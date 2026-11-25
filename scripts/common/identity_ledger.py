"""Read the verified identity evidence for exported block signatures."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast

import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from s2and.incremental_linking.contracts import canonical_json_digest
from scripts.common.arrow_roots import read_json_object, resolved_manifest_artifact_path
from scripts.common.block_roles import BlockRoles, normalize_block_key
from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS

EVIDENCE_SHA256_FIELD = "identity_evidence_sha256"
ROLE_SHA256_FIELD = "block_role_assignments_sha256"
FREEZE_BINDING = ".role_freeze.json"
INTAKE_SOURCE_RECORD = "request_source.json"


def excluded_orcid_groups_digest() -> str:
    """Bind evidence to the current identity exclusions."""
    return canonical_json_digest(sorted(EXCLUDED_ORCID_GROUPS))


class GoldEvidence(TypedDict):
    orcid: str
    scored: bool
    split: str | None


class RejectedGoldEvidence(GoldEvidence):
    reason: str


class MentionEvidence(TypedDict):
    block_key: str
    signature_id: str
    corpus_paper_id: str
    position: int
    gold: list[GoldEvidence]
    rejected_gold: NotRequired[list[RejectedGoldEvidence]]
    deposited_orcid: str | None
    excluded: bool
    reviewed_error: bool
    evidence_state: Literal["none", "single", "conflict"]


class EvidenceBlock(TypedDict):
    path: str
    sha256: str
    signature_count: int
    dataset_manifest_sha256: str
    dataset: str
    request_source_sha256: str
    permission_sha256: dict[str, str]
    reviewed_errors_sha256: str


def evidence_identities(row: MentionEvidence) -> frozenset[str]:
    """Return the distinct normalized identities from matched gold and deposits."""
    deposited = row["deposited_orcid"]
    gold = {entry["orcid"] for entry in row["gold"]}
    if deposited:
        gold.add(deposited)
    return frozenset(gold)


@dataclass(frozen=True)
class IdentityEvidence:
    """Read evidence by block and verify each partition against the table manifest."""

    path: Path
    sha256: str
    snapshot_sha256: str
    intake_sha256: str
    name_counts_index_sha256: str
    blocks: dict[str, EvidenceBlock]
    snapshot_root: Path

    def block_digests(self) -> dict[str, str]:
        """Return the binding for each evidence partition and its intake inputs."""
        return {block: canonical_json_digest(entry) for block, entry in self.blocks.items()}

    def read_block(self, block: str, *, dataset_dir: Path | None = None) -> dict[str, MentionEvidence]:
        """Return evidence keyed by signature and reject a different dataset generation."""
        block = normalize_block_key(block)
        if block not in self.blocks:
            raise ValueError(f"No identity evidence for {block!r}")
        metadata = self.blocks[block]
        path = (self.path.parent / metadata["path"]).resolve()
        if path.parent != self.path.parent or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Identity evidence digest mismatch for {block!r}")
        if dataset_dir is not None:
            manifest = dataset_dir / "manifest.json"
            if sha256_file(manifest) != metadata["dataset_manifest_sha256"]:
                raise ValueError(f"Identity evidence was built for a different dataset generation: {block!r}")
            if sha256_file(dataset_dir / INTAKE_SOURCE_RECORD) != metadata["request_source_sha256"]:
                raise ValueError(f"Identity evidence intake permissions changed: {block!r}")
            arrow_manifest = read_json_object(manifest)
            signature_path = resolved_manifest_artifact_path(
                manifest, arrow_manifest["paths"]["signatures"], key="signatures"
            )
            if sha256_file(signature_path) != arrow_manifest["files"]["signatures"]["sha256"]:
                raise ValueError(f"Identity evidence signature digest mismatch: {block!r}")
            index = resolved_manifest_artifact_path(
                manifest, arrow_manifest["paths"]["name_counts_index"], key="name_counts_index"
            )
            if sha256_file(index / "manifest.json") != self.name_counts_index_sha256:
                raise ValueError(f"Identity evidence name-counts index digest mismatch: {block!r}")
        rows = cast(list[MentionEvidence], pq.read_table(path).to_pylist())
        by_signature = {row["signature_id"]: row for row in rows}
        if (
            len(by_signature) != len(rows)
            or len(rows) != metadata["signature_count"]
            or any(row["block_key"] != block for row in rows)
        ):
            raise ValueError(f"Invalid identity evidence coverage for {block!r}")
        return by_signature


def load_identity_evidence(path: Path) -> IdentityEvidence:
    """Load one evidence manifest and verify the exclusion policy."""
    path = path.resolve()
    payload = read_json_object(path)
    if payload.get("format_version") != 1 or not isinstance(payload.get("blocks"), dict) or not payload["blocks"]:
        raise ValueError(f"Invalid identity evidence manifest: {path}")
    if payload.get("excluded_orcid_groups_sha256") != excluded_orcid_groups_digest():
        raise ValueError(f"Identity evidence uses different excluded ORCID groups: {path}")
    return IdentityEvidence(
        path,
        sha256_file(path),
        payload["snapshot_sha256"],
        payload["intake_sha256"],
        payload["name_counts_index_sha256"],
        payload["blocks"],
        Path(payload["snapshot_root"]),
    )


@dataclass(frozen=True)
class Eligibility:
    """Record permitted fitting and scoring uses for one mention."""

    fit_ok: bool
    score_orcid: str | None
    identities: frozenset[str]
    identity_veto: bool


@dataclass(frozen=True)
class BlockEligibility:
    """Evaluate every mention against one role and its expected gold split."""

    block: str
    role: str
    expected_split: str
    rows: dict[str, MentionEvidence]
    roles: BlockRoles

    def require_signatures(self, signatures: Collection[str]) -> None:
        """Check that a consumer sees the same signature population as the ledger."""
        keys = set(signatures)
        if keys != self.rows.keys():
            raise ValueError(f"Ledger signature coverage differs for {self.block!r}")

    def eligibility(self, signature_id: str) -> Eligibility:
        """Return fit eligibility and the affirmative gold identity, if any."""
        if signature_id not in self.rows:
            raise ValueError(f"No identity evidence for {self.block!r} signature {signature_id!r}")
        row = self.rows[signature_id]
        identities = evidence_identities(row)
        if row["excluded"] or not identities.isdisjoint(EXCLUDED_ORCID_GROUPS):
            return Eligibility(False, None, identities, True)
        if not row["reviewed_error"]:
            for orcid in identities:
                if orcid not in self.roles.block_orcids[self.block] or orcid not in self.roles.identity_roles:
                    raise ValueError(f"Unknown evidence identity {orcid!r} in frozen block {self.block!r}")
        foreign = any(self.roles.identity_roles.get(orcid, frozenset()) - {self.role} for orcid in identities)
        fit_ok = not row["reviewed_error"] and not row.get("rejected_gold") and len(identities) <= 1 and not foreign
        score_orcid = None
        if fit_ok and identities:
            score_orcid = next(
                (gold["orcid"] for gold in row["gold"] if gold["scored"] and gold["split"] == self.expected_split),
                None,
            )
        return Eligibility(fit_ok, score_orcid, identities, foreign)

    def scored(self) -> dict[str, str]:
        """Return score-eligible gold labels for this block."""
        return {sid: orcid for sid in self.rows if (orcid := self.eligibility(sid).score_orcid) is not None}

    @staticmethod
    def _component_is_safe(decisions: list[Eligibility]) -> bool:
        return (
            all(decision.fit_ok for decision in decisions)
            and len(set().union(*(decision.identities for decision in decisions))) <= 1
        )

    def component_is_safe(self, members: Collection[str]) -> bool:
        """Check every member for fit eligibility and mixed evidence."""
        return self._component_is_safe([self.eligibility(sid) for sid in members])

    def component_label(self, members: Collection[str]) -> str | None:
        """Label a component when all members pass and its evidence identifies one author."""
        decisions = [self.eligibility(sid) for sid in members]
        if not self._component_is_safe(decisions):
            return None
        return next((decision.score_orcid for decision in decisions if decision.score_orcid is not None), None)


@dataclass(frozen=True)
class IdentityLedger:
    """Bind a verified evidence table to the frozen role assignments."""

    evidence: IdentityEvidence
    roles: BlockRoles

    @property
    def bindings(self) -> dict[str, str]:
        """Return the provenance fields shared by downstream manifests."""
        return {EVIDENCE_SHA256_FIELD: self.evidence.sha256, ROLE_SHA256_FIELD: self.roles.sha256}

    def for_block(self, block: str, *, dataset_dir: Path | None = None) -> BlockEligibility:
        """Return the eligibility view for an assigned block."""
        block = normalize_block_key(block)
        role = self.roles.role_for(block)
        split = self.roles.source_splits[block]
        rows = self.evidence.read_block(block, dataset_dir=dataset_dir)
        return BlockEligibility(block, role, split, rows, self.roles)


def ledger_for_roles(path: Path | None, roles: BlockRoles | None) -> IdentityLedger | None:
    """Require the identity ledger whenever frozen ledger-backed roles apply."""
    if path is None and (roles is None or roles.evidence_sha256 is None):
        return None
    if path is None or roles is None:
        raise ValueError("Ledger-backed roles require --identity-evidence and --block-role-assignments")
    evidence = load_identity_evidence(path)
    if (
        evidence.sha256 != roles.evidence_sha256
        or evidence.snapshot_sha256 != roles.snapshot_sha256
        or evidence.block_digests() != roles.evidence_blocks
    ):
        raise ValueError("Frozen roles do not match the identity evidence and intake snapshot")
    return IdentityLedger(evidence, roles)

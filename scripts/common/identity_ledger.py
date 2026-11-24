"""Read the verified identity evidence for exported block signatures."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from scripts.common.arrow_roots import read_json_object
from scripts.common.block_roles import FINAL_TEST_ROLES, BlockRoles, normalize_block_key

EVIDENCE_SHA256_FIELD = "identity_evidence_sha256"
ROLE_SHA256_FIELD = "block_role_assignments_sha256"


class GoldEvidence(TypedDict):
    orcid: str
    scored: bool
    split: str | None


class MentionEvidence(TypedDict):
    block_key: str
    signature_id: str
    corpus_paper_id: str
    position: int
    gold: list[GoldEvidence]
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


def evidence_identities(row: MentionEvidence) -> frozenset[str]:
    """Return the distinct normalized identities from both evidence sources."""
    deposited = row["deposited_orcid"]
    return frozenset([*(entry["orcid"] for entry in row["gold"]), *([deposited] if deposited else [])])


@dataclass(frozen=True)
class IdentityEvidence:
    """Read evidence by block and verify each partition against the table manifest."""

    path: Path
    sha256: str
    snapshot_sha256: str
    intake_sha256: str
    blocks: dict[str, EvidenceBlock]

    def read_block(self, block: str, *, dataset_dir: Path | None = None) -> dict[str, MentionEvidence]:
        """Return evidence keyed by signature and reject a different dataset generation."""
        block = normalize_block_key(block)
        if block not in self.blocks:
            raise ValueError(f"No identity evidence for {block!r}")
        metadata = self.blocks[block]
        path = (self.path.parent / metadata["path"]).resolve()
        if path.parent != self.path.parent or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Identity evidence digest mismatch for {block!r}")
        if (
            dataset_dir is not None
            and sha256_file(dataset_dir / "manifest.json") != metadata["dataset_manifest_sha256"]
        ):
            raise ValueError(f"Identity evidence was built for a different dataset generation: {block!r}")
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
    """Load the manifest of one immutable evidence table."""
    path = path.resolve()
    payload = read_json_object(path)
    if payload.get("format_version") != 1 or not isinstance(payload.get("blocks"), dict) or not payload["blocks"]:
        raise ValueError(f"Invalid identity evidence manifest: {path}")
    return IdentityEvidence(
        path, sha256_file(path), payload["snapshot_sha256"], payload["intake_sha256"], payload["blocks"]
    )


@dataclass(frozen=True)
class Eligibility:
    """Record permitted fitting and scoring uses for one mention."""

    fit_ok: bool
    score_orcid: str | None
    identities: frozenset[str]


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
        if row["excluded"] or row["reviewed_error"]:
            return Eligibility(False, None, identities)
        for orcid in identities:
            if orcid not in self.roles.block_orcids[self.block] or orcid not in self.roles.identity_roles:
                raise ValueError(f"Unknown evidence identity {orcid!r} in frozen block {self.block!r}")
        fit_ok = row["evidence_state"] != "conflict" and all(
            self.roles.identity_roles[orcid] == {self.role} for orcid in identities
        )
        score_orcid = None
        if fit_ok and row["evidence_state"] == "single":
            score_orcid = next(
                (gold["orcid"] for gold in row["gold"] if gold["scored"] and gold["split"] == self.expected_split),
                None,
            )
        return Eligibility(fit_ok, score_orcid, identities)

    def scored(self) -> dict[str, str]:
        """Return score-eligible gold labels for this block."""
        return {sid: orcid for sid in self.rows if (orcid := self.eligibility(sid).score_orcid) is not None}

    def component_label(self, members: Collection[str]) -> str | None:
        """Label a component when all members pass and its evidence identifies one author."""
        decisions = [self.eligibility(sid) for sid in members]
        if any(not decision.fit_ok for decision in decisions):
            return None
        identities = set().union(*(decision.identities for decision in decisions))
        if len(identities) != 1:
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

    def require_bindings(self, payload: Mapping[str, object], *, context: str) -> None:
        """Reject outputs built against any other evidence or role assignment."""
        for key, expected in self.bindings.items():
            if payload.get(key) != expected:
                raise ValueError(f"{context} {key} does not match the identity ledger")

    def for_block(self, block: str, *, dataset_dir: Path | None = None) -> BlockEligibility:
        """Return the eligibility view for an assigned block."""
        block = normalize_block_key(block)
        role = self.roles.role_for(block)
        split = "holdout" if role in FINAL_TEST_ROLES else "development"
        rows = self.evidence.read_block(block, dataset_dir=dataset_dir)
        return BlockEligibility(block, role, split, rows, self.roles)


def ledger_for_roles(path: Path | None, roles: BlockRoles | None) -> IdentityLedger | None:
    """Require the identity ledger whenever frozen ledger-backed roles apply."""
    if path is None and (roles is None or roles.evidence_sha256 is None):
        return None
    if path is None or roles is None:
        raise ValueError("Ledger-backed roles require --identity-evidence and --block-role-assignments")
    evidence = load_identity_evidence(path)
    if evidence.sha256 != roles.evidence_sha256 or evidence.snapshot_sha256 != roles.snapshot_sha256:
        raise ValueError("Frozen roles do not match the identity evidence and intake snapshot")
    return IdentityLedger(evidence, roles)

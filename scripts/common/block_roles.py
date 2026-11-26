"""Read frozen block roles and their identity coverage."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from s2and._sha256 import is_lowercase_sha256
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.text import normalize_orcid

BLOCK_BINDINGS_FIELD = "block_bindings"
ROLE_SHA256_FIELD = "block_role_assignments_sha256"


def assignment_digest(path: Path) -> str:
    """Hash a role assignment by its JSON content, so whitespace or key order never changes its identity."""
    return canonical_json_digest(json.loads(path.read_text(encoding="utf-8")))


GATE_BINDINGS_FIELD = "gate_thresh_block_bindings"
GATE_CANDIDATE_ORCIDS = "gate_thresh_candidate_orcids"
FINAL_TEST_ROLE = "final_test"
VALID_ROLES = ("fit_pool", "eps_cal", "gate_thresh", "dev_test", FINAL_TEST_ROLE)
_WHITESPACE_OR_UNDERSCORE = re.compile(r"[\s_]+")


def normalize_block_key(value: str | None) -> str:
    """Normalize block keys for comparison across underscore and space conventions."""
    return _WHITESPACE_OR_UNDERSCORE.sub(" ", value.strip().lower()) if value else ""


def assigned_identity_roles(
    block_orcids: Mapping[str, Collection[str]], block_to_role: Mapping[str, str]
) -> dict[str, frozenset[str]]:
    """Index identities by the frozen roles of their assigned blocks."""
    roles_by_orcid: dict[str, set[str]] = defaultdict(set)
    for block, role in block_to_role.items():
        for orcid in block_orcids[block]:
            roles_by_orcid[orcid].add(role)
    return {orcid: frozenset(roles) for orcid, roles in roles_by_orcid.items()}


@dataclass(frozen=True)
class BlockRoles:
    """Frozen block assignments and the roles holding each identity."""

    block_to_role: dict[str, str]
    sha256: str
    block_orcids: dict[str, frozenset[str]]
    identity_roles: dict[str, frozenset[str]] = field(default_factory=dict)
    evidence_sha256: str | None = None
    snapshot_sha256: str | None = None
    source_splits: dict[str, str] = field(default_factory=dict)
    evidence_blocks: dict[str, str] = field(default_factory=dict)
    excluded_orcid_groups_sha256: str | None = None
    gate_block_orcids: dict[str, frozenset[str]] = field(default_factory=dict)
    pairwise_validation_blocks: frozenset[str] = frozenset()

    def require_assignment(self, payload: Mapping[str, object], *, context: str) -> None:
        """Check that an artifact was built with this exact role assignment."""
        if payload.get(ROLE_SHA256_FIELD) != self.sha256:
            raise ValueError(f"{context} does not match the master block role assignment")

    def require_current_exclusions(self) -> None:
        """Reject roles built with a different policy or containing excluded ORCIDs."""
        from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS, excluded_orcid_groups_digest

        if (
            self.evidence_sha256 is not None and self.excluded_orcid_groups_sha256 != excluded_orcid_groups_digest()
        ) or any(not orcids.isdisjoint(EXCLUDED_ORCID_GROUPS) for orcids in self.block_orcids.values()):
            raise ValueError("Frozen roles use different excluded ORCID groups. Rebuild the evidence and freeze.")

    def bindings_for(self, blocks: Collection[str]) -> dict[str, dict[str, str]]:
        """Bind each requested block to its evidence and effective eligibility."""
        bindings = {}
        for block in sorted({normalize_block_key(key) for key in blocks}):
            role = self.role_for(block)
            outside_role = sorted(orcid for orcid in self.block_orcids[block] if self.identity_roles[orcid] != {role})
            role_inputs = [role, self.source_splits[block], outside_role]
            if role == "gate_thresh":
                if block not in self.gate_block_orcids:
                    raise ValueError("Frozen gate roles lack candidate identity coverage. Re-freeze the roles.")
                # Rejected gold and reviewed errors can veto without contributing to identity_roles.
                role_inputs.append(
                    [
                        (orcid, sorted(self.identity_roles.get(orcid, ())))
                        for orcid in sorted(self.gate_block_orcids[block])
                    ]
                )
            bindings[block] = {
                "evidence_sha256": self.evidence_blocks[block],
                "role_sha256": canonical_json_digest(role_inputs),
            }
        return bindings

    def require_bindings(self, payload: Mapping[str, object], blocks: Collection[str], *, context: str) -> None:
        """Check that an artifact has current bindings for exactly its blocks."""
        if payload.get(BLOCK_BINDINGS_FIELD) != self.bindings_for(blocks):
            raise ValueError(f"{context} block provenance does not match the identity ledger")

    def gate_judgment_bindings(self, candidates: Mapping[str, Collection[str]]) -> dict[str, dict[str, str]]:
        """Bind gate blocks and the assigned blocks holding their candidates' identities."""
        for block, identities in candidates.items():
            if self.role_for(block) != "gate_thresh" or not set(identities).issubset(self.gate_block_orcids[block]):
                raise ValueError(f"Unknown gate candidate identities in {block!r}")
        named = {orcid for identities in candidates.values() for orcid in identities}
        holders = {block for block in self.block_to_role if not named.isdisjoint(self.block_orcids[block])}
        return self.bindings_for(set(candidates) | holders)

    def role_for(self, block_key: str) -> str:
        """Return the assigned role or reject an unassigned block."""
        key = normalize_block_key(block_key)
        if key not in self.block_to_role:
            raise ValueError(f"Block {block_key!r} is absent from the frozen role assignments")
        return self.block_to_role[key]

    def scores(self, block_key: str, orcid: str) -> bool:
        """Return whether a normalized ORCID can score in the assigned block."""
        role = self.role_for(block_key)
        if orcid not in self.block_orcids[normalize_block_key(block_key)]:
            raise ValueError(f"ORCID {orcid!r} is absent from frozen block {block_key!r}")
        return self.identity_roles[orcid] == {role}


def load_block_roles(path: Path) -> BlockRoles:
    """Load the scored assignments and reject inconsistent manifests."""
    content = path.read_bytes()
    payload = json.loads(content.decode("utf-8"))
    roles = payload.get("roles")
    block_to_role = payload.get("block_to_role")
    if (isinstance(roles, dict) and "gate_fit" in roles) or (
        isinstance(block_to_role, dict) and "gate_fit" in block_to_role.values()
    ):
        raise ValueError(f"Block role manifest {path} uses the retired gate_fit role. Reassign blocks.")
    verification = payload.get("verification", {})
    if not isinstance(verification, dict) or verification.get("cross_role_orcid_overlap") != 0:
        raise ValueError(f"Block role manifest {path} does not certify zero cross-role ORCID overlap for scoring")
    if not isinstance(roles, dict):
        raise ValueError(f"Block role manifest {path} is missing roles")
    by_block: dict[str, str] = {}
    for role, blocks in roles.items():
        if role not in VALID_ROLES:
            raise ValueError(f"Unknown block role {role!r} in {path}. Expected one of {VALID_ROLES}.")
        if not isinstance(blocks, list):
            raise ValueError(f"Expected a list of blocks for role {role!r} in {path}")
        for block in blocks:
            key = normalize_block_key(block)
            if not key or key in by_block:
                raise ValueError(f"Empty or duplicate block {block!r} in role manifest {path}")
            by_block[key] = role
    if "block_to_role" in payload and payload["block_to_role"] != by_block:
        raise ValueError(f"Inconsistent block_to_role and roles in {path}")
    validation = payload.get("pairwise_validation_blocks")
    if (
        not isinstance(validation, list)
        or any(not isinstance(block, str) or by_block.get(block) != "fit_pool" for block in validation)
        or len(set(validation)) != len(validation)
    ):
        raise ValueError(f"Expected unique fit_pool keys in pairwise_validation_blocks in {path}")
    bound = payload.get("format_version") == 2
    raw_orcids = payload.get("block_orcids")
    evidence_blocks = payload.get("evidence_blocks", {}) if bound else {}
    if bound and (
        not isinstance(evidence_blocks, dict)
        or not evidence_blocks
        or not all(is_lowercase_sha256(digest) for digest in evidence_blocks.values())
    ):
        raise ValueError(f"Invalid identity evidence partition digests in {path}")
    expected = set(evidence_blocks) if bound else set(by_block)
    if not isinstance(raw_orcids, dict) or set(raw_orcids) != expected or by_block.keys() - raw_orcids.keys():
        raise ValueError(f"Expected identity coverage for every frozen block in {path}")
    block_orcids: dict[str, frozenset[str]] = {}
    for block, orcids in raw_orcids.items():
        if not isinstance(orcids, list) or any(
            not isinstance(orcid, str) or not orcid or normalize_orcid(orcid) != orcid for orcid in orcids
        ):
            raise ValueError(f"Invalid frozen ORCID list for {block!r} in {path}")
        block_orcids[block] = frozenset(str(orcid) for orcid in orcids)
    digests = {key: payload.get(key) for key in ("identity_evidence_sha256", "snapshot_sha256")} if bound else {}
    if bound and not all(is_lowercase_sha256(value) for value in digests.values()):
        raise ValueError(f"Invalid identity evidence or snapshot digest in {path}")
    source_splits = payload.get("block_source_splits", {}) if bound else {}
    if bound and (
        not isinstance(source_splits, dict)
        or source_splits.keys() != expected
        or any(split not in {"development", "holdout"} for split in source_splits.values())
    ):
        raise ValueError(f"Expected source splits for every evidence block in {path}")
    gate_block_orcids: dict[str, frozenset[str]] = {}
    if bound and "gate_block_orcids" in payload:
        gate_coverage = payload["gate_block_orcids"]
        gate_blocks = {block for block, role in by_block.items() if role == "gate_thresh"}
        if not isinstance(gate_coverage, dict) or gate_coverage.keys() != gate_blocks:
            raise ValueError(f"Invalid gate identity coverage in {path}")
        for block, orcids in gate_coverage.items():
            if (
                not isinstance(orcids, list)
                or any(not isinstance(orcid, str) or not orcid or normalize_orcid(orcid) != orcid for orcid in orcids)
                or not block_orcids[block].issubset(orcids)
            ):
                raise ValueError(f"Invalid gate ORCID list for {block!r} in {path}")
            gate_block_orcids[block] = frozenset(cast(list[str], orcids))
    result = BlockRoles(
        by_block,
        canonical_json_digest(payload),
        block_orcids,
        assigned_identity_roles(block_orcids, by_block),
        digests.get("identity_evidence_sha256"),
        digests.get("snapshot_sha256"),
        source_splits,
        evidence_blocks,
        payload.get("excluded_orcid_groups_sha256") if bound else None,
        gate_block_orcids,
        frozenset(cast(list[str], validation)),
    )
    result.require_current_exclusions()
    return result

"""Read frozen block roles and their identity coverage."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from s2and._sha256 import is_lowercase_sha256, sha256_file
from s2and.text import normalize_orcid

FINAL_TEST_ROLES = ("final_test_a", "final_test_b")
_WHITESPACE_OR_UNDERSCORE = re.compile(r"[\s_]+")


def normalize_block_key(value: str | None) -> str:
    """Normalize block keys for comparison across underscore and space conventions."""
    return _WHITESPACE_OR_UNDERSCORE.sub(" ", value.strip().lower()) if value else ""


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    verification = payload.get("verification", {})
    if not isinstance(verification, dict) or verification.get("cross_role_orcid_overlap") != 0:
        raise ValueError(f"Block role manifest {path} does not certify zero cross-role ORCID overlap for scoring")
    roles = payload.get("roles")
    if not isinstance(roles, dict):
        raise ValueError(f"Block role manifest {path} is missing roles")
    by_block: dict[str, str] = {}
    for role, blocks in roles.items():
        if not isinstance(blocks, list):
            raise ValueError(f"Expected a list of blocks for role {role!r} in {path}")
        for block in blocks:
            key = normalize_block_key(block)
            if not key or key in by_block:
                raise ValueError(f"Empty or duplicate block {block!r} in role manifest {path}")
            by_block[key] = role
    if "block_to_role" in payload and payload["block_to_role"] != by_block:
        raise ValueError(f"Inconsistent block_to_role and roles in {path}")
    bound = payload.get("format_version") == 2
    raw_orcids = payload.get("block_orcids")
    expected = set(payload["evidence_blocks"]) if bound else set(by_block)
    if not isinstance(raw_orcids, dict) or set(raw_orcids) != expected or by_block.keys() - raw_orcids.keys():
        raise ValueError(f"Expected identity coverage for every frozen block in {path}")
    block_orcids: dict[str, frozenset[str]] = {}
    roles_by_orcid: dict[str, set[str]] = defaultdict(set)
    for block, orcids in raw_orcids.items():
        if not isinstance(orcids, list) or any(
            not isinstance(orcid, str) or not orcid or normalize_orcid(orcid) != orcid for orcid in orcids
        ):
            raise ValueError(f"Invalid frozen ORCID list for {block!r} in {path}")
        block_orcids[block] = frozenset(str(orcid) for orcid in orcids)
        for orcid in orcids:
            roles_by_orcid[orcid].add(by_block.get(block, "unassigned"))
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
    return BlockRoles(
        by_block,
        sha256_file(path),
        block_orcids,
        {orcid: frozenset(roles) for orcid, roles in roles_by_orcid.items()},
        digests.get("identity_evidence_sha256"),
        digests.get("snapshot_sha256"),
        source_splits,
    )

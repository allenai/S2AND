"""Read frozen block roles and apply their ORCID scoring policy."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from s2and._sha256 import sha256_file
from s2and.text import normalize_orcid

FINAL_TEST_ROLES = ("final_test_a", "final_test_b")
_WHITESPACE_OR_UNDERSCORE = re.compile(r"[\s_]+")


def normalize_block_key(value: str | None) -> str:
    """Normalize block keys for comparison across underscore and space conventions."""
    return _WHITESPACE_OR_UNDERSCORE.sub(" ", value.strip().lower()) if value else ""


def in_orcid_home_role(orcid: str, role: str, home_roles: Mapping[str, str]) -> bool:
    """Return whether a normalized ORCID is eligible for scoring in this role."""
    return home_roles.get(orcid, role) == role


@dataclass(frozen=True)
class BlockRoles:
    """Frozen block assignments and the home role of each shared ORCID."""

    block_to_role: dict[str, str]
    orcid_home_role: dict[str, str]
    sha256: str
    closure_mode: str
    block_orcids: dict[str, frozenset[str]]
    identity_roles: dict[str, frozenset[str]] = field(default_factory=dict)
    evidence_sha256: str | None = None
    snapshot_sha256: str | None = None

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
        if self.evidence_sha256 is not None:
            return self.identity_roles[orcid] == {role}
        return in_orcid_home_role(orcid, role, self.orcid_home_role)


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
    if payload.get("format_version") == 2:
        return _load_evidence_roles(path, payload, by_block)
    mode = payload.get("closure_mode", verification.get("closure_mode", "strict"))
    if mode not in {"strict", "home_pool"}:
        raise ValueError(f"Invalid closure_mode in {path}")
    homes = payload.get("orcid_home_role", {} if mode == "strict" else None)
    assigned_roles = set(by_block.values())
    if not isinstance(homes, dict) or any(
        normalize_orcid(orcid) != orcid or role not in assigned_roles for orcid, role in homes.items()
    ):
        raise ValueError(f"Invalid orcid_home_role in {path}")
    raw_orcids = payload.get("block_orcids")
    if not isinstance(raw_orcids, dict) or set(raw_orcids) != by_block.keys():
        raise ValueError(f"Expected block_orcids for every assigned block in {path}. Re-freeze the role assignments.")
    block_orcids: dict[str, frozenset[str]] = {}
    roles_by_orcid: dict[str, set[str]] = defaultdict(set)
    for block, orcids in raw_orcids.items():
        if not isinstance(orcids, list) or any(
            not isinstance(orcid, str) or not orcid or normalize_orcid(orcid) != orcid for orcid in orcids
        ):
            raise ValueError(f"Invalid frozen ORCID list for {block!r} in {path}")
        block_orcids[block] = frozenset(str(orcid) for orcid in orcids)
        for orcid in orcids:
            roles_by_orcid[orcid].add(by_block[block])
    overlap = {orcid: roles for orcid, roles in roles_by_orcid.items() if len(roles) > 1}
    if (
        mode == "strict"
        and overlap
        or homes.keys() != overlap.keys()
        or any(home not in overlap[orcid] for orcid, home in homes.items())
    ):
        raise ValueError(f"ORCID home roles do not match frozen block ORCIDs in {path}")
    return BlockRoles(by_block, homes, sha256_file(path), mode, block_orcids)


def _load_evidence_roles(path: Path, payload: dict, by_block: dict[str, str]) -> BlockRoles:
    block_orcids = {}
    roles_by_orcid: dict[str, set[str]] = defaultdict(set)
    if (
        set(payload["block_orcids"]) != set(payload["evidence_blocks"])
        or by_block.keys() - payload["block_orcids"].keys()
    ):
        raise ValueError(f"Frozen roles and evidence do not cover the same exported blocks: {path}")
    for block, orcids in payload["block_orcids"].items():
        if not isinstance(orcids, list) or any(normalize_orcid(orcid) != orcid for orcid in orcids):
            raise ValueError(f"Invalid frozen ORCID list for {block!r} in {path}")
        block_orcids[block] = frozenset(orcids)
        for orcid in orcids:
            roles_by_orcid[orcid].add(by_block.get(block, "unassigned"))
    return BlockRoles(
        by_block,
        {},
        sha256_file(path),
        "block_only",
        block_orcids,
        {orcid: frozenset(roles) for orcid, roles in roles_by_orcid.items()},
        payload["identity_evidence_sha256"],
        payload["snapshot_sha256"],
    )

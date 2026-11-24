"""Read frozen block roles and apply their ORCID scoring policy."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
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

    def role_for(self, block_key: str) -> str:
        """Return the assigned role or reject an unassigned block."""
        key = normalize_block_key(block_key)
        if key not in self.block_to_role:
            raise ValueError(f"Block {block_key!r} is absent from the frozen role assignments")
        return self.block_to_role[key]

    def scores(self, block_key: str, orcid: str) -> bool:
        """Return whether a normalized ORCID can score in the assigned block."""
        return in_orcid_home_role(orcid, self.role_for(block_key), self.orcid_home_role)


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
    mode = payload.get("closure_mode", verification.get("closure_mode", "strict"))
    if mode not in {"strict", "home_pool"}:
        raise ValueError(f"Invalid closure_mode in {path}")
    homes = payload.get("orcid_home_role", {} if mode == "strict" else None)
    assigned_roles = set(by_block.values())
    if not isinstance(homes, dict) or any(
        normalize_orcid(orcid) != orcid or role not in assigned_roles for orcid, role in homes.items()
    ):
        raise ValueError(f"Invalid orcid_home_role in {path}")
    if mode == "home_pool" and verification.get("raw_cross_role_orcid_overlap") != len(homes):
        raise ValueError(f"ORCID home roles do not cover the verified raw overlap in {path}")
    return BlockRoles(by_block, homes, sha256_file(path), mode)

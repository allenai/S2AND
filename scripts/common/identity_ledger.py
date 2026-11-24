"""Read the verified identity evidence for exported block signatures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from scripts.common.arrow_roots import read_json_object
from scripts.common.block_roles import normalize_block_key

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

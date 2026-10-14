"""Keep giant-block ORCID references outside prediction and score known identities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from s2and.arrow_inputs import ArrowDataset
from s2and.text import normalize_orcid
from scripts.production.counts.generate_orcid_name_prefix_counts import EXCLUDED_ORCID_GROUPS
from scripts.production.model.train_linker_and_finalize import _load_initial_query_sources


def read_reference(
    arrow_dataset: ArrowDataset, snapshot_root: Path, dataset: str, exclusions_path: Path | None = None
) -> tuple[dict[str, str], frozenset[str], dict[str, Any]]:
    """Exclude derived query copies and reviewed mixed-identity ORCID groups."""
    origins = _load_initial_query_sources(snapshot_root).get(dataset, {})
    clones = frozenset(origins)
    with arrow_dataset.use() as lease, lease.open_file("signatures") as infile:
        with pa.PythonFile(infile, mode="r") as source:
            table = ipc.open_file(source).read_all().select(["signature_id", "author_orcid"])
    rows = table.to_pydict()
    reference: dict[str, str] = {}
    excluded_rows = 0
    for signature_id, raw_orcid in zip(rows["signature_id"], rows["author_orcid"], strict=True):
        signature_id = str(signature_id)
        if signature_id.startswith("visibleq:") and signature_id not in clones:
            raise ValueError("Derived query copies require their source map in the snapshot")
        if signature_id in clones or raw_orcid is None:
            continue
        orcid = normalize_orcid(raw_orcid)
        if orcid in EXCLUDED_ORCID_GROUPS:
            excluded_rows += 1
        elif orcid:
            reference[signature_id] = orcid
    excluded_groups: set[str] = set()
    if exclusions_path is not None:
        excluded_ids = {
            line.strip() for line in exclusions_path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        excluded_groups = {
            reference[origins.get(signature, signature)]
            for signature in excluded_ids
            if origins.get(signature, signature) in reference
        }
        reference = {signature: orcid for signature, orcid in reference.items() if orcid not in excluded_groups}
    counts = Counter(reference.values())
    reference = {signature: orcid for signature, orcid in reference.items() if counts[orcid] >= 2}
    if not reference:
        raise ValueError(f"No repeated ORCID references for dataset={dataset}")
    return (
        reference,
        clones,
        {
            "reference_kind": "repeated_orcid_identity",
            "reference_signatures": len(reference),
            "reference_groups": len(set(reference.values())),
            "excluded_initial_query_copies": len(clones),
            "excluded_mixed_orcid_rows": excluded_rows,
            "excluded_holdout_orcid_groups": len(excluded_groups),
            "excluded_orcid_groups": sorted(EXCLUDED_ORCID_GROUPS),
        },
    )


def planning_targets(dataset: str, reference: Mapping[str, str]) -> pd.DataFrame:
    """Supply target IDs to the existing planner without an expanded pair table."""
    signature_ids = sorted(reference)
    return pd.DataFrame(
        {
            "dataset": dataset,
            "query_signature_id": signature_ids,
            "member_signature_id": signature_ids,
            "label": 1,
        }
    )


def score_assignments(reference: Mapping[str, str], assignments: Mapping[str, str], eps: float) -> dict[str, Any]:
    """Compute unrounded B3 on covered references, including cross-subblock splits."""
    known = {signature: orcid for signature, orcid in reference.items() if signature in assignments}
    if not known:
        raise ValueError("The selected subblocks cover no ORCID references")
    truth_sizes = Counter(known.values())
    predicted_sizes = Counter(assignments[signature] for signature in known)
    intersections = Counter((orcid, assignments[signature]) for signature, orcid in known.items())
    count = len(known)
    precision = sum(size * size / predicted_sizes[cluster] for (_, cluster), size in intersections.items()) / count
    recall = sum(size * size / truth_sizes[orcid] for (orcid, _), size in intersections.items()) / count
    return {
        "eps": eps,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        "reference_signatures": len(reference),
        "scored_signatures": count,
        "reference_coverage": count / len(reference),
        "scored_reference_groups": len(truth_sizes),
    }

"""Keep giant-block ORCID references outside prediction and final tests, and score known identities."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking_training.classic import _load_classic_stratified_eval_rows, load_bundle
from s2and.text import normalize_orcid
from scripts.production.counts.generate_orcid_name_prefix_counts import EXCLUDED_ORCID_GROUPS
from scripts.production.model.train_linker_and_finalize import _load_initial_query_sources

# Selection references exclude final-test identities and reviewed ORCID errors.
REFERENCE_POLICY = "final_test_orcid_groups_and_reviewed_error_records_excluded"
ORCID_ERROR_RECORDS = "orcid_error_query_signatures.json"


@dataclass(frozen=True)
class OrcidReference:
    """Known ORCID identities for EPS curves, with the populations they exclude.

    ``repeated`` selects EPS. ``with_singletons`` is a diagnostic: one-record
    ORCID groups add merge penalties that the repeated view omits.
    """

    repeated: dict[str, str]
    with_singletons: dict[str, str]
    excluded_copies: frozenset[str]
    stats: dict[str, Any]


def final_test_query_identities(snapshot_root: Path, dataset: str) -> frozenset[str]:
    """Return the dataset's frozen final-test query records from a snapshot.

    Args:
        snapshot_root: Snapshot with the linker bundle and its frozen split assignments.
        dataset: Incremental dataset name.

    Returns:
        Source signature IDs of every final-test query.
    """
    bundle = load_bundle(snapshot_root)
    spec = dict(bundle.models["classic"])
    split_spec = dict(spec["stratified_eval_test_split"])
    rows, assignments = _load_classic_stratified_eval_rows(
        bundle, spec, split_spec, splits=(str(split_spec["test_split"]),), include_calibration_source=True
    )
    return resolve_test_query_identities(
        rows, assignments, _load_initial_query_sources(snapshot_root).get(dataset, {}), dataset
    )


def resolve_test_query_identities(
    rows: pd.DataFrame, assignments: pd.DataFrame, origins: Mapping[str, str], dataset: str
) -> frozenset[str]:
    """Resolve assigned test query groups to source records.

    Args:
        rows: Label rows of the test split, with query group and query signature IDs.
        assignments: Frozen test-split assignments.
        origins: Initial-only query copies mapped to their source records.
        dataset: Incremental dataset name.

    Returns:
        Source signature IDs of every assigned test query.

    Raises:
        ValueError: If a test group has no label rows or a copy has no source record.
    """
    rows = rows[rows["dataset"].astype(str).eq(dataset)]
    assigned = set(assignments.loc[assignments["dataset"].astype(str).eq(dataset), "query_group_id"].astype(str))
    if missing := assigned - set(rows["query_group_id"].astype(str)):
        raise ValueError(f"Final-test query groups have no label rows: {dataset} {sorted(missing)[:5]}")
    queries = set(rows["query_signature_id"].astype(str))
    if unmapped := sorted(query for query in queries if query.startswith("visibleq:") and query not in origins):
        raise ValueError(f"Final-test query copies have no source record: {dataset} {unmapped[:5]}")
    return frozenset(origins.get(query, query) for query in queries)


def reviewed_error_records(snapshot_root: Path, dataset: str) -> frozenset[str]:
    """Return records whose ORCID the snapshot review marks as wrong."""
    rows = json.loads((snapshot_root / ORCID_ERROR_RECORDS).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(
        not isinstance(row, list) or len(row) != 2 or not all(isinstance(value, str) and value for value in row)
        for row in rows
    ):
        raise ValueError(f"{ORCID_ERROR_RECORDS} must contain [dataset, signature_id] rows")
    return frozenset(signature for name, signature in rows if name == dataset)


def read_reference(
    arrow_dataset: ArrowDataset,
    snapshot_root: Path,
    dataset: str,
    *,
    test_identities: frozenset[str],
    error_records: frozenset[str],
) -> OrcidReference:
    """Build ORCID references outside final-test identities.

    A final-test record excludes its complete ORCID group, so test identities do not
    select EPS. Reviewed wrong-ORCID records leave only their own row.
    """
    origins = _load_initial_query_sources(snapshot_root).get(dataset, {})
    clones = frozenset(origins)
    with arrow_dataset.use() as lease, lease.open_file("signatures") as infile:
        with pa.PythonFile(infile, mode="r") as source:
            table = ipc.open_file(source).read_all().select(["signature_id", "author_orcid"])
    rows = table.to_pydict()
    candidates: dict[str, str] = {}
    excluded_rows = 0
    excluded_errors = 0
    for signature_id, raw_orcid in zip(rows["signature_id"], rows["author_orcid"], strict=True):
        signature_id = str(signature_id)
        if signature_id.startswith("visibleq:") and signature_id not in clones:
            raise ValueError("Derived query copies require their source map in the snapshot")
        if signature_id in clones or raw_orcid is None:
            continue
        orcid = normalize_orcid(raw_orcid)
        if orcid in EXCLUDED_ORCID_GROUPS:
            excluded_rows += 1
        elif signature_id in error_records:
            excluded_errors += bool(orcid)
        elif orcid:
            candidates[signature_id] = orcid
    test_groups = {candidates[signature] for signature in test_identities if signature in candidates}
    with_singletons = {signature: orcid for signature, orcid in candidates.items() if orcid not in test_groups}
    counts = Counter(with_singletons.values())
    repeated = {signature: orcid for signature, orcid in with_singletons.items() if counts[orcid] >= 2}
    if overlap := (set(with_singletons) | set(repeated)) & test_identities:
        raise ValueError(f"EPS reference contains final-test records: {dataset} {sorted(overlap)[:5]}")
    if not repeated:
        raise ValueError(f"No repeated ORCID references for dataset={dataset}")
    return OrcidReference(
        repeated=repeated,
        with_singletons=with_singletons,
        excluded_copies=clones,
        stats={
            "reference_kind": "repeated_orcid_identity",
            "reference_policy": REFERENCE_POLICY,
            "reference_signatures": len(repeated),
            "reference_groups": len(set(repeated.values())),
            "singleton_inclusive_signatures": len(with_singletons),
            "singleton_inclusive_groups": len(counts),
            "final_test_identities": len(test_identities),
            "final_test_reference_overlap": 0,
            "excluded_final_test_orcid_groups": len(test_groups),
            "excluded_final_test_group_signatures": len(candidates) - len(with_singletons),
            "excluded_final_test_signature_fraction": (len(candidates) - len(with_singletons)) / len(candidates),
            "excluded_reviewed_error_records": excluded_errors,
            "excluded_initial_query_copies": len(clones),
            "excluded_mixed_orcid_rows": excluded_rows,
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

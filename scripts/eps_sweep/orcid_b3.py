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
from s2and.incremental_linking_training.classic import (
    _load_classic_stratified_eval_rows,
    _promoted_stratified_gate_spec,
    load_bundle,
)
from s2and.text import normalize_orcid
from scripts.production.counts.generate_orcid_name_prefix_counts import EXCLUDED_ORCID_GROUPS
from scripts.production.data.block_roles import BlockRoles, in_orcid_home_role, normalize_block_key
from scripts.production.model.initial_query_sources import (
    MAP_SHA256_FIELD,
    InitialQuerySources,
    load_initial_query_sources,
)

# Selection references exclude final-test identities and reviewed ORCID errors.
REFERENCE_POLICY = "final_test_orcid_groups_and_reviewed_error_records_excluded"
REFERENCE_POLICY_ALL_IDENTITIES = "all_reference_identities_block_disjoint"
BLOCK_ROLE_SHA256_FIELD = "block_role_assignments_sha256"
ORCID_ERROR_RECORDS = "orcid_error_query_signatures.json"


def load_external_reference_labels(path: Path, *, dataset: str | None = None) -> dict[str, str]:
    """Load verified external (signature_id, orcid) reference labels from parquet.

    If a ``dataset`` column is present and ``dataset`` is given, filters to rows matching
    that dataset (preventing signature-ID collisions across multi-block label files).
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"signature_id", "orcid"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"External reference labels missing required columns: {missing}")
    if dataset is not None and "dataset" in frame.columns:
        frame = frame[frame["dataset"].astype(str).eq(str(dataset))]
    if "scored" in frame:
        frame = frame[frame["scored"].eq(True).fillna(False)]
    labels: dict[str, str] = {}
    for signature_id, raw_orcid in zip(frame["signature_id"], frame["orcid"], strict=True):
        if signature_id is None or raw_orcid is None or pd.isna(signature_id) or pd.isna(raw_orcid):
            continue
        sid = str(signature_id)
        orcid = normalize_orcid(str(raw_orcid))
        if not sid or not orcid:
            continue
        existing = labels.get(sid)
        if existing is not None and existing != orcid:
            raise ValueError(f"Conflicting external ORCID labels for signature_id={sid!r}: {existing} vs {orcid}")
        labels[sid] = orcid
    return labels


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


def final_test_query_identities(
    snapshot_root: Path, dataset: str, query_sources: InitialQuerySources | None = None
) -> frozenset[str]:
    """Return the dataset's frozen final-test query records from a snapshot.

    Args:
        snapshot_root: Snapshot with the linker bundle and its frozen split assignments.
        dataset: Incremental dataset name.
        query_sources: Validated source map from the caller, if available.

    Returns:
        Source signature IDs of every final-test query.
    """
    if query_sources is None:
        query_sources = load_initial_query_sources(snapshot_root, dataset_names=(dataset,))
    bundle = load_bundle(snapshot_root)
    spec = dict(bundle.models["classic"])
    gate = _promoted_stratified_gate_spec(spec)
    if gate is None:
        raise ValueError("classic.promoted_stratified_gate is required")
    split_spec = dict(spec["stratified_eval_test_split"])
    rows, _assignments = _load_classic_stratified_eval_rows(
        bundle,
        spec,
        split_spec,
        splits=(str(gate["test_split"]),),
        include_calibration_source=False,
    )
    return resolve_test_query_identities(rows, query_sources.origins.get(dataset, {}), dataset)


def resolve_test_query_identities(rows: pd.DataFrame, origins: Mapping[str, str], dataset: str) -> frozenset[str]:
    """Resolve assigned test query groups to source records.

    Args:
        rows: Label rows of the test split, with query group and query signature IDs.
        origins: Initial-only query copies mapped to their source records.
        dataset: Incremental dataset name.

    Returns:
        Source signature IDs of every assigned test query.

    Raises:
        ValueError: If a copy has no source record.
    """
    rows = rows[rows["dataset"].astype(str).eq(dataset)]
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
    query_sources: InitialQuerySources | None = None,
    external_labels: Mapping[str, str] | None = None,
    allow_all_identities: bool = False,
    block_roles: BlockRoles | None = None,
) -> OrcidReference:
    """Build ORCID references outside final-test identities (or all identities for block-disjoint pools).

    A final-test record excludes its complete ORCID group, so test identities do not
    select EPS. Reviewed wrong-ORCID records leave only their own row. When
    ``external_labels`` is supplied, verified ``(signature_id, orcid)`` pairs are read
    from that mapping instead of the Arrow ``author_orcid`` column.
    """
    if allow_all_identities and test_identities:
        raise ValueError("allow_all_identities cannot be set when test_identities is non-empty")
    role = block_roles.role_for(dataset) if block_roles is not None else None
    if query_sources is None:
        query_sources = load_initial_query_sources(snapshot_root, arrow_datasets={dataset: arrow_dataset})
    origins = query_sources.origins.get(dataset, {})
    clones = frozenset(origins)
    with arrow_dataset.use() as lease, lease.open_file("signatures") as infile:
        with pa.PythonFile(infile, mode="r") as source:
            reader = ipc.open_file(source)
            has_arrow_orcid = "author_orcid" in reader.schema.names
            columns = ["signature_id", "author_orcid"] if has_arrow_orcid else ["signature_id"]
            if block_roles is not None:
                columns.append("author_block")
            table = reader.read_all().select(columns)
    rows = table.to_pydict()
    sig_ids = [str(sid) for sid in rows["signature_id"]]
    arrow_orcids = rows["author_orcid"] if has_arrow_orcid else [None] * len(sig_ids)
    candidates: dict[str, str] = {}
    arrow_identity_map: dict[str, str] = {}
    excluded_rows = 0
    excluded_errors = 0
    excluded_non_home = 0
    matched_external = 0
    for index, (signature_id, raw_arrow_orcid) in enumerate(zip(sig_ids, arrow_orcids, strict=True)):
        if signature_id.startswith("visibleq:") and signature_id not in clones:
            raise ValueError("Derived query copies require their source map in the snapshot")
        if signature_id in clones:
            continue
        if block_roles is not None and normalize_block_key(rows["author_block"][index]) != normalize_block_key(dataset):
            raise ValueError(f"ORCID reference for {dataset!r} contains a signature from another block: {signature_id}")
        if raw_arrow_orcid is not None:
            norm_arrow = normalize_orcid(raw_arrow_orcid)
            if norm_arrow and norm_arrow not in EXCLUDED_ORCID_GROUPS:
                arrow_identity_map[signature_id] = norm_arrow
        raw_orcid = external_labels.get(signature_id) if external_labels is not None else raw_arrow_orcid
        if raw_orcid is None:
            continue
        if external_labels is not None:
            matched_external += 1
        orcid = normalize_orcid(raw_orcid)
        if orcid in EXCLUDED_ORCID_GROUPS:
            excluded_rows += 1
        elif signature_id in error_records:
            excluded_errors += bool(orcid)
        elif orcid:
            if (
                block_roles is not None
                and role is not None
                and not in_orcid_home_role(orcid, role, block_roles.orcid_home_role)
            ):
                excluded_non_home += 1
            else:
                candidates[signature_id] = orcid
    # Resolve final-test identities from BOTH external labels AND Arrow author_orcid so an
    # unlabeled final-test record sharing an Arrow ORCID with dev records cannot leak its author group.
    if allow_all_identities:
        test_groups: set[str] = set()
    else:
        test_groups = set()
        for signature in test_identities:
            if signature in candidates:
                test_groups.add(candidates[signature])
            if signature in arrow_identity_map:
                test_groups.add(arrow_identity_map[signature])
            if external_labels is not None and signature in external_labels:
                ext_orcid = normalize_orcid(external_labels[signature])
                if ext_orcid and ext_orcid not in EXCLUDED_ORCID_GROUPS:
                    test_groups.add(ext_orcid)
    with_singletons = {signature: orcid for signature, orcid in candidates.items() if orcid not in test_groups}
    counts = Counter(with_singletons.values())
    repeated = {signature: orcid for signature, orcid in with_singletons.items() if counts[orcid] >= 2}
    if overlap := (set(with_singletons) | set(repeated)) & test_identities:
        raise ValueError(f"EPS reference contains final-test records: {dataset} {sorted(overlap)[:5]}")
    if not repeated:
        raise ValueError(f"No repeated ORCID references for dataset={dataset}")
    excluded_fraction = (len(candidates) - len(with_singletons)) / len(candidates) if candidates else 0.0
    extra_external_stats: dict[str, Any] = {}
    if external_labels is not None:
        extra_external_stats = {
            "external_labels_total": len(external_labels),
            "external_labels_matched": matched_external,
            "external_labels_unmatched": max(0, len(external_labels) - matched_external),
        }
    return OrcidReference(
        repeated=repeated,
        with_singletons=with_singletons,
        excluded_copies=clones,
        stats={
            "reference_kind": "repeated_orcid_identity",
            "reference_policy": REFERENCE_POLICY_ALL_IDENTITIES if allow_all_identities else REFERENCE_POLICY,
            "reference_source": "external_labels" if external_labels is not None else "arrow_author_orcid",
            **extra_external_stats,
            **({BLOCK_ROLE_SHA256_FIELD: block_roles.sha256} if block_roles is not None else {}),
            MAP_SHA256_FIELD: query_sources.sha256,
            "reference_signatures": len(repeated),
            "reference_groups": len(set(repeated.values())),
            "singleton_inclusive_signatures": len(with_singletons),
            "singleton_inclusive_groups": len(counts),
            "final_test_identities": len(test_identities),
            "final_test_reference_overlap": 0,
            "excluded_final_test_orcid_groups": len(test_groups),
            "excluded_final_test_group_signatures": len(candidates) - len(with_singletons),
            "excluded_final_test_signature_fraction": excluded_fraction,
            "excluded_reviewed_error_records": excluded_errors,
            "excluded_non_home_role_records": excluded_non_home,
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

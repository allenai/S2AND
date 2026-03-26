"""Build dual single-letter target artifacts against cached multi-letter seeds.

This runner consumes the persisted step-2 multi-letter artifacts and builds the
step-3 dual target set for retrieval evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from s2and.model import _bump_cluster_seeds_version, _signature_first_for_rules, _sync_rust_cluster_seeds
from s2and.text import ORCID_PATTERN

try:
    from scripts.giant_block_cluster_retrieval_task import (
        DEFAULT_MODEL_PATH,
        DEFAULT_TOTAL_RAM_BYTES,
        _read_json,
        _required_file,
        _write_json,
        load_clusterer,
        load_dataset,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from giant_block_cluster_retrieval_task import (  # type: ignore
        DEFAULT_MODEL_PATH,
        DEFAULT_TOTAL_RAM_BYTES,
        _read_json,
        _required_file,
        _write_json,
        load_clusterer,
        load_dataset,
    )

DEFAULT_JOINT_BATCHING_THRESHOLD = 64
DEFAULT_MAX_DISAGREEMENT_EXAMPLES = 25


def _build_per_query_progress_identity(
    *,
    data_dir: Path,
    step2_dir: Path,
    model_path: Path,
    target_block: str,
    total_ram_bytes: int | None,
    query_ids: list[str],
) -> dict[str, Any]:
    """Build the strict identity used to validate resumed per-query progress."""

    return {
        "data_dir": str(Path(data_dir).resolve()),
        "step2_dir": str(Path(step2_dir).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "target_block": str(target_block),
        "total_ram_bytes": int(total_ram_bytes) if total_ram_bytes is not None else None,
        "query_ids": [str(query_id) for query_id in query_ids],
    }


def _normalize_orcid(orcid: str | None) -> str | None:
    """Normalize ORCID to the compact uppercase 16-character form."""

    if not orcid:
        return None
    matches = ORCID_PATTERN.findall(orcid)
    if not matches:
        return None
    return matches[0].upper().replace("-", "")


def _extract_signature_orcid(signature_payload: dict[str, Any]) -> str | None:
    """Return the normalized ORCID for a raw extracted signature payload."""

    author_info = signature_payload.get("author_info", {})
    if str(author_info.get("source_id_source", "")) != "ORCID":
        return None
    source_ids = author_info.get("source_ids") or []
    if len(source_ids) == 0:
        return None
    return _normalize_orcid(str(source_ids[0]))


def _invert_subblocks(subblocks: dict[str, list[str]]) -> dict[str, str]:
    """Invert a subblock manifest to a signature -> subblock mapping."""

    signature_to_subblock: dict[str, str] = {}
    for subblock_key, signature_ids in subblocks.items():
        for signature_id in signature_ids:
            signature_to_subblock[str(signature_id)] = str(subblock_key)
    return signature_to_subblock


def _orcid_group_size_bucket(orcid_group_size: int) -> str:
    """Bucket ORCID group sizes for disagreement slicing."""

    if orcid_group_size <= 2:
        return "2"
    if orcid_group_size == 3:
        return "3"
    return "4+"


def _subblock_type(subblock_key: str) -> str:
    """Derive a coarse subblock category from the manifest key."""

    if "|specter=" in subblock_key:
        return "specter"
    if "|middle=" in subblock_key:
        return "middle_split"
    return "initial_only"


def _estimate_next_cluster_id_start(cluster_ids: set[str]) -> int:
    """Return a safe numeric starting point for new incremental cluster IDs."""

    numeric_ids: list[int] = []
    for cluster_id in cluster_ids:
        try:
            numeric_ids.append(int(cluster_id))
        except (TypeError, ValueError):
            continue
    if numeric_ids:
        return max(numeric_ids) + 1
    return len(cluster_ids)


def _select_query_ids(
    eligible_query_ids: list[str],
    *,
    limit_queries: int | None,
    random_seed: int,
) -> list[str]:
    """Select a deterministic query subset for pilot runs."""

    if limit_queries is None or limit_queries >= len(eligible_query_ids):
        return list(eligible_query_ids)
    rng = random.Random(int(random_seed))
    return sorted(rng.sample(eligible_query_ids, int(limit_queries)))


def _build_query_metadata(
    raw_signatures: dict[str, Any],
    dataset: Any,
    *,
    target_block: str,
    subblock_manifest: dict[str, Any],
    seed_signature_ids: set[str],
    limit_queries: int | None,
    random_seed: int,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Build the repeated-ORCID single-letter query set and metadata."""

    grouped_orcid_ids: dict[str, list[str]] = defaultdict(list)
    for signature_id, signature_payload in raw_signatures.items():
        signature_block = str(signature_payload.get("author_info", {}).get("block", ""))
        if signature_block != target_block:
            continue
        normalized_orcid = _extract_signature_orcid(signature_payload)
        if normalized_orcid is not None:
            grouped_orcid_ids[normalized_orcid].append(str(signature_id))

    repeated_orcid_groups = {
        normalized_orcid: sorted(signature_ids)
        for normalized_orcid, signature_ids in grouped_orcid_ids.items()
        if len(signature_ids) > 1
    }
    repeated_orcid_signature_ids = {
        signature_id for signature_ids in repeated_orcid_groups.values() for signature_id in signature_ids
    }
    signature_to_orcid = {
        signature_id: normalized_orcid
        for normalized_orcid, signature_ids in repeated_orcid_groups.items()
        for signature_id in signature_ids
    }

    single_letter_ids = sorted(
        signature_id
        for signature_id, signature in dataset.signatures.items()
        if len(_signature_first_for_rules(signature)) <= 1
    )
    eligible_before_seed_overlap = [
        signature_id for signature_id in single_letter_ids if signature_id in repeated_orcid_signature_ids
    ]
    seed_overlap_query_ids = [
        signature_id for signature_id in eligible_before_seed_overlap if signature_id in seed_signature_ids
    ]
    eligible_query_ids = [
        signature_id for signature_id in eligible_before_seed_overlap if signature_id not in seed_signature_ids
    ]
    selected_query_ids = _select_query_ids(
        eligible_query_ids,
        limit_queries=limit_queries,
        random_seed=random_seed,
    )

    signature_to_subblock = _invert_subblocks(dict(subblock_manifest["subblocks"]))
    metadata_by_query: dict[str, dict[str, Any]] = {}
    for query_id in selected_query_ids:
        if query_id not in signature_to_subblock:
            raise RuntimeError(f"Query signature {query_id!r} missing from subblock manifest")
        signature = dataset.signatures[query_id]
        normalized_orcid = signature_to_orcid[query_id]
        middle_normalized = (
            signature.author_info_middle_normalized_without_apostrophe or signature.author_info_middle or ""
        )
        subblock_key = signature_to_subblock[query_id]
        metadata_by_query[query_id] = {
            "query_id": str(query_id),
            "normalized_orcid": normalized_orcid,
            "orcid_group_size": len(repeated_orcid_groups[normalized_orcid]),
            "orcid_group_size_bucket": _orcid_group_size_bucket(len(repeated_orcid_groups[normalized_orcid])),
            "first_name": signature.author_info_first or "",
            "first_name_normalized": _signature_first_for_rules(signature),
            "middle_name": signature.author_info_middle or "",
            "middle_name_normalized": middle_normalized,
            "has_middle_name": bool(middle_normalized),
            "query_subblock_key": subblock_key,
            "query_subblock_type": _subblock_type(subblock_key),
            "query_in_specter_subblock": "|specter=" in subblock_key,
        }

    query_set_payload = {
        "target_block": str(target_block),
        "repeated_orcid_group_count": len(repeated_orcid_groups),
        "repeated_orcid_signature_count": len(repeated_orcid_signature_ids),
        "eligible_query_count_before_seed_overlap_filter": len(eligible_before_seed_overlap),
        "seed_overlap_filtered_query_count": len(seed_overlap_query_ids),
        "eligible_query_count": len(eligible_query_ids),
        "selected_query_count": len(selected_query_ids),
        "selection": {
            "limit_queries": int(limit_queries) if limit_queries is not None else None,
            "random_seed": int(random_seed),
        },
        "query_ids": list(selected_query_ids),
        "query_rows": [metadata_by_query[query_id] for query_id in selected_query_ids],
    }
    return selected_query_ids, metadata_by_query, query_set_payload


def _install_cached_seed_map(dataset: Any, signature_to_cluster_id: dict[str, str]) -> dict[str, Any]:
    """Replace the dataset seed state with the cached multi-letter clusters."""

    normalized_seed_map = {
        str(signature_id): str(cluster_id) for signature_id, cluster_id in signature_to_cluster_id.items()
    }
    unique_cluster_ids = set(normalized_seed_map.values())
    dataset.cluster_seeds_require = dict(sorted(normalized_seed_map.items()))
    dataset.cluster_seeds_disallow = set()
    dataset.altered_cluster_signatures = []
    dataset.max_seed_cluster_id = _estimate_next_cluster_id_start(unique_cluster_ids)
    _bump_cluster_seeds_version(dataset)
    _sync_rust_cluster_seeds(dataset, use_cache=False)
    return {
        "seed_signature_count": len(normalized_seed_map),
        "seed_cluster_count": len(unique_cluster_ids),
        "max_seed_cluster_id": int(dataset.max_seed_cluster_id or 0),
    }


def _extract_query_targets(
    clusters: dict[str, list[str]],
    *,
    query_ids: list[str],
    metadata_by_query: dict[str, dict[str, Any]],
    seed_signature_ids: set[str],
    existing_seed_cluster_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Extract per-query target metadata from an incremental clustering result."""

    normalized_clusters = {
        str(cluster_id): [str(signature_id) for signature_id in members] for cluster_id, members in clusters.items()
    }
    query_id_set = set(query_ids)
    query_to_cluster_id: dict[str, str] = {}
    for cluster_id, members in normalized_clusters.items():
        for signature_id in members:
            if signature_id in query_id_set:
                if signature_id in query_to_cluster_id:
                    raise RuntimeError(f"Query signature {signature_id!r} appeared in more than one cluster")
                query_to_cluster_id[signature_id] = cluster_id

    missing_query_ids = [query_id for query_id in query_ids if query_id not in query_to_cluster_id]
    if missing_query_ids:
        raise RuntimeError(f"Missing query targets for {len(missing_query_ids)} signatures: {missing_query_ids[:5]}")

    targets_by_query: dict[str, dict[str, Any]] = {}
    for query_id in query_ids:
        cluster_id = query_to_cluster_id[query_id]
        members = normalized_clusters[cluster_id]
        query_members = sorted(signature_id for signature_id in members if signature_id in query_id_set)
        seed_member_count = sum(1 for signature_id in members if signature_id in seed_signature_ids)
        target_payload = dict(metadata_by_query[query_id])
        target_payload.update(
            {
                "target_cluster_id": cluster_id,
                "target_cluster_size": len(members),
                "target_is_existing_seed_cluster": cluster_id in existing_seed_cluster_ids,
                "target_seed_member_count": int(seed_member_count),
                "target_query_member_count": len(query_members),
                "target_query_member_ids": query_members,
                "target_other_query_ids": [signature_id for signature_id in query_members if signature_id != query_id],
            }
        )
        targets_by_query[query_id] = target_payload
    return targets_by_query


def _run_joint_targets(
    *,
    clusterer: Any,
    dataset: Any,
    query_ids: list[str],
    metadata_by_query: dict[str, dict[str, Any]],
    joint_batching_threshold: int,
    total_ram_bytes: int | None,
    seed_signature_ids: set[str],
    existing_seed_cluster_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the joint single-letter target build."""

    start = time.perf_counter()
    result = clusterer.predict_incremental(
        query_ids,
        dataset,
        batching_threshold=int(joint_batching_threshold),
        total_ram_bytes=total_ram_bytes,
    )
    elapsed = time.perf_counter() - start
    clusters_payload = result.get("clusters")
    if not isinstance(clusters_payload, dict):
        raise RuntimeError(
            "predict_incremental returned invalid joint clusters payload; "
            f"expected dict got {type(clusters_payload).__name__}"
        )
    targets_by_query = _extract_query_targets(
        clusters_payload,
        query_ids=query_ids,
        metadata_by_query=metadata_by_query,
        seed_signature_ids=seed_signature_ids,
        existing_seed_cluster_ids=existing_seed_cluster_ids,
    )
    summary = {
        "query_count": len(query_ids),
        "joint_batching_threshold": int(joint_batching_threshold),
        "elapsed_seconds": round(elapsed, 6),
        "phase_b_mode": str(result["phase_b_mode"]),
        "phase_b_budget_bytes": int(result["phase_b_budget_bytes"]),
        "phase_b_required_bytes": int(result["phase_b_required_bytes"]),
        "phase_a_accumulator_overflow_early_stop": bool(result["phase_a_accumulator_overflow_early_stop"]),
        "phase_a_adaptive_halvings_max": int(result["phase_a_adaptive_halvings_max"]),
    }
    return targets_by_query, summary


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a JSON line to `path`."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_existing_per_query_progress(
    progress_path: Path,
    *,
    progress_identity_path: Path,
    expected_progress_identity: dict[str, Any],
    query_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Load any existing per-query progress rows for resume support."""

    if not progress_path.exists():
        return {}
    if not progress_identity_path.exists():
        raise RuntimeError(
            "Cannot resume per-query progress from "
            f"{progress_path} without identity metadata at {progress_identity_path}"
        )

    progress_identity = _read_json(progress_identity_path)
    if progress_identity != expected_progress_identity:
        raise RuntimeError(
            f"Per-query progress at {progress_path} does not match the current run identity; "
            "clear the progress artifacts before resuming."
        )

    expected_query_ids = set(query_ids)
    targets_by_query: dict[str, dict[str, Any]] = {}
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        query_id = str(payload["query_id"])
        if query_id not in expected_query_ids:
            continue
        if query_id in targets_by_query:
            raise RuntimeError(f"Duplicate per-query progress row for query_id={query_id!r}")
        targets_by_query[query_id] = payload
    return targets_by_query


def _run_per_query_targets(
    *,
    clusterer: Any,
    dataset: Any,
    query_ids: list[str],
    metadata_by_query: dict[str, dict[str, Any]],
    total_ram_bytes: int | None,
    seed_signature_ids: set[str],
    existing_seed_cluster_ids: set[str],
    progress_path: Path,
    progress_identity_path: Path,
    progress_identity: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the per-query target build and persist progress rows."""

    if progress_path.exists():
        targets_by_query = _load_existing_per_query_progress(
            progress_path,
            progress_identity_path=progress_identity_path,
            expected_progress_identity=progress_identity,
            query_ids=query_ids,
        )
    else:
        _write_json(progress_identity_path, progress_identity)
        targets_by_query = {}
    resumed_query_count = len(targets_by_query)
    phase_b_mode_counts: Counter[str] = Counter()
    overflow_count = 0
    elapsed_seconds_by_query: list[float] = []

    for target_payload in targets_by_query.values():
        phase_b_mode_counts[str(target_payload["phase_b_mode"])] += 1
        overflow_count += int(bool(target_payload["phase_a_accumulator_overflow_early_stop"]))
        elapsed_seconds_by_query.append(float(target_payload["per_query_elapsed_seconds"]))

    for query_index, query_id in enumerate(query_ids, start=1):
        if query_id in targets_by_query:
            continue
        start = time.perf_counter()
        result = clusterer.predict_incremental(
            [query_id],
            dataset,
            batching_threshold=None,
            total_ram_bytes=total_ram_bytes,
        )
        elapsed = time.perf_counter() - start
        clusters_payload = result.get("clusters")
        if not isinstance(clusters_payload, dict):
            raise RuntimeError(
                "predict_incremental returned invalid per-query clusters payload; "
                f"expected dict got {type(clusters_payload).__name__}"
            )
        target_payload = _extract_query_targets(
            clusters_payload,
            query_ids=[query_id],
            metadata_by_query=metadata_by_query,
            seed_signature_ids=seed_signature_ids,
            existing_seed_cluster_ids=existing_seed_cluster_ids,
        )[query_id]
        target_payload.update(
            {
                "per_query_index": int(query_index),
                "per_query_elapsed_seconds": round(elapsed, 6),
                "phase_b_mode": str(result["phase_b_mode"]),
                "phase_b_budget_bytes": int(result["phase_b_budget_bytes"]),
                "phase_b_required_bytes": int(result["phase_b_required_bytes"]),
                "phase_a_accumulator_overflow_early_stop": bool(result["phase_a_accumulator_overflow_early_stop"]),
                "phase_a_adaptive_halvings_max": int(result["phase_a_adaptive_halvings_max"]),
            }
        )
        targets_by_query[query_id] = target_payload
        elapsed_seconds_by_query.append(float(elapsed))
        phase_b_mode_counts[str(result["phase_b_mode"])] += 1
        overflow_count += int(bool(result["phase_a_accumulator_overflow_early_stop"]))
        _append_jsonl(progress_path, target_payload)

    total_elapsed_seconds = float(sum(elapsed_seconds_by_query))
    summary = {
        "query_count": len(query_ids),
        "elapsed_seconds_total": round(total_elapsed_seconds, 6),
        "elapsed_seconds_mean": round(total_elapsed_seconds / max(1, len(query_ids)), 6),
        "elapsed_seconds_max": round(max(elapsed_seconds_by_query) if elapsed_seconds_by_query else 0.0, 6),
        "resumed_query_count": int(resumed_query_count),
        "phase_b_mode_counts": dict(sorted(phase_b_mode_counts.items())),
        "phase_a_accumulator_overflow_count": int(overflow_count),
        "progress_path": str(progress_path.resolve()),
    }
    return targets_by_query, summary


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize disagreement rows for a slice."""

    query_count = len(rows)
    disagreement_count = sum(int(row["targets_disagree"]) for row in rows)
    joint_existing_seed_count = sum(int(row["joint_target_is_existing_seed_cluster"]) for row in rows)
    per_query_existing_seed_count = sum(int(row["per_query_target_is_existing_seed_cluster"]) for row in rows)
    return {
        "query_count": int(query_count),
        "disagreement_count": int(disagreement_count),
        "disagreement_fraction": float(disagreement_count / query_count) if query_count > 0 else 0.0,
        "joint_existing_seed_target_count": int(joint_existing_seed_count),
        "joint_new_cluster_target_count": int(query_count - joint_existing_seed_count),
        "per_query_existing_seed_target_count": int(per_query_existing_seed_count),
        "per_query_new_cluster_target_count": int(query_count - per_query_existing_seed_count),
    }


def _slice_rows(
    rows: list[dict[str, Any]],
    *,
    key_name: str,
) -> dict[str, Any]:
    """Group disagreement rows by a metadata key."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key_name])].append(row)
    return {bucket: _aggregate_rows(grouped[bucket]) for bucket in sorted(grouped)}


def _transition_label(joint_target: dict[str, Any], per_query_target: dict[str, Any]) -> str:
    """Label how the joint and per-query targets differ."""

    if joint_target["target_cluster_id"] == per_query_target["target_cluster_id"]:
        return "same_cluster"
    if joint_target["target_is_existing_seed_cluster"] and per_query_target["target_is_existing_seed_cluster"]:
        return "different_existing_seed_clusters"
    if joint_target["target_is_existing_seed_cluster"] and not per_query_target["target_is_existing_seed_cluster"]:
        return "joint_existing_seed_per_query_new_cluster"
    if not joint_target["target_is_existing_seed_cluster"] and per_query_target["target_is_existing_seed_cluster"]:
        return "joint_new_cluster_per_query_existing_seed"
    return "different_new_clusters"


def _build_disagreement_report(
    *,
    query_ids: list[str],
    joint_targets: dict[str, dict[str, Any]],
    per_query_targets: dict[str, dict[str, Any]],
    joint_summary: dict[str, Any],
    per_query_summary: dict[str, Any],
    max_examples: int,
) -> dict[str, Any]:
    """Build the informative dual-target disagreement report."""

    comparison_rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        joint_target = joint_targets[query_id]
        per_query_target = per_query_targets[query_id]
        comparison_rows.append(
            {
                "query_id": query_id,
                "normalized_orcid": joint_target["normalized_orcid"],
                "orcid_group_size": int(joint_target["orcid_group_size"]),
                "orcid_group_size_bucket": str(joint_target["orcid_group_size_bucket"]),
                "first_name_normalized": joint_target["first_name_normalized"],
                "middle_name_normalized": joint_target["middle_name_normalized"],
                "has_middle_name": bool(joint_target["has_middle_name"]),
                "query_subblock_key": joint_target["query_subblock_key"],
                "query_subblock_type": joint_target["query_subblock_type"],
                "query_in_specter_subblock": bool(joint_target["query_in_specter_subblock"]),
                "joint_target_cluster_id": joint_target["target_cluster_id"],
                "joint_target_cluster_size": int(joint_target["target_cluster_size"]),
                "joint_target_is_existing_seed_cluster": bool(joint_target["target_is_existing_seed_cluster"]),
                "joint_target_query_member_ids": list(joint_target["target_query_member_ids"]),
                "per_query_target_cluster_id": per_query_target["target_cluster_id"],
                "per_query_target_cluster_size": int(per_query_target["target_cluster_size"]),
                "per_query_target_is_existing_seed_cluster": bool(per_query_target["target_is_existing_seed_cluster"]),
                "per_query_target_query_member_ids": list(per_query_target["target_query_member_ids"]),
                "targets_disagree": joint_target["target_cluster_id"] != per_query_target["target_cluster_id"],
                "transition_label": _transition_label(joint_target, per_query_target),
            }
        )

    overall = _aggregate_rows(comparison_rows)
    transition_counts = Counter(row["transition_label"] for row in comparison_rows)
    disagreement_examples = [
        {
            "query_id": row["query_id"],
            "normalized_orcid": row["normalized_orcid"],
            "orcid_group_size": row["orcid_group_size"],
            "first_name_normalized": row["first_name_normalized"],
            "middle_name_normalized": row["middle_name_normalized"],
            "query_subblock_key": row["query_subblock_key"],
            "query_subblock_type": row["query_subblock_type"],
            "query_in_specter_subblock": row["query_in_specter_subblock"],
            "joint_target_cluster_id": row["joint_target_cluster_id"],
            "joint_target_cluster_size": row["joint_target_cluster_size"],
            "joint_target_is_existing_seed_cluster": row["joint_target_is_existing_seed_cluster"],
            "joint_target_query_member_ids": row["joint_target_query_member_ids"],
            "per_query_target_cluster_id": row["per_query_target_cluster_id"],
            "per_query_target_cluster_size": row["per_query_target_cluster_size"],
            "per_query_target_is_existing_seed_cluster": row["per_query_target_is_existing_seed_cluster"],
            "per_query_target_query_member_ids": row["per_query_target_query_member_ids"],
            "transition_label": row["transition_label"],
        }
        for row in comparison_rows
        if row["targets_disagree"]
    ][: int(max_examples)]

    joint_primary_reference_valid = bool(
        joint_summary["phase_b_mode"] == "exact" and not joint_summary["phase_a_accumulator_overflow_early_stop"]
    )
    return {
        **overall,
        "joint_primary_reference_valid": joint_primary_reference_valid,
        "joint_primary_reference_gate": {
            "phase_b_mode": joint_summary["phase_b_mode"],
            "phase_a_accumulator_overflow_early_stop": joint_summary["phase_a_accumulator_overflow_early_stop"],
        },
        "joint_run_summary": joint_summary,
        "per_query_run_summary": per_query_summary,
        "transition_counts": dict(sorted(transition_counts.items())),
        "slices": {
            "query_subblock_type": _slice_rows(comparison_rows, key_name="query_subblock_type"),
            "query_in_specter_subblock": _slice_rows(comparison_rows, key_name="query_in_specter_subblock"),
            "has_middle_name": _slice_rows(comparison_rows, key_name="has_middle_name"),
            "orcid_group_size_bucket": _slice_rows(comparison_rows, key_name="orcid_group_size_bucket"),
            "joint_target_is_existing_seed_cluster": _slice_rows(
                comparison_rows,
                key_name="joint_target_is_existing_seed_cluster",
            ),
            "per_query_target_is_existing_seed_cluster": _slice_rows(
                comparison_rows,
                key_name="per_query_target_is_existing_seed_cluster",
            ),
        },
        "disagreement_examples": disagreement_examples,
    }


def _validate_step2_artifacts(
    *,
    data_dir: Path,
    expected_target_block: str,
    subblock_manifest: dict[str, Any],
    signature_to_cluster_id: dict[str, Any],
    dataset: Any,
) -> None:
    """Fail closed when step-3 inputs do not match the current step-2 artifacts."""

    if "data_dir" not in subblock_manifest:
        raise RuntimeError("Step-2 subblock_manifest.json is missing required data_dir metadata")
    manifest_data_dir = str(Path(subblock_manifest["data_dir"]).resolve())
    current_data_dir = str(Path(data_dir).resolve())
    if manifest_data_dir != current_data_dir:
        raise RuntimeError(
            f"Step-2 artifacts were built from data_dir={manifest_data_dir!r}, expected {current_data_dir!r}"
        )
    manifest_target_block = str(subblock_manifest.get("target_block", ""))
    if not manifest_target_block:
        raise RuntimeError("Step-2 subblock_manifest.json is missing required target_block metadata")
    if manifest_target_block != str(expected_target_block):
        raise RuntimeError(
            f"Step-2 artifacts target block {manifest_target_block!r} does not match current block "
            f"{expected_target_block!r}"
        )
    manifest_subblocks = subblock_manifest.get("subblocks")
    if not isinstance(manifest_subblocks, dict):
        raise RuntimeError("Step-2 subblock_manifest.json is missing the required subblocks mapping")

    manifest_signature_ids = {str(signature_id) for members in manifest_subblocks.values() for signature_id in members}
    dataset_signature_ids = {str(signature_id) for signature_id in dataset.signatures}
    seed_signature_ids = {str(signature_id) for signature_id in signature_to_cluster_id}
    unknown_seed_signature_ids = sorted(seed_signature_ids.difference(dataset_signature_ids))
    if unknown_seed_signature_ids:
        raise RuntimeError(
            "Step-2 signature_to_cluster_id.json contains signatures that are not present in the current dataset: "
            f"{unknown_seed_signature_ids[:10]}"
        )
    missing_from_manifest = sorted(seed_signature_ids.difference(manifest_signature_ids))
    if missing_from_manifest:
        raise RuntimeError(
            "Step-2 signature_to_cluster_id.json contains signatures that are missing from the current "
            f"subblock manifest: {missing_from_manifest[:10]}"
        )


def run_task(
    *,
    data_dir: Path,
    step2_dir: Path,
    output_dir: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    block_key: str | None = None,
    n_jobs: int = 20,
    total_ram_bytes: int | None = DEFAULT_TOTAL_RAM_BYTES,
    joint_batching_threshold: int = DEFAULT_JOINT_BATCHING_THRESHOLD,
    limit_queries: int | None = None,
    random_seed: int = 0,
    max_disagreement_examples: int = DEFAULT_MAX_DISAGREEMENT_EXAMPLES,
) -> dict[str, Any]:
    """Run the dual-target build and persist its artifacts."""

    if joint_batching_threshold <= 0:
        raise ValueError("joint_batching_threshold must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_signatures = _read_json(_required_file(Path(data_dir), "signatures.json"))
    subblock_manifest = _read_json(_required_file(Path(step2_dir), "subblock_manifest.json"))
    signature_to_cluster_id = _read_json(_required_file(Path(step2_dir), "signature_to_cluster_id.json"))

    model_start = time.perf_counter()
    clusterer = load_clusterer(Path(model_path), n_jobs=int(n_jobs))
    model_load_seconds = time.perf_counter() - model_start
    dataset_start = time.perf_counter()
    dataset, load_info = load_dataset(Path(data_dir), block_key=block_key, n_jobs=int(n_jobs), clusterer=clusterer)
    dataset_load_seconds = time.perf_counter() - dataset_start
    signature_to_cluster_id = {
        str(signature_id): str(cluster_id) for signature_id, cluster_id in dict(signature_to_cluster_id).items()
    }
    _validate_step2_artifacts(
        data_dir=Path(data_dir),
        expected_target_block=str(load_info["target_block"]),
        subblock_manifest=dict(subblock_manifest),
        signature_to_cluster_id=signature_to_cluster_id,
        dataset=dataset,
    )

    seed_install_summary = _install_cached_seed_map(dataset, dict(signature_to_cluster_id))
    seed_signature_ids = set(str(signature_id) for signature_id in signature_to_cluster_id)
    existing_seed_cluster_ids = set(str(cluster_id) for cluster_id in signature_to_cluster_id.values())

    query_ids, query_metadata_by_id, query_set_payload = _build_query_metadata(
        raw_signatures,
        dataset,
        target_block=str(load_info["target_block"]),
        subblock_manifest=subblock_manifest,
        seed_signature_ids=seed_signature_ids,
        limit_queries=limit_queries,
        random_seed=random_seed,
    )
    if len(query_ids) == 0:
        raise RuntimeError("Query selection produced zero repeated-ORCID single-letter signatures")

    joint_targets, joint_summary = _run_joint_targets(
        clusterer=clusterer,
        dataset=dataset,
        query_ids=query_ids,
        metadata_by_query=query_metadata_by_id,
        joint_batching_threshold=int(joint_batching_threshold),
        total_ram_bytes=total_ram_bytes,
        seed_signature_ids=seed_signature_ids,
        existing_seed_cluster_ids=existing_seed_cluster_ids,
    )
    per_query_progress_path = output_dir / "per_query_progress.jsonl"
    per_query_progress_identity_path = output_dir / "per_query_progress.meta.json"
    per_query_progress_identity = _build_per_query_progress_identity(
        data_dir=Path(data_dir),
        step2_dir=Path(step2_dir),
        model_path=Path(model_path),
        target_block=str(load_info["target_block"]),
        total_ram_bytes=total_ram_bytes,
        query_ids=query_ids,
    )
    per_query_targets, per_query_summary = _run_per_query_targets(
        clusterer=clusterer,
        dataset=dataset,
        query_ids=query_ids,
        metadata_by_query=query_metadata_by_id,
        total_ram_bytes=total_ram_bytes,
        seed_signature_ids=seed_signature_ids,
        existing_seed_cluster_ids=existing_seed_cluster_ids,
        progress_path=per_query_progress_path,
        progress_identity_path=per_query_progress_identity_path,
        progress_identity=per_query_progress_identity,
    )
    disagreement_report = _build_disagreement_report(
        query_ids=query_ids,
        joint_targets=joint_targets,
        per_query_targets=per_query_targets,
        joint_summary=joint_summary,
        per_query_summary=per_query_summary,
        max_examples=int(max_disagreement_examples),
    )

    query_set_payload.update(
        {
            "data_dir": str(Path(data_dir).resolve()),
            "step2_dir": str(Path(step2_dir).resolve()),
        }
    )
    _write_json(output_dir / "query_set.json", query_set_payload)
    joint_targets_payload = {
        "data_dir": str(Path(data_dir).resolve()),
        "step2_dir": str(Path(step2_dir).resolve()),
        "query_count": len(query_ids),
        "summary": joint_summary,
        "targets": joint_targets,
    }
    _write_json(output_dir / "joint_targets.json", joint_targets_payload)
    per_query_targets_payload = {
        "data_dir": str(Path(data_dir).resolve()),
        "step2_dir": str(Path(step2_dir).resolve()),
        "query_count": len(query_ids),
        "summary": per_query_summary,
        "targets": per_query_targets,
    }
    run_summary = {
        "data_dir": str(Path(data_dir).resolve()),
        "step2_dir": str(Path(step2_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "target_block": str(load_info["target_block"]),
        "total_ram_bytes": int(total_ram_bytes) if total_ram_bytes is not None else None,
        "joint_batching_threshold": int(joint_batching_threshold),
        "dataset_load_seconds": round(dataset_load_seconds, 6),
        "model_load_seconds": round(model_load_seconds, 6),
        "seed_install_summary": seed_install_summary,
        "query_count": len(query_ids),
        "joint_primary_reference_valid": bool(disagreement_report["joint_primary_reference_valid"]),
        "disagreement_count": int(disagreement_report["disagreement_count"]),
        "disagreement_fraction": float(disagreement_report["disagreement_fraction"]),
        "artifact_paths": {
            "query_set": str(output_dir / "query_set.json"),
            "joint_targets": str(output_dir / "joint_targets.json"),
            "per_query_targets": str(output_dir / "per_query_targets.json"),
            "per_query_progress": str(per_query_progress_path),
            "per_query_progress_identity": str(per_query_progress_identity_path),
            "target_disagreement_report": str(output_dir / "target_disagreement_report.json"),
            "run_summary": str(output_dir / "run_summary.json"),
        },
    }

    _write_json(output_dir / "per_query_targets.json", per_query_targets_payload)
    _write_json(output_dir / "target_disagreement_report.json", disagreement_report)
    _write_json(output_dir / "run_summary.json", run_summary)
    return run_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing extracted giant-block files.",
    )
    parser.add_argument(
        "--step2-dir",
        type=Path,
        required=True,
        help="Directory containing cached step-2 artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for persisted step-3 artifacts.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--block-key", type=str, default=None, help="Optional target block override.")
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--total-ram-bytes", type=int, default=DEFAULT_TOTAL_RAM_BYTES)
    parser.add_argument("--joint-batching-threshold", type=int, default=DEFAULT_JOINT_BATCHING_THRESHOLD)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-disagreement-examples", type=int, default=DEFAULT_MAX_DISAGREEMENT_EXAMPLES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    summary = run_task(
        data_dir=args.data_dir,
        step2_dir=args.step2_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        block_key=args.block_key,
        n_jobs=int(args.n_jobs),
        total_ram_bytes=int(args.total_ram_bytes) if args.total_ram_bytes is not None else None,
        joint_batching_threshold=int(args.joint_batching_threshold),
        limit_queries=int(args.limit_queries) if args.limit_queries is not None else None,
        random_seed=int(args.random_seed),
        max_disagreement_examples=int(args.max_disagreement_examples),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

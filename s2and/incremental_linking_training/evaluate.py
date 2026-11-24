"""Score and evaluate linker predictions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s2and.incremental_linking.artifact import IncrementalLinkingArtifact
from s2and.incremental_linking.contracts import DEFAULT_RETRIEVAL_TOP_K
from s2and.incremental_linking.logistic_gate import NumpyLogisticGate

from .gate import (
    _apply_promoted_logistic_gate_to_candidate_rows,
    _gate_bucket_split_counts,
    _promoted_stratified_gate_spec,
    _query_gate_bucket,
    _summarize_predictions,
)
from .ranker import (
    DEFAULT_CLASSIC_N_JOBS,
    CalibratedClassicModel,
    _classic_feature_matrix,
    _drop_unlabeled_singleton_orcid_rows,
    _filter_candidate_rows_to_retrieval_top_k,
    _score_query_choices,
)
from .source_bundle import (
    OfficialBundle,
    _iter_extra_eval_paths,
    _normalize_dataset_slug,
    _read_csv,
    _resolve_path,
    _summary_key_for_eval_dataset,
)


@dataclass(frozen=True)
class ClassicStratifiedEvalScores:
    """Loaded stratified eval rows, model probabilities, and scored choices."""

    rows: pd.DataFrame
    probabilities: np.ndarray
    choices: pd.DataFrame
    assignments: pd.DataFrame


ClassicStratifiedEvalScores.__module__ = "s2and.incremental_linking_training.classic"


@dataclass(frozen=True)
class ClassicEvaluation:
    """Evaluation written from one reloaded complete-bundle artifact."""

    summary: dict[str, Any]
    query_predictions: dict[str, Any]


ClassicEvaluation.__module__ = "s2and.incremental_linking_training.classic"


CALIBRATION_DATASET_SOURCE_KEY_BY_DATASET = {
    "a_khan": "a_khan_eval",
    "a_silva": "a_silva_eval",
    "h_wang": "hwang_eval",
    "j_smith": "j_smith_eval",
    "s_gupta": "s_gupta_eval",
    "s_lee": "s_lee_eval",
    "s_park": "s_park_eval",
}


def _apply_clean_hwang_overrides(predictions: pd.DataFrame, override_df: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        override_df[["query_group_id", "manual_safe_target"]].rename(columns={"query_group_id": "query_case_id"}),
        on="query_case_id",
        how="left",
        validate="one_to_one",
    )
    manual_override = merged["manual_safe_target"].notna()
    if "query_safe_target_source" not in merged.columns:
        merged["query_safe_target_source"] = "retrieved_window"
    merged.loc[manual_override, "query_safe_target_source"] = "manual_safe_target"
    merged["query_safe_target"] = merged["manual_safe_target"].fillna(merged["query_safe_target"]).astype(int)
    merged = merged.drop(columns=["manual_safe_target"])
    merged["correct"] = (
        ((merged["predicted_action"] == "abstain") & (merged["query_safe_target"] == 0))
        | (
            (merged["predicted_action"] == "link_candidate")
            & (merged["query_safe_target"] == 1)
            & (merged["chosen_candidate_target"] == 1)
        )
    ).astype(int)
    return merged


def _evaluate_logistic_scored_windows(
    candidate_rows: pd.DataFrame,
    probabilities: np.ndarray,
    gate: NumpyLogisticGate,
    *,
    override_df: pd.DataFrame | None = None,
    limits: tuple[int, ...] = (5, 25),
) -> dict[str, dict[str, Any]]:
    """Score and summarize retrieval windows with the fitted logistic gate."""

    results: dict[str, dict[str, Any]] = {}
    if len(probabilities) != len(candidate_rows):
        raise ValueError(f"probabilities length must match rows: {len(probabilities)} != {len(candidate_rows)}")
    for limit in sorted(set(int(limit) for limit in limits)):
        mask = (candidate_rows["retrieval_rank"] <= limit).to_numpy(dtype=bool)
        limited = candidate_rows.loc[mask].copy()
        limited_probabilities = probabilities[np.flatnonzero(mask)]
        choices = _score_query_choices(
            limited.rename(columns={"query_group_id": "query_case_id"}),
            limited_probabilities,
            query_id_column="query_case_id",
            include_margin=True,
        )
        predictions = _apply_promoted_logistic_gate_to_candidate_rows(
            limited,
            choices,
            limited_probabilities,
            gate,
        )
        if override_df is not None:
            predictions = _apply_clean_hwang_overrides(predictions, override_df)
        results[str(limit)] = {"overall": _summarize_predictions(predictions)}
    return results


def _evaluation_retrieval_limits(retrieval_top_k: int) -> tuple[int, ...]:
    """Return the diagnostic windows that are reachable by one linker artifact."""

    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive")
    return tuple(sorted({limit for limit in (5, retrieval_top_k) if limit <= retrieval_top_k}))


def _evaluate_logistic_manual_holdout(
    manual_holdout: pd.DataFrame,
    probabilities: np.ndarray,
    gate: NumpyLogisticGate,
) -> dict[str, Any]:
    """Score and summarize manual holdout candidates with the logistic gate."""

    candidate_rows = manual_holdout.rename(columns={"binary_safe_link_target": "label"}).copy()
    if "query_group_id" not in candidate_rows and "query_case_id" in candidate_rows:
        candidate_rows["query_group_id"] = candidate_rows["query_case_id"].astype(str)
    choices = _score_query_choices(
        candidate_rows,
        probabilities,
        query_id_column="query_case_id",
        include_margin=True,
        bucket_column="review_bucket",
    )
    predictions = _apply_promoted_logistic_gate_to_candidate_rows(
        candidate_rows,
        choices,
        probabilities,
        gate,
    )
    return {
        "overall": _summarize_predictions(predictions),
        "by_bucket": {
            str(bucket): _summarize_predictions(group.copy())
            for bucket, group in predictions.groupby("review_bucket", sort=False)
        },
    }


def _classic_stratified_eval_source_specs(spec: dict[str, Any]) -> tuple[dict[str, str], ...]:
    """Return source files used by the promoted eval/test stratified split."""

    sources = [
        {
            "source_key": "calibration_source",
            "table_key": "classic_gate_source_path",
            "path": str(spec["classic_gate_source_path"]),
            "source_kind": "calibration_source",
        },
    ]
    for path_key, source_key in (
        ("s2and_eval_path", "s2and_eval"),
        ("hwang_eval_path", "hwang_eval"),
        ("s_park_eval_path", "s_park_eval"),
        ("s_lee_eval_path", "s_lee_eval"),
    ):
        if path_key in spec or source_key in {"s2and_eval", "hwang_eval"}:
            sources.append(
                {
                    "source_key": source_key,
                    "table_key": path_key,
                    "path": str(spec[path_key]),
                    "source_kind": "public_test",
                }
            )
    for dataset_name, path_like in sorted(dict(spec.get("extra_eval_paths") or {}).items()):
        sources.append(
            {
                "source_key": f"{_normalize_dataset_slug(dataset_name)}_eval",
                "table_key": f"extra_eval_paths.{dataset_name}",
                "path": str(path_like),
                "source_kind": "public_test",
            }
        )
    return tuple(sources)


def _drop_shadowed_calibration_source_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop calibration rows when the active public source has the same query."""

    query_source_key = ["query_group_id", "source_key"]
    public_query_sources = rows.loc[
        ~rows["source_kind"].astype(str).eq("calibration_source"),
        query_source_key,
    ].drop_duplicates()
    if public_query_sources.empty:
        return rows

    marked = rows.merge(
        public_query_sources.assign(_has_public_source_rows=True),
        on=query_source_key,
        how="left",
    )
    shadowed = marked["source_kind"].astype(str).eq("calibration_source") & marked["_has_public_source_rows"].eq(True)
    return marked.loc[~shadowed].drop(columns=["_has_public_source_rows"]).copy()


def _validate_unique_stratified_candidate_rows(rows: pd.DataFrame) -> None:
    """Fail if one selected query/source/candidate has multiple active rows."""

    key_columns = ["query_group_id", "source_key", "candidate_component_key"]
    duplicate_mask = rows.duplicated(key_columns, keep=False)
    if not duplicate_mask.any():
        return

    duplicate_rows = rows.loc[duplicate_mask, key_columns + ["label"]].copy()
    duplicate_summary = (
        duplicate_rows.groupby(key_columns, dropna=False)
        .agg(row_count=("label", "size"), labels=("label", lambda values: sorted({str(value) for value in values})))
        .reset_index()
    )
    conflict_count = int(duplicate_summary["labels"].map(len).gt(1).sum())
    sample = duplicate_summary.head(5).to_dict(orient="records")
    raise ValueError(
        "Promoted stratified eval rows contain duplicate query/source/candidate rows: "
        f"duplicate_pairs={len(duplicate_summary)}, conflicting_pairs={conflict_count}, sample={sample}"
    )


def _validate_stratified_base_identity_split_disjointness(assignments: pd.DataFrame) -> None:
    """Fail if one base identity is assigned to more than one split.

    Query group ids are masked views (``full``, ``initial_only``, ...) of one
    base query; splitting views of the same ``base_group_id`` across splits
    leaks the same author between calibration and test. Split assignment must
    group by ``base_group_id``, mirroring the identity notion the classic
    train/holdout filter enforces.
    """

    base_group_ids = assignments["base_group_id"]
    invalid_base_group_ids = base_group_ids.isna() | base_group_ids.astype(str).str.strip().eq("")
    if bool(invalid_base_group_ids.any()):
        raise ValueError(
            "Stratified split assignments require a nonempty base_group_id for every row: "
            f"invalid_rows={int(invalid_base_group_ids.sum())}"
        )
    split_values = assignments["split"]
    invalid_splits = split_values.isna() | split_values.astype(str).str.strip().eq("")
    if bool(invalid_splits.any()):
        raise ValueError(
            "Stratified split assignments require a nonempty split for every row: "
            f"invalid_rows={int(invalid_splits.sum())}"
        )

    splits_per_base_identity = assignments.groupby("base_group_id", dropna=False)["split"].nunique()
    leaking_base_identities = splits_per_base_identity[splits_per_base_identity > 1]
    if leaking_base_identities.empty:
        return

    raise ValueError(
        "Stratified split assignments place base_group_id values in multiple splits; "
        "masked views of one base query would leak between calibration and test. "
        "Regenerate the assignments with splits assigned per base_group_id: "
        f"count={len(leaking_base_identities)}, sample={leaking_base_identities.head(5).to_dict()}"
    )


def _validate_stratified_split_assignments(assignments: pd.DataFrame) -> None:
    """Validate the identity columns needed for leakage-safe split joins."""

    required = {"query_group_id", "source_key", "split", "base_group_id"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Stratified split assignments missing required columns: {missing}")
    _validate_stratified_base_identity_split_disjointness(assignments)


def _active_stratified_label_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    """Return query/source metadata recomputed from active candidate labels."""

    metadata_input = rows[["query_group_id", "source_key", "candidate_component_key", "retrieval_rank", "label"]].copy()
    metadata_input["_label"] = pd.to_numeric(metadata_input["label"], errors="coerce").fillna(0).astype(int)
    metadata_input["_retrieval_rank"] = pd.to_numeric(metadata_input["retrieval_rank"], errors="coerce").fillna(
        np.iinfo(np.int32).max
    )
    grouped = metadata_input.groupby(["query_group_id", "source_key"], sort=False)
    metadata = grouped.agg(
        candidate_count=("candidate_component_key", "nunique"),
        min_retrieval_rank=("_retrieval_rank", "min"),
        max_retrieval_rank=("_retrieval_rank", "max"),
        positive_candidate_rows=("_label", "sum"),
    ).reset_index()
    min_positive_rank = (
        metadata_input.loc[metadata_input["_label"].eq(1)]
        .groupby(["query_group_id", "source_key"], sort=False)["_retrieval_rank"]
        .min()
    )
    positive_rank_frame = min_positive_rank.rename("min_positive_rank").reset_index()
    metadata = metadata.merge(positive_rank_frame, on=["query_group_id", "source_key"], how="left")
    has_positive = metadata["positive_candidate_rows"].astype(int).gt(0)
    positive_first = has_positive & metadata["min_positive_rank"].eq(metadata["min_retrieval_rank"])
    metadata["has_positive_candidate"] = has_positive
    metadata["positive_first"] = positive_first
    metadata["positive_rank_bucket"] = np.select(
        [~has_positive, positive_first],
        ["no_positive", "positive_first"],
        default="positive_not_first",
    )
    metadata["raw_has_positive_candidate"] = metadata["has_positive_candidate"]
    metadata["raw_positive_first"] = metadata["positive_first"]
    metadata["manual_safe_target"] = has_positive.astype(int)
    metadata["multiple_candidates"] = metadata["candidate_count"].astype(int).gt(1)
    metadata["min_positive_rank"] = metadata["min_positive_rank"].astype(object)
    metadata.loc[~has_positive, "min_positive_rank"] = ""
    for column in ("candidate_count", "min_retrieval_rank", "max_retrieval_rank", "positive_candidate_rows"):
        metadata[column] = metadata[column].astype(int)
    return metadata


def _refresh_stratified_metadata_from_active_labels(
    rows: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh split metadata from the selected active candidate rows."""

    metadata = _active_stratified_label_metadata(rows)
    metadata_columns = [
        "has_positive_candidate",
        "positive_first",
        "positive_rank_bucket",
        "raw_has_positive_candidate",
        "raw_positive_first",
        "manual_safe_target",
        "multiple_candidates",
        "candidate_count",
        "min_positive_rank",
        "min_retrieval_rank",
        "max_retrieval_rank",
        "positive_candidate_rows",
    ]
    refreshed_rows = rows.drop(columns=[column for column in metadata_columns if column in rows.columns]).merge(
        metadata,
        on=["query_group_id", "source_key"],
        how="left",
    )
    refreshed_assignments = assignments.drop(
        columns=[column for column in metadata_columns if column in assignments.columns]
    ).merge(
        metadata,
        on=["query_group_id", "source_key"],
        how="left",
    )
    if "source_stratum" in refreshed_assignments.columns and "first_name_bucket" in refreshed_assignments.columns:
        refreshed_assignments["stratum_key"] = refreshed_assignments.apply(
            lambda row: (
                f"{row['source_stratum']}|has_pos={int(bool(row['has_positive_candidate']))}|"
                f"{row['positive_rank_bucket']}|{row['first_name_bucket']}|"
                f"multi_cand={int(bool(row['multiple_candidates']))}"
            ),
            axis=1,
        )
    if "stratum_key" in refreshed_assignments.columns:
        refreshed_rows = refreshed_rows.drop(columns=["stratum_key"], errors="ignore").merge(
            refreshed_assignments[["query_group_id", "source_key", "stratum_key"]],
            on=["query_group_id", "source_key"],
            how="left",
        )
    return refreshed_rows, refreshed_assignments


def _select_classic_stratified_eval_rows(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    assignments: pd.DataFrame,
    *,
    include_calibration_source: bool,
    read_source_rows: Callable[[Path, set[str]], pd.DataFrame],
) -> pd.DataFrame:
    """Resolve assignments to eligible source rows for fit, evaluation, and preflight."""
    sources = _classic_stratified_eval_source_specs(spec)
    known_keys = {source["source_key"] for source in sources if source["source_kind"] != "calibration_source"}
    if unknown_keys := set(assignments["source_key"].astype(str)) - known_keys:
        raise ValueError(f"Stratified split assignments use unknown source_key values: {sorted(unknown_keys)}")
    assigned_query_ids = set(assignments["query_group_id"].astype(str))
    source_frames: list[pd.DataFrame] = []
    for source_spec in sources:
        if source_spec["source_kind"] == "calibration_source":
            if not include_calibration_source:
                continue
            source_query_ids = assigned_query_ids
        else:
            source_query_ids = set(
                assignments.loc[
                    assignments["source_key"].astype(str).eq(str(source_spec["source_key"])),
                    "query_group_id",
                ].astype(str)
            )
        if not source_query_ids:
            continue
        source_path = _resolve_path(bundle, source_spec["path"])
        rows = read_source_rows(source_path, source_query_ids)
        if str(source_spec["source_key"]) == "calibration_source":
            rows["source_key"] = (
                rows["dataset"].astype(str).map(CALIBRATION_DATASET_SOURCE_KEY_BY_DATASET).fillna("s2and_eval")
            )
        else:
            rows["source_key"] = str(source_spec["source_key"])
        rows["source_kind"] = str(source_spec["source_kind"])
        rows, _filter_summary = _drop_unlabeled_singleton_orcid_rows(
            rows,
            context=f"stratified_eval:{source_spec['source_key']}",
        )
        source_frames.append(rows)
    all_rows = _drop_shadowed_calibration_source_rows(pd.concat(source_frames, ignore_index=True))
    join_keys = ["query_group_id", "source_key"]
    rows = all_rows.merge(assignments, on=join_keys, how="inner", suffixes=("", "_split"))
    matched_assignments = rows[join_keys].drop_duplicates()
    if len(matched_assignments) != len(assignments):
        missing = assignments[join_keys].merge(matched_assignments, on=join_keys, how="left", indicator=True)
        examples = missing.loc[missing["_merge"].eq("left_only"), join_keys].head(5).to_dict(orient="records")
        raise ValueError(
            "Stratified split assignments did not all match source rows: "
            f"matched={len(matched_assignments)}, expected={len(assignments)}, sample={examples}"
        )
    if "base_group_id_split" not in rows.columns:
        raise ValueError("Stratified eval source rows must contain base_group_id")
    assignment_base_column = "base_group_id_split"
    source_base_ids = rows["base_group_id"]
    assignment_base_ids = rows[assignment_base_column]
    invalid_source_base_ids = source_base_ids.isna() | source_base_ids.astype(str).str.strip().eq("")
    if bool(invalid_source_base_ids.any()):
        raise ValueError(
            "Stratified eval source rows require nonempty base_group_id values: "
            f"invalid_rows={int(invalid_source_base_ids.sum())}"
        )
    mismatched_base_ids = source_base_ids.astype(str) != assignment_base_ids.astype(str)
    if bool(mismatched_base_ids.any()):
        sample = rows.loc[
            mismatched_base_ids,
            ["query_group_id", "source_key", "base_group_id", assignment_base_column],
        ].head(5)
        raise ValueError(
            "Stratified split assignment base_group_id does not match source rows: "
            f"mismatched_rows={int(mismatched_base_ids.sum())}, sample={sample.to_dict(orient='records')}"
        )
    _validate_unique_stratified_candidate_rows(rows)
    source_keys_per_query = rows.groupby("query_group_id")["source_key"].nunique()
    cross_source_queries = source_keys_per_query[source_keys_per_query > 1]
    if not cross_source_queries.empty:
        raise ValueError(
            "Stratified eval rows contain query_group_id values present under multiple source_keys; "
            "downstream scoring groups by query_group_id only and would silently collapse them: "
            f"count={len(cross_source_queries)}, sample={cross_source_queries.head(5).to_dict()}"
        )
    return rows.drop(columns=[assignment_base_column])


def _read_classic_stratified_source_rows(source_path: Path, source_query_ids: set[str]) -> pd.DataFrame:
    """Read classic label rows for the assigned query groups."""
    if source_path.suffix == ".parquet":
        rows = pd.read_parquet(
            source_path,
            filters=[("query_group_id", "in", sorted(source_query_ids))],
        )
    else:
        rows = _read_csv(source_path, compression="gzip")
        rows = rows[rows["query_group_id"].astype(str).isin(source_query_ids)].copy()
    return rows


def _load_classic_stratified_eval_rows(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    split_spec: dict[str, Any],
    *,
    splits: Sequence[str],
    include_calibration_source: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load rows assigned to the requested splits."""

    assignments = _read_csv(_resolve_path(bundle, str(split_spec["assignments_path"])))
    _validate_stratified_split_assignments(assignments)
    requested_splits = {str(split) for split in splits}
    assignments = assignments[assignments["split"].astype(str).isin(requested_splits)].copy()
    if assignments.empty:
        raise ValueError(f"Stratified assignments contain no rows for splits={sorted(requested_splits)}")
    assignment_columns = [
        "query_group_id",
        "source_key",
        "split",
        "base_group_id",
        "stratum_key",
        "source_stratum",
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
        "manual_safe_target",
        "correction_type",
    ]
    selected_assignment_columns = [column for column in assignment_columns if column in assignments.columns]
    rows = _select_classic_stratified_eval_rows(
        bundle,
        spec,
        assignments[selected_assignment_columns],
        include_calibration_source=include_calibration_source,
        read_source_rows=_read_classic_stratified_source_rows,
    )
    for column in selected_assignment_columns:
        if column in {"query_group_id", "source_key"}:
            continue
        assignment_column = f"{column}_split"
        if assignment_column in rows.columns:
            rows[column] = rows[assignment_column]
            rows = rows.drop(columns=[assignment_column])
    rows, assignments = _refresh_stratified_metadata_from_active_labels(rows, assignments)
    return rows, assignments


def _breakdown_predictions(predictions: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    """Summarize predictions by one column for JSON reports."""

    return {
        str(value): _summarize_predictions(group.copy())
        for value, group in predictions.groupby(column, dropna=False, sort=True)
    }


def _score_classic_stratified_rows(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    split_spec: dict[str, Any],
    predict_probabilities: Callable[[np.ndarray], np.ndarray],
    feature_columns: tuple[str, ...],
    *,
    splits: Sequence[str],
    include_calibration_source: bool,
    retrieval_top_k: int | None = None,
) -> ClassicStratifiedEvalScores:
    """Load and score selected promoted stratified splits."""

    split_rows, assignments = _load_classic_stratified_eval_rows(
        bundle,
        spec,
        split_spec,
        splits=splits,
        include_calibration_source=include_calibration_source,
    )
    active_retrieval_top_k = int(
        spec.get("retrieval_top_k", DEFAULT_RETRIEVAL_TOP_K) if retrieval_top_k is None else retrieval_top_k
    )
    split_rows = _filter_candidate_rows_to_retrieval_top_k(
        split_rows,
        retrieval_top_k=active_retrieval_top_k,
        context="classic_stratified_eval",
    )
    retained_query_ids = set(split_rows["query_group_id"].astype(str))
    missing_query_ids = set(assignments["query_group_id"].astype(str)) - retained_query_ids
    if missing_query_ids:
        raise ValueError(
            "classic_stratified_eval has assigned queries with no candidates inside "
            f"retrieval_top_k={active_retrieval_top_k}: "
            f"count={len(missing_query_ids)}, sample={sorted(missing_query_ids)[:5]}"
        )
    split_rows, assignments = _refresh_stratified_metadata_from_active_labels(split_rows, assignments)
    matrix = np.ascontiguousarray(
        _classic_feature_matrix(split_rows, feature_columns).to_numpy(dtype=np.float32),
    )
    probabilities = np.asarray(predict_probabilities(matrix), dtype=np.float64).reshape(-1)
    choices = _score_query_choices(
        split_rows,
        probabilities,
        query_id_column="query_group_id",
        include_margin=True,
    )
    metadata_columns = [
        "query_group_id",
        "source_key",
        "source_kind",
        "split",
        "stratum_key",
        "source_stratum",
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
        "manual_safe_target",
        "correction_type",
    ]
    metadata = split_rows[[column for column in metadata_columns if column in split_rows.columns]].drop_duplicates(
        "query_group_id"
    )
    metadata = metadata.rename(columns={"query_group_id": "query_case_id"})
    choices = choices.merge(metadata, on="query_case_id", how="left", suffixes=("", "_split"))
    if "first_name_bucket_split" in choices.columns:
        choices["first_name_bucket"] = choices["first_name_bucket_split"].fillna(choices["first_name_bucket"])
        choices = choices.drop(columns=["first_name_bucket_split"])
    if "query_safe_target_source" not in choices.columns:
        choices["query_safe_target_source"] = "retrieved_window"
    if "manual_safe_target" in choices.columns:
        manual_target = pd.to_numeric(choices["manual_safe_target"], errors="coerce")
        choices["manual_safe_target_matches_active_label"] = manual_target.isna() | manual_target.astype("Int64").eq(
            choices["query_safe_target"].astype("Int64")
        )
    return ClassicStratifiedEvalScores(
        rows=split_rows,
        probabilities=probabilities,
        choices=choices,
        assignments=assignments,
    )


def _summarize_classic_stratified_predictions(
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    split_spec: dict[str, Any],
) -> dict[str, Any]:
    """Build promoted stratified split summary and test breakdowns from predictions."""

    predictions = predictions.copy()
    predictions["gate_bucket"] = _query_gate_bucket(predictions) if not predictions.empty else pd.Series(dtype="string")
    split_order = tuple(
        str(value)
        for value in split_spec.get(
            "split_order",
            ("calibration_fit", "calibration_check", "test"),
        )
    )
    overall_by_split = {
        split: _summarize_predictions(predictions[predictions["split"] == split].copy()) for split in split_order
    }
    test_predictions = predictions[predictions["split"] == str(split_spec.get("test_split", "test"))].copy()
    factor_columns = [
        "gate_bucket",
        "source_key",
        "source_stratum",
        "has_positive_candidate",
        "positive_rank_bucket",
        "first_name_bucket",
        "multiple_candidates",
    ]
    return {
        "assignment_query_counts": {
            str(split): int(count) for split, count in assignments["split"].value_counts().sort_index().items()
        },
        "scored_query_counts": {
            str(split): int(count) for split, count in predictions["split"].value_counts().sort_index().items()
        },
        "overall": overall_by_split,
        "gate_bucket_split_counts": _gate_bucket_split_counts(predictions, split_order),
        "test_breakdowns": {
            column: _breakdown_predictions(test_predictions, column)
            for column in factor_columns
            if column in test_predictions.columns
        },
    }


def _write_query_predictions(predictions: pd.DataFrame, path: Path | None = None) -> dict[str, Any]:
    """Inventory exact query-level decisions, optionally writing their CSV."""

    required = {
        "base_group_id",
        "chosen_candidate_component_key",
        "chosen_probability",
        "correct",
        "predicted_action",
        "query_case_id",
        "query_safe_target",
        "source_key",
        "split",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Linker query predictions are missing required columns: {missing}")
    frame = predictions.copy()
    identity_columns = ["source_key", "query_case_id"]
    if frame.duplicated(identity_columns).any():
        raise ValueError("Linker query predictions contain duplicate source/query identities")
    frame["label"] = pd.to_numeric(frame["query_safe_target"], errors="raise").astype(int)
    leading_columns = [*identity_columns, "base_group_id", "split", "label"]
    remaining_columns = sorted(set(frame.columns) - set(leading_columns))
    frame = frame[[*leading_columns, *remaining_columns]].sort_values(
        ["split", "source_key", "base_group_id", "query_case_id"],
        kind="mergesort",
    )
    payload = frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    inventory = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": len(frame),
        "columns": list(frame.columns),
    }
    if path is not None:
        inventory["path"] = str(path)
    return inventory


def _score_eval_candidate_rows(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    include_margin: bool,
    limits: tuple[int, ...] = (5, 25),
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if len(probabilities) != len(df):
        raise ValueError(f"probabilities length must match df rows: {len(probabilities)} != {len(df)}")
    for limit in sorted(set(int(limit) for limit in limits)):
        mask = (df["retrieval_rank"] <= limit).to_numpy(dtype=bool)
        limited = df.loc[mask].copy()
        limited_probabilities = probabilities[np.flatnonzero(mask)]
        choices = _score_query_choices(
            limited.rename(columns={"query_group_id": "query_case_id"}),
            limited_probabilities,
            query_id_column="query_case_id",
            include_margin=include_margin,
        )
        choices["retrieval_rank_limit"] = limit
        frames.append(choices)
    return pd.concat(frames, ignore_index=True)


def _score_frozen_gate_rows(
    task_rows: Mapping[str, pd.DataFrame],
    predict: Callable[[Any], np.ndarray],
    feature_columns: tuple[str, ...],
) -> ClassicStratifiedEvalScores:
    """Score complete windows and keep only verified outcomes for gate fitting."""
    rows = pd.concat(
        [task_rows[split].assign(split=split) for split in ("gate_fit", "gate_thresh")],
        ignore_index=True,
    )
    probabilities = predict(_classic_feature_matrix(rows, feature_columns))
    choices = _score_query_choices(rows, probabilities, query_id_column="query_group_id", include_margin=True)
    splits = rows.drop_duplicates("query_group_id").set_index("query_group_id")["split"]
    choices["split"] = choices["query_case_id"].map(splits)
    keep_choices = choices["split"].eq("gate_thresh") | choices["chosen_candidate_gold_scored"]
    query_ids = set(choices.loc[keep_choices, "query_case_id"])
    keep_rows = rows["query_group_id"].isin(query_ids).to_numpy(dtype=bool)
    return ClassicStratifiedEvalScores(
        rows.loc[keep_rows], probabilities[keep_rows], choices.loc[keep_choices], pd.DataFrame()
    )


def _classic_internal_eval(
    bundle: OfficialBundle,
    spec: dict[str, Any],
    predict: Callable[[Any], np.ndarray],
    feature_columns: tuple[str, ...],
    retrieval_top_k: int,
    gate: NumpyLogisticGate,
) -> dict[str, Any]:
    """Evaluate the classic gate's existing internal holdout."""
    source_path = _resolve_path(bundle, spec.get("classic_gate_source_path", spec["hwang_eval_path"]))
    rows = _read_csv(source_path, compression="gzip")
    rows, _ = _drop_unlabeled_singleton_orcid_rows(rows, context="classic_gate_source")
    rows = _filter_candidate_rows_to_retrieval_top_k(
        rows,
        retrieval_top_k=retrieval_top_k,
        context="classic_gate_source",
    )
    probabilities = predict(_classic_feature_matrix(rows, feature_columns))
    choices = _score_eval_candidate_rows(rows, probabilities, include_margin=True, limits=(retrieval_top_k,))
    groups_path = _resolve_path(bundle, spec["classic_gate_internal_eval_base_groups_path"])
    groups = set(_read_csv(groups_path)["base_group_id"].astype(str))
    choices = choices[choices["base_group_id"].isin(groups)].drop(columns=["retrieval_rank_limit"])
    internal = rows[rows["base_group_id"].astype(str).isin(groups)]
    predictions = _apply_promoted_logistic_gate_to_candidate_rows(
        internal, choices, probabilities[internal.index.to_numpy(dtype=np.int64)], gate
    )
    return _summarize_predictions(predictions)


def evaluate_linker(
    bundle: OfficialBundle,
    *,
    calibrated: CalibratedClassicModel,
    artifact: IncrementalLinkingArtifact,
    n_jobs: int = DEFAULT_CLASSIC_N_JOBS,
) -> ClassicEvaluation:
    """Evaluate the exact linker artifact loaded from a complete bundle."""

    if artifact.retrieval_top_k != calibrated.retrieval_top_k:
        raise ValueError(
            "Reloaded linker retrieval_top_k differs from the calibrated model: "
            f"{artifact.retrieval_top_k} != {calibrated.retrieval_top_k}"
        )
    spec = bundle.models["classic"]
    feature_columns = tuple(spec["feature_columns"])
    retrieval_top_k = artifact.retrieval_top_k
    gate = artifact.gate_model

    def predict_rows(rows: pd.DataFrame) -> np.ndarray:
        matrix = np.ascontiguousarray(
            _classic_feature_matrix(rows, feature_columns).to_numpy(dtype=np.float32),
        )
        return artifact.predict_probabilities(matrix, num_threads=n_jobs)

    promoted_gate_config = _promoted_stratified_gate_spec(spec)
    if promoted_gate_config is None:
        raise ValueError("classic.promoted_stratified_gate is required")
    split_spec = dict(spec["stratified_eval_test_split"])
    test_split = str(promoted_gate_config["test_split"])
    stratified_scores = _score_classic_stratified_rows(
        bundle,
        spec,
        split_spec,
        lambda matrix: artifact.predict_probabilities(matrix, num_threads=n_jobs),
        feature_columns,
        splits=(test_split,),
        include_calibration_source=False,
        retrieval_top_k=retrieval_top_k,
    )
    test_predictions = _apply_promoted_logistic_gate_to_candidate_rows(
        stratified_scores.rows,
        stratified_scores.choices,
        stratified_scores.probabilities,
        gate,
    )
    stratified_eval_test_summary = _summarize_classic_stratified_predictions(
        test_predictions,
        stratified_scores.assignments,
        split_spec,
    )

    evaluation_limits = _evaluation_retrieval_limits(retrieval_top_k)
    s2and_eval = _filter_candidate_rows_to_retrieval_top_k(
        _read_csv(_resolve_path(bundle, spec["s2and_eval_path"]), compression="gzip"),
        retrieval_top_k=retrieval_top_k,
        context="classic_s2and_eval",
    )
    s2and_eval_summary = _evaluate_logistic_scored_windows(
        s2and_eval,
        predict_rows(s2and_eval),
        gate,
        limits=evaluation_limits,
    )
    hwang_eval = _filter_candidate_rows_to_retrieval_top_k(
        _read_csv(_resolve_path(bundle, spec["hwang_eval_path"]), compression="gzip"),
        retrieval_top_k=retrieval_top_k,
        context="classic_hwang_eval",
    )
    hwang_probabilities = predict_rows(hwang_eval)
    optional_eval_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_name, path_like in _iter_extra_eval_paths(spec):
        eval_df = _filter_candidate_rows_to_retrieval_top_k(
            _read_csv(_resolve_path(bundle, path_like), compression="gzip"),
            retrieval_top_k=retrieval_top_k,
            context=f"classic_extra_eval:{dataset_name}",
        )
        optional_eval_summaries[_summary_key_for_eval_dataset(dataset_name)] = _evaluate_logistic_scored_windows(
            eval_df,
            predict_rows(eval_df),
            gate,
            limits=evaluation_limits,
        )

    override_path = spec.get("hwang_clean_override_path")
    override_df = _read_csv(_resolve_path(bundle, str(override_path))) if override_path else None
    hwang_eval_summary = _evaluate_logistic_scored_windows(
        hwang_eval,
        hwang_probabilities,
        gate,
        override_df=override_df,
        limits=evaluation_limits,
    )
    hwang_cleaned_eval = {
        f"w{limit}": {
            "cleaned_balanced_accuracy": window_summary["overall"]["balanced_accuracy"],
            "cleaned_positive_recall": window_summary["overall"]["positive_recall"],
            "cleaned_negative_recall": window_summary["overall"]["negative_recall"],
        }
        for limit, window_summary in hwang_eval_summary.items()
    }

    summary = {
        "model": "classic",
        "training_summary": calibrated.training_summary,
        "abstain_rule": calibrated.abstain_rule_summary,
        "overall_s2and_eval": s2and_eval_summary,
        "hwang_cleaned_eval": hwang_cleaned_eval,
        "stratified_eval_test_split": stratified_eval_test_summary,
    }
    manual_holdout_path = spec.get("manual_holdout_candidates_path")
    if manual_holdout_path:
        manual_holdout = _read_csv(_resolve_path(bundle, manual_holdout_path), low_memory=False)
        manual_holdout = _filter_candidate_rows_to_retrieval_top_k(
            manual_holdout,
            retrieval_top_k=retrieval_top_k,
            context="classic_manual_holdout",
        )
        summary["manual_holdout"] = _evaluate_logistic_manual_holdout(
            manual_holdout,
            predict_rows(manual_holdout),
            gate,
        )
    for summary_key, eval_summary in optional_eval_summaries.items():
        summary[summary_key] = eval_summary
    query_predictions = _write_query_predictions(test_predictions)
    return ClassicEvaluation(
        summary=summary,
        query_predictions=query_predictions,
    )

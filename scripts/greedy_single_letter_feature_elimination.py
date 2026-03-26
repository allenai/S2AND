"""Run greedy feature elimination for the single-letter reranker.

This script freezes one corrected S2AND-only ``LGBMRanker`` configuration,
uses the same cached ``h_wang`` top-50 slice to choose feature removals, and
records the held-out S2AND trajectory for diagnosis only.
Only the baseline full-feature point is a fresh blind transfer check.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from lightgbm import LGBMRanker

matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from scripts.eval_single_letter_ranker import (
        _load_dataset_rows,
        _ordered_group_rows,
        _predict_scores,
        _top1_accuracy,
        _training_matrix_group_sizes,
        _write_group_csv,
    )
    from scripts.single_letter_reranker_utils import (
        DEFAULT_LABELED_DATASETS,
        FEATURE_PRESETS,
        build_feature_matrix,
        build_training_matrix,
        materialize_derived_rows,
        read_rows_csv,
        resolve_feature_columns,
        select_rows,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from eval_single_letter_ranker import (  # type: ignore[no-redef]
        _load_dataset_rows,
        _ordered_group_rows,
        _predict_scores,
        _top1_accuracy,
        _training_matrix_group_sizes,
        _write_group_csv,
    )
    from single_letter_reranker_utils import (  # type: ignore[no-redef]
        DEFAULT_LABELED_DATASETS,
        FEATURE_PRESETS,
        build_feature_matrix,
        build_training_matrix,
        materialize_derived_rows,
        read_rows_csv,
        resolve_feature_columns,
        select_rows,
        write_json,
    )


DEFAULT_FIXED_SUMMARY_JSON = Path("scratch/s2and_ranker_all6_k50_v8_roundrobinmatched_full_20260325/summary.json")
DEFAULT_DATASET_ROOT = Path("scratch/s2and_round_robin_matched_20260325_rows")
DEFAULT_H_WANG_ROW_FILE = Path("scratch/single_letter_reranker_h_wang_full_20260322/rows.csv")


@dataclass(frozen=True)
class PreparedTrainingData:
    """Prepared grouped training data with the full feature matrix."""

    features: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray
    group_sizes: list[int]
    rows_seen: int
    rows_used: int
    groups_used: int
    dropped_all_negative_group_count: int


@dataclass(frozen=True)
class PreparedEvaluationData:
    """Prepared grouped evaluation data with the full feature matrix."""

    ordered_rows: list[dict[str, Any]]
    features: np.ndarray
    labels: np.ndarray
    group_sizes: list[int]


@dataclass(frozen=True)
class PreparedHeldoutSplit:
    """One leave-dataset-out diagnostic split."""

    heldout_dataset: str
    train_data: PreparedTrainingData
    eval_data: PreparedEvaluationData


@dataclass(frozen=True)
class CandidateResult:
    """One ``h_wang`` top-50 result for a candidate feature removal."""

    position: int
    removed_feature: str
    remaining_features: tuple[str, ...]
    h_wang_accuracy: float
    fit_seconds: float
    predict_seconds: float


def _configure_single_thread_runtime() -> None:
    """Force single-threaded model work inside each concurrent job."""

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[variable] = "1"


def _read_fixed_best_params(summary_json: Path) -> dict[str, Any]:
    """Read the fixed LightGBM parameter blob from one saved summary."""

    with summary_json.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    params = summary["h_wang"]["train_summary"]["best_params"]
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Expected non-empty h_wang.train_summary.best_params in {summary_json}")
    return dict(params)


def _prepare_training_data(
    rows: Sequence[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
    seed: int,
) -> PreparedTrainingData:
    """Prepare the grouped training matrix for one fixed row set."""

    training_matrix = build_training_matrix(
        rows,
        seed=int(seed),
        feature_columns=feature_columns,
        enrichment_profile="none",
        enrichment_rounds=0,
    )
    return PreparedTrainingData(
        features=np.asarray(training_matrix.features, dtype=np.float32),
        labels=np.asarray(training_matrix.labels, dtype=np.int32),
        sample_weights=np.asarray(training_matrix.sample_weights, dtype=np.float32),
        group_sizes=_training_matrix_group_sizes(training_matrix),
        rows_seen=int(len(rows)),
        rows_used=int(training_matrix.features.shape[0]),
        groups_used=int(len(training_matrix.group_ids)),
        dropped_all_negative_group_count=int(len(training_matrix.dropped_all_negative_group_ids)),
    )


def _prepare_evaluation_data(
    rows: Sequence[dict[str, Any]],
    *,
    feature_columns: tuple[str, ...],
) -> PreparedEvaluationData:
    """Prepare the ordered evaluation rows and feature matrix."""

    ordered_rows, group_sizes, _group_ids = _ordered_group_rows(list(rows))
    features = build_feature_matrix(ordered_rows, feature_columns=feature_columns)
    labels = np.asarray([int(row["label"]) for row in ordered_rows], dtype=np.int32)
    return PreparedEvaluationData(
        ordered_rows=ordered_rows,
        features=np.asarray(features, dtype=np.float32),
        labels=labels,
        group_sizes=list(group_sizes),
    )


def _build_ranker(*, fixed_params: dict[str, Any], seed: int, n_jobs: int) -> Any:
    """Build one fixed-parameter LightGBM ranker."""

    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_jobs=int(n_jobs),
        verbose=-1,
        tree_learner="data",
        random_state=int(seed),
        data_random_seed=int(seed),
        feature_fraction_seed=int(seed),
        **fixed_params,
    )


def _column_indices(
    columns: Sequence[str],
    *,
    feature_index: dict[str, int],
) -> np.ndarray:
    """Resolve the numeric column indices for one feature subset."""

    return np.asarray([int(feature_index[column]) for column in columns], dtype=np.int64)


def _evaluate_subset_accuracy(
    *,
    train_data: PreparedTrainingData,
    eval_data: PreparedEvaluationData,
    fixed_params: dict[str, Any],
    seed: int,
    lgbm_n_jobs: int,
    feature_indices: np.ndarray,
) -> tuple[float, float, float]:
    """Fit one model on the selected subset and return accuracy plus timings."""

    ranker = _build_ranker(
        fixed_params=fixed_params,
        seed=int(seed),
        n_jobs=int(lgbm_n_jobs),
    )
    train_features = np.ascontiguousarray(train_data.features[:, feature_indices], dtype=np.float32)
    eval_features = np.ascontiguousarray(eval_data.features[:, feature_indices], dtype=np.float32)
    fit_start = time.perf_counter()
    ranker.fit(
        train_features,
        train_data.labels,
        group=train_data.group_sizes,
        sample_weight=train_data.sample_weights,
    )
    fit_seconds = float(time.perf_counter() - fit_start)
    predict_start = time.perf_counter()
    scores = _predict_scores(ranker, eval_features)
    predict_seconds = float(time.perf_counter() - predict_start)
    accuracy = _top1_accuracy(scores, eval_data.labels, eval_data.group_sizes)
    return float(accuracy), float(fit_seconds), float(predict_seconds)


def _score_candidate_removal(
    *,
    position: int,
    removed_feature: str,
    remaining_features: tuple[str, ...],
    feature_index: dict[str, int],
    train_data: PreparedTrainingData,
    eval_data: PreparedEvaluationData,
    fixed_params: dict[str, Any],
    seed: int,
    lgbm_n_jobs: int,
) -> CandidateResult:
    """Score one candidate feature removal on the cached ``h_wang`` slice."""

    accuracy, fit_seconds, predict_seconds = _evaluate_subset_accuracy(
        train_data=train_data,
        eval_data=eval_data,
        fixed_params=fixed_params,
        seed=int(seed),
        lgbm_n_jobs=int(lgbm_n_jobs),
        feature_indices=_column_indices(remaining_features, feature_index=feature_index),
    )
    return CandidateResult(
        position=int(position),
        removed_feature=str(removed_feature),
        remaining_features=tuple(remaining_features),
        h_wang_accuracy=round(float(accuracy), 6),
        fit_seconds=round(float(fit_seconds), 6),
        predict_seconds=round(float(predict_seconds), 6),
    )


def _choose_best_candidate_result(results: Sequence[CandidateResult]) -> CandidateResult:
    """Select the accepted removal using the current ``h_wang`` slice only.

    Ties break by original feature order so the path stays deterministic without
    consulting held-out S2AND diagnostics.
    """

    if not results:
        raise ValueError("Expected at least one candidate result")
    ordered = sorted(results, key=lambda result: (-float(result.h_wang_accuracy), int(result.position)))
    return ordered[0]


def _evaluate_heldout_splits(
    *,
    splits: Sequence[PreparedHeldoutSplit],
    feature_columns: tuple[str, ...],
    feature_index: dict[str, int],
    fixed_params: dict[str, Any],
    seed: int,
    lgbm_n_jobs: int,
    max_workers: int,
) -> dict[str, Any]:
    """Evaluate the accepted feature set on leave-dataset-out S2AND splits."""

    feature_indices = _column_indices(feature_columns, feature_index=feature_index)

    def evaluate_split(split: PreparedHeldoutSplit) -> tuple[str, float]:
        accuracy, _fit_seconds, _predict_seconds = _evaluate_subset_accuracy(
            train_data=split.train_data,
            eval_data=split.eval_data,
            fixed_params=fixed_params,
            seed=int(seed),
            lgbm_n_jobs=int(lgbm_n_jobs),
            feature_indices=feature_indices,
        )
        return str(split.heldout_dataset), float(accuracy)

    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(splits))) as executor:
        split_results = list(executor.map(evaluate_split, splits))

    by_dataset_raw = {dataset_name: accuracy for dataset_name, accuracy in split_results}
    overall_queries = sum(int(len(split.eval_data.group_sizes)) for split in splits)
    overall_correct = sum(
        float(by_dataset_raw[str(split.heldout_dataset)]) * float(len(split.eval_data.group_sizes)) for split in splits
    )
    overall_accuracy = float(overall_correct / max(1, overall_queries))
    return {
        "overall_accuracy": round(float(overall_accuracy), 6),
        "by_dataset": {
            dataset_name: round(float(accuracy), 6) for dataset_name, accuracy in sorted(by_dataset_raw.items())
        },
        "queries": int(overall_queries),
    }


def _candidate_result_rows(
    *,
    step_index: int,
    results: Sequence[CandidateResult],
) -> list[dict[str, Any]]:
    """Convert one step's candidate results into tabular rows."""

    ordered = sorted(results, key=lambda result: (-float(result.h_wang_accuracy), int(result.position)))
    return [
        {
            "step_index": int(step_index),
            "rank": int(rank),
            "removed_feature": str(result.removed_feature),
            "h_wang_top50_accuracy": round(float(result.h_wang_accuracy), 6),
            "remaining_feature_count": int(len(result.remaining_features)),
            "fit_seconds": round(float(result.fit_seconds), 6),
            "predict_seconds": round(float(result.predict_seconds), 6),
            "remaining_features": "|".join(result.remaining_features),
        }
        for rank, result in enumerate(ordered, start=1)
    ]


def _trajectory_row(
    *,
    step_index: int,
    removed_feature: str | None,
    feature_columns: tuple[str, ...],
    h_wang_accuracy: float,
    heldout_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build one accepted-step trajectory row."""

    return {
        "step_index": int(step_index),
        "removed_feature": None if removed_feature is None else str(removed_feature),
        "remaining_feature_count": int(len(feature_columns)),
        "h_wang_top50_accuracy": round(float(h_wang_accuracy), 6),
        "heldout_s2and_accuracy": round(float(heldout_summary["overall_accuracy"]), 6),
        "heldout_queries": int(heldout_summary["queries"]),
        "feature_columns": list(feature_columns),
    }


def _write_trajectory_plot(path: Path, trajectory: Sequence[dict[str, Any]]) -> None:
    """Write the required two-line elimination trajectory plot."""

    x_values = [int(row["remaining_feature_count"]) for row in trajectory]
    h_wang_values = [float(row["h_wang_top50_accuracy"]) for row in trajectory]
    heldout_values = [float(row["heldout_s2and_accuracy"]) for row in trajectory]
    baseline_h_wang = float(trajectory[0]["h_wang_top50_accuracy"])
    plt.figure(figsize=(9, 5))
    plt.plot(x_values, heldout_values, marker="s", linewidth=1.8, label="held-out S2AND overall")
    plt.plot(
        x_values,
        h_wang_values,
        marker="o",
        linewidth=1.8,
        label="accepted-path h_wang top-50 (only step 0 is blind)",
    )
    plt.axhline(
        baseline_h_wang,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        color="#DD8452",
        label="blind full-v8 baseline",
    )
    plt.gca().invert_xaxis()
    plt.xlabel("Remaining features")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_trajectory_csv(path: Path, trajectory: Sequence[dict[str, Any]]) -> None:
    """Write the accepted elimination path to CSV."""

    rows = [
        {
            "step_index": int(row["step_index"]),
            "removed_feature": row["removed_feature"],
            "remaining_feature_count": int(row["remaining_feature_count"]),
            "h_wang_top50_accuracy": round(float(row["h_wang_top50_accuracy"]), 6),
            "heldout_s2and_accuracy": round(float(row["heldout_s2and_accuracy"]), 6),
            "heldout_queries": int(row["heldout_queries"]),
            "feature_columns": "|".join(row["feature_columns"]),
        }
        for row in trajectory
    ]
    _write_group_csv(path, rows)


def _materialize_filtered_rows(
    rows: Sequence[dict[str, Any]],
    *,
    query_view: str,
    window_size: int,
) -> list[dict[str, Any]]:
    """Filter to the locked operating point and materialize derived features once."""

    filtered_rows = select_rows(
        rows,
        query_view=str(query_view),
        window_size=int(window_size),
    )
    return materialize_derived_rows(filtered_rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_LABELED_DATASETS))
    parser.add_argument("--rows-source", choices=("auto", "base", "derived"), default="auto")
    parser.add_argument("--query-view", default="initial_only")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--feature-preset", choices=sorted(FEATURE_PRESETS), default="generalized_v8")
    parser.add_argument("--fixed-summary-json", type=Path, default=DEFAULT_FIXED_SUMMARY_JSON)
    parser.add_argument("--h-wang-row-file", type=Path, default=DEFAULT_H_WANG_ROW_FILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lgbm-n-jobs", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--min-features", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Run the greedy elimination experiment and write artifacts."""

    args = parse_args()
    _configure_single_thread_runtime()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    fixed_params = _read_fixed_best_params(args.fixed_summary_json)
    full_feature_columns = resolve_feature_columns(feature_preset=str(args.feature_preset))
    if int(args.min_features) < 1 or int(args.min_features) >= len(full_feature_columns):
        raise ValueError(
            f"--min-features must be between 1 and {len(full_feature_columns) - 1}, got {args.min_features}"
        )

    dataset_rows, rows_source_by_dataset = _load_dataset_rows(
        args.dataset_root,
        [str(value) for value in args.datasets],
        rows_source=str(args.rows_source),
        selected_query_group_ids=None,
    )
    materialized_dataset_rows = _materialize_filtered_rows(
        dataset_rows,
        query_view=str(args.query_view),
        window_size=int(args.window_size),
    )
    h_wang_rows = read_rows_csv(args.h_wang_row_file)
    materialized_h_wang_rows = _materialize_filtered_rows(
        h_wang_rows,
        query_view=str(args.query_view),
        window_size=int(args.window_size),
    )

    h_wang_train_data = _prepare_training_data(
        materialized_dataset_rows,
        feature_columns=full_feature_columns,
        seed=int(args.seed),
    )
    h_wang_eval_data = _prepare_evaluation_data(
        materialized_h_wang_rows,
        feature_columns=full_feature_columns,
    )

    heldout_splits: list[PreparedHeldoutSplit] = []
    for heldout_dataset in [str(value) for value in args.datasets]:
        heldout_eval_rows = select_rows(materialized_dataset_rows, datasets=[heldout_dataset])
        heldout_train_rows = select_rows(
            materialized_dataset_rows,
            datasets=[dataset_name for dataset_name in args.datasets if str(dataset_name) != heldout_dataset],
        )
        heldout_splits.append(
            PreparedHeldoutSplit(
                heldout_dataset=str(heldout_dataset),
                train_data=_prepare_training_data(
                    heldout_train_rows,
                    feature_columns=full_feature_columns,
                    seed=int(args.seed),
                ),
                eval_data=_prepare_evaluation_data(
                    heldout_eval_rows,
                    feature_columns=full_feature_columns,
                ),
            )
        )

    feature_index = {column: index for index, column in enumerate(full_feature_columns)}
    current_features = tuple(full_feature_columns)
    trajectory: list[dict[str, Any]] = []
    candidate_scan_rows: list[dict[str, Any]] = []

    baseline_accuracy, baseline_fit_seconds, baseline_predict_seconds = _evaluate_subset_accuracy(
        train_data=h_wang_train_data,
        eval_data=h_wang_eval_data,
        fixed_params=fixed_params,
        seed=int(args.seed),
        lgbm_n_jobs=int(args.lgbm_n_jobs),
        feature_indices=_column_indices(current_features, feature_index=feature_index),
    )
    baseline_heldout = _evaluate_heldout_splits(
        splits=heldout_splits,
        feature_columns=current_features,
        feature_index=feature_index,
        fixed_params=fixed_params,
        seed=int(args.seed),
        lgbm_n_jobs=int(args.lgbm_n_jobs),
        max_workers=int(args.max_workers),
    )
    trajectory.append(
        _trajectory_row(
            step_index=0,
            removed_feature=None,
            feature_columns=current_features,
            h_wang_accuracy=baseline_accuracy,
            heldout_summary=baseline_heldout,
        )
    )
    candidate_scan_rows.append(
        {
            "step_index": 0,
            "rank": 1,
            "removed_feature": None,
            "h_wang_top50_accuracy": round(float(baseline_accuracy), 6),
            "remaining_feature_count": int(len(current_features)),
            "fit_seconds": round(float(baseline_fit_seconds), 6),
            "predict_seconds": round(float(baseline_predict_seconds), 6),
            "remaining_features": "|".join(current_features),
        }
    )
    _write_trajectory_csv(args.output_dir / "accepted_trajectory.csv", trajectory)
    _write_group_csv(args.output_dir / "candidate_scans.csv", candidate_scan_rows)

    step_index = 0
    while len(current_features) > int(args.min_features):
        step_index += 1
        step_start = time.perf_counter()
        candidate_specs = [
            (
                position,
                feature_name,
                tuple(column for column in current_features if column != feature_name),
            )
            for position, feature_name in enumerate(current_features)
        ]
        with ThreadPoolExecutor(max_workers=min(int(args.max_workers), len(candidate_specs))) as executor:
            step_results = list(
                executor.map(
                    lambda spec: _score_candidate_removal(
                        position=int(spec[0]),
                        removed_feature=str(spec[1]),
                        remaining_features=tuple(spec[2]),
                        feature_index=feature_index,
                        train_data=h_wang_train_data,
                        eval_data=h_wang_eval_data,
                        fixed_params=fixed_params,
                        seed=int(args.seed),
                        lgbm_n_jobs=int(args.lgbm_n_jobs),
                    ),
                    candidate_specs,
                )
            )
        candidate_scan_rows.extend(_candidate_result_rows(step_index=step_index, results=step_results))
        accepted = _choose_best_candidate_result(step_results)
        heldout_summary = _evaluate_heldout_splits(
            splits=heldout_splits,
            feature_columns=accepted.remaining_features,
            feature_index=feature_index,
            fixed_params=fixed_params,
            seed=int(args.seed),
            lgbm_n_jobs=int(args.lgbm_n_jobs),
            max_workers=int(args.max_workers),
        )
        trajectory_row = _trajectory_row(
            step_index=step_index,
            removed_feature=accepted.removed_feature,
            feature_columns=accepted.remaining_features,
            h_wang_accuracy=accepted.h_wang_accuracy,
            heldout_summary=heldout_summary,
        )
        trajectory_row["step_seconds"] = round(float(time.perf_counter() - step_start), 6)
        trajectory.append(trajectory_row)
        current_features = tuple(accepted.remaining_features)
        _write_trajectory_csv(args.output_dir / "accepted_trajectory.csv", trajectory)
        _write_group_csv(args.output_dir / "candidate_scans.csv", candidate_scan_rows)

    _write_trajectory_plot(args.output_dir / "accepted_trajectory.png", trajectory)
    best_h_wang_row = max(
        trajectory,
        key=lambda row: (float(row["h_wang_top50_accuracy"]), -int(row["step_index"])),
    )
    summary = {
        "config": {
            "dataset_root": str(args.dataset_root),
            "datasets": [str(value) for value in args.datasets],
            "rows_source": str(args.rows_source),
            "rows_source_by_dataset": dict(rows_source_by_dataset),
            "query_view": str(args.query_view),
            "window_size": int(args.window_size),
            "feature_preset": str(args.feature_preset),
            "fixed_summary_json": str(args.fixed_summary_json),
            "seed": int(args.seed),
            "lgbm_n_jobs": int(args.lgbm_n_jobs),
            "max_workers": int(args.max_workers),
            "min_features": int(args.min_features),
            "objective": "lambdarank",
            "selection_metric": "blind_h_wang_top50_accuracy",
            "diagnostic_metric": "heldout_s2and_leave_one_dataset_out_accuracy",
        },
        "fixed_params": dict(fixed_params),
        "baseline": dict(trajectory[0]),
        "best_h_wang_step": dict(best_h_wang_row),
        "final_step": dict(trajectory[-1]),
        "trajectory": list(trajectory),
        "data_summary": {
            "full_train_rows_seen": int(h_wang_train_data.rows_seen),
            "full_train_rows_used": int(h_wang_train_data.rows_used),
            "full_train_groups_used": int(h_wang_train_data.groups_used),
            "full_train_dropped_all_negative_group_count": int(h_wang_train_data.dropped_all_negative_group_count),
            "h_wang_queries": int(len(h_wang_eval_data.group_sizes)),
            "heldout_queries_total": int(sum(len(split.eval_data.group_sizes) for split in heldout_splits)),
        },
        "artifacts": {
            "accepted_trajectory_csv": str(args.output_dir / "accepted_trajectory.csv"),
            "candidate_scans_csv": str(args.output_dir / "candidate_scans.csv"),
            "accepted_trajectory_plot": str(args.output_dir / "accepted_trajectory.png"),
        },
        "elapsed_seconds": round(float(time.perf_counter() - start), 6),
    }
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()

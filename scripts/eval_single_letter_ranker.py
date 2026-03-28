"""Evaluate an S2AND-only LightGBM ranker on held-out datasets and blind h_wang."""

from __future__ import annotations

import argparse
import copy
import csv
import pickle
import random
import statistics
import time
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from hyperopt import Trials, fmin, space_eval, tpe
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupShuffleSplit

from s2and import shap_utils
from s2and.model_pairwise import PairwiseModeler, intify

try:
    from scripts.single_letter_reranker_utils import (
        DEFAULT_H_WANG_WINDOW_SENSITIVITY,
        DEFAULT_LABELED_DATASETS,
        ENRICHMENT_PROFILES,
        FEATURE_PRESETS,
        build_feature_matrix,
        build_training_matrix,
        choose_generic_heuristic,
        choose_retrieval_top1,
        group_rows,
        read_materialized_rows_csv,
        read_rows_csv,
        resolve_feature_columns,
        select_rows,
        write_json,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from single_letter_reranker_utils import (  # type: ignore
        DEFAULT_H_WANG_WINDOW_SENSITIVITY,
        DEFAULT_LABELED_DATASETS,
        ENRICHMENT_PROFILES,
        FEATURE_PRESETS,
        build_feature_matrix,
        build_training_matrix,
        choose_generic_heuristic,
        choose_retrieval_top1,
        group_rows,
        read_materialized_rows_csv,
        read_rows_csv,
        resolve_feature_columns,
        select_rows,
        write_json,
    )


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRanker was fitted with feature names",
)

TRAINING_SOURCE_MODES = ("s2and_only", "h_wang_only", "mixed")


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank <= 1:
        return "1"
    if rank <= 3:
        return "2_3"
    if rank <= 10:
        return "4_10"
    if rank <= 25:
        return "11_25"
    if rank <= 50:
        return "26_50"
    if rank <= 100:
        return "51_100"
    return "100_plus"


def _resolve_query_views(*, query_view: str, query_views: list[str] | None) -> list[str]:
    """Resolve the effective ordered query-view list."""

    raw_values = list(query_views) if query_views else [str(query_view)]
    ordered: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _query_base_group_id(row: dict[str, Any]) -> str:
    """Return the base query identity used to keep mixed views together."""

    return f"{str(row.get('query_source', row['source']))}:" f"{str(row['dataset'])}:" f"{str(row['query_id'])}"


def _read_string_id_file(path: Path) -> set[str]:
    """Read a newline-delimited query-group selection file."""

    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError(f"Expected at least one query-group ID in {path}")
    return values


def _resolve_hyperopt_evals(*, requested_hyperopt_evals: int | None, run_mode: str) -> int:
    """Resolve the effective hyperopt budget for the selected run mode."""

    if requested_hyperopt_evals is not None:
        return int(requested_hyperopt_evals)
    if str(run_mode) == "screen":
        return 0
    return 20


def _resolve_rows_source(dataset_dir: Path, *, rows_source: str) -> tuple[Path, str]:
    """Resolve which cached row file to load for one dataset."""

    derived_path = dataset_dir / "rows_derived.csv"
    base_path = dataset_dir / "rows.csv"
    if str(rows_source) == "derived":
        if not derived_path.exists():
            raise FileNotFoundError(f"Missing derived row cache: {derived_path}")
        return derived_path, "derived"
    if str(rows_source) == "base":
        return base_path, "base"
    if derived_path.exists():
        return derived_path, "derived"
    return base_path, "base"


def _ordered_group_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    grouped = group_rows(rows)
    ordered: list[dict[str, Any]] = []
    group_sizes: list[int] = []
    group_ids: list[str] = []
    for group_id in sorted(grouped):
        group_rows_for_id = grouped[group_id]
        ordered.extend(group_rows_for_id)
        group_sizes.append(int(len(group_rows_for_id)))
        group_ids.append(str(group_id))
    return ordered, group_sizes, group_ids


def _training_matrix_group_sizes(training_matrix: Any) -> list[int]:
    sizes: list[int] = []
    for group_id in training_matrix.group_ids:
        repeat_count = int(training_matrix.group_repeat_counts.get(str(group_id), 1))
        group_size = int(training_matrix.kept_group_sizes[str(group_id)])
        sizes.extend([group_size] * repeat_count)
    if sum(sizes) != len(training_matrix.ordered_rows):
        raise RuntimeError(
            "Training group sizes did not match ordered rows: "
            f"sum_group_sizes={sum(sizes)} ordered_rows={len(training_matrix.ordered_rows)}"
        )
    return sizes


def _predict_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    return np.asarray(model.predict(features), dtype=np.float64)


def _top1_accuracy(scores: np.ndarray, labels: np.ndarray, group_sizes: list[int]) -> float:
    correct = 0
    start = 0
    for group_size in group_sizes:
        end = start + int(group_size)
        group_scores = scores[start:end]
        group_labels = labels[start:end]
        winner = int(np.argmax(group_scores))
        correct += int(group_labels[winner] == 1)
        start = end
    if start != len(labels):
        raise RuntimeError(f"Expected grouped labels to consume all rows; consumed={start} rows={len(labels)}")
    return float(correct / max(1, len(group_sizes)))


def _pairwise_search_space() -> dict[str, Any]:
    modeler = PairwiseModeler(n_iter=1)
    return copy.deepcopy(modeler.search_space)


def _fit_ranker_with_hyperopt(
    *,
    training_matrix: Any,
    validation_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    seed: int,
    hyperopt_evals: int,
    n_jobs: int,
) -> tuple[Any, dict[str, Any]]:
    validation_ordered_rows, validation_group_sizes, _validation_group_ids = _ordered_group_rows(validation_rows)
    validation_features = build_feature_matrix(validation_ordered_rows, feature_columns=feature_columns)
    validation_labels = np.asarray([int(row["label"]) for row in validation_ordered_rows], dtype=np.int32)
    train_group_sizes = _training_matrix_group_sizes(training_matrix)
    train_features = training_matrix.features
    train_labels = training_matrix.labels
    train_weights = training_matrix.sample_weights

    best_params: dict[str, Any] = {}
    best_validation_accuracy: float | None = None
    trials_count = 0
    trials: Trials | None = None
    search_space = _pairwise_search_space()

    def build_ranker(params: dict[str, Any]) -> Any:
        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            label_gain=[0, 1],
            n_jobs=int(n_jobs),
            verbose=-1,
            tree_learner="data",
            random_state=int(seed),
            data_random_seed=int(seed),
            feature_fraction_seed=int(seed),
        )
        ranker.set_params(**params)
        return ranker

    if hyperopt_evals > 0:

        def objective(params: dict[str, Any]) -> float:
            resolved = {key: intify(value) for key, value in params.items()}
            ranker = build_ranker(resolved)
            ranker.fit(
                train_features,
                train_labels,
                group=train_group_sizes,
                sample_weight=train_weights,
            )
            validation_scores = _predict_scores(ranker, validation_features)
            return -_top1_accuracy(validation_scores, validation_labels, validation_group_sizes)

        trials = Trials()
        _ = fmin(
            fn=objective,
            space=search_space,
            algo=partial(tpe.suggest, n_startup_jobs=5),
            max_evals=int(hyperopt_evals),
            trials=trials,
            rstate=np.random.default_rng(int(seed)),
        )
        best_params = {key: intify(value) for key, value in space_eval(search_space, trials.argmin).items()}
        trials_count = int(len(trials.trials))
        best_validation_accuracy = float(-trials.best_trial["result"]["loss"])

    ranker = build_ranker(best_params)
    start = time.perf_counter()
    ranker.fit(
        train_features,
        train_labels,
        group=train_group_sizes,
        sample_weight=train_weights,
    )
    train_seconds = float(time.perf_counter() - start)
    summary = {
        "model_type": "lgbm_ranker",
        "best_params": dict(best_params),
        "best_validation_accuracy": round(float(best_validation_accuracy), 6)
        if best_validation_accuracy is not None
        else None,
        "hyperopt_evals": int(hyperopt_evals),
        "hyperopt_trials_ran": int(trials_count),
        "train_seconds": round(float(train_seconds), 6),
    }
    return ranker, summary


def _evaluate_rows(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    feature_preset: str,
    window_size: int,
    fold_index: int,
    train_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped = group_rows(rows)
    per_group_rows: list[dict[str, Any]] = []
    for group_id, group_values in grouped.items():
        eligible_rows = [row for row in group_values if int(row["retrieval_rank"]) <= int(window_size)]
        if not eligible_rows:
            continue
        features = build_feature_matrix(eligible_rows, feature_preset=feature_preset)
        predict_start = time.perf_counter()
        scores = _predict_scores(model, features)
        predict_seconds = float(time.perf_counter() - predict_start)
        ranked_pairs = sorted(
            zip(scores.tolist(), eligible_rows, strict=True),
            key=lambda item: (-float(item[0]), int(item[1]["retrieval_rank"]), str(item[1]["candidate_component_key"])),
        )
        top3_hit = int(any(int(row["label"]) == 1 for _score, row in ranked_pairs[:3]))
        chosen_score, chosen_row = ranked_pairs[0]
        has_runner_up = int(len(ranked_pairs) > 1)
        second_score = float(ranked_pairs[1][0]) if has_runner_up else None
        retrieval_top1 = choose_retrieval_top1(eligible_rows, window_size=window_size)
        heuristic_choice, _heuristic_score = choose_generic_heuristic(eligible_rows, window_size=window_size)
        per_group_rows.append(
            {
                "fold_index": int(fold_index),
                "dataset": str(chosen_row["dataset"]),
                "query_source": str(chosen_row.get("query_source", chosen_row["source"])),
                "query_view": str(chosen_row["query_view"]),
                "natural_query_view": str(chosen_row.get("natural_query_view", chosen_row["query_view"])),
                "window_size": int(window_size),
                "query_group_id": str(group_id),
                "query_id": str(chosen_row["query_id"]),
                "support_type": str(chosen_row["support_type"]),
                "supervision_type": str(chosen_row.get("supervision_type", "labeled")),
                "split": str(chosen_row.get("split", "all")),
                "query_in_seed_before_holdout": int(chosen_row.get("query_in_seed_before_holdout", 0)),
                "candidate_count": int(len(eligible_rows)),
                "group_has_positive": int(chosen_row["group_has_positive"]),
                "best_positive_retrieval_rank": (
                    int(chosen_row["best_positive_retrieval_rank"])
                    if chosen_row["best_positive_retrieval_rank"] is not None
                    else None
                ),
                "best_positive_rank_bucket": _rank_bucket(chosen_row["best_positive_retrieval_rank"]),
                "model_prediction": str(chosen_row["candidate_component_key"]),
                "model_correct": int(chosen_row["label"]),
                "model_top3_hit": int(top3_hit),
                "model_score": round(float(chosen_score), 6),
                "model_margin": (
                    round(float(chosen_score - float(second_score)), 6) if second_score is not None else None
                ),
                "has_runner_up": int(has_runner_up),
                "retrieval_top1_prediction": retrieval_top1,
                "retrieval_top1_correct": int(
                    any(
                        str(row["candidate_component_key"]) == str(retrieval_top1) and int(row["label"]) == 1
                        for row in eligible_rows
                    )
                ),
                "heuristic_prediction": heuristic_choice,
                "heuristic_correct": int(
                    any(
                        str(row["candidate_component_key"]) == str(heuristic_choice) and int(row["label"]) == 1
                        for row in eligible_rows
                    )
                ),
                "predict_seconds": round(float(predict_seconds), 6),
                "train_seconds": float(train_summary["train_seconds"]),
            }
        )
    return per_group_rows


def _summarize_per_group_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_bucket[str(row["best_positive_rank_bucket"])].append(int(row["model_correct"]))
    return {
        "queries": int(len(rows)),
        "accuracy": round(float(statistics.mean(int(row["model_correct"]) for row in rows)), 6) if rows else 0.0,
        "top3_accuracy": round(float(statistics.mean(int(row["model_top3_hit"]) for row in rows)), 6) if rows else 0.0,
        "retrieval_top1_accuracy": round(
            float(statistics.mean(int(row["retrieval_top1_correct"]) for row in rows)),
            6,
        )
        if rows
        else 0.0,
        "heuristic_accuracy": round(float(statistics.mean(int(row["heuristic_correct"]) for row in rows)), 6)
        if rows
        else 0.0,
        "improvements_vs_heuristic": int(
            sum(1 for row in rows if int(row["model_correct"]) == 1 and int(row["heuristic_correct"]) == 0)
        ),
        "regressions_vs_heuristic": int(
            sum(1 for row in rows if int(row["model_correct"]) == 0 and int(row["heuristic_correct"]) == 1)
        ),
        "accuracy_by_supported_rank_bucket": {
            bucket: round(float(statistics.mean(values)), 6) for bucket, values in sorted(by_bucket.items())
        },
        "mean_candidate_count": round(float(statistics.mean(int(row["candidate_count"]) for row in rows)), 6)
        if rows
        else 0.0,
        "mean_predict_seconds": round(float(statistics.mean(float(row["predict_seconds"]) for row in rows)), 6)
        if rows
        else 0.0,
        "mean_train_seconds": round(float(statistics.mean(float(row["train_seconds"]) for row in rows)), 6)
        if rows
        else 0.0,
    }


def _count_query_groups(rows: list[dict[str, Any]]) -> int:
    """Count persisted query groups in a row slice."""

    return int(len(group_rows(rows)))


def _select_h_wang_any_input_rows(
    rows: list[dict[str, Any]],
    *,
    query_views: list[str],
    splits: list[str] | None = None,
    supervision_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select the any-input `h_wang` rows needed for training or dev."""

    return select_rows(
        rows,
        query_views=query_views,
        query_sources=["orcid_any_input"],
        splits=splits,
        supervision_types=supervision_types,
    )


def _compose_training_rows(
    *,
    s2and_rows: list[dict[str, Any]],
    h_wang_rows: list[dict[str, Any]],
    training_source_mode: str,
) -> list[dict[str, Any]]:
    """Assemble the requested training source mixture."""

    if str(training_source_mode) == "s2and_only":
        return list(s2and_rows)
    if str(training_source_mode) == "h_wang_only":
        return list(h_wang_rows)
    if str(training_source_mode) == "mixed":
        return [*s2and_rows, *h_wang_rows]
    raise ValueError(f"Unknown training_source_mode: {training_source_mode}")


def _summarize_training_source_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the training-source composition."""

    by_source: dict[str, dict[str, int]] = {}
    for source in sorted({str(row["source"]) for row in rows}):
        source_rows = [row for row in rows if str(row["source"]) == source]
        by_source[source] = {
            "rows": int(len(source_rows)),
            "query_groups": _count_query_groups(source_rows),
        }
    return {
        "rows": int(len(rows)),
        "query_groups": _count_query_groups(rows),
        "by_source": by_source,
    }


def _reject_threshold_eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that have a real runner-up margin for thresholding."""

    return [row for row in rows if int(row.get("has_runner_up", 1)) == 1 and row.get("model_margin") is not None]


def _score_reject_threshold_core(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    """Score one reject-all threshold over dev per-group rows."""

    eligible_rows = _reject_threshold_eligible_rows(rows)
    positives = [row for row in eligible_rows if str(row["supervision_type"]) == "positive_repeat_orcid"]
    negatives = [row for row in eligible_rows if str(row["supervision_type"]) == "negative_singleton_orcid"]
    accepted = {str(row["query_group_id"]): float(row["model_margin"]) > float(threshold) for row in eligible_rows}
    positive_correct = sum(
        int(accepted[str(row["query_group_id"])] and int(row["model_correct"]) == 1) for row in positives
    )
    negative_correct = sum(int(not accepted[str(row["query_group_id"])]) for row in negatives)
    positive_accuracy = float(positive_correct / len(positives)) if positives else 0.0
    negative_reject_accuracy = float(negative_correct / len(negatives)) if negatives else 0.0
    if positives and negatives:
        balanced_accuracy = float((positive_accuracy + negative_reject_accuracy) / 2.0)
    elif positives:
        balanced_accuracy = float(positive_accuracy)
    elif negatives:
        balanced_accuracy = float(negative_reject_accuracy)
    else:
        balanced_accuracy = 0.0
    total_correct = int(positive_correct + negative_correct)
    return {
        "threshold": round(float(threshold), 6),
        "queries": int(len(rows)),
        "eligible_queries": int(len(eligible_rows)),
        "singleton_candidate_group_count": int(len(rows) - len(eligible_rows)),
        "positive_queries": int(len(positives)),
        "negative_queries": int(len(negatives)),
        "balanced_accuracy": round(float(balanced_accuracy), 6),
        "overall_accuracy": round(float(total_correct / len(eligible_rows)), 6) if eligible_rows else 0.0,
        "positive_accuracy": round(float(positive_accuracy), 6) if positives else None,
        "negative_reject_accuracy": round(float(negative_reject_accuracy), 6) if negatives else None,
        "rejection_rate": round(
            float(sum(int(not accepted[str(row["query_group_id"])]) for row in eligible_rows) / len(eligible_rows)),
            6,
        )
        if eligible_rows
        else 0.0,
        "positive_accept_rate": round(
            float(sum(int(accepted[str(row["query_group_id"])]) for row in positives) / len(positives)),
            6,
        )
        if positives
        else None,
    }


def _select_reject_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sweep a reject-all threshold and keep the best dev setting."""

    if not rows:
        return {
            "threshold": 0.0,
            "queries": 0,
            "eligible_queries": 0,
            "singleton_candidate_group_count": 0,
            "positive_queries": 0,
            "negative_queries": 0,
            "balanced_accuracy": 0.0,
            "overall_accuracy": 0.0,
            "positive_accuracy": None,
            "negative_reject_accuracy": None,
            "rejection_rate": 0.0,
            "positive_accept_rate": None,
            "per_view": {},
        }
    eligible_rows = _reject_threshold_eligible_rows(rows)
    if not eligible_rows:
        summary = _score_reject_threshold_core(rows, threshold=0.0)
        summary["per_view"] = {
            query_view: _score_reject_threshold_core(
                [row for row in rows if str(row["query_view"]) == query_view],
                threshold=0.0,
            )
            for query_view in sorted({str(row["query_view"]) for row in rows})
        }
        return summary
    margins = sorted({float(row["model_margin"]) for row in eligible_rows})
    epsilon = 1e-6
    candidate_thresholds = [margins[0] - epsilon, *margins, margins[-1] + epsilon]
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in candidate_thresholds:
        metrics = _score_reject_threshold_core(rows, threshold=float(threshold))
        ranking_key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["positive_accuracy"] if metrics["positive_accuracy"] is not None else -1.0),
            -float(metrics["rejection_rate"]),
            -float(metrics["threshold"]),
        )
        if best_key is None or ranking_key > best_key:
            best_key = ranking_key
            best_metrics = metrics
    assert best_metrics is not None
    best_metrics["per_view"] = {}
    for query_view in sorted({str(row["query_view"]) for row in rows}):
        best_metrics["per_view"][query_view] = _score_reject_threshold_core(
            [row for row in rows if str(row["query_view"]) == query_view],
            threshold=float(best_metrics["threshold"]),
        )
    return best_metrics


def _summarize_negative_singletons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize singleton-negative dev groups."""

    eligible_rows = _reject_threshold_eligible_rows(rows)
    return {
        "queries": int(len(rows)),
        "eligible_margin_queries": int(len(eligible_rows)),
        "singleton_candidate_group_count": int(len(rows) - len(eligible_rows)),
        "mean_margin": (
            round(float(statistics.mean(float(row["model_margin"]) for row in eligible_rows)), 6)
            if eligible_rows
            else None
        ),
        "mean_candidate_count": round(float(statistics.mean(int(row["candidate_count"]) for row in rows)), 6)
        if rows
        else 0.0,
    }


def _load_dataset_rows(
    dataset_root: Path,
    datasets: list[str],
    *,
    rows_source: str,
    selected_query_group_ids: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    rows_source_by_dataset: dict[str, str] = {}
    for dataset_name in datasets:
        dataset_dir = dataset_root / dataset_name
        rows_path, resolved_source = _resolve_rows_source(dataset_dir, rows_source=rows_source)
        dataset_rows = (
            read_materialized_rows_csv(rows_path) if resolved_source == "derived" else read_rows_csv(rows_path)
        )
        rows.extend(
            select_rows(
                dataset_rows,
                selected_query_group_ids=selected_query_group_ids,
            )
        )
        rows_source_by_dataset[str(dataset_name)] = str(resolved_source)
    return rows, rows_source_by_dataset


def _write_group_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sample_rows_for_shap(rows: list[dict[str, Any]], *, max_rows: int, seed: int) -> list[dict[str, Any]]:
    """Sample whole query groups reproducibly for SHAP."""

    if max_rows <= 0 or len(rows) <= int(max_rows):
        return list(rows)
    grouped = group_rows(rows)
    group_ids = sorted(grouped)
    rng = random.Random(int(seed))
    rng.shuffle(group_ids)
    sampled_rows: list[dict[str, Any]] = []
    for group_id in group_ids:
        candidate_rows = list(grouped[str(group_id)])
        if sampled_rows and (len(sampled_rows) + len(candidate_rows)) > int(max_rows):
            continue
        sampled_rows.extend(candidate_rows)
        if len(sampled_rows) >= int(max_rows):
            break
    if sampled_rows:
        return sampled_rows
    first_group_id = str(group_ids[0])
    return list(grouped[first_group_id])


def _ranker_shap_values(model: Any, features: np.ndarray) -> np.ndarray:
    """Compute SHAP values for a fitted LightGBM ranker."""

    shap_module = shap_utils.shap
    explainer = shap_module.TreeExplainer(model)
    values = explainer.shap_values(features)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    if hasattr(values, "values"):
        values = values.values
    return np.asarray(values, dtype=np.float32)


def _write_pooled_shap_artifacts(
    *,
    shap_runs: list[dict[str, Any]],
    feature_preset: str,
    output_dir: Path,
    max_rows: int,
    seed: int,
    shap_plot_type: str,
) -> dict[str, Any]:
    """Write one pooled out-of-fold SHAP beeswarm from held-out S2AND rows."""

    if not shap_runs:
        return {
            "enabled": False,
            "reason": "no_shap_runs",
        }
    feature_columns = resolve_feature_columns(feature_preset=feature_preset)
    shap_dir = output_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    ordered_runs = sorted(shap_runs, key=lambda row: str(row["heldout_dataset"]))
    base_quota = max(1, int(max_rows) // len(ordered_runs))
    extra = max(0, int(max_rows) - (base_quota * len(ordered_runs)))
    sampled_records: list[dict[str, Any]] = []
    pooled_features: list[np.ndarray] = []
    pooled_values: list[np.ndarray] = []
    selected_rows_total = 0

    for index, run in enumerate(ordered_runs):
        quota = base_quota + (1 if index < extra else 0)
        sampled_rows = _sample_rows_for_shap(
            list(run["rows"]),
            max_rows=quota,
            seed=int(seed) + index,
        )
        if not sampled_rows:
            continue
        features = build_feature_matrix(sampled_rows, feature_columns=feature_columns)
        shap_values = _ranker_shap_values(run["model"], features)
        pooled_features.append(np.asarray(features, dtype=np.float32))
        pooled_values.append(np.asarray(shap_values, dtype=np.float32))
        selected_rows_total += len(sampled_rows)
        for row, feature_row in zip(sampled_rows, features.tolist(), strict=True):
            record = {
                "heldout_dataset": str(run["heldout_dataset"]),
                "query_group_id": str(row["query_group_id"]),
                "query_id": str(row["query_id"]),
                "candidate_component_key": str(row["candidate_component_key"]),
                "retrieval_rank": int(row["retrieval_rank"]),
                "label": int(row["label"]),
            }
            for column, value in zip(feature_columns, feature_row, strict=True):
                record[str(column)] = float(value)
            sampled_records.append(record)

    if not pooled_features or not pooled_values:
        return {
            "enabled": False,
            "reason": "no_rows_selected",
        }

    feature_matrix = np.vstack(pooled_features)
    shap_matrix = np.vstack(pooled_values)
    plot_path = shap_dir / "pooled_validation_shap.png"
    shap_utils._safe_summary_plot(  # noqa: SLF001
        shap_matrix,
        feature_matrix,
        feature_columns,
        shap_plot_type,
        str(plot_path),
    )
    rows_path = shap_dir / "pooled_validation_rows.csv"
    _write_group_csv(rows_path, sampled_records)
    values_path = shap_dir / "pooled_validation_shap_values.npz"
    np.savez_compressed(
        values_path,
        shap_values=shap_matrix,
        features=feature_matrix,
        feature_names=np.asarray(feature_columns, dtype=object),
    )
    return {
        "enabled": True,
        "selected_rows": int(selected_rows_total),
        "feature_count": int(len(feature_columns)),
        "plot_type": str(shap_plot_type),
        "plot_path": str(plot_path),
        "rows_path": str(rows_path),
        "values_path": str(values_path),
        "heldout_datasets": [str(run["heldout_dataset"]) for run in ordered_runs],
    }


def _fit_ranker_for_split(
    *,
    train_rows: list[dict[str, Any]],
    query_views: list[str],
    window_size: int,
    seed: int,
    feature_preset: str,
    enrichment_profile: str,
    enrichment_rounds: int,
    hyperopt_evals: int,
    inner_validation_fraction: float,
    n_jobs: int,
) -> tuple[Any, dict[str, Any]]:
    filtered_rows = select_rows(train_rows, query_views=query_views, window_size=window_size)
    grouped = group_rows(filtered_rows)
    base_group_ids = sorted({_query_base_group_id(group_rows_for_id[0]) for group_rows_for_id in grouped.values()})
    if len(base_group_ids) < 2:
        raise RuntimeError(f"Need at least two base train groups for ranker fit; found {len(base_group_ids)}")
    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=float(inner_validation_fraction),
        random_state=int(seed),
    )
    inner_train_index, inner_validation_index = next(
        inner_splitter.split(
            [[0]] * len(base_group_ids),
            [0] * len(base_group_ids),
            groups=base_group_ids,
        )
    )
    inner_train_ids = {str(base_group_ids[idx]) for idx in inner_train_index}
    inner_validation_ids = {str(base_group_ids[idx]) for idx in inner_validation_index}
    search_train_rows = [row for row in filtered_rows if _query_base_group_id(row) in inner_train_ids]
    search_validation_rows = [row for row in filtered_rows if _query_base_group_id(row) in inner_validation_ids]
    feature_columns = resolve_feature_columns(feature_preset=feature_preset)

    search_training_matrix = build_training_matrix(
        search_train_rows,
        seed=int(seed),
        feature_columns=feature_columns,
        enrichment_profile=enrichment_profile,
        enrichment_rounds=int(enrichment_rounds),
    )
    if search_training_matrix.features.shape[0] == 0:
        raise RuntimeError("No trainable rows remained in search training matrix")

    outer_training_matrix = build_training_matrix(
        filtered_rows,
        seed=int(seed),
        feature_columns=feature_columns,
        enrichment_profile=enrichment_profile,
        enrichment_rounds=int(enrichment_rounds),
    )
    if outer_training_matrix.features.shape[0] == 0:
        raise RuntimeError("No trainable rows remained in outer training matrix")

    _search_model, hyperopt_summary = _fit_ranker_with_hyperopt(
        training_matrix=search_training_matrix,
        validation_rows=search_validation_rows,
        feature_columns=feature_columns,
        seed=int(seed),
        hyperopt_evals=int(hyperopt_evals),
        n_jobs=int(n_jobs),
    )
    best_params = dict(hyperopt_summary["best_params"])
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_jobs=int(n_jobs),
        verbose=-1,
        tree_learner="data",
        random_state=int(seed),
        data_random_seed=int(seed),
        feature_fraction_seed=int(seed),
        **best_params,
    )
    ranker.fit(
        outer_training_matrix.features,
        outer_training_matrix.labels,
        group=_training_matrix_group_sizes(outer_training_matrix),
        sample_weight=outer_training_matrix.sample_weights,
    )
    train_summary = {
        "query_views": [str(value) for value in query_views],
        "window_size": int(window_size),
        "feature_preset": str(feature_preset),
        "feature_columns": list(feature_columns),
        "enrichment_profile": str(enrichment_profile),
        "enrichment_rounds": int(enrichment_rounds),
        "rows_seen": int(len(filtered_rows)),
        "rows_used": int(outer_training_matrix.features.shape[0]),
        "groups_used": int(len(outer_training_matrix.group_ids)),
        "dropped_all_negative_group_count": int(len(outer_training_matrix.dropped_all_negative_group_ids)),
        "groups_with_extra_copies": int(outer_training_matrix.groups_with_extra_copies),
        "extra_group_copies": int(outer_training_matrix.extra_group_copies),
        "positive_rows_used": int(outer_training_matrix.labels.sum()),
        "positive_rate_used": round(float(outer_training_matrix.labels.mean()), 6),
        "mean_group_size": round(float(statistics.mean(outer_training_matrix.kept_group_sizes.values())), 6),
        "hyperopt": dict(hyperopt_summary),
        "best_params": best_params,
        "train_seconds": float(hyperopt_summary["train_seconds"]),
    }
    return ranker, train_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_LABELED_DATASETS))
    parser.add_argument("--query-view", default="initial_only")
    parser.add_argument("--query-views", nargs="+", default=None)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--feature-preset", default="generalized_v8", choices=sorted(FEATURE_PRESETS))
    parser.add_argument("--enrichment-profile", default="none", choices=list(ENRICHMENT_PROFILES))
    parser.add_argument("--enrichment-rounds", type=int, default=0)
    parser.add_argument("--training-source-mode", choices=TRAINING_SOURCE_MODES, default="s2and_only")
    parser.add_argument("--hyperopt-evals", type=int, default=None)
    parser.add_argument("--run-mode", choices=("full", "screen"), default="full")
    parser.add_argument("--rows-source", choices=("auto", "base", "derived"), default="auto")
    parser.add_argument("--selected-query-groups-file", type=Path, default=None)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.2)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--h-wang-row-file", type=Path, default=None)
    parser.add_argument("--h-wang-window-sizes", nargs="+", type=int, default=list(DEFAULT_H_WANG_WINDOW_SENSITIVITY))
    parser.add_argument("--write-shap", action="store_true")
    parser.add_argument("--shap-max-rows", type=int, default=6000)
    parser.add_argument("--shap-plot-type", default="dot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.run_mode) == "screen" and args.write_shap:
        raise ValueError("Screen mode does not support --write-shap")
    query_views = _resolve_query_views(query_view=str(args.query_view), query_views=args.query_views)
    resolved_hyperopt_evals = _resolve_hyperopt_evals(
        requested_hyperopt_evals=args.hyperopt_evals,
        run_mode=str(args.run_mode),
    )
    selected_query_group_ids = (
        _read_string_id_file(args.selected_query_groups_file) if args.selected_query_groups_file is not None else None
    )
    dataset_rows, rows_source_by_dataset = _load_dataset_rows(
        args.dataset_root,
        [str(value) for value in args.datasets],
        rows_source=str(args.rows_source),
        selected_query_group_ids=selected_query_group_ids,
    )
    h_wang_rows = read_rows_csv(args.h_wang_row_file) if args.h_wang_row_file is not None else []
    h_wang_any_input_rows = select_rows(h_wang_rows, query_sources=["orcid_any_input"]) if h_wang_rows else []
    h_wang_train_rows = _select_h_wang_any_input_rows(
        h_wang_rows,
        query_views=query_views,
        splits=["train"],
        supervision_types=["positive_repeat_orcid"],
    )
    if str(args.training_source_mode) in {"h_wang_only", "mixed"} and not h_wang_train_rows:
        raise ValueError(f"training_source_mode={args.training_source_mode!r} requires any-input `h_wang` train rows")
    write_shap = bool(args.write_shap)
    save_models = str(args.run_mode) == "full"

    heldout_summary: dict[str, Any] = {}
    all_per_group_rows: list[dict[str, Any]] = []
    shap_runs: list[dict[str, Any]] = []

    for heldout_dataset in [str(value) for value in args.datasets]:
        train_datasets = [dataset_name for dataset_name in args.datasets if str(dataset_name) != heldout_dataset]
        s2and_train_rows = select_rows(dataset_rows, datasets=train_datasets)
        train_rows = _compose_training_rows(
            s2and_rows=s2and_train_rows,
            h_wang_rows=h_wang_train_rows,
            training_source_mode=str(args.training_source_mode),
        )
        eval_rows = select_rows(dataset_rows, datasets=[heldout_dataset], query_views=query_views)
        model, train_summary = _fit_ranker_for_split(
            train_rows=train_rows,
            query_views=query_views,
            window_size=int(args.window_size),
            seed=int(args.seed),
            feature_preset=str(args.feature_preset),
            enrichment_profile=str(args.enrichment_profile),
            enrichment_rounds=int(args.enrichment_rounds),
            hyperopt_evals=int(resolved_hyperopt_evals),
            inner_validation_fraction=float(args.inner_validation_fraction),
            n_jobs=int(args.n_jobs),
        )
        per_group_rows = _evaluate_rows(
            model=model,
            rows=eval_rows,
            feature_preset=str(args.feature_preset),
            window_size=int(args.window_size),
            fold_index=0,
            train_summary=train_summary,
        )
        if write_shap:
            shap_runs.append(
                {
                    "heldout_dataset": str(heldout_dataset),
                    "model": model,
                    "rows": [row for row in eval_rows if int(row["retrieval_rank"]) <= int(args.window_size)],
                }
            )
        all_per_group_rows.extend(per_group_rows)
        heldout_summary[heldout_dataset] = {
            "train_summary": train_summary,
            "evaluation": _summarize_per_group_rows(per_group_rows),
        }
        if save_models:
            model_dir = args.output_dir / "models" / heldout_dataset
            model_dir.mkdir(parents=True, exist_ok=True)
            with (model_dir / "model.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "model": model,
                        "train_summary": train_summary,
                        "feature_preset": str(args.feature_preset),
                        "enrichment_profile": str(args.enrichment_profile),
                        "enrichment_rounds": int(args.enrichment_rounds),
                        "window_size": int(args.window_size),
                        "query_views": list(query_views),
                        "training_source_mode": str(args.training_source_mode),
                        "training_source_summary": _summarize_training_source_rows(train_rows),
                        "datasets": [str(value) for value in train_datasets],
                    },
                    handle,
                )

    summary: dict[str, Any] = {
        "config": {
            "datasets": [str(value) for value in args.datasets],
            "query_views": list(query_views),
            "window_size": int(args.window_size),
            "seed": int(args.seed),
            "feature_preset": str(args.feature_preset),
            "enrichment_profile": str(args.enrichment_profile),
            "enrichment_rounds": int(args.enrichment_rounds),
            "training_source_mode": str(args.training_source_mode),
            "hyperopt_evals": int(resolved_hyperopt_evals),
            "run_mode": str(args.run_mode),
            "rows_source": str(args.rows_source),
            "rows_source_by_dataset": dict(rows_source_by_dataset),
            "selected_query_group_filter_applied": bool(selected_query_group_ids is not None),
            "inner_validation_fraction": float(args.inner_validation_fraction),
            "n_jobs": int(args.n_jobs),
            "objective": "lambdarank",
            "search_space_source": "s2and.model_pairwise.PairwiseModeler",
        },
        "heldout": heldout_summary,
        "overall": _summarize_per_group_rows(all_per_group_rows),
    }
    _write_group_csv(args.output_dir / "leave_one_dataset_out_per_group.csv", all_per_group_rows)
    if write_shap:
        summary["shap"] = _write_pooled_shap_artifacts(
            shap_runs=shap_runs,
            feature_preset=str(args.feature_preset),
            output_dir=args.output_dir,
            max_rows=int(args.shap_max_rows),
            seed=int(args.seed),
            shap_plot_type=str(args.shap_plot_type),
        )

    if args.h_wang_row_file is not None:
        full_train_rows = _compose_training_rows(
            s2and_rows=dataset_rows,
            h_wang_rows=h_wang_train_rows,
            training_source_mode=str(args.training_source_mode),
        )
        final_model, final_train_summary = _fit_ranker_for_split(
            train_rows=full_train_rows,
            query_views=query_views,
            window_size=int(args.window_size),
            seed=int(args.seed),
            feature_preset=str(args.feature_preset),
            enrichment_profile=str(args.enrichment_profile),
            enrichment_rounds=int(args.enrichment_rounds),
            hyperopt_evals=int(resolved_hyperopt_evals),
            inner_validation_fraction=float(args.inner_validation_fraction),
            n_jobs=int(args.n_jobs),
        )
        h_wang_summary: dict[str, Any] = {
            "train_summary": final_train_summary,
            "training_source_summary": _summarize_training_source_rows(full_train_rows),
        }
        if h_wang_any_input_rows:
            dev_rows = _select_h_wang_any_input_rows(
                h_wang_rows,
                query_views=query_views,
                splits=["dev"],
                supervision_types=["positive_repeat_orcid", "negative_singleton_orcid"],
            )
            unresolved_dev_rows = _select_h_wang_any_input_rows(
                h_wang_rows,
                query_views=query_views,
                splits=["dev"],
                supervision_types=["unresolved_repeat_orcid"],
            )
            h_wang_per_group_rows: list[dict[str, Any]] = []
            for window_size in [int(value) for value in args.h_wang_window_sizes]:
                h_wang_per_group_rows.extend(
                    _evaluate_rows(
                        model=final_model,
                        rows=dev_rows,
                        feature_preset=str(args.feature_preset),
                        window_size=int(window_size),
                        fold_index=0,
                        train_summary=final_train_summary,
                    )
                )
            _write_group_csv(args.output_dir / "h_wang_per_group.csv", h_wang_per_group_rows)
            h_wang_summary["evaluation"] = {
                "by_window": {},
                "unresolved_group_counts": {
                    "queries": _count_query_groups(unresolved_dev_rows),
                    "by_view": {
                        query_view: _count_query_groups(
                            [row for row in unresolved_dev_rows if str(row["query_view"]) == query_view]
                        )
                        for query_view in query_views
                    },
                },
            }
            for window_size in [int(value) for value in args.h_wang_window_sizes]:
                window_rows = [row for row in h_wang_per_group_rows if int(row["window_size"]) == int(window_size)]
                positive_rows = [row for row in window_rows if str(row["supervision_type"]) == "positive_repeat_orcid"]
                negative_rows = [
                    row for row in window_rows if str(row["supervision_type"]) == "negative_singleton_orcid"
                ]
                h_wang_summary["evaluation"]["by_window"][str(window_size)] = {
                    "positive_repeat_orcid": _summarize_per_group_rows(positive_rows),
                    "negative_singleton_orcid": _summarize_negative_singletons(negative_rows),
                    "reject_threshold": _select_reject_threshold(window_rows),
                }
            regressions = [
                row
                for row in h_wang_per_group_rows
                if int(row["window_size"]) == int(args.window_size)
                and str(row["supervision_type"]) == "positive_repeat_orcid"
                and int(row["model_correct"]) == 0
                and int(row["heuristic_correct"]) == 1
            ]
            _write_group_csv(args.output_dir / "h_wang_regressions.csv", regressions)
            h_wang_summary["regression_count"] = int(len(regressions))
        else:
            legacy_eval_rows = select_rows(h_wang_rows, query_views=query_views)
            h_wang_per_group_rows: list[dict[str, Any]] = []
            for window_size in [int(value) for value in args.h_wang_window_sizes]:
                h_wang_per_group_rows.extend(
                    _evaluate_rows(
                        model=final_model,
                        rows=legacy_eval_rows,
                        feature_preset=str(args.feature_preset),
                        window_size=int(window_size),
                        fold_index=0,
                        train_summary=final_train_summary,
                    )
                )
            regressions = [
                row
                for row in h_wang_per_group_rows
                if int(row["window_size"]) == int(args.window_size)
                and int(row["model_correct"]) == 0
                and int(row["heuristic_correct"]) == 1
            ]
            _write_group_csv(args.output_dir / "h_wang_per_group.csv", h_wang_per_group_rows)
            _write_group_csv(args.output_dir / "h_wang_regressions.csv", regressions)
            h_wang_summary["evaluation"] = {
                "by_window": {
                    str(window_size): _summarize_per_group_rows(
                        [row for row in h_wang_per_group_rows if int(row["window_size"]) == int(window_size)]
                    )
                    for window_size in [int(value) for value in args.h_wang_window_sizes]
                }
            }
            h_wang_summary["regression_count"] = int(len(regressions))
        summary["h_wang"] = h_wang_summary
        if save_models:
            final_model_dir = args.output_dir / "models" / "final_h_wang"
            final_model_dir.mkdir(parents=True, exist_ok=True)
            with (final_model_dir / "model.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "model": final_model,
                        "train_summary": final_train_summary,
                        "feature_preset": str(args.feature_preset),
                        "enrichment_profile": str(args.enrichment_profile),
                        "enrichment_rounds": int(args.enrichment_rounds),
                        "window_size": int(args.window_size),
                        "query_views": list(query_views),
                        "training_source_mode": str(args.training_source_mode),
                        "training_source_summary": _summarize_training_source_rows(full_train_rows),
                        "datasets": [str(value) for value in args.datasets],
                    },
                    handle,
                )

    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()

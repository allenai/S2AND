from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightgbm as lgb
import numpy as np
import pytest

import s2and.incremental_linking.runtime as runtime_module
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import load_incremental_linking_artifact, save_incremental_linking_artifact
from s2and.incremental_linking.features import (
    PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS,
    LinkerFeatureMatrix,
    assemble_linker_feature_matrix,
    promoted_linker_feature_columns,
)
from s2and.incremental_linking.linker_pairwise import (
    PROMOTED_PAIRWISE_AGG_FEATURE_INDICES,
    LinkerCandidateBatch,
    promoted_pairwise_aggregate_columns,
)
from s2and.incremental_linking.retrieval import LinkerRetrievalBatch
from s2and.incremental_linking.runtime import (
    CandidateBatchPairwiseModelResult,
    _predict_incremental_link_or_abstain_compact,
    _predict_incremental_link_or_abstain_production_private,
    _predict_incremental_link_or_abstain_retrieved_candidates,
    compute_candidate_batch_pairwise_model_and_aggregate_stats,
    naturalize_incremental_clusters,
    signature_id_to_index_map,
)
from tests.helpers import build_dummy_dataset


class StaticPairwiseStats:
    def __init__(self, row_count: int) -> None:
        self.aggregate_feature_columns = promoted_pairwise_aggregate_columns()
        self._matrix = np.zeros((row_count, len(self.aggregate_feature_columns)), dtype=np.float32)

    def feature_matrix(self) -> np.ndarray:
        return self._matrix


class StaticArtifact:
    def __init__(self, probabilities: np.ndarray, gate_config: dict[str, object]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.metadata = SimpleNamespace(
            feature_columns=promoted_linker_feature_columns(),
            gate_config=gate_config,
            retrieval_top_k=25,
        )

    def predict_probabilities(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.shape[0] == len(self.probabilities)
        return self.probabilities


class FirstColumnDistanceClassifier:
    def predict_proba(self, features: np.ndarray, num_threads: int | None = None) -> np.ndarray:
        distances = np.asarray(features, dtype=np.float64)[:, 0]
        return np.column_stack((distances, 1.0 - distances))


class RejectsNumThreadsClassifier:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        distances = np.asarray(features, dtype=np.float64)[:, 0]
        return np.column_stack((distances, 1.0 - distances))


def test_pairwise_predict_class0_does_not_require_num_threads_keyword_support() -> None:
    predictions = runtime_module._predict_pairwise_class0(
        RejectsNumThreadsClassifier(),
        np.asarray([[0.25], [0.75]], dtype=np.float64),
        num_threads=2,
    )

    assert np.allclose(predictions, [0.25, 0.75])


class FakeRuntimeFeaturizer:
    def __init__(self, signature_ids: list[str], *, default_label: float = float("nan")) -> None:
        self._signature_ids = tuple(signature_ids)
        self.default_label = float(default_label)

    def signature_ids(self) -> list[str]:
        return list(self._signature_ids)

    def linker_pair_index_arrays_constraint_labels(
        self,
        left_signature_indices: np.ndarray,
        right_signature_indices: np.ndarray,
        low_value: float,
        high_value: float,
        dont_merge_cluster_seeds: bool,
        incremental_dont_use_cluster_seeds: bool,
        num_threads: int | None,
        suppress_orcid: bool,
        large_integer: float,
    ) -> np.ndarray:
        del (
            right_signature_indices,
            low_value,
            high_value,
            dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds,
            num_threads,
            suppress_orcid,
            large_integer,
        )
        return np.full(len(left_signature_indices), self.default_label, dtype=np.float64)


def _python_distance_accumulators(
    *,
    row_indices: np.ndarray,
    row_count: int,
    model_distances: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    counts = np.zeros(int(row_count), dtype=np.uint32)
    sums = np.zeros(int(row_count), dtype=np.float64)
    mins = np.full(int(row_count), np.inf, dtype=np.float64)
    top = np.full((int(row_count), 5), np.inf, dtype=np.float64)
    hard_disallow = 0
    for row_raw, model_distance, label in zip(row_indices, model_distances, labels, strict=True):
        row = int(row_raw)
        value = float(model_distance if np.isnan(label) else label + LARGE_INTEGER)
        if np.isnan(value):
            raise ValueError("pairwise model returned NaN distance")
        counts[row] += 1
        sums[row] += value
        mins[row] = min(mins[row], value)
        if value >= LARGE_DISTANCE:
            hard_disallow += 1
        if value < top[row, -1]:
            top[row, -1] = value
            top[row].sort()
    return counts, sums, mins, top, hard_disallow


class FakeProductionClusterer:
    def __init__(
        self,
        seed_map: dict[str, str],
        recluster_map: dict[str, str] | None = None,
        *,
        default_label: float = float("nan"),
    ) -> None:
        self.seed_map = dict(seed_map)
        self.recluster_map = dict(recluster_map or {})
        self.default_label = float(default_label)
        self.n_jobs = 1
        self.use_cache = False
        self.classifier = FirstColumnDistanceClassifier()
        self.featurizer_info = FeaturizationInfo(features_to_use=["name_similarity"])
        self.nameless_classifier = None
        self.nameless_featurizer_info = None
        self.resolved_pair_ids: list[tuple[str, str]] = []
        self.resolve_incremental_flags: list[bool] = []

    def _build_incremental_seed_setup(
        self,
        _dataset: object,
        _partial_supervision: dict[tuple[str, str], int | float],
        _runtime_context: object,
        total_ram_bytes: int | None = None,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
        del total_ram_bytes
        inverse: dict[str, list[str]] = {}
        for signature_id, cluster_id in self.seed_map.items():
            inverse.setdefault(cluster_id, []).append(signature_id)
        return dict(self.seed_map), dict(self.recluster_map), inverse

    def _resolve_constraint_batch(
        self,
        _dataset: object,
        pair_ids: list[tuple[str, str]],
        partial_supervision: dict[tuple[str, str], int | float],
        runtime_context: object,
        *,
        incremental_dont_use_cluster_seeds: bool,
        constraint_backend: object | None,
    ) -> tuple[list[float], SimpleNamespace]:
        assert runtime_context is not None
        assert incremental_dont_use_cluster_seeds is False
        assert constraint_backend is None
        self.resolved_pair_ids = list(pair_ids)
        self.resolve_incremental_flags.append(bool(incremental_dont_use_cluster_seeds))
        labels: list[float] = []
        partial_hits = 0
        for left, right in pair_ids:
            if (left, right) in partial_supervision:
                labels.append(float(partial_supervision[(left, right)] - LARGE_INTEGER))
                partial_hits += 1
            elif (right, left) in partial_supervision:
                labels.append(float(partial_supervision[(right, left)] - LARGE_INTEGER))
                partial_hits += 1
            else:
                labels.append(float(self.default_label))
        return labels, SimpleNamespace(
            total_pairs=len(pair_ids),
            partial_supervision_hits=partial_hits,
            unresolved_pairs=len(pair_ids) - partial_hits,
            rust_batch_call_count=0,
            api_mode="fake",
            elapsed_seconds=0.0,
        )


def _tiny_booster() -> tuple[lgb.Booster, np.ndarray]:
    columns = promoted_linker_feature_columns()
    matrix = np.zeros((8, len(columns)), dtype=np.float32)
    matrix[:, columns.index("retrieval_score")] = np.linspace(0.0, 1.0, len(matrix), dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    dataset = lgb.Dataset(matrix, label=labels, free_raw_data=False)
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_threads": 1,
            "learning_rate": 0.3,
            "num_leaves": 3,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=6,
    )
    return booster, matrix[:3]


def _row_features(retrieval_scores: np.ndarray) -> dict[str, np.ndarray]:
    row_count = len(retrieval_scores)
    row_features = {column: np.zeros(row_count, dtype=np.float32) for column in PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS}
    row_features["retrieval_score"] = np.asarray(retrieval_scores, dtype=np.float32)
    return row_features


def _row_features_with_telemetry(retrieval_scores: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    return _row_features(retrieval_scores), {
        "generated_family_id_count": 0,
        "generic_family_override_count": 0,
    }


def _retrieval_batch(
    *,
    row_query_signature_indices: np.ndarray,
    row_component_keys: tuple[str, ...],
    retrieval_ranks: np.ndarray | None = None,
) -> LinkerRetrievalBatch:
    row_count = len(row_query_signature_indices)
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=row_query_signature_indices,
        row_component_keys=row_component_keys,
        retrieval_ranks=retrieval_ranks,
    )
    return LinkerRetrievalBatch(
        candidate_batch=candidate_batch,
        row_signals={
            "retrieval_score": np.linspace(0.1, 0.9, row_count, dtype=np.float32),
            "retrieval_rank": (
                np.arange(1, row_count + 1, dtype=np.float32)
                if retrieval_ranks is None
                else retrieval_ranks.astype(np.float32)
            ),
            "candidate_component_key": np.asarray(row_component_keys, dtype=object),
            "query_view": np.asarray(["initial_only"] * row_count, dtype=object),
            "query_first_token": np.asarray(["alice"] * row_count, dtype=object),
        },
    )


def _empty_feature_matrix(candidate_batch: LinkerCandidateBatch) -> LinkerFeatureMatrix:
    return LinkerFeatureMatrix(
        matrix=np.empty((candidate_batch.row_count, len(promoted_linker_feature_columns())), dtype=np.float32),
        feature_columns=promoted_linker_feature_columns(),
        candidate_batch=candidate_batch,
        pairwise_stats=StaticPairwiseStats(candidate_batch.row_count),
    )


def _production_retrieval_batch(
    *,
    row_query_signature_indices: np.ndarray,
    row_component_keys: tuple[str, ...],
    left_signature_indices: np.ndarray | None = None,
    right_signature_indices: np.ndarray | None = None,
    pair_row_indices: np.ndarray | None = None,
) -> LinkerRetrievalBatch:
    row_count = len(row_query_signature_indices)
    left = np.zeros(0, dtype=np.uint32) if left_signature_indices is None else left_signature_indices
    right = np.zeros(0, dtype=np.uint32) if right_signature_indices is None else right_signature_indices
    rows = np.zeros(0, dtype=np.uint32) if pair_row_indices is None else pair_row_indices
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=left,
        right_signature_indices=right,
        pair_row_indices=rows,
        row_query_signature_indices=row_query_signature_indices,
        row_component_keys=row_component_keys,
        retrieval_scores=np.ones(row_count, dtype=np.float32),
        retrieval_ranks=np.arange(1, row_count + 1, dtype=np.uint16),
    )
    return LinkerRetrievalBatch(
        candidate_batch=candidate_batch,
        row_signals={
            "retrieval_score": np.ones(row_count, dtype=np.float32),
            "retrieval_rank": np.arange(1, row_count + 1, dtype=np.float32),
            "candidate_component_key": np.asarray(row_component_keys, dtype=object),
            "query_view": np.asarray(["initial_only"] * row_count, dtype=object),
            "query_first_token": np.asarray(["alice"] * row_count, dtype=object),
        },
    )


def _fake_pairwise_result(candidate_batch: LinkerCandidateBatch) -> CandidateBatchPairwiseModelResult:
    row_count = candidate_batch.row_count
    return CandidateBatchPairwiseModelResult(
        row_signals={
            "min_distance": np.zeros(row_count, dtype=np.float32),
            "mean_distance": np.zeros(row_count, dtype=np.float32),
            "top3_mean_distance": np.zeros(row_count, dtype=np.float32),
            "top5_mean_distance": np.zeros(row_count, dtype=np.float32),
            "pair_count": np.asarray([candidate_batch.pair_count] * row_count, dtype=np.float32),
        },
        pairwise_stats=StaticPairwiseStats(row_count),
        telemetry={
            "candidate_row_count": row_count,
            "pair_count": candidate_batch.pair_count,
            "chunk_count": 1 if candidate_batch.pair_count else 0,
            "matrix_feature_count": 1,
            "aggregate_feature_count": 1,
            "feature_seconds": 0.0,
            "predict_seconds": 0.0,
            "total_seconds": 0.0,
        },
    )


def test_fused_pairwise_model_and_aggregates_preserve_existing_distance_semantics(monkeypatch) -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=4,
        left_signature_indices=np.asarray([0, 1, 2, 3, 4], dtype=np.uint32),
        right_signature_indices=np.asarray([10, 11, 12, 13, 14], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 0, 1, 2, 2], dtype=np.uint32),
    )
    main_distances = np.asarray([0.2, 0.5, 0.1, 0.9, 0.4], dtype=np.float64)
    nameless_distances = np.asarray([0.4, 0.7, 0.3, 0.7, 0.2], dtype=np.float64)
    calls: list[dict[str, object]] = []

    def fake_build_arrays(
        _dataset,
        left_signature_indices,
        _right_signature_indices,
        row_indices,
        row_count,
        *,
        matrix_indices,
        aggregate_indices,
        num_threads,
        nan_value,
        aggregate_nan_value,
        runtime_context=None,
        use_cache=False,
        featurizer=None,
    ):
        assert runtime_context is None
        assert use_cache is False
        assert featurizer is not None
        calls.append(
            {
                "matrix_indices": tuple(matrix_indices),
                "aggregate_indices": tuple(aggregate_indices),
                "num_threads": num_threads,
                "nan_value": nan_value,
                "aggregate_nan_value": aggregate_nan_value,
            }
        )
        offsets = np.asarray(left_signature_indices, dtype=np.int64)
        position_by_index = {int(index): position for position, index in enumerate(matrix_indices)}
        matrix = np.zeros((len(offsets), len(matrix_indices)), dtype=np.float64)
        matrix[:, position_by_index[0]] = main_distances[offsets]
        matrix[:, position_by_index[6]] = nameless_distances[offsets]
        counts = np.zeros(int(row_count), dtype=np.uint32)
        sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
        mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
        maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
        for offset, local_row in zip(offsets, row_indices, strict=True):
            values = offset.astype(np.float64) + np.arange(len(aggregate_indices), dtype=np.float64) / 100.0
            if int(offset) == 1:
                values[0] = aggregate_nan_value
            row = int(local_row)
            counts[row] += 1
            sums[row] += values
            mins[row] = np.minimum(mins[row], values)
            maxs[row] = np.maximum(maxs[row], values)
        return matrix, counts, sums, mins, maxs

    monkeypatch.setattr(
        runtime_module.feature_port,
        "build_linker_pair_features_and_aggregate_stats_arrays_rust",
        fake_build_arrays,
    )
    monkeypatch.setattr(
        runtime_module,
        "_accumulate_pairwise_distance_chunk",
        lambda **kwargs: _python_distance_accumulators(
            row_indices=kwargs["row_indices"],
            row_count=kwargs["row_count"],
            model_distances=kwargs["model_distances"],
            labels=kwargs["labels"],
        ),
    )

    labels = np.full(candidate_batch.pair_count, np.nan, dtype=np.float64)
    labels[1] = -float(LARGE_INTEGER)
    result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        SimpleNamespace(),
        candidate_batch,
        classifier=FirstColumnDistanceClassifier(),
        featurizer_info=FeaturizationInfo(features_to_use=["name_similarity"]),
        nameless_classifier=FirstColumnDistanceClassifier(),
        nameless_featurizer_info=FeaturizationInfo(features_to_use=["affiliation_similarity"]),
        pair_labels=labels,
        n_jobs=3,
        featurizer=object(),
    )

    assert len(calls) == 1
    assert 0 in calls[0]["matrix_indices"]
    assert 6 in calls[0]["matrix_indices"]
    assert tuple(calls[0]["aggregate_indices"]) == tuple(PROMOTED_PAIRWISE_AGG_FEATURE_INDICES)
    assert np.isnan(float(calls[0]["nan_value"]))
    assert float(calls[0]["aggregate_nan_value"]) == 0.0
    np.testing.assert_array_equal(result.pairwise_stats.counts, np.asarray([2, 1, 2, 0], dtype=np.uint64))
    np.testing.assert_array_equal(result.pairwise_stats.valid_counts[:, 0], np.asarray([2, 1, 2, 0]))
    np.testing.assert_allclose(result.pairwise_stats.sums[:, 0], np.asarray([0.7, 0.1, 1.3, 0.0]))
    np.testing.assert_allclose(result.pairwise_stats.mins[:, 0], np.asarray([0.2, 0.1, 0.4, np.inf]))
    np.testing.assert_allclose(result.pairwise_stats.maxs[:, 0], np.asarray([0.5, 0.1, 0.9, -np.inf]))
    np.testing.assert_allclose(result.row_signals["min_distance"], np.asarray([0.0, 0.2, 0.3, 1.0]))
    np.testing.assert_allclose(result.row_signals["mean_distance"], np.asarray([0.15, 0.2, 0.55, 1.0]))
    np.testing.assert_allclose(result.row_signals["top3_mean_distance"], np.asarray([0.15, 0.2, 0.55, 1.0]))
    np.testing.assert_allclose(result.row_signals["top5_mean_distance"], np.asarray([0.15, 0.2, 0.55, 1.0]))
    np.testing.assert_array_equal(
        result.row_signals["pair_count"],
        np.asarray([2.0, 1.0, 2.0, 0.0], dtype=np.float32),
    )
    assert result.telemetry["pair_count"] == 5
    assert result.telemetry["chunk_count"] == 1


def test_fused_pairwise_model_uses_configurable_nan_policies(monkeypatch) -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.asarray([0, 1], dtype=np.uint32),
        right_signature_indices=np.asarray([10, 11], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 0], dtype=np.uint32),
    )
    calls: list[tuple[float, float]] = []

    def fake_build_arrays(
        _dataset,
        left_signature_indices,
        _right_signature_indices,
        row_indices,
        row_count,
        *,
        matrix_indices,
        aggregate_indices,
        num_threads,
        nan_value,
        aggregate_nan_value,
        runtime_context=None,
        use_cache=False,
        featurizer=None,
    ):
        del _right_signature_indices, runtime_context, use_cache, featurizer
        assert num_threads == 1
        calls.append((float(nan_value), float(aggregate_nan_value)))
        offsets = np.asarray(left_signature_indices, dtype=np.int64)
        position_by_index = {int(index): position for position, index in enumerate(matrix_indices)}
        matrix = np.full((len(offsets), len(matrix_indices)), 0.25, dtype=np.float64)
        last_aggregate_position = position_by_index[int(aggregate_indices[-1])]
        matrix[0, last_aggregate_position] = np.nan
        matrix[1, last_aggregate_position] = 0.75
        counts = np.zeros(int(row_count), dtype=np.uint32)
        sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
        mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
        maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
        for local_row in row_indices:
            row = int(local_row)
            counts[row] += 1
            sums[row] += 0.0
            mins[row] = np.minimum(mins[row], 0.0)
            maxs[row] = np.maximum(maxs[row], 0.0)
        return matrix, counts, sums, mins, maxs

    monkeypatch.setattr(
        runtime_module.feature_port,
        "build_linker_pair_features_and_aggregate_stats_arrays_rust",
        fake_build_arrays,
    )
    monkeypatch.setattr(
        runtime_module,
        "_accumulate_pairwise_distance_chunk",
        lambda **kwargs: _python_distance_accumulators(
            row_indices=kwargs["row_indices"],
            row_count=kwargs["row_count"],
            model_distances=kwargs["model_distances"],
            labels=kwargs["labels"],
        ),
    )

    result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        SimpleNamespace(),
        candidate_batch,
        classifier=FirstColumnDistanceClassifier(),
        featurizer_info=FeaturizationInfo(features_to_use=["name_similarity"]),
        pair_labels=np.full(candidate_batch.pair_count, np.nan, dtype=np.float64),
        pairwise_model_nan_value=0.0,
        pairwise_aggregate_nan_value=0.0,
        featurizer=object(),
    )

    assert len(calls) == 1
    assert calls[0][0] == 0.0
    assert calls[0][1] == 0.0
    assert result.pairwise_stats.valid_counts[0, -1] == 2
    assert result.pairwise_stats.mean_matrix()[0, -1] == pytest.approx(0.375)

    calls.clear()
    result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        SimpleNamespace(),
        candidate_batch,
        classifier=FirstColumnDistanceClassifier(),
        featurizer_info=FeaturizationInfo(features_to_use=["name_similarity"]),
        pair_labels=np.full(candidate_batch.pair_count, np.nan, dtype=np.float64),
        featurizer=object(),
    )

    assert len(calls) == 1
    assert np.isnan(calls[0][0])
    assert calls[0][1] == 0.0
    assert result.pairwise_stats.valid_counts[0, -1] == 2
    assert result.pairwise_stats.mean_matrix()[0, -1] == pytest.approx(0.375)

    calls.clear()
    result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        SimpleNamespace(),
        candidate_batch,
        classifier=FirstColumnDistanceClassifier(),
        featurizer_info=FeaturizationInfo(features_to_use=["name_similarity"]),
        pair_labels=np.full(candidate_batch.pair_count, np.nan, dtype=np.float64),
        pairwise_model_nan_value=np.nan,
        pairwise_aggregate_nan_value=np.nan,
        featurizer=object(),
    )

    assert len(calls) == 1
    assert np.isnan(calls[0][0])
    assert np.isnan(calls[0][1])
    assert result.pairwise_stats.valid_counts[0, -1] == 1
    assert result.pairwise_stats.mean_matrix()[0, -1] == pytest.approx(0.75)


def test_fused_pairwise_model_preserves_true_hard_disallow_distances(monkeypatch) -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=2,
        left_signature_indices=np.asarray([0, 1, 2], dtype=np.uint32),
        right_signature_indices=np.asarray([10, 11, 12], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 0, 1], dtype=np.uint32),
    )

    def fake_build_arrays(
        _dataset,
        left_signature_indices,
        _right_signature_indices,
        row_indices,
        row_count,
        *,
        matrix_indices,
        aggregate_indices,
        num_threads,
        nan_value,
        aggregate_nan_value,
        runtime_context=None,
        use_cache=False,
        featurizer=None,
    ):
        assert runtime_context is None
        assert use_cache is False
        assert featurizer is not None
        assert num_threads == 2
        assert np.isnan(float(nan_value))
        assert float(aggregate_nan_value) == 0.0
        offsets = np.asarray(left_signature_indices, dtype=np.int64)
        position_by_index = {int(index): position for position, index in enumerate(matrix_indices)}
        matrix = np.zeros((len(offsets), len(matrix_indices)), dtype=np.float64)
        matrix[:, position_by_index[0]] = np.asarray([0.2, 0.4, 0.6], dtype=np.float64)[offsets]
        counts = np.zeros(int(row_count), dtype=np.uint32)
        sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
        mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
        maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
        for local_row in row_indices:
            row = int(local_row)
            counts[row] += 1
            mins[row] = 0.0
            maxs[row] = 0.0
        return matrix, counts, sums, mins, maxs

    monkeypatch.setattr(
        runtime_module.feature_port,
        "build_linker_pair_features_and_aggregate_stats_arrays_rust",
        fake_build_arrays,
    )
    monkeypatch.setattr(
        runtime_module,
        "_accumulate_pairwise_distance_chunk",
        lambda **kwargs: _python_distance_accumulators(
            row_indices=kwargs["row_indices"],
            row_count=kwargs["row_count"],
            model_distances=kwargs["model_distances"],
            labels=kwargs["labels"],
        ),
    )

    labels = np.full(candidate_batch.pair_count, np.nan, dtype=np.float64)
    labels[1:] = float(LARGE_DISTANCE - LARGE_INTEGER)
    result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        SimpleNamespace(),
        candidate_batch,
        classifier=FirstColumnDistanceClassifier(),
        featurizer_info=FeaturizationInfo(features_to_use=["name_similarity"]),
        pair_labels=labels,
        n_jobs=2,
        featurizer=object(),
    )

    np.testing.assert_allclose(result.row_signals["min_distance"], np.asarray([0.2, LARGE_DISTANCE]))
    np.testing.assert_allclose(
        result.row_signals["mean_distance"],
        np.asarray([(0.2 + LARGE_DISTANCE) / 2.0, LARGE_DISTANCE], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result.row_signals["top3_mean_distance"],
        np.asarray([(0.2 + LARGE_DISTANCE) / 2.0, LARGE_DISTANCE], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result.row_signals["top5_mean_distance"],
        np.asarray([(0.2 + LARGE_DISTANCE) / 2.0, LARGE_DISTANCE], dtype=np.float32),
    )
    np.testing.assert_array_equal(result.row_signals["pair_count"], np.asarray([2.0, 1.0], dtype=np.float32))
    assert result.telemetry["hard_disallow_distance_pair_count"] == 2


def test_fused_pairwise_model_requires_rust_distance_accumulator(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.asarray([0], dtype=np.uint32),
        right_signature_indices=np.asarray([1], dtype=np.uint32),
        pair_row_indices=np.asarray([0], dtype=np.uint32),
    )

    def fake_build_arrays(*_args, matrix_indices, aggregate_indices, **_kwargs):
        matrix = np.zeros((1, len(matrix_indices)), dtype=np.float64)
        matrix[:, tuple(matrix_indices).index(0)] = 0.2
        return (
            matrix,
            np.asarray([1], dtype=np.uint32),
            np.zeros((1, len(aggregate_indices)), dtype=np.float64),
            np.zeros((1, len(aggregate_indices)), dtype=np.float64),
            np.zeros((1, len(aggregate_indices)), dtype=np.float64),
        )

    monkeypatch.setattr(
        runtime_module.feature_port,
        "build_linker_pair_features_and_aggregate_stats_arrays_rust",
        fake_build_arrays,
    )

    with pytest.raises(RuntimeError, match="linker_pair_distance_accumulators is required"):
        compute_candidate_batch_pairwise_model_and_aggregate_stats(
            SimpleNamespace(),
            candidate_batch,
            classifier=FirstColumnDistanceClassifier(),
            featurizer_info=FeaturizationInfo(features_to_use=["name_similarity"]),
            featurizer=object(),
        )


def test_fused_pairwise_model_rust_distance_accumulator_matches_python_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = build_dummy_dataset("dummy_linker_rust_distance_accumulator_parity", load_name_counts=True)
    rust_featurizer = runtime_module.feature_port._get_rust_featurizer(dataset)  # noqa: SLF001
    if not hasattr(rust_featurizer, "linker_pair_distance_accumulators"):
        pytest.skip("linker_pair_distance_accumulators is unavailable")

    signature_count = len(rust_featurizer.signature_ids())
    pair_count = 4096
    row_count = 257
    offsets = np.arange(pair_count, dtype=np.uint32)
    left_indices = offsets % np.uint32(signature_count)
    right_indices = (left_indices + (offsets % np.uint32(max(1, signature_count - 1))) + np.uint32(1)) % np.uint32(
        signature_count
    )
    row_indices = ((offsets * np.uint32(37)) % np.uint32(row_count)).astype(np.uint32)
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=left_indices,
        right_signature_indices=right_indices,
        pair_row_indices=row_indices,
    )
    labels = runtime_module.feature_port.get_constraint_labels_index_arrays_rust(
        dataset,
        left_indices,
        right_indices,
        featurizer=rust_featurizer,
        num_threads=2,
    )
    labels = np.asarray(labels, dtype=np.float64)
    labels[::31] = -float(LARGE_INTEGER)
    labels[::43] = float(LARGE_DISTANCE - LARGE_INTEGER)

    common_kwargs = {
        "classifier": FirstColumnDistanceClassifier(),
        "featurizer_info": FeaturizationInfo(features_to_use=["name_similarity"]),
        "pair_labels": labels,
        "n_jobs": 2,
        "featurizer": rust_featurizer,
    }
    with monkeypatch.context() as scoped:
        scoped.setattr(
            runtime_module,
            "_accumulate_pairwise_distance_chunk",
            lambda **kwargs: _python_distance_accumulators(
                row_indices=kwargs["row_indices"],
                row_count=kwargs["row_count"],
                model_distances=kwargs["model_distances"],
                labels=kwargs["labels"],
            ),
        )
        python_result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
            dataset,
            candidate_batch,
            **common_kwargs,
        )
    rust_result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        dataset,
        candidate_batch,
        **common_kwargs,
    )

    for name, expected_values in python_result.row_signals.items():
        np.testing.assert_allclose(rust_result.row_signals[name], expected_values)
    np.testing.assert_array_equal(rust_result.pairwise_stats.counts, python_result.pairwise_stats.counts)
    np.testing.assert_allclose(rust_result.pairwise_stats.sums, python_result.pairwise_stats.sums)
    np.testing.assert_allclose(rust_result.pairwise_stats.mins, python_result.pairwise_stats.mins)
    np.testing.assert_allclose(rust_result.pairwise_stats.maxs, python_result.pairwise_stats.maxs)
    assert (
        rust_result.telemetry["hard_disallow_distance_pair_count"]
        == python_result.telemetry["hard_disallow_distance_pair_count"]
    )


def test_compact_link_or_abstain_scores_artifact_rows_and_applies_gate(tmp_path: Path) -> None:
    booster, fixture = _tiny_booster()
    save_incremental_linking_artifact(
        booster,
        tmp_path,
        prediction_fixture_matrix=fixture,
        gate_config={"score_threshold": 0.0},
    )
    artifact = load_incremental_linking_artifact(tmp_path)
    candidate_batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        row_component_keys=("c_low", "c_high", "c_single"),
        retrieval_ranks=np.asarray([2, 1, 1], dtype=np.uint16),
    )
    feature_matrix = assemble_linker_feature_matrix(
        candidate_batch,
        _row_features(np.asarray([0.1, 0.9, 0.8], dtype=np.float32)),
        pairwise_stats=StaticPairwiseStats(row_count=3),
    )

    result = _predict_incremental_link_or_abstain_compact(artifact, feature_matrix)

    assert len(result.probabilities) == 3
    assert [decision.action for decision in result.decisions] == ["link", "link"]
    assert result.decisions[0].query_signature_index == 10
    assert result.decisions[0].component_key == "c_high"
    assert result.decisions[1].query_signature_index == 11
    assert result.decisions[1].component_key == "c_single"


def test_compact_link_or_abstain_abstains_when_artifact_score_threshold_too_high(tmp_path: Path) -> None:
    booster, fixture = _tiny_booster()
    save_incremental_linking_artifact(
        booster,
        tmp_path,
        prediction_fixture_matrix=fixture,
        gate_config={"score_threshold": 1.1, "single_candidate_score_threshold": 1.1},
    )
    artifact = load_incremental_linking_artifact(tmp_path)
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10], dtype=np.uint32),
        row_component_keys=("c0",),
    )
    feature_matrix = assemble_linker_feature_matrix(
        candidate_batch,
        _row_features(np.asarray([0.9], dtype=np.float32)),
        pairwise_stats=StaticPairwiseStats(row_count=1),
    )

    result = _predict_incremental_link_or_abstain_compact(artifact, feature_matrix)

    assert result.decisions[0].action == "abstain"
    assert result.decisions[0].component_key is None


def test_compact_link_or_abstain_applies_bucketed_classic_gate_parity() -> None:
    artifact = StaticArtifact(
        np.asarray([0.60, 0.55, 0.40], dtype=np.float64),
        gate_config={
            "score_threshold": 0.99,
            "margin_threshold": 0.99,
            "single_candidate_score_threshold": 0.99,
            "bucketed_score_thresholds": {
                "multi_candidate|single_letter_first": 0.99,
                "multi_candidate|multi_letter_first": 0.99,
                "single_candidate|single_letter_first": 0.30,
                "single_candidate|multi_letter_first": 0.50,
            },
            "bucketed_margin_thresholds": {
                "multi_candidate|single_letter_first": 0.04,
                "multi_candidate|multi_letter_first": 0.99,
            },
        },
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        row_component_keys=("c_margin", "c_runner_up", "c_single"),
        retrieval_ranks=np.asarray([1, 2, 1], dtype=np.uint16),
    )
    feature_matrix = _empty_feature_matrix(candidate_batch)
    row_signals: dict[str, Any] = {
        "query_first_token": np.asarray(["a", "a", "alice"], dtype=object),
        "query_view": np.asarray(["initial_only", "initial_only", "full"], dtype=object),
    }

    result = _predict_incremental_link_or_abstain_compact(
        artifact,  # type: ignore[arg-type]
        feature_matrix,
        row_signals=row_signals,
    )

    assert [decision.action for decision in result.decisions] == ["link", "abstain"]
    assert result.decisions[0].component_key == "c_margin"
    assert result.decisions[0].score_margin == pytest.approx(0.05)


def test_private_retrieved_candidate_slice_scores_matrix_and_records_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = StaticArtifact(
        np.asarray([0.1, 0.9, 0.8], dtype=np.float64),
        gate_config={"score_threshold": 0.0},
    )
    retrieval_batch = _retrieval_batch(
        row_query_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        row_component_keys=("c_low", "c_high", "c_single"),
        retrieval_ranks=np.asarray([2, 1, 1], dtype=np.uint16),
    )

    monkeypatch.setattr(
        runtime_module,
        "build_promoted_non_pairwise_row_features_with_telemetry",
        lambda _candidate_batch, _row_signals: _row_features_with_telemetry(
            np.asarray([0.1, 0.9, 0.8], dtype=np.float32)
        ),
    )

    result = _predict_incremental_link_or_abstain_retrieved_candidates(
        artifact,  # type: ignore[arg-type]
        retrieval_batch,
        pairwise_stats=StaticPairwiseStats(row_count=3),
    )

    assert result.feature_matrix.matrix.shape == (3, len(promoted_linker_feature_columns()))
    assert [decision.component_key for decision in result.compact_result.decisions] == ["c_high", "c_single"]
    assert result.telemetry == {
        "candidate_row_count": 3,
        "pair_count": 0,
        "no_candidate_query_count": 0,
        "decision_count": 2,
        "link_count": 2,
        "abstain_count": 0,
        "row_feature_generated_family_id_count": 0,
        "row_feature_generic_family_override_count": 0,
    }


def test_private_retrieved_candidate_slice_returns_no_candidate_abstains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = StaticArtifact(np.asarray([], dtype=np.float64), gate_config={"score_threshold": 0.0})
    retrieval_batch = _retrieval_batch(
        row_query_signature_indices=np.asarray([], dtype=np.uint32),
        row_component_keys=(),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_promoted_non_pairwise_row_features_with_telemetry",
        lambda _candidate_batch, _row_signals: _row_features_with_telemetry(np.asarray([], dtype=np.float32)),
    )

    result = _predict_incremental_link_or_abstain_retrieved_candidates(
        artifact,  # type: ignore[arg-type]
        retrieval_batch,
        pairwise_stats=StaticPairwiseStats(row_count=0),
        no_candidate_query_signature_indices=np.asarray([42], dtype=np.uint32),
    )

    assert len(result.compact_result.probabilities) == 0
    assert result.compact_result.decisions[0].query_signature_index == 42
    assert result.compact_result.decisions[0].action == "abstain"
    assert result.telemetry["no_candidate_query_count"] == 1


def test_private_retrieved_candidate_slice_rejects_partial_supervision() -> None:
    artifact = StaticArtifact(np.asarray([], dtype=np.float64), gate_config={"score_threshold": 0.0})
    retrieval_batch = _retrieval_batch(
        row_query_signature_indices=np.asarray([], dtype=np.uint32),
        row_component_keys=(),
    )

    with pytest.raises(NotImplementedError, match="partial supervision"):
        _predict_incremental_link_or_abstain_retrieved_candidates(
            artifact,  # type: ignore[arg-type]
            retrieval_batch,
            pairwise_stats=StaticPairwiseStats(row_count=0),
            partial_supervision={("q", "m"): "require"},
        )


def test_signature_id_to_index_map_returns_zero_indexed_map_from_featurizer() -> None:
    featurizer = SimpleNamespace(signature_ids=lambda: ["s1", "s2", "s3"])

    assert signature_id_to_index_map(featurizer) == {"s1": 0, "s2": 1, "s3": 2}


def test_naturalize_incremental_clusters_maps_split_ids() -> None:
    assert naturalize_incremental_clusters(
        {"s1": "7_0", "s2": "9"},
        {"7_0": "7"},
    ) == {"s1": "7", "s2": "9"}


def test_private_production_slice_links_abstains_and_naturalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1", "q2"])
    clusterer = FakeProductionClusterer({"s1": "7_0"}, recluster_map={"7_0": "7"})
    artifact = StaticArtifact(np.asarray([0.9], dtype=np.float64), gate_config={"score_threshold": 0.0})

    def fake_retrieval(**kwargs: object) -> LinkerRetrievalBatch:
        assert kwargs["component_member_indices_by_key"] == {"7_0": np.asarray([1], dtype=np.uint32)}
        np.testing.assert_array_equal(kwargs["query_signature_indices"], np.asarray([0, 2], dtype=np.uint32))
        return _production_retrieval_batch(
            row_query_signature_indices=np.asarray([0], dtype=np.uint32),
            row_component_keys=("7_0",),
            left_signature_indices=np.asarray([0], dtype=np.uint32),
            right_signature_indices=np.asarray([1], dtype=np.uint32),
            pair_row_indices=np.asarray([0], dtype=np.uint32),
        )

    monkeypatch.setattr(runtime_module, "build_linker_retrieval_batch_rust", fake_retrieval)
    monkeypatch.setattr(
        runtime_module,
        "compute_candidate_batch_pairwise_model_and_aggregate_stats",
        lambda _dataset, candidate_batch, **_kwargs: _fake_pairwise_result(candidate_batch),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_promoted_non_pairwise_row_features_with_telemetry",
        lambda _candidate_batch, _row_signals: _row_features_with_telemetry(np.asarray([0.9], dtype=np.float32)),
    )

    result = _predict_incremental_link_or_abstain_production_private(
        clusterer,
        artifact,  # type: ignore[arg-type]
        dataset=dataset,  # type: ignore[arg-type]
        featurizer=featurizer,
        retriever=object(),
        queries=[object(), object()],
        query_signature_ids=["q1", "q2"],
    )

    assert result.linked_signature_clusters == {"q1": "7"}
    assert [decision.action for decision in result.compact_result.decisions] == ["link", "abstain"]
    assert result.telemetry["no_candidate_query_count"] == 1
    assert result.telemetry["link_count"] == 1
    assert result.telemetry["abstain_count"] == 1


def test_private_production_slice_preserves_partial_supervision_constraint_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1", "s2"])
    clusterer = FakeProductionClusterer({"s1": "c1", "s2": "c2"})
    artifact = StaticArtifact(np.asarray([0.9, 0.1], dtype=np.float64), gate_config={"score_threshold": 0.0})
    captured_pair_labels: list[np.ndarray] = []

    monkeypatch.setattr(
        runtime_module,
        "build_linker_retrieval_batch_rust",
        lambda **_kwargs: _production_retrieval_batch(
            row_query_signature_indices=np.asarray([0, 0], dtype=np.uint32),
            row_component_keys=("c1", "c2"),
            left_signature_indices=np.asarray([0, 0], dtype=np.uint32),
            right_signature_indices=np.asarray([1, 2], dtype=np.uint32),
            pair_row_indices=np.asarray([0, 1], dtype=np.uint32),
        ),
    )

    def fake_pairwise(
        _dataset: object,
        candidate_batch: LinkerCandidateBatch,
        **kwargs: object,
    ) -> CandidateBatchPairwiseModelResult:
        captured_pair_labels.append(np.asarray(kwargs["pair_labels"], dtype=np.float64))
        return _fake_pairwise_result(candidate_batch)

    monkeypatch.setattr(runtime_module, "compute_candidate_batch_pairwise_model_and_aggregate_stats", fake_pairwise)
    monkeypatch.setattr(
        runtime_module,
        "build_promoted_non_pairwise_row_features_with_telemetry",
        lambda _candidate_batch, _row_signals: _row_features_with_telemetry(np.asarray([0.9, 0.1], dtype=np.float32)),
    )

    _predict_incremental_link_or_abstain_production_private(
        clusterer,
        artifact,  # type: ignore[arg-type]
        dataset=dataset,  # type: ignore[arg-type]
        featurizer=featurizer,
        retriever=object(),
        queries=[object()],
        query_signature_ids=["q1"],
        partial_supervision={
            ("q1", "s1"): 0,
            ("s2", "q1"): 10_000,
        },
    )

    np.testing.assert_allclose(
        captured_pair_labels[0],
        np.asarray([-float(LARGE_INTEGER), 10_000.0 - float(LARGE_INTEGER)]),
    )


def test_private_production_slice_keeps_seed_disallow_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    disallow_label = float(LARGE_DISTANCE - LARGE_INTEGER)
    featurizer = FakeRuntimeFeaturizer(["q1", "s1", "s2"], default_label=disallow_label)
    clusterer = FakeProductionClusterer(
        {"q1": "c1", "s1": "c1", "s2": "c2"},
        default_label=disallow_label,
    )
    artifact = StaticArtifact(np.asarray([0.9, 0.1], dtype=np.float64), gate_config={"score_threshold": 0.0})
    captured_pair_labels: list[np.ndarray] = []

    monkeypatch.setattr(
        runtime_module,
        "build_linker_retrieval_batch_rust",
        lambda **_kwargs: _production_retrieval_batch(
            row_query_signature_indices=np.asarray([0, 0], dtype=np.uint32),
            row_component_keys=("c1", "c2"),
            left_signature_indices=np.asarray([0, 0], dtype=np.uint32),
            right_signature_indices=np.asarray([1, 2], dtype=np.uint32),
            pair_row_indices=np.asarray([0, 1], dtype=np.uint32),
        ),
    )

    def fake_pairwise(
        _dataset: object,
        candidate_batch: LinkerCandidateBatch,
        **kwargs: object,
    ) -> CandidateBatchPairwiseModelResult:
        captured_pair_labels.append(np.asarray(kwargs["pair_labels"], dtype=np.float64))
        return _fake_pairwise_result(candidate_batch)

    monkeypatch.setattr(runtime_module, "compute_candidate_batch_pairwise_model_and_aggregate_stats", fake_pairwise)
    monkeypatch.setattr(
        runtime_module,
        "build_promoted_non_pairwise_row_features_with_telemetry",
        lambda _candidate_batch, _row_signals: _row_features_with_telemetry(np.asarray([0.9, 0.1], dtype=np.float32)),
    )

    result = _predict_incremental_link_or_abstain_production_private(
        clusterer,
        artifact,  # type: ignore[arg-type]
        dataset=dataset,  # type: ignore[arg-type]
        featurizer=featurizer,
        retriever=object(),
        queries=[object()],
        query_signature_ids=["q1"],
    )

    assert result.telemetry["constraint_api_mode"] == "rust_index_arrays"
    assert "constraint_disallow_ignored_pair_count" not in result.telemetry
    assert "constraint_seed_bypass_pair_count" not in result.telemetry
    assert captured_pair_labels[0][0] == pytest.approx(disallow_label)
    assert captured_pair_labels[0][1] == pytest.approx(disallow_label)


def test_private_production_slice_rejects_require_outside_retrieval_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = StaticArtifact(np.asarray([], dtype=np.float64), gate_config={"score_threshold": 0.0})
    monkeypatch.setattr(
        runtime_module,
        "build_linker_retrieval_batch_rust",
        lambda **_kwargs: _production_retrieval_batch(
            row_query_signature_indices=np.asarray([], dtype=np.uint32),
            row_component_keys=(),
        ),
    )

    with pytest.raises(ValueError, match="partial_supervision_require_outside_retrieval_window"):
        _predict_incremental_link_or_abstain_production_private(
            clusterer,
            artifact,  # type: ignore[arg-type]
            dataset=dataset,  # type: ignore[arg-type]
            featurizer=featurizer,
            retriever=object(),
            queries=[object()],
            query_signature_ids=["q1"],
            partial_supervision={("q1", "s1"): 0},
        )


def test_private_production_slice_records_disallow_outside_retrieval_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = StaticArtifact(np.asarray([], dtype=np.float64), gate_config={"score_threshold": 0.0})
    monkeypatch.setattr(
        runtime_module,
        "build_linker_retrieval_batch_rust",
        lambda **_kwargs: _production_retrieval_batch(
            row_query_signature_indices=np.asarray([], dtype=np.uint32),
            row_component_keys=(),
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "compute_candidate_batch_pairwise_model_and_aggregate_stats",
        lambda _dataset, candidate_batch, **_kwargs: _fake_pairwise_result(candidate_batch),
    )
    monkeypatch.setattr(
        runtime_module,
        "build_promoted_non_pairwise_row_features_with_telemetry",
        lambda _candidate_batch, _row_signals: _row_features_with_telemetry(np.asarray([], dtype=np.float32)),
    )

    result = _predict_incremental_link_or_abstain_production_private(
        clusterer,
        artifact,  # type: ignore[arg-type]
        dataset=dataset,  # type: ignore[arg-type]
        featurizer=featurizer,
        retriever=object(),
        queries=[object()],
        query_signature_ids=["q1"],
        partial_supervision={("q1", "s1"): 10_000},
    )

    assert result.compact_result.decisions[0].action == "abstain"
    assert result.telemetry["partial_supervision_disallow_outside_retrieval_window"] == 1


def test_private_production_slice_rejects_require_between_residual_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "q2", "s1"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = StaticArtifact(np.asarray([], dtype=np.float64), gate_config={"score_threshold": 0.0})
    monkeypatch.setattr(
        runtime_module,
        "build_linker_retrieval_batch_rust",
        lambda **_kwargs: _production_retrieval_batch(
            row_query_signature_indices=np.asarray([], dtype=np.uint32),
            row_component_keys=(),
        ),
    )

    with pytest.raises(ValueError, match="partial_supervision_require_between_residual_queries"):
        _predict_incremental_link_or_abstain_production_private(
            clusterer,
            artifact,  # type: ignore[arg-type]
            dataset=dataset,  # type: ignore[arg-type]
            featurizer=featurizer,
            retriever=object(),
            queries=[object(), object()],
            query_signature_ids=["q1", "q2"],
            partial_supervision={("q1", "q2"): 0},
        )

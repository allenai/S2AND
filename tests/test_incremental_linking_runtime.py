from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS,
    PROMOTED_PAIRWISE_AGG_FEATURE_INDICES,
    LinkerCandidateBatch,
    promoted_pairwise_aggregate_columns,
)
from s2and.incremental_linking.logistic_gate import load_logistic_gate_config, logistic_gate_config
from s2and.incremental_linking.retrieval import (
    RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS,
    LinkerRetrievalBatch,
    RawArrowPlanBundle,
    build_linker_retrieval_batch_from_raw_plan_bundle,
)
from s2and.incremental_linking.runtime import (
    CandidateBatchPairwiseModelResult,
    _predict_incremental_link_or_abstain_compact,
    _predict_incremental_link_or_abstain_retrieved_candidates,
    compute_candidate_batch_pairwise_model_and_aggregate_stats,
)
from s2and.model_pairwise import predict_pairwise_class0
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset
from tests.model_helpers import ConstantDistanceClassifier
from tests.promoted_linking_helpers import build_tiny_promoted_booster, synthetic_pairwise_bundle_binding

runtime_module: Any = runtime_module


class StaticPairwiseStats:
    def __init__(self, row_count: int) -> None:
        self.aggregate_feature_columns = promoted_pairwise_aggregate_columns()
        self._matrix = np.zeros((row_count, len(self.aggregate_feature_columns)), dtype=np.float32)

    def feature_matrix(self) -> np.ndarray:
        return self._matrix


class StaticArtifact:
    def __init__(self, probabilities: np.ndarray, gate_config: dict[str, Any]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.last_num_threads: int | None = None
        self.gate_model = load_logistic_gate_config(gate_config)
        self.feature_columns = promoted_linker_feature_columns()
        self.retrieval_top_k = 25

    def predict_probabilities(
        self,
        matrix: np.ndarray,
        *,
        num_threads: int | None = None,
        max_rows_per_chunk: int | None = None,
    ) -> np.ndarray:
        assert matrix.shape[0] == len(self.probabilities)
        assert max_rows_per_chunk is None or max_rows_per_chunk >= 1
        self.last_num_threads = num_threads
        return self.probabilities


class FirstColumnDistanceClassifier:
    def predict_proba(self, features: np.ndarray, num_threads: int | None = None) -> np.ndarray:
        distances = np.asarray(features, dtype=np.float64)[:, 0]
        return np.column_stack((distances, 1.0 - distances))


class RejectsNumThreadsClassifier:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        distances = np.asarray(features, dtype=np.float64)[:, 0]
        return np.column_stack((distances, 1.0 - distances))


class PositiveProbabilityClassifier:
    def predict_proba_positive(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64)[:, 0]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise AssertionError("predict_proba should not be called when predict_proba_positive is available")


class TwoFeatureClassifier:
    n_features_in_ = 2

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raise AssertionError("feature-count mismatch must fail before prediction")


def test_pairwise_predict_class0_does_not_require_num_threads_keyword_support() -> None:
    predictions = predict_pairwise_class0(
        RejectsNumThreadsClassifier(),
        np.asarray([[0.25], [0.75]], dtype=np.float64),
    )

    assert np.allclose(predictions, [0.25, 0.75])


def test_pairwise_predict_class0_uses_native_positive_probability_fast_path() -> None:
    predictions = predict_pairwise_class0(
        PositiveProbabilityClassifier(),
        np.asarray([[0.25], [0.75]], dtype=np.float64),
    )

    assert np.allclose(predictions, [0.75, 0.25])


def test_pairwise_predict_class0_rejects_fitted_feature_count_mismatch() -> None:
    with pytest.raises(ValueError, match="feature count does not match fitted schema"):
        predict_pairwise_class0(
            TwoFeatureClassifier(),
            np.asarray([[0.25], [0.75]], dtype=np.float64),
        )


def test_pairwise_model_feature_indices_match_sorted_featurizer_order() -> None:
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    featurizer_info.features_to_use = ["second", "first"]
    featurizer_info.feature_group_to_index = {"first": [3, 1], "second": [5, 1]}

    assert runtime_module._pairwise_model_feature_indices(featurizer_info) == (1, 3, 5)  # noqa: SLF001


def test_distance_row_signals_distinguish_top3_and_top5_means() -> None:
    signals = runtime_module._distance_row_signals(
        counts=np.asarray([6], dtype=np.uint64),
        sums=np.asarray([2.1], dtype=np.float64),
        mins=np.asarray([0.1], dtype=np.float64),
        top_distances=np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float64),
    )

    assert signals["mean_distance"][0] == pytest.approx(0.35)
    assert signals["top3_mean_distance"][0] == pytest.approx(0.2)
    assert signals["top5_mean_distance"][0] == pytest.approx(0.3)


def _minimal_raw_candidate_plan(**overrides: Any) -> dict[str, Any]:
    raw_plan = {
        "query_signature_ids": ["q0"],
        "query_views": ["full"],
        "query_authors": ["Alice"],
        "row_count": 1,
        "pair_count": 1,
        "row_query_signature_indices": np.asarray([0], dtype=np.uint32),
        "row_component_keys": ["c1"],
        "retrieval_scores": np.asarray([0.9], dtype=np.float32),
        "retrieval_ranks": np.asarray([1], dtype=np.uint16),
        "pair_row_indices": np.asarray([0], dtype=np.uint32),
        "left_signature_ids": ["q0"],
        "right_signature_ids": ["s1"],
        "component_members": {"c1": ["s1"]},
    }
    for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        raw_plan[raw_key] = np.asarray([""] if dtype is object else [0], dtype=dtype)
    raw_plan.update(overrides)
    return raw_plan


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"pair_row_indices": [5]}, "pair_row_indices.*row_count=1"),
        ({"left_signature_indices": [0]}, "legacy numeric pair indices"),
        ({"row_query_signature_indices": [1]}, "row_query_signature_indices.*query_signature_ids length=1"),
        ({"retrieval_ranks": [-1]}, "retrieval_ranks"),
        ({"retrieval_ranks": [0]}, "retrieval_ranks"),
        ({"row_query_year_missing": [-1]}, "row_query_year_missing.*non-0/1"),
        ({"query_views": []}, "query_views length must match query_signature_ids"),
        ({"query_views": ["typo"]}, "Unknown retrieval query_view"),
    ],
)
def test_raw_candidate_plan_rejects_invalid_rows(overrides, message):
    with pytest.raises(ValueError, match=message):
        build_linker_retrieval_batch_from_raw_plan_bundle(
            RawArrowPlanBundle.from_native_mapping(_minimal_raw_candidate_plan(**overrides)),
            signature_id_to_index={"q0": 0, "s1": 1},
        )


def test_subset_row_signals_rejects_non_1d_signals() -> None:
    with pytest.raises(ValueError, match="row signal 'bad' must be 1D"):
        runtime_module._subset_row_signals(
            {"bad": np.zeros((2, 2), dtype=np.float32)},
            np.asarray([0], dtype=np.int64),
            2,
        )


def test_constraint_row_signals_summarize_require_and_disallow_labels() -> None:
    candidate_batch = LinkerCandidateBatch(
        row_count=3,
        left_signature_indices=np.asarray([10, 10, 11, 11, 12], dtype=np.uint32),
        right_signature_indices=np.asarray([1, 2, 3, 4, 5], dtype=np.uint32),
        pair_row_indices=np.asarray([0, 0, 1, 1, 2], dtype=np.uint32),
    )
    labels = np.asarray(
        [
            -float(LARGE_INTEGER),
            float(LARGE_DISTANCE - LARGE_INTEGER),
            np.nan,
            float(LARGE_DISTANCE - LARGE_INTEGER),
            float(LARGE_DISTANCE - LARGE_INTEGER),
        ],
        dtype=np.float64,
    )

    signals = runtime_module._constraint_row_signals(candidate_batch, labels)

    np.testing.assert_allclose(signals["constraint_pair_count"], [2.0, 2.0, 1.0])
    np.testing.assert_allclose(signals["constraint_hit_count"], [2.0, 1.0, 1.0])
    np.testing.assert_allclose(signals["constraint_require_count"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(signals["constraint_disallow_count"], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(signals["constraint_disallow_fraction"], [0.5, 0.5, 1.0])


def test_runtime_pair_constraint_features_suppress_orcid_like_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, bool] = {}

    def fake_constraint_labels(
        _left: np.ndarray,
        _right: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        observed["suppress_orcid"] = bool(kwargs["suppress_orcid"])
        return np.asarray([np.nan], dtype=np.float64)

    monkeypatch.setattr(
        runtime_module.feature_port,
        "get_constraint_labels_index_arrays_rust",
        fake_constraint_labels,
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.asarray([0], dtype=np.uint32),
        right_signature_indices=np.asarray([1], dtype=np.uint32),
        pair_row_indices=np.asarray([0], dtype=np.uint32),
    )

    labels, _telemetry = runtime_module._resolve_candidate_batch_pair_labels_rust(
        candidate_batch=candidate_batch,
        signature_ids_by_index=("query", "seed"),
        partial_supervision={},
        use_default_constraints_as_supervision=True,
        dont_merge_cluster_seeds=True,
        n_jobs=1,
        featurizer=object(),
    )

    assert observed == {"suppress_orcid": True}
    assert np.isnan(labels[0])


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
    ) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
        del total_ram_bytes
        inverse: dict[str, list[str]] = {}
        for signature_id, cluster_id in self.seed_map.items():
            inverse.setdefault(cluster_id, []).append(signature_id)
        return dict(self.seed_map), dict(self.recluster_map), inverse, inverse

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


def _row_features(retrieval_scores: np.ndarray) -> dict[str, np.ndarray]:
    row_count = len(retrieval_scores)
    row_features = {column: np.zeros(row_count, dtype=np.float32) for column in PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS}
    row_features["min_distance"] = 1.0 - np.asarray(retrieval_scores, dtype=np.float32)
    return row_features


def _row_features_with_telemetry(retrieval_scores: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    return _row_features(retrieval_scores), {
        "generated_family_id_count": 0,
        "generic_family_override_count": 0,
    }


def _promoted_gate_config(score: float = 0.0) -> dict[str, Any]:
    scale = 200.0
    return logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[-scale, 0.0, scale]], dtype=np.float64),
        bias=np.asarray([scale * float(score), -10.0, -scale * float(score)], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="test",
    )


def _retrieval_batch(
    *,
    row_query_signature_indices: np.ndarray,
    row_component_keys: tuple[str, ...],
    retrieval_ranks: np.ndarray | None = None,
) -> LinkerRetrievalBatch:
    row_count = len(row_query_signature_indices)
    candidate_batch = _row_only_candidate_batch(
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
            "first_name_bucket": np.asarray(["multi_letter_first"] * row_count, dtype=object),
        },
    )


def _row_only_candidate_batch(
    *,
    row_query_signature_indices: np.ndarray,
    row_component_keys: tuple[str, ...],
    retrieval_ranks: np.ndarray | None = None,
) -> LinkerCandidateBatch:
    return LinkerCandidateBatch(
        row_count=len(row_query_signature_indices),
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=row_query_signature_indices,
        row_component_keys=row_component_keys,
        retrieval_ranks=retrieval_ranks,
    )


def _empty_feature_matrix(candidate_batch: LinkerCandidateBatch) -> LinkerFeatureMatrix:
    return LinkerFeatureMatrix(
        matrix=np.zeros((candidate_batch.row_count, len(promoted_linker_feature_columns())), dtype=np.float32),
        feature_columns=promoted_linker_feature_columns(),
        candidate_batch=candidate_batch,
        pairwise_stats=StaticPairwiseStats(candidate_batch.row_count),
    )


def _compact_decisions(probabilities, *, gate=0.5, queries=None, row_signals=None, hard_excluded_rows=None):
    """Run the real gate with explicit candidate probabilities and hard evidence."""
    count = len(probabilities)
    batch = _row_only_candidate_batch(
        row_query_signature_indices=np.asarray([10] * count if queries is None else queries, dtype=np.uint32),
        row_component_keys=tuple(f"c{index}" for index in range(count)),
        retrieval_ranks=np.arange(1, count + 1, dtype=np.uint16),
    )
    signals = {"first_name_bucket": np.asarray(["multi_letter_first"] * count, dtype=object)}
    signals.update({name: np.asarray(values, dtype=np.float32) for name, values in (row_signals or {}).items()})
    return _predict_incremental_link_or_abstain_compact(
        StaticArtifact(np.asarray(probabilities), _promoted_gate_config(gate)),
        _empty_feature_matrix(batch),
        row_signals=signals,
        hard_excluded_rows=None if hard_excluded_rows is None else np.asarray(hard_excluded_rows),
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
            "first_name_bucket": np.asarray(["multi_letter_first"] * row_count, dtype=object),
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


@pytest.mark.parametrize("model_nan,aggregate_nan", [(0.0, np.nan), (np.nan, 0.0)])
def test_fused_native_features_scoring_and_constraints(tmp_path, monkeypatch, model_nan, aggregate_nan):
    dataset = build_arrow_training_dataset(build_dummy_dataset("fused", name_counts_index=True), tmp_path)
    native = runtime_module.feature_port._get_rust_featurizer(dataset)
    pairs = [(0, 1), (0, 2), (3, 4), (0, 3), (1, 4), (2, 5)]
    rows = np.asarray([0, 0, 1, 2, 2, 2], dtype=np.uint32)
    candidate_batch = LinkerCandidateBatch(
        row_count=4,
        left_signature_indices=np.asarray([left for left, _ in pairs], dtype=np.uint32),
        right_signature_indices=np.asarray([right for _, right in pairs], dtype=np.uint32),
        pair_row_indices=rows,
    )
    labels = np.asarray(
        [np.nan, -LARGE_INTEGER, LARGE_DISTANCE - LARGE_INTEGER, np.nan, LARGE_DISTANCE - LARGE_INTEGER, np.nan],
        dtype=np.float64,
    )
    raw_features = np.asarray(native.featurize_pairs_matrix_indexed(pairs, None, 1, np.nan))
    assert np.isnan(raw_features[np.isnan(labels), :7]).any()

    class RecordingClassifier(ConstantDistanceClassifier):
        def predict_proba(self, features):
            self.features = features.copy()
            return super().predict_proba(features)

    main, nameless = RecordingClassifier(0.2), RecordingClassifier(0.6)
    # Observe both native boundaries while still executing their actual implementations.
    monkeypatch.setattr("s2and.thread_config.os.cpu_count", lambda: 5)
    threads = []
    for name in (
        "build_linker_pair_features_and_aggregate_stats_arrays_rust",
        "build_linker_pair_distance_accumulators_rust",
    ):
        original = getattr(runtime_module.feature_port, name)

        def record(*args, _original=original, **kwargs):
            threads.append(kwargs["num_threads"])
            return _original(*args, **kwargs)

        monkeypatch.setattr(runtime_module.feature_port, name, record)

    result = compute_candidate_batch_pairwise_model_and_aggregate_stats(
        dataset,
        candidate_batch,
        classifier=main,
        featurizer_info=FeaturizationInfo(features_to_use=["name_similarity"]),
        nameless_classifier=nameless,
        nameless_featurizer_info=FeaturizationInfo(features_to_use=["affiliation_similarity"]),
        pair_labels=labels,
        n_jobs=-1,
        featurizer=native,
        pairwise_model_nan_value=model_nan,
        pairwise_aggregate_nan_value=aggregate_nan,
    )
    assert threads == [5, 5]
    for scorer, indices in ((main, range(6)), (nameless, [6])):
        expected = raw_features[np.isnan(labels)][:, indices].copy()
        if not np.isnan(model_nan):
            expected[np.isnan(expected)] = model_nan
        np.testing.assert_array_equal(scorer.features, expected)

    # Dense reduction is an independent oracle for the fused native aggregation.
    aggregate = raw_features[:, PROMOTED_PAIRWISE_AGG_FEATURE_INDICES].copy()
    assert np.isnan(aggregate).any()
    if not np.isnan(aggregate_nan):
        aggregate[np.isnan(aggregate)] = aggregate_nan
    stats = result.pairwise_stats
    assert tuple(stats.aggregate_feature_columns) == tuple(PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS)
    np.testing.assert_array_equal(stats.counts, [2, 1, 3, 0])
    for row in range(4):
        values = aggregate[rows == row]
        valid = ~np.isnan(values)
        np.testing.assert_array_equal(stats.valid_counts[row], valid.sum(axis=0))
        np.testing.assert_allclose(stats.sums[row], np.nansum(values, axis=0))
        np.testing.assert_allclose(stats.mins[row], np.min(np.where(valid, values, np.inf), axis=0, initial=np.inf))
        np.testing.assert_allclose(stats.maxs[row], np.max(np.where(valid, values, -np.inf), axis=0, initial=-np.inf))
    np.testing.assert_allclose(result.row_signals["min_distance"], [0, LARGE_DISTANCE, 0.4, 1])
    for signal in ("mean_distance", "top3_mean_distance", "top5_mean_distance"):
        np.testing.assert_allclose(result.row_signals[signal], [0.2, LARGE_DISTANCE, (LARGE_DISTANCE + 0.8) / 3, 1])
    np.testing.assert_array_equal(result.row_signals["pair_count"], [2, 1, 3, 0])
    assert result.telemetry["hard_disallow_distance_pair_count"] == 2
    assert result.telemetry["index_remap_bytes_per_pair"] == 8
    assert result.telemetry["predicted_index_remap_bytes"] == stats.chunk_plan.chunk_pairs * 8


def test_fused_pairwise_model_rust_distance_accumulator_matches_python_large(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = build_dummy_dataset("dummy_linker_rust_distance_accumulator_parity", name_counts_index=True)
    dataset = build_arrow_training_dataset(dataset, tmp_path)
    rust_featurizer = runtime_module.feature_port._get_rust_featurizer(dataset)  # noqa: SLF001
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


@pytest.mark.parametrize("gate", [0.0, 1.1], ids=["accept", "reject"])
def test_compact_link_or_abstain_scores_artifact_rows_and_applies_gate(tmp_path: Path, gate: float) -> None:
    booster, _fixture = build_tiny_promoted_booster()
    artifact_dir = tmp_path / "artifact"
    save_incremental_linking_artifact(
        booster,
        artifact_dir,
        gate_config=_promoted_gate_config(gate),
        target_spec={},
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
    )
    artifact = load_incremental_linking_artifact(artifact_dir)
    candidate_batch = _row_only_candidate_batch(
        row_query_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        row_component_keys=("c_low", "c_high", "c_single"),
        retrieval_ranks=np.asarray([2, 1, 1], dtype=np.uint16),
    )
    feature_matrix = assemble_linker_feature_matrix(
        candidate_batch,
        _row_features(np.asarray([0.1, 0.9, 0.8], dtype=np.float32)),
        pairwise_stats=StaticPairwiseStats(row_count=3),
    )

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        feature_matrix,
        row_signals={"first_name_bucket": np.asarray(["multi_letter_first"] * 3, dtype=object)},
    )

    assert len(result.probabilities) == 3
    assert [decision.query_signature_index for decision in result.decisions] == [10, 11]
    assert [(decision.action, decision.component_key) for decision in result.decisions] == (
        [("link", "c_high"), ("link", "c_single")] if gate == 0 else [("abstain", None)] * 2
    )


def test_compact_link_or_abstain_applies_numpy_logistic_gate_feature() -> None:
    scale = 200.0
    artifact = StaticArtifact(
        np.asarray([0.60, 0.55, 0.40], dtype=np.float64),
        gate_config=logistic_gate_config(
            feature_names=("score_margin",),
            weights=np.asarray([[-scale, 0.0, scale]], dtype=np.float64),
            bias=np.asarray([scale * 0.04, -10.0, -scale * 0.04], dtype=np.float64),
            missing_values=np.asarray([0.0], dtype=np.float64),
            calibration_mode="test",
        ),
    )
    candidate_batch = _row_only_candidate_batch(
        row_query_signature_indices=np.asarray([10, 10, 11], dtype=np.uint32),
        row_component_keys=("c_margin", "c_runner_up", "c_single"),
        retrieval_ranks=np.asarray([1, 2, 1], dtype=np.uint16),
    )
    feature_matrix = _empty_feature_matrix(candidate_batch)
    row_signals: dict[str, Any] = {
        "first_name_bucket": np.asarray(
            ["single_letter_first", "single_letter_first", "multi_letter_first"],
            dtype=object,
        ),
    }

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        feature_matrix,
        row_signals=row_signals,
    )

    assert [decision.action for decision in result.decisions] == ["link", "abstain"]
    assert result.decisions[0].component_key == "c_margin"
    assert result.decisions[0].score_margin == pytest.approx(0.05)


def test_compact_logistic_gate_can_use_materialized_bucket_feature() -> None:
    artifact = StaticArtifact(
        np.asarray([0.60], dtype=np.float64),
        gate_config=logistic_gate_config(
            feature_names=("first_name_bucket_multi_letter_first",),
            weights=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float64),
            bias=np.asarray([0.0, 0.0, -2.5], dtype=np.float64),
            missing_values=np.asarray([0.0], dtype=np.float64),
            calibration_mode="test",
        ),
    )
    candidate_batch = _row_only_candidate_batch(
        row_query_signature_indices=np.asarray([10], dtype=np.uint32),
        row_component_keys=("c_single",),
        retrieval_ranks=np.asarray([1], dtype=np.uint16),
    )

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        _empty_feature_matrix(candidate_batch),
        row_signals={"first_name_bucket": np.asarray(["multi_letter_first"], dtype=object)},
    )

    assert result.decisions[0].action == "link"


@pytest.mark.parametrize("force", ["constraint_require_count", "orcid_match"])
def test_compact_forced_link_keeps_full_group_runner_up(force):
    """A forced low-score row beats both the model gate and a high-score rival."""
    result = _compact_decisions(
        [0.95, 0.10],
        gate=0.99,
        row_signals={
            force: [0, 1],
            "constraint_pair_count": [1, 1],
            "constraint_disallow_count": [0, int(force == "orcid_match")],
            "constraint_disallow_fraction": [0, int(force == "orcid_match")],
        },
    )
    decision = result.decisions[0]
    assert (decision.action, decision.component_key) == ("link", "c1")
    assert decision.score == pytest.approx(0.10)
    assert decision.runner_up_score == pytest.approx(0.95)
    assert decision.score_margin == pytest.approx(-0.85)


@pytest.mark.parametrize(
    "excluded,message",
    [
        (None, "constraint_require_conflicting_candidate_components"),
        ([True, False], "cluster_seed_disallow_conflicts_with_require_constraint"),
    ],
)
def test_compact_rejects_conflicting_requirements(excluded, message):
    requires = [1, 1] if excluded is None else [1, 0]
    with pytest.raises(ValueError, match=message):
        _compact_decisions(
            [0.95, 0.60],
            hard_excluded_rows=excluded,
            row_signals={"constraint_pair_count": requires, "constraint_require_count": requires},
        )


def test_compact_constraint_veto_recomputes_gate_only_for_affected_query(monkeypatch):
    gate_row_counts = []
    original = runtime_module.build_runtime_logistic_gate_matrix

    def recording_gate_builder(*args, **kwargs):
        gate_row_counts.append(args[1].candidate_batch.row_count)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "build_runtime_logistic_gate_matrix", recording_gate_builder)
    result = _compact_decisions(
        [0.95, 0.80, 0.90, 0.10],
        queries=[10, 10, 11, 11],
        row_signals={
            "constraint_pair_count": [1, 1, 1, 1],
            "constraint_disallow_count": [1, 0, 0, 0],
            "constraint_disallow_fraction": [1, 0, 0, 0],
        },
    )
    assert gate_row_counts == [4, 1]
    assert [decision.component_key for decision in result.decisions] == ["c1", "c2"]
    assert result.decisions[0].score == pytest.approx(0.80)


@pytest.mark.parametrize("hard", [False, True], ids=["constraint-veto", "hard-exclusion"])
def test_compact_abstains_when_every_candidate_is_excluded(hard):
    result = _compact_decisions(
        [0.95, 0.80],
        hard_excluded_rows=[True, True] if hard else None,
        row_signals={
            "constraint_pair_count": [1, 1],
            "constraint_disallow_count": [1, 1],
            "constraint_disallow_fraction": [1, 1],
        },
    )
    decision = result.decisions[0]
    assert (decision.action, decision.component_key, decision.row_index) == ("abstain", None, None)
    if hard:
        assert decision.score is None


def test_runtime_orcid_force_link_policy_is_independent_of_model_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = StaticArtifact(
        np.asarray([0.95, 0.10], dtype=np.float64),
        gate_config=_promoted_gate_config(0.99),
    )
    retrieval_batch = _retrieval_batch(
        row_query_signature_indices=np.asarray([10, 10], dtype=np.uint32),
        row_component_keys=("non_orcid_high_score", "orcid_low_score"),
        retrieval_ranks=np.asarray([1, 2], dtype=np.uint16),
    )
    retrieval_batch.row_signals["orcid_match"] = np.asarray([0, 1], dtype=np.uint8)
    monkeypatch.setattr(
        runtime_module,
        "build_promoted_non_pairwise_row_features_with_telemetry",
        lambda _candidate_batch, _row_signals: _row_features_with_telemetry(np.asarray([0.95, 0.10], dtype=np.float32)),
    )

    force_enabled = _predict_incremental_link_or_abstain_retrieved_candidates(
        artifact,
        retrieval_batch,
        pairwise_stats=StaticPairwiseStats(row_count=2),
        orcid_force_link_enabled=True,
    )
    force_disabled = _predict_incremental_link_or_abstain_retrieved_candidates(
        artifact,
        retrieval_batch,
        pairwise_stats=StaticPairwiseStats(row_count=2),
        orcid_force_link_enabled=False,
    )

    assert force_enabled.compact_result.decisions[0].component_key == "orcid_low_score"
    assert force_disabled.compact_result.decisions[0].action == "abstain"


def test_constraint_disallow_veto_policy_pins_two_pair_half_disallow_fall_through() -> None:
    """Pin the intentional veto policy shape.

    Veto fires on unanimous disallow evidence (any sample size) or >=80% with
    pair_count >= 3. The 2-pair/1-disallow case (50%) deliberately falls
    through to the model score: derived constraint disallows are noisy
    evidence, and vetoing at 50% for n=2 while requiring 80% for n>=3 would be
    non-monotonic strictness. Request-level hard disallows are enforced by
    candidate exclusion upstream, not by this veto layer.
    """

    row_signals = {
        "constraint_pair_count": np.asarray([1.0, 2.0, 2.0, 3.0, 3.0, 5.0], dtype=np.float32),
        "constraint_disallow_count": np.asarray([1.0, 1.0, 2.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "constraint_disallow_fraction": np.asarray([1.0, 0.5, 1.0, 2.0 / 3.0, 1.0, 0.8], dtype=np.float32),
    }

    veto = runtime_module._constraint_disallow_veto_signal(row_signals, 6)

    assert veto is not None
    assert veto.tolist() == [True, False, True, False, True, True]


def test_hard_exclusion_beats_orcid_and_preserves_eligible_candidate():
    result = _compact_decisions(
        [0.95, 0.60],
        row_signals={"orcid_match": [1, 0]},
        hard_excluded_rows=[True, False],
    )
    assert (result.decisions[0].action, result.decisions[0].component_key) == ("link", "c1")
    assert result.decision_telemetry["cluster_seed_disallow_excluded_row_count"] == 1
    assert result.decision_telemetry["cluster_seed_disallow_excluded_query_count"] == 1


def test_cluster_seed_disallow_excluded_rows_builds_query_component_mask() -> None:
    candidate_batch = _row_only_candidate_batch(
        row_query_signature_indices=np.asarray([0, 0, 1], dtype=np.uint32),
        row_component_keys=("c1", "c2", "c1"),
    )
    signature_id_to_index = {"q1": 0, "q2": 1}

    mask = runtime_module._cluster_seed_disallow_excluded_rows(
        candidate_batch,
        signature_id_to_index=signature_id_to_index,
        excluded_components_by_query_id={"q1": {"c1"}, "unknown_query": {"c1"}, "q2": set()},
    )

    assert mask is not None
    # Only q1's rows in c1 are excluded; q2's c1 row is untouched.
    assert mask.tolist() == [True, False, False]

    no_match = runtime_module._cluster_seed_disallow_excluded_rows(
        candidate_batch,
        signature_id_to_index=signature_id_to_index,
        excluded_components_by_query_id={"q1": {"c_absent"}},
    )
    assert no_match is None


def test_query_disallow_partner_ids_collects_query_pairs_from_both_channels() -> None:
    import s2and.incremental_linking.production as production_module

    partners = production_module._query_disallow_partner_ids(
        ["q1", "q2", "q3"],
        {("q1", "q2"), ("q1", "seed9"), ("q3", "q3")},
        {
            ("q2", "q3"): float(LARGE_DISTANCE),  # disallow between two queries
            ("q1", "q3"): 0.0,  # require pair: not a disallow, ignored here
            ("q2", "seed9"): float(LARGE_DISTANCE),  # query-vs-seed: planner-enforced
        },
    )

    assert partners == {"q1": {"q2"}, "q2": {"q1", "q3"}, "q3": {"q2"}}


def test_private_retrieved_candidate_slice_scores_matrix_and_records_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = StaticArtifact(
        np.asarray([0.1, 0.9, 0.8], dtype=np.float64),
        gate_config=_promoted_gate_config(0.0),
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
        artifact,
        retrieval_batch,
        pairwise_stats=StaticPairwiseStats(row_count=3),
    )

    assert result.feature_matrix.matrix.shape == (3, len(promoted_linker_feature_columns()))
    assert [decision.component_key for decision in result.compact_result.decisions] == ["c_high", "c_single"]
    assert {key: value for key, value in result.telemetry.items() if not key.startswith("native_scorer_")} == {
        "candidate_row_count": 3,
        "pair_count": 0,
        "no_candidate_query_count": 0,
        "decision_count": 2,
        "link_count": 2,
        "abstain_count": 0,
        "cluster_seed_disallow_excluded_row_count": 0,
        "cluster_seed_disallow_excluded_query_count": 0,
        "row_feature_generated_family_id_count": 0,
        "row_feature_generic_family_override_count": 0,
    }
    assert result.telemetry["native_scorer_chunk_rows"] == 3
    assert result.telemetry["native_scorer_chunk_count"] == 1
    assert result.telemetry["native_scorer_predicted_peak_delta_bytes"] == 3 * (53 * 4 + 8)


def test_private_retrieved_candidate_slice_returns_no_candidate_abstains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = StaticArtifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
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
        artifact,
        retrieval_batch,
        pairwise_stats=StaticPairwiseStats(row_count=0),
        no_candidate_query_signature_indices=np.asarray([42], dtype=np.uint32),
    )

    assert len(result.compact_result.probabilities) == 0
    assert result.compact_result.decisions[0].query_signature_index == 42
    assert result.compact_result.decisions[0].action == "abstain"
    assert result.telemetry["no_candidate_query_count"] == 1


def test_private_retrieved_candidate_slice_rejects_partial_supervision() -> None:
    artifact = StaticArtifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
    retrieval_batch = _retrieval_batch(
        row_query_signature_indices=np.asarray([], dtype=np.uint32),
        row_component_keys=(),
    )

    with pytest.raises(NotImplementedError, match="partial supervision"):
        _predict_incremental_link_or_abstain_retrieved_candidates(
            artifact,
            retrieval_batch,
            pairwise_stats=StaticPairwiseStats(row_count=0),
            partial_supervision={("q", "m"): "require"},
        )


def test_production_query_author_row_signals_reuses_retrieval_signal() -> None:
    retrieval_batch = _production_retrieval_batch(
        row_query_signature_indices=np.asarray([0, 0], dtype=np.uint32),
        row_component_keys=("c1", "c2"),
    )
    retrieval_batch.row_signals["query_author"] = np.asarray(["Ada Lovelace", "Ada Lovelace"], dtype=object)

    assert (
        runtime_module._production_query_author_row_signals(
            retrieval_batch,
            query_signature_id_by_index={0: "q1"},
            query_by_signature_id={"q1": SimpleNamespace(query_author="ignored")},
        )
        == {}
    )


def test_query_author_for_gate_fallback_includes_full_signature_name() -> None:
    query = SimpleNamespace(
        query_author="",
        author_info_first="Ada",
        author_info_middle="Byron",
        author_info_last="Lovelace",
        author_info_suffix="PhD",
    )

    assert runtime_module._query_author_for_gate(query) == "Ada Byron Lovelace PhD"


def test_from_retrieval_validates_partial_supervision_against_full_seed_map() -> None:
    featurizer = FakeRuntimeFeaturizer(["q1", "s1", "s2"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = StaticArtifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
    retrieval_batch = _production_retrieval_batch(
        row_query_signature_indices=np.asarray([], dtype=np.uint32),
        row_component_keys=(),
    )

    with pytest.raises(ValueError, match="partial_supervision_require_outside_retrieval_window"):
        runtime_module._predict_incremental_link_or_abstain_production_from_retrieval_private(  # noqa: SLF001
            clusterer,
            artifact,
            featurizer=featurizer,
            retrieval_batch=retrieval_batch,
            queries=[object()],
            query_signature_ids=["q1"],
            partial_supervision={("q1", "s2"): 0},
            cluster_seeds_require={"s1": "c1"},
            partial_supervision_seed_signature_to_component={"s1": "c1", "s2": "c2"},
        )


def test_from_retrieval_records_artifact_retrieval_top_k_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    featurizer = FakeRuntimeFeaturizer(["q1", "s1"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = StaticArtifact(np.asarray([0.9], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
    artifact.retrieval_top_k = 37
    retrieval_batch = _production_retrieval_batch(
        row_query_signature_indices=np.asarray([0], dtype=np.uint32),
        row_component_keys=("c1",),
        left_signature_indices=np.asarray([0], dtype=np.uint32),
        right_signature_indices=np.asarray([1], dtype=np.uint32),
        pair_row_indices=np.asarray([0], dtype=np.uint32),
    )
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

    result = runtime_module._predict_incremental_link_or_abstain_production_from_retrieval_private(  # noqa: SLF001
        clusterer,
        artifact,
        featurizer=featurizer,
        retrieval_batch=retrieval_batch,
        queries=[object()],
        query_signature_ids=["q1"],
        cluster_seeds_require={"s1": "c1"},
    )

    assert result.telemetry["retrieval_top_k"] == 37

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.incremental_linking.runtime as runtime_module
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import load_incremental_linking_artifact, save_incremental_linking_artifact
from s2and.incremental_linking.features import (
    PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS,
    LinkerFeatureMatrix,
    promoted_linker_feature_columns,
)
from s2and.incremental_linking.features import (
    assemble_linker_feature_matrix as _assemble_linker_feature_matrix_impl,
)
from s2and.incremental_linking.linker_pairwise import (
    PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS,
    PROMOTED_PAIRWISE_AGG_FEATURE_INDICES,
    LinkerCandidateBatch,
    promoted_pairwise_aggregate_columns,
)
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from s2and.incremental_linking.retrieval import (
    RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS,
    RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
    LinkerRetrievalBatch,
    build_linker_retrieval_batch_from_raw_candidate_plan,
)
from s2and.incremental_linking.runtime import (
    CandidateBatchPairwiseModelResult,
    _predict_incremental_link_or_abstain_compact,
    naturalize_incremental_clusters,
    signature_id_to_index_map,
)
from s2and.incremental_linking.runtime import (
    _predict_incremental_link_or_abstain_production_private as _prod_private_impl,
)
from s2and.incremental_linking.runtime import (
    _predict_incremental_link_or_abstain_retrieved_candidates as _retrieved_candidates_impl,
)
from s2and.incremental_linking.runtime import (
    compute_candidate_batch_pairwise_model_and_aggregate_stats as _pairwise_model_stats_impl,
)
from tests.helpers import attach_arrow_featurizer_bundle, build_dummy_dataset, import_s2and_rust
from tests.promoted_linking_helpers import build_tiny_promoted_booster

runtime_module: Any = runtime_module

_HAS_RUST_LIGHTGBM, _RUST_LIGHTGBM_PAYLOAD = import_s2and_rust(
    required_module_attrs=("RustLightGBMBooster",),
    prefer_site_packages=True,
)
requires_rust_lightgbm = pytest.mark.skipif(
    not _HAS_RUST_LIGHTGBM,
    reason=f"RustLightGBMBooster unavailable: {_RUST_LIGHTGBM_PAYLOAD!r}",
)


def assemble_linker_feature_matrix(*args: Any, **kwargs: Any) -> Any:
    return _assemble_linker_feature_matrix_impl(*args, **kwargs)


def compute_candidate_batch_pairwise_model_and_aggregate_stats(*args: Any, **kwargs: Any) -> Any:
    return _pairwise_model_stats_impl(*args, **kwargs)


def _predict_incremental_link_or_abstain_retrieved_candidates(*args: Any, **kwargs: Any) -> Any:
    return _retrieved_candidates_impl(*args, **kwargs)


def _predict_incremental_link_or_abstain_production_private(*args: Any, **kwargs: Any) -> Any:
    return _prod_private_impl(*args, **kwargs)


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
        self.metadata = SimpleNamespace(
            feature_columns=promoted_linker_feature_columns(),
            gate_config=gate_config,
            retrieval_top_k=25,
        )

    def predict_probabilities(self, matrix: np.ndarray, *, num_threads: int | None = None) -> np.ndarray:
        assert matrix.shape[0] == len(self.probabilities)
        self.last_num_threads = num_threads
        return self.probabilities


def _static_pairwise_stats(row_count: int) -> Any:
    return StaticPairwiseStats(row_count)


def _static_artifact(probabilities: np.ndarray, gate_config: dict[str, Any]) -> Any:
    return StaticArtifact(probabilities, gate_config)


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


def test_pairwise_predict_class0_does_not_require_num_threads_keyword_support() -> None:
    predictions = runtime_module._predict_pairwise_class0(
        RejectsNumThreadsClassifier(),
        np.asarray([[0.25], [0.75]], dtype=np.float64),
    )

    assert np.allclose(predictions, [0.25, 0.75])


def test_pairwise_predict_class0_uses_native_positive_probability_fast_path() -> None:
    predictions = runtime_module._predict_pairwise_class0(
        PositiveProbabilityClassifier(),
        np.asarray([[0.25], [0.75]], dtype=np.float64),
    )

    assert np.allclose(predictions, [0.75, 0.25])


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


def test_raw_candidate_plan_telemetry_preserves_seed_counts_for_window_reuse() -> None:
    fields = runtime_module._raw_candidate_plan_telemetry_fields(  # noqa: SLF001
        {
            "telemetry": {
                "window_plan_reused": 1,
                "signature_count": 9,
                "seed_signature_count": 4,
                "cluster_count": 2,
                "timings": {"total_secs": 0.5},
            }
        }
    )

    assert fields["raw_arrow_plan_signature_count"] == 0
    assert fields["raw_arrow_plan_seed_signature_count"] == 4
    assert fields["raw_arrow_plan_cluster_count"] == 2
    assert fields["raw_arrow_plan_total_secs"] == 0.5


def test_subset_raw_candidate_plan_preserves_unretrieved_component_members() -> None:
    raw_plan = {
        "schema_version": RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
        "query_signature_ids": ["q0", "q1"],
        "query_views": ["full", "full"],
        "query_authors": ["Alice", "Alice"],
        "row_count": 1,
        "pair_count": 1,
        "row_query_signature_indices": np.asarray([0], dtype=np.uint32),
        "row_component_keys": ["c1"],
        "retrieval_scores": np.asarray([0.9], dtype=np.float32),
        "retrieval_ranks": np.asarray([1], dtype=np.uint16),
        "pair_row_indices": np.asarray([0], dtype=np.uint32),
        "left_signature_ids": ["q0"],
        "right_signature_ids": ["s1"],
        "component_members": {"c1": ["s1"], "c2": ["s2"]},
        "telemetry": {},
    }
    for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        raw_plan[raw_key] = np.asarray([""] if dtype is object else [0], dtype=dtype)

    subset = runtime_module.subset_raw_candidate_plan_for_query_ids(raw_plan, ["q0"])

    assert subset["row_component_keys"] == ["c1"]
    assert subset["component_members"] == {"c1": ["s1"], "c2": ["s2"]}


def _minimal_raw_candidate_plan(**overrides: Any) -> dict[str, Any]:
    raw_plan = {
        "schema_version": RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
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


def test_raw_candidate_plan_rejects_pair_row_index_outside_row_count() -> None:
    raw_plan = _minimal_raw_candidate_plan(pair_row_indices=np.asarray([5], dtype=np.uint32))

    with pytest.raises(ValueError, match="pair_row_indices.*row_count=1"):
        runtime_module.subset_raw_candidate_plan_for_query_ids(raw_plan, ["q0"])


def test_raw_candidate_plan_rejects_legacy_numeric_pair_indices() -> None:
    raw_plan = _minimal_raw_candidate_plan(left_signature_indices=np.asarray([0], dtype=np.uint32))

    with pytest.raises(ValueError, match="legacy numeric pair indices"):
        build_linker_retrieval_batch_from_raw_candidate_plan(
            raw_plan,
            signature_id_to_index={"q0": 0, "s1": 1},
        )


def test_raw_candidate_plan_rejects_pair_left_id_that_disagrees_with_row_query() -> None:
    raw_plan = _minimal_raw_candidate_plan(left_signature_ids=["other-query"])

    with pytest.raises(ValueError, match="left_signature_ids must match"):
        build_linker_retrieval_batch_from_raw_candidate_plan(
            raw_plan,
            signature_id_to_index={"q0": 0, "other-query": 1, "s1": 2},
        )


def test_raw_candidate_plan_rejects_row_query_index_outside_query_count() -> None:
    raw_plan = _minimal_raw_candidate_plan(row_query_signature_indices=np.asarray([1], dtype=np.uint32))

    with pytest.raises(ValueError, match="row_query_signature_indices.*query_signature_ids length=1"):
        build_linker_retrieval_batch_from_raw_candidate_plan(
            raw_plan,
            signature_id_to_index={"q0": 0, "s1": 1},
        )


@pytest.mark.parametrize("retrieval_rank", [-1, 0])
def test_raw_candidate_plan_rejects_invalid_retrieval_rank(retrieval_rank: int) -> None:
    raw_plan = {
        "schema_version": RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
        "query_signature_ids": ["q0"],
        "query_views": ["full"],
        "query_authors": ["Alice"],
        "row_count": 1,
        "pair_count": 1,
        "row_query_signature_indices": np.asarray([0], dtype=np.uint32),
        "row_component_keys": ["c1"],
        "retrieval_scores": np.asarray([0.9], dtype=np.float32),
        "retrieval_ranks": [retrieval_rank],
        "pair_row_indices": np.asarray([0], dtype=np.uint32),
        "left_signature_ids": ["q0"],
        "right_signature_ids": ["s1"],
        "component_members": {"c1": ["s1"]},
    }
    for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        raw_plan[raw_key] = np.asarray([""] if dtype is object else [0], dtype=dtype)

    with pytest.raises(ValueError, match="retrieval_ranks"):
        build_linker_retrieval_batch_from_raw_candidate_plan(raw_plan, signature_id_to_index={"q0": 0, "s1": 1})


def test_raw_candidate_plan_rejects_invalid_uint8_flag() -> None:
    raw_plan = _minimal_raw_candidate_plan(row_query_year_missing=[-1])

    with pytest.raises(ValueError, match="row_query_year_missing.*non-0/1"):
        build_linker_retrieval_batch_from_raw_candidate_plan(
            raw_plan,
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


def test_private_production_forwards_four_element_seed_setup_to_retrieval_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_inverse = {"c1_0": ["s1"]}
    captured: dict[str, Any] = {}

    class FourElementSeedSetupClusterer(FakeProductionClusterer):
        def _build_incremental_seed_setup(
            self,
            _dataset: object,
            _partial_supervision: dict[tuple[str, str], int | float],
            _runtime_context: object,
            total_ram_bytes: int | None = None,
        ):
            del total_ram_bytes
            return {"s1": "c1_0"}, {"c1_0": "c1"}, {"c1": ["s1"]}, split_inverse

    def fake_from_retrieval_private(*_args: Any, **kwargs: Any) -> Any:
        captured["seed_setup"] = kwargs["seed_setup"]
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_from_retrieval_private",
        fake_from_retrieval_private,
    )

    result = cast(
        Any,
        runtime_module._predict_incremental_link_or_abstain_production_private(  # noqa: SLF001
            FourElementSeedSetupClusterer({"s1": "c1"}),
            _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0)),
            dataset=cast(Any, SimpleNamespace()),
            featurizer=FakeRuntimeFeaturizer(["s1"]),
            retriever=object(),
            queries=[],
            query_signature_ids=[],
        ),
    )

    assert result.ok is True
    assert captured["seed_setup"][3] is split_inverse


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


def _promoted_gate_config(score: float = 0.0, margin: float = 0.0) -> dict[str, Any]:
    del margin
    scale = 200.0
    return logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[-scale, 0.0, scale]], dtype=np.float64),
        bias=np.asarray([scale * float(score), -10.0, -scale * float(score)], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="test",
    )


def test_raw_arrow_runtime_rejects_none_path_before_rust_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_require_rust_runtime():
        raise AssertionError("invalid Arrow paths should be rejected before raw Arrow planning")

    monkeypatch.setattr(runtime_module.feature_port, "_require_rust_runtime", fail_require_rust_runtime)
    clusterer = SimpleNamespace(
        n_jobs=1,
        featurizer_info=FeaturizationInfo(features_to_use=[]),
        nameless_featurizer_info=None,
    )

    with pytest.raises(ValueError, match="signatures.*None"):
        runtime_module.predict_incremental_link_or_abstain_from_raw_arrow_paths(
            clusterer,
            _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0)),
            arrow_paths={"signatures": None, "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"},
        )


def test_raw_arrow_runtime_rejects_mismatched_query_view_length_before_featurizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_build_rust_featurizer_from_arrow_paths(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("invalid raw candidate plans should fail before featurizer construction")

    monkeypatch.setattr(
        runtime_module.feature_port,
        "build_rust_featurizer_from_arrow_paths",
        fail_build_rust_featurizer_from_arrow_paths,
    )
    clusterer = SimpleNamespace(
        n_jobs=1,
        featurizer_info=FeaturizationInfo(features_to_use=[]),
        nameless_featurizer_info=None,
    )

    with pytest.raises(ValueError, match="query_views length must match query_signature_ids"):
        runtime_module._predict_incremental_link_or_abstain_from_preplanned_raw_arrow(  # noqa: SLF001
            clusterer,
            _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0)),
            arrow_paths={"signatures": tmp_path / "signatures.arrow"},
            query_signature_ids=["q"],
            raw_candidate_plan=_minimal_raw_candidate_plan(
                query_signature_ids=["q"],
                left_signature_ids=["q"],
                query_views=[],
            ),
            rust_featurizer=None,
            allow_featurizer_build=False,
        )


def test_raw_arrow_runtime_rejects_unknown_query_view_before_featurizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_build_rust_featurizer_from_arrow_paths(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("invalid raw candidate plans should fail before featurizer construction")

    monkeypatch.setattr(
        runtime_module.feature_port,
        "build_rust_featurizer_from_arrow_paths",
        fail_build_rust_featurizer_from_arrow_paths,
    )
    clusterer = SimpleNamespace(
        n_jobs=1,
        featurizer_info=FeaturizationInfo(features_to_use=[]),
        nameless_featurizer_info=None,
    )

    with pytest.raises(ValueError, match="Unknown retrieval query_view"):
        runtime_module._predict_incremental_link_or_abstain_from_preplanned_raw_arrow(  # noqa: SLF001
            clusterer,
            _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0)),
            arrow_paths={"signatures": tmp_path / "signatures.arrow"},
            query_signature_ids=["q"],
            raw_candidate_plan=_minimal_raw_candidate_plan(
                query_signature_ids=["q"],
                left_signature_ids=["q"],
                query_views=["typo"],
            ),
            rust_featurizer=None,
            allow_featurizer_build=False,
        )


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
            "first_name_bucket": np.asarray(["multi_letter_first"] * row_count, dtype=object),
        },
    )


def _empty_feature_matrix(candidate_batch: LinkerCandidateBatch) -> LinkerFeatureMatrix:
    return LinkerFeatureMatrix(
        matrix=np.zeros((candidate_batch.row_count, len(promoted_linker_feature_columns())), dtype=np.float32),
        feature_columns=promoted_linker_feature_columns(),
        candidate_batch=candidate_batch,
        pairwise_stats=_static_pairwise_stats(candidate_batch.row_count),
    )


def test_compact_artifact_scoring_forwards_num_threads() -> None:
    artifact = _static_artifact(
        np.asarray([0.9], dtype=np.float64),
        gate_config=_promoted_gate_config(0.0),
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10], dtype=np.uint32),
        row_component_keys=("c0",),
    )

    _predict_incremental_link_or_abstain_compact(
        artifact,
        _empty_feature_matrix(candidate_batch),
        num_threads=7,
    )

    assert artifact.last_num_threads == 7


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
        pairwise_stats=_static_pairwise_stats(row_count),
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
    calls: list[dict[str, Any]] = []

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
        valid_counts = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.uint64)
        sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
        mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
        maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
        aggregate_positions = [position_by_index[int(feature_index)] for feature_index in aggregate_indices]
        for pair_offset, local_row in enumerate(row_indices):
            row = int(local_row)
            counts[row] += 1
            values = matrix[pair_offset, aggregate_positions].copy()
            valid = ~np.isnan(values)
            if not np.isnan(float(aggregate_nan_value)):
                values[~valid] = float(aggregate_nan_value)
                valid = np.ones_like(valid, dtype=bool)
            if np.any(valid):
                valid_counts[row, valid] += 1
                sums[row, valid] += values[valid]
                mins[row, valid] = np.minimum(mins[row, valid], values[valid])
                maxs[row, valid] = np.maximum(maxs[row, valid], values[valid])
        return matrix, counts, valid_counts, sums, mins, maxs

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
    assert tuple(result.pairwise_stats.aggregate_feature_columns) == tuple(PROMOTED_PAIRWISE_AGG_FEATURE_COLUMNS)
    np.testing.assert_array_equal(result.pairwise_stats.counts, np.asarray([2, 1, 2, 0], dtype=np.uint64))
    np.testing.assert_array_equal(result.pairwise_stats.valid_counts[:, 0], np.asarray([2, 1, 2, 0]))
    np.testing.assert_allclose(result.pairwise_stats.sums[:, 0], np.asarray([1.1, 0.3, 0.9, 0.0]))
    np.testing.assert_allclose(result.pairwise_stats.mins[:, 0], np.asarray([0.4, 0.3, 0.2, np.inf]))
    np.testing.assert_allclose(result.pairwise_stats.maxs[:, 0], np.asarray([0.7, 0.3, 0.7, -np.inf]))
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
        valid_counts = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.uint64)
        sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
        mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
        maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
        aggregate_positions = [position_by_index[int(feature_index)] for feature_index in aggregate_indices]
        for pair_offset, local_row in enumerate(row_indices):
            row = int(local_row)
            counts[row] += 1
            values = matrix[pair_offset, aggregate_positions].copy()
            valid = ~np.isnan(values)
            if not np.isnan(float(aggregate_nan_value)):
                values[~valid] = float(aggregate_nan_value)
                valid = np.ones_like(valid, dtype=bool)
            if np.any(valid):
                valid_counts[row, valid] += 1
                sums[row, valid] += values[valid]
                mins[row, valid] = np.minimum(mins[row, valid], values[valid])
                maxs[row, valid] = np.maximum(maxs[row, valid], values[valid])
        return matrix, counts, valid_counts, sums, mins, maxs

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
        valid_counts = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.uint64)
        sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
        mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
        maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
        for local_row in row_indices:
            row = int(local_row)
            counts[row] += 1
            valid_counts[row] += 1
            mins[row] = 0.0
            maxs[row] = 0.0
        return matrix, counts, valid_counts, sums, mins, maxs

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


def test_fused_pairwise_model_resolves_negative_n_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("s2and.thread_config.os.cpu_count", lambda: 5)
    seen_threads: list[int] = []
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.asarray([0], dtype=np.uint32),
        right_signature_indices=np.asarray([1], dtype=np.uint32),
        pair_row_indices=np.asarray([0], dtype=np.uint32),
    )

    class FeaturizerWithDistanceAccumulator:
        def linker_pair_distance_accumulators(self):
            return None

    def fake_build_arrays(
        _dataset,
        _left_signature_indices,
        _right_signature_indices,
        _row_indices,
        row_count,
        *,
        matrix_indices,
        aggregate_indices,
        num_threads,
        **_kwargs,
    ):
        seen_threads.append(int(num_threads))
        matrix = np.zeros((1, len(matrix_indices)), dtype=np.float64)
        matrix[:, 0] = 0.2
        return (
            matrix,
            np.ones(int(row_count), dtype=np.uint32),
            np.ones((int(row_count), len(aggregate_indices)), dtype=np.uint64),
            np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64),
            np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64),
            np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64),
        )

    def fake_distance_accumulators(
        _dataset,
        _row_indices,
        row_count,
        model_distances,
        *,
        num_threads,
        **_kwargs,
    ):
        seen_threads.append(int(num_threads))
        return (
            np.ones(int(row_count), dtype=np.uint64),
            np.asarray([float(model_distances[0])], dtype=np.float64),
            np.asarray([float(model_distances[0])], dtype=np.float64),
            np.asarray([[float(model_distances[0]), np.inf, np.inf, np.inf, np.inf]], dtype=np.float64),
            0,
        )

    monkeypatch.setattr(
        runtime_module.feature_port,
        "build_linker_pair_features_and_aggregate_stats_arrays_rust",
        fake_build_arrays,
    )
    monkeypatch.setattr(
        runtime_module.feature_port,
        "build_linker_pair_distance_accumulators_rust",
        fake_distance_accumulators,
    )

    compute_candidate_batch_pairwise_model_and_aggregate_stats(
        SimpleNamespace(),
        candidate_batch,
        classifier=FirstColumnDistanceClassifier(),
        featurizer_info=FeaturizationInfo(features_to_use=["name_similarity"]),
        n_jobs=-1,
        featurizer=FeaturizerWithDistanceAccumulator(),
    )

    assert seen_threads == [5, 5]


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
            np.ones((1, len(aggregate_indices)), dtype=np.uint64),
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
    tmp_path: Path,
) -> None:
    if not runtime_module.feature_port.rust_featurizer_available():
        raise pytest.skip.Exception("s2and_rust core runtime is unavailable")
    dataset = build_dummy_dataset("dummy_linker_rust_distance_accumulator_parity", load_name_counts=True)
    attach_arrow_featurizer_bundle(dataset, tmp_path)
    rust_featurizer = runtime_module.feature_port._get_rust_featurizer(dataset)  # noqa: SLF001
    if not hasattr(rust_featurizer, "linker_pair_distance_accumulators"):
        raise pytest.skip.Exception("linker_pair_distance_accumulators is unavailable")

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


@requires_rust_lightgbm
def test_compact_link_or_abstain_scores_artifact_rows_and_applies_gate(tmp_path: Path) -> None:
    booster, fixture = build_tiny_promoted_booster()
    save_incremental_linking_artifact(
        booster,
        tmp_path,
        prediction_fixture_matrix=fixture,
        gate_config=_promoted_gate_config(0.0),
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
        pairwise_stats=_static_pairwise_stats(row_count=3),
    )

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        feature_matrix,
        row_signals={"first_name_bucket": np.asarray(["multi_letter_first"] * 3, dtype=object)},
    )

    assert len(result.probabilities) == 3
    assert [decision.action for decision in result.decisions] == ["link", "link"]
    assert result.decisions[0].query_signature_index == 10
    assert result.decisions[0].component_key == "c_high"
    assert result.decisions[1].query_signature_index == 11
    assert result.decisions[1].component_key == "c_single"


@requires_rust_lightgbm
def test_compact_link_or_abstain_abstains_when_artifact_score_threshold_too_high(tmp_path: Path) -> None:
    booster, fixture = build_tiny_promoted_booster()
    save_incremental_linking_artifact(
        booster,
        tmp_path,
        prediction_fixture_matrix=fixture,
        gate_config=_promoted_gate_config(1.1),
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
        pairwise_stats=_static_pairwise_stats(row_count=1),
    )

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        feature_matrix,
        row_signals={"first_name_bucket": np.asarray(["multi_letter_first"], dtype=object)},
    )

    assert result.decisions[0].action == "abstain"
    assert result.decisions[0].component_key is None


def test_compact_link_or_abstain_single_candidate_uses_logistic_score_feature() -> None:
    artifact = _static_artifact(
        np.asarray([0.9], dtype=np.float64),
        gate_config=_promoted_gate_config(0.95),
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10], dtype=np.uint32),
        row_component_keys=("c0",),
    )
    feature_matrix = _empty_feature_matrix(candidate_batch)

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        feature_matrix,
        row_signals={"first_name_bucket": np.asarray(["multi_letter_first"], dtype=object)},
    )

    assert result.decisions[0].action == "abstain"


def test_compact_link_or_abstain_applies_numpy_logistic_gate_feature() -> None:
    scale = 200.0
    artifact = _static_artifact(
        np.asarray([0.60, 0.55, 0.40], dtype=np.float64),
        gate_config=logistic_gate_config(
            feature_names=("score_margin",),
            weights=np.asarray([[-scale, 0.0, scale]], dtype=np.float64),
            bias=np.asarray([scale * 0.04, -10.0, -scale * 0.04], dtype=np.float64),
            missing_values=np.asarray([0.0], dtype=np.float64),
            calibration_mode="test",
        ),
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


def test_compact_gate_rejects_non_logistic_config() -> None:
    artifact = _static_artifact(
        np.asarray([0.60], dtype=np.float64),
        gate_config={
            "model_type": "legacy_thresholds",
        },
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10], dtype=np.uint32),
        row_component_keys=("c_single",),
        retrieval_ranks=np.asarray([1], dtype=np.uint16),
    )

    with pytest.raises(ValueError, match="Unsupported logistic gate model_type"):
        _predict_incremental_link_or_abstain_compact(
            artifact,
            _empty_feature_matrix(candidate_batch),
            row_signals={},
        )


def test_compact_logistic_gate_can_use_materialized_bucket_feature() -> None:
    artifact = _static_artifact(
        np.asarray([0.60], dtype=np.float64),
        gate_config=logistic_gate_config(
            feature_names=("first_name_bucket_multi_letter_first",),
            weights=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float64),
            bias=np.asarray([0.0, 0.0, -2.5], dtype=np.float64),
            missing_values=np.asarray([0.0], dtype=np.float64),
            calibration_mode="test",
        ),
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=1,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
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


def test_compact_constraint_require_forces_link() -> None:
    artifact = _static_artifact(
        np.asarray([0.95, 0.10], dtype=np.float64),
        gate_config=_promoted_gate_config(0.99),
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=2,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10], dtype=np.uint32),
        row_component_keys=("non_require_high_score", "require_low_score"),
        retrieval_ranks=np.asarray([1, 2], dtype=np.uint16),
    )

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        _empty_feature_matrix(candidate_batch),
        row_signals={
            "constraint_pair_count": np.asarray([1.0, 1.0], dtype=np.float32),
            "constraint_require_count": np.asarray([0.0, 1.0], dtype=np.float32),
            "constraint_disallow_count": np.asarray([0.0, 0.0], dtype=np.float32),
            "constraint_disallow_fraction": np.asarray([0.0, 0.0], dtype=np.float32),
        },
    )

    assert result.decisions[0].action == "link"
    assert result.decisions[0].component_key == "require_low_score"
    assert result.decisions[0].score == pytest.approx(0.10)
    # Runner-up margin is reported against the full query group (the non-required
    # candidate at 0.95), not just within the constraint-required subset.
    assert result.decisions[0].runner_up_score == pytest.approx(0.95)
    assert result.decisions[0].score_margin == pytest.approx(-0.85)


def test_compact_constraint_require_rejects_conflicting_candidate_components() -> None:
    artifact = _static_artifact(
        np.asarray([0.95, 0.90], dtype=np.float64),
        gate_config=_promoted_gate_config(0.0),
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=2,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10], dtype=np.uint32),
        row_component_keys=("required_component_a", "required_component_b"),
        retrieval_ranks=np.asarray([1, 2], dtype=np.uint16),
    )

    with pytest.raises(ValueError, match="constraint_require_conflicting_candidate_components"):
        _predict_incremental_link_or_abstain_compact(
            artifact,
            _empty_feature_matrix(candidate_batch),
            row_signals={
                "constraint_pair_count": np.asarray([1.0, 1.0], dtype=np.float32),
                "constraint_require_count": np.asarray([1.0, 1.0], dtype=np.float32),
                "constraint_disallow_count": np.asarray([0.0, 0.0], dtype=np.float32),
                "constraint_disallow_fraction": np.asarray([0.0, 0.0], dtype=np.float32),
            },
        )


def test_compact_constraint_disallow_vetoes_single_member_candidate_and_chooses_next() -> None:
    artifact = _static_artifact(
        np.asarray([0.95, 0.80], dtype=np.float64),
        gate_config=_promoted_gate_config(0.5),
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=2,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10], dtype=np.uint32),
        row_component_keys=("disallowed_high_score", "eligible_lower_score"),
        retrieval_ranks=np.asarray([1, 2], dtype=np.uint16),
    )

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        _empty_feature_matrix(candidate_batch),
        row_signals={
            "constraint_pair_count": np.asarray([1.0, 1.0], dtype=np.float32),
            "constraint_require_count": np.asarray([0.0, 0.0], dtype=np.float32),
            "constraint_disallow_count": np.asarray([1.0, 0.0], dtype=np.float32),
            "constraint_disallow_fraction": np.asarray([1.0, 0.0], dtype=np.float32),
        },
    )

    assert result.decisions[0].action == "link"
    assert result.decisions[0].component_key == "eligible_lower_score"
    assert result.decisions[0].score == pytest.approx(0.80)


def test_compact_constraint_veto_recomputes_gate_only_for_affected_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _static_artifact(
        np.asarray([0.95, 0.80, 0.90, 0.10], dtype=np.float64),
        gate_config=_promoted_gate_config(0.5),
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=4,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10, 11, 11], dtype=np.uint32),
        row_component_keys=("vetoed_high_score", "eligible_lower_score", "q2_high_score", "q2_low_score"),
        retrieval_ranks=np.asarray([1, 2, 1, 2], dtype=np.uint16),
    )
    gate_row_counts: list[int] = []
    original_gate_builder = runtime_module.build_runtime_logistic_gate_matrix

    def recording_gate_builder(*args: Any, **kwargs: Any):
        feature_matrix = args[1]
        gate_row_counts.append(int(feature_matrix.candidate_batch.row_count))
        return original_gate_builder(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "build_runtime_logistic_gate_matrix", recording_gate_builder)

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        _empty_feature_matrix(candidate_batch),
        row_signals={
            "constraint_pair_count": np.ones(4, dtype=np.float32),
            "constraint_require_count": np.zeros(4, dtype=np.float32),
            "constraint_disallow_count": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "constraint_disallow_fraction": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        },
    )

    assert gate_row_counts == [4, 1]
    assert result.decisions[0].component_key == "eligible_lower_score"
    assert result.decisions[1].component_key == "q2_high_score"


def test_compact_constraint_disallow_abstains_when_all_candidate_rows_vetoed() -> None:
    artifact = _static_artifact(
        np.asarray([0.95, 0.80], dtype=np.float64),
        gate_config=_promoted_gate_config(0.5),
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=2,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10], dtype=np.uint32),
        row_component_keys=("disallowed_high_score", "disallowed_lower_score"),
        retrieval_ranks=np.asarray([1, 2], dtype=np.uint16),
    )

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        _empty_feature_matrix(candidate_batch),
        row_signals={
            "constraint_pair_count": np.asarray([1.0, 1.0], dtype=np.float32),
            "constraint_require_count": np.asarray([0.0, 0.0], dtype=np.float32),
            "constraint_disallow_count": np.asarray([1.0, 1.0], dtype=np.float32),
            "constraint_disallow_fraction": np.asarray([1.0, 1.0], dtype=np.float32),
        },
    )

    assert result.decisions[0].action == "abstain"
    assert result.decisions[0].component_key is None
    assert result.decisions[0].row_index is None


def test_compact_orcid_match_forces_link_and_beats_non_orcid_rows() -> None:
    artifact = _static_artifact(
        np.asarray([0.95, 0.10], dtype=np.float64),
        gate_config=_promoted_gate_config(0.99),
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=2,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        row_query_signature_indices=np.asarray([10, 10], dtype=np.uint32),
        row_component_keys=("non_orcid_high_score", "orcid_low_score"),
        retrieval_ranks=np.asarray([1, 2], dtype=np.uint16),
    )

    result = _predict_incremental_link_or_abstain_compact(
        artifact,
        _empty_feature_matrix(candidate_batch),
        row_signals={
            "orcid_match": np.asarray([0.0, 1.0], dtype=np.float32),
            "constraint_pair_count": np.asarray([1.0, 1.0], dtype=np.float32),
            "constraint_require_count": np.asarray([0.0, 0.0], dtype=np.float32),
            "constraint_disallow_count": np.asarray([0.0, 1.0], dtype=np.float32),
            "constraint_disallow_fraction": np.asarray([0.0, 1.0], dtype=np.float32),
        },
    )

    assert result.decisions[0].action == "link"
    assert result.decisions[0].component_key == "orcid_low_score"
    assert result.decisions[0].score == pytest.approx(0.10)
    # Runner-up margin is reported against the full query group (the non-ORCID
    # candidate at 0.95), not just within the ORCID-forced subset.
    assert result.decisions[0].runner_up_score == pytest.approx(0.95)
    assert result.decisions[0].score_margin == pytest.approx(-0.85)


def test_private_retrieved_candidate_slice_scores_matrix_and_records_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _static_artifact(
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
        pairwise_stats=_static_pairwise_stats(row_count=3),
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
    artifact = _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
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
        pairwise_stats=_static_pairwise_stats(row_count=0),
        no_candidate_query_signature_indices=np.asarray([42], dtype=np.uint32),
    )

    assert len(result.compact_result.probabilities) == 0
    assert result.compact_result.decisions[0].query_signature_index == 42
    assert result.compact_result.decisions[0].action == "abstain"
    assert result.telemetry["no_candidate_query_count"] == 1


def test_private_retrieved_candidate_slice_rejects_partial_supervision() -> None:
    artifact = _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
    retrieval_batch = _retrieval_batch(
        row_query_signature_indices=np.asarray([], dtype=np.uint32),
        row_component_keys=(),
    )

    with pytest.raises(NotImplementedError, match="partial supervision"):
        _predict_incremental_link_or_abstain_retrieved_candidates(
            artifact,
            retrieval_batch,
            pairwise_stats=_static_pairwise_stats(row_count=0),
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


def test_private_production_slice_preserves_split_ids_for_incremental_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1", "q2"])
    clusterer = FakeProductionClusterer({"s1": "7_0"}, recluster_map={"7_0": "7"})
    artifact = _static_artifact(np.asarray([0.9], dtype=np.float64), gate_config=_promoted_gate_config(0.0))

    def fake_retrieval(**kwargs: Any) -> LinkerRetrievalBatch:
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
        artifact,
        dataset=dataset,
        featurizer=featurizer,
        retriever=object(),
        queries=[object(), object()],
        query_signature_ids=["q1", "q2"],
    )

    assert result.linked_signature_clusters == {"q1": "7_0"}
    assert [decision.action for decision in result.compact_result.decisions] == ["link", "abstain"]
    assert result.telemetry["no_candidate_query_count"] == 1
    assert result.telemetry["link_count"] == 1
    assert result.telemetry["abstain_count"] == 1


def test_private_production_slice_supplies_query_author_to_logistic_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    scale = 10.0
    artifact = _static_artifact(
        np.asarray([0.9], dtype=np.float64),
        gate_config=logistic_gate_config(
            feature_names=("top_meta_query_author_len",),
            weights=np.asarray([[-scale, 0.0, scale]], dtype=np.float64),
            bias=np.asarray([scale * 5.0, -10.0, -scale * 5.0], dtype=np.float64),
            missing_values=np.asarray([0.0], dtype=np.float64),
            calibration_mode="test",
        ),
    )

    monkeypatch.setattr(
        runtime_module,
        "build_linker_retrieval_batch_rust",
        lambda **_kwargs: _production_retrieval_batch(
            row_query_signature_indices=np.asarray([0], dtype=np.uint32),
            row_component_keys=("c1",),
            left_signature_indices=np.asarray([0], dtype=np.uint32),
            right_signature_indices=np.asarray([1], dtype=np.uint32),
            pair_row_indices=np.asarray([0], dtype=np.uint32),
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
        lambda _candidate_batch, _row_signals: _row_features_with_telemetry(np.asarray([0.9], dtype=np.float32)),
    )

    result = _predict_incremental_link_or_abstain_production_private(
        clusterer,
        artifact,
        dataset=dataset,
        featurizer=featurizer,
        retriever=object(),
        queries=[SimpleNamespace(query_author="Ada Lovelace")],
        query_signature_ids=["q1"],
    )

    assert result.compact_result.decisions[0].action == "link"


def test_private_production_slice_uses_explicit_retrieval_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
    captured: dict[str, int] = {}
    sentinel = object()

    def fake_retrieval(**kwargs: Any) -> LinkerRetrievalBatch:
        captured["top_k"] = int(kwargs["top_k"])
        return runtime_module._empty_retrieval_batch()  # noqa: SLF001

    def fake_from_retrieval(*args: Any, **kwargs: Any) -> object:
        captured["forwarded_retrieval_top_k"] = int(kwargs["retrieval_top_k"])
        return sentinel

    monkeypatch.setattr(runtime_module, "build_linker_retrieval_batch_rust", fake_retrieval)
    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_production_from_retrieval_private",
        fake_from_retrieval,
    )

    result = _predict_incremental_link_or_abstain_production_private(
        clusterer,
        artifact,
        dataset=dataset,
        featurizer=featurizer,
        retriever=object(),
        queries=[object()],
        query_signature_ids=["q1"],
        retrieval_top_k=7,
    )

    assert result is sentinel
    assert captured == {"top_k": 7, "forwarded_retrieval_top_k": 7}


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


def test_private_production_slice_preserves_partial_supervision_constraint_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1", "s2"])
    clusterer = FakeProductionClusterer({"s1": "c1", "s2": "c2"})
    artifact = _static_artifact(np.asarray([0.9, 0.1], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
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
        **kwargs: Any,
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
        artifact,
        dataset=dataset,
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


def test_private_production_slice_rejects_conflicting_partial_require_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1", "s2"])
    clusterer = FakeProductionClusterer({"s1": "c1", "s2": "c2"})
    artifact = _static_artifact(np.asarray([0.9, 0.8], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
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

    with pytest.raises(ValueError, match="partial_supervision_require_conflicting_seed_components"):
        _predict_incremental_link_or_abstain_production_private(
            clusterer,
            artifact,
            dataset=dataset,
            featurizer=featurizer,
            retriever=object(),
            queries=[object()],
            query_signature_ids=["q1"],
            partial_supervision={
                ("q1", "s1"): 0,
                ("q1", "s2"): 0,
            },
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
    artifact = _static_artifact(np.asarray([0.9, 0.1], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
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
        **kwargs: Any,
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
        artifact,
        dataset=dataset,
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
    assert result.linked_signature_clusters == {}


def test_private_production_slice_rejects_require_outside_retrieval_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
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
            artifact,
            dataset=dataset,
            featurizer=featurizer,
            retriever=object(),
            queries=[object()],
            query_signature_ids=["q1"],
            partial_supervision={("q1", "s1"): 0},
        )


def test_from_retrieval_validates_partial_supervision_against_full_seed_map() -> None:
    featurizer = FakeRuntimeFeaturizer(["q1", "s1", "s2"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
    retrieval_batch = _production_retrieval_batch(
        row_query_signature_indices=np.asarray([], dtype=np.uint32),
        row_component_keys=(),
    )

    with pytest.raises(ValueError, match="partial_supervision_require_outside_retrieval_window"):
        runtime_module._predict_incremental_link_or_abstain_production_from_retrieval_private(  # noqa: SLF001
            clusterer,
            artifact,
            dataset=None,
            featurizer=featurizer,
            retrieval_batch=retrieval_batch,
            queries=[object()],
            query_signature_ids=["q1"],
            partial_supervision={("q1", "s2"): 0},
            seed_setup=({"s1": "c1"}, {}, {"c1": ["s1"]}),
            partial_supervision_seed_signature_to_component={"s1": "c1", "s2": "c2"},
        )


def test_from_retrieval_records_artifact_retrieval_top_k_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    featurizer = FakeRuntimeFeaturizer(["q1", "s1"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = _static_artifact(np.asarray([0.9], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
    artifact.metadata.retrieval_top_k = 37
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
        dataset=None,
        featurizer=featurizer,
        retrieval_batch=retrieval_batch,
        queries=[object()],
        query_signature_ids=["q1"],
        seed_setup=({"s1": "c1"}, {}, {"c1": ["s1"]}),
    )

    assert result.telemetry["retrieval_top_k"] == 37


def test_private_production_slice_records_disallow_outside_retrieval_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = SimpleNamespace()
    featurizer = FakeRuntimeFeaturizer(["q1", "s1"])
    clusterer = FakeProductionClusterer({"s1": "c1"})
    artifact = _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
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
        artifact,
        dataset=dataset,
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
    artifact = _static_artifact(np.asarray([], dtype=np.float64), gate_config=_promoted_gate_config(0.0))
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
            artifact,
            dataset=dataset,
            featurizer=featurizer,
            retriever=object(),
            queries=[object(), object()],
            query_signature_ids=["q1", "q2"],
            partial_supervision={("q1", "q2"): 0},
        )

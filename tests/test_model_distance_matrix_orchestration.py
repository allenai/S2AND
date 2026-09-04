from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.model as model_module
from s2and.consts import LARGE_DISTANCE
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.prediction_state import PredictionState
from s2and.runtime import RuntimeContext
from tests.helpers import build_dummy_dataset


def _clusterer(
    *,
    cluster_model: object | None,
    use_default_constraints_as_supervision: bool = False,
) -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        cluster_model=cluster_model,
        n_jobs=1,
        use_default_constraints_as_supervision=use_default_constraints_as_supervision,
        batch_size=2,
    )


def _rust_dataset(name: str) -> ANDData:
    dataset = build_dummy_dataset(name, name_counts_index=None)
    cast(Any, dataset).arrow_dataset = object()
    dataset.runtime_context = RuntimeContext(
        operation="test_distance_matrix_orchestration",
        backend="rust",
        run_id=name,
    )
    return dataset


def test_block_feature_union_is_built_once_and_projected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def build_features(
        block_signature_indices: list[int],
        *,
        start_offset: int,
        max_pairs: int,
        selected_indices: list[int],
        num_threads: int,
        nan_value: float,
        featurizer: object,
    ) -> np.ndarray:
        calls.append(
            {
                "block_signature_indices": block_signature_indices,
                "start_offset": start_offset,
                "max_pairs": max_pairs,
                "selected_indices": selected_indices,
                "num_threads": num_threads,
                "nan_value": nan_value,
                "featurizer": featurizer,
            }
        )
        return np.asarray(
            [[float(index) for index in selected_indices], [100.0 + float(index) for index in selected_indices]],
            dtype=np.float64,
        )

    monkeypatch.setattr(
        model_module,
        "build_block_upper_triangle_feature_matrix_indexed_rust",
        build_features,
    )
    featurizer = object()

    main, nameless = model_module._build_block_feature_matrices_indexed_rust(
        [7, 8, 9],
        featurizer=featurizer,
        start_offset=3,
        max_pairs=2,
        main_indices=[4, 1, 4],
        nameless_indices=[9, 1, 0],
        num_threads=5,
    )

    assert len(calls) == 1
    call = calls[0]
    nan_value = call.pop("nan_value")
    assert np.isnan(nan_value)
    assert call == {
        "block_signature_indices": [7, 8, 9],
        "start_offset": 3,
        "max_pairs": 2,
        "selected_indices": [4, 1, 9, 0],
        "num_threads": 5,
        "featurizer": featurizer,
    }
    np.testing.assert_array_equal(main, [[4.0, 1.0, 4.0], [104.0, 101.0, 104.0]])
    assert nameless is not None
    np.testing.assert_array_equal(nameless, [[9.0, 1.0, 0.0], [109.0, 101.0, 100.0]])
    assert main.flags.c_contiguous
    assert nameless.flags.c_contiguous


def test_rust_distance_matrix_propagates_ram_bounded_chunk_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def chunk_plan(_num_features: int, **kwargs: Any) -> SimpleNamespace:
        captured["plan_total_pairs"] = kwargs["total_pairs"]
        captured["plan_total_ram_bytes"] = kwargs["total_ram_bytes"]
        return SimpleNamespace(chunk_pairs=1)

    def allocation_guard(**kwargs: Any) -> None:
        captured["guard_total_ram_bytes"] = kwargs["total_ram_bytes"]

    def chunk_helper(*_args: Any, **kwargs: Any):
        captured["helper_pair_chunk_size"] = kwargs["pair_chunk_size"]
        yield model_module._DistanceMatrixChunk(
            block_key="block",
            block_size=2,
            start_offset=0,
            index_i=np.asarray([0], dtype=np.intp),
            index_j=np.asarray([1], dtype=np.intp),
            pair_ids=[("0", "1")],
            labels=np.asarray([np.nan], dtype=np.float64),
        )

    def predict_chunk(
        _self: Clusterer,
        chunk: Any,
        _dataset: ANDData,
        _runtime_context: RuntimeContext,
        batch_label: int | str,
        total_ram_bytes: int | None = None,
    ) -> tuple[np.ndarray, float]:
        del batch_label
        captured["scorer_total_ram_bytes"] = total_ram_bytes
        return np.full(len(chunk.labels), 0.25, dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "_compute_predict_batch_chunk_plan", chunk_plan)
    monkeypatch.setattr(model_module, "_guard_predict_block_matrix_allocation", allocation_guard)
    monkeypatch.setattr(Clusterer, "_distance_matrix_chunk_helper_rust", chunk_helper)
    monkeypatch.setattr(Clusterer, "_predict_distance_matrix_chunk", predict_chunk)

    output = _clusterer(cluster_model=object()).make_distance_matrices(
        {"block": ["0", "1"]},
        _rust_dataset("ram-bounded-chunks"),
        disable_tqdm=True,
        total_ram_bytes=123,
    )

    assert captured == {
        "guard_total_ram_bytes": 123,
        "plan_total_pairs": 1,
        "plan_total_ram_bytes": 123,
        "helper_pair_chunk_size": 1,
        "scorer_total_ram_bytes": 123,
    }
    np.testing.assert_allclose(
        output["block"],
        np.asarray([[0.0, 0.25], [0.25, 0.0]], dtype=np.float16),
    )


def test_fastcluster_rejects_native_constraint_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    featurizer = SimpleNamespace(
        signature_ids=lambda: ["0", "1", "2"],
        featurize_block_upper_triangle_matrix_indexed=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("features must not be built after a constraint count mismatch")
        ),
    )
    monkeypatch.setattr(
        model_module,
        "get_constraints_block_upper_triangle_indexed_rust",
        lambda _indices, *, max_pairs, **_kwargs: ([], [], [None] * (max_pairs - 1)),
    )

    with pytest.raises(RuntimeError, match="Rust constraint row count mismatch"):
        _clusterer(
            cluster_model=model_module.FastCluster(linkage="average"),
            use_default_constraints_as_supervision=True,
        ).make_distance_matrices_from_rust_featurizer(
            {"block": ["0", "1", "2"]},
            featurizer,
        )


def test_predict_from_rust_featurizer_streams_only_nontrivial_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_state = PredictionState()
    make_calls: list[tuple[str, ...]] = []
    cluster_calls: list[tuple[str, tuple[str, ...]]] = []
    shared_index = {str(index): index for index in range(5)}
    index_calls = 0

    def build_index(_featurizer: object) -> dict[str, int]:
        nonlocal index_calls
        index_calls += 1
        return shared_index

    def make_dists(
        self: Clusterer,
        block_dict: dict[str, list[str]],
        _featurizer: object,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        make_calls.append(tuple(block_dict))
        assert kwargs["signature_index_by_id"] is shared_index
        block_key, signatures = next(iter(block_dict.items()))
        kwargs["prediction_state"].telemetry["rust_featurizer_make_dists"] = {
            "block_count": 1,
            "pair_count": len(signatures) * (len(signatures) - 1) // 2,
        }
        return {block_key: np.zeros((len(signatures), len(signatures)), dtype=np.float16)}

    def cluster_block(
        _self: Clusterer,
        signatures: list[str],
        _pairwise_proba: np.ndarray,
        _cluster_model_params: dict[str, Any],
        _dataset: object,
        _disallowed_ids: set[str],
        *,
        block_key: str,
        incremental_dont_use_cluster_seeds: bool,
    ) -> list[int]:
        cluster_calls.append((block_key, tuple(signatures)))
        return [0] * len(signatures)

    monkeypatch.setattr(model_module, "_build_signature_index_by_id", build_index)
    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", make_dists)
    monkeypatch.setattr(Clusterer, "_cluster_one_block_with_logging", cluster_block)

    clusterer = _clusterer(cluster_model=None)
    result, dists = clusterer._predict_from_rust_featurizer(
        {
            "empty": [],
            "singleton": ["4"],
            "a": ["0", "1"],
            "b": ["2", "3"],
        },
        object(),
        cluster_seeds_require={},
        prediction_state=prediction_state,
    )

    assert dists is None
    assert index_calls == 1
    assert make_calls == [("a",), ("b",)]
    assert cluster_calls == [("a", ("0", "1")), ("b", ("2", "3"))]
    assert result == {
        "singleton_0": ["4"],
        "a_0": ["0", "1"],
        "b_0": ["2", "3"],
    }
    telemetry = prediction_state.telemetry["rust_featurizer_predict"]
    assert telemetry["make_dists_block_count"] == 4
    assert telemetry["make_dists_pair_count"] == 2


def test_predict_from_rust_featurizer_skips_setup_for_only_trivial_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_state = PredictionState()

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("trivial blocks must not run pairwise prediction setup")

    monkeypatch.setattr(model_module, "_build_signature_index_by_id", unexpected)
    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", unexpected)
    monkeypatch.setattr(Clusterer, "_cluster_one_block_with_logging", unexpected)

    clusterer = _clusterer(cluster_model=None)
    result, dists = clusterer._predict_from_rust_featurizer(
        {"empty": [], "first": ["0"], "second": ["1"]},
        object(),
        cluster_seeds_require={},
        prediction_state=prediction_state,
    )

    assert dists is None
    assert result == {"first_0": ["0"], "second_0": ["1"]}
    telemetry = prediction_state.telemetry["rust_featurizer_predict"]
    assert telemetry["make_dists_block_count"] == 3
    assert telemetry["make_dists_pair_count"] == 0


def test_predict_from_rust_featurizer_rejects_seeds_with_precomputed_dists() -> None:
    for seed_mode in ("explicit_disallow", "explicit_require", "native_require"):
        native_seeds = [("0", "c0"), ("1", "c1")] if seed_mode == "native_require" else []
        featurizer = SimpleNamespace(cluster_seeds_require=lambda _seeds=native_seeds: _seeds)
        kwargs: dict[str, Any] = {}
        if seed_mode == "explicit_disallow":
            kwargs = {"cluster_seeds_require": {}, "cluster_seeds_disallow": {("0", "1")}}
        elif seed_mode == "explicit_require":
            kwargs = {"cluster_seeds_require": {"0": "c0", "1": "c0"}}

        with pytest.raises(ValueError, match="precomputed dists"):
            _clusterer(cluster_model=None).predict_from_rust_featurizer(
                {"block": ["0", "1"]},
                featurizer,
                dists={"block": np.zeros((2, 2), dtype=np.float64)},
                **kwargs,
            )


def test_seed_overrides_preserve_existing_pairs_and_add_missing_constraints() -> None:
    merged = model_module._partial_supervision_with_cluster_seed_overrides(
        ["0", "1", "2", "3"],
        {("2", "0"): 42.0},
        cluster_seeds_require={"0": "c0", "1": "c0", "2": "c1", "3": "c1"},
        cluster_seeds_disallow={("1", "2")},
    )

    assert merged == {
        ("2", "0"): 42.0,
        ("0", "3"): LARGE_DISTANCE,
        ("1", "2"): LARGE_DISTANCE,
        ("1", "3"): LARGE_DISTANCE,
        ("0", "1"): 0,
        ("2", "3"): 0,
    }


def test_native_seed_map_is_forwarded_without_python_pair_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def prepared_predict(
        _self: Clusterer,
        _block_dict: dict[str, list[str]],
        _featurizer: object,
        **kwargs: Any,
    ) -> tuple[dict[str, list[str]], None]:
        captured.update(kwargs)
        return {}, None

    monkeypatch.setattr(Clusterer, "_predict_from_prepared_rust_featurizer", prepared_predict)
    featurizer = SimpleNamespace(
        cluster_seeds_require=lambda: [
            ("0", "c0"),
            ("1", "c0"),
            ("2", "c1"),
            ("3", "c1"),
        ]
    )

    _clusterer(cluster_model=None).predict_from_rust_featurizer(
        {"block": ["0", "1", "2", "3"]},
        featurizer,
    )

    assert captured["partial_supervision"] == {}
    assert captured["incremental_dont_use_cluster_seeds"] is False
    assert captured["explicit_cluster_seeds_require"] is None
    assert captured["proxy_dataset"].cluster_seeds_require == {
        "0": "c0",
        "1": "c0",
        "2": "c1",
        "3": "c1",
    }


@pytest.mark.parametrize("fail_nested", [False, True])
def test_rust_prediction_telemetry_isolated_during_nested_request(
    monkeypatch: pytest.MonkeyPatch, fail_nested: bool
) -> None:
    """An interleaved request, including failure, cannot overwrite outer diagnostics."""
    clusterer = _clusterer(cluster_model=None)
    original_attributes = dict(vars(clusterer))
    outer_state = PredictionState()
    nested_state = PredictionState()
    featurizer = SimpleNamespace(signature_ids=lambda: ["0", "1", "2"])

    def make_dists(self: Clusterer, blocks: dict[str, list[str]], native: object, **kwargs: Any):
        state = kwargs["prediction_state"]
        block_key, signatures = next(iter(blocks.items()))
        state.telemetry["rust_featurizer_make_dists"] = {"request_marker": block_key}
        if block_key == "outer":
            try:
                self._predict_from_rust_featurizer(
                    {"nested": ["0", "1", "2"]},
                    native,
                    cluster_seeds_require={},
                    prediction_state=nested_state,
                )
            except RuntimeError as error:
                assert fail_nested
                assert str(error) == "injected nested failure"
        elif fail_nested:
            raise RuntimeError("injected nested failure")
        return {block_key: np.zeros((len(signatures), len(signatures)))}

    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", make_dists)
    monkeypatch.setattr(
        Clusterer, "_cluster_one_block_with_logging", lambda _self, signatures, *_args, **_kwargs: [0] * len(signatures)
    )
    clusters, _ = clusterer._predict_from_rust_featurizer(
        {"outer": ["0", "1"]},
        featurizer,
        cluster_seeds_require={},
        prediction_state=outer_state,
    )
    assert clusters == {"outer_0": ["0", "1"]}
    assert outer_state.telemetry["rust_featurizer_predict"]["make_dists_request_marker"] == "outer"
    assert nested_state.telemetry["rust_featurizer_make_dists"]["request_marker"] == "nested"
    assert vars(clusterer) == original_attributes

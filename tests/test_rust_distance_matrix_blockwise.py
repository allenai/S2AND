from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.model as model_module
from s2and import feature_port
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import write_name_counts_index
from s2and.model import Clusterer
from s2and.runtime import RuntimeContext
from tests.helpers import (
    tiny_name_counts_provenance,
    tiny_name_counts_tuple,
    write_minimal_arrow_prediction_bundle,
    write_test_arrow_artifact_manifest,
)

s2and_rust = cast(Any, feature_port.s2and_rust)


def _dummy_dataset(name: str) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        name_counts_index=None,
        n_jobs=1,
    )


def _mark_arrow_backed(dataset: ANDData) -> ANDData:
    cast(Any, dataset).arrow_paths = {"signatures": "mock-signatures.arrow"}
    dataset.runtime_context = RuntimeContext(
        operation="test_arrow_training_dataset",
        backend="rust",
        run_id="test-arrow-training-dataset",
    )
    return dataset


def _specter_dataset(name: str, specter_embeddings: Any) -> ANDData:
    signatures = {
        "s1": {
            "signature_id": "s1",
            "paper_id": "p1",
            "author_info": {
                "first": "Ada",
                "middle": "",
                "last": "Lovelace",
                "suffix": "",
                "affiliations": [],
                "email": "",
                "position": 0,
                "block": "a lovelace",
                "source_ids": [],
            },
        },
        "s2": {
            "signature_id": "s2",
            "paper_id": "p2",
            "author_info": {
                "first": "Ada",
                "middle": "",
                "last": "Lovelace",
                "suffix": "",
                "affiliations": [],
                "email": "",
                "position": 0,
                "block": "a lovelace",
                "source_ids": [],
            },
        },
    }
    papers = {
        "p1": {
            "paper_id": "p1",
            "title": "Graph Models",
            "abstract": "",
            "venue": "",
            "journal_name": "",
            "year": 2020,
            "authors": [{"position": 0, "author_name": "Ada Lovelace"}],
            "references": [],
        },
        "p2": {
            "paper_id": "p2",
            "title": "Graph Models",
            "abstract": "",
            "venue": "",
            "journal_name": "",
            "year": 2020,
            "authors": [{"position": 0, "author_name": "Ada Lovelace"}],
            "references": [],
        },
    }
    return ANDData(
        signatures=signatures,
        papers=papers,
        name=name,
        mode="inference",
        specter_embeddings=specter_embeddings,
        name_counts_index=None,
        preprocess=True,
        name_tuples=set(),
        n_jobs=1,
    )


def _empty_first_constraint_dataset() -> ANDData:
    def signature(signature_id: str, first: str, paper_id: str) -> dict[str, Any]:
        return {
            "signature_id": signature_id,
            "paper_id": paper_id,
            "author_info": {
                "first": first,
                "middle": "",
                "last": "Smith",
                "suffix": "",
                "affiliations": [],
                "email": "",
                "position": 0,
                "block": "smith",
            },
            "sourced_author_ids": [],
            "sourced_author_source": "Extracted",
        }

    def paper(paper_id: str) -> dict[str, Any]:
        return {
            "paper_id": paper_id,
            "title": "Untitled",
            "abstract": "",
            "venue": "",
            "journal_name": "",
            "year": 2020,
            "authors": [{"position": 0, "author_name": "Smith"}],
            "references": [],
        }

    return ANDData(
        signatures={
            "empty": signature("empty", "", "p1"),
            "named": signature("named", "Alice", "p2"),
        },
        papers={"p1": paper("p1"), "p2": paper("p2")},
        name="empty_first_constraint_dataset",
        mode="inference",
        name_counts_index=None,
        preprocess=True,
        name_tuples=set(),
        n_jobs=1,
    )


def _indexed_pair_matrix(featurizer: Any, pairs: list[tuple[str, str]]) -> np.ndarray:
    signature_id_to_index = {str(signature_id): index for index, signature_id in enumerate(featurizer.signature_ids())}
    indexed_pairs = [(signature_id_to_index[left], signature_id_to_index[right]) for left, right in pairs]
    return np.asarray(featurizer.featurize_pairs_matrix_indexed(indexed_pairs, None, 1, np.nan))


def _dummy_clusterer(
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


def _with_fake_batch_indexes(arrow_paths: dict[str, str], tmp_path: Path) -> dict[str, str]:
    indexed = write_minimal_arrow_prediction_bundle(tmp_path, include_specter="specter" in arrow_paths)
    for key, value in arrow_paths.items():
        if key not in {"signatures", "papers", "paper_authors", "specter"}:
            indexed[key] = value
    return indexed


def _partial_supervision_for_upper_triangle(signatures: list[str]) -> tuple[dict[tuple[str, str], float], np.ndarray]:
    values: list[float] = []
    partial_supervision: dict[tuple[str, str], float] = {}
    next_value = 11
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            value = float(next_value) / 100.0
            partial_supervision[(signatures[i], signatures[j])] = value
            values.append(value)
            next_value += 11
    return partial_supervision, np.asarray(values, dtype=np.float64)


def test_block_feature_union_builds_once_and_preserves_requested_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_build(
        block_signature_indices,
        *,
        start_offset,
        max_pairs,
        selected_indices,
        num_threads,
        nan_value,
        featurizer,
    ):
        captured.update(
            block_signature_indices=block_signature_indices,
            start_offset=start_offset,
            max_pairs=max_pairs,
            selected_indices=selected_indices,
            num_threads=num_threads,
            nan_value=nan_value,
            featurizer=featurizer,
        )
        return np.asarray(
            [[float(index) for index in selected_indices], [100.0 + float(index) for index in selected_indices]],
            dtype=np.float64,
        )

    monkeypatch.setattr(model_module, "build_block_upper_triangle_feature_matrix_indexed_rust", fake_build)
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

    assert captured["block_signature_indices"] == [7, 8, 9]
    assert captured["start_offset"] == 3
    assert captured["max_pairs"] == 2
    assert captured["selected_indices"] == [4, 1, 9, 0]
    assert captured["num_threads"] == 5
    assert np.isnan(captured["nan_value"])
    assert captured["featurizer"] is featurizer
    np.testing.assert_array_equal(main, [[4.0, 1.0, 4.0], [104.0, 101.0, 104.0]])
    assert nameless is not None
    np.testing.assert_array_equal(nameless, [[9.0, 1.0, 0.0], [109.0, 101.0, 100.0]])
    assert main.flags.c_contiguous
    assert nameless.flags.c_contiguous


@pytest.mark.parametrize("square_matrix", [False, True], ids=["fastcluster", "square"])
def test_make_distance_matrices_rust_blockwise_output_formats(monkeypatch, square_matrix: bool):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(
        model_module.Clusterer,
        "distance_matrix_helper",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy pair helper should not be called")),
    )

    featurize_call_sizes: list[int] = []

    def fake_many_pairs_featurize(signature_pairs, *_args, **_kwargs):
        featurize_call_sizes.append(len(signature_pairs))
        labels = np.asarray([float(pair[2]) for pair in signature_pairs], dtype=np.float64)
        features = np.zeros((len(signature_pairs), 1), dtype=np.float64)
        return features, labels, None

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        _features,
        labels,
        _nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del runtime_context, _kwargs
        return np.asarray(labels + LARGE_INTEGER, dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "many_pairs_featurize", fake_many_pairs_featurize)
    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    dataset = _mark_arrow_backed(_dummy_dataset("dummy_rust_blockwise_fastcluster"))
    clusterer = _dummy_clusterer(cluster_model=object() if square_matrix else None)
    signatures = ["0", "1", "2", "3"]
    partial_supervision, expected_flat = _partial_supervision_for_upper_triangle(signatures)

    output = clusterer.make_distance_matrices(
        {"block": signatures},
        dataset,
        partial_supervision=partial_supervision,
    )
    matrix = output["block"]

    if square_matrix:
        expected_square = np.zeros((4, 4), dtype=np.float16)
        expected_square[np.triu_indices(4, k=1)] = expected_flat.astype(np.float16)
        expected_square = expected_square + expected_square.T
        np.fill_diagonal(expected_square, 0)
        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(matrix, expected_square, rtol=0, atol=0)
    else:
        assert matrix.dtype == np.float64
        np.testing.assert_allclose(matrix, expected_flat, rtol=1e-10, atol=1e-12)
    assert featurize_call_sizes == [2, 2, 2]


def test_make_distance_matrices_rust_honors_ram_bounded_pair_chunk_size(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_chunk_plan(_num_features: int, **kwargs: Any) -> SimpleNamespace:
        captured["plan_total_pairs"] = kwargs["total_pairs"]
        captured["plan_total_ram_bytes"] = kwargs["total_ram_bytes"]
        return SimpleNamespace(chunk_pairs=1)

    def fake_guard(**kwargs: Any) -> None:
        captured["guard_total_ram_bytes"] = kwargs["total_ram_bytes"]

    def fake_chunk_helper(*_args: Any, **kwargs: Any):
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

    def fake_predict_chunk(
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

    monkeypatch.setattr(model_module, "_compute_predict_batch_chunk_plan", fake_chunk_plan)
    monkeypatch.setattr(model_module, "_guard_predict_block_matrix_allocation", fake_guard)
    monkeypatch.setattr(Clusterer, "_distance_matrix_chunk_helper_rust", fake_chunk_helper)
    monkeypatch.setattr(Clusterer, "_predict_distance_matrix_chunk", fake_predict_chunk)

    dataset = _mark_arrow_backed(_dummy_dataset("dummy_rust_ram_bounded_chunks"))
    clusterer = _dummy_clusterer(cluster_model=object())
    output = clusterer.make_distance_matrices(
        {"block": ["0", "1"]},
        dataset,
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
    np.testing.assert_allclose(output["block"], np.asarray([[0.0, 0.25], [0.25, 0.0]], dtype=np.float16))


def test_make_distance_matrices_rust_fused_upper_triangle_api(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(
        model_module.Clusterer,
        "distance_matrix_helper",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy pair helper should not be called")),
    )
    monkeypatch.setattr(
        model_module,
        "many_pairs_featurize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("many_pairs_featurize should not be called")),
    )
    monkeypatch.setattr(
        model_module,
        "get_constraints_matrix_indexed_rust",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy indexed constraint API should not be called")
        ),
    )

    captured = {"constraint_calls": 0, "feature_calls": 0}

    class _FakeFusedFeaturizer:
        def signature_ids(self):
            return ["0", "1", "2", "3"]

        def get_constraints_block_upper_triangle_indexed(
            self,
            block_signature_indices,
            start_offset=0,
            max_pairs=None,
            *_args,
            **_kwargs,
        ):
            captured["constraint_calls"] += 1
            block_size = len(block_signature_indices)
            all_pairs = [(i, j) for i in range(block_size) for j in range(i + 1, block_size)]
            pair_slice = all_pairs[start_offset : start_offset + int(max_pairs or len(all_pairs))]
            left = [int(i) for i, _ in pair_slice]
            right = [int(j) for _, j in pair_slice]
            return left, right, [None] * len(pair_slice)

        def featurize_block_upper_triangle_matrix_indexed(
            self,
            block_signature_indices,
            start_offset=0,
            max_pairs=None,
            selected_indices=None,
            *_args,
            **_kwargs,
        ):
            captured["feature_calls"] += 1
            block_size = len(block_signature_indices)
            all_pairs = [(i, j) for i in range(block_size) for j in range(i + 1, block_size)]
            pair_slice = all_pairs[start_offset : start_offset + int(max_pairs or len(all_pairs))]
            out_cols = len(selected_indices) if selected_indices is not None else 33
            out = np.zeros((len(pair_slice), out_cols), dtype=np.float64)
            for row_offset in range(len(pair_slice)):
                out[row_offset, :] = float(start_offset + row_offset + 1) / 10.0
            return out

    monkeypatch.setattr(model_module, "_get_rust_featurizer", lambda *_args, **_kwargs: _FakeFusedFeaturizer())

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        features,
        labels,
        nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del labels, runtime_context, _kwargs
        assert nameless_features is not None
        np.testing.assert_array_equal(nameless_features[:, 0], features[:, 0])
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    dataset = _mark_arrow_backed(_dummy_dataset("dummy_rust_blockwise_fused"))
    clusterer = _dummy_clusterer(
        cluster_model=None,
        use_default_constraints_as_supervision=True,
    )
    clusterer.nameless_classifier = object()
    clusterer.nameless_featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    signatures = ["0", "1", "2", "3"]
    output = clusterer.make_distance_matrices(
        {"block": signatures},
        dataset,
        partial_supervision={},
    )
    matrix = output["block"]

    np.testing.assert_allclose(matrix, np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64), rtol=0, atol=0)
    assert captured["constraint_calls"] == 3
    assert captured["feature_calls"] == 3


def test_make_distance_matrices_releases_chunk_payloads_before_next_build(monkeypatch):
    live_feature_chunks = []
    live_native_chunks = []

    class _TrackedFeatureChunk:
        def __init__(self, chunk_id):
            self.chunk_id = chunk_id
            live_feature_chunks.append(chunk_id)

        def __del__(self):
            live_feature_chunks.remove(self.chunk_id)

    class _TrackedNativeChunk(list):
        def __init__(self, chunk_id, values):
            super().__init__(values)
            self.chunk_id = chunk_id
            live_native_chunks.append(chunk_id)

        def __del__(self):
            live_native_chunks.remove(self.chunk_id)

    def fake_constraints(_block_signature_indices, *, start_offset, max_pairs, **_kwargs):
        assert live_feature_chunks == []
        assert live_native_chunks == []
        count = int(max_pairs)
        return (
            _TrackedNativeChunk((start_offset, "left"), range(count)),
            _TrackedNativeChunk((start_offset, "right"), range(count)),
            _TrackedNativeChunk((start_offset, "constraints"), [None] * count),
        )

    def fake_features(_block_signature_indices, *, start_offset, **_kwargs):
        assert live_feature_chunks == []
        return _TrackedFeatureChunk(start_offset)

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        _features,
        labels,
        _nameless_features,
        _batch_label,
        **_kwargs,
    ):
        return np.full(len(labels), 0.25, dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "get_constraints_block_upper_triangle_indexed_rust", fake_constraints)
    monkeypatch.setattr(model_module, "build_block_upper_triangle_feature_matrix_indexed_rust", fake_features)
    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    clusterer = _dummy_clusterer(
        cluster_model=model_module.FastCluster(linkage="average"),
        use_default_constraints_as_supervision=True,
    )
    output = clusterer.make_distance_matrices_from_rust_featurizer(
        {"block": ["0", "1", "2", "3"]},
        type("FakeFeaturizer", (), {"signature_ids": lambda self: ["0", "1", "2", "3"]})(),
    )

    np.testing.assert_allclose(output["block"], np.full(6, 0.25, dtype=np.float64), rtol=0, atol=0)
    assert live_feature_chunks == []
    assert live_native_chunks == []


def test_make_distance_matrices_from_rust_featurizer_skips_fastcluster_indices_without_constraints(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "_upper_triangle_indices_for_range",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("FastCluster vector writes do not need upper-triangle index arrays")
        ),
    )
    monkeypatch.setattr(
        model_module,
        "get_constraints_block_upper_triangle_indexed_rust",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("constraints are disabled")),
    )

    captured = {"feature_calls": 0}

    class _FakeFeaturizer:
        def signature_ids(self):
            return ["0", "1", "2", "3"]

        def featurize_block_upper_triangle_matrix_indexed(
            self,
            _block_signature_indices,
            start_offset=0,
            max_pairs=None,
            selected_indices=None,
            *_args,
            **_kwargs,
        ):
            captured["feature_calls"] += 1
            assert max_pairs is not None
            out_cols = len(selected_indices) if selected_indices is not None else 1
            out = np.zeros((int(max_pairs), out_cols), dtype=np.float64)
            out[:, 0] = np.arange(start_offset, start_offset + int(max_pairs), dtype=np.float64)
            return out

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        features,
        labels,
        nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del _classifier, _nameless_classifier, _batch_label, runtime_context, _kwargs
        assert np.isnan(labels).all()
        assert nameless_features is not None
        np.testing.assert_array_equal(nameless_features[:, 0], features[:, 0])
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    clusterer = _dummy_clusterer(
        cluster_model=model_module.FastCluster(linkage="average"),
        use_default_constraints_as_supervision=False,
    )
    clusterer.nameless_classifier = object()
    clusterer.nameless_featurizer_info = FeaturizationInfo(features_to_use=["year_diff"])
    output = clusterer.make_distance_matrices_from_rust_featurizer(
        {"block": ["0", "1", "2", "3"]},
        _FakeFeaturizer(),
    )

    np.testing.assert_allclose(output["block"], np.arange(6, dtype=np.float64), rtol=0, atol=0)
    assert captured["feature_calls"] == 3
    telemetry = clusterer._last_rust_featurizer_make_dists_telemetry
    assert telemetry["chunk_count"] == 3
    assert telemetry["upper_triangle_index_seconds"] == 0.0


def test_make_distance_matrices_from_rust_featurizer_checks_fastcluster_constraint_count(monkeypatch):
    class _FakeFeaturizer:
        def signature_ids(self):
            return ["0", "1", "2"]

        def featurize_block_upper_triangle_matrix_indexed(self, *_args, **_kwargs):
            raise AssertionError("feature rows should not be built after a constraint count mismatch")

    monkeypatch.setattr(
        model_module,
        "get_constraints_block_upper_triangle_indexed_rust",
        lambda _block_signature_indices, *, max_pairs, **_kwargs: (
            [],
            [],
            [None] * (int(max_pairs) - 1),
        ),
    )

    clusterer = _dummy_clusterer(
        cluster_model=model_module.FastCluster(linkage="average"),
        use_default_constraints_as_supervision=True,
    )

    with pytest.raises(RuntimeError, match="Rust constraint row count mismatch"):
        clusterer.make_distance_matrices_from_rust_featurizer(
            {"block": ["0", "1", "2"]},
            _FakeFeaturizer(),
        )


def test_predict_from_rust_featurizer_builds_and_clusters_one_block_at_a_time(monkeypatch):
    make_calls = []
    cluster_calls = []
    live_distance_blocks = []
    signature_index_calls = 0
    shared_signature_index = {str(index): index for index in range(4)}

    class _FakeFeaturizer:
        def signature_rule_metadata(self):
            return [(str(index), f"First{index}", None) for index in range(4)]

    class _TrackedBlockDists(dict):
        def __init__(self, block_key, pairwise_proba):
            super().__init__({block_key: pairwise_proba})
            self.block_key = block_key
            live_distance_blocks.append(block_key)

        def __del__(self):
            live_distance_blocks.remove(self.block_key)

    def fake_make_dists(
        self,
        block_dict,
        _rust_featurizer,
        **kwargs,
    ):
        assert live_distance_blocks == []
        make_calls.append(tuple(block_dict))
        assert kwargs["signature_index_by_id"] is shared_signature_index
        assert len(block_dict) == 1
        block_key, signatures = next(iter(block_dict.items()))
        self._last_rust_featurizer_make_dists_telemetry = {
            "total_seconds": 0.25,
            "constraint_seconds": 0.1,
            "feature_matrix_seconds": 0.2,
            "nameless_feature_matrix_seconds": 0.0,
            "model_predict_seconds": 0.3,
            "matrix_write_seconds": 0.4,
            "block_count": 1,
            "pair_count": len(signatures) * (len(signatures) - 1) // 2,
        }
        return _TrackedBlockDists(
            block_key,
            np.zeros((len(signatures), len(signatures)), dtype=np.float16),
        )

    def fake_cluster_one_block(
        self,
        signatures,
        pairwise_proba,
        effective_cluster_model_params,
        dataset,
        all_disallow_signature_ids,
        *,
        block_key,
    ):
        del self, pairwise_proba, effective_cluster_model_params, dataset, all_disallow_signature_ids
        cluster_calls.append((block_key, tuple(signatures)))
        return [0 for _signature in signatures]

    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", fake_make_dists)
    monkeypatch.setattr(Clusterer, "_cluster_one_block_with_logging", fake_cluster_one_block)

    def fake_build_signature_index(_rust_featurizer):
        nonlocal signature_index_calls
        signature_index_calls += 1
        return shared_signature_index

    monkeypatch.setattr(model_module, "_build_signature_index_by_id", fake_build_signature_index)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict_from_rust_featurizer(
        {"a": ["0", "1"], "b": ["2", "3"]},
        _FakeFeaturizer(),
        cluster_seeds_require={},
    )

    assert dists is None
    assert make_calls == [("a",), ("b",)]
    assert live_distance_blocks == []
    assert signature_index_calls == 1
    assert cluster_calls == [("a", ("0", "1")), ("b", ("2", "3"))]
    assert result == {"a_0": ["0", "1"], "b_0": ["2", "3"]}
    telemetry = clusterer._last_rust_featurizer_predict_telemetry
    assert float(telemetry["make_dists_total_seconds"]) >= 0.0
    assert telemetry["make_dists_constraint_seconds"] == 0.2
    assert telemetry["make_dists_block_count"] == 2
    assert telemetry["make_dists_pair_count"] == 2


def test_predict_from_rust_featurizer_skips_pairwise_work_for_empty_and_singleton_blocks(monkeypatch):
    make_calls = []
    cluster_calls = []
    shared_signature_index = {"0": 0, "1": 1, "2": 2}

    def fake_make_dists(self, block_dict, _rust_featurizer, **kwargs):
        make_calls.append(tuple(block_dict))
        assert kwargs["signature_index_by_id"] is shared_signature_index
        block_key, signatures = next(iter(block_dict.items()))
        self._last_rust_featurizer_make_dists_telemetry = {
            "block_count": 1,
            "pair_count": len(signatures) * (len(signatures) - 1) // 2,
        }
        return {block_key: np.zeros((len(signatures), len(signatures)), dtype=np.float16)}

    def fake_cluster_one_block(
        self,
        signatures,
        pairwise_proba,
        effective_cluster_model_params,
        dataset,
        all_disallow_signature_ids,
        *,
        block_key,
    ):
        del self, pairwise_proba, effective_cluster_model_params, dataset, all_disallow_signature_ids
        cluster_calls.append((block_key, tuple(signatures)))
        return [0 for _signature in signatures]

    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", fake_make_dists)
    monkeypatch.setattr(Clusterer, "_cluster_one_block_with_logging", fake_cluster_one_block)
    monkeypatch.setattr(model_module, "_build_signature_index_by_id", lambda _featurizer: shared_signature_index)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict_from_rust_featurizer(
        {"empty": [], "singleton": ["2"], "multi": ["0", "1"]},
        object(),
        cluster_seeds_require={},
    )

    assert dists is None
    assert make_calls == [("multi",)]
    assert cluster_calls == [("multi", ("0", "1"))]
    assert result == {"singleton_0": ["2"], "multi_0": ["0", "1"]}
    telemetry = clusterer._last_rust_featurizer_predict_telemetry
    assert telemetry["make_dists_block_count"] == 3
    assert telemetry["make_dists_pair_count"] == 1


def test_predict_from_rust_featurizer_avoids_signature_index_for_only_singletons(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("empty and singleton blocks must not run pairwise prediction setup")

    monkeypatch.setattr(model_module, "_build_signature_index_by_id", unexpected)
    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", unexpected)
    monkeypatch.setattr(Clusterer, "_cluster_one_block_with_logging", unexpected)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict_from_rust_featurizer(
        {"empty": [], "first": ["0"], "second": ["1"]},
        object(),
        cluster_seeds_require={},
    )

    assert dists is None
    assert result == {"first_0": ["0"], "second_0": ["1"]}
    telemetry = clusterer._last_rust_featurizer_predict_telemetry
    assert telemetry["make_dists_block_count"] == 3
    assert telemetry["make_dists_pair_count"] == 0


@pytest.mark.parametrize("seed_mode", ["explicit_disallow", "explicit_require", "implicit_require"])
def test_predict_from_rust_featurizer_rejects_seed_constraints_with_precomputed_dists(seed_mode: str) -> None:
    class _FakeFeaturizer:
        def signature_rule_metadata(self):
            return [("0", "First0", None), ("1", "First1", None)]

        def cluster_seeds_require(self):
            return [("0", "c0"), ("1", "c1")] if seed_mode == "implicit_require" else []

    clusterer = _dummy_clusterer(cluster_model=None)
    kwargs = {}
    if seed_mode == "explicit_disallow":
        kwargs = {"cluster_seeds_require": {}, "cluster_seeds_disallow": {("0", "1")}}
    elif seed_mode == "explicit_require":
        kwargs = {"cluster_seeds_require": {"0": "c0", "1": "c0"}}

    with pytest.raises(ValueError, match="precomputed dists"):
        clusterer.predict_from_rust_featurizer(
            {"block": ["0", "1"]},
            _FakeFeaturizer(),
            dists={"block": np.zeros((2, 2), dtype=np.float64)},
            **kwargs,
        )


def test_predict_from_rust_featurizer_injects_seed_overrides_into_distance_build(monkeypatch):
    captured_partial_supervision: list[dict[tuple[str, str], int | float]] = []
    captured_incremental_flags: list[bool] = []

    class _FakeFeaturizer:
        def signature_ids(self):
            return [str(index) for index in range(4)]

        def signature_rule_metadata(self):
            return [(str(index), f"First{index}", None) for index in range(4)]

    def fake_make_dists(self, block_dict, _rust_featurizer, **kwargs):
        block_key, signatures = next(iter(block_dict.items()))
        captured_partial_supervision.append(dict(kwargs["partial_supervision"]))
        captured_incremental_flags.append(bool(kwargs["incremental_dont_use_cluster_seeds"]))
        return {block_key: np.zeros((len(signatures), len(signatures)), dtype=np.float64)}

    def fake_cluster_one_block(
        self,
        signatures,
        pairwise_proba,
        effective_cluster_model_params,
        dataset,
        all_disallow_signature_ids,
        *,
        block_key,
    ):
        del self, pairwise_proba, effective_cluster_model_params, dataset, all_disallow_signature_ids, block_key
        return [0 for _signature in signatures]

    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", fake_make_dists)
    monkeypatch.setattr(Clusterer, "_cluster_one_block_with_logging", fake_cluster_one_block)

    clusterer = _dummy_clusterer(cluster_model=None)
    clusterer.predict_from_rust_featurizer(
        {"block": ["0", "1", "2", "3"]},
        _FakeFeaturizer(),
        partial_supervision={("0", "2"): 0},
        cluster_seeds_require={"0": "c0", "1": "c0", "2": "c1", "3": "c1"},
        cluster_seeds_disallow={("1", "2")},
    )

    assert captured_partial_supervision == [
        {
            ("0", "2"): 0,
            ("0", "3"): LARGE_DISTANCE,
            ("0", "1"): 0,
            ("1", "2"): LARGE_DISTANCE,
            ("1", "3"): LARGE_DISTANCE,
            ("2", "3"): 0,
        }
    ]
    assert captured_incremental_flags == [True]


def test_predict_from_rust_featurizer_uses_native_seed_constraints_without_python_pair_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeFeaturizer:
        def signature_ids(self):
            return ["0", "1", "2", "3"]

        def cluster_seeds_require(self):
            return [("0", "c0"), ("1", "c0"), ("2", "c1"), ("3", "c1")]

    def fake_make_dists(self, block_dict, _rust_featurizer, **kwargs):
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
        captured["incremental_dont_use_cluster_seeds"] = kwargs["incremental_dont_use_cluster_seeds"]
        block_key, signatures = next(iter(block_dict.items()))
        return {block_key: np.zeros((len(signatures), len(signatures)), dtype=np.float64)}

    def fake_cluster_one_block(
        self,
        signatures,
        pairwise_proba,
        effective_cluster_model_params,
        dataset,
        all_disallow_signature_ids,
        *,
        block_key,
    ):
        del self, pairwise_proba, effective_cluster_model_params, all_disallow_signature_ids, block_key
        captured["proxy_cluster_seeds_require"] = dict(dataset.cluster_seeds_require)
        return [0 for _signature in signatures]

    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", fake_make_dists)
    monkeypatch.setattr(Clusterer, "_cluster_one_block_with_logging", fake_cluster_one_block)

    _dummy_clusterer(cluster_model=None).predict_from_rust_featurizer(
        {"block": ["0", "1", "2", "3"]},
        _FakeFeaturizer(),
    )

    assert captured["partial_supervision"] == {}
    assert captured["incremental_dont_use_cluster_seeds"] is False
    assert captured["proxy_cluster_seeds_require"] == {
        "0": "c0",
        "1": "c0",
        "2": "c1",
        "3": "c1",
    }


def test_seed_override_partial_supervision_respects_existing_reverse_pair() -> None:
    merged = model_module._partial_supervision_with_cluster_seed_overrides(
        ["0", "1"],
        {("1", "0"): 42.0},
        cluster_seeds_require={"0": "c0", "1": "c1"},
        cluster_seeds_disallow={("0", "1")},
    )

    assert merged == {("1", "0"): 42.0}


def test_predict_from_arrow_paths_builds_filtered_arrow_featurizer(monkeypatch, tmp_path):
    import pyarrow as pa

    captured = {}
    arrow_paths = {
        "signatures": str(tmp_path / "signatures.arrow"),
        "papers": str(tmp_path / "papers.arrow"),
        "paper_authors": str(tmp_path / "paper_authors.arrow"),
    }
    for path in arrow_paths.values():
        Path(path).touch()
    arrow_paths = _with_fake_batch_indexes(arrow_paths, tmp_path)
    name_counts_index, _metrics = write_name_counts_index(
        tmp_path, tiny_name_counts_tuple(), tiny_name_counts_provenance()
    )
    disallow_path = tmp_path / "cluster_seed_disallows.arrow"
    disallow_table = pa.table(
        {
            "signature_id_1": pa.array(["0"], type=pa.string()),
            "signature_id_2": pa.array(["2"], type=pa.string()),
        }
    )
    with pa.OSFile(str(disallow_path), "wb") as sink:
        with pa.ipc.new_file(sink, disallow_table.schema) as writer:
            writer.write_table(disallow_table)

    class _FakeFeaturizer:
        def signature_ids(self):
            return ["0", "1", "2"]

    def fake_build_from_arrow_paths(paths, **kwargs):
        captured["paths"] = paths
        captured["signature_ids"] = tuple(kwargs["signature_ids"])
        captured["load_name_counts"] = kwargs["load_name_counts"]
        return _FakeFeaturizer()

    def fake_predict_from_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        captured["block_dict"] = block_dict
        captured["rust_featurizer"] = rust_featurizer
        captured["total_ram_bytes"] = kwargs["total_ram_bytes"]
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
        captured["cluster_seeds_disallow"] = kwargs.get("cluster_seeds_disallow")
        return {"block_0": ["0", "1", "2"]}, None

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_from_arrow_paths)
    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", fake_predict_from_rust_featurizer)

    prediction_paths = {
        **arrow_paths,
        "name_counts_index": name_counts_index,
        "cluster_seed_disallows": str(disallow_path),
    }
    write_test_arrow_artifact_manifest(tmp_path, prediction_paths)
    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict_from_arrow_paths(
        {"block": ["0", "1", "2"]},
        prediction_paths,
        load_name_counts=True,
        total_ram_bytes=123,
    )

    assert result == {"block_0": ["0", "1", "2"]}
    assert dists is None
    assert captured["signature_ids"] == ("0", "1", "2")
    assert captured["load_name_counts"] is True
    assert captured["total_ram_bytes"] == 123
    assert captured["partial_supervision"] == {("0", "2"): LARGE_DISTANCE}
    assert captured["cluster_seeds_disallow"] == {("0", "2")}
    telemetry = clusterer._last_arrow_predict_telemetry
    assert telemetry["signature_count"] == 3
    assert telemetry["block_count"] == 1
    assert telemetry["pair_count"] == 3
    assert telemetry["arrow_featurizer_seconds"] >= 0
    assert telemetry["rust_featurizer_predict_seconds"] >= 0


def test_predict_from_arrow_paths_omits_unused_specter_for_non_embedding_model(monkeypatch, tmp_path):
    captured = {}
    arrow_paths = {
        "signatures": str(tmp_path / "signatures.arrow"),
        "papers": str(tmp_path / "papers.arrow"),
        "paper_authors": str(tmp_path / "paper_authors.arrow"),
        "specter": str(tmp_path / "specter.arrow"),
    }
    for path in arrow_paths.values():
        Path(path).touch()
    arrow_paths = _with_fake_batch_indexes(arrow_paths, tmp_path)
    arrow_paths.pop("specter_batch_index")

    class _FakeFeaturizer:
        def signature_ids(self):
            return ["0", "1"]

    def fake_build_from_arrow_paths(paths, **kwargs):
        captured["paths"] = dict(paths)
        captured["signature_ids"] = tuple(kwargs["signature_ids"])
        return _FakeFeaturizer()

    def fake_predict_from_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        captured["block_dict"] = block_dict
        captured["rust_featurizer"] = rust_featurizer
        return {"block": ["0", "1"]}, None

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_from_arrow_paths)
    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", fake_predict_from_rust_featurizer)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict_from_arrow_paths({"block": ["0", "1"]}, arrow_paths)

    assert result == {"block": ["0", "1"]}
    assert dists is None
    assert captured["signature_ids"] == ("0", "1")
    assert "specter" not in captured["paths"]
    assert "specter_batch_index" not in captured["paths"]


def test_predict_from_arrow_paths_merges_explicit_disallows(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_build_from_arrow_paths(*_args, **_kwargs):
        return object()

    def fake_predict_from_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        captured["self"] = self
        captured["block_dict"] = block_dict
        captured["rust_featurizer"] = rust_featurizer
        captured["partial_supervision"] = dict(kwargs["partial_supervision"])
        captured["cluster_seeds_disallow"] = kwargs.get("cluster_seeds_disallow")
        return {"block": ["0", "1", "2"]}, None

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_from_arrow_paths)
    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", fake_predict_from_rust_featurizer)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict_from_arrow_paths(
        {"block": ["0", "1", "2"]},
        write_minimal_arrow_prediction_bundle(tmp_path),
        partial_supervision={("0", "1"): 0, ("0", "2"): 0},
        cluster_seeds_disallow={("0", "1"), ("1", "2")},
    )

    assert result == {"block": ["0", "1", "2"]}
    assert dists is None
    assert captured["partial_supervision"] == {
        ("0", "1"): 0,
        ("0", "2"): 0,
    }
    assert captured["cluster_seeds_disallow"] == {("0", "1"), ("1", "2")}


def test_predict_from_arrow_paths_reports_structured_missing_artifacts(tmp_path):
    signatures_path = tmp_path / "signatures.arrow"
    signatures_path.touch()
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
    )

    with pytest.raises(model_module.MissingArrowArtifactError) as exc_info:
        clusterer.predict_from_arrow_paths(
            {"block": ["0", "1"]},
            {
                "signatures": str(signatures_path),
                "papers": str(tmp_path / "missing_papers.arrow"),
                "paper_authors": str(tmp_path / "missing_paper_authors.arrow"),
            },
        )

    error = exc_info.value
    assert error.context == "Clusterer.predict_from_arrow_paths"
    assert error.required_keys == (
        "name_counts_index",
        "paper_authors",
        "paper_authors_batch_index",
        "papers",
        "papers_batch_index",
        "signatures",
        "signatures_batch_index",
    )
    assert error.missing_keys == (
        "name_counts_index",
        "paper_authors_batch_index",
        "papers_batch_index",
        "signatures_batch_index",
    )
    assert error.missing_files == {
        "papers": str(tmp_path / "missing_papers.arrow"),
        "paper_authors": str(tmp_path / "missing_paper_authors.arrow"),
    }
    assert "producer hint" in str(error)


def test_predict_from_arrow_paths_rejects_disabling_model_required_name_counts() -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
    )

    with pytest.raises(
        ValueError,
        match="cannot run with load_name_counts=False when the clusterer selects name_counts features",
    ):
        clusterer.predict_from_arrow_paths(
            {"block": ["0", "1"]},
            {},
            load_name_counts=False,
        )


def test_predict_from_arrow_paths_rejects_declared_missing_optional_sidecar(tmp_path):
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    arrow_paths = _with_fake_batch_indexes(arrow_paths, tmp_path)

    clusterer = _dummy_clusterer(cluster_model=None)
    with pytest.raises(model_module.MissingArrowArtifactError) as exc_info:
        clusterer.predict_from_arrow_paths(
            {"block": ["0", "1"]},
            {
                **arrow_paths,
                "cluster_seed_disallows": str(tmp_path / "missing_cluster_seed_disallows.arrow"),
            },
        )

    assert exc_info.value.missing_files == {
        "cluster_seed_disallows": str(tmp_path / "missing_cluster_seed_disallows.arrow")
    }


def test_predict_from_arrow_paths_routes_only_oversized_blocks_through_native_subblocking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    native_calls: list[dict[str, Any]] = []
    predicted_blocks: list[str] = []
    predict_calls: list[tuple[str, ...]] = []
    events: list[str] = []

    class FakeRustFeaturizer:
        def cluster_seeds_require(self):
            return []

        def signature_rule_metadata(self):
            raise AssertionError("subblocking must not export all signature metadata")

    def fake_native_subblocking(paths, signature_ids, **kwargs):
        events.append("subblocking")
        native_calls.append({"paths": paths, "signature_ids": list(signature_ids), **kwargs})
        return {"beta": ["s3", "s4"], "alpha": ["s1", "s2"]}, {"final_subblock_count": 2}

    def fake_build_rust_featurizer(*_args, **_kwargs):
        events.append("featurizer")
        return FakeRustFeaturizer()

    def fake_load_representative_metadata(_paths, signature_ids):
        events.append("representatives")
        assert list(signature_ids) == ["s1", "s3", "s0"]
        return {
            signature_id: SimpleNamespace(
                author_info_first="john",
                author_info_first_normalized_without_apostrophe="john",
            )
            for signature_id in signature_ids
        }

    def fake_predict_from_prepared_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        del self, rust_featurizer, kwargs
        events.append("predict")
        predict_calls.append(tuple(block_dict))
        predicted_blocks.extend(block_dict)
        return {f"cluster:{block_key}": list(signature_ids) for block_key, signature_ids in block_dict.items()}, None

    monkeypatch.setattr(
        model_module,
        "build_rust_featurizer_from_arrow_paths",
        fake_build_rust_featurizer,
    )
    monkeypatch.setattr(model_module, "_make_subblocks_with_telemetry_arrow_rust", fake_native_subblocking)
    monkeypatch.setattr(model_module, "_load_arrow_incremental_signature_info", fake_load_representative_metadata)
    monkeypatch.setattr(
        Clusterer,
        "_predict_from_prepared_rust_featurizer",
        fake_predict_from_prepared_rust_featurizer,
    )
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Python subblocking must not run")),
    )

    clusterer = _dummy_clusterer(cluster_model=None)
    clusterer.random_state = 17
    result, dists = clusterer.predict_from_arrow_paths(
        {"small": ["s0"], "large": ["s1", "s2", "s3", "s4"]},
        arrow_paths,
        batching_threshold=2,
    )

    assert dists is None
    assert native_calls[0]["signature_ids"] == ["s1", "s2", "s3", "s4"]
    assert native_calls[0]["maximum_size"] == 2
    assert native_calls[0]["graph_subblocking_random_seed"] == 17
    assert native_calls[0]["use_orcid_subblocking"] is True
    assert predicted_blocks == ["large|subblock=alpha", "large|subblock=beta", "small"]
    assert predict_calls == [("large|subblock=alpha", "large|subblock=beta", "small")]
    assert events == ["subblocking", "representatives", "featurizer", "predict"]
    assert {signature_id for cluster in result.values() for signature_id in cluster} == {
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
    }
    assert clusterer._last_rust_arrow_subblocking_telemetry["oversized_block_count"] == 1


def test_predict_from_arrow_paths_uses_request_featurizer_cluster_seeds_without_python_pair_expansion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path)
    captured: dict[str, Any] = {}

    class FakeRustFeaturizer:
        def cluster_seeds_require(self):
            return [("s1", "current"), ("s2", "current")]

    def fake_build_rust_featurizer(paths, **_kwargs):
        captured["build_seed_map"] = model_module._read_cluster_seeds_arrow(Path(paths["cluster_seeds"]))
        return FakeRustFeaturizer()

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_rust_featurizer)

    def fake_predict_from_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        del self, rust_featurizer
        captured.update(kwargs)
        return {"cluster": list(block_dict["block"])}, None

    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", fake_predict_from_rust_featurizer)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, _ = clusterer.predict_from_arrow_paths(
        {"block": ["s1", "s2"]},
        arrow_paths,
        cluster_seeds_require={"s1": "current", "s2": "current"},
    )

    assert result == {"cluster": ["s1", "s2"]}
    assert captured["build_seed_map"] == {"s1": "current", "s2": "current"}
    assert captured["cluster_seeds_require"] is None


def test_arrow_subblocking_presplits_altered_profiles_before_partition_and_uses_rewritten_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    events: list[str] = []
    captured: dict[str, Any] = {}

    class FakeRustFeaturizer:
        def cluster_seeds_require(self):
            return [("s1", "claimed_0"), ("s2", "claimed_0"), ("s3", "claimed_1")]

    def fake_seed_setup(self, dataset, *_args, **_kwargs):
        del self
        events.append("presplit")
        captured["presplit_input_seeds"] = dict(dataset.cluster_seeds_require)
        captured["presplit_altered"] = list(dataset.altered_cluster_signatures or [])
        split = {"s1": "claimed_0", "s2": "claimed_0", "s3": "claimed_1"}
        return split, {"claimed_0": "claimed", "claimed_1": "claimed"}, {}, {}

    def fake_native_subblocking(paths, signature_ids, **_kwargs):
        events.append("subblocking")
        captured["native_seed_path"] = paths["cluster_seeds"]
        captured["native_seed_map"] = model_module._read_cluster_seeds_arrow(Path(paths["cluster_seeds"]))
        assert captured["native_seed_map"] == {"s1": "claimed_0", "s2": "claimed_0", "s3": "claimed_1"}
        return {"left": [signature_ids[0], signature_ids[2]], "right": [signature_ids[1]]}, {}

    def fake_build_rust_featurizer(paths, **_kwargs):
        captured["build_seed_path"] = paths["cluster_seeds"]
        captured["build_seed_map"] = model_module._read_cluster_seeds_arrow(Path(paths["cluster_seeds"]))
        return FakeRustFeaturizer()

    def fake_predict_from_prepared_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        del self, rust_featurizer
        captured["predicted_blocks"] = dict(block_dict)
        captured["explicit_cluster_seeds_require"] = kwargs["explicit_cluster_seeds_require"]
        captured["proxy_cluster_seeds_require"] = dict(kwargs["proxy_dataset"].cluster_seeds_require)
        return {f"cluster:{key}": list(values) for key, values in block_dict.items()}, None

    monkeypatch.setattr(Clusterer, "_build_incremental_seed_setup", fake_seed_setup)
    monkeypatch.setattr(model_module, "_make_subblocks_with_telemetry_arrow_rust", fake_native_subblocking)
    monkeypatch.setattr(
        model_module,
        "build_rust_featurizer_from_arrow_paths",
        fake_build_rust_featurizer,
    )
    monkeypatch.setattr(
        model_module,
        "_load_arrow_incremental_signature_info",
        lambda _paths, signature_ids: {
            str(signature_id): SimpleNamespace(
                author_info_first="john",
                author_info_first_normalized_without_apostrophe="john",
                author_info_orcid=None,
            )
            for signature_id in signature_ids
        },
    )
    monkeypatch.setattr(
        Clusterer,
        "_predict_from_prepared_rust_featurizer",
        fake_predict_from_prepared_rust_featurizer,
    )

    clusterer = _dummy_clusterer(cluster_model=None)
    clusterer.predict_from_arrow_paths(
        {"block": ["s1", "s2", "s3"]},
        arrow_paths,
        batching_threshold=2,
        cluster_seeds_require={"s1": "claimed", "s2": "claimed", "s3": "claimed"},
        altered_cluster_signatures=["s1"],
    )

    assert events == ["presplit", "subblocking"]
    assert captured["presplit_input_seeds"] == {"s1": "claimed", "s2": "claimed", "s3": "claimed"}
    assert captured["presplit_altered"] == ["s1"]
    expected_rewritten_seeds = {"s1": "claimed_0", "s2": "claimed_0", "s3": "claimed_1"}
    assert captured["native_seed_map"] == expected_rewritten_seeds
    assert captured["build_seed_map"] == expected_rewritten_seeds
    assert captured["native_seed_path"] == captured["build_seed_path"]
    assert captured["explicit_cluster_seeds_require"] is None
    assert captured["proxy_cluster_seeds_require"] == expected_rewritten_seeds
    assert ["s1", "s2"] in captured["predicted_blocks"].values()


def test_arrow_subblocking_attaches_initial_only_groups_sequentially(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    incremental_calls: list[dict[str, Any]] = []
    featurizer_name_tuples: list[object] = []
    featurizer_signature_ids: list[tuple[str, ...]] = []
    release_events: list[str] = []

    class FakeRustFeaturizer:
        def cluster_seeds_require(self):
            return []

        def __del__(self):
            release_events.append("released")

        def signature_rule_metadata(self):
            raise AssertionError("subblocking must not export all signature metadata")

    def fake_build_rust_featurizer(*_args, **kwargs):
        featurizer_name_tuples.append(kwargs["name_tuples"])
        featurizer_signature_ids.append(tuple(kwargs["signature_ids"]))
        return FakeRustFeaturizer()

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_rust_featurizer)
    monkeypatch.setattr(
        model_module,
        "_make_subblocks_with_telemetry_arrow_rust",
        lambda *_args, **_kwargs: (
            {"multi": ["m0", "m1"], "initial_a": ["i0"], "initial_b": ["i1"]},
            {},
        ),
    )
    monkeypatch.setattr(
        model_module,
        "_load_arrow_incremental_signature_info",
        lambda _paths, signature_ids: {
            signature_id: SimpleNamespace(
                author_info_first=("j" if str(signature_id).startswith("i") else "john"),
                author_info_first_normalized_without_apostrophe=("j" if str(signature_id).startswith("i") else "john"),
            )
            for signature_id in signature_ids
        },
    )

    def fake_predict_from_prepared_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        del self, rust_featurizer, kwargs
        assert block_dict == {"block|subblock=multi": ["m0", "m1"]}
        return {"seed": ["m0", "m1"]}, None

    def fake_predict_incremental_from_arrow_paths(self, block_signatures, paths, **kwargs):
        del self, paths
        assert release_events == ["released"]
        incremental_calls.append(
            {
                "block_signatures": list(block_signatures),
                **kwargs,
                "cluster_seeds_require": dict(kwargs["cluster_seeds_require"]),
            }
        )
        clusters = {"seed": list(kwargs["cluster_seeds_require"])}
        clusters[f"new_{len(incremental_calls)}"] = list(block_signatures)
        return {"clusters": clusters}

    monkeypatch.setattr(
        Clusterer,
        "_predict_from_prepared_rust_featurizer",
        fake_predict_from_prepared_rust_featurizer,
    )
    monkeypatch.setattr(
        Clusterer,
        "predict_incremental_from_arrow_paths",
        fake_predict_incremental_from_arrow_paths,
    )

    clusterer = _dummy_clusterer(cluster_model=None)
    mutable_name_tuples = {("john", "jon")}
    result, _ = clusterer.predict_from_arrow_paths(
        {"block": ["m0", "m1", "i0", "i1"]},
        arrow_paths,
        batching_threshold=2,
        name_tuples=mutable_name_tuples,
    )

    assert [call["block_signatures"] for call in incremental_calls] == [["i0"], ["i1"]]
    assert incremental_calls[0]["cluster_seeds_require"] == {"m0": "seed", "m1": "seed"}
    assert incremental_calls[1]["cluster_seeds_require"] == {
        "m0": "seed",
        "m1": "seed",
        "i0": "new_1",
    }
    assert all(call["altered_cluster_signatures"] == [] for call in incremental_calls)
    assert all(call["batching_threshold"] == 2 for call in incremental_calls)
    assert featurizer_signature_ids == [("m0", "m1")]
    assert clusterer._last_arrow_predict_telemetry["featurizer_signature_count"] == 2
    resolved_name_tuples = featurizer_name_tuples[0]
    assert isinstance(resolved_name_tuples, frozenset)
    assert all(call["name_tuples"] is resolved_name_tuples for call in incremental_calls)
    assert {signature_id for cluster in result.values() for signature_id in cluster} == {"m0", "m1", "i0", "i1"}


def test_arrow_subblocking_keeps_singleton_seed_initial_group_on_sequential_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    featurizer_signature_ids: list[tuple[str, ...]] = []
    incremental_calls: list[dict[str, Any]] = []

    class FakeRustFeaturizer:
        def cluster_seeds_require(self):
            return [("i0", "singleton")]

    def fake_build_rust_featurizer(*_args, **kwargs):
        featurizer_signature_ids.append(tuple(kwargs["signature_ids"]))
        return FakeRustFeaturizer()

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_rust_featurizer)
    monkeypatch.setattr(
        model_module,
        "_make_subblocks_with_telemetry_arrow_rust",
        lambda *_args, **_kwargs: ({"multi": ["m0", "m1"], "initial": ["i0"]}, {}),
    )
    monkeypatch.setattr(
        model_module,
        "_load_arrow_incremental_signature_info",
        lambda _paths, signature_ids: {
            signature_id: SimpleNamespace(
                author_info_first=("j" if signature_id == "i0" else "john"),
                author_info_first_normalized_without_apostrophe=("j" if signature_id == "i0" else "john"),
            )
            for signature_id in signature_ids
        },
    )

    def fake_predict_from_prepared_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        del self, rust_featurizer, kwargs
        assert block_dict == {"block|subblock=multi": ["m0", "m1"]}
        return {"bulk": ["m0", "m1"]}, None

    def fake_predict_incremental_from_arrow_paths(self, block_signatures, paths, **kwargs):
        del self, paths
        incremental_calls.append(
            {
                "block_signatures": list(block_signatures),
                "cluster_seeds_require": dict(kwargs["cluster_seeds_require"]),
            }
        )
        return {
            "clusters": {
                "bulk": ["m0", "m1"],
                "singleton": ["i0"],
            }
        }

    monkeypatch.setattr(
        Clusterer,
        "_predict_from_prepared_rust_featurizer",
        fake_predict_from_prepared_rust_featurizer,
    )
    monkeypatch.setattr(
        Clusterer,
        "predict_incremental_from_arrow_paths",
        fake_predict_incremental_from_arrow_paths,
    )

    result, _ = _dummy_clusterer(cluster_model=None).predict_from_arrow_paths(
        {"block": ["m0", "m1", "i0"]},
        arrow_paths,
        batching_threshold=2,
        cluster_seeds_require={"i0": "singleton"},
    )

    assert featurizer_signature_ids == [("m0", "m1")]
    assert incremental_calls == [
        {
            "block_signatures": ["i0"],
            "cluster_seeds_require": {
                "m0": "bulk",
                "m1": "bulk",
                "i0": "singleton",
            },
        }
    ]
    assert result == {"bulk": ["m0", "m1"], "singleton": ["i0"]}


def test_arrow_subblocking_all_initial_groups_do_not_require_incremental_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    featurizer_signature_ids: list[tuple[str, ...]] = []

    class FakeRustFeaturizer:
        def cluster_seeds_require(self):
            return []

        def signature_rule_metadata(self):
            raise AssertionError("subblocking must not export all signature metadata")

    def fake_build_rust_featurizer(*_args, **kwargs):
        featurizer_signature_ids.append(tuple(kwargs["signature_ids"]))
        return FakeRustFeaturizer()

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_rust_featurizer)
    monkeypatch.setattr(
        model_module,
        "_make_subblocks_with_telemetry_arrow_rust",
        lambda *_args, **_kwargs: ({"a": ["i0", "i1"], "b": ["i2"]}, {}),
    )
    monkeypatch.setattr(
        model_module,
        "_load_arrow_incremental_signature_info",
        lambda _paths, signature_ids: {
            signature_id: SimpleNamespace(
                author_info_first="j",
                author_info_first_normalized_without_apostrophe="j",
            )
            for signature_id in signature_ids
        },
    )
    monkeypatch.setattr(
        Clusterer,
        "predict_incremental_from_arrow_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("seedless incremental must not run")),
    )

    def fake_predict_from_prepared_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        del self, rust_featurizer, kwargs
        return {block_key: list(signature_ids) for block_key, signature_ids in block_dict.items()}, None

    monkeypatch.setattr(
        Clusterer,
        "_predict_from_prepared_rust_featurizer",
        fake_predict_from_prepared_rust_featurizer,
    )

    clusterer = _dummy_clusterer(cluster_model=None)
    result, _ = clusterer.predict_from_arrow_paths(
        {"block": ["i0", "i1", "i2"]},
        arrow_paths,
        batching_threshold=2,
    )

    assert {signature_id for cluster in result.values() for signature_id in cluster} == {"i0", "i1", "i2"}
    assert featurizer_signature_ids == [("i0", "i1", "i2")]


@pytest.mark.parametrize("batching_threshold", [0, -1])
def test_predict_from_arrow_paths_rejects_nonpositive_batching_threshold(batching_threshold: int) -> None:
    clusterer = _dummy_clusterer(cluster_model=None)

    with pytest.raises(ValueError, match="batching_threshold must be positive"):
        clusterer.predict_from_arrow_paths({}, {}, batching_threshold=batching_threshold)


def test_predict_from_arrow_paths_rejects_subblocking_with_precomputed_dists() -> None:
    clusterer = _dummy_clusterer(cluster_model=None)

    with pytest.raises(ValueError, match="batching_threshold cannot be used with precomputed dists"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            {},
            dists={"block": np.zeros((2, 2))},
            batching_threshold=1,
        )


def test_predict_from_arrow_paths_native_subblocking_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        model_module,
        "_make_subblocks_with_telemetry_arrow_rust",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("native subblocking failed")),
    )
    clusterer = _dummy_clusterer(cluster_model=None)

    with pytest.raises(RuntimeError, match="native subblocking failed"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            arrow_paths,
            batching_threshold=1,
        )


def test_predict_from_arrow_paths_requires_specter_when_subblocking_needs_graph(tmp_path: Path) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path)
    clusterer = _dummy_clusterer(cluster_model=None)

    with pytest.raises(model_module.MissingArrowArtifactError) as exc_info:
        clusterer.predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            arrow_paths,
            batching_threshold=1,
        )

    assert "specter" in exc_info.value.required_keys

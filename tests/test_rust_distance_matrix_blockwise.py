from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from s2and import feature_port

if not feature_port.rust_featurizer_available():
    raise pytest.skip.Exception("s2and_rust extension is unavailable", allow_module_level=True)

import numpy as np

import s2and.model as model_module
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow, write_name_counts_index
from s2and.model import Clusterer
from tests.helpers import patch_tiny_name_counts_loader

s2and_rust = cast(Any, feature_port.s2and_rust)


def _dummy_dataset(name: str) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        load_name_counts=False,
        n_jobs=1,
    )


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
        load_name_counts=False,
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
                "given_block": "smith",
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
        load_name_counts=False,
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
        use_cache=False,
        use_default_constraints_as_supervision=use_default_constraints_as_supervision,
        batch_size=2,
    )


def _with_fake_batch_indexes(arrow_paths: dict[str, str], tmp_path: Path) -> dict[str, str]:
    indexed = dict(arrow_paths)
    for key in ("signatures", "papers", "paper_authors"):
        index_key = f"{key}_batch_index"
        index_path = tmp_path / f"{key}.{index_key}.bin"
        index_path.touch()
        indexed[index_key] = str(index_path)
    if "specter" in indexed:
        index_path = tmp_path / "specter.specter_batch_index.bin"
        index_path.touch()
        indexed["specter_batch_index"] = str(index_path)
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


def test_rust_from_dataset_expects_python_normalized_specter_dict():
    tuple_input_dataset = _specter_dataset(
        "tuple_specter_python_normalized_dataset",
        (np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32), ["p1", "p2"]),
    )
    dict_dataset = _specter_dataset(
        "dict_specter_rust_dataset",
        {
            "p1": np.asarray([1.0, 0.0], dtype=np.float32),
            "p2": np.asarray([1.0, 0.0], dtype=np.float32),
        },
    )
    no_specter_dataset = _specter_dataset("no_specter_rust_dataset", None)

    tuple_featurizer = s2and_rust.RustFeaturizer.from_dataset(tuple_input_dataset, 0.0, 10000.0, 1)
    dict_featurizer = s2and_rust.RustFeaturizer.from_dataset(dict_dataset, 0.0, 10000.0, 1)
    no_specter_featurizer = s2and_rust.RustFeaturizer.from_dataset(no_specter_dataset, 0.0, 10000.0, 1)
    pairs = [("s1", "s2")]
    tuple_features = _indexed_pair_matrix(tuple_featurizer, pairs)
    dict_features = _indexed_pair_matrix(dict_featurizer, pairs)
    no_specter_features = _indexed_pair_matrix(no_specter_featurizer, pairs)

    np.testing.assert_allclose(tuple_features, dict_features, equal_nan=True)
    assert not np.allclose(tuple_features, no_specter_features, equal_nan=True)

    dict_dataset.specter_embeddings = (
        np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        ["p1", "p2"],
    )
    with pytest.raises(TypeError, match="specter_embeddings to be a dict"):
        s2and_rust.RustFeaturizer.from_dataset(dict_dataset, 0.0, 10000.0, 1)


def test_rust_constraint_allows_missing_first_name_like_python():
    dataset = _empty_first_constraint_dataset()

    assert dataset.get_constraint("empty", "named", low_value=0, high_value=LARGE_DISTANCE, suppress_orcid=True) is None

    featurizer = s2and_rust.RustFeaturizer.from_dataset(dataset, 0.0, float(LARGE_DISTANCE), 1)
    signature_id_to_index = {str(signature_id): index for index, signature_id in enumerate(featurizer.signature_ids())}

    assert featurizer.get_constraints_matrix_indexed(
        [(signature_id_to_index["empty"], signature_id_to_index["named"])],
        suppress_orcid=True,
    ) == [None]


def test_make_distance_matrices_rust_blockwise_fastcluster(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
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

    dataset = _dummy_dataset("dummy_rust_blockwise_fastcluster")
    clusterer = _dummy_clusterer(cluster_model=None)
    signatures = ["0", "1", "2", "3"]
    partial_supervision, expected_flat = _partial_supervision_for_upper_triangle(signatures)

    output = clusterer.make_distance_matrices(
        {"block": signatures},
        dataset,
        partial_supervision=partial_supervision,
    )
    matrix = output["block"]

    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix, expected_flat, rtol=1e-10, atol=1e-12)
    assert featurize_call_sizes == [2, 2, 2]


def test_make_distance_matrices_rust_blockwise_square_matrix(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
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

    dataset = _dummy_dataset("dummy_rust_blockwise_square")
    clusterer = _dummy_clusterer(cluster_model=object())
    signatures = ["0", "1", "2", "3"]
    partial_supervision, expected_flat = _partial_supervision_for_upper_triangle(signatures)

    output = clusterer.make_distance_matrices(
        {"block": signatures},
        dataset,
        partial_supervision=partial_supervision,
    )
    matrix = output["block"]

    expected_square = np.zeros((4, 4), dtype=np.float16)
    expected_square[np.triu_indices(4, k=1)] = expected_flat.astype(np.float16)
    expected_square = expected_square + expected_square.T
    np.fill_diagonal(expected_square, 0)

    assert matrix.shape == (4, 4)
    np.testing.assert_allclose(matrix, expected_square, rtol=0, atol=0)
    assert featurize_call_sizes == [2, 2, 2]


def test_make_distance_matrices_rust_fused_upper_triangle_api(monkeypatch):
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
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
            out_cols = len(selected_indices) if selected_indices is not None else 39
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
        _nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del labels, runtime_context, _kwargs
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    dataset = _dummy_dataset("dummy_rust_blockwise_fused")
    clusterer = _dummy_clusterer(
        cluster_model=None,
        use_default_constraints_as_supervision=True,
    )
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


def test_make_distance_matrices_from_rust_featurizer_avoids_anddata_lookup(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "_get_rust_featurizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ANDData featurizer lookup is not needed")),
    )
    monkeypatch.setattr(
        model_module,
        "many_pairs_featurize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Python pair featurization is not needed")),
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
            out_cols = len(selected_indices) if selected_indices is not None else 39
            out = np.zeros((len(pair_slice), out_cols), dtype=np.float64)
            for row_offset in range(len(pair_slice)):
                out[row_offset, :] = float(start_offset + row_offset + 1) / 10.0
            return out

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        features,
        labels,
        _nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del labels, runtime_context, _kwargs
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    clusterer = _dummy_clusterer(
        cluster_model=None,
        use_default_constraints_as_supervision=True,
    )
    output = clusterer.make_distance_matrices_from_rust_featurizer(
        {"block": ["0", "1", "2", "3"]},
        _FakeFusedFeaturizer(),
    )

    np.testing.assert_allclose(output["block"], np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), rtol=0, atol=0)
    assert captured["constraint_calls"] == 3
    assert captured["feature_calls"] == 3


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
        _nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del _classifier, _nameless_classifier, _nameless_features, _batch_label, runtime_context, _kwargs
        assert np.isnan(labels).all()
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    clusterer = _dummy_clusterer(
        cluster_model=model_module.FastCluster(linkage="average"),
        use_default_constraints_as_supervision=False,
    )
    output = clusterer.make_distance_matrices_from_rust_featurizer(
        {"block": ["0", "1", "2", "3"]},
        _FakeFeaturizer(),
    )

    np.testing.assert_allclose(output["block"], np.arange(6, dtype=np.float64), rtol=0, atol=0)
    assert captured["feature_calls"] == 3
    telemetry = clusterer._last_rust_featurizer_make_dists_telemetry
    assert telemetry["chunk_count"] == 3
    assert telemetry["upper_triangle_index_seconds"] == 0.0


def test_make_distance_matrices_from_rust_featurizer_skips_fastcluster_constraint_index_conversion(monkeypatch):
    class _IndexValuesThatShouldNotBeConverted:
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("FastCluster vector writes do not need converted constraint index arrays")

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
        _nameless_features,
        _batch_label,
        runtime_context=None,
        **_kwargs,
    ):
        del _classifier, _nameless_classifier, _nameless_features, _batch_label, runtime_context, _kwargs
        assert np.isnan(labels).all()
        return np.asarray(features[:, 0], dtype=np.float64), 0.0

    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)
    monkeypatch.setattr(
        model_module,
        "get_constraints_block_upper_triangle_indexed_rust",
        lambda _dataset, _block_signature_indices, *, max_pairs, **_kwargs: (
            _IndexValuesThatShouldNotBeConverted(),
            _IndexValuesThatShouldNotBeConverted(),
            [None] * int(max_pairs),
        ),
    )

    clusterer = _dummy_clusterer(
        cluster_model=model_module.FastCluster(linkage="average"),
        use_default_constraints_as_supervision=True,
    )
    output = clusterer.make_distance_matrices_from_rust_featurizer(
        {"block": ["0", "1", "2", "3"]},
        _FakeFeaturizer(),
    )

    np.testing.assert_allclose(output["block"], np.arange(6, dtype=np.float64), rtol=0, atol=0)
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
        lambda _dataset, _block_signature_indices, *, max_pairs, **_kwargs: (
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

    class _FakeFeaturizer:
        def signature_rule_metadata(self):
            return [(str(index), f"First{index}", None) for index in range(4)]

    def fake_make_dists(
        self,
        block_dict,
        _rust_featurizer,
        **_kwargs,
    ):
        make_calls.append(tuple(block_dict))
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

    monkeypatch.setattr(Clusterer, "make_distance_matrices_from_rust_featurizer", fake_make_dists)
    monkeypatch.setattr(Clusterer, "_cluster_one_block_with_logging", fake_cluster_one_block)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict_from_rust_featurizer(
        {"a": ["0", "1"], "b": ["2", "3"]},
        _FakeFeaturizer(),
        cluster_seeds_require={},
    )

    assert dists is None
    assert make_calls == [("a",), ("b",)]
    assert cluster_calls == [("a", ("0", "1")), ("b", ("2", "3"))]
    assert result == {"a_0": ["0", "1"], "b_0": ["2", "3"]}
    telemetry = clusterer._last_rust_featurizer_predict_telemetry
    assert float(telemetry["make_dists_total_seconds"]) >= 0.0
    assert telemetry["make_dists_constraint_seconds"] == 0.2
    assert telemetry["make_dists_block_count"] == 2
    assert telemetry["make_dists_pair_count"] == 2


def test_predict_from_rust_featurizer_rejects_disallows_with_precomputed_dists() -> None:
    class _FakeFeaturizer:
        def signature_rule_metadata(self):
            return [("0", "First0", None), ("1", "First1", None)]

    clusterer = _dummy_clusterer(cluster_model=None)

    with pytest.raises(ValueError, match="precomputed dists"):
        clusterer.predict_from_rust_featurizer(
            {"block": ["0", "1"]},
            _FakeFeaturizer(),
            dists={"block": np.zeros((2, 2), dtype=np.float64)},
            cluster_seeds_require={},
            cluster_seeds_disallow={("0", "1")},
        )


def test_predict_from_rust_featurizer_rejects_require_overrides_with_precomputed_dists() -> None:
    class _FakeFeaturizer:
        def signature_rule_metadata(self):
            return [("0", "First0", None), ("1", "First1", None)]

    clusterer = _dummy_clusterer(cluster_model=None)

    with pytest.raises(ValueError, match="precomputed dists"):
        clusterer.predict_from_rust_featurizer(
            {"block": ["0", "1"]},
            _FakeFeaturizer(),
            dists={"block": np.zeros((2, 2), dtype=np.float64)},
            cluster_seeds_require={"0": "c0", "1": "c0"},
        )


def test_predict_from_rust_featurizer_rejects_implicit_require_with_precomputed_dists() -> None:
    class _FakeFeaturizer:
        def cluster_seeds_require(self):
            return [("0", "c0"), ("1", "c1")]

        def signature_rule_metadata(self):
            return [("0", "First0", None), ("1", "First1", None)]

    clusterer = _dummy_clusterer(cluster_model=None)

    with pytest.raises(ValueError, match="cluster_seeds_require cannot be used with precomputed dists"):
        clusterer.predict_from_rust_featurizer(
            {"block": ["0", "1"]},
            _FakeFeaturizer(),
            dists={"block": np.zeros((2, 2), dtype=np.float64)},
        )


def test_predict_from_rust_featurizer_injects_seed_overrides_into_distance_build(monkeypatch):
    captured_partial_supervision: list[dict[tuple[str, str], int | float]] = []
    captured_incremental_flags: list[bool] = []

    class _FakeFeaturizer:
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

    monkeypatch.setattr(Clusterer, "make_distance_matrices_from_rust_featurizer", fake_make_dists)
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
    patch_tiny_name_counts_loader(monkeypatch)
    name_counts_index, _metrics = write_name_counts_index(tmp_path)
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

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict_from_arrow_paths(
        {"block": ["0", "1", "2"]},
        {
            **arrow_paths,
            "name_counts_index": name_counts_index,
            "cluster_seed_disallows": str(disallow_path),
        },
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


def test_predict_from_arrow_paths_rejects_reference_features(monkeypatch):
    def fail_build_from_arrow_paths(*_args, **_kwargs):
        raise AssertionError("Arrow featurizer build should fail fast before touching inputs")

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fail_build_from_arrow_paths)

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["reference_features"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
    )
    with pytest.raises(ValueError, match="reference_features"):
        clusterer.predict_from_arrow_paths(
            {"block": ["0", "1"]},
            {"signatures": "signatures.arrow", "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"},
        )


def test_predict_from_arrow_paths_merges_explicit_disallows(monkeypatch):
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
        {
            "signatures": __file__,
            "papers": __file__,
            "paper_authors": __file__,
            "signatures_batch_index": __file__,
            "papers_batch_index": __file__,
            "paper_authors_batch_index": __file__,
        },
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


def test_predict_auto_routes_to_arrow_paths_when_dataset_advertises_them(tmp_path, monkeypatch):
    captured = {}
    dataset = _dummy_dataset("dummy_predict_auto_arrow")
    dataset.name_tuples = {("alice", "a")}
    dataset.cluster_seeds_disallow = {("0", "1")}
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
    cast(Any, dataset).arrow_paths = arrow_paths
    runtime_context = type(
        "RuntimeContext",
        (),
        {
            "operation": "cluster_predict",
            "requested_backend": "rust",
            "resolved_backend": "rust",
            "use_rust": True,
            "run_id": "test-auto-arrow-predict",
            "source": "test",
        },
    )()

    def fail_predict_helper(*_args, **_kwargs):
        raise AssertionError("predict should route to Arrow/Rust before legacy predict_helper")

    def fake_predict_from_arrow_paths(self, block_dict, paths, **kwargs):
        captured["self"] = self
        captured["block_dict"] = block_dict
        captured["paths"] = dict(paths)
        captured["runtime_context"] = kwargs["runtime_context"]
        captured["name_tuples"] = kwargs["name_tuples"]
        captured["cluster_seeds_disallow"] = set(kwargs["cluster_seeds_disallow"])
        return {"arrow": ["0", "1"]}, None

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(Clusterer, "predict_helper", fail_predict_helper)
    monkeypatch.setattr(Clusterer, "predict_from_arrow_paths", fake_predict_from_arrow_paths)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict({"block": ["0", "1"]}, dataset)

    assert result == {"arrow": ["0", "1"]}
    assert dists is None
    assert captured["self"] is clusterer
    assert captured["block_dict"] == {"block": ["0", "1"]}
    assert captured["paths"] == arrow_paths
    assert captured["runtime_context"] is runtime_context
    assert captured["name_tuples"] == {("alice", "a")}
    assert captured["cluster_seeds_disallow"] == {("0", "1")}


def test_predict_auto_requires_arrow_paths_with_name_counts_index(tmp_path, monkeypatch):
    monkeypatch.setattr(model_module, "PROJECT_ROOT_PATH", str(tmp_path / "project"))
    dataset = _dummy_dataset("dummy_predict_auto_arrow_missing_name_counts_index")
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
    cast(Any, dataset).arrow_paths = arrow_paths
    runtime_context = type(
        "RuntimeContext",
        (),
        {
            "operation": "cluster_predict",
            "requested_backend": "rust",
            "resolved_backend": "rust",
            "use_rust": True,
            "run_id": "test-auto-arrow-predict-missing-name-counts-index",
            "source": "test",
        },
    )()

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(
        Clusterer,
        "predict_helper",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy predict_helper should not run")),
    )
    monkeypatch.setattr(
        Clusterer,
        "predict_from_arrow_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("incomplete Arrow paths should fail first")),
    )

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
        use_cache=False,
        batch_size=2,
    )
    with pytest.raises(model_module.MissingArrowArtifactError) as exc_info:
        clusterer.predict({"block": ["0", "1"]}, dataset)

    error = exc_info.value
    assert error.context == "Clusterer.predict Rust prediction"
    assert error.missing_keys == ("name_counts_index",)


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


def test_predict_subblocked_rust_requires_arrow_paths(monkeypatch):
    dataset = _dummy_dataset("dummy_predict_subblocked_missing_arrow")
    runtime_context = type(
        "RuntimeContext",
        (),
        {
            "operation": "cluster_predict",
            "requested_backend": "rust",
            "resolved_backend": "rust",
            "use_rust": True,
            "run_id": "test-subblocked-missing-arrow-predict",
            "source": "test",
        },
    )()

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(
        Clusterer,
        "_predict_subblocked",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subblocked fallback should not run")),
    )

    clusterer = _dummy_clusterer(cluster_model=None)
    with pytest.raises(model_module.MissingArrowArtifactError) as exc_info:
        clusterer.predict({"block": ["0", "1"]}, dataset, batching_threshold=10)

    error = exc_info.value
    assert error.context == "Clusterer.predict Rust prediction"
    assert error.missing_keys == ("signatures", "papers", "paper_authors")


def test_predict_rust_requires_explicit_batch_indexes(tmp_path, monkeypatch):
    dataset = _dummy_dataset("dummy_predict_missing_batch_indexes")
    arrow_paths = {}
    for key, filename in {
        "signatures": "signatures.arrow",
        "papers": "papers.arrow",
        "paper_authors": "paper_authors.arrow",
    }.items():
        path = tmp_path / filename
        path.touch()
        arrow_paths[key] = str(path)
    cast(Any, dataset).arrow_paths = arrow_paths
    runtime_context = type(
        "RuntimeContext",
        (),
        {
            "operation": "cluster_predict",
            "requested_backend": "rust",
            "resolved_backend": "rust",
            "use_rust": True,
            "run_id": "test-subblocked-missing-subblocking-index",
            "source": "test",
        },
    )()

    def fail_make_subblocks(*_args, **_kwargs):
        raise AssertionError("Rust production subblocking should not fall back to Python partitioning")

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "make_subblocks", fail_make_subblocks)

    clusterer = _dummy_clusterer(cluster_model=None)
    with pytest.raises(model_module.MissingArrowArtifactError) as exc_info:
        clusterer.predict({"block": ["0", "1"]}, dataset, batching_threshold=1)

    error = exc_info.value
    assert error.context == "Clusterer.predict Rust prediction"
    assert error.missing_keys == ("paper_authors_batch_index", "papers_batch_index", "signatures_batch_index")


def test_predict_subblocked_uses_arrow_featurizer_for_multiple_letter_groups(tmp_path, monkeypatch):
    captured = {"predict_calls": []}
    dataset = _dummy_dataset("dummy_predict_subblocked_arrow")
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
    cast(Any, dataset).arrow_paths = arrow_paths
    runtime_context = type(
        "RuntimeContext",
        (),
        {
            "operation": "cluster_predict",
            "requested_backend": "rust",
            "resolved_backend": "rust",
            "use_rust": True,
            "run_id": "test-subblocked-arrow-predict",
            "source": "test",
        },
    )()

    class _FakeFeaturizer:
        pass

    def fake_build_from_arrow_paths(paths, **kwargs):
        captured["build_paths"] = dict(paths)
        captured["build_signature_ids"] = tuple(kwargs["signature_ids"])
        return _FakeFeaturizer()

    def fail_predict_helper(*_args, **_kwargs):
        raise AssertionError("subblocked Arrow predict should not call legacy predict_helper")

    def fake_predict_from_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        del self, rust_featurizer, kwargs
        captured["predict_calls"].append(dict(block_dict))
        return {f"{next(iter(block_dict))}_0": list(next(iter(block_dict.values())))}, None

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_from_arrow_paths)
    monkeypatch.setattr(Clusterer, "predict_helper", fail_predict_helper)
    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", fake_predict_from_rust_featurizer)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict({"block": ["0", "1"]}, dataset, batching_threshold=10)

    assert result == {"block_0": ["0", "1"]}
    assert dists is None
    assert captured["build_paths"] == arrow_paths
    assert captured["build_signature_ids"] == ("0", "1")
    assert captured["predict_calls"] == [{"block": ["0", "1"]}]


def test_predict_subblocked_materializes_current_cluster_seeds_without_altered_presplit(
    tmp_path,
    monkeypatch,
):
    from contextlib import contextmanager

    captured: dict[str, Any] = {"temporary_seed_context_calls": 0}
    dataset = _dummy_dataset("dummy_predict_subblocked_arrow_current_seeds")
    dataset.cluster_seeds_require = {"0": "c0", "1": "c0"}
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
    cast(Any, dataset).arrow_paths = arrow_paths
    runtime_context = type(
        "RuntimeContext",
        (),
        {
            "operation": "cluster_predict",
            "requested_backend": "rust",
            "resolved_backend": "rust",
            "use_rust": True,
            "run_id": "test-subblocked-current-seeds",
            "source": "test",
        },
    )()

    class _FakeFeaturizer:
        pass

    @contextmanager
    def fake_temporary_seed_paths(dataset_arg, arrow_paths_arg, **_kwargs):
        captured["temporary_seed_context_calls"] += 1
        captured["temporary_seed_map"] = dict(dataset_arg.cluster_seeds_require)
        cluster_seeds_path = tmp_path / "request_cluster_seeds.arrow"
        write_cluster_seeds_arrow(cluster_seeds_path, dataset_arg.cluster_seeds_require)
        yield {**dict(arrow_paths_arg), "cluster_seeds": str(cluster_seeds_path)}

    def fake_build_from_arrow_paths(paths, **kwargs):
        captured["build_paths"] = dict(paths)
        captured["build_signature_ids"] = tuple(kwargs["signature_ids"])
        return _FakeFeaturizer()

    def fake_predict_from_rust_featurizer(self, block_dict, rust_featurizer, **kwargs):
        del self, rust_featurizer, kwargs
        return {f"{next(iter(block_dict))}_0": list(next(iter(block_dict.values())))}, None

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_temporary_arrow_paths_with_current_cluster_seeds", fake_temporary_seed_paths)
    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_from_arrow_paths)
    monkeypatch.setattr(Clusterer, "predict_from_rust_featurizer", fake_predict_from_rust_featurizer)

    clusterer = _dummy_clusterer(cluster_model=None)
    result, dists = clusterer.predict({"block": ["0", "1"]}, dataset, batching_threshold=10)

    assert result == {"block_0": ["0", "1"]}
    assert dists is None
    assert captured["temporary_seed_context_calls"] == 1
    assert captured["temporary_seed_map"] == {"0": "c0", "1": "c0"}
    assert captured["build_paths"] == {**arrow_paths, "cluster_seeds": str(tmp_path / "request_cluster_seeds.arrow")}
    assert captured["build_signature_ids"] == ("0", "1")


def test_compat_arrow_path_discovery_uses_original_signature_path_after_filtering(tmp_path):
    json_dataset_dir = tmp_path / "s2and_mini" / "demo"
    filtered_dir = tmp_path / "filtered"
    arrow_dataset_dir = tmp_path / "s2and_mini_arrow" / "demo"
    json_dataset_dir.mkdir(parents=True)
    filtered_dir.mkdir()
    arrow_dataset_dir.mkdir(parents=True)
    for filename in ("signatures.arrow", "papers.arrow", "paper_authors.arrow", "specter2.arrow"):
        (arrow_dataset_dir / filename).touch()
    dataset = type(
        "Dataset",
        (),
        {
            "original_signatures_path": str(json_dataset_dir / "demo_signatures.json"),
            "signatures_path": str(filtered_dir / "signatures_filtered.json"),
            "specter_embeddings_path": str(json_dataset_dir / "demo_specter2.pkl"),
        },
    )()

    resolved = model_module._resolve_dataset_arrow_paths_for_compat_discovery(
        dataset,
        require_specter=True,
    )

    assert resolved == {
        "signatures": str(arrow_dataset_dir / "signatures.arrow"),
        "papers": str(arrow_dataset_dir / "papers.arrow"),
        "paper_authors": str(arrow_dataset_dir / "paper_authors.arrow"),
        "specter": str(arrow_dataset_dir / "specter2.arrow"),
    }

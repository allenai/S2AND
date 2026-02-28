from __future__ import annotations

import numpy as np

import s2and.model as model_module
from s2and.consts import LARGE_INTEGER
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer


def _dummy_dataset(name: str) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        load_name_counts=False,
        n_jobs=1,
    )


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
        _rust_failure_counts,
        runtime_context=None,
    ):
        del runtime_context
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
        _rust_failure_counts,
        runtime_context=None,
    ):
        del runtime_context
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
        _rust_failure_counts,
        runtime_context=None,
    ):
        del labels, runtime_context
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

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.data import ANDData
from s2and.feature_port import _get_rust_featurizer
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from s2and.prediction_state import PredictionState
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset, import_s2and_rust

HAS_RUST, _RUST_IMPORT_PAYLOAD = import_s2and_rust()
if not HAS_RUST:
    raise pytest.skip.Exception(
        f"s2and_rust extension not built/installed: {_RUST_IMPORT_PAYLOAD}",
        allow_module_level=True,
    )


class _DeterministicClassifier:
    """Return feature-sensitive, deterministic class probabilities."""

    def predict_proba(self, features: object) -> NDArray[np.float64]:
        matrix = np.asarray(features, dtype=np.float64)
        weights = np.linspace(0.1, 0.6, matrix.shape[1], dtype=np.float64)
        scores = np.nan_to_num(matrix, nan=0.0, posinf=1_000.0, neginf=-1_000.0) @ weights
        distances = 0.5 + 0.25 * np.tanh(scores / 10.0)
        return np.column_stack((distances, 1.0 - distances))


class _ConstantDistanceClassifier:
    """Return a precise distance independent of feature values or batch size."""

    def __init__(self, distance: float) -> None:
        self.distance = distance

    def predict_proba(self, features: object) -> NDArray[np.float64]:
        distances = np.full(len(np.asarray(features)), self.distance, dtype=np.float64)
        return np.column_stack((distances, 1.0 - distances))


@pytest.fixture(scope="module")
def parity_datasets(tmp_path_factory: pytest.TempPathFactory) -> tuple[ANDData, ANDData]:
    """Build matching Python and Arrow-backed Rust dummy datasets."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("S2AND_BACKEND", "python")
        python_dataset = build_dummy_dataset(
            "distance_matrix_python_parity",
            name_counts_index=True,
        )
        rust_dataset = build_arrow_training_dataset(
            python_dataset,
            tmp_path_factory.mktemp("distance_matrix_rust_parity"),
        )
    cluster_seeds_require = {"3": "seed-a", "4": "seed-a", "5": "seed-b"}
    cluster_seeds_disallow = {("5", "6")}
    for dataset in (python_dataset, rust_dataset):
        dataset.cluster_seeds_require = dict(cluster_seeds_require)
        dataset.cluster_seeds_disallow = set(cluster_seeds_disallow)
    return python_dataset, rust_dataset


def _clusterer(
    *,
    fastcluster: bool,
    batch_size: int,
    use_default_constraints_as_supervision: bool = False,
) -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=_DeterministicClassifier(),
        cluster_model=FastCluster(linkage="average") if fastcluster else object(),
        n_jobs=1,
        use_default_constraints_as_supervision=use_default_constraints_as_supervision,
        batch_size=batch_size,
    )


@pytest.mark.parametrize("pair_chunk_size", [1, 6], ids=["one-pair", "whole-block"])
def test_python_and_rust_square_distance_matrices_and_constraints_match(
    parity_datasets: tuple[ANDData, ANDData],
    pair_chunk_size: int,
) -> None:
    python_dataset, rust_dataset = parity_datasets
    block = {"mixed": ["3", "4", "5", "6"]}

    python_matrix = _clusterer(
        fastcluster=False,
        batch_size=pair_chunk_size,
        use_default_constraints_as_supervision=True,
    ).make_distance_matrices(
        block,
        python_dataset,
        disable_tqdm=True,
    )["mixed"]
    rust_entry_matrix = _clusterer(
        fastcluster=False,
        batch_size=pair_chunk_size,
        use_default_constraints_as_supervision=True,
    ).make_distance_matrices(
        block,
        rust_dataset,
        disable_tqdm=True,
    )["mixed"]
    rust_direct_matrix = _clusterer(
        fastcluster=False,
        batch_size=pair_chunk_size,
        use_default_constraints_as_supervision=True,
    ).make_distance_matrices_from_rust_featurizer(
        block,
        _get_rust_featurizer(rust_dataset),
        pair_chunk_size=pair_chunk_size,
    )["mixed"]

    np.testing.assert_allclose(rust_entry_matrix, python_matrix, rtol=0.0, atol=np.finfo(np.float16).eps)
    np.testing.assert_array_equal(rust_direct_matrix, rust_entry_matrix)
    assert rust_entry_matrix[0, 1] == 0.0  # same required seed
    assert rust_entry_matrix[0, 2] == LARGE_DISTANCE  # different required seeds
    assert rust_entry_matrix[2, 3] == LARGE_DISTANCE  # explicit disallow
    assert 0.0 < rust_entry_matrix[0, 3] < 1.0  # unconstrained classifier score


def test_rust_fastcluster_condensed_order_overrides_and_chunk_telemetry(
    parity_datasets: tuple[ANDData, ANDData],
) -> None:
    prediction_state = PredictionState()
    _, rust_dataset = parity_datasets
    signatures = ["0", "1", "2", "3"]
    block = {"mixed": signatures}
    rust_featurizer: Any = _get_rust_featurizer(rust_dataset)

    square_matrix = _clusterer(
        fastcluster=False,
        batch_size=6,
    ).make_distance_matrices_from_rust_featurizer(
        block,
        rust_featurizer,
        pair_chunk_size=6,
    )["mixed"]
    expected = np.asarray(square_matrix[np.triu_indices(len(signatures), k=1)], dtype=np.float64)
    expected[1] = 0.125  # (0, 2)
    expected[4] = 0.875  # (1, 3), supplied below in reverse order

    clusterer = _clusterer(fastcluster=True, batch_size=6)
    condensed = clusterer._make_distance_matrices_from_verified_rust_featurizer(
        block,
        rust_featurizer,
        partial_supervision={("0", "2"): 0.125, ("3", "1"): 0.875},
        pair_chunk_size=2,
        prediction_state=prediction_state,
    )["mixed"]

    np.testing.assert_allclose(condensed, expected, rtol=0.0, atol=np.finfo(np.float16).eps)
    telemetry = prediction_state.telemetry["rust_featurizer_make_dists"]
    assert telemetry["block_count"] == 1
    assert telemetry["pair_count"] == 6
    assert telemetry["chunk_count"] == 3
    assert telemetry["resolved_pair_chunk_size"] == 2
    assert telemetry["upper_triangle_index_seconds"] == 0.0


def test_predict_from_arrow_filters_real_rust_featurizer(
    parity_datasets: tuple[ANDData, ANDData],
) -> None:
    prediction_state = PredictionState()
    _, rust_dataset = parity_datasets
    signatures = ["0", "1", "2", "3"]
    block = {"mixed": signatures}
    assert not rust_dataset.arrow_dataset.has("specter")

    clusterer = _clusterer(fastcluster=True, batch_size=2)
    predicted_clusters, dists = clusterer._predict_from_arrow_request(
        block,
        rust_dataset.arrow_dataset,
        prediction_state=prediction_state,
    )

    predicted_ids = [signature_id for members in predicted_clusters.values() for signature_id in members]
    assert sorted(predicted_ids) == sorted(signatures)
    assert dists is None
    telemetry = prediction_state.telemetry["arrow_predict"]
    assert telemetry["signature_count"] == 4
    assert telemetry["featurizer_signature_count"] == 4
    assert telemetry["pair_count"] == 6


@pytest.mark.parametrize("pair_chunk_size", [1, 3], ids=["one-pair", "whole-block"])
@pytest.mark.parametrize("distance", [0.4999, 0.5, 0.5001])
def test_fastcluster_stored_and_streaming_precision_at_threshold(
    parity_datasets: tuple[ANDData, ANDData],
    pair_chunk_size: int,
    distance: float,
) -> None:
    """Keep classifier distances and threshold decisions exact across backends."""
    _assert_fastcluster_precision_parity(
        parity_datasets,
        pair_chunk_size=pair_chunk_size,
        distance=distance,
        partial_supervision={},
        expected_distances=[distance, distance, distance],
        expected_clusters={frozenset({"0", "1", "2"})}
        if distance <= 0.5
        else {frozenset({"0"}), frozenset({"1"}), frozenset({"2"})},
    )


@pytest.mark.parametrize("pair_chunk_size", [1, 3], ids=["one-pair", "whole-block"])
def test_fastcluster_precision_preserves_merge_order(
    parity_datasets: tuple[ANDData, ANDData],
    pair_chunk_size: int,
) -> None:
    """Avoid a float16 tie that merges the wrong first pair at the threshold."""
    _assert_fastcluster_precision_parity(
        parity_datasets,
        pair_chunk_size=pair_chunk_size,
        distance=0.9,
        partial_supervision={("0", "1"): 0.5001, ("0", "2"): 0.4999},
        # Supervised values pass through the existing negative-label encoding.
        expected_distances=[0.5001 - LARGE_INTEGER + LARGE_INTEGER, 0.4999 - LARGE_INTEGER + LARGE_INTEGER, 0.9],
        expected_clusters={frozenset({"0", "2"}), frozenset({"1"})},
    )


def _assert_fastcluster_precision_parity(
    parity_datasets: tuple[ANDData, ANDData],
    *,
    pair_chunk_size: int,
    distance: float,
    partial_supervision: dict[tuple[str, str], float],
    expected_distances: list[float],
    expected_clusters: set[frozenset[str]],
) -> None:
    """Exercise stored matrices and streaming prediction with real featurizers."""
    python_dataset, rust_dataset = parity_datasets
    block = {"threshold": ["0", "1", "2"]}
    clusterer = _clusterer(fastcluster=True, batch_size=pair_chunk_size)
    clusterer.classifier = _ConstantDistanceClassifier(distance)
    rust_featurizer = _get_rust_featurizer(rust_dataset)
    matrices = [
        clusterer.make_distance_matrices(block, dataset, partial_supervision=partial_supervision, disable_tqdm=True)
        for dataset in (python_dataset, rust_dataset)
    ]
    matrices.append(
        clusterer.make_distance_matrices_from_rust_featurizer(
            block,
            rust_featurizer,
            partial_supervision=partial_supervision,
            pair_chunk_size=pair_chunk_size,
        )
    )
    for stored in matrices:
        assert stored["threshold"].dtype == np.float64
        np.testing.assert_array_equal(stored["threshold"], expected_distances)

    params = {"eps": 0.5}
    for stored in matrices:
        for clusters, _ in (
            clusterer.predict(block, python_dataset, dists=stored, cluster_model_params=params),
            clusterer.predict_from_rust_featurizer(
                block, rust_featurizer, dists=stored, cluster_model_params=params, cluster_seeds_require={}
            ),
        ):
            assert {frozenset(members) for members in clusters.values()} == expected_clusters
    for clusters, returned_distances in (
        clusterer.predict(block, python_dataset, cluster_model_params=params, partial_supervision=partial_supervision),
        clusterer.predict_from_rust_featurizer(
            block,
            rust_featurizer,
            cluster_model_params=params,
            partial_supervision=partial_supervision,
            cluster_seeds_require={},
        ),
    ):
        assert {frozenset(members) for members in clusters.values()} == expected_clusters
        assert returned_distances is None

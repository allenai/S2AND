"""Check exact bulk label bits across native constraints and supervision."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

import s2and.model as model_module
from s2and.consts import LARGE_INTEGER
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster, distances


# Signaling NaN inputs raise NumPy's invalid-value flag; real constraint values are finite.
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
@pytest.mark.parametrize(
    ("chunk_size", "constraints_enabled", "overrides_enabled", "fastcluster"),
    [
        (7, False, False, True),
        (7, False, True, True),
        (7, True, False, True),
        (7, True, True, True),
        (10000, True, True, False),
    ],
)
def test_bulk_labels_preserve_scalar_bits(
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int,
    constraints_enabled: bool,
    overrides_enabled: bool,
    fastcluster: bool,
) -> None:
    """Preserve float subtraction, missing values, and directional precedence."""
    signatures = [str(index) for index in range(12)]
    pairs = [(left, right) for index, left in enumerate(signatures) for right in signatures[index + 1 :]]
    rng = np.random.default_rng(917)
    values: list[float | None] = rng.integers(0, 2**64, size=len(pairs), dtype=np.uint64).view(np.float64).tolist()
    values[:11] = [None, 0.0, -0.0, float(LARGE_INTEGER), np.inf, -np.inf, np.nan, 1e-300, 1e300, None, 0.5]
    values[11:15] = (
        np.asarray(
            [0x7FF0000000000001, 0x7FF8000000000042, 0xFFF0000000000001, 0xFFF8000000000042],
            dtype=np.uint64,
        )
        .view(np.float64)
        .tolist()
    )
    supervision: dict[tuple[str, str], Any] = {}
    if overrides_enabled:
        supervision = {
            pairs[0]: 0.25,
            pairs[0][::-1]: 0.75,
            pairs[1][::-1]: -0.0,
            pairs[2]: np.nan,
            pairs[3][::-1]: np.inf,
            pairs[4]: -np.inf,
        }
    expected = np.full(len(pairs), np.nan, dtype=np.float64)
    for index, pair in enumerate(pairs):
        override = supervision.get(pair)
        if override is None:
            override = supervision.get(pair[::-1])
        if override is not None:
            expected[index] = float(override - LARGE_INTEGER)
        elif constraints_enabled and values[index] is not None:
            expected[index] = float(values[index] - LARGE_INTEGER)

    observed: list[np.ndarray] = []

    def constraints(_indices: list[int], *, start_offset: int, max_pairs: int, **kwargs: Any):
        assert kwargs["num_threads"] == 1
        left, right = model_module._upper_triangle_indices_for_range(len(signatures), start_offset, max_pairs)
        return left.tolist(), right.tolist(), values[start_offset : start_offset + max_pairs]

    def features(_indices: list[int], *, max_pairs: int, **_kwargs: Any):
        return np.zeros((max_pairs, 1)), None

    def predict(_classifier: Any, _nameless: Any, _features: Any, labels: np.ndarray, *_args: Any, **_kwargs: Any):
        observed.append(labels.copy())
        return np.zeros(len(labels)), 0.0

    def constraint_labels(*args: Any, large_integer: float, **kwargs: Any) -> np.ndarray:
        # Mirror the native label arithmetic so this test checks chunking and override order.
        chunk_values = constraints(*args, **kwargs)[2]
        return np.asarray([np.nan if value is None else value - large_integer for value in chunk_values])

    monkeypatch.setattr(distances, "get_constraints_block_upper_triangle_indexed_rust", constraints)
    monkeypatch.setattr(distances, "_get_constraint_labels_block_upper_triangle_indexed_rust", constraint_labels)
    monkeypatch.setattr(distances, "_build_block_feature_matrices_indexed_rust", features)
    monkeypatch.setattr(distances, "_predict_and_combine", predict)
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        cluster_model=FastCluster(linkage="average") if fastcluster else DBSCAN(),
        n_jobs=1,
        batch_size=chunk_size,
        use_default_constraints_as_supervision=constraints_enabled,
    )
    clusterer._make_distance_matrices_from_verified_rust_featurizer(
        {"block": signatures},
        SimpleNamespace(),
        signature_index_by_id={value: index for index, value in enumerate(signatures)},
        partial_supervision=supervision,
    )
    np.testing.assert_array_equal(np.concatenate(observed).view(np.uint64), expected.view(np.uint64))

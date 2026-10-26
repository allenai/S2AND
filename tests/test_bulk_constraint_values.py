"""Verify the coordinate-free constraint label path against the public native API."""

from typing import Any

import numpy as np
import pytest

from s2and.consts import LARGE_INTEGER
from s2and.rust_calls import _get_constraint_labels_block_upper_triangle_indexed_rust


def _labels_from_values(values: list[float | None]) -> np.ndarray:
    """Apply the Python label arithmetic that the native labels must match bit for bit."""
    return np.asarray([np.nan if value is None else float(value - LARGE_INTEGER) for value in values], dtype=np.float64)


@pytest.fixture(scope="module")
def native_constraints(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Create a real Arrow-backed featurizer once for the parity cases."""
    from s2and.feature_port import _get_rust_featurizer
    from tests.helpers import build_arrow_training_dataset, build_dummy_dataset

    dataset = build_dummy_dataset("bulk-constraint-values", name_counts_index=True)
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path_factory.mktemp("constraint-values"))
    return _get_rust_featurizer(arrow_dataset)


@pytest.mark.parametrize(
    ("start_offset", "max_pairs", "flags"),
    [(0, None, 7), (0, 0, 7), (1, 1, 7), (2, 3, 7), (10, 7, 7), (0, None, 0)],
)
def test_labels_match_values_bit_for_bit(
    native_constraints: Any, start_offset: int, max_pairs: int | None, flags: int
) -> None:
    """Preserve ordering, duplicate indices, chunk boundaries and label bits."""
    native = native_constraints
    indices = list(range(min(4, len(native.signature_ids()))))[::-1]
    indices.append(indices[0])
    kwargs = {
        "start_offset": start_offset,
        "max_pairs": max_pairs,
        "low_value": -0.0,
        "high_value": float("inf"),
        "dont_merge_cluster_seeds": bool(flags & 1),
        "incremental_dont_use_cluster_seeds": bool(flags & 2),
        "suppress_orcid": bool(flags & 4),
        "num_threads": 10,
    }
    values = native.get_constraints_block_upper_triangle_indexed(indices, **kwargs)[2]
    actual = _get_constraint_labels_block_upper_triangle_indexed_rust(
        indices, featurizer=native, large_integer=LARGE_INTEGER, **kwargs
    )
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float64
    assert actual.flags.writeable
    np.testing.assert_array_equal(actual.view(np.uint64), _labels_from_values(values).view(np.uint64))


def test_labels_cover_both_constraint_values(native_constraints: Any) -> None:
    """Exercise the real production constraint values, not only missing ones."""
    native = native_constraints
    indices = list(range(len(native.signature_ids())))
    values = native.get_constraints_block_upper_triangle_indexed(indices, num_threads=10)[2]
    assert any(value is not None for value in values), "the fixture must produce constraint values"
    actual = _get_constraint_labels_block_upper_triangle_indexed_rust(
        indices, featurizer=native, large_integer=LARGE_INTEGER, num_threads=10
    )
    np.testing.assert_array_equal(actual.view(np.uint64), _labels_from_values(values).view(np.uint64))


@pytest.mark.parametrize(
    ("indices", "start_offset"),
    [(indices, 0) for indices in ([], [0], [999999], [0, 1], [0, 999999], [-1, 0], [2**32, 0])]
    + [([], -1), ([0, 1], -1), ([0, 1], 99)],
)
def test_labels_preserve_boundary_results_and_errors(
    native_constraints: Any, indices: list[int], start_offset: int
) -> None:
    """Keep extraction errors and even the existing singleton early return."""
    native = native_constraints
    try:
        values = native.get_constraints_block_upper_triangle_indexed(indices, start_offset=start_offset)[2]
    except (IndexError, OverflowError, ValueError) as error:
        with pytest.raises(type(error)) as caught:
            native._constraint_labels_block_upper_triangle_indexed(
                indices, float(LARGE_INTEGER), start_offset=start_offset
            )
        assert str(caught.value) == str(error)
    else:
        actual = native._constraint_labels_block_upper_triangle_indexed(
            indices, float(LARGE_INTEGER), start_offset=start_offset
        )
        np.testing.assert_array_equal(actual.view(np.uint64), _labels_from_values(values).view(np.uint64))

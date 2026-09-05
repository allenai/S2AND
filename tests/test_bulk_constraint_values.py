"""Verify the coordinate-free constraint path against the public native API."""

from typing import Any

import numpy as np
import pytest

from s2and.rust_calls import _get_constraints_block_upper_triangle_values_indexed_rust


@pytest.fixture(scope="module")
def native_constraints(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Create a real Arrow-backed featurizer once for the parity cases."""
    from s2and.feature_port import _get_rust_featurizer
    from tests.helpers import build_arrow_training_dataset, build_dummy_dataset

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("S2AND_BACKEND", "python")
        dataset = build_dummy_dataset("bulk-constraint-values", name_counts_index=True)
        arrow_dataset = build_arrow_training_dataset(dataset, tmp_path_factory.mktemp("constraint-values"))
        return _get_rust_featurizer(arrow_dataset)


@pytest.mark.parametrize("start_offset,max_pairs", [(0, None), (0, 0), (1, 1), (2, 3), (10, 7)])
@pytest.mark.parametrize("flags", range(8))
def test_values_only_preserves_optional_float_bits(
    native_constraints: Any, start_offset: int, max_pairs: int | None, flags: int
) -> None:
    """Preserve ordering, duplicate indices, chunk boundaries and all rule flags."""
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
    expected = native.get_constraints_block_upper_triangle_indexed(indices, **kwargs)[2]
    actual = _get_constraints_block_upper_triangle_values_indexed_rust(indices, featurizer=native, **kwargs)
    assert type(actual) is list
    assert [value is None for value in actual] == [value is None for value in expected]
    assert all(value is None or type(value) is float for value in actual)
    np.testing.assert_array_equal(
        np.asarray([value for value in actual if value is not None], dtype=np.float64).view(np.uint64),
        np.asarray([value for value in expected if value is not None], dtype=np.float64).view(np.uint64),
    )


@pytest.mark.parametrize("indices", [[], [0], [999999], [0, 1], [0, 999999], [-1, 0], [2**32, 0]])
@pytest.mark.parametrize("start_offset", [0, -1, 99])
def test_values_only_preserves_boundary_results_and_errors(
    native_constraints: Any, indices: list[int], start_offset: int
) -> None:
    """Keep extraction errors and even the existing singleton early return."""
    native = native_constraints
    try:
        expected = native.get_constraints_block_upper_triangle_indexed(indices, start_offset=start_offset)[2]
    except (IndexError, OverflowError, ValueError) as error:
        with pytest.raises(type(error)) as caught:
            native._get_constraints_block_upper_triangle_values_indexed(indices, start_offset=start_offset)
        assert str(caught.value) == str(error)
    else:
        assert (
            native._get_constraints_block_upper_triangle_values_indexed(indices, start_offset=start_offset) == expected
        )

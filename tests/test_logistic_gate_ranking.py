"""Array ranking of gate query rows against the loops it replaces."""

from __future__ import annotations

import numpy as np
import pytest

from s2and.incremental_linking import logistic_gate


def _loop_ranking(query_array, probability_array, rank_array, component_keys):
    groups = logistic_gate._ordered_group_indices(query_array)  # noqa: SLF001
    ranked = tuple(
        logistic_gate._sort_group(  # noqa: SLF001
            group,
            probabilities=probability_array,
            retrieval_ranks=rank_array,
            component_keys=component_keys,
        )
        for group in groups
    )
    return groups, ranked


@pytest.mark.parametrize("seed", range(40))
def test_array_ranking_matches_the_loops(seed: int) -> None:
    rng = np.random.default_rng(seed)
    row_count = int(rng.integers(1, 400))
    query_array = rng.integers(0, max(1, row_count // 3), size=row_count).astype(
        [np.uint32, np.int64, np.int32][seed % 3]
    )
    # Coarse probabilities make ties that the rank and key must break.
    probability_array = np.round(rng.random(row_count), int(rng.integers(1, 4)))
    probability_array[rng.random(row_count) < 0.05] = 0.0
    if seed % 4 == 0:
        rank_array = None
    elif seed % 4 == 1:
        rank_array = rng.integers(1, 30, size=row_count).astype(np.uint16)
    else:
        rank_array = rng.integers(-5, 30, size=row_count) + np.round(rng.random(row_count), 1)
    component_keys = (
        None
        if seed % 5 == 0
        else np.asarray([f"p_{int(value)}" for value in rng.integers(0, 50, size=row_count)], dtype=object)
    )
    array_groups = logistic_gate._array_ranked_groups(  # noqa: SLF001
        query_array, probability_array, rank_array, component_keys
    )
    assert array_groups is not None
    expected_groups, expected_ranked = _loop_ranking(query_array, probability_array, rank_array, component_keys)
    groups, ranked = array_groups
    assert [group.tolist() for group in groups] == [group.tolist() for group in expected_groups]
    assert [group.tolist() for group in ranked] == [group.tolist() for group in expected_ranked]
    assert all(group.dtype == np.int64 for group in (*groups, *ranked))


def test_array_ranking_leaves_non_finite_and_object_inputs_to_the_loops() -> None:
    queries = np.asarray([0, 0, 1], dtype=np.int64)
    probabilities = np.asarray([0.5, 0.25, 0.75])
    assert logistic_gate._array_ranked_groups(queries, np.asarray([0.5, np.nan, 0.1]), None, None) is None  # noqa: SLF001
    assert logistic_gate._array_ranked_groups(queries, probabilities, np.asarray([1.0, np.inf, 2.0]), None) is None  # noqa: SLF001
    assert logistic_gate._array_ranked_groups(queries.astype(object), probabilities, None, None) is None  # noqa: SLF001
    rows = logistic_gate.ranked_query_rows(queries.astype(object), probabilities)
    assert [group.tolist() for group in rows.ranked_groups] == [[0, 1], [2]]

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from s2and.incremental_linking import features
from s2and.incremental_linking.linker_pairwise import (
    LinkerCandidateBatch,
    promoted_pairwise_aggregate_columns,
)
from s2and.incremental_linking.logistic_gate import feature_values_from_runtime


class StaticPairwiseStats:
    def __init__(
        self,
        matrix: np.ndarray,
        columns: tuple[str, ...],
    ) -> None:
        self._matrix = np.asarray(matrix, dtype=np.float32)
        self.aggregate_feature_columns = columns
        self.feature_matrix_call_count = 0

    def feature_matrix(self) -> np.ndarray:
        self.feature_matrix_call_count += 1
        return self._matrix


def _static_pairwise_stats(matrix: np.ndarray, columns: tuple[str, ...]) -> Any:
    return StaticPairwiseStats(matrix, columns)


def _row_feature_fixture(row_count: int) -> dict[str, np.ndarray]:
    return {
        column: np.linspace(0.1, 0.9, row_count, dtype=np.float32) + np.float32(column_index)
        for column_index, column in enumerate(features.PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS)
    }


def test_assemble_linker_feature_matrix_rejects_nonfinite_pairwise_values() -> None:
    row_count = 2
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
    )
    pairwise_columns = promoted_pairwise_aggregate_columns()
    pairwise_matrix = np.zeros((row_count, len(pairwise_columns)), dtype=np.float32)
    column = pairwise_columns.index("pw_mean_middle_names_equal")
    row_features = _row_feature_fixture(row_count)
    stats = _static_pairwise_stats(pairwise_matrix, pairwise_columns)

    pairwise_matrix[0, column] = np.nan
    with pytest.raises(ValueError, match="NaN values"):
        features.assemble_linker_feature_matrix(candidate_batch, row_features, pairwise_stats=stats)

    pairwise_matrix[0, column] = np.inf
    with pytest.raises(ValueError, match="infinite values"):
        features.assemble_linker_feature_matrix(candidate_batch, row_features, pairwise_stats=stats)


def test_assemble_linker_feature_matrix_matches_tracked_target_order() -> None:
    target_path = Path("tests/fixtures/incremental_linker_training_target.json")
    target_columns = tuple(json.loads(target_path.read_text(encoding="utf-8"))["features"])
    row_count = 4
    pairwise_columns = promoted_pairwise_aggregate_columns()
    row_offsets = np.arange(row_count, dtype=np.float32) * np.float32(1000)
    row_features = {
        column: row_offsets + np.float32(column_index + 0.25)
        for column_index, column in enumerate(target_columns)
        if not column.startswith("pw_")
    }
    pairwise_features = {
        column: row_offsets + np.float32(column_index + 100.25) for column_index, column in enumerate(pairwise_columns)
    }
    pairwise_matrix = np.column_stack([pairwise_features[column] for column in pairwise_columns])
    original_pairwise_matrix = pairwise_matrix.copy()
    original_row_features = {column: values.copy() for column, values in row_features.items()}
    expected = np.column_stack(
        [pairwise_features[column] if column.startswith("pw_") else row_features[column] for column in target_columns]
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
    )

    assembled = features.assemble_linker_feature_matrix(
        candidate_batch,
        row_features,
        pairwise_stats=_static_pairwise_stats(pairwise_matrix, pairwise_columns),
    )

    assert not any(column.endswith("_rank_fraction") for column in assembled.feature_columns)
    assert assembled.matrix.dtype == np.float32
    assert assembled.matrix.shape == (row_count, len(target_columns))
    assert assembled.feature_columns == target_columns == features.promoted_linker_feature_columns()
    np.testing.assert_array_equal(assembled.matrix, expected)
    np.testing.assert_array_equal(pairwise_matrix, original_pairwise_matrix)
    assert row_features.keys() == original_row_features.keys()
    for column, values in row_features.items():
        np.testing.assert_array_equal(values, original_row_features[column])


def test_feature_values_from_runtime_prefers_assembled_pairwise_and_live_row_signals() -> None:
    row_count = 2
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
    )
    pairwise_columns = promoted_pairwise_aggregate_columns()
    pairwise_column = "pw_mean_specter_cosine_sim"
    pairwise_index = pairwise_columns.index(pairwise_column)
    pairwise_matrix = np.zeros((row_count, len(pairwise_columns)), dtype=np.float32)
    pairwise_matrix[:, pairwise_index] = 1.0
    pairwise_stats = StaticPairwiseStats(pairwise_matrix, pairwise_columns)
    assembled = features.assemble_linker_feature_matrix(
        candidate_batch,
        _row_feature_fixture(row_count),
        pairwise_stats=cast(Any, pairwise_stats),
        feature_columns=features.promoted_linker_feature_columns(),
    )
    assert pairwise_stats.feature_matrix_call_count == 1

    # A changed overlay must never replace the pairwise values the artifact scored.
    pairwise_matrix[:, pairwise_index] = 3.0
    override = np.asarray([42.0, 99.0], dtype=np.float32)
    values = feature_values_from_runtime(
        assembled, {pairwise_column: np.asarray([2.0, 2.0], dtype=np.float32), "min_distance": override}
    )

    assert pairwise_stats.feature_matrix_call_count == 1
    np.testing.assert_array_equal(values[pairwise_column], [1.0, 1.0])
    np.testing.assert_array_equal(values["min_distance"], override)

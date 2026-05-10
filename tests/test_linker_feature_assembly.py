from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2and.incremental_linking import features
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch, promoted_pairwise_aggregate_columns


class StaticPairwiseStats:
    def __init__(self, matrix: np.ndarray, columns: tuple[str, ...]) -> None:
        self._matrix = np.asarray(matrix, dtype=np.float32)
        self.aggregate_feature_columns = columns

    def feature_matrix(self) -> np.ndarray:
        return self._matrix


def _row_feature_fixture(row_count: int) -> dict[str, np.ndarray]:
    return {
        column: np.linspace(0.1, 0.9, row_count, dtype=np.float32) + np.float32(column_index)
        for column_index, column in enumerate(features.PROMOTED_NON_PAIRWISE_FEATURE_COLUMNS)
    }


def test_promoted_linker_feature_columns_match_promoted_target_file() -> None:
    target_path = Path("s2and/data/production_incremental_linker_v1.2/training_target.json")
    target = json.loads(target_path.read_text(encoding="utf-8"))

    assert features.promoted_linker_feature_columns() == tuple(target["features"])


def test_promoted_linker_feature_columns_are_promoted_70_without_rank_fractions() -> None:
    promoted = features.promoted_linker_feature_columns()

    assert len(promoted) == 70
    assert not any(column.endswith("_rank_fraction") for column in promoted)
    assert not features.PROMOTED_LINKER_DROPPED_FEATURE_COLUMNS.intersection(promoted)


def test_assemble_linker_feature_matrix_places_row_and_pairwise_columns() -> None:
    row_count = 3
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
    )
    pairwise_columns = promoted_pairwise_aggregate_columns()
    pairwise_matrix = np.arange(row_count * len(pairwise_columns), dtype=np.float32).reshape(
        row_count, len(pairwise_columns)
    )
    assembled = features.assemble_linker_feature_matrix(
        candidate_batch,
        _row_feature_fixture(row_count),
        pairwise_stats=StaticPairwiseStats(pairwise_matrix, pairwise_columns),
    )

    assert assembled.matrix.dtype == np.float32
    assert assembled.matrix.shape == (row_count, len(features.promoted_linker_feature_columns()))
    assert assembled.feature_columns == features.promoted_linker_feature_columns()
    np.testing.assert_array_equal(
        assembled.matrix[:, assembled.feature_columns.index("min_distance")],
        np.asarray(_row_feature_fixture(row_count)["min_distance"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        assembled.matrix[:, assembled.feature_columns.index("pw_mean_first_names_equal")],
        pairwise_matrix[:, pairwise_columns.index("pw_mean_first_names_equal")],
    )


def test_assemble_linker_feature_matrix_allows_pairwise_nans() -> None:
    row_count = 2
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
    )
    pairwise_columns = promoted_pairwise_aggregate_columns()
    pairwise_matrix = np.zeros((row_count, len(pairwise_columns)), dtype=np.float32)
    pairwise_matrix[0, pairwise_columns.index("pw_mean_middle_names_equal")] = np.nan

    assembled = features.assemble_linker_feature_matrix(
        candidate_batch,
        _row_feature_fixture(row_count),
        pairwise_stats=StaticPairwiseStats(pairwise_matrix, pairwise_columns),
    )

    assert np.isnan(assembled.matrix[0, assembled.feature_columns.index("pw_mean_middle_names_equal")])


def test_assemble_linker_feature_matrix_rejects_pairwise_infinities() -> None:
    row_count = 1
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
    )
    pairwise_columns = promoted_pairwise_aggregate_columns()
    pairwise_matrix = np.zeros((row_count, len(pairwise_columns)), dtype=np.float32)
    pairwise_matrix[0, pairwise_columns.index("pw_mean_middle_names_equal")] = np.inf

    with pytest.raises(ValueError, match="infinite values"):
        features.assemble_linker_feature_matrix(
            candidate_batch,
            _row_feature_fixture(row_count),
            pairwise_stats=StaticPairwiseStats(pairwise_matrix, pairwise_columns),
        )


def test_assemble_linker_feature_matrix_matches_tracked_target_order() -> None:
    target_path = Path("s2and/data/production_incremental_linker_v1.2/training_target.json")
    target_columns = tuple(json.loads(target_path.read_text(encoding="utf-8"))["features"])
    row_count = 4
    pairwise_columns = promoted_pairwise_aggregate_columns()
    frame = pd.DataFrame(
        {
            column: np.full(row_count, column_index + 0.25, dtype=np.float32)
            for column_index, column in enumerate(target_columns)
            if not column.startswith("pw_")
        }
    )
    pairwise_frame = pd.DataFrame(
        {
            column: np.full(row_count, column_index + 100.25, dtype=np.float32)
            for column_index, column in enumerate(pairwise_columns)
        }
    )
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
    )

    assembled = features.assemble_linker_feature_matrix(
        candidate_batch,
        frame,
        pairwise_stats=StaticPairwiseStats(
            pairwise_frame.loc[:, list(pairwise_columns)].to_numpy(np.float32),
            pairwise_columns,
        ),
        feature_columns=target_columns,
    )
    expected = pd.concat([frame, pairwise_frame], axis=1).loc[:, list(target_columns)].to_numpy(np.float32)

    np.testing.assert_array_equal(assembled.matrix, expected)

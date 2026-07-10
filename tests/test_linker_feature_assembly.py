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


def test_promoted_linker_feature_columns_match_promoted_target_file() -> None:
    target_path = Path("s2and/data/production_model_v1.21/reproducibility/incremental_linker_training_target.json")
    target = json.loads(target_path.read_text(encoding="utf-8"))

    assert features.promoted_linker_feature_columns() == tuple(target["features"])


def test_promoted_linker_feature_columns_are_promoted_53_without_rank_fractions() -> None:
    promoted = features.promoted_linker_feature_columns()

    assert len(promoted) == 53
    assert not any(column.endswith("_rank_fraction") for column in promoted)


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
        pairwise_stats=_static_pairwise_stats(pairwise_matrix, pairwise_columns),
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


def test_assemble_linker_feature_matrix_rejects_pairwise_nans() -> None:
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

    with pytest.raises(ValueError, match="NaN values"):
        features.assemble_linker_feature_matrix(
            candidate_batch,
            _row_feature_fixture(row_count),
            pairwise_stats=_static_pairwise_stats(pairwise_matrix, pairwise_columns),
        )


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
            pairwise_stats=_static_pairwise_stats(pairwise_matrix, pairwise_columns),
        )


def test_assemble_linker_feature_matrix_matches_tracked_target_order() -> None:
    target_path = Path("s2and/data/production_model_v1.21/reproducibility/incremental_linker_training_target.json")
    target_columns = tuple(json.loads(target_path.read_text(encoding="utf-8"))["features"])
    row_count = 4
    pairwise_columns = promoted_pairwise_aggregate_columns()
    row_features = {
        column: np.full(row_count, column_index + 0.25, dtype=np.float32)
        for column_index, column in enumerate(target_columns)
        if not column.startswith("pw_")
    }
    pairwise_features = {
        column: np.full(row_count, column_index + 100.25, dtype=np.float32)
        for column_index, column in enumerate(pairwise_columns)
    }
    pairwise_matrix = np.column_stack([pairwise_features[column] for column in pairwise_columns])
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
        feature_columns=target_columns,
    )
    expected = np.column_stack(
        [pairwise_features[column] if column.startswith("pw_") else row_features[column] for column in target_columns]
    )

    np.testing.assert_array_equal(assembled.matrix, expected)


def test_feature_values_from_runtime_reuses_assembled_pairwise_columns() -> None:
    row_count = 3
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
        retrieval_scores=np.asarray([0.3, 0.2, 0.1], dtype=np.float32),
        retrieval_ranks=np.asarray([1, 2, 3], dtype=np.uint16),
    )
    pairwise_columns = promoted_pairwise_aggregate_columns()
    pairwise_matrix = np.arange(row_count * len(pairwise_columns), dtype=np.float32).reshape(
        row_count, len(pairwise_columns)
    )
    pairwise_stats = StaticPairwiseStats(pairwise_matrix, pairwise_columns)
    pairwise_stats_for_call: Any = pairwise_stats
    assembled = features.assemble_linker_feature_matrix(
        candidate_batch,
        _row_feature_fixture(row_count),
        pairwise_stats=cast(Any, pairwise_stats_for_call),
    )

    assert pairwise_stats.feature_matrix_call_count == 1
    values = feature_values_from_runtime(assembled, None)

    assert pairwise_stats.feature_matrix_call_count == 1
    np.testing.assert_array_equal(
        values["pw_mean_first_names_equal"],
        assembled.matrix[:, assembled.feature_columns.index("pw_mean_first_names_equal")],
    )


def test_feature_values_from_runtime_keeps_pairwise_columns_over_row_signals() -> None:
    row_count = 2
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
    pairwise_stats = StaticPairwiseStats(pairwise_matrix, pairwise_columns)
    assembled = features.assemble_linker_feature_matrix(
        candidate_batch,
        _row_feature_fixture(row_count),
        pairwise_stats=cast(Any, pairwise_stats),
    )

    values = feature_values_from_runtime(
        assembled,
        {"pw_mean_first_names_equal": np.asarray([999.0, 999.0], dtype=np.float32)},
    )

    np.testing.assert_array_equal(
        values["pw_mean_first_names_equal"],
        assembled.matrix[:, assembled.feature_columns.index("pw_mean_first_names_equal")],
    )


def test_feature_values_from_runtime_precedence_three_way_overlap() -> None:
    """Lock the column-precedence contract when matrix, row_signals, and pairwise_stats all
    carry the same column name.

    Contract (validated against production data flow):
      - ``feature_matrix.matrix`` is the artifact-ordered, schema-validated source that the
        LightGBM artifact consumed. For any column present in ``feature_columns`` it MUST
        be the value surfaced to the logistic gate; otherwise the gate sees inputs that
        disagree with the probabilities it is gating on.
      - ``row_signals`` may carry redundant copies of pairwise columns (e.g. from runtime
        plumbing). Where the column also appears in ``pairwise_stats`` (i.e. it is a
        ``pw_*`` aggregate), ``row_signals`` must NOT override the matrix value.
      - ``pairwise_stats`` is an unvalidated overlay; it must only fill columns missing
        from ``values``.

    Net precedence for a column that exists in all three sources AND is a pairwise
    aggregate column: ``feature_matrix.matrix`` wins (the canonical assembled value).
    """

    row_count = 2
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
    )
    pairwise_columns = promoted_pairwise_aggregate_columns()
    target_column = "pw_mean_specter_cosine_sim"
    target_index = pairwise_columns.index(target_column)

    # Build a pairwise matrix where the target column carries value 1.0 (this is what
    # gets baked into ``feature_matrix.matrix`` by ``assemble_linker_feature_matrix``).
    matrix_value = np.float32(1.0)
    pairwise_matrix = np.zeros((row_count, len(pairwise_columns)), dtype=np.float32)
    pairwise_matrix[:, target_index] = matrix_value
    pairwise_stats = StaticPairwiseStats(pairwise_matrix, pairwise_columns)

    assembled = features.assemble_linker_feature_matrix(
        candidate_batch,
        _row_feature_fixture(row_count),
        pairwise_stats=cast(Any, pairwise_stats),
    )

    # Sanity check: matrix actually carries the canonical 1.0 for the target column.
    assembled_column_index = assembled.feature_columns.index(target_column)
    np.testing.assert_array_equal(
        assembled.matrix[:, assembled_column_index],
        np.full(row_count, matrix_value, dtype=np.float32),
    )

    # Then mutate the pairwise overlay AFTER assembly so the pairwise_stats overlay value
    # (3.0) differs from the matrix-baked value (1.0). This is the only way to construct
    # a three-way disagreement and is exactly the silent-precedence footgun this test
    # locks down.
    overlay_value = np.float32(3.0)
    pairwise_stats._matrix[:, target_index] = overlay_value
    row_signal_value = np.asarray([2.0, 2.0], dtype=np.float32)

    values = feature_values_from_runtime(
        assembled,
        {target_column: row_signal_value},
    )

    # Matrix (1.0) wins over row_signals (2.0) and pairwise_stats overlay (3.0).
    np.testing.assert_array_equal(
        values[target_column],
        np.full(row_count, matrix_value, dtype=np.float32),
    )
    # Explicitly assert the losers are NOT visible.
    assert not np.array_equal(values[target_column], row_signal_value)
    assert not np.array_equal(values[target_column], np.full(row_count, overlay_value, dtype=np.float32))


def test_feature_values_from_runtime_row_signal_wins_for_non_pairwise_overlap() -> None:
    """When a row_signal key collides with a matrix column but is NOT a pairwise aggregate
    column, ``row_signals`` overrides the matrix value.

    This documents the second half of the precedence contract: the matrix-wins rule is
    scoped to pairwise columns (where matrix is the canonical assembled form populated
    from pairwise_stats). For non-pairwise columns, callers who pass a row_signal of the
    same name are deliberately overriding -- e.g. constraint-eligibility resubsetting in
    ``runtime._predict_incremental_link_or_abstain_compact``.
    """

    row_count = 2
    candidate_batch = LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.zeros(0, dtype=np.uint32),
        right_signature_indices=np.zeros(0, dtype=np.uint32),
        pair_row_indices=np.zeros(0, dtype=np.uint32),
    )
    pairwise_columns = promoted_pairwise_aggregate_columns()
    pairwise_matrix = np.zeros((row_count, len(pairwise_columns)), dtype=np.float32)
    pairwise_stats = StaticPairwiseStats(pairwise_matrix, pairwise_columns)
    assembled = features.assemble_linker_feature_matrix(
        candidate_batch,
        _row_feature_fixture(row_count),
        pairwise_stats=cast(Any, pairwise_stats),
    )

    target_column = "min_distance"
    assert target_column not in pairwise_columns  # precondition: non-pairwise column
    override = np.asarray([42.0, 99.0], dtype=np.float32)

    values = feature_values_from_runtime(assembled, {target_column: override})

    np.testing.assert_array_equal(values[target_column], override)

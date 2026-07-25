from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from s2and import feature_port, memory_budget
from s2and.incremental_linking import linker_pairwise
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset, import_s2and_rust

HAS_RUST, RUST_IMPORT_ERROR = import_s2and_rust()


def _mock_chunk_plan(chunk_pairs: int, total_pairs: int) -> memory_budget.RustBatchChunkPlan:
    return memory_budget.RustBatchChunkPlan(
        total_ram_bytes=2 * 1024 * 1024 * 1024,
        total_ram_source="test",
        current_rss_bytes=128 * 1024 * 1024,
        current_rss_source="test",
        available_bytes=1024 * 1024 * 1024,
        effective_available_fraction=0.5,
        safety_margin_bytes=128 * 1024 * 1024,
        stage_budget_fraction=0.25,
        stage_budget_bytes=256 * 1024 * 1024,
        base_chunk_pairs=int(chunk_pairs),
        max_chunk_pairs=int(chunk_pairs),
        row_overhead_bytes=128,
        persistent_row_overhead_bytes=52,
        fixed_overhead_bytes=16 * (1 << 20),
        bytes_per_pair_row=256,
        chunk_pairs=int(chunk_pairs),
        total_rows=int(total_pairs),
        full_feature_count=39,
        selected_feature_count=39,
        nameless_feature_count=0,
        predicted_chunk_bytes=int(chunk_pairs) * 256,
        predicted_features_matrix_bytes=int(total_pairs) * 39 * 8,
        predicted_labels_bytes=int(total_pairs) * 8,
        predicted_persistent_row_overhead_bytes=int(total_pairs) * 52,
        predicted_fixed_overhead_bytes=16 * (1 << 20),
        predicted_stage_peak_delta_bytes=16 * (1 << 20),
        predicted_stage_peak_rss_bytes=144 * 1024 * 1024,
    )


def _candidate_batch_from_index_arrays(
    *,
    left_signature_indices: list[int],
    right_signature_indices: list[int],
    row_indices: list[int],
    row_count: int,
) -> linker_pairwise.LinkerCandidateBatch:
    return linker_pairwise.LinkerCandidateBatch(
        row_count=row_count,
        left_signature_indices=np.asarray(left_signature_indices, dtype=np.uint32),
        right_signature_indices=np.asarray(right_signature_indices, dtype=np.uint32),
        pair_row_indices=np.asarray(row_indices, dtype=np.uint32),
    )


def test_candidate_batch_rejects_uint32_wraparound_indices() -> None:
    with pytest.raises(ValueError, match="uint32 range"):
        linker_pairwise.LinkerCandidateBatch(
            row_count=1,
            left_signature_indices=np.asarray([-1], dtype=np.int64),
            right_signature_indices=np.asarray([0], dtype=np.int64),
            pair_row_indices=np.asarray([0], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="uint32 range"):
        linker_pairwise.LinkerCandidateBatch(
            row_count=1,
            left_signature_indices=np.asarray([0], dtype=np.int64),
            right_signature_indices=np.asarray([int(np.iinfo(np.uint32).max) + 1], dtype=np.int64),
            pair_row_indices=np.asarray([0], dtype=np.int64),
        )


@pytest.mark.parametrize(
    "retrieval_ranks",
    (
        np.asarray([0], dtype=np.uint16),
        np.asarray([-1], dtype=np.int64),
        np.asarray([int(np.iinfo(np.uint16).max) + 1], dtype=np.int64),
    ),
)
def test_candidate_batch_rejects_invalid_retrieval_ranks(retrieval_ranks: np.ndarray) -> None:
    with pytest.raises(ValueError, match="retrieval_ranks"):
        linker_pairwise.LinkerCandidateBatch(
            row_count=1,
            left_signature_indices=np.asarray([0], dtype=np.uint32),
            right_signature_indices=np.asarray([1], dtype=np.uint32),
            pair_row_indices=np.asarray([0], dtype=np.uint32),
            retrieval_ranks=retrieval_ranks,
        )


def test_pairwise_featurizer_resolver_prefers_explicit_featurizer() -> None:
    featurizer = object()

    resolved = linker_pairwise.resolve_linker_pairwise_featurizer(None, featurizer)

    assert resolved is featurizer


def test_pairwise_featurizer_resolver_requires_dataset_without_featurizer() -> None:
    with pytest.raises(ValueError, match="dataset is required"):
        linker_pairwise.resolve_linker_pairwise_featurizer(None, None)


def test_array_feature_wrappers_pass_separate_nan_policies() -> None:
    calls: list[tuple[float, float, bool]] = []

    class FakeRustFeaturizer:
        def linker_pair_index_arrays_and_aggregate_stats(
            self,
            left_signature_indices,
            right_signature_indices,
            row_indices,
            row_count,
            matrix_indices,
            aggregate_indices,
            num_threads,
            nan_value,
            aggregate_nan_value,
            emit_matrix=True,
        ):
            del right_signature_indices, row_indices, row_count, matrix_indices, aggregate_indices, num_threads
            calls.append((float(nan_value), float(aggregate_nan_value), bool(emit_matrix)))
            pair_count = len(left_signature_indices)
            return (
                np.zeros((pair_count, 1), dtype=np.float64),
                np.ones(1, dtype=np.uint32),
                np.ones((1, 1), dtype=np.uint64),
                np.zeros((1, 1), dtype=np.float64),
                np.zeros((1, 1), dtype=np.float64),
                np.zeros((1, 1), dtype=np.float64),
            )

    matrix, counts, valid_counts, sums, mins, maxs = (
        feature_port.build_linker_pair_features_and_aggregate_stats_arrays_rust(
            np.asarray([0, 1], dtype=np.uint32),
            np.asarray([1, 2], dtype=np.uint32),
            np.asarray([0, 0], dtype=np.uint32),
            1,
            matrix_indices=[0],
            aggregate_indices=[0],
            num_threads=2,
            nan_value=np.nan,
            aggregate_nan_value=0.0,
            featurizer=FakeRustFeaturizer(),
        )
    )

    assert len(calls) == 1
    assert np.isnan(calls[0][0])
    assert calls[0][1] == 0.0
    assert calls[0][2] is True
    assert matrix.shape == (2, 1)
    assert counts.tolist() == [1]
    assert valid_counts.shape == (1, 1)
    assert sums.shape == mins.shape == maxs.shape == (1, 1)

    feature_port.build_linker_pair_aggregate_stats_arrays_rust(
        np.asarray([0, 1], dtype=np.uint32),
        np.asarray([1, 2], dtype=np.uint32),
        np.asarray([0, 0], dtype=np.uint32),
        1,
        aggregate_indices=[0],
        num_threads=2,
        nan_value=np.nan,
        aggregate_nan_value=0.0,
        featurizer=FakeRustFeaturizer(),
    )

    assert len(calls) == 2
    assert np.isnan(calls[1][0])
    assert calls[1][1] == 0.0
    assert calls[1][2] is False


def test_combined_array_feature_wrapper_requires_exact_six_array_result() -> None:
    class FakeRustFeaturizer:
        def linker_pair_index_arrays_and_aggregate_stats(self, *args):
            del args
            return (
                np.zeros((1, 1), dtype=np.float64),
                np.ones(1, dtype=np.uint32),
                np.zeros((1, 1), dtype=np.float64),
                np.zeros((1, 1), dtype=np.float64),
                np.zeros((1, 1), dtype=np.float64),
            )

    with pytest.raises(RuntimeError, match="violated its six-array result contract"):
        feature_port.build_linker_pair_features_and_aggregate_stats_arrays_rust(
            np.asarray([0], dtype=np.uint32),
            np.asarray([1], dtype=np.uint32),
            np.asarray([0], dtype=np.uint32),
            1,
            matrix_indices=[0],
            aggregate_indices=[0],
            featurizer=FakeRustFeaturizer(),
        )


def test_combined_array_feature_wrapper_passes_result_arrays_through() -> None:
    class FakeRustFeaturizer:
        def linker_pair_index_arrays_and_aggregate_stats(self, *args):
            del args
            return (
                [[0.25]],
                [1],
                [[1]],
                [[0.25]],
                [[0.25]],
                [[0.25]],
            )

    matrix, counts, valid_counts, sums, mins, maxs = (
        feature_port.build_linker_pair_features_and_aggregate_stats_arrays_rust(
            np.asarray([0], dtype=np.uint32),
            np.asarray([1], dtype=np.uint32),
            np.asarray([0], dtype=np.uint32),
            1,
            matrix_indices=[0],
            aggregate_indices=[0],
            featurizer=FakeRustFeaturizer(),
        )
    )

    assert matrix.dtype == np.float64
    assert counts.dtype == np.uint32
    assert valid_counts.dtype == np.uint64
    assert sums.dtype == mins.dtype == maxs.dtype == np.float64


def test_linker_pairwise_aggregates_use_memory_chunk_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = build_dummy_dataset("dummy_linker_pairwise_fake", name_counts_index=True)
    candidate_batch = _candidate_batch_from_index_arrays(
        left_signature_indices=[0, 0, 0, 1, 2],
        right_signature_indices=[1, 2, 3, 3, 3],
        row_indices=[0, 0, 1, 1, 1],
        row_count=2,
    )
    call_sizes: list[int] = []
    aggregate_indices_seen: list[tuple[int, ...]] = []

    class FakeRustFeaturizer:
        def linker_pair_index_arrays_and_aggregate_stats(
            self,
            left_signature_indices,
            right_signature_indices,
            local_row_indices,
            row_count,
            matrix_indices,
            aggregate_indices,
            num_threads,
            nan_value,
            aggregate_nan_value,
            emit_matrix,
        ):
            del matrix_indices, num_threads, nan_value, aggregate_nan_value
            assert emit_matrix is False
            call_sizes.append(len(left_signature_indices))
            aggregate_indices_seen.append(tuple(aggregate_indices))
            counts = np.zeros(int(row_count), dtype=np.uint32)
            valid_counts = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.uint64)
            sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
            mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
            maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
            for pair_offset, local_row_index in enumerate(local_row_indices):
                counts[int(local_row_index)] += 1
                left = int(left_signature_indices[pair_offset])
                right = int(right_signature_indices[pair_offset])
                values = np.asarray(
                    [float(left * 10 + right + feature_index) for feature_index in aggregate_indices],
                    dtype=np.float64,
                )
                valid_counts[int(local_row_index)] += 1
                sums[int(local_row_index)] += values
                mins[int(local_row_index)] = np.minimum(mins[int(local_row_index)], values)
                maxs[int(local_row_index)] = np.maximum(maxs[int(local_row_index)], values)
            return np.zeros((len(left_signature_indices), 0), dtype=np.float64), counts, valid_counts, sums, mins, maxs

    fake_featurizer = FakeRustFeaturizer()
    plan_call_count = 0
    plan_kwargs: dict[str, Any] = {}

    def fake_chunk_plan(**kwargs):
        nonlocal plan_call_count
        plan_call_count += 1
        plan_kwargs.update(kwargs)
        return _mock_chunk_plan(chunk_pairs=2, total_pairs=candidate_batch.pair_count)

    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        fake_chunk_plan,
    )
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, runtime_context=None: fake_featurizer,
    )

    stats = linker_pairwise.compute_candidate_batch_pairwise_aggregate_stats_rust(
        dataset,
        candidate_batch,
        aggregate_feature_names=("first_names_equal", "affiliation_overlap"),
        n_jobs=2,
        total_ram_bytes=2 * 1024 * 1024 * 1024,
    )

    assert call_sizes == [2, 2, 1]
    assert plan_call_count == 1
    assert plan_kwargs["index_remap_bytes_per_pair"] == 8
    assert all(0 in seen and 6 in seen for seen in aggregate_indices_seen)
    assert stats.counts.tolist() == [2, 3]
    assert stats.feature_matrix().shape == (2, 6)


def test_pairwise_aggregate_feature_matrix_preserves_missing_values() -> None:
    stats = linker_pairwise.PairwiseAggregateStats(
        counts=np.asarray([2, 0], dtype=np.uint64),
        sums=np.asarray([[np.nan, 6.0], [0.0, 0.0]], dtype=np.float64),
        mins=np.asarray([[np.nan, 2.0], [np.inf, np.inf]], dtype=np.float64),
        maxs=np.asarray([[np.nan, 4.0], [-np.inf, -np.inf]], dtype=np.float64),
        base_feature_names=("middle_names_equal", "affiliation_overlap"),
        aggregate_feature_columns=(
            "pw_min_middle_names_equal",
            "pw_min_affiliation_overlap",
            "pw_mean_middle_names_equal",
            "pw_mean_affiliation_overlap",
            "pw_max_middle_names_equal",
            "pw_max_affiliation_overlap",
        ),
        chunk_plan=_mock_chunk_plan(chunk_pairs=2, total_pairs=2),
        chunk_count=1,
        matrix_indices=(0, 1),
        aggregate_indices=(0, 1),
    )

    matrix = stats.feature_matrix()

    assert np.isnan(matrix[0, 0])
    assert np.isnan(matrix[0, 2])
    assert np.isnan(matrix[0, 4])
    np.testing.assert_allclose(matrix[0, [1, 3, 5]], np.asarray([2.0, 3.0, 4.0]))
    assert np.isnan(matrix[1]).all()


def test_pairwise_aggregate_feature_matrix_uses_per_feature_valid_counts() -> None:
    stats = linker_pairwise.PairwiseAggregateStats(
        counts=np.asarray([2], dtype=np.uint64),
        sums=np.asarray([[3.0, 0.0]], dtype=np.float64),
        mins=np.asarray([[1.0, 7.0]], dtype=np.float64),
        maxs=np.asarray([[2.0, 8.0]], dtype=np.float64),
        valid_counts=np.asarray([[2, 0]], dtype=np.uint64),
        base_feature_names=("middle_names_equal", "affiliation_overlap"),
        aggregate_feature_columns=(
            "pw_min_middle_names_equal",
            "pw_min_affiliation_overlap",
            "pw_mean_middle_names_equal",
            "pw_mean_affiliation_overlap",
            "pw_max_middle_names_equal",
            "pw_max_affiliation_overlap",
        ),
        chunk_plan=_mock_chunk_plan(chunk_pairs=2, total_pairs=2),
        chunk_count=1,
        matrix_indices=(0, 1),
        aggregate_indices=(0, 1),
    )

    matrix = stats.feature_matrix()

    np.testing.assert_allclose(matrix[0, [0, 2, 4]], np.asarray([1.0, 1.5, 2.0]))
    assert np.isnan(matrix[0, [1, 3, 5]]).all()


def test_candidate_batch_aggregates_use_array_api(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = build_dummy_dataset("dummy_linker_candidate_batch_fake", name_counts_index=True)
    candidate_batch = linker_pairwise.build_candidate_batch_from_members(
        [0, 1],
        [np.asarray([1, 2], dtype=np.uint32), np.asarray([3], dtype=np.uint32)],
        row_component_keys=("c0", "c1"),
        labels=np.asarray([1, 0], dtype=np.int8),
    )
    call_sizes: list[int] = []

    class FakeRustFeaturizer:
        def linker_pair_index_arrays_and_aggregate_stats(
            self,
            left_signature_indices,
            right_signature_indices,
            row_indices,
            row_count,
            matrix_indices,
            aggregate_indices,
            num_threads,
            nan_value,
            aggregate_nan_value,
            emit_matrix,
        ):
            del matrix_indices, num_threads, nan_value, aggregate_nan_value
            assert emit_matrix is False
            call_sizes.append(len(left_signature_indices))
            counts = np.zeros(int(row_count), dtype=np.uint32)
            valid_counts = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.uint64)
            sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
            mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
            maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
            for pair_offset, local_row_index in enumerate(row_indices):
                left = int(left_signature_indices[pair_offset])
                right = int(right_signature_indices[pair_offset])
                values = np.asarray(
                    [float(left + right + feature_index) for feature_index in aggregate_indices],
                    dtype=np.float64,
                )
                counts[int(local_row_index)] += 1
                valid_counts[int(local_row_index)] += 1
                sums[int(local_row_index)] += values
                mins[int(local_row_index)] = np.minimum(mins[int(local_row_index)], values)
                maxs[int(local_row_index)] = np.maximum(maxs[int(local_row_index)], values)
            return np.zeros((len(left_signature_indices), 0), dtype=np.float64), counts, valid_counts, sums, mins, maxs

    fake_featurizer = FakeRustFeaturizer()
    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        lambda **_kwargs: _mock_chunk_plan(chunk_pairs=2, total_pairs=candidate_batch.pair_count),
    )
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, runtime_context=None: fake_featurizer,
    )

    stats = linker_pairwise.compute_candidate_batch_pairwise_aggregate_stats_rust(
        dataset,
        candidate_batch,
        aggregate_feature_names=("first_names_equal",),
        n_jobs=2,
    )

    assert call_sizes == [2, 1]
    assert stats.counts.tolist() == [2, 1]
    assert cast(Any, candidate_batch.labels).tolist() == [1, 0]
    assert candidate_batch.row_component_keys == ("c0", "c1")


def test_candidate_batch_aggregates_trust_rust_aggregate_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = build_dummy_dataset("dummy_linker_candidate_batch_rust_aggregate_authority", name_counts_index=True)
    candidate_batch = _candidate_batch_from_index_arrays(
        left_signature_indices=[0],
        right_signature_indices=[1],
        row_indices=[0],
        row_count=1,
    )

    class FakeRustFeaturizer:
        def linker_pair_index_arrays_and_aggregate_stats(
            self,
            left_signature_indices,
            right_signature_indices,
            row_indices,
            row_count,
            matrix_indices,
            aggregate_indices,
            num_threads,
            nan_value,
            aggregate_nan_value,
            emit_matrix,
        ):
            del (
                left_signature_indices,
                right_signature_indices,
                row_indices,
                matrix_indices,
                aggregate_indices,
                num_threads,
                nan_value,
                aggregate_nan_value,
            )
            assert emit_matrix is False
            return (
                np.zeros((1, 0), dtype=np.float64),
                np.ones(int(row_count), dtype=np.uint32),
                np.ones((int(row_count), 1), dtype=np.uint64),
                np.asarray([[42.0]], dtype=np.float64),
                np.asarray([[40.0]], dtype=np.float64),
                np.asarray([[44.0]], dtype=np.float64),
            )

    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        lambda **_kwargs: _mock_chunk_plan(chunk_pairs=1, total_pairs=1),
    )
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, runtime_context=None: FakeRustFeaturizer(),
    )

    stats = linker_pairwise.compute_candidate_batch_pairwise_aggregate_stats_rust(
        dataset,
        candidate_batch,
        aggregate_feature_names=("first_names_equal",),
    )

    np.testing.assert_allclose(stats.feature_matrix()[0], np.asarray([40.0, 42.0, 44.0]))


def test_localize_row_indices_handles_grouped_sparse_and_ungrouped_chunks() -> None:
    cases = [
        ([5, 5, 6, 8, 8], [5, 6, 8], [0, 0, 1, 2, 2]),
        ([0, 1_000_000], [0, 1_000_000], [0, 1]),
        ([5, 2, 5, 3], [2, 3, 5], [2, 0, 2, 1]),
    ]
    for row_indices, expected_global, expected_local in cases:
        global_rows, local_rows = linker_pairwise._localize_row_indices(  # noqa: SLF001
            np.asarray(row_indices, dtype=np.uint32)
        )
        np.testing.assert_array_equal(global_rows, np.asarray(expected_global, dtype=np.int64))
        np.testing.assert_array_equal(local_rows, np.asarray(expected_local, dtype=np.uint32))


@pytest.mark.skipif(
    not HAS_RUST,
    reason=f"s2and_rust unavailable: {RUST_IMPORT_ERROR}",
)
def test_candidate_batch_aggregates_match_existing_rust_matrix_path(tmp_path) -> None:
    dataset = build_dummy_dataset("dummy_linker_candidate_batch_real", name_counts_index=True)
    dataset = build_arrow_training_dataset(dataset, tmp_path)
    pairs = [("0", "1"), ("0", "2"), ("3", "4"), ("0", "3"), ("1", "4")]
    row_indices = [0, 0, 1, 1, 1]
    feature_names = ("first_names_equal", "affiliation_overlap", "title_overlap_words")
    feature_indices = [
        linker_pairwise.PROD_PAIRWISE_FEATURE_INDICES[linker_pairwise.PROD_PAIRWISE_FEATURE_NAMES.index(feature_name)]
        for feature_name in feature_names
    ]
    rust_featurizer = feature_port._get_rust_featurizer(dataset)  # noqa: SLF001
    signature_id_to_index = {
        str(signature_id): index for index, signature_id in enumerate(rust_featurizer.signature_ids())
    }
    candidate_batch = linker_pairwise.LinkerCandidateBatch(
        row_count=2,
        left_signature_indices=np.asarray([signature_id_to_index[left] for left, _right in pairs], dtype=np.uint32),
        right_signature_indices=np.asarray([signature_id_to_index[right] for _left, right in pairs], dtype=np.uint32),
        pair_row_indices=np.asarray(row_indices, dtype=np.uint32),
    )

    matrix, *_ = feature_port.build_linker_pair_features_and_aggregate_stats_arrays_rust(
        candidate_batch.left_signature_indices,
        candidate_batch.right_signature_indices,
        candidate_batch.pair_row_indices,
        candidate_batch.row_count,
        matrix_indices=feature_indices,
        aggregate_indices=feature_indices,
        num_threads=2,
        nan_value=0.0,
        aggregate_nan_value=0.0,
        featurizer=rust_featurizer,
    )
    expected_counts = np.zeros(2, dtype=np.uint64)
    expected_sums = np.zeros((2, len(feature_indices)), dtype=np.float64)
    expected_mins = np.full((2, len(feature_indices)), np.inf, dtype=np.float64)
    expected_maxs = np.full((2, len(feature_indices)), -np.inf, dtype=np.float64)
    for pair_offset, row_index in enumerate(row_indices):
        expected_counts[row_index] += 1
        expected_sums[row_index] += matrix[pair_offset]
        expected_mins[row_index] = np.minimum(expected_mins[row_index], matrix[pair_offset])
        expected_maxs[row_index] = np.maximum(expected_maxs[row_index], matrix[pair_offset])

    stats = linker_pairwise.compute_candidate_batch_pairwise_aggregate_stats_rust(
        dataset,
        candidate_batch,
        aggregate_feature_names=feature_names,
        n_jobs=2,
        nan_value=0.0,
        featurizer=rust_featurizer,
    )

    np.testing.assert_array_equal(stats.counts, expected_counts)
    np.testing.assert_allclose(stats.sums, expected_sums, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats.mins, expected_mins, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats.maxs, expected_maxs, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(
    not HAS_RUST,
    reason=f"s2and_rust unavailable: {RUST_IMPORT_ERROR}",
)
def test_native_combined_matrix_and_aggregates_apply_independent_nan_policies(tmp_path) -> None:
    dataset = build_dummy_dataset("dummy_linker_independent_nan_policies", name_counts_index=True)
    dataset = build_arrow_training_dataset(dataset, tmp_path)
    rust_featurizer = feature_port._get_rust_featurizer(dataset)  # noqa: SLF001

    raw_matrix = np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed(
            [(0, 1)],
            None,
            1,
            np.nan,
        )
    )
    nan_positions = np.flatnonzero(np.isnan(raw_matrix[0]))
    assert nan_positions.size > 0, "dummy pair must exercise a genuinely missing native feature"
    selected_indices = [int(nan_positions[0])]
    left = np.asarray([0], dtype=np.uint32)
    right = np.asarray([1], dtype=np.uint32)
    rows = np.asarray([0], dtype=np.uint32)

    def combined(matrix_nan_value: float, aggregate_nan_value: float):
        return tuple(
            np.asarray(value)
            for value in rust_featurizer.linker_pair_index_arrays_and_aggregate_stats(
                left,
                right,
                rows,
                1,
                selected_indices,
                selected_indices,
                1,
                matrix_nan_value,
                aggregate_nan_value,
                True,
            )
        )

    matrix, counts, valid_counts, sums, mins, maxs = combined(0.0, np.nan)
    np.testing.assert_array_equal(matrix, np.asarray([[0.0]]))
    np.testing.assert_array_equal(counts, np.asarray([1], dtype=np.uint32))
    np.testing.assert_array_equal(valid_counts, np.asarray([[0]], dtype=np.uint64))
    np.testing.assert_array_equal(sums, np.asarray([[0.0]]))
    assert np.isposinf(mins[0, 0])
    assert np.isneginf(maxs[0, 0])

    matrix, counts, valid_counts, sums, mins, maxs = combined(np.nan, 0.0)
    assert np.isnan(matrix[0, 0])
    np.testing.assert_array_equal(counts, np.asarray([1], dtype=np.uint32))
    np.testing.assert_array_equal(valid_counts, np.asarray([[1]], dtype=np.uint64))
    np.testing.assert_array_equal(sums, np.asarray([[0.0]]))
    np.testing.assert_array_equal(mins, np.asarray([[0.0]]))
    np.testing.assert_array_equal(maxs, np.asarray([[0.0]]))


@pytest.mark.skipif(
    not HAS_RUST,
    reason=f"s2and_rust unavailable: {RUST_IMPORT_ERROR}",
)
def test_dense_and_sparse_signature_index_layouts_produce_identical_features(tmp_path) -> None:
    dataset = build_dummy_dataset("dummy_linker_signature_index_layouts", name_counts_index=True)
    template_signature = next(iter(dataset.signatures.values()))
    template_paper = dataset.papers[str(template_signature.paper_id)]
    next_paper_id = max(int(paper_id) for paper_id in dataset.papers) + 1
    while len(dataset.signatures) < 128:
        offset = len(dataset.signatures)
        signature_id = f"layout_{offset:04d}"
        paper_id = next_paper_id + offset
        dataset.papers[str(paper_id)] = template_paper._replace(paper_id=paper_id)
        dataset.signatures[signature_id] = template_signature._replace(
            signature_id=signature_id,
            paper_id=paper_id,
        )

    dataset = build_arrow_training_dataset(dataset, tmp_path)
    rust_featurizer = feature_port._get_rust_featurizer(dataset)  # noqa: SLF001
    signature_count = len(rust_featurizer.signature_ids())
    target_pair = (0, signature_count - 1)
    dense_pairs = [target_pair, *((index, index + 1) for index in range(signature_count - 1))]
    selected_indices = [0, 6, 10]

    dense_owned = np.asarray(rust_featurizer.featurize_pairs_matrix_indexed(dense_pairs, selected_indices, 1, 0.0))
    sparse_owned = np.asarray(rust_featurizer.featurize_pairs_matrix_indexed([target_pair], selected_indices, 1, 0.0))
    np.testing.assert_allclose(sparse_owned[0], dense_owned[0], rtol=0.0, atol=0.0)

    def array_result(pairs: list[tuple[int, int]]):
        left = np.asarray([pair[0] for pair in pairs], dtype=np.uint32)
        right = np.asarray([pair[1] for pair in pairs], dtype=np.uint32)
        rows = np.arange(len(pairs), dtype=np.uint32)
        return rust_featurizer.linker_pair_index_arrays_and_aggregate_stats(
            left,
            right,
            rows,
            len(pairs),
            selected_indices,
            selected_indices,
            1,
            0.0,
            0.0,
            True,
        )

    dense_matrix, dense_counts, dense_valid, dense_sums, dense_mins, dense_maxs = array_result(dense_pairs)
    sparse_matrix, sparse_counts, sparse_valid, sparse_sums, sparse_mins, sparse_maxs = array_result([target_pair])
    np.testing.assert_allclose(np.asarray(sparse_matrix)[0], np.asarray(dense_matrix)[0], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(np.asarray(sparse_counts), np.asarray(dense_counts)[:1])
    np.testing.assert_array_equal(np.asarray(sparse_valid), np.asarray(dense_valid)[:1])
    np.testing.assert_allclose(np.asarray(sparse_sums), np.asarray(dense_sums)[:1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(sparse_mins), np.asarray(dense_mins)[:1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(sparse_maxs), np.asarray(dense_maxs)[:1], rtol=0.0, atol=0.0)

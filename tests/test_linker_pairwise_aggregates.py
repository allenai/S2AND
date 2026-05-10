from __future__ import annotations

import numpy as np
import pytest

from s2and import feature_port, memory_budget
from s2and.feature_port import build_pair_feature_matrix_rust
from s2and.incremental_linking import linker_pairwise
from tests.helpers import build_dummy_dataset, import_s2and_rust

HAS_LINKER_RUST, LINKER_RUST_IMPORT_ERROR = import_s2and_rust(
    required_method="linker_pair_features_and_aggregate_stats_indexed"
)
HAS_LINKER_ARRAY_AGG_RUST, LINKER_ARRAY_AGG_RUST_IMPORT_ERROR = import_s2and_rust(
    required_method="linker_pair_index_arrays_aggregate_stats"
)


def _mock_chunk_plan(chunk_pairs: int, total_pairs: int) -> dict[str, int | str | float]:
    return {
        "total_ram_bytes": 2 * 1024 * 1024 * 1024,
        "total_ram_source": "test",
        "current_rss_bytes": 128 * 1024 * 1024,
        "current_rss_source": "test",
        "available_bytes": 1024 * 1024 * 1024,
        "effective_available_fraction": 0.5,
        "safety_margin_bytes": 128 * 1024 * 1024,
        "stage_budget_fraction": 0.25,
        "stage_budget_bytes": 256 * 1024 * 1024,
        "base_chunk_pairs": int(chunk_pairs),
        "row_overhead_bytes": 128,
        "persistent_row_overhead_bytes": 52,
        "fixed_overhead_bytes": 16 * (1 << 20),
        "bytes_per_pair_row": 256,
        "derived_chunk_pairs": int(chunk_pairs),
        "chunk_pairs": int(chunk_pairs),
        "total_rows": int(total_pairs),
        "full_feature_count": 39,
        "selected_feature_count": 39,
        "nameless_feature_count": 0,
        "predicted_chunk_bytes": int(chunk_pairs) * 256,
        "predicted_features_matrix_bytes": int(total_pairs) * 39 * 8,
        "predicted_labels_bytes": int(total_pairs) * 8,
        "predicted_persistent_row_overhead_bytes": int(total_pairs) * 52,
        "predicted_fixed_overhead_bytes": 16 * (1 << 20),
        "predicted_selected_features_bytes": int(total_pairs) * 39 * 8,
        "predicted_nameless_features_bytes": 0,
        "predicted_stage_peak_delta_bytes": 16 * (1 << 20),
        "predicted_stage_peak_rss_bytes": 144 * 1024 * 1024,
        "predicted_stage_peak_bytes": 16 * (1 << 20),
    }


def test_combined_array_feature_wrapper_passes_separate_nan_policies() -> None:
    calls: list[tuple[float, float]] = []

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
        ):
            del right_signature_indices, row_indices, row_count, matrix_indices, aggregate_indices, num_threads
            calls.append((float(nan_value), float(aggregate_nan_value)))
            pair_count = len(left_signature_indices)
            return (
                np.zeros((pair_count, 1), dtype=np.float64),
                np.ones(1, dtype=np.uint32),
                np.zeros((1, 1), dtype=np.float64),
                np.zeros((1, 1), dtype=np.float64),
                np.zeros((1, 1), dtype=np.float64),
            )

    matrix, counts, sums, mins, maxs = feature_port.build_linker_pair_features_and_aggregate_stats_arrays_rust(
        object(),
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

    assert len(calls) == 1
    assert np.isnan(calls[0][0])
    assert calls[0][1] == 0.0
    assert matrix.shape == (2, 1)
    assert counts.tolist() == [1]
    assert sums.shape == mins.shape == maxs.shape == (1, 1)


def test_linker_pairwise_aggregates_use_memory_chunk_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = build_dummy_dataset("dummy_linker_pairwise_fake", load_name_counts=True)
    pairs = [("0", "1"), ("0", "2"), ("0", "3"), ("1", "3"), ("2", "3")]
    row_indices = [0, 0, 1, 1, 1]
    call_sizes: list[int] = []
    matrix_indices_seen: list[tuple[int, ...]] = []

    class FakeRustFeaturizer:
        def signature_ids(self):
            return ["0", "1", "2", "3"]

        def linker_pair_features_and_aggregate_stats_indexed(
            self,
            indexed_pairs,
            local_row_indices,
            row_count,
            matrix_indices,
            aggregate_indices,
            num_threads,
            nan_value,
        ):
            del num_threads, nan_value
            call_sizes.append(len(indexed_pairs))
            matrix_indices_seen.append(tuple(matrix_indices))
            matrix = np.asarray(
                [
                    [float(left * 10 + right + feature_index) for feature_index in matrix_indices]
                    for left, right in indexed_pairs
                ],
                dtype=np.float64,
            )
            counts = np.zeros(int(row_count), dtype=np.uint32)
            sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
            mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
            maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
            aggregate_positions = [matrix_indices.index(feature_index) for feature_index in aggregate_indices]
            for pair_offset, local_row_index in enumerate(local_row_indices):
                counts[int(local_row_index)] += 1
                values = matrix[pair_offset, aggregate_positions]
                sums[int(local_row_index)] += values
                mins[int(local_row_index)] = np.minimum(mins[int(local_row_index)], values)
                maxs[int(local_row_index)] = np.maximum(maxs[int(local_row_index)], values)
            return matrix, counts, sums, mins, maxs

    fake_featurizer = FakeRustFeaturizer()
    plan_call_count = 0

    def fake_chunk_plan(**_kwargs):
        nonlocal plan_call_count
        plan_call_count += 1
        return _mock_chunk_plan(chunk_pairs=2, total_pairs=len(pairs))

    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        fake_chunk_plan,
    )
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, runtime_context=None, use_cache=False: fake_featurizer,
    )

    stats = linker_pairwise.compute_pairwise_aggregate_stats_rust(
        dataset,
        pairs,
        row_indices,
        row_count=2,
        aggregate_feature_names=("first_names_equal", "affiliation_overlap"),
        n_jobs=2,
        total_ram_bytes=2 * 1024 * 1024 * 1024,
    )

    assert call_sizes == [2, 2, 1]
    assert plan_call_count == 1
    assert all(0 in seen and 6 in seen for seen in matrix_indices_seen)
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


def test_linker_pairwise_aggregates_accept_indexed_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = build_dummy_dataset("dummy_linker_pairwise_indexed_fake", load_name_counts=True)
    pairs = [(0, 1), (0, 2), (1, 2)]
    row_indices = [0, 0, 1]
    indexed_pairs_seen: list[tuple[int, int]] = []

    class FakeRustFeaturizer:
        def signature_ids(self):
            raise AssertionError("indexed pair path should not request signature_ids")

        def linker_pair_features_and_aggregate_stats_indexed(
            self,
            indexed_pairs,
            local_row_indices,
            row_count,
            matrix_indices,
            aggregate_indices,
            num_threads,
            nan_value,
        ):
            del local_row_indices, num_threads, nan_value
            indexed_pairs_seen.extend(indexed_pairs)
            matrix = np.asarray(
                [
                    [float(left + right + feature_index) for feature_index in matrix_indices]
                    for left, right in indexed_pairs
                ],
                dtype=np.float64,
            )
            return (
                matrix,
                np.ones(int(row_count), dtype=np.uint32),
                np.ones((int(row_count), len(aggregate_indices)), dtype=np.float64),
                np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64),
                np.ones((int(row_count), len(aggregate_indices)), dtype=np.float64),
            )

    fake_featurizer = FakeRustFeaturizer()
    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        lambda **_kwargs: _mock_chunk_plan(chunk_pairs=2, total_pairs=len(pairs)),
    )
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, runtime_context=None, use_cache=False: fake_featurizer,
    )

    linker_pairwise.compute_pairwise_aggregate_stats_rust(
        dataset,
        pairs,
        row_indices,
        row_count=2,
        aggregate_feature_names=("first_names_equal",),
        n_jobs=2,
        pairs_are_indices=True,
    )

    assert indexed_pairs_seen == pairs


def test_candidate_batch_aggregates_use_array_api(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = build_dummy_dataset("dummy_linker_candidate_batch_fake", load_name_counts=True)
    candidate_batch = linker_pairwise.build_candidate_batch_from_members(
        [0, 1],
        [np.asarray([1, 2], dtype=np.uint32), np.asarray([3], dtype=np.uint32)],
        row_component_keys=("c0", "c1"),
        labels=np.asarray([1, 0], dtype=np.int8),
    )
    call_sizes: list[int] = []

    class FakeRustFeaturizer:
        def linker_pair_index_arrays_aggregate_stats(
            self,
            left_signature_indices,
            right_signature_indices,
            row_indices,
            row_count,
            aggregate_indices,
            num_threads,
            nan_value,
        ):
            del num_threads, nan_value
            call_sizes.append(len(left_signature_indices))
            counts = np.zeros(int(row_count), dtype=np.uint32)
            sums = np.zeros((int(row_count), len(aggregate_indices)), dtype=np.float64)
            mins = np.full((int(row_count), len(aggregate_indices)), np.inf, dtype=np.float64)
            maxs = np.full((int(row_count), len(aggregate_indices)), -np.inf, dtype=np.float64)
            for left, right, local_row_index in zip(
                left_signature_indices,
                right_signature_indices,
                row_indices,
                strict=True,
            ):
                counts[int(local_row_index)] += 1
                values = np.asarray(
                    [float(left + right + feature_index) for feature_index in aggregate_indices],
                    dtype=np.float64,
                )
                sums[int(local_row_index)] += values
                mins[int(local_row_index)] = np.minimum(mins[int(local_row_index)], values)
                maxs[int(local_row_index)] = np.maximum(maxs[int(local_row_index)], values)
            return counts, sums, mins, maxs

    fake_featurizer = FakeRustFeaturizer()
    monkeypatch.setattr(
        memory_budget,
        "compute_rust_batch_chunk_plan",
        lambda **_kwargs: _mock_chunk_plan(chunk_pairs=2, total_pairs=candidate_batch.pair_count),
    )
    monkeypatch.setattr(
        feature_port,
        "_get_rust_featurizer",
        lambda _dataset, runtime_context=None, use_cache=False: fake_featurizer,
    )

    stats = linker_pairwise.compute_candidate_batch_pairwise_aggregate_stats_rust(
        dataset,
        candidate_batch,
        aggregate_feature_names=("first_names_equal",),
        n_jobs=2,
    )

    assert call_sizes == [2, 1]
    assert stats.counts.tolist() == [2, 1]
    assert candidate_batch.labels.tolist() == [1, 0]
    assert candidate_batch.row_component_keys == ("c0", "c1")


def test_localize_row_indices_keeps_grouped_chunks_fast() -> None:
    global_rows, local_rows = linker_pairwise._localize_row_indices(  # noqa: SLF001
        np.asarray([5, 5, 6, 8, 8], dtype=np.uint32)
    )

    np.testing.assert_array_equal(global_rows, np.asarray([5, 6, 7, 8], dtype=np.int64))
    np.testing.assert_array_equal(local_rows, np.asarray([0, 0, 1, 3, 3], dtype=np.uint32))


def test_localize_row_indices_handles_ungrouped_chunks() -> None:
    global_rows, local_rows = linker_pairwise._localize_row_indices(  # noqa: SLF001
        np.asarray([5, 2, 5, 3], dtype=np.uint32)
    )

    np.testing.assert_array_equal(global_rows, np.asarray([2, 3, 5], dtype=np.int64))
    np.testing.assert_array_equal(local_rows, np.asarray([2, 0, 2, 1], dtype=np.uint32))


@pytest.mark.skipif(
    not HAS_LINKER_RUST,
    reason=f"s2and_rust linker aggregate API unavailable: {LINKER_RUST_IMPORT_ERROR}",
)
def test_linker_pairwise_aggregates_match_existing_rust_matrix_path() -> None:
    dataset = build_dummy_dataset("dummy_linker_pairwise_real", load_name_counts=True)
    pairs = [("0", "1"), ("0", "2"), ("3", "4"), ("0", "3"), ("1", "4")]
    row_indices = [0, 0, 1, 1, 1]
    feature_names = ("first_names_equal", "affiliation_overlap", "title_overlap_words")
    feature_indices = [
        linker_pairwise.PROD_PAIRWISE_FEATURE_INDICES[linker_pairwise.PROD_PAIRWISE_FEATURE_NAMES.index(feature_name)]
        for feature_name in feature_names
    ]

    matrix = build_pair_feature_matrix_rust(
        dataset,
        pairs,
        selected_indices=feature_indices,
        num_threads=2,
        nan_value=0.0,
        use_cache=False,
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

    stats = linker_pairwise.compute_pairwise_aggregate_stats_rust(
        dataset,
        pairs,
        row_indices,
        row_count=2,
        aggregate_feature_names=feature_names,
        n_jobs=2,
        nan_value=0.0,
        use_cache=False,
    )

    np.testing.assert_array_equal(stats.counts, expected_counts)
    np.testing.assert_allclose(stats.sums, expected_sums, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats.mins, expected_mins, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats.maxs, expected_maxs, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(
    not HAS_LINKER_ARRAY_AGG_RUST,
    reason=f"s2and_rust linker array aggregate-only API unavailable: {LINKER_ARRAY_AGG_RUST_IMPORT_ERROR}",
)
def test_candidate_batch_aggregates_match_existing_rust_matrix_path() -> None:
    dataset = build_dummy_dataset("dummy_linker_candidate_batch_real", load_name_counts=True)
    pairs = [("0", "1"), ("0", "2"), ("3", "4"), ("0", "3"), ("1", "4")]
    row_indices = [0, 0, 1, 1, 1]
    feature_names = ("first_names_equal", "affiliation_overlap", "title_overlap_words")
    feature_indices = [
        linker_pairwise.PROD_PAIRWISE_FEATURE_INDICES[linker_pairwise.PROD_PAIRWISE_FEATURE_NAMES.index(feature_name)]
        for feature_name in feature_names
    ]
    rust_featurizer = feature_port._get_rust_featurizer(dataset, use_cache=False)  # noqa: SLF001
    signature_id_to_index = {
        str(signature_id): index for index, signature_id in enumerate(rust_featurizer.signature_ids())
    }
    candidate_batch = linker_pairwise.LinkerCandidateBatch(
        row_count=2,
        left_signature_indices=np.asarray([signature_id_to_index[left] for left, _right in pairs], dtype=np.uint32),
        right_signature_indices=np.asarray([signature_id_to_index[right] for _left, right in pairs], dtype=np.uint32),
        pair_row_indices=np.asarray(row_indices, dtype=np.uint32),
    )

    matrix = build_pair_feature_matrix_rust(
        dataset,
        pairs,
        selected_indices=feature_indices,
        num_threads=2,
        nan_value=0.0,
        use_cache=False,
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
        use_cache=False,
        featurizer=rust_featurizer,
    )

    np.testing.assert_array_equal(stats.counts, expected_counts)
    np.testing.assert_allclose(stats.sums, expected_sums, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats.mins, expected_mins, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stats.maxs, expected_maxs, rtol=1e-9, atol=1e-9)

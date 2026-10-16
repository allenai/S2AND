from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from s2and import feature_port, memory_budget
from s2and.incremental_linking import linker_pairwise
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset


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
    ids=("zero", "negative", "overflow"),
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


def test_pairwise_featurizer_resolver_requires_dataset_without_featurizer() -> None:
    with pytest.raises(ValueError, match="dataset is required"):
        linker_pairwise.resolve_linker_pairwise_featurizer(None, None)


def test_combined_array_feature_wrapper_requires_exact_six_array_result():
    featurizer = SimpleNamespace(linker_pair_index_arrays_and_aggregate_stats=lambda *args: ([],) * 5)
    with pytest.raises(RuntimeError, match="violated its six-array result contract"):
        feature_port.build_linker_pair_features_and_aggregate_stats_arrays_rust(
            np.asarray([0], dtype=np.uint32),
            np.asarray([1], dtype=np.uint32),
            np.asarray([0], dtype=np.uint32),
            1,
            matrix_indices=[0],
            aggregate_indices=[0],
            featurizer=featurizer,
        )


@pytest.mark.parametrize(
    "sums,valid_counts,expected",
    [
        ([[np.nan, 6], [0, 0]], None, [np.nan, 2, np.nan, 3, np.nan, 4]),
        ([[3, 6], [0, 0]], [[2, 0], [0, 0]], [1, np.nan, 1.5, np.nan, 2, np.nan]),
    ],
    ids=["legacy-row-counts", "per-feature-valid-counts"],
)
def test_pairwise_aggregate_feature_matrix_handles_missing_values(sums, valid_counts, expected):
    stats = linker_pairwise.PairwiseAggregateStats(
        counts=np.asarray([2, 0], dtype=np.uint64),
        sums=np.asarray(sums, dtype=np.float64),
        mins=np.asarray([[np.nan if valid_counts is None else 1, 2], [np.inf, np.inf]]),
        maxs=np.asarray([[np.nan if valid_counts is None else 2, 4], [-np.inf, -np.inf]]),
        valid_counts=None if valid_counts is None else np.asarray(valid_counts, dtype=np.uint64),
        base_feature_names=("middle_names_equal", "affiliation_overlap"),
        aggregate_feature_columns=tuple(
            f"pw_{stat}_{name}"
            for stat in ("min", "mean", "max")
            for name in ("middle_names_equal", "affiliation_overlap")
        ),
        chunk_plan=memory_budget.compute_rust_batch_chunk_plan(num_features=2, total_pairs=2, total_ram_bytes=1 << 40),
        chunk_count=1,
        matrix_indices=(0, 1),
        aggregate_indices=(0, 1),
    )
    matrix = stats.feature_matrix()
    np.testing.assert_allclose(matrix[0], expected)
    assert np.isnan(matrix[1]).all()


def test_candidate_batch_expands_members_and_preserves_row_metadata():
    batch = linker_pairwise.build_candidate_batch_from_members(
        [0, 1],
        [np.asarray([1, 2], dtype=np.uint32), np.asarray([3], dtype=np.uint32)],
        row_component_keys=("c0", "c1"),
        labels=np.asarray([1, 0], dtype=np.int8),
    )
    np.testing.assert_array_equal(batch.left_signature_indices, [0, 0, 1])
    np.testing.assert_array_equal(batch.right_signature_indices, [1, 2, 3])
    np.testing.assert_array_equal(batch.pair_row_indices, [0, 0, 1])
    np.testing.assert_array_equal(batch.labels, [1, 0])
    assert batch.row_component_keys == ("c0", "c1")


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


@pytest.mark.parametrize("chunk_size", [2, 100], ids=["chunked", "one-batch"])
def test_native_candidate_aggregates_preserve_sparse_rows_across_chunks(tmp_path, monkeypatch, chunk_size):
    dataset = build_arrow_training_dataset(build_dummy_dataset("native_aggregates", name_counts_index=True), tmp_path)
    native = feature_port._get_rust_featurizer(dataset)
    pairs = [(0, 1), (0, 2), (3, 4), (0, 3), (1, 4)]
    rows = np.asarray([5, 0, 5, 2, 0], dtype=np.uint32)
    feature_names = ("first_names_equal", "affiliation_overlap", "title_overlap_words")
    feature_indices = [
        linker_pairwise.PROD_PAIRWISE_FEATURE_INDICES[linker_pairwise.PROD_PAIRWISE_FEATURE_NAMES.index(name)]
        for name in feature_names
    ]
    batch = linker_pairwise.LinkerCandidateBatch(
        row_count=6,
        left_signature_indices=np.asarray([left for left, _ in pairs], dtype=np.uint32),
        right_signature_indices=np.asarray([right for _, right in pairs], dtype=np.uint32),
        pair_row_indices=rows,
    )
    matrix = np.asarray(native.featurize_pairs_matrix_indexed(pairs, feature_indices, 1, 0.0))
    monkeypatch.setattr(memory_budget, "RUST_BATCH_MAX_CHUNK_PAIRS", chunk_size)
    stats = linker_pairwise.compute_candidate_batch_pairwise_aggregate_stats_rust(
        dataset,
        batch,
        aggregate_feature_names=feature_names,
        n_jobs=2,
        nan_value=0.0,
        featurizer=native if chunk_size == 2 else None,
        total_ram_bytes=1 << 40,
    )
    assert stats.chunk_count == (3 if chunk_size == 2 else 1)
    assert stats.chunk_plan.index_remap_bytes_per_pair == 8
    np.testing.assert_array_equal(stats.counts, [2, 0, 1, 0, 0, 2])
    for row in range(6):
        values = matrix[rows == row]
        np.testing.assert_array_equal(stats.valid_counts[row], [len(values)] * len(feature_names))
        np.testing.assert_allclose(stats.sums[row], values.sum(axis=0))
        np.testing.assert_allclose(stats.mins[row], np.min(values, axis=0, initial=np.inf))
        np.testing.assert_allclose(stats.maxs[row], np.max(values, axis=0, initial=-np.inf))
    assert np.isnan(stats.feature_matrix()[[1, 3, 4]]).all()


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

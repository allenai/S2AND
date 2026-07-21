from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.eval as eval_module
import s2and.model as model_module
import s2and.subblocking as subblocking_module
from s2and.data import ANDData, _ordered_coauthors_for_signature
from s2and.eval import incremental_cluster_eval
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.runtime import RuntimeContext
from s2and.sampling import sampling
from tests.helpers import write_minimal_arrow_prediction_bundle


def _as_anddata(dataset: object) -> ANDData:
    if not hasattr(dataset, "runtime_context"):
        cast(Any, dataset).runtime_context = RuntimeContext(
            operation="test_classic_dataset",
            backend="python",
            run_id="test-classic-dataset",
        )
    return cast(ANDData, dataset)


def _subblocking_signature(first_name: str, *, middle_name: str = "", orcid: str | None = None):
    return SimpleNamespace(
        author_info_first_normalized_without_apostrophe=first_name,
        author_info_middle_normalized_without_apostrophe=middle_name,
        author_info_first=first_name,
        author_info_middle=middle_name,
        author_info_orcid=orcid,
    )


def test_ordered_coauthors_rejects_missing_author_position() -> None:
    signature = SimpleNamespace(signature_id="s1", paper_id="p1", author_info_position=None)
    paper = SimpleNamespace(authors=[SimpleNamespace(author_name="Ada Lovelace", position=0)])

    with pytest.raises(ValueError, match="missing author_info_position"):
        _ordered_coauthors_for_signature(cast(Any, signature), {"p1": cast(Any, paper)})


VALID_ORCID_1 = "0000-0001-2345-6789"
VALID_ORCID_2 = "0000-0001-2345-679X"


def test_cacheable_value_preserves_list_order_but_sorts_sets():
    assert model_module._cacheable_value(["year_diff", "name_counts"]) != model_module._cacheable_value(
        ["name_counts", "year_diff"]
    )
    assert model_module._cacheable_value({"year_diff", "name_counts"}) == model_module._cacheable_value(
        {"name_counts", "year_diff"}
    )


def _expected_upper_triangle_pairs_for_range(
    block_size: int,
    start_offset: int,
    max_pairs: int | None,
) -> list[tuple[int, int]]:
    total_pairs = block_size * (block_size - 1) // 2
    count = total_pairs - start_offset if max_pairs is None else min(max_pairs, total_pairs - start_offset)
    row = 0
    remaining_offset = start_offset
    while row < block_size - 1:
        row_len = block_size - row - 1
        if remaining_offset < row_len:
            break
        remaining_offset -= row_len
        row += 1
    col = row + 1 + remaining_offset
    pairs = []
    for _ in range(count):
        pairs.append((row, col))
        col += 1
        if col >= block_size:
            row += 1
            col = row + 1
    return pairs


@pytest.mark.parametrize(
    ("block_size", "start_offset", "max_pairs"),
    [
        (6, 0, 4),
        (6, 1, 4),
        (6, 14, 4),
        (6, 15, 4),
        (2000, 0, 7),
        (2000, 1998, 7),
        (2000, 1999, 7),
        (2000, 999_500, 7),
        (2000, 1_998_995, 7),
        (2000, 1_999_000, 7),
    ],
)
def test_upper_triangle_indices_for_range_matches_row_major_order(
    block_size: int,
    start_offset: int,
    max_pairs: int | None,
):
    left, right = model_module._upper_triangle_indices_for_range(block_size, start_offset, max_pairs)
    assert list(zip(left.tolist(), right.tolist(), strict=True)) == _expected_upper_triangle_pairs_for_range(
        block_size,
        start_offset,
        max_pairs,
    )


def test_python_predicted_batches_use_effective_pair_chunk_size(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        batch_size=10,
    )
    helper_items = [((f"s{i}", f"s{i + 1}", float("nan")), (0, i + 1), "block") for i in range(5)]

    def fake_distance_matrix_helper(self, *_args, **_kwargs):
        del self, _args, _kwargs
        yield from helper_items

    def fake_many_pairs_featurize(pairs, *_args, **_kwargs):
        row_count = len(pairs)
        return np.zeros((row_count, 1), dtype=np.float64), np.zeros(row_count, dtype=np.float64), None

    def fake_predict_and_combine(
        _classifier,
        _nameless_classifier,
        features,
        labels,
        _nameless_features,
        _batch_label,
        **_kwargs,
    ):
        del labels, _kwargs
        return np.arange(len(features), dtype=np.float64), 0.0

    monkeypatch.setattr(Clusterer, "distance_matrix_helper", fake_distance_matrix_helper)
    monkeypatch.setattr(model_module, "many_pairs_featurize", fake_many_pairs_featurize)
    monkeypatch.setattr(model_module, "_predict_and_combine", fake_predict_and_combine)

    batches = list(
        clusterer._iter_python_predicted_distance_matrix_batches(
            {"block": ["s0", "s1", "s2", "s3", "s4", "s5"]},
            dataset,
            {},
            incremental_dont_use_cluster_seeds=False,
            runtime_context=RuntimeContext(
                operation="model_predict",
                backend="python",
                run_id="test-python-batches",
            ),
            num_pairs=len(helper_items),
            pair_chunk_size=2,
        )
    )

    assert [len(batch.predictions) for batch in batches] == [2, 2, 1]


def test_fused_constraint_fallback_resumes_at_failed_offset(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
        batch_size=2,
        use_default_constraints_as_supervision=True,
    )
    signatures = ["s0", "s1", "s2", "s3"]
    signature_index_by_id = {signature_id: idx for idx, signature_id in enumerate(signatures)}

    class FakeRustFeaturizer:
        def get_constraints_block_upper_triangle_indexed(self):
            raise AssertionError("only checked with hasattr")

        def featurize_block_upper_triangle_matrix_indexed(self):
            raise AssertionError("only checked with hasattr")

    backend = model_module._IncrementalConstraintBackend(
        rust_featurizer=FakeRustFeaturizer(),
        use_rust_constraints=True,
        constraint_api_mode="indexed",
        signature_index_by_id=signature_index_by_id,
        suppress_orcid=False,
    )

    def fake_build_backend(*_args, **_kwargs):
        return backend

    def fake_constraints(
        _block_signature_indices,
        *,
        start_offset,
        max_pairs,
        **_kwargs,
    ):
        if start_offset == 0:
            local_i, local_j = model_module._upper_triangle_indices_for_range(4, start_offset, max_pairs)
            return local_i.tolist(), local_j.tolist(), [None] * len(local_i)
        raise RuntimeError("optional fused failure")

    def fake_resolve_constraint_batch(self, _dataset, pair_batch_ids, **_kwargs):
        del self, _dataset, _kwargs
        return [float("nan")] * len(pair_batch_ids), model_module._ConstraintBatchTelemetry(
            total_pairs=len(pair_batch_ids),
            partial_supervision_hits=0,
            unresolved_pairs=len(pair_batch_ids),
            rust_batch_call_count=0,
            api_mode="test",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(model_module, "_build_incremental_constraint_backend", fake_build_backend)
    monkeypatch.setattr(model_module, "get_constraints_block_upper_triangle_indexed_rust", fake_constraints)
    monkeypatch.setattr(model_module, "_handle_optional_rust_exception", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(Clusterer, "_resolve_constraint_batch", fake_resolve_constraint_batch)

    chunks = list(
        clusterer._distance_matrix_chunk_helper_rust(
            {"block": signatures},
            dataset,
            {},
            runtime_context=RuntimeContext(
                operation="constraints",
                backend="rust",
                run_id="test-fused-fallback",
            ),
        )
    )

    assert [chunk.start_offset for chunk in chunks] == [0, 2, 4]


def test_predict_from_arrow_paths_rejects_disallows_with_precomputed_dists_before_build(monkeypatch, tmp_path):
    def fake_build_from_arrow_paths(*_args, **_kwargs):
        raise AssertionError("precomputed dists with disallows should be rejected before Rust featurizer build")

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_from_arrow_paths)

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
    )
    dists = {"block": np.asarray([0.5], dtype=np.float64)}
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path)
    with pytest.raises(ValueError, match="cluster_seeds_disallow cannot be used with precomputed dists"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            arrow_paths,
            dists=dists,
            cluster_seeds_disallow={("s0", "s1")},
        )


@pytest.mark.parametrize("bad_path", ["", "   ", Path()])
def test_predict_from_arrow_paths_rejects_empty_path_before_rust_builder(monkeypatch, bad_path):
    def fail_build_from_arrow_paths(*_args, **_kwargs):
        raise AssertionError("invalid Arrow paths should be rejected before Rust featurizer build")

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fail_build_from_arrow_paths)

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
    )

    with pytest.raises(ValueError, match="signatures"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            {"signatures": bad_path, "papers": "papers.arrow", "paper_authors": "paper_authors.arrow"},
        )


def test_rust_featurizer_distance_matrix_guards_allocation_before_matrix_build():
    class FakeRustFeaturizer:
        def signature_ids(self):
            return [str(i) for i in range(100)]

        def get_constraints_block_upper_triangle_indexed(self, *_args, **_kwargs):
            raise AssertionError("guard should run before constraint evaluation")

        def featurize_block_upper_triangle_matrix_indexed(self, *_args, **_kwargs):
            raise AssertionError("guard should run before feature evaluation")

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
    )

    with pytest.raises(MemoryError, match="Predict exact block exceeds memory budget"):
        clusterer.make_distance_matrices_from_rust_featurizer(
            {"block": [str(i) for i in range(100)]},
            FakeRustFeaturizer(),
            total_ram_bytes=1,
        )


def test_make_distance_matrices_guards_allocation_before_pair_featurization(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        n_jobs=1,
    )

    monkeypatch.setattr(
        model_module,
        "build_runtime_context",
        lambda _operation: RuntimeContext(
            operation="model_predict",
            backend="python",
            run_id="test-matrix-guard",
        ),
    )
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model_module,
        "many_pairs_featurize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("guard should run before pair featurization")),
    )

    with pytest.raises(MemoryError, match="Predict exact block exceeds memory budget"):
        clusterer.make_distance_matrices(
            {"block": [str(i) for i in range(100)]},
            dataset,
            total_ram_bytes=1,
        )


def test_residual_first_initial_groups_union_normalized_orcids():
    dataset = SimpleNamespace(
        signatures={
            "s1": _subblocking_signature("alice", orcid="0000-0000-0000-0001"),
            "s2": _subblocking_signature("bob", orcid="0000000000000001"),
            "s3": _subblocking_signature("carol", orcid=None),
        }
    )
    clusterer = SimpleNamespace(use_default_constraints_as_supervision=True, suppress_orcid=False)

    groups = model_module._residual_phase_b_first_initial_groups(clusterer, dataset, ["s1", "s2", "s3"], {})

    assert {frozenset(group) for group in groups} == {frozenset({"s1", "s2"}), frozenset({"s3"})}


def test_residual_first_initial_groups_rejects_whitespace_only_first_initials():
    assert model_module._signature_first_initials_for_rules("  ") == frozenset()

    dataset = SimpleNamespace(
        signatures={
            "s1": SimpleNamespace(
                author_info_first_normalized_without_apostrophe="",
                author_info_first="  ",
                author_info_orcid=None,
            ),
            "s2": SimpleNamespace(
                author_info_first_normalized_without_apostrophe="",
                author_info_first="\t ",
                author_info_orcid=None,
            ),
            "s3": _subblocking_signature("alice", orcid=None),
        }
    )
    clusterer = SimpleNamespace(use_default_constraints_as_supervision=True, suppress_orcid=True)

    groups = model_module._residual_phase_b_first_initial_groups(clusterer, dataset, ["s1", "s2", "s3"], {})

    assert groups == [["s1", "s2", "s3"]]


def test_sampling_balanced_homonym_synonym_respects_sample_size():
    all_pairs = [
        ("a", "b", 0),
        ("c", "d", 1),
        ("e", "f", 1),
        ("g", "h", 0),
    ]
    sampled = sampling(
        same_name_different_cluster=[all_pairs[0]],
        different_name_same_cluster=[all_pairs[1]],
        same_name_same_cluster=[all_pairs[2]],
        different_name_different_cluster=[all_pairs[3]],
        sample_size=1,
        balanced_homonyms_and_synonyms=True,
        random_seed=3,
    )
    assert len(sampled) == 1
    assert sampled[0] in all_pairs


def test_incremental_cluster_eval_val_uses_val_block_for_pairwise_metrics(monkeypatch):
    class DummyDataset:
        def __init__(self):
            self.signature_to_cluster_id = {"s_train": "c_train", "s_val": "c_val", "s_test": "c_test"}

        def get_blocks(self):
            return {"b": ["s_train", "s_val", "s_test"]}

        def split_cluster_signatures(self):
            return {"b": ["s_train"]}, {"b": ["s_val"]}, {"b": ["s_test"]}

        def construct_cluster_to_signatures(self, block_dict):
            output = {}
            for signatures in block_dict.values():
                for signature in signatures:
                    cluster_id = self.signature_to_cluster_id[signature]
                    output.setdefault(cluster_id, []).append(signature)
            return output

    class DummyClusterer:
        def predict(self, block_dict, dataset, partial_supervision=None):
            all_signatures = []
            for signatures in block_dict.values():
                all_signatures.extend(signatures)
            return {"pred_cluster": all_signatures}, None

    captured_test_blocks = []

    def fake_pairwise_precision_recall_fscore(true_clus, pred_clus, test_block, strategy="clusters"):
        captured_test_blocks.append(test_block)
        return 0.0, 0.0, 0.0

    monkeypatch.setattr(eval_module, "pairwise_precision_recall_fscore", fake_pairwise_precision_recall_fscore)

    dataset = DummyDataset()
    clusterer = DummyClusterer()
    incremental_cluster_eval(cast(ANDData, dataset), cast(Clusterer, clusterer), split="val")

    assert len(captured_test_blocks) == 2
    assert captured_test_blocks[0] == {"b": ["s_val"]}
    assert captured_test_blocks[1] == {"b": ["s_val"]}


def test_make_subblocks_handles_specter_edge_case_without_unbound_local(monkeypatch):
    class Signature:
        def __init__(self, first_name, middle_name, orcid=None):
            self.author_info_first = first_name
            self.author_info_middle = middle_name
            self.author_info_first_normalized_without_apostrophe = first_name
            self.author_info_middle_normalized_without_apostrophe = middle_name
            self.author_info_orcid = orcid

    anddata = SimpleNamespace(signatures={"s1": Signature("ab", "cd")})

    call_count = {"value": 0}

    def fake_subdivide_helper(names, sig_ids, maximum_size, starting_k=2):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return {}, {"ab": np.array(["s1"])}
        if call_count["value"] == 2:
            return {}, {"cd": np.array(["s1"])}
        raise AssertionError("Unexpected extra call to subdivide_helper")

    monkeypatch.setattr(subblocking_module, "subdivide_helper", fake_subdivide_helper)
    monkeypatch.setattr(subblocking_module, "cluster_with_specter", lambda *args, **kwargs: {"0": ["s1"]})

    output = subblocking_module.make_subblocks(["s1"], anddata, maximum_size=2, first_k_letter_counts_sorted={})
    assert output == {"ab|middle=cd": ["s1"]}


def test_clusterer_predict_does_not_forward_batch_threshold_to_python_incremental(monkeypatch):
    class Signature:
        def __init__(self, first_name):
            self.author_info_first_normalized_without_apostrophe = first_name

    dataset = _as_anddata(
        SimpleNamespace(
            signatures={
                "m1": Signature("alex"),
                "m2": Signature("alex"),
                "m3": Signature("alex"),
                "m4": Signature("alex"),
                "m5": Signature("alex"),
                "m6": Signature("alex"),
                "s1": Signature("a"),
                "s2": Signature("a"),
            },
            cluster_seeds_require={},
        )
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1)

    monkeypatch.setattr(
        model_module,
        "_sync_rust_cluster_seeds",
        lambda _dataset, runtime_context=None: None,
    )
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda block_signatures, _dataset, maximum_size, **_kwargs: {
            "multi_1": ["m1", "m2"],
            "multi_2": ["m3", "m4"],
            "multi_3": ["m5", "m6"],
            "single_1": ["s1", "s2"],
        },
    )

    def fake_predict_helper(self, block_dict, _dataset, *args, **kwargs):
        predicted = {}
        for block_key, signatures in block_dict.items():
            predicted[f"cluster_{block_key}"] = list(signatures)
        return predicted, None

    captured_kwargs = {}
    incremental_calls: list[tuple[str, ...]] = []

    def fake_predict_incremental(self, block_signatures, dataset, *args, **kwargs):
        incremental_calls.append(tuple(block_signatures))
        captured_kwargs.update(kwargs)
        return {
            "clusters": {"merged": list(dataset.cluster_seeds_require.keys()) + list(block_signatures)},
            "phase_b_mode": "exact",
            "phase_b_budget_bytes": 0,
            "phase_b_required_bytes": 0,
        }

    monkeypatch.setattr(Clusterer, "predict_helper", fake_predict_helper)
    monkeypatch.setattr(Clusterer, "predict_incremental", fake_predict_incremental)

    clusterer.predict(
        {"block": ["m1", "m2", "m3", "m4", "m5", "m6", "s1", "s2"]},
        dataset,
        batching_threshold=2,
    )

    assert incremental_calls == [("s1", "s2")]
    assert "batching_threshold" not in captured_kwargs


def test_sync_rust_cluster_seeds_skips_unchanged_and_detects_mutations(monkeypatch):
    calls = {"count": 0}

    def fake_update(_dataset, runtime_context=None, **_kwargs):
        del runtime_context
        calls["count"] += 1

    monkeypatch.setattr(model_module, "update_rust_cluster_seeds", fake_update)

    dataset = _as_anddata(
        SimpleNamespace(
            cluster_seeds_require={"s1": "c1", "s2": "c1"},
            cluster_seeds_disallow={("s1", "s3")},
            _cluster_seeds_version=1,
            arrow_paths={"signatures": "mock-signatures.arrow"},
        )
    )
    runtime_context = RuntimeContext(
        operation="constraints",
        backend="rust",
        run_id="run-1",
    )

    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    assert calls["count"] == 1

    dataset.cluster_seeds_require["s2"] = "c2"
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    assert calls["count"] == 2

    dataset.cluster_seeds_disallow.remove(("s1", "s3"))
    dataset.cluster_seeds_disallow.add(("s2", "s3"))
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    assert calls["count"] == 3

    dataset._cluster_seeds_version += 1
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context)
    assert calls["count"] == 4


def test_make_distance_matrices_fastcluster_cross_batch_preserves_per_block_order(monkeypatch):
    dataset = _as_anddata(SimpleNamespace(cluster_seeds_require={}, cluster_seeds_disallow=set()))
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=None,
        n_jobs=1,
        batch_size=2,
    )

    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(model_module, "stage_uses_rust", lambda _runtime_context: False)

    batches = [
        model_module._PredictedDistanceMatrixBatch(
            batch_num=0,
            blocks=["a", "a"],
            indices=[(0, 1), (0, 2)],
            predictions=np.asarray([0.1, 0.2], dtype=np.float64),
            batch_seconds=0.0,
        ),
        model_module._PredictedDistanceMatrixBatch(
            batch_num=1,
            blocks=["b", "a"],
            indices=[(0, 1), (0, 3)],
            predictions=np.asarray([9.9, 0.3], dtype=np.float64),
            batch_seconds=0.0,
        ),
        model_module._PredictedDistanceMatrixBatch(
            batch_num=2,
            blocks=["a", "a", "a"],
            indices=[(1, 2), (1, 3), (2, 3)],
            predictions=np.asarray([0.4, 0.5, 0.6], dtype=np.float64),
            batch_seconds=0.0,
        ),
    ]

    def fake_iter_python_batches(self, *_args, **_kwargs):
        del self, _args, _kwargs
        yield from batches

    monkeypatch.setattr(
        model_module.Clusterer,
        "_iter_python_predicted_distance_matrix_batches",
        fake_iter_python_batches,
    )

    output = clusterer.make_distance_matrices(
        {"a": ["s1", "s2", "s3", "s4"], "b": ["t1", "t2"]},
        dataset,
        partial_supervision={},
    )

    expected_a = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    expected_b = np.asarray([9.9], dtype=np.float64)
    np.testing.assert_array_equal(output["a"], expected_a.astype(output["a"].dtype))
    np.testing.assert_array_equal(output["b"], expected_b.astype(output["b"].dtype))


def test_propagate_n_jobs_re_raises_unexpected_set_params_error():
    class _ExplodingEstimator:
        def set_params(self, **_kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        model_module._propagate_n_jobs(_ExplodingEstimator(), 4)

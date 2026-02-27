import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import s2and.eval as eval_module
import s2and.model as model_module
import s2and.subblocking as subblocking_module
from s2and.eval import incremental_cluster_eval
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.runtime import RuntimeContext
from s2and.sampling import sampling

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    incremental_cluster_eval(dataset, clusterer, split="val")

    assert len(captured_test_blocks) == 2
    assert captured_test_blocks[0] == {"b": ["s_val"]}
    assert captured_test_blocks[1] == {"b": ["s_val"]}


def test_make_subblocks_handles_specter_edge_case_without_unbound_local(monkeypatch):
    class Signature:
        def __init__(self, first_name, middle_name, orcid=None):
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


def test_transform_signature_file_handles_empty_email_field(tmp_path):
    transform_module = _load_script_module(
        "scripts/archive/transform_all_datasets.py",
        "transform_all_datasets_regression",
    )

    signatures = {
        "1": {
            "paperid": 1,
            "authorinfo": {
                "position": 0,
                "block": "a smith",
                "first": "A",
                "middle": "",
                "last": "Smith",
                "suffix": "",
                "emails": "{}",
                "affiliations": "",
                "given-block": "",
                "ethnicity": "",
                "gender": "",
            },
            "actual_name": "",
        }
    }
    input_path = tmp_path / "signatures.json"
    with open(input_path, "w") as input_file:
        json.dump(signatures, input_file)

    transformed = transform_module.transform_signature_file(str(input_path))
    assert transformed["1"]["author_info"]["email"] is None


@pytest.mark.parametrize(
    ("script_path", "module_name"),
    [
        ("scripts/transfer_experiment_seed_paper.py", "transfer_seed_regression"),
        ("scripts/internal/transfer_experiment_internal.py", "transfer_internal_regression"),
        ("scripts/custom_block_transfer_experiment_seed_paper.py", "transfer_custom_regression"),
    ],
)
def test_transfer_script_facet_helpers_handle_empty_groups(script_path, module_name):
    module = _load_script_module(script_path, module_name)

    disparity_scores = module.disparity_analysis({"group": []}, {"group": []})
    assert math.isnan(disparity_scores["S2AND std"])
    assert disparity_scores["S2AND max-perf-group"] is None

    empty_feature = ([], [])
    s2and_scores, s2_scores = module.summary_features_analysis(
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
        empty_feature,
    )
    assert math.isnan(s2and_scores["first_name_diff"])
    assert math.isnan(s2_scores["first_name_diff"])


def test_clusterer_predict_uses_minimum_one_for_incremental_batch_threshold(monkeypatch):
    class Signature:
        def __init__(self, first_name):
            self.author_info_first_normalized_without_apostrophe = first_name

    dataset = SimpleNamespace(
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

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(featurizer_info=featurizer_info, classifier=None, n_jobs=1, use_cache=False)

    monkeypatch.setattr(
        model_module,
        "_sync_rust_cluster_seeds",
        lambda _dataset, runtime_context=None, use_cache=False: None,
    )
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda block_signatures, _dataset, maximum_size: {
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

    captured = {"batching_threshold": None}

    def fake_predict_incremental(self, block_signatures, dataset, *args, **kwargs):
        captured["batching_threshold"] = kwargs["batching_threshold"]
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
        desired_memory_use=4,
    )

    assert captured["batching_threshold"] == 1


def test_distance_matrix_helper_uses_indexed_constraint_api(monkeypatch):
    dataset = SimpleNamespace()
    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=None,
        n_jobs=1,
        use_cache=False,
        use_default_constraints_as_supervision=True,
    )

    class _FakeFeaturizer:
        def signature_ids(self):
            return ["s1", "s2"]

        def get_constraints_matrix_indexed(self, *_args, **_kwargs):
            return [None]

    captured = {
        "featurizer_use_cache": None,
        "batch_use_cache": None,
        "batch_calls": 0,
        "fallback_calls": 0,
    }

    monkeypatch.setattr(model_module, "_use_rust_constraints", lambda runtime_context=None: True)
    monkeypatch.setattr(
        model_module,
        "_get_rust_featurizer",
        lambda _dataset, runtime_context=None, use_cache=False: captured.__setitem__("featurizer_use_cache", use_cache)
        or _FakeFeaturizer(),
    )

    def fake_get_constraints_matrix_indexed_rust(*args, use_cache=False, **kwargs):
        del args, kwargs
        captured["batch_calls"] += 1
        captured["batch_use_cache"] = use_cache
        return [0.0]

    def fake_get_constraint_rust(*args, **kwargs):
        del args, kwargs
        captured["fallback_calls"] += 1
        return None

    monkeypatch.setattr(model_module, "get_constraints_matrix_indexed_rust", fake_get_constraints_matrix_indexed_rust)
    monkeypatch.setattr(model_module, "get_constraint_rust", fake_get_constraint_rust)

    helper = clusterer.distance_matrix_helper({"b": ["s1", "s2"]}, dataset, partial_supervision={})
    next(helper)

    assert captured["featurizer_use_cache"] is False
    assert captured["batch_use_cache"] is False
    assert captured["batch_calls"] == 1
    assert captured["fallback_calls"] == 0


def test_sync_rust_cluster_seeds_skips_when_unchanged(monkeypatch):
    calls = {"count": 0}

    def fake_update(_dataset, runtime_context=None, use_cache=False):
        del runtime_context, use_cache
        calls["count"] += 1

    monkeypatch.setattr(model_module, "update_rust_cluster_seeds", fake_update)

    dataset = SimpleNamespace(
        cluster_seeds_require={},
        cluster_seeds_disallow=set(),
        _cluster_seeds_version=1,
    )
    runtime_context = RuntimeContext(
        operation="constraints",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="run-1",
        source="default",
    )

    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=False)
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=False)
    assert calls["count"] == 1

    dataset._cluster_seeds_version += 1
    model_module._sync_rust_cluster_seeds(dataset, runtime_context=runtime_context, use_cache=False)
    assert calls["count"] == 2

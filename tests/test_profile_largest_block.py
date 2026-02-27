from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from s2and.consts import PROJECT_ROOT_PATH
from s2and.eval import cluster_precision_recall_fscore


def _load_module():
    module_path = Path(PROJECT_ROOT_PATH) / "scripts" / "rust_suite.py"
    spec = importlib.util.spec_from_file_location("rust_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cluster_membership_digest_is_label_invariant():
    module = _load_module()
    clusters_a = {"0": ["s1", "s2"], "1": ["s3"]}
    clusters_b = {"x": ["s3"], "y": ["s2", "s1"]}
    assert module._cluster_membership_digest(clusters_a) == module._cluster_membership_digest(clusters_b)


def test_signature_to_cluster_fingerprint_map_is_label_invariant():
    module = _load_module()
    clusters_a = {"0": ["s1", "s2"], "1": ["s3"]}
    clusters_b = {"x": ["s3"], "y": ["s2", "s1"]}
    assert module._signature_to_cluster_fingerprint_map(clusters_a) == module._signature_to_cluster_fingerprint_map(
        clusters_b
    )


def test_pairwise_precision_recall_matches_eval_singleton_fix():
    module = _load_module()
    true_clusters = {"t0": ["s1", "s2"], "t1": ["s3"], "t2": ["s4", "s5", "s6"]}
    pred_clusters = {"p0": ["s1", "s3"], "p1": ["s2"], "p2": ["s4", "s5", "s6"]}

    expected = cluster_precision_recall_fscore(true_clusters, pred_clusters)
    got = module._pairwise_precision_recall_fscore_with_singleton_fix(true_clusters, pred_clusters)
    assert got == tuple(round(x, 3) for x in expected)


def test_pairwise_precision_recall_rejects_signature_set_mismatch():
    module = _load_module()
    true_clusters = {"t0": ["s1", "s2"]}
    pred_clusters = {"p0": ["s1"]}
    with pytest.raises(ValueError, match="cover all the signatures"):
        module._pairwise_precision_recall_fscore_with_singleton_fix(true_clusters, pred_clusters)

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from s2and.consts import PROJECT_ROOT_PATH


def _load_module():
    module_path = Path(PROJECT_ROOT_PATH) / "scripts" / "rust_suite.py"
    spec = importlib.util.spec_from_file_location("rust_suite", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_effective_seed_cluster_count_caps_to_half():
    module = _load_module()
    assert module._effective_seed_cluster_count(10, 20) == 5
    assert module._effective_seed_cluster_count(10, 3) == 3


def test_build_cluster_seeds_covers_all_seed_signatures():
    module = _load_module()
    seed_signature_ids = [f"s{i}" for i in range(12)]
    cluster_seeds = module._build_cluster_seeds(seed_signature_ids, seed_cluster_count=4)

    covered = set()
    for root, members in cluster_seeds.items():
        assert len(members) >= 1
        covered.add(root)
        covered.update(members.keys())

    assert covered == set(seed_signature_ids)
    assert len(cluster_seeds) == 4


def test_cluster_membership_digest_is_label_invariant():
    module = _load_module()
    clusters_a = {"0": ["s1", "s2"], "1": ["s3"]}
    clusters_b = {"x": ["s3"], "y": ["s2", "s1"]}
    assert module._cluster_membership_digest(clusters_a) == module._cluster_membership_digest(clusters_b)


def test_validate_args_requires_full_run_for_large_signature_count():
    module = _load_module()
    args = argparse.Namespace(
        total_signatures=5001,
        seed_signatures=3000,
        seed_cluster_count=500,
        n_jobs=4,
        batching_threshold=1000,
        full_run=False,
    )
    with pytest.raises(ValueError, match="--full-run"):
        module._validate_args(args)


def test_paper_block_safety_rejects_empty_author_names():
    module = _load_module()
    assert module._paper_has_block_safe_author_names({"authors": [{"author_name": "Jane Doe"}]}) is True
    assert module._paper_has_block_safe_author_names({"authors": [{"author_name": ""}]}) is False

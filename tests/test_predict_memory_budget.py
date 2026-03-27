from __future__ import annotations

import numpy as np
import pytest
from lightgbm import LGBMClassifier

import s2and.model as model_module
from s2and import memory_budget
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer


def _build_dummy_clusterer_and_dataset(*, name: str = "dummy_predict_memory") -> tuple[Clusterer, ANDData]:
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        cluster_seeds="tests/dummy/cluster_seeds.json",
        name=name,
        load_name_counts=True,
    )

    featurizer_info = FeaturizationInfo(features_to_use=["year_diff", "misc_features"])
    rng = np.random.RandomState(1)
    x_random = rng.random((10, 6))
    y_random = rng.randint(0, 6, 10)
    clusterer = Clusterer(
        featurizer_info=featurizer_info,
        classifier=LGBMClassifier(random_state=1, data_random_seed=1, feature_fraction_seed=1).fit(x_random, y_random),
        n_jobs=1,
        use_cache=False,
        use_default_constraints_as_supervision=False,
    )
    return clusterer, dataset


def _snapshot(*, available_bytes: int, total_ram_bytes: int = 1_000) -> memory_budget.MemorySnapshot:
    return memory_budget.MemorySnapshot(
        total_ram_bytes=total_ram_bytes,
        total_ram_source="test",
        current_rss_bytes=100,
        current_rss_source="rss:test",
        safety_margin_bytes=100,
        available_bytes=available_bytes,
        effective_available_fraction=float(available_bytes) / float(total_ram_bytes),
    )


def test_predict_helper_raises_before_matrix_allocation_when_budget_too_small(monkeypatch):
    clusterer, dataset = _build_dummy_clusterer_and_dataset(name="dummy_predict_memory_too_small")
    block = {"a sattar": ["0", "1", "2"]}

    monkeypatch.setattr(
        model_module.memory_budget,
        "memory_snapshot_for_stage",
        lambda **_kwargs: _snapshot(available_bytes=16),
    )

    with pytest.raises(MemoryError, match="Predict exact block exceeds memory budget before matrix allocation"):
        clusterer.predict_helper(block, dataset, total_ram_bytes=1_000)


def test_predict_helper_matches_baseline_when_budget_allows(monkeypatch):
    clusterer, dataset = _build_dummy_clusterer_and_dataset(name="dummy_predict_memory_large_budget")
    block = {"a sattar": ["0", "1", "2", "3", "4", "5", "6", "7", "8"]}

    baseline_clusters, baseline_dists = clusterer.predict_helper(block, dataset)
    assert baseline_dists is None

    monkeypatch.setattr(
        model_module.memory_budget,
        "memory_snapshot_for_stage",
        lambda **_kwargs: _snapshot(available_bytes=10_000_000, total_ram_bytes=20_000_000),
    )
    budgeted_clusters, budgeted_dists = clusterer.predict_helper(block, dataset, total_ram_bytes=20_000_000)

    assert budgeted_dists is None
    assert budgeted_clusters == baseline_clusters

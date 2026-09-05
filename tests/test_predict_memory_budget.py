from __future__ import annotations

import pytest

import s2and.model as model_module
from s2and import memory_budget
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from tests.helpers import build_dummy_dataset
from tests.model_helpers import ConstantDistanceClassifier


def _build_dummy_clusterer_and_dataset(*, name: str = "dummy_predict_memory") -> tuple[Clusterer, ANDData]:
    clusterer = Clusterer(
        FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        ConstantDistanceClassifier(),
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )
    return clusterer, build_dummy_dataset(name, name_counts_index=True)


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


@pytest.mark.parametrize("route", ["make_distance_matrices", "predict_helper"])
def test_fastcluster_matrix_budget_uses_float64(monkeypatch, route):
    """Stored Python matrices must budget eight bytes for each condensed pair."""
    clusterer, dataset = _build_dummy_clusterer_and_dataset(name="stored_matrix_precision_budget")
    block = {"a sattar": ["0", "1", "2"]}
    monkeypatch.setattr(
        model_module.memory_budget,
        "memory_snapshot_for_stage",
        lambda **_kwargs: _snapshot(available_bytes=16),
    )

    with pytest.raises(MemoryError, match="matrix_bytes=24 available_bytes=16"):
        getattr(clusterer, route)(block, dataset, total_ram_bytes=1_000)


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

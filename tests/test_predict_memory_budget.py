from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import s2and.model as model_module
from s2and import memory_budget
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.production_model import NativeLightGBMBinaryClassifier
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


def test_python_calibration_bounds_features_and_native_scoring_without_changing_distances(monkeypatch):
    """Cached calibration and streamed prediction share the caller's RAM budget."""
    info = FeaturizationInfo()
    classifier = NativeLightGBMBinaryClassifier(
        Path(__file__).parent / "fixtures" / "rust_lightgbm" / "production_main.lgb",
        n_jobs=1,
    )
    clusterer = Clusterer(
        info,
        classifier,
        nameless_classifier=classifier,
        nameless_featurizer_info=info,
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )
    dataset = build_dummy_dataset("calibration_memory_budget", name_counts_index=True)
    clusterer.feature_contract["name_counts_manifest_sha256"] = dataset.name_counts_manifest_sha256
    blocks = {"a sattar": [str(index) for index in range(9)]}
    baseline = clusterer.make_distance_matrices(blocks, dataset, disable_tqdm=True)
    assert np.ptp(baseline["a sattar"]) > 0

    total_ram_bytes = 20_000_000
    feature_batches: list[tuple[int, int | None]] = []
    scorer_batches: list[tuple[int, int | None]] = []
    real_featurize = model_module.many_pairs_featurize
    real_scorer_plan = memory_budget.compute_native_scorer_chunk_plan

    def record_features(pairs, *args: Any, **kwargs: Any):
        feature_batches.append((len(pairs), kwargs.get("total_ram_bytes")))
        return real_featurize(pairs, *args, **kwargs)

    def record_scorer_plan(**kwargs: Any):
        scorer_batches.append((kwargs["row_count"], kwargs.get("total_ram_bytes")))
        return real_scorer_plan(**kwargs)

    monkeypatch.setattr(
        memory_budget,
        "memory_snapshot_for_stage",
        lambda **_kwargs: _snapshot(available_bytes=20_000, total_ram_bytes=total_ram_bytes),
    )
    monkeypatch.setattr(model_module, "many_pairs_featurize", record_features)
    monkeypatch.setattr(memory_budget, "compute_native_scorer_chunk_plan", record_scorer_plan)

    distances = clusterer.make_distance_matrices(blocks, dataset, disable_tqdm=True, total_ram_bytes=total_ram_bytes)

    assert feature_batches == [(1, total_ram_bytes)] * 36
    assert scorer_batches == [(1, total_ram_bytes)] * 72
    np.testing.assert_array_equal(distances["a sattar"], baseline["a sattar"])
    cached_clusters, _ = clusterer.predict_helper(blocks, dataset, dists=distances)

    streamed_matrices: list[np.ndarray] = []
    real_cluster_block = clusterer._cluster_one_block_with_logging

    def record_cluster_block(signature_ids, matrix, *args: Any, **kwargs: Any):
        streamed_matrices.append(matrix.copy())
        return real_cluster_block(signature_ids, matrix, *args, **kwargs)

    monkeypatch.setattr(clusterer, "_cluster_one_block_with_logging", record_cluster_block)
    streamed_clusters, _ = clusterer.predict_helper(blocks, dataset, total_ram_bytes=total_ram_bytes)

    assert feature_batches == [(1, total_ram_bytes)] * 72
    assert scorer_batches == [(1, total_ram_bytes)] * 144
    assert len(streamed_matrices) == 1
    np.testing.assert_array_equal(streamed_matrices[0], distances["a sattar"])
    assert streamed_clusters == cached_clusters

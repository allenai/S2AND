from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import s2and.runtime as runtime
from s2and.arrow_inputs import ArrowDataset, MissingArrowArtifactError
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from tests.helpers import write_minimal_arrow_prediction_bundle


def _year_diff_clusterer() -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
        batch_size=2,
    )


def test_filtered_arrow_prediction_validates_requested_profiles_and_subblocking(tmp_path: Path) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path)
    with ArrowDataset.open(tmp_path) as arrow_dataset:
        blocks = {"block": ["s0", "s1"]}
        name_counts_clusterer = Clusterer(
            FeaturizationInfo(["name_counts"]),
            classifier=None,
            n_jobs=1,
        )
        with pytest.raises(ValueError, match="load_name_counts=False"):
            name_counts_clusterer.predict_from_arrow(blocks, arrow_dataset, load_name_counts=False)

        clusterer = _year_diff_clusterer()
        with pytest.raises(ValueError, match="batching_threshold must be positive"):
            clusterer.predict_from_arrow(blocks, arrow_dataset, batching_threshold=0)
        with pytest.raises(ValueError, match="cannot be used with precomputed dists"):
            clusterer.predict_from_arrow(blocks, arrow_dataset, batching_threshold=1, dists={"block": np.zeros((2, 2))})
        with pytest.raises(MissingArrowArtifactError) as exc_info:
            clusterer.predict_from_arrow(blocks, arrow_dataset, batching_threshold=1)
        assert "specter" in exc_info.value.required_keys


def test_classic_predict_keeps_rust_and_incremental_routes_out() -> None:
    clusterer = _year_diff_clusterer()
    runtime_context = runtime.RuntimeContext(
        operation="cluster_predict",
        backend="rust",
        run_id="test-explicit-routing",
    )

    with pytest.raises(ValueError, match="predict_from_arrow"):
        clusterer.predict(
            {"block": ["s1"]},
            SimpleNamespace(name="json_dataset"),  # type: ignore[arg-type]
            runtime_context=runtime_context,
        )

    assert not hasattr(clusterer, "predict_incremental")
    assert not hasattr(clusterer, "_predict_incremental_python")
    for name in (
        "incremental_precluster_broadcast_mode",
        "incremental_seed_score_mode",
        "incremental_mean_min_hybrid_weight",
    ):
        assert not hasattr(clusterer, name)
    with pytest.raises(ValueError, match="retired; use predict_from_arrow"):
        clusterer.predict({}, None, batching_threshold=10)

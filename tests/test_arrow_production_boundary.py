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


def test_filtered_arrow_prediction_rejects_disabling_required_name_counts(tmp_path: Path) -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
    )
    write_minimal_arrow_prediction_bundle(tmp_path)
    arrow_dataset = ArrowDataset.open(tmp_path)

    with pytest.raises(
        ValueError,
        match="cannot run with load_name_counts=False when the clusterer selects name_counts features",
    ):
        clusterer.predict_from_arrow(
            {"block": ["s1"]},
            arrow_dataset,
            load_name_counts=False,
        )


def test_filtered_arrow_prediction_rejects_invalid_subblocking_arguments(tmp_path: Path) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path)
    arrow_dataset = ArrowDataset.open(tmp_path)
    cases = (
        ("zero-threshold", 0, None, "batching_threshold must be positive"),
        ("negative-threshold", -1, None, "batching_threshold must be positive"),
        ("precomputed-dists", 1, {"block": np.zeros((2, 2))}, "cannot be used with precomputed dists"),
    )
    for _case_id, batching_threshold, dists, message in cases:
        with pytest.raises(ValueError, match=message):
            _year_diff_clusterer().predict_from_arrow(
                {"block": ["s0", "s1"]},
                arrow_dataset,
                dists=dists,
                batching_threshold=batching_threshold,
            )


def test_filtered_arrow_prediction_requires_specter_for_subblocking(tmp_path: Path) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path)
    arrow_dataset = ArrowDataset.open(tmp_path)
    with pytest.raises(MissingArrowArtifactError) as exc_info:
        _year_diff_clusterer().predict_from_arrow(
            {"block": ["s0", "s1"]},
            arrow_dataset,
            batching_threshold=1,
        )

    assert "specter" in exc_info.value.required_keys


def test_classic_predict_rejects_rust_context() -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
        batch_size=2,
    )
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


def test_classic_incremental_rejects_rust_context() -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
        batch_size=2,
    )
    runtime_context = runtime.RuntimeContext(
        operation="cluster_predict_incremental",
        backend="rust",
        run_id="test-explicit-routing",
    )

    with pytest.raises(ValueError, match="predict_incremental_from_arrow"):
        clusterer.predict_incremental(
            ["s1"],
            SimpleNamespace(name="json_dataset"),  # type: ignore[arg-type]
            runtime_context=runtime_context,
        )

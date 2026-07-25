from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import s2and.runtime as runtime
from s2and.arrow_inputs import MissingArrowArtifactError
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


def _unindexed_arrow_paths(tmp_path: Path) -> dict[str, str]:
    paths = {
        "signatures": tmp_path / "signatures.arrow",
        "papers": tmp_path / "papers.arrow",
        "paper_authors": tmp_path / "paper_authors.arrow",
    }
    for path in paths.values():
        path.touch()
    return {key: str(path) for key, path in paths.items()}


def test_filtered_arrow_prediction_rejects_unindexed_input(tmp_path: Path) -> None:
    clusterer = _year_diff_clusterer()

    with pytest.raises(MissingArrowArtifactError, match="batch_index"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s1"]},
            _unindexed_arrow_paths(tmp_path),
        )


def test_incremental_arrow_prediction_rejects_unindexed_input(tmp_path: Path) -> None:
    clusterer = _year_diff_clusterer()

    with pytest.raises(MissingArrowArtifactError, match="batch_index"):
        clusterer.predict_incremental_from_arrow_paths(
            ["s1"],
            _unindexed_arrow_paths(tmp_path),
        )


def test_filtered_arrow_prediction_rejects_disabling_required_name_counts() -> None:
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["name_counts"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
    )

    with pytest.raises(
        ValueError,
        match="cannot run with load_name_counts=False when the clusterer selects name_counts features",
    ):
        clusterer.predict_from_arrow_paths(
            {"block": ["s1"]},
            {},
            load_name_counts=False,
        )


@pytest.mark.parametrize(
    ("batching_threshold", "dists", "message"),
    [
        (0, None, "batching_threshold must be positive"),
        (-1, None, "batching_threshold must be positive"),
        (1, {"block": np.zeros((2, 2))}, "batching_threshold cannot be used with precomputed dists"),
    ],
)
def test_filtered_arrow_prediction_rejects_invalid_subblocking_arguments(
    batching_threshold: int,
    dists: dict[str, np.ndarray] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _year_diff_clusterer().predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            {},
            dists=dists,
            batching_threshold=batching_threshold,
        )


def test_filtered_arrow_prediction_requires_specter_for_subblocking(tmp_path: Path) -> None:
    with pytest.raises(MissingArrowArtifactError) as exc_info:
        _year_diff_clusterer().predict_from_arrow_paths(
            {"block": ["s0", "s1"]},
            write_minimal_arrow_prediction_bundle(tmp_path),
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

    with pytest.raises(ValueError, match="predict_from_arrow_paths"):
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

    with pytest.raises(ValueError, match="predict_incremental_from_arrow_paths"):
        clusterer.predict_incremental(
            ["s1"],
            SimpleNamespace(name="json_dataset"),  # type: ignore[arg-type]
            runtime_context=runtime_context,
        )

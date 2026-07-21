from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import s2and.runtime as runtime
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer


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
    from s2and.arrow_inputs import MissingArrowArtifactError

    clusterer = _year_diff_clusterer()

    with pytest.raises(MissingArrowArtifactError, match="batch_index"):
        clusterer.predict_from_arrow_paths(
            {"block": ["s1"]},
            _unindexed_arrow_paths(tmp_path),
        )


def test_incremental_arrow_prediction_rejects_unindexed_input(tmp_path: Path) -> None:
    from s2and.arrow_inputs import MissingArrowArtifactError

    clusterer = _year_diff_clusterer()

    with pytest.raises(MissingArrowArtifactError, match="batch_index"):
        clusterer.predict_incremental_from_arrow_paths(
            ["s1"],
            _unindexed_arrow_paths(tmp_path),
        )


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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import s2and.feature_port as feature_port
import s2and.runtime as runtime
from s2and.arrow_inputs import ValidatedArrowInputs
from s2and.consts import NORMALIZATION_VERSION
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer


class ArrowOnlyRustFeaturizer:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    @classmethod
    def from_arrow_paths(cls, *args: Any, **kwargs: Any) -> ArrowOnlyRustFeaturizer:
        cls.calls.append((args, kwargs))
        return cls()

    def signature_ids(self) -> list[str]:
        return []

    def get_constraints_matrix_indexed(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def featurize_pairs_matrix_indexed(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def update_signature_name_counts(self, signatures: dict[str, Any]) -> int:
        return len(signatures)


class ArrowOnlyRustModule:
    __version__ = runtime.REQUIRED_RUST_EXTENSION_VERSION
    RustFeaturizer = ArrowOnlyRustFeaturizer


@pytest.fixture(autouse=True)
def _reset_arrow_only_rust(monkeypatch: pytest.MonkeyPatch):
    ArrowOnlyRustFeaturizer.calls = []
    monkeypatch.setattr(feature_port, "s2and_rust", ArrowOnlyRustModule)
    yield


def _touch_arrow_bundle(tmp_path: Path) -> dict[str, str]:
    paths = {
        "signatures": tmp_path / "signatures.arrow",
        "papers": tmp_path / "papers.arrow",
        "paper_authors": tmp_path / "paper_authors.arrow",
        "signatures_batch_index": tmp_path / "signatures.signatures_batch_index.bin",
        "papers_batch_index": tmp_path / "papers.papers_batch_index.bin",
        "paper_authors_batch_index": tmp_path / "paper_authors.paper_authors_batch_index.bin",
    }
    for path in paths.values():
        path.touch()
    return {key: str(path) for key, path in paths.items()}


def test_arrow_production_builder_calls_only_arrow_constructor(
    tmp_path: Path,
) -> None:
    paths = _touch_arrow_bundle(tmp_path)
    validated_paths = ValidatedArrowInputs._from_verified(
        paths=paths,
        generation_id="test-generation",
        normalization_version=NORMALIZATION_VERSION,
    )

    featurizer = feature_port.build_rust_featurizer_from_arrow_paths(
        validated_paths,
        expected_normalization_version=NORMALIZATION_VERSION,
        signature_ids=[1, "2"],
        name_tuples={("ada", "a")},
        preprocess=False,
        cluster_seed_require_value=7.0,
        cluster_seed_disallow_value=9.0,
        num_threads=1,
    )

    assert isinstance(featurizer, ArrowOnlyRustFeaturizer)
    assert len(ArrowOnlyRustFeaturizer.calls) == 1
    args, kwargs = ArrowOnlyRustFeaturizer.calls[0]
    assert kwargs == {}
    assert args == (
        paths,
        ["1", "2"],
        {("ada", "a")},
        False,
        7.0,
        9.0,
        1,
    )


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

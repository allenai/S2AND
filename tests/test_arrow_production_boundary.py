from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import s2and.feature_port as feature_port
import s2and.model as model_module
import s2and.runtime as runtime
from s2and.arrow_inputs import MissingArrowArtifactError
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
    __version__ = runtime.min_supported_rust_extension_version_string()
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
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _touch_arrow_bundle(tmp_path)
    monkeypatch.setattr(
        feature_port,
        "validate_arrow_prediction_artifacts",
        lambda paths_arg, **_kwargs: dict(paths_arg),
    )

    featurizer = feature_port.build_rust_featurizer_from_arrow_paths(
        paths,
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


def test_rust_predict_requires_arrow_artifacts_before_legacy_dataset_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_context = SimpleNamespace(
        operation="cluster_predict",
        requested_backend="rust",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-arrow-boundary",
        source="test",
    )
    dataset = SimpleNamespace(
        name="missing_arrow_dataset",
        mode="inference",
        signatures_path=None,
        original_signatures_path=None,
        papers_path=None,
        specter_embeddings_path=None,
        name_tuples=set(),
        cluster_seeds_require={},
        cluster_seeds_disallow=set(),
    )

    def fail_legacy_path(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Rust predict must fail before legacy dataset featurizer paths")

    monkeypatch.setattr(model_module, "build_runtime_context", lambda _operation, **_kwargs: runtime_context)
    monkeypatch.setattr(model_module, "_get_rust_featurizer", fail_legacy_path)
    monkeypatch.setattr(Clusterer, "predict_helper", fail_legacy_path)
    monkeypatch.setattr(Clusterer, "predict_from_arrow_paths", fail_legacy_path)

    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
        use_cache=False,
        batch_size=2,
    )

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        clusterer.predict({"block": ["0", "1"]}, dataset)  # type: ignore[arg-type]

    error = exc_info.value
    assert error.context == "Clusterer.predict Rust prediction"
    assert error.missing_keys == ("signatures", "papers", "paper_authors")


def test_predict_incremental_auto_degrades_to_python_without_arrow_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_context = runtime.RuntimeContext(
        operation="cluster_predict_incremental",
        requested_backend="auto",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-incremental-degrade",
        source="argument",
    )
    dataset = SimpleNamespace(
        name="json_dataset",
        cluster_seeds_require={},
        cluster_seeds_disallow=set(),
    )
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
        use_cache=False,
        batch_size=2,
    )
    observed: dict[str, Any] = {}

    monkeypatch.setattr(model_module, "_apply_dataset_name_count_semantics_for_prediction", lambda *_args: None)
    monkeypatch.setattr(model_module, "_sync_rust_cluster_seeds", lambda *_args, **_kwargs: None)

    def fake_python_fallback(self: Clusterer, *_args: Any, **kwargs: Any) -> dict[str, Any]:
        del self
        observed["runtime_context"] = kwargs["runtime_context"]
        return {"clusters": {"c0": ["s1"]}}

    def fail_promoted(self: Clusterer, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        del self
        raise AssertionError("auto mode without Arrow paths must use Python incremental fallback")

    monkeypatch.setattr(Clusterer, "_predict_incremental_python_fallback", fake_python_fallback)
    monkeypatch.setattr(Clusterer, "_predict_incremental_promoted_linker", fail_promoted)

    result = clusterer.predict_incremental(["s1"], dataset, runtime_context=runtime_context)  # type: ignore[arg-type]

    assert result == {"clusters": {"c0": ["s1"]}}
    degraded_context = observed["runtime_context"]
    assert degraded_context.resolved_backend == "python"
    assert degraded_context.use_rust is False
    assert degraded_context.run_id == runtime_context.run_id


def test_predict_subblocked_receives_python_context_after_auto_arrow_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_context = runtime.RuntimeContext(
        operation="cluster_predict",
        requested_backend="auto",
        resolved_backend="rust",
        use_rust=True,
        run_id="test-subblocked-degrade",
        source="argument",
    )
    dataset = SimpleNamespace(name="json_dataset")
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
        use_cache=False,
        batch_size=2,
    )
    observed: dict[str, Any] = {}

    def fake_subblocked(self: Clusterer, *_args: Any, **kwargs: Any) -> tuple[dict[str, list[str]], None]:
        del self
        observed["runtime_context"] = kwargs["runtime_context"]
        observed["arrow_paths"] = kwargs["arrow_paths"]
        return {"c0": ["s1"]}, None

    monkeypatch.setattr(Clusterer, "_predict_subblocked", fake_subblocked)

    clusters, dists = clusterer.predict(
        {"block": ["s1", "s2"]},
        dataset,  # type: ignore[arg-type]
        batching_threshold=1,
        runtime_context=runtime_context,
    )

    assert clusters == {"c0": ["s1"]}
    assert dists is None
    degraded_context = observed["runtime_context"]
    assert degraded_context.resolved_backend == "python"
    assert degraded_context.use_rust is False
    assert observed["arrow_paths"] is None

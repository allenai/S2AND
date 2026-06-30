from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import s2and.feature_port as feature_port
import s2and.model as model_module
import s2and.runtime as runtime
from s2and.arrow_inputs import MissingArrowArtifactError
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer

# Derived from the guard so fixtures track the lockstep version bump instead of going stale.
_SUPPORTED_VERSION = runtime.min_supported_rust_extension_version_string()


class ArrowOnlyRustFeaturizer:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    @classmethod
    def from_arrow_paths(cls, *args: Any, **kwargs: Any) -> ArrowOnlyRustFeaturizer:
        cls.calls.append((args, kwargs))
        return cls()

    @classmethod
    def from_dataset(cls, *_args: Any, **_kwargs: Any) -> ArrowOnlyRustFeaturizer:
        raise AssertionError("production Arrow build must not call RustFeaturizer.from_dataset")

    def signature_ids(self) -> list[str]:
        return []

    def get_constraints_matrix_indexed(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def featurize_pairs_matrix_indexed(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def update_signature_name_counts(self, signatures: dict[str, Any]) -> int:
        return len(signatures)


class ArrowOnlyRustModule:
    __version__ = _SUPPORTED_VERSION
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


def test_arrow_production_builder_calls_only_arrow_constructor(tmp_path: Path) -> None:
    paths = _touch_arrow_bundle(tmp_path)

    featurizer = feature_port.build_rust_featurizer_from_arrow_paths(
        paths,
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

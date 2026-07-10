from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import s2and.incremental_linking.policy as policy_module
import s2and.model as model_module
from s2and import feature_port
from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.runtime import build_runtime_context
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset, tiny_name_counts_provenance


class _ConstantClassifier:
    def predict_proba(self, features: Any, **_kwargs: Any) -> np.ndarray:
        row_count = int(np.asarray(features).shape[0])
        return np.tile(np.asarray([[0.5, 0.5]], dtype=np.float64), (row_count, 1))


def _feature_contract(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "name_counts_last_first_initial_semantics": "initial_char",
        "name_counts_generation_id": provenance["generation_id"],
        "name_counts_pickle_sha256": provenance["pickle_sha256"],
        "name_counts_source_snapshot_id": provenance["source_snapshot_id"],
        "name_counts_selected_rows_sha256": provenance["selected_rows_sha256"],
    }


def _name_count_clusterer(provenance: dict[str, Any]) -> Clusterer:
    clusterer = Clusterer(
        FeaturizationInfo(["name_counts"], featurizer_version=FEATURIZER_VERSION),
        _ConstantClassifier(),
        n_jobs=1,
        use_cache=False,
    )
    clusterer.feature_contract = _feature_contract(provenance)
    return clusterer


def _write_index_manifest(index_dir: Path, provenance: dict[str, Any]) -> None:
    index_dir.mkdir()
    (index_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "name_counts_index_v1",
                "source_provenance": provenance,
            }
        ),
        encoding="utf-8",
    )


def _binding_tuple(provenance: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        provenance["generation_id"],
        provenance["pickle_sha256"],
        provenance["source_snapshot_id"],
        provenance["selected_rows_sha256"],
    )


class _PrebuiltRustFeaturizer:
    def __init__(self, provenance: dict[str, Any]) -> None:
        self._binding = _binding_tuple(provenance)
        self.binding_read_count = 0

    @property
    def name_counts_provenance_binding(self) -> tuple[str, str, str, str]:
        self.binding_read_count += 1
        return self._binding

    def cluster_seeds_require(self) -> list[tuple[str, str]]:
        return []

    def signature_rule_metadata(self) -> list[tuple[str, str, None]]:
        return []


def test_python_prediction_accepts_exact_name_count_binding() -> None:
    provenance = tiny_name_counts_provenance()
    clusterer = _name_count_clusterer(provenance)
    dataset = build_dummy_dataset(
        "name-count-binding-python-match",
        mode="inference",
        name_counts_index=True,
    )

    clusters, dists = clusterer.predict_helper(
        {"block": ["1"]},
        dataset,
        runtime_context=build_runtime_context("name_count_binding_test", backend="python"),
    )

    assert clusters == {"block_0": ["1"]}
    assert dists is None


def test_python_prediction_rejects_mismatched_name_count_binding_before_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusterer = _name_count_clusterer(tiny_name_counts_provenance())
    dataset = build_dummy_dataset(
        "name-count-binding-python-mismatch",
        mode="inference",
        name_counts_index=True,
    )
    assert dataset.name_counts_provenance is not None
    dataset.name_counts_provenance = {
        **dataset.name_counts_provenance,
        "generation_id": "different-name-count-generation",
    }
    monkeypatch.setattr(
        model_module,
        "many_pairs_featurize",
        lambda *_args, **_kwargs: pytest.fail("feature work started before the binding check"),
    )

    with pytest.raises(ValueError, match="name-count binding mismatch.*ANDData.name_counts_provenance"):
        clusterer.predict_helper(
            {"block": ["1", "2"]},
            dataset,
            runtime_context=build_runtime_context("name_count_binding_test", backend="python"),
        )


@pytest.mark.parametrize("generation_id", ["test-tiny-name-counts", "different-name-count-generation"])
def test_arrow_prediction_checks_exact_name_count_binding_before_featurizer_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generation_id: str,
) -> None:
    model_provenance = tiny_name_counts_provenance()
    index_provenance = {**model_provenance, "generation_id": generation_id}
    clusterer = _name_count_clusterer(model_provenance)
    index_dir = tmp_path / "name_counts_index"
    _write_index_manifest(index_dir, index_provenance)
    arrow_paths = {"name_counts_index": str(index_dir)}
    build_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        model_module,
        "validate_arrow_prediction_artifacts",
        lambda paths, **_kwargs: dict(paths),
    )
    monkeypatch.setattr(
        policy_module,
        "require_name_counts_index_artifact",
        lambda path, **_kwargs: str(path),
    )

    def fake_build(paths: dict[str, Any], **_kwargs: Any) -> object:
        build_calls.append(paths)
        return object()

    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build)
    monkeypatch.setattr(
        clusterer,
        "predict_from_rust_featurizer",
        lambda *_args, **_kwargs: ({"block_0": ["1"]}, None),
    )

    if generation_id == model_provenance["generation_id"]:
        clusters, dists = clusterer.predict_from_arrow_paths({"block": ["1"]}, arrow_paths)
        assert clusters == {"block_0": ["1"]}
        assert dists is None
        assert len(build_calls) == 1
    else:
        with pytest.raises(ValueError, match="name-count binding mismatch.*name_counts_index"):
            clusterer.predict_from_arrow_paths({"block": ["1"]}, arrow_paths)
        assert build_calls == []


def test_prebuilt_rust_featurizer_prediction_checks_binding_once_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = tiny_name_counts_provenance()
    clusterer = _name_count_clusterer(provenance)
    featurizer = _PrebuiltRustFeaturizer(provenance)
    make_calls: list[str] = []

    def fake_make_dists(
        _self: Clusterer,
        block_dict: dict[str, list[str]],
        _rust_featurizer: object,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        assert "_name_counts_binding_verified" not in kwargs
        block_key = next(iter(block_dict))
        make_calls.append(block_key)
        return {block_key: np.asarray([], dtype=np.float64)}

    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", fake_make_dists)
    monkeypatch.setattr(
        Clusterer,
        "_cluster_one_block_with_logging",
        lambda _self, signatures, *_args, **_kwargs: [0] * len(signatures),
    )

    clusters, dists = clusterer.predict_from_rust_featurizer(
        {"a": ["1"], "b": ["2"]},
        featurizer,
        cluster_seeds_require={},
    )

    assert clusters == {"a_0": ["1"], "b_0": ["2"]}
    assert dists is None
    assert make_calls == ["a", "b"]
    assert featurizer.binding_read_count == 1


def test_prebuilt_rust_featurizer_prediction_rejects_mismatch_before_metadata() -> None:
    clusterer = _name_count_clusterer(tiny_name_counts_provenance())
    featurizer_provenance = {**tiny_name_counts_provenance(), "generation_id": "different-name-count-generation"}
    featurizer = _PrebuiltRustFeaturizer(featurizer_provenance)

    with pytest.raises(ValueError, match="name-count binding mismatch.*RustFeaturizer"):
        clusterer.predict_from_rust_featurizer({"block": ["1", "2"]}, featurizer)

    assert featurizer.binding_read_count == 1


def test_prebuilt_rust_featurizer_distance_boundary_accepts_exact_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = tiny_name_counts_provenance()
    clusterer = _name_count_clusterer(provenance)
    featurizer = _PrebuiltRustFeaturizer(provenance)
    monkeypatch.setattr(model_module, "_build_signature_index_by_id", lambda _featurizer: {})

    assert clusterer.make_distance_matrices_from_rust_featurizer({}, featurizer) == {}
    assert featurizer.binding_read_count == 1


def test_prebuilt_rust_featurizer_distance_boundary_rejects_mismatch_before_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusterer = _name_count_clusterer(tiny_name_counts_provenance())
    featurizer_provenance = {**tiny_name_counts_provenance(), "generation_id": "different-name-count-generation"}
    featurizer = _PrebuiltRustFeaturizer(featurizer_provenance)
    monkeypatch.setattr(
        model_module,
        "_build_signature_index_by_id",
        lambda _featurizer: pytest.fail("feature work started before the binding check"),
    )

    with pytest.raises(ValueError, match="name-count binding mismatch.*RustFeaturizer"):
        clusterer.make_distance_matrices_from_rust_featurizer({"block": ["1", "2"]}, featurizer)


def test_arrow_built_rust_featurizer_retains_verified_name_count_binding(tmp_path: Path) -> None:
    provenance = tiny_name_counts_provenance()
    dataset = build_dummy_dataset(
        "name-count-binding-real-rust-featurizer",
        mode="inference",
        name_counts_index=True,
    )
    dataset = build_arrow_training_dataset(dataset, tmp_path)

    rust_featurizer = feature_port._get_rust_featurizer(dataset)  # noqa: SLF001

    assert rust_featurizer.name_counts_provenance_binding == _binding_tuple(provenance)

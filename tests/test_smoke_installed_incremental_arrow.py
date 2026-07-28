from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from s2and import memory_budget
from s2and import production_model as production_model_module
from s2and.arrow_inputs import build_arrow_artifact_manifest, write_arrow_artifact_manifest
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from scripts.verification import smoke_installed_incremental_arrow as smoke_module
from tests.helpers import tiny_name_counts_tuple


@pytest.fixture(autouse=True)
def _stable_smoke_rss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory_budget,
        "current_rss_bytes_best_effort",
        lambda _total: (100_000_000, "test"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_release_data_root(root: Path, *, dataset: str) -> None:
    name_counts_index, _metrics = write_name_counts_index(
        root,
        tiny_name_counts_tuple(),
    )
    dataset_root = root / dataset
    paths = smoke_module._write_arrow_request(dataset_root)  # noqa: SLF001
    paths.pop("manifest")
    paths["name_counts_index"] = name_counts_index
    dataset_manifest = build_arrow_artifact_manifest(paths, dataset_root)
    dataset_manifest_path = write_arrow_artifact_manifest(dataset_manifest, dataset_root)
    root_manifest = {
        "schema": smoke_module.RELEASE_DATA_MANIFEST_SCHEMA,
        "datasets": [dataset],
        "dataset_manifests": [
            {
                "dataset": dataset,
                "manifest_path": f"{dataset}/manifest.json",
                "manifest_sha256": _sha256(dataset_manifest_path),
            }
        ],
    }
    root_manifest_path = root / "manifest.json"
    root_manifest_path.write_text(json.dumps(root_manifest, sort_keys=True) + "\n", encoding="utf-8")


def _write_release_smoke_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    artifact_hashes = {
        "name_tuples_data_sha256": "1" * 64,
        "orcid_prefix_counts_data_sha256": "2" * 64,
    }
    monkeypatch.setattr(smoke_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    monkeypatch.setattr(production_model_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    model_dir = smoke_module._write_synthetic_bundle(tmp_path / "model-inputs")  # noqa: SLF001
    data_root = tmp_path / "data"
    _write_release_data_root(data_root, dataset="smoke")
    monkeypatch.setattr(smoke_module, "NAME_COUNTS_INDEX_PATH", data_root / "name_counts_index")
    return model_dir, data_root


def _run_release_candidate(model_dir: Path, data_root: Path) -> dict[str, object]:
    return smoke_module.run_release_candidate_smoke(
        model_dir=model_dir,
        data_root=data_root,
        dataset="smoke",
        signature_ids=("s1", "s2", "q1"),
    )


def test_promoted_incremental_arrow_smoke_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_hashes = {
        "name_tuples_data_sha256": "1" * 64,
        "orcid_prefix_counts_data_sha256": "2" * 64,
    }
    monkeypatch.setattr(smoke_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    monkeypatch.setattr(production_model_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    summary = smoke_module.run_smoke(tmp_path)

    assert summary == {
        "arrow_promoted_incremental": 1,
        "bulk_pair_count": 3,
        "bulk_signature_count": 3,
        "cluster_count": 2,
        "query_view": "raw_arrow",
        "signature_count": 3,
    }


def test_smoke_summary_rejects_duplicate_signature_membership() -> None:
    result = {
        "incremental_linker_query_view": "raw_arrow",
        "incremental_linker_telemetry": {"arrow_promoted_incremental": 1},
        "clusters": {"a": ["q1", "s1"], "b": ["q1", "s2"]},
    }

    with pytest.raises(RuntimeError, match="lost signatures"):
        smoke_module._smoke_result_summary(  # noqa: SLF001
            result,
            {"q1", "s1", "s2"},
            label="duplicate smoke",
        )


def test_release_candidate_smoke_uses_exact_downloaded_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir, data_root = _write_release_smoke_inputs(tmp_path, monkeypatch)

    smoke_module.main(
        [
            "release-candidate",
            "--model-dir",
            str(model_dir),
            "--data-root",
            str(data_root),
            "--dataset",
            "smoke",
            "--signature-ids",
            "s1",
            "s2",
            "q1",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert summary == {
        "arrow_promoted_incremental": 1,
        "bulk_pair_count": 3,
        "bulk_signature_count": 3,
        "cluster_count": 2,
        "configured_name_counts_index": str((data_root / "name_counts_index").resolve()),
        "dataset": "smoke",
        "query_view": "raw_arrow",
        "signature_count": 3,
    }


def test_release_candidate_smoke_rejects_modified_nested_dataset_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, data_root = _write_release_smoke_inputs(tmp_path, monkeypatch)
    manifest_path = data_root / "smoke" / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="Release dataset 'smoke' manifest SHA-256 mismatch"):
        _run_release_candidate(model_dir, data_root)


def test_release_candidate_smoke_rejects_modified_name_count_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, data_root = _write_release_smoke_inputs(tmp_path, monkeypatch)
    index_root = data_root / "name_counts_index"
    manifest = json.loads((index_root / "manifest.json").read_text(encoding="utf-8"))
    material_path = index_root / next(iter(manifest["files"].values()))["path"]
    material_path.write_bytes(material_path.read_bytes() + b"\0")

    with pytest.raises(ValueError, match="mismatch"):
        _run_release_candidate(model_dir, data_root)


def test_release_candidate_smoke_rejects_modified_model_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, data_root = _write_release_smoke_inputs(tmp_path, monkeypatch)
    clusterer_path = model_dir / "clusterer.json"
    clusterer_path.write_bytes(clusterer_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="mismatch"):
        _run_release_candidate(model_dir, data_root)


def test_release_candidate_smoke_rejects_dataset_bound_to_other_name_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, data_root = _write_release_smoke_inputs(tmp_path, monkeypatch)
    dataset_root = data_root / "smoke"
    alternate_name_counts, _metrics = write_name_counts_index(
        dataset_root / "alternate",
        tiny_name_counts_tuple(),
    )
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {key: (dataset_root / value).resolve() for key, value in manifest["paths"].items()}
    paths["name_counts_index"] = alternate_name_counts
    write_arrow_artifact_manifest(build_arrow_artifact_manifest(paths, dataset_root), dataset_root)
    root_manifest_path = data_root / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_manifest["dataset_manifests"][0]["manifest_sha256"] = _sha256(manifest_path)
    root_manifest_path.write_text(json.dumps(root_manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not bind the configured name-count index"):
        _run_release_candidate(model_dir, data_root)


def test_release_candidate_smoke_rejects_wrong_configured_name_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, data_root = _write_release_smoke_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(smoke_module, "NAME_COUNTS_INDEX_PATH", tmp_path / "wrong-name-counts")
    monkeypatch.setattr(
        smoke_module,
        "load_production_model",
        lambda _path: pytest.fail("model load must not run after a selector mismatch"),
    )

    with pytest.raises(ValueError, match="Configured NAME_COUNTS_INDEX_PATH"):
        _run_release_candidate(model_dir, data_root)

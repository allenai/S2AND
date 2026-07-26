from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from s2and import production_model as production_model_module
from s2and.arrow_inputs import build_arrow_artifact_manifest, write_arrow_artifact_manifest
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from scripts.verification import smoke_installed_incremental_arrow as smoke_module
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_release_data_root(root: Path, *, dataset: str) -> dict[str, str]:
    name_counts_index, _metrics = write_name_counts_index(
        root,
        tiny_name_counts_tuple(),
        tiny_name_counts_provenance(),
    )
    dataset_root = root / dataset
    paths = smoke_module._write_arrow_request(dataset_root)  # noqa: SLF001
    paths.pop("cluster_seeds")
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
    return {
        "data": _sha256(root_manifest_path),
        "name_counts": _sha256(Path(name_counts_index) / "manifest.json"),
    }


def _write_release_smoke_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, str]]:
    artifact_hashes = {
        "name_tuples_data_sha256": "1" * 64,
        "orcid_prefix_counts_data_sha256": "2" * 64,
    }
    monkeypatch.setattr(smoke_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    monkeypatch.setattr(production_model_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    model_dir = smoke_module._write_synthetic_bundle(tmp_path / "model-inputs")  # noqa: SLF001
    data_root = tmp_path / "data"
    expected = _write_release_data_root(data_root, dataset="smoke")
    expected["model"] = _sha256(model_dir / "manifest.json")
    monkeypatch.setattr(smoke_module, "NAME_COUNTS_INDEX_PATH", data_root / "name_counts_index")
    return model_dir, data_root, expected


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
        "cluster_count": 2,
        "query_view": "raw_arrow",
        "signature_count": 3,
    }


def test_release_candidate_smoke_uses_exact_downloaded_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, data_root, expected = _write_release_smoke_inputs(tmp_path, monkeypatch)

    summary = smoke_module.run_release_candidate_smoke(
        model_dir=model_dir,
        data_root=data_root,
        dataset="smoke",
        signature_ids=("s1", "s2", "q1"),
        expected_model_manifest_sha256=expected["model"],
        expected_data_manifest_sha256=expected["data"],
        expected_name_counts_manifest_sha256=expected["name_counts"],
    )

    assert summary["arrow_promoted_incremental"] == 1
    assert summary["query_view"] == "raw_arrow"
    assert summary["signature_count"] == 3
    assert summary["model_manifest_sha256"] == expected["model"]
    assert summary["data_manifest_sha256"] == expected["data"]
    assert summary["name_counts_manifest_sha256"] == expected["name_counts"]
    assert summary["configured_name_counts_index"] == str((data_root / "name_counts_index").resolve())


@pytest.mark.parametrize(
    ("wrong_binding", "error"),
    (
        ("model", "Production model manifest SHA-256 mismatch"),
        ("data", "Release data manifest SHA-256 mismatch"),
        ("name_counts", "Name-count manifest SHA-256 mismatch"),
    ),
)
def test_release_candidate_smoke_rejects_wrong_manifest_digest_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_binding: str,
    error: str,
) -> None:
    model_dir, data_root, expected = _write_release_smoke_inputs(tmp_path, monkeypatch)
    expected[wrong_binding] = "f" * 64
    monkeypatch.setattr(
        smoke_module,
        "load_production_model",
        lambda _path: pytest.fail("model load must not run after a manifest mismatch"),
    )

    with pytest.raises(ValueError, match=error):
        smoke_module.run_release_candidate_smoke(
            model_dir=model_dir,
            data_root=data_root,
            dataset="smoke",
            signature_ids=("s1", "s2", "q1"),
            expected_model_manifest_sha256=expected["model"],
            expected_data_manifest_sha256=expected["data"],
            expected_name_counts_manifest_sha256=expected["name_counts"],
        )


def test_release_candidate_smoke_rejects_wrong_configured_name_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir, data_root, expected = _write_release_smoke_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(smoke_module, "NAME_COUNTS_INDEX_PATH", tmp_path / "wrong-name-counts")
    monkeypatch.setattr(
        smoke_module,
        "load_production_model",
        lambda _path: pytest.fail("model load must not run after a selector mismatch"),
    )

    with pytest.raises(ValueError, match="Configured NAME_COUNTS_INDEX_PATH"):
        smoke_module.run_release_candidate_smoke(
            model_dir=model_dir,
            data_root=data_root,
            dataset="smoke",
            signature_ids=("s1", "s2", "q1"),
            expected_model_manifest_sha256=expected["model"],
            expected_data_manifest_sha256=expected["data"],
            expected_name_counts_manifest_sha256=expected["name_counts"],
        )

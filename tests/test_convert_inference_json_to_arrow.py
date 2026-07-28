from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import scripts.convert_to_arrow as convert_module
from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block import FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION
from scripts.convert_to_arrow import convert_service_json_to_arrow


def _read_table(path: str) -> pa.Table:
    with pa.memory_map(path, "r") as source:
        return pa.ipc.open_file(source).read_all()


def _manifest_path(manifest: Mapping[str, Any], dataset_dir: Path, key: str) -> Path:
    path = Path(str(manifest["paths"][key]))
    if path.is_absolute():
        return path
    return dataset_dir / path


def _minimal_service_payload(signature_id: str = "s1", paper_id: int = 1) -> dict[str, Any]:
    return {
        "signatures": [
            {
                "signature_id": signature_id,
                "paper_id": paper_id,
                "author_info": {
                    "position": 0,
                    "block": "a smith",
                    "first": "Alice",
                    "middle": None,
                    "last": "Smith",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                    "source_ids": [],
                },
            }
        ],
        "papers": [
            {
                "paper_id": paper_id,
                "title": "One",
                "abstract": "",
                "journal_name": "",
                "venue": "",
                "year": 2020,
                "authors": [{"position": 0, "author_name": "Alice Smith"}],
            }
        ],
        "cluster_seeds": {},
        "altered_cluster_signatures": [],
    }


def test_convert_service_json_to_arrow_rejects_altered_without_seed(tmp_path: Path) -> None:
    payload = _minimal_service_payload()
    payload["altered_cluster_signatures"] = ["s1"]
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Altered cluster signature s1 not in cluster_seeds_require"):
        convert_service_json_to_arrow(
            input_json=input_json,
            output_root=tmp_path / "arrow",
            dataset_name="service_payload",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=True,
            skip_name_counts_index=True,
        )


@pytest.mark.parametrize("source_author_name", ["", "   ", "24"])
def test_convert_service_json_to_arrow_preserves_author_rows_that_normalize_empty(
    tmp_path: Path,
    source_author_name: str,
) -> None:
    payload = _minimal_service_payload()
    payload["papers"][0]["authors"][0]["author_name"] = source_author_name
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    manifest = convert_service_json_to_arrow(
        input_json=input_json,
        output_root=tmp_path / "arrow",
        dataset_name="service_payload",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
    )

    dataset_dir = tmp_path / "arrow" / "service_payload"
    paper_authors = _read_table(str(_manifest_path(manifest, dataset_dir, "paper_authors")))
    assert paper_authors.to_pylist() == [{"paper_id": "1", "position": 0, "author_name": ""}]
    assert manifest["validation"]["paper_author_count"] == 1


def test_convert_service_json_to_arrow_preserves_seed_and_altered_tables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = {
        "signatures": [
            {
                "signature_id": "s1",
                "paper_id": 1,
                "author_info": {
                    "position": 0,
                    "block": "a smith",
                    "first": "Alice",
                    "middle": None,
                    "last": "Smith",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                    "source_ids": [],
                },
            },
            {
                "signature_id": "s2",
                "paper_id": 2,
                "author_info": {
                    "position": 0,
                    "block": "a smith",
                    "first": "Alice",
                    "middle": None,
                    "last": "Smith",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                    "source_ids": [],
                },
            },
            {
                "signature_id": "q",
                "paper_id": 3,
                "author_info": {
                    "position": 0,
                    "block": "a smith",
                    "first": "Alex",
                    "middle": None,
                    "last": "Smith",
                    "suffix": None,
                    "email": None,
                    "affiliations": [],
                    "source_ids": [],
                },
            },
        ],
        "papers": [
            {
                "paper_id": 1,
                "title": "One",
                "abstract": "Has Abstract",
                "journal_name": "",
                "venue": "",
                "year": 2020,
                "authors": [{"position": 0, "author_name": "Alice Smith"}],
            },
            {
                "paper_id": 2,
                "title": "Two",
                "abstract": "Has Abstract",
                "journal_name": "",
                "venue": "",
                "year": 2021,
                "authors": [{"position": 0, "author_name": "Alice Smith"}],
            },
            {
                "paper_id": 3,
                "title": "Three",
                "abstract": "",
                "journal_name": "",
                "venue": "",
                "year": 2022,
                "authors": [{"position": 0, "author_name": "Alex Smith"}],
            },
        ],
        "paper_embeddings": {"1": [0.1, 0.2], "2": [0.2, 0.3], "3": [0.3, 0.4]},
        "cluster_seeds": {"s1": {"s2": "require", "q": "disallow"}},
        "altered_cluster_signatures": ["s1"],
    }
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    manifest = convert_service_json_to_arrow(
        input_json=input_json,
        output_root=tmp_path / "arrow",
        dataset_name="service_payload",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
    )

    assert manifest["schema"] == FEATURE_BLOCK_ARROW_MANIFEST_SCHEMA_VERSION
    assert manifest["signature_count"] == 3
    assert manifest["paper_count"] == 3
    assert manifest["cluster_seeds_require_count"] == 2
    assert manifest["cluster_seeds_disallow_count"] == 1

    dataset_dir = tmp_path / "arrow" / "service_payload"
    cluster_seed_rows = _read_table(str(_manifest_path(manifest, dataset_dir, "cluster_seeds"))).to_pydict()
    assert cluster_seed_rows == {"signature_id": ["s1", "s2"], "cluster_id": ["0", "0"]}
    cluster_seed_disallow_rows = _read_table(
        str(_manifest_path(manifest, dataset_dir, "cluster_seed_disallows"))
    ).to_pydict()
    assert cluster_seed_disallow_rows == {"signature_id_1": ["q"], "signature_id_2": ["s1"]}
    altered_path = _manifest_path(manifest, dataset_dir, "altered_cluster_signatures")
    assert altered_path.name == "altered_cluster_signatures.arrow"
    assert _read_table(str(altered_path)).to_pydict() == {"signature_id": ["s1"]}

    assert _read_table(str(_manifest_path(manifest, dataset_dir, "signatures"))).num_rows == 3
    assert _read_table(str(_manifest_path(manifest, dataset_dir, "papers"))).num_rows == 3
    assert _read_table(str(_manifest_path(manifest, dataset_dir, "paper_authors"))).num_rows == 3
    assert _read_table(str(_manifest_path(manifest, dataset_dir, "specter"))).num_rows == 3
    assert Path(manifest["paths"]["signatures_batch_index"]).name == "signatures.signatures_batch_index.bin"
    assert _manifest_path(manifest, dataset_dir, "papers_batch_index").exists()
    assert "signatures_json" not in manifest["paths"]
    assert "papers_json" not in manifest["paths"]
    assert "cluster_seeds_json" not in manifest["paths"]
    assert not (_manifest_path(manifest, dataset_dir, "signatures").parent / "signatures.json").exists()
    assert manifest["physical_layout"]["schema"] == "s2and_arrow_physical_v1"
    assert manifest["physical_layout"]["tables"]["signatures"]["batch_index_present"] is True
    assert manifest["raw_planner_batch_indexes"]["signatures_batch_index"]["record_count"] == 3


def test_convert_service_json_to_arrow_accepts_service_shaped_cluster_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _minimal_service_payload("s1", 1)
    payload["signatures"] = [
        payload["signatures"][0],
        {
            **payload["signatures"][0],
            "signature_id": "s2",
            "paper_id": 2,
        },
        {
            **payload["signatures"][0],
            "signature_id": "q",
            "paper_id": 3,
        },
    ]
    payload["papers"] = [
        payload["papers"][0],
        {**payload["papers"][0], "paper_id": 2, "title": "Two"},
        {**payload["papers"][0], "paper_id": 3, "title": "Three"},
    ]
    payload["cluster_seeds"] = {
        "require": {"c0": ["s1", "s2"]},
        "disallow": [["q", "s1"]],
    }
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    manifest = convert_service_json_to_arrow(
        input_json=input_json,
        output_root=tmp_path / "arrow",
        dataset_name="service_payload",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
    )

    assert manifest["cluster_seeds_require_count"] == 2
    assert manifest["cluster_seeds_disallow_count"] == 1


def test_convert_service_json_to_arrow_falls_back_from_explicit_null_paper_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _minimal_service_payload()
    payload["paper_embeddings"] = None
    payload["specter_embeddings"] = {"1": [0.1, 0.2]}
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    manifest = convert_service_json_to_arrow(
        input_json=input_json,
        output_root=tmp_path / "arrow",
        dataset_name="service_payload",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
    )

    assert manifest["paper_embedding_count"] == 1
    assert _read_table(str(_manifest_path(manifest, tmp_path / "arrow" / "service_payload", "specter"))).num_rows == 1


def test_convert_service_json_to_arrow_emits_empty_specter_for_empty_embeddings(tmp_path: Path) -> None:
    payload = _minimal_service_payload("s1", 1)
    payload["paper_embeddings"] = {}
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    manifest = convert_service_json_to_arrow(
        input_json=input_json,
        output_root=tmp_path / "arrow",
        dataset_name="service_payload",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
    )
    dataset_dir = tmp_path / "arrow" / "service_payload"
    resolved_paths = {key: str(_manifest_path(manifest, dataset_dir, key)) for key in manifest["paths"]}

    specter = _read_table(resolved_paths["specter"])
    embedding_type = specter.schema.field("embedding").type
    assert specter.num_rows == 0
    assert pa.types.is_fixed_size_list(embedding_type)
    assert embedding_type.list_size == 1
    assert manifest["validation"]["specter_count"] == 0
    assert manifest["validation"]["missing_specter_paper_count"] == 1
    assert "specter_batch_index" in resolved_paths
    with ArrowDataset.open(dataset_dir, require_specter=True):
        pass


def test_root_manifest_upsert_uses_bounded_shared_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "manifest.json.lock"
    observed: dict[str, object] = {}

    class FailingLock:
        def __enter__(self) -> None:
            raise TimeoutError("simulated contention")

        def __exit__(self, *_args: object) -> None:
            return None

    def failing_lock(path: Path, *, timeout_seconds: float) -> FailingLock:
        observed["path"] = path
        observed["timeout_seconds"] = timeout_seconds
        return FailingLock()

    monkeypatch.setattr(convert_module, "exclusive_file_lock", failing_lock)

    with pytest.raises(TimeoutError, match="simulated contention"):
        convert_module._upsert_root_manifest(
            tmp_path,
            dataset_name="tiny",
            dataset_dir=tmp_path / "tiny",
        )

    assert observed == {
        "path": lock_path,
        "timeout_seconds": convert_module._ROOT_MANIFEST_LOCK_TIMEOUT_SECONDS,
    }


def test_convert_service_json_to_arrow_reports_missing_specter_embeddings(
    tmp_path: Path,
) -> None:
    payload = _minimal_service_payload("s1", 1)
    payload["signatures"].append(
        {
            **payload["signatures"][0],
            "signature_id": "s2",
            "paper_id": 2,
        }
    )
    payload["papers"].append({**payload["papers"][0], "paper_id": 2, "title": "Two"})
    payload["paper_embeddings"] = {"1": [0.1, 0.2]}
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    manifest = convert_service_json_to_arrow(
        input_json=input_json,
        output_root=tmp_path / "arrow",
        dataset_name="service_payload",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
    )

    assert manifest["validation"]["missing_specter_paper_count"] == 1
    assert manifest["validation"]["missing_specter_paper_examples"] == ["2"]


def test_convert_service_json_to_arrow_rejects_ambiguous_service_shaped_cluster_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _minimal_service_payload()
    payload["cluster_seeds"] = {"require": {"c0": ["s1"]}, "disallow": [], "unexpected": []}
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported keys"):
        convert_service_json_to_arrow(
            input_json=input_json,
            output_root=tmp_path / "arrow",
            dataset_name="service_payload",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=True,
            skip_name_counts_index=True,
        )


def test_convert_service_json_to_arrow_source_json_is_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")

    manifest = convert_service_json_to_arrow(
        input_json=input_json,
        output_root=tmp_path / "arrow",
        dataset_name="service_payload",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
        copy_source_json=True,
    )

    for key in ("signatures_json", "papers_json", "cluster_seeds_json"):
        assert _manifest_path(manifest, tmp_path / "arrow" / "service_payload", key).exists()


def test_convert_service_json_to_arrow_rejects_duplicate_list_ids(tmp_path: Path) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(
        json.dumps(
            {
                "signatures": [
                    {"signature_id": "s1"},
                    {"signature_id": "s1"},
                ],
                "papers": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate signature_id"):
        convert_service_json_to_arrow(
            input_json=input_json,
            output_root=tmp_path / "arrow",
            dataset_name="service_payload",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=False,
            skip_name_counts_index=True,
        )


def test_convert_service_json_to_arrow_rejects_stale_output_without_overwrite(tmp_path: Path) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "arrow" / "service_payload"
    output_dir.mkdir(parents=True)
    (output_dir / "signatures.arrow").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        convert_service_json_to_arrow(
            input_json=input_json,
            output_root=tmp_path / "arrow",
            dataset_name="service_payload",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=False,
            skip_name_counts_index=True,
        )


def test_service_json_main_dispatches_bounded_cli_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}

    def fake_convert_service_json_to_arrow(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "dataset": kwargs["dataset_name"],
            "signature_count": 0,
            "paper_count": 0,
            "paths": {},
        }

    monkeypatch.setattr(convert_module, "convert_service_json_to_arrow", fake_convert_service_json_to_arrow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_arrow.py",
            "service-json",
            "--input-json",
            str(tmp_path / "payload.json"),
            "--output-root",
            str(tmp_path / "arrow"),
            "--dataset-name",
            "service_payload",
            "--n-jobs",
            "1",
            "--skip-name-counts-index",
            "--skip-validation",
            "--copy-source-json",
        ],
    )

    convert_module.main()

    assert captured["dataset_name"] == "service_payload"
    assert captured["n_jobs"] == 1
    assert captured["skip_name_counts_index"] is True
    assert captured["copy_source_json"] is True
    assert captured["validate"] is False
    assert json.loads(capsys.readouterr().out)["dataset"] == "service_payload"


def test_convert_service_json_to_arrow_overwrite_preserves_other_root_manifest_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")
    output_root = tmp_path / "arrow"
    output_root.mkdir()
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "inference_arrow_bundle_v1",
                "output_root": str(output_root),
                "datasets": ["existing_dataset"],
                "dataset_manifests": [
                    {
                        "dataset": "existing_dataset",
                        "dataset_dir": "existing_dataset",
                        "manifest_path": "existing_dataset/manifest.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    convert_service_json_to_arrow(
        input_json=input_json,
        output_root=output_root,
        dataset_name="new_dataset",
        name_counts_index_root=tmp_path,
        n_jobs=1,
        overwrite=True,
        skip_name_counts_index=True,
    )

    root_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert root_manifest["schema"] == "inference_arrow_bundle_v1"
    assert "source_path" not in root_manifest
    assert "reports" not in root_manifest
    assert root_manifest["datasets"] == ["existing_dataset", "new_dataset"]
    assert [report["dataset"] for report in root_manifest["dataset_manifests"]] == [
        "existing_dataset",
        "new_dataset",
    ]
    entries = {entry["dataset"]: entry for entry in root_manifest["dataset_manifests"]}
    assert entries["existing_dataset"]["manifest_exists"] is False
    assert entries["new_dataset"]["manifest_exists"] is True
    assert entries["new_dataset"]["manifest_size_bytes"] > 0
    assert len(entries["new_dataset"]["manifest_sha256"]) == 64
    assert entries["new_dataset"]["audit"]["conversion_kind"] == "service-json"
    assert entries["new_dataset"]["audit"]["signature_count"] == 1
    assert root_manifest["audit"]["datasets_with_missing_manifests"] == ["existing_dataset"]
    assert root_manifest["audit"]["total_signature_count"] == 1
    new_dataset_dir = convert_module._manifest_relative_path(
        output_root / "new_dataset",
        convert_module._PROJECT_ROOT,
    ).replace("\\", "/")
    assert root_manifest["validation_commands"] == [
        f"uv run python scripts/convert_to_arrow.py validate --dataset-dir {new_dataset_dir}"
    ]


def test_convert_service_json_to_arrow_rejects_malformed_root_manifest_before_dataset_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")
    output_root = tmp_path / "arrow"
    output_root.mkdir()
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "inference_arrow_bundle_v1",
                "output_root": str(output_root),
                "datasets": ["existing_dataset"],
                "dataset_manifests": [{"manifest_path": "existing_dataset/manifest.json"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"dataset_manifests\[0\].*dataset"):
        convert_service_json_to_arrow(
            input_json=input_json,
            output_root=output_root,
            dataset_name="new_dataset",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=True,
            skip_name_counts_index=True,
        )

    assert not (output_root / "new_dataset" / "manifest.json").exists()

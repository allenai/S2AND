from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import scripts.convert_to_arrow as convert_module
from s2and.arrow_inputs import ARROW_COLLECTION_KIND, ARROW_DATASET_KIND, ArrowDataset
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
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


def _convert_service_payload(tmp_path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], Path]:
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
    return manifest, tmp_path / "arrow" / "service_payload"


def test_convert_service_json_to_arrow_rejects_altered_without_seed(tmp_path: Path) -> None:
    payload = _minimal_service_payload()
    payload["altered_cluster_signatures"] = ["s1"]

    with pytest.raises(ValueError, match="Altered cluster signature s1 not in cluster_seeds_require"):
        _convert_service_payload(tmp_path, payload)


def test_convert_service_json_to_arrow_preserves_author_rows_that_normalize_empty(tmp_path: Path) -> None:
    for case_id, source_author_name in (("empty", ""), ("blank", "   "), ("digits", "24")):
        case_root = tmp_path / case_id
        case_root.mkdir()
        payload = _minimal_service_payload()
        payload["papers"][0]["authors"][0]["author_name"] = source_author_name

        manifest, dataset_dir = _convert_service_payload(case_root, payload)

        paper_authors = _read_table(str(_manifest_path(manifest, dataset_dir, "paper_authors")))
        assert paper_authors.to_pylist() == [{"paper_id": "1", "position": 0, "author_name": ""}], case_id
        assert manifest["validation"]["paper_author_count"] == 1, case_id


def test_convert_service_json_to_arrow_preserves_seed_and_altered_tables(tmp_path: Path) -> None:
    payload = _minimal_service_payload()
    first_signature = payload["signatures"][0]
    first_paper = payload["papers"][0]
    first_paper["abstract"] = "Has Abstract"
    payload["signatures"] = [
        first_signature,
        {**first_signature, "signature_id": "s2", "paper_id": 2},
        {
            **first_signature,
            "signature_id": "q",
            "paper_id": 3,
            "author_info": {**first_signature["author_info"], "first": "Alex"},
        },
    ]
    payload["papers"] = [
        first_paper,
        {**first_paper, "paper_id": 2, "title": "Two", "year": 2021},
        {
            **first_paper,
            "paper_id": 3,
            "title": "Three",
            "abstract": "",
            "year": 2022,
            "authors": [{"position": 0, "author_name": "Alex Smith"}],
        },
    ]
    payload["paper_embeddings"] = {"1": [0.1, 0.2], "2": [0.2, 0.3], "3": [0.3, 0.4]}
    payload["cluster_seeds"] = {"s1": {"s2": "require", "q": "disallow"}}
    payload["altered_cluster_signatures"] = ["s1"]

    manifest, dataset_dir = _convert_service_payload(tmp_path, payload)

    assert manifest["signature_count"] == 3
    assert manifest["paper_count"] == 3

    persisted_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert persisted_manifest["kind"] == ARROW_DATASET_KIND
    assert persisted_manifest["format_version"] == PUBLIC_DATA_FORMAT_VERSION
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


def test_convert_service_json_to_arrow_accepts_service_shaped_cluster_seeds(tmp_path: Path) -> None:
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
    manifest, dataset_dir = _convert_service_payload(tmp_path, payload)

    assert _read_table(str(_manifest_path(manifest, dataset_dir, "cluster_seeds"))).num_rows == 2
    assert _read_table(str(_manifest_path(manifest, dataset_dir, "cluster_seed_disallows"))).num_rows == 1


def test_convert_service_json_to_arrow_falls_back_from_explicit_null_paper_embeddings(tmp_path: Path) -> None:
    payload = _minimal_service_payload()
    payload["paper_embeddings"] = None
    payload["specter_embeddings"] = {"1": [0.1, 0.2]}
    manifest, dataset_dir = _convert_service_payload(tmp_path, payload)

    assert _read_table(str(_manifest_path(manifest, dataset_dir, "specter"))).num_rows == 1


def test_convert_service_json_to_arrow_emits_empty_specter_for_empty_embeddings(tmp_path: Path) -> None:
    payload = _minimal_service_payload("s1", 1)
    payload["paper_embeddings"] = {}

    manifest, dataset_dir = _convert_service_payload(tmp_path, payload)
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

    manifest, _dataset_dir = _convert_service_payload(tmp_path, payload)

    assert manifest["validation"]["missing_specter_paper_count"] == 1
    assert manifest["validation"]["missing_specter_paper_examples"] == ["2"]


def test_convert_service_json_to_arrow_rejects_ambiguous_service_shaped_cluster_seeds(tmp_path: Path) -> None:
    payload = _minimal_service_payload()
    payload["cluster_seeds"] = {"require": {"c0": ["s1"]}, "disallow": [], "unexpected": []}

    with pytest.raises(ValueError, match="unsupported keys"):
        _convert_service_payload(tmp_path, payload)


def test_convert_service_json_to_arrow_source_json_is_opt_in(
    tmp_path: Path,
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
) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")
    output_root = tmp_path / "arrow"
    output_root.mkdir()
    existing_dataset_dir = output_root / "existing_dataset"
    existing_dataset_dir.mkdir()
    (existing_dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "existing_dataset", "paths": {}}),
        encoding="utf-8",
    )
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": ARROW_COLLECTION_KIND,
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "dataset_manifests": {
                    "existing_dataset": {
                        "path": "existing_dataset/manifest.json",
                        "sha256": hashlib.sha256((existing_dataset_dir / "manifest.json").read_bytes()).hexdigest(),
                    },
                },
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
    assert set(root_manifest) == {"kind", "format_version", "dataset_manifests"}
    assert root_manifest["kind"] == ARROW_COLLECTION_KIND
    assert root_manifest["format_version"] == PUBLIC_DATA_FORMAT_VERSION
    assert list(root_manifest["dataset_manifests"]) == ["existing_dataset", "new_dataset"]
    for binding in root_manifest["dataset_manifests"].values():
        assert set(binding) == {"path", "sha256"}
        assert len(binding["sha256"]) == 64


def test_convert_service_json_to_arrow_rejects_missing_referenced_manifest(
    tmp_path: Path,
) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")
    output_root = tmp_path / "arrow"
    output_root.mkdir()
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": ARROW_COLLECTION_KIND,
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "dataset_manifests": {
                    "missing": {
                        "path": "missing/manifest.json",
                        "sha256": "0" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset_manifests.missing manifest does not exist"):
        convert_service_json_to_arrow(
            input_json=input_json,
            output_root=output_root,
            dataset_name="new_dataset",
            name_counts_index_root=tmp_path,
            n_jobs=1,
            overwrite=True,
            skip_name_counts_index=True,
        )

    assert not (output_root / "new_dataset").exists()


def test_convert_service_json_to_arrow_rejects_malformed_root_manifest_before_dataset_manifest(
    tmp_path: Path,
) -> None:
    input_json = tmp_path / "service_payload.json"
    input_json.write_text(json.dumps(_minimal_service_payload()), encoding="utf-8")
    output_root = tmp_path / "arrow"
    output_root.mkdir()
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": ARROW_COLLECTION_KIND,
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "dataset_manifests": {"existing_dataset": {"path": "existing_dataset/manifest.json"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"dataset_manifests\.existing_dataset.*path and sha256"):
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

"""Protect derived query origins before native feature allocation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pytest

from scripts.linker.train_linker_and_finalize import (
    _copy_bundle_support_files,
    load_bundle,
)
from scripts.production.model.initial_query_sources import derived_query_id, load_initial_query_sources
from tests.raw_arrow_helpers import write_ipc


def _snapshot(root: Path, *, clones: bool = True) -> tuple[Path, list[dict[str, str]]]:
    dataset = root / "datasets" / "unit"
    clone = derived_query_id("visibleq:", "q", "paper")
    clone_paper = derived_query_id("visiblep:", "q", "paper")
    ids, papers = ["q", "other"], ["paper", "other-paper"]
    if clones:
        ids.append(clone)
        papers.append(clone_paper)
    write_ipc(dataset / "signatures.arrow", pa.table({"signature_id": ids, "paper_id": papers}))
    (dataset / "manifest.json").write_text(json.dumps({"paths": {"signatures": "signatures.arrow"}}), encoding="utf-8")
    path = root / "incremental_patent_tasks" / "initial_query_source_map.jsonl"
    rows = [
        {
            "dataset": "unit",
            "source_query_signature_id": "q",
            "initial_query_signature_id": clone,
            "source_paper_id": "paper",
            "initial_paper_id": clone_paper,
        }
    ]
    return path, rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_derived_queries_require_valid_origins(tmp_path: Path) -> None:
    path, rows = _snapshot(tmp_path)
    _write(path, [])
    with pytest.raises(ValueError, match="no source-map origin"):
        load_initial_query_sources(tmp_path)

    for field, value, message in (
        ("source_query_signature_id", "unbound-source", "absent"),
        ("source_paper_id", "wrong-paper", "source paper differs"),
        ("initial_paper_id", "wrong-copy-paper", "initial paper differs"),
    ):
        _write(path, [{**rows[0], field: value}])
        with pytest.raises(ValueError, match=message):
            load_initial_query_sources(tmp_path)

    compact = {key: value for key, value in rows[0].items() if key not in {"source_paper_id", "initial_paper_id"}}
    _write(path, [compact])
    assert load_initial_query_sources(tmp_path).origins == {"unit": {compact["initial_query_signature_id"]: "q"}}
    _write(path, [{**compact, "source_query_signature_id": "other"}])
    with pytest.raises(ValueError, match="Derived query IDs"):
        load_initial_query_sources(tmp_path)


def test_source_map_rejects_invalid_or_duplicate_identities(tmp_path: Path) -> None:
    path, [row] = _snapshot(tmp_path)
    _write(path, [row, {**row, "initial_query_signature_id": "different-copy"}])
    with pytest.raises(ValueError, match="Duplicate"):
        load_initial_query_sources(tmp_path)
    _write(path, [{**row, "source_query_signature_id": row["initial_query_signature_id"]}])
    with pytest.raises(ValueError, match="original signature"):
        load_initial_query_sources(tmp_path)
    _write(path, [{**row, "dataset": ".."}])
    with pytest.raises(ValueError, match="Invalid source-map dataset"):
        load_initial_query_sources(tmp_path)
    _write(path, [{**row, "dataset": "different-dataset"}])
    with pytest.raises(ValueError, match="no signature manifest"):
        load_initial_query_sources(tmp_path)


def test_source_map_digest_binds_the_exact_bytes(tmp_path: Path) -> None:
    path, rows = _snapshot(tmp_path)
    _write(path, rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "bundle.json"
    manifest.write_text(json.dumps({"initial_query_sources_sha256": 12}), encoding="utf-8")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_initial_query_sources(tmp_path)
    manifest.write_text(json.dumps({"initial_query_sources_sha256": digest}), encoding="utf-8")
    verified = load_initial_query_sources(tmp_path, expected_sha256=digest)
    assert verified.sha256 == digest
    assert verified.origins == {"unit": {rows[0]["initial_query_signature_id"]: "q"}}

    with pytest.raises(ValueError, match="conflict"):
        load_initial_query_sources(tmp_path, expected_sha256="0" * 64)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        load_initial_query_sources(tmp_path)


def test_feature_bundle_preserves_map_bytes_and_records_digest(tmp_path: Path) -> None:
    root = tmp_path / "source"
    path, rows = _snapshot(root)
    _write(path, rows)
    (root / "splits").mkdir()
    (root / "bundle.json").write_text(
        json.dumps({"bundle_name": "source", "assets": {}, "models": {}, "expected_metrics": {}}), encoding="utf-8"
    )
    sources = load_initial_query_sources(root)
    output = tmp_path / "features"
    payload = _copy_bundle_support_files(load_bundle(root), output, sources)
    assert payload["initial_query_sources_sha256"] == sources.sha256
    assert (output / path.relative_to(root)).read_bytes() == path.read_bytes()
    path.write_bytes(path.read_bytes() + b" ")
    other_output = tmp_path / "changed"
    with pytest.raises(ValueError, match="changes after validation"):
        _copy_bundle_support_files(load_bundle(root), other_output, sources)
    assert not other_output.exists()


def test_original_signature_cannot_be_declared_as_a_derived_query(tmp_path: Path) -> None:
    path, rows = _snapshot(tmp_path, clones=False)
    assert load_initial_query_sources(tmp_path).origins == {}
    with pytest.raises(ValueError, match="SHA-256"):
        load_initial_query_sources(tmp_path, expected_sha256="0" * 64)

    rows[0]["initial_query_signature_id"] = "other"
    rows[0]["initial_paper_id"] = "other-paper"
    _write(path, rows)
    with pytest.raises(ValueError, match="Derived query IDs"):
        load_initial_query_sources(tmp_path)

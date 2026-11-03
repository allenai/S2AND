"""Protect derived query origins before native feature allocation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pytest

from scripts.production.model.initial_query_sources import derived_query_id, load_initial_query_sources
from scripts.production.model.train_linker_and_finalize import (
    _copy_bundle_support_files,
    load_bundle,
)
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


@pytest.mark.parametrize("missing", ["file", "entry"])
def test_derived_queries_require_complete_origins(tmp_path: Path, missing: str) -> None:
    path, _rows = _snapshot(tmp_path)
    if missing == "entry":
        _write(path, [])
    with pytest.raises(ValueError, match="[Mm]ap|origin"):
        load_initial_query_sources(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_query_signature_id", "unbound-source"),
        ("source_query_signature_id", "other"),
        ("source_paper_id", "wrong-paper"),
        ("initial_paper_id", "wrong-copy-paper"),
        ("dataset", "different-dataset"),
    ],
)
def test_origin_must_match_source_and_derived_records(tmp_path: Path, field: str, value: str) -> None:
    path, rows = _snapshot(tmp_path)
    rows[0][field] = value
    _write(path, rows)
    with pytest.raises(ValueError, match="[Mm]ap|origin|[Ss]ource|[Dd]erived"):
        load_initial_query_sources(tmp_path)


def test_declared_map_digest_is_enforced(tmp_path: Path) -> None:
    path, rows = _snapshot(tmp_path)
    _write(path, rows)
    (tmp_path / "bundle.json").write_text(json.dumps({"initial_query_sources_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="digest|SHA-256"):
        load_initial_query_sources(tmp_path)


def test_valid_map_and_plain_source_preserve_origins(tmp_path: Path) -> None:
    path, rows = _snapshot(tmp_path)
    _write(path, rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "bundle.json").write_text(json.dumps({"initial_query_sources_sha256": digest}), encoding="utf-8")
    assert load_initial_query_sources(tmp_path).origins == {"unit": {rows[0]["initial_query_signature_id"]: "q"}}
    plain = tmp_path / "plain"
    _snapshot(plain, clones=False)
    assert load_initial_query_sources(plain).origins == {}


def test_supplied_digest_requires_exact_bytes_and_agrees_with_declaration(tmp_path: Path) -> None:
    path, rows = _snapshot(tmp_path)
    _write(path, rows)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_initial_query_sources(tmp_path, expected_sha256=digest).sha256 == digest
    with pytest.raises(ValueError, match="SHA-256"):
        load_initial_query_sources(tmp_path, expected_sha256="0" * 64)
    (tmp_path / "bundle.json").write_text(json.dumps({"initial_query_sources_sha256": digest}), encoding="utf-8")
    with pytest.raises(ValueError, match="conflict"):
        load_initial_query_sources(tmp_path, expected_sha256="0" * 64)


@pytest.mark.parametrize("digest", [12])
def test_declared_digest_has_a_strict_type_and_format(tmp_path: Path, digest: object) -> None:
    _snapshot(tmp_path, clones=False)
    (tmp_path / "bundle.json").write_text(json.dumps({"initial_query_sources_sha256": digest}), encoding="utf-8")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        load_initial_query_sources(tmp_path)


def test_supplied_digest_requires_a_map_even_without_derived_queries(tmp_path: Path) -> None:
    _snapshot(tmp_path, clones=False)
    with pytest.raises(ValueError, match="SHA-256"):
        load_initial_query_sources(tmp_path, expected_sha256="0" * 64)


def test_three_field_map_uses_actual_paper_identities(tmp_path: Path) -> None:
    path, rows = _snapshot(tmp_path)
    rows[0].pop("source_paper_id")
    rows[0].pop("initial_paper_id")
    _write(path, rows)
    assert load_initial_query_sources(tmp_path).origins["unit"][rows[0]["initial_query_signature_id"]] == "q"
    rows[0]["source_query_signature_id"] = "other"
    _write(path, rows)
    with pytest.raises(ValueError, match="Derived query IDs"):
        load_initial_query_sources(tmp_path)


@pytest.mark.parametrize("variant", ["duplicate-clone", "duplicate-source", "self", "clone-source", "dataset-parent"])
def test_ambiguous_and_invalid_origin_records_are_rejected(tmp_path: Path, variant: str) -> None:
    path, rows = _snapshot(tmp_path)
    if variant.startswith("duplicate"):
        rows.append(dict(rows[0]))
        if variant == "duplicate-source":
            rows[1]["initial_query_signature_id"] = "different-copy"
    elif variant == "self":
        rows[0]["source_query_signature_id"] = rows[0]["initial_query_signature_id"]
    elif variant == "clone-source":
        rows[0]["source_query_signature_id"] = "visibleq:source"
    else:
        rows[0]["dataset"] = ".."
    _write(path, rows)
    with pytest.raises(ValueError, match="[Ss]ource-map|[Dd]uplicate"):
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
    rows[0]["initial_query_signature_id"] = "other"
    rows[0]["initial_paper_id"] = "other-paper"
    _write(path, rows)
    with pytest.raises(ValueError, match="Derived query IDs"):
        load_initial_query_sources(tmp_path)

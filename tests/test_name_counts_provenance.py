from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import s2and.name_counts_index as name_counts_index_module
import s2and.name_counts_manifest as name_counts_manifest_module
from s2and.data import ANDData
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_counts_index import NameCountsIndex
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple


def _write_index(root: Path, *, generation_id: str, first_count: int = 10) -> str:
    mappings = tiny_name_counts_tuple()
    mappings[0]["abdul"] = first_count
    provenance = {**tiny_name_counts_provenance(), "generation_id": generation_id}
    path, _metrics = write_name_counts_index(root, mappings, provenance, overwrite=True)
    return path


def test_name_counts_index_open_is_shared_for_one_manifest_generation(tmp_path: Path) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")

    first = NameCountsIndex.open(path)
    second = NameCountsIndex.open(path)

    assert first is second
    assert dict(first.source_provenance) == {
        **tiny_name_counts_provenance(),
        "generation_id": "generation-one",
    }


def test_name_counts_index_open_does_not_repeat_native_material_validation_in_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")

    def unexpected_python_hash(*args, **kwargs):
        pytest.fail("normal NameCountsIndex.open must use the native validation result")

    monkeypatch.setattr(name_counts_manifest_module, "_sha256_file", unexpected_python_hash)

    first = NameCountsIndex.open(path)
    second = NameCountsIndex.open(path)

    assert first is second


def test_name_counts_index_concurrent_open_parses_generation_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")
    real_parser = name_counts_index_module.json.loads
    parsing_started = threading.Event()
    release_parsing = threading.Event()
    parse_count = 0

    def block_parsing(*args, **kwargs):
        nonlocal parse_count
        parse_count += 1
        parsing_started.set()
        if not release_parsing.wait(timeout=10):
            raise TimeoutError("test did not release name-count manifest parsing")
        return real_parser(*args, **kwargs)

    monkeypatch.setattr(name_counts_index_module.json, "loads", block_parsing)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(NameCountsIndex.open, path)
        assert parsing_started.wait(timeout=10)
        second_future = pool.submit(NameCountsIndex.open, path)
        release_parsing.set()
        first = first_future.result(timeout=10)
        second = second_future.result(timeout=10)

    assert first is second
    assert parse_count == 1


def test_name_counts_index_manifest_replacement_opens_new_generation(tmp_path: Path) -> None:
    path = _write_index(tmp_path, generation_id="generation-one", first_count=10)
    first = NameCountsIndex.open(path)
    assert first.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [10.0]

    _write_index(tmp_path, generation_id="generation-two", first_count=99)
    second = NameCountsIndex.open(path)

    assert second is not first
    assert second.source_provenance["generation_id"] == "generation-two"
    assert second.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [99.0]
    assert first.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [10.0]


def test_name_counts_index_revalidates_replaced_generation_files(tmp_path: Path) -> None:
    path = _write_index(tmp_path, generation_id="generation-one", first_count=10)
    first = NameCountsIndex.open(path)

    _write_index(tmp_path, generation_id="generation-two", first_count=99)
    manifest_path = Path(path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_path = Path(path) / manifest["files"]["first"]["path"]
    payload = bytearray(first_path.read_bytes())
    payload[-1] ^= 1
    first_path.write_bytes(payload)
    manifest["test_generation_nonce"] = "corrupt-replacement"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        NameCountsIndex.open(path)
    assert first.lookup_many(["abdul"], [None], [None], [None])[0].tolist() == [10.0]


def test_name_counts_index_constructor_revalidates_native_binding() -> None:
    provenance = tiny_name_counts_provenance()
    native = SimpleNamespace(
        normalization_version="canonical_v2",
        name_counts_provenance_binding=("wrong-generation", "0" * 64, "snapshot", "1" * 64),
    )

    with pytest.raises(ValueError, match="native provenance mismatch"):
        NameCountsIndex(
            native=native,
            path="unused",
            manifest_sha256="2" * 64,
            normalization_version="canonical_v2",
            source_provenance=provenance,
        )


def test_anddata_exposes_read_only_index_provenance(tmp_path: Path) -> None:
    path = _write_index(tmp_path, generation_id="generation-one")
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        name="index-provenance",
        mode="inference",
        name_counts_index=path,
        preprocess=False,
    )

    assert dataset.name_counts_provenance is not None
    with pytest.raises(TypeError):
        dataset.name_counts_provenance["generation_id"] = "mutated"  # type: ignore[index]


def test_index_writer_rejects_incomplete_provenance(tmp_path: Path) -> None:
    provenance = tiny_name_counts_provenance()
    provenance.pop("selected_rows_sha256")

    with pytest.raises(ValueError, match="selected_rows_sha256"):
        write_name_counts_index(tmp_path, tiny_name_counts_tuple(), provenance)

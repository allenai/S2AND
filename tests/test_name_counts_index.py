"""Name-count index publication, native loading, and bounded writer contracts."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import weakref
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from s2and.arrow_inputs import MissingArrowArtifactError, require_name_counts_index_artifact
from s2and.incremental_linking import feature_block_arrow
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.name_counts_index import NameCountsIndex
from s2and.runtime import load_s2and_rust_extension
from tests.helpers import tiny_name_counts_tuple


@pytest.fixture(scope="module")
def write_index(tmp_path_factory: pytest.TempPathFactory) -> Callable[[Path], Path]:
    """Return a writer that copies one tiny index, so each case gets its own mutable index."""

    # The index writer is slow relative to a copy, and these tests read or corrupt only the published files.
    template_path, _metrics = write_name_counts_index(tmp_path_factory.mktemp("name_counts"), tiny_name_counts_tuple())
    template = Path(template_path)

    def write(root: Path) -> Path:
        target = root / "name_counts_index"
        shutil.copytree(template, target)
        return target

    return write


def test_open_cache_retains_only_four_paths(tmp_path: Path, write_index: Callable[[Path], Path]) -> None:
    references: list[weakref.ReferenceType[NameCountsIndex]] = []
    for index in range(5):
        path = write_index(tmp_path / str(index))
        opened = NameCountsIndex.open(path)
        assert NameCountsIndex.open(path) is opened
        references.append(weakref.ref(opened))
    del opened
    gc.collect()

    assert references[0]() is None
    assert all(reference() is not None for reference in references[1:])


_REQUIRED_MANIFEST_FIELDS = ("kind", "format_version", "files")


def _read_manifest(index_dir: Path) -> dict[str, Any]:
    return json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(index_dir: Path, manifest: dict[str, Any]) -> None:
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_native_validation_rejects(index_dir: Path, expected_field: str, *, verify: bool = False) -> None:
    with pytest.raises((OSError, RuntimeError, ValueError), match=expected_field):
        NameCountsIndex.open(index_dir, verify=verify)


def _remove_manifest_field(index_dir: Path, field: str) -> None:
    manifest = _read_manifest(index_dir)
    manifest.pop(field)
    _write_manifest(index_dir, manifest)


def test_valid_manifest_has_identical_python_and_rust_identity(
    tmp_path: Path, write_index: Callable[[Path], Path]
) -> None:
    index_dir = write_index(tmp_path)

    python_index = NameCountsIndex.open(index_dir)
    rust_index = load_s2and_rust_extension().NameCountsIndex.open(str(index_dir))
    manifest_sha256 = hashlib.sha256((index_dir / "manifest.json").read_bytes()).hexdigest()

    assert python_index.manifest_sha256 == manifest_sha256
    assert rust_index.name_counts_manifest_sha256 == manifest_sha256
    assert Path(python_index.path) == index_dir.resolve()


def test_native_manifest_validation_rejects_structural_identity_and_material_corruption(
    tmp_path: Path, write_index: Callable[[Path], Path]
) -> None:
    for field in _REQUIRED_MANIFEST_FIELDS:
        index_dir = write_index(tmp_path / f"missing-{field}")
        _remove_manifest_field(index_dir, field)
        _assert_native_validation_rejects(index_dir, field)

    cases = (
        ("wrong-kind", "kind", "other", "kind"),
        ("unsupported-format", "format_version", 2, "format_version"),
        ("string-format", "format_version", "1", "expected u32"),
    )
    for case_id, field, value, expected in cases:
        index_dir = write_index(tmp_path / case_id)
        manifest = _read_manifest(index_dir)
        manifest[field] = value
        _write_manifest(index_dir, manifest)

        _assert_native_validation_rejects(index_dir, expected)
    parent = tmp_path / "parent"
    write_index(parent)
    _assert_native_validation_rejects(parent, "does not contain manifest.json")

    index_dir = write_index(tmp_path / "missing-first")
    manifest = _read_manifest(index_dir)
    manifest["files"].pop("first")
    _write_manifest(index_dir, manifest)
    _assert_native_validation_rejects(index_dir, "first")

    index_dir = write_index(tmp_path / "missing-sha256")
    manifest = _read_manifest(index_dir)
    manifest["files"]["first"].pop("sha256")
    _write_manifest(index_dir, manifest)
    _assert_native_validation_rejects(index_dir, "sha256")

    cases = (
        ("root", "source_provenance"),
        ("files", "middle"),
        ("file-entry", "record_count"),
    )
    for mutation, expected_field in cases:
        index_dir = write_index(tmp_path / mutation)
        manifest = _read_manifest(index_dir)
        if mutation == "root":
            manifest["source_provenance"] = {}
        elif mutation == "files":
            manifest["files"]["middle"] = manifest["files"]["first"]
        else:
            manifest["files"]["first"]["record_count"] = 0
        _write_manifest(index_dir, manifest)

        _assert_native_validation_rejects(index_dir, expected_field)
    mutations = (
        ("byte_count_mismatch", "byte_count"),
        ("sha256_mismatch", "SHA-256"),
    )
    for mutation, expected_field in mutations:
        index_dir = write_index(tmp_path / mutation / "index")
        manifest = _read_manifest(index_dir)
        if mutation == "byte_count_mismatch":
            manifest["files"]["first"]["byte_count"] += 1
        elif mutation == "sha256_mismatch":
            manifest["files"]["first"]["sha256"] = "f" * 64
        else:  # pragma: no cover - mutation invariant
            raise AssertionError(f"unknown mutation {mutation}")
        _write_manifest(index_dir, manifest)
        # A fast open checks file sizes; only verification hashes the file contents.
        _assert_native_validation_rejects(index_dir, expected_field, verify=mutation == "sha256_mismatch")

    index_dir = write_index(tmp_path / "renamed-file")
    (index_dir / "first.bin").rename(index_dir / "renamed.bin")
    _assert_native_validation_rejects(index_dir, "first.bin")

    index_dir = write_index(tmp_path / "format-before-payload")
    manifest = _read_manifest(index_dir)
    manifest["format_version"] = 2
    _write_manifest(index_dir, manifest)
    (index_dir / "first.bin").unlink()
    _assert_native_validation_rejects(index_dir, "format_version")


def test_fast_open_checks_layout_and_verify_checks_contents(
    tmp_path: Path, write_index: Callable[[Path], Path]
) -> None:
    corrupt = write_index(tmp_path / "corrupt-name")
    first = corrupt / "first.bin"
    payload = bytearray(first.read_bytes())
    payload[-1] ^= 0x01
    first.write_bytes(bytes(payload))

    opened = NameCountsIndex.open(corrupt)
    with pytest.raises(ValueError, match="SHA-256"):
        opened.verify()
    with pytest.raises(ValueError, match="SHA-256"):
        NameCountsIndex.open(corrupt, verify=True)
    with pytest.raises(ValueError, match="SHA-256"):
        load_s2and_rust_extension().NameCountsIndex.open(str(corrupt), verify=True)
    with pytest.raises(MissingArrowArtifactError, match="SHA-256"):
        require_name_counts_index_artifact(corrupt, context="publication check", producer_hint="rebuild", verify=True)

    bad_header = write_index(tmp_path / "corrupt-header")
    header = bytearray((bad_header / "last.bin").read_bytes())
    header[0:8] = b"NOTANIDX"
    (bad_header / "last.bin").write_bytes(bytes(header))
    _assert_native_validation_rejects(bad_header, "magic")

    valid = NameCountsIndex.open(write_index(tmp_path / "valid"), verify=True)
    assert valid.verify() is None


def test_arrow_boundary_translates_representative_manifest_failure(
    tmp_path: Path, write_index: Callable[[Path], Path]
) -> None:
    index_dir = write_index(tmp_path)
    _remove_manifest_field(index_dir, "format_version")

    with pytest.raises(MissingArrowArtifactError, match="format_version"):
        require_name_counts_index_artifact(
            index_dir,
            context="test Arrow boundary",
            producer_hint="write a complete test index",
        )


class _DuplicateNameMapping(Mapping[str, float]):
    """Deliberately violate Mapping uniqueness to exercise streaming validation."""

    def __getitem__(self, key: str) -> float:
        if key != "ada":
            raise KeyError(key)
        return 2.0

    def __iter__(self) -> Iterator[str]:
        return iter(("ada", "ada"))

    def __len__(self) -> int:
        return 2


def _read_index(path: Path) -> dict[str, float]:
    header_struct = feature_block_arrow._NAME_COUNTS_INDEX_HEADER_STRUCT
    record_struct = feature_block_arrow._NAME_COUNTS_INDEX_RECORD_STRUCT
    payload = path.read_bytes()
    magic, record_count, blob_offset, blob_size = header_struct.unpack_from(payload)
    assert magic == feature_block_arrow._NAME_COUNTS_INDEX_MAGIC
    blob = payload[blob_offset : blob_offset + blob_size]
    observed: dict[str, float] = {}
    for index in range(record_count):
        offset = header_struct.size + index * record_struct.size
        _hash_1, _hash_2, name_offset, name_length, reserved, count = record_struct.unpack_from(payload, offset)
        assert reserved == 0
        observed[blob[name_offset : name_offset + name_length].decode("utf-8")] = count
    return observed


def test_bounded_sort_runs_preserve_exact_binary_lookup_values(tmp_path: Path) -> None:
    names = ("ada", "bea", "cy", "dee", "eve", "fay", "gia", "hal", "ian", "jay", "kim")
    mapping = {name: float(index + 2) for index, name in enumerate(names)}
    output_path = tmp_path / "first.bin"
    metrics = feature_block_arrow._write_name_count_index_file(
        output_path,
        "first",
        mapping,
        max_records_in_memory=3,
    )

    assert _read_index(output_path) == mapping
    assert metrics["sort_run_count"] == 4
    assert metrics["peak_buffered_records"] == 3
    assert metrics["record_count"] == len(mapping)
    assert metrics["temporary_byte_count"] > metrics["byte_count"]
    assert list(tmp_path.glob(".first.bin.*")) == []


def test_writer_rejects_duplicate_logical_utf8_name_without_unbounded_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate UTF-8 name 'ada'"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            (_DuplicateNameMapping(), {}, {}, {}),
        )


def test_writer_rejects_keys_that_would_collide_after_stringification(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="name-count first keys must be strings"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            ({1: 2.0, "1": 3.0}, {}, {}, {}),
        )


@pytest.mark.parametrize(
    ("kind", "name"),
    (
        ("first", "Ada"),
        ("first", "ada!"),
        ("first", "ada  marie"),
        ("first", "a"),
        ("last", ""),
        ("first_last", "adasmith"),
        ("first_last", "a smith"),
        ("last_first_initial", "smith ad"),
    ),
    ids=(
        "first-uppercase",
        "first-punctuation",
        "first-repeated-space",
        "first-single-letter",
        "last-empty",
        "first-last-unseparated",
        "first-last-short-first",
        "last-first-initial-long",
    ),
)
def test_writer_rejects_noncanonical_name_count_keys(tmp_path: Path, kind: str, name: str) -> None:
    mapping_kinds = ("first", "last", "first_last", "last_first_initial")
    mappings = {mapping_kind: {} for mapping_kind in mapping_kinds}
    mappings[kind] = {name: 2.0}

    with pytest.raises(ValueError) as exc_info:
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            tuple(mappings[mapping_kind] for mapping_kind in mapping_kinds),
        )
    assert f"name-count {kind} key" in str(exc_info.value)
    assert "canonical_v2" in str(exc_info.value)


def test_name_count_key_contract_accepts_producible_canonical_keys() -> None:
    cases = (
        ("first-single-token", "first", "ada"),
        ("first-multiple-token", "first", "a b"),
        ("last-particle", "last", "van der berg"),
        ("first-last", "first_last", "ada smith"),
        ("first-multiple-last", "first_last", "a b smith"),
        ("first-last-particle", "first_last", "ada van der berg"),
        ("last-particle-initial", "last_first_initial", "van der berg a"),
    )
    for case_id, kind, name in cases:
        assert feature_block_arrow._validated_name_count_entry(kind, name, 2.0) == (name, 2.0), case_id


@pytest.mark.parametrize("count", (float("nan"), 0.0), ids=("nan", "zero"))
def test_writer_rejects_nonfinite_and_nonpositive_counts(tmp_path: Path, count: float) -> None:
    with pytest.raises(ValueError, match="must be a finite positive number"):
        feature_block_arrow.write_name_counts_index(
            tmp_path,
            ({"ada": count}, {}, {}, {}),
        )


def test_fresh_writer_validates_each_name_count_entry_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mappings = ({"ada": 2.0, "grace": 3.0}, {"smith": 4.0}, {"ada smith": 2.0}, {"smith a": 2.0})
    validation_calls = 0
    original_validator = feature_block_arrow._validated_name_count_entry

    def counted_validator(kind: str, raw_name: object, raw_count: object) -> tuple[str, float]:
        nonlocal validation_calls
        validation_calls += 1
        return original_validator(kind, raw_name, raw_count)

    monkeypatch.setattr(feature_block_arrow, "_validated_name_count_entry", counted_validator)

    feature_block_arrow.write_name_counts_index(
        tmp_path,
        mappings,
    )

    assert validation_calls == sum(len(mapping) for mapping in mappings)


@pytest.mark.parametrize(
    ("mappings", "expected_row_count", "expected_first_count"),
    [
        (({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7}), 4, 1),
        (
            (
                {"ada": 2.0, "grace": 3.0},
                {"lovelace": 4.0, "hopper": 5.0},
                {"ada lovelace": 2.0, "grace hopper": 3.0},
                {"lovelace a": 2.0, "hopper g": 3.0},
            ),
            8,
            2,
        ),
    ],
    ids=["singleton-integer-counts", "two-rows-per-kind"],
)
def test_identical_counts_produce_byte_identical_flat_indexes(
    tmp_path: Path,
    mappings: tuple[dict[str, int | float], ...],
    expected_row_count: int,
    expected_first_count: int,
) -> None:
    reversed_mappings = tuple(dict(reversed(tuple(mapping.items()))) for mapping in mappings)
    outputs = [
        feature_block_arrow.write_name_counts_index(tmp_path / label, values)
        for label, values in (("first", mappings), ("second", reversed_mappings))
    ]
    indexes = [Path(index_path) for index_path, _metrics in outputs]
    for _index_path, metrics in outputs:
        assert "reused" not in metrics
        assert metrics["row_count"] == expected_row_count
        assert metrics["first_count"] == expected_first_count
    kinds = ("first", "last", "first_last", "last_first_initial")
    expected_files = {"manifest.json", *(f"{kind}.bin" for kind in kinds)}

    assert all({path.name for path in index.iterdir()} == expected_files for index in indexes)
    manifests = [(index / "manifest.json").read_bytes() for index in indexes]
    assert manifests[0] == manifests[1]
    manifest = json.loads(manifests[0])
    assert manifest["kind"] == "s2and_name_counts"
    assert manifest["format_version"] == 1
    assert all(set(entry) == {"byte_count", "sha256"} for entry in manifest["files"].values())


def test_reused_kind_hash_seed_preserves_binary_index_hashes() -> None:
    for case_id, kind, name in (
        ("first-ascii", "first", "ada"),
        ("last-unicode", "last", "李"),
        ("first-last-spaces", "first_last", "anne marie o connor"),
    ):
        name_bytes = name.encode("utf-8")
        expected = (
            feature_block_arrow._fnv64_bytes(name_bytes),
            feature_block_arrow._fnv64_bytes(
                feature_block_arrow._NAME_COUNTS_INDEX_HASH_DOMAIN + kind.encode("utf-8") + b"\x00" + name_bytes
            ),
        )

        assert feature_block_arrow._name_counts_index_hashes(kind, name_bytes) == expected, case_id


def _temporary_indexes(root: Path) -> list[Path]:
    return list(root.glob(".name_counts_index.*"))


def test_existing_target_is_never_reused_or_replaced(tmp_path: Path) -> None:
    index_path, _metrics = write_name_counts_index(
        tmp_path,
        tiny_name_counts_tuple(),
    )
    manifest_path = Path(index_path) / "manifest.json"
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(FileExistsError, match="target already exists"):
        write_name_counts_index(
            tmp_path,
            tiny_name_counts_tuple(),
        )

    assert manifest_path.read_bytes() == original_manifest
    assert _temporary_indexes(tmp_path) == []


def test_all_material_is_built_before_target_appears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_writer = feature_block_arrow._write_name_count_index_file
    written_kinds: list[str] = []

    def observe_write(path: Path, kind: str, mapping: Any, **kwargs: Any) -> dict[str, int]:
        assert not (tmp_path / "name_counts_index").exists()
        assert path.is_relative_to(tmp_path)
        assert path.parent.name.startswith(".name_counts_index.")
        assert path.name == f"{kind}.bin"
        written_kinds.append(kind)
        return real_writer(path, kind, mapping, **kwargs)

    monkeypatch.setattr(feature_block_arrow, "_write_name_count_index_file", observe_write)

    index_path, _metrics = write_name_counts_index(
        tmp_path,
        tiny_name_counts_tuple(),
    )

    assert written_kinds == ["first", "last", "first_last", "last_first_initial"]
    assert Path(index_path).is_dir()
    assert _temporary_indexes(tmp_path) == []


def test_failed_material_write_leaves_target_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_writer = feature_block_arrow._write_name_count_index_file

    def fail_on_last(path: Path, kind: str, mapping: Any, **kwargs: Any) -> dict[str, int]:
        if kind == "last":
            raise OSError("injected material failure")
        return real_writer(path, kind, mapping, **kwargs)

    monkeypatch.setattr(feature_block_arrow, "_write_name_count_index_file", fail_on_last)

    with pytest.raises(OSError, match="injected material failure"):
        write_name_counts_index(tmp_path, tiny_name_counts_tuple())

    assert not (tmp_path / "name_counts_index").exists()
    assert _temporary_indexes(tmp_path) == []


def test_failed_final_rename_leaves_target_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_rename = Path.rename

    def fail_publication(path: Path, target: str | Path) -> Path:
        if Path(target) == tmp_path / "name_counts_index":
            raise OSError("injected publication failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publication)

    with pytest.raises(OSError, match="injected publication failure"):
        write_name_counts_index(tmp_path, tiny_name_counts_tuple())

    assert not (tmp_path / "name_counts_index").exists()
    assert _temporary_indexes(tmp_path) == []

from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import threading
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from s2and.consts import NORMALIZATION_VERSION, PROJECT_ROOT_PATH
from s2and.subblocking import _LazyCanonicalOrcidPrefixCounts, _load_canonical_orcid_prefix_counts

ORCID_1 = "0000-0000-0000-0001"
ORCID_2 = "0000-0000-0000-0002"


def _load_module() -> ModuleType:
    module_path = Path(PROJECT_ROOT_PATH) / "scripts" / "production" / "counts" / "generate_orcid_name_prefix_counts.py"
    spec = importlib.util.spec_from_file_location("generate_orcid_name_prefix_counts", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _publish_in_process(output_dir: str, barrier: Any, results: Any) -> None:
    module = _load_module()
    barrier.wait(timeout=10)
    try:
        module.publish_generation(
            {"al": {"am": 7}},
            output_dir=Path(output_dir),
            source_snapshot_id="fixture",
            source_digest="a" * 64,
            metrics={"source_rows": 2},
            overwrite=False,
        )
    except Exception as error:  # noqa: BLE001 - exception class is the subprocess result under test
        results.put(("error", type(error).__name__))
    else:
        results.put(("ok", None))


def test_import_is_side_effect_free_without_internal_pys2() -> None:
    module = _load_module()

    assert callable(module.main)


def test_package_data_includes_versioned_orcid_count_generation_files() -> None:
    with (Path(PROJECT_ROOT_PATH) / "pyproject.toml").open("rb") as stream:
        setuptools = tomllib.load(stream)["tool"]["setuptools"]
    package_data = setuptools["package-data"]["s2and"]
    excluded_package_data = setuptools["exclude-package-data"]["s2and"]

    assert "data/first_k_letter_counts_from_orcid.json" not in package_data
    assert "data/first_k_letter_counts_from_orcid.meta.json" not in package_data
    assert "data/first_k_letter_counts_from_orcid.json" in excluded_package_data
    assert "data/first_k_letter_counts_from_orcid.meta.json" in excluded_package_data
    assert "data/first_k_letter_counts_from_orcid.manifest.json" in package_data
    assert "data/orcid-prefix-counts-*/*.json" in package_data


def test_empty_canonical_names_are_rejected_with_metrics() -> None:
    module = _load_module()
    groups, metrics = module.canonical_orcid_name_groups(
        [
            {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
            {"orcid": ORCID_1, "first_name": "", "middle": None},
            {"orcid": ORCID_1, "first_name": "...", "middle": None},
            {"orcid": "", "first_name": "Amy", "middle": None},
            {"orcid": "not-an-orcid", "first_name": "Ava", "middle": None},
        ]
    )

    assert groups == {ORCID_1: {"alice"}}
    assert metrics["accepted_rows"] == 1
    assert metrics["rejected_empty_canonical_first"] == 2
    assert metrics["rejected_missing_orcid"] == 1
    assert metrics["rejected_invalid_orcid"] == 1


def test_prefix_counts_are_unordered_and_deterministic() -> None:
    module = _load_module()
    forward, _ = module.build_prefix_counts(
        {"o1": ["alice", "amy"]},
        {("alicia", "amanda")},
        min_orcid_count=1,
        min_alias_count=1,
    )
    reverse, _ = module.build_prefix_counts(
        {"o1": ["amy", "alice"]},
        {("amanda", "alicia")},
        min_orcid_count=1,
        min_alias_count=1,
    )

    assert forward == reverse
    assert all(left <= right for left, nested in forward.items() for right in nested)


def test_fixture_cli_publishes_generation_then_manifest(tmp_path: Path) -> None:
    module = _load_module()
    fixture_path = tmp_path / "rows.json"
    fixture_path.write_text(
        json.dumps(
            [
                {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
                {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    assert (
        module.main(
            [
                "--input-json",
                str(fixture_path),
                "--output-dir",
                str(output_dir),
                "--source-snapshot-id",
                "fixture-2026-07-09",
            ]
        )
        == 0
    )

    pointer_path = output_dir / "first_k_letter_counts_from_orcid.manifest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    generation_dir = output_dir / pointer["generation_dir"]
    metadata = json.loads((generation_dir / "first_k_letter_counts_from_orcid.meta.json").read_text(encoding="utf-8"))
    assert metadata["normalization_version"] == NORMALIZATION_VERSION
    assert metadata["pair_key_semantics"] == "unordered_lexicographic"
    assert (generation_dir / "first_k_letter_counts_from_orcid.json").is_file()

    with pytest.raises(FileExistsError, match="Manifest already exists"):
        module.main(
            [
                "--input-json",
                str(fixture_path),
                "--output-dir",
                str(output_dir),
                "--source-snapshot-id",
                "fixture-2026-07-09",
            ]
        )


def test_runtime_loader_is_lazy_and_verifies_the_published_generation(tmp_path: Path) -> None:
    module = _load_module()
    lazy_counts = _LazyCanonicalOrcidPrefixCounts(tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing canonical ORCID prefix-count manifest"):
        len(lazy_counts)

    module.publish_generation(
        {"al": {"am": 7}},
        output_dir=tmp_path,
        source_snapshot_id="fixture",
        source_digest="a" * 64,
        metrics={"source_rows": 2},
        overwrite=False,
    )
    assert _load_canonical_orcid_prefix_counts(tmp_path) == {"al": {"am": 7}}
    assert dict(lazy_counts) == {"al": {"am": 7}}
    binding = lazy_counts.binding()
    assert binding["schema_version"] == "s2and_orcid_prefix_counts_binding_v1"
    assert binding["generation_id"].startswith("fixture-")
    assert binding["data_sha256"]

    pointer = json.loads((tmp_path / "first_k_letter_counts_from_orcid.manifest.json").read_text(encoding="utf-8"))
    data_path = tmp_path / pointer["generation_dir"] / "first_k_letter_counts_from_orcid.json"
    data_path.write_text('{"al":{"az":9}}', encoding="utf-8")
    with pytest.raises(ValueError, match="data SHA-256"):
        _load_canonical_orcid_prefix_counts(tmp_path)


def test_runtime_loader_rejects_manifest_path_escape(tmp_path: Path) -> None:
    (tmp_path / "first_k_letter_counts_from_orcid.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_id": "fixture-aaaaaaaaaaaa",
                "generation_dir": "../orcid-prefix-counts-fixture-aaaaaaaaaaaa",
                "metadata_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generation_dir must exactly match"):
        _load_canonical_orcid_prefix_counts(tmp_path)


def test_runtime_loader_rejects_boolean_cardinality(tmp_path: Path) -> None:
    module = _load_module()
    module.publish_generation(
        {"al": {"am": 7}},
        output_dir=tmp_path,
        source_snapshot_id="fixture",
        source_digest="a" * 64,
        metrics={},
        overwrite=False,
    )
    pointer_path = tmp_path / "first_k_letter_counts_from_orcid.manifest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    metadata_path = tmp_path / pointer["generation_dir"] / "first_k_letter_counts_from_orcid.meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["outer_key_cardinality"] = True
    metadata_bytes = json.dumps(metadata, sort_keys=True, indent=2).encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)
    pointer["metadata_sha256"] = hashlib.sha256(metadata_bytes).hexdigest()
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(ValueError, match="outer_key_cardinality must be a nonnegative integer"):
        _load_canonical_orcid_prefix_counts(tmp_path)


def test_publish_rechecks_no_overwrite_under_the_manifest_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    arrivals = threading.Barrier(2)
    serialization_lock = threading.Lock()

    @contextmanager
    def gated_publish_lock(_output_dir: Path) -> Iterator[None]:
        arrivals.wait(timeout=5)
        with serialization_lock:
            yield

    monkeypatch.setattr(module, "_publish_lock", gated_publish_lock)

    def publish() -> Path:
        return module.publish_generation(
            {"al": {"am": 7}},
            output_dir=tmp_path,
            source_snapshot_id="fixture",
            source_digest="a" * 64,
            metrics={"source_rows": 2},
            overwrite=False,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(publish) for _ in range(2)]
    failures = [future.exception() for future in futures if future.exception() is not None]
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    generation_dirs = list(tmp_path.glob("orcid-prefix-counts-*"))
    assert len(generation_dirs) == 1


def test_publish_lock_serializes_real_processes(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [context.Process(target=_publish_in_process, args=(str(tmp_path), barrier, results)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    observed = sorted(results.get(timeout=5) for _ in processes)
    assert observed == [("error", "FileExistsError"), ("ok", None)]
    assert len(list(tmp_path.glob("orcid-prefix-counts-*"))) == 1


@pytest.mark.parametrize("failure_kind", ["read_error", "malformed_json"])
def test_post_replace_pointer_inspection_failure_retains_published_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    module = _load_module()
    pointer_path = tmp_path / "first_k_letter_counts_from_orcid.manifest.json"
    if failure_kind == "read_error":
        original_read_text = Path.read_text

        def fail_published_pointer_read(path: Path, *args: object, **kwargs: object) -> str:
            if path == pointer_path and path.exists():
                raise OSError("injected post-replace read failure")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_published_pointer_read)
        expected_error = OSError
        expected_message = "Unable to read published ORCID prefix-count pointer"
    else:
        original_replace = module.os.replace

        def corrupt_replaced_pointer(source: Path, target: Path) -> None:
            original_replace(source, target)
            if Path(target) == pointer_path:
                Path(target).write_text("{", encoding="utf-8")

        monkeypatch.setattr(module.os, "replace", corrupt_replaced_pointer)
        expected_error = ValueError
        expected_message = "pointer is invalid JSON"

    with pytest.raises(expected_error, match=expected_message):
        module.publish_generation(
            {"al": {"am": 7}},
            output_dir=tmp_path,
            source_snapshot_id="fixture",
            source_digest="a" * 64,
            metrics={"source_rows": 2},
            overwrite=False,
        )

    generation_dirs = list(tmp_path.glob("orcid-prefix-counts-*"))
    assert len(generation_dirs) == 1
    assert generation_dirs[0].is_dir()
    assert pointer_path.is_file()


def test_invalid_pointer_during_failed_publication_does_not_mask_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    pointer_path = tmp_path / "first_k_letter_counts_from_orcid.manifest.json"
    pointer_path.write_text("{", encoding="utf-8")
    original_replace = module.os.replace

    def fail_pointer_replace(source: Path, target: Path) -> None:
        if Path(target) == pointer_path:
            raise OSError("injected primary replace failure")
        original_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_pointer_replace)
    with pytest.raises(OSError, match="injected primary replace failure") as exc_info:
        module.publish_generation(
            {"al": {"am": 7}},
            output_dir=tmp_path,
            source_snapshot_id="fixture",
            source_digest="a" * 64,
            metrics={"source_rows": 2},
            overwrite=True,
        )

    assert "Retained generation" in "\n".join(exc_info.value.__notes__)
    assert len(list(tmp_path.glob("orcid-prefix-counts-*"))) == 1


def test_streaming_source_digest_covers_selected_row_content() -> None:
    module = _load_module()
    rows = [
        {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
    ]
    counts, metrics, digest = module.build_prefix_counts_from_sorted_rows(
        rows,
        [],
        min_orcid_count=1,
    )
    changed_counts, _, changed_digest = module.build_prefix_counts_from_sorted_rows(
        [*rows[:1], {"orcid": ORCID_1, "first_name": "Ava", "middle": None}],
        [],
        min_orcid_count=1,
    )

    assert counts != changed_counts
    assert digest != changed_digest
    assert metrics["source_rows"] == 2
    assert metrics["max_unique_names_per_orcid"] == 2


def test_compact_json_writer_matches_canonical_encoding(tmp_path: Path) -> None:
    module = _load_module()
    payload = {"zo": {"gian ": 2, "amy": 4}, "al": {"bob": 3}}
    path = tmp_path / "counts.json"

    digest, byte_count = module._write_compact_json(path, payload)
    expected = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    assert path.read_bytes() == expected
    assert byte_count == len(expected)
    assert digest == hashlib.sha256(expected).hexdigest()


def test_source_digest_covers_deduplicated_rows_and_name_tuple_content() -> None:
    module = _load_module()
    rows = [
        {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
    ]
    counts, _, digest = module.build_prefix_counts_from_sorted_rows(rows, [("alicia", "amanda")], min_orcid_count=1)
    duplicate_counts, _, duplicate_digest = module.build_prefix_counts_from_sorted_rows(
        [*rows, rows[-1]],
        [("alicia", "amanda")],
        min_orcid_count=1,
    )
    alias_counts, _, alias_digest = module.build_prefix_counts_from_sorted_rows(
        rows,
        [("alicia", "ava")],
        min_orcid_count=1,
        min_alias_count=1,
    )

    assert duplicate_counts == counts
    assert duplicate_digest == digest
    assert alias_counts != counts
    assert alias_digest != digest


def test_publication_rejects_noncanonical_count_pairs_before_writing(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="lexicographically ordered"):
        module.publish_generation(
            {"am": {"al": 7}},
            output_dir=tmp_path,
            source_snapshot_id="fixture",
            source_digest="a" * 64,
            metrics={},
            overwrite=False,
        )

    assert not list(tmp_path.iterdir())


def test_publication_rejects_non_ascii_prefixes_before_writing(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="lowercase printable ASCII prefixes"):
        module.publish_generation(
            {"ál": {"amy": 7}},
            output_dir=tmp_path,
            source_snapshot_id="fixture",
            source_digest="a" * 64,
            metrics={},
            overwrite=False,
        )

    assert not list(tmp_path.iterdir())


def test_name_pair_expansion_has_an_explicit_per_orcid_bound() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="max_names_per_orcid=2"):
        module.build_prefix_counts(
            {"o1": ["alice", "amy", "ava"]},
            [],
            min_orcid_count=1,
            max_names_per_orcid=2,
        )

    with pytest.raises(ValueError, match="max_names_per_orcid=2"):
        module.build_prefix_counts_from_sorted_rows(
            [
                {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
                {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
                {"orcid": ORCID_1, "first_name": "Ava", "middle": None},
            ],
            [],
            min_orcid_count=1,
            max_names_per_orcid=2,
        )


def test_cli_refuses_implicit_warehouse_access(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="explicitly authorize warehouse access"):
        module.main(
            [
                "--output-dir",
                str(tmp_path),
                "--source-snapshot-id",
                "fixture",
            ]
        )

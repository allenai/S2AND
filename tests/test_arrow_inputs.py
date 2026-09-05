from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import BinaryIO

import pytest

import s2and.arrow_inputs as arrow_inputs
from s2and.arrow_inputs import (
    ArrowDataset,
    MissingArrowArtifactError,
    build_arrow_artifact_manifest,
    normalize_arrow_paths,
    require_name_counts_index_artifact,
    write_arrow_artifact_manifest,
)
from s2and.incremental_linking.feature_block import (
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_name_counts_index,
)
from tests.helpers import tiny_name_counts_tuple


@pytest.fixture(autouse=True)
def _fake_native_arrow_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    def construct(retained, native_name_counts_index):
        for source in retained.values():
            os.fstat(source.handle.fileno())
        return SimpleNamespace(
            keys=frozenset(retained),
            name_counts_index=native_name_counts_index,
            batch_indices=lambda _table_key, _values: [0],
        )

    monkeypatch.setattr(arrow_inputs, "_construct_native_arrow_dataset", construct)


def _write_dataset(
    root: Path,
    *,
    include_specter: bool = False,
    include_name_counts: bool = False,
    signature_id: str = "s1",
) -> dict[str, str]:
    import pyarrow as pa

    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "signatures": pa.table(
            {
                "signature_id": pa.array([signature_id], type=pa.string()),
                "paper_id": pa.array(["p1"], type=pa.string()),
                "author_first": pa.array(["a"], type=pa.string()),
                "author_middle": pa.array([""], type=pa.string()),
                "author_last": pa.array(["b"], type=pa.string()),
                "author_suffix": pa.array([""], type=pa.string()),
                "author_affiliations": pa.array([[]], type=pa.list_(pa.string())),
                "author_position": pa.array([0], type=pa.int64()),
            }
        ),
        "papers": pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "title": pa.array(["title"], type=pa.string()),
                "venue": pa.array(["venue"], type=pa.string()),
                "journal_name": pa.array(["journal"], type=pa.string()),
            }
        ),
        "paper_authors": pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
                "author_name": pa.array(["a b"], type=pa.string()),
            }
        ),
    }
    if include_specter:
        tables["specter"] = pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.array([[0.1, 0.2]], type=pa.list_(pa.float32(), 2)),
            }
        )
    paths: dict[str, str] = {}
    for table_name, table in tables.items():
        table_path = root / f"{table_name}.arrow"
        index_path = root / f"{table_name}.index"
        write_arrow_ipc_table(table, table_path)
        key_column = "signature_id" if table_name == "signatures" else "paper_id"
        write_arrow_batch_lookup_index(table_path, index_path, key_column=key_column)
        paths[table_name] = str(table_path)
        paths[f"{table_name}_batch_index"] = str(index_path)
    if include_name_counts:
        name_counts_path, _metrics = write_name_counts_index(root, tiny_name_counts_tuple())
        paths["name_counts_index"] = name_counts_path
    write_arrow_artifact_manifest(build_arrow_artifact_manifest(paths, root), root)
    return paths


def _rewrite_manifest(root: Path, transform) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    transform(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_writer_owns_runtime_fields_and_publishes_atomically(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    signatures_path.write_bytes(b"signatures")

    manifest = build_arrow_artifact_manifest(
        {"signatures": signatures_path},
        tmp_path,
    )
    manifest_path = write_arrow_artifact_manifest(manifest, tmp_path)

    assert manifest["kind"] == "s2and_arrow_dataset"
    assert manifest["format_version"] == 1
    assert manifest["paths"] == {"signatures": "signatures.arrow"}
    assert set(manifest["files"]["signatures"]) == {"byte_count", "sha256"}
    assert set(manifest) == {"kind", "format_version", "paths", "files"}
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize("failure_stage", ["write", "replace"])
def test_failed_manifest_publication_preserves_previous_generation_and_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """A failed commit must leave the previous manifest usable for retry."""
    artifact = tmp_path / "signatures.arrow"
    artifact.write_bytes(b"first generation")
    original = build_arrow_artifact_manifest({"signatures": artifact}, tmp_path)
    manifest_path = write_arrow_artifact_manifest(original, tmp_path)
    original_bytes = manifest_path.read_bytes()
    artifact.write_bytes(b"second generation")
    replacement = build_arrow_artifact_manifest({"signatures": artifact}, tmp_path)

    def fail_replace(source: Path, target: Path) -> None:
        assert target == manifest_path
        assert json.loads(source.read_text(encoding="utf-8")) == replacement
        raise OSError("injected manifest publication failure")

    real_temporary_file = arrow_inputs.tempfile.NamedTemporaryFile

    def fail_write(*args, **kwargs):
        output = real_temporary_file(*args, **kwargs)
        original_write = output.write

        def write_partial_manifest(text: str) -> None:
            original_write(text[:1])
            raise OSError("injected manifest publication failure")

        output.write = write_partial_manifest
        return output

    with monkeypatch.context() as patch:
        if failure_stage == "write":
            patch.setattr(arrow_inputs.tempfile, "NamedTemporaryFile", fail_write)
        else:
            patch.setattr(Path, "replace", fail_replace)
        with pytest.raises(OSError, match="injected manifest publication failure"):
            write_arrow_artifact_manifest(replacement, tmp_path)

    assert manifest_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".manifest.json.*.tmp")) == []
    write_arrow_artifact_manifest(replacement, tmp_path)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == replacement


def test_manifest_writer_rejects_artifacts_outside_its_authority(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.arrow"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="must remain within manifest directory"):
        build_arrow_artifact_manifest({"signatures": outside}, root)


def test_manifest_writer_allows_relative_shared_name_counts_layouts(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    dataset_root = release_root / "replay" / "datasets" / "tiny"
    dataset_root.mkdir(parents=True)
    signatures = dataset_root / "signatures.arrow"
    signatures.write_bytes(b"signatures")
    name_counts = release_root / "name_counts_index"
    name_counts.mkdir()
    (name_counts / "manifest.json").write_text("{}", encoding="utf-8")

    manifest = build_arrow_artifact_manifest(
        {"signatures": signatures, "name_counts_index": name_counts},
        dataset_root,
    )

    assert manifest["paths"]["name_counts_index"] == "../../../name_counts_index"
    assert (
        arrow_inputs._portable_manifest_path(  # noqa: SLF001
            PureWindowsPath("..") / ".." / ".." / "name_counts_index"
        )
        == "../../../name_counts_index"
    )
    other = release_root / "other"
    other.mkdir()
    (other / "manifest.json").write_text("{}", encoding="utf-8")
    other_manifest = build_arrow_artifact_manifest({"name_counts_index": other}, dataset_root)
    assert other_manifest["paths"]["name_counts_index"] == "../../../other"


def test_collection_root_requires_nonempty_confined_child_bindings(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    child_manifest = root / "dataset" / "manifest.json"
    child_manifest.parent.mkdir(parents=True)
    child_manifest.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(child_manifest.read_bytes()).hexdigest()

    for child_path, message in (
        (str(child_manifest.resolve()), "must be relative"),
        ("../outside.json", "escapes its root"),
    ):
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "kind": arrow_inputs.ARROW_COLLECTION_KIND,
                    "format_version": 1,
                    "dataset_manifests": {"dataset": {"path": child_path, "sha256": digest}},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=message):
            arrow_inputs.read_arrow_collection_root(root / "manifest.json")

    valid_dataset_binding = {
        "dataset": {
            "path": "dataset/manifest.json",
            "sha256": digest,
        }
    }
    for kind, release_fields, replay_bundles, message in (
        (
            arrow_inputs.ARROW_COLLECTION_KIND,
            {},
            valid_dataset_binding,
            r"fields mismatch.*extra=\['replay_bundles'\]",
        ),
        (
            arrow_inputs.PUBLIC_DATA_KIND,
            {"release_version": "1.3"},
            {},
            "replay_bundles must be a nonempty object",
        ),
    ):
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "kind": kind,
                    **release_fields,
                    "format_version": 1,
                    "dataset_manifests": valid_dataset_binding,
                    "replay_bundles": replay_bundles,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=message):
            arrow_inputs.read_arrow_collection_root(root / "manifest.json")


def test_generation_identity_uses_role_and_content_not_filename(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = first_root / "specter2.arrow"
    renamed = second_root / "renamed.arrow"
    first.write_bytes(b"same-content")
    renamed.write_bytes(b"same-content")

    first_manifest = build_arrow_artifact_manifest({"specter": first}, first_root)
    renamed_manifest = build_arrow_artifact_manifest({"specter": renamed}, second_root)

    assert first_manifest["paths"]["specter"] != renamed_manifest["paths"]["specter"]
    assert first_manifest["files"] == renamed_manifest["files"]


def test_cold_open_hashes_each_file_once_and_reuse_never_rehashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_dataset(tmp_path)
    original = arrow_inputs._hash_retained_file  # noqa: SLF001
    hashed: list[Path] = []

    def record(source):
        hashed.append(source.path)
        return original(source)

    monkeypatch.setattr(arrow_inputs, "_hash_retained_file", record)
    dataset = ArrowDataset.open(tmp_path)
    expected = {Path(path).resolve() for path in paths.values()}
    assert set(hashed) == expected
    assert len(hashed) == len(expected)
    monkeypatch.setattr(
        arrow_inputs,
        "_hash_retained_file",
        lambda _source: (_ for _ in ()).throw(AssertionError("dataset reuse rehashed files")),
    )

    for _ in range(3):
        with dataset.use() as lease:
            assert lease.native is dataset.native
    dataset.close()


def test_six_open_roots_are_owned_without_global_eviction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = []
    for index in range(6):
        root = tmp_path / str(index)
        _write_dataset(root)
        datasets.append(ArrowDataset.open(root))
    monkeypatch.setattr(
        arrow_inputs,
        "_hash_retained_file",
        lambda _source: (_ for _ in ()).throw(AssertionError("retained root was reopened")),
    )

    for dataset in datasets:
        with dataset.use() as lease:
            assert lease.native is dataset.native
    for dataset in datasets:
        dataset.close()


def test_concurrent_use_leases_block_close(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    dataset = ArrowDataset.open(tmp_path)
    barrier = threading.Barrier(5)

    def use_dataset() -> str:
        with dataset.use() as lease:
            barrier.wait(timeout=10)
            with lease.open_file("signatures") as source:
                digest = hashlib.sha256(source.read()).hexdigest()
            batches = tuple(lease.native.batch_indices("signatures", ["s1"]))
            barrier.wait(timeout=10)
            return f"{digest}:{batches}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(use_dataset) for _ in range(4)]
        barrier.wait(timeout=10)
        with pytest.raises(RuntimeError, match="active use lease"):
            dataset.close()
        barrier.wait(timeout=10)
        assert len({future.result(timeout=10) for future in futures}) == 1
    dataset.close()


def test_close_is_deterministic_and_use_after_close_is_rejected(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    with ArrowDataset.open(tmp_path) as managed_dataset:
        assert not managed_dataset.closed
    assert managed_dataset.closed

    dataset = ArrowDataset.open(tmp_path)
    descriptors = [retained.handle.fileno() for retained in dataset._files.values()]  # noqa: SLF001

    dataset.close()
    dataset.close()

    assert dataset.closed
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    with pytest.raises(RuntimeError, match="closed"):
        dataset.use()
    with pytest.raises(RuntimeError, match="closed"):
        _ = dataset.native


@pytest.mark.parametrize("failure_stage", ["open", "checksum", "schema", "native"])
def test_failed_open_closes_all_acquired_files_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Partial acquisition and validation errors must not leak retained handles."""
    paths = _write_dataset(tmp_path)
    opened: list[BinaryIO] = []
    real_open = arrow_inputs._open_retained_path

    def track_open(path: Path) -> BinaryIO:
        if failure_stage == "open" and len(opened) == 1:
            raise OSError("injected retained-file open failure")
        source = real_open(path)
        opened.append(source)
        return source

    def fail_validation(*_args, **_kwargs) -> None:
        raise ValueError(f"injected {failure_stage} failure")

    failure_points = {
        "checksum": "_hash_retained_file",
        "schema": "_validate_retained_arrow_schema",
        "native": "_construct_native_arrow_dataset",
    }
    with monkeypatch.context() as patch:
        patch.setattr(arrow_inputs, "_open_retained_path", track_open)
        if failure_stage in failure_points:
            patch.setattr(arrow_inputs, failure_points[failure_stage], fail_validation)
        with pytest.raises(OSError if failure_stage == "open" else ValueError, match="injected"):
            ArrowDataset.open(tmp_path)

    assert len(opened) == (1 if failure_stage == "open" else len(paths))
    assert all(source.closed for source in opened)
    with ArrowDataset.open(tmp_path) as dataset:
        with dataset.use() as lease:
            with lease.open_file("signatures") as source:
                assert source.read() == Path(paths["signatures"]).read_bytes()


def test_reader_exception_releases_duplicate_and_lease_without_closing_dataset(tmp_path: Path) -> None:
    """A failed reader must leave the retained generation reusable and closable."""
    paths = _write_dataset(tmp_path)
    dataset = ArrowDataset.open(tmp_path)

    with pytest.raises(RuntimeError, match="injected reader failure"):
        with dataset.use() as failed_lease:
            with failed_lease.open_file("signatures") as failed_source:
                assert failed_source.read(16)
                raise RuntimeError("injected reader failure")

    assert failed_source.closed
    with pytest.raises(RuntimeError, match="lease is not active"):
        _ = failed_lease.native
    with dataset.use() as lease:
        with lease.open_file("signatures") as source:
            assert source.read() == Path(paths["signatures"]).read_bytes()
    dataset.close()
    assert dataset.closed


@pytest.mark.parametrize("requirement", ["require_specter", "require_name_counts_index"])
def test_rejected_use_profile_does_not_leak_active_lease(tmp_path: Path, requirement: str) -> None:
    """A model requesting unavailable features must not prevent dataset cleanup."""
    _write_dataset(tmp_path)
    dataset = ArrowDataset.open(tmp_path)

    with pytest.raises(MissingArrowArtifactError):
        with dataset.use(**{requirement: True}):
            pytest.fail("incomplete dataset accepted required model material")

    with dataset.use() as lease:
        assert lease.has("signatures")
    dataset.close()
    assert dataset.closed


def test_same_size_corruption_before_open_is_rejected(tmp_path: Path) -> None:
    paths = _write_dataset(tmp_path)
    signatures_path = Path(paths["signatures"])
    payload = bytearray(signatures_path.read_bytes())
    payload[len(payload) // 2] ^= 1
    signatures_path.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        ArrowDataset.open(tmp_path)


def test_open_validates_schema_before_native_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow as pa

    paths = _write_dataset(tmp_path)
    write_arrow_ipc_table(pa.table({"signature_id": ["s1"]}), paths["signatures"])
    write_arrow_artifact_manifest(build_arrow_artifact_manifest(paths, tmp_path), tmp_path)
    monkeypatch.setattr(
        arrow_inputs,
        "_construct_native_arrow_dataset",
        lambda *_args: (_ for _ in ()).throw(AssertionError("native constructed before schema validation")),
    )

    with pytest.raises(ValueError, match="missing required column 'paper_id'"):
        ArrowDataset.open(tmp_path)


def test_open_requires_the_fixed_base_shape_and_paired_specter(tmp_path: Path) -> None:
    _write_dataset(tmp_path, include_specter=True)

    def remove_papers(manifest):
        manifest["paths"].pop("papers")
        manifest["files"].pop("papers")

    _rewrite_manifest(tmp_path, remove_papers)
    with pytest.raises(MissingArrowArtifactError, match="papers"):
        ArrowDataset.open(tmp_path)

    _write_dataset(tmp_path / "other", include_specter=True)

    def remove_specter_index(manifest):
        manifest["paths"].pop("specter_batch_index")
        manifest["files"].pop("specter_batch_index")

    _rewrite_manifest(tmp_path / "other", remove_specter_index)
    with pytest.raises(ValueError, match="both specter and specter_batch_index"):
        ArrowDataset.open(tmp_path / "other")


def test_open_can_require_optional_material(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    with pytest.raises(MissingArrowArtifactError, match="specter"):
        ArrowDataset.open(tmp_path, require_specter=True)
    with pytest.raises(MissingArrowArtifactError, match="name_counts_index"):
        ArrowDataset.open(tmp_path, require_name_counts_index=True)


def test_name_counts_state_is_retained_by_the_dataset(tmp_path: Path) -> None:
    _write_dataset(tmp_path, include_name_counts=True, include_specter=True)

    dataset = ArrowDataset.open(tmp_path, require_name_counts_index=True, require_specter=True)

    assert dataset.name_counts_index is not None
    assert dataset.native_name_counts_index is not None
    assert dataset.native.name_counts_index is dataset.native_name_counts_index
    dataset.close()


def test_manifest_rejects_declared_file_path_and_sidecar_inventory(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    def add_legacy_path(manifest):
        manifest["files"]["signatures"]["path"] = "other.arrow"

    _rewrite_manifest(tmp_path, add_legacy_path)
    with pytest.raises(ValueError, match=r"files\.signatures field mismatch.*extra=\['path'\]"):
        ArrowDataset.open(tmp_path)

    other = tmp_path / "other"
    _write_dataset(other)
    sidecar = other / "cluster_seeds.arrow"
    sidecar.write_bytes(b"sidecar")

    def add_sidecar(manifest):
        manifest["paths"]["cluster_seeds"] = "cluster_seeds.arrow"
        manifest["files"]["cluster_seeds"] = {
            "byte_count": len(b"sidecar"),
            "sha256": hashlib.sha256(b"sidecar").hexdigest(),
        }

    _rewrite_manifest(other, add_sidecar)
    with pytest.raises(ValueError, match="unsupported immutable file keys"):
        ArrowDataset.open(other)


def test_manifest_rejects_wrong_kind_or_format_before_opening_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        arrow_inputs,
        "_open_retained_path",
        lambda *_args: (_ for _ in ()).throw(AssertionError("payload opened before format validation")),
    )
    cases = (
        ("wrong-kind", "kind", "other"),
        ("unsupported-format", "format_version", 2),
        ("boolean-format", "format_version", True),
        ("string-format", "format_version", "1"),
    )
    for case_id, field, value in cases:
        case_root = tmp_path / case_id
        _write_dataset(case_root)

        def mutate_contract(manifest, field=field, value=value):
            manifest[field] = value

        _rewrite_manifest(case_root, mutate_contract)
        try:
            ArrowDataset.open(case_root)
        except ValueError as error:
            assert field in str(error), f"{case_id}: {error}"
        else:
            raise AssertionError(f"{case_id}: wrong manifest identity was accepted")


def test_normalize_path_helper_remains_a_writer_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    other_dir = tmp_path / "other"
    source_dir.mkdir()
    other_dir.mkdir()
    source = source_dir / "signatures.arrow"
    source.write_bytes(b"source")
    monkeypatch.chdir(source_dir)
    normalized = normalize_arrow_paths({"signatures": "signatures.arrow"})
    monkeypatch.chdir(other_dir)
    assert Path(normalized["signatures"]).read_bytes() == b"source"
    with pytest.raises(ValueError, match="is None"):
        normalize_arrow_paths({"signatures": None})
    with pytest.raises(ValueError, match="is empty"):
        normalize_arrow_paths({"signatures": " "})
    with pytest.raises(ValueError, match="current directory"):
        normalize_arrow_paths({"signatures": "."})


def test_require_name_counts_index_artifact_reports_invalid_manifest(tmp_path: Path) -> None:
    index_dir = tmp_path / "name_counts_index"
    index_dir.mkdir()

    with pytest.raises(MissingArrowArtifactError, match="name_counts_index"):
        require_name_counts_index_artifact(index_dir, context="test", producer_hint="write it")

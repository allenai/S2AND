from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import s2and.arrow_inputs as arrow_inputs_module
from s2and.arrow_inputs import (
    MissingArrowArtifactError,
    normalize_arrow_paths,
    require_arrow_artifacts,
    require_filtered_arrow_batch_indexes,
    validate_arrow_prediction_artifacts,
    verified_arrow_artifact_generation,
)
from s2and.incremental_linking.feature_block import (
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_name_counts_index,
)
from tests.helpers import patch_tiny_name_counts_loader, write_test_arrow_artifact_manifest


def _touch_paths(tmp_path: Path, keys: tuple[str, ...], *, suffix: str = ".arrow") -> dict[str, str]:
    paths = {}
    for key in keys:
        path = tmp_path / f"{key}{suffix}"
        path.touch()
        paths[key] = str(path)
    return paths


def _write_artifact_generation_manifest(root: Path, paths: dict[str, str]) -> str:
    files: dict[str, dict[str, object]] = {}
    for key, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        payload = path.read_bytes()
        files[key] = {
            "path": path.name,
            "kind": "file",
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    generation_id = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_generation": {
                    "schema_version": "s2and_arrow_artifact_generation_v1",
                    "generation_id": generation_id,
                    "files": files,
                }
            }
        ),
        encoding="utf-8",
    )
    return generation_id


def _discard_cached_arrow_generations(manifest_path: Path) -> None:
    watches = []
    with arrow_inputs_module._ARROW_ARTIFACT_GENERATION_CACHE_LOCK:  # noqa: SLF001
        for cache_key in list(arrow_inputs_module._ARROW_ARTIFACT_GENERATION_CACHE):  # noqa: SLF001
            if cache_key[0] != str(manifest_path):
                continue
            _generation, watch = arrow_inputs_module._ARROW_ARTIFACT_GENERATION_CACHE.pop(cache_key)  # noqa: SLF001
            watches.append(watch)
    for watch in watches:
        watch.close()


@pytest.mark.parametrize("warm_with_specter", (False, True))
def test_generation_cache_binds_exact_immutable_key_paths_in_either_order(
    tmp_path: Path,
    warm_with_specter: bool,
) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    specter_path = tmp_path / "specter.arrow"
    mismatched_specter_path = tmp_path / "other-specter.arrow"
    signatures_path.write_bytes(b"signatures")
    specter_path.write_bytes(b"specter")
    mismatched_specter_path.write_bytes(b"specter")
    full_paths = {
        "signatures": str(signatures_path),
        "specter": str(specter_path),
    }
    generation_id = _write_artifact_generation_manifest(tmp_path, full_paths)
    manifest_path = tmp_path / "manifest.json"
    warm_paths = full_paths if warm_with_specter else {"signatures": str(signatures_path)}
    other_valid_paths = {"signatures": str(signatures_path)} if warm_with_specter else full_paths

    try:
        assert verified_arrow_artifact_generation(warm_paths) == generation_id
        assert verified_arrow_artifact_generation(other_valid_paths) == generation_id
        with pytest.raises(ValueError, match="path does not match supplied path"):
            verified_arrow_artifact_generation(
                {
                    "signatures": str(signatures_path),
                    "specter": str(mismatched_specter_path),
                }
            )
    finally:
        _discard_cached_arrow_generations(manifest_path)


def test_generation_cache_excludes_request_local_sidecars_from_base_binding(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper-authors.arrow"
    first_seed_path = tmp_path / "first-cluster-seeds.arrow"
    second_seed_path = tmp_path / "second-cluster-seeds.arrow"
    signatures_path.write_bytes(b"signatures")
    papers_path.write_bytes(b"papers")
    paper_authors_path.write_bytes(b"paper-authors")
    first_seed_path.write_bytes(b"first")
    second_seed_path.write_bytes(b"second")
    base_paths = {
        "signatures": str(signatures_path),
        "papers": str(papers_path),
        "paper_authors": str(paper_authors_path),
    }
    generation_id = _write_artifact_generation_manifest(tmp_path, base_paths)
    manifest_path = tmp_path / "manifest.json"

    try:
        assert (
            verified_arrow_artifact_generation({**base_paths, "cluster_seeds": str(first_seed_path)}) == generation_id
        )
        assert verified_arrow_artifact_generation(base_paths) == generation_id
        assert (
            verified_arrow_artifact_generation({**base_paths, "cluster_seeds": str(second_seed_path)}) == generation_id
        )
        with arrow_inputs_module._ARROW_ARTIFACT_GENERATION_CACHE_LOCK:  # noqa: SLF001
            matching_keys = [
                cache_key
                for cache_key in arrow_inputs_module._ARROW_ARTIFACT_GENERATION_CACHE  # noqa: SLF001
                if cache_key[0] == str(manifest_path)
            ]
        assert len(matching_keys) == 1
        second_seed_path.unlink()
        with pytest.raises(MissingArrowArtifactError, match="second-cluster-seeds"):
            validate_arrow_prediction_artifacts(
                {**base_paths, "cluster_seeds": str(second_seed_path)},
                require_specter=False,
                require_name_counts_index=False,
            )
    finally:
        _discard_cached_arrow_generations(manifest_path)


def test_generation_manifest_rejects_request_sidecars_in_immutable_inventory(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    cluster_seeds_path = tmp_path / "cluster-seeds.arrow"
    signatures_path.write_bytes(b"signatures")
    cluster_seeds_path.write_bytes(b"seeds")
    paths = {
        "signatures": str(signatures_path),
        "cluster_seeds": str(cluster_seeds_path),
    }
    _write_artifact_generation_manifest(tmp_path, paths)

    with pytest.raises(ValueError, match=r"request_sidecars=\['cluster_seeds'\]"):
        verified_arrow_artifact_generation(paths)


def test_concurrent_consumable_change_watch_cannot_return_stale_generation(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    signatures_path.write_bytes(b"original")
    paths = {"signatures": str(signatures_path)}
    generation_id = _write_artifact_generation_manifest(tmp_path, paths)
    manifest_path = tmp_path / "manifest.json"
    assert verified_arrow_artifact_generation(paths) == generation_id

    class ConsumableChangeWatch:
        def __init__(self) -> None:
            self.first_entered = threading.Event()
            self.second_entered = threading.Event()
            self.release_first = threading.Event()
            self._call_lock = threading.Lock()
            self._call_count = 0

        def changed(self) -> bool:
            with self._call_lock:
                self._call_count += 1
                call_count = self._call_count
            if call_count == 1:
                self.first_entered.set()
                if not self.release_first.wait(timeout=5):
                    raise RuntimeError("test did not release the first watcher read")
                return True
            self.second_entered.set()
            return False

        def close(self) -> None:
            return None

    watch = ConsumableChangeWatch()
    with arrow_inputs_module._ARROW_ARTIFACT_GENERATION_CACHE_LOCK:  # noqa: SLF001
        matching_keys = [
            cache_key
            for cache_key in arrow_inputs_module._ARROW_ARTIFACT_GENERATION_CACHE  # noqa: SLF001
            if cache_key[0] == str(manifest_path)
        ]
        assert len(matching_keys) == 1
        cache_key = matching_keys[0]
        cached_generation, original_watch = arrow_inputs_module._ARROW_ARTIFACT_GENERATION_CACHE[  # noqa: SLF001
            cache_key
        ]
        arrow_inputs_module._ARROW_ARTIFACT_GENERATION_CACHE[cache_key] = (  # noqa: SLF001
            cached_generation,
            watch,
        )
    original_watch.close()
    signatures_path.write_bytes(b"tampered")
    second_started = threading.Event()

    def second_lookup() -> str | None:
        second_started.set()
        return verified_arrow_artifact_generation(paths)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(verified_arrow_artifact_generation, paths)
            assert watch.first_entered.wait(timeout=5)
            second_future = executor.submit(second_lookup)
            assert second_started.wait(timeout=5)
            assert not watch.second_entered.wait(timeout=1)
            watch.release_first.set()
            for future in (first_future, second_future):
                with pytest.raises(ValueError, match="checksum mismatch"):
                    future.result(timeout=5)
    finally:
        watch.release_first.set()
        _discard_cached_arrow_generations(manifest_path)


def test_darwin_polling_generation_guard_tracks_exact_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    signatures_path.write_bytes(b"original")
    paths = {"signatures": str(signatures_path)}
    generation_id = _write_artifact_generation_manifest(tmp_path, paths)
    manifest_path = tmp_path / "manifest.json"
    change_tokens: dict[str, int] = {}

    monkeypatch.setattr(
        arrow_inputs_module,
        "_file_change_token",
        lambda path: change_tokens.get(str(path.resolve()), 0),
    )
    monkeypatch.setattr(
        arrow_inputs_module,
        "_guard_tokens_unchanged",
        lambda tokens: all(change_tokens.get(str(Path(path).resolve()), 0) == expected for path, expected in tokens),
    )
    monkeypatch.setattr(
        arrow_inputs_module,
        "_artifact_change_watch",
        lambda directories, *, files=(): arrow_inputs_module._PollingArtifactWatch(  # noqa: SLF001
            (*directories, *files)
        ),
    )

    try:
        assert verified_arrow_artifact_generation(paths) == generation_id
        original_stat = signatures_path.stat()
        time.sleep(0.02)
        signatures_path.write_bytes(b"tampered")
        os.utime(
            signatures_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        change_tokens[str(signatures_path.resolve())] = 1
        assert signatures_path.stat().st_size == original_stat.st_size
        assert signatures_path.stat().st_mtime_ns == original_stat.st_mtime_ns
        with pytest.raises(ValueError, match="checksum mismatch"):
            verified_arrow_artifact_generation(paths)
    finally:
        _discard_cached_arrow_generations(manifest_path)


def test_normalize_arrow_paths_resolves_relative_paths_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    other_dir = tmp_path / "other"
    source_dir.mkdir()
    other_dir.mkdir()
    source_path = source_dir / "signatures.arrow"
    source_path.write_bytes(b"source")
    monkeypatch.chdir(source_dir)
    normalized = normalize_arrow_paths({"signatures": "signatures.arrow"})

    monkeypatch.chdir(other_dir)
    resolved = Path(normalized["signatures"])
    assert resolved.is_absolute()
    assert resolved.read_bytes() == b"source"


def test_manifest_normalization_is_checked_without_name_counts(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors"))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"normalization_version": "legacy_compat", "paths": paths}),
        encoding="utf-8",
    )

    with pytest.raises(MissingArrowArtifactError, match="normalization_version mismatch"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
            expected_normalization_version="canonical_v2",
        )


def test_existing_non_object_manifest_is_never_treated_as_legacy(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors"))
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
            expected_normalization_version="canonical_v2",
        )


def test_canonical_manifest_requirement_rejects_legacy_inventory(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors"))
    (tmp_path / "manifest.json").write_text(
        json.dumps({"normalization_version": "canonical_v2", "paths": paths}),
        encoding="utf-8",
    )

    with pytest.raises(MissingArrowArtifactError, match="manifest.artifact_generation"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
            expected_normalization_version="canonical_v2",
            require_canonical_manifest=True,
        )


def test_strict_index_cache_does_not_skip_later_optional_specter_index(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    paths: dict[str, str] = {}
    for table_key, key_column in (
        ("signatures", "signature_id"),
        ("papers", "paper_id"),
        ("paper_authors", "paper_id"),
    ):
        table_path = tmp_path / f"{table_key}.arrow"
        index_path = tmp_path / f"{table_key}.index"
        write_arrow_ipc_table(pa.table({key_column: ["a"]}), table_path)
        write_arrow_batch_lookup_index(table_path, index_path, key_column=key_column)
        paths[table_key] = str(table_path)
        paths[f"{table_key}_batch_index"] = str(index_path)
    specter_path = tmp_path / "specter.arrow"
    specter_index_path = tmp_path / "specter.index"
    write_arrow_ipc_table(pa.table({"paper_id": ["a"]}), specter_path)
    specter_index_path.write_bytes(b"not-an-index")
    full_paths = {
        **paths,
        "specter": str(specter_path),
        "specter_batch_index": str(specter_index_path),
    }
    write_test_arrow_artifact_manifest(tmp_path, full_paths)

    validate_arrow_prediction_artifacts(
        paths,
        require_specter=False,
        require_name_counts_index=False,
        require_batch_indexes=True,
        strict_batch_index_validation=True,
        require_canonical_manifest=True,
    )
    with pytest.raises(ValueError, match="batch lookup index"):
        validate_arrow_prediction_artifacts(
            full_paths,
            require_specter=True,
            require_name_counts_index=False,
            require_batch_indexes=True,
            strict_batch_index_validation=True,
            require_canonical_manifest=True,
        )


def test_manifest_backed_strict_attachment_rejects_same_size_same_mtime_rewrite(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    paths: dict[str, str] = {}
    for table_key, key_column in (
        ("signatures", "signature_id"),
        ("papers", "paper_id"),
        ("paper_authors", "paper_id"),
    ):
        source_path = tmp_path / f"{table_key}.arrow"
        index_path = tmp_path / f"{table_key}.index"
        write_arrow_ipc_table(pa.table({key_column: ["a"]}), source_path)
        write_arrow_batch_lookup_index(source_path, index_path, key_column=key_column)
        paths[table_key] = str(source_path)
        paths[f"{table_key}_batch_index"] = str(index_path)
    _write_artifact_generation_manifest(tmp_path, paths)

    validate_arrow_prediction_artifacts(
        paths,
        require_specter=False,
        require_name_counts_index=False,
        require_batch_indexes=True,
        strict_batch_index_validation=True,
    )
    original_stat = Path(paths["signatures"]).stat()
    time.sleep(0.02)
    write_arrow_ipc_table(pa.table({"signature_id": ["b"]}), paths["signatures"])
    os.utime(
        paths["signatures"],
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    rewritten_stat = Path(paths["signatures"]).stat()
    assert rewritten_stat.st_size == original_stat.st_size
    assert rewritten_stat.st_mtime_ns == original_stat.st_mtime_ns

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
            require_batch_indexes=True,
            strict_batch_index_validation=True,
        )


def test_require_arrow_artifacts_reports_missing_keys_and_files(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    signatures_path.touch()
    missing_index_path = tmp_path / "signatures.signatures_batch_index.bin"

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        require_arrow_artifacts(
            {
                "signatures": signatures_path,
                "signatures_batch_index": missing_index_path,
            },
            required_keys=("signatures", "signatures_batch_index", "papers"),
            context="test context",
            producer_hint="test hint",
        )

    error = exc_info.value
    assert error.context == "test context"
    assert error.required_keys == ("signatures", "signatures_batch_index", "papers")
    assert error.missing_keys == ("papers",)
    assert error.missing_files == {"signatures_batch_index": str(missing_index_path)}
    assert "test hint" in str(error)


def test_validate_arrow_prediction_artifacts_requires_filtered_read_indexes(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors", "specter"))

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=True,
            require_name_counts_index=False,
            require_batch_indexes=True,
        )

    assert exc_info.value.missing_keys == (
        "paper_authors_batch_index",
        "papers_batch_index",
        "signatures_batch_index",
        "specter_batch_index",
    )


def test_validate_arrow_prediction_artifacts_rejects_missing_declared_seed_sidecar(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors"))
    seed_path = tmp_path / "missing_cluster_seeds.arrow"
    paths["cluster_seeds"] = str(seed_path)

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
        )

    assert exc_info.value.missing_files == {"cluster_seeds": str(seed_path)}


def test_validate_arrow_prediction_artifacts_rejects_wrong_path_kinds(tmp_path: Path) -> None:
    signatures_dir = tmp_path / "signatures.arrow"
    signatures_dir.mkdir()
    papers_path = tmp_path / "papers.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    name_counts_file = tmp_path / "name_counts_index"
    papers_path.touch()
    paper_authors_path.touch()
    name_counts_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        validate_arrow_prediction_artifacts(
            {
                "signatures": str(signatures_dir),
                "papers": str(papers_path),
                "paper_authors": str(paper_authors_path),
                "name_counts_index": str(name_counts_file),
            },
            require_specter=False,
            require_name_counts_index=True,
        )

    assert exc_info.value.missing_files == {
        "name_counts_index": f"{name_counts_file} (expected directory)",
        "signatures": f"{signatures_dir} (expected file)",
    }


def test_validate_arrow_prediction_artifacts_requires_manifest_backed_name_counts_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors"))
    empty_index_dir = tmp_path / "empty_name_counts_index"
    empty_index_dir.mkdir()

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        validate_arrow_prediction_artifacts(
            {**paths, "name_counts_index": str(empty_index_dir)},
            require_specter=False,
            require_name_counts_index=True,
        )

    assert exc_info.value.missing_files["name_counts_index"].endswith("manifest.json (missing manifest.json)")

    patch_tiny_name_counts_loader(monkeypatch)
    valid_index_dir, _metrics = write_name_counts_index(tmp_path / "valid_index")
    assert (
        validate_arrow_prediction_artifacts(
            {**paths, "name_counts_index": valid_index_dir},
            require_specter=False,
            require_name_counts_index=True,
        )["name_counts_index"]
        == valid_index_dir
    )

    manifest_path = Path(valid_index_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "unexpected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MissingArrowArtifactError) as bad_schema_exc:
        validate_arrow_prediction_artifacts(
            {**paths, "name_counts_index": valid_index_dir},
            require_specter=False,
            require_name_counts_index=True,
        )
    assert "schema_version" in bad_schema_exc.value.missing_files["name_counts_index"]


def test_name_counts_index_rejects_same_size_same_mtime_binary_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors"))
    patch_tiny_name_counts_loader(monkeypatch)
    index_dir, _metrics = write_name_counts_index(tmp_path / "index")
    prediction_paths = {**paths, "name_counts_index": index_dir}
    validate_arrow_prediction_artifacts(
        prediction_paths,
        require_specter=False,
        require_name_counts_index=True,
    )

    manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
    binary_path = Path(index_dir) / manifest["files"]["first"]["path"]
    original_stat = binary_path.stat()
    payload = bytearray(binary_path.read_bytes())
    payload[-1] ^= 1
    time.sleep(0.02)
    binary_path.write_bytes(payload)
    os.utime(binary_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(MissingArrowArtifactError, match="SHA-256 mismatch"):
        validate_arrow_prediction_artifacts(
            prediction_paths,
            require_specter=False,
            require_name_counts_index=True,
        )


def test_validate_arrow_prediction_artifacts_ignores_unused_specter_path(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors", "specter"))
    paths.update(
        _touch_paths(
            tmp_path, ("signatures_batch_index", "papers_batch_index", "paper_authors_batch_index"), suffix=".bin"
        )
    )

    normalized = validate_arrow_prediction_artifacts(
        paths,
        require_specter=False,
        require_name_counts_index=False,
        require_batch_indexes=True,
    )

    assert "specter" not in normalized
    assert "specter_batch_index" not in normalized


def test_validate_arrow_prediction_artifacts_accepts_selected_specter2_alias(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors", "specter2"))
    paths.update(
        _touch_paths(
            tmp_path,
            ("signatures_batch_index", "papers_batch_index", "paper_authors_batch_index", "specter2_batch_index"),
            suffix=".bin",
        )
    )

    normalized = validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=False,
        require_batch_indexes=True,
    )

    assert normalized["specter"] == paths["specter2"]
    assert normalized["specter_batch_index"] == paths["specter2_batch_index"]
    assert "specter2" not in normalized
    assert "specter2_batch_index" not in normalized


def test_validate_arrow_prediction_artifacts_clears_invalid_legacy_specter_after_alias(tmp_path: Path) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors", "specter2"))
    paths.update(
        _touch_paths(
            tmp_path,
            ("signatures_batch_index", "papers_batch_index", "paper_authors_batch_index", "specter2_batch_index"),
            suffix=".bin",
        )
    )
    paths["specter"] = None  # type: ignore[assignment]
    paths["specter_batch_index"] = None  # type: ignore[assignment]

    normalized = validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=False,
        require_batch_indexes=True,
    )

    assert normalized["specter"] == paths["specter2"]
    assert normalized["specter_batch_index"] == paths["specter2_batch_index"]


def test_validate_arrow_prediction_artifacts_clears_invalid_specter2_after_canonical(
    tmp_path: Path,
) -> None:
    paths = _touch_paths(tmp_path, ("signatures", "papers", "paper_authors", "specter"))
    paths.update(
        _touch_paths(
            tmp_path,
            ("signatures_batch_index", "papers_batch_index", "paper_authors_batch_index", "specter_batch_index"),
            suffix=".bin",
        )
    )
    paths["specter2"] = None  # type: ignore[assignment]
    paths["specter2_batch_index"] = None  # type: ignore[assignment]

    normalized = validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=False,
        require_batch_indexes=True,
    )

    assert normalized["specter"] == paths["specter"]
    assert normalized["specter_batch_index"] == paths["specter_batch_index"]


def test_require_filtered_arrow_batch_indexes_ignores_specter_index_without_specter(tmp_path: Path) -> None:
    paths = {}
    for key in ("signatures", "papers", "paper_authors"):
        path = tmp_path / f"{key}.arrow"
        path.touch()
        paths[key] = str(path)
        index_path = tmp_path / f"{key}.{key}_batch_index.bin"
        index_path.touch()
        paths[f"{key}_batch_index"] = str(index_path)

    require_filtered_arrow_batch_indexes(paths)


def test_normalize_arrow_paths_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="is None"):
        normalize_arrow_paths({"signatures": None})
    with pytest.raises(ValueError, match="is empty"):
        normalize_arrow_paths({"signatures": " "})
    with pytest.raises(ValueError, match="current directory"):
        normalize_arrow_paths({"signatures": "."})

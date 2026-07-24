from __future__ import annotations

import hashlib
import json
import operator
from pathlib import Path, PureWindowsPath

import pytest

import s2and.arrow_inputs as arrow_inputs_module
import s2and.incremental_linking.feature_block_arrow as feature_block_arrow_module
from s2and.arrow_inputs import (
    MissingArrowArtifactError,
    ValidatedArrowInputs,
    build_arrow_artifact_manifest,
    normalize_arrow_paths,
    require_arrow_artifacts,
    require_filtered_arrow_batch_indexes,
    validate_arrow_prediction_artifacts,
    validate_arrow_publication_artifacts,
    write_arrow_artifact_manifest,
)
from s2and.incremental_linking.feature_block import (
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_name_counts_index,
)
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple, write_test_arrow_artifact_manifest


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
                "normalization_version": "canonical_v2",
                "artifact_generation": {
                    "schema_version": "s2and_arrow_artifact_generation_v1",
                    "generation_id": generation_id,
                    "files": files,
                },
            }
        ),
        encoding="utf-8",
    )
    return generation_id


def _write_valid_prediction_bundle(
    tmp_path: Path,
    *,
    specter_key: str | None = None,
    include_name_counts: bool = False,
) -> dict[str, str]:
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
    if specter_key is not None:
        table_path = tmp_path / f"{specter_key}.arrow"
        index_path = tmp_path / f"{specter_key}.index"
        write_arrow_ipc_table(pa.table({"paper_id": ["a"]}), table_path)
        write_arrow_batch_lookup_index(table_path, index_path, key_column="paper_id")
        paths[specter_key] = str(table_path)
        paths[f"{specter_key}_batch_index"] = str(index_path)
    if include_name_counts:
        index_path, _metrics = write_name_counts_index(
            tmp_path,
            tiny_name_counts_tuple(),
            tiny_name_counts_provenance(),
        )
        paths["name_counts_index"] = index_path
    write_test_arrow_artifact_manifest(tmp_path, paths)
    return paths


@pytest.mark.parametrize("warm_with_specter", (False, True))
def test_generation_verification_binds_exact_immutable_key_paths_in_either_order(
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
    warm_paths = full_paths if warm_with_specter else {"signatures": str(signatures_path)}
    other_valid_paths = {"signatures": str(signatures_path)} if warm_with_specter else full_paths

    assert arrow_inputs_module._verified_arrow_artifact_manifest(warm_paths).generation_id == generation_id
    assert arrow_inputs_module._verified_arrow_artifact_manifest(other_valid_paths).generation_id == generation_id
    with pytest.raises(ValueError, match="path does not match supplied path"):
        arrow_inputs_module._verified_arrow_artifact_manifest(
            {
                "signatures": str(signatures_path),
                "specter": str(mismatched_specter_path),
            }
        )


def test_with_request_sidecars_validates_only_sidecars_and_preserves_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_paths = _write_valid_prediction_bundle(tmp_path)
    validated = validate_arrow_prediction_artifacts(
        base_paths,
        require_specter=False,
        require_name_counts_index=False,
    )
    cluster_seeds_path = tmp_path / "cluster-seeds.arrow"
    cluster_seeds_path.write_bytes(b"seeds")
    monkeypatch.setattr(
        arrow_inputs_module,
        "_sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("base generation was rehashed")),
    )

    extended = validated.with_request_sidecars(
        {**base_paths, "cluster_seeds": cluster_seeds_path},
        required_keys=("cluster_seeds",),
        context="test sidecars",
        producer_hint="write the sidecar",
    )

    assert extended.generation_id == validated.generation_id
    assert extended.normalization_version == validated.normalization_version
    assert extended["cluster_seeds"] == str(cluster_seeds_path.resolve())
    assert "cluster_seeds" not in validated

    cluster_seeds_path.unlink()
    with pytest.raises(MissingArrowArtifactError, match="cluster-seeds.arrow"):
        validated.with_request_sidecars(
            {"cluster_seeds": cluster_seeds_path},
            required_keys=("cluster_seeds",),
            context="test sidecars",
            producer_hint="write the sidecar",
        )


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
        arrow_inputs_module._verified_arrow_artifact_manifest(paths)


def test_artifact_generation_writer_rejects_path_outside_manifest_directory(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    outside_path = tmp_path / "outside.arrow"
    outside_path.write_bytes(b"outside")

    with pytest.raises(ValueError, match="must remain within manifest directory"):
        build_arrow_artifact_manifest(
            {"signatures": outside_path},
            bundle_dir,
        )


def test_canonical_manifest_accepts_linker_replay_shared_name_counts_index(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    bundle_dir = release_root / "replay-bundle" / "datasets" / "tiny"
    bundle_dir.mkdir(parents=True)
    signatures_path = bundle_dir / "signatures.arrow"
    signatures_path.write_bytes(b"signatures")
    name_counts_path = release_root / "name_counts_index"
    name_counts_path.mkdir()
    (name_counts_path / "manifest.json").write_text("{}", encoding="utf-8")

    paths = {
        "signatures": signatures_path,
        "name_counts_index": name_counts_path,
    }
    manifest = build_arrow_artifact_manifest(paths, bundle_dir)
    write_arrow_artifact_manifest(manifest, bundle_dir)

    assert manifest["paths"]["name_counts_index"] == "../../../name_counts_index"
    assert manifest["artifact_generation"]["files"]["name_counts_index"]["path"] == "../../../name_counts_index"
    verified = arrow_inputs_module._verified_arrow_artifact_manifest(
        {**paths, "manifest": bundle_dir / "manifest.json"}
    )
    assert verified is not None
    assert verified.generation_id == manifest["artifact_generation"]["generation_id"]


def test_manifest_path_serialization_normalizes_windows_separators() -> None:
    relative_path = PureWindowsPath("..") / ".." / ".." / "name_counts_index"

    assert arrow_inputs_module._portable_manifest_path(relative_path) == "../../../name_counts_index"


def test_linker_replay_shared_name_counts_exception_rejects_other_escaped_directories(tmp_path: Path) -> None:
    release_root = tmp_path / "release"
    bundle_dir = release_root / "replay-bundle" / "datasets" / "tiny"
    bundle_dir.mkdir(parents=True)
    outside_index = release_root / "other-index"
    outside_index.mkdir()
    (outside_index / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must remain within manifest directory"):
        build_arrow_artifact_manifest({"name_counts_index": outside_index}, bundle_dir)


def test_canonical_manifest_builder_owns_runtime_fields_and_publication(tmp_path: Path) -> None:
    signatures_path = tmp_path / "signatures.arrow"
    signatures_path.write_bytes(b"signatures")

    manifest = build_arrow_artifact_manifest(
        {"signatures": signatures_path},
        tmp_path,
        metadata={"dataset": "tiny"},
    )
    manifest_path = write_arrow_artifact_manifest(manifest, tmp_path)

    assert manifest["normalization_version"] == "canonical_v2"
    assert manifest["paths"] == {"signatures": "signatures.arrow"}
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    with pytest.raises(ValueError, match="cannot override canonical fields"):
        build_arrow_artifact_manifest(
            {"signatures": signatures_path},
            tmp_path,
            metadata={"paths": {}},
        )


@pytest.mark.parametrize("declared_path_kind", ("absolute", "parent_escape"))
def test_generation_manifest_rejects_paths_outside_manifest_authority(
    tmp_path: Path,
    declared_path_kind: str,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    outside_path = tmp_path / "outside.arrow"
    outside_path.write_bytes(b"outside")
    declared_path = str(outside_path.resolve()) if declared_path_kind == "absolute" else "../outside.arrow"
    files = {
        "signatures": {
            "path": declared_path,
            "kind": "file",
            "byte_count": outside_path.stat().st_size,
            "sha256": hashlib.sha256(outside_path.read_bytes()).hexdigest(),
        }
    }
    generation_id = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "normalization_version": "canonical_v2",
                "artifact_generation": {
                    "schema_version": "s2and_arrow_artifact_generation_v1",
                    "generation_id": generation_id,
                    "files": files,
                },
            }
        ),
        encoding="utf-8",
    )

    expected_message = "must be manifest-relative" if declared_path_kind == "absolute" else "escapes"
    with pytest.raises(ValueError, match=expected_message):
        arrow_inputs_module._verified_arrow_artifact_manifest(
            {
                "manifest": str(manifest_path),
                "signatures": str(outside_path),
            }
        )


def test_retained_validated_profile_reuses_in_memory_contract_without_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_valid_prediction_bundle(tmp_path, include_name_counts=True)
    validated = validate_arrow_prediction_artifacts(
        paths,
        require_specter=False,
        require_name_counts_index=True,
        expected_normalization_version="canonical_v2",
    )
    assert validated.name_counts_manifest is not None
    original_signature_path = validated["signatures"]
    paths["signatures"] = "other.arrow"
    assert validated["signatures"] == original_signature_path
    with pytest.raises(TypeError):
        operator.setitem(validated, "signatures", "other.arrow")
    with pytest.raises(TypeError):
        operator.setitem(validated.paths, "signatures", "other.arrow")
    with pytest.raises(AttributeError):
        validated.generation_id = "other"  # type: ignore[misc]
    without_name_counts = validated.without("name_counts_index")
    assert "name_counts_index" not in without_name_counts
    assert without_name_counts.name_counts_manifest is None
    assert without_name_counts._name_counts_index is None  # noqa: SLF001

    def fail_io(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("retained validation performed filesystem I/O")

    monkeypatch.setattr(arrow_inputs_module.ValidatedNameCountsManifest, "load", fail_io)
    for helper_name in (
        "_missing_or_wrong_kind_artifacts",
        "_name_counts_index_error",
        "_normalize_arrow_path_values",
        "_sha256_file",
        "_validate_batch_indexes",
        "_verified_arrow_artifact_manifest",
    ):
        monkeypatch.setattr(arrow_inputs_module, helper_name, fail_io)

    assert (
        validate_arrow_prediction_artifacts(
            validated,
            require_specter=False,
            require_name_counts_index=True,
            expected_normalization_version="canonical_v2",
        )
        is validated
    )
    with pytest.raises(MissingArrowArtifactError, match="papers_batch_index"):
        validate_arrow_prediction_artifacts(
            validated.without("papers_batch_index"),
            require_specter=False,
            require_name_counts_index=True,
            expected_normalization_version="canonical_v2",
        )


def test_retained_validated_profile_projects_unused_specter_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_valid_prediction_bundle(tmp_path, specter_key="specter")
    validated = validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=False,
    )
    monkeypatch.setattr(
        arrow_inputs_module,
        "_normalize_arrow_path_values",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("projection touched the filesystem")),
    )

    projected = validate_arrow_prediction_artifacts(
        validated,
        require_specter=False,
        require_name_counts_index=False,
    )

    assert not {"specter", "specter_batch_index"}.intersection(projected)
    assert projected.generation_id == validated.generation_id
    assert projected.normalization_version == validated.normalization_version
    assert "specter" in validated


def test_validated_arrow_inputs_is_a_normal_immutable_value_object() -> None:
    source_paths = {"signatures": "signatures.arrow"}
    inputs = ValidatedArrowInputs(
        paths=source_paths,
        generation_id="generation",
        normalization_version="canonical_v2",
    )

    source_paths["signatures"] = "changed.arrow"
    assert inputs["signatures"] == "signatures.arrow"
    with pytest.raises(TypeError):
        inputs.paths["signatures"] = "changed.arrow"  # type: ignore[index]


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


def test_legacy_manifest_is_rejected_without_name_counts(tmp_path: Path) -> None:
    paths = _write_valid_prediction_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["normalization_version"] = "legacy_compat"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="normalization_version is invalid"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
            expected_normalization_version="canonical_v2",
        )


def test_legacy_name_counts_are_rejected_without_model_expectation(tmp_path: Path) -> None:
    paths = _write_valid_prediction_bundle(tmp_path, include_name_counts=True)
    name_counts_manifest_path = Path(paths["name_counts_index"]) / "manifest.json"
    name_counts_manifest = json.loads(name_counts_manifest_path.read_text(encoding="utf-8"))
    name_counts_manifest["normalization_version"] = "legacy_compat"
    name_counts_manifest["source_provenance"]["normalization_version"] = "legacy_compat"
    name_counts_manifest_path.write_text(json.dumps(name_counts_manifest), encoding="utf-8")
    write_test_arrow_artifact_manifest(tmp_path, paths)

    with pytest.raises(MissingArrowArtifactError, match="unsupported normalization_version"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=True,
            expected_normalization_version=None,
        )


def test_existing_non_object_manifest_is_never_treated_as_legacy(tmp_path: Path) -> None:
    paths = _write_valid_prediction_bundle(tmp_path)
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
            expected_normalization_version="canonical_v2",
        )


def test_canonical_manifest_requirement_rejects_legacy_inventory(tmp_path: Path) -> None:
    paths = _write_valid_prediction_bundle(tmp_path)
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
        )


def test_repeated_validation_checks_newly_required_specter_index(tmp_path: Path) -> None:
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
    )
    with pytest.raises(ValueError, match="batch lookup index"):
        validate_arrow_prediction_artifacts(
            full_paths,
            require_specter=True,
            require_name_counts_index=False,
        )


def test_repeated_raw_validation_reuses_immutable_generation(tmp_path: Path) -> None:
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

    validated = validate_arrow_prediction_artifacts(
        paths,
        require_specter=False,
        require_name_counts_index=False,
    )
    write_arrow_ipc_table(pa.table({"signature_id": ["b"]}), paths["signatures"])
    repeated = validate_arrow_prediction_artifacts(
        paths,
        require_specter=False,
        require_name_counts_index=False,
    )
    assert repeated.generation_id == validated.generation_id
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_arrow_publication_artifacts(
            validated,
            require_specter=False,
            require_name_counts_index=False,
        )
    assert validated.generation_id


def test_complete_bundle_streams_each_generation_file_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_valid_prediction_bundle(tmp_path, specter_key="specter")
    paired_keys = {
        "signatures",
        "signatures_batch_index",
        "papers",
        "papers_batch_index",
        "paper_authors",
        "paper_authors_batch_index",
        "specter",
        "specter_batch_index",
    }
    paired_paths = {Path(paths[key]).resolve() for key in paired_keys}
    index_paths = {
        Path(paths[key]).resolve()
        for key in (
            "signatures_batch_index",
            "papers_batch_index",
            "paper_authors_batch_index",
            "specter_batch_index",
        )
    }
    table_reads: list[Path] = []
    index_reads: list[Path] = []
    original_manifest_sha256 = arrow_inputs_module._sha256_file  # noqa: SLF001
    original_table_sha256 = feature_block_arrow_module._source_file_sha256_once  # noqa: SLF001
    original_path_open = Path.open

    def reject_duplicate_manifest_hash(path: Path) -> str:
        assert path.resolve() not in paired_paths, f"paired artifact was hashed separately: {path}"
        return original_manifest_sha256(path)

    def record_table_read(path: Path) -> str:
        table_reads.append(path.resolve())
        return original_table_sha256(path)

    def record_index_read(path: Path, *args: object, **kwargs: object):
        if path.resolve() in index_paths:
            index_reads.append(path.resolve())
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(arrow_inputs_module, "_sha256_file", reject_duplicate_manifest_hash)
    monkeypatch.setattr(feature_block_arrow_module, "_source_file_sha256_once", record_table_read)
    monkeypatch.setattr(Path, "open", record_index_read)

    validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=False,
    )
    validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=False,
    )

    assert table_reads == [Path(paths[key]).resolve() for key in ("signatures", "papers", "paper_authors", "specter")]
    assert index_reads == [
        Path(paths[key]).resolve()
        for key in (
            "signatures_batch_index",
            "papers_batch_index",
            "paper_authors_batch_index",
            "specter_batch_index",
        )
    ]


def test_raw_validation_rejects_generation_bound_index_checksum_mutation(tmp_path: Path) -> None:
    paths = _write_valid_prediction_bundle(tmp_path)
    index_path = Path(paths["signatures_batch_index"])
    payload = bytearray(index_path.read_bytes())
    header_size = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size  # noqa: SLF001
    record_struct = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT  # noqa: SLF001
    key_hash, batch_index, reserved = record_struct.unpack_from(payload, header_size)
    record_struct.pack_into(payload, header_size, key_hash, batch_index, reserved + 1)
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match="index checksum mismatch"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
        )


def test_publication_validation_rejects_stale_index_in_regenerated_generation(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    paths = _write_valid_prediction_bundle(tmp_path)
    write_arrow_ipc_table(pa.table({"signature_id": ["b"]}), paths["signatures"])
    write_test_arrow_artifact_manifest(tmp_path, paths)

    with pytest.raises(ValueError, match="is stale"):
        validate_arrow_publication_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
        )


def test_publication_validation_rejects_checksummed_semantically_invalid_batch_index(tmp_path: Path) -> None:
    paths = _write_valid_prediction_bundle(tmp_path)
    index_path = Path(paths["signatures_batch_index"])
    payload = bytearray(index_path.read_bytes())
    header_size = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size  # noqa: SLF001
    record_struct = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT  # noqa: SLF001
    key_hash, _batch_index, reserved = record_struct.unpack_from(payload, header_size)
    record_struct.pack_into(payload, header_size, key_hash, 1, reserved)
    index_path.write_bytes(payload)
    write_test_arrow_artifact_manifest(tmp_path, paths)

    with pytest.raises(ValueError, match="batch index 1 is out of bounds"):
        validate_arrow_publication_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
        )


def test_prediction_validation_rejects_legacy_name_counts_arrow_path(tmp_path: Path) -> None:
    for legacy_key in ("name_counts", "name_counts_index_dir"):
        paths = _write_valid_prediction_bundle(tmp_path)
        legacy_path = tmp_path / legacy_key
        legacy_path.touch()
        paths[legacy_key] = str(legacy_path)

        with pytest.raises(MissingArrowArtifactError, match="use name_counts_index"):
            validate_arrow_prediction_artifacts(
                paths,
                require_specter=False,
                require_name_counts_index=False,
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
    del monkeypatch
    paths = _write_valid_prediction_bundle(tmp_path)
    empty_index_dir = tmp_path / "empty_name_counts_index"
    empty_index_dir.mkdir()

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        validate_arrow_prediction_artifacts(
            {**paths, "name_counts_index": str(empty_index_dir)},
            require_specter=False,
            require_name_counts_index=True,
        )

    assert exc_info.value.missing_files["name_counts_index"].endswith("manifest.json (missing manifest.json)")

    valid_index_dir, _metrics = write_name_counts_index(
        tmp_path / "valid_index", tiny_name_counts_tuple(), tiny_name_counts_provenance()
    )
    valid_paths = {**paths, "name_counts_index": valid_index_dir}
    write_test_arrow_artifact_manifest(tmp_path, valid_paths)
    assert (
        validate_arrow_prediction_artifacts(
            valid_paths,
            require_specter=False,
            require_name_counts_index=True,
        )["name_counts_index"]
        == valid_index_dir
    )

    bad_index_dir, _metrics = write_name_counts_index(
        tmp_path / "bad_index", tiny_name_counts_tuple(), tiny_name_counts_provenance()
    )
    manifest_path = Path(bad_index_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "unexpected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bad_paths = {**paths, "name_counts_index": bad_index_dir}
    write_test_arrow_artifact_manifest(tmp_path, bad_paths)
    with pytest.raises(MissingArrowArtifactError) as bad_schema_exc:
        validate_arrow_prediction_artifacts(
            bad_paths,
            require_specter=False,
            require_name_counts_index=True,
        )
    assert "schema_version" in bad_schema_exc.value.missing_files["name_counts_index"]


def test_publication_validation_detects_name_counts_file_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction_paths = _write_valid_prediction_bundle(tmp_path, include_name_counts=True)
    index_dir = Path(prediction_paths["name_counts_index"])
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    binary_path = index_dir / manifest["files"]["first"]["path"]
    payload = bytearray(binary_path.read_bytes())
    payload[-1] ^= 1
    binary_path.write_bytes(payload)

    def reject_redundant_python_material_scan(*_args: object, **_kwargs: object):
        raise AssertionError("publication validation must use the native material-validation pass")

    monkeypatch.setattr(
        arrow_inputs_module.ValidatedNameCountsManifest,
        "load",
        classmethod(reject_redundant_python_material_scan),
    )
    with pytest.raises(MissingArrowArtifactError, match="SHA-256 mismatch"):
        validate_arrow_publication_artifacts(
            prediction_paths,
            require_specter=False,
            require_name_counts_index=True,
        )


def test_validate_arrow_prediction_artifacts_rejects_specter2_aliases(tmp_path: Path) -> None:
    paths = _write_valid_prediction_bundle(tmp_path, specter_key="specter2")

    with pytest.raises(MissingArrowArtifactError, match="unsupported embedding path key"):
        validate_arrow_prediction_artifacts(
            paths,
            require_specter=True,
            require_name_counts_index=False,
        )


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

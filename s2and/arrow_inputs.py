"""Canonical validation helpers for Arrow-backed runtime inputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from s2and.consts import NORMALIZATION_VERSION


@dataclass(frozen=True)
class ArrowBatchIndexContract:
    """One Arrow table and its raw-planner batch lookup index contract."""

    table_key: str
    key_column: str
    index_key: str
    max_record_batch_rows: int


@dataclass(frozen=True)
class _VerifiedArrowArtifactGeneration:
    """Manifest facts retained after one full immutable-generation check."""

    generation_id: str
    normalization_version: str | None


class _VerifiedArrowInputsCapability:
    """Unexported authority to create an already-verified Arrow path set."""


_VERIFIED_ARROW_INPUTS_CAPABILITY = _VerifiedArrowInputsCapability()


@dataclass(frozen=True, init=False)
class ValidatedArrowInputs(Mapping[str, str]):
    """Immutable, integrity-checked Arrow paths for one artifact generation."""

    paths: Mapping[str, str]
    generation_id: str
    normalization_version: str

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "ValidatedArrowInputs cannot be constructed directly; use an Arrow artifact validation function"
        )

    def __init_subclass__(cls, **_kwargs: Any) -> None:
        raise TypeError("ValidatedArrowInputs cannot be subclassed")

    @classmethod
    def _from_verified(
        cls,
        *,
        paths: Mapping[str, str],
        generation_id: str,
        normalization_version: str,
        capability: object,
    ) -> ValidatedArrowInputs:
        """Create an instance after a trusted internal validation or projection."""

        if capability is not _VERIFIED_ARROW_INPUTS_CAPABILITY:
            raise TypeError("ValidatedArrowInputs creation requires the internal verified capability")
        instance = object.__new__(cls)
        object.__setattr__(instance, "paths", MappingProxyType(dict(paths)))
        object.__setattr__(instance, "generation_id", str(generation_id))
        object.__setattr__(instance, "normalization_version", str(normalization_version))
        return instance

    def __getitem__(self, key: str) -> str:
        return self.paths[key]

    def __iter__(self):
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def without(self, *keys: str) -> ValidatedArrowInputs:
        """Return this verified generation without request-local path entries."""

        removed = set(keys)
        return self._from_verified(
            paths={key: value for key, value in self.paths.items() if key not in removed},
            generation_id=self.generation_id,
            normalization_version=self.normalization_version,
            capability=_VERIFIED_ARROW_INPUTS_CAPABILITY,
        )

    def with_request_sidecars(
        self,
        sidecars: Mapping[str, Any],
        *,
        required_keys: Sequence[str] = (),
        context: str,
        producer_hint: str,
    ) -> ValidatedArrowInputs:
        """Validate request-local sidecars while retaining the verified base generation."""

        requested = {str(key): value for key, value in sidecars.items() if key in _ARROW_REQUEST_SIDECAR_KEYS}
        required = tuple(str(key) for key in required_keys)
        unsupported_required = sorted(set(required) - _ARROW_REQUEST_SIDECAR_KEYS)
        if unsupported_required:
            raise ValueError(f"request sidecar keys are unsupported: {unsupported_required}")
        normalized, invalid_paths = _normalize_arrow_path_values(requested)
        invalid_paths.update(_missing_or_wrong_kind_artifacts(normalized, tuple(normalized)))
        missing_keys = tuple(key for key in required if key not in normalized)
        if missing_keys or invalid_paths:
            raise MissingArrowArtifactError(
                context=context,
                required_keys=required,
                missing_keys=missing_keys,
                missing_files=invalid_paths,
                producer_hint=producer_hint,
            )
        paths = dict(self.paths)
        paths.update(normalized)
        return self._from_verified(
            paths=paths,
            generation_id=self.generation_id,
            normalization_version=self.normalization_version,
            capability=_VERIFIED_ARROW_INPUTS_CAPABILITY,
        )


RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS = (
    ArrowBatchIndexContract("signatures", "signature_id", "signatures_batch_index", 16_384),
    ArrowBatchIndexContract("papers", "paper_id", "papers_batch_index", 16_384),
    ArrowBatchIndexContract("paper_authors", "paper_id", "paper_authors_batch_index", 16_384),
    ArrowBatchIndexContract("specter", "paper_id", "specter_batch_index", 2_048),
)
RAW_PLANNER_ARROW_KEY_COLUMNS = {
    contract.table_key: contract.key_column for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS
}
RAW_PLANNER_ARROW_BATCH_INDEX_KEYS = {
    contract.table_key: contract.index_key for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS
}
RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS = {
    contract.table_key: contract.max_record_batch_rows for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS
}
FILTERED_READ_ARROW_TABLE_KEYS = ("signatures", "papers", "paper_authors")
DECLARED_ARROW_SIDECAR_KEYS = (
    "cluster_seeds",
    "cluster_seed_disallows",
    "altered_cluster_signatures",
    "name_counts_index",
)
UNSUPPORTED_ARROW_NAME_ALIAS_KEYS = frozenset({"name_pairs", "name_tuples"})
DIRECTORY_ARTIFACT_KEYS = frozenset({"name_counts_index"})
NAME_COUNTS_INDEX_MANIFEST_FILES = ("first", "last", "first_last", "last_first_initial")
NAME_COUNTS_INDEX_SCHEMA_VERSION = "name_counts_index_v1"
ARROW_ARTIFACT_GENERATION_SCHEMA_VERSION = "s2and_arrow_artifact_generation_v1"
_ARROW_REQUEST_SIDECAR_KEYS = frozenset(
    {
        "query_signatures",
        "cluster_seeds",
        "cluster_seed_disallows",
        "altered_cluster_signatures",
    }
)
_ARROW_IMMUTABLE_ARTIFACT_FILE_KEYS = frozenset(
    {
        "signatures",
        "papers",
        "paper_authors",
        "specter",
        "name_counts_index",
    }
)
_ARROW_IMMUTABLE_BATCH_INDEX_KEYS = frozenset(RAW_PLANNER_ARROW_BATCH_INDEX_KEYS.values())
_SPECTER_PATH_KEYS = frozenset(
    {
        "specter",
        "specter_batch_index",
        "specter2",
        "specter2_batch_index",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_relative_path(path_value: Any, manifest_dir: Path, *, artifact_key: str) -> str:
    path = Path(os.fspath(path_value)).resolve()
    root = manifest_dir.resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        shared_name_counts_path = root.parent / "name_counts_index"
        if artifact_key == "name_counts_index" and path == shared_name_counts_path:
            return os.fspath(Path("..") / "name_counts_index")
        raise ValueError(f"Arrow artifact path must remain within manifest directory: path={path} root={root}") from exc
    return os.fspath(relative_path)


def _build_arrow_artifact_generation(paths: Mapping[str, Any], manifest_dir: str | Path) -> dict[str, Any]:
    """Build the canonical content inventory used by Arrow bundle writers."""

    root = Path(manifest_dir)
    canonical_paths = {str(key): value for key, value in paths.items()}
    if "specter" not in canonical_paths and "specter2" in canonical_paths:
        canonical_paths["specter"] = canonical_paths["specter2"]
    if "specter_batch_index" not in canonical_paths and "specter2_batch_index" in canonical_paths:
        canonical_paths["specter_batch_index"] = canonical_paths["specter2_batch_index"]
    files: dict[str, dict[str, Any]] = {}
    for key, raw_path in sorted(canonical_paths.items()):
        if key not in _ARROW_IMMUTABLE_ARTIFACT_FILE_KEYS and key not in _ARROW_IMMUTABLE_BATCH_INDEX_KEYS:
            continue
        declared_path = Path(os.fspath(raw_path)).resolve()
        artifact_path = declared_path / "manifest.json" if declared_path.is_dir() else declared_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"cannot inventory Arrow artifact {key}={artifact_path}")
        files[key] = {
            "path": _manifest_relative_path(declared_path, root, artifact_key=key),
            "kind": "directory_manifest" if declared_path.is_dir() else "file",
            "byte_count": artifact_path.stat().st_size,
            "sha256": _sha256_file(artifact_path),
        }
    encoded_files = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": ARROW_ARTIFACT_GENERATION_SCHEMA_VERSION,
        "generation_id": hashlib.sha256(encoded_files).hexdigest(),
        "files": files,
    }


def _stable_file_digest_token(path: Path) -> tuple[str, int, int, str]:
    for _attempt in range(3):
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            return str(path.resolve()), int(after.st_size), int(after.st_mtime_ns), digest
    raise RuntimeError(f"artifact changed during all 3 checksum attempts: {path}")


def _arrow_artifact_manifest_path(paths: Mapping[str, str]) -> Path | None:
    explicit = paths.get("manifest")
    if explicit is not None:
        return Path(os.path.abspath(explicit))
    signatures = paths.get("signatures")
    if signatures is None:
        return None
    return Path(os.path.abspath(signatures)).parent / "manifest.json"


def _validate_arrow_bundle_manifest(
    paths: Mapping[str, str],
    *,
    expected_normalization_version: str | None,
    context: str,
    required_keys: Sequence[str],
    producer_hint: str,
) -> _VerifiedArrowArtifactGeneration:
    manifest_path = _arrow_artifact_manifest_path(paths)
    verified = _verified_arrow_artifact_manifest(paths)
    if verified is None:
        manifest_exists = manifest_path is not None and manifest_path.is_file()
        missing_key = "manifest.artifact_generation" if manifest_exists else "manifest"
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required_keys,
            missing_keys=(missing_key,),
            missing_files={} if manifest_exists else {"manifest": str(manifest_path or "<unresolved>")},
            producer_hint=producer_hint,
        )
    normalization_version = verified.normalization_version
    if normalization_version is not None:
        if expected_normalization_version is not None and normalization_version != expected_normalization_version:
            raise MissingArrowArtifactError(
                context=context,
                required_keys=required_keys,
                missing_keys=(),
                missing_files={
                    "manifest": (
                        f"normalization_version mismatch: artifact is {normalization_version!r} but the model "
                        f"feature contract requires {expected_normalization_version!r}"
                    )
                },
                producer_hint=producer_hint,
            )
    else:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required_keys,
            missing_keys=("manifest.normalization_version",),
            missing_files={},
            producer_hint=producer_hint,
        )
    return verified


def _verified_arrow_artifact_manifest(
    paths: Mapping[str, str],
) -> _VerifiedArrowArtifactGeneration | None:
    """Verify one exact immutable generation."""

    manifest_path = _arrow_artifact_manifest_path(paths)
    if manifest_path is None:
        return None
    if not manifest_path.is_file():
        return None
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Arrow artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Arrow artifact manifest must be a JSON object: {manifest_path}")
    normalization_version = manifest.get("normalization_version")
    if normalization_version is not None and normalization_version != NORMALIZATION_VERSION:
        raise ValueError(
            f"Arrow artifact manifest normalization_version is invalid: {normalization_version!r}; "
            f"expected {NORMALIZATION_VERSION!r}"
        )
    generation = manifest.get("artifact_generation")
    if generation is None:
        return None
    if not isinstance(generation, Mapping):
        raise ValueError(f"Arrow artifact manifest artifact_generation must be an object: {manifest_path}")
    if generation.get("schema_version") != ARROW_ARTIFACT_GENERATION_SCHEMA_VERSION:
        raise ValueError(f"Arrow artifact manifest has unsupported artifact_generation schema: {manifest_path}")
    files = generation.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"Arrow artifact manifest artifact_generation is missing files: {manifest_path}")
    invalid_generation_keys = sorted(
        str(key)
        for key in files
        if key not in _ARROW_IMMUTABLE_ARTIFACT_FILE_KEYS and key not in _ARROW_IMMUTABLE_BATCH_INDEX_KEYS
    )
    if invalid_generation_keys:
        request_sidecars = sorted(set(invalid_generation_keys).intersection(_ARROW_REQUEST_SIDECAR_KEYS))
        unsupported_keys = sorted(set(invalid_generation_keys) - set(request_sidecars))
        raise ValueError(
            "Arrow artifact generation files must contain only immutable dataset artifacts; "
            f"request_sidecars={request_sidecars} unsupported={unsupported_keys}: {manifest_path}"
        )
    declared_generation_id = generation.get("generation_id")
    computed_generation_id = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if declared_generation_id != computed_generation_id:
        raise ValueError(f"Arrow artifact manifest generation_id mismatch: {manifest_path}")

    material_keys = sorted(
        key for key in paths if key in _ARROW_IMMUTABLE_ARTIFACT_FILE_KEYS or key in _ARROW_IMMUTABLE_BATCH_INDEX_KEYS
    )
    verification_items: list[tuple[Path, str, int, str]] = []
    for key in material_keys:
        entry = files.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"Arrow artifact generation is missing files.{key}: {manifest_path}")
        raw_declared_path = entry.get("path")
        if not isinstance(raw_declared_path, str) or not raw_declared_path.strip():
            raise ValueError(f"Arrow artifact generation files.{key}.path is invalid: {manifest_path}")
        supplied_path = Path(os.path.abspath(os.fspath(paths[key])))
        declared_path = Path(raw_declared_path)
        if declared_path.is_absolute():
            raise ValueError(
                f"Arrow artifact generation files.{key}.path must be manifest-relative: {raw_declared_path!r}"
            )
        manifest_root = manifest_path.parent.resolve()
        declared_path = (manifest_root / declared_path).resolve()
        try:
            declared_path.relative_to(manifest_root)
        except ValueError as exc:
            shared_name_counts_path = manifest_root.parent / "name_counts_index"
            if (
                key == "name_counts_index"
                and Path(raw_declared_path) == Path("..") / "name_counts_index"
                and declared_path == shared_name_counts_path
            ):
                pass
            else:
                raise ValueError(
                    f"Arrow artifact generation files.{key}.path escapes the manifest directory: "
                    f"{raw_declared_path!r}"
                ) from exc
        actual_path = supplied_path.resolve()
        if declared_path != actual_path:
            raise ValueError(
                f"Arrow artifact generation files.{key}.path does not match supplied path: "
                f"declared={declared_path} supplied={actual_path}"
            )
        kind = entry.get("kind")
        if kind == "directory_manifest":
            artifact_path = actual_path / "manifest.json"
        elif kind == "file":
            artifact_path = actual_path
        else:
            raise ValueError(f"Arrow artifact generation files.{key}.kind is invalid: {kind!r}")
        expected_bytes = entry.get("byte_count")
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"Arrow artifact generation files.{key}.byte_count is invalid")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"Arrow artifact generation files.{key}.sha256 is invalid")
        stat = artifact_path.stat()
        if stat.st_size != expected_bytes:
            raise ValueError(f"Arrow artifact generation files.{key}.byte_count mismatch: {artifact_path}")
        verification_items.append((artifact_path, key, expected_bytes, expected_sha256))
    for artifact_path, key, expected_bytes, expected_sha256 in verification_items:
        _resolved, observed_bytes, _mtime_ns, observed_sha256 = _stable_file_digest_token(artifact_path)
        if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
            raise ValueError(f"Arrow artifact generation files.{key} checksum mismatch: {artifact_path}")
    if manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError(f"Arrow artifact manifest changed during verification: {manifest_path}")

    return _VerifiedArrowArtifactGeneration(
        generation_id=computed_generation_id,
        normalization_version=None if normalization_version is None else str(normalization_version),
    )


def verified_arrow_artifact_generation(paths: Mapping[str, str]) -> str | None:
    """Return the retained generation ID or verify a raw path mapping."""

    if isinstance(paths, ValidatedArrowInputs):
        return paths.generation_id
    verified = _verified_arrow_artifact_manifest(paths)
    return None if verified is None else verified.generation_id


def _name_counts_index_error(path: Path) -> str | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return f"{manifest_path} (missing manifest.json)"
    try:
        manifest_stat_before = manifest_path.stat()
        manifest_bytes = manifest_path.read_bytes()
        manifest_stat_after = manifest_path.stat()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"{manifest_path} (invalid manifest: {exc})"
    if not isinstance(manifest, Mapping):
        return f"{manifest_path} (invalid manifest: expected a JSON object)"
    manifest_stat_token = (
        int(manifest_stat_after.st_dev),
        int(manifest_stat_after.st_ino),
        int(manifest_stat_after.st_size),
        int(manifest_stat_after.st_mtime_ns),
        int(manifest_stat_after.st_ctime_ns),
    )
    if manifest_stat_token != (
        int(manifest_stat_before.st_dev),
        int(manifest_stat_before.st_ino),
        int(manifest_stat_before.st_size),
        int(manifest_stat_before.st_mtime_ns),
        int(manifest_stat_before.st_ctime_ns),
    ):
        return f"{manifest_path} (manifest changed while reading)"
    schema_version = manifest.get("schema_version")
    if schema_version != NAME_COUNTS_INDEX_SCHEMA_VERSION:
        return (
            f"{manifest_path} (unsupported schema_version {schema_version!r}; "
            f"expected {NAME_COUNTS_INDEX_SCHEMA_VERSION!r})"
        )
    normalization_version = manifest.get("normalization_version")
    if normalization_version != NORMALIZATION_VERSION:
        return (
            f"{manifest_path} (invalid normalization_version {normalization_version!r}; "
            f"expected {NORMALIZATION_VERSION!r})"
        )
    source_provenance = manifest.get("source_provenance")
    if not isinstance(source_provenance, Mapping):
        return f"{manifest_path} (missing source_provenance mapping)"
    if source_provenance.get("normalization_version") != normalization_version:
        return f"{manifest_path} (source_provenance normalization_version mismatch)"
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return f"{manifest_path} (missing files mapping)"
    verified_files: list[tuple[Path, str, int]] = []
    for file_key in NAME_COUNTS_INDEX_MANIFEST_FILES:
        entry = files.get(file_key)
        if not isinstance(entry, Mapping):
            return f"{manifest_path} (missing files.{file_key})"
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            return f"{manifest_path} (missing files.{file_key}.path)"
        resolved = Path(path_value)
        if not resolved.is_absolute():
            resolved = path / resolved
        try:
            resolved.resolve().relative_to(path.resolve())
        except ValueError:
            return f"{resolved} (files.{file_key}.path escapes the name_counts_index directory)"
        if not resolved.is_file():
            return f"{resolved} (missing files.{file_key}.path target)"
        file_stat = resolved.stat()
        byte_count = entry.get("byte_count")
        if isinstance(byte_count, int) and file_stat.st_size != byte_count:
            return f"{resolved} (files.{file_key}.byte_count mismatch)"
        if not isinstance(byte_count, int) or byte_count < 0:
            return f"{manifest_path} (missing files.{file_key}.byte_count)"
        expected_sha256 = entry.get("sha256")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            return f"{manifest_path} (missing files.{file_key}.sha256)"
        if not (resolved.parent / ".published").is_file():
            return f"{resolved.parent / '.published'} (missing published-generation marker)"
        verified_files.append(
            (
                resolved,
                expected_sha256 if isinstance(expected_sha256, str) else "",
                int(file_stat.st_size),
            )
        )

    material_stat_tokens = tuple(
        (
            str(resolved.resolve()),
            int(file_stat.st_dev),
            int(file_stat.st_ino),
            int(file_stat.st_size),
            int(file_stat.st_mtime_ns),
            int(file_stat.st_ctime_ns),
        )
        for resolved, _expected_sha256, _expected_bytes in verified_files
        for file_stat in (resolved.stat(),)
    )
    for resolved, expected_sha256, expected_bytes in verified_files:
        token = _stable_file_digest_token(resolved)
        if token[1] != expected_bytes or token[3] != expected_sha256:
            return f"{resolved} (declared SHA-256 mismatch)"
    try:
        if manifest_path.read_bytes() != manifest_bytes:
            return f"{manifest_path} (manifest changed during verification)"
    except OSError as exc:
        return f"{manifest_path} (manifest changed during verification: {exc})"
    final_manifest_stat = manifest_path.stat()
    if manifest_stat_token != (
        int(final_manifest_stat.st_dev),
        int(final_manifest_stat.st_ino),
        int(final_manifest_stat.st_size),
        int(final_manifest_stat.st_mtime_ns),
        int(final_manifest_stat.st_ctime_ns),
    ):
        return f"{manifest_path} (manifest metadata changed during verification)"
    final_material_stat_tokens = tuple(
        (
            str(resolved.resolve()),
            int(file_stat.st_dev),
            int(file_stat.st_ino),
            int(file_stat.st_size),
            int(file_stat.st_mtime_ns),
            int(file_stat.st_ctime_ns),
        )
        for resolved, _expected_sha256, _expected_bytes in verified_files
        for file_stat in (resolved.stat(),)
    )
    if final_material_stat_tokens != material_stat_tokens:
        return f"{manifest_path} (name-count index files changed during verification)"
    return None


def read_name_counts_index_normalization_version(path: Any) -> str:
    """Read the normalization_version recorded in a name_counts_index/ manifest.

    Only the package's current canonical normalization contract is executable.
    """

    manifest_path = Path(os.fspath(path)) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = manifest.get("normalization_version")
    if value != NORMALIZATION_VERSION:
        raise ValueError(
            f"{manifest_path} has invalid normalization_version {value!r}; " f"expected {NORMALIZATION_VERSION!r}"
        )
    return str(value)


def _require_name_counts_index_normalization(
    paths: Mapping[str, str],
    *,
    expected_normalization_version: str | None,
    context: str,
    required_keys: Sequence[str],
    producer_hint: str,
) -> None:
    if expected_normalization_version is None or "name_counts_index" not in paths:
        return
    artifact_version = read_name_counts_index_normalization_version(paths["name_counts_index"])
    if artifact_version != expected_normalization_version:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required_keys,
            missing_keys=(),
            missing_files={
                "name_counts_index": (
                    f"normalization_version mismatch: artifact is {artifact_version!r} but the model "
                    f"feature contract requires {expected_normalization_version!r}"
                )
            },
            producer_hint=producer_hint,
        )


def require_normalization_version(value: Any, *, context: str) -> str:
    """Return the explicitly declared canonical normalization version."""

    if value != NORMALIZATION_VERSION:
        raise ValueError(f"{context} requires normalization_version={NORMALIZATION_VERSION!r}, got {value!r}")
    return NORMALIZATION_VERSION


def require_feature_contract_normalization_version(owner: Any, *, context: str) -> str:
    """Return the explicitly declared normalization version for an owner.

    Rust Arrow construction must be tied to the model or training contract that
    owns the artifacts.  Treating an absent value as the package's current
    version would allow legacy artifacts to be relabeled during construction.
    """

    contract = getattr(owner, "feature_contract", None)
    if not isinstance(contract, Mapping):
        raise ValueError(f"{context} requires a feature_contract mapping")
    return require_normalization_version(
        contract.get("normalization_version"),
        context=f"{context} feature_contract",
    )


def require_name_counts_index_artifact(
    path: Any,
    *,
    context: str,
    producer_hint: str,
) -> str:
    """Require a manifest-backed name-count index directory and return its normalized path."""

    path_text = os.fspath(path) if isinstance(path, os.PathLike) else str(path)
    if not path_text.strip():
        missing_files = {"name_counts_index": "<empty>"}
    elif path_text == ".":
        missing_files = {"name_counts_index": "."}
    else:
        index_path = Path(path_text)
        if not index_path.exists():
            missing_files = {"name_counts_index": str(index_path)}
        elif not index_path.is_dir():
            missing_files = {"name_counts_index": f"{index_path} (expected directory)"}
        else:
            index_error = _name_counts_index_error(index_path)
            missing_files = {"name_counts_index": index_error} if index_error is not None else {}
    if missing_files:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=("name_counts_index",),
            missing_keys=(),
            missing_files=missing_files,
            producer_hint=producer_hint,
        )
    return path_text


def _missing_or_wrong_kind_artifacts(paths: Mapping[str, str], keys: Iterable[str]) -> dict[str, str]:
    missing: dict[str, str] = {}
    for key in keys:
        if key not in paths:
            continue
        path = Path(paths[key])
        if not path.exists():
            missing[key] = str(path)
        elif key in DIRECTORY_ARTIFACT_KEYS and not path.is_dir():
            missing[key] = f"{path} (expected directory)"
        elif key == "name_counts_index":
            index_error = _name_counts_index_error(path)
            if index_error is not None:
                missing[key] = index_error
        elif key not in DIRECTORY_ARTIFACT_KEYS and not path.is_file():
            missing[key] = f"{path} (expected file)"
    return missing


class MissingArrowArtifactError(ValueError):
    """Raised when a strict Arrow production route is missing required artifacts."""

    def __init__(
        self,
        *,
        context: str,
        required_keys: Sequence[str],
        missing_keys: Sequence[str],
        missing_files: Mapping[str, str],
        producer_hint: str,
    ) -> None:
        self.context = str(context)
        self.required_keys = tuple(str(key) for key in required_keys)
        self.missing_keys = tuple(str(key) for key in missing_keys)
        self.missing_files = {str(key): str(value) for key, value in missing_files.items()}
        self.producer_hint = str(producer_hint)
        details = [f"{self.context} is missing required Arrow artifacts"]
        if self.missing_keys:
            details.append(f"missing mapping keys: {', '.join(self.missing_keys)}")
        if self.missing_files:
            formatted_files = "; ".join(f"{key}={value}" for key, value in sorted(self.missing_files.items()))
            details.append(f"missing files: {formatted_files}")
        if self.producer_hint:
            details.append(f"producer hint: {self.producer_hint}")
        super().__init__(". ".join(details))


def _normalize_arrow_path_values(
    paths: Mapping[Any, Any],
    *,
    omit_none: bool = False,
) -> tuple[dict[str, str], dict[str, str]]:
    normalized: dict[str, str] = {}
    invalid: dict[str, str] = {}
    for key, value in paths.items():
        key_text = str(key)
        if value is None:
            if omit_none:
                continue
            invalid[key_text] = "<None>"
            continue
        path_text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
        if not path_text.strip():
            invalid[key_text] = "<empty>"
            continue
        if path_text == ".":
            invalid[key_text] = "."
            continue
        try:
            normalized[key_text] = str(Path(path_text).expanduser().resolve(strict=False))
        except OSError as exc:
            invalid[key_text] = f"{path_text} ({exc})"
    return normalized, invalid


def normalize_arrow_paths(paths: Mapping[Any, Any], *, omit_none: bool = False) -> dict[str, str]:
    """Return Arrow path mappings with explicit rejection of missing path values."""

    normalized, invalid = _normalize_arrow_path_values(paths, omit_none=omit_none)
    if invalid:
        key, reason = next(iter(invalid.items()))
        if reason == "<None>":
            raise ValueError(f"Arrow path for {key!r} is None")
        if reason == "<empty>":
            raise ValueError(f"Arrow path for {key!r} is empty")
        if reason == ".":
            raise ValueError(f"Arrow path for {key!r} resolves to the current directory")
        raise ValueError(f"Arrow path for {key!r} is invalid: {reason}")
    return normalized


def required_filtered_read_batch_index_keys(paths: Mapping[str, str]) -> tuple[str, ...]:
    """Return required batch-index keys for filtered Arrow reads over these paths."""

    required = [RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[table_key] for table_key in FILTERED_READ_ARROW_TABLE_KEYS]
    if "specter" in paths:
        required.append(RAW_PLANNER_ARROW_BATCH_INDEX_KEYS["specter"])
    return tuple(required)


def require_filtered_arrow_batch_indexes(
    paths: Mapping[str, str],
    *,
    context: str = "RustFeaturizer.from_arrow_paths",
    producer_hint: str = "generate raw-planner batch indexes with scripts/convert_to_arrow.py",
) -> None:
    """Require batch indexes for production filtered Arrow featurizer builds."""

    required = required_filtered_read_batch_index_keys(paths)
    missing_keys = [key for key in required if key not in paths]
    missing_files = _missing_or_wrong_kind_artifacts(paths, required)
    if missing_keys or missing_files:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required,
            missing_keys=missing_keys,
            missing_files=missing_files,
            producer_hint=producer_hint,
        )


def _validate_batch_indexes(paths: Mapping[str, str]) -> None:
    """Strictly validate the batch indexes for one immutable Arrow generation."""

    # Local import avoids a module cycle: feature_block_arrow consumes the
    # canonical path helpers defined in this module.
    from s2and.incremental_linking.feature_block_arrow import validate_arrow_batch_lookup_index

    for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS:
        if contract.table_key not in paths:
            continue
        validate_arrow_batch_lookup_index(
            paths[contract.table_key],
            paths[contract.index_key],
            key_column=contract.key_column,
        )


def require_arrow_artifacts(
    arrow_paths: Mapping[str, Any],
    *,
    required_keys: Sequence[str],
    context: str,
    producer_hint: str,
) -> dict[str, str]:
    """Require specific Arrow path keys and existing files, returning normalized paths."""

    required = tuple(str(key) for key in required_keys)
    missing_keys = [key for key in required if key not in arrow_paths]
    normalized, invalid_paths = _normalize_arrow_path_values(arrow_paths)
    missing_files = _missing_or_wrong_kind_artifacts(normalized, required)
    missing_files.update({key: invalid_paths[key] for key in required if key in invalid_paths})
    if missing_keys or missing_files:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required,
            missing_keys=missing_keys,
            missing_files=missing_files,
            producer_hint=producer_hint,
        )
    return normalized


def _validate_complete_arrow_artifacts(
    arrow_paths: Mapping[str, Any],
    *,
    require_specter: bool,
    require_name_counts_index: bool,
    require_cluster_seeds: bool = False,
    required_request_sidecars: Sequence[str] = (),
    expected_normalization_version: str | None = None,
    context: str,
    producer_hint: str,
) -> ValidatedArrowInputs:
    """Apply the shared complete-bundle contract used by the fixed profiles."""

    required = {"signatures", "papers", "paper_authors"}
    if require_specter:
        required.add("specter")
    if require_name_counts_index:
        required.add("name_counts_index")
    if require_cluster_seeds:
        required.add("cluster_seeds")
    unsupported_request_sidecars = sorted(set(required_request_sidecars) - _ARROW_REQUEST_SIDECAR_KEYS)
    if unsupported_request_sidecars:
        raise ValueError(f"required request sidecar keys are unsupported: {unsupported_request_sidecars}")
    required.update(str(key) for key in required_request_sidecars)

    if isinstance(arrow_paths, ValidatedArrowInputs):
        profiled_paths = arrow_paths
        if require_specter:
            canonical_paths = dict(arrow_paths)
            if "specter" not in canonical_paths and "specter2" in canonical_paths:
                canonical_paths["specter"] = canonical_paths["specter2"]
            if "specter_batch_index" not in canonical_paths and "specter2_batch_index" in canonical_paths:
                canonical_paths["specter_batch_index"] = canonical_paths["specter2_batch_index"]
            canonical_paths.pop("specter2", None)
            canonical_paths.pop("specter2_batch_index", None)
            if canonical_paths != dict(arrow_paths):
                profiled_paths = ValidatedArrowInputs._from_verified(
                    paths=canonical_paths,
                    generation_id=arrow_paths.generation_id,
                    normalization_version=arrow_paths.normalization_version,
                    capability=_VERIFIED_ARROW_INPUTS_CAPABILITY,
                )
        elif _SPECTER_PATH_KEYS.intersection(arrow_paths):
            profiled_paths = arrow_paths.without(*_SPECTER_PATH_KEYS)

        invalid_profile_keys = sorted(UNSUPPORTED_ARROW_NAME_ALIAS_KEYS.intersection(profiled_paths))
        if "name_counts" in profiled_paths:
            invalid_profile_keys.append("name_counts")
        if "name_counts_index_dir" in profiled_paths:
            invalid_profile_keys.append("name_counts_index_dir")
        if invalid_profile_keys:
            raise MissingArrowArtifactError(
                context=context,
                required_keys=sorted(required),
                missing_keys=(),
                missing_files={key: "unsupported Arrow input profile key" for key in invalid_profile_keys},
                producer_hint=producer_hint,
            )

        required.update(required_filtered_read_batch_index_keys(profiled_paths))
        missing_keys = sorted(key for key in required if key not in profiled_paths)
        if missing_keys:
            raise MissingArrowArtifactError(
                context=context,
                required_keys=sorted(required),
                missing_keys=missing_keys,
                missing_files={},
                producer_hint=producer_hint,
            )
        if (
            expected_normalization_version is not None
            and profiled_paths.normalization_version != expected_normalization_version
        ):
            raise MissingArrowArtifactError(
                context=context,
                required_keys=sorted(required),
                missing_keys=(),
                missing_files={
                    "manifest": (
                        f"normalization_version mismatch: artifact is {profiled_paths.normalization_version!r} "
                        f"but the model feature contract requires {expected_normalization_version!r}"
                    )
                },
                producer_hint=producer_hint,
            )
        return profiled_paths

    missing_keys = sorted(key for key in required if key not in arrow_paths)
    normalized, invalid_paths = _normalize_arrow_path_values(arrow_paths)
    for key in UNSUPPORTED_ARROW_NAME_ALIAS_KEYS.intersection(normalized):
        invalid_paths[key] = "name aliases must be supplied via the name_tuples argument, not Arrow path bundles"
    if "name_counts" in normalized:
        invalid_paths["name_counts"] = "legacy name_counts Arrow tables are unsupported; use name_counts_index"
    if "name_counts_index_dir" in normalized:
        invalid_paths["name_counts_index_dir"] = (
            "legacy name-count index aliases are unsupported; use name_counts_index"
        )

    if require_specter and "specter" not in normalized and "specter2" in normalized:
        normalized["specter"] = normalized["specter2"]
        invalid_paths.pop("specter", None)
        if "specter" in missing_keys:
            missing_keys.remove("specter")
    if require_specter and "specter" in normalized:
        invalid_paths.pop("specter2", None)
    if require_specter and "specter_batch_index" not in normalized and "specter2_batch_index" in normalized:
        normalized["specter_batch_index"] = normalized["specter2_batch_index"]
        invalid_paths.pop("specter_batch_index", None)
    if require_specter and "specter_batch_index" in normalized:
        invalid_paths.pop("specter2_batch_index", None)
    if require_specter:
        normalized.pop("specter2", None)
        normalized.pop("specter2_batch_index", None)
        invalid_paths.pop("specter2", None)
        invalid_paths.pop("specter2_batch_index", None)
    if not require_specter:
        normalized.pop("specter", None)
        normalized.pop("specter_batch_index", None)
        normalized.pop("specter2", None)
        normalized.pop("specter2_batch_index", None)
        invalid_paths.pop("specter", None)
        invalid_paths.pop("specter_batch_index", None)
        invalid_paths.pop("specter2", None)
        invalid_paths.pop("specter2_batch_index", None)

    for key in required_filtered_read_batch_index_keys(normalized):
        required.add(key)
        if key not in normalized:
            missing_keys.append(key)

    required_or_declared_keys = {
        key
        for key in normalized
        if key in required or key.endswith("_batch_index") or key in _ARROW_REQUEST_SIDECAR_KEYS
    }
    missing_files = _missing_or_wrong_kind_artifacts(normalized, required_or_declared_keys)
    missing_files.update(invalid_paths)
    if missing_keys or missing_files:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=sorted(required),
            missing_keys=sorted(set(missing_keys)),
            missing_files=missing_files,
            producer_hint=producer_hint,
        )
    verified = _validate_arrow_bundle_manifest(
        normalized,
        expected_normalization_version=expected_normalization_version,
        context=context,
        required_keys=sorted(required),
        producer_hint=producer_hint,
    )
    if verified.normalization_version is None:  # pragma: no cover - manifest validation rejects this
        raise RuntimeError("validated Arrow artifact generation is missing normalization_version")
    _require_name_counts_index_normalization(
        normalized,
        expected_normalization_version=verified.normalization_version,
        context=context,
        required_keys=sorted(required),
        producer_hint=producer_hint,
    )
    _validate_batch_indexes(normalized)
    return ValidatedArrowInputs._from_verified(
        paths=normalized,
        generation_id=verified.generation_id,
        normalization_version=verified.normalization_version,
        capability=_VERIFIED_ARROW_INPUTS_CAPABILITY,
    )


def validate_arrow_training_artifacts(
    arrow_paths: Mapping[str, Any],
    *,
    require_specter: bool,
    require_name_counts_index: bool,
    expected_normalization_version: str,
    context: str = "Arrow training ingest",
    producer_hint: str = "generate a complete training bundle with scripts/convert_to_arrow.py",
) -> ValidatedArrowInputs:
    """Validate the fixed canonical Arrow training-ingest profile."""

    return _validate_complete_arrow_artifacts(
        arrow_paths,
        require_specter=require_specter,
        require_name_counts_index=require_name_counts_index,
        expected_normalization_version=expected_normalization_version,
        context=context,
        producer_hint=producer_hint,
    )


def validate_arrow_prediction_artifacts(
    arrow_paths: Mapping[str, Any],
    *,
    require_specter: bool,
    require_name_counts_index: bool,
    require_cluster_seeds: bool = False,
    required_request_sidecars: Sequence[str] = (),
    expected_normalization_version: str | None = None,
    context: str = "Canonical Arrow prediction",
    producer_hint: str = (
        "generate a complete Arrow bundle with scripts/convert_to_arrow.py or use the published "
        "s2and-release-arrow bundle"
    ),
) -> ValidatedArrowInputs:
    """Validate the fixed canonical prediction profile."""

    return _validate_complete_arrow_artifacts(
        arrow_paths,
        require_specter=require_specter,
        require_name_counts_index=require_name_counts_index,
        require_cluster_seeds=require_cluster_seeds,
        required_request_sidecars=required_request_sidecars,
        expected_normalization_version=expected_normalization_version,
        context=context,
        producer_hint=producer_hint,
    )


def validate_arrow_publication_artifacts(
    arrow_paths: Mapping[str, Any],
    *,
    require_specter: bool,
    require_name_counts_index: bool,
    expected_normalization_version: str | None = None,
    context: str = "Arrow publication integrity",
    producer_hint: str = "publish a complete immutable Arrow bundle with all batch indexes",
) -> ValidatedArrowInputs:
    """Validate the fixed publication/integrity profile."""

    return _validate_complete_arrow_artifacts(
        arrow_paths,
        require_specter=require_specter,
        require_name_counts_index=require_name_counts_index,
        expected_normalization_version=expected_normalization_version,
        context=context,
        producer_hint=producer_hint,
    )

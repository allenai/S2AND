"""Canonical validation helpers for Arrow-backed runtime inputs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any

from s2and.consts import NORMALIZATION_VERSION
from s2and.name_counts_manifest import ValidatedNameCountsManifest


@dataclass(frozen=True)
class ArrowBatchIndexContract:
    """One Arrow table and its raw-planner batch lookup index contract."""

    table_key: str
    key_column: str
    index_key: str
    max_record_batch_rows: int


@dataclass(frozen=True)
class _ArrowArtifactGenerationFile:
    """One manifest-bound immutable file retained for combined validation."""

    path: Path
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class _VerifiedArrowArtifactGeneration:
    """Manifest facts retained after one full immutable-generation check."""

    generation_id: str
    normalization_version: str | None
    files: Mapping[str, _ArrowArtifactGenerationFile]


@dataclass(frozen=True, init=False)
class ValidatedArrowInputs(Mapping[str, str]):
    """Immutable, integrity-checked Arrow paths for one artifact generation."""

    paths: Mapping[str, str]
    generation_id: str
    normalization_version: str
    name_counts_manifest: ValidatedNameCountsManifest | None
    _name_counts_index: Any | None

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
        name_counts_manifest: ValidatedNameCountsManifest | None = None,
        name_counts_index: Any | None = None,
    ) -> ValidatedArrowInputs:
        """Create an instance from facts established by internal validation."""

        instance = object.__new__(cls)
        object.__setattr__(instance, "paths", MappingProxyType(dict(paths)))
        object.__setattr__(instance, "generation_id", str(generation_id))
        object.__setattr__(instance, "normalization_version", str(normalization_version))
        object.__setattr__(instance, "name_counts_manifest", name_counts_manifest)
        object.__setattr__(instance, "_name_counts_index", name_counts_index)
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
            name_counts_manifest=(None if "name_counts_index" in removed else self.name_counts_manifest),
            name_counts_index=(None if "name_counts_index" in removed else self._name_counts_index),
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
            name_counts_manifest=self.name_counts_manifest,
            name_counts_index=self._name_counts_index,
        )

    def _retained_native_name_counts_index(self) -> Any | None:
        """Return the exact native snapshot bound to retained manifest facts."""

        index = self._name_counts_index
        return None if index is None else index._native  # noqa: SLF001 - retained internal snapshot


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
ARROW_ARTIFACT_GENERATION_SCHEMA_VERSION = "s2and_arrow_artifact_generation_v1"
_ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS = frozenset(
    {
        "normalization_version",
        "paths",
        "artifact_generation",
    }
)
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
    }
)
_UNSUPPORTED_SPECTER_PATH_KEYS = frozenset({"specter2", "specter2_batch_index"})
_RUNTIME_ARROW_GENERATION_CACHE_SIZE = 4
_RuntimeArrowGenerationCacheKey = tuple[str, tuple[tuple[str, str], ...]]
_RUNTIME_ARROW_GENERATION_CACHE: OrderedDict[_RuntimeArrowGenerationCacheKey, ValidatedArrowInputs] = OrderedDict()
_RUNTIME_ARROW_GENERATION_KEY_LOCKS: dict[_RuntimeArrowGenerationCacheKey, threading.Lock] = {}
_RUNTIME_ARROW_GENERATION_CACHE_LOCK = threading.Lock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _shared_name_counts_relative_path(path: Path, manifest_root: Path, *, artifact_key: str) -> Path | None:
    """Return an allowed relative path for a known shared name-count layout."""

    if artifact_key != "name_counts_index":
        return None
    candidates = [(manifest_root.parent / "name_counts_index", Path("..") / "name_counts_index")]
    if manifest_root.parent.name == "datasets":
        candidates.extend(
            [
                (
                    manifest_root.parent.parent / "name_counts_index",
                    Path("..") / ".." / "name_counts_index",
                ),
                (
                    manifest_root.parent.parent.parent / "name_counts_index",
                    Path("..") / ".." / ".." / "name_counts_index",
                ),
            ]
        )
    return next((relative for candidate, relative in candidates if path == candidate), None)


def _portable_manifest_path(path: PurePath) -> str:
    """Serialize a manifest path with platform-independent separators."""

    return path.as_posix()


def _manifest_relative_path(path_value: Any, manifest_dir: Path, *, artifact_key: str) -> str:
    path = Path(os.fspath(path_value)).resolve()
    root = manifest_dir.resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        relative_path = _shared_name_counts_relative_path(path, root, artifact_key=artifact_key)
        if relative_path is None:
            raise ValueError(
                f"Arrow artifact path must remain within manifest directory: path={path} root={root}"
            ) from exc
    return _portable_manifest_path(relative_path)


def _build_arrow_artifact_generation(paths: Mapping[str, Any], manifest_dir: str | Path) -> dict[str, Any]:
    """Build the canonical content inventory used by Arrow bundle writers."""

    root = Path(manifest_dir)
    canonical_paths = {str(key): value for key, value in paths.items()}
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


def build_arrow_artifact_manifest(
    paths: Mapping[str, Any],
    manifest_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one canonical, portable Arrow artifact manifest.

    ``metadata`` may add dataset- or command-specific fields, but the runtime
    contract fields are owned here so every producer emits the same paths,
    normalization version, and immutable-generation inventory.
    """

    root = Path(manifest_dir)
    canonical_paths = {str(key): value for key, value in paths.items() if str(key) != "manifest"}
    extra = {} if metadata is None else dict(metadata)
    conflicting_fields = sorted(_ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS.intersection(extra))
    if conflicting_fields:
        raise ValueError(f"Arrow artifact manifest metadata cannot override canonical fields: {conflicting_fields}")
    return {
        **extra,
        "normalization_version": NORMALIZATION_VERSION,
        "paths": {
            key: _manifest_relative_path(value, root, artifact_key=key)
            for key, value in sorted(canonical_paths.items())
        },
        "artifact_generation": _build_arrow_artifact_generation(canonical_paths, root),
    }


def write_arrow_artifact_manifest(
    manifest: Mapping[str, Any],
    manifest_dir: str | Path,
) -> Path:
    """Atomically publish a manifest built by :func:`build_arrow_artifact_manifest`."""

    root = Path(manifest_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    missing_fields = sorted(_ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS.difference(manifest))
    if missing_fields:
        raise ValueError(f"Arrow artifact manifest is missing canonical fields: {missing_fields}")
    encoded = json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=root,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as output:
            output.write(encoded)
            temporary_path = Path(output.name)
        temporary_path.replace(manifest_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return manifest_path


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
    defer_batch_material: bool = False,
) -> _VerifiedArrowArtifactGeneration:
    manifest_path = _arrow_artifact_manifest_path(paths)
    verified = _verified_arrow_artifact_manifest(paths, defer_batch_material=defer_batch_material)
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
    *,
    defer_batch_material: bool = False,
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
    deferred_keys: set[str] = set()
    if defer_batch_material:
        for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS:
            if contract.table_key in paths and contract.index_key in paths:
                deferred_keys.update((contract.table_key, contract.index_key))
    verified_files: dict[str, _ArrowArtifactGenerationFile] = {}
    for key in material_keys:
        entry = files.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"Arrow artifact generation is missing files.{key}: {manifest_path}")
        raw_declared_path = entry.get("path")
        if not isinstance(raw_declared_path, str) or not raw_declared_path.strip():
            raise ValueError(f"Arrow artifact generation files.{key}.path is invalid: {manifest_path}")
        supplied_path = Path(os.path.abspath(os.fspath(paths[key])))
        declared_relative_path = Path(raw_declared_path)
        if declared_relative_path.is_absolute():
            raise ValueError(
                f"Arrow artifact generation files.{key}.path must be manifest-relative: {raw_declared_path!r}"
            )
        manifest_root = manifest_path.parent.resolve()
        declared_path = (manifest_root / declared_relative_path).resolve()
        try:
            declared_path.relative_to(manifest_root)
        except ValueError as exc:
            allowed_relative_path = _shared_name_counts_relative_path(
                declared_path,
                manifest_root,
                artifact_key=key,
            )
            if declared_relative_path != allowed_relative_path:
                raise ValueError(
                    f"Arrow artifact generation files.{key}.path escapes the manifest directory: {raw_declared_path!r}"
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
        if artifact_path.stat().st_size != expected_bytes:
            raise ValueError(f"Arrow artifact generation files.{key}.byte_count mismatch: {artifact_path}")
        if key not in deferred_keys and _sha256_file(artifact_path) != expected_sha256:
            raise ValueError(f"Arrow artifact generation files.{key} checksum mismatch: {artifact_path}")
        verified_files[key] = _ArrowArtifactGenerationFile(
            path=artifact_path,
            byte_count=expected_bytes,
            sha256=expected_sha256,
        )

    return _VerifiedArrowArtifactGeneration(
        generation_id=computed_generation_id,
        normalization_version=None if normalization_version is None else str(normalization_version),
        files=MappingProxyType(verified_files),
    )


def _name_counts_index_error(path: Path) -> str | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return f"{manifest_path} (missing manifest.json)"
    try:
        ValidatedNameCountsManifest.load(path, context="name-count index")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return str(exc)
    return None


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


def _validate_batch_indexes(
    paths: Mapping[str, str],
    generation_files: Mapping[str, _ArrowArtifactGenerationFile],
    *,
    validate_source_fingerprint: bool,
) -> None:
    """Strictly validate paired files with one full pass per immutable artifact."""

    # Local import avoids a module cycle: feature_block_arrow consumes the
    # canonical path helpers defined in this module.
    from s2and.incremental_linking.feature_block_arrow import _validate_arrow_batch_lookup_index

    for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS:
        if contract.table_key not in paths:
            continue
        arrow_file = generation_files[contract.table_key]
        index_file = generation_files[contract.index_key]
        _validate_arrow_batch_lookup_index(
            arrow_path=Path(paths[contract.table_key]),
            index_path=Path(paths[contract.index_key]),
            key_column=contract.key_column,
            expected_arrow_byte_count=arrow_file.byte_count,
            expected_arrow_sha256=arrow_file.sha256,
            expected_index_byte_count=index_file.byte_count,
            expected_index_sha256=index_file.sha256,
            validate_source_fingerprint=validate_source_fingerprint,
        )


def _runtime_arrow_generation_cache_key(
    paths: Mapping[str, str],
    generation_id: str,
) -> _RuntimeArrowGenerationCacheKey:
    """Return the identity of one immutable runtime bundle projection."""

    return generation_id, tuple(sorted((str(key), str(value)) for key, value in paths.items()))


def _open_validated_arrow_generation(
    paths: Mapping[str, str],
    verified: _VerifiedArrowArtifactGeneration,
    *,
    strict_integrity: bool,
    required_keys: Sequence[str],
    context: str,
    producer_hint: str,
    retained_name_counts: ValidatedArrowInputs | None = None,
) -> ValidatedArrowInputs:
    """Open retained native state and validate material at one generation boundary."""

    name_counts_manifest: ValidatedNameCountsManifest | None = None
    name_counts_index: Any | None = None
    if "name_counts_index" in paths:
        index_path = Path(paths["name_counts_index"])
        try:
            generation_file = verified.files.get("name_counts_index")
            if generation_file is None:
                raise ValueError("Arrow artifact generation is missing name_counts_index material facts")
            if retained_name_counts is not None:
                retained_manifest = retained_name_counts.name_counts_manifest
                retained_index = retained_name_counts._name_counts_index
                if retained_manifest is None or retained_index is None:
                    raise ValueError("retained Arrow generation does not contain a validated name-count index")
                if retained_manifest.index_dir != index_path.resolve():
                    raise ValueError(
                        "retained name-count index path mismatch: "
                        f"retained={retained_manifest.index_dir} requested={index_path.resolve()}"
                    )
                name_counts_manifest = retained_manifest
                name_counts_index = retained_index
            else:
                # Local import avoids the name_counts_index -> arrow_inputs
                # error-reporting cycle.
                from s2and.name_counts_index import NameCountsIndex

                name_counts_index, name_counts_manifest = NameCountsIndex._open_with_manifest(
                    index_path,
                    context=f"{context} name_counts_index",
                )
            if name_counts_manifest.manifest_sha256 != generation_file.sha256:
                raise ValueError("opened name-count manifest does not match the Arrow artifact generation")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise MissingArrowArtifactError(
                context=context,
                required_keys=required_keys,
                missing_keys=(),
                missing_files={"name_counts_index": str(exc)},
                producer_hint=producer_hint,
            ) from exc

    normalization_version = verified.normalization_version
    if normalization_version is None:  # pragma: no cover - manifest validation rejects this
        raise RuntimeError("validated Arrow artifact generation is missing normalization_version")
    if name_counts_manifest is not None and name_counts_manifest.normalization_version != normalization_version:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required_keys,
            missing_keys=(),
            missing_files={
                "name_counts_index": (
                    "normalization_version mismatch: artifact is "
                    f"{name_counts_manifest.normalization_version!r} but the Arrow generation "
                    f"requires {normalization_version!r}"
                )
            },
            producer_hint=producer_hint,
        )
    _validate_batch_indexes(
        paths,
        verified.files,
        validate_source_fingerprint=strict_integrity,
    )
    return ValidatedArrowInputs._from_verified(
        paths=paths,
        generation_id=verified.generation_id,
        normalization_version=normalization_version,
        name_counts_manifest=name_counts_manifest,
        name_counts_index=name_counts_index,
    )


def _open_runtime_arrow_generation(
    paths: Mapping[str, str],
    verified: _VerifiedArrowArtifactGeneration,
    *,
    required_keys: Sequence[str],
    context: str,
    producer_hint: str,
) -> ValidatedArrowInputs:
    """Open and cache one runtime generation."""

    cache_key = _runtime_arrow_generation_cache_key(paths, verified.generation_id)
    with _RUNTIME_ARROW_GENERATION_CACHE_LOCK:
        cached = _RUNTIME_ARROW_GENERATION_CACHE.get(cache_key)
        if cached is not None:
            _RUNTIME_ARROW_GENERATION_CACHE.move_to_end(cache_key)
            return cached
        key_lock = _RUNTIME_ARROW_GENERATION_KEY_LOCKS.setdefault(cache_key, threading.Lock())

    with key_lock:
        with _RUNTIME_ARROW_GENERATION_CACHE_LOCK:
            cached = _RUNTIME_ARROW_GENERATION_CACHE.get(cache_key)
            if cached is not None:
                _RUNTIME_ARROW_GENERATION_CACHE.move_to_end(cache_key)
                return cached

        opened = _open_validated_arrow_generation(
            paths,
            verified,
            strict_integrity=False,
            required_keys=required_keys,
            context=context,
            producer_hint=producer_hint,
        )
        with _RUNTIME_ARROW_GENERATION_CACHE_LOCK:
            _RUNTIME_ARROW_GENERATION_CACHE[cache_key] = opened
            while len(_RUNTIME_ARROW_GENERATION_CACHE) > _RUNTIME_ARROW_GENERATION_CACHE_SIZE:
                _RUNTIME_ARROW_GENERATION_CACHE.popitem(last=False)
            return opened


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
    strict_integrity: bool = False,
    context: str,
    producer_hint: str,
    retained_name_counts: ValidatedArrowInputs | None = None,
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

    if isinstance(arrow_paths, ValidatedArrowInputs) and not strict_integrity:
        profiled_paths = arrow_paths
        if not require_specter and _SPECTER_PATH_KEYS.intersection(arrow_paths):
            profiled_paths = arrow_paths.without(*_SPECTER_PATH_KEYS)

        invalid_profile_keys = sorted(
            (UNSUPPORTED_ARROW_NAME_ALIAS_KEYS | _UNSUPPORTED_SPECTER_PATH_KEYS).intersection(profiled_paths)
        )
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
    for key in _UNSUPPORTED_SPECTER_PATH_KEYS.intersection(normalized):
        canonical_key = "specter_batch_index" if key.endswith("_batch_index") else "specter"
        invalid_paths[key] = f"unsupported embedding path key; use {canonical_key}"
    if "name_counts" in normalized:
        invalid_paths["name_counts"] = "legacy name_counts Arrow tables are unsupported; use name_counts_index"
    if "name_counts_index_dir" in normalized:
        invalid_paths["name_counts_index_dir"] = (
            "legacy name-count index aliases are unsupported; use name_counts_index"
        )

    if not require_specter:
        normalized.pop("specter", None)
        normalized.pop("specter_batch_index", None)
        invalid_paths.pop("specter", None)
        invalid_paths.pop("specter_batch_index", None)

    for key in required_filtered_read_batch_index_keys(normalized):
        required.add(key)
        if key not in normalized:
            missing_keys.append(key)

    required_or_declared_keys = {
        key
        for key in normalized
        if key in required or key.endswith("_batch_index") or key in _ARROW_REQUEST_SIDECAR_KEYS
    }
    missing_files = _missing_or_wrong_kind_artifacts(
        normalized,
        required_or_declared_keys.difference({"name_counts_index"}),
    )
    if "name_counts_index" in normalized:
        index_path = Path(normalized["name_counts_index"])
        if not index_path.exists():
            missing_files["name_counts_index"] = str(index_path)
        elif not index_path.is_dir():
            missing_files["name_counts_index"] = f"{index_path} (expected directory)"
        elif not (index_path / "manifest.json").is_file():
            missing_files["name_counts_index"] = f"{index_path / 'manifest.json'} (missing manifest.json)"
    missing_files.update(invalid_paths)
    if missing_keys or missing_files:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=sorted(required),
            missing_keys=sorted(set(missing_keys)),
            missing_files=missing_files,
            producer_hint=producer_hint,
        )
    request_sidecars = {key: normalized[key] for key in _ARROW_REQUEST_SIDECAR_KEYS if key in normalized}
    generation_paths = {key: value for key, value in normalized.items() if key not in _ARROW_REQUEST_SIDECAR_KEYS}
    verified = _validate_arrow_bundle_manifest(
        generation_paths,
        expected_normalization_version=expected_normalization_version,
        context=context,
        required_keys=sorted(required),
        producer_hint=producer_hint,
        defer_batch_material=True,
    )
    sorted_required = sorted(required)
    if strict_integrity:
        validated_generation = _open_validated_arrow_generation(
            generation_paths,
            verified,
            strict_integrity=True,
            required_keys=sorted_required,
            context=context,
            producer_hint=producer_hint,
            retained_name_counts=retained_name_counts,
        )
    else:
        if retained_name_counts is not None:
            raise ValueError("retained_name_counts is only supported for strict publication validation")
        validated_generation = _open_runtime_arrow_generation(
            generation_paths,
            verified,
            required_keys=sorted_required,
            context=context,
            producer_hint=producer_hint,
        )

    required_sidecars = set(str(key) for key in required_request_sidecars)
    if require_cluster_seeds:
        required_sidecars.add("cluster_seeds")
    return validated_generation.with_request_sidecars(
        request_sidecars,
        required_keys=sorted(required_sidecars),
        context=context,
        producer_hint=producer_hint,
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

    return _validate_arrow_publication_artifacts_with_retained_name_counts(
        arrow_paths,
        require_specter=require_specter,
        require_name_counts_index=require_name_counts_index,
        expected_normalization_version=expected_normalization_version,
        context=context,
        producer_hint=producer_hint,
        retained_name_counts=None,
    )


def _validate_arrow_publication_artifacts_with_retained_name_counts(
    arrow_paths: Mapping[str, Any],
    *,
    require_specter: bool,
    require_name_counts_index: bool,
    expected_normalization_version: str | None = None,
    context: str = "Arrow publication integrity",
    producer_hint: str = "publish a complete immutable Arrow bundle with all batch indexes",
    retained_name_counts: ValidatedArrowInputs | None,
) -> ValidatedArrowInputs:
    """Validate publication material while reusing an exact retained name-count generation."""

    return _validate_complete_arrow_artifacts(
        arrow_paths,
        require_specter=require_specter,
        require_name_counts_index=require_name_counts_index,
        expected_normalization_version=expected_normalization_version,
        strict_integrity=True,
        context=context,
        producer_hint=producer_hint,
        retained_name_counts=retained_name_counts,
    )

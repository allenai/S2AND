"""Canonical Arrow artifact manifests and the owning runtime dataset handle."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any, BinaryIO

from s2and._sha256 import is_lowercase_sha256
from s2and.arrow_schema import validate_arrow_schema
from s2and.consts import NORMALIZATION_VERSION
from s2and.name_counts_manifest import ValidatedNameCountsManifest

INFERENCE_ARROW_BUNDLE_SCHEMA_VERSION = "inference_arrow_bundle_v1"
ARROW_ARTIFACT_GENERATION_SCHEMA_VERSION = "s2and_arrow_artifact_generation_v1"


@dataclass(frozen=True)
class ArrowBatchIndexContract:
    """One Arrow table and its raw-planner batch lookup index contract."""

    table_key: str
    key_column: str
    index_key: str
    max_record_batch_rows: int


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
_ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS = frozenset(
    {
        "normalization_version",
        "paths",
        "artifact_generation",
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
_ARROW_IMMUTABLE_KEYS = _ARROW_IMMUTABLE_ARTIFACT_FILE_KEYS | _ARROW_IMMUTABLE_BATCH_INDEX_KEYS
_REQUIRED_RUNTIME_KEYS = frozenset(
    {
        "signatures",
        "signatures_batch_index",
        "papers",
        "papers_batch_index",
        "paper_authors",
        "paper_authors_batch_index",
    }
)
_NATIVE_HANDLE_KEYS = _REQUIRED_RUNTIME_KEYS | frozenset({"specter", "specter_batch_index"})


@dataclass(frozen=True)
class _ArtifactSpec:
    path: Path
    content_path: Path
    byte_count: int
    sha256: str


@dataclass
class _RetainedFile:
    path: Path
    handle: BinaryIO
    lock: threading.Lock = field(default_factory=threading.Lock)


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
    files: dict[str, dict[str, Any]] = {}
    for key, raw_path in sorted((str(key), value) for key, value in paths.items()):
        if key not in _ARROW_IMMUTABLE_KEYS:
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
    """Build one canonical, portable Arrow artifact manifest."""

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


class MissingArrowArtifactError(ValueError):
    """Raised when a strict Arrow route is missing required artifacts."""

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
            details.append(f"missing keys: {', '.join(self.missing_keys)}")
        if self.missing_files:
            formatted = "; ".join(f"{key}={value}" for key, value in sorted(self.missing_files.items()))
            details.append(f"missing files: {formatted}")
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
    """Resolve path values and reject missing values."""

    normalized, invalid = _normalize_arrow_path_values(paths, omit_none=omit_none)
    if invalid:
        key, reason = next(iter(invalid.items()))
        labels = {"<None>": "is None", "<empty>": "is empty", ".": "resolves to the current directory"}
        raise ValueError(f"Arrow path for {key!r} {labels.get(reason, f'is invalid: {reason}')}")
    return normalized


def require_normalization_version(value: Any, *, context: str) -> str:
    """Return the explicitly declared canonical normalization version."""

    if value != NORMALIZATION_VERSION:
        raise ValueError(f"{context} requires normalization_version={NORMALIZATION_VERSION!r}, got {value!r}")
    return NORMALIZATION_VERSION


def require_feature_contract_normalization_version(owner: Any, *, context: str) -> str:
    """Return the explicit normalization version from an owning feature contract."""

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
    """Require a manifest-backed name-count index directory."""

    path_text = os.fspath(path) if isinstance(path, os.PathLike) else str(path)
    index_path = Path(path_text)
    error: str | None = None
    if not path_text.strip() or path_text == ".":
        error = path_text or "<empty>"
    elif not index_path.is_dir():
        error = f"{index_path} (expected directory)"
    else:
        try:
            ValidatedNameCountsManifest.load(index_path)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            error = str(exc)
    if error is not None:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=("name_counts_index",),
            missing_keys=(),
            missing_files={"name_counts_index": error},
            producer_hint=producer_hint,
        )
    return path_text


def _resolve_manifest_artifact_path(root: Path, raw_path: str, *, key: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError(f"Arrow artifact generation files.{key}.path must be manifest-relative: {raw_path!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        if relative != _shared_name_counts_relative_path(resolved, root, artifact_key=key):
            raise ValueError(f"Arrow artifact generation files.{key}.path escapes the manifest directory") from exc
    return resolved


def _load_artifact_specs(root: Path) -> tuple[str, str, Mapping[str, _ArtifactSpec]]:
    manifest_path = root / "manifest.json"
    try:
        with manifest_path.open("rb") as source:
            manifest = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Arrow artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Arrow artifact manifest must be a JSON object: {manifest_path}")
    normalization_version = require_normalization_version(
        manifest.get("normalization_version"),
        context=f"Arrow artifact manifest {manifest_path}",
    )
    paths = manifest.get("paths")
    generation = manifest.get("artifact_generation")
    if not isinstance(paths, Mapping):
        raise ValueError(f"Arrow artifact manifest is missing paths: {manifest_path}")
    if not isinstance(generation, Mapping):
        raise ValueError(f"Arrow artifact manifest is missing artifact_generation: {manifest_path}")
    if generation.get("schema_version") != ARROW_ARTIFACT_GENERATION_SCHEMA_VERSION:
        raise ValueError(f"Arrow artifact manifest has unsupported artifact_generation schema: {manifest_path}")
    files = generation.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"Arrow artifact manifest artifact_generation is missing files: {manifest_path}")
    invalid_keys = sorted(str(key) for key in files if str(key) not in _ARROW_IMMUTABLE_KEYS)
    if invalid_keys:
        raise ValueError(f"Arrow artifact generation contains unsupported immutable keys: {invalid_keys}")
    missing_base = sorted(_REQUIRED_RUNTIME_KEYS.difference(str(key) for key in files))
    if missing_base:
        raise MissingArrowArtifactError(
            context="ArrowDataset.open",
            required_keys=sorted(_REQUIRED_RUNTIME_KEYS),
            missing_keys=missing_base,
            missing_files={},
            producer_hint="generate a complete Arrow bundle with scripts/convert_to_arrow.py",
        )
    file_keys = {str(key) for key in files}
    if ("specter" in file_keys) != ("specter_batch_index" in file_keys):
        raise ValueError("Arrow artifact generation must contain both specter and specter_batch_index or neither")
    declared_generation_id = generation.get("generation_id")
    computed_generation_id = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if declared_generation_id != computed_generation_id:
        raise ValueError(f"Arrow artifact manifest generation_id mismatch: {manifest_path}")

    specs: dict[str, _ArtifactSpec] = {}
    for raw_key, raw_entry in files.items():
        key = str(raw_key)
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Arrow artifact generation files.{key} must be an object")
        raw_path = raw_entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Arrow artifact generation files.{key}.path is invalid")
        if paths.get(key) != raw_path:
            raise ValueError(f"Arrow artifact generation files.{key}.path does not match manifest paths")
        path = _resolve_manifest_artifact_path(root, raw_path, key=key)
        kind = raw_entry.get("kind")
        expected_kind = "directory_manifest" if key == "name_counts_index" else "file"
        if kind != expected_kind:
            raise ValueError(f"Arrow artifact generation files.{key}.kind must be {expected_kind!r}")
        byte_count = raw_entry.get("byte_count")
        sha256 = raw_entry.get("sha256")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"Arrow artifact generation files.{key}.byte_count is invalid")
        if not is_lowercase_sha256(sha256):
            raise ValueError(f"Arrow artifact generation files.{key}.sha256 is invalid")
        specs[key] = _ArtifactSpec(
            path=path,
            content_path=path / "manifest.json" if key == "name_counts_index" else path,
            byte_count=byte_count,
            sha256=sha256,
        )
    return computed_generation_id, normalization_version, MappingProxyType(specs)


def _hash_retained_file(retained: _RetainedFile) -> tuple[int, str]:
    with retained.lock:
        before = os.fstat(retained.handle.fileno())
        retained.handle.seek(0)
        digest = hashlib.sha256()
        while chunk := retained.handle.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(retained.handle.fileno())
        retained.handle.seek(0)
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError(f"Arrow artifact changed while opening: {retained.path}")
    return int(after.st_size), digest.hexdigest()


def _open_retained_path(path: Path) -> BinaryIO:
    """Open a file while permitting atomic path replacement on Windows."""

    if os.name != "nt":
        return path.open("rb", buffering=0)

    import ctypes
    import msvcrt
    from ctypes import wintypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x80000000,  # GENERIC_READ
        0x00000001 | 0x00000002 | 0x00000004,  # FILE_SHARE_READ | WRITE | DELETE
        None,
        3,  # OPEN_EXISTING
        0x00000080,  # FILE_ATTRIBUTE_NORMAL
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error(), filename=str(path))
    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except Exception:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
        raise
    return os.fdopen(descriptor, "rb", buffering=0)


def _validate_retained_arrow_schema(retained: _RetainedFile, *, table_name: str) -> None:
    import mmap

    import pyarrow as pa

    with retained.lock:
        source_map = mmap.mmap(retained.handle.fileno(), 0, access=mmap.ACCESS_READ)
        source = pa.BufferReader(source_map)
        reader = None
        error: tuple[type[Exception], str] | None = None
        try:
            reader = pa.ipc.open_file(source)
            validate_arrow_schema(reader.schema, table_name=table_name)
        except Exception as exc:
            error = type(exc), str(exc)
        finally:
            reader = None
            source.close()
            del source
            source_map.close()
            retained.handle.seek(0)
        if error is not None:
            error_type, message = error
            raise error_type(message)


def _raw_os_handle(handle: BinaryIO) -> int:
    if os.name == "nt":
        import msvcrt

        return int(msvcrt.get_osfhandle(handle.fileno()))
    return int(handle.fileno())


def _construct_native_arrow_dataset(
    retained: Mapping[str, _RetainedFile],
    native_name_counts_index: Any | None,
) -> Any:
    from s2and.runtime import load_s2and_rust_extension

    handles = {
        key: (_raw_os_handle(retained[key].handle), str(retained[key].path))
        for key in sorted(_NATIVE_HANDLE_KEYS.intersection(retained))
    }
    return load_s2and_rust_extension()._ArrowDataset(handles, native_name_counts_index)


class ArrowDataset:
    """One validated, owning handle for an immutable Arrow dataset root."""

    _active_uses: int
    _closed: bool
    _files: dict[str, _RetainedFile]
    _generation_id: str
    _name_counts_index: Any | None
    _name_counts_manifest: ValidatedNameCountsManifest | None
    _native: Any | None
    _normalization_version: str
    _root: Path
    _state_lock: Any

    __slots__ = (
        "_active_uses",
        "_closed",
        "_files",
        "_generation_id",
        "_name_counts_index",
        "_name_counts_manifest",
        "_native",
        "_normalization_version",
        "_root",
        "_state_lock",
    )

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("ArrowDataset must be created with ArrowDataset.open(root)")

    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str],
        *,
        require_specter: bool = False,
        require_name_counts_index: bool = False,
        expected_normalization_version: str | None = None,
    ) -> ArrowDataset:
        """Open, validate, and retain one exact immutable Arrow generation."""

        resolved_root = Path(root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"Arrow dataset root is not a directory: {resolved_root}")
        generation_id, normalization_version, specs = _load_artifact_specs(resolved_root)
        keys = set(specs)
        if require_specter and "specter" not in keys:
            raise MissingArrowArtifactError(
                context="ArrowDataset.open",
                required_keys=("specter", "specter_batch_index"),
                missing_keys=("specter", "specter_batch_index"),
                missing_files={},
                producer_hint="generate the dataset with SPECTER embeddings",
            )
        if require_name_counts_index and "name_counts_index" not in keys:
            raise MissingArrowArtifactError(
                context="ArrowDataset.open",
                required_keys=("name_counts_index",),
                missing_keys=("name_counts_index",),
                missing_files={},
                producer_hint="generate and bind a manifest-backed name-count index",
            )
        if expected_normalization_version is not None:
            require_normalization_version(
                expected_normalization_version,
                context="ArrowDataset.open expected feature contract",
            )

        retained: dict[str, _RetainedFile] = {}
        try:
            for key, spec in specs.items():
                retained[key] = _RetainedFile(spec.content_path, _open_retained_path(spec.content_path))
            for key, spec in specs.items():
                byte_count, sha256 = _hash_retained_file(retained[key])
                if byte_count != spec.byte_count:
                    raise ValueError(f"Arrow artifact generation files.{key}.byte_count mismatch: {spec.content_path}")
                if sha256 != spec.sha256:
                    raise ValueError(f"Arrow artifact generation files.{key} checksum mismatch: {spec.content_path}")
            for table_name in ("signatures", "papers", "paper_authors", "specter"):
                if table_name in retained:
                    _validate_retained_arrow_schema(retained[table_name], table_name=table_name)

            name_counts_index: Any | None = None
            name_counts_manifest: ValidatedNameCountsManifest | None = None
            if "name_counts_index" in specs:
                from s2and.name_counts_index import NameCountsIndex

                name_counts_index, name_counts_manifest = NameCountsIndex._open_with_manifest(
                    specs["name_counts_index"].path
                )
                if name_counts_manifest.manifest_sha256 != specs["name_counts_index"].sha256:
                    raise ValueError("opened name-count manifest does not match the Arrow artifact generation")
                if name_counts_manifest.normalization_version != normalization_version:
                    raise ValueError("name-count index normalization_version does not match the Arrow dataset")
            native_name_counts = None if name_counts_index is None else name_counts_index._native
            native = _construct_native_arrow_dataset(retained, native_name_counts)
        except Exception:
            for retained_file in retained.values():
                retained_file.handle.close()
            raise

        dataset = object.__new__(cls)
        dataset._root = resolved_root
        dataset._generation_id = generation_id
        dataset._normalization_version = normalization_version
        dataset._files = retained
        dataset._name_counts_index = name_counts_index
        dataset._name_counts_manifest = name_counts_manifest
        dataset._native = native
        dataset._state_lock = threading.Lock()
        dataset._active_uses = 0
        dataset._closed = False
        return dataset

    @property
    def root(self) -> Path:
        return self._root

    @property
    def generation_id(self) -> str:
        return self._generation_id

    @property
    def normalization_version(self) -> str:
        return self._normalization_version

    @property
    def name_counts_manifest(self) -> ValidatedNameCountsManifest | None:
        return self._name_counts_manifest

    @property
    def closed(self) -> bool:
        with self._state_lock:
            return self._closed

    @property
    def native(self) -> Any:
        with self._state_lock:
            self._ensure_open()
            return self._native

    @property
    def native_name_counts_index(self) -> Any | None:
        with self._state_lock:
            self._ensure_open()
            index = self._name_counts_index
            return None if index is None else index._native

    def has(self, key: str) -> bool:
        with self._state_lock:
            self._ensure_open()
            return str(key) in self._files

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ArrowDataset is closed")

    def _require_profile(
        self,
        *,
        require_specter: bool,
        require_name_counts_index: bool,
        expected_normalization_version: str | None,
    ) -> None:
        missing: list[str] = []
        if require_specter:
            missing.extend(key for key in ("specter", "specter_batch_index") if key not in self._files)
        if require_name_counts_index and "name_counts_index" not in self._files:
            missing.append("name_counts_index")
        if missing:
            raise MissingArrowArtifactError(
                context="ArrowDataset use",
                required_keys=missing,
                missing_keys=missing,
                missing_files={},
                producer_hint="open a complete dataset root for this model",
            )
        if expected_normalization_version is not None and expected_normalization_version != self._normalization_version:
            raise ValueError(
                "ArrowDataset normalization_version mismatch: "
                f"dataset={self._normalization_version!r} expected={expected_normalization_version!r}"
            )

    def use(
        self,
        *,
        require_specter: bool = False,
        require_name_counts_index: bool = False,
        expected_normalization_version: str | None = None,
    ) -> _ArrowDatasetUse:
        """Create one concurrency-safe use lease."""

        with self._state_lock:
            self._ensure_open()
        return _ArrowDatasetUse(
            self,
            require_specter=require_specter,
            require_name_counts_index=require_name_counts_index,
            expected_normalization_version=expected_normalization_version,
        )

    def _acquire(
        self,
        *,
        require_specter: bool,
        require_name_counts_index: bool,
        expected_normalization_version: str | None,
    ) -> None:
        with self._state_lock:
            self._ensure_open()
            self._require_profile(
                require_specter=require_specter,
                require_name_counts_index=require_name_counts_index,
                expected_normalization_version=expected_normalization_version,
            )
            self._active_uses += 1

    def _release(self) -> None:
        with self._state_lock:
            if self._active_uses <= 0:
                raise RuntimeError("ArrowDataset use lease is not active")
            self._active_uses -= 1

    @contextmanager
    def _open_retained_file(self, key: str) -> Generator[BinaryIO, None, None]:
        retained = self._files[str(key)]
        with retained.lock:
            duplicate = os.fdopen(os.dup(retained.handle.fileno()), "rb", closefd=True)
            try:
                duplicate.seek(0)
                yield duplicate
            finally:
                if not duplicate.closed:
                    duplicate.close()
                retained.handle.seek(0)

    def close(self) -> None:
        """Release all retained files; reject closing a dataset in active use."""

        with self._state_lock:
            if self._closed:
                return
            if self._active_uses:
                raise RuntimeError(f"cannot close ArrowDataset with {self._active_uses} active use lease(s)")
            self._closed = True
            self._native = None
            self._name_counts_index = None
            retained = tuple(self._files.values())
            self._files = {}
        for retained_file in retained:
            retained_file.handle.close()

    def __enter__(self) -> ArrowDataset:
        with self._state_lock:
            self._ensure_open()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if not hasattr(self, "_closed"):
            return
        try:
            self.close()
        except Exception:
            pass


class _ArrowDatasetUse:
    """One active-use lease over an :class:`ArrowDataset`."""

    __slots__ = (
        "_dataset",
        "_entered",
        "_expected_normalization_version",
        "_require_name_counts_index",
        "_require_specter",
    )

    def __init__(
        self,
        dataset: ArrowDataset,
        *,
        require_specter: bool,
        require_name_counts_index: bool,
        expected_normalization_version: str | None,
    ) -> None:
        self._dataset = dataset
        self._require_specter = require_specter
        self._require_name_counts_index = require_name_counts_index
        self._expected_normalization_version = expected_normalization_version
        self._entered = False

    def __enter__(self) -> _ArrowDatasetUse:
        if self._entered:
            raise RuntimeError("ArrowDataset use lease cannot be entered twice")
        self._dataset._acquire(
            require_specter=self._require_specter,
            require_name_counts_index=self._require_name_counts_index,
            expected_normalization_version=self._expected_normalization_version,
        )
        self._entered = True
        return self

    def __exit__(self, *_args: Any) -> None:
        if self._entered:
            self._entered = False
            self._dataset._release()

    def _ensure_active(self) -> None:
        if not self._entered:
            raise RuntimeError("ArrowDataset use lease is not active")

    @property
    def native(self) -> Any:
        self._ensure_active()
        return self._dataset.native

    @property
    def native_name_counts_index(self) -> Any | None:
        self._ensure_active()
        return self._dataset.native_name_counts_index

    def has(self, key: str) -> bool:
        self._ensure_active()
        return self._dataset.has(key)

    @contextmanager
    def open_file(self, key: str) -> Generator[BinaryIO, None, None]:
        """Yield a duplicate of the retained file, even if its path was replaced."""

        self._ensure_active()
        with self._dataset._open_retained_file(key) as source:
            yield source

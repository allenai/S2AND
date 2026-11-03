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
from uuid import uuid4

import numpy as np

from s2and._sha256 import is_lowercase_sha256
from s2and._sha256 import sha256_file as _sha256_file
from s2and.arrow_schema import validate_arrow_schema
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.name_counts_index import NameCountsIndex

ARROW_DATASET_KIND = "s2and_arrow_dataset"
ARROW_COLLECTION_KIND = "s2and_arrow_collection"
PUBLIC_DATA_KIND = "s2and_public_data"


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
        "kind",
        "format_version",
        "paths",
        "files",
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
# Path keys that 0.51 read as request inputs, mapped to the prediction argument that replaces each.
_REQUEST_LOCAL_PATH_ARGUMENTS = {
    "cluster_seeds": "cluster_seeds_require (see read_cluster_seeds_arrow)",
    "cluster_seed_disallows": "cluster_seeds_disallow (see read_cluster_seed_disallows_arrow)",
    "altered_cluster_signatures": "altered_cluster_signatures (see read_altered_cluster_signatures_arrow)",
}


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


def _portable_manifest_path(path: PurePath) -> str:
    """Serialize a manifest path with platform-independent separators."""

    return path.as_posix()


def _manifest_relative_path(path_value: Any, manifest_dir: Path, *, artifact_key: str) -> str:
    path = Path(os.fspath(path_value)).resolve()
    root = manifest_dir.resolve()
    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        if artifact_key != "name_counts_index":
            raise ValueError(
                f"Arrow artifact path must remain within manifest directory: path={path} root={root}"
            ) from exc
        try:
            relative_path = Path(os.path.relpath(path, root))
        except ValueError:
            raise ValueError(
                f"Arrow name-count index must be serializable relative to manifest directory: path={path} root={root}"
            ) from exc
    return _portable_manifest_path(relative_path)


def _build_arrow_artifact_files(paths: Mapping[str, Any]) -> dict[str, Any]:
    """Build the canonical content inventory used by Arrow bundle writers."""

    files: dict[str, dict[str, Any]] = {}
    for key, raw_path in sorted((str(key), value) for key, value in paths.items()):
        if key not in _ARROW_IMMUTABLE_KEYS:
            continue
        declared_path = Path(os.fspath(raw_path)).resolve()
        artifact_path = declared_path / "manifest.json" if declared_path.is_dir() else declared_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"cannot inventory Arrow artifact {key}={artifact_path}")
        files[key] = {
            "byte_count": artifact_path.stat().st_size,
            "sha256": _sha256_file(artifact_path),
        }
    return files


def build_arrow_artifact_manifest(
    paths: Mapping[str, Any],
    manifest_dir: str | Path,
) -> dict[str, Any]:
    """Build one canonical, portable Arrow artifact manifest."""

    root = Path(manifest_dir)
    canonical_paths = {str(key): value for key, value in paths.items() if str(key) != "manifest"}
    return {
        "kind": ARROW_DATASET_KIND,
        "format_version": PUBLIC_DATA_FORMAT_VERSION,
        "paths": {
            key: _manifest_relative_path(value, root, artifact_key=key)
            for key, value in sorted(canonical_paths.items())
        },
        "files": _build_arrow_artifact_files(canonical_paths),
    }


def write_arrow_artifact_manifest(
    manifest: Mapping[str, Any],
    manifest_dir: str | Path,
) -> Path:
    """Atomically publish a manifest built by :func:`build_arrow_artifact_manifest`."""

    root = Path(manifest_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if set(manifest) != _ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS:
        raise ValueError(
            "Arrow artifact manifest fields mismatch: "
            f"missing={sorted(_ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS - set(manifest))} "
            f"extra={sorted(set(manifest) - _ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS)}"
        )
    if manifest["kind"] != ARROW_DATASET_KIND:
        raise ValueError(f"Arrow artifact manifest kind must be {ARROW_DATASET_KIND!r}")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != PUBLIC_DATA_FORMAT_VERSION:
        raise ValueError(f"Arrow artifact manifest format_version must be {PUBLIC_DATA_FORMAT_VERSION}")
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
            temporary_path = Path(output.name)
            output.write(encoded)
        temporary_path.replace(manifest_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return manifest_path


def _collection_manifest_bindings(
    root_manifest: Mapping[str, Any],
    manifest_path: Path,
    field: str,
    *,
    ignore_name: str | None = None,
) -> dict[str, Path]:
    raw_bindings = root_manifest.get(field)
    if not isinstance(raw_bindings, Mapping) or not raw_bindings:
        raise ValueError(f"{manifest_path} {field} must be a nonempty object")

    root = manifest_path.parent.resolve()
    bindings: dict[str, Path] = {}
    for raw_name, raw_binding in raw_bindings.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(f"{manifest_path} {field} keys must be nonempty strings")
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != {"path", "sha256"}:
            raise ValueError(f"{manifest_path} {field}.{raw_name} must contain exactly path and sha256")
        raw_path = raw_binding["path"]
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{manifest_path} {field}.{raw_name}.path must be a nonempty string")
        relative_path = Path(raw_path.replace("\\", "/"))
        if relative_path.is_absolute():
            raise ValueError(f"{manifest_path} {field}.{raw_name}.path must be relative to its root")
        child_manifest = (root / relative_path).resolve()
        try:
            child_manifest.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{manifest_path} {field}.{raw_name}.path escapes its root") from exc
        if not child_manifest.is_file():
            raise ValueError(f"{manifest_path} {field}.{raw_name} manifest does not exist: {child_manifest}")

        expected_sha256 = raw_binding["sha256"]
        if not is_lowercase_sha256(expected_sha256):
            raise ValueError(f"{manifest_path} {field}.{raw_name}.sha256 must be a lowercase SHA-256")
        if raw_name != ignore_name:
            observed_sha256 = _sha256_file(child_manifest)
            if observed_sha256 != expected_sha256:
                raise ValueError(
                    f"{manifest_path} {field}.{raw_name}.sha256 mismatch: "
                    f"expected={expected_sha256} observed={observed_sha256}"
                )
        bindings[raw_name] = child_manifest
    return bindings


def read_arrow_collection_root(
    manifest_path: str | Path,
    *,
    ignore_dataset: str | None = None,
) -> tuple[dict[str, Path], dict[str, Path], str | None]:
    """Validate one collection root and return its bound child manifests."""

    resolved_manifest_path = Path(manifest_path).resolve()
    try:
        with resolved_manifest_path.open(encoding="utf-8") as source:
            root_manifest = json.load(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"root manifest is invalid JSON: {resolved_manifest_path}") from exc
    if not isinstance(root_manifest, Mapping):
        raise TypeError(f"root manifest must contain an object: {resolved_manifest_path}")

    kind = root_manifest.get("kind")
    release_version: str | None
    if kind == ARROW_COLLECTION_KIND:
        release_version = None
        expected_fields = {"kind", "format_version", "dataset_manifests"}
    elif kind == PUBLIC_DATA_KIND:
        release_version = root_manifest.get("release_version")
        if not isinstance(release_version, str) or not release_version or release_version.strip() != release_version:
            raise ValueError(f"{resolved_manifest_path} release_version must be a nonempty trimmed string")
        expected_fields = {"kind", "release_version", "format_version", "dataset_manifests"}
    else:
        raise ValueError(f"{resolved_manifest_path} kind must be {ARROW_COLLECTION_KIND!r} or {PUBLIC_DATA_KIND!r}")
    if kind == PUBLIC_DATA_KIND and "replay_bundles" in root_manifest:
        expected_fields.add("replay_bundles")
    if set(root_manifest) != expected_fields:
        raise ValueError(
            f"{resolved_manifest_path} fields mismatch: "
            f"missing={sorted(expected_fields - set(root_manifest))} "
            f"extra={sorted(set(root_manifest) - expected_fields)}"
        )
    format_version = root_manifest["format_version"]
    if type(format_version) is not int or format_version != PUBLIC_DATA_FORMAT_VERSION:
        raise ValueError(f"{resolved_manifest_path} format_version must be {PUBLIC_DATA_FORMAT_VERSION}")

    dataset_manifests = _collection_manifest_bindings(
        root_manifest,
        resolved_manifest_path,
        "dataset_manifests",
        ignore_name=ignore_dataset,
    )
    replay_bundles = (
        _collection_manifest_bindings(root_manifest, resolved_manifest_path, "replay_bundles")
        if kind == PUBLIC_DATA_KIND and "replay_bundles" in root_manifest
        else {}
    )
    return dataset_manifests, replay_bundles, release_version


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


def arrow_string_array(values: Sequence[str]) -> Any:
    """Build an Arrow string array of non-null strings from its UTF-8 bytes and offsets.

    ``pa.array`` imports pandas to check its input type, which adds 0.3 s to a new prediction process.

    Raises:
        ValueError: The strings need more than 2 GiB, the limit of 32-bit string offsets.
    """
    import pyarrow as pa

    # ASCII strings, the usual ids, have one byte per character, so one join
    # and one encode give the buffers that per-string encoding would.
    joined = "".join(values)
    if joined.isascii():
        data = joined.encode("ascii")
        lengths = np.fromiter(map(len, values), dtype=np.int64, count=len(values))
    else:
        encoded = [value.encode("utf-8") for value in values]
        data = b"".join(encoded)
        lengths = np.fromiter(map(len, encoded), dtype=np.int64, count=len(encoded))
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError(f"Arrow string array needs {int(offsets[-1])} bytes, more than 32-bit offsets allow")
    return pa.StringArray.from_buffers(
        len(values),
        pa.py_buffer(offsets.astype(np.int32)),
        pa.py_buffer(data),
    )


def require_name_counts_index_artifact(
    path: Any,
    *,
    context: str,
    producer_hint: str,
    verify: bool = False,
) -> str:
    """Require a manifest-backed name-count index directory.

    Pass ``verify=True`` at publication checks to also check file digests and every record.
    """

    path_text = os.fspath(path) if isinstance(path, os.PathLike) else str(path)
    index_path = Path(path_text)
    error: str | None = None
    if not path_text.strip() or path_text == ".":
        error = path_text or "<empty>"
    elif not index_path.is_dir():
        error = f"{index_path} (expected directory)"
    else:
        try:
            NameCountsIndex.open(index_path, verify=verify)
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
        raise ValueError(f"Arrow artifact manifest paths.{key} must be manifest-relative: {raw_path!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        if key != "name_counts_index":
            raise ValueError(f"Arrow artifact manifest paths.{key} escapes the manifest directory") from exc
    return resolved


def _load_artifact_specs(root: Path) -> tuple[str, Mapping[str, _ArtifactSpec]]:
    manifest_path = root / "manifest.json"
    try:
        with manifest_path.open("rb") as source:
            manifest = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Arrow artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Arrow artifact manifest must be a JSON object: {manifest_path}")
    if set(manifest) != _ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS:
        raise ValueError(
            f"Arrow artifact manifest {manifest_path} fields mismatch: "
            f"missing={sorted(_ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS - set(manifest))} "
            f"extra={sorted(set(manifest) - _ARROW_ARTIFACT_MANIFEST_OWNED_FIELDS)}"
        )
    if manifest.get("kind") != ARROW_DATASET_KIND:
        raise ValueError(
            f"Arrow artifact manifest {manifest_path} requires kind={ARROW_DATASET_KIND!r}, "
            f"got {manifest.get('kind')!r}"
        )
    format_version = manifest.get("format_version")
    if type(format_version) is not int or format_version != PUBLIC_DATA_FORMAT_VERSION:
        raise ValueError(
            f"Arrow artifact manifest {manifest_path} requires "
            f"format_version={PUBLIC_DATA_FORMAT_VERSION}, got {format_version!r}"
        )
    paths = manifest.get("paths")
    files = manifest.get("files")
    if not isinstance(paths, Mapping):
        raise ValueError(f"Arrow artifact manifest is missing paths: {manifest_path}")
    if not isinstance(files, Mapping):
        raise ValueError(f"Arrow artifact manifest is missing files: {manifest_path}")
    invalid_keys = sorted(str(key) for key in files if str(key) not in _ARROW_IMMUTABLE_KEYS)
    if invalid_keys:
        raise ValueError(f"Arrow artifact manifest contains unsupported immutable file keys: {invalid_keys}")
    file_keys = {str(key) for key in files}
    declared_immutable_keys = {str(key) for key in paths if str(key) in _ARROW_IMMUTABLE_KEYS}
    if file_keys != declared_immutable_keys:
        missing = sorted(declared_immutable_keys - file_keys)
        extra = sorted(file_keys - declared_immutable_keys)
        raise ValueError(f"Arrow artifact files and paths immutable key mismatch: missing={missing} extra={extra}")
    missing_base = sorted(_REQUIRED_RUNTIME_KEYS.difference(str(key) for key in files))
    if missing_base:
        raise MissingArrowArtifactError(
            context="ArrowDataset.open",
            required_keys=sorted(_REQUIRED_RUNTIME_KEYS),
            missing_keys=missing_base,
            missing_files={},
            producer_hint="generate a complete Arrow bundle with scripts/convert_to_arrow.py",
        )
    if ("specter" in file_keys) != ("specter_batch_index" in file_keys):
        raise ValueError("Arrow artifact files must contain both specter and specter_batch_index or neither")
    computed_generation_id = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    specs: dict[str, _ArtifactSpec] = {}
    for raw_key, raw_entry in files.items():
        key = str(raw_key)
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Arrow artifact manifest files.{key} must be an object")
        expected_fields = {"byte_count", "sha256"}
        entry_fields = {str(field) for field in raw_entry}
        if entry_fields != expected_fields:
            raise ValueError(
                f"Arrow artifact manifest files.{key} field mismatch: "
                f"missing={sorted(expected_fields - entry_fields)} "
                f"extra={sorted(entry_fields - expected_fields)}"
            )
        raw_path = paths.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Arrow artifact manifest paths.{key} is invalid")
        path = _resolve_manifest_artifact_path(root, raw_path, key=key)
        byte_count = raw_entry.get("byte_count")
        sha256 = raw_entry.get("sha256")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"Arrow artifact manifest files.{key}.byte_count is invalid")
        if not is_lowercase_sha256(sha256):
            raise ValueError(f"Arrow artifact manifest files.{key}.sha256 is invalid")
        specs[key] = _ArtifactSpec(
            path=path,
            content_path=path / "manifest.json" if key == "name_counts_index" else path,
            byte_count=byte_count,
            sha256=sha256,
        )
    return computed_generation_id, MappingProxyType(specs)


def _hash_retained_file(retained: _RetainedFile) -> tuple[int, str]:
    from s2and.runtime import load_s2and_rust_extension

    with retained.lock:
        before = os.fstat(retained.handle.fileno())
        sha256 = load_s2and_rust_extension()._arrow_retained_file_sha256(
            _raw_os_handle(retained.handle), str(retained.path)
        )
        after = os.fstat(retained.handle.fileno())
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError(f"Arrow artifact changed while opening: {retained.path}")
    return int(after.st_size), sha256


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
        error = ctypes.WinError(ctypes.get_last_error())
        error.filename = str(path)
        raise error
    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except Exception:
        ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
        raise
    return os.fdopen(descriptor, "rb", buffering=0)


def _clear_exception_chain_tracebacks(error: BaseException) -> None:
    """Release buffer-owning frames without changing exception-chain identities."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def _validate_retained_arrow_schema(retained: _RetainedFile, *, table_name: str) -> None:
    import mmap

    import pyarrow as pa

    with retained.lock:
        try:
            source_map = mmap.mmap(retained.handle.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError as exc:
            raise ValueError(
                f"Invalid Arrow {table_name} IPC file in {retained.path} ({type(exc).__name__}): {exc}"
            ) from exc
        source = pa.BufferReader(source_map)
        reader = None
        error: Exception | None = None
        try:
            reader = pa.ipc.open_file(source)
            validate_arrow_schema(reader.schema, table_name=table_name)
        except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
            error = ValueError(f"Invalid Arrow {table_name} schema in {retained.path} ({type(exc).__name__}): {exc}")
            _clear_exception_chain_tracebacks(exc)
        except Exception as exc:
            # Traceback frames can retain buffers into the mapping. Release them
            # before closing it, but preserve non-schema errors and their data.
            _clear_exception_chain_tracebacks(exc)
            error = exc
        finally:
            reader = None
            source.close()
            del source
            source_map.close()
            retained.handle.seek(0)
        if error is not None:
            raise error


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
    """An owning handle for immutable Arrow files, verified or caller-trusted."""

    _active_uses: int
    _closed: bool
    _files: dict[str, _RetainedFile]
    _generation_id: str
    _name_counts_index: NameCountsIndex | None
    _native: Any | None
    _root: Path
    _state_lock: Any

    __slots__ = (
        "_active_uses",
        "_closed",
        "_files",
        "_generation_id",
        "_name_counts_index",
        "_native",
        "_root",
        "_state_lock",
    )

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("ArrowDataset must be created with ArrowDataset.open(root) or ArrowDataset.open_trusted(paths)")

    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str],
        *,
        require_specter: bool = False,
        require_name_counts_index: bool = False,
    ) -> ArrowDataset:
        """Open, validate, and retain one exact immutable Arrow generation."""

        resolved_root = Path(root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise FileNotFoundError(f"Arrow dataset root is not a directory: {resolved_root}")
        generation_id, specs = _load_artifact_specs(resolved_root)
        return cls._open_files(
            {key: spec.path for key, spec in specs.items()},
            root=resolved_root,
            generation_id=generation_id,
            specs=specs,
            require_specter=require_specter,
            require_name_counts_index=require_name_counts_index,
        )

    @classmethod
    def open_trusted(
        cls,
        paths: Mapping[str, str | os.PathLike[str]],
        *,
        require_specter: bool = False,
        require_name_counts_index: bool = False,
    ) -> ArrowDataset:
        """Retain caller-owned files without a dataset manifest or checksum scans.

        The caller guarantees that tables and indexes belong together and stay
        immutable until the handle closes. Schemas, index structure, and source
        sizes are still checked. Optional name-count indexes retain their own
        validation. Unrelated path keys are ignored, as in dataset manifests.
        Request-local seed tables raise ``ValueError``; pass their contents as
        prediction arguments instead.
        Each open gets a fresh generation ID, isolating prediction caches.
        """
        # S2AND 0.51 read these tables from the path mapping. Ignoring them here
        # would silently drop seeds and exclusions from a caller ported from 0.51.
        request_local = sorted(key for key in paths if key in _REQUEST_LOCAL_PATH_ARGUMENTS)
        if request_local:
            raise ValueError(
                "ArrowDataset.open_trusted does not read request-local tables; "
                + "; ".join(f"for {key!r} pass {_REQUEST_LOCAL_PATH_ARGUMENTS[key]}" for key in request_local)
            )
        resolved = {
            key: Path(value)
            for key, value in normalize_arrow_paths(
                {key: value for key, value in paths.items() if key in _ARROW_IMMUTABLE_KEYS}
            ).items()
        }
        missing = sorted(_REQUIRED_RUNTIME_KEYS - resolved.keys())
        if missing:
            raise MissingArrowArtifactError(
                context="ArrowDataset.open_trusted",
                required_keys=sorted(_REQUIRED_RUNTIME_KEYS),
                missing_keys=missing,
                missing_files={},
                producer_hint="provide tables and their batch lookup indexes",
            )
        if ("specter" in resolved) != ("specter_batch_index" in resolved):
            raise ValueError("Arrow paths must contain both specter and specter_batch_index or neither")
        return cls._open_files(
            resolved,
            root=resolved["signatures"].parent,
            generation_id=f"trusted:{uuid4().hex}",
            specs=None,
            require_specter=require_specter,
            require_name_counts_index=require_name_counts_index,
        )

    @classmethod
    def _open_files(
        cls,
        paths: Mapping[str, Path],
        *,
        root: Path,
        generation_id: str,
        specs: Mapping[str, _ArtifactSpec] | None,
        require_specter: bool,
        require_name_counts_index: bool,
    ) -> ArrowDataset:
        keys = set(paths)
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

        retained: dict[str, _RetainedFile] = {}
        try:
            for key, path in paths.items():
                content_path = path / "manifest.json" if key == "name_counts_index" else path
                retained[key] = _RetainedFile(content_path, _open_retained_path(content_path))
            if specs is not None:
                for key, spec in specs.items():
                    byte_count, sha256 = _hash_retained_file(retained[key])
                    if byte_count != spec.byte_count:
                        raise ValueError(
                            f"Arrow artifact manifest files.{key}.byte_count mismatch: {spec.content_path}"
                        )
                    if sha256 != spec.sha256:
                        raise ValueError(f"Arrow artifact manifest files.{key} checksum mismatch: {spec.content_path}")
            for table_name in ("signatures", "papers", "paper_authors", "specter"):
                if table_name in retained:
                    _validate_retained_arrow_schema(retained[table_name], table_name=table_name)

            name_counts_index: NameCountsIndex | None = None
            if "name_counts_index" in paths:
                name_counts_index = NameCountsIndex.open(paths["name_counts_index"])
                if specs is not None and name_counts_index.manifest_sha256 != specs["name_counts_index"].sha256:
                    raise ValueError("opened name-count manifest does not match the Arrow artifact manifest")
            native_name_counts = None if name_counts_index is None else name_counts_index._native
            native = _construct_native_arrow_dataset(retained, native_name_counts)
        except Exception:
            for retained_file in retained.values():
                retained_file.handle.close()
            raise

        dataset = object.__new__(cls)
        dataset._root = root
        dataset._generation_id = generation_id
        dataset._files = retained
        dataset._name_counts_index = name_counts_index
        dataset._native = native
        dataset._state_lock = threading.Lock()
        dataset._active_uses = 0
        dataset._closed = False
        return dataset

    @property
    def root(self) -> Path:
        """Manifest root, or the signatures directory for a trusted open."""
        return self._root

    @property
    def generation_id(self) -> str:
        return self._generation_id

    @property
    def name_counts_index(self) -> NameCountsIndex | None:
        return self._name_counts_index

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

    def use(
        self,
        *,
        require_specter: bool = False,
        require_name_counts_index: bool = False,
    ) -> _ArrowDatasetUse:
        """Create one concurrency-safe use lease."""

        with self._state_lock:
            self._ensure_open()
        return _ArrowDatasetUse(
            self,
            require_specter=require_specter,
            require_name_counts_index=require_name_counts_index,
        )

    def _acquire(
        self,
        *,
        require_specter: bool,
        require_name_counts_index: bool,
    ) -> None:
        with self._state_lock:
            self._ensure_open()
            self._require_profile(
                require_specter=require_specter,
                require_name_counts_index=require_name_counts_index,
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
        "_require_name_counts_index",
        "_require_specter",
    )

    def __init__(
        self,
        dataset: ArrowDataset,
        *,
        require_specter: bool,
        require_name_counts_index: bool,
    ) -> None:
        self._dataset = dataset
        self._require_specter = require_specter
        self._require_name_counts_index = require_name_counts_index
        self._entered = False

    def __enter__(self) -> _ArrowDatasetUse:
        if self._entered:
            raise RuntimeError("ArrowDataset use lease cannot be entered twice")
        self._dataset._acquire(
            require_specter=self._require_specter,
            require_name_counts_index=self._require_name_counts_index,
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

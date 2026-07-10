"""Canonical validation helpers for Arrow-backed runtime inputs."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from s2and.consts import NORMALIZATION_VERSION_LEGACY_COMPAT, VALID_NORMALIZATION_VERSIONS


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
_STRICT_BATCH_INDEX_VALIDATION_CACHE_MAX_ENTRIES = 128
_STRICT_BATCH_INDEX_VALIDATION_CACHE: OrderedDict[tuple[Any, ...], None] = OrderedDict()
_STRICT_BATCH_INDEX_VALIDATION_CACHE_LOCK = threading.Lock()
_NAME_COUNTS_INDEX_VALIDATION_CACHE: OrderedDict[tuple[Any, ...], None] = OrderedDict()
_NAME_COUNTS_INDEX_VALIDATION_CACHE_LOCK = threading.Lock()
_ArrowArtifactMaterialBinding = tuple[tuple[str, str], ...]
_ArrowArtifactGenerationCacheKey = tuple[str, _ArrowArtifactMaterialBinding]
_ARROW_ARTIFACT_GENERATION_CACHE: OrderedDict[
    _ArrowArtifactGenerationCacheKey,
    tuple[str, Any],
] = OrderedDict()
_ARROW_ARTIFACT_GENERATION_CACHE_LOCK = threading.Lock()
_WINDOWS_CHANGE_HANDLE_CACHE: OrderedDict[str, int] = OrderedDict()
_WINDOWS_CHANGE_HANDLE_CACHE_LOCK = threading.Lock()
_WINDOWS_CHANGE_HANDLE_CACHE_MAX_ENTRIES = 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_relative_path(path_value: Any, manifest_dir: Path) -> str:
    path = Path(os.fspath(path_value))
    try:
        return os.path.relpath(str(path.resolve()), str(manifest_dir.resolve()))
    except ValueError:
        return str(path)


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
            "path": _manifest_relative_path(declared_path, root),
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


@lru_cache(maxsize=1)
def _windows_change_token_api() -> tuple[Any, Any, Any, Any, Any, int]:
    import ctypes
    from ctypes import wintypes

    class _FileBasicInfo(ctypes.Structure):
        _fields_ = (
            ("creation_time", ctypes.c_longlong),
            ("last_access_time", ctypes.c_longlong),
            ("last_write_time", ctypes.c_longlong),
            ("change_time", ctypes.c_longlong),
            ("file_attributes", wintypes.DWORD),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
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
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = (wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD)
    get_info.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    invalid_handle = wintypes.HANDLE(-1).value
    if invalid_handle is None:  # pragma: no cover - Windows HANDLE invariant
        raise RuntimeError("Windows INVALID_HANDLE_VALUE is unavailable")
    return ctypes, _FileBasicInfo, create_file, get_info, close_handle, int(invalid_handle)


def _file_change_token(path: Path) -> int:
    """Return an OS change timestamp that cannot be restored with ``utime``."""

    if os.name != "nt":
        return int(path.lstat().st_ctime_ns)

    ctypes, file_basic_info, create_file, get_info, close_handle, invalid_handle = _windows_change_token_api()
    cache_key = os.path.abspath(os.fspath(path))
    with _WINDOWS_CHANGE_HANDLE_CACHE_LOCK:
        handle = _WINDOWS_CHANGE_HANDLE_CACHE.get(cache_key)
        if handle is None:
            handle = create_file(
                cache_key,
                0x0080,  # FILE_READ_ATTRIBUTES
                0x0001 | 0x0002 | 0x0004,  # share read/write/delete
                None,
                3,  # OPEN_EXISTING
                0x02000000 if path.is_dir() else 0,  # FILE_FLAG_BACKUP_SEMANTICS
                None,
            )
            if handle == invalid_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            _WINDOWS_CHANGE_HANDLE_CACHE[cache_key] = int(handle)
        else:
            _WINDOWS_CHANGE_HANDLE_CACHE.move_to_end(cache_key)
        while len(_WINDOWS_CHANGE_HANDLE_CACHE) > _WINDOWS_CHANGE_HANDLE_CACHE_MAX_ENTRIES:
            _discarded_path, discarded_handle = _WINDOWS_CHANGE_HANDLE_CACHE.popitem(last=False)
            close_handle(discarded_handle)
        info = file_basic_info()
        if not get_info(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.change_time)


def _guard_tokens_unchanged(tokens: tuple[tuple[str, int], ...]) -> bool:
    if os.name != "nt":
        return all(Path(path).lstat().st_ctime_ns == expected for path, expected in tokens)

    ctypes, file_basic_info, _create_file, get_info, _close_handle, _invalid_handle = _windows_change_token_api()
    with _WINDOWS_CHANGE_HANDLE_CACHE_LOCK:
        for path, expected in tokens:
            handle = _WINDOWS_CHANGE_HANDLE_CACHE.get(os.path.abspath(path))
            if handle is None:
                return False
            info = file_basic_info()
            if not get_info(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
                return False
            if int(info.change_time) != expected:
                return False
    return True


class _PollingArtifactWatch:
    def __init__(self, paths: Iterable[Path]) -> None:
        resolved_paths: set[Path] = set()
        for path in paths:
            resolved_paths.add(path.absolute())
        self._tokens = tuple(
            (str(path), _file_change_token(path)) for path in sorted(resolved_paths, key=lambda value: str(value))
        )

    def changed(self) -> bool:
        return not _guard_tokens_unchanged(self._tokens)

    def close(self) -> None:
        return None


class _InotifyArtifactWatch:
    _MASK = 0x00000002 | 0x00000004 | 0x00000008 | 0x00000080 | 0x00000100 | 0x00000200 | 0x00000400

    def __init__(self, directories: Iterable[Path]) -> None:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        nonblocking = cast(int, vars(os)["O_NONBLOCK"])
        close_on_exec = cast(int, vars(os)["O_CLOEXEC"])
        self._fd = int(libc.inotify_init1(nonblocking | close_on_exec))
        if self._fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1 failed")
        try:
            for directory in sorted({path.resolve() for path in directories}, key=str):
                watch_descriptor = int(
                    libc.inotify_add_watch(
                        self._fd,
                        os.fsencode(str(directory)),
                        self._MASK,
                    )
                )
                if watch_descriptor < 0:
                    raise OSError(ctypes.get_errno(), f"inotify_add_watch failed: {directory}")
        except Exception:
            os.close(self._fd)
            self._fd = -1
            raise

    def changed(self) -> bool:
        try:
            return bool(os.read(self._fd, 65_536))
        except BlockingIOError:
            return False

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class _WindowsArtifactWatch:
    _NOTIFY_FILTER = 0x00000001 | 0x00000002 | 0x00000004 | 0x00000008 | 0x00000010 | 0x00000040

    def __init__(self, directories: Iterable[Path]) -> None:
        import ctypes
        from ctypes import wintypes

        class _Overlapped(ctypes.Structure):
            _fields_ = (
                ("internal", ctypes.c_void_p),
                ("internal_high", ctypes.c_void_p),
                ("offset", wintypes.DWORD),
                ("offset_high", wintypes.DWORD),
                ("event", wintypes.HANDLE),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ctypes = ctypes
        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = (wintypes.HANDLE,)
        self._close_handle.restype = wintypes.BOOL
        self._cancel_io = kernel32.CancelIoEx
        self._cancel_io.argtypes = (wintypes.HANDLE, wintypes.LPVOID)
        self._cancel_io.restype = wintypes.BOOL
        self._wait = kernel32.WaitForSingleObject
        self._wait.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        self._wait.restype = wintypes.DWORD
        create_file = kernel32.CreateFileW
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
        create_event = kernel32.CreateEventW
        create_event.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR)
        create_event.restype = wintypes.HANDLE
        read_changes = kernel32.ReadDirectoryChangesW
        read_changes.argtypes = (
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.LPVOID,
        )
        read_changes.restype = wintypes.BOOL
        invalid_handle_value = wintypes.HANDLE(-1).value
        if invalid_handle_value is None:  # pragma: no cover - Windows HANDLE invariant
            raise RuntimeError("Windows INVALID_HANDLE_VALUE is unavailable")
        invalid_handle = int(invalid_handle_value)
        self._states: list[tuple[int, int, Any, Any]] = []
        try:
            for directory in sorted({path.resolve() for path in directories}, key=str):
                handle = create_file(
                    str(directory),
                    0x0001,  # FILE_LIST_DIRECTORY
                    0x0001 | 0x0002 | 0x0004,
                    None,
                    3,
                    0x02000000 | 0x40000000,  # backup semantics + overlapped
                    None,
                )
                if handle == invalid_handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                event = create_event(None, True, False, None)
                if not event:
                    self._close_handle(handle)
                    raise ctypes.WinError(ctypes.get_last_error())
                buffer = ctypes.create_string_buffer(8192)
                overlapped = _Overlapped(event=event)
                if not read_changes(
                    handle,
                    buffer,
                    len(buffer),
                    True,
                    self._NOTIFY_FILTER,
                    None,
                    ctypes.byref(overlapped),
                    None,
                ):
                    error = ctypes.get_last_error()
                    if error != 997:  # ERROR_IO_PENDING is the expected overlapped result.
                        self._close_handle(event)
                        self._close_handle(handle)
                        raise ctypes.WinError(error)
                self._states.append((int(handle), int(event), buffer, overlapped))
        except Exception:
            self.close()
            raise

    def changed(self) -> bool:
        return any(self._wait(event, 0) == 0 for _handle, event, _buffer, _overlapped in self._states)

    def close(self) -> None:
        for handle, event, _buffer, overlapped in getattr(self, "_states", ()):  # pragma: no branch
            self._cancel_io(handle, self._ctypes.byref(overlapped))
            self._close_handle(event)
            self._close_handle(handle)
        self._states = []


def _artifact_change_watch(
    directories: Iterable[Path],
    *,
    files: Iterable[Path] = (),
) -> Any:
    if os.name == "nt":
        return _WindowsArtifactWatch(directories)
    if sys.platform.startswith("linux"):
        return _InotifyArtifactWatch(directories)
    return _PollingArtifactWatch((*directories, *files))


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
    require_canonical_manifest: bool,
    context: str,
    required_keys: Sequence[str],
    producer_hint: str,
) -> None:
    manifest_path = _arrow_artifact_manifest_path(paths)
    if manifest_path is None or not manifest_path.is_file():
        if require_canonical_manifest:
            raise MissingArrowArtifactError(
                context=context,
                required_keys=required_keys,
                missing_keys=("manifest",),
                missing_files={"manifest": str(manifest_path or "<unresolved>")},
                producer_hint=producer_hint,
            )
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Arrow artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Arrow artifact manifest must be a JSON object: {manifest_path}")
    normalization_version = manifest.get("normalization_version")
    if normalization_version is not None:
        if normalization_version not in VALID_NORMALIZATION_VERSIONS:
            raise ValueError(f"Arrow artifact manifest normalization_version is invalid: {normalization_version!r}")
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
    elif require_canonical_manifest:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required_keys,
            missing_keys=("manifest.normalization_version",),
            missing_files={},
            producer_hint=producer_hint,
        )
    if require_canonical_manifest and verified_arrow_artifact_generation(paths) is None:
        raise MissingArrowArtifactError(
            context=context,
            required_keys=required_keys,
            missing_keys=("manifest.artifact_generation",),
            missing_files={},
            producer_hint=producer_hint,
        )


def _arrow_artifact_material_binding(paths: Mapping[str, str]) -> _ArrowArtifactMaterialBinding:
    """Return the exact supplied material-key/path set authorized by one cache entry."""

    return tuple(
        sorted(
            (
                str(key),
                str(os.fspath(paths[key])),
            )
            for key in paths
            if key in _ARROW_IMMUTABLE_ARTIFACT_FILE_KEYS or key in _ARROW_IMMUTABLE_BATCH_INDEX_KEYS
        )
    )


def verified_arrow_artifact_generation(paths: Mapping[str, str]) -> str | None:
    """Verify and return an immutable content-addressed Arrow generation.

    Legacy path bundles without an artifact inventory return ``None``. Callers
    must then fingerprint material files directly rather than minting a trusted
    generation token.
    """

    manifest_path = _arrow_artifact_manifest_path(paths)
    if manifest_path is None:
        return None
    material_binding = _arrow_artifact_material_binding(paths)
    manifest_cache_key = (str(manifest_path), material_binding)
    stale_watch = None
    with _ARROW_ARTIFACT_GENERATION_CACHE_LOCK:
        cached_guard = _ARROW_ARTIFACT_GENERATION_CACHE.get(manifest_cache_key)
        if cached_guard is not None:
            cached_generation, change_watch = cached_guard
            try:
                guard_unchanged = not change_watch.changed()
            except OSError:
                guard_unchanged = False
            if guard_unchanged:
                _ARROW_ARTIFACT_GENERATION_CACHE.move_to_end(manifest_cache_key)
                return cached_generation
            _discarded_generation, stale_watch = _ARROW_ARTIFACT_GENERATION_CACHE.pop(manifest_cache_key)
    if stale_watch is not None:
        stale_watch.close()
    if not manifest_path.is_file():
        return None
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Arrow artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Arrow artifact manifest must be a JSON object: {manifest_path}")
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
    guarded_directories = {manifest_path.parent}
    guarded_files = {manifest_path}
    for key in material_keys:
        entry = files.get(key)
        if not isinstance(entry, Mapping):
            raise ValueError(f"Arrow artifact generation is missing files.{key}: {manifest_path}")
        raw_declared_path = entry.get("path")
        if not isinstance(raw_declared_path, str) or not raw_declared_path.strip():
            raise ValueError(f"Arrow artifact generation files.{key}.path is invalid: {manifest_path}")
        supplied_path = Path(os.path.abspath(os.fspath(paths[key])))
        declared_path = Path(raw_declared_path)
        if not declared_path.is_absolute():
            declared_path = manifest_path.parent / declared_path
        declared_path = declared_path.resolve()
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
        guarded_directories.add(supplied_path.parent)
        guarded_directories.add(artifact_path.parent)
        guarded_files.add(supplied_path)
        guarded_files.add(artifact_path)
    change_watch = _artifact_change_watch(guarded_directories, files=guarded_files)
    try:
        for artifact_path, key, expected_bytes, expected_sha256 in verification_items:
            _resolved, observed_bytes, _mtime_ns, observed_sha256 = _stable_file_digest_token(artifact_path)
            if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
                raise ValueError(f"Arrow artifact generation files.{key} checksum mismatch: {artifact_path}")
        if manifest_path.read_bytes() != manifest_bytes or change_watch.changed():
            raise RuntimeError(f"Arrow artifact generation changed during verification: {manifest_path}")
    except Exception:
        change_watch.close()
        raise
    with _ARROW_ARTIFACT_GENERATION_CACHE_LOCK:
        previous = _ARROW_ARTIFACT_GENERATION_CACHE.pop(manifest_cache_key, None)
        if previous is not None:
            previous[1].close()
        _ARROW_ARTIFACT_GENERATION_CACHE[manifest_cache_key] = (
            computed_generation_id,
            change_watch,
        )
        _ARROW_ARTIFACT_GENERATION_CACHE.move_to_end(manifest_cache_key)
        while len(_ARROW_ARTIFACT_GENERATION_CACHE) > _STRICT_BATCH_INDEX_VALIDATION_CACHE_MAX_ENTRIES:
            _discarded_key, (_discarded_generation, discarded_watch) = _ARROW_ARTIFACT_GENERATION_CACHE.popitem(
                last=False
            )
            discarded_watch.close()
    return computed_generation_id


def _name_counts_index_error(path: Path) -> str | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return f"{manifest_path} (missing manifest.json)"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"{manifest_path} (invalid manifest: {exc})"
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    schema_version = manifest.get("schema_version")
    if schema_version != NAME_COUNTS_INDEX_SCHEMA_VERSION:
        return (
            f"{manifest_path} (unsupported schema_version {schema_version!r}; "
            f"expected {NAME_COUNTS_INDEX_SCHEMA_VERSION!r})"
        )
    normalization_version = manifest.get("normalization_version", NORMALIZATION_VERSION_LEGACY_COMPAT)
    if normalization_version not in VALID_NORMALIZATION_VERSIONS:
        return (
            f"{manifest_path} (invalid normalization_version {normalization_version!r}; "
            f"expected one of {sorted(VALID_NORMALIZATION_VERSIONS)})"
        )
    source_provenance = manifest.get("source_provenance")
    strict_integrity = isinstance(source_provenance, Mapping)
    if not strict_integrity and normalization_version != NORMALIZATION_VERSION_LEGACY_COMPAT:
        return f"{manifest_path} (missing source_provenance mapping)"
    if strict_integrity and source_provenance.get("normalization_version") != normalization_version:
        return f"{manifest_path} (source_provenance normalization_version mismatch)"
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        return f"{manifest_path} (missing files mapping)"
    verified_files: list[tuple[Path, str, int, int, int]] = []
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
        if strict_integrity and (not isinstance(byte_count, int) or byte_count < 0):
            return f"{manifest_path} (missing files.{file_key}.byte_count)"
        expected_sha256 = entry.get("sha256")
        if strict_integrity and (not isinstance(expected_sha256, str) or len(expected_sha256) != 64):
            return f"{manifest_path} (missing files.{file_key}.sha256)"
        if strict_integrity and not (resolved.parent / ".published").is_file():
            return f"{resolved.parent / '.published'} (missing published-generation marker)"
        verified_files.append(
            (
                resolved,
                expected_sha256 if isinstance(expected_sha256, str) else "",
                int(file_stat.st_size),
                int(file_stat.st_mtime_ns),
                _file_change_token(resolved),
            )
        )
    verified_parents: set[Path] = set()
    for file_path, *_rest in verified_files:
        verified_parents.add(file_path.parent)
    cache_key = (
        str(manifest_path.resolve()),
        manifest_digest,
        tuple(
            (str(file_path.resolve()), size, mtime_ns, change_token)
            for file_path, _digest, size, mtime_ns, change_token in verified_files
        ),
        tuple(
            (str(parent), _file_change_token(parent))
            for parent in sorted(verified_parents, key=lambda value: str(value))
        ),
    )
    with _NAME_COUNTS_INDEX_VALIDATION_CACHE_LOCK:
        if cache_key in _NAME_COUNTS_INDEX_VALIDATION_CACHE:
            _NAME_COUNTS_INDEX_VALIDATION_CACHE.move_to_end(cache_key)
            return None
    for resolved, expected_sha256, _size, _mtime_ns, _change_token in verified_files:
        if not strict_integrity:
            continue
        if _sha256_file(resolved) != expected_sha256:
            return f"{resolved} (declared SHA-256 mismatch)"
    with _NAME_COUNTS_INDEX_VALIDATION_CACHE_LOCK:
        _NAME_COUNTS_INDEX_VALIDATION_CACHE[cache_key] = None
        _NAME_COUNTS_INDEX_VALIDATION_CACHE.move_to_end(cache_key)
        while len(_NAME_COUNTS_INDEX_VALIDATION_CACHE) > _STRICT_BATCH_INDEX_VALIDATION_CACHE_MAX_ENTRIES:
            _NAME_COUNTS_INDEX_VALIDATION_CACHE.popitem(last=False)
    return None


def read_name_counts_index_normalization_version(path: Any) -> str:
    """Read the normalization_version recorded in a name_counts_index/ manifest.

    An absent field means the artifact predates the normalization contract and is
    treated as "legacy_compat". Invalid tokens raise ValueError.
    """

    manifest_path = Path(os.fspath(path)) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = manifest.get("normalization_version", NORMALIZATION_VERSION_LEGACY_COMPAT)
    if value not in VALID_NORMALIZATION_VERSIONS:
        raise ValueError(
            f"{manifest_path} has invalid normalization_version {value!r}; "
            f"expected one of {sorted(VALID_NORMALIZATION_VERSIONS)}"
        )
    return str(value)


def require_normalization_version(value: Any, *, context: str) -> str:
    """Return an explicitly declared, supported normalization version."""

    if not isinstance(value, str) or value not in VALID_NORMALIZATION_VERSIONS:
        raise ValueError(
            f"{context} requires normalization_version to be one of "
            f"{sorted(VALID_NORMALIZATION_VERSIONS)}, got {value!r}"
        )
    return value


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


def _strict_batch_index_validation_key(paths: Mapping[str, str]) -> tuple[Any, ...]:
    verified_generation = verified_arrow_artifact_generation(paths)
    if verified_generation is not None:
        return (
            "verified_arrow_artifact_generation",
            verified_generation,
            _arrow_artifact_material_binding(paths),
        )
    tokens: list[tuple[str, str, int, int, str]] = []
    for contract in RAW_PLANNER_ARROW_BATCH_INDEX_CONTRACTS:
        if contract.table_key not in paths:
            continue
        for key in (contract.table_key, contract.index_key):
            path = Path(paths[key])
            resolved, size, mtime_ns, digest = _stable_file_digest_token(path)
            tokens.append((key, resolved, size, mtime_ns, digest))
    return tuple(tokens)


def _validate_batch_indexes_once(paths: Mapping[str, str]) -> None:
    """Strictly validate one immutable Arrow generation once per process."""

    cache_key = _strict_batch_index_validation_key(paths)
    with _STRICT_BATCH_INDEX_VALIDATION_CACHE_LOCK:
        if cache_key in _STRICT_BATCH_INDEX_VALIDATION_CACHE:
            _STRICT_BATCH_INDEX_VALIDATION_CACHE.move_to_end(cache_key)
            return

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
    with _STRICT_BATCH_INDEX_VALIDATION_CACHE_LOCK:
        _STRICT_BATCH_INDEX_VALIDATION_CACHE[cache_key] = None
        _STRICT_BATCH_INDEX_VALIDATION_CACHE.move_to_end(cache_key)
        while len(_STRICT_BATCH_INDEX_VALIDATION_CACHE) > _STRICT_BATCH_INDEX_VALIDATION_CACHE_MAX_ENTRIES:
            _STRICT_BATCH_INDEX_VALIDATION_CACHE.popitem(last=False)


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


def validate_arrow_prediction_artifacts(
    arrow_paths: Mapping[str, Any],
    *,
    require_specter: bool,
    require_name_counts_index: bool,
    require_cluster_seeds: bool = False,
    require_batch_indexes: bool = False,
    strict_batch_index_validation: bool = False,
    expected_normalization_version: str | None = None,
    require_canonical_manifest: bool = False,
    context: str = "Arrow prediction",
    producer_hint: str = (
        "generate a complete Arrow bundle with scripts/convert_to_arrow.py or use the published "
        "s2and-release-arrow bundle"
    ),
) -> dict[str, str]:
    """Validate strict production Arrow prediction artifacts and return normalized paths."""

    required = {"signatures", "papers", "paper_authors"}
    if require_specter:
        required.add("specter")
    if require_name_counts_index:
        required.add("name_counts_index")
    if require_cluster_seeds:
        required.add("cluster_seeds")

    missing_keys = sorted(key for key in required if key not in arrow_paths)
    normalized, invalid_paths = _normalize_arrow_path_values(arrow_paths)
    for key in UNSUPPORTED_ARROW_NAME_ALIAS_KEYS.intersection(normalized):
        invalid_paths[key] = "name aliases must be supplied via the name_tuples argument, not Arrow path bundles"

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

    if require_batch_indexes:
        for key in required_filtered_read_batch_index_keys(normalized):
            required.add(key)
            if key not in normalized:
                missing_keys.append(key)

    required_or_declared_keys = {
        key
        for key in normalized
        if key in required or key.endswith("_batch_index") or key in DECLARED_ARROW_SIDECAR_KEYS
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
    _validate_arrow_bundle_manifest(
        normalized,
        expected_normalization_version=expected_normalization_version,
        require_canonical_manifest=require_canonical_manifest,
        context=context,
        required_keys=sorted(required),
        producer_hint=producer_hint,
    )
    if expected_normalization_version is not None and "name_counts_index" in normalized:
        artifact_version = read_name_counts_index_normalization_version(normalized["name_counts_index"])
        if artifact_version != expected_normalization_version:
            raise MissingArrowArtifactError(
                context=context,
                required_keys=sorted(required),
                missing_keys=(),
                missing_files={
                    "name_counts_index": (
                        f"normalization_version mismatch: artifact is {artifact_version!r} but the model "
                        f"feature contract requires {expected_normalization_version!r}; regenerate the "
                        "artifact bundle and model as one release unit "
                        "(docs/normalization_migration_blocked.md)"
                    )
                },
                producer_hint=producer_hint,
            )
    if strict_batch_index_validation:
        if not require_batch_indexes:
            raise ValueError("strict_batch_index_validation requires require_batch_indexes=True")
        _validate_batch_indexes_once(normalized)
    return normalized

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENV_KEYS = (
    "S2AND_BACKEND",
    "S2AND_SKIP_FASTTEXT",
    "S2AND_RUST_FEATURIZER_MAX_INMEM",
    "S2AND_RUST_BATCH_RSS_SAMPLER_MS",
    "S2AND_NORMALIZATION_VERSION",
    "S2AND_RUST_NAME_COUNTS_JSON",
    "PYTHONHASHSEED",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
)


def _run_git_command(project_root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def build_run_metadata(
    *,
    script_path: str | Path | None = None,
    argv: list[str] | None = None,
    env_keys: tuple[str, ...] = DEFAULT_ENV_KEYS,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    env_snapshot = {key: os.environ[key] for key in env_keys if key in os.environ}
    git_status = _run_git_command(project_root, ["status", "--porcelain"])
    resolved_script = Path(script_path).resolve() if script_path is not None else Path(__file__).resolve()
    resolved_argv = list(argv) if argv is not None else list(sys.argv)
    return {
        "generated_at_utc": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat(),
        "script": str(resolved_script),
        "argv": resolved_argv,
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(project_root),
        "git_commit": _run_git_command(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _run_git_command(project_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": (None if git_status is None else bool(git_status)),
        "env": env_snapshot,
    }


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def compute_file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as infile:
        while True:
            chunk = infile.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stats = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": compute_file_sha256(resolved),
        "size_bytes": int(stats.st_size),
        "mtime_utc": datetime.datetime.fromtimestamp(stats.st_mtime, datetime.UTC).isoformat(),
    }


def collect_rust_extension_identity(
    *,
    require_release: bool = False,
    fail_if_unavailable: bool = False,
) -> dict[str, Any]:
    try:
        import s2and_rust
    except Exception as exc:
        if fail_if_unavailable or require_release:
            raise RuntimeError(f"Failed to import s2and_rust: {exc}") from exc
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    module_path_raw = getattr(s2and_rust, "__file__", None)
    if module_path_raw is None:
        message = "s2and_rust.__file__ is unavailable"
        if fail_if_unavailable or require_release:
            raise RuntimeError(message)
        return {"available": False, "error_type": "MissingModulePath", "error": message}

    module_path = Path(str(module_path_raw)).resolve()
    if not module_path.exists():
        message = f"s2and_rust module path does not exist: {module_path}"
        if fail_if_unavailable or require_release:
            raise RuntimeError(message)
        return {"available": False, "error_type": "MissingBinary", "error": message}

    build_info: dict[str, Any] | None = None
    get_build_info = getattr(s2and_rust, "get_build_info", None)
    if callable(get_build_info):
        raw_build_info = get_build_info()
        if isinstance(raw_build_info, dict):
            build_info = {str(key): value for key, value in raw_build_info.items()}

    debug_assertions: bool | None = None
    if isinstance(build_info, dict) and "debug_assertions" in build_info:
        debug_assertions = bool(build_info["debug_assertions"])

    if require_release and debug_assertions is not False:
        raise RuntimeError(
            "Rust release build is required but debug_assertions is not false. "
            "Rebuild with: uv run maturin develop -m s2and_rust/Cargo.toml --release"
        )

    identity = {
        "available": True,
        "module_name": str(getattr(s2and_rust, "__name__", "s2and_rust")),
        "module_version": str(getattr(s2and_rust, "__version__", "unknown")),
        "module_file": str(module_path),
        "binary": collect_file_identity(module_path),
        "build_info": build_info,
        "debug_assertions": debug_assertions,
    }
    return identity


class ProcessTreeRSSMonitor:
    """Monitor peak RSS across current process and all child workers."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self.peak_rss_bytes = 0

    def _tree_rss_bytes(self) -> int:
        rss_total = 0
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.Error):
            pass
        for proc in processes:
            try:
                rss_total += int(proc.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return rss_total

    def sample_rss_bytes(self) -> int:
        rss = self._tree_rss_bytes()
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        return rss

    def sample_gb(self) -> float:
        return self.sample_rss_bytes() / (1024**3)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_rss_bytes()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self.peak_rss_bytes = self.sample_rss_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> ProcessTreeRSSMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def peak_gb(self) -> float:
        return self.peak_rss_bytes / (1024**3)


class RSSMonitor:
    """Monitor peak RSS for current process only."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self.peak_rss_bytes = 0

    def _rss_bytes(self) -> int:
        return int(self._process.memory_info().rss)

    def sample_rss_bytes(self) -> int:
        rss = self._rss_bytes()
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        return rss

    def sample_gb(self) -> float:
        return self.sample_rss_bytes() / (1024**3)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_rss_bytes()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self.peak_rss_bytes = self.sample_rss_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> RSSMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def peak_gb(self) -> float:
        return self.peak_rss_bytes / (1024**3)


def compute_rss_growth_fraction(rss_peak_gb_by_iteration: list[float]) -> float | None:
    if len(rss_peak_gb_by_iteration) < 2:
        return None
    first = float(rss_peak_gb_by_iteration[0])
    last = float(rss_peak_gb_by_iteration[-1])
    if first <= 0:
        return None
    return (last - first) / first

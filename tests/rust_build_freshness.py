"""Detect a loaded native extension built from different Rust sources.

Tests import the installed ``s2and_rust`` package. ``uv run`` uses timestamp
heuristics, so backdated source additions, deletions, preserved mtimes,
``--no-sync`` runs and direct interpreter calls can load an older build. The
extension embeds a digest of its build inputs (``s2and_rust/build.rs``); this
module recomputes it from the checkout. Matching source snapshots do not prove
that Cargo recompiled every dependency: preserved mtimes require cleaning both
local crates before rebuilding.
"""

import hashlib
import importlib
import importlib.metadata
import stat
import sys
from collections.abc import Callable
from pathlib import Path

RUST_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "s2and_rust"
REBUILD_COMMAND = "uv sync --extra dev --reinstall-package s2and"
CLEAN_COMMAND = (
    "uv run --no-sync cargo clean -p s2and_rust -p cld2 -p cld2-sys --manifest-path s2and_rust/Cargo.toml --release"
)

# Mirrors SOURCE_DIGEST_FILES and SOURCE_DIGEST_DIRS in s2and_rust/build.rs.
_SOURCE_DIGEST_FILES = (
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    "vendor/cld2/Cargo.toml",
    "vendor/cld2-sys/Cargo.toml",
    "vendor/cld2-sys/build.rs",
)
_SOURCE_DIGEST_DIRS = ("src", "vendor/cld2/src", "vendor/cld2-sys/src", "vendor/cld2-sys/cld2")


def _rust_source_files(directory: Path) -> list[Path]:
    """List native sources, rejecting directory links on every supported OS."""
    metadata = directory.lstat()
    if stat.S_ISLNK(metadata.st_mode) or (
        getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ValueError(f"linked native source directories are unsupported: {directory}")
    files: list[Path] = []
    for path in directory.iterdir():
        if path.is_dir():
            files.extend(_rust_source_files(path))
        elif path.is_file() and path.suffix in {".rs", ".cc", ".cpp", ".h"}:
            files.append(path)
    return files


def rust_source_digest(root: Path = RUST_PACKAGE_ROOT) -> str:
    """Hash native build inputs exactly as ``build.rs`` embeds them.

    Args:
        root: Rust package directory containing ``Cargo.toml``.

    Returns:
        The versioned digest string, for example ``"v1:<sha256 hex>"``.
    """
    files = [(name, root / name) for name in _SOURCE_DIGEST_FILES]
    for name in _SOURCE_DIGEST_DIRS:
        files.extend((path.relative_to(root).as_posix(), path) for path in _rust_source_files(root / name))
    hasher = hashlib.sha256()
    for relative, path in sorted(files):
        content = path.read_bytes()
        try:
            encoded_path = relative.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError(f"native source paths must be valid UTF-8: {relative!r}") from None
        hasher.update(encoded_path)
        hasher.update(b"\0")
        hasher.update(len(content).to_bytes(8, "little"))
        hasher.update(content)
    return f"v1:{hasher.hexdigest()}"


def duplicate_package_metadata_message(path: list[str] | None = None) -> str | None:
    """Explain extra ``s2and`` metadata that can set the wrong package version.

    Old setuptools builds left ``s2and.egg-info`` in the checkout root, and maturin
    never refreshes it. The editable install puts that root on ``sys.path``, so the
    stale copy can supply ``s2and.__version__`` and fail the native version check.

    Args:
        path: Directories to search; ``None`` searches ``sys.path``.

    Returns:
        None when one metadata copy names ``s2and``; otherwise a message listing them.
    """
    # importlib.metadata searches sys.path only when the path option is absent.
    search = {"name": "s2and"} if path is None else {"name": "s2and", "path": path}
    found = {
        (str(Path(str(dist.locate_file(""))).resolve()), dist.version)
        for dist in importlib.metadata.distributions(**search)
    }
    if len(found) <= 1:
        return None
    copies = ", ".join(f"{version} in {location}" for location, version in sorted(found))
    return (
        f"Found several s2and metadata copies ({copies}). The first one on sys.path sets s2and.__version__. "
        "Delete the stale copy, such as s2and.egg-info in the checkout root."
    )


def stale_rust_extension_message(root: Path = RUST_PACKAGE_ROOT) -> str | None:
    """Explain why the importable extension does not match the checkout.

    Args:
        root: Rust package directory whose sources the extension must match.

    Returns:
        None when the loaded extension embeds the checkout digest; otherwise a
        message naming the loaded file and the rebuild command.
    """
    try:
        s2and_rust = importlib.import_module("s2and._s2and_rust")
    except ImportError as exc:
        return f"Cannot import the s2and_rust extension ({exc}). Rebuild it with: {REBUILD_COMMAND}"
    loaded_paths = sorted(
        str(module.__file__)
        for name, module in sys.modules.items()
        if name.rsplit(".", 1)[-1] == "_s2and_rust" and getattr(module, "__file__", None)
    )
    loaded = ", ".join(loaded_paths) or "<unknown>"
    get_build_info: Callable[[], dict[str, object]] | None = vars(s2and_rust).get("get_build_info")
    if get_build_info is None:
        # Without an installed package, the repository's s2and_rust/ directory
        # imports as an empty namespace package.
        locations = ", ".join(str(path) for path in getattr(s2and_rust, "__path__", ())) or "<unknown>"
        return f"s2and_rust resolved to {locations}, not the compiled extension. Install it with: {REBUILD_COMMAND}"
    embedded = get_build_info().get("source_digest")
    try:
        expected = rust_source_digest(root)
    except (OSError, ValueError) as exc:
        return f"Cannot verify s2and_rust source freshness: {exc}. Correct the native source inputs before rebuilding."
    if embedded == expected:
        return None
    return (
        f"Loaded s2and_rust extension {loaded} was built from different sources "
        f"(embedded={embedded!r}, checkout={expected!r}). Rebuild it with: {REBUILD_COMMAND}. "
        f"If the mismatch persists, clear Cargo's cached release artifacts with: {CLEAN_COMMAND}, "
        f"then repeat: {REBUILD_COMMAND}"
    )

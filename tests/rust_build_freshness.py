"""Detect a loaded native extension built from different Rust sources.

Tests run from the repository import whichever ``s2and_rust`` resolves first,
which can be an old in-tree ``maturin develop`` build. The extension embeds a
digest of its build inputs (``s2and_rust/build.rs``); this module recomputes it
from the checkout. It establishes source freshness, not compiler provenance.
"""

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path

RUST_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "s2and_rust"
REBUILD_COMMAND = "uv run maturin develop -m s2and_rust/Cargo.toml --release"

# Mirrors SOURCE_DIGEST_FILES and SOURCE_DIGEST_DIRS in s2and_rust/build.rs.
_SOURCE_DIGEST_FILES = ("Cargo.toml", "Cargo.lock", "build.rs")
_SOURCE_DIGEST_DIRS = ("src", "vendor")


def rust_source_digest(root: Path = RUST_PACKAGE_ROOT) -> str:
    """Hash native build inputs exactly as ``build.rs`` embeds them.

    Args:
        root: Rust package directory containing ``Cargo.toml``.

    Returns:
        The versioned digest string, for example ``"v1:<sha256 hex>"``.
    """
    files = [(name, root / name) for name in _SOURCE_DIGEST_FILES]
    for name in _SOURCE_DIGEST_DIRS:
        files.extend((path.relative_to(root).as_posix(), path) for path in (root / name).rglob("*") if path.is_file())
    hasher = hashlib.sha256()
    for relative, path in sorted(files):
        content = path.read_bytes()
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(len(content).to_bytes(8, "little"))
        hasher.update(content)
    return f"v1:{hasher.hexdigest()}"


def stale_rust_extension_message(root: Path = RUST_PACKAGE_ROOT) -> str | None:
    """Explain why the importable extension does not match the checkout.

    Args:
        root: Rust package directory whose sources the extension must match.

    Returns:
        None when the loaded extension embeds the checkout digest; otherwise a
        message naming the loaded file and the rebuild command.
    """
    try:
        import s2and_rust
    except ImportError as exc:
        return f"Cannot import the s2and_rust extension ({exc}). Rebuild it with: {REBUILD_COMMAND}"
    loaded_paths = sorted(
        str(module.__file__)
        for name, module in sys.modules.items()
        if name.rsplit(".", 1)[-1] == "_s2and_rust" and getattr(module, "__file__", None)
    )
    loaded = ", ".join(loaded_paths) or "<unknown>"
    get_build_info: Callable[[], dict[str, object]] = vars(s2and_rust)["get_build_info"]
    embedded = get_build_info().get("source_digest")
    expected = rust_source_digest(root)
    if embedded == expected:
        return None
    return (
        f"Loaded s2and_rust extension {loaded} was built from different sources "
        f"(embedded={embedded!r}, checkout={expected!r}). Rebuild it with: {REBUILD_COMMAND}"
    )

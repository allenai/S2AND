"""Workspace import shim for the Rust extension package."""

from .s2and_rust import *  # noqa: F401,F403

# Expose the native extension version for capability gating.
#
# Note: ``from ... import *`` does not import dunder names like ``__version__``,
# so we set it explicitly here.
try:  # pragma: no cover
    from .s2and_rust._s2and_rust import __version__ as __version__
except Exception:  # pragma: no cover
    try:
        from ._s2and_rust import __version__ as __version__
    except Exception:
        __version__ = None

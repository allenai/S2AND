"""Workspace import shim for the Rust extension package."""

from . import s2and_rust as _s2and_rust_pkg
from .s2and_rust import *  # noqa: F401,F403

# Expose the native extension version for capability gating.
#
# Note: ``from ... import *`` does not import dunder names like ``__version__``,
# so we set it explicitly here.
__version__ = getattr(_s2and_rust_pkg, "__version__", None)

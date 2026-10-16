"""Python package wrapper for the Rust extension module."""

from . import _s2and_rust as _native
from ._s2and_rust import *  # type: ignore  # noqa: F401,F403

_ArrowDataset = _native._ArrowDataset
_arrow_retained_file_sha256 = _native._arrow_retained_file_sha256

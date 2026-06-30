"""FeatureBlock conversion helpers for writing Arrow tables from ANDData.

The implementation now lives in the packaged module ``s2and.arrow_service_io`` so
services that install s2and (without the ``scripts/`` tree) can import it. This
module re-exports the public helpers for backwards compatibility.
"""

from __future__ import annotations

from s2and.arrow_service_io import (
    feature_block_from_anddata,
    write_feature_block_arrow_from_anddata,
)

__all__ = [
    "feature_block_from_anddata",
    "write_feature_block_arrow_from_anddata",
]

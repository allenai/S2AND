"""Checksummed JSON artifacts have the same bytes on every operating system."""

from __future__ import annotations

from pathlib import Path

from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.production_bundle import _write_json
from scripts.eps_sweep.common import write_json


def test_bundle_and_review_json_use_lf_line_endings(tmp_path: Path) -> None:
    for writer, path in ((_write_json, tmp_path / "bundle.json"), (write_json, tmp_path / "review.json")):
        writer(path, {"nested": {"value": 1}, "name": "x"})
        contents = path.read_bytes()
        assert b"\r" not in contents
        assert contents.endswith(b"}\n")


def test_name_count_manifest_uses_lf_line_endings(tmp_path: Path) -> None:
    index_dir, _counts = write_name_counts_index(
        tmp_path, ({"alex": 2}, {"smith": 3}, {"alex smith": 2}, {"smith a": 2})
    )
    assert b"\r" not in (Path(index_dir) / "manifest.json").read_bytes()

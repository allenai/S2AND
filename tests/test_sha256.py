"""Tests for shared SHA-256 lexical validation."""

from __future__ import annotations

from pathlib import Path

from s2and._sha256 import is_lowercase_sha256, sha256_file


def test_sha256_file_and_lexical_validation(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    cases = (
        ("mixed-hex-digest", "0123456789abcdef" * 4, True),
        ("uppercase", "A" * 64, False),
        ("non-hex", "g" * 64, False),
        ("short", "0" * 63, False),
        ("none", None, False),
    )
    for case_id, value, expected in cases:
        assert is_lowercase_sha256(value) is expected, case_id

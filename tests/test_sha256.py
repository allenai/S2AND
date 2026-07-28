"""Tests for shared SHA-256 lexical validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from s2and._sha256 import is_lowercase_sha256, sha256_file


def test_sha256_file_known_content(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0" * 64, True),
        ("0123456789abcdef" * 4, True),
        ("A" * 64, False),
        ("g" * 64, False),
        ("0" * 63, False),
        ("0" * 65, False),
        ("", False),
        (None, False),
        (b"0" * 64, False),
    ],
)
def test_is_lowercase_sha256(value: object, expected: bool) -> None:
    assert is_lowercase_sha256(value) is expected

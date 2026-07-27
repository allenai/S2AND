"""Tests for shared SHA-256 lexical validation."""

from __future__ import annotations

import pytest

from s2and._sha256 import is_lowercase_sha256


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

"""Verify public Python and native name helpers against reviewed examples."""

from __future__ import annotations

import json
from pathlib import Path

from s2and.runtime import load_s2and_rust_extension
from s2and.text import NAME_PREFIXES, canonical_name_count_keys, canonicalize_name_parts, canonicalize_name_text


def test_public_native_helpers_match_reviewed_names() -> None:
    native = load_s2and_rust_extension()
    fixture = json.loads((Path(__file__).parent / "fixtures/canonical_name_examples.json").read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        raw = case["input"]
        parts = native.canonicalize_name_parts(raw["first"], raw["middle"], raw["last"])
        assert parts == tuple(canonicalize_name_parts(raw["first"], raw["middle"], raw["last"])), case["id"]
        assert native.canonical_name_count_keys(parts) == case["canonical"]["count_keys"], case["id"]


def test_public_native_helpers_preserve_missing_names_and_title_policy() -> None:
    native = load_s2and_rust_extension()
    assert set(native.NAME_PREFIXES) == NAME_PREFIXES
    parts = native.canonicalize_name_parts(None, None, None)
    assert parts == ("", "", "")
    assert native.canonical_name_count_keys(parts) == canonical_name_count_keys(
        canonicalize_name_parts(None, None, None)
    )
    for raw in (None, "", "Dr. Anne-Marie Claire", "O’Connor", "Ou—Yang", "王明"):
        assert native.canonicalize_name_text(raw) == canonicalize_name_text(raw)

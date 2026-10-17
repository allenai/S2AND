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


# D9 edge cases beyond the reviewed fixture: whitespace classes, the dotted
# abbreviation list, titles, letter counts, and the transliterated case guard.
D9_EDGE_CASES = [
    (("JH\u00a0", None, "Lee"), ("j", "h", "lee")),
    (("J\x1cH", None, "Lee"), ("j", "h", "lee")),
    ((" J.H. ", "", "Lee"), ("j", "h", "lee")),
    (("PH", None, "Dupont"), ("p", "h", "dupont")),
    (("PH. ", None, "Dupont"), ("ph", "", "dupont")),
    (("P.H.", None, "Dupont"), ("p", "h", "dupont")),
    (("GY.", None, "Nagy"), ("gy", "", "nagy")),
    (("E.D.", None, "Wood"), ("e", "d", "wood")),
    (("MRS", None, "Smith"), ("", "", "smith")),
    (("PHD", None, "Smith"), ("", "", "smith")),
    (("JHMK", None, "Smith"), ("jhmk", "", "smith")),
    (("H-G", None, "Wells"), ("h g", "", "wells")),
    (("HG", None, "\u00d3lafsson"), ("h", "g", "olafsson")),
    (("HG", None, "\u00d3LAFSSON"), ("hg", "", "olafsson")),
    (("HG", None, "DeLa"), ("h", "g", "dela")),
    (("HG", None, "McDONALD"), ("hg", "", "mcdonald")),
    (("HG", None, "\u91d1"), ("h", "g", "jin")),
    (("HG", None, None), ("hg", "", "")),
    # The mixed-case branch: whitespace classes, periods, vowels, the list, titles, and letter counts.
    (("J\u00a0h", None, "Lee"), ("j", "h", "lee")),
    (("J\x1ch", None, "Lee"), ("j", "h", "lee")),
    (("Jhm", None, "Lee"), ("j", "h m", "lee")),
    (("Jh.", None, "Lee"), ("jh", "", "lee")),
    (("Jhmk", None, "Lee"), ("jhmk", "", "lee")),
    (("By", None, "Lee"), ("by", "", "lee")),
    (("Wm", None, "Lee"), ("wm", "", "lee")),
    (("Ms", None, "Lee"), ("", "", "lee")),
    (("Hj", None, "\u00d3lafsson"), ("h", "j", "olafsson")),
    (("Hj", None, "McDONALD"), ("hj", "", "mcdonald")),
    (("Hj", None, None), ("hj", "", "")),
]


def test_public_native_helpers_match_d9_edge_cases() -> None:
    native = load_s2and_rust_extension()
    for raw, expected in D9_EDGE_CASES:
        assert tuple(canonicalize_name_parts(*raw)) == expected, raw
        assert native.canonicalize_name_parts(*raw) == expected, raw

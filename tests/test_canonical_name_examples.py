"""Frozen canonical name-normalization example table (migration step 1).

This test module enforces ``tests/fixtures/canonical_name_examples.json``, the
step-1 artifact of ``docs/normalization_migration_blocked.md``, in three
layers:

- Legacy pins (run today): the fixture's ``legacy`` values are asserted
  against the current normalizer and count-key shims, so accidental legacy
  drift is caught while the canonical migration branch is in flight.
- Table coherence (run today): equivalence groups, decision references, and
  normalized-form invariants of the hand-authored ``canonical`` values.
- Canonical contract (skipped until implemented): the fixture's ``canonical``
  values are asserted against ``s2and.text.canonicalize_name_parts`` and
  ``s2and.text.canonical_name_count_keys`` once migration step 2 lands. The
  function names are the proposed step-2 surface; adjust here and in the
  fixture description together if the API lands under different names.

The JSON fixture is the frozen source of truth. The generator that produced it
(``scratch/generate_canonical_name_examples.py``) exists only to add cases;
decided values must not be regenerated silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import s2and.text as s2and_text
from s2and.data import _canonicalize_last_for_counts
from s2and.text import (
    NAME_PREFIXES,
    has_name_dash,
    normalize_text,
    split_first_middle_hyphen_aware,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_name_examples.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
CASE_PARAMS = [pytest.param(case, id=case["id"]) for case in CASES]

CANONICALIZE_NAME_PARTS = getattr(s2and_text, "canonicalize_name_parts", None)
CANONICAL_NAME_COUNT_KEYS = getattr(s2and_text, "canonical_name_count_keys", None)
needs_canonical_api = pytest.mark.skipif(
    CANONICALIZE_NAME_PARTS is None or CANONICAL_NAME_COUNT_KEYS is None,
    reason="canonical normalizer not implemented yet (migration step 2)",
)


def _legacy_first_normalized_token(first_raw: str, middle_raw: str) -> str:
    # Mirrors ANDData.preprocess_signatures' legacy single-token construction.
    first_normalized = normalize_text(first_raw)
    middle_normalized = normalize_text(middle_raw)
    first_middle_split = (first_normalized + " " + middle_normalized).split(" ")
    if first_middle_split and first_middle_split[0] in NAME_PREFIXES:
        first_middle_split = first_middle_split[1:]
    return first_middle_split[0] if first_middle_split else ""


def _legacy_count_keys(first_raw: str, last_raw: str, first_without_apostrophe: str, last_normalized: str) -> dict:
    # Mirrors ANDData._compute_signature_name_counts key construction
    # (initial_char semantics).
    first_for_counts = first_without_apostrophe.split(" ")[0] if first_without_apostrophe else ""
    if has_name_dash(first_raw):
        joined = first_without_apostrophe.replace(" ", "")
        if joined:
            first_for_counts = joined
    last_for_counts = _canonicalize_last_for_counts(last_raw, last_normalized)
    first_last_for_count = (first_for_counts + " " + last_for_counts).strip()
    first_initial = first_for_counts[0] if first_for_counts else ""
    last_first_initial_for_count = (last_for_counts + " " + first_initial).strip()
    return {
        "first": first_for_counts,
        "last": last_for_counts,
        "first_last": first_last_for_count,
        "last_first_initial": last_first_initial_for_count,
    }


@pytest.mark.parametrize("case", CASE_PARAMS)
def test_legacy_normalized_fields_match_current_code(case):
    raw = case["input"]
    expected = case["legacy"]

    first_wo, middle_wo = split_first_middle_hyphen_aware(raw["first"], raw["middle"])
    assert first_wo == expected["first_normalized_without_apostrophe"]
    assert middle_wo == expected["middle_normalized_without_apostrophe"]
    assert normalize_text(raw["last"]) == expected["last_normalized"]
    assert _legacy_first_normalized_token(raw["first"], raw["middle"]) == expected["first_normalized"]


@pytest.mark.parametrize("case", CASE_PARAMS)
def test_legacy_count_keys_match_current_shims(case):
    raw = case["input"]
    expected = case["legacy"]
    computed = _legacy_count_keys(
        raw["first"],
        raw["last"],
        expected["first_normalized_without_apostrophe"],
        expected["last_normalized"],
    )
    assert computed == expected["count_keys"]


def test_equivalence_groups_share_canonical_fields():
    groups: dict[str, list[dict]] = {}
    for case in CASES:
        group = case["equivalence_group"]
        if group is not None:
            groups.setdefault(group, []).append(case)
    assert groups, "fixture must define at least one equivalence group"
    for group, members in groups.items():
        assert len(members) >= 2, f"equivalence group {group!r} has fewer than two members"
        triples = {(m["canonical"]["first"], m["canonical"]["middle"], m["canonical"]["last"]) for m in members}
        assert len(triples) == 1, f"equivalence group {group!r} disagrees on canonical fields: {triples}"


def test_decision_references_are_wellformed():
    registry = FIXTURE["decisions"]
    assert registry, "fixture must carry the decisions registry"
    for decision_id, decision in registry.items():
        assert decision["status"] in {"open", "decided"}, f"{decision_id}: bad status {decision['status']!r}"
        assert decision["title"] and decision["description"]
    referenced = {decision_id for case in CASES for decision_id in case["decisions"]}
    unknown = referenced - set(registry)
    assert not unknown, f"cases reference unknown decisions: {sorted(unknown)}"
    unreferenced = set(registry) - referenced
    assert not unreferenced, f"decisions never exercised by any case: {sorted(unreferenced)}"


@pytest.mark.parametrize("case", CASE_PARAMS)
def test_canonical_values_are_in_normalized_form(case):
    canonical = case["canonical"]
    for field in ("first", "middle", "last"):
        value = canonical[field]
        assert value == " ".join(value.split()), f"{field} not whitespace-normalized: {value!r}"
        assert all(ch.islower() or ch == " " for ch in value), f"{field} has non [a-z ] chars: {value!r}"
    for key_name, key_value in canonical["count_keys"].items():
        if key_value is not None:
            assert key_value == " ".join(key_value.split()), f"count key {key_name} malformed: {key_value!r}"
            assert key_value != "", f"count key {key_name} must be null instead of empty"


@needs_canonical_api
@pytest.mark.parametrize("case", CASE_PARAMS)
def test_canonical_fields(case):
    raw = case["input"]
    expected = case["canonical"]
    parts = CANONICALIZE_NAME_PARTS(raw["first"], raw["middle"], raw["last"])
    assert (parts.first, parts.middle, parts.last) == (expected["first"], expected["middle"], expected["last"])


@needs_canonical_api
@pytest.mark.parametrize("case", CASE_PARAMS)
def test_canonical_count_keys(case):
    raw = case["input"]
    expected = case["canonical"]["count_keys"]
    parts = CANONICALIZE_NAME_PARTS(raw["first"], raw["middle"], raw["last"])
    assert CANONICAL_NAME_COUNT_KEYS(parts) == expected

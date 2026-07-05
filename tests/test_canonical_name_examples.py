"""Frozen canonical name-normalization example table (migration step 1).

This test module enforces ``tests/fixtures/canonical_name_examples.json``, the
step-1 artifact of ``docs/normalization_migration_blocked.md``, in four
layers:

- Legacy pins (run today): the fixture's ``legacy`` values are asserted
  against the current normalizer, and the legacy count keys are asserted
  through the LIVE ``ANDData._compute_signature_name_counts`` path with
  seeded count dicts, so drift in the real method fails here.
- Compare-time contract (run today): canonical first fields across variant
  groups (``Jo`` / ``Jo Ann`` / ``JoAnn`` etc.) must be pairwise compatible
  under the live ``same_prefix_tokens`` — this is issue #39's real invariant
  after the D1 ruling (spill on space, keep together on dash).
- Table coherence (run today): equivalence groups, decision references, and
  normalized-form invariants of the ``canonical`` values.
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

import itertools
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest

import s2and.text as s2and_text
from s2and.data import (
    NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
    ANDData,
    Signature,
)
from s2and.text import (
    NAME_PREFIXES,
    normalize_text,
    same_prefix_tokens,
    split_first_middle_hyphen_aware,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "canonical_name_examples.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE["cases"]
CASE_PARAMS = [pytest.param(case, id=case["id"]) for case in CASES]
CASES_BY_ID = {case["id"]: case for case in CASES}

CANONICALIZE_NAME_PARTS = cast(Any, getattr(s2and_text, "canonicalize_name_parts", None))
CANONICAL_NAME_COUNT_KEYS = cast(Any, getattr(s2and_text, "canonical_name_count_keys", None))
needs_canonical_api = pytest.mark.skipif(
    CANONICALIZE_NAME_PARTS is None or CANONICAL_NAME_COUNT_KEYS is None,
    reason="canonical normalizer not implemented yet (migration step 2)",
)


def _legacy_first_normalized_token(first_raw: str, middle_raw: str) -> str:
    # Mirrors ANDData.preprocess_signatures' legacy single-token construction.
    # Documentation-grade only: the field is vestigial (write-only) and is
    # scheduled for removal with the dual-field unification.
    first_normalized = normalize_text(first_raw)
    middle_normalized = normalize_text(middle_raw)
    first_middle_split = (first_normalized + " " + middle_normalized).split(" ")
    if first_middle_split and first_middle_split[0] in NAME_PREFIXES:
        first_middle_split = first_middle_split[1:]
    return first_middle_split[0] if first_middle_split else ""


def _skeleton_signature(author_info_last: str | None) -> Signature:
    return Signature(
        author_info_first=None,
        author_info_first_normalized_without_apostrophe=None,
        author_info_middle=None,
        author_info_middle_normalized_without_apostrophe=None,
        author_info_last_normalized=None,
        author_info_last=cast(str, author_info_last),
        author_info_suffix_normalized=None,
        author_info_suffix=None,
        author_info_first_normalized=None,
        author_info_coauthors=None,
        author_info_coauthor_blocks=None,
        author_info_full_name=None,
        author_info_affiliations=[],
        author_info_affiliations_n_grams=None,
        author_info_coauthor_n_grams=None,
        author_info_email=None,
        author_info_orcid=None,
        author_info_name_counts=None,
        author_info_position=0,
        author_info_block="",
        author_info_given_block=None,
        author_info_estimated_gender=None,
        author_info_estimated_ethnicity=None,
        paper_id=0,
        sourced_author_source=None,
        sourced_author_ids=[],
        author_id=None,
        signature_id="",
    )


# Sentinel count values, one per key, so a wrong-key lookup returns the default
# (1) or NaN rather than the sentinel and fails the assertion.
_COUNT_SENTINELS = {"first": 7.0, "last": 11.0, "first_last": 13.0, "last_first_initial": 17.0}


def _live_legacy_name_counts(case):
    """Run the real ANDData count path with dicts seeded at the fixture's keys.

    Each dict maps only the fixture's expected legacy key (when non-None) to a
    sentinel value, so if ``_compute_signature_name_counts`` ever builds a
    different key string the lookup falls back to the default (1) and the
    assertions fail. A None fixture key means the method should do no lookup and
    return NaN. ``first_without_apostrophe=None`` forces the method to recompute
    the normalized fields itself, exercising the full live path.
    """
    raw = case["input"]
    keys = case["legacy"]["count_keys"]
    dataset = ANDData.__new__(ANDData)
    dataset.first_dict = {} if keys["first"] is None else {keys["first"]: _COUNT_SENTINELS["first"]}
    dataset.last_dict = {} if keys["last"] is None else {keys["last"]: _COUNT_SENTINELS["last"]}
    dataset.first_last_dict = {} if keys["first_last"] is None else {keys["first_last"]: _COUNT_SENTINELS["first_last"]}
    dataset.last_first_initial_dict = (
        {}
        if keys["last_first_initial"] is None
        else {keys["last_first_initial"]: _COUNT_SENTINELS["last_first_initial"]}
    )
    dataset.name_counts_last_first_initial_semantics = NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR
    return dataset._compute_signature_name_counts(
        _skeleton_signature(raw["last"]),
        first_raw=raw["first"],
        middle_raw=raw["middle"],
        first_without_apostrophe=None,
        last_normalized=None,
    )


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
def test_legacy_count_keys_via_live_anddata_path(case):
    counts = _live_legacy_name_counts(case)
    keys = case["legacy"]["count_keys"]
    for name, count_value in [
        ("first", counts.first),
        ("last", counts.last),
        ("first_last", counts.first_last),
        ("last_first_initial", counts.last_first_initial),
    ]:
        if keys[name] is None:
            assert math.isnan(count_value), f"{name} should be NaN (no lookup) but was {count_value}"
        else:
            assert count_value == _COUNT_SENTINELS[name], f"{name} lookup did not hit the seeded key {keys[name]!r}"


def test_compare_time_first_name_compatibility():
    groups = FIXTURE["compare_compatibility"]["compatible_groups"]
    assert groups, "fixture must define compare-compatible groups"
    for group in groups:
        firsts = {case_id: CASES_BY_ID[case_id]["canonical"]["first"] for case_id in group}
        for (id_a, first_a), (id_b, first_b) in itertools.combinations(firsts.items(), 2):
            assert same_prefix_tokens(
                first_a, first_b
            ), f"{id_a} ({first_a!r}) and {id_b} ({first_b!r}) must be compare-time compatible"


def test_compare_time_first_name_incompatibility():
    pairs = FIXTURE["compare_compatibility"]["incompatible_pairs"]
    assert pairs, "fixture must define incompatible pairs"
    for first_a, first_b in pairs:
        assert not same_prefix_tokens(first_a, first_b), f"{first_a!r} vs {first_b!r} must NOT be compatible"


def test_compare_time_first_name_truth_table():
    rows = FIXTURE["compare_compatibility"]["truth_table"]
    assert rows, "fixture must define the compare-time truth table"
    for row in rows:
        actual = same_prefix_tokens(row["a"], row["b"])
        actual_reversed = same_prefix_tokens(row["b"], row["a"])
        assert actual is row["compatible"], f"{row['a']!r} vs {row['b']!r}: {row['notes']}"
        assert actual_reversed is row["compatible"], f"truth table must be symmetric for {row['a']!r}/{row['b']!r}"


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

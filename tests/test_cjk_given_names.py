"""Regressions for the hard name rule on Han, kana and hangul given names.

Transliteration writes each Han character as one pinyin syllable. The first
syllable became the canonical first name and the rest the middle name, so
different given names such as 重喜 ("zhong xi") and 重郷 ("zhong xiang") passed
the prefix rules, and production merged them (blocks 重 鈴木, 勝 井上, 孝 清水).
"""

from pathlib import Path

import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.model import DEFAULT_ORCID_NAME_WAIVER
from s2and.text import (
    canonical_names_conflict,
    canonicalize_name_parts,
    cjk_given_name_for_constraint,
    is_cjk_given_name,
)
from tests.incremental_test_helpers import _dataset, _predict, prepare_seed_arrow_bundle
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors


@pytest.mark.parametrize(
    ("first", "middle", "expected"),
    [
        ("重喜", None, True),
        ("孝", "", True),
        ("重", "喜", True),
        ("ひろし", None, True),
        ("ヒロシ", None, True),
        ("민수", None, True),
        ("佐々木", None, True),
        ("重喜.", " ", True),
        (None, None, False),
        ("Wei", None, False),
        ("重喜", "Shigeki", False),
        ("Shigeki 重喜", None, False),
        ("Жан", None, False),
    ],
)
def test_is_cjk_given_name(first, middle, expected):
    assert is_cjk_given_name(first, middle) is expected


def _conflict(first_a: str, first_b: str, last_a: str = "鈴木", last_b: str = "鈴木") -> bool:
    name_a = canonicalize_name_parts(first_a, None, last_a)
    name_b = canonicalize_name_parts(first_b, None, last_b)
    given_names = cjk_given_name_for_constraint(first_a, None), cjk_given_name_for_constraint(first_b, None)
    return canonical_names_conflict(name_a, name_b, set(), cjk_given_names=given_names)


@pytest.mark.parametrize(
    ("first_a", "first_b", "expected"),
    [
        # The production merges: pinyin syllables that match by prefix, or a missing second syllable.
        ("重喜", "重郷", True),
        ("重郷", "重憲", True),
        ("重喜", "重憲", True),
        ("勝央", "勝彦", True),
        ("孝徳", "孝", True),
        # Chinese given names in Han script, kana and hangul.
        ("伟", "伟明", True),
        ("ひろ", "ひろし", True),
        ("민", "민수", True),
        # One given name in two forms stays compatible.
        ("孝徳", "孝徳", False),
        ("张伟", "張偉", False),
        ("ひろし", "ヒロシ", False),
        ("민수", "민 수", False),
        # A romanized side keeps the plain rules, which allow this pair.
        ("Xiao", "孝徳", False),
    ],
)
def test_cjk_given_names_conflict_only_when_they_differ(first_a, first_b, expected):
    assert _conflict(first_a, first_b) is expected


def test_last_name_rule_still_applies_to_equal_cjk_given_names():
    assert _conflict("重喜", "重喜", last_b="佐藤")


def _partition(clusters, names):
    return sorted(sorted(names[int(signature_id)] for signature_id in members) for members in clusters.values())


# Every pair gets model distance 0.1, under EPS 0.5, so only the name rule keeps people apart.
@pytest.mark.parametrize("route", ["arrow", "python"])
@pytest.mark.parametrize(
    ("given_names", "last", "expected"),
    [
        (["重喜", "重喜", "重郷", "重郷", "重憲"], "鈴木", [["重喜", "重喜"], ["重憲"], ["重郷", "重郷"]]),
        (["勝央", "勝央", "勝彦"], "井上", [["勝央", "勝央"], ["勝彦"]]),
        (["孝徳", "孝徳", "孝"], "清水", [["孝"], ["孝徳", "孝徳"]]),
        (["孝徳", "孝徳", "孝徳"], "清水", [["孝徳", "孝徳", "孝徳"]]),
        (["민수", "민 수", "민"], "김", [["민"], ["민 수", "민수"]]),
    ],
)
def test_full_block_prediction_keeps_different_cjk_given_names_apart(
    tmp_path: Path, route, given_names, last, expected
):
    dataset = _dataset([(first, "", last) for first in given_names], {})
    signature_ids = [str(index) for index in range(len(given_names))]
    clusters = _predict(tmp_path, dataset, {"block": signature_ids}, route, threshold=None)
    assert _partition(clusters, given_names) == expected


@pytest.mark.parametrize("route", ["arrow", "python"])
@pytest.mark.parametrize(
    ("left", "right", "cluster_count"),
    [
        (("拓海", ""), ("拓海", ""), 1),
        (("奈々", ""), ("奈奈", ""), 1),
        (("奈々", ""), ("奈", ""), 2),
        (("重喜", ""), ("重", "喜"), 1),
        (("ひ", "ろし"), ("ひろ", "し"), 1),
        (("𠮷秀", ""), ("吉秀", ""), 1),
        (("しけ", "\u3099る"), ("しげる", ""), 1),
    ],
)
def test_whole_given_name_orthography(tmp_path: Path, route, left, right, cluster_count):
    dataset = _dataset([(*left, "鈴木"), (*right, "鈴木")], {})
    clusters = _predict(tmp_path, dataset, {"block": ["0", "1"]}, route, threshold=None)
    assert len(clusters) == cluster_count


def test_orcid_name_waiver_is_independent_of_mixed_script_input_order(tmp_path: Path):
    names = ["孝", *(["孝徳"] * 10), "Xiao"]
    dataset = _dataset([(name, "", "鈴木") for name in names], {})
    for index in range(10):
        signature_id = str(index)
        dataset.signatures[signature_id] = dataset.signatures[signature_id]._replace(
            author_info_orcid="0000-0002-1825-0097"
        )
    arrow_dir = tmp_path / "arrow"
    prepare_seed_arrow_bundle(dataset, arrow_dir)
    with ArrowDataset.open(arrow_dir) as arrow:
        native = build_rust_featurizer_from_arrow_dataset(arrow, name_tuples=set(), num_threads=1)
        ids = native.signature_ids()

        def values(order: list[str], waiver: tuple[int, float] | None):
            left, right, distances = native.get_constraints_block_upper_triangle_indexed(
                [ids.index(signature_id) for signature_id in order],
                orcid_name_waiver=waiver,
                num_threads=1,
            )
            return {
                frozenset((order[i], order[j])): distance for i, j, distance in zip(left, right, distances, strict=True)
            }

        order = list(map(str, range(len(names))))
        pair = frozenset(("0", "10"))
        assert values(order, None)[pair] == LARGE_DISTANCE
        expected = values(order, DEFAULT_ORCID_NAME_WAIVER)
        assert expected[pair] is None
        assert values([order[-1], *order[:-1]], DEFAULT_ORCID_NAME_WAIVER) == expected


def test_incremental_new_link_name_check_applies_the_cjk_rule():
    from s2and.incremental_linking.production import _new_link_name_conflict

    dataset = _dataset(
        [
            ("重喜", "", "鈴木"),
            ("重郷", "", "鈴木"),
            ("重喜", "", "鈴木"),
            ("Zhong", "Xiang", "鈴木"),
            ("拓", "", "鈴木"),
            ("拓海", "", "鈴木"),
            ("拓海", "", "鈴木"),
        ],
        {},
    )
    dataset.name_tuples = frozenset()
    conflict, conflict_key = _new_link_name_conflict(dataset, {})  # type: ignore[arg-type]
    assert conflict("0", "1")
    assert not conflict("0", "2")
    # The romanized spelling has the same canonical parts as 重郷 but keeps the plain rules.
    assert not conflict("0", "3")
    assert conflict_key("1") != conflict_key("3")
    # The legacy parts of 拓 and 拓海 match after dropping 海.
    assert conflict_key("4") != conflict_key("5")
    assert conflict("4", "6")
    assert not conflict("5", "6")

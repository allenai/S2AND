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
        ("ひろし", None, True),
        ("민수", None, True),
        ("佐々木", None, True),
        ("重喜.", " ", True),
        (None, None, False),
        ("重喜", "Shigeki", False),
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
        ("孝徳", "孝", True),
        # One given name in two forms stays compatible.
        ("张伟", "張偉", False),
        ("ひろし", "ヒロシ", False),
        # A romanized side keeps the plain rules, which allow this pair.
        ("Xiao", "孝徳", False),
        # An incomplete transliteration (the long-vowel mark ー, a halfwidth voiced mark, a Han
        # character outside the table) keeps the ordinary rules, which veto these pairs.
        ("ピーター", "ジョン", True),
        ("ｼﾞｭﾝ", "ｹﾝ", True),
        # Hangul and Han forms of one name romanize differently, so they keep the ordinary rules.
        ("민수", "敏洙", False),
    ],
)
def test_cjk_given_names_conflict_only_when_they_differ(first_a, first_b, expected):
    assert _conflict(first_a, first_b) is expected


def test_last_name_rule_still_applies_to_equal_cjk_given_names():
    assert _conflict("重喜", "重喜", last_b="佐藤")


def _partition(clusters, names):
    return sorted(sorted(names[int(signature_id)] for signature_id in members) for members in clusters.values())


# Set model distances below the cut to exercise the name rule.
@pytest.mark.parametrize("route", ["arrow", "python"])
def test_full_block_prediction_keeps_different_cjk_given_names_apart(tmp_path: Path, route: str) -> None:
    names = ["重喜", "重喜", "重郷", "重郷", "重憲", "민수", "민 수", "민"]
    dataset = _dataset([(first, "", "鈴木") for first in names[:5]] + [(first, "", "김") for first in names[5:]], {})

    clusters = _predict(tmp_path, dataset, {"suzuki": list("01234"), "kim": list("567")}, route, threshold=None)

    assert _partition(clusters, names) == sorted(
        [["重喜", "重喜"], ["重憲"], ["重郷", "重郷"], ["민"], ["민 수", "민수"]]
    )


@pytest.mark.parametrize("route", ["arrow", "python"])
def test_whole_given_name_orthography(tmp_path: Path, route: str) -> None:
    pairs = [
        (("拓海", ""), ("拓海", ""), True),
        (("奈々", ""), ("奈奈", ""), True),
        (("奈々", ""), ("奈", ""), False),
        (("重喜", ""), ("重", "喜"), True),
        # Keep the ordinary name rules when transliteration omits a Han character.
        (("𠮷秀", ""), ("吉秀", ""), False),
        (("しけ", "\u3099る"), ("しげる", ""), True),
        # Treat U+323B0 as incomplete across Unicode versions.
        (("孝𲎰", ""), ("孝徳", ""), True),
        (("ｶﾞｸ", ""), ("ガク", ""), True),
        (("しか゛", ""), ("しが", ""), True),
        (("ｶﾞｸ", ""), ("ヒロシ", ""), False),
    ]
    names = [(*given, "鈴木") for left, right, _ in pairs for given in (left, right)]
    blocks = {str(i): [str(2 * i), str(2 * i + 1)] for i in range(len(pairs))}
    expected = set()
    for (_, _, joins), (left_id, right_id) in zip(pairs, blocks.values(), strict=True):
        if joins:
            expected.add(frozenset((left_id, right_id)))
        else:
            expected.update((frozenset((left_id,)), frozenset((right_id,))))

    clusters = _predict(tmp_path, _dataset(names, {}), blocks, route, threshold=None)

    assert {frozenset(members) for members in clusters.values()} == expected


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

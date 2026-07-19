from __future__ import annotations

import pandas as pd
import pytest

from scripts._pair_ablation.study import (
    ALL13_DOMAINS,
    ALL_DOMAINS,
    BASE_FAMILY,
    BASELINE,
    BIG7_DOMAINS,
    GOLD_DOMAINS,
    LINKER_FAMILY,
    PAIR_COLUMNS,
    PAIR_DOMAINS,
    PROXY_DOMAINS,
    AdditiveDose,
    pair_digest,
    parse_arm_name,
    select_pairs,
    validate_catalog,
)


def _frame(rows: list[tuple[object, ...]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _catalog() -> pd.DataFrame:
    rows: list[tuple[object, ...]] = [
        ("aminer", BASE_FAMILY, "a0", "b0", 0),
        ("arnetminer", BASE_FAMILY, "a1", "b1", 1),
        ("pubmed", BASE_FAMILY, "a2", "b2", 0),
        ("arnetminer", LINKER_FAMILY, "a1", "b1", 1),  # base overlap
    ]
    for domain in ("a_khan", "a_silva", "arnetminer"):
        for label, count in ((0, 4), (1, 2)):
            rows.extend(
                (domain, LINKER_FAMILY, f"{domain}-{label}-{index}-a", f"{domain}-{label}-{index}-b", label)
                for index in range(count)
            )
    return _frame(rows)


def test_domain_sets_and_arm_names_are_fixed() -> None:
    assert GOLD_DOMAINS == ("aminer", "arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")
    assert PAIR_DOMAINS == ("medline",)
    assert PROXY_DOMAINS == ("a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park")
    assert ALL_DOMAINS == (*GOLD_DOMAINS, *PAIR_DOMAINS, *PROXY_DOMAINS)
    assert BIG7_DOMAINS == PROXY_DOMAINS
    assert ALL13_DOMAINS == ("arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath", *PROXY_DOMAINS)
    assert parse_arm_name("baseline") is BASELINE
    arm = AdditiveDose("all13", 20)
    assert arm.name == "linker_all13_20"
    assert parse_arm_name(arm.name) == arm


@pytest.mark.parametrize(
    ("column", "value", "match"),
    (
        ("source_domain", "", "nonempty strings"),
        ("source_domain", 1, "nonempty strings"),
        ("source_domain", "unknown", "unknown source domains"),
        ("source_family", "gold", "unknown source families"),
        ("pair1", "", "nonempty strings"),
        ("label", True, "exact integral"),
        ("label", "1", "exact integral"),
        ("label", 1.0, "exact integral"),
        ("label", 2, "exact integral"),
    ),
)
def test_catalog_rejects_invalid_scalar_values(column: str, value: object, match: str) -> None:
    frame = _frame([("aminer", BASE_FAMILY, "a", "b", 0)])
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        validate_catalog(frame)


def test_catalog_requires_exact_columns_and_family_domains() -> None:
    frame = _frame([("aminer", BASE_FAMILY, "a", "b", 0)])
    with pytest.raises(ValueError, match="columns must be exactly"):
        validate_catalog(frame.assign(extra=1))
    with pytest.raises(ValueError, match="columns must be exactly"):
        validate_catalog(frame.loc[:, list(reversed(PAIR_COLUMNS))])
    with pytest.raises(ValueError, match="family/domain"):
        validate_catalog(_frame([("a_khan", BASE_FAMILY, "a", "b", 0)]))
    with pytest.raises(ValueError, match="family/domain"):
        validate_catalog(_frame([("aminer", LINKER_FAMILY, "a", "b", 0)]))


@pytest.mark.parametrize(
    ("rows", "match"),
    (
        ([("aminer", BASE_FAMILY, "a", "a", 0)], "self-pair"),
        ([("aminer", BASE_FAMILY, "b", "a", 0)], "pair1 < pair2"),
        (
            [
                ("aminer", BASE_FAMILY, "a", "b", 0),
                ("aminer", BASE_FAMILY, "a", "b", 0),
            ],
            "duplicate family",
        ),
        (
            [
                ("arnetminer", BASE_FAMILY, "a", "b", 0),
                ("arnetminer", LINKER_FAMILY, "a", "b", 1),
            ],
            "conflicting labels",
        ),
    ),
)
def test_catalog_rejects_invalid_pair_identities(
    rows: list[tuple[object, ...]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_catalog(_frame(rows))


def test_catalog_feature_rows_and_digest_follow_content() -> None:
    frame = _catalog()
    validated = validate_catalog(frame)
    assert validated["feature_row"].tolist() == list(range(len(frame)))
    assert pair_digest(validated) == pair_digest(frame.sample(frac=1, random_state=7))


def test_baseline_is_unchanged_lodo() -> None:
    selected, audit = select_pairs(validate_catalog(_catalog()), BASELINE, "aminer", 4)
    assert selected["source_family"].eq(BASE_FAMILY).all()
    assert selected["source_domain"].tolist() == ["arnetminer", "pubmed"]
    assert audit == {
        "heldout": "aminer",
        "base_digest": pair_digest(selected),
        "base_rows": 2,
        "training_digest": pair_digest(selected),
        "training_rows": 2,
        "linker_rows": 0,
        "linker_by_domain": {},
    }


def test_lodo_baseline_requires_both_labels() -> None:
    catalog = validate_catalog(
        _frame(
            [
                ("aminer", BASE_FAMILY, "a", "b", 0),
                ("arnetminer", BASE_FAMILY, "c", "d", 1),
            ]
        )
    )
    with pytest.raises(ValueError, match="both labels"):
        select_pairs(catalog, BASELINE, "aminer", 4)


def test_additive_selection_is_balanced_nested_and_shuffle_stable() -> None:
    catalog = validate_catalog(_catalog())
    small, small_audit = select_pairs(catalog, AdditiveDose("big7", 2), "aminer", 19)
    large, large_audit = select_pairs(
        catalog.sample(frac=1, random_state=3),
        AdditiveDose("big7", 4),
        "aminer",
        19,
    )
    repeated, _ = select_pairs(
        catalog.sample(frac=1, random_state=5),
        AdditiveDose("big7", 4),
        "aminer",
        19,
    )

    pd.testing.assert_frame_equal(large, repeated)
    small_linker = small.loc[small["source_family"].eq(LINKER_FAMILY)]
    large_linker = large.loc[large["source_family"].eq(LINKER_FAMILY)]
    assert set(small_linker["feature_row"]).issubset(set(large_linker["feature_row"]))
    assert small_linker.groupby("source_domain")["label"].value_counts().to_dict() == {
        ("a_khan", 0): 1,
        ("a_khan", 1): 1,
        ("a_silva", 0): 1,
        ("a_silva", 1): 1,
    }
    assert large_linker.groupby("source_domain")["label"].value_counts().to_dict() == {
        ("a_khan", 0): 2,
        ("a_khan", 1): 2,
        ("a_silva", 0): 2,
        ("a_silva", 1): 2,
    }
    assert small_audit["linker_rows"] == 4
    assert large_audit["linker_rows"] == 8
    assert large_audit["linker_by_domain"] == {
        "a_khan": 4,
        "a_silva": 4,
        "h_wang": 0,
        "j_smith": 0,
        "s_gupta": 0,
        "s_lee": 0,
        "s_park": 0,
    }


def test_additive_selection_does_not_backfill_or_overlap_base() -> None:
    selected, audit = select_pairs(
        validate_catalog(_catalog()),
        AdditiveDose("all13", 6),
        "aminer",
        11,
    )
    linker = selected.loc[selected["source_family"].eq(LINKER_FAMILY)]
    assert audit["linker_by_domain"]["arnetminer"] == 4
    assert linker.loc[linker["source_domain"].eq("arnetminer"), "label"].value_counts().to_dict() == {
        0: 2,
        1: 2,
    }
    assert not (linker["source_domain"].eq("arnetminer") & linker["pair1"].eq("a1") & linker["pair2"].eq("b1")).any()


def test_heldout_linker_domain_is_excluded() -> None:
    selected, audit = select_pairs(
        validate_catalog(_catalog()),
        AdditiveDose("big7", 2),
        "a_khan",
        1,
    )
    assert not selected["source_domain"].eq("a_khan").any()
    assert "a_khan" not in audit["linker_by_domain"]

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts._pair_ablation.modeling import (
    ANCHOR_UNIFORM_FAMILY,
    AUGMENTED_FAMILY,
    BALANCED_RANDOM_FAMILY,
    BASE_FAMILY,
    LINKER_BIG_POSITIVE_FAMILY,
    LINKER_PROXY_NEGATIVE_FAMILY,
    LINKER_PUBLIC_FAMILY,
    MEDLINE_FAMILY,
    AblationArm,
    AdditiveLinkerRecipe,
    ExactBudgetRecipe,
    ablation_arm_registry,
    additive_linker_arm,
    assemble_additive_linker_recipe,
    assemble_exact_budget_recipe,
    catalog_for_arm,
    default_ablation_arms,
    pair_catalog_diversity_diagnostics,
    pairwise_metrics,
    select_feature_rows,
    train_pairwise_models,
)
from scripts._pair_ablation.pair_sources import PAIR_COLUMNS, canonicalize_pairs


def _row(domain: str, family: str, pair1: str, pair2: str, label: int) -> dict[str, object]:
    return {
        "source_domain": domain,
        "source_family": family,
        "pair1": pair1,
        "pair2": pair2,
        "label": label,
        "label_rule": "fixture",
        "origin": "fixture",
        "group_id": f"{domain}:g",
    }


def _exact_arm(
    *auxiliary_families: str,
    balanced_linker: bool = False,
    capped_proxy_negative: bool = False,
) -> AblationArm:
    source_families = {BASE_FAMILY, *auxiliary_families}
    if balanced_linker:
        source_families.update(
            {
                LINKER_PUBLIC_FAMILY,
                LINKER_BIG_POSITIVE_FAMILY,
                LINKER_PROXY_NEGATIVE_FAMILY,
            }
        )
    if capped_proxy_negative:
        source_families.add(LINKER_PROXY_NEGATIVE_FAMILY)
    return AblationArm(
        "fixture_exact",
        frozenset(source_families),
        ExactBudgetRecipe(
            tuple(auxiliary_families),
            balanced_gold_dose=("low" if BALANCED_RANDOM_FAMILY in auxiliary_families else None),
            balanced_linker=balanced_linker,
            capped_proxy_negative=capped_proxy_negative,
        ),
    )


def test_default_arms_have_unique_names_and_a_gold_sampling_base() -> None:
    arms = default_ablation_arms()
    assert len({arm.name for arm in arms}) == len(arms)
    gold_bases = {
        BASE_FAMILY,
        ANCHOR_UNIFORM_FAMILY,
    }
    assert all(arm.source_families & gold_bases for arm in arms)

    by_name = {arm.name: arm for arm in arms}
    factorial = {
        "uniform_100k": (False, False, False),
        "uniform_budget_balanced_random": (True, False, False),
        "uniform_budget_pairwise_all": (False, True, False),
        "uniform_budget_linker_balanced": (False, False, True),
        "uniform_budget_balanced_plus_pairwise_all": (True, True, False),
        "uniform_budget_balanced_plus_linker_balanced": (True, False, True),
        "uniform_budget_pairwise_linker_balanced": (False, True, True),
        "uniform_budget_balanced_plus_pairwise_linker_balanced": (True, True, True),
    }
    assert set(by_name) == {
        *factorial,
        "uniform_budget_balanced_random_50k",
        "uniform_budget_balanced_random_100k",
        "uniform_budget_linker_proxy_negative_only",
    }
    for name, expected_factors in factorial.items():
        recipe = by_name[name].exact_budget_recipe
        auxiliary_families = () if recipe is None else recipe.auxiliary_families
        observed_factors = (
            BALANCED_RANDOM_FAMILY in auxiliary_families,
            {MEDLINE_FAMILY, AUGMENTED_FAMILY}.issubset(auxiliary_families),
            False if recipe is None else recipe.balanced_linker,
        )
        assert observed_factors == expected_factors
        if expected_factors[0]:
            assert recipe is not None and recipe.balanced_gold_dose == "low"

    pairwise = (BALANCED_RANDOM_FAMILY, MEDLINE_FAMILY, AUGMENTED_FAMILY)
    assert by_name["uniform_budget_balanced_plus_pairwise_all"].exact_budget_recipe == ExactBudgetRecipe(
        pairwise, balanced_gold_dose="low"
    )
    assert by_name["uniform_budget_balanced_plus_linker_balanced"].exact_budget_recipe == ExactBudgetRecipe(
        (BALANCED_RANDOM_FAMILY,),
        balanced_gold_dose="low",
        balanced_linker=True,
    )
    assert by_name["uniform_budget_balanced_plus_pairwise_linker_balanced"].exact_budget_recipe == ExactBudgetRecipe(
        pairwise, balanced_gold_dose="low", balanced_linker=True
    )
    assert by_name["uniform_budget_balanced_random_50k"].exact_budget_recipe == ExactBudgetRecipe(
        (BALANCED_RANDOM_FAMILY,), balanced_gold_dose="medium"
    )
    assert by_name["uniform_budget_balanced_random_100k"].exact_budget_recipe == ExactBudgetRecipe(
        (BALANCED_RANDOM_FAMILY,), balanced_gold_dose="max"
    )
    proxy_challenge = by_name["uniform_budget_linker_proxy_negative_only"]
    assert proxy_challenge.source_families == frozenset({BASE_FAMILY, LINKER_PROXY_NEGATIVE_FAMILY})
    assert proxy_challenge.exact_budget_recipe == ExactBudgetRecipe((), capped_proxy_negative=True)

    registry = {arm.name: arm for arm in ablation_arm_registry()}
    assert set(registry).difference(by_name) == {
        "uniform_budget_linker_balanced_50k",
        "uniform_budget_balanced_plus_linker_balanced_50k",
        "uniform_budget_pairwise_linker_balanced_50k",
        "uniform_budget_balanced_plus_pairwise_linker_balanced_50k",
    }
    assert all(
        registry[name].exact_budget_recipe is not None
        and registry[name].exact_budget_recipe.linker_pairs_per_domain == 50_000
        for name in set(registry).difference(by_name)
    )


def test_exact_budget_recipe_rejects_overlapping_linker_modes() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ExactBudgetRecipe((), balanced_linker=True, capped_proxy_negative=True)
    with pytest.raises(ValueError, match="uncapped auxiliary"):
        ExactBudgetRecipe((LINKER_PROXY_NEGATIVE_FAMILY,), capped_proxy_negative=True)
    with pytest.raises(ValueError, match="configured together"):
        ExactBudgetRecipe((BALANCED_RANDOM_FAMILY,))
    with pytest.raises(ValueError, match="B-only"):
        ExactBudgetRecipe(
            (BALANCED_RANDOM_FAMILY, MEDLINE_FAMILY),
            balanced_gold_dose="max",
        )
    with pytest.raises(ValueError, match="requires balanced_linker"):
        ExactBudgetRecipe((), linker_pairs_per_domain=50_000)


def test_additive_linker_recipe_requires_named_source_set_and_even_cap() -> None:
    arm = additive_linker_arm("big7", 2_500)
    assert arm.name == "uniform_100k_plus_linker_big7_2500"
    assert arm.exact_budget_recipe is None
    assert arm.additive_linker_recipe is not None
    assert arm.additive_linker_recipe.linker_pairs_per_domain == 2_500

    with pytest.raises(ValueError, match="positive even"):
        additive_linker_arm("big7", 625)
    with pytest.raises(ValueError, match="do not match"):
        AdditiveLinkerRecipe("big7", ("h_wang",), 2_500)


def test_catalog_for_arm_excludes_every_heldout_source() -> None:
    catalog = pd.DataFrame(
        [
            _row("qian", BASE_FAMILY, "a", "b", 0),
            _row("pubmed", BASE_FAMILY, "c", "d", 1),
            _row("medline", MEDLINE_FAMILY, "e", "f", 0),
        ],
        columns=list(PAIR_COLUMNS),
    )
    arm = AblationArm("with_medline", frozenset({BASE_FAMILY, MEDLINE_FAMILY}))
    selected, audit = catalog_for_arm(
        catalog,
        arm,
        held_out_domain="qian",
        random_seed=17,
        linker_pairs_per_domain=10,
    )
    assert set(selected["source_domain"]) == {"pubmed", "medline"}
    assert audit["mode"] == "additive"


def test_catalog_for_arm_filters_family_before_cross_family_deduplication() -> None:
    catalog = pd.DataFrame(
        [
            _row("qian", "historical_augmented", "a", "b", 1),
            _row("qian", BASE_FAMILY, "a", "b", 1),
            _row("pubmed", BASE_FAMILY, "c", "d", 0),
        ],
        columns=list(PAIR_COLUMNS),
    )
    selected, _audit = catalog_for_arm(
        catalog,
        AblationArm("baseline", frozenset({BASE_FAMILY})),
        held_out_domain="medline",
        random_seed=17,
        linker_pairs_per_domain=10,
    )
    assert len(selected) == 2


def test_catalog_for_arm_rejects_unknown_family() -> None:
    catalog = pd.DataFrame([_row("qian", BASE_FAMILY, "a", "b", 0)], columns=list(PAIR_COLUMNS))
    with pytest.raises(ValueError, match="unavailable source families"):
        catalog_for_arm(
            catalog,
            AblationArm("bad", frozenset({"missing"})),
            held_out_domain="pubmed",
            random_seed=17,
            linker_pairs_per_domain=10,
        )


def test_exact_budget_assembler_applies_lodo_before_computing_target() -> None:
    catalog = pd.DataFrame(
        [
            _row("qian", BASE_FAMILY, "q0", "q1", 0),
            _row("qian", BASE_FAMILY, "q2", "q3", 1),
            *[_row("pubmed", BASE_FAMILY, f"p{i}", f"p{i + 10}", i % 2) for i in range(4)],
            _row("qian", BALANCED_RANDOM_FAMILY, "qx", "qy", 0),
            _row("pubmed", BALANCED_RANDOM_FAMILY, "px", "py", 1),
            _row("medline", MEDLINE_FAMILY, "m0", "m1", 0),
        ],
        columns=list(PAIR_COLUMNS),
    )

    selected, audit = catalog_for_arm(
        catalog,
        _exact_arm(BALANCED_RANDOM_FAMILY, MEDLINE_FAMILY),
        held_out_domain="qian",
        random_seed=23,
        linker_pairs_per_domain=10,
        balanced_pairs_per_domain=2,
        balanced_pool_pairs_per_domain=2,
    )

    assert len(selected) == 4
    assert not selected["source_domain"].eq("qian").any()
    assert set(selected["source_family"]) >= {BALANCED_RANDOM_FAMILY, MEDLINE_FAMILY}
    assert audit["target_rows"] == 4
    assert audit["base_rows_after_lodo"] == 4
    assert audit["base_filler_rows"] == 2
    assert audit["held_out_rows"] == 0


def test_exact_budget_assembler_deduplicates_cross_family_pairs_with_auxiliary_priority() -> None:
    catalog = pd.DataFrame(
        [
            _row("pubmed", BASE_FAMILY, "a", "b", 1),
            _row("pubmed", BASE_FAMILY, "c", "d", 0),
            _row("pubmed", BASE_FAMILY, "e", "f", 0),
            _row("pubmed", BALANCED_RANDOM_FAMILY, "b", "a", 1),
            _row("pubmed", AUGMENTED_FAMILY, "a", "b", 1),
        ],
        columns=list(PAIR_COLUMNS),
    )

    selected, audit = assemble_exact_budget_recipe(
        catalog,
        ExactBudgetRecipe(
            (BALANCED_RANDOM_FAMILY, AUGMENTED_FAMILY),
            balanced_gold_dose="low",
        ),
        held_out_domain="medline",
        random_seed=7,
        linker_pairs_per_domain=10,
        balanced_pairs_per_domain=2,
        balanced_pool_pairs_per_domain=2,
    )

    duplicate = selected.loc[selected["pair1"].eq("a") & selected["pair2"].eq("b")]
    assert len(selected) == 3
    assert len(duplicate) == 1
    assert duplicate.iloc[0]["source_family"] == BALANCED_RANDOM_FAMILY
    assert audit["non_linker_duplicates_removed"] == 1
    assert audit["base_overlap_rows"] == 1

    conflicting = catalog.copy()
    conflicting.loc[conflicting["source_family"].eq(AUGMENTED_FAMILY), "label"] = 0
    with pytest.raises(ValueError, match="Conflicting labels"):
        assemble_exact_budget_recipe(
            conflicting,
            ExactBudgetRecipe(
                (BALANCED_RANDOM_FAMILY, AUGMENTED_FAMILY),
                balanced_gold_dose="low",
            ),
            held_out_domain="medline",
            random_seed=7,
            linker_pairs_per_domain=10,
            balanced_pairs_per_domain=2,
            balanced_pool_pairs_per_domain=2,
        )


def test_exact_budget_assembler_is_input_order_invariant_and_seed_deterministic() -> None:
    rows = [
        *[_row("pubmed", BASE_FAMILY, f"a{i}", f"b{i}", i % 2) for i in range(12)],
        _row("pubmed", BALANCED_RANDOM_FAMILY, "x0", "y0", 0),
        _row("pubmed", BALANCED_RANDOM_FAMILY, "x1", "y1", 1),
    ]
    catalog = pd.DataFrame(rows, columns=list(PAIR_COLUMNS))
    recipe = ExactBudgetRecipe((BALANCED_RANDOM_FAMILY,), balanced_gold_dose="low")

    first, first_audit = assemble_exact_budget_recipe(
        catalog,
        recipe,
        held_out_domain="qian",
        random_seed=31,
        linker_pairs_per_domain=10,
        balanced_pairs_per_domain=2,
        balanced_pool_pairs_per_domain=2,
    )
    shuffled, shuffled_audit = assemble_exact_budget_recipe(
        catalog.sample(frac=1, random_state=11),
        recipe,
        held_out_domain="qian",
        random_seed=31,
        linker_pairs_per_domain=10,
        balanced_pairs_per_domain=2,
        balanced_pool_pairs_per_domain=2,
    )
    another_seed, another_audit = assemble_exact_budget_recipe(
        catalog,
        recipe,
        held_out_domain="qian",
        random_seed=32,
        linker_pairs_per_domain=10,
        balanced_pairs_per_domain=2,
        balanced_pool_pairs_per_domain=2,
    )

    pd.testing.assert_frame_equal(first, shuffled)
    assert first_audit == shuffled_audit
    assert first_audit["selection_sha256"] != another_audit["selection_sha256"]
    assert len(first) == len(another_seed) == 12


def test_exact_budget_assembler_rejects_required_auxiliary_over_budget() -> None:
    catalog = pd.DataFrame(
        [
            _row("pubmed", BASE_FAMILY, "a", "b", 0),
            _row("pubmed", BASE_FAMILY, "c", "d", 1),
            *[_row("pubmed", BALANCED_RANDOM_FAMILY, f"x{i}", f"y{i}", i % 2) for i in range(3)],
        ],
        columns=list(PAIR_COLUMNS),
    )

    with pytest.raises(ValueError, match="required exact-budget auxiliaries exceed"):
        assemble_exact_budget_recipe(
            catalog,
            ExactBudgetRecipe((BALANCED_RANDOM_FAMILY,), balanced_gold_dose="low"),
            held_out_domain="qian",
            random_seed=3,
            linker_pairs_per_domain=10,
            balanced_pairs_per_domain=4,
            balanced_pool_pairs_per_domain=4,
        )


def _balanced_gold_fixture(*, base_rows: int, domains: dict[str, tuple[int, int]]) -> pd.DataFrame:
    rows = [_row("base", BASE_FAMILY, f"base-{i}", f"base-{i + 1000}", i % 2) for i in range(base_rows)]
    for domain, (negatives, positives) in domains.items():
        rows.extend(
            _row(domain, BALANCED_RANDOM_FAMILY, f"n-{domain}-{i}", f"nx-{domain}-{i}", 0) for i in range(negatives)
        )
        rows.extend(
            _row(domain, BALANCED_RANDOM_FAMILY, f"p-{domain}-{i}", f"px-{domain}-{i}", 1) for i in range(positives)
        )
    return pd.DataFrame(rows, columns=list(PAIR_COLUMNS))


def test_balanced_gold_doses_are_nested_lodo_safe_and_exact_budget() -> None:
    catalog = _balanced_gold_fixture(
        base_rows=100,
        domains={"d1": (50, 50), "held": (50, 50)},
    )
    selections: dict[int, set[tuple[str, str, str]]] = {}
    audits: dict[int, dict[str, object]] = {}
    recipes = {
        10: ExactBudgetRecipe((BALANCED_RANDOM_FAMILY,), balanced_gold_dose="low"),
        50: ExactBudgetRecipe((BALANCED_RANDOM_FAMILY,), balanced_gold_dose="medium"),
        100: ExactBudgetRecipe((BALANCED_RANDOM_FAMILY,), balanced_gold_dose="max"),
    }
    for dose, recipe in recipes.items():
        selected, audit = assemble_exact_budget_recipe(
            catalog,
            recipe,
            held_out_domain="held",
            random_seed=53,
            linker_pairs_per_domain=10,
            balanced_pairs_per_domain=dose,
            balanced_pool_pairs_per_domain=100,
        )
        balanced = selected.loc[selected["source_family"].eq(BALANCED_RANDOM_FAMILY)]
        selections[dose] = set(balanced[["source_domain", "pair1", "pair2"]].itertuples(index=False, name=None))
        audits[dose] = audit
        assert len(selected) == 100
        assert "held" not in set(selected["source_domain"])
        assert balanced["label"].value_counts().to_dict() == {0: dose // 2, 1: dose // 2}
        assert audit["balanced_requested_rows"] == dose
        assert audit["balanced_available_capped_rows"] == dose
        assert audit["balanced_selected_rows"] == dose
        assert not audit["balanced_source_limited"]

    assert selections[10] < selections[50] < selections[100]
    assert len({audit["balanced_pool_sha256"] for audit in audits.values()}) == 1
    assert len({audit["balanced_selection_sha256"] for audit in audits.values()}) == 3
    assert audits[10]["base_filler_rows"] == 90
    assert audits[50]["base_filler_rows"] == 50
    assert audits[100]["base_filler_rows"] == 0


def test_balanced_gold_prefix_has_no_majority_backfill_when_pool_is_source_limited() -> None:
    catalog = _balanced_gold_fixture(base_rows=10, domains={"d1": (8, 2)})

    selected, audit = assemble_exact_budget_recipe(
        catalog,
        ExactBudgetRecipe((BALANCED_RANDOM_FAMILY,), balanced_gold_dose="low"),
        held_out_domain="held",
        random_seed=59,
        linker_pairs_per_domain=10,
        balanced_pairs_per_domain=10,
        balanced_pool_pairs_per_domain=20,
    )

    balanced = selected.loc[selected["source_family"].eq(BALANCED_RANDOM_FAMILY)]
    assert len(selected) == 10
    assert balanced["label"].value_counts().to_dict() == {0: 5, 1: 2}
    assert audit["balanced_requested_rows"] == 10
    assert audit["balanced_available_capped_rows"] == 7
    assert audit["balanced_selected_rows"] == 7
    assert audit["balanced_source_limited"]
    assert audit["base_filler_rows"] == 3
    domain_audit = audit["balanced_domains"][0]
    assert not domain_audit["source_limited_negative"]
    assert domain_audit["source_limited_positive"]


def _linker_fixture(*, base_rows: int, availability: dict[str, tuple[int, int]]) -> pd.DataFrame:
    rows = [_row("base", BASE_FAMILY, f"base-{i}", f"base-{i + 1000}", i % 2) for i in range(base_rows)]
    for domain, (negatives, positives) in availability.items():
        rows.extend(
            _row(domain, LINKER_PROXY_NEGATIVE_FAMILY, f"n-{domain}-{i}", f"nx-{domain}-{i}", 0)
            for i in range(negatives)
        )
        rows.extend(
            _row(domain, LINKER_BIG_POSITIVE_FAMILY, f"p-{domain}-{i}", f"px-{domain}-{i}", 1) for i in range(positives)
        )
    return pd.DataFrame(rows, columns=list(PAIR_COLUMNS))


def test_additive_linker_assembler_preserves_base_balances_sources_and_applies_lodo() -> None:
    base = pd.DataFrame(
        [
            _row("pubmed", BASE_FAMILY, "a", "b", 0),
            _row("qian", BASE_FAMILY, "c", "d", 1),
            _row("qian", BASE_FAMILY, "e", "f", 0),
        ],
        columns=list(PAIR_COLUMNS),
    )
    linker = _linker_fixture(
        base_rows=0,
        availability={
            "a_khan": (4, 5),
            "a_silva": (5, 4),
            "h_wang": (6, 6),
            "j_smith": (6, 6),
            "s_gupta": (6, 6),
            "s_lee": (6, 6),
            "s_park": (6, 6),
        },
    )
    catalog = pd.concat([base, linker], ignore_index=True)
    selected, audit = assemble_additive_linker_recipe(
        catalog,
        additive_linker_arm("big7", 6).additive_linker_recipe,  # type: ignore[arg-type]
        held_out_domain="h_wang",
        random_seed=17,
    )

    selected_base = selected.loc[selected["source_family"].eq(BASE_FAMILY), list(PAIR_COLUMNS)]
    pd.testing.assert_frame_equal(
        selected_base.reset_index(drop=True),
        canonicalize_pairs(base).reset_index(drop=True),
    )
    linker_selected = selected.loc[
        selected["source_family"].isin({LINKER_BIG_POSITIVE_FAMILY, LINKER_PROXY_NEGATIVE_FAMILY})
    ]
    assert "h_wang" not in set(selected["source_domain"])
    assert linker_selected.groupby(["source_domain", "label"]).size().to_dict() == {
        (domain, label): 3
        for domain in ("a_khan", "a_silva", "j_smith", "s_gupta", "s_lee", "s_park")
        for label in (0, 1)
    }
    assert audit["mode"] == "additive_linker"
    assert audit["base_rows_after_lodo"] == 3
    assert audit["linker_selected_rows"] == 36
    assert audit["final_rows"] == 39
    assert audit["final_rows"] == audit["base_rows_after_lodo"] + audit["linker_selected_rows"]
    assert audit["held_out_rows"] == 0


def test_additive_linker_doses_are_nested_and_deduplicate_against_base() -> None:
    domains = ("a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park")
    catalog = _linker_fixture(base_rows=20, availability={domain: (8, 8) for domain in domains})
    overlap_rows = pd.DataFrame(
        [
            _row("a_khan", BASE_FAMILY, "overlap-a", "overlap-b", 0),
            _row("a_khan", LINKER_PROXY_NEGATIVE_FAMILY, "overlap-b", "overlap-a", 0),
        ],
        columns=list(PAIR_COLUMNS),
    )
    catalog = pd.concat([catalog, overlap_rows], ignore_index=True)

    low, low_audit = assemble_additive_linker_recipe(
        catalog,
        additive_linker_arm("big7", 4).additive_linker_recipe,  # type: ignore[arg-type]
        held_out_domain="held",
        random_seed=29,
    )
    high, high_audit = assemble_additive_linker_recipe(
        catalog,
        additive_linker_arm("big7", 8).additive_linker_recipe,  # type: ignore[arg-type]
        held_out_domain="held",
        random_seed=29,
    )
    linker_families = {LINKER_BIG_POSITIVE_FAMILY, LINKER_PROXY_NEGATIVE_FAMILY}
    low_keys = set(
        low.loc[low["source_family"].isin(linker_families), ["source_domain", "pair1", "pair2"]].itertuples(
            index=False,
            name=None,
        )
    )
    high_keys = set(
        high.loc[high["source_family"].isin(linker_families), ["source_domain", "pair1", "pair2"]].itertuples(
            index=False,
            name=None,
        )
    )

    assert low_keys < high_keys
    assert low_audit["base_pair_digest"] == high_audit["base_pair_digest"]
    assert low_audit["linker_base_overlap_rows"] == high_audit["linker_base_overlap_rows"] == 1
    assert not low.duplicated(["source_domain", "pair1", "pair2"]).any()
    assert not high.duplicated(["source_domain", "pair1", "pair2"]).any()


def test_balanced_linker_uses_shared_min_without_majority_backfill() -> None:
    catalog = _linker_fixture(base_rows=20, availability={"d1": (2, 5), "d2": (7, 1), "held": (4, 4)})

    selected, audit = assemble_exact_budget_recipe(
        catalog,
        ExactBudgetRecipe((), balanced_linker=True),
        held_out_domain="held",
        random_seed=13,
        linker_pairs_per_domain=6,
    )

    linker = selected.loc[selected["source_family"].isin({LINKER_BIG_POSITIVE_FAMILY, LINKER_PROXY_NEGATIVE_FAMILY})]
    counts = linker.groupby(["source_domain", "label"]).size().to_dict()
    assert counts == {("d1", 0): 2, ("d1", 1): 2, ("d2", 0): 1, ("d2", 1): 1}
    assert "held" not in set(selected["source_domain"])
    assert audit["linker_requested_rows"] == 6
    assert audit["linker_selected_rows"] == 6
    assert not audit["linker_capped"]


def test_balanced_linker_capacity_cap_is_domain_fair_and_label_balanced() -> None:
    catalog = _linker_fixture(base_rows=7, availability={"d1": (5, 5), "d2": (5, 5)})

    selected, audit = assemble_exact_budget_recipe(
        catalog,
        ExactBudgetRecipe((), balanced_linker=True),
        held_out_domain="held",
        random_seed=19,
        linker_pairs_per_domain=10,
    )

    linker_audit = {row["source_domain"]: row for row in audit["linker_domains"]}
    assert len(selected) == 7
    assert audit["linker_requested_rows"] == 20
    assert audit["linker_selected_rows"] == 6
    assert audit["linker_capped"]
    assert audit["base_filler_rows"] == 1
    assert {domain: row["selected_per_class"] for domain, row in linker_audit.items()} == {"d1": 2, "d2": 1}
    linker = selected.loc[selected["source_family"].isin({LINKER_BIG_POSITIVE_FAMILY, LINKER_PROXY_NEGATIVE_FAMILY})]
    assert linker["label"].value_counts().to_dict() == {0: 3, 1: 3}


def test_balanced_linker_doses_are_nested_when_capacity_does_not_bind() -> None:
    catalog = _linker_fixture(base_rows=80, availability={"d1": (30, 30)})
    recipe = ExactBudgetRecipe((), balanced_linker=True)
    dose_10, audit_10 = assemble_exact_budget_recipe(
        catalog,
        recipe,
        held_out_domain="held",
        random_seed=29,
        linker_pairs_per_domain=10,
    )
    dose_50, audit_50 = assemble_exact_budget_recipe(
        catalog,
        recipe,
        held_out_domain="held",
        random_seed=29,
        linker_pairs_per_domain=50,
    )

    linker_families = {LINKER_BIG_POSITIVE_FAMILY, LINKER_PROXY_NEGATIVE_FAMILY}
    keys_10 = set(
        dose_10.loc[dose_10["source_family"].isin(linker_families), ["source_domain", "pair1", "pair2"]].itertuples(
            index=False, name=None
        )
    )
    keys_50 = set(
        dose_50.loc[dose_50["source_family"].isin(linker_families), ["source_domain", "pair1", "pair2"]].itertuples(
            index=False, name=None
        )
    )
    assert keys_10 < keys_50
    assert audit_10["linker_selected_rows"] == 10
    assert audit_50["linker_selected_rows"] == 50
    json.dumps(audit_50, allow_nan=False)


def test_linker_recipe_override_beats_config_default_and_records_capacity_cap() -> None:
    catalog = pd.concat(
        [
            _balanced_gold_fixture(base_rows=20, domains={"gold": (2, 2)}),
            pd.DataFrame(
                [_row("medline", MEDLINE_FAMILY, f"m-{i}", f"mx-{i}", i % 2) for i in range(4)],
                columns=list(PAIR_COLUMNS),
            ),
            _linker_fixture(base_rows=0, availability={"linker": (20, 20)}),
        ],
        ignore_index=True,
    )
    recipe = ExactBudgetRecipe(
        (BALANCED_RANDOM_FAMILY, MEDLINE_FAMILY),
        balanced_gold_dose="low",
        balanced_linker=True,
        linker_pairs_per_domain=50,
    )

    selected, audit = assemble_exact_budget_recipe(
        catalog,
        recipe,
        held_out_domain="held",
        random_seed=61,
        linker_pairs_per_domain=10,
        balanced_pairs_per_domain=4,
        balanced_pool_pairs_per_domain=4,
    )

    assert len(selected) == 20
    assert len(selected.loc[selected["source_family"].eq(BALANCED_RANDOM_FAMILY)]) == 4
    assert len(selected.loc[selected["source_family"].eq(MEDLINE_FAMILY)]) == 4
    assert audit["linker_cap_per_domain"] == 50
    assert audit["linker_requested_rows"] == 40
    assert audit["linker_selected_rows"] == 12
    assert audit["linker_capped"]
    assert audit["base_filler_rows"] == 0


def test_capped_proxy_negative_is_deterministic_lodo_capped_and_never_backfills() -> None:
    catalog = _linker_fixture(
        base_rows=30,
        availability={"d1": (2, 4), "d2": (20, 4), "held": (10, 10)},
    )
    recipe = ExactBudgetRecipe((), capped_proxy_negative=True)

    first, audit = assemble_exact_budget_recipe(
        catalog,
        recipe,
        held_out_domain="held",
        random_seed=37,
        linker_pairs_per_domain=5,
    )
    shuffled, shuffled_audit = assemble_exact_budget_recipe(
        catalog.sample(frac=1, random_state=9),
        recipe,
        held_out_domain="held",
        random_seed=37,
        linker_pairs_per_domain=5,
    )
    another_seed, _ = assemble_exact_budget_recipe(
        catalog,
        recipe,
        held_out_domain="held",
        random_seed=38,
        linker_pairs_per_domain=5,
    )

    pd.testing.assert_frame_equal(first, shuffled)
    assert audit == shuffled_audit
    proxy = first.loc[first["source_family"].eq(LINKER_PROXY_NEGATIVE_FAMILY)]
    another_proxy = another_seed.loc[another_seed["source_family"].eq(LINKER_PROXY_NEGATIVE_FAMILY)]
    assert proxy.groupby("source_domain").size().to_dict() == {"d1": 2, "d2": 5}
    assert set(proxy["label"]) == {0}
    assert "held" not in set(first["source_domain"])
    assert len(first) == 30
    assert set(proxy[["source_domain", "pair1", "pair2"]].itertuples(index=False, name=None)) != set(
        another_proxy[["source_domain", "pair1", "pair2"]].itertuples(index=False, name=None)
    )
    assert audit["proxy_cap_per_domain"] == 5
    assert audit["proxy_requested_rows"] == 10
    assert audit["proxy_available_capped_rows"] == 7
    assert audit["proxy_selected_rows"] == 7
    assert audit["proxy_source_limited"]
    assert not audit["proxy_capacity_capped"]
    assert audit["base_filler_rows"] == 23
    domain_audit = {row["source_domain"]: row for row in audit["proxy_domains"]}
    assert domain_audit["d1"]["source_limited"]
    assert not domain_audit["d2"]["source_limited"]
    assert all(row["selected_rows"] <= row["cap_rows"] for row in domain_audit.values())
    json.dumps(audit, allow_nan=False)


def test_capped_proxy_negative_respects_exact_budget_capacity_fairly() -> None:
    catalog = _linker_fixture(base_rows=5, availability={"d1": (10, 2), "d2": (10, 2)})

    selected, audit = assemble_exact_budget_recipe(
        catalog,
        ExactBudgetRecipe((), capped_proxy_negative=True),
        held_out_domain="held",
        random_seed=41,
        linker_pairs_per_domain=4,
    )

    proxy = selected.loc[selected["source_family"].eq(LINKER_PROXY_NEGATIVE_FAMILY)]
    domain_audit = {row["source_domain"]: row for row in audit["proxy_domains"]}
    assert len(selected) == 5
    assert len(proxy) == 5
    assert proxy.groupby("source_domain").size().to_dict() == {"d1": 3, "d2": 2}
    assert audit["proxy_requested_rows"] == 8
    assert audit["proxy_available_capped_rows"] == 8
    assert audit["proxy_selected_rows"] == 5
    assert not audit["proxy_source_limited"]
    assert audit["proxy_capacity_capped"]
    assert audit["base_filler_rows"] == 0
    assert {domain: row["capacity_capped"] for domain, row in domain_audit.items()} == {
        "d1": True,
        "d2": True,
    }


def test_capped_proxy_negative_deduplicates_against_uniform_base() -> None:
    catalog = pd.DataFrame(
        [
            _row("d1", BASE_FAMILY, "a", "b", 0),
            _row("d1", BASE_FAMILY, "c", "d", 1),
            _row("d1", BASE_FAMILY, "e", "f", 0),
            _row("d1", LINKER_PROXY_NEGATIVE_FAMILY, "b", "a", 0),
            _row("d1", LINKER_PROXY_NEGATIVE_FAMILY, "x", "y", 0),
        ],
        columns=list(PAIR_COLUMNS),
    )

    selected, audit = assemble_exact_budget_recipe(
        catalog,
        ExactBudgetRecipe((), capped_proxy_negative=True),
        held_out_domain="held",
        random_seed=43,
        linker_pairs_per_domain=2,
    )

    assert len(selected) == 3
    assert not selected.duplicated(["source_domain", "pair1", "pair2"]).any()
    overlap = selected.loc[selected["pair1"].eq("a") & selected["pair2"].eq("b")]
    assert len(overlap) == 1
    assert overlap.iloc[0]["source_family"] == LINKER_PROXY_NEGATIVE_FAMILY
    assert audit["base_overlap_rows"] == 1


def test_capped_proxy_negative_rejects_nonzero_labels() -> None:
    catalog = pd.DataFrame(
        [
            _row("d1", BASE_FAMILY, "a", "b", 0),
            _row("d1", BASE_FAMILY, "c", "d", 1),
            _row("d1", LINKER_PROXY_NEGATIVE_FAMILY, "x", "y", 1),
        ],
        columns=list(PAIR_COLUMNS),
    )

    with pytest.raises(ValueError, match="nonzero label"):
        assemble_exact_budget_recipe(
            catalog,
            ExactBudgetRecipe((), capped_proxy_negative=True),
            held_out_domain="held",
            random_seed=47,
            linker_pairs_per_domain=2,
        )


def test_pairwise_metrics_reports_prevalence_and_scores() -> None:
    metrics = pairwise_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.2, 0.8, 0.9]),
        oracle_kind="gold",
    )
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["prevalence"] == 0.5
    with pytest.raises(ValueError, match="both classes"):
        pairwise_metrics(np.asarray([1, 1]), np.asarray([0.7, 0.8]), oracle_kind="proxy")


@pytest.mark.parametrize(
    "labels",
    (
        np.asarray([0.0, 0.5]),
        np.asarray([0.0, np.nan]),
        np.asarray([0.0, np.inf]),
    ),
)
def test_pairwise_metrics_rejects_non_binary_labels_before_int8_cast(labels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite exact binary values 0 or 1"):
        pairwise_metrics(labels, np.asarray([0.2, 0.8]), oracle_kind="gold")


@pytest.mark.parametrize(
    "labels",
    (
        np.asarray([0.0, 0.5]),
        np.asarray([0.0, np.nan]),
        np.asarray([0.0, np.inf]),
    ),
)
def test_pairwise_training_rejects_non_binary_labels_before_int8_cast(tmp_path, labels: np.ndarray) -> None:
    features = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="finite exact binary values 0 or 1"):
        train_pairwise_models(
            features,
            features,
            labels,
            main_featurizer_info=None,  # type: ignore[arg-type]
            nameless_featurizer_info=None,  # type: ignore[arg-type]
            donor_model_dir=tmp_path / "donor",
            output_dir=tmp_path / "output",
            n_jobs=1,
            random_seed=1,
        )


def test_select_feature_rows_uses_canonical_pair_orientation() -> None:
    catalog = pd.DataFrame([_row("qian", BASE_FAMILY, "b", "a", 1)], columns=list(PAIR_COLUMNS))
    indices = select_feature_rows(catalog, {("qian", "a", "b"): 17})
    np.testing.assert_array_equal(indices, np.asarray([17]))


def test_pair_catalog_diversity_diagnostics_reports_diversity_and_overlap() -> None:
    rows = [
        _row("qian", BASE_FAMILY, "a", "b", 1),
        _row("qian", BASE_FAMILY, "a", "c", 0),
        _row("qian", BASE_FAMILY, "d", "e", 0),
        _row("qian", "extra", "b", "a", 1),
        _row("qian", "extra", "x", "y", 0),
        _row("qian", "extra", "y", "x", 0),
        _row("pubmed", "extra", "a", "b", 1),
    ]
    rows[0]["label_rule"] = "same_cluster"
    rows[1]["label_rule"] = "different_cluster"
    rows[2]["label_rule"] = "different_cluster"
    rows[2]["group_id"] = "qian:g2"
    catalog = pd.DataFrame(rows, columns=list(PAIR_COLUMNS))

    diagnostics = pair_catalog_diversity_diagnostics(catalog, reference_family=BASE_FAMILY)
    json.dumps(diagnostics, allow_nan=False)
    for table in ("domain_family", "label_rules", "reference_overlap"):
        assert not pd.DataFrame(diagnostics[table]).to_csv(index=False).startswith("\n")

    domain_family = {
        (record["source_domain"], record["source_family"]): record for record in diagnostics["domain_family"]
    }
    baseline = domain_family[("qian", BASE_FAMILY)]
    assert baseline["rows"] == 3
    assert baseline["positives"] == 1
    assert baseline["negatives"] == 2
    assert baseline["unique_signatures"] == 5
    assert baseline["unique_group_ids"] == 2
    assert baseline["pair_degree_p50"] == 1.0
    assert baseline["pair_degree_max"] == 2

    overlap = {
        (record["source_domain"], record["source_family"]): record for record in diagnostics["reference_overlap"]
    }
    qian_extra = overlap[("qian", "extra")]
    assert qian_extra["source_pairs"] == 2
    assert qian_extra["reference_pairs"] == 3
    assert qian_extra["overlapping_pairs"] == 1
    assert qian_extra["source_overlap_fraction"] == 0.5
    assert qian_extra["reference_coverage_fraction"] == pytest.approx(1 / 3)
    assert overlap[("pubmed", "extra")]["overlapping_pairs"] == 0
    assert overlap[("pubmed", "extra")]["reference_coverage_fraction"] is None

    label_rules = {
        (record["source_domain"], record["source_family"], record["label_rule"]): record["rows"]
        for record in diagnostics["label_rules"]
    }
    assert label_rules[("qian", BASE_FAMILY, "different_cluster")] == 2
    assert label_rules[("qian", BASE_FAMILY, "same_cluster")] == 1


def test_pair_catalog_diversity_diagnostics_rejects_bad_reference_and_conflicts() -> None:
    catalog = pd.DataFrame([_row("qian", BASE_FAMILY, "a", "b", 0)], columns=list(PAIR_COLUMNS))
    with pytest.raises(ValueError, match="is absent"):
        pair_catalog_diversity_diagnostics(catalog, reference_family="missing")

    conflict = pd.DataFrame(
        [
            _row("qian", BASE_FAMILY, "a", "b", 0),
            _row("qian", BASE_FAMILY, "b", "a", 1),
        ],
        columns=list(PAIR_COLUMNS),
    )
    with pytest.raises(ValueError, match="conflicting labels"):
        pair_catalog_diversity_diagnostics(conflict, reference_family=BASE_FAMILY)

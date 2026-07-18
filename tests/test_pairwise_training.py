from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from s2and.pairwise_training import (
    BASE_FAMILY,
    BIG_LINKER_SOURCE_DOMAINS,
    LINKER_BIG_POSITIVE_FAMILY,
    LINKER_PROXY_NEGATIVE_FAMILY,
    AdditiveLinkerRecipe,
    PairwiseTrainingRecipe,
    pair_identity_digest,
    resolve_pairwise_training_recipe,
)


def _row(
    domain: str,
    family: str,
    pair1: str,
    pair2: str,
    label: int,
) -> dict[str, object]:
    return {
        "source_domain": domain,
        "source_family": family,
        "pair1": pair1,
        "pair2": pair2,
        "label": label,
        "label_rule": "fixture",
        "origin": "fixture",
        "group_id": f"{domain}:fixture",
    }


def _fixture_recipe() -> PairwiseTrainingRecipe:
    return PairwiseTrainingRecipe(
        name="fixture_big7",
        uniform_source_domains=("gold_a", "gold_b"),
        uniform_pairs_per_domain=2,
        pair_sampling_seed=1111,
        additive_linker=AdditiveLinkerRecipe(
            source_set="big7",
            source_domains=BIG_LINKER_SOURCE_DOMAINS,
            linker_pairs_per_domain=2,
        ),
    )


def _complete_catalog() -> pd.DataFrame:
    rows = [
        _row("gold_a", BASE_FAMILY, "a1", "a2", 1),
        _row("gold_a", BASE_FAMILY, "a1", "a3", 0),
        _row("gold_b", BASE_FAMILY, "b1", "b2", 1),
        _row("gold_b", BASE_FAMILY, "b1", "b3", 0),
    ]
    for domain in BIG_LINKER_SOURCE_DOMAINS:
        rows.extend(
            [
                _row(domain, LINKER_BIG_POSITIVE_FAMILY, f"{domain}_p0", f"{domain}_p1", 1),
                _row(domain, LINKER_BIG_POSITIVE_FAMILY, f"{domain}_p2", f"{domain}_p3", 1),
                _row(domain, LINKER_PROXY_NEGATIVE_FAMILY, f"{domain}_n0", f"{domain}_n1", 0),
                _row(domain, LINKER_PROXY_NEGATIVE_FAMILY, f"{domain}_n2", f"{domain}_n3", 0),
            ]
        )
    return pd.DataFrame(rows)


def test_resolve_pairwise_training_recipe_selects_exact_balanced_big7_rows() -> None:
    recipe = _fixture_recipe()
    catalog = _complete_catalog()

    result = resolve_pairwise_training_recipe(catalog, recipe)

    assert len(result.pairs) == recipe.nominal_training_rows == 18
    assert result.audit["base_rows_after_lodo"] == 4
    assert result.audit["linker_selected_rows"] == 14
    assert result.audit["training_pair_digest"] == pair_identity_digest(result.pairs)
    assert result.audit["recipe"] == recipe.as_dict()
    linker = result.pairs.loc[result.pairs["source_domain"].isin(BIG_LINKER_SOURCE_DOMAINS)]
    assert linker.groupby(["source_domain", "label"]).size().to_dict() == {
        (domain, label): 1 for domain in BIG_LINKER_SOURCE_DOMAINS for label in (0, 1)
    }


def test_resolve_pairwise_training_recipe_is_deterministic_and_seed_bound() -> None:
    recipe = _fixture_recipe()
    catalog = _complete_catalog()

    first = resolve_pairwise_training_recipe(catalog, recipe)
    repeated = resolve_pairwise_training_recipe(catalog.sample(frac=1, random_state=9), recipe)
    other_seed = resolve_pairwise_training_recipe(
        catalog,
        replace(recipe, pair_sampling_seed=2222),
    )

    assert first.audit["training_pair_digest"] == repeated.audit["training_pair_digest"]
    assert first.audit["training_pair_digest"] != other_seed.audit["training_pair_digest"]


def test_resolve_pairwise_training_recipe_rejects_incomplete_base() -> None:
    recipe = _fixture_recipe()
    catalog = _complete_catalog()
    incomplete = catalog.drop(
        catalog.index[catalog["source_domain"].eq("gold_b") & catalog["source_family"].eq(BASE_FAMILY)][0]
    )

    with pytest.raises(ValueError, match="uniform base does not match"):
        resolve_pairwise_training_recipe(incomplete, recipe)


def test_resolve_pairwise_training_recipe_rejects_source_limited_linker_domain() -> None:
    recipe = _fixture_recipe()
    catalog = _complete_catalog()
    limited = catalog.loc[
        ~(catalog["source_domain"].eq("h_wang") & catalog["source_family"].eq(LINKER_PROXY_NEGATIVE_FAMILY))
    ]

    with pytest.raises(ValueError, match="cannot satisfy"):
        resolve_pairwise_training_recipe(limited, recipe)


def test_pairwise_training_recipe_serializes_derived_quotas() -> None:
    recipe = _fixture_recipe()

    assert recipe.as_dict() == {
        "name": "fixture_big7",
        "uniform_source_domains": ["gold_a", "gold_b"],
        "uniform_pairs_per_domain": 2,
        "pair_sampling_seed": 1111,
        "additive_linker": {
            "source_set": "big7",
            "source_domains": list(BIG_LINKER_SOURCE_DOMAINS),
            "linker_pairs_per_domain": 2,
            "linker_pairs_per_label": 1,
        },
        "base_sampler": "uniform_within_blocks_without_replacement",
        "balancing": "linker_per_domain_shared_min_binary_no_backfill",
        "budget_policy": "additive_to_unchanged_uniform",
        "exclude_base_overlaps": True,
        "nominal_training_rows": 18,
    }

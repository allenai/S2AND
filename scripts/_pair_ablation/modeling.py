"""Fixed-model ablation arms, LightGBM fitting, and pairwise metrics."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from s2and.featurizer import FeaturizationInfo
from s2and.pairwise_training import (
    ADDITIVE_LINKER_SOURCE_SETS,
    ALL_LINKER_SOURCE_DOMAINS,
    BASE_FAMILY,
    BIG_LINKER_SOURCE_DOMAINS,
    LINKER_BIG_POSITIVE_FAMILY,
    LINKER_FAMILIES,
    LINKER_PROXY_NEGATIVE_FAMILY,
    LINKER_PUBLIC_FAMILY,
    PUBLIC_LINKER_SOURCE_DOMAINS,
    AdditiveLinkerRecipe,
    AdditiveLinkerSourceSet,
    additive_linker_recipe,
    assemble_additive_linker_pairs,
    select_balanced_linker_pairs,
)
from s2and.production_model import NativeLightGBMBinaryClassifier
from scripts._pair_ablation.pair_sources import PAIR_COLUMNS, canonicalize_pairs

BalancedGoldDose = Literal["low", "medium", "max"]

__all__ = [
    "ADDITIVE_LINKER_SOURCE_SETS",
    "ALL_LINKER_SOURCE_DOMAINS",
    "AdditiveLinkerRecipe",
    "AdditiveLinkerSourceSet",
    "BASE_FAMILY",
    "BIG_LINKER_SOURCE_DOMAINS",
    "LINKER_BIG_POSITIVE_FAMILY",
    "LINKER_FAMILIES",
    "LINKER_PROXY_NEGATIVE_FAMILY",
    "LINKER_PUBLIC_FAMILY",
    "PUBLIC_LINKER_SOURCE_DOMAINS",
]


@dataclass(frozen=True, slots=True)
class ExactBudgetRecipe:
    """Fold-local auxiliary sources that replace rows from the uniform base."""

    auxiliary_families: tuple[str, ...]
    balanced_gold_dose: BalancedGoldDose | None = None
    balanced_linker: bool = False
    linker_pairs_per_domain: int | None = None
    capped_proxy_negative: bool = False

    def __post_init__(self) -> None:
        if len(set(self.auxiliary_families)) != len(self.auxiliary_families):
            raise ValueError("exact-budget auxiliary families must be unique and ordered")
        has_balanced_gold = BALANCED_RANDOM_FAMILY in self.auxiliary_families
        if has_balanced_gold != (self.balanced_gold_dose is not None):
            raise ValueError("balanced gold family and balanced_gold_dose must be configured together")
        if self.balanced_linker and self.capped_proxy_negative:
            raise ValueError("balanced_linker and capped_proxy_negative are mutually exclusive")
        if self.linker_pairs_per_domain is not None:
            if not self.balanced_linker:
                raise ValueError("linker_pairs_per_domain override requires balanced_linker")
            if self.linker_pairs_per_domain <= 0 or self.linker_pairs_per_domain % 2:
                raise ValueError("linker_pairs_per_domain override must be a positive even value")
        if self.capped_proxy_negative and LINKER_PROXY_NEGATIVE_FAMILY in self.auxiliary_families:
            raise ValueError("capped_proxy_negative must not also be an uncapped auxiliary family")
        if self.balanced_gold_dose in {"medium", "max"}:
            other_auxiliaries = set(self.auxiliary_families).difference({BALANCED_RANDOM_FAMILY})
            if other_auxiliaries or self.balanced_linker or self.capped_proxy_negative:
                raise ValueError("medium and max balanced-gold doses are reserved for B-only arms")


@dataclass(frozen=True, slots=True)
class AblationArm:
    """One named pair-source recipe."""

    name: str
    source_families: frozenset[str]
    exact_budget_recipe: ExactBudgetRecipe | None = None
    additive_linker_recipe: AdditiveLinkerRecipe | None = None

    def __post_init__(self) -> None:
        if self.exact_budget_recipe is not None and self.additive_linker_recipe is not None:
            raise ValueError("an ablation arm cannot use exact-budget and additive-linker assembly together")


EXTRA_UNIFORM_FAMILY = "gold_cluster_uniform_extra"
ANCHOR_UNIFORM_FAMILY = "gold_cluster_anchor_uniform"
BALANCED_RANDOM_FAMILY = "gold_cluster_balanced_random"
MEDLINE_FAMILY = "pairwise_only"
AUGMENTED_FAMILY = "historical_augmented"
NAME_CHALLENGE_FAMILY = "gold_name_challenge"


def additive_linker_arm(source_set: AdditiveLinkerSourceSet, linker_pairs_per_domain: int) -> AblationArm:
    """Build one dynamically named low-dose additive linker arm."""

    recipe = additive_linker_recipe(source_set, linker_pairs_per_domain)
    return AblationArm(
        name=f"uniform_100k_plus_linker_{source_set}_{linker_pairs_per_domain}",
        source_families=frozenset({BASE_FAMILY, *LINKER_FAMILIES}),
        additive_linker_recipe=recipe,
    )


def ablation_arm_registry() -> tuple[AblationArm, ...]:
    """Return the frozen selectable registry, including optional dose arms."""

    base = frozenset({BASE_FAMILY})

    def exact_arm(
        name: str,
        *auxiliary_families: str,
        balanced_gold_dose: BalancedGoldDose | None = None,
        balanced_linker: bool = False,
        linker_pairs_per_domain: int | None = None,
        capped_proxy_negative: bool = False,
    ) -> AblationArm:
        families = base | frozenset(auxiliary_families)
        if balanced_linker:
            families |= frozenset(LINKER_FAMILIES)
        if capped_proxy_negative:
            families |= frozenset({LINKER_PROXY_NEGATIVE_FAMILY})
        return AblationArm(
            name,
            families,
            ExactBudgetRecipe(
                tuple(auxiliary_families),
                balanced_gold_dose=balanced_gold_dose,
                balanced_linker=balanced_linker,
                linker_pairs_per_domain=linker_pairs_per_domain,
                capped_proxy_negative=capped_proxy_negative,
            ),
        )

    pairwise = (MEDLINE_FAMILY, AUGMENTED_FAMILY)
    return (
        AblationArm("uniform_100k", base),
        exact_arm(
            "uniform_budget_balanced_random",
            BALANCED_RANDOM_FAMILY,
            balanced_gold_dose="low",
        ),
        exact_arm(
            "uniform_budget_balanced_random_50k",
            BALANCED_RANDOM_FAMILY,
            balanced_gold_dose="medium",
        ),
        exact_arm(
            "uniform_budget_balanced_random_100k",
            BALANCED_RANDOM_FAMILY,
            balanced_gold_dose="max",
        ),
        exact_arm("uniform_budget_pairwise_all", *pairwise),
        exact_arm("uniform_budget_linker_balanced", balanced_linker=True),
        exact_arm(
            "uniform_budget_balanced_plus_pairwise_all",
            BALANCED_RANDOM_FAMILY,
            *pairwise,
            balanced_gold_dose="low",
        ),
        exact_arm(
            "uniform_budget_balanced_plus_linker_balanced",
            BALANCED_RANDOM_FAMILY,
            balanced_gold_dose="low",
            balanced_linker=True,
        ),
        exact_arm(
            "uniform_budget_pairwise_linker_balanced",
            *pairwise,
            balanced_linker=True,
        ),
        exact_arm(
            "uniform_budget_balanced_plus_pairwise_linker_balanced",
            BALANCED_RANDOM_FAMILY,
            *pairwise,
            balanced_gold_dose="low",
            balanced_linker=True,
        ),
        exact_arm(
            "uniform_budget_linker_proxy_negative_only",
            capped_proxy_negative=True,
        ),
        exact_arm(
            "uniform_budget_linker_balanced_50k",
            balanced_linker=True,
            linker_pairs_per_domain=50_000,
        ),
        exact_arm(
            "uniform_budget_balanced_plus_linker_balanced_50k",
            BALANCED_RANDOM_FAMILY,
            balanced_gold_dose="low",
            balanced_linker=True,
            linker_pairs_per_domain=50_000,
        ),
        exact_arm(
            "uniform_budget_pairwise_linker_balanced_50k",
            *pairwise,
            balanced_linker=True,
            linker_pairs_per_domain=50_000,
        ),
        exact_arm(
            "uniform_budget_balanced_plus_pairwise_linker_balanced_50k",
            BALANCED_RANDOM_FAMILY,
            *pairwise,
            balanced_gold_dose="low",
            balanced_linker=True,
            linker_pairs_per_domain=50_000,
        ),
    )


def default_ablation_arms() -> tuple[AblationArm, ...]:
    """Return the primary 11-arm seed-1111 screen from the frozen registry."""

    return ablation_arm_registry()[:11]


def _stable_pair_rank(frame: pd.DataFrame, *, random_seed: int, scope: str) -> pd.DataFrame:
    """Return canonical rows in a deterministic content-hash order."""

    canonical = canonicalize_pairs(frame.loc[:, PAIR_COLUMNS])
    if canonical.empty:
        return canonical
    ranked = canonical.copy()
    ranked["_selection_rank"] = [
        hashlib.sha256(
            "\0".join(
                (
                    str(int(random_seed)),
                    scope,
                    str(source_domain),
                    str(int(label)),
                    str(pair1),
                    str(pair2),
                )
            ).encode("utf-8")
        ).hexdigest()
        for source_domain, pair1, pair2, label in ranked[["source_domain", "pair1", "pair2", "label"]].itertuples(
            index=False, name=None
        )
    ]
    return (
        ranked.sort_values(
            ["_selection_rank", "source_domain", "pair1", "pair2", "source_family"],
            kind="stable",
        )
        .drop(columns="_selection_rank")
        .reset_index(drop=True)
    )


def _pair_key_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return {
        (str(domain), str(pair1), str(pair2))
        for domain, pair1, pair2 in frame[["source_domain", "pair1", "pair2"]].itertuples(
            index=False,
            name=None,
        )
    }


def _without_pair_keys(frame: pd.DataFrame, excluded: set[tuple[str, str, str]]) -> pd.DataFrame:
    if frame.empty or not excluded:
        return frame.reset_index(drop=True)
    keep = [
        (str(domain), str(pair1), str(pair2)) not in excluded
        for domain, pair1, pair2 in frame[["source_domain", "pair1", "pair2"]].itertuples(
            index=False,
            name=None,
        )
    ]
    return frame.loc[keep, PAIR_COLUMNS].reset_index(drop=True)


def _selection_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    canonical = canonicalize_pairs(frame.loc[:, PAIR_COLUMNS]).sort_values(
        ["source_domain", "pair1", "pair2", "label"],
        kind="stable",
    )
    for domain, pair1, pair2, label in canonical[["source_domain", "pair1", "pair2", "label"]].itertuples(
        index=False, name=None
    ):
        digest.update(f"{domain}\0{pair1}\0{pair2}\0{int(label)}\n".encode())
    return digest.hexdigest()


def _domain_fair_quotas(requested: Mapping[str, int], total_units: int) -> dict[str, int]:
    """Water-fill requested units across domains in stable domain order."""

    allocated = {domain: 0 for domain in sorted(requested)}
    remaining = min(int(total_units), sum(requested.values()))
    while remaining:
        progressed = False
        for domain in allocated:
            if allocated[domain] >= requested[domain]:
                continue
            allocated[domain] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise AssertionError("domain-fair quota allocation stalled")
    return allocated


def _balanced_gold_selection(
    eligible_catalog: pd.DataFrame,
    *,
    dose_per_domain: int,
    pool_per_domain: int,
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int, int, str, str]:
    """Select nested per-domain, per-label prefixes from one fixed gold pool."""

    if dose_per_domain <= 0 or pool_per_domain <= 0:
        raise ValueError("balanced gold dose and pool sizes must be positive")
    if dose_per_domain % 2 or pool_per_domain % 2:
        raise ValueError("balanced gold dose and pool sizes must be even")
    if dose_per_domain > pool_per_domain:
        raise ValueError("balanced gold dose cannot exceed its fixed candidate pool")

    pool = canonicalize_pairs(
        eligible_catalog.loc[
            eligible_catalog["source_family"].eq(BALANCED_RANDOM_FAMILY),
            PAIR_COLUMNS,
        ]
    )
    domains = sorted(str(domain) for domain in pool["source_domain"].unique())
    requested_per_class = dose_per_domain // 2
    pool_per_class = pool_per_domain // 2
    selected_frames: list[pd.DataFrame] = []
    domain_audit: list[dict[str, Any]] = []
    available_capped_rows = 0

    for domain in domains:
        domain_rows = pool.loc[pool["source_domain"].eq(domain), PAIR_COLUMNS]
        counts = domain_rows["label"].value_counts().to_dict()
        available_negatives = int(counts.get(0, 0))
        available_positives = int(counts.get(1, 0))
        if available_negatives > pool_per_class or available_positives > pool_per_class:
            raise ValueError(
                "balanced gold catalog exceeds its declared fixed pool: "
                f"domain={domain!r}, negatives={available_negatives}, positives={available_positives}, "
                f"pool_per_class={pool_per_class}"
            )
        selected_negatives = min(requested_per_class, available_negatives)
        selected_positives = min(requested_per_class, available_positives)
        available_capped_rows += selected_negatives + selected_positives
        domain_pool_sha256 = _selection_digest(domain_rows)
        selected_by_label: dict[int, int] = {0: selected_negatives, 1: selected_positives}
        for label in (0, 1):
            selected_frames.append(
                _stable_pair_rank(
                    domain_rows.loc[domain_rows["label"].eq(label), PAIR_COLUMNS],
                    random_seed=random_seed,
                    scope=f"balanced_gold_prefix:{domain}:label={label}:pool={domain_pool_sha256}",
                ).head(selected_by_label[label])
            )
        domain_audit.append(
            {
                "source_domain": domain,
                "pool_rows": len(domain_rows),
                "pool_sha256": domain_pool_sha256,
                "available_negatives": available_negatives,
                "available_positives": available_positives,
                "requested_per_class": requested_per_class,
                "selected_negatives": selected_negatives,
                "selected_positives": selected_positives,
                "selected_rows": selected_negatives + selected_positives,
                "source_limited_negative": available_negatives < requested_per_class,
                "source_limited_positive": available_positives < requested_per_class,
            }
        )

    selected = (
        canonicalize_pairs(pd.concat(selected_frames, ignore_index=True))
        if selected_frames
        else canonicalize_pairs(pool.iloc[0:0])
    )
    if len(selected) != available_capped_rows:
        raise AssertionError("balanced gold prefix selection did not preserve label quotas")
    requested_rows = dose_per_domain * len(domains)
    return (
        selected,
        domain_audit,
        requested_rows,
        available_capped_rows,
        _selection_digest(pool),
        _selection_digest(selected),
    )


def _balanced_linker_selection(
    eligible_catalog: pd.DataFrame,
    *,
    excluded_pair_keys: set[tuple[str, str, str]],
    cap_per_domain: int,
    capacity_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Select shared-min positive/negative linker quotas within fold capacity."""

    selected, audit, requested_rows = select_balanced_linker_pairs(
        eligible_catalog.loc[:, PAIR_COLUMNS],
        excluded_pair_keys=excluded_pair_keys,
        cap_per_domain=cap_per_domain,
        capacity_rows=capacity_rows,
        random_seed=random_seed,
    )
    return canonicalize_pairs(selected.loc[:, PAIR_COLUMNS]), audit, requested_rows


def assemble_additive_linker_recipe(
    catalog: pd.DataFrame,
    recipe: AdditiveLinkerRecipe,
    *,
    held_out_domain: str,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append balanced linker prefixes while preserving every base row."""

    result = assemble_additive_linker_pairs(
        catalog.loc[:, PAIR_COLUMNS],
        recipe,
        held_out_domain=held_out_domain,
        random_seed=random_seed,
    )
    return canonicalize_pairs(result.pairs.loc[:, PAIR_COLUMNS]), result.audit


def _capped_proxy_negative_selection(
    eligible_catalog: pd.DataFrame,
    *,
    excluded_pair_keys: set[tuple[str, str, str]],
    cap_per_domain: int,
    capacity_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int, int]:
    """Select deterministic negative-only quotas without cross-domain backfill."""

    if cap_per_domain < 0:
        raise ValueError("linker_pairs_per_domain must be non-negative")
    raw = canonicalize_pairs(
        eligible_catalog.loc[
            eligible_catalog["source_family"].eq(LINKER_PROXY_NEGATIVE_FAMILY),
            PAIR_COLUMNS,
        ]
    )
    if not raw.empty and bool(raw["label"].ne(0).any()):
        raise ValueError("linker proxy-negative source family contains a nonzero label")
    domains = sorted(str(domain) for domain in raw["source_domain"].unique())
    proxy = _without_pair_keys(raw, excluded_pair_keys)
    available = {domain: int(proxy["source_domain"].eq(domain).sum()) for domain in domains}
    availability_capped = {
        domain: min(cap_per_domain, domain_available) for domain, domain_available in available.items()
    }
    allocations = _domain_fair_quotas(availability_capped, max(0, capacity_rows))

    selected_frames: list[pd.DataFrame] = []
    domain_audit: list[dict[str, Any]] = []
    for domain in domains:
        selected_rows = allocations[domain]
        domain_rows = proxy.loc[proxy["source_domain"].eq(domain), PAIR_COLUMNS]
        selected_frames.append(
            _stable_pair_rank(
                domain_rows,
                random_seed=random_seed,
                scope=f"capped_proxy_negative:{domain}",
            ).head(selected_rows)
        )
        domain_audit.append(
            {
                "source_domain": domain,
                "available_rows": available[domain],
                "cap_rows": cap_per_domain,
                "available_capped_rows": availability_capped[domain],
                "selected_rows": selected_rows,
                "source_limited": available[domain] < cap_per_domain,
                "capacity_capped": selected_rows < availability_capped[domain],
            }
        )

    selected = (
        canonicalize_pairs(pd.concat(selected_frames, ignore_index=True))
        if selected_frames
        else canonicalize_pairs(raw.iloc[0:0])
    )
    if len(selected) != sum(allocations.values()):
        raise AssertionError("capped proxy-negative selection did not preserve domain quotas")
    nominal_requested_rows = cap_per_domain * len(domains)
    available_capped_rows = sum(availability_capped.values())
    return selected, domain_audit, nominal_requested_rows, available_capped_rows


def assemble_exact_budget_recipe(
    catalog: pd.DataFrame,
    recipe: ExactBudgetRecipe,
    *,
    held_out_domain: str,
    random_seed: int,
    linker_pairs_per_domain: int,
    balanced_pairs_per_domain: int | None = None,
    balanced_pool_pairs_per_domain: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble one fold after LODO to exactly match its uniform-base row budget."""

    raw = catalog.loc[:, PAIR_COLUMNS]
    eligible = raw.loc[~raw["source_domain"].eq(held_out_domain), PAIR_COLUMNS]
    base = canonicalize_pairs(eligible.loc[eligible["source_family"].eq(BASE_FAMILY), PAIR_COLUMNS])
    target_rows = len(base)
    if target_rows == 0:
        raise ValueError(f"uniform base is empty after holding out {held_out_domain!r}")

    balanced = canonicalize_pairs(base.iloc[0:0])
    balanced_domain_audit: list[dict[str, Any]] = []
    balanced_requested_rows = 0
    balanced_available_capped_rows = 0
    balanced_pool_sha256: str | None = None
    balanced_selection_sha256: str | None = None
    if recipe.balanced_gold_dose is not None:
        if balanced_pairs_per_domain is None or balanced_pool_pairs_per_domain is None:
            raise ValueError("balanced gold recipes require explicit dose and pool sizes")
        (
            balanced,
            balanced_domain_audit,
            balanced_requested_rows,
            balanced_available_capped_rows,
            balanced_pool_sha256,
            balanced_selection_sha256,
        ) = _balanced_gold_selection(
            eligible,
            dose_per_domain=balanced_pairs_per_domain,
            pool_per_domain=balanced_pool_pairs_per_domain,
            random_seed=random_seed,
        )
        if recipe.balanced_gold_dose == "max" and len(balanced) > target_rows:
            raise ValueError(
                "max balanced-gold dose does not fit the B-only exact budget: "
                f"selected={len(balanced)}, target={target_rows}"
            )

    auxiliary_frames: list[pd.DataFrame] = [balanced]
    auxiliary_input_rows = len(balanced)
    for family in recipe.auxiliary_families:
        if family == BALANCED_RANDOM_FAMILY:
            continue
        family_rows = canonicalize_pairs(eligible.loc[eligible["source_family"].eq(family), PAIR_COLUMNS])
        auxiliary_input_rows += len(family_rows)
        auxiliary_frames.append(family_rows)
    non_linker = canonicalize_pairs(pd.concat(auxiliary_frames, ignore_index=True))
    non_linker = _stable_pair_rank(
        non_linker,
        random_seed=random_seed,
        scope=f"required_auxiliary:heldout={held_out_domain}",
    )
    if len(non_linker) > target_rows:
        raise ValueError(
            "required exact-budget auxiliaries exceed the held-out uniform budget: "
            f"required={len(non_linker)}, target={target_rows}, "
            f"balanced_gold_dose={recipe.balanced_gold_dose!r}"
        )

    non_linker_keys = _pair_key_set(non_linker)
    linker = canonicalize_pairs(base.iloc[0:0])
    linker_domain_audit: list[dict[str, Any]] = []
    linker_requested_rows = 0
    effective_linker_pairs_per_domain = (
        recipe.linker_pairs_per_domain if recipe.linker_pairs_per_domain is not None else linker_pairs_per_domain
    )
    if recipe.balanced_linker:
        linker, linker_domain_audit, linker_requested_rows = _balanced_linker_selection(
            eligible,
            excluded_pair_keys=non_linker_keys,
            cap_per_domain=effective_linker_pairs_per_domain,
            capacity_rows=target_rows - len(non_linker),
            random_seed=random_seed,
        )

    proxy = canonicalize_pairs(base.iloc[0:0])
    proxy_domain_audit: list[dict[str, Any]] = []
    proxy_requested_rows = 0
    proxy_available_capped_rows = 0
    if recipe.capped_proxy_negative:
        proxy, proxy_domain_audit, proxy_requested_rows, proxy_available_capped_rows = _capped_proxy_negative_selection(
            eligible,
            excluded_pair_keys=non_linker_keys,
            cap_per_domain=linker_pairs_per_domain,
            capacity_rows=target_rows - len(non_linker),
            random_seed=random_seed,
        )

    auxiliary = canonicalize_pairs(pd.concat([non_linker, linker, proxy], ignore_index=True))
    if len(auxiliary) != len(non_linker) + len(linker) + len(proxy):
        raise AssertionError("exact-budget auxiliary selections overlap")
    auxiliary_keys = _pair_key_set(auxiliary)
    base_keys = _pair_key_set(base)
    base_overlap_rows = len(auxiliary_keys & base_keys)
    filler_rows = target_rows - len(auxiliary)
    base_candidates = _without_pair_keys(base, auxiliary_keys)
    if len(base_candidates) < filler_rows:
        raise AssertionError(
            f"uniform base cannot fill exact budget: required={filler_rows}, available={len(base_candidates)}"
        )
    filler = _stable_pair_rank(
        base_candidates,
        random_seed=random_seed,
        scope=f"uniform_base_filler:heldout={held_out_domain}",
    ).head(filler_rows)
    selected = canonicalize_pairs(pd.concat([auxiliary, filler], ignore_index=True))
    held_out_rows = int(selected["source_domain"].eq(held_out_domain).sum())
    if len(selected) != target_rows or held_out_rows:
        raise AssertionError(
            "exact-budget assembly invariant failed: "
            f"target={target_rows}, selected={len(selected)}, held_out_rows={held_out_rows}"
        )

    audit = {
        "mode": "exact_budget",
        "held_out_domain": held_out_domain,
        "target_rows": target_rows,
        "base_rows_after_lodo": len(base),
        "auxiliary_families": list(recipe.auxiliary_families),
        "auxiliary_input_rows": auxiliary_input_rows,
        "non_linker_auxiliary_rows": len(non_linker),
        "non_linker_duplicates_removed": auxiliary_input_rows - len(non_linker),
        "balanced_gold_dose": recipe.balanced_gold_dose,
        "balanced_pool_pairs_per_domain": (
            balanced_pool_pairs_per_domain if recipe.balanced_gold_dose is not None else None
        ),
        "balanced_pairs_per_domain": (balanced_pairs_per_domain if recipe.balanced_gold_dose is not None else None),
        "balanced_requested_rows": balanced_requested_rows,
        "balanced_available_capped_rows": balanced_available_capped_rows,
        "balanced_selected_rows": len(balanced),
        "balanced_source_limited": balanced_available_capped_rows < balanced_requested_rows,
        "balanced_pool_sha256": balanced_pool_sha256,
        "balanced_selection_sha256": balanced_selection_sha256,
        "balanced_domains": balanced_domain_audit,
        "balanced_linker": recipe.balanced_linker,
        "linker_cap_per_domain": effective_linker_pairs_per_domain if recipe.balanced_linker else None,
        "linker_nominal_requested_rows": (
            effective_linker_pairs_per_domain * len(linker_domain_audit) if recipe.balanced_linker else 0
        ),
        "linker_requested_rows": linker_requested_rows,
        "linker_selected_rows": len(linker),
        "linker_source_limited": (
            recipe.balanced_linker
            and linker_requested_rows < effective_linker_pairs_per_domain * len(linker_domain_audit)
        ),
        "linker_capacity_capped": len(linker) < linker_requested_rows,
        "linker_capped": len(linker) < linker_requested_rows,
        "linker_domains": linker_domain_audit,
        "capped_proxy_negative": recipe.capped_proxy_negative,
        "proxy_cap_per_domain": linker_pairs_per_domain if recipe.capped_proxy_negative else None,
        "proxy_requested_rows": proxy_requested_rows,
        "proxy_available_capped_rows": proxy_available_capped_rows,
        "proxy_selected_rows": len(proxy),
        "proxy_source_limited": proxy_available_capped_rows < proxy_requested_rows,
        "proxy_capacity_capped": len(proxy) < proxy_available_capped_rows,
        "proxy_domains": proxy_domain_audit,
        "auxiliary_rows": len(auxiliary),
        "base_overlap_rows": base_overlap_rows,
        "base_filler_rows": len(filler),
        "final_rows": len(selected),
        "held_out_rows": held_out_rows,
        "selection_sha256": _selection_digest(selected),
    }
    return selected.reset_index(drop=True), audit


def catalog_for_arm(
    catalog: pd.DataFrame,
    arm: AblationArm,
    *,
    held_out_domain: str,
    random_seed: int,
    linker_pairs_per_domain: int,
    balanced_pairs_per_domain: int | None = None,
    balanced_pool_pairs_per_domain: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select one arm after LODO and return its auditable fold recipe."""

    raw = catalog.loc[:, PAIR_COLUMNS]
    unknown_families = sorted(arm.source_families.difference(set(raw["source_family"].astype(str))))
    if unknown_families:
        raise ValueError(f"arm={arm.name!r} references unavailable source families: {unknown_families}")
    if arm.additive_linker_recipe is not None:
        required_families = frozenset({BASE_FAMILY, *LINKER_FAMILIES})
        if arm.source_families != required_families:
            raise ValueError(
                f"arm={arm.name!r} additive-linker metadata disagrees with source_families: "
                f"required={sorted(required_families)}, observed={sorted(arm.source_families)}"
            )
        return assemble_additive_linker_recipe(
            raw,
            arm.additive_linker_recipe,
            held_out_domain=held_out_domain,
            random_seed=random_seed,
        )
    if arm.exact_budget_recipe is not None:
        required_families = {BASE_FAMILY, *arm.exact_budget_recipe.auxiliary_families}
        if arm.exact_budget_recipe.balanced_linker:
            required_families.update(LINKER_FAMILIES)
        if arm.exact_budget_recipe.capped_proxy_negative:
            required_families.add(LINKER_PROXY_NEGATIVE_FAMILY)
        if arm.source_families != frozenset(required_families):
            raise ValueError(
                f"arm={arm.name!r} exact-budget metadata disagrees with source_families: "
                f"required={sorted(required_families)}, observed={sorted(arm.source_families)}"
            )
        return assemble_exact_budget_recipe(
            raw,
            arm.exact_budget_recipe,
            held_out_domain=held_out_domain,
            random_seed=random_seed,
            linker_pairs_per_domain=linker_pairs_per_domain,
            balanced_pairs_per_domain=balanced_pairs_per_domain,
            balanced_pool_pairs_per_domain=balanced_pool_pairs_per_domain,
        )
    selected = raw.loc[
        raw["source_family"].isin(arm.source_families) & ~raw["source_domain"].eq(held_out_domain),
        PAIR_COLUMNS,
    ]
    selected = canonicalize_pairs(selected)
    if bool(selected["source_domain"].eq(held_out_domain).any()):
        raise AssertionError(f"held-out domain leaked into arm={arm.name!r}: {held_out_domain}")
    if selected.empty:
        raise ValueError(f"arm={arm.name!r} has no training pairs after holding out {held_out_domain!r}")
    selected = selected.reset_index(drop=True)
    return selected, {
        "mode": "additive",
        "held_out_domain": held_out_domain,
        "final_rows": len(selected),
        "held_out_rows": 0,
        "selection_sha256": _selection_digest(selected),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def donor_lightgbm_params(model_path: str | Path, *, estimator_scale: float = 1.0) -> dict[str, Any]:
    """Translate a released native booster into sklearn training parameters."""

    if estimator_scale <= 0:
        raise ValueError("estimator_scale must be positive")
    params = lgb.Booster(model_file=str(model_path)).params
    iterations = max(2, round(int(params["num_iterations"]) * estimator_scale))
    return {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": iterations,
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params["max_depth"]),
        "min_child_samples": int(params["min_data_in_leaf"]),
        "min_child_weight": float(params["min_sum_hessian_in_leaf"]),
        "subsample": float(params["bagging_fraction"]),
        "subsample_freq": int(params["bagging_freq"]),
        "colsample_bytree": float(params["feature_fraction"]),
        "reg_alpha": float(params["lambda_l1"]),
        "reg_lambda": float(params["lambda_l2"]),
        "min_split_gain": float(params["min_gain_to_split"]),
        "monotone_penalty": float(params.get("monotone_penalty", 0.0)),
    }


@dataclass(frozen=True, slots=True)
class TrainedPairwiseModels:
    """Saved main and nameless model paths plus Rust-native scoring wrappers."""

    main_path: Path
    nameless_path: Path
    main: NativeLightGBMBinaryClassifier
    nameless: NativeLightGBMBinaryClassifier
    metadata: dict[str, Any]


def _fit_one_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    featurizer_info: FeaturizationInfo,
    donor_path: Path,
    output_path: Path,
    n_jobs: int,
    random_seed: int,
    estimator_scale: float,
) -> tuple[NativeLightGBMBinaryClassifier, dict[str, Any]]:
    params = donor_lightgbm_params(donor_path, estimator_scale=estimator_scale)
    params.update(
        {
            "n_jobs": int(n_jobs),
            "random_state": int(random_seed),
            "tree_learner": "data",
            "verbosity": -1,
            "monotone_constraints": featurizer_info.lightgbm_monotone_constraints,
            "monotone_constraints_method": "advanced",
        }
    )
    model = LGBMClassifier(**params)
    model.fit(features, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output_path))
    native = NativeLightGBMBinaryClassifier(output_path, n_jobs=n_jobs, n_features=features.shape[1])
    metadata = {
        "donor_path": str(donor_path),
        "donor_sha256": _sha256_file(donor_path),
        "model_path": str(output_path),
        "model_sha256": _sha256_file(output_path),
        "feature_count": int(features.shape[1]),
        "parameters": params,
    }
    return native, metadata


def _validated_binary_labels(labels: np.ndarray, *, context: str) -> np.ndarray:
    """Validate exact finite binary values before narrowing them to int8."""

    raw = np.asarray(labels)
    if raw.ndim != 1:
        raise ValueError(f"{context} labels must be a one-dimensional vector")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"{context} labels must be finite exact binary values 0 or 1")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} labels must be finite exact binary values 0 or 1") from exc
    if not np.isfinite(numeric).all() or not bool(np.isin(numeric, (0.0, 1.0)).all()):
        raise ValueError(f"{context} labels must be finite exact binary values 0 or 1")
    return numeric.astype(np.int8)


def train_pairwise_models(
    main_features: np.ndarray,
    nameless_features: np.ndarray,
    labels: np.ndarray,
    *,
    main_featurizer_info: FeaturizationInfo,
    nameless_featurizer_info: FeaturizationInfo,
    donor_model_dir: str | Path,
    output_dir: str | Path,
    n_jobs: int,
    random_seed: int,
    estimator_scale: float = 1.0,
) -> TrainedPairwiseModels:
    """Fit both production pairwise views with one fixed donor configuration."""

    main = np.asarray(main_features)
    nameless = np.asarray(nameless_features)
    raw_target = np.asarray(labels)
    if main.ndim != 2 or nameless.ndim != 2 or raw_target.ndim != 1:
        raise ValueError("pairwise training arrays must be two matrices and one label vector")
    if len(main) != len(nameless) or len(main) != len(raw_target):
        raise ValueError("pairwise training arrays must have equal row counts")
    target = _validated_binary_labels(raw_target, context="pairwise training")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("pairwise training labels must contain both classes")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")

    donor_root = Path(donor_model_dir)
    output_root = Path(output_dir)
    main_native, main_metadata = _fit_one_model(
        main,
        target,
        featurizer_info=main_featurizer_info,
        donor_path=donor_root / "main.lgb",
        output_path=output_root / "main.lgb",
        n_jobs=n_jobs,
        random_seed=random_seed,
        estimator_scale=estimator_scale,
    )
    nameless_native, nameless_metadata = _fit_one_model(
        nameless,
        target,
        featurizer_info=nameless_featurizer_info,
        donor_path=donor_root / "nameless.lgb",
        output_path=output_root / "nameless.lgb",
        n_jobs=n_jobs,
        random_seed=random_seed,
        estimator_scale=estimator_scale,
    )
    return TrainedPairwiseModels(
        output_root / "main.lgb",
        output_root / "nameless.lgb",
        main_native,
        nameless_native,
        {"main": main_metadata, "nameless": nameless_metadata},
    )


def load_pairwise_models(model_dir: str | Path, *, n_jobs: int) -> TrainedPairwiseModels:
    """Load an already-saved ablation model pair through Rust-native scorers."""

    root = Path(model_dir)
    main_path = root / "main.lgb"
    nameless_path = root / "nameless.lgb"
    return TrainedPairwiseModels(
        main_path,
        nameless_path,
        NativeLightGBMBinaryClassifier(main_path, n_jobs=n_jobs),
        NativeLightGBMBinaryClassifier(nameless_path, n_jobs=n_jobs),
        {},
    )


def averaged_positive_probability(
    models: TrainedPairwiseModels,
    main_features: np.ndarray,
    nameless_features: np.ndarray,
) -> np.ndarray:
    """Score both released-model views in Rust and average positive probabilities."""

    main = np.ascontiguousarray(main_features)
    nameless = np.ascontiguousarray(nameless_features)
    if len(main) != len(nameless):
        raise ValueError("main and nameless evaluation matrices must have equal row counts")
    return (models.main.predict_proba_positive(main) + models.nameless.predict_proba_positive(nameless)) / 2.0


def pairwise_metrics(labels: np.ndarray, positive_probability: np.ndarray, *, oracle_kind: str) -> dict[str, Any]:
    """Return AUROC/AUPRC with the prevalence needed to interpret AUPRC."""

    raw_target = np.asarray(labels)
    probability = np.asarray(positive_probability, dtype=np.float64)
    if raw_target.ndim != 1 or probability.ndim != 1 or len(raw_target) != len(probability):
        raise ValueError("pairwise metric labels and probabilities must be equal-length vectors")
    target = _validated_binary_labels(raw_target, context="pairwise metric")
    classes = set(np.unique(target))
    if classes != {0, 1}:
        raise ValueError(f"AUROC/AUPRC require both classes, observed={sorted(classes)}")
    if not np.isfinite(probability).all() or bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("pairwise probabilities must be finite values in [0, 1]")
    positives = int(target.sum())
    return {
        "oracle_kind": str(oracle_kind),
        "rows": int(len(target)),
        "positives": positives,
        "negatives": int(len(target) - positives),
        "prevalence": float(positives / len(target)),
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
    }


def source_counts(catalog: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize training rows and class balance by domain and source family."""

    canonical_frames = [
        canonicalize_pairs(frame.loc[:, PAIR_COLUMNS]) for _, frame in catalog.groupby("source_family", sort=False)
    ]
    canonical = pd.concat(canonical_frames, ignore_index=True) if canonical_frames else catalog.loc[[], PAIR_COLUMNS]
    grouped = canonical.groupby(["source_domain", "source_family"], sort=True)["label"]
    output = []
    for (domain, family), labels in grouped:
        positives = int(labels.sum())
        output.append(
            {
                "source_domain": str(domain),
                "source_family": str(family),
                "rows": int(len(labels)),
                "positives": positives,
                "negatives": int(len(labels) - positives),
            }
        )
    return output


def _canonical_diagnostic_view(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return a vectorized, within-family canonical view for diagnostics."""

    missing = sorted(set(PAIR_COLUMNS) - set(catalog.columns))
    if missing:
        raise ValueError(f"pair catalog is missing columns: {missing}")
    diagnostic_columns = (
        "source_domain",
        "source_family",
        "pair1",
        "pair2",
        "label",
        "label_rule",
        "group_id",
    )
    frame = catalog.loc[:, diagnostic_columns].copy()
    text_columns = ("source_domain", "source_family", "pair1", "pair2", "label_rule", "group_id")
    for column in text_columns:
        if bool(frame[column].isna().any()):
            raise ValueError(f"pair catalog column {column!r} contains missing values")
        frame[column] = frame[column].astype(str).str.strip()
        if bool(frame[column].eq("").any()):
            raise ValueError(f"pair catalog column {column!r} contains empty values")

    first = frame["pair1"].where(frame["pair1"] <= frame["pair2"], frame["pair2"])
    second = frame["pair2"].where(frame["pair1"] <= frame["pair2"], frame["pair1"])
    if bool(first.eq(second).any()):
        raise ValueError("pair catalog contains self-pairs")
    frame["pair1"] = first
    frame["pair2"] = second

    numeric_labels = pd.to_numeric(frame["label"], errors="raise")
    if not bool(numeric_labels.isin((0, 1)).all()):
        raise ValueError("pair catalog labels must be binary")
    frame["label"] = numeric_labels.astype("int8")

    pair_key = ["source_domain", "source_family", "pair1", "pair2"]
    duplicate_rows = frame.loc[frame.duplicated(pair_key, keep=False)]
    if not duplicate_rows.empty:
        conflicting = duplicate_rows.groupby(pair_key, sort=False, observed=True)["label"].nunique().gt(1)
        if bool(conflicting.any()):
            example = conflicting.index[conflicting][0]
            raise ValueError(f"pair catalog contains conflicting labels within one source family: {example}")
    return frame.drop_duplicates(pair_key, keep="first").reset_index(drop=True)


def pair_catalog_diversity_diagnostics(
    catalog: pd.DataFrame,
    *,
    reference_family: str,
) -> dict[str, Any]:
    """Summarize pair-source diversity and exact overlap with one family.

    Rows are canonicalized and de-duplicated within each domain/family without
    materializing a global pair universe. Pair degree counts both endpoints of
    each retained pair. The returned object contains three bounded, flat record
    tables (``domain_family``, ``label_rules``, and ``reference_overlap``), so
    each table can be written directly to JSON or CSV.

    Args:
        catalog: Pair catalog containing the canonical pair-source columns.
        reference_family: Existing source family against which exact canonical
            ``(source_domain, pair1, pair2)`` overlap is measured.

    Returns:
        JSON-friendly metadata and flat diagnostic record tables.

    Raises:
        ValueError: If the catalog is malformed, internally contradictory, or
            does not contain ``reference_family``.
    """

    reference = str(reference_family).strip()
    if not reference:
        raise ValueError("reference_family must be non-empty")
    canonical = _canonical_diagnostic_view(catalog)
    reference_rows = canonical.loc[canonical["source_family"].eq(reference)]
    if reference_rows.empty:
        raise ValueError(f"reference_family={reference!r} is absent from the pair catalog")

    reference_pairs_by_domain: dict[str, set[tuple[str, str]]] = {}
    for domain, rows in reference_rows.groupby("source_domain", sort=False, observed=True):
        reference_pairs_by_domain[str(domain)] = set(
            zip(rows["pair1"].astype(str), rows["pair2"].astype(str), strict=True)
        )

    domain_family_records: list[dict[str, Any]] = []
    overlap_records: list[dict[str, Any]] = []
    grouped = canonical.groupby(["source_domain", "source_family"], sort=True, observed=True)
    for raw_key, rows in grouped:
        raw_domain, raw_family = cast(tuple[object, object], raw_key)
        domain = str(raw_domain)
        family = str(raw_family)
        labels = rows["label"].to_numpy(dtype=np.int8, copy=False)
        positives = int(labels.sum())
        endpoints = pd.concat([rows["pair1"], rows["pair2"]], ignore_index=True)
        degrees = endpoints.value_counts(sort=False).to_numpy(dtype=np.int64, copy=False)
        quantiles = np.quantile(degrees, (0.5, 0.9, 0.95, 0.99))
        domain_family_records.append(
            {
                "source_domain": domain,
                "source_family": family,
                "rows": int(len(rows)),
                "positives": positives,
                "negatives": int(len(rows) - positives),
                "unique_signatures": int(len(degrees)),
                "unique_group_ids": int(rows["group_id"].nunique()),
                "pair_degree_mean": float(degrees.mean()),
                "pair_degree_p50": float(quantiles[0]),
                "pair_degree_p90": float(quantiles[1]),
                "pair_degree_p95": float(quantiles[2]),
                "pair_degree_p99": float(quantiles[3]),
                "pair_degree_max": int(degrees.max()),
            }
        )

        reference_pairs = reference_pairs_by_domain.get(domain, set())
        overlapping = sum(
            (pair1, pair2) in reference_pairs for pair1, pair2 in zip(rows["pair1"], rows["pair2"], strict=True)
        )
        overlap_records.append(
            {
                "source_domain": domain,
                "source_family": family,
                "reference_family": reference,
                "source_pairs": int(len(rows)),
                "reference_pairs": int(len(reference_pairs)),
                "overlapping_pairs": int(overlapping),
                "source_overlap_fraction": float(overlapping / len(rows)),
                "reference_coverage_fraction": (float(overlapping / len(reference_pairs)) if reference_pairs else None),
            }
        )

    label_rule_records: list[dict[str, Any]] = []
    rule_counts = canonical.groupby(["source_domain", "source_family", "label_rule"], sort=True, observed=True).size()
    for raw_key, rows in rule_counts.items():
        domain, family, label_rule = cast(tuple[object, object, object], raw_key)
        label_rule_records.append(
            {
                "source_domain": str(domain),
                "source_family": str(family),
                "label_rule": str(label_rule),
                "rows": int(rows),
            }
        )
    return {
        "reference_family": reference,
        "domain_family": domain_family_records,
        "label_rules": label_rule_records,
        "reference_overlap": overlap_records,
    }


def select_feature_rows(
    selected_catalog: pd.DataFrame,
    row_index_by_domain_and_pair: Mapping[tuple[str, str, str], int],
) -> np.ndarray:
    """Resolve canonical catalog rows to one global feature-matrix index."""

    canonical = canonicalize_pairs(selected_catalog.loc[:, PAIR_COLUMNS])
    indices = []
    for source_domain, _source_family, pair1, pair2, _label, _label_rule, _origin, _group_id in canonical.itertuples(
        index=False, name=None
    ):
        key = (str(source_domain), str(pair1), str(pair2))
        try:
            indices.append(int(row_index_by_domain_and_pair[key]))
        except KeyError as exc:
            raise ValueError(f"Feature store is missing pair key: {key}") from exc
    return np.asarray(indices, dtype=np.int64)

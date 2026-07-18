"""Deterministic pairwise-training recipes and pair selection.

This module contains the production-safe policy shared by release training and
pair-source experiments.  It deliberately operates on an explicit pair catalog
so the exact selected identities, class counts, and provenance can be audited
before featurization.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import pandas as pd

PAIR_KEY_COLUMNS = ("source_domain", "pair1", "pair2")
PAIR_IDENTITY_COLUMNS = (*PAIR_KEY_COLUMNS, "label")
REQUIRED_PAIR_COLUMNS = ("source_domain", "source_family", "pair1", "pair2", "label")
PAIR_IDENTITY_DIGEST_SCHEMA = "s2and-pair-identity-v1"

BASE_FAMILY = "gold_cluster_uniform"
LINKER_PUBLIC_FAMILY = "linker_public_gold"
LINKER_BIG_POSITIVE_FAMILY = "linker_big_orcid_positive"
LINKER_PROXY_NEGATIVE_FAMILY = "linker_component_proxy_negative"
LINKER_FAMILIES = (
    LINKER_PUBLIC_FAMILY,
    LINKER_BIG_POSITIVE_FAMILY,
    LINKER_PROXY_NEGATIVE_FAMILY,
)

GOLD_BASE_SOURCE_DOMAINS = ("aminer", "arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")
BIG_LINKER_SOURCE_DOMAINS = ("a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park")
PUBLIC_LINKER_SOURCE_DOMAINS = ("arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")
ALL_LINKER_SOURCE_DOMAINS = (*PUBLIC_LINKER_SOURCE_DOMAINS, *BIG_LINKER_SOURCE_DOMAINS)

AdditiveLinkerSourceSet = Literal["all13", "big7"]
ADDITIVE_LINKER_SOURCE_SETS: dict[AdditiveLinkerSourceSet, tuple[str, ...]] = {
    "all13": ALL_LINKER_SOURCE_DOMAINS,
    "big7": BIG_LINKER_SOURCE_DOMAINS,
}


@dataclass(frozen=True, slots=True)
class AdditiveLinkerRecipe:
    """Balanced linker rows appended to an unchanged uniform-gold base."""

    source_set: AdditiveLinkerSourceSet
    source_domains: tuple[str, ...]
    linker_pairs_per_domain: int

    def __post_init__(self) -> None:
        expected_domains = ADDITIVE_LINKER_SOURCE_SETS.get(self.source_set)
        if expected_domains is None:
            raise ValueError(f"unknown additive linker source set: {self.source_set!r}")
        if self.source_domains != expected_domains:
            raise ValueError(
                "additive linker source domains do not match their named source set: "
                f"source_set={self.source_set!r}, expected={expected_domains}, observed={self.source_domains}"
            )
        if self.linker_pairs_per_domain <= 0 or self.linker_pairs_per_domain % 2:
            raise ValueError("additive linker cap must be a positive even number of rows per source domain")

    @property
    def linker_pairs_per_label(self) -> int:
        """Return the requested positive and negative quota per domain."""

        return self.linker_pairs_per_domain // 2


@dataclass(frozen=True, slots=True)
class PairwiseTrainingRecipe:
    """One fully resolved production pairwise-training recipe."""

    name: str
    uniform_source_domains: tuple[str, ...]
    uniform_pairs_per_domain: int
    pair_sampling_seed: int
    additive_linker: AdditiveLinkerRecipe
    base_sampler: str = "uniform_within_blocks_without_replacement"
    balancing: str = "linker_per_domain_shared_min_binary_no_backfill"
    budget_policy: str = "additive_to_unchanged_uniform"
    exclude_base_overlaps: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("pairwise training recipe name must be non-empty")
        if not self.uniform_source_domains or len(set(self.uniform_source_domains)) != len(self.uniform_source_domains):
            raise ValueError("uniform source domains must be non-empty and unique")
        if any(not domain.strip() for domain in self.uniform_source_domains):
            raise ValueError("uniform source domains must contain non-empty names")
        if self.uniform_pairs_per_domain <= 0:
            raise ValueError("uniform_pairs_per_domain must be positive")
        if not isinstance(self.pair_sampling_seed, int):
            raise TypeError("pair_sampling_seed must be an integer")
        if not self.exclude_base_overlaps:
            raise ValueError("production pairwise recipes must exclude linker rows overlapping the base")

    @property
    def nominal_training_rows(self) -> int:
        """Return the expected row count when every source reaches its quota."""

        return self.uniform_pairs_per_domain * len(self.uniform_source_domains) + (
            self.additive_linker.linker_pairs_per_domain * len(self.additive_linker.source_domains)
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, fully resolved recipe."""

        payload = asdict(self)
        payload["uniform_source_domains"] = list(self.uniform_source_domains)
        payload["additive_linker"]["source_domains"] = list(self.additive_linker.source_domains)
        payload["additive_linker"]["linker_pairs_per_label"] = self.additive_linker.linker_pairs_per_label
        payload["nominal_training_rows"] = self.nominal_training_rows
        return payload


@dataclass(frozen=True, slots=True)
class PairAssemblyResult:
    """Selected canonical pair rows and their reproducibility audit."""

    pairs: pd.DataFrame
    audit: dict[str, Any]


def additive_linker_recipe(
    source_set: AdditiveLinkerSourceSet,
    linker_pairs_per_domain: int,
) -> AdditiveLinkerRecipe:
    """Build a validated additive-linker recipe from a named source set."""

    domains = ADDITIVE_LINKER_SOURCE_SETS.get(source_set)
    if domains is None:
        raise ValueError(f"unknown additive linker source set: {source_set!r}")
    return AdditiveLinkerRecipe(
        source_set=source_set,
        source_domains=domains,
        linker_pairs_per_domain=linker_pairs_per_domain,
    )


def _canonicalize_core_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize pair identity while preserving provenance columns."""

    missing = sorted(set(REQUIRED_PAIR_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"pair catalog is missing required columns: {missing}")
    canonical = frame.copy()
    if canonical.empty:
        canonical["label"] = canonical["label"].astype("int8")
        return canonical.reset_index(drop=True)

    for column in ("source_domain", "source_family", "pair1", "pair2"):
        if bool(canonical[column].isna().any()):
            raise ValueError(f"{column} must be non-empty")
        canonical[column] = canonical[column].astype(str).str.strip()
        if bool(canonical[column].eq("").any()):
            raise ValueError(f"{column} must be non-empty")

    numeric_labels = pd.to_numeric(canonical["label"], errors="coerce")
    if bool(numeric_labels.isna().any()) or not bool(numeric_labels.isin((0, 1)).all()):
        raise ValueError("pair labels must be exactly 0 or 1")
    canonical["label"] = numeric_labels.astype("int8")

    left = canonical["pair1"].copy()
    right = canonical["pair2"].copy()
    if bool(left.eq(right).any()):
        example = str(left.loc[left.eq(right)].iloc[0])
        raise ValueError(f"self-pairs are not valid training examples: {example!r}")
    swap = left.gt(right)
    canonical.loc[swap, "pair1"] = right.loc[swap]
    canonical.loc[swap, "pair2"] = left.loc[swap]

    duplicate_rows = canonical.loc[canonical.duplicated(list(PAIR_KEY_COLUMNS), keep=False)]
    if not duplicate_rows.empty:
        conflicting = duplicate_rows.groupby(list(PAIR_KEY_COLUMNS), sort=False)["label"].nunique().gt(1)
        if bool(conflicting.any()):
            source_domain, pair1, pair2 = conflicting.index[conflicting][0]
            raise ValueError(f"conflicting labels for source_domain={source_domain!r}, pair=({pair1!r}, {pair2!r})")
    return canonical.drop_duplicates(list(PAIR_KEY_COLUMNS), keep="first").reset_index(drop=True)


def _canonicalize_catalog_preserving_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize within each source family without erasing overlaps."""

    missing = sorted(set(REQUIRED_PAIR_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"pair catalog is missing required columns: {missing}")
    canonical_frames = [
        _canonicalize_core_pairs(rows) for _, rows in frame.groupby("source_family", sort=False, observed=True)
    ]
    if canonical_frames:
        return pd.concat(canonical_frames, ignore_index=True)
    return _canonicalize_core_pairs(frame)


def pair_identity_digest(frame: pd.DataFrame) -> str:
    """Digest canonical pair identities and labels, independent of provenance."""

    canonical = _canonicalize_core_pairs(frame).sort_values(list(PAIR_IDENTITY_COLUMNS), kind="stable")
    digest = hashlib.sha256(f"{PAIR_IDENTITY_DIGEST_SCHEMA}\0".encode())
    for source_domain, pair1, pair2, label in canonical.loc[:, PAIR_IDENTITY_COLUMNS].itertuples(
        index=False,
        name=None,
    ):
        digest.update(f"{source_domain}\0{pair1}\0{pair2}\0{int(label)}\n".encode())
    return digest.hexdigest()


def source_counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize rows and class balance by source domain and family."""

    canonical_frames = [
        _canonicalize_core_pairs(rows) for _, rows in frame.groupby("source_family", sort=False, observed=True)
    ]
    canonical = pd.concat(canonical_frames, ignore_index=True) if canonical_frames else frame.iloc[0:0]
    output: list[dict[str, Any]] = []
    grouped = canonical.groupby(["source_domain", "source_family"], sort=True, observed=True)["label"]
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


def _stable_pair_rank(frame: pd.DataFrame, *, random_seed: int, scope: str) -> pd.DataFrame:
    canonical = _canonicalize_core_pairs(frame)
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
            ).encode()
        ).hexdigest()
        for source_domain, pair1, pair2, label in ranked.loc[:, PAIR_IDENTITY_COLUMNS].itertuples(
            index=False,
            name=None,
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
        for domain, pair1, pair2 in frame.loc[:, PAIR_KEY_COLUMNS].itertuples(index=False, name=None)
    }


def _without_pair_keys(
    frame: pd.DataFrame,
    excluded: set[tuple[str, str, str]],
) -> pd.DataFrame:
    if frame.empty or not excluded:
        return frame.reset_index(drop=True)
    keep = [
        (str(domain), str(pair1), str(pair2)) not in excluded
        for domain, pair1, pair2 in frame.loc[:, PAIR_KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    return frame.loc[keep].reset_index(drop=True)


def _selection_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    canonical = _canonicalize_core_pairs(frame).sort_values(list(PAIR_IDENTITY_COLUMNS), kind="stable")
    for domain, pair1, pair2, label in canonical.loc[:, PAIR_IDENTITY_COLUMNS].itertuples(
        index=False,
        name=None,
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


def select_balanced_linker_pairs(
    eligible_catalog: pd.DataFrame,
    *,
    excluded_pair_keys: set[tuple[str, str, str]],
    cap_per_domain: int,
    capacity_rows: int,
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Select shared-min positive/negative linker quotas within capacity."""

    if cap_per_domain < 0 or cap_per_domain % 2:
        raise ValueError("linker_pairs_per_domain must be a non-negative even value")
    raw = eligible_catalog.loc[eligible_catalog["source_family"].isin(LINKER_FAMILIES)]
    linker = _without_pair_keys(_canonicalize_core_pairs(raw), excluded_pair_keys)

    requested_units: dict[str, int] = {}
    available_by_domain: dict[str, tuple[int, int]] = {}
    for domain, rows in linker.groupby("source_domain", sort=True, observed=True):
        label_counts = rows["label"].value_counts().to_dict()
        available_negative = int(label_counts.get(0, 0))
        available_positive = int(label_counts.get(1, 0))
        domain_text = str(domain)
        available_by_domain[domain_text] = (available_negative, available_positive)
        requested_units[domain_text] = min(
            cap_per_domain // 2,
            available_negative,
            available_positive,
        )

    allocations = _domain_fair_quotas(requested_units, max(0, capacity_rows) // 2)
    selected_frames: list[pd.DataFrame] = []
    domain_audit: list[dict[str, Any]] = []
    for domain in sorted(available_by_domain):
        available_negative, available_positive = available_by_domain[domain]
        requested_per_class = requested_units[domain]
        selected_per_class = allocations[domain]
        domain_rows = linker.loc[linker["source_domain"].eq(domain)]
        for label in (0, 1):
            selected_frames.append(
                _stable_pair_rank(
                    domain_rows.loc[domain_rows["label"].eq(label)],
                    random_seed=random_seed,
                    scope=f"balanced_linker:{domain}:label={label}",
                ).head(selected_per_class)
            )
        domain_audit.append(
            {
                "source_domain": domain,
                "available_negatives": available_negative,
                "available_positives": available_positive,
                "cap_rows": cap_per_domain,
                "requested_per_class": requested_per_class,
                "selected_per_class": selected_per_class,
                "selected_rows": 2 * selected_per_class,
                "source_limited": requested_per_class < cap_per_domain // 2,
                "capacity_capped": selected_per_class < requested_per_class,
            }
        )

    selected = (
        _canonicalize_core_pairs(pd.concat(selected_frames, ignore_index=True))
        if selected_frames
        else _canonicalize_core_pairs(raw.iloc[0:0])
    )
    requested_rows = 2 * sum(requested_units.values())
    if len(selected) != 2 * sum(allocations.values()):
        raise AssertionError("balanced-linker selection did not preserve paired class quotas")
    return selected, domain_audit, requested_rows


def assemble_additive_linker_pairs(
    catalog: pd.DataFrame,
    recipe: AdditiveLinkerRecipe,
    *,
    held_out_domain: str | None,
    random_seed: int,
) -> PairAssemblyResult:
    """Append balanced linker prefixes while preserving every base row."""

    raw = _canonicalize_catalog_preserving_families(catalog)
    eligible = (
        raw if held_out_domain is None else raw.loc[~raw["source_domain"].eq(held_out_domain)].reset_index(drop=True)
    )
    base = _canonicalize_core_pairs(eligible.loc[eligible["source_family"].eq(BASE_FAMILY)])
    if base.empty:
        suffix = "" if held_out_domain is None else f" after holding out {held_out_domain!r}"
        raise ValueError(f"uniform base is empty{suffix}")

    expected_linker_domains = tuple(domain for domain in recipe.source_domains if domain != held_out_domain)
    raw_linker = _canonicalize_core_pairs(
        eligible.loc[
            eligible["source_domain"].isin(expected_linker_domains) & eligible["source_family"].isin(LINKER_FAMILIES)
        ]
    )
    observed_linker_domains = set(raw_linker["source_domain"].astype(str))
    missing_linker_domains = sorted(set(expected_linker_domains).difference(observed_linker_domains))
    if missing_linker_domains:
        raise ValueError("additive linker source set is missing eligible catalog rows: " f"{missing_linker_domains}")

    base_keys = _pair_key_set(base)
    eligible_linker = _without_pair_keys(raw_linker, base_keys)
    linker, linker_domain_audit, linker_requested_rows = select_balanced_linker_pairs(
        eligible_linker,
        excluded_pair_keys=set(),
        cap_per_domain=recipe.linker_pairs_per_domain,
        capacity_rows=len(eligible_linker),
        random_seed=random_seed,
    )
    observed_audit_domains = tuple(record["source_domain"] for record in linker_domain_audit)
    if observed_audit_domains != tuple(sorted(expected_linker_domains)):
        raise AssertionError(
            "balanced additive linker audit does not cover the expected source domains: "
            f"expected={tuple(sorted(expected_linker_domains))}, observed={observed_audit_domains}"
        )

    selected = _canonicalize_core_pairs(pd.concat([base, linker], ignore_index=True))
    if len(selected) != len(base) + len(linker):
        raise AssertionError("additive linker rows overlap the preserved uniform base")
    selected_base = _canonicalize_core_pairs(selected.loc[selected["source_family"].eq(BASE_FAMILY)])
    if pair_identity_digest(selected_base) != pair_identity_digest(base):
        raise AssertionError("additive linker assembly changed the uniform base")
    held_out_rows = 0 if held_out_domain is None else int(selected["source_domain"].eq(held_out_domain).sum())
    if held_out_rows:
        raise AssertionError(f"additive linker assembly retained {held_out_rows} held-out rows")

    nominal_linker_rows = recipe.linker_pairs_per_domain * len(expected_linker_domains)
    audit = {
        "mode": "additive_linker",
        "held_out_domain": held_out_domain,
        "target_rows": len(selected),
        "base_rows_after_lodo": len(base),
        "base_pair_digest": pair_identity_digest(base),
        "base_selection_sha256": _selection_digest(base),
        "base_filler_rows": 0,
        "linker_source_set": recipe.source_set,
        "linker_source_domains": list(recipe.source_domains),
        "linker_eligible_domains_after_lodo": list(expected_linker_domains),
        "linker_cap_per_domain": recipe.linker_pairs_per_domain,
        "linker_nominal_requested_rows": nominal_linker_rows,
        "linker_requested_rows": linker_requested_rows,
        "linker_selected_rows": len(linker),
        "linker_source_limited": linker_requested_rows < nominal_linker_rows,
        "linker_base_overlap_rows": len(raw_linker) - len(eligible_linker),
        "linker_domains": linker_domain_audit,
        "final_rows": len(selected),
        "held_out_rows": held_out_rows,
        "selection_sha256": _selection_digest(selected),
        "training_pair_digest_schema": PAIR_IDENTITY_DIGEST_SCHEMA,
        "training_pair_digest": pair_identity_digest(selected),
        "training_source_counts": source_counts(selected),
    }
    if audit["final_rows"] != audit["base_rows_after_lodo"] + audit["linker_selected_rows"]:
        raise AssertionError("additive linker final row count does not equal unchanged base plus linker rows")
    return PairAssemblyResult(selected.reset_index(drop=True), audit)


def resolve_pairwise_training_recipe(
    catalog: pd.DataFrame,
    recipe: PairwiseTrainingRecipe,
) -> PairAssemblyResult:
    """Resolve and strictly verify a no-heldout production training recipe."""

    canonical = _canonicalize_catalog_preserving_families(catalog)
    allowed_domains = set(recipe.uniform_source_domains) | set(recipe.additive_linker.source_domains)
    allowed_families = {BASE_FAMILY, *LINKER_FAMILIES}
    relevant = canonical.loc[
        canonical["source_domain"].isin(allowed_domains) & canonical["source_family"].isin(allowed_families)
    ].reset_index(drop=True)

    base = relevant.loc[relevant["source_family"].eq(BASE_FAMILY)]
    base_counts = {
        str(domain): int(len(rows)) for domain, rows in base.groupby("source_domain", sort=True, observed=True)
    }
    expected_base_counts = {domain: recipe.uniform_pairs_per_domain for domain in recipe.uniform_source_domains}
    if base_counts != expected_base_counts:
        raise ValueError(
            "uniform base does not match the production recipe: "
            f"expected={expected_base_counts}, observed={base_counts}"
        )

    result = assemble_additive_linker_pairs(
        relevant,
        recipe.additive_linker,
        held_out_domain=None,
        random_seed=recipe.pair_sampling_seed,
    )
    if result.audit["linker_source_limited"]:
        raise ValueError("linker sources cannot satisfy the production recipe quotas")
    if len(result.pairs) != recipe.nominal_training_rows:
        raise AssertionError(
            "resolved production recipe row count mismatch: "
            f"expected={recipe.nominal_training_rows}, observed={len(result.pairs)}"
        )
    audit = {
        **result.audit,
        "recipe": recipe.as_dict(),
        "nominal_training_rows": recipe.nominal_training_rows,
    }
    return PairAssemblyResult(result.pairs, audit)


__all__ = [
    "ADDITIVE_LINKER_SOURCE_SETS",
    "ALL_LINKER_SOURCE_DOMAINS",
    "AdditiveLinkerRecipe",
    "AdditiveLinkerSourceSet",
    "BASE_FAMILY",
    "BIG_LINKER_SOURCE_DOMAINS",
    "GOLD_BASE_SOURCE_DOMAINS",
    "LINKER_BIG_POSITIVE_FAMILY",
    "LINKER_FAMILIES",
    "LINKER_PROXY_NEGATIVE_FAMILY",
    "LINKER_PUBLIC_FAMILY",
    "PAIR_IDENTITY_DIGEST_SCHEMA",
    "PUBLIC_LINKER_SOURCE_DOMAINS",
    "PairAssemblyResult",
    "PairwiseTrainingRecipe",
    "additive_linker_recipe",
    "assemble_additive_linker_pairs",
    "pair_identity_digest",
    "resolve_pairwise_training_recipe",
    "select_balanced_linker_pairs",
    "source_counts",
]

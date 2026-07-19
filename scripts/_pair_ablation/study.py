"""The complete design and deterministic sampling policy for pair ablation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal, TypeAlias, cast

import pandas as pd

PAIR_COLUMNS = ("source_domain", "source_family", "pair1", "pair2", "label")
BASE_FAMILY = "base"
LINKER_FAMILY = "linker"

GOLD_DOMAINS = ("aminer", "arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")
PAIR_DOMAINS = ("medline",)
PROXY_DOMAINS = ("a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park")
ALL_DOMAINS = (*GOLD_DOMAINS, *PAIR_DOMAINS, *PROXY_DOMAINS)
BIG7_DOMAINS = PROXY_DOMAINS
ALL13_DOMAINS = (*tuple(domain for domain in GOLD_DOMAINS if domain != "aminer"), *PROXY_DOMAINS)

SourceSet = Literal["all13", "big7"]
SOURCE_SETS: dict[SourceSet, tuple[str, ...]] = {
    "all13": ALL13_DOMAINS,
    "big7": BIG7_DOMAINS,
}
BASELINE_NAME = "baseline"
_ADDITIVE_NAME = re.compile(r"linker_(all13|big7)_([1-9][0-9]*)")


@dataclass(frozen=True, slots=True)
class Baseline:
    """Unchanged gold pairs with the evaluation domain held out."""

    @property
    def name(self) -> str:
        """Return the stable command-line name."""

        return BASELINE_NAME


@dataclass(frozen=True, slots=True)
class AdditiveDose:
    """A balanced per-domain linker dose added to the baseline."""

    source_set: SourceSet
    pairs_per_domain: int

    def __post_init__(self) -> None:
        valid_dose = (
            isinstance(self.pairs_per_domain, Integral)
            and not isinstance(self.pairs_per_domain, bool)
            and self.pairs_per_domain > 0
            and self.pairs_per_domain % 2 == 0
        )
        if self.source_set not in SOURCE_SETS or not valid_dose:
            raise ValueError("source_set must be all13/big7 and dose must be a positive even integer")

    @property
    def source_domains(self) -> tuple[str, ...]:
        """Return linker domains in their canonical order."""

        return SOURCE_SETS[self.source_set]

    @property
    def name(self) -> str:
        """Return the stable command-line name."""

        return f"linker_{self.source_set}_{self.pairs_per_domain}"


StudyArm: TypeAlias = Baseline | AdditiveDose
BASELINE = Baseline()


def parse_arm_name(name: str) -> StudyArm:
    """Parse a canonical arm name."""

    if name == BASELINE_NAME:
        return BASELINE
    match = _ADDITIVE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"unknown pair-ablation arm: {name!r}")
    source_set, dose = match.groups()
    return AdditiveDose(cast(SourceSet, source_set), int(dose))


def _strict_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _strict_label(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and int(value) in (0, 1)


def validate_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the five-column catalog and attach its feature-matrix row."""

    if tuple(frame.columns) != PAIR_COLUMNS:
        raise ValueError(f"catalog columns must be exactly {PAIR_COLUMNS}")
    output = frame.copy()
    for column in PAIR_COLUMNS[:-1]:
        if not all(_strict_string(value) for value in output[column].array):
            raise ValueError(f"catalog {column} values must be nonempty strings")
    if not all(_strict_label(value) for value in output["label"].array):
        raise ValueError("catalog labels must be exact integral 0 or 1 values")
    output["label"] = output["label"].astype("int8")

    unknown_domains = sorted(set(output["source_domain"]).difference(ALL_DOMAINS))
    if unknown_domains:
        raise ValueError(f"catalog contains unknown source domains: {unknown_domains}")
    unknown_families = sorted(set(output["source_family"]).difference((BASE_FAMILY, LINKER_FAMILY)))
    if unknown_families:
        raise ValueError(f"catalog contains unknown source families: {unknown_families}")
    invalid_base = sorted(
        set(output.loc[output["source_family"].eq(BASE_FAMILY), "source_domain"]).difference(GOLD_DOMAINS)
    )
    invalid_linker = sorted(
        set(output.loc[output["source_family"].eq(LINKER_FAMILY), "source_domain"]).difference(ALL13_DOMAINS)
    )
    if invalid_base or invalid_linker:
        raise ValueError(f"invalid family/domain combinations: base={invalid_base}, linker={invalid_linker}")
    if output["pair1"].eq(output["pair2"]).any():
        raise ValueError("catalog contains a self-pair")
    if output["pair1"].ge(output["pair2"]).any():
        raise ValueError("catalog pairs must use canonical pair1 < pair2 orientation")

    provenance_key = ["source_family", "source_domain", "pair1", "pair2"]
    if output.duplicated(provenance_key).any():
        raise ValueError("catalog contains a duplicate family/domain pair")
    pair_key = ["source_domain", "pair1", "pair2"]
    if output.groupby(pair_key, sort=False, observed=True)["label"].nunique().gt(1).any():
        raise ValueError("catalog contains conflicting labels for a pair")
    output = output.reset_index(drop=True)
    output["feature_row"] = range(len(output))
    return output


def _catalog_with_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) == PAIR_COLUMNS:
        return validate_catalog(frame)
    if tuple(frame.columns) != (*PAIR_COLUMNS, "feature_row"):
        raise ValueError(f"catalog columns must be {PAIR_COLUMNS} with optional feature_row")
    feature_rows = list(frame["feature_row"])
    valid = validate_catalog(frame.loc[:, PAIR_COLUMNS])
    if not all(isinstance(value, Integral) and not isinstance(value, bool) for value in feature_rows) or {
        int(value) for value in feature_rows
    } != set(range(len(valid))):
        raise ValueError("feature_row must be a permutation of the catalog row numbers")
    valid["feature_row"] = [int(value) for value in feature_rows]
    return valid


def pair_digest(frame: pd.DataFrame) -> str:
    """Hash pair content independently of row order and feature-row position."""

    missing = set(PAIR_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"cannot digest catalog missing columns: {sorted(missing)}")
    valid = validate_catalog(frame.loc[:, PAIR_COLUMNS])
    rows = sorted(valid.loc[:, PAIR_COLUMNS].itertuples(index=False, name=None))
    digest = hashlib.sha256(b"pair-ablation-catalog-v1\0")
    for row in rows:
        for value in row:
            encoded = str(int(value) if isinstance(value, Integral) else value).encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _rank(rows: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    ranked = rows.copy()
    ranked["_rank"] = [
        hashlib.sha256(f"{seed}\0{domain}\0{label}\0{left}\0{right}".encode()).digest()
        for domain, left, right, label in ranked[["source_domain", "pair1", "pair2", "label"]].itertuples(
            index=False, name=None
        )
    ]
    return ranked.sort_values(["_rank", "source_domain", "label", "pair1", "pair2"], kind="stable").drop(
        columns="_rank"
    )


def select_pairs(
    catalog: pd.DataFrame,
    arm: StudyArm,
    held_out_domain: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select unchanged LODO base pairs and an optional balanced linker dose."""

    if held_out_domain not in ALL_DOMAINS:
        raise ValueError(f"unknown held-out domain: {held_out_domain!r}")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    catalog = _catalog_with_rows(catalog)
    base = catalog.loc[catalog["source_family"].eq(BASE_FAMILY) & ~catalog["source_domain"].eq(held_out_domain)]
    if base.empty:
        raise ValueError("LODO baseline is empty")
    if set(base["label"]) != {0, 1}:
        raise ValueError(f"LODO baseline must contain both labels after holding out {held_out_domain!r}")

    linker_parts: list[pd.DataFrame] = []
    linker_counts: dict[str, int] = {}
    if isinstance(arm, AdditiveDose):
        base_keys = set(base[["source_domain", "pair1", "pair2"]].itertuples(index=False, name=None))
        quota = arm.pairs_per_domain // 2
        for domain in arm.source_domains:
            if domain == held_out_domain:
                continue
            rows = catalog.loc[catalog["source_family"].eq(LINKER_FAMILY) & catalog["source_domain"].eq(domain)]
            rows = rows.loc[
                [
                    key not in base_keys
                    for key in rows[["source_domain", "pair1", "pair2"]].itertuples(index=False, name=None)
                ]
            ]
            per_class = min(
                quota,
                int(rows["label"].eq(0).sum()),
                int(rows["label"].eq(1).sum()),
            )
            chosen = pd.concat(
                [_rank(rows.loc[rows["label"].eq(label)], seed=int(seed)).head(per_class) for label in (0, 1)],
                ignore_index=True,
            )
            linker_parts.append(chosen)
            linker_counts[domain] = len(chosen)
    elif not isinstance(arm, Baseline):
        raise TypeError(f"unsupported pair-ablation arm: {type(arm).__name__}")

    linker = pd.concat(linker_parts, ignore_index=True) if linker_parts else base.iloc[0:0]
    if isinstance(arm, AdditiveDose) and linker.empty:
        raise ValueError(f"additive arm {arm.name!r} selected no linker rows")
    selected = pd.concat([base, linker], ignore_index=True).sort_values(list(PAIR_COLUMNS), kind="stable")
    selected = selected.reset_index(drop=True)
    audit = {
        "heldout": held_out_domain,
        "base_digest": pair_digest(base),
        "base_rows": len(base),
        "training_digest": pair_digest(selected),
        "training_rows": len(selected),
        "linker_rows": len(linker),
        "linker_by_domain": linker_counts,
    }
    return selected, audit

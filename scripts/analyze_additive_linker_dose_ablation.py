"""Apply the three paired B³ safety gates to fresh pair-ablation results."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._pair_ablation.study import (  # noqa: E402
    ALL_DOMAINS,
    BASELINE_NAME,
    GOLD_DOMAINS,
    parse_arm_name,
)

MIN_MEAN_DELTA_B3 = 0.0
MIN_Q05_DELTA_B3 = -0.002
MIN_WORST_DOMAIN_DELTA_B3 = -0.010
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVALUATION_IDENTITY_KEYS = (
    "evaluation_digest",
    "evaluation_identity",
    "evaluation_pair_digest",
    "evaluation_seed",
    "b3_evaluation_digest",
)


@dataclass(frozen=True, slots=True)
class Fold:
    """The fields needed to pair one candidate fold with its baseline."""

    seed: int
    arm: str
    domain: str
    study_digest: str
    prepared_digest: str
    training_pair_digest: str
    baseline_pair_digest: str
    b3_f1: float | None
    evaluation_identity: dict[str, Any]


def _required_string(row: dict[str, Any], key: str, path: Path) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: {key} must be a nonempty string")
    return value


def _digest(row: dict[str, Any], key: str, path: Path) -> str:
    value = _required_string(row, key, path)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path}: {key} must be a lowercase SHA-256 digest")
    return value


def _fold(path: Path) -> Fold:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: cannot read result JSON: {error}") from error
    if not isinstance(row, dict):
        raise ValueError(f"{path}: result must be a JSON object")
    seed = row.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{path}: training_seed must be a nonnegative integer")
    arm = _required_string(row, "arm", path)
    try:
        parse_arm_name(arm)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    domain = _required_string(row, "held_out_domain", path)
    if domain not in ALL_DOMAINS:
        raise ValueError(f"{path}: unknown held_out_domain {domain!r}")
    if "b3_f1" not in row:
        raise ValueError(f"{path}: missing b3_f1")
    b3_f1 = row["b3_f1"]
    if b3_f1 is not None:
        if (
            isinstance(b3_f1, bool)
            or not isinstance(b3_f1, int | float)
            or not math.isfinite(b3_f1)
            or not 0 <= b3_f1 <= 1
        ):
            raise ValueError(f"{path}: b3_f1 must be a finite probability or null")
        b3_f1 = float(b3_f1)
    if (domain in GOLD_DOMAINS) != (b3_f1 is not None):
        raise ValueError(f"{path}: b3_f1 must be present exactly for gold domains")
    identities = {key: row[key] for key in _EVALUATION_IDENTITY_KEYS if key in row}
    if any(value is None for value in identities.values()):
        raise ValueError(f"{path}: evaluation identity fields cannot be null")
    return Fold(
        seed=seed,
        arm=arm,
        domain=domain,
        study_digest=_digest(row, "study_digest", path),
        prepared_digest=_digest(row, "prepared_digest", path),
        training_pair_digest=_digest(row, "training_pair_digest", path),
        baseline_pair_digest=_digest(row, "baseline_pair_digest", path),
        b3_f1=b3_f1,
        evaluation_identity=identities,
    )


def load_folds(inputs: list[Path]) -> dict[tuple[int, str, str], Fold]:
    """Recursively load result files and reject duplicate study cells."""

    paths: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.resolve()
        if path.is_dir():
            for candidate in path.rglob("*.json"):
                if not candidate.parent.parent.name.isdecimal() or candidate.stem not in ALL_DOMAINS:
                    continue
                try:
                    parse_arm_name(candidate.parent.name)
                except ValueError:
                    continue
                paths.add(candidate.resolve())
        elif path.is_file():
            paths.add(path)
        else:
            raise ValueError(f"result input does not exist: {raw_path}")
    if not paths:
        raise ValueError("no result JSON files found")
    folds: dict[tuple[int, str, str], Fold] = {}
    for path in sorted(paths):
        fold = _fold(path)
        key = (fold.seed, fold.arm, fold.domain)
        if key in folds:
            raise ValueError(f"duplicate result cell seed={fold.seed}, arm={fold.arm}, domain={fold.domain}")
        folds[key] = fold
    return folds


def _paired_grids(
    folds: dict[tuple[int, str, str], Fold],
) -> tuple[dict[tuple[int, str], Fold], dict[str, dict[tuple[int, str], Fold]]]:
    baseline = {(fold.seed, fold.domain): fold for fold in folds.values() if fold.arm == BASELINE_NAME}
    if not baseline:
        raise ValueError("no baseline results found")
    arms = sorted({fold.arm for fold in folds.values()} - {BASELINE_NAME})
    if not arms:
        raise ValueError("no candidate results found")
    seeds = {seed for seed, _ in baseline}
    domains = {domain for _, domain in baseline}
    expected = {(seed, domain) for seed in seeds for domain in domains}
    if set(baseline) != expected:
        raise ValueError("baseline result grid is incomplete")
    prepared = {fold.prepared_digest for fold in baseline.values()}
    if len(prepared) != 1:
        raise ValueError("baseline folds use different prepared inputs")
    if len({fold.study_digest for fold in baseline.values()}) != 1:
        raise ValueError("baseline folds use different study recipes")
    for domain in domains:
        if len({baseline[(seed, domain)].training_pair_digest for seed in seeds}) != 1:
            raise ValueError(f"baseline pairs changed across seeds for domain={domain!r}")
    candidates: dict[str, dict[tuple[int, str], Fold]] = {}
    for arm in arms:
        grid = {(fold.seed, fold.domain): fold for fold in folds.values() if fold.arm == arm}
        if set(grid) != expected:
            missing = sorted(expected - set(grid))
            extra = sorted(set(grid) - expected)
            raise ValueError(f"candidate arm={arm!r} grid mismatch; missing={missing}, extra={extra}")
        for cell, candidate in grid.items():
            base = baseline[cell]
            if candidate.study_digest != base.study_digest:
                raise ValueError(f"candidate arm={arm!r}, cell={cell} uses a different study recipe")
            if candidate.prepared_digest != base.prepared_digest:
                raise ValueError(f"candidate arm={arm!r}, cell={cell} uses different prepared inputs")
            if base.baseline_pair_digest != base.training_pair_digest:
                raise ValueError(f"baseline cell={cell} has inconsistent pair digests")
            if candidate.baseline_pair_digest != base.training_pair_digest:
                raise ValueError(f"candidate arm={arm!r}, cell={cell} uses the wrong baseline pairs")
            if candidate.training_pair_digest == base.training_pair_digest:
                raise ValueError(f"candidate arm={arm!r}, cell={cell} added no training pairs")
            keys = candidate.evaluation_identity.keys() | base.evaluation_identity.keys()
            if any(candidate.evaluation_identity.get(key) != base.evaluation_identity.get(key) for key in keys):
                raise ValueError(f"candidate arm={arm!r}, cell={cell} uses different evaluation inputs")
        candidates[arm] = grid
    return baseline, candidates


def evaluate_gates(
    mean_delta_b3: float,
    q05_delta_b3: float,
    worst_domain_delta_b3: float,
) -> dict[str, dict[str, float | bool]]:
    """Return the three inclusive release decisions."""

    values = (
        ("mean_delta_b3", mean_delta_b3, MIN_MEAN_DELTA_B3),
        ("q05_delta_b3", q05_delta_b3, MIN_Q05_DELTA_B3),
        ("worst_domain_delta_b3", worst_domain_delta_b3, MIN_WORST_DOMAIN_DELTA_B3),
    )
    return {
        name: {"value": value, "threshold": threshold, "passed": value >= threshold}
        for name, value, threshold in values
    }


def _score_arm(
    arm: str,
    baseline: dict[tuple[int, str], Fold],
    candidate: dict[tuple[int, str], Fold],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    seeds = sorted({seed for seed, _ in baseline})
    domains = [domain for domain in GOLD_DOMAINS if (seeds[0], domain) in baseline]
    if not domains:
        raise ValueError("no gold-domain B3 folds found")
    missing_gold = sorted(set(GOLD_DOMAINS).difference(domains))
    deltas = np.empty((len(seeds), len(domains)))
    for seed_index, seed in enumerate(seeds):
        for domain_index, domain in enumerate(domains):
            base_value = baseline[(seed, domain)].b3_f1
            candidate_value = candidate[(seed, domain)].b3_f1
            if base_value is None or candidate_value is None:
                raise ValueError(f"gold-domain B3 is missing for arm={arm!r}, seed={seed}, domain={domain}")
            deltas[seed_index, domain_index] = candidate_value - base_value
    rng = np.random.default_rng(bootstrap_seed)
    seed_draws = rng.integers(len(seeds), size=(bootstrap_samples, len(seeds)))
    domain_draws = rng.integers(len(domains), size=(bootstrap_samples, len(domains)))
    draws = deltas[seed_draws[:, :, None], domain_draws[:, None, :]].mean(axis=(1, 2))
    mean = float(deltas.mean())
    q05 = float(np.quantile(draws, 0.05))
    domain_means = {domain: float(deltas[:, index].mean()) for index, domain in enumerate(domains)}
    worst = min(domain_means.values())
    gates = evaluate_gates(mean, q05, worst)
    return {
        "arm": arm,
        "training_seeds": seeds,
        "gold_domains": domains,
        "mean_delta_b3": mean,
        "q05_delta_b3": q05,
        "worst_domain_delta_b3": worst,
        "domain_mean_delta_b3": domain_means,
        "missing_gold_domains": missing_gold,
        "coverage_complete": not missing_gold,
        "gates": gates,
        "passed": not missing_gold and all(gate["passed"] for gate in gates.values()),
    }


def analyze(
    inputs: list[Path],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    """Load, pair, and score every candidate arm in the supplied results."""

    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    folds = load_folds(inputs)
    baseline, candidates = _paired_grids(folds)
    return {
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed, "method": "two_way_paired"},
        "arms": [
            _score_arm(
                arm,
                baseline,
                candidates[arm],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for arm in sorted(candidates)
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="Result JSON files or directories")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output is not None and args.output.exists():
        raise SystemExit(f"analysis output already exists: {args.output}")
    payload = analyze(
        args.results,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()

"""Audit and score additive linker-dose results against the frozen baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._pair_ablation.modeling import (  # noqa: E402
    ADDITIVE_LINKER_SOURCE_SETS,
    LINKER_FAMILIES,
)
from scripts._pair_ablation.ranking import (  # noqa: E402
    ALL_DOMAINS,
    B3_DOMAINS,
    BOUNDARY_ABS_TOLERANCE,
    PAIR_DOMAINS,
    PROXY_DOMAINS,
    LoadedFold,
    load_ranking_input,
    make_bootstrap_draws,
    score_candidate,
)
from scripts._pair_ablation.results import strict_json_digest  # noqa: E402
from scripts._pair_ablation.run_identity import load_run_manifest  # noqa: E402

SCHEMA_VERSION = "s2and_additive_linker_dose_analysis_v1"
INITIAL_DOSES = (2_500, 5_000, 10_000)
REFINEMENT_TOLERANCE = 500
EXPECTED_TRAINING_SEEDS = (1111, 2222, 3333)
MIN_MEAN_DELTA_B3 = 0.0
MIN_Q05_DELTA_B3 = -0.002
MIN_WORST_DOMAIN_DELTA_B3 = -0.010
_ARM_PATTERN = re.compile(r"uniform_100k_plus_linker_(all13|big7)_([1-9][0-9]*)")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _at_least(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value,
        threshold,
        rel_tol=0,
        abs_tol=BOUNDARY_ABS_TOLERANCE,
    )


def _arm_identity(arm: str) -> tuple[str, int]:
    match = _ARM_PATTERN.fullmatch(arm)
    if match is None:
        raise ValueError(f"candidate arm is not an additive linker-dose recipe: {arm!r}")
    source_set, raw_dose = match.groups()
    dose = int(raw_dose)
    if dose <= 0 or dose % 2:
        raise ValueError(f"candidate additive linker dose must be positive and even: {arm!r}")
    return source_set, dose


def _load_many(paths: Iterable[Path]) -> tuple[LoadedFold, ...]:
    loaded: list[LoadedFold] = []
    for path in paths:
        loaded.extend(load_ranking_input(path.resolve()))
    return tuple(loaded)


def _frozen_baseline_index(
    folds: Sequence[LoadedFold],
) -> tuple[dict[int, dict[str, LoadedFold]], tuple[int, ...]]:
    indexed: dict[int, dict[str, LoadedFold]] = defaultdict(dict)
    for fold in folds:
        payload = fold.payload
        if payload["arm"] != "uniform_100k":
            continue
        seed = int(payload["training_seed"])
        domain = str(payload["held_out_domain"])
        if domain in indexed[seed]:
            raise ValueError(f"duplicate frozen baseline cell seed={seed}, domain={domain}")
        indexed[seed][domain] = fold
    seeds = tuple(sorted(indexed))
    if seeds != EXPECTED_TRAINING_SEEDS:
        raise ValueError(
            f"frozen baseline seeds must be exactly {EXPECTED_TRAINING_SEEDS}; observed={seeds}"
        )
    for seed in seeds:
        if set(indexed[seed]) != set(ALL_DOMAINS):
            raise ValueError(
                f"frozen baseline seed={seed} domain grid mismatch: "
                f"expected={sorted(ALL_DOMAINS)}, observed={sorted(indexed[seed])}"
            )
    return dict(indexed), seeds


def _candidate_index(
    folds: Sequence[LoadedFold],
    *,
    seeds: Sequence[int],
) -> tuple[dict[str, dict[int, dict[str, LoadedFold]]], str]:
    indexed: dict[str, dict[int, dict[str, LoadedFold]]] = defaultdict(lambda: defaultdict(dict))
    comparison_identities = {fold.comparison_identity_sha256 for fold in folds}
    if len(comparison_identities) != 1:
        raise ValueError(
            "candidate additive runs must share one comparison identity; "
            f"observed={sorted(comparison_identities)}"
        )
    for fold in folds:
        payload = fold.payload
        arm = str(payload["arm"])
        _arm_identity(arm)
        seed = int(payload["training_seed"])
        domain = str(payload["held_out_domain"])
        if seed not in seeds:
            raise ValueError(f"candidate seed={seed} is absent from the frozen baseline")
        if domain in indexed[arm][seed]:
            raise ValueError(f"duplicate candidate cell arm={arm}, seed={seed}, domain={domain}")
        indexed[arm][seed][domain] = fold
    for arm, by_seed in indexed.items():
        if set(by_seed) != set(seeds):
            raise ValueError(f"candidate arm={arm!r} seed grid mismatch")
        for seed in seeds:
            if set(by_seed[seed]) != set(ALL_DOMAINS):
                raise ValueError(f"candidate arm={arm!r}, seed={seed} domain grid mismatch")
    return {arm: dict(by_seed) for arm, by_seed in indexed.items()}, next(iter(comparison_identities))


def _normalized_run_contract(fold: LoadedFold) -> dict[str, Any]:
    manifest = load_run_manifest(fold.run_manifest_path)
    config = dict(manifest["config"])
    config.pop("training_seed", None)
    config.pop("arm_names", None)
    input_identity = manifest["input_identity"]
    raw_input_identity = {
        key: input_identity[key]
        for key in ("roots", "catalog_files", "feature_artifacts")
    }
    return {
        "adapter": manifest["adapter"],
        "config": config,
        "donor_model_sha256": manifest["donor_model_sha256"],
        "featurizer_version": manifest["featurizer_version"],
        "input_identity": raw_input_identity,
        "runtime_versions": manifest["runtime_versions"],
        "rust_extension_sha256": manifest["rust_extension_sha256"],
        "rust_version": manifest["rust_version"],
        "thread_environment": manifest["thread_environment"],
    }


def _validate_run_contracts(
    baseline: dict[int, dict[str, LoadedFold]],
    candidates: dict[str, dict[int, dict[str, LoadedFold]]],
    seeds: Sequence[int],
) -> None:
    baseline_contracts = {
        seed: _normalized_run_contract(baseline[seed][ALL_DOMAINS[0]])
        for seed in seeds
    }
    if any(contract != baseline_contracts[seeds[0]] for contract in baseline_contracts.values()):
        raise ValueError("frozen baseline run contract differs across training seeds")
    expected = baseline_contracts[seeds[0]]
    observed_manifests: dict[Path, dict[str, Any]] = {}
    for by_seed in candidates.values():
        for seed in seeds:
            for fold in by_seed[seed].values():
                observed_manifests.setdefault(fold.run_manifest_path, _normalized_run_contract(fold))
    mismatches = sorted(str(path) for path, contract in observed_manifests.items() if contract != expected)
    if mismatches:
        raise ValueError(f"candidate model/evaluation run contract differs from frozen baseline: {mismatches}")


def _validate_additive_audit(
    candidate: LoadedFold,
    baseline: LoadedFold,
    *,
    source_set: str,
    dose: int,
) -> tuple[int, float, list[dict[str, Any]]]:
    payload = candidate.payload
    base_payload = baseline.payload
    if (
        payload["evaluation_seed"],
        payload["evaluation_pair_digest"],
        payload["b3_evaluation_digest"],
    ) != (
        base_payload["evaluation_seed"],
        base_payload["evaluation_pair_digest"],
        base_payload["b3_evaluation_digest"],
    ):
        raise ValueError(
            "candidate evaluation identity differs from frozen baseline: "
            f"arm={payload['arm']}, seed={payload['training_seed']}, heldout={payload['held_out_domain']}"
        )
    audit = payload["pair_recipe_assembly"]
    if not isinstance(audit, dict) or audit.get("mode") != "additive_linker":
        raise ValueError("candidate result lacks an additive-linker assembly audit")
    if audit.get("base_pair_digest") != base_payload["training_pair_digest"]:
        raise ValueError("candidate additive base digest differs from frozen baseline")
    if audit.get("training_pair_digest") != payload["training_pair_digest"]:
        raise ValueError("candidate additive audit training digest differs from the result")
    if audit.get("base_rows_after_lodo") != base_payload["training_rows"]:
        raise ValueError("candidate additive base row count differs from frozen baseline")
    if audit.get("linker_source_set") != source_set or audit.get("linker_cap_per_domain") != dose:
        raise ValueError("candidate additive audit source set or cap differs from its arm name")
    expected_domains = ADDITIVE_LINKER_SOURCE_SETS[source_set]  # type: ignore[index]
    if tuple(audit.get("linker_source_domains", ())) != expected_domains:
        raise ValueError("candidate additive audit has the wrong source-domain set")
    held_out = str(payload["held_out_domain"])
    expected_after_lodo = tuple(domain for domain in expected_domains if domain != held_out)
    if tuple(audit.get("linker_eligible_domains_after_lodo", ())) != expected_after_lodo:
        raise ValueError("candidate additive audit did not apply held-out exclusion to linker domains")
    linker_rows = int(audit.get("linker_selected_rows", -1))
    if (
        audit.get("held_out_rows") != 0
        or audit.get("final_rows") != payload["training_rows"]
        or payload["training_rows"] != base_payload["training_rows"] + linker_rows
    ):
        raise ValueError("candidate additive final rows do not equal frozen base plus linker rows")
    linker_source_counts = [
        record
        for record in payload["training_source_counts"]
        if record["source_family"] in LINKER_FAMILIES
    ]
    unexpected_count_domains = sorted(
        {
            str(record["source_domain"])
            for record in linker_source_counts
            if record["source_domain"] not in expected_after_lodo
        }
    )
    if unexpected_count_domains:
        raise ValueError(f"candidate linker source counts contain excluded domains: {unexpected_count_domains}")
    linker_count_rows = sum(int(record["rows"]) for record in linker_source_counts)
    if linker_count_rows != linker_rows:
        raise ValueError("candidate linker source counts do not match the assembly audit")
    domain_audits = audit.get("linker_domains")
    if not isinstance(domain_audits, list):
        raise ValueError("candidate additive audit linker_domains must be a list")
    if tuple(record.get("source_domain") for record in domain_audits) != tuple(sorted(expected_after_lodo)):
        raise ValueError("candidate additive linker-domain audit grid mismatch")
    enriched_domain_audits = []
    for record in domain_audits:
        source_domain = str(record.get("source_domain"))
        selected_per_class = int(record.get("selected_per_class", -1))
        if int(record.get("selected_rows", -1)) != 2 * selected_per_class:
            raise ValueError("candidate additive linker-domain selection is not label-balanced")
        source_records = [
            source_record
            for source_record in linker_source_counts
            if source_record["source_domain"] == source_domain
        ]
        actual_positives = sum(int(source_record["positives"]) for source_record in source_records)
        actual_negatives = sum(int(source_record["negatives"]) for source_record in source_records)
        actual_rows = sum(int(source_record["rows"]) for source_record in source_records)
        if (actual_positives, actual_negatives, actual_rows) != (
            selected_per_class,
            selected_per_class,
            2 * selected_per_class,
        ):
            raise ValueError(
                "candidate actual linker source counts do not match balanced audit quotas: "
                f"source_domain={source_domain!r}"
            )
        enriched_domain_audits.append(
            {
                **record,
                "actual_positives": actual_positives,
                "actual_negatives": actual_negatives,
                "actual_rows": actual_rows,
                "source_family_counts_json": json.dumps(
                    sorted(source_records, key=lambda value: value["source_family"]),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
    if audit.get("frozen_baseline_result_sha256") != baseline.sha256:
        raise ValueError("candidate audit references the wrong frozen baseline result hash")
    if audit.get("frozen_baseline_run_manifest_sha256") != baseline.run_manifest_sha256:
        raise ValueError("candidate audit references the wrong frozen baseline run-manifest hash")
    linker_share = linker_rows / int(payload["training_rows"])
    return linker_rows, linker_share, enriched_domain_audits


def _score_arm(
    arm: str,
    candidate: dict[int, dict[str, LoadedFold]],
    baseline: dict[int, dict[str, LoadedFold]],
    seeds: Sequence[int],
    *,
    draws: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_set, dose = _arm_identity(arm)
    delta_auprc = np.empty((len(seeds), len(PAIR_DOMAINS)), dtype=np.float64)
    delta_auroc = np.empty_like(delta_auprc)
    delta_b3 = np.empty((len(seeds), len(B3_DOMAINS)), dtype=np.float64)
    absolute_b3 = np.empty_like(delta_b3)
    proxy_delta_auprc = np.empty((len(seeds), len(PROXY_DOMAINS)), dtype=np.float64)
    proxy_delta_auroc = np.empty_like(proxy_delta_auprc)
    training_rows = np.empty((len(seeds), len(ALL_DOMAINS)), dtype=np.float64)
    paired_rows: list[dict[str, Any]] = []
    realization_rows: list[dict[str, Any]] = []
    recipe_ids: set[str] = set()
    recipes: list[dict[str, Any]] = []
    linker_rows_values: list[int] = []
    linker_share_values: list[float] = []

    for seed_index, seed in enumerate(seeds):
        for domain_index, domain in enumerate(ALL_DOMAINS):
            row = candidate[seed][domain]
            base = baseline[seed][domain]
            payload = row.payload
            base_payload = base.payload
            linker_rows, linker_share, domain_audits = _validate_additive_audit(
                row,
                base,
                source_set=source_set,
                dose=dose,
            )
            linker_rows_values.append(linker_rows)
            linker_share_values.append(linker_share)
            recipe_ids.add(str(payload["recipe_id"]))
            recipes.append(payload["recipe"])
            ap_delta = float(payload["pairwise"]["auprc"] - base_payload["pairwise"]["auprc"])
            auc_delta = float(payload["pairwise"]["auroc"] - base_payload["pairwise"]["auroc"])
            b3_value = None
            if domain in PAIR_DOMAINS:
                pair_index = PAIR_DOMAINS.index(domain)
                delta_auprc[seed_index, pair_index] = ap_delta
                delta_auroc[seed_index, pair_index] = auc_delta
            if domain in B3_DOMAINS:
                b3_index = B3_DOMAINS.index(domain)
                b3_value = float(payload["b3"]["f1"] - base_payload["b3"]["f1"])
                delta_b3[seed_index, b3_index] = b3_value
                absolute_b3[seed_index, b3_index] = float(payload["b3"]["f1"])
            if domain in PROXY_DOMAINS:
                proxy_index = PROXY_DOMAINS.index(domain)
                proxy_delta_auprc[seed_index, proxy_index] = ap_delta
                proxy_delta_auroc[seed_index, proxy_index] = auc_delta
            training_rows[seed_index, domain_index] = float(payload["training_rows"])
            paired_rows.append(
                {
                    "arm": arm,
                    "source_set": source_set,
                    "dose_per_source": dose,
                    "training_seed": seed,
                    "held_out_domain": domain,
                    "role": (
                        "cluster_gold"
                        if domain in B3_DOMAINS
                        else "pair_gold"
                        if domain == "medline"
                        else "proxy"
                    ),
                    "delta_auprc": ap_delta,
                    "delta_auroc": auc_delta,
                    "delta_b3": b3_value,
                    "training_rows": int(payload["training_rows"]),
                    "linker_rows": linker_rows,
                    "linker_share": linker_share,
                }
            )
            for domain_audit in domain_audits:
                realization_rows.append(
                    {
                        "arm": arm,
                        "source_set": source_set,
                        "dose_per_source": dose,
                        "training_seed": seed,
                        "held_out_domain": domain,
                        **domain_audit,
                    }
                )

    if len(recipe_ids) != 1 or any(recipe != recipes[0] for recipe in recipes[1:]):
        raise ValueError(f"candidate arm={arm!r} recipe identity varies across folds")
    score = score_candidate(
        recipe_id=next(iter(recipe_ids)),
        arm=arm,
        recipe=recipes[0],
        seeds=seeds,
        delta_auprc=delta_auprc,
        delta_auroc=delta_auroc,
        delta_b3=delta_b3,
        absolute_b3=absolute_b3,
        proxy_delta_auprc=proxy_delta_auprc,
        proxy_delta_auroc=proxy_delta_auroc,
        training_rows=training_rows,
        draws=draws,
    )
    b3_gates = {
        "mean_delta_b3": {
            "value": score["mean_delta_b3"],
            "threshold": MIN_MEAN_DELTA_B3,
            "comparator": ">=",
            "passed": _at_least(score["mean_delta_b3"], MIN_MEAN_DELTA_B3),
        },
        "q05_delta_b3": {
            "value": score["bootstrap"]["delta_b3"]["q05"],
            "threshold": MIN_Q05_DELTA_B3,
            "comparator": ">=",
            "passed": _at_least(score["bootstrap"]["delta_b3"]["q05"], MIN_Q05_DELTA_B3),
        },
        "worst_domain_delta_b3": {
            "value": score["worst_domain_delta_b3"],
            "threshold": MIN_WORST_DOMAIN_DELTA_B3,
            "comparator": ">=",
            "passed": _at_least(score["worst_domain_delta_b3"], MIN_WORST_DOMAIN_DELTA_B3),
        },
    }
    score["additive_linker"] = {
        "source_set": source_set,
        "dose_per_source": dose,
        "b3_safety_gates": b3_gates,
        "b3_safety_eligible": all(gate["passed"] for gate in b3_gates.values()),
        "realized_linker_rows": {
            "minimum": min(linker_rows_values),
            "maximum": max(linker_rows_values),
            "mean": float(np.mean(linker_rows_values)),
        },
        "realized_linker_share": {
            "minimum": min(linker_share_values),
            "maximum": max(linker_share_values),
            "mean": float(np.mean(linker_share_values)),
        },
    }
    return score, paired_rows, realization_rows


def _even_midpoint(lower: int, upper: int) -> int | None:
    if upper - lower <= REFINEMENT_TOLERANCE:
        return None
    midpoint = (lower + upper) // 2
    midpoint -= midpoint % 2
    if midpoint <= lower:
        midpoint = lower + 2
    if midpoint >= upper:
        midpoint = upper - 2
    return midpoint if lower < midpoint < upper else None


def plan_next_doses(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the adaptive next arms or the final conservative boundary."""

    by_set: dict[str, dict[int, bool]] = defaultdict(dict)
    for score in scores:
        additive = score["additive_linker"]
        by_set[additive["source_set"]][int(additive["dose_per_source"])] = bool(
            additive["b3_safety_eligible"]
        )

    next_requests: list[tuple[str, int, str]] = []
    boundaries: dict[str, Any] = {}
    for source_set in ("all13", "big7"):
        observed = by_set[source_set]
        missing_initial = [dose for dose in INITIAL_DOSES if dose not in observed]
        if missing_initial:
            next_requests.extend((source_set, dose, "initial_sweep") for dose in missing_initial)
            continue
        if observed[10_000]:
            if 25_000 not in observed:
                next_requests.append((source_set, 25_000, "ten_thousand_passed"))
                continue
            if observed[25_000] and 50_000 not in observed:
                next_requests.append((source_set, 50_000, "twenty_five_thousand_passed"))
                continue

        tested = sorted(observed)
        failing = [dose for dose in tested if not observed[dose]]
        if not failing:
            if max(tested) < 50_000:
                raise AssertionError(f"source_set={source_set} has no failure but the 50k cap was not tested")
            boundaries[source_set] = {
                "largest_passing_dose": 50_000,
                "smallest_failing_dose": None,
                "bracket_width": None,
                "non_monotonic": False,
            }
            continue

        first_failure = min(failing)
        passing_below = [0, *(dose for dose in tested if dose < first_failure and observed[dose])]
        largest_pass = max(passing_below)
        midpoint = _even_midpoint(largest_pass, first_failure)
        non_monotonic = any(observed[dose] for dose in tested if dose > first_failure)
        if midpoint is not None and midpoint not in observed:
            next_requests.append((source_set, midpoint, "refine_first_pass_fail_bracket"))
            continue
        boundaries[source_set] = {
            "largest_passing_dose": largest_pass,
            "smallest_failing_dose": first_failure,
            "bracket_width": first_failure - largest_pass,
            "non_monotonic": non_monotonic,
        }

    unique_requests = sorted(set(next_requests), key=lambda item: (item[1], item[0], item[2]))
    return {
        "complete": not unique_requests and set(boundaries) == {"all13", "big7"},
        "next_arms": [
            {
                "arm": f"uniform_100k_plus_linker_{source_set}_{dose}",
                "source_set": source_set,
                "dose_per_source": dose,
                "reason": reason,
            }
            for source_set, dose, reason in unique_requests
        ],
        "boundaries": boundaries,
    }


def analyze(
    *,
    baseline_paths: Sequence[Path],
    candidate_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    baseline_folds = _load_many(baseline_paths)
    candidate_folds = _load_many(candidate_paths)
    baseline, seeds = _frozen_baseline_index(baseline_folds)
    candidates, candidate_comparison_identity = _candidate_index(candidate_folds, seeds=seeds)
    _validate_run_contracts(baseline, candidates, seeds)
    draws = make_bootstrap_draws(len(seeds))
    scores: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    realization_rows: list[dict[str, Any]] = []
    for arm in sorted(candidates, key=lambda value: (_arm_identity(value)[1], _arm_identity(value)[0])):
        score, arm_paired_rows, arm_realization_rows = _score_arm(
            arm,
            candidates[arm],
            baseline,
            seeds,
            draws=draws,
        )
        scores.append(score)
        paired_rows.extend(arm_paired_rows)
        realization_rows.extend(arm_realization_rows)

    adaptive_plan = plan_next_doses(scores)
    baseline_inputs = [
        {"path": str(path.resolve()), "sha256": _sha256(path.resolve())} for path in baseline_paths
    ]
    candidate_inputs = [
        {"path": str(path.resolve()), "sha256": _sha256(path.resolve())} for path in candidate_paths
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": "",
        "training_seeds": list(seeds),
        "baseline_ranking_inputs": baseline_inputs,
        "candidate_ranking_inputs": candidate_inputs,
        "candidate_comparison_identity_sha256": candidate_comparison_identity,
        "gates": {
            "minimum_mean_delta_b3": MIN_MEAN_DELTA_B3,
            "minimum_q05_delta_b3": MIN_Q05_DELTA_B3,
            "minimum_worst_domain_delta_b3": MIN_WORST_DOMAIN_DELTA_B3,
            "boundary_absolute_tolerance": BOUNDARY_ABS_TOLERANCE,
        },
        "scores": scores,
        "adaptive_plan": adaptive_plan,
    }
    payload["analysis_id"] = strict_json_digest({key: value for key, value in payload.items() if key != "analysis_id"})
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "analysis.json", payload)
    summary_rows = []
    for score in scores:
        additive = score["additive_linker"]
        summary_rows.append(
            {
                "arm": score["arm"],
                "source_set": additive["source_set"],
                "dose_per_source": additive["dose_per_source"],
                "b3_safety_eligible": additive["b3_safety_eligible"],
                "mean_delta_b3": score["mean_delta_b3"],
                "q05_delta_b3": score["bootstrap"]["delta_b3"]["q05"],
                "worst_domain_delta_b3": score["worst_domain_delta_b3"],
                "mean_delta_auprc": score["mean_delta_auprc"],
                "mean_delta_auroc": score["mean_delta_auroc"],
                "mean_linker_rows": additive["realized_linker_rows"]["mean"],
                "min_linker_rows": additive["realized_linker_rows"]["minimum"],
                "max_linker_rows": additive["realized_linker_rows"]["maximum"],
                "mean_linker_share": additive["realized_linker_share"]["mean"],
                "min_linker_share": additive["realized_linker_share"]["minimum"],
                "max_linker_share": additive["realized_linker_share"]["maximum"],
            }
        )
    _write_csv(output_dir / "arm_summary.csv", summary_rows)
    _write_csv(output_dir / "paired_metrics.csv", paired_rows)
    _write_csv(output_dir / "linker_source_realization.csv", realization_rows)
    _write_report(output_dir / "REPORT.md", payload, summary_rows)
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _write_report(path: Path, payload: dict[str, Any], summary_rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Additive linker-dose study",
        "",
        f"- complete: `{payload['adaptive_plan']['complete']}`",
        f"- analysis id: `{payload['analysis_id']}`",
        f"- training seeds: `{payload['training_seeds']}`",
        "",
        "| source set | cap/source | mean ΔB³ | q05 ΔB³ | worst-domain ΔB³ | safe | mean linker share |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["source_set"]),
                    str(row["dose_per_source"]),
                    _metric(row["mean_delta_b3"]),
                    _metric(row["q05_delta_b3"]),
                    _metric(row["worst_domain_delta_b3"]),
                    str(bool(row["b3_safety_eligible"])),
                    _metric(row["mean_linker_share"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Adaptive plan",
            "",
            "```json",
            json.dumps(payload["adaptive_plan"], allow_nan=False, indent=2, sort_keys=True),
            "```",
            "",
            "Name-block metrics are proxy diagnostics and do not enter the B³ safety decision.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ranking-input", type=Path, action="append", required=True)
    parser.add_argument("--candidate-ranking-input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = analyze(
        baseline_paths=args.baseline_ranking_input,
        candidate_paths=args.candidate_ranking_input,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(payload["adaptive_plan"], allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

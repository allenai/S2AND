"""Authoritative multi-run ranking for pair-source ablation results.

The runner does not yet emit a multi-run study manifest, so this module uses a
small strict input contract rather than inferring expected identities from the
results being ranked.  The input JSON has this shape::

    {
      "schema_version": "s2and_pair_ablation_ranking_input_v2",
      "folds": [
        {
          "path": "results/arm/domain.json",
          "result_sha256": "<64 lowercase hex characters>",
          "run_manifest_path": "run_manifest.json",
          "run_manifest_sha256": "<64 lowercase hex characters>",
          "expected": {
            "run_id": "...",
            "arm": "...",
            "source_families": ["..."],
            "held_out_domain": "...",
            "training_seed": 1111,
            "evaluation_seed": 1111,
            "recipe": {...},
            "training_pair_digest": "...",
            "evaluation_pair_digest": "...",
            "b3_evaluation_digest": "... or null",
            "oracle_kind": "...",
            "b3_scope": "test or null",
            "requires_recipe_audit": false
          }
        }
      ]
    }

Relative fold paths are resolved against the input manifest directory.  Every
fold is loaded through :func:`scripts._pair_ablation.results.load_fold_result`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts._pair_ablation.results import (
    FoldResultExpectation,
    load_fold_result,
    load_strict_json,
    recipe_id_for,
    strict_json_digest,
)
from scripts._pair_ablation.run_identity import (
    COMPARISON_IDENTITY_SCHEMA_VERSION,
    load_run_manifest,
)

RANKING_INPUT_SCHEMA_VERSION = "s2and_pair_ablation_ranking_input_v2"
RANKING_OUTPUT_SCHEMA_VERSION = "s2and_pair_ablation_ranking_output_v2"
RANKING_POLICY_VERSION = "s2and_pair_ablation_final_policy_v1"

BASELINE_ARM = "uniform_100k"
PAIR_DOMAINS = ("aminer", "arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath", "medline")
B3_DOMAINS = PAIR_DOMAINS[:7]
PROXY_DOMAINS = ("a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park")
ALL_DOMAINS = (*PAIR_DOMAINS, *PROXY_DOMAINS)

BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_SEED = 20_260_711
BOOTSTRAP_QUANTILE_METHOD = "linear"
MIN_TRAINING_SEEDS = 3

MIN_Q05_DELTA_B3 = -0.002
MIN_WORST_DOMAIN_DELTA_B3 = -0.010
MIN_WORST_DOMAIN_DELTA_AUPRC = -0.010
MIN_WORST_DOMAIN_DELTA_AUROC = -0.010
MIN_POSITIVE_SEED_FRACTION = 2 / 3
MIN_POSITIVE_AUPRC_DOMAINS = 5
PRACTICAL_AUPRC_TIE = 0.0005
PROXY_MANUAL_REVIEW_LOSS = -0.05
BOUNDARY_ABS_TOLERANCE = 1e-12

_MANIFEST_KEYS = {"folds", "schema_version"}
RANKING_INPUT_FOLD_ENTRY_KEYS = {
    "expected",
    "path",
    "result_sha256",
    "run_manifest_path",
    "run_manifest_sha256",
}
_EXPECTATION_KEYS = {
    "arm",
    "b3_evaluation_digest",
    "b3_scope",
    "evaluation_pair_digest",
    "evaluation_seed",
    "held_out_domain",
    "oracle_kind",
    "recipe",
    "requires_recipe_audit",
    "run_id",
    "source_families",
    "training_pair_digest",
    "training_seed",
}


@dataclass(frozen=True, slots=True)
class LoadedFold:
    """One strictly validated fold plus its content identity."""

    path: Path
    sha256: str
    payload: dict[str, Any]
    run_manifest_path: Path
    run_manifest_sha256: str
    comparison_identity_sha256: str


@dataclass(frozen=True, slots=True)
class BootstrapDraws:
    """Candidate-independent two-way bootstrap indices."""

    seed_indices: np.ndarray
    pair_domain_indices: np.ndarray
    b3_domain_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class RankingArtifacts:
    """Machine-readable ranking plus its flat output tables."""

    ranking: dict[str, Any]
    ranking_rows: tuple[dict[str, Any], ...]
    paired_delta_rows: tuple[dict[str, Any], ...]
    domain_rows: tuple[dict[str, Any], ...]
    bootstrap_rows: tuple[dict[str, Any], ...]
    concentration: dict[str, Any]
    input_manifest: dict[str, Any]


def _require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{context} schema mismatch: missing={missing}, extra={extra}")


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    return value


def _require_digest(value: Any, context: str) -> str:
    digest = _require_text(value, context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expectation_from_dict(value: Any, *, context: str) -> FoldResultExpectation:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(value, _EXPECTATION_KEYS, context)
    source_families = value["source_families"]
    if not isinstance(source_families, list) or any(not isinstance(item, str) or not item for item in source_families):
        raise ValueError(f"{context}.source_families must be a list of non-empty strings")
    if len(source_families) != len(set(source_families)):
        raise ValueError(f"{context}.source_families contains duplicates")
    recipe = value["recipe"]
    if not isinstance(recipe, dict):
        raise ValueError(f"{context}.recipe must be an object")
    requires_recipe_audit = value["requires_recipe_audit"]
    if not isinstance(requires_recipe_audit, bool):
        raise ValueError(f"{context}.requires_recipe_audit must be Boolean")
    b3_digest = value["b3_evaluation_digest"]
    if b3_digest is not None and not isinstance(b3_digest, str):
        raise ValueError(f"{context}.b3_evaluation_digest must be a string or null")
    b3_scope = value["b3_scope"]
    if b3_scope is not None and not isinstance(b3_scope, str):
        raise ValueError(f"{context}.b3_scope must be a string or null")
    return FoldResultExpectation(
        run_id=_require_text(value["run_id"], f"{context}.run_id"),
        arm=_require_text(value["arm"], f"{context}.arm"),
        source_families=tuple(source_families),
        held_out_domain=_require_text(value["held_out_domain"], f"{context}.held_out_domain"),
        training_seed=_require_int(value["training_seed"], f"{context}.training_seed"),
        evaluation_seed=_require_int(value["evaluation_seed"], f"{context}.evaluation_seed"),
        recipe=recipe,
        training_pair_digest=_require_text(value["training_pair_digest"], f"{context}.training_pair_digest"),
        evaluation_pair_digest=_require_text(value["evaluation_pair_digest"], f"{context}.evaluation_pair_digest"),
        b3_evaluation_digest=b3_digest,
        oracle_kind=_require_text(value["oracle_kind"], f"{context}.oracle_kind"),
        b3_scope=b3_scope,
        requires_recipe_audit=requires_recipe_audit,
    )


def load_ranking_input(path: Path) -> tuple[LoadedFold, ...]:
    """Load and strictly validate every fold declared by a ranking manifest."""

    manifest_path = path.resolve()
    payload = load_strict_json(manifest_path)
    _require_exact_keys(payload, _MANIFEST_KEYS, "ranking input")
    if payload["schema_version"] != RANKING_INPUT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported ranking-input schema version: {payload['schema_version']!r}")
    entries = payload["folds"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("ranking input folds must be a non-empty list")

    loaded: list[LoadedFold] = []
    seen_paths: set[Path] = set()
    run_manifest_cache: dict[Path, tuple[str, dict[str, Any]]] = {}
    for index, entry in enumerate(entries):
        context = f"folds[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{context} must be an object")
        _require_exact_keys(entry, RANKING_INPUT_FOLD_ENTRY_KEYS, context)
        raw_path = Path(_require_text(entry["path"], f"{context}.path"))
        result_path = raw_path.resolve() if raw_path.is_absolute() else (manifest_path.parent / raw_path).resolve()
        if result_path in seen_paths:
            raise ValueError(f"ranking input contains duplicate result path: {result_path}")
        seen_paths.add(result_path)
        expected_sha = _require_digest(entry["result_sha256"], f"{context}.result_sha256")
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing declared fold result: {result_path}")
        observed_sha = _sha256_file(result_path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"Fold result SHA-256 mismatch for {result_path}: expected={expected_sha}, observed={observed_sha}"
            )
        expectation = _expectation_from_dict(entry["expected"], context=f"{context}.expected")
        result = load_fold_result(result_path, expected=expectation)

        raw_run_manifest_path = Path(_require_text(entry["run_manifest_path"], f"{context}.run_manifest_path"))
        run_manifest_path = (
            raw_run_manifest_path.resolve()
            if raw_run_manifest_path.is_absolute()
            else (manifest_path.parent / raw_run_manifest_path).resolve()
        )
        expected_run_manifest_sha = _require_digest(
            entry["run_manifest_sha256"],
            f"{context}.run_manifest_sha256",
        )
        if not run_manifest_path.is_file():
            raise FileNotFoundError(f"Missing declared run manifest: {run_manifest_path}")
        observed_run_manifest_sha = _sha256_file(run_manifest_path)
        if observed_run_manifest_sha != expected_run_manifest_sha:
            raise ValueError(
                "Run manifest SHA-256 mismatch for "
                f"{run_manifest_path}: expected={expected_run_manifest_sha}, observed={observed_run_manifest_sha}"
            )
        cached = run_manifest_cache.get(run_manifest_path)
        if cached is None:
            run_manifest = load_run_manifest(run_manifest_path)
            run_manifest_cache[run_manifest_path] = (observed_run_manifest_sha, run_manifest)
        else:
            cached_sha, run_manifest = cached
            if cached_sha != observed_run_manifest_sha:
                raise ValueError(f"Run manifest changed while ranking inputs were loaded: {run_manifest_path}")

        if run_manifest["run_id"] != result["run_id"]:
            raise ValueError(
                f"Fold result run_id does not match {run_manifest_path}: "
                f"result={result['run_id']}, manifest={run_manifest['run_id']}"
            )
        run_config = run_manifest["config"]
        for key in ("training_seed", "evaluation_seed"):
            if run_config.get(key) != result[key]:
                raise ValueError(
                    f"Fold result {key} does not match {run_manifest_path}: "
                    f"result={result[key]!r}, manifest={run_config.get(key)!r}"
                )
        if result["arm"] not in run_config["arm_names"]:
            raise ValueError(f"Fold result arm={result['arm']!r} is absent from {run_manifest_path}")
        matching_recipes = [
            recipe_entry for recipe_entry in run_manifest["recipes"] if recipe_entry["recipe_id"] == result["recipe_id"]
        ]
        if len(matching_recipes) != 1 or matching_recipes[0]["recipe"] != result["recipe"]:
            raise ValueError(f"Fold result recipe is absent or inconsistent in {run_manifest_path}")
        loaded.append(
            LoadedFold(
                path=result_path,
                sha256=observed_sha,
                payload=result,
                run_manifest_path=run_manifest_path,
                run_manifest_sha256=observed_run_manifest_sha,
                comparison_identity_sha256=run_manifest["comparison_identity"]["sha256"],
            )
        )
    return tuple(loaded)


def make_bootstrap_draws(n_seeds: int) -> BootstrapDraws:
    """Create the frozen candidate-independent PCG64 bootstrap draws."""

    if n_seeds < MIN_TRAINING_SEEDS:
        raise ValueError(f"Final ranking requires at least {MIN_TRAINING_SEEDS} training seeds")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return BootstrapDraws(
        seed_indices=rng.integers(0, n_seeds, size=(BOOTSTRAP_REPLICATES, n_seeds)),
        pair_domain_indices=rng.integers(
            0,
            len(PAIR_DOMAINS),
            size=(BOOTSTRAP_REPLICATES, len(PAIR_DOMAINS)),
        ),
        b3_domain_indices=rng.integers(
            0,
            len(B3_DOMAINS),
            size=(BOOTSTRAP_REPLICATES, len(B3_DOMAINS)),
        ),
    )


def two_way_paired_bootstrap(
    deltas: np.ndarray,
    *,
    seed_indices: np.ndarray,
    domain_indices: np.ndarray,
) -> np.ndarray:
    """Resample both seed and domain axes after paired deltas are formed."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap deltas must be one finite seed-by-domain matrix")
    if seed_indices.ndim != 2 or domain_indices.ndim != 2:
        raise ValueError("bootstrap indices must be two matrices")
    if len(seed_indices) != len(domain_indices):
        raise ValueError("bootstrap seed and domain draws must have equal replicate counts")
    if seed_indices.shape[1] != values.shape[0] or domain_indices.shape[1] != values.shape[1]:
        raise ValueError("bootstrap draw shapes do not match the delta matrix")
    sampled = values[seed_indices[:, :, None], domain_indices[:, None, :]]
    return sampled.mean(axis=(1, 2))


def _bootstrap_summary(values: np.ndarray) -> dict[str, float]:
    q05, median, q95 = np.quantile(
        values,
        (0.05, 0.5, 0.95),
        method=BOOTSTRAP_QUANTILE_METHOD,
    )
    return {
        "q05": float(q05),
        "mean": float(values.mean()),
        "median": float(median),
        "q95": float(q95),
    }


def _gate(value: float | int, threshold: float | int, comparator: str, passed: bool) -> dict[str, Any]:
    return {"comparator": comparator, "passed": bool(passed), "threshold": threshold, "value": value}


def _at_least(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(value, threshold, rel_tol=0, abs_tol=BOUNDARY_ABS_TOLERANCE)


def _strictly_less(value: float, threshold: float) -> bool:
    return value < threshold and not math.isclose(
        value,
        threshold,
        rel_tol=0,
        abs_tol=BOUNDARY_ABS_TOLERANCE,
    )


def concentration_diagnostics(domain_mean_delta_auprc: np.ndarray) -> dict[str, Any]:
    """Return positive-gain concentration and leave-one-domain-out diagnostics."""

    values = np.asarray(domain_mean_delta_auprc, dtype=np.float64)
    if values.shape != (len(PAIR_DOMAINS),) or not np.isfinite(values).all():
        raise ValueError("concentration values must cover the eight pair domains")
    positive = np.maximum(values, 0)
    positive_sum = float(positive.sum())
    order = sorted(range(len(PAIR_DOMAINS)), key=lambda index: (-positive[index], index))
    lodo = {domain: float(np.delete(values, index).mean()) for index, domain in enumerate(PAIR_DOMAINS)}
    min_lodo_domain = min(PAIR_DOMAINS, key=lambda domain: (lodo[domain], PAIR_DOMAINS.index(domain)))
    return {
        "positive_gain_sum": positive_sum,
        "positive_domain_count": int((values > 0).sum()),
        "top1_domain": None if positive_sum == 0 else PAIR_DOMAINS[order[0]],
        "top1_share": None if positive_sum == 0 else float(positive[order[0]] / positive_sum),
        "top2_domains": [] if positive_sum == 0 else [PAIR_DOMAINS[index] for index in order[:2]],
        "top2_share": None if positive_sum == 0 else float(positive[order[:2]].sum() / positive_sum),
        "leave_one_domain_out": lodo,
        "minimum_leave_one_domain_out": float(lodo[min_lodo_domain]),
        "minimum_leave_one_domain_out_removed_domain": min_lodo_domain,
    }


def score_candidate(
    *,
    recipe_id: str,
    arm: str,
    recipe: dict[str, Any],
    seeds: Sequence[int],
    delta_auprc: np.ndarray,
    delta_auroc: np.ndarray,
    delta_b3: np.ndarray,
    absolute_b3: np.ndarray,
    proxy_delta_auprc: np.ndarray,
    proxy_delta_auroc: np.ndarray,
    training_rows: np.ndarray,
    draws: BootstrapDraws,
) -> dict[str, Any]:
    """Score one complete recipe against its same-seed baseline."""

    n_seeds = len(seeds)
    pair_shape = (n_seeds, len(PAIR_DOMAINS))
    b3_shape = (n_seeds, len(B3_DOMAINS))
    proxy_shape = (n_seeds, len(PROXY_DOMAINS))
    arrays = {
        "delta_auprc": (np.asarray(delta_auprc, dtype=np.float64), pair_shape),
        "delta_auroc": (np.asarray(delta_auroc, dtype=np.float64), pair_shape),
        "delta_b3": (np.asarray(delta_b3, dtype=np.float64), b3_shape),
        "absolute_b3": (np.asarray(absolute_b3, dtype=np.float64), b3_shape),
        "proxy_delta_auprc": (np.asarray(proxy_delta_auprc, dtype=np.float64), proxy_shape),
        "proxy_delta_auroc": (np.asarray(proxy_delta_auroc, dtype=np.float64), proxy_shape),
    }
    for name, (values, shape) in arrays.items():
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite with shape={shape}, observed={values.shape}")
    row_values = np.asarray(training_rows, dtype=np.float64)
    if row_values.shape != (n_seeds, len(ALL_DOMAINS)) or not np.isfinite(row_values).all():
        raise ValueError("training_rows must cover every seed and held-out domain")

    ap_bootstrap = two_way_paired_bootstrap(
        arrays["delta_auprc"][0],
        seed_indices=draws.seed_indices,
        domain_indices=draws.pair_domain_indices,
    )
    auc_bootstrap = two_way_paired_bootstrap(
        arrays["delta_auroc"][0],
        seed_indices=draws.seed_indices,
        domain_indices=draws.pair_domain_indices,
    )
    b3_bootstrap = two_way_paired_bootstrap(
        arrays["delta_b3"][0],
        seed_indices=draws.seed_indices,
        domain_indices=draws.b3_domain_indices,
    )
    bootstrap = {
        "delta_auprc": _bootstrap_summary(ap_bootstrap),
        "delta_auroc": _bootstrap_summary(auc_bootstrap),
        "delta_b3": _bootstrap_summary(b3_bootstrap),
    }

    domain_ap = arrays["delta_auprc"][0].mean(axis=0)
    domain_auc = arrays["delta_auroc"][0].mean(axis=0)
    domain_b3 = arrays["delta_b3"][0].mean(axis=0)
    absolute_domain_b3 = arrays["absolute_b3"][0].mean(axis=0)
    seed_macro_ap = arrays["delta_auprc"][0].mean(axis=1)
    positive_seed_count = int((seed_macro_ap > 0).sum())
    positive_domain_count = int((domain_ap > 0).sum())

    gates = {
        "q05_delta_b3": _gate(
            bootstrap["delta_b3"]["q05"],
            MIN_Q05_DELTA_B3,
            ">=",
            _at_least(bootstrap["delta_b3"]["q05"], MIN_Q05_DELTA_B3),
        ),
        "worst_domain_delta_b3": _gate(
            float(domain_b3.min()),
            MIN_WORST_DOMAIN_DELTA_B3,
            ">=",
            _at_least(float(domain_b3.min()), MIN_WORST_DOMAIN_DELTA_B3),
        ),
        "worst_domain_delta_auprc": _gate(
            float(domain_ap.min()),
            MIN_WORST_DOMAIN_DELTA_AUPRC,
            ">=",
            _at_least(float(domain_ap.min()), MIN_WORST_DOMAIN_DELTA_AUPRC),
        ),
        "worst_domain_delta_auroc": _gate(
            float(domain_auc.min()),
            MIN_WORST_DOMAIN_DELTA_AUROC,
            ">=",
            _at_least(float(domain_auc.min()), MIN_WORST_DOMAIN_DELTA_AUROC),
        ),
        "positive_seed_fraction": _gate(
            float(positive_seed_count / n_seeds),
            MIN_POSITIVE_SEED_FRACTION,
            ">=",
            positive_seed_count * 3 >= n_seeds * 2,
        ),
        "positive_auprc_domains": _gate(
            positive_domain_count,
            MIN_POSITIVE_AUPRC_DOMAINS,
            ">=",
            positive_domain_count >= MIN_POSITIVE_AUPRC_DOMAINS,
        ),
    }
    safety_eligible = all(record["passed"] for record in gates.values())
    replacement_evidence = _gate(
        bootstrap["delta_auprc"]["q05"],
        0.0,
        ">",
        bootstrap["delta_auprc"]["q05"] > 0,
    )

    proxy_domain_ap = arrays["proxy_delta_auprc"][0].mean(axis=0)
    proxy_domain_auc = arrays["proxy_delta_auroc"][0].mean(axis=0)
    proxy_flags = []
    for index, domain in enumerate(PROXY_DOMAINS):
        for metric, value in (
            ("auprc", float(proxy_domain_ap[index])),
            ("auroc", float(proxy_domain_auc[index])),
        ):
            if _strictly_less(value, PROXY_MANUAL_REVIEW_LOSS):
                proxy_flags.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "threshold": PROXY_MANUAL_REVIEW_LOSS,
                        "value": value,
                    }
                )

    expected_recipe_id = recipe_id_for(recipe)
    if recipe_id != expected_recipe_id:
        raise ValueError(f"recipe_id does not match recipe metadata: {recipe_id!r} != {expected_recipe_id!r}")
    if recipe["arm"] != arm:
        raise ValueError(f"recipe arm does not match result arm: {recipe['arm']!r} != {arm!r}")
    return {
        "arm": arm,
        "bootstrap": bootstrap,
        "complexity_rank": int(recipe["complexity_rank"]),
        "concentration": concentration_diagnostics(domain_ap),
        "domain_mean_absolute_b3": {
            domain: float(absolute_domain_b3[index]) for index, domain in enumerate(B3_DOMAINS)
        },
        "domain_mean_delta_auprc": {domain: float(domain_ap[index]) for index, domain in enumerate(PAIR_DOMAINS)},
        "domain_mean_delta_auroc": {domain: float(domain_auc[index]) for index, domain in enumerate(PAIR_DOMAINS)},
        "domain_mean_delta_b3": {domain: float(domain_b3[index]) for index, domain in enumerate(B3_DOMAINS)},
        "fixed_budget": bool(recipe["fixed_budget"]),
        "gates": gates,
        "manual_review_required": bool(proxy_flags),
        "mean_delta_auprc": float(arrays["delta_auprc"][0].mean()),
        "mean_delta_auroc": float(arrays["delta_auroc"][0].mean()),
        "mean_delta_b3": float(arrays["delta_b3"][0].mean()),
        "mean_training_rows": float(row_values.mean()),
        "positive_auprc_domain_count": positive_domain_count,
        "positive_seed_count": positive_seed_count,
        "proxy_flags": proxy_flags,
        "proxy_mean_delta_auprc": {domain: float(proxy_domain_ap[index]) for index, domain in enumerate(PROXY_DOMAINS)},
        "proxy_mean_delta_auroc": {
            domain: float(proxy_domain_auc[index]) for index, domain in enumerate(PROXY_DOMAINS)
        },
        "recipe": recipe,
        "recipe_id": recipe_id,
        "replacement_evidence": replacement_evidence,
        "replacement_eligible": bool(safety_eligible and replacement_evidence["passed"]),
        "safety_eligible": bool(safety_eligible),
        "seed_macro_delta_auprc": {str(seed): float(seed_macro_ap[index]) for index, seed in enumerate(seeds)},
        "training_seeds": list(seeds),
        "worst_absolute_domain_mean_b3": float(absolute_domain_b3.min()),
        "worst_domain_delta_auprc": float(domain_ap.min()),
        "worst_domain_delta_auroc": float(domain_auc.min()),
        "worst_domain_delta_b3": float(domain_b3.min()),
    }


def _normal_rank_key(score: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -score["bootstrap"]["delta_auprc"]["q05"],
        -score["mean_delta_auprc"],
        -score["mean_delta_auroc"],
        -score["mean_delta_b3"],
        -score["worst_absolute_domain_mean_b3"],
        score["mean_training_rows"],
        score["complexity_rank"],
        score["recipe_id"],
    )


def select_recommendation(
    scores: Sequence[dict[str, Any]],
    *,
    baseline_recipe_id: str,
) -> dict[str, Any]:
    """Apply frozen ranking, practical-tie, baseline, and proxy-review policy."""

    eligible = sorted((score for score in scores if score["replacement_eligible"]), key=_normal_rank_key)
    if not eligible:
        return {
            "decision": "retain_baseline_no_eligible_replacement",
            "practical_tie_recipe_ids": [],
            "provisional_arm": None,
            "provisional_recipe_id": None,
            "recommended_arm": BASELINE_ARM,
            "recommended_recipe_id": baseline_recipe_id,
        }

    leader_q05 = eligible[0]["bootstrap"]["delta_auprc"]["q05"]
    practical_ties = [
        score for score in eligible if leader_q05 - score["bootstrap"]["delta_auprc"]["q05"] <= PRACTICAL_AUPRC_TIE
    ]
    selected = min(
        practical_ties,
        key=lambda score: (
            not score["fixed_budget"],
            score["mean_training_rows"],
            score["complexity_rank"],
            _normal_rank_key(score),
        ),
    )
    base = {
        "practical_tie_recipe_ids": [score["recipe_id"] for score in practical_ties],
        "provisional_arm": selected["arm"],
        "provisional_recipe_id": selected["recipe_id"],
    }
    if selected["manual_review_required"]:
        return {
            **base,
            "decision": "manual_review_required",
            "recommended_arm": BASELINE_ARM,
            "recommended_recipe_id": baseline_recipe_id,
        }
    return {
        **base,
        "decision": "replace_baseline",
        "recommended_arm": selected["arm"],
        "recommended_recipe_id": selected["recipe_id"],
    }


def _policy_payload() -> dict[str, Any]:
    return {
        "b3_domains": list(B3_DOMAINS),
        "baseline_arm": BASELINE_ARM,
        "bootstrap_quantile_method": BOOTSTRAP_QUANTILE_METHOD,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_rng": "numpy.random.PCG64",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inclusive_boundary_absolute_tolerance": BOUNDARY_ABS_TOLERANCE,
        "gates": {
            "minimum_positive_auprc_domains": MIN_POSITIVE_AUPRC_DOMAINS,
            "minimum_positive_seed_fraction": MIN_POSITIVE_SEED_FRACTION,
            "minimum_q05_delta_b3": MIN_Q05_DELTA_B3,
            "minimum_worst_domain_delta_auprc": MIN_WORST_DOMAIN_DELTA_AUPRC,
            "minimum_worst_domain_delta_auroc": MIN_WORST_DOMAIN_DELTA_AUROC,
            "minimum_worst_domain_delta_b3": MIN_WORST_DOMAIN_DELTA_B3,
            "replacement_q05_delta_auprc_strictly_positive": True,
        },
        "minimum_training_seeds": MIN_TRAINING_SEEDS,
        "pair_domains": list(PAIR_DOMAINS),
        "practical_auprc_tie": PRACTICAL_AUPRC_TIE,
        "proxy_domains": list(PROXY_DOMAINS),
        "proxy_manual_review_loss": PROXY_MANUAL_REVIEW_LOSS,
        "proxy_metrics_receive_rank_credit": False,
        "schema_version": RANKING_POLICY_VERSION,
    }


def _validate_grid(
    folds: Sequence[LoadedFold],
) -> tuple[
    dict[str, dict[int, dict[str, LoadedFold]]],
    str,
    tuple[int, ...],
]:
    comparison_identities = {fold.comparison_identity_sha256 for fold in folds}
    if len(comparison_identities) != 1:
        raise ValueError(
            "All final-ranking folds must share one comparison identity, " f"observed={sorted(comparison_identities)}"
        )
    by_recipe: dict[str, dict[int, dict[str, LoadedFold]]] = {}
    recipes: dict[str, dict[str, Any]] = {}
    arms: dict[str, str] = {}
    cell_paths: dict[tuple[str, int, str], Path] = {}
    evaluation_seeds: set[int] = set()
    for fold in folds:
        row = fold.payload
        recipe_id = str(row["recipe_id"])
        seed = int(row["training_seed"])
        domain = str(row["held_out_domain"])
        if domain not in ALL_DOMAINS:
            raise ValueError(f"Unsupported held-out domain in ranking input: {domain!r}")
        key = (recipe_id, seed, domain)
        if key in cell_paths:
            raise ValueError(f"Duplicate recipe/seed/domain cell {key}: {cell_paths[key]} and {fold.path}")
        cell_paths[key] = fold.path
        by_recipe.setdefault(recipe_id, {}).setdefault(seed, {})[domain] = fold
        previous_recipe = recipes.setdefault(recipe_id, row["recipe"])
        if previous_recipe != row["recipe"]:
            raise ValueError(f"recipe_id={recipe_id} maps to inconsistent recipe metadata")
        if row["recipe"]["arm"] != row["arm"]:
            raise ValueError(f"recipe_id={recipe_id} has recipe.arm inconsistent with fold arm")
        previous_arm = arms.setdefault(recipe_id, str(row["arm"]))
        if previous_arm != row["arm"]:
            raise ValueError(f"recipe_id={recipe_id} maps to inconsistent arm names")
        evaluation_seeds.add(int(row["evaluation_seed"]))

    if len(evaluation_seeds) != 1:
        raise ValueError(f"All final-ranking folds must use one evaluation seed, observed={sorted(evaluation_seeds)}")
    baseline_ids = sorted(recipe_id for recipe_id, arm in arms.items() if arm == BASELINE_ARM)
    if len(baseline_ids) != 1:
        raise ValueError(f"Final ranking requires exactly one {BASELINE_ARM!r} recipe, observed={baseline_ids}")
    baseline_id = baseline_ids[0]
    seeds = tuple(sorted(by_recipe[baseline_id]))
    if len(seeds) < MIN_TRAINING_SEEDS:
        raise ValueError(f"Final ranking requires at least {MIN_TRAINING_SEEDS} common training seeds")
    expected_cells = {(seed, domain) for seed in seeds for domain in ALL_DOMAINS}
    for recipe_id, by_seed in by_recipe.items():
        observed_cells = {(seed, domain) for seed, by_domain in by_seed.items() for domain in by_domain}
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        if missing or extra:
            raise ValueError(
                f"Incomplete recipe/seed/15-domain grid for recipe_id={recipe_id}: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        for seed in seeds:
            run_ids = {by_seed[seed][domain].payload["run_id"] for domain in ALL_DOMAINS}
            if len(run_ids) != 1:
                raise ValueError(f"recipe_id={recipe_id} seed={seed} spans multiple run IDs")

    baseline = by_recipe[baseline_id]
    for domain in ALL_DOMAINS:
        baseline_identities = {
            (
                baseline[seed][domain].payload["evaluation_seed"],
                baseline[seed][domain].payload["evaluation_pair_digest"],
                baseline[seed][domain].payload["b3_evaluation_digest"],
            )
            for seed in seeds
        }
        if len(baseline_identities) != 1:
            raise ValueError(f"Baseline evaluation identity changes across seeds for domain={domain!r}")
        for recipe_id, by_seed in by_recipe.items():
            for seed in seeds:
                candidate = by_seed[seed][domain].payload
                reference = baseline[seed][domain].payload
                candidate_identity = (
                    candidate["evaluation_seed"],
                    candidate["evaluation_pair_digest"],
                    candidate["b3_evaluation_digest"],
                )
                reference_identity = (
                    reference["evaluation_seed"],
                    reference["evaluation_pair_digest"],
                    reference["b3_evaluation_digest"],
                )
                if candidate_identity != reference_identity:
                    raise ValueError(f"Evaluation identity mismatch recipe_id={recipe_id} seed={seed} domain={domain}")
    return by_recipe, baseline_id, seeds


def rank_folds(folds: Sequence[LoadedFold], *, source_manifest: Path | None = None) -> RankingArtifacts:
    """Validate a complete study and apply the frozen final-ranking policy."""

    by_recipe, baseline_id, seeds = _validate_grid(folds)
    baseline = by_recipe[baseline_id]
    draws = make_bootstrap_draws(len(seeds))
    scores: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    for recipe_id in sorted(recipe for recipe in by_recipe if recipe != baseline_id):
        candidate = by_recipe[recipe_id]
        first = candidate[seeds[0]][ALL_DOMAINS[0]].payload
        delta_ap = np.empty((len(seeds), len(PAIR_DOMAINS)), dtype=np.float64)
        delta_auc = np.empty_like(delta_ap)
        delta_b3 = np.empty((len(seeds), len(B3_DOMAINS)), dtype=np.float64)
        absolute_b3 = np.empty_like(delta_b3)
        proxy_ap = np.empty((len(seeds), len(PROXY_DOMAINS)), dtype=np.float64)
        proxy_auc = np.empty_like(proxy_ap)
        training_rows = np.empty((len(seeds), len(ALL_DOMAINS)), dtype=np.float64)

        for seed_index, seed in enumerate(seeds):
            for domain_index, domain in enumerate(ALL_DOMAINS):
                row = candidate[seed][domain].payload
                base = baseline[seed][domain].payload
                ap_delta = float(row["pairwise"]["auprc"] - base["pairwise"]["auprc"])
                auc_delta = float(row["pairwise"]["auroc"] - base["pairwise"]["auroc"])
                b3_delta: float | None = None
                if domain in PAIR_DOMAINS:
                    pair_index = PAIR_DOMAINS.index(domain)
                    delta_ap[seed_index, pair_index] = ap_delta
                    delta_auc[seed_index, pair_index] = auc_delta
                if domain in B3_DOMAINS:
                    b3_index = B3_DOMAINS.index(domain)
                    b3_delta = float(row["b3"]["f1"] - base["b3"]["f1"])
                    delta_b3[seed_index, b3_index] = b3_delta
                    absolute_b3[seed_index, b3_index] = float(row["b3"]["f1"])
                if domain in PROXY_DOMAINS:
                    proxy_index = PROXY_DOMAINS.index(domain)
                    proxy_ap[seed_index, proxy_index] = ap_delta
                    proxy_auc[seed_index, proxy_index] = auc_delta
                training_rows[seed_index, domain_index] = float(row["training_rows"])
                paired_rows.append(
                    {
                        "arm": first["arm"],
                        "baseline_arm": BASELINE_ARM,
                        "delta_auprc": ap_delta,
                        "delta_auroc": auc_delta,
                        "delta_b3": b3_delta,
                        "held_out_domain": domain,
                        "recipe_id": recipe_id,
                        "role": (
                            "cluster_gold" if domain in B3_DOMAINS else "pair_gold" if domain == "medline" else "proxy"
                        ),
                        "training_rows": int(row["training_rows"]),
                        "training_seed": seed,
                    }
                )

        score = score_candidate(
            recipe_id=recipe_id,
            arm=str(first["arm"]),
            recipe=first["recipe"],
            seeds=seeds,
            delta_auprc=delta_ap,
            delta_auroc=delta_auc,
            delta_b3=delta_b3,
            absolute_b3=absolute_b3,
            proxy_delta_auprc=proxy_ap,
            proxy_delta_auroc=proxy_auc,
            training_rows=training_rows,
            draws=draws,
        )
        scores.append(score)
        for metric, summary in score["bootstrap"].items():
            bootstrap_rows.append({"arm": score["arm"], "metric": metric, "recipe_id": recipe_id, **summary})
        for domain in ALL_DOMAINS:
            if domain in PAIR_DOMAINS:
                pair_index = PAIR_DOMAINS.index(domain)
                mean_delta_auprc = float(delta_ap[:, pair_index].mean())
                mean_delta_auroc = float(delta_auc[:, pair_index].mean())
            else:
                proxy_index = PROXY_DOMAINS.index(domain)
                mean_delta_auprc = float(proxy_ap[:, proxy_index].mean())
                mean_delta_auroc = float(proxy_auc[:, proxy_index].mean())
            domain_rows.append(
                {
                    "arm": score["arm"],
                    "held_out_domain": domain,
                    "mean_delta_auprc": mean_delta_auprc,
                    "mean_delta_auroc": mean_delta_auroc,
                    "mean_delta_b3": (score["domain_mean_delta_b3"][domain] if domain in B3_DOMAINS else None),
                    "recipe_id": recipe_id,
                    "role": (
                        "cluster_gold" if domain in B3_DOMAINS else "pair_gold" if domain == "medline" else "proxy"
                    ),
                }
            )

    normal_order = sorted(scores, key=_normal_rank_key)
    for rank, score in enumerate(normal_order, start=1):
        score["rank_before_practical_tie"] = rank
    decision = select_recommendation(scores, baseline_recipe_id=baseline_id)
    baseline_fold = baseline[seeds[0]][ALL_DOMAINS[0]].payload
    baseline_summary = {
        "arm": BASELINE_ARM,
        "mean_training_rows": float(
            np.mean([baseline[seed][domain].payload["training_rows"] for seed in seeds for domain in ALL_DOMAINS])
        ),
        "recipe": baseline_fold["recipe"],
        "recipe_id": baseline_id,
        "training_seeds": list(seeds),
    }
    policy = _policy_payload()
    input_files = [
        {
            "comparison_identity_sha256": fold.comparison_identity_sha256,
            "path": str(fold.path),
            "recipe_id": fold.payload["recipe_id"],
            "run_manifest_path": str(fold.run_manifest_path),
            "run_manifest_sha256": fold.run_manifest_sha256,
            "run_id": fold.payload["run_id"],
            "sha256": fold.sha256,
        }
        for fold in sorted(folds, key=lambda item: str(item.path))
    ]
    run_manifests_by_path: dict[Path, dict[str, Any]] = {}
    for fold in folds:
        run_manifests_by_path.setdefault(
            fold.run_manifest_path,
            {
                "path": str(fold.run_manifest_path),
                "run_id": fold.payload["run_id"],
                "sha256": fold.run_manifest_sha256,
            },
        )
    evaluation_identity = {
        domain: {
            "b3_evaluation_digest": baseline[seeds[0]][domain].payload["b3_evaluation_digest"],
            "evaluation_pair_digest": baseline[seeds[0]][domain].payload["evaluation_pair_digest"],
            "evaluation_seed": baseline[seeds[0]][domain].payload["evaluation_seed"],
        }
        for domain in ALL_DOMAINS
    }
    input_manifest = {
        "comparison_identity": {
            "schema_version": COMPARISON_IDENTITY_SCHEMA_VERSION,
            "sha256": folds[0].comparison_identity_sha256,
        },
        "evaluation_identity": evaluation_identity,
        "fold_count": len(folds),
        "input_files": input_files,
        "ranking_input_manifest": None if source_manifest is None else str(source_manifest.resolve()),
        "ranking_input_manifest_sha256": None if source_manifest is None else _sha256_file(source_manifest.resolve()),
        "recipe_ids": sorted(by_recipe),
        "run_manifests": [run_manifests_by_path[path] for path in sorted(run_manifests_by_path)],
        "run_ids": sorted({fold.payload["run_id"] for fold in folds}),
        "training_seeds": list(seeds),
    }
    ranking = {
        "baseline": baseline_summary,
        "candidates": sorted(scores, key=lambda score: score["rank_before_practical_tie"]),
        "decision": decision,
        "input_manifest_digest": strict_json_digest(input_manifest),
        "policy": policy,
        "schema_version": RANKING_OUTPUT_SCHEMA_VERSION,
    }

    ranking_rows: list[dict[str, Any]] = [
        {
            "arm": BASELINE_ARM,
            "complexity_rank": baseline_fold["recipe"]["complexity_rank"],
            "decision": decision["decision"] if decision["recommended_recipe_id"] == baseline_id else "baseline",
            "fixed_budget": baseline_fold["recipe"]["fixed_budget"],
            "manual_review_required": False,
            "mean_delta_auprc": 0.0,
            "mean_delta_auroc": 0.0,
            "mean_delta_b3": 0.0,
            "mean_training_rows": baseline_summary["mean_training_rows"],
            "q05_delta_auprc": 0.0,
            "q05_delta_b3": 0.0,
            "rank_before_practical_tie": 0,
            "recipe_id": baseline_id,
            "replacement_eligible": False,
            "safety_eligible": True,
            "worst_absolute_domain_mean_b3": None,
            "worst_domain_delta_auprc": 0.0,
            "worst_domain_delta_auroc": 0.0,
            "worst_domain_delta_b3": 0.0,
        }
    ]
    for score in sorted(scores, key=lambda item: item["rank_before_practical_tie"]):
        ranking_rows.append(
            {
                "arm": score["arm"],
                "complexity_rank": score["complexity_rank"],
                "decision": (
                    decision["decision"] if decision["provisional_recipe_id"] == score["recipe_id"] else "not_selected"
                ),
                "fixed_budget": score["fixed_budget"],
                "manual_review_required": score["manual_review_required"],
                "mean_delta_auprc": score["mean_delta_auprc"],
                "mean_delta_auroc": score["mean_delta_auroc"],
                "mean_delta_b3": score["mean_delta_b3"],
                "mean_training_rows": score["mean_training_rows"],
                "q05_delta_auprc": score["bootstrap"]["delta_auprc"]["q05"],
                "q05_delta_b3": score["bootstrap"]["delta_b3"]["q05"],
                "rank_before_practical_tie": score["rank_before_practical_tie"],
                "recipe_id": score["recipe_id"],
                "replacement_eligible": score["replacement_eligible"],
                "safety_eligible": score["safety_eligible"],
                "worst_absolute_domain_mean_b3": score["worst_absolute_domain_mean_b3"],
                "worst_domain_delta_auprc": score["worst_domain_delta_auprc"],
                "worst_domain_delta_auroc": score["worst_domain_delta_auroc"],
                "worst_domain_delta_b3": score["worst_domain_delta_b3"],
            }
        )
    concentration = {
        "schema_version": RANKING_OUTPUT_SCHEMA_VERSION,
        "recipes": {score["recipe_id"]: score["concentration"] for score in scores},
    }
    return RankingArtifacts(
        ranking=ranking,
        ranking_rows=tuple(ranking_rows),
        paired_delta_rows=tuple(paired_rows),
        domain_rows=tuple(domain_rows),
        bootstrap_rows=tuple(bootstrap_rows),
        concentration=concentration,
        input_manifest=input_manifest,
    )


def rank_manifest(path: Path) -> RankingArtifacts:
    """Load a strict input manifest and rank its complete fold grid."""

    return rank_folds(load_ranking_input(path), source_manifest=path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("CSV output contains a non-finite value")
    return value


def _write_csv_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV output: {path}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"CSV rows have inconsistent schemas: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: _csv_scalar(value) for key, value in row.items()} for row in rows)
    temporary.replace(path)


def _format_metric(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _report(artifacts: RankingArtifacts) -> str:
    decision = artifacts.ranking["decision"]
    lines = [
        "# Authoritative pair-source ranking",
        "",
        f"- decision: `{decision['decision']}`",
        f"- recommended arm: `{decision['recommended_arm']}`",
        f"- provisional replacement: `{decision['provisional_arm']}`",
        f"- bootstrap: `{BOOTSTRAP_REPLICATES}` PCG64 replicates with seed `{BOOTSTRAP_SEED}`",
        "",
        "Big-block proxy metrics receive no ranking credit. A proxy loss below -0.05 requires manual review.",
        "",
        "| arm | q05 ΔAUPRC | mean ΔAUPRC | mean ΔAUROC | mean ΔB³ | eligible | proxy review |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in artifacts.ranking_rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["arm"]),
                    _format_metric(row["q05_delta_auprc"]),
                    _format_metric(row["mean_delta_auprc"]),
                    _format_metric(row["mean_delta_auroc"]),
                    _format_metric(row["mean_delta_b3"]),
                    str(row["replacement_eligible"]).lower(),
                    str(row["manual_review_required"]).lower(),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_ranking_outputs(output_dir: Path, artifacts: RankingArtifacts) -> None:
    """Atomically write every authoritative ranking output."""

    root = output_dir.resolve()
    _write_json_atomic(root / "ranking.json", artifacts.ranking)
    _write_csv_atomic(root / "ranking.csv", artifacts.ranking_rows)
    _write_csv_atomic(root / "paired_deltas.csv", artifacts.paired_delta_rows)
    _write_csv_atomic(root / "domain_summary.csv", artifacts.domain_rows)
    _write_csv_atomic(root / "bootstrap_summary.csv", artifacts.bootstrap_rows)
    _write_json_atomic(root / "concentration.json", artifacts.concentration)
    _write_json_atomic(root / "input_manifest.json", artifacts.input_manifest)
    _write_text_atomic(root / "report.md", _report(artifacts))


__all__ = [
    "ALL_DOMAINS",
    "B3_DOMAINS",
    "BASELINE_ARM",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "BootstrapDraws",
    "LoadedFold",
    "PAIR_DOMAINS",
    "PROXY_DOMAINS",
    "RANKING_INPUT_FOLD_ENTRY_KEYS",
    "RANKING_INPUT_SCHEMA_VERSION",
    "RANKING_OUTPUT_SCHEMA_VERSION",
    "RankingArtifacts",
    "concentration_diagnostics",
    "load_ranking_input",
    "make_bootstrap_draws",
    "rank_folds",
    "rank_manifest",
    "score_candidate",
    "select_recommendation",
    "two_way_paired_bootstrap",
    "write_ranking_outputs",
]

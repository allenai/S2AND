"""Strict, versioned fold-result artifacts for pair-source ablations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FOLD_RESULT_SCHEMA_VERSION = "s2and_pair_ablation_fold_result_v1"

_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
_RECIPE_KEYS = {
    "arm",
    "assembly_version",
    "auxiliary_sources",
    "balancing",
    "base_sampler",
    "budget_policy",
    "complexity_rank",
    "fixed_budget",
    "source_caps",
}
_TOP_LEVEL_KEYS = {
    "arm",
    "b3",
    "b3_cache_digests",
    "b3_evaluation_digest",
    "elapsed_seconds",
    "evaluation_pair_digest",
    "evaluation_seed",
    "held_out_domain",
    "model_cache_hit",
    "models",
    "pair_recipe_assembly",
    "pairwise",
    "recipe",
    "recipe_id",
    "run_id",
    "schema_version",
    "source_families",
    "training_negatives",
    "training_pair_digest",
    "training_positives",
    "training_rows",
    "training_seed",
    "training_source_counts",
}
_PAIRWISE_KEYS = {"auprc", "auroc", "negatives", "oracle_kind", "positives", "prevalence", "rows"}
_B3_KEYS = {
    "f1",
    "heldout_blocks",
    "heldout_pairs",
    "heldout_signatures",
    "precision",
    "recall",
    "scope",
    "scoring_backend",
    "threshold",
    "threshold_calibration",
}
_B3_CALIBRATION_KEYS = {"f1", "precision", "recall"}
_SOURCE_COUNT_KEYS = {"negatives", "positives", "rows", "source_domain", "source_family"}


@dataclass(frozen=True, slots=True)
class FoldResultExpectation:
    """Identity and role that a result at one arm/domain path must satisfy."""

    run_id: str
    arm: str
    source_families: tuple[str, ...]
    held_out_domain: str
    training_seed: int
    evaluation_seed: int
    recipe: dict[str, Any]
    training_pair_digest: str
    evaluation_pair_digest: str
    b3_evaluation_digest: str | None
    oracle_kind: str
    b3_scope: str | None
    requires_recipe_audit: bool


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value!r}")


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load one JSON object while rejecting duplicate keys and non-finite values."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid strict JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain one object: {path}")
    return payload


def strict_json_digest(value: Any) -> str:
    """Return a stable digest and reject values outside strict JSON."""

    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def recipe_id_for(recipe: dict[str, Any]) -> str:
    """Validate and digest seed-independent recipe metadata."""

    _require_exact_keys(recipe, _RECIPE_KEYS, "recipe")
    _require_nonempty_text(recipe["arm"], "recipe.arm")
    _require_nonempty_text(recipe["assembly_version"], "recipe.assembly_version")
    _require_nonempty_text(recipe["base_sampler"], "recipe.base_sampler")
    _require_nonempty_text(recipe["budget_policy"], "recipe.budget_policy")
    _require_nonempty_text(recipe["balancing"], "recipe.balancing")
    if not isinstance(recipe["fixed_budget"], bool):
        raise ValueError("recipe.fixed_budget must be Boolean")
    _require_int(recipe["complexity_rank"], "recipe.complexity_rank", minimum=0)
    auxiliaries = recipe["auxiliary_sources"]
    if not isinstance(auxiliaries, list) or any(not isinstance(value, str) or not value for value in auxiliaries):
        raise ValueError("recipe.auxiliary_sources must be a list of non-empty strings")
    if len(auxiliaries) != len(set(auxiliaries)):
        raise ValueError("recipe.auxiliary_sources contains duplicates")
    caps = recipe["source_caps"]
    if not isinstance(caps, dict) or any(not isinstance(key, str) or not key for key in caps):
        raise ValueError("recipe.source_caps must be an object with non-empty string keys")
    for key, value in caps.items():
        _require_int(value, f"recipe.source_caps.{key}", minimum=0)
    return strict_json_digest(recipe)


def _require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"{context} schema mismatch: missing={missing}, extra={extra}")


def _require_nonempty_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _HEX_DIGEST.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return value


def _require_finite(value: Any, context: str, *, lower: float | None = None, upper: float | None = None) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{context} must be finite")
    number = float(value)
    if lower is not None and number < lower:
        raise ValueError(f"{context} must be >= {lower}")
    if upper is not None and number > upper:
        raise ValueError(f"{context} must be <= {upper}")
    return number


def _validate_pairwise(pairwise: Any, expectation: FoldResultExpectation) -> None:
    if not isinstance(pairwise, dict):
        raise ValueError("pairwise must be an object")
    _require_exact_keys(pairwise, _PAIRWISE_KEYS, "pairwise")
    if pairwise["oracle_kind"] != expectation.oracle_kind:
        raise ValueError(
            f"pairwise.oracle_kind mismatch: expected={expectation.oracle_kind!r}, "
            f"observed={pairwise['oracle_kind']!r}"
        )
    rows = _require_int(pairwise["rows"], "pairwise.rows", minimum=1)
    positives = _require_int(pairwise["positives"], "pairwise.positives")
    negatives = _require_int(pairwise["negatives"], "pairwise.negatives")
    if rows != positives + negatives or positives == 0 or negatives == 0:
        raise ValueError("pairwise counts must reconcile and contain both classes")
    prevalence = _require_finite(pairwise["prevalence"], "pairwise.prevalence", lower=0, upper=1)
    if not math.isclose(prevalence, positives / rows, rel_tol=0, abs_tol=1e-15):
        raise ValueError("pairwise.prevalence does not match pairwise counts")
    _require_finite(pairwise["auroc"], "pairwise.auroc", lower=0, upper=1)
    _require_finite(pairwise["auprc"], "pairwise.auprc", lower=0, upper=1)


def _validate_b3(b3: Any, expectation: FoldResultExpectation) -> None:
    required = expectation.b3_evaluation_digest is not None
    if not required:
        if b3 is not None or expectation.b3_scope is not None:
            raise ValueError("B3 must be absent for a non-cluster-gold fold")
        return
    if not isinstance(b3, dict):
        raise ValueError("B3 must be present for a cluster-gold fold")
    if expectation.b3_scope is None:
        raise ValueError("B3 expectation is missing its scope")
    _require_exact_keys(b3, _B3_KEYS, "b3")
    if b3["scope"] != expectation.b3_scope:
        raise ValueError(f"b3.scope mismatch: expected={expectation.b3_scope!r}, observed={b3['scope']!r}")
    _require_finite(b3["threshold"], "b3.threshold", lower=0, upper=1)
    for key in ("precision", "recall", "f1"):
        _require_finite(b3[key], f"b3.{key}", lower=0, upper=1)
    for key in ("heldout_blocks", "heldout_signatures"):
        _require_int(b3[key], f"b3.{key}", minimum=1)
    _require_int(b3["heldout_pairs"], "b3.heldout_pairs")
    _require_nonempty_text(b3["scoring_backend"], "b3.scoring_backend")
    calibration = b3["threshold_calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("b3.threshold_calibration must be an object")
    _require_exact_keys(calibration, _B3_CALIBRATION_KEYS, "b3.threshold_calibration")
    for key in _B3_CALIBRATION_KEYS:
        _require_finite(calibration[key], f"b3.threshold_calibration.{key}", lower=0, upper=1)


def _validate_training_counts(payload: dict[str, Any], expectation: FoldResultExpectation) -> None:
    rows = _require_int(payload["training_rows"], "training_rows", minimum=1)
    positives = _require_int(payload["training_positives"], "training_positives")
    negatives = _require_int(payload["training_negatives"], "training_negatives")
    if rows != positives + negatives or positives == 0 or negatives == 0:
        raise ValueError("training counts must reconcile and contain both classes")
    counts = payload["training_source_counts"]
    if not isinstance(counts, list) or not counts:
        raise ValueError("training_source_counts must be a non-empty list")
    allowed_families = set(expectation.source_families)
    seen: set[tuple[str, str]] = set()
    observed_rows = observed_positives = observed_negatives = 0
    for index, record in enumerate(counts):
        if not isinstance(record, dict):
            raise ValueError(f"training_source_counts[{index}] must be an object")
        _require_exact_keys(record, _SOURCE_COUNT_KEYS, f"training_source_counts[{index}]")
        domain = _require_nonempty_text(record["source_domain"], f"training_source_counts[{index}].source_domain")
        family = _require_nonempty_text(record["source_family"], f"training_source_counts[{index}].source_family")
        if domain == expectation.held_out_domain:
            raise ValueError(f"held-out source domain appears in training counts: {domain}")
        if family not in allowed_families:
            raise ValueError(f"unexpected training source family: {family}")
        key = (domain, family)
        if key in seen:
            raise ValueError(f"duplicate training source-count key: {key}")
        seen.add(key)
        source_rows = _require_int(record["rows"], f"training_source_counts[{index}].rows", minimum=1)
        source_positives = _require_int(record["positives"], f"training_source_counts[{index}].positives")
        source_negatives = _require_int(record["negatives"], f"training_source_counts[{index}].negatives")
        if source_rows != source_positives + source_negatives:
            raise ValueError(f"training_source_counts[{index}] does not reconcile")
        observed_rows += source_rows
        observed_positives += source_positives
        observed_negatives += source_negatives
    if (observed_rows, observed_positives, observed_negatives) != (rows, positives, negatives):
        raise ValueError("training source counts do not reconcile with top-level training totals")


def _validate_recipe_audit(payload: dict[str, Any], expectation: FoldResultExpectation) -> None:
    audit = payload["pair_recipe_assembly"]
    if not expectation.requires_recipe_audit:
        if audit is not None:
            raise ValueError("pair_recipe_assembly must be null for a non-exact-budget recipe")
        return
    if not isinstance(audit, dict):
        raise ValueError("pair_recipe_assembly is required for an exact-budget recipe")
    for key in ("target_rows", "final_rows", "held_out_rows", "base_filler_rows"):
        if key not in audit:
            raise ValueError(f"pair_recipe_assembly is missing {key!r}")
        _require_int(audit[key], f"pair_recipe_assembly.{key}")
    if audit["target_rows"] != payload["training_rows"] or audit["final_rows"] != payload["training_rows"]:
        raise ValueError("pair_recipe_assembly does not match training_rows")
    if audit["held_out_rows"] != 0:
        raise ValueError("pair_recipe_assembly contains held-out rows")
    _require_digest(audit.get("selection_sha256"), "pair_recipe_assembly.selection_sha256")


def validate_fold_result(payload: dict[str, Any], *, expected: FoldResultExpectation) -> dict[str, Any]:
    """Validate one parsed result and return it unchanged."""

    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "fold result")
    if payload["schema_version"] != FOLD_RESULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported fold-result schema version: {payload['schema_version']!r}")
    identity_pairs = {
        "run_id": expected.run_id,
        "arm": expected.arm,
        "held_out_domain": expected.held_out_domain,
        "training_seed": expected.training_seed,
        "evaluation_seed": expected.evaluation_seed,
        "training_pair_digest": expected.training_pair_digest,
        "evaluation_pair_digest": expected.evaluation_pair_digest,
        "b3_evaluation_digest": expected.b3_evaluation_digest,
    }
    for key, expected_value in identity_pairs.items():
        if payload[key] != expected_value:
            raise ValueError(f"{key} mismatch: expected={expected_value!r}, observed={payload[key]!r}")
    _require_digest(payload["run_id"], "run_id")
    _require_digest(payload["training_pair_digest"], "training_pair_digest")
    _require_digest(payload["evaluation_pair_digest"], "evaluation_pair_digest")
    if payload["b3_evaluation_digest"] is not None:
        _require_digest(payload["b3_evaluation_digest"], "b3_evaluation_digest")
    expected_recipe_id = recipe_id_for(expected.recipe)
    if payload["recipe"] != expected.recipe or payload["recipe_id"] != expected_recipe_id:
        raise ValueError("fold result recipe identity does not match expectation")
    _require_digest(payload["recipe_id"], "recipe_id")
    expected_families = sorted(expected.source_families)
    if payload["source_families"] != expected_families:
        raise ValueError(
            f"source_families mismatch: expected={expected_families!r}, observed={payload['source_families']!r}"
        )
    if not isinstance(payload["model_cache_hit"], bool):
        raise ValueError("model_cache_hit must be Boolean")
    _require_finite(payload["elapsed_seconds"], "elapsed_seconds", lower=0)
    _validate_training_counts(payload, expected)
    _validate_recipe_audit(payload, expected)
    _validate_pairwise(payload["pairwise"], expected)
    _validate_b3(payload["b3"], expected)
    cache_digests = payload["b3_cache_digests"]
    if not isinstance(cache_digests, list):
        raise ValueError("b3_cache_digests must be a list")
    for index, digest in enumerate(cache_digests):
        _require_digest(digest, f"b3_cache_digests[{index}]")
    if (expected.b3_evaluation_digest is None) != (not cache_digests):
        raise ValueError("b3_cache_digests must be non-empty exactly for B3 folds")
    models = payload["models"]
    if not isinstance(models, dict) or set(models) != {"main", "nameless"}:
        raise ValueError("models must contain exactly main and nameless metadata")
    for name, metadata in models.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"models.{name} must be an object")
        _require_nonempty_text(metadata.get("model_path"), f"models.{name}.model_path")
        _require_digest(metadata.get("model_sha256"), f"models.{name}.model_sha256")
    return payload


def load_fold_result(path: Path, *, expected: FoldResultExpectation) -> dict[str, Any]:
    """Strictly load and validate one fold result."""

    return validate_fold_result(load_strict_json(path), expected=expected)


def write_fold_result(path: Path, payload: dict[str, Any], *, expected: FoldResultExpectation) -> None:
    """Validate and atomically write one strict fold result."""

    validate_fold_result(payload, expected=expected)
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)

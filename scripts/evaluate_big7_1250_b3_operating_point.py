"""Evaluate fixed-epsilon B-cubed tradeoffs for the big7_1250 recipe.

This is a practice-data post-hoc analysis.  Public S2AND datasets use their
existing held-out B-cubed plans.  Giant linker datasets use repeated ORCID as a
masked label over bounded candidate-component samples; ORCID constraints are
disabled while features and predictions are produced.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.incremental_linking.feature_block_arrow import _read_arrow_ipc_table  # noqa: E402
from s2and.subblocking import normalize_orcid_for_subblocking  # noqa: E402
from scripts._pair_ablation.b3_cache import (  # noqa: E402
    B3RawFeatureStore,
    build_or_load_b3_raw_feature_store,
    score_b3_raw_feature_store,
)
from scripts._pair_ablation.evaluation import (  # noqa: E402
    B3EvaluationPlan,
    b3_for_threshold,
    build_b3_evaluation_plans,
    build_block_linkages,
    load_gold_block_data,
)
from scripts._pair_ablation.legacy_rust import (  # noqa: E402
    build_legacy_rust_featurizer,
    resolve_legacy_arrow_manifest,
)
from scripts._pair_ablation.masked_b3 import (  # noqa: E402
    MaskedB3Selection,
    masked_b3_for_threshold,
    masked_component_ceiling,
    select_masked_orcid_components,
)
from scripts._pair_ablation.modeling import load_pairwise_models  # noqa: E402
from scripts.run_pair_source_ablation import _clusterer_for_models  # noqa: E402

LOGGER = logging.getLogger("big7_1250_b3_operating_point")
SCHEMA = "s2and_big7_1250_b3_operating_point_v1"
BASELINE_ARM = "uniform_100k"
DEFAULT_CANDIDATE_ARM = "uniform_100k_plus_linker_big7_1250"
DEFAULT_SEEDS = (1111, 2222, 3333)
DEFAULT_PUBLIC_DOMAINS = ("aminer", "arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")
DEFAULT_GIANT_DOMAINS = ("a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _thresholds(start: float, stop: float, step: float) -> tuple[float, ...]:
    if not 0 <= start <= stop <= 1:
        raise ValueError("threshold range must satisfy 0 <= start <= stop <= 1")
    if step <= 0:
        raise ValueError("threshold step must be positive")
    count = int(round((stop - start) / step))
    values = tuple(round(start + index * step, 10) for index in range(count + 1))
    if not values or abs(values[-1] - stop) > 1e-9:
        raise ValueError("threshold stop must lie on the requested step grid")
    return values


def _candidate_result_path(
    study_root: Path,
    *,
    stage: int,
    seed: int,
    arm: str,
    domain: str,
) -> Path:
    return study_root / "stages" / f"stage_{stage:02d}" / f"seed_{seed}" / "results" / arm / f"{domain}.json"


def _candidate_run_manifest_path(study_root: Path, *, stage: int, seed: int) -> Path:
    return study_root / "stages" / f"stage_{stage:02d}" / f"seed_{seed}" / "run_manifest.json"


def _result_pair(
    study_root: Path,
    *,
    stage: int,
    seed: int,
    candidate_arm: str,
    domain: str,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path]:
    candidate_path = _candidate_result_path(
        study_root,
        stage=stage,
        seed=seed,
        arm=candidate_arm,
        domain=domain,
    )
    candidate = _read_json(candidate_path)
    try:
        baseline_path = Path(candidate["pair_recipe_assembly"]["frozen_baseline_result_path"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Candidate result lacks a frozen baseline pointer: {candidate_path}") from exc
    baseline = _read_json(baseline_path)
    if baseline.get("held_out_domain") != domain or candidate.get("held_out_domain") != domain:
        raise RuntimeError(f"Held-out domain mismatch for candidate/baseline pair: {candidate_path}")
    return baseline, baseline_path, candidate, candidate_path


def _load_clusterer(result: Mapping[str, Any], *, n_jobs: int, suppress_orcid: bool) -> Any:
    try:
        main_path = Path(result["models"]["main"]["model_path"])
        nameless_path = Path(result["models"]["nameless"]["model_path"])
        expected_main = str(result["models"]["main"]["model_sha256"])
        expected_nameless = str(result["models"]["nameless"]["model_sha256"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Result has malformed model metadata") from exc
    if main_path.parent != nameless_path.parent:
        raise RuntimeError("Main and nameless models must share a model directory")
    if _sha256_file(main_path) != expected_main or _sha256_file(nameless_path) != expected_nameless:
        raise RuntimeError(f"Model content hash mismatch: {main_path.parent}")
    models = load_pairwise_models(main_path.parent, n_jobs=n_jobs)
    clusterer = _clusterer_for_models(models, n_jobs=n_jobs)
    clusterer.suppress_orcid = bool(suppress_orcid)
    return clusterer


def _model_identity(result: Mapping[str, Any]) -> dict[str, str]:
    return {
        "main_sha256": str(result["models"]["main"]["model_sha256"]),
        "nameless_sha256": str(result["models"]["nameless"]["model_sha256"]),
        "training_pair_digest": str(result["training_pair_digest"]),
    }


def _load_masked_selection(
    *,
    bundle_root: Path,
    domain: str,
    pair_budget: int,
    max_block_size: int,
    evaluation_seed: int,
) -> tuple[MaskedB3Selection, Any]:
    artifacts = resolve_legacy_arrow_manifest(bundle_root / "datasets" / domain / "manifest.json")
    table = _read_arrow_ipc_table(pa, artifacts.source_paths["signatures"]).select(["signature_id", "author_orcid"])
    signature_ids = [str(value) for value in table["signature_id"].to_pylist()]
    orcids = [normalize_orcid_for_subblocking(value) for value in table["author_orcid"].to_pylist()]
    orcid_by_signature = {
        signature_id: orcid for signature_id, orcid in zip(signature_ids, orcids, strict=True) if orcid is not None
    }
    members_path = bundle_root / "components" / f"{domain}_members.parquet"
    members_table = pq.read_table(
        members_path,
        columns=["candidate_component_key", "member_index", "signature_id"],
    )
    components: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in members_table.to_pylist():
        components[str(row["candidate_component_key"])].append((int(row["member_index"]), str(row["signature_id"])))
    ordered_components = {
        key: [signature_id for _, signature_id in sorted(values)] for key, values in components.items()
    }
    selection = select_masked_orcid_components(
        dataset=domain,
        components=ordered_components,
        orcid_by_signature=orcid_by_signature,
        pair_budget=pair_budget,
        max_block_size=max_block_size,
        random_seed=evaluation_seed,
    )
    return selection, artifacts


def _public_plan(
    *,
    data_root: Path,
    public_domains: Sequence[str],
    heldout_domain: str,
    evaluation_seed: int,
    threshold_pairs_per_domain: int,
    b3_scope: str,
) -> B3EvaluationPlan:
    gold = {
        domain: load_gold_block_data(
            domain,
            data_root / domain / "signatures.arrow",
            data_root / domain / f"{domain}_clusters.json",
        )
        for domain in public_domains
    }
    plans = build_b3_evaluation_plans(
        gold,
        evaluation_seed=evaluation_seed,
        threshold_pairs_per_domain=threshold_pairs_per_domain,
        b3_scope=b3_scope,
    )
    return plans[heldout_domain].heldout


def _curve_for_public(
    *,
    domain: str,
    plan: B3EvaluationPlan,
    distances: Mapping[str, Any],
    thresholds: Sequence[float],
) -> list[dict[str, float]]:
    blocks = plan.blocks_dict()
    linkages = {domain: build_block_linkages(blocks, distances)}
    gold = {domain: plan.gold_dict()}
    return [
        {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        for threshold in thresholds
        for precision, recall, f1 in [b3_for_threshold(linkages, {domain: blocks}, gold, threshold)]
    ]


def _curve_for_masked(
    *,
    domain: str,
    selection: MaskedB3Selection,
    distances: Mapping[str, Any],
    thresholds: Sequence[float],
) -> list[dict[str, float]]:
    linkages = build_block_linkages(selection.plan.blocks_dict(), distances)
    return [
        {
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        for threshold in thresholds
        for precision, recall, f1 in [
            masked_b3_for_threshold(
                linkages,
                selection.target_gold,
                threshold,
                dataset_prefix=domain,
            )
        ]
    ]


def _curve_value(curve: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, float]:
    for row in curve:
        if abs(float(row["threshold"]) - threshold) < 1e-9:
            return {
                "threshold": float(row["threshold"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
    raise ValueError(f"threshold={threshold} is absent from curve")


def _result_is_reusable(
    path: Path,
    *,
    plan_digest: str,
    model_identity: Mapping[str, str],
    thresholds: Sequence[float],
) -> bool:
    if not path.is_file():
        return False
    payload = _read_json(path)
    expected_thresholds = [float(value) for value in thresholds]
    observed_thresholds = [float(row["threshold"]) for row in payload.get("curve", [])]
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("evaluation_plan_digest") != plan_digest
        or payload.get("model_identity") != dict(model_identity)
        or observed_thresholds != expected_thresholds
    ):
        raise RuntimeError(f"Existing operating-point result has incompatible identity: {path}")
    return True


def _build_store(
    *,
    cache_root: Path,
    plan: B3EvaluationPlan,
    rust_featurizer: Any,
    feature_artifact_identity: Mapping[str, Any],
    rust_featurizer_identity: Mapping[str, Any],
    clusterer: Any,
    run_manifest: Mapping[str, Any],
    pair_chunk_size: int,
    validated_stores: dict[str, B3RawFeatureStore],
) -> B3RawFeatureStore:
    return build_or_load_b3_raw_feature_store(
        cache_root,
        plan=plan,
        rust_featurizer=rust_featurizer,
        feature_artifact_identity=feature_artifact_identity,
        rust_featurizer_identity=rust_featurizer_identity,
        clusterer=clusterer,
        rust_version=str(run_manifest["rust_version"]),
        rust_extension_sha256=str(run_manifest["rust_extension_sha256"]),
        cache_builder_identity=str(run_manifest["input_identity"]["b3_cache_builder_identity"]),
        pair_chunk_size=pair_chunk_size,
        validated_stores=validated_stores,
    )


def evaluate_masked_giant(
    *,
    study_root: Path,
    stage: int,
    candidate_arm: str,
    seeds: Sequence[int],
    domains: Sequence[str],
    thresholds: Sequence[float],
    output_dir: Path,
    cache_root: Path,
    pair_budget: int,
    max_block_size: int,
    evaluation_seed: int,
    n_jobs: int,
    total_ram_bytes: int,
    pair_chunk_size: int,
    resume: bool,
) -> None:
    """Build one masked store per giant domain and score every paired model."""

    run_manifest = _read_json(_candidate_run_manifest_path(study_root, stage=stage, seed=int(seeds[0])))
    bundle_root = Path(run_manifest["input_identity"]["roots"]["linker_bundle_root"])
    validated_stores: dict[str, B3RawFeatureStore] = {}
    for domain in domains:
        LOGGER.info("masked selection domain=%s", domain)
        selection, artifacts = _load_masked_selection(
            bundle_root=bundle_root,
            domain=domain,
            pair_budget=pair_budget,
            max_block_size=max_block_size,
            evaluation_seed=evaluation_seed,
        )
        ceiling = masked_component_ceiling(
            selection.plan,
            selection.target_gold,
            dataset_prefix=domain,
        )
        selection_payload = {
            "schema_version": SCHEMA,
            "evaluation_plan_digest": selection.plan.plan_digest,
            "stats": selection.stats,
            "component_ceiling": {
                "precision": ceiling[0],
                "recall": ceiling[1],
                "f1": ceiling[2],
            },
        }
        _write_json_atomic(output_dir / "masked_giant" / "selections" / f"{domain}.json", selection_payload)

        first_baseline, _, _, _ = _result_pair(
            study_root,
            stage=stage,
            seed=int(seeds[0]),
            candidate_arm=candidate_arm,
            domain=domain,
        )
        cache_clusterer = _load_clusterer(first_baseline, n_jobs=n_jobs, suppress_orcid=True)
        selected_signature_ids = [signature_id for block in selection.plan.blocks for signature_id in block.signatures]
        rust_featurizer = build_legacy_rust_featurizer(
            artifacts,
            n_jobs=n_jobs,
            signature_ids=selected_signature_ids,
        )
        store = _build_store(
            cache_root=cache_root,
            plan=selection.plan,
            rust_featurizer=rust_featurizer,
            feature_artifact_identity=run_manifest["input_identity"]["feature_artifacts"][domain],
            rust_featurizer_identity={
                "adapter": "practice_only_legacy_arrow_rust_v1",
                "preprocess": False,
                "name_tuples": "filtered",
                "signature_selection": "evaluation_plan",
                "evaluation_plan_digest": selection.plan.plan_digest,
            },
            clusterer=cache_clusterer,
            run_manifest=run_manifest,
            pair_chunk_size=pair_chunk_size,
            validated_stores=validated_stores,
        )
        del rust_featurizer, cache_clusterer
        gc.collect()

        for seed in seeds:
            baseline, baseline_path, candidate, candidate_path = _result_pair(
                study_root,
                stage=stage,
                seed=int(seed),
                candidate_arm=candidate_arm,
                domain=domain,
            )
            for arm, result, source_path in (
                (BASELINE_ARM, baseline, baseline_path),
                (candidate_arm, candidate, candidate_path),
            ):
                destination = output_dir / "masked_giant" / f"seed_{seed}" / arm / f"{domain}.json"
                model_identity = _model_identity(result)
                if resume and _result_is_reusable(
                    destination,
                    plan_digest=selection.plan.plan_digest,
                    model_identity=model_identity,
                    thresholds=thresholds,
                ):
                    LOGGER.info("masked result already complete seed=%s arm=%s domain=%s", seed, arm, domain)
                    continue
                LOGGER.info("masked score seed=%s arm=%s domain=%s", seed, arm, domain)
                clusterer = _load_clusterer(result, n_jobs=n_jobs, suppress_orcid=True)
                distances = score_b3_raw_feature_store(
                    store,
                    clusterer=clusterer,
                    total_ram_bytes=total_ram_bytes,
                )
                curve = _curve_for_masked(
                    domain=domain,
                    selection=selection,
                    distances=distances,
                    thresholds=thresholds,
                )
                _write_json_atomic(
                    destination,
                    {
                        "schema_version": SCHEMA,
                        "space": "masked_giant",
                        "arm": arm,
                        "training_seed": int(seed),
                        "held_out_domain": domain,
                        "source_result_path": str(source_path),
                        "source_result_sha256": _sha256_file(source_path),
                        "model_identity": model_identity,
                        "evaluation_plan_digest": selection.plan.plan_digest,
                        "cache_digest": store.cache_digest,
                        "evaluated_signature_count": len(selection.target_gold),
                        "curve": curve,
                    },
                )
                del clusterer, distances
                gc.collect()


def evaluate_public_s2and(
    *,
    study_root: Path,
    stage: int,
    candidate_arm: str,
    seeds: Sequence[int],
    domains: Sequence[str],
    thresholds: Sequence[float],
    output_dir: Path,
    cache_root: Path,
    n_jobs: int,
    total_ram_bytes: int,
    pair_chunk_size: int,
    resume: bool,
) -> None:
    """Score fixed-epsilon curves on each existing public held-out plan."""

    run_manifest = _read_json(_candidate_run_manifest_path(study_root, stage=stage, seed=int(seeds[0])))
    config = run_manifest["config"]
    data_root = Path(run_manifest["input_identity"]["roots"]["data_root"])
    all_public_domains = tuple(str(value) for value in config["public_domains"])
    public_gold = {
        domain: load_gold_block_data(
            domain,
            data_root / domain / "signatures.arrow",
            data_root / domain / f"{domain}_clusters.json",
        )
        for domain in all_public_domains
    }
    plans = build_b3_evaluation_plans(
        public_gold,
        evaluation_seed=int(config["evaluation_seed"]),
        threshold_pairs_per_domain=int(config["threshold_pairs_per_domain"]),
        b3_scope=str(config["b3_scope"]),
    )
    validated_stores: dict[str, B3RawFeatureStore] = {}
    for domain in domains:
        plan = plans[domain].heldout
        artifacts = resolve_legacy_arrow_manifest(data_root / domain / "manifest.json")
        first_baseline, _, _, _ = _result_pair(
            study_root,
            stage=stage,
            seed=int(seeds[0]),
            candidate_arm=candidate_arm,
            domain=domain,
        )
        cache_clusterer = _load_clusterer(first_baseline, n_jobs=n_jobs, suppress_orcid=False)
        selected_signature_ids = [signature_id for block in plan.blocks for signature_id in block.signatures]
        rust_featurizer = build_legacy_rust_featurizer(
            artifacts,
            n_jobs=n_jobs,
            signature_ids=selected_signature_ids,
        )
        heldout_pair_count = sum(len(block.signatures) * (len(block.signatures) - 1) // 2 for block in plan.blocks)
        LOGGER.info("public cache load domain=%s pairs=%d", domain, heldout_pair_count)
        store = _build_store(
            cache_root=cache_root,
            plan=plan,
            rust_featurizer=rust_featurizer,
            feature_artifact_identity=run_manifest["input_identity"]["feature_artifacts"][domain],
            rust_featurizer_identity={
                "adapter": "practice_only_legacy_arrow_rust_v1",
                "preprocess": False,
                "name_tuples": "filtered",
            },
            clusterer=cache_clusterer,
            run_manifest=run_manifest,
            pair_chunk_size=pair_chunk_size,
            validated_stores=validated_stores,
        )
        del rust_featurizer, cache_clusterer
        gc.collect()

        for seed in seeds:
            baseline, baseline_path, candidate, candidate_path = _result_pair(
                study_root,
                stage=stage,
                seed=int(seed),
                candidate_arm=candidate_arm,
                domain=domain,
            )
            for arm, result, source_path in (
                (BASELINE_ARM, baseline, baseline_path),
                (candidate_arm, candidate, candidate_path),
            ):
                destination = output_dir / "s2and" / f"seed_{seed}" / arm / f"{domain}.json"
                model_identity = _model_identity(result)
                if resume and _result_is_reusable(
                    destination,
                    plan_digest=plan.plan_digest,
                    model_identity=model_identity,
                    thresholds=thresholds,
                ):
                    LOGGER.info("public result already complete seed=%s arm=%s domain=%s", seed, arm, domain)
                    continue
                LOGGER.info("public score seed=%s arm=%s domain=%s", seed, arm, domain)
                clusterer = _load_clusterer(result, n_jobs=n_jobs, suppress_orcid=False)
                distances = score_b3_raw_feature_store(
                    store,
                    clusterer=clusterer,
                    total_ram_bytes=total_ram_bytes,
                )
                curve = _curve_for_public(
                    domain=domain,
                    plan=plan,
                    distances=distances,
                    thresholds=thresholds,
                )
                stored_threshold = float(result["b3"]["threshold"])
                stored_curve_value = _curve_value(curve, stored_threshold)
                if abs(stored_curve_value["f1"] - float(result["b3"]["f1"])) > 1e-9:
                    raise RuntimeError(
                        "Public B3 curve does not reproduce the audited fold result: "
                        f"seed={seed} arm={arm} domain={domain} "
                        f"curve={stored_curve_value['f1']} audited={result['b3']['f1']}"
                    )
                _write_json_atomic(
                    destination,
                    {
                        "schema_version": SCHEMA,
                        "space": "s2and",
                        "arm": arm,
                        "training_seed": int(seed),
                        "held_out_domain": domain,
                        "source_result_path": str(source_path),
                        "source_result_sha256": _sha256_file(source_path),
                        "model_identity": model_identity,
                        "evaluation_plan_digest": plan.plan_digest,
                        "cache_digest": store.cache_digest,
                        "evaluated_signature_count": sum(len(block.signatures) for block in plan.blocks),
                        "audited_threshold": stored_threshold,
                        "audited_metrics": {
                            "precision": float(result["b3"]["precision"]),
                            "recall": float(result["b3"]["recall"]),
                            "f1": float(result["b3"]["f1"]),
                        },
                        "curve": curve,
                    },
                )
                del clusterer, distances
                gc.collect()


def _load_records(
    *,
    output_dir: Path,
    space: str,
    seeds: Sequence[int],
    arms: Sequence[str],
    domains: Sequence[str],
) -> list[dict[str, Any]]:
    records = []
    for seed in seeds:
        for arm in arms:
            for domain in domains:
                path = output_dir / space / f"seed_{seed}" / arm / f"{domain}.json"
                payload = _read_json(path)
                if payload.get("schema_version") != SCHEMA:
                    raise RuntimeError(f"Result schema mismatch: {path}")
                records.append(payload)
    return records


def _aggregate_curves(records: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    by_threshold: dict[float, list[tuple[float, float, float, int]]] = defaultdict(list)
    for record in records:
        weight = int(record["evaluated_signature_count"])
        for row in record["curve"]:
            by_threshold[float(row["threshold"])].append(
                (float(row["precision"]), float(row["recall"]), float(row["f1"]), weight)
            )
    output = []
    for threshold in sorted(by_threshold):
        values = by_threshold[threshold]
        total_weight = sum(value[3] for value in values)
        weighted_precision = sum(value[0] * value[3] for value in values) / total_weight
        weighted_recall = sum(value[1] * value[3] for value in values) / total_weight
        weighted_f1 = (
            0.0
            if weighted_precision + weighted_recall == 0
            else 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall)
        )
        output.append(
            {
                "threshold": threshold,
                "macro_precision": sum(value[0] for value in values) / len(values),
                "macro_recall": sum(value[1] for value in values) / len(values),
                "macro_f1": sum(value[2] for value in values) / len(values),
                "signature_weighted_precision": weighted_precision,
                "signature_weighted_recall": weighted_recall,
                "signature_weighted_f1": weighted_f1,
            }
        )
    return output


def _best_curve_row(curve: Sequence[Mapping[str, float]], metric: str = "macro_f1") -> dict[str, float]:
    return dict(max(curve, key=lambda row: (float(row[metric]), -float(row["threshold"]))))


def _row_at(curve: Sequence[Mapping[str, float]], threshold: float) -> dict[str, float]:
    for row in curve:
        if abs(float(row["threshold"]) - threshold) < 1e-9:
            return dict(row)
    raise ValueError(f"Aggregate curve lacks threshold={threshold}")


def _domain_lodo_common_threshold(
    *,
    raw_records: Mapping[str, Sequence[Mapping[str, Any]]],
    arms: Sequence[str],
    candidate_arm: str,
    seeds: Sequence[int],
    domains_by_space: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Calibrate one cross-space epsilon while leaving out the scored domain."""

    record_index = {
        (
            space,
            str(record["held_out_domain"]),
            int(record["training_seed"]),
            str(record["arm"]),
        ): record
        for space, records in raw_records.items()
        for record in records
    }
    folds = []
    for heldout_space, heldout_domains in domains_by_space.items():
        for heldout_domain in heldout_domains:
            arm_results = {}
            for arm in arms:
                calibration_curves = {}
                for calibration_space, records in raw_records.items():
                    selected = [
                        record
                        for record in records
                        if record["arm"] == arm
                        and not (calibration_space == heldout_space and record["held_out_domain"] == heldout_domain)
                    ]
                    if not selected:
                        raise ValueError(
                            "Domain-LODO common-threshold calibration has no records for "
                            f"space={calibration_space!r} heldout={heldout_space}:{heldout_domain}"
                        )
                    calibration_curves[calibration_space] = _aggregate_curves(selected)
                optima = {
                    space: max(float(row["macro_f1"]) for row in curve) for space, curve in calibration_curves.items()
                }
                threshold_rows = []
                for reference_row in next(iter(calibration_curves.values())):
                    threshold = float(reference_row["threshold"])
                    fractions = {
                        space: float(_row_at(curve, threshold)["macro_f1"]) / optima[space]
                        for space, curve in calibration_curves.items()
                    }
                    threshold_rows.append(
                        {
                            "threshold": threshold,
                            "fractions_of_space_optimum": fractions,
                            "minimum_fraction_of_space_optimum": min(fractions.values()),
                        }
                    )
                selected_threshold = max(
                    threshold_rows,
                    key=lambda row: (
                        row["minimum_fraction_of_space_optimum"],
                        -row["threshold"],
                    ),
                )
                heldout_values = [
                    _curve_value(
                        record_index[(heldout_space, heldout_domain, int(seed), arm)]["curve"],
                        float(selected_threshold["threshold"]),
                    )
                    for seed in seeds
                ]
                arm_results[arm] = {
                    "threshold": float(selected_threshold["threshold"]),
                    "calibration_minimum_fraction_of_space_optimum": float(
                        selected_threshold["minimum_fraction_of_space_optimum"]
                    ),
                    "mean_precision": sum(row["precision"] for row in heldout_values) / len(heldout_values),
                    "mean_recall": sum(row["recall"] for row in heldout_values) / len(heldout_values),
                    "mean_f1": sum(row["f1"] for row in heldout_values) / len(heldout_values),
                }
            folds.append(
                {
                    "space": heldout_space,
                    "held_out_domain": heldout_domain,
                    "arms": arm_results,
                    "candidate_minus_baseline_f1": (
                        arm_results[candidate_arm]["mean_f1"] - arm_results[BASELINE_ARM]["mean_f1"]
                    ),
                }
            )
    by_space = {}
    for space in domains_by_space:
        selected = [fold for fold in folds if fold["space"] == space]
        by_space[space] = {
            "mean_baseline_f1": sum(fold["arms"][BASELINE_ARM]["mean_f1"] for fold in selected) / len(selected),
            "mean_candidate_f1": sum(fold["arms"][candidate_arm]["mean_f1"] for fold in selected) / len(selected),
            "mean_candidate_minus_baseline_f1": sum(fold["candidate_minus_baseline_f1"] for fold in selected)
            / len(selected),
            "worst_domain_candidate_minus_baseline_f1": min(fold["candidate_minus_baseline_f1"] for fold in selected),
        }
    return {
        "method": (
            "For each held-out domain and arm, maximize the minimum normalized macro-F1 "
            "retained across S2AND and masked-giant calibration curves after excluding "
            "that domain, then score the held-out domain at the selected epsilon."
        ),
        "folds": folds,
        "by_space": by_space,
    }


def summarize(
    *,
    output_dir: Path,
    candidate_arm: str,
    seeds: Sequence[int],
    public_domains: Sequence[str],
    giant_domains: Sequence[str],
) -> dict[str, Any]:
    """Aggregate both spaces and select a transparent maximin-regret epsilon."""

    arms = (BASELINE_ARM, candidate_arm)
    raw_records = {
        "s2and": _load_records(
            output_dir=output_dir,
            space="s2and",
            seeds=seeds,
            arms=arms,
            domains=public_domains,
        ),
        "masked_giant": _load_records(
            output_dir=output_dir,
            space="masked_giant",
            seeds=seeds,
            arms=arms,
            domains=giant_domains,
        ),
    }
    aggregates: dict[str, dict[str, list[dict[str, float]]]] = {}
    for space, records in raw_records.items():
        aggregates[space] = {
            arm: _aggregate_curves([record for record in records if record["arm"] == arm]) for arm in arms
        }

    best = {space: {arm: _best_curve_row(curves[arm]) for arm in arms} for space, curves in aggregates.items()}
    candidate_s2and = aggregates["s2and"][candidate_arm]
    candidate_giant = aggregates["masked_giant"][candidate_arm]
    s2and_best_f1 = float(best["s2and"][candidate_arm]["macro_f1"])
    giant_best_f1 = float(best["masked_giant"][candidate_arm]["macro_f1"])
    compromise_rows = []
    for s2and_row in candidate_s2and:
        threshold = float(s2and_row["threshold"])
        giant_row = _row_at(candidate_giant, threshold)
        s2and_ratio = float(s2and_row["macro_f1"]) / s2and_best_f1
        giant_ratio = float(giant_row["macro_f1"]) / giant_best_f1
        compromise_rows.append(
            {
                "threshold": threshold,
                "s2and_macro_f1": float(s2and_row["macro_f1"]),
                "masked_giant_macro_f1": float(giant_row["macro_f1"]),
                "s2and_fraction_of_space_optimum": s2and_ratio,
                "masked_giant_fraction_of_space_optimum": giant_ratio,
                "min_fraction_of_space_optimum": min(s2and_ratio, giant_ratio),
            }
        )
    compromise = dict(
        max(
            compromise_rows,
            key=lambda row: (
                row["min_fraction_of_space_optimum"],
                -row["threshold"],
            ),
        )
    )
    common_threshold = float(compromise["threshold"])
    common = {space: {arm: _row_at(aggregates[space][arm], common_threshold) for arm in arms} for space in aggregates}

    matched_baseline_rows = []
    public_records_by_key = {
        (int(record["training_seed"]), str(record["held_out_domain"]), str(record["arm"])): record
        for record in raw_records["s2and"]
    }
    for seed in seeds:
        for domain in public_domains:
            baseline = public_records_by_key[(int(seed), domain, BASELINE_ARM)]
            candidate = public_records_by_key[(int(seed), domain, candidate_arm)]
            threshold = float(baseline["audited_threshold"])
            baseline_value = _curve_value(baseline["curve"], threshold)
            candidate_value = _curve_value(candidate["curve"], threshold)
            matched_baseline_rows.append(
                {
                    "training_seed": int(seed),
                    "held_out_domain": domain,
                    "baseline_threshold": threshold,
                    "baseline_f1": baseline_value["f1"],
                    "candidate_f1_at_baseline_threshold": candidate_value["f1"],
                    "candidate_minus_baseline_f1": candidate_value["f1"] - baseline_value["f1"],
                }
            )

    domain_lodo = _domain_lodo_common_threshold(
        raw_records=raw_records,
        arms=arms,
        candidate_arm=candidate_arm,
        seeds=seeds,
        domains_by_space={
            "s2and": public_domains,
            "masked_giant": giant_domains,
        },
    )
    selection_reports = [
        _read_json(output_dir / "masked_giant" / "selections" / f"{domain}.json") for domain in giant_domains
    ]
    summary = {
        "schema_version": SCHEMA,
        "candidate_arm": candidate_arm,
        "training_seeds": [int(seed) for seed in seeds],
        "public_domains": list(public_domains),
        "giant_domains": list(giant_domains),
        "aggregate_curves": aggregates,
        "space_optima": best,
        "maximin_common_threshold": compromise,
        "metrics_at_common_threshold": common,
        "matched_baseline_threshold": {
            "folds": matched_baseline_rows,
            "mean_candidate_minus_baseline_f1": sum(row["candidate_minus_baseline_f1"] for row in matched_baseline_rows)
            / len(matched_baseline_rows),
            "worst_candidate_minus_baseline_f1": min(
                row["candidate_minus_baseline_f1"] for row in matched_baseline_rows
            ),
        },
        "domain_lodo_common_threshold": domain_lodo,
        "masked_selection_coverage": [
            {
                **report["stats"],
                "component_ceiling": report["component_ceiling"],
            }
            for report in selection_reports
        ],
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary)
    return summary


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    candidate_arm = str(summary["candidate_arm"])
    optimum = summary["space_optima"]
    compromise = summary["maximin_common_threshold"]
    common = summary["metrics_at_common_threshold"]
    matched = summary["matched_baseline_threshold"]
    domain_lodo = summary["domain_lodo_common_threshold"]
    lines = [
        "# big7_1250 B-cubed operating-point analysis",
        "",
        "ORCID was hidden from giant-block features and constraints and used only after clustering.",
        "",
        "## Space-specific optima",
        "",
        "| space | arm | eps | macro B3 F1 |",
        "| --- | --- | ---: | ---: |",
    ]
    for space in ("s2and", "masked_giant"):
        for arm in (BASELINE_ARM, candidate_arm):
            row = optimum[space][arm]
            lines.append(f"| {space} | {arm} | {row['threshold']:.2f} | {row['macro_f1']:.6f} |")
    lines.extend(
        [
            "",
            "## One shared epsilon",
            "",
            (
                f"Maximin normalized-regret epsilon: **{compromise['threshold']:.2f}** "
                f"(retains {compromise['s2and_fraction_of_space_optimum']:.4%} of the S2AND optimum "
                f"and {compromise['masked_giant_fraction_of_space_optimum']:.4%} of the masked-giant optimum)."
            ),
            "",
            "| space | arm | macro B3 F1 at common eps |",
            "| --- | --- | ---: |",
        ]
    )
    for space in ("s2and", "masked_giant"):
        for arm in (BASELINE_ARM, candidate_arm):
            lines.append(f"| {space} | {arm} | {common[space][arm]['macro_f1']:.6f} |")
    lines.extend(
        [
            "",
            "## Candidate at each baseline fold's audited epsilon",
            "",
            (
                f"Mean candidate-minus-baseline B3 F1: "
                f"**{matched['mean_candidate_minus_baseline_f1']:+.6f}**; "
                f"worst fold: **{matched['worst_candidate_minus_baseline_f1']:+.6f}**."
            ),
            "",
            "## Domain-LODO cross-space calibration",
            "",
            (
                "Each arm's epsilon is selected on the other domains while balancing both "
                "evaluation spaces, then applied to the held-out domain."
            ),
            "",
            "| space | baseline mean F1 | candidate mean F1 | delta | worst domain delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for space in ("s2and", "masked_giant"):
        row = domain_lodo["by_space"][space]
        lines.append(
            f"| {space} | {row['mean_baseline_f1']:.6f} | {row['mean_candidate_f1']:.6f} | "
            f"{row['mean_candidate_minus_baseline_f1']:+.6f} | "
            f"{row['worst_domain_candidate_minus_baseline_f1']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Masked coverage",
            "",
            "| dataset | repeated ORCID coverage | selected groups | selected targets | pairs | component ceiling F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["masked_selection_coverage"]:
        lines.append(
            f"| {row['dataset']} | {row['component_covered_repeated_orcid_signature_fraction']:.4%} | "
            f"{row['selected_orcid_group_count']} | {row['selected_orcid_signature_count']} | "
            f"{row['selected_pair_count']} | {row['component_ceiling']['f1']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("scratch/additive_linker_dose_study_v1_20260716"),
    )
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--candidate-arm", default=DEFAULT_CANDIDATE_ARM)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--public-domains", nargs="+", default=list(DEFAULT_PUBLIC_DOMAINS))
    parser.add_argument("--giant-domains", nargs="+", default=list(DEFAULT_GIANT_DOMAINS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/big7_1250_b3_operating_point_v1_20260718"),
    )
    parser.add_argument(
        "--public-cache-dir",
        type=Path,
        default=Path("scratch/pair_source_ablation_b3_raw_cache_release_v4_20260712"),
    )
    parser.add_argument(
        "--masked-cache-dir",
        type=Path,
        default=Path("scratch/masked_big_block_b3_raw_cache_v1_20260718"),
    )
    parser.add_argument("--masked-pair-budget-per-domain", type=int, default=250_000)
    parser.add_argument("--masked-max-block-size", type=int, default=256)
    parser.add_argument("--evaluation-seed", type=int, default=1111)
    parser.add_argument("--threshold-start", type=float, default=0.3)
    parser.add_argument("--threshold-stop", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--total-ram-gib", type=int, default=200)
    parser.add_argument("--pair-chunk-size", type=int, default=1_000_000)
    parser.add_argument(
        "--phase",
        choices=("all", "masked", "s2and", "summarize"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    thresholds = _thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)
    if args.n_jobs <= 0 or args.total_ram_gib <= 0 or args.pair_chunk_size <= 0:
        raise ValueError("n_jobs, total_ram_gib, and pair_chunk_size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA,
        "study_root": str(args.study_root.resolve()),
        "stage": int(args.stage),
        "candidate_arm": args.candidate_arm,
        "seeds": list(args.seeds),
        "public_domains": list(args.public_domains),
        "giant_domains": list(args.giant_domains),
        "masked_pair_budget_per_domain": int(args.masked_pair_budget_per_domain),
        "masked_max_block_size": int(args.masked_max_block_size),
        "evaluation_seed": int(args.evaluation_seed),
        "thresholds": list(thresholds),
        "n_jobs": int(args.n_jobs),
        "total_ram_gib": int(args.total_ram_gib),
        "pair_chunk_size": int(args.pair_chunk_size),
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists() and _read_json(config_path) != config:
        raise RuntimeError(f"Output directory belongs to a different configuration: {config_path}")
    _write_json_atomic(config_path, config)

    if args.phase in {"all", "masked"}:
        evaluate_masked_giant(
            study_root=args.study_root,
            stage=args.stage,
            candidate_arm=args.candidate_arm,
            seeds=args.seeds,
            domains=args.giant_domains,
            thresholds=thresholds,
            output_dir=args.output_dir,
            cache_root=args.masked_cache_dir,
            pair_budget=args.masked_pair_budget_per_domain,
            max_block_size=args.masked_max_block_size,
            evaluation_seed=args.evaluation_seed,
            n_jobs=args.n_jobs,
            total_ram_bytes=args.total_ram_gib * 1024**3,
            pair_chunk_size=args.pair_chunk_size,
            resume=args.resume,
        )
    if args.phase in {"all", "s2and"}:
        evaluate_public_s2and(
            study_root=args.study_root,
            stage=args.stage,
            candidate_arm=args.candidate_arm,
            seeds=args.seeds,
            domains=args.public_domains,
            thresholds=thresholds,
            output_dir=args.output_dir,
            cache_root=args.public_cache_dir,
            n_jobs=args.n_jobs,
            total_ram_bytes=args.total_ram_gib * 1024**3,
            pair_chunk_size=args.pair_chunk_size,
            resume=args.resume,
        )
    if args.phase in {"all", "summarize"}:
        result = summarize(
            output_dir=args.output_dir,
            candidate_arm=args.candidate_arm,
            seeds=args.seeds,
            public_domains=args.public_domains,
            giant_domains=args.giant_domains,
        )
        print(json.dumps(result["maximin_common_threshold"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

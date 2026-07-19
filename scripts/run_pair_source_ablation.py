"""Run a fresh leave-one-domain-out pair-source ablation from prepared arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and import model as model_module  # noqa: E402
from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION  # noqa: E402
from s2and.eval import b3_precision_recall_fscore  # noqa: E402
from s2and.featurizer import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_NAMELESS_FEATURE_GROUPS,
    FeaturizationInfo,
)
from scripts._pair_ablation.modeling import (  # noqa: E402
    TrainedPairwiseModels,
    averaged_positive_probability,
    pairwise_metrics,
    train_pairwise_models,
)
from scripts._pair_ablation.prepared import B3Evaluation, PreparedStudy, load_prepared  # noqa: E402
from scripts._pair_ablation.study import (  # noqa: E402
    ALL_DOMAINS,
    BASELINE,
    SOURCE_SETS,
    AdditiveDose,
    SourceSet,
    StudyArm,
    select_pairs,
)

logger = logging.getLogger("pair_source_ablation")
_THRESHOLDS = tuple(float(value) for value in np.linspace(0.3, 0.9, 61))


def _main_featurizer_info() -> FeaturizationInfo:
    return FeaturizationInfo(
        features_to_use=list(DEFAULT_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )


def _nameless_featurizer_info() -> FeaturizationInfo:
    return FeaturizationInfo(
        features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_fresh_output(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError as exc:
        raise FileExistsError(f"pair-ablation output directory already exists: {output_dir}") from exc


def _study_digest(
    prepared_digest: str,
    donor_model_dir: Path,
    *,
    estimator_scale: float,
    n_jobs: int,
) -> str:
    payload = {
        "prepared_digest": prepared_digest,
        "donor_main": _sha256_file(donor_model_dir / "main.lgb"),
        "donor_nameless": _sha256_file(donor_model_dir / "nameless.lgb"),
        "estimator_scale": estimator_scale,
        "n_jobs": n_jobs,
        "featurizer_version": FEATURIZER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "main_feature_groups": list(DEFAULT_FEATURE_GROUPS),
        "nameless_feature_groups": list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        "b3_thresholds": _THRESHOLDS,
    }
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _score_b3_rows(
    models: TrainedPairwiseModels,
    evaluation: B3Evaluation,
    *,
    domain: str,
    n_jobs: int,
    total_ram_bytes: int,
) -> np.ndarray:
    distances, _seconds = model_module._predict_and_combine(
        models.main,
        models.nameless,
        evaluation.main,
        evaluation.staged_labels,
        evaluation.nameless,
        f"pair-ablation:{domain}",
        num_threads=n_jobs,
        total_ram_bytes=total_ram_bytes,
    )
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != evaluation.staged_labels.shape:
        raise RuntimeError(
            f"B3 prediction shape mismatch for {domain}: " f"{values.shape} != {evaluation.staged_labels.shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"B3 predictions for {domain} must be finite distances")
    return values


def _build_linkages(
    evaluation: B3Evaluation,
    distances: np.ndarray,
) -> tuple[np.ndarray | None, ...]:
    trees: list[np.ndarray | None] = []
    for block_index in range(len(evaluation.pair_offsets) - 1):
        pair_start = int(evaluation.pair_offsets[block_index])
        pair_stop = int(evaluation.pair_offsets[block_index + 1])
        signature_start = int(evaluation.signature_offsets[block_index])
        signature_stop = int(evaluation.signature_offsets[block_index + 1])
        count = signature_stop - signature_start
        values = np.asarray(distances[pair_start:pair_stop], dtype=np.float64)
        trees.append(None if count <= 1 else linkage(values, "average", preserve_input=True))
    return tuple(trees)


def _b3_at_threshold(
    evaluations: Mapping[str, B3Evaluation],
    trees_by_domain: Mapping[str, tuple[np.ndarray | None, ...]],
    threshold: float,
) -> tuple[float, float, float]:
    predicted: dict[str, list[tuple[str, str]]] = defaultdict(list)
    truth: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for domain, evaluation in evaluations.items():
        trees = trees_by_domain[domain]
        for block_index, tree in enumerate(trees):
            start = int(evaluation.signature_offsets[block_index])
            stop = int(evaluation.signature_offsets[block_index + 1])
            labels = (
                np.ones(stop - start, dtype=np.int64)
                if tree is None
                else fcluster(tree, t=threshold, criterion="distance")
            )
            for signature, gold, label in zip(
                evaluation.signature_ids[start:stop],
                evaluation.gold_cluster_ids[start:stop],
                labels,
                strict=True,
            ):
                member = (domain, str(signature))
                predicted[f"{domain}:{block_index}:{int(label)}"].append(member)
                truth[f"{domain}:{gold}"].append(member)
    precision, recall, f1, *_ = b3_precision_recall_fscore(dict(truth), dict(predicted))
    return float(precision), float(recall), float(f1)


def b3_metrics(
    models: TrainedPairwiseModels,
    prepared: PreparedStudy,
    held_out_domain: str,
    *,
    n_jobs: int,
    total_ram_bytes: int,
) -> dict[str, float | None]:
    """Calibrate on the other gold domains and evaluate the held-out gold domain."""

    if held_out_domain not in prepared.b3:
        return {
            "b3_threshold": None,
            "b3_precision": None,
            "b3_recall": None,
            "b3_f1": None,
        }
    calibration = {domain: evaluation for domain, evaluation in prepared.b3.items() if domain != held_out_domain}
    if not calibration:
        raise ValueError("B3 evaluation needs at least one calibration domain")

    scored = {
        domain: _score_b3_rows(
            models,
            evaluation,
            domain=domain,
            n_jobs=n_jobs,
            total_ram_bytes=total_ram_bytes,
        )
        for domain, evaluation in prepared.b3.items()
    }
    trees = {domain: _build_linkages(evaluation, scored[domain]) for domain, evaluation in prepared.b3.items()}
    calibration_trees = {domain: trees[domain] for domain in calibration}
    threshold = min(
        _THRESHOLDS,
        key=lambda value: (
            -_b3_at_threshold(calibration, calibration_trees, value)[2],
            value,
        ),
    )
    heldout = {held_out_domain: prepared.b3[held_out_domain]}
    precision, recall, f1 = _b3_at_threshold(
        heldout,
        {held_out_domain: trees[held_out_domain]},
        threshold,
    )
    return {
        "b3_threshold": threshold,
        "b3_precision": precision,
        "b3_recall": recall,
        "b3_f1": f1,
    }


def _fold_result(
    prepared: PreparedStudy,
    arm: StudyArm,
    held_out_domain: str,
    *,
    donor_model_dir: Path,
    model_dir: Path,
    training_seed: int,
    n_jobs: int,
    estimator_scale: float,
    total_ram_bytes: int,
    study_digest: str,
) -> dict[str, Any]:
    selected, audit = select_pairs(
        prepared.catalog,
        arm,
        held_out_domain=held_out_domain,
        seed=training_seed,
    )
    rows = selected["feature_row"].to_numpy(dtype=np.int64, copy=False)
    labels = selected["label"].to_numpy(dtype=np.int8, copy=False)
    models = train_pairwise_models(
        np.asarray(prepared.training_main[rows]),
        np.asarray(prepared.training_nameless[rows]),
        labels,
        main_featurizer_info=_main_featurizer_info(),
        nameless_featurizer_info=_nameless_featurizer_info(),
        donor_model_dir=donor_model_dir,
        output_dir=model_dir,
        n_jobs=n_jobs,
        random_seed=training_seed,
        estimator_scale=estimator_scale,
    )
    evaluation = prepared.evaluation[held_out_domain]
    probability = averaged_positive_probability(
        models,
        evaluation.main,
        evaluation.nameless,
    )
    pair = pairwise_metrics(
        evaluation.labels,
        probability,
        oracle_kind="prepared_pair_labels",
    )
    b3 = b3_metrics(
        models,
        prepared,
        held_out_domain,
        n_jobs=n_jobs,
        total_ram_bytes=total_ram_bytes,
    )
    return {
        "study_digest": study_digest,
        "prepared_digest": prepared.prepared_digest,
        "training_seed": training_seed,
        "arm": arm.name,
        "held_out_domain": held_out_domain,
        "training_pair_digest": audit["training_digest"],
        "baseline_pair_digest": audit["base_digest"],
        "baseline_rows": audit["base_rows"],
        "training_rows": audit["training_rows"],
        "linker_rows": audit["linker_rows"],
        "linker_by_domain": audit["linker_by_domain"],
        "pair_rows": pair["rows"],
        "auroc": pair["auroc"],
        "auprc": pair["auprc"],
        **b3,
        "main_model_sha256": _sha256_file(models.main_path),
        "nameless_model_sha256": _sha256_file(models.nameless_path),
    }


def run_ablation(
    prepared: PreparedStudy,
    *,
    donor_model_dir: Path,
    output_dir: Path,
    domains: Sequence[str],
    arms: Sequence[StudyArm],
    training_seed: int,
    n_jobs: int,
    estimator_scale: float = 1.0,
    total_ram_bytes: int = 32 * 1024**3,
) -> list[dict[str, Any]]:
    """Train and evaluate each requested fold exactly once in a fresh directory."""

    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
    if n_jobs <= 0 or estimator_scale <= 0 or total_ram_bytes <= 0:
        raise ValueError("n_jobs, estimator_scale, and total_ram_bytes must be positive")
    if not domains or len(domains) != len(set(domains)):
        raise ValueError("domains must be non-empty and unique")
    unavailable = sorted(set(domains).difference(prepared.evaluation))
    if unavailable:
        raise ValueError(f"prepared evaluation data is missing domains: {unavailable}")
    if any(domain in prepared.b3 for domain in domains) and len(prepared.b3) < 2:
        raise ValueError("gold-domain folds need at least two prepared B3 domains")
    if not arms or arms[0] != BASELINE:
        raise ValueError("arms must start with the baseline")
    arm_names = [arm.name for arm in arms]
    if len(arm_names) != len(set(arm_names)):
        raise ValueError("arms must be unique")

    study_digest = _study_digest(
        prepared.prepared_digest,
        donor_model_dir,
        estimator_scale=estimator_scale,
        n_jobs=n_jobs,
    )
    _require_fresh_output(output_dir)
    results: list[dict[str, Any]] = []
    baseline_by_domain: dict[str, tuple[str, int]] = {}
    for arm in arms:
        for domain in domains:
            logger.info("Running arm=%s held_out=%s", arm.name, domain)
            result = _fold_result(
                prepared,
                arm,
                domain,
                donor_model_dir=donor_model_dir,
                model_dir=output_dir / "models" / str(training_seed) / arm.name / domain,
                training_seed=training_seed,
                n_jobs=n_jobs,
                estimator_scale=estimator_scale,
                total_ram_bytes=total_ram_bytes,
                study_digest=study_digest,
            )
            base = (
                str(result["baseline_pair_digest"]),
                int(result["baseline_rows"]),
            )
            if arm == BASELINE:
                if result["training_pair_digest"] != result["baseline_pair_digest"]:
                    raise AssertionError("baseline training digest changed")
                baseline_by_domain[domain] = base
            elif baseline_by_domain.get(domain) != base:
                raise AssertionError(f"additive arm changed the base pairs for {domain}")
            result_path = output_dir / "results" / str(training_seed) / arm.name / f"{domain}.json"
            _write_json(result_path, result)
            results.append(result)
    return results


def _arms(doses: Sequence[int], source_sets: Sequence[str]) -> tuple[StudyArm, ...]:
    if not doses:
        if source_sets:
            raise ValueError("--source-set requires at least one --dose")
        return (BASELINE,)
    if len(doses) != len(set(doses)) or len(source_sets) != len(set(source_sets)):
        raise ValueError("doses and source sets must not contain duplicates")
    selected_sources = tuple(source_sets) or tuple(SOURCE_SETS)
    additive = tuple(
        AdditiveDose(source_set=cast(SourceSet, source_set), pairs_per_domain=dose)
        for source_set in selected_sources
        for dose in sorted(doses)
    )
    return (BASELINE, *additive)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--donor-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--domain",
        action="append",
        choices=ALL_DOMAINS,
        required=True,
        help="Explicit fold limit; repeat for each held-out domain.",
    )
    parser.add_argument("--dose", action="append", type=int, default=[])
    parser.add_argument(
        "--source-set",
        action="append",
        choices=tuple(SOURCE_SETS),
        default=[],
    )
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--estimator-scale", type=float, default=1.0)
    parser.add_argument("--total-ram-gib", type=float, default=32.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    if args.total_ram_gib <= 0:
        raise SystemExit("--total-ram-gib must be positive")
    try:
        arms = _arms(args.dose, args.source_set)
        prepared = load_prepared(args.prepared_dir.resolve())
        results = run_ablation(
            prepared,
            donor_model_dir=args.donor_model_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            domains=args.domain,
            arms=arms,
            training_seed=args.seed,
            n_jobs=args.n_jobs,
            estimator_scale=args.estimator_scale,
            total_ram_bytes=int(args.total_ram_gib * 1024**3),
        )
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "folds": len(results),
                "output_dir": str(args.output_dir.resolve()),
                "prepared_digest": prepared.prepared_digest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

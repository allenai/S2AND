"""Apply a manually reviewed EPS to a fresh complete model without another fit."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s2and._sha256 import sha256_file
from s2and.production_bundle import _staged_production_bundle
from s2and.production_model import load_production_model
from scripts.eps_sweep.common import (
    CLUSTERING_SOURCE,
    RELEASE_GIANT_DATASETS,
    SINGLETON_INCLUSIVE_CURVE,
    read_singleton_inclusive_curve,
    read_sweep_curve,
    write_json,
)
from scripts.eps_sweep.orcid_b3 import REFERENCE_POLICY, final_test_query_identities
from scripts.production.model.initial_query_sources import (
    MAP_SHA256_FIELD,
    InitialQuerySources,
    load_initial_query_sources,
)

TestIdentityLoader = Callable[[Path, str, InitialQuerySources], frozenset[str]]


def _require_test_separated_reference(
    sweep_root: Path,
    dataset: str,
    plan: dict[str, Any],
    test_identities: TestIdentityLoader,
    query_sources: InitialQuerySources,
) -> dict[str, Any]:
    """Recheck that the reference used for the curve contains no final-test record."""
    stats = plan.get("reference")
    if not isinstance(stats, dict) or stats.get("reference_policy") != REFERENCE_POLICY:
        raise ValueError(f"The EPS reference does not exclude final-test identities: {dataset}")
    snapshot_root = Path(plan["arrow_root"])
    expected_digest = plan.get(MAP_SHA256_FIELD)
    if query_sources.sha256 != expected_digest or stats.get(MAP_SHA256_FIELD) != expected_digest:
        raise ValueError(f"The EPS source map changes after its sweep: {dataset}")
    reference_path = sweep_root / dataset / "orcid_reference.parquet"
    if sha256_file(reference_path) != stats.get("sha256"):
        raise ValueError(f"The EPS reference changes after its sweep: {dataset}")
    reference = set(pd.read_parquet(reference_path, columns=["signature_id"])["signature_id"].astype(str))
    if overlap := reference & test_identities(snapshot_root, dataset, query_sources):
        raise ValueError(f"The EPS reference contains final-test records: {dataset} {sorted(overlap)[:5]}")
    return stats


def _validated_sweep_plans(
    sweep_root: Path,
    datasets: Sequence[str],
    eps: float,
    source_hash: str,
    release_datasets: Sequence[str],
    test_identities: TestIdentityLoader,
) -> dict[str, dict[str, Any]]:
    """Reject stale, bounded, partial, or test-overlapping snapshot curves."""
    if len(set(datasets)) != len(datasets) or set(datasets) != set(release_datasets):
        raise ValueError(f"Release EPS review requires exactly the frozen datasets: {sorted(release_datasets)}")
    plans: dict[str, dict[str, Any]] = {}
    source_maps: dict[Path, InitialQuerySources] = {}
    for dataset in dict.fromkeys(datasets):
        plan_path = sweep_root / dataset / "plan.json"
        raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan, curve = read_sweep_curve(sweep_root, dataset, raw_plan["score_mode"])
        if plan["score_mode"] != "orcid-b3":
            raise ValueError("Release EPS requires identity B3 curves; safe-action curves are diagnostic")
        if plan["model_manifest_sha256"] != source_hash:
            raise ValueError(f"The EPS sweep uses a different model: {dataset}")
        if plan["limited_run"]:
            raise ValueError(f"A bounded smoke curve cannot select release EPS: {dataset}")
        if plan.get("clustering_source") != CLUSTERING_SOURCE:
            raise ValueError(f"The EPS sweep did not cluster with production block clustering; rerun it: {dataset}")
        if plan["use_orcid_subblocking"] or not plan["suppress_orcid_constraints"]:
            raise ValueError(f"EPS reference labels must remain hidden from prediction: {dataset}")
        if not any(math.isclose(eps, float(value), abs_tol=1e-10, rel_tol=0) for value in plan["eps_values"]):
            raise ValueError(f"The selected EPS is absent from the sweep: {dataset}")
        score_path = sweep_root / dataset / ("eps_b3.tsv" if plan["score_mode"] == "orcid-b3" else "eps_overall.tsv")
        if not score_path.is_file():
            raise FileNotFoundError(score_path)
        if plan["score_mode"] == "orcid-b3":
            metric_columns = ["precision", "recall", "f1"]
            if not curve["reference_coverage"].eq(1).all():
                raise ValueError(f"Incomplete ORCID reference coverage: {dataset}")
        elif plan["score_mode"] == "safe-action-proxy":
            metric_columns = ["positive_recall", "negative_separation"]
        else:
            raise ValueError(f"Unknown EPS score mode: {dataset}")
        if not curve[metric_columns].map(lambda value: math.isfinite(value) and 0 <= value <= 1).all().all():
            raise ValueError(f"Invalid EPS metrics: {dataset}")
        snapshot_root = Path(plan["arrow_root"]).resolve()
        if snapshot_root not in source_maps:
            source_maps[snapshot_root] = load_initial_query_sources(
                snapshot_root, dataset_names=release_datasets, expected_sha256=plan.get(MAP_SHA256_FIELD)
            )
        reference = _require_test_separated_reference(
            sweep_root, dataset, plan, test_identities, source_maps[snapshot_root]
        )
        diagnostic = read_singleton_inclusive_curve(sweep_root, dataset, plan)
        if not diagnostic["reference_coverage"].eq(1).all():
            raise ValueError(f"Incomplete singleton-inclusive ORCID reference coverage: {dataset}")
        plans[dataset] = {
            "plan_sha256": sha256_file(plan_path),
            "scores_sha256": sha256_file(score_path),
            "singleton_inclusive_scores_sha256": sha256_file(sweep_root / dataset / SINGLETON_INCLUSIVE_CURVE),
            "score_mode": plan["score_mode"],
            "reference": reference,
        }
    return plans


def _pooled_curve(sweep_root: Path, datasets: Sequence[str]) -> pd.DataFrame:
    """Return giant-block precision, recall and F1 by EPS, each scored reference weighted equally."""

    weights: dict[str, int] = {}
    curves: dict[str, pd.DataFrame] = {}
    for dataset in datasets:
        _, curve = read_sweep_curve(sweep_root, dataset, "orcid-b3")
        curves[dataset] = curve.assign(eps=curve["eps"].round(10)).set_index("eps")
        # The identity curve scores the repeated references; the file also lists one-record groups.
        reference = pd.read_parquet(sweep_root / dataset / "orcid_reference.parquet", columns=["repeated"])
        weights[dataset] = int(reference["repeated"].sum())
    grids = {dataset: sorted(curve.index) for dataset, curve in curves.items()}
    shared = grids[datasets[0]]
    # A value that only some blocks swept cannot be pooled, and dropping it would weaken the check silently.
    if any(grid != shared for grid in grids.values()):
        raise ValueError(f"The giant-block sweeps cover different EPS values: {grids}")
    total = sum(weights.values())
    pooled = pd.DataFrame(
        {
            metric: [sum(weights[d] * float(curves[d].at[eps, metric]) for d in datasets) / total for eps in shared]
            for metric in ("precision", "recall")
        },
        index=pd.Index(shared, name="eps"),
    )
    pooled["f1"] = 2 * pooled["precision"] * pooled["recall"] / (pooled["precision"] + pooled["recall"])
    return pooled


def _giant_f1_tolerance(sweep_root: Path, datasets: Sequence[str], eps: float, tolerance: float) -> dict[str, Any]:
    """Require pooled giant-block F1 at the reviewed EPS to be within ``tolerance`` of the swept best.

    This is the giant-block half of the plan's ``precision_f1_tolerance``: calibration takes the
    most precise setting within the tolerance on validation, and this check keeps that choice
    from costing more than the same tolerance on giant blocks.

    Raises:
        ValueError: F1 at ``eps`` is more than ``tolerance`` below the best swept F1.
    """

    pooled = _pooled_curve(sweep_root, datasets)
    best_eps = float(pooled["f1"].idxmax())
    record = {
        "tolerance": tolerance,
        "best_eps": best_eps,
        "best_f1": float(pooled["f1"].max()),
        "selected_f1": float(pooled.at[round(eps, 10), "f1"]),
    }
    if record["selected_f1"] < record["best_f1"] - tolerance:
        raise ValueError(
            f"Giant-block F1 {record['selected_f1']:.6f} at EPS {eps} is more than {tolerance:g} below "
            f"the best swept F1 {record['best_f1']:.6f} at EPS {best_eps}"
        )
    return record


def apply_reviewed_eps(
    source: Path,
    output: Path,
    sweep_root: Path | None,
    datasets: Sequence[str],
    eps: float,
    review_note: str,
    report_path: Path,
    *,
    release_datasets: Sequence[str] = RELEASE_GIANT_DATASETS,
    test_identities: TestIdentityLoader = final_test_query_identities,
    close_pair: tuple[float, float] | None = None,
    release_version: str | None = None,
    precision_f1_tolerance: float | None = None,
) -> dict[str, Any]:
    """Write a fresh complete model with the reviewed EPS and preserve all trained artifact bytes.

    ``close_pair`` is the reviewed (close_pair_bonus, close_pair_distance); None keeps the source's
    setting. ``release_version`` names a model that differs from its source only in linkage settings.
    ``precision_f1_tolerance`` is the model plan's value; see ``_giant_f1_tolerance``.

    With a sweep root, check the snapshot giant curves first.
    Without one, the review note must cite the internal evidence, because the curves are not on disk here.
    """
    source, output, report_path = source.resolve(), output.resolve(), report_path.resolve()
    if not math.isfinite(eps) or not 0 <= eps <= 1:
        raise ValueError("EPS must be finite and within [0, 1]")
    if not review_note.strip():
        raise ValueError("A manual review note is required")
    if (sweep_root is None) != (not datasets):
        raise ValueError("Pass the sweep root and the dataset list together, or neither")
    if close_pair is not None:
        bonus, close = close_pair
        if not math.isfinite(bonus) or bonus <= 0 or not 0 < close <= 1:
            raise ValueError("The close-pair bonus must be positive and its distance within (0, 1]")
        if sweep_root is not None:
            # Snapshot curves were cut from trees built with the linkage settings of the source model.
            raise ValueError("Snapshot sweeps review EPS only; set the close-pair bonus before the sweep")
    if precision_f1_tolerance is not None:
        if sweep_root is None:
            raise ValueError("The giant-block F1 tolerance needs a snapshot sweep")
        if not math.isfinite(precision_f1_tolerance) or not 0 <= precision_f1_tolerance <= 1:
            raise ValueError("The F1 tolerance must be within [0, 1]")
    if output.exists() or report_path.exists():
        raise FileExistsError("Use new model and review-report paths")
    if source in output.parents or output in source.parents:
        raise ValueError("The model output must be separate from its source")
    if report_path in {source, output} or source in report_path.parents or output in report_path.parents:
        raise ValueError("The review report must remain outside both model directories")
    source_hash = sha256_file(source / "manifest.json")
    plans = (
        None
        if sweep_root is None
        else _validated_sweep_plans(sweep_root, datasets, eps, source_hash, release_datasets, test_identities)
    )
    giant_f1_tolerance = (
        None
        if precision_f1_tolerance is None or sweep_root is None
        else _giant_f1_tolerance(sweep_root.resolve(), datasets, eps, precision_f1_tolerance)
    )
    load_production_model(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((source / "clusterer.json").read_text(encoding="utf-8"))
    if sweep_root is not None and precision_f1_tolerance is None and "close_pair_bonus" in config["cluster_model"]:
        # The 1.3 review score has no peak on a bonus bundle, so only this check tests its EPS.
        raise ValueError("A sweep review of a close-pair bonus bundle needs --precision-f1-tolerance")
    previous_eps = float(config["cluster_model"]["eps"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with _staged_production_bundle(output) as staging:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        config["cluster_model"]["eps"] = eps
        if close_pair is not None:
            config["cluster_model"]["close_pair_bonus"], config["cluster_model"]["close_pair_distance"] = close_pair
        if release_version is not None:
            manifest["release_version"] = release_version
        write_json(staging / "clusterer.json", config)
        manifest["sha256"]["clusterer.json"] = sha256_file(staging / "clusterer.json")
        write_json(staging / "manifest.json", manifest)
        for relative, expected in manifest["sha256"].items():
            if sha256_file(staging / relative) != expected:
                raise RuntimeError(f"EPS review changes a trained artifact: {relative}")
        load_production_model(staging)
    report = {
        "source_manifest_sha256": source_hash,
        "output_manifest_sha256": sha256_file(output / "manifest.json"),
        "source_eps": previous_eps,
        "selected_eps": eps,
        "selected_cluster_model": config["cluster_model"],
        "release_version": manifest["release_version"],
        "review_note": review_note,
        "sweep_plans": plans,
        "giant_f1_tolerance": giant_f1_tolerance,
        "model_output": str(output),
        "unchanged_artifacts": sorted(set(manifest["sha256"]) - {"clusterer.json"}),
    }
    write_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Apply the explicit human choice after review of complete curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        help="Snapshot giant-block sweep root; omit it with --datasets when internal exports supply the review",
    )
    parser.add_argument("--datasets", nargs="+", default=[])
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument(
        "--close-pair",
        type=float,
        nargs=2,
        metavar=("BONUS", "DISTANCE"),
        help="Reviewed close-pair bonus and distance; only without --sweep-root",
    )
    parser.add_argument("--release-version", help="Version of a model that differs from its source only in linkage")
    parser.add_argument(
        "--precision-f1-tolerance",
        type=float,
        help=(
            "The model plan's precision_f1_tolerance. Reject an EPS whose pooled giant-block F1 is more than "
            "this below the best swept F1."
        ),
    )
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = apply_reviewed_eps(
        args.source_model,
        args.output_model,
        args.sweep_root,
        args.datasets,
        args.eps,
        args.review_note,
        args.output_report,
        close_pair=(args.close_pair[0], args.close_pair[1]) if args.close_pair else None,
        release_version=args.release_version,
        precision_f1_tolerance=args.precision_f1_tolerance,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

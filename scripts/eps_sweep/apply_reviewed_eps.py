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
) -> dict[str, Any]:
    """Write a fresh complete model with the reviewed EPS and preserve all trained artifact bytes.

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
    load_production_model(source)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((source / "clusterer.json").read_text(encoding="utf-8"))
    previous_eps = float(config["cluster_model"]["eps"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with _staged_production_bundle(output) as staging:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        config["cluster_model"]["eps"] = eps
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
        "review_note": review_note,
        "sweep_plans": plans,
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
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

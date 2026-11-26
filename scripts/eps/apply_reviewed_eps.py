"""Apply a reviewed EPS, and optionally reviewed gate cutoffs, to a fresh complete model without another fit."""

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
from s2and.incremental_linking_training.classic import (
    _promoted_stratified_gate_spec,
    _read_classic_stratified_source_rows,
    _read_csv,
    _resolve_path,
    _select_classic_stratified_eval_rows,
    _validate_stratified_split_assignments,
    load_bundle,
)
from s2and.incremental_linking_training.gate import cutoff_gate_config
from s2and.production_bundle import _staged_production_bundle
from s2and.production_bundle_contract import COMPLETE_MANIFEST_FILES
from s2and.production_model import load_production_model
from scripts.common.block_roles import BlockRoles, load_block_roles, normalize_block_key
from scripts.common.identity_ledger import ledger_for_roles
from scripts.common.initial_query_sources import (
    MAP_SHA256_FIELD,
    InitialQuerySources,
    load_initial_query_sources,
)
from scripts.eps.orcid_b3 import (
    REFERENCE_POLICY,
    REFERENCE_POLICY_ALL_IDENTITIES,
    final_test_query_identities,
)
from scripts.eps.sweep_curves import (
    CLUSTERING_SOURCE,
    DISTANCE_POLICY_VERSION,
    RELEASE_ROUTE,
    SINGLETON_INCLUSIVE_CURVE,
    read_singleton_inclusive_curve,
    read_sweep_curve,
    write_json,
)

TestIdentityLoader = Callable[[Path, str, InitialQuerySources], frozenset[str]]


def _eps_cal_blocks(block_roles: BlockRoles) -> frozenset[str]:
    """Read the disjoint EPS calibration pool from verified roles."""
    eps_cal = frozenset(block for block, role in block_roles.block_to_role.items() if role == "eps_cal")
    if not eps_cal:
        raise ValueError("Block role manifest has an empty eps_cal pool")
    return eps_cal


def _require_test_separated_reference(
    sweep_root: Path,
    dataset: str,
    plan: dict[str, Any],
    test_identities: TestIdentityLoader,
    query_sources: InitialQuerySources,
    *,
    verified_eps_cal_blocks: frozenset[str] | None = None,
    block_roles: BlockRoles | None = None,
) -> dict[str, Any]:
    """Recheck that the reference used for the curve contains no final-test record."""
    stats = plan.get("reference")
    if not isinstance(stats, dict):
        raise ValueError(f"The EPS reference does not exclude final-test identities: {dataset}")
    if unmatched := stats.get("external_labels_unmatched", 0):
        raise ValueError(f"Unmatched external ORCID labels for {dataset!r}: {unmatched}")
    policy = stats.get("reference_policy")
    if policy == REFERENCE_POLICY_ALL_IDENTITIES:
        if verified_eps_cal_blocks is None:
            raise ValueError(
                f"Block-disjoint policy {REFERENCE_POLICY_ALL_IDENTITIES!r} "
                f"requires --block-role-assignments: {dataset}"
            )
        if normalize_block_key(dataset) not in verified_eps_cal_blocks:
            raise ValueError(f"Dataset {dataset!r} is not in the verified eps_cal block pool")
    elif policy != REFERENCE_POLICY:
        raise ValueError(f"The EPS reference does not exclude final-test identities: {dataset}")
    snapshot_root = Path(plan["arrow_root"])
    expected_digest = plan.get(MAP_SHA256_FIELD)
    if query_sources.sha256 != expected_digest or stats.get(MAP_SHA256_FIELD) != expected_digest:
        raise ValueError(f"The EPS source map changes after its sweep: {dataset}")
    reference_path = sweep_root / dataset / "orcid_reference.parquet"
    if sha256_file(reference_path) != stats.get("sha256"):
        raise ValueError(f"The EPS reference changes after its sweep: {dataset}")
    if block_roles is not None:
        block_roles.require_assignment(stats, context=f"{dataset} EPS sweep reference")
        orcids = pd.read_parquet(reference_path, columns=["orcid"])["orcid"].unique()
        if any(not block_roles.scores(dataset, orcid) for orcid in orcids):
            raise ValueError(f"The EPS reference contains an ORCID outside its home role: {dataset}")
    if policy == REFERENCE_POLICY:
        reference = set(pd.read_parquet(reference_path, columns=["signature_id"])["signature_id"].astype(str))
        if overlap := reference & test_identities(snapshot_root, dataset, query_sources):
            raise ValueError(f"The EPS reference contains final-test records: {dataset} {sorted(overlap)[:5]}")
    return stats


def _release_datasets_from_snapshot(snapshot_root: Path) -> list[str]:
    """Read the giant datasets assigned to linker calibration in the snapshot."""
    bundle = load_bundle(snapshot_root)
    classic = dict(bundle.models["classic"])
    gate = _promoted_stratified_gate_spec(classic)
    if gate is None:
        raise ValueError("classic.promoted_stratified_gate is required")
    split_spec = classic["stratified_eval_test_split"]
    assignments = _read_csv(_resolve_path(bundle, str(split_spec["assignments_path"])))
    _validate_stratified_split_assignments(assignments)
    is_calibration = assignments["split"].astype(str).isin(gate["calibration_splits"])
    is_giant = assignments["source_key"].ne("s2and_eval")
    selected = assignments.loc[is_calibration & is_giant, ["query_group_id", "source_key", "base_group_id", "dataset"]]
    if selected.empty:
        return []
    rows = _select_classic_stratified_eval_rows(
        bundle,
        classic,
        selected,
        include_calibration_source=True,
        read_source_rows=_read_classic_stratified_source_rows,
    )
    mismatched = rows["dataset"].astype(str).ne(rows["dataset_split"].astype(str))
    if mismatched.any():
        sample = rows.loc[mismatched, ["query_group_id", "source_key", "dataset", "dataset_split"]]
        raise ValueError(
            f"Calibration assignment dataset does not match its eval source: {sample.head(5).to_dict(orient='records')}"
        )
    return sorted(set(rows["dataset"].astype(str)))


def _validated_sweep_plans(
    sweep_root: Path,
    datasets: Sequence[str],
    eps: float,
    source_hash: str,
    expected_batching_threshold: int,
    test_identities: TestIdentityLoader,
    *,
    block_role_assignments: Path | None = None,
    identity_evidence: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Reject stale, bounded, partial, or test-overlapping snapshot curves."""
    verified_eps_cal: frozenset[str] | None = None
    block_roles = None
    if block_role_assignments is not None:
        block_roles = load_block_roles(block_role_assignments)
        verified_eps_cal = _eps_cal_blocks(block_roles)
    ledger = ledger_for_roles(identity_evidence, block_roles)
    folders = {path.name for path in sweep_root.iterdir() if path.is_dir()}
    if ignored := sorted(folders - set(datasets)):
        print(f"Ignoring EPS sweep folders outside --datasets: {ignored}", file=sys.stderr)
    if missing := sorted(set(datasets) - folders):
        raise ValueError(f"Release EPS review datasets have no sweep folders: {missing}")
    plans: dict[str, dict[str, Any]] = {}
    snapshot_root: Path | None = None
    query_sources: InitialQuerySources | None = None
    for dataset in datasets:
        plan_path = sweep_root / dataset / "plan.json"
        plan, curve = read_sweep_curve(sweep_root, dataset)
        if ledger is not None:
            ledger.roles.require_bindings(plan, {dataset}, context=f"{dataset} EPS sweep")
            ledger.roles.require_bindings(plan["reference"], {dataset}, context=f"{dataset} EPS reference")
        if plan["model_manifest_sha256"] != source_hash:
            raise ValueError(f"The EPS sweep uses a different model: {dataset}")
        if plan["limited_run"]:
            raise ValueError(f"A bounded smoke curve cannot select release EPS: {dataset}")
        if plan.get("clustering_source") != CLUSTERING_SOURCE:
            raise ValueError(f"The EPS sweep did not cluster with production block clustering; rerun it: {dataset}")
        if plan.get("route") != RELEASE_ROUTE:
            raise ValueError(
                f"The EPS sweep skipped the production attachment passes; rerun with --route production: {dataset}"
            )
        batching_threshold = plan.get("batching_threshold")
        if batching_threshold != expected_batching_threshold:
            raise ValueError(
                f"The EPS sweep used batching_threshold={batching_threshold!r}, "
                f"expected {expected_batching_threshold}; rerun it: {dataset}"
            )
        if plan.get("distance_policy_version") != DISTANCE_POLICY_VERSION:
            raise ValueError(f"The EPS sweep used another distance policy; rerun it: {dataset}")
        if plan["use_orcid_subblocking"] or not plan["suppress_orcid_constraints"]:
            raise ValueError(f"EPS reference labels must remain hidden from prediction: {dataset}")
        if not any(math.isclose(eps, float(value), abs_tol=1e-10, rel_tol=0) for value in plan["eps_values"]):
            raise ValueError(f"The selected EPS is absent from the sweep: {dataset}")
        score_path = sweep_root / dataset / "eps_b3.tsv"
        if not curve["reference_coverage"].eq(1).all():
            raise ValueError(f"Incomplete ORCID reference coverage: {dataset}")
        plan_snapshot = Path(plan["arrow_root"]).resolve()
        if query_sources is None:
            snapshot_root = plan_snapshot
            if verified_eps_cal is not None:
                requested_norm = {normalize_block_key(d) for d in datasets}
                if requested_norm != verified_eps_cal:
                    raise ValueError(
                        "Block-disjoint EPS review --datasets must match the verified eps_cal pool: "
                        f"expected {sorted(verified_eps_cal)}, found {sorted(requested_norm)}"
                    )
                release_datasets = sorted(datasets)
            else:
                release_datasets = _release_datasets_from_snapshot(snapshot_root)
                if sorted(datasets) != release_datasets:
                    raise ValueError(
                        "Release EPS review --datasets must match the snapshot's calibration assignments outside "
                        f"s2and_eval: expected {release_datasets}, found {sorted(datasets)}"
                    )
            query_sources = load_initial_query_sources(
                snapshot_root, dataset_names=release_datasets, expected_sha256=plan.get(MAP_SHA256_FIELD)
            )
        elif plan_snapshot != snapshot_root:
            raise ValueError(
                "Release EPS sweep plans must use one snapshot: "
                f"{dataset} uses {plan_snapshot}, expected {snapshot_root}"
            )
        reference = _require_test_separated_reference(
            sweep_root,
            dataset,
            plan,
            test_identities,
            query_sources,
            verified_eps_cal_blocks=verified_eps_cal,
            block_roles=block_roles,
        )
        diagnostic = read_singleton_inclusive_curve(sweep_root, dataset, plan)
        if not diagnostic["reference_coverage"].eq(1).all():
            raise ValueError(f"Incomplete singleton-inclusive ORCID reference coverage: {dataset}")
        labels = pd.read_parquet(
            sweep_root / dataset / "orcid_reference.parquet", columns=["signature_id", "orcid", "repeated"]
        )
        if (
            labels.empty
            or labels.isna().any().any()
            or labels["signature_id"].duplicated().any()
            or not pd.api.types.is_bool_dtype(labels["repeated"])
            or not labels["repeated"].any()
        ):
            raise ValueError(f"Expected a nonempty scored EPS reference: {dataset}")
        if ledger is not None:
            expected = ledger.for_block(dataset, dataset_dir=Path(plan["arrow_manifest"]).parent).scored()
            if dict(zip(labels["signature_id"], labels["orcid"], strict=True)) != expected:
                raise ValueError(f"EPS reference labels differ from the identity ledger: {dataset}")
        repeats = labels.groupby("orcid")["signature_id"].transform("size").gt(1)
        if not labels["repeated"].eq(repeats).all():
            raise ValueError(f"Invalid repeated identities in EPS reference: {dataset}")
        for scores, count in ((curve, int(repeats.sum())), (diagnostic, len(labels))):
            columns = ["reference_signatures", "scored_signatures"]
            if not set(columns) <= set(scores.columns) or not scores[columns].eq(count).all().all():
                raise ValueError(f"EPS curve counts do not match the scored reference: {dataset}")
        plans[dataset] = {
            "plan_sha256": sha256_file(plan_path),
            "scores_sha256": sha256_file(score_path),
            "singleton_inclusive_scores_sha256": sha256_file(sweep_root / dataset / SINGLETON_INCLUSIVE_CURVE),
            "score_mode": plan["score_mode"],
            **({"identity_bindings": ledger.bindings} if ledger is not None else {}),
            "reference": reference,
        }
    return plans


def _pooled_curve(sweep_root: Path, datasets: Sequence[str]) -> pd.DataFrame:
    """Return giant-block precision, recall and F1 by EPS, each scored reference weighted equally."""

    weights: dict[str, int] = {}
    curves: dict[str, pd.DataFrame] = {}
    for dataset in dict.fromkeys(datasets):
        _, curve = read_sweep_curve(sweep_root, dataset)
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
            metric: [
                sum(weight * float(curves[dataset].at[eps, metric]) for dataset, weight in weights.items()) / total
                for eps in shared
            ]
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
    expected_batching_threshold: int | None = None,
    test_identities: TestIdentityLoader = final_test_query_identities,
    close_pair: tuple[float, float] | None = None,
    release_version: str | None = None,
    precision_f1_tolerance: float | None = None,
    block_role_assignments: Path | None = None,
    identity_evidence: Path | None = None,
    gate_cutoffs: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Write a fresh complete model with the reviewed EPS and preserve all trained artifact bytes.

    ``close_pair`` is the reviewed (close_pair_bonus, close_pair_distance); None keeps the source's
    setting. ``release_version`` names a model that differs from its source only in linkage settings.
    ``gate_cutoffs`` is the reviewed (t_full, t_init). It replaces only the linker gate config, because
    judged cutoffs are set after the ranker is trained; the ranker booster bytes stay unchanged.
    ``precision_f1_tolerance`` is the model plan's value; see ``_giant_f1_tolerance``.

    With a sweep root, check the expected subblock limit and require the requested datasets
    to match the snapshot's calibration assignments outside ``s2and_eval``.
    Without one, the review note must cite the internal evidence, because the curves are not on disk here.
    """
    source, output, report_path = source.resolve(), output.resolve(), report_path.resolve()
    if not math.isfinite(eps) or not 0 <= eps <= 1:
        raise ValueError("EPS must be finite and within [0, 1]")
    if not review_note.strip():
        raise ValueError("A manual review note is required")
    if (sweep_root is None) != (not datasets):
        raise ValueError("Pass the sweep root and the dataset list together, or neither")
    if len({normalize_block_key(dataset) for dataset in datasets}) != len(datasets):
        raise ValueError("EPS review datasets must be unique")
    if sweep_root is None and (identity_evidence is not None or block_role_assignments is not None):
        raise ValueError("Frozen block roles require a bound EPS sweep")
    if sweep_root is None and expected_batching_threshold is not None:
        raise ValueError("--expected-batching-threshold requires --sweep-root")
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
    plans = None
    if sweep_root is not None:
        if expected_batching_threshold is None or expected_batching_threshold <= 0:
            raise ValueError("A sweep review needs a positive --expected-batching-threshold")
        plans = _validated_sweep_plans(
            sweep_root,
            datasets,
            eps,
            source_hash,
            expected_batching_threshold,
            test_identities,
            block_role_assignments=block_role_assignments,
            identity_evidence=identity_evidence,
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
        if gate_cutoffs is not None:
            gate_path = COMPLETE_MANIFEST_FILES["incremental_linker_metadata"]
            if gate_path not in manifest["sha256"]:
                raise ValueError("Gate cutoffs need a complete bundle with an incremental linker")
            metadata = json.loads((staging / gate_path).read_text(encoding="utf-8"))
            metadata["gate_config"] = cutoff_gate_config(*gate_cutoffs)
            write_json(staging / gate_path, metadata)
            manifest["sha256"][gate_path] = sha256_file(staging / gate_path)
        write_json(staging / "manifest.json", manifest)
        for relative, expected in manifest["sha256"].items():
            if sha256_file(staging / relative) != expected:
                raise RuntimeError(f"EPS review changes a trained artifact: {relative}")
        load_production_model(staging)
    report = {
        **(next(iter(plans.values()))["identity_bindings"] if plans and identity_evidence is not None else {}),
        "source_manifest_sha256": source_hash,
        "output_manifest_sha256": sha256_file(output / "manifest.json"),
        "source_eps": previous_eps,
        "selected_eps": eps,
        "selected_cluster_model": config["cluster_model"],
        "release_version": manifest["release_version"],
        "review_note": review_note,
        "sweep_plans": plans,
        "giant_f1_tolerance": giant_f1_tolerance,
        "gate_cutoffs": list(gate_cutoffs) if gate_cutoffs is not None else None,
        "model_output": str(output),
        "unchanged_artifacts": sorted(
            set(manifest["sha256"])
            - {"clusterer.json"}
            - ({COMPLETE_MANIFEST_FILES["incremental_linker_metadata"]} if gate_cutoffs is not None else set())
        ),
    }
    write_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Apply the explicit human choice after review of complete curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-evidence", type=Path)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        help="Snapshot giant-block sweep root; omit it with --datasets when internal exports supply the review",
    )
    parser.add_argument("--datasets", nargs="+", default=[], help="Datasets in the sweep being reviewed.")
    parser.add_argument(
        "--expected-batching-threshold", type=int, help="Production subblock limit required for a snapshot sweep."
    )
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
    parser.add_argument("--allow-wide-tolerance", action="store_true", help="Allow a tolerance above 0.005.")
    parser.add_argument(
        "--block-role-assignments",
        type=Path,
        default=None,
        help="Optional verified assign_block_roles.py assignments.json authorizing block-disjoint eps_cal sweeps.",
    )
    parser.add_argument(
        "--gate-cutoffs",
        type=float,
        nargs=2,
        metavar=("T_FULL", "T_INIT"),
        help="Reviewed ranker-probability gate cutoffs for full-name and initial-only queries.",
    )
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.precision_f1_tolerance is not None
        and args.precision_f1_tolerance > 0.005
        and not args.allow_wide_tolerance
    ):
        parser.error("--precision-f1-tolerance above 0.005 requires --allow-wide-tolerance")
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
        expected_batching_threshold=args.expected_batching_threshold,
        block_role_assignments=args.block_role_assignments,
        identity_evidence=args.identity_evidence,
        gate_cutoffs=(args.gate_cutoffs[0], args.gate_cutoffs[1]) if args.gate_cutoffs else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

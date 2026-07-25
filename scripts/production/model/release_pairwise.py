"""Release-only pairwise calibration, finalization, and one-shot evaluation."""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.consts import NAME_COUNTS_INDEX_PATH  # noqa: E402
from s2and.data import ANDData  # noqa: E402
from s2and.eval import b3_precision_recall_fscore, pairwise_probability_metrics  # noqa: E402
from s2and.featurizer import many_pairs_featurize  # noqa: E402
from s2and.incremental_linking.contracts import canonical_json_digest  # noqa: E402
from s2and.production_bundle import finalize_pairwise_eps  # noqa: E402
from s2and.production_model import _load_pairwise_staging_model  # noqa: E402

PAIR_MANIFEST_SCHEMA = "s2and_pairwise_test_manifest_v1"
CLUSTER_MANIFEST_SCHEMA = "s2and_cluster_test_manifest_v1"


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def _verified_manifest(path: Path, expected_sha256: str, schema_version: str) -> dict[str, Any]:
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"Manifest SHA-256 mismatch: expected={expected_sha256} observed={observed}")
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != schema_version:
        raise ValueError(f"Expected {schema_version!r} manifest at {path}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Evaluation manifest datasets must be a nonempty list")
    dataset_names: set[str] = set()
    for spec in datasets:
        if not isinstance(spec, Mapping):
            raise ValueError("Each evaluation manifest dataset must be a named object")
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Evaluation manifest dataset names must be nonempty strings")
        if name in dataset_names:
            raise ValueError(f"Evaluation manifest contains duplicate dataset name {name!r}")
        dataset_names.add(name)
    return payload


def _resolved_dataset_files(manifest_path: Path, spec: Mapping[str, Any], roles: Sequence[str]) -> dict[str, Path]:
    declared_files = spec.get("files")
    if not isinstance(declared_files, Mapping) or set(declared_files) != set(roles):
        raise ValueError(f"Dataset {spec.get('name')!r} must declare exact file roles {sorted(roles)}")
    resolved: dict[str, Path] = {}
    for role in roles:
        file_spec = declared_files[role]
        if not isinstance(file_spec, Mapping) or set(file_spec) != {"path", "sha256"}:
            raise ValueError(f"Dataset {spec.get('name')!r} file {role!r} must contain exactly path and sha256")
        raw_path = file_spec["path"]
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"Dataset {spec.get('name')!r} file {role!r} path must be a nonempty string")
        expected_sha256 = file_spec["sha256"]
        if not isinstance(expected_sha256, str) or not expected_sha256:
            raise ValueError(f"Dataset {spec.get('name')!r} file {role!r} sha256 must be a nonempty string")
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        observed = _sha256_file(path)
        if observed != expected_sha256:
            raise ValueError(
                f"Dataset {spec.get('name')!r} {role} SHA-256 mismatch: expected={expected_sha256} observed={observed}"
            )
        resolved[role] = path
    return resolved


def _record_unblind(path: Path, *, manifest_path: Path, manifest_sha256: str, model_path: Path) -> None:
    _write_fresh_json(
        path,
        {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "model_path": str(model_path.resolve()),
            "unblinded_at": datetime.datetime.now(datetime.UTC).isoformat(),
        },
    )


def pairwise_metrics(
    labels: np.ndarray,
    main_positive: np.ndarray,
    nameless_positive: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Return the fixed release metric contract and averaged probabilities."""

    return pairwise_probability_metrics(labels, main_positive, nameless_positive)


def _anddata(name: str, files: Mapping[str, Path], *, mode: str, random_seed: int = 1111) -> ANDData:
    return ANDData(
        signatures=str(files["signatures"]),
        papers=str(files["papers"]),
        clusters=str(files["clusters"]) if "clusters" in files else None,
        specter_embeddings=str(files["specter_embeddings"]),
        name=name,
        mode=mode,
        random_seed=random_seed,
        name_counts_index=NAME_COUNTS_INDEX_PATH,
        preprocess=True,
    )


def evaluate_pairs(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a pairwise-only bundle once on one sealed pair manifest."""

    output_path = Path(args.output_json)
    if output_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")
    manifest_path = Path(args.manifest).resolve()
    manifest = _verified_manifest(manifest_path, args.expected_manifest_sha256, PAIR_MANIFEST_SCHEMA)
    verified: list[tuple[Mapping[str, Any], dict[str, Path]]] = []
    roles = ("signatures", "papers", "specter_embeddings", "pairs")
    for spec in manifest["datasets"]:
        verified.append((spec, _resolved_dataset_files(manifest_path, spec, roles)))
    model_path = Path(args.pairwise_model).resolve()
    clusterer = _load_pairwise_staging_model(model_path)
    _record_unblind(
        Path(args.unblind_record),
        manifest_path=manifest_path,
        manifest_sha256=args.expected_manifest_sha256,
        model_path=model_path,
    )

    dataset_reports: dict[str, Any] = {}
    all_labels: list[np.ndarray] = []
    all_main: list[np.ndarray] = []
    all_nameless: list[np.ndarray] = []
    for spec, files in verified:
        name = str(spec["name"])
        raw_pairs = _read_json(files["pairs"])
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError(f"Pair file for {name!r} must contain a nonempty JSON list")
        pairs: list[tuple[str, str, int | float]] = []
        identities: list[dict[str, Any]] = []
        for row in raw_pairs:
            if not isinstance(row, Mapping):
                raise ValueError(f"Pair row for {name!r} must be an object")
            left = str(row["signature_id_1"])
            right = str(row["signature_id_2"])
            label = row.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
                raise ValueError(f"Pair label for {name!r} must be the JSON integer 0 or 1")
            pairs.append((left, right, label))
            identities.append({"signature_id_1": left, "signature_id_2": right, "label": label})
        dataset = _anddata(name, files, mode="inference")
        features, labels, nameless_features = many_pairs_featurize(
            pairs,
            dataset,
            clusterer.featurizer_info,
            n_jobs=int(args.n_jobs),
            nameless_featurizer_info=clusterer.nameless_featurizer_info,
            nan_value=np.nan,
            total_ram_bytes=args.total_ram_bytes,
        )
        if nameless_features is None or clusterer.nameless_classifier is None:
            raise RuntimeError("Release pair evaluation requires the nameless model")
        main = clusterer.classifier.predict_proba(features)[:, 1]
        nameless = clusterer.nameless_classifier.predict_proba(nameless_features)[:, 1]
        metrics, probabilities = pairwise_metrics(labels, main, nameless)
        dataset_reports[name] = {
            "metrics": metrics,
            "pairs": [
                {
                    **identity,
                    "main_probability": float(main[index]),
                    "nameless_probability": float(nameless[index]),
                    "probability": float(probabilities[index]),
                }
                for index, identity in enumerate(identities)
            ],
        }
        all_labels.append(np.asarray(labels))
        all_main.append(np.asarray(main))
        all_nameless.append(np.asarray(nameless))

    aggregate, _ = pairwise_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_main),
        np.concatenate(all_nameless),
    )
    report = {
        "schema_version": "s2and_pairwise_evaluation_report_v1",
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregation": "all_pairs",
        "aggregate": aggregate,
        "datasets": dataset_reports,
    }
    _write_fresh_json(output_path, report)
    return report


def _b3_report(
    true_clusters: Mapping[str, Sequence[str]],
    predicted_clusters: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    precision, recall, f1, per_signature, _, _ = b3_precision_recall_fscore(
        dict(true_clusters),
        dict(predicted_clusters),
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "signature_count": len(per_signature),
    }


def _aggregate_b3(dataset_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(dataset_metrics)
    weights = np.asarray([dataset_metrics[name]["signature_count"] for name in names], dtype=np.float64)
    if not names or np.any(weights <= 0):
        raise ValueError("B3 aggregation requires positive evaluated signature counts")
    return {
        "dataset_macro": {
            metric: float(np.mean([dataset_metrics[name][metric] for name in names]))
            for metric in ("precision", "recall", "f1")
        },
        "signature_weighted": {
            metric: float(np.average([dataset_metrics[name][metric] for name in names], weights=weights))
            for metric in ("precision", "recall", "f1")
        },
        "signature_count": int(weights.sum()),
    }


def _training_config(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / "reproducibility" / "pairwise_training_config.json"
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("training_scope") != "production_full":
        raise ValueError("EPS calibration requires production_full pairwise training provenance")
    return payload


def calibrate_eps(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate explicit EPS values on the immutable training validation identities.

    Every cheap precondition — output absence, EPS domain, bundle load, and input
    digests — is checked before any distance matrix is built, so an operator
    error cannot surface only after hours of featurization.

    One dataset's condensed distance matrices are resident at a time: the
    dataset loop is outer and the EPS loop is inner. Matrices are still built
    exactly once per dataset and reused across every EPS value, so this bounds
    peak RSS without adding recomputation.

    Validation blocks are filtered only for singletons. Canonical pairwise
    bundles no longer persist ``val_blocks_size``, and the production trainer
    constructs its ``Clusterer`` without one, so training and calibration both
    score all non-singleton validation blocks. Passing the loaded attribute here
    would silently cap calibration for a noncanonical bundle that still carries
    a finite value.
    """

    output_path = Path(args.output_json)
    if output_path.exists():
        raise FileExistsError(f"Calibration output already exists: {output_path}")
    eps_values = sorted({float(value) for value in args.eps})
    for eps in eps_values:
        if not math.isfinite(eps) or not 0.0 <= eps <= 1.0:
            raise ValueError(f"EPS values must be finite and in [0, 1], got {eps!r}")

    bundle_dir = Path(args.pairwise_model).resolve()
    clusterer = _load_pairwise_staging_model(bundle_dir)
    clusterer.n_jobs = args.n_jobs
    config = _training_config(bundle_dir)

    resolved_inputs: list[tuple[str, dict[str, Path]]] = []
    for name, dataset_spec in config["dataset_inputs"].items():
        if dataset_spec["split_mode"] != "random_blocks":
            continue
        raw_files = dataset_spec["files"]
        files: dict[str, Path] = {}
        for role in ("signatures", "papers", "specter_embeddings", "clusters"):
            source = raw_files[role]
            path = Path(source["path"]).resolve()
            observed = _sha256_file(path)
            if observed != source["sha256"]:
                raise ValueError(f"Calibration input drift for {name}:{role}")
            files[role] = path
        resolved_inputs.append((str(name), files))
    if not resolved_inputs:
        raise ValueError("EPS calibration found no clustered validation datasets")

    identities: dict[str, Any] = {}
    metrics_by_eps: dict[float, dict[str, Any]] = {eps: {} for eps in eps_values}
    original_eps = float(clusterer.cluster_model.eps)
    try:
        for name, files in resolved_inputs:
            dataset = _anddata(
                name,
                files,
                mode="train",
                random_seed=int(config["data_random_seed"]),
            )
            _, val_blocks, _ = dataset.split_cluster_signatures()
            val_blocks = clusterer.filter_blocks(val_blocks)
            true_clusters = dataset.construct_cluster_to_signatures(val_blocks)
            distances = clusterer.make_distance_matrices(
                val_blocks,
                dataset,
                total_ram_bytes=args.total_ram_bytes,
            )
            identities[name] = {
                "blocks": val_blocks,
                "digest": canonical_json_digest(val_blocks),
            }
            for eps in eps_values:
                clusterer.cluster_model.eps = eps
                predicted, _ = clusterer.predict(
                    val_blocks,
                    dataset,
                    dists=distances,
                    total_ram_bytes=args.total_ram_bytes,
                )
                metrics_by_eps[eps][name] = _b3_report(true_clusters, predicted)
            del distances, dataset, val_blocks, true_clusters
            gc.collect()
    finally:
        clusterer.cluster_model.eps = original_eps

    trials = [{"eps": eps, "datasets": metrics_by_eps[eps], **_aggregate_b3(metrics_by_eps[eps])} for eps in eps_values]
    selected = max(trials, key=lambda trial: (trial["signature_weighted"]["f1"], -trial["eps"]))
    report = {
        "schema_version": "s2and_eps_calibration_report_v1",
        "objective": "maximize_signature_weighted_b3_f1",
        "tie_break": "smallest_eps",
        "selected_eps": selected["eps"],
        "validation_identities": identities,
        "trials": trials,
    }
    _write_fresh_json(output_path, report)
    return report


def evaluate_clusters(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate a frozen pairwise stage once on sealed cluster identities."""

    output_path = Path(args.output_json)
    if output_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")
    manifest_path = Path(args.manifest).resolve()
    manifest = _verified_manifest(manifest_path, args.expected_manifest_sha256, CLUSTER_MANIFEST_SCHEMA)
    roles = ("signatures", "papers", "specter_embeddings", "clusters", "blocks")
    verified: list[tuple[Mapping[str, Any], dict[str, Path]]] = []
    for spec in manifest["datasets"]:
        verified.append((spec, _resolved_dataset_files(manifest_path, spec, roles)))
    model_path = Path(args.pairwise_model).resolve()
    clusterer = _load_pairwise_staging_model(model_path)
    clusterer.n_jobs = args.n_jobs
    _record_unblind(
        Path(args.unblind_record),
        manifest_path=manifest_path,
        manifest_sha256=args.expected_manifest_sha256,
        model_path=model_path,
    )

    dataset_reports: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for spec, files in verified:
        name = str(spec["name"])
        blocks = _read_json(files["blocks"])
        if not isinstance(blocks, dict) or not blocks:
            raise ValueError(f"Cluster blocks for {name!r} must be a nonempty object")
        if any(not isinstance(values, list) for values in blocks.values()):
            raise ValueError(f"Cluster blocks for {name!r} must map block ids to signature lists")
        normalized_blocks = {str(key): [str(value) for value in values] for key, values in blocks.items()}
        flattened = [signature for values in normalized_blocks.values() for signature in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"Cluster blocks for {name!r} contain duplicate signatures")
        dataset = _anddata(name, files, mode="train")
        true_clusters = dataset.construct_cluster_to_signatures(normalized_blocks)
        predicted, _ = clusterer.predict(normalized_blocks, dataset)
        metrics[name] = _b3_report(true_clusters, predicted)
        dataset_reports[name] = {
            "metrics": metrics[name],
            "blocks_digest": canonical_json_digest(normalized_blocks),
            "predicted_clusters": predicted,
        }
    report = {
        "schema_version": "s2and_cluster_evaluation_report_v1",
        "manifest_sha256": args.expected_manifest_sha256,
        "datasets": dataset_reports,
        **_aggregate_b3(metrics),
    }
    _write_fresh_json(output_path, report)
    return report


def finalize_eps(args: argparse.Namespace) -> dict[str, Any]:
    """Run the atomic fresh-output EPS finalizer."""

    summary = finalize_pairwise_eps(
        source_bundle_dir=Path(args.source_bundle),
        output_bundle_dir=Path(args.output_bundle),
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_old_eps=args.expected_old_eps,
        new_eps=args.new_eps,
    )
    return {
        "bundle_dir": str(summary.bundle_dir),
        "bundle_version": summary.bundle_version,
        "manifest_path": str(summary.manifest_path),
    }


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate-eps", help="Select EPS on validation identities only.")
    calibrate.add_argument("--pairwise-model", type=Path, required=True)
    calibrate.add_argument("--eps", type=float, nargs="+", required=True)
    calibrate.add_argument("--output-json", type=Path, required=True)
    calibrate.add_argument("--n-jobs", type=_positive_int, default=1)
    calibrate.add_argument("--total-ram-bytes", type=_positive_int)
    calibrate.set_defaults(handler=calibrate_eps)

    finalizer = commands.add_parser("finalize-eps", help="Write a fresh pairwise stage with reviewed EPS.")
    finalizer.add_argument("--source-bundle", type=Path, required=True)
    finalizer.add_argument("--output-bundle", type=Path, required=True)
    finalizer.add_argument("--expected-manifest-sha256", required=True)
    finalizer.add_argument("--expected-old-eps", type=float, required=True)
    finalizer.add_argument("--new-eps", type=float, required=True)
    finalizer.set_defaults(handler=finalize_eps)

    pair_evaluator = commands.add_parser("evaluate-pairs", help="Run the one-shot sealed pair evaluation.")
    pair_evaluator.add_argument("--total-ram-bytes", type=_positive_int)
    cluster_evaluator = commands.add_parser(
        "evaluate-clusters",
        help="Run the one-shot sealed cluster evaluation.",
    )
    for evaluator, handler in (
        (pair_evaluator, evaluate_pairs),
        (cluster_evaluator, evaluate_clusters),
    ):
        evaluator.add_argument("--pairwise-model", type=Path, required=True)
        evaluator.add_argument("--manifest", type=Path, required=True)
        evaluator.add_argument("--expected-manifest-sha256", required=True)
        evaluator.add_argument("--unblind-record", type=Path, required=True)
        evaluator.add_argument("--output-json", type=Path, required=True)
        evaluator.add_argument("--n-jobs", type=_positive_int, default=1)
        evaluator.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

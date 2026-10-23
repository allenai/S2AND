"""Calibrate EPS on validation blocks and score a model on the model plan's outer test."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._atomic_io import fsync_directory  # noqa: E402
from s2and._sha256 import sha256_file as _sha256_file  # noqa: E402
from s2and.arrow_inputs import ArrowDataset  # noqa: E402
from s2and.arrow_training import load_signatures_from_arrow  # noqa: E402
from s2and.data import ANDData  # noqa: E402
from s2and.eval import b3_precision_recall_fscore, pairwise_probability_metrics  # noqa: E402
from s2and.feature_port import build_verified_rust_featurizer, featurize_pairs_with_rust  # noqa: E402
from s2and.metrics import _validated_cluster_partition_coverage  # noqa: E402
from s2and.production_bundle import finalize_pairwise_eps  # noqa: E402
from s2and.production_model import (  # noqa: E402
    _load_pairwise_staging_model,
    load_production_model,
)
from s2and.production_training_contract import (  # noqa: E402
    ModelDataset,
    ModelPlan,
    ProductionArtifactAuthority,
    load_model_plan,
    load_packaged_artifact_authority,
    require_recorded_population,
)
from scripts.production.model.augmented_leakage import (  # noqa: E402
    reject_heldout_members,
    require_declared_sources,
    source_members,
)
from scripts.production.model.plan_datasets import (  # noqa: E402
    excluded_signature_ids,
    heldout_blocks,
    heldout_signature_ids,
    load_development_dataset,
    load_heldout_dataset,
    open_plan_arrow,
    plan_unknown_signatures,
    require_arrow_plan,
)
from scripts.production.model.unknown_signatures import (  # noqa: E402
    augmented_unknown_ids,
    exclude_unknown_pairs,
    same_unknown_content,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one new JSON file without replacing an existing file."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, staging_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staging_path = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.link(staging_path, path)
        fsync_directory(path.parent)
    finally:
        staging_path.unlink(missing_ok=True)


def _reject_nonfinite_json(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite_json(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_nonfinite_json(nested, label=f"{label}[{index}]")


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _unit_float(value: Any, *, label: str) -> float:
    numeric = _number(value, label=label)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return numeric


def _signature_id(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{context} signature id must be a nonempty trimmed string")
    return value


def _pair_identity(left: Any, right: Any, *, context: str) -> tuple[str, str]:
    left_id = _signature_id(left, context=context)
    right_id = _signature_id(right, context=context)
    if left_id == right_id:
        raise ValueError(f"{context} contains a self-pair")
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def _csv_pair_identities(path: Path, *, context: str) -> set[tuple[str, str]]:
    """Validate training pairs and return unordered identities."""

    identities: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"signature_id_1", "signature_id_2"} <= set(reader.fieldnames):
            raise ValueError(f"{context} must contain signature_id_1 and signature_id_2")
        for row_number, row in enumerate(reader, start=2):
            identity = _pair_identity(
                row.get("signature_id_1"),
                row.get("signature_id_2"),
                context=f"{context} row {row_number}",
            )
            if identity in identities:
                raise ValueError(f"{context} contains a duplicate unordered pair")
            identities.add(identity)
    if not identities:
        raise ValueError(f"{context} must contain at least one pair")
    return identities


def _heldout_pair_identities(path: Path, *, context: str) -> set[tuple[str, str]]:
    """Read only held-out pair identities; labels are intentionally ignored."""

    rows = _read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{context} must contain a nonempty JSON list")
    identities: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{context} row {row_number} must be an object")
        identity = _pair_identity(
            row.get("signature_id_1"),
            row.get("signature_id_2"),
            context=f"{context} row {row_number}",
        )
        if identity in identities:
            raise ValueError(f"{context} contains a duplicate unordered pair")
        identities.add(identity)
    return identities


def _validated_probability_vector(values: np.ndarray, *, label: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{label} probabilities must be finite and in [0, 1]")
    return probabilities


def _validated_unit_interval_metrics(metrics: Mapping[str, Any], *, label: str) -> None:
    for name, value in metrics.items():
        if name in {"rows", "signature_count"}:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(f"{label} {name} must be a positive integer")
        elif not 0.0 <= _number(value, label=f"{label} {name}") <= 1.0:
            raise RuntimeError(f"{label} {name} must be in [0, 1]")


def pairwise_metrics(
    labels: np.ndarray,
    main_positive: np.ndarray,
    nameless_positive: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Return the fixed release metric contract and averaged probabilities."""

    main = _validated_probability_vector(main_positive, label="main")
    nameless = _validated_probability_vector(nameless_positive, label="nameless")
    metrics, probabilities = pairwise_probability_metrics(labels, main, nameless)
    _validated_unit_interval_metrics(metrics, label="pairwise metric")
    return metrics, _validated_probability_vector(probabilities, label="averaged")


def _load_release_artifacts(args: argparse.Namespace) -> ProductionArtifactAuthority:
    return load_packaged_artifact_authority(
        name_counts_index_root=Path(args.name_counts_index_root),
    )


def _labeled_pairs(path: Path, *, name: str) -> list[tuple[str, str, int]]:
    rows = _read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Pair file for {name!r} must contain a nonempty list")
    pairs: list[tuple[str, str, int]] = []
    identities: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or set(row) != {"signature_id_1", "signature_id_2", "label"}:
            raise ValueError(f"Pair row {row_number} for {name!r} has invalid fields")
        typed_row = cast(Mapping[str, Any], row)
        identity = _pair_identity(
            typed_row["signature_id_1"],
            typed_row["signature_id_2"],
            context=f"{name} row {row_number}",
        )
        label = typed_row["label"]
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"Pair label for {name!r} must be the JSON integer 0 or 1")
        if identity in identities:
            raise ValueError(f"Pair file for {name!r} contains a duplicate unordered pair")
        identities.add(identity)
        pairs.append((identity[0], identity[1], label))
    return pairs


def _pairwise_predictions(
    pairs: Sequence[tuple[str, str, int | float]],
    rust_featurizer: Any,
    clusterer: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    features, labels, nameless_features = featurize_pairs_with_rust(
        rust_featurizer,
        list(pairs),
        clusterer.featurizer_info,
        nameless_featurizer_info=clusterer.nameless_featurizer_info,
        n_jobs=int(args.n_jobs),
        nan_value=np.nan,
        total_ram_bytes=args.total_ram_bytes,
    )
    if nameless_features is None or clusterer.nameless_classifier is None:
        raise RuntimeError("Release pair evaluation requires the nameless model")
    main = clusterer.classifier.predict_proba(features)[:, 1]
    nameless = clusterer.nameless_classifier.predict_proba(nameless_features)[:, 1]
    metrics, _ = pairwise_metrics(labels, main, nameless)
    return metrics, np.asarray(labels), np.asarray(main), np.asarray(nameless)


def _evaluation_model(args: argparse.Namespace) -> tuple[ProductionArtifactAuthority, Any]:
    artifacts = _load_release_artifacts(args)
    model = load_production_model(
        Path(args.model).resolve(),
        expected_artifact_hashes=artifacts.hashes,
    )
    return artifacts, model


def _b3_report(
    true_clusters: Mapping[str, Sequence[str]],
    predicted_clusters: Mapping[str, Sequence[str]],
    unknown_signature_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if unknown_signature_ids:
        # Full inference must cover unknowns before the score removes their fallback labels.
        _validated_cluster_partition_coverage(dict(true_clusters), dict(predicted_clusters))
        true_clusters, predicted_clusters = (
            {
                cluster: known
                for cluster, members in partition.items()
                if (known := [signature for signature in members if signature not in unknown_signature_ids])
            }
            for partition in (true_clusters, predicted_clusters)
        )
    precision, recall, f1, per_signature, _, _ = b3_precision_recall_fscore(
        dict(true_clusters),
        dict(predicted_clusters),
    )
    if unknown_signature_ids and not per_signature:
        raise ValueError("B3 requires at least one known signature")
    report = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "signature_count": len(per_signature),
    }
    _validated_unit_interval_metrics(report, label="B3 metric")
    return report


def _aggregate_b3(dataset_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(dataset_metrics)
    weights = np.asarray([dataset_metrics[name]["signature_count"] for name in names], dtype=np.float64)
    if not names or np.any(weights <= 0):
        raise ValueError("B3 aggregation requires positive evaluated signature counts")
    report = {
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
    _validated_unit_interval_metrics(report["dataset_macro"], label="B3 dataset-macro metric")
    _validated_unit_interval_metrics(report["signature_weighted"], label="B3 signature-weighted metric")
    return report


def _signature_ids_from_arrow(arrow_dataset: ArrowDataset) -> set[str]:
    with arrow_dataset.use() as lease, lease.open_file("signatures") as source:
        return set(load_signatures_from_arrow(source))


def _plan_populations(plan: ModelPlan) -> dict[str, dict[str, Any]]:
    """Read each dataset's signature populations without reading held-out labels."""

    populations: dict[str, dict[str, Any]] = {}
    for name, dataset in plan.datasets.items():
        files = dataset.files
        pairs = _heldout_pair_identities(files["heldout_pairs"], context=f"held-out pairs {name!r}")
        if dataset.split_mode == "fixed_pairs":
            populations[name] = {
                "pairs": pairs,
                "train": _csv_pair_identities(files["train_pairs"], context=f"{name!r} train pairs"),
                "validation": _csv_pair_identities(files["val_pairs"], context=f"{name!r} validation pairs"),
            }
            continue
        with open_plan_arrow(dataset) as arrow_dataset:
            complete = _signature_ids_from_arrow(arrow_dataset)
        heldout = set(heldout_signature_ids(dataset))
        if not heldout <= complete:
            raise ValueError(f"Held-out blocks name signatures outside the Arrow dataset {name!r}")
        populations[name] = {
            "pairs": pairs,
            "heldout": heldout,
            "development": complete - heldout - excluded_signature_ids(dataset),
        }
    return populations


def _check_plan_leakage(plan: ModelPlan, populations: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject development and outer-test overlap without inspecting held-out answers."""

    for name, population in populations.items():
        if "development" not in population:
            for left_name, right_name in (("train", "validation"), ("train", "pairs"), ("validation", "pairs")):
                overlap = population[left_name] & population[right_name]
                if overlap:
                    raise ValueError(
                        f"Dataset {name!r} {left_name}/{right_name} unordered pairs overlap: "
                        f"count={len(overlap)} sample={sorted(overlap)[:5]}"
                    )
            continue
        # Outer-test pairs must come from the held-out blocks, because every other signature can train.
        outside = {signature_id for pair in population["pairs"] for signature_id in pair} - population["heldout"]
        if outside:
            raise ValueError(
                f"Held-out pairs for {name!r} use signatures outside the held-out blocks: "
                f"count={len(outside)} sample={sorted(outside)[:5]}"
            )
    if "augmented" not in populations:
        return
    development = {
        name: (
            population["development"]
            if "development" in population
            else {
                signature_id for split in ("train", "validation") for pair in population[split] for signature_id in pair
            }
        )
        for name, population in populations.items()
        if name != "augmented"
    }
    evaluation_members = source_members(populations["augmented"]["pairs"])
    for name, members in development.items():
        overlap = evaluation_members.get(name, set()) & members
        if overlap:
            raise ValueError(
                f"Augmented outer-test pairs contain model {name!r} signatures: "
                f"count={len(overlap)} sample={sorted(overlap)[:5]}. "
                "Exclude the complete benchmark development population from augmented evaluation."
            )
    unknown_ids = augmented_unknown_ids(
        {name: plan_unknown_signatures(dataset).ids for name, dataset in plan.datasets.items() if name != "augmented"}
    )
    heldout = {
        name: population["heldout"] | {signature_id for pair in population["pairs"] for signature_id in pair}
        for name, population in populations.items()
        if "heldout" in population
    }
    # Validation pairs select the model, so held-out copies there leak as surely as in training.
    for split, label in (("train", "training"), ("validation", "validation")):
        augmented = source_members(
            (left, right)
            for left, right in populations["augmented"][split]
            if left not in unknown_ids and right not in unknown_ids
        )
        require_declared_sources(augmented, tuple(plan.datasets), context=f"Augmented {label}")
        reject_heldout_members(augmented, heldout, context="Outer-test evaluation", population=label)


def _data_random_seed(config: Mapping[str, Any]) -> int:
    """Return the recorded seed that split the model's development blocks."""

    seed = config.get("data_random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Pairwise training data_random_seed must be an integer")
    return seed


def _training_config(bundle_dir: Path) -> dict[str, Any]:
    payload = _read_json(bundle_dir / "reproducibility" / "pairwise_training_config.json")
    if not isinstance(payload, dict) or payload.get("training_scope") != "production_full":
        raise ValueError("Pairwise release commands require a production_full training configuration")
    _data_random_seed(payload)
    return payload


def _recorded_populations(bundle_dir: Path) -> Mapping[str, Any]:
    payload = _read_json(bundle_dir / "reproducibility" / "pairwise_training_summary.json")
    splits = payload.get("cluster_test_splits") if isinstance(payload, Mapping) else None
    if not isinstance(splits, Mapping):
        raise ValueError("Pairwise model records no training populations")
    return splits


def _heldout_dataset(
    name: str,
    dataset: ModelDataset,
    arrow_dataset: ArrowDataset,
    *,
    recorded: Mapping[str, Any],
    random_seed: int,
    name_tuples: frozenset[tuple[str, str]],
    n_jobs: int,
) -> tuple[ANDData, dict[str, list[str]] | None]:
    """Load the outer-test population after binding its complement to the model."""

    development = load_development_dataset(
        name, dataset, arrow_dataset, name_tuples=name_tuples, random_seed=random_seed, n_jobs=n_jobs
    )
    if dataset.split_mode == "fixed_pairs":
        # Fixed-pair datasets score only their frozen held-out pairs.
        return development, None
    require_recorded_population(recorded.get(name), development.get_blocks(), name=name)
    del development
    heldout = load_heldout_dataset(name, dataset, arrow_dataset, name_tuples=name_tuples, n_jobs=n_jobs)
    blocks = heldout_blocks(dataset)
    if {key: sorted(members) for key, members in blocks.items()} != {
        key: sorted(members) for key, members in heldout.get_blocks().items()
    }:
        raise ValueError(f"Held-out blocks differ from the held-out Arrow population for {name!r}")
    return heldout, blocks


def evaluate_heldout(args: argparse.Namespace) -> dict[str, Any]:
    """Score the plan's outer-test blocks and pairs with one complete model and no baseline."""

    output_path = Path(args.output_report).resolve()
    if output_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")
    plan = load_model_plan(Path(args.model_plan).resolve())
    require_arrow_plan(plan.datasets)
    model_dir = Path(args.model).resolve()
    config = _training_config(model_dir)
    recorded = _recorded_populations(model_dir)
    _check_plan_leakage(plan, _plan_populations(plan))
    artifacts, clusterer = _evaluation_model(args)
    if clusterer.production_model_release_version != plan.release_version:
        raise ValueError("Model plan and complete model release versions differ")
    clusterer.n_jobs = int(args.n_jobs)
    name_tuples = artifacts.name_tuples.pairs

    unknown_sources = {name: plan_unknown_signatures(dataset) for name, dataset in plan.datasets.items()}
    augmented_unknowns = augmented_unknown_ids(
        {name: source.ids for name, source in unknown_sources.items() if name != "augmented"}
    )
    pair_reports: dict[str, Any] = {}
    cluster_reports: dict[str, Any] = {}
    b3_metrics: dict[str, dict[str, Any]] = {}
    all_labels: list[np.ndarray] = []
    all_main: list[np.ndarray] = []
    all_nameless: list[np.ndarray] = []
    for name, dataset_spec in plan.datasets.items():
        pairs = _labeled_pairs(dataset_spec.files["heldout_pairs"], name=name)
        unknown_ids = augmented_unknowns if name == "augmented" else unknown_sources[name].ids
        selected_pairs = exclude_unknown_pairs(pairs, unknown_ids)
        with open_plan_arrow(dataset_spec) as arrow_dataset:
            dataset, blocks = _heldout_dataset(
                name,
                dataset_spec,
                arrow_dataset,
                recorded=recorded,
                random_seed=int(config["data_random_seed"]),
                name_tuples=name_tuples,
                n_jobs=int(args.n_jobs),
            )
            selected_ids = {str(pair[index]) for pair in selected_pairs for index in (0, 1)}
            if blocks is not None:
                selected_ids.update(signature for members in blocks.values() for signature in members)
            rust_featurizer = build_verified_rust_featurizer(
                dataset,
                arrow_dataset,
                signature_ids=selected_ids,
                name_counts_manifest_sha256=artifacts.name_counts_index.manifest_sha256,
                num_threads=int(args.n_jobs),
            )
            metrics, labels, main, nameless = _pairwise_predictions(selected_pairs, rust_featurizer, clusterer, args)
            pair_reports[name] = {"metrics": metrics, "unknown_pairs_removed": len(pairs) - len(selected_pairs)}
            all_labels.append(labels)
            all_main.append(main)
            all_nameless.append(nameless)
            if blocks is not None:
                true_clusters = dataset.construct_cluster_to_signatures(blocks)
                seeded = rust_featurizer.with_cluster_seeds(
                    dataset.cluster_seeds_require, dataset.cluster_seeds_disallow
                )
                predicted, _ = clusterer.predict_from_rust_featurizer(
                    blocks,
                    seeded,
                    cluster_seeds_disallow=(
                        dataset.cluster_seeds_disallow if clusterer.use_default_constraints_as_supervision else None
                    ),
                    total_ram_bytes=args.total_ram_bytes,
                )
                scored_unknowns = unknown_ids & set(dataset.signatures)
                b3_metrics[name] = _b3_report(true_clusters, predicted, scored_unknowns)
                cluster_reports[name] = {
                    "metrics": b3_metrics[name],
                    "blocks": len(blocks),
                    "inference_signatures": sum(map(len, blocks.values())),
                    "unknown_signatures": len(scored_unknowns),
                }
            del rust_featurizer, dataset
        print(json.dumps({"event": "heldout_dataset_complete", "dataset": name}), flush=True)
    aggregate, _ = pairwise_metrics(np.concatenate(all_labels), np.concatenate(all_main), np.concatenate(all_nameless))
    report = {
        "clusterer_config": {
            "eps": float(clusterer.cluster_model.eps),
            "use_default_constraints_as_supervision": bool(clusterer.use_default_constraints_as_supervision),
            "suppress_orcid": bool(clusterer.suppress_orcid),
            "dont_merge_cluster_seeds": bool(clusterer.dont_merge_cluster_seeds),
        },
        "pairwise": {"aggregation": "all_pairs", "aggregate": aggregate, "datasets": pair_reports},
        "cluster": {"datasets": cluster_reports, **_aggregate_b3(b3_metrics)},
    }
    _reject_nonfinite_json(report, label="heldout metrics")
    _write_fresh_json(output_path, report)
    return report


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _calibration_preflight(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, ModelPlan, dict[str, Any], str]:
    source_bundle = Path(args.source_bundle).resolve()
    model_plan_path = Path(args.model_plan).resolve()
    output_bundle = Path(args.output_bundle).resolve()
    output_report = Path(args.output_report).resolve()
    if not source_bundle.is_dir():
        raise FileNotFoundError(f"Calibration source bundle does not exist: {source_bundle}")
    if not model_plan_path.is_file():
        raise FileNotFoundError(f"Model plan does not exist: {model_plan_path}")
    if output_bundle.exists() or output_report.exists():
        raise FileExistsError("Calibration outputs must both be fresh")
    if _paths_overlap(source_bundle, output_bundle):
        raise ValueError("Calibration source and output bundle paths must not overlap")
    if output_report.is_relative_to(source_bundle) or _paths_overlap(output_report, output_bundle):
        raise ValueError("Calibration report must be outside source and output bundles")
    plan = load_model_plan(model_plan_path)
    require_arrow_plan(plan.datasets)
    config = _training_config(source_bundle)
    manifest_sha256 = _sha256_file(source_bundle / "manifest.json")
    return source_bundle, output_bundle, output_report, plan, config, manifest_sha256


def calibrate_eps(args: argparse.Namespace) -> dict[str, Any]:
    """Select EPS using each random-block validation population once."""

    source_bundle, output_bundle, output_report, plan, config, manifest_sha256 = _calibration_preflight(args)
    recorded = _recorded_populations(source_bundle)
    artifacts = _load_release_artifacts(args)
    clusterer = _load_pairwise_staging_model(
        source_bundle,
        expected_eps_calibration="pending",
        expected_artifact_hashes=artifacts.hashes,
    )
    clusterer.n_jobs = args.n_jobs
    datasets = [
        (name, dataset) for name, dataset in sorted(plan.datasets.items()) if dataset.split_mode == "random_blocks"
    ]
    if not datasets:
        raise ValueError("EPS calibration found no random-block validation datasets")

    identities: dict[str, Any] = {}
    metrics_by_eps: dict[float, dict[str, Any]] = {eps: {} for eps in plan.eps.grid}
    original_eps = _unit_float(clusterer.cluster_model.eps, label="Source EPS")
    try:
        for name, dataset_spec in datasets:
            unknown_source = plan_unknown_signatures(dataset_spec)
            unknown_bindings = config.get("unknown_signature_inputs")
            if unknown_bindings is None:
                if unknown_source.ids:
                    raise ValueError("EPS calibration requires a bundle trained with unknown-signature exclusions")
            elif not same_unknown_content(unknown_bindings.get(name), unknown_source):
                raise ValueError(f"Unknown-signature sidecar changed after training for {name!r}")
            with open_plan_arrow(dataset_spec) as arrow_dataset:
                dataset = load_development_dataset(
                    name,
                    dataset_spec,
                    arrow_dataset,
                    name_tuples=artifacts.name_tuples.pairs,
                    random_seed=int(config["data_random_seed"]),
                    n_jobs=int(args.n_jobs),
                )
                require_recorded_population(recorded.get(name), dataset.get_blocks(), name=name)
                _, val_blocks, _ = dataset.split_cluster_signatures()
                val_blocks = clusterer.filter_blocks(val_blocks)
                true_clusters = dataset.construct_cluster_to_signatures(val_blocks)
                rust_featurizer = build_verified_rust_featurizer(
                    dataset,
                    arrow_dataset,
                    signature_ids={signature for signatures in val_blocks.values() for signature in signatures},
                    name_counts_manifest_sha256=artifacts.name_counts_index.manifest_sha256,
                    num_threads=int(args.n_jobs),
                )
                distances = clusterer.make_distance_matrices_from_rust_featurizer(
                    val_blocks,
                    rust_featurizer.with_cluster_seeds(dataset.cluster_seeds_require, dataset.cluster_seeds_disallow),
                    total_ram_bytes=args.total_ram_bytes,
                )
                del rust_featurizer
                identities[name] = {
                    "block_count": len(val_blocks),
                    "signature_count": sum(len(signatures) for signatures in val_blocks.values()),
                }
                # Prediction can read the Arrow-backed dataset, so it stays inside the open handle.
                for eps in plan.eps.grid:
                    clusterer.cluster_model.eps = eps
                    predicted, _ = clusterer.predict(
                        val_blocks,
                        dataset,
                        dists=distances,
                        total_ram_bytes=args.total_ram_bytes,
                    )
                    metrics_by_eps[eps][name] = _b3_report(true_clusters, predicted, unknown_source.ids)
                del distances, dataset, val_blocks, true_clusters
            gc.collect()
    finally:
        clusterer.cluster_model.eps = original_eps

    trials: list[dict[str, Any]] = [
        {"eps": eps, "datasets": metrics_by_eps[eps], **_aggregate_b3(metrics_by_eps[eps])} for eps in plan.eps.grid
    ]
    eligible = [
        trial
        for trial in trials
        if trial["signature_weighted"]["f1"] >= plan.eps.minimum_signature_weighted_f1
        and all(metrics["f1"] >= plan.eps.minimum_dataset_f1 for metrics in trial["datasets"].values())
    ]
    if not eligible:
        raise RuntimeError("No EPS calibration trial met all quality floors")
    selected = max(eligible, key=lambda trial: (trial["signature_weighted"]["f1"], -trial["eps"]))
    selected_eps = float(selected["eps"])
    finalize_pairwise_eps(
        source_bundle_dir=source_bundle,
        output_bundle_dir=output_bundle,
        expected_manifest_sha256=manifest_sha256,
        new_eps=selected_eps,
        expected_artifact_hashes=artifacts.hashes,
    )
    report = {
        "source_eps": original_eps,
        "selected_eps": selected_eps,
        "validation_identities": identities,
        "trials": trials,
    }
    _write_fresh_json(output_report, report)
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate-eps", help="Calibrate EPS and write a fresh pairwise bundle.")
    calibrate.add_argument("--source-bundle", type=Path, required=True)
    calibrate.add_argument("--model-plan", type=Path, required=True)
    calibrate.add_argument("--output-bundle", type=Path, required=True)
    calibrate.add_argument("--output-report", type=Path, required=True)
    calibrate.add_argument("--name-counts-index-root", type=Path, required=True)
    calibrate.add_argument("--n-jobs", type=_positive_int, default=1)
    calibrate.add_argument("--total-ram-bytes", type=_positive_int)
    calibrate.set_defaults(handler=calibrate_eps)

    heldout = commands.add_parser("evaluate-heldout", help="Score the model plan's outer test without a baseline.")
    heldout.add_argument("--model-plan", type=Path, required=True)
    heldout.add_argument("--model", type=Path, required=True)
    heldout.add_argument("--name-counts-index-root", type=Path, required=True)
    heldout.add_argument("--output-report", type=Path, required=True)
    heldout.add_argument("--n-jobs", type=_positive_int, default=1)
    heldout.add_argument("--total-ram-bytes", type=_positive_int)
    heldout.set_defaults(handler=evaluate_heldout)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

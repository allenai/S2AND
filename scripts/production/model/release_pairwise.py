"""Prepare, calibrate, and evaluate one production release run."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._atomic_io import fsync_directory  # noqa: E402
from s2and._sha256 import sha256_file as _sha256_file  # noqa: E402
from s2and.data import ANDData  # noqa: E402
from s2and.eval import b3_precision_recall_fscore, pairwise_probability_metrics  # noqa: E402
from s2and.featurizer import many_pairs_featurize  # noqa: E402
from s2and.name_counts_index import NameCountsIndex  # noqa: E402
from s2and.production_bundle import finalize_pairwise_eps  # noqa: E402
from s2and.production_model import _load_pairwise_staging_model, load_production_model  # noqa: E402
from s2and.production_training_contract import (  # noqa: E402
    ModelPlan,
    ProductionArtifactAuthority,
    load_model_plan,
    load_packaged_artifact_authority,
)

EPS_CALIBRATION_REPORT_SCHEMA = "s2and_eps_calibration_report_v1"
EVALUATION_REPORT_SCHEMA = "s2and_release_evaluation_report_v1"

_COMMON_MODEL_ROLES = frozenset({"signatures", "papers", "specter_embeddings"})
_RANDOM_MODEL_ROLES = _COMMON_MODEL_ROLES | {"clusters"}
_FIXED_MODEL_ROLES = _COMMON_MODEL_ROLES | {"train_pairs", "val_pairs"}
_PAIR_ROLES = frozenset({"signatures", "papers", "specter_embeddings", "pairs"})
_CLUSTER_ROLES = frozenset({"signatures", "papers", "specter_embeddings", "clusters", "blocks"})
_RELEASE_GATE_INPUTS = {
    "cluster_evaluation_report": ("cluster_evaluation_report.json", "s2and_cluster_evaluation_report_v1"),
    "pairwise_evaluation_report": ("pairwise_evaluation_report.json", "s2and_pairwise_evaluation_report_v1"),
    "parity_evaluation_report": ("parity_evaluation_report.json", "s2and_parity_evaluation_report_v1"),
    "performance_evaluation_report": ("performance_evaluation_report.json", "s2and_performance_evaluation_report_v1"),
    "subblocking_evaluation_report": ("subblocking_evaluation_report.json", "s2and_subblocking_evaluation_report_v1"),
}
_NORMATIVE_GATE_MAXIMA = {
    "cluster_signature_weighted_b3_f1_max_drop": 0.005,
    "pairwise_aggregate_auroc_max_drop": 0.001,
    "pairwise_aggregate_macro_f1_max_drop": 0.005,
    "pairwise_dataset_auroc_max_drop": 0.001,
    "pairwise_dataset_macro_f1_max_drop": 0.005,
    "runtime_max_ratio": 1.1,
}


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """The held-out populations and release policy prepared for one run."""

    pairwise: Mapping[str, Mapping[str, tuple[Path, str]]]
    cluster: Mapping[str, Mapping[str, tuple[Path, str]]]
    arrow_root: Path
    workload: Mapping[str, Any]
    baselines: Mapping[str, Any]
    gates: Mapping[str, Any]


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


def _object(value: Any, *, label: str, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)}")
    return cast(Mapping[str, Any], value)


def _named_objects(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a nonempty object")
    for name, spec in value.items():
        if not isinstance(name, str) or not name or name.strip() != name or not isinstance(spec, Mapping):
            raise ValueError(f"{label} must map nonempty dataset names to objects")
    return cast(Mapping[str, Any], value)


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


def _eps_payload(value: Any) -> dict[str, Any]:
    keys = {"grid", "minimum_dataset_f1", "minimum_signature_weighted_f1"}
    eps = _object(value, label="model.eps", keys=keys)
    raw_grid = eps["grid"]
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError("model.eps.grid must be a nonempty list")
    grid = sorted(_unit_float(item, label="model.eps.grid item") for item in raw_grid)
    if len(grid) != len(set(grid)):
        raise ValueError("model.eps.grid values must be unique")
    return {
        "grid": grid,
        "minimum_dataset_f1": _unit_float(
            eps["minimum_dataset_f1"],
            label="model.eps.minimum_dataset_f1",
        ),
        "minimum_signature_weighted_f1": _unit_float(
            eps["minimum_signature_weighted_f1"],
            label="model.eps.minimum_signature_weighted_f1",
        ),
    }


def _resolve_path(raw_path: Any, *, base: Path, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} must be a nonempty path string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _snapshot_dataset(
    value: Any,
    *,
    base: Path,
    label: str,
    allowed_roles: Sequence[frozenset[str]],
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    if not isinstance(value, Mapping) or set(value) not in allowed_roles:
        choices = [sorted(roles) for roles in allowed_roles]
        raise ValueError(f"{label} has unsupported file roles; expected one of {choices}")
    paths = {
        role: _resolve_path(raw_path, base=base, label=f"{label}.{role}")
        for role, raw_path in cast(Mapping[str, Any], value).items()
    }
    snapshots = {role: {"path": str(path), "sha256": _sha256_file(path)} for role, path in sorted(paths.items())}
    return paths, snapshots


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


def _signature_ids(path: Path, *, context: str) -> set[str]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{context} must contain a nonempty object")
    return {_signature_id(value, context=context) for value in payload}


def _cluster_signature_ids(path: Path, *, context: str) -> set[str]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{context} must contain a nonempty object")
    signature_ids: set[str] = set()
    for cluster_id, spec in payload.items():
        if not isinstance(spec, Mapping) or not isinstance(spec.get("signature_ids"), list):
            raise ValueError(f"{context} cluster {cluster_id!r} must contain signature_ids")
        for member in spec["signature_ids"]:
            signature_id = _signature_id(member, context=f"{context} cluster {cluster_id!r}")
            if signature_id in signature_ids:
                raise ValueError(f"{context} contains a duplicate signature")
            signature_ids.add(signature_id)
    if not signature_ids:
        raise ValueError(f"{context} must contain signatures")
    return signature_ids


def _block_signature_ids(path: Path, *, context: str) -> set[str]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{context} must contain a nonempty object")
    signature_ids: set[str] = set()
    for block_id, members in payload.items():
        if not isinstance(members, list) or not members:
            raise ValueError(f"{context} block {block_id!r} must contain signatures")
        for member in members:
            signature_id = _signature_id(member, context=f"{context} block {block_id!r}")
            if signature_id in signature_ids:
                raise ValueError(f"{context} contains a duplicate signature")
            signature_ids.add(signature_id)
    return signature_ids


def _validate_baselines(value: Any, *, pairwise_names: set[str]) -> dict[str, Any]:
    keys = {
        "cluster_signature_weighted_b3_f1",
        "pairwise_aggregate",
        "pairwise_datasets",
        "predict_seconds_p50",
    }
    baselines = _object(value, label="evaluation.baselines", keys=keys)
    pairwise = baselines["pairwise_datasets"]
    if not isinstance(pairwise, Mapping) or set(pairwise) != pairwise_names:
        raise ValueError("evaluation.baselines.pairwise_datasets must match pairwise evaluation datasets")
    normalized_pairwise: dict[str, Any] = {}
    for name, metrics in {"aggregate": baselines["pairwise_aggregate"], **pairwise}.items():
        metric_map = _object(metrics, label=f"pairwise baseline {name}", keys={"auroc", "macro_f1"})
        normalized = {metric: _unit_float(raw, label=f"{name}.{metric}") for metric, raw in metric_map.items()}
        if name == "aggregate":
            aggregate = normalized
        else:
            normalized_pairwise[name] = normalized
    runtime = _number(baselines["predict_seconds_p50"], label="predict_seconds_p50")
    if runtime <= 0:
        raise ValueError("predict_seconds_p50 must be positive")
    return {
        "cluster_signature_weighted_b3_f1": _unit_float(
            baselines["cluster_signature_weighted_b3_f1"],
            label="cluster_signature_weighted_b3_f1",
        ),
        "pairwise_aggregate": aggregate,
        "pairwise_datasets": normalized_pairwise,
        "predict_seconds_p50": runtime,
    }


def _validate_gates(value: Any) -> dict[str, Any]:
    keys = {*_NORMATIVE_GATE_MAXIMA, "peak_rss_absolute_max_gb", "subblocking_maximum_size"}
    gates = _object(value, label="evaluation.gates", keys=keys)
    normalized: dict[str, Any] = {}
    for name, maximum in _NORMATIVE_GATE_MAXIMA.items():
        threshold = _number(gates[name], label=f"evaluation.gates.{name}")
        if threshold < 0 or threshold > maximum:
            raise ValueError(f"evaluation.gates.{name} must be in [0, {maximum}]")
        normalized[name] = threshold
    peak_rss = _number(gates["peak_rss_absolute_max_gb"], label="peak_rss_absolute_max_gb")
    if peak_rss <= 0:
        raise ValueError("peak_rss_absolute_max_gb must be positive")
    maximum_size = gates["subblocking_maximum_size"]
    if isinstance(maximum_size, bool) or not isinstance(maximum_size, int) or maximum_size <= 0:
        raise ValueError("subblocking_maximum_size must be a positive integer")
    normalized["peak_rss_absolute_max_gb"] = peak_rss
    normalized["subblocking_maximum_size"] = maximum_size
    return normalized


def _check_release_leakage(
    model_paths: Mapping[str, Mapping[str, Path]],
    pairwise_paths: Mapping[str, Mapping[str, Path]],
    cluster_paths: Mapping[str, Mapping[str, Path]],
) -> None:
    """Reject model/evaluation overlap without inspecting held-out answers."""

    pair_identities = {
        name: _heldout_pair_identities(files["pairs"], context=f"pair evaluation {name!r}")
        for name, files in pairwise_paths.items()
    }
    pair_signatures = {
        name: {signature_id for pair in pairs for signature_id in pair} for name, pairs in pair_identities.items()
    }
    block_signatures = {
        name: _block_signature_ids(files["blocks"], context=f"cluster evaluation {name!r}")
        for name, files in cluster_paths.items()
    }
    for name, files in model_paths.items():
        if set(files) == _FIXED_MODEL_ROLES:
            if name not in pair_identities:
                raise ValueError(f"Fixed-pair dataset {name!r} has no pairwise evaluation population")
            populations = {
                "train": _csv_pair_identities(files["train_pairs"], context=f"{name!r} train pairs"),
                "validation": _csv_pair_identities(files["val_pairs"], context=f"{name!r} validation pairs"),
                "test": pair_identities[name],
            }
            for left_name, right_name in (("train", "validation"), ("train", "test"), ("validation", "test")):
                overlap = populations[left_name] & populations[right_name]
                if overlap:
                    raise ValueError(
                        f"Dataset {name!r} {left_name}/{right_name} unordered pairs overlap: "
                        f"count={len(overlap)} sample={sorted(overlap)[:5]}"
                    )
            continue

        if name not in pair_signatures or name not in block_signatures:
            raise ValueError(f"Random-block dataset {name!r} needs pairwise and cluster evaluation populations")
        training = _signature_ids(files["signatures"], context=f"training signatures {name!r}")
        clustered = _cluster_signature_ids(files["clusters"], context=f"training clusters {name!r}")
        if training != clustered:
            raise ValueError(f"Training signatures and clusters disagree for dataset {name!r}")
        overlap = training & (pair_signatures[name] | block_signatures[name])
        if overlap:
            raise ValueError(
                f"Random-block dataset {name!r} contains test signatures: "
                f"count={len(overlap)} sample={sorted(overlap)[:5]}"
            )


def prepare_run(args: argparse.Namespace) -> dict[str, str]:
    """Validate one owner-authored release file and prepare a fresh run directory."""

    release_path = Path(args.release).resolve()
    if release_path.name != "release.json" or not release_path.is_file():
        raise ValueError("--release must name an existing release.json")
    run_dir = release_path.parent
    if {path.name for path in run_dir.iterdir()} != {"release.json"}:
        raise ValueError("A release run directory must initially contain only release.json")

    source = _object(_read_json(release_path), label="release", keys={"model", "evaluation"})
    model = _object(source["model"], label="model", keys={"datasets", "eps"})
    model_paths: dict[str, dict[str, Path]] = {}
    model_datasets: dict[str, Any] = {}
    for name, dataset in _named_objects(model["datasets"], label="model.datasets").items():
        paths, snapshots = _snapshot_dataset(
            dataset,
            base=run_dir,
            label=f"model.datasets.{name}",
            allowed_roles=(_RANDOM_MODEL_ROLES, _FIXED_MODEL_ROLES),
        )
        model_paths[name] = paths
        model_datasets[name] = snapshots
    model_plan = {"datasets": model_datasets, "eps": _eps_payload(model["eps"])}

    evaluation = _object(
        source["evaluation"],
        label="evaluation",
        keys={"pairwise", "cluster", "performance", "baselines", "gates"},
    )
    evaluation_paths: dict[str, dict[str, dict[str, Path]]] = {"pairwise": {}, "cluster": {}}
    evaluation_datasets: dict[str, dict[str, Any]] = {"pairwise": {}, "cluster": {}}
    for kind, roles in (("pairwise", _PAIR_ROLES), ("cluster", _CLUSTER_ROLES)):
        for name, dataset in _named_objects(evaluation[kind], label=f"evaluation.{kind}").items():
            paths, snapshots = _snapshot_dataset(
                dataset,
                base=run_dir,
                label=f"evaluation.{kind}.{name}",
                allowed_roles=(roles,),
            )
            evaluation_paths[kind][name] = paths
            evaluation_datasets[kind][name] = snapshots
    performance = _object(
        evaluation["performance"],
        label="evaluation.performance",
        keys={"arrow_root", "workload"},
    )
    if not isinstance(performance["workload"], Mapping) or not performance["workload"]:
        raise ValueError("evaluation.performance.workload must be a nonempty object")
    evaluation_plan = {
        **evaluation_datasets,
        "performance": {
            "arrow_root": str(
                _resolve_path(performance["arrow_root"], base=run_dir, label="evaluation.performance.arrow_root")
            ),
            "workload": dict(performance["workload"]),
        },
        "baselines": _validate_baselines(
            evaluation["baselines"],
            pairwise_names=set(evaluation_datasets["pairwise"]),
        ),
        "gates": _validate_gates(evaluation["gates"]),
    }
    _reject_nonfinite_json(evaluation_plan, label="evaluation")
    _check_release_leakage(
        model_paths,
        evaluation_paths["pairwise"],
        evaluation_paths["cluster"],
    )

    model_plan_path = run_dir / "model_plan.json"
    evaluation_plan_path = run_dir / "evaluation_plan.json"
    _write_fresh_json(model_plan_path, model_plan)
    _write_fresh_json(evaluation_plan_path, evaluation_plan)
    for name in ("stages", "reports", "final"):
        (run_dir / name).mkdir()
    return {
        "model_plan": str(model_plan_path),
        "evaluation_plan": str(evaluation_plan_path),
    }


def _plan_file_spec(value: Any, *, label: str) -> tuple[Path, str]:
    spec = _object(value, label=label, keys={"path", "sha256"})
    raw_path, digest = spec["path"], spec["sha256"]
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    return Path(raw_path), digest


def _load_plan_datasets(
    value: Any,
    *,
    label: str,
    roles: frozenset[str],
) -> dict[str, dict[str, tuple[Path, str]]]:
    datasets: dict[str, dict[str, tuple[Path, str]]] = {}
    for name, dataset in _named_objects(value, label=label).items():
        if set(dataset) != roles:
            raise ValueError(f"{label}.{name} must contain exactly {sorted(roles)}")
        datasets[name] = {role: _plan_file_spec(spec, label=f"{label}.{name}.{role}") for role, spec in dataset.items()}
    return datasets


def _load_evaluation_plan(path: Path) -> EvaluationPlan:
    payload = _object(
        _read_json(path),
        label="evaluation plan",
        keys={"pairwise", "cluster", "performance", "baselines", "gates"},
    )
    pairwise = _load_plan_datasets(payload["pairwise"], label="pairwise", roles=_PAIR_ROLES)
    cluster = _load_plan_datasets(payload["cluster"], label="cluster", roles=_CLUSTER_ROLES)
    performance = _object(payload["performance"], label="performance", keys={"arrow_root", "workload"})
    raw_arrow_root = performance["arrow_root"]
    if not isinstance(raw_arrow_root, str) or not Path(raw_arrow_root).is_absolute():
        raise ValueError("performance.arrow_root must be absolute")
    if not isinstance(performance["workload"], Mapping) or not performance["workload"]:
        raise ValueError("performance.workload must be a nonempty object")
    _reject_nonfinite_json(payload, label="evaluation plan")
    return EvaluationPlan(
        pairwise=pairwise,
        cluster=cluster,
        arrow_root=Path(raw_arrow_root),
        workload=cast(Mapping[str, Any], performance["workload"]),
        baselines=_validate_baselines(payload["baselines"], pairwise_names=set(pairwise)),
        gates=_validate_gates(payload["gates"]),
    )


def _verified_files(files: Mapping[str, tuple[Path, str]], *, label: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for role, (path, expected_digest) in files.items():
        observed = _sha256_file(path)
        if observed != expected_digest:
            raise ValueError(f"{label}.{role} changed after release preparation")
        paths[role] = path
    return paths


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


def _load_anddata(
    name: str,
    files: Mapping[str, Path],
    *,
    mode: str,
    name_counts_index: NameCountsIndex,
    name_tuples: frozenset[tuple[str, str]],
    random_seed: int = 1111,
) -> ANDData:
    return ANDData(
        signatures=str(files["signatures"]),
        papers=str(files["papers"]),
        clusters=str(files["clusters"]) if "clusters" in files else None,
        specter_embeddings=str(files["specter_embeddings"]),
        name=name,
        mode=mode,
        random_seed=random_seed,
        name_counts_index=name_counts_index,
        name_tuples=name_tuples,
        preprocess=True,
    )


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
        identity = _pair_identity(row["signature_id_1"], row["signature_id_2"], context=f"{name} row {row_number}")
        label = row["label"]
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"Pair label for {name!r} must be the JSON integer 0 or 1")
        if identity in identities:
            raise ValueError(f"Pair file for {name!r} contains a duplicate unordered pair")
        identities.add(identity)
        pairs.append((identity[0], identity[1], label))
    return pairs


def _pairwise_predictions(
    pairs: list[tuple[str, str, int | float]],
    dataset: ANDData,
    clusterer: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
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
    metrics, _ = pairwise_metrics(labels, main, nameless)
    return metrics, np.asarray(labels), np.asarray(main), np.asarray(nameless)


def _evaluation_model(args: argparse.Namespace) -> tuple[ProductionArtifactAuthority, Any]:
    artifacts = _load_release_artifacts(args)
    model = load_production_model(
        Path(args.model).resolve(),
        expected_artifact_hashes=artifacts.hashes,
    )
    return artifacts, model


def evaluate_pairs(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one complete bundle on the held-out pair populations."""

    output_path = Path(args.output_report).resolve()
    if output_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")
    artifacts, clusterer = _evaluation_model(args)
    plan = _load_evaluation_plan(Path(args.evaluation_plan).resolve())
    prepared = [
        (name, files, _labeled_pairs(files["pairs"], name=name))
        for name, specs in plan.pairwise.items()
        if (files := _verified_files(specs, label=f"pairwise.{name}"))
    ]

    dataset_reports: dict[str, Any] = {}
    all_labels: list[np.ndarray] = []
    all_main: list[np.ndarray] = []
    all_nameless: list[np.ndarray] = []
    for name, files, pairs in prepared:
        dataset = _load_anddata(
            name,
            files,
            mode="inference",
            name_counts_index=artifacts.name_counts_index,
            name_tuples=artifacts.name_tuples.pairs,
        )
        metrics, labels, main, nameless = _pairwise_predictions(pairs, dataset, clusterer, args)
        dataset_reports[name] = {"metrics": metrics}
        all_labels.append(labels)
        all_main.append(main)
        all_nameless.append(nameless)
    aggregate, _ = pairwise_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_main),
        np.concatenate(all_nameless),
    )
    report = {
        "schema_version": "s2and_pairwise_evaluation_report_v1",
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


def _normalized_blocks(path: Path, *, name: str) -> dict[str, list[str]]:
    blocks = _read_json(path)
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError(f"Cluster blocks for {name!r} must be a nonempty object")
    normalized: dict[str, list[str]] = {}
    signatures: set[str] = set()
    for block_id, members in blocks.items():
        if not isinstance(members, list) or not members:
            raise ValueError(f"Cluster block {block_id!r} for {name!r} must be nonempty")
        normalized[str(block_id)] = []
        for member in members:
            signature_id = _signature_id(member, context=f"cluster blocks {name!r}")
            if signature_id in signatures:
                raise ValueError(f"Cluster blocks for {name!r} contain duplicate signatures")
            signatures.add(signature_id)
            normalized[str(block_id)].append(signature_id)
    return normalized


def evaluate_clusters(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one complete bundle on the held-out cluster populations."""

    output_path = Path(args.output_report).resolve()
    if output_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")
    artifacts, clusterer = _evaluation_model(args)
    plan = _load_evaluation_plan(Path(args.evaluation_plan).resolve())
    clusterer.n_jobs = args.n_jobs
    dataset_reports: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for name, specs in plan.cluster.items():
        files = _verified_files(specs, label=f"cluster.{name}")
        blocks = _normalized_blocks(files["blocks"], name=name)
        dataset = _load_anddata(
            name,
            files,
            mode="train",
            name_counts_index=artifacts.name_counts_index,
            name_tuples=artifacts.name_tuples.pairs,
        )
        true_clusters = dataset.construct_cluster_to_signatures(blocks)
        predicted, _ = clusterer.predict(blocks, dataset)
        metrics[name] = _b3_report(true_clusters, predicted)
        dataset_reports[name] = {"metrics": metrics[name]}
    report = {
        "schema_version": "s2and_cluster_evaluation_report_v1",
        "datasets": dataset_reports,
        **_aggregate_b3(metrics),
    }
    _write_fresh_json(output_path, report)
    return report


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _training_config(bundle_dir: Path, *, model_plan_sha256: str) -> dict[str, Any]:
    payload = _read_json(bundle_dir / "reproducibility" / "pairwise_training_config.json")
    if not isinstance(payload, dict) or payload.get("training_scope") != "production_full":
        raise ValueError("EPS calibration requires a production_full training configuration")
    if payload.get("model_plan_sha256") != model_plan_sha256:
        raise ValueError("Source bundle was trained from a different model plan")
    if isinstance(payload.get("data_random_seed"), bool) or not isinstance(payload.get("data_random_seed"), int):
        raise ValueError("EPS calibration data_random_seed must be an integer")
    return payload


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
    config = _training_config(source_bundle, model_plan_sha256=plan.sha256)
    manifest_sha256 = _sha256_file(source_bundle / "manifest.json")
    return source_bundle, output_bundle, output_report, plan, config, manifest_sha256


def calibrate_eps(args: argparse.Namespace) -> dict[str, Any]:
    """Select EPS using each random-block validation population once."""

    source_bundle, output_bundle, output_report, plan, config, manifest_sha256 = _calibration_preflight(args)
    artifacts = _load_release_artifacts(args)
    clusterer = _load_pairwise_staging_model(source_bundle, expected_artifact_hashes=artifacts.hashes)
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
            dataset = _load_anddata(
                name,
                dataset_spec.files,
                mode="train",
                random_seed=int(config["data_random_seed"]),
                name_counts_index=artifacts.name_counts_index,
                name_tuples=artifacts.name_tuples.pairs,
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
                "block_count": len(val_blocks),
                "signature_count": sum(len(signatures) for signatures in val_blocks.values()),
            }
            for eps in plan.eps.grid:
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

    trials = [
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
    finalize_pairwise_eps(
        source_bundle_dir=source_bundle,
        output_bundle_dir=output_bundle,
        expected_manifest_sha256=manifest_sha256,
        expected_old_eps=original_eps,
        new_eps=selected["eps"],
        expected_artifact_hashes=artifacts.hashes,
    )
    report = {
        "schema_version": EPS_CALIBRATION_REPORT_SCHEMA,
        "model_plan_sha256": plan.sha256,
        "source_eps": original_eps,
        "selected_eps": selected["eps"],
        "validation_identities": identities,
        "trials": trials,
    }
    _write_fresh_json(output_report, report)
    return report


def _load_release_gate_inputs(report_dir: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for role, (filename, expected_schema) in _RELEASE_GATE_INPUTS.items():
        payload = _read_json(report_dir / filename)
        if not isinstance(payload, dict) or payload.get("schema_version") != expected_schema:
            raise ValueError(f"Release report {role!r} has the wrong schema")
        _reject_nonfinite_json(payload, label=f"Release report {role}")
        reports[role] = payload
    return reports


def _report_field(reports: Mapping[str, Any], role: str, *path: str) -> Any:
    value: Any = reports[role]
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Release report {role!r} is missing {'.'.join(path)!r}")
        value = value[part]
    return value


def _report_unit_metric(reports: Mapping[str, Any], role: str, *path: str) -> float:
    return _unit_float(
        _report_field(reports, role, *path),
        label=f"Release report metric {role}.{'.'.join(path)}",
    )


def _drop_check(check_id: str, candidate: float, baseline: Any, threshold: Any) -> dict[str, Any]:
    baseline_value = _unit_float(baseline, label=f"{check_id} baseline")
    threshold_value = _number(threshold, label=f"{check_id} threshold")
    observed = baseline_value - candidate
    return {
        "id": check_id,
        "candidate": candidate,
        "baseline": baseline_value,
        "observed_drop": observed,
        "threshold": threshold_value,
        "passed": observed <= threshold_value,
    }


def _maximum_check(check_id: str, candidate: float, threshold: Any) -> dict[str, Any]:
    threshold_value = _number(threshold, label=f"{check_id} threshold")
    return {
        "id": check_id,
        "candidate": candidate,
        "threshold": threshold_value,
        "passed": candidate <= threshold_value,
    }


def _validate_release_report_inputs(plan: EvaluationPlan, reports: Mapping[str, Any]) -> None:
    for role, expected in (
        ("pairwise_evaluation_report", set(plan.pairwise)),
        ("cluster_evaluation_report", set(plan.cluster)),
    ):
        datasets = _report_field(reports, role, "datasets")
        if not isinstance(datasets, Mapping) or set(datasets) != expected:
            raise ValueError(f"Release report {role}.datasets must exactly match the evaluation plan")
    workload = _report_field(reports, "performance_evaluation_report", "workload")
    if workload != plan.workload:
        raise ValueError("Performance workload does not match the evaluation plan")
    raw_arrow_root = _report_field(reports, "performance_evaluation_report", "arrow_root")
    if not isinstance(raw_arrow_root, str) or Path(raw_arrow_root).resolve() != plan.arrow_root.resolve():
        raise ValueError("Performance arrow_root does not match the evaluation plan")


def _release_checks(plan: EvaluationPlan, reports: Mapping[str, Any]) -> list[dict[str, Any]]:
    _validate_release_report_inputs(plan, reports)
    baselines, gates = plan.baselines, plan.gates
    drop_gates = [
        (
            "cluster.signature_weighted.b3_f1_drop",
            ("cluster_evaluation_report", "signature_weighted", "f1"),
            baselines["cluster_signature_weighted_b3_f1"],
            "cluster_signature_weighted_b3_f1_max_drop",
        ),
        (
            "pairwise.aggregate.auroc_drop",
            ("pairwise_evaluation_report", "aggregate", "auroc"),
            baselines["pairwise_aggregate"]["auroc"],
            "pairwise_aggregate_auroc_max_drop",
        ),
        (
            "pairwise.aggregate.macro_f1_drop",
            ("pairwise_evaluation_report", "aggregate", "macro_f1"),
            baselines["pairwise_aggregate"]["macro_f1"],
            "pairwise_aggregate_macro_f1_max_drop",
        ),
    ]
    for dataset in sorted(plan.pairwise):
        for metric, threshold_key in (
            ("auroc", "pairwise_dataset_auroc_max_drop"),
            ("macro_f1", "pairwise_dataset_macro_f1_max_drop"),
        ):
            drop_gates.append(
                (
                    f"pairwise.dataset.{dataset}.{metric}_drop",
                    ("pairwise_evaluation_report", "datasets", dataset, "metrics", metric),
                    baselines["pairwise_datasets"][dataset][metric],
                    threshold_key,
                )
            )
    checks = [
        _drop_check(
            check_id,
            _report_unit_metric(reports, *candidate_path),
            baseline,
            gates[threshold_key],
        )
        for check_id, candidate_path, baseline, threshold_key in drop_gates
    ]

    runtime = _number(
        _report_field(reports, "performance_evaluation_report", "summary", "predict_seconds", "p50"),
        label="performance runtime",
    )
    baseline_runtime = float(baselines["predict_seconds_p50"])
    runtime_ratio = runtime / baseline_runtime
    checks.append(
        {
            "id": "performance.runtime_ratio",
            "candidate": runtime,
            "baseline": baseline_runtime,
            "observed_ratio": runtime_ratio,
            "threshold": gates["runtime_max_ratio"],
            "passed": runtime_ratio <= gates["runtime_max_ratio"],
        }
    )
    peak_rss = _number(
        _report_field(reports, "performance_evaluation_report", "summary", "peak_rss_gb", "max"),
        label="performance peak RSS",
    )
    maximum_size = _report_field(reports, "subblocking_evaluation_report", "rust", "partition", "max_subblock_size")
    if runtime <= 0 or peak_rss <= 0:
        raise ValueError("Performance measurements must be positive")
    if isinstance(maximum_size, bool) or not isinstance(maximum_size, int) or maximum_size <= 0:
        raise ValueError("Subblocking maximum size must be a positive integer")
    checks.extend(
        (
            _maximum_check("performance.peak_rss_absolute_gb", peak_rss, gates["peak_rss_absolute_max_gb"]),
            _maximum_check("subblocking.maximum_size", maximum_size, gates["subblocking_maximum_size"]),
        )
    )
    component_recall = _report_unit_metric(
        reports,
        "subblocking_evaluation_report",
        "rust",
        "component_preservation",
        "component_pair_recall",
    )
    parity_exact_match = _report_field(reports, "parity_evaluation_report", "clusters_exact_match")
    if not isinstance(parity_exact_match, bool):
        raise ValueError("Parity clusters_exact_match must be boolean")
    checks.extend(
        (
            {
                "id": "subblocking.member_preservation",
                "candidate": component_recall,
                "threshold": 1.0,
                "passed": component_recall == 1.0,
            },
            {
                "id": "parity.clusters_exact_match",
                "candidate": parity_exact_match,
                "passed": parity_exact_match,
            },
        )
    )
    return sorted(checks, key=lambda check: check["id"])


def build_evaluation_report(args: argparse.Namespace) -> dict[str, Any]:
    """Apply every release gate to the five fixed component reports."""

    output_path = Path(args.output_report).resolve()
    if output_path.exists():
        raise FileExistsError(f"Evaluation report output already exists: {output_path}")
    plan = _load_evaluation_plan(Path(args.evaluation_plan).resolve())
    reports = _load_release_gate_inputs(Path(args.report_dir).resolve())
    checks = _release_checks(plan, reports)
    report = {
        "schema_version": EVALUATION_REPORT_SCHEMA,
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    _write_fresh_json(output_path, report)
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-run", help="Validate release.json and prepare a fresh run directory.")
    prepare.add_argument("--release", type=Path, required=True)
    prepare.set_defaults(handler=prepare_run)

    release = commands.add_parser("evaluate-release", help="Apply release gates to the five component reports.")
    release.add_argument("--evaluation-plan", type=Path, required=True)
    release.add_argument("--report-dir", type=Path, required=True)
    release.add_argument("--output-report", type=Path, required=True)
    release.set_defaults(handler=build_evaluation_report)

    calibrate = commands.add_parser("calibrate-eps", help="Calibrate EPS and write a fresh pairwise bundle.")
    calibrate.add_argument("--source-bundle", type=Path, required=True)
    calibrate.add_argument("--model-plan", type=Path, required=True)
    calibrate.add_argument("--output-bundle", type=Path, required=True)
    calibrate.add_argument("--output-report", type=Path, required=True)
    calibrate.add_argument("--name-counts-index-root", type=Path, required=True)
    calibrate.add_argument("--n-jobs", type=_positive_int, default=1)
    calibrate.add_argument("--total-ram-bytes", type=_positive_int)
    calibrate.set_defaults(handler=calibrate_eps)

    pair_evaluator = commands.add_parser("evaluate-pairs", help="Run the held-out pair evaluation.")
    pair_evaluator.add_argument("--total-ram-bytes", type=_positive_int)
    cluster_evaluator = commands.add_parser("evaluate-clusters", help="Run the held-out cluster evaluation.")
    for evaluator, handler in (
        (pair_evaluator, evaluate_pairs),
        (cluster_evaluator, evaluate_clusters),
    ):
        evaluator.add_argument("--model", type=Path, required=True)
        evaluator.add_argument("--evaluation-plan", type=Path, required=True)
        evaluator.add_argument("--name-counts-index-root", type=Path, required=True)
        evaluator.add_argument("--output-report", type=Path, required=True)
        evaluator.add_argument("--n-jobs", type=_positive_int, default=1)
        evaluator.set_defaults(handler=handler)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "evaluate-release" and result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

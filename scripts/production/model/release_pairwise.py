"""Release-only pairwise calibration, finalization, and one-shot evaluation."""

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
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._atomic_io import fsync_directory  # noqa: E402
from s2and._sha256 import is_lowercase_sha256  # noqa: E402
from s2and._sha256 import sha256_file as _sha256_file  # noqa: E402
from s2and.arrow_inputs import INFERENCE_ARROW_BUNDLE_SCHEMA_VERSION  # noqa: E402
from s2and.consts import _PACKAGE_DATA_DIR  # noqa: E402
from s2and.data import ANDData  # noqa: E402
from s2and.eval import b3_precision_recall_fscore, pairwise_probability_metrics  # noqa: E402
from s2and.featurizer import many_pairs_featurize  # noqa: E402
from s2and.incremental_linking.contracts import canonical_json_digest  # noqa: E402
from s2and.name_counts_index import NameCountsIndex  # noqa: E402
from s2and.name_tuple_artifact import load_packaged_name_tuple_artifact  # noqa: E402
from s2and.orcid_prefix_counts import load_canonical_orcid_prefix_counts  # noqa: E402
from s2and.production_bundle import finalize_pairwise_eps  # noqa: E402
from s2and.production_bundle_contract import PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION  # noqa: E402
from s2and.production_model import _load_pairwise_staging_model, load_production_model  # noqa: E402
from s2and.production_training_contract import (  # noqa: E402
    FLOAT_OFFICIAL_METRIC_KEYS,
    INTEGER_OFFICIAL_METRIC_KEYS,
    LINKER_EVALUATION_REPORT_SCHEMA,
    PAIRWISE_TRAINING_PLAN_SCHEMA_VERSION,
    SUPPORTED_OFFICIAL_METRIC_KEYS,
    ProductionArtifactAuthority,
)
from s2and.release_evidence import (  # noqa: E402
    RELEASE_EVIDENCE_ROLES,
    validate_release_evidence_manifest,
)

PAIR_MANIFEST_SCHEMA = "s2and_pairwise_test_manifest_v1"
CLUSTER_MANIFEST_SCHEMA = "s2and_cluster_test_manifest_v1"
TRAINING_INPUTS_MANIFEST_SCHEMA = "s2and_pairwise_training_inputs_manifest_v1"
TRAINING_PLAN_SCHEMA = PAIRWISE_TRAINING_PLAN_SCHEMA_VERSION
EPS_CALIBRATION_SPEC_SCHEMA = "s2and_eps_calibration_spec_v1"
EPS_CALIBRATION_REPORT_SCHEMA = "s2and_eps_calibration_report_v1"
RELEASE_SPEC_SCHEMA = "s2and_release_spec_v1"
EVALUATION_REPORT_SCHEMA = "s2and_release_evaluation_report_v1"

_RELEASE_EVIDENCE_SCHEMAS = {
    "cluster_evaluation_report": "s2and_cluster_evaluation_report_v1",
    "complete_model_manifest": PRODUCTION_MODEL_BUNDLE_SCHEMA_VERSION,
    "data_manifest": INFERENCE_ARROW_BUNDLE_SCHEMA_VERSION,
    "linker_evaluation_report": LINKER_EVALUATION_REPORT_SCHEMA,
    "pairwise_evaluation_report": "s2and_pairwise_evaluation_report_v1",
    "parity_evaluation_report": "s2and_parity_evaluation_report_v1",
    "performance_evaluation_report": "s2and_performance_evaluation_report_v1",
    "release_spec": RELEASE_SPEC_SCHEMA,
    "subblocking_evaluation_report": "s2and_subblocking_evaluation_report_v1",
}
assert set(_RELEASE_EVIDENCE_SCHEMAS) == RELEASE_EVIDENCE_ROLES


def _verified_sha256_file(path: Path, expected_sha256: Any, *, label: str) -> str:
    """Verify one lowercase SHA-256 expectation against one file."""

    if not is_lowercase_sha256(expected_sha256):
        raise ValueError(f"{label} SHA-256 must be a lowercase SHA-256 digest")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected_sha256} observed={observed_sha256}")
    return observed_sha256


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one new JSON report without replacing an existing file."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Report output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
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


def _verified_manifest(path: Path, expected_sha256: str, schema_version: str) -> dict[str, Any]:
    _verified_sha256_file(path, expected_sha256, label="Manifest")
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


def _reject_nonfinite_json(value: Any, *, label: str) -> None:
    """Reject non-finite JSON numbers recursively."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite_json(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_nonfinite_json(nested, label=f"{label}[{index}]")


def _release_number(value: Any, *, label: str) -> float:
    """Return a finite non-boolean release value."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return numeric


def _release_dataset_names(value: Any, *, label: str) -> list[str]:
    """Validate one nonempty, sorted, duplicate-free population name list."""

    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    names: list[str] = []
    for name in value:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError(f"{label} entries must be nonempty strings without surrounding whitespace")
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        if any(character not in allowed for character in name):
            raise ValueError(f"{label} contains unsupported dataset name {name!r}")
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError(f"{label} must be sorted and contain no duplicates")
    return names


def _validate_release_spec_payload(payload: Any) -> dict[str, Any]:
    """Validate the frozen release identity, inputs, baselines, and gates."""

    root_keys = {
        "baselines",
        "inputs",
        "populations",
        "release_identity",
        "schema_version",
        "thresholds",
    }
    if not isinstance(payload, dict) or set(payload) != root_keys:
        raise ValueError(f"Release spec must contain exactly {sorted(root_keys)}")
    _reject_nonfinite_json(payload, label="Release spec")
    if payload["schema_version"] != RELEASE_SPEC_SCHEMA:
        raise ValueError(f"Release spec schema_version must be {RELEASE_SPEC_SCHEMA!r}")

    release_identity = payload["release_identity"]
    release_keys = {"normalization_version", "python_version", "release_commit", "rust_version"}
    if not isinstance(release_identity, dict) or set(release_identity) != release_keys:
        raise ValueError(f"Release spec release_identity must contain exactly {sorted(release_keys)}")
    release_commit = release_identity["release_commit"]
    if (
        not isinstance(release_commit, str)
        or len(release_commit) != 40
        or any(character not in "0123456789abcdef" for character in release_commit)
    ):
        raise ValueError("Release spec release_commit must be a lowercase 40-character Git commit")
    if release_identity["normalization_version"] != "canonical_v2":
        raise ValueError("Release spec normalization_version must be 'canonical_v2'")
    for key in ("python_version", "rust_version"):
        value = release_identity[key]
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError(f"Release spec {key} must be a nonempty string without surrounding whitespace")

    populations = payload["populations"]
    population_keys = {
        "cluster_datasets",
        "cluster_manifest_sha256",
        "pairwise_datasets",
        "pairwise_manifest_sha256",
    }
    if not isinstance(populations, dict) or set(populations) != population_keys:
        raise ValueError(f"Release spec populations must contain exactly {sorted(population_keys)}")
    for key in ("cluster_datasets", "pairwise_datasets"):
        _release_dataset_names(populations[key], label=f"Release spec populations.{key}")
    for key in ("cluster_manifest_sha256", "pairwise_manifest_sha256"):
        if not is_lowercase_sha256(populations[key]):
            raise ValueError(f"Release spec populations.{key} must be a lowercase SHA-256 digest")

    inputs = payload["inputs"]
    input_keys = {
        "data_manifest_sha256",
        "parity_fixture_manifest_sha256",
        "performance_data_manifest_sha256",
        "performance_workload_sha256",
        "subblocking_input_manifest_sha256",
    }
    if not isinstance(inputs, dict) or set(inputs) != input_keys:
        raise ValueError(f"Release spec inputs must contain exactly {sorted(input_keys)}")
    for key, digest in inputs.items():
        if not is_lowercase_sha256(digest):
            raise ValueError(f"Release spec inputs.{key} must be a lowercase SHA-256 digest")

    baselines = payload["baselines"]
    baseline_keys = {
        "cluster_signature_weighted_b3_f1",
        "pairwise_aggregate",
        "pairwise_datasets",
        "predict_seconds_p50",
    }
    if not isinstance(baselines, dict) or set(baselines) != baseline_keys:
        raise ValueError(f"Release spec baselines must contain exactly {sorted(baseline_keys)}")
    pairwise_dataset_baselines = baselines["pairwise_datasets"]
    if not isinstance(pairwise_dataset_baselines, dict):
        raise ValueError("Release spec baseline pairwise_datasets must be an object")
    baseline_pairwise = {
        "aggregate": baselines["pairwise_aggregate"],
        **pairwise_dataset_baselines,
    }
    if set(pairwise_dataset_baselines) != set(populations["pairwise_datasets"]):
        raise ValueError("Release spec baseline pairwise datasets must match the frozen population")
    for label, metrics in baseline_pairwise.items():
        if not isinstance(metrics, dict) or set(metrics) != {"auroc", "macro_f1"}:
            raise ValueError(f"Release spec baseline {label} must contain exactly auroc and macro_f1")
        for metric, value in metrics.items():
            numeric = _release_number(value, label=f"Release spec baseline {label}.{metric}")
            if not 0 <= numeric <= 1:
                raise ValueError(f"Release spec baseline {label}.{metric} must be in [0, 1]")
    cluster_baseline = _release_number(
        baselines["cluster_signature_weighted_b3_f1"],
        label="Release spec baseline cluster_signature_weighted_b3_f1",
    )
    if not 0 <= cluster_baseline <= 1:
        raise ValueError("Release spec baseline cluster_signature_weighted_b3_f1 must be in [0, 1]")
    runtime_baseline = _release_number(
        baselines["predict_seconds_p50"],
        label="Release spec baseline predict_seconds_p50",
    )
    if runtime_baseline <= 0:
        raise ValueError("Release spec baseline predict_seconds_p50 must be positive")

    thresholds = payload["thresholds"]
    threshold_bounds = {
        "cluster_signature_weighted_b3_f1_max_drop": 0.005,
        "pairwise_aggregate_auroc_max_drop": 0.001,
        "pairwise_aggregate_macro_f1_max_drop": 0.005,
        "pairwise_dataset_auroc_max_drop": 0.001,
        "pairwise_dataset_macro_f1_max_drop": 0.005,
        "runtime_max_ratio": 1.1,
    }
    threshold_keys = {
        *threshold_bounds,
        "peak_rss_absolute_max_gb",
        "subblocking_maximum_size",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != threshold_keys:
        raise ValueError(f"Release spec thresholds must contain exactly {sorted(threshold_keys)}")
    for key, maximum in threshold_bounds.items():
        value = _release_number(thresholds[key], label=f"Release spec thresholds.{key}")
        if value < 0 or value > maximum:
            raise ValueError(f"Release spec threshold {key!r} weakens the normative maximum {maximum}")
        thresholds[key] = value
    peak_rss = _release_number(
        thresholds["peak_rss_absolute_max_gb"],
        label="Release spec thresholds.peak_rss_absolute_max_gb",
    )
    if peak_rss <= 0:
        raise ValueError("Release spec peak_rss_absolute_max_gb must be positive")
    maximum_size = thresholds["subblocking_maximum_size"]
    if isinstance(maximum_size, bool) or not isinstance(maximum_size, int) or maximum_size <= 0:
        raise ValueError("Release spec subblocking_maximum_size must be a positive integer")
    thresholds["peak_rss_absolute_max_gb"] = peak_rss
    return payload


def _verified_release_spec(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and strictly validate one digest-bound release spec."""

    _verified_sha256_file(path, expected_sha256, label="Release-spec")
    return _validate_release_spec_payload(_read_json(path))


def _verified_complete_model_path(path: Path, expected_manifest_sha256: str) -> Path:
    """Validate the expected complete-bundle manifest identity."""

    model_path = path.resolve()
    manifest_path = model_path / "manifest.json"
    _verified_sha256_file(manifest_path, expected_manifest_sha256, label="Model-manifest")
    return model_path


def _calibration_unit_float(value: Any, *, label: str) -> float:
    """Return one finite unit-interval calibration value."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number in [0, 1], got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1], got {value!r}")
    return numeric


def _verified_calibration_spec(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load and normalize the fixed digest-bound EPS calibration spec."""

    _verified_sha256_file(path, expected_sha256, label="Calibration-spec")
    payload = _read_json(path)
    expected_keys = {
        "aggregation",
        "eps_grid",
        "floors",
        "objective",
        "schema_version",
        "source_manifest_sha256",
        "tie_break",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"Calibration spec must contain exactly {sorted(expected_keys)}")
    if payload["schema_version"] != EPS_CALIBRATION_SPEC_SCHEMA:
        raise ValueError(f"Calibration spec schema_version must be {EPS_CALIBRATION_SPEC_SCHEMA!r}")
    if payload["objective"] != "signature_weighted_b3_f1":
        raise ValueError("Calibration spec objective must be 'signature_weighted_b3_f1'")
    if payload["aggregation"] != "dataset_macro_and_signature_weighted":
        raise ValueError("Calibration spec aggregation must be 'dataset_macro_and_signature_weighted'")
    if payload["tie_break"] != "smallest_eps":
        raise ValueError("Calibration spec tie_break must be 'smallest_eps'")
    source_manifest_sha256 = payload["source_manifest_sha256"]
    if not is_lowercase_sha256(source_manifest_sha256):
        raise ValueError("Calibration spec source_manifest_sha256 must be a lowercase SHA-256 digest")
    raw_grid = payload["eps_grid"]
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError("Calibration spec eps_grid must be a nonempty list")
    eps_grid = [_calibration_unit_float(value, label="Calibration EPS") for value in raw_grid]
    if len(eps_grid) != len(set(eps_grid)):
        raise ValueError("Calibration spec eps_grid values must be unique")
    floors = payload["floors"]
    expected_floor_keys = {"minimum_dataset_f1", "minimum_signature_weighted_f1"}
    if not isinstance(floors, Mapping) or set(floors) != expected_floor_keys:
        raise ValueError(f"Calibration spec floors must contain exactly {sorted(expected_floor_keys)}")
    return {
        **payload,
        "eps_grid": sorted(eps_grid),
        "floors": {
            name: _calibration_unit_float(floors[name], label=f"Calibration floor {name}")
            for name in sorted(expected_floor_keys)
        },
    }


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
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        _verified_sha256_file(
            path,
            expected_sha256,
            label=f"Dataset {spec.get('name')!r} {role}",
        )
        resolved[role] = path
    return resolved


def _signature_id(value: Any, *, context: str) -> str:
    """Return one exact nonempty signature id."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} signature id must be a nonempty string")
    if value != value.strip():
        raise ValueError(f"{context} signature id must not have surrounding whitespace")
    return value


def _pair_identity(left: Any, right: Any, *, context: str) -> tuple[str, str]:
    """Return one non-self unordered signature pair."""

    left_id = _signature_id(left, context=context)
    right_id = _signature_id(right, context=context)
    if left_id == right_id:
        raise ValueError(f"{context} contains a self-pair for signature {left_id!r}")
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def _csv_pair_identities(path: Path, *, context: str) -> set[tuple[str, str]]:
    """Validate one fixed-pair CSV and return its unordered identities."""

    identities: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        expected_columns = ["signature_id_1", "signature_id_2", "label"]
        if reader.fieldnames != expected_columns:
            raise ValueError(f"{context} must contain exactly {expected_columns}")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"{context} row {row_number} must contain exactly {expected_columns}")
            label = row["label"]
            if label is None:
                raise ValueError(f"{context} row {row_number} label must be 0, 1, NO, or YES")
            if label != label.strip():
                raise ValueError(f"{context} row {row_number} label must not have surrounding whitespace")
            if label not in {"0", "1", "NO", "YES"}:
                raise ValueError(f"{context} row {row_number} label must be 0, 1, NO, or YES")
            identity = _pair_identity(
                row["signature_id_1"],
                row["signature_id_2"],
                context=f"{context} row {row_number}",
            )
            if identity in identities:
                raise ValueError(f"{context} contains duplicate unordered pair {identity!r}")
            identities.add(identity)
    if not identities:
        raise ValueError(f"{context} must contain at least one pair")
    return identities


def _json_pair_identities(path: Path, *, context: str) -> set[tuple[str, str]]:
    """Validate one sealed pair JSON member and return its identities."""

    rows = _read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{context} must contain a nonempty JSON list")
    identities: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or set(row) != {"signature_id_1", "signature_id_2", "label"}:
            raise ValueError(f"{context} row {row_number} must contain exactly signature_id_1, signature_id_2, label")
        row = cast(Mapping[str, Any], row)
        label = row["label"]
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError(f"{context} row {row_number} label must be the JSON integer 0 or 1")
        identity = _pair_identity(
            row["signature_id_1"],
            row["signature_id_2"],
            context=f"{context} row {row_number}",
        )
        if identity in identities:
            raise ValueError(f"{context} contains duplicate unordered pair {identity!r}")
        identities.add(identity)
    return identities


def _signature_ids(path: Path, *, context: str) -> set[str]:
    """Return the nonempty signature ids declared by one signatures JSON."""

    payload = _read_json(path)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{context} must contain a nonempty JSON object")
    return {_signature_id(value, context=context) for value in payload}


def _cluster_signature_ids(path: Path, *, context: str) -> set[str]:
    """Validate cluster membership and return its unique signature ids."""

    payload = _read_json(path)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{context} must contain a nonempty JSON object")
    signature_ids: set[str] = set()
    for cluster_id, spec in payload.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"{context} cluster {cluster_id!r} must be an object")
        members = spec.get("signature_ids")
        if not isinstance(members, list) or not members:
            raise ValueError(f"{context} cluster {cluster_id!r} must contain a nonempty signature_ids list")
        for member in members:
            signature_id = _signature_id(member, context=f"{context} cluster {cluster_id!r}")
            if signature_id in signature_ids:
                raise ValueError(f"{context} contains duplicate signature {signature_id!r}")
            signature_ids.add(signature_id)
    return signature_ids


def _block_signature_ids(path: Path, *, context: str) -> set[str]:
    """Validate a sealed block mapping and return its unique signature ids."""

    payload = _read_json(path)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{context} must contain a nonempty JSON object")
    signature_ids: set[str] = set()
    for block_id, members in payload.items():
        if not isinstance(members, list) or not members:
            raise ValueError(f"{context} block {block_id!r} must contain a nonempty signature list")
        for member in members:
            signature_id = _signature_id(member, context=f"{context} block {block_id!r}")
            if signature_id in signature_ids:
                raise ValueError(f"{context} contains duplicate signature {signature_id!r}")
            signature_ids.add(signature_id)
    return signature_ids


def _sealed_manifest_binding(manifest: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    """Return digest-only provenance for a sealed evaluation manifest."""

    return {
        "manifest_sha256": manifest_sha256,
        "members": {
            str(spec["name"]): {
                str(role): str(file_spec["sha256"]) for role, file_spec in sorted(spec["files"].items())
            }
            for spec in manifest["datasets"]
        },
    }


def _sealed_pair_identities(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, set[tuple[str, str]]]:
    """Validate sealed pair members and return identities by dataset."""

    roles = ("signatures", "papers", "specter_embeddings", "pairs")
    return {
        str(spec["name"]): _json_pair_identities(
            _resolved_dataset_files(manifest_path, spec, roles)["pairs"],
            context=f"sealed pair test {spec['name']!r}",
        )
        for spec in manifest["datasets"]
    }


def _sealed_cluster_identities(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, set[str]]:
    """Validate sealed block members and return identities by dataset."""

    roles = ("signatures", "papers", "specter_embeddings", "clusters", "blocks")
    return {
        str(spec["name"]): _block_signature_ids(
            _resolved_dataset_files(manifest_path, spec, roles)["blocks"],
            context=f"sealed cluster blocks {spec['name']!r}",
        )
        for spec in manifest["datasets"]
    }


def _sealed_manifest(
    source_manifest_path: Path,
    ref: Any,
    *,
    schema: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    """Resolve and validate one sealed manifest reference."""

    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{label} reference must contain exactly path and sha256")
    raw_path = ref["path"]
    digest = ref["sha256"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path must be a nonempty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = source_manifest_path.parent / path
    path = path.resolve()
    return path, _verified_manifest(path, digest, schema), digest


def preflight_training_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Validate release training/test isolation and emit a test-path-free plan."""

    source_manifest_path = Path(args.manifest).resolve()
    source = _verified_manifest(
        source_manifest_path,
        args.expected_manifest_sha256,
        TRAINING_INPUTS_MANIFEST_SCHEMA,
    )
    if set(source) != {"schema_version", "datasets", "sealed_test_manifests"}:
        raise ValueError("Training-input manifest must contain exactly schema_version, datasets, sealed_test_manifests")
    sealed_refs = source["sealed_test_manifests"]
    if not isinstance(sealed_refs, Mapping) or set(sealed_refs) != {"pairwise", "cluster"}:
        raise ValueError("sealed_test_manifests must contain exactly pairwise and cluster")

    pair_path, pair_manifest, pair_sha256 = _sealed_manifest(
        source_manifest_path,
        sealed_refs["pairwise"],
        schema=PAIR_MANIFEST_SCHEMA,
        label="Pairwise test manifest",
    )
    cluster_path, cluster_manifest, cluster_sha256 = _sealed_manifest(
        source_manifest_path,
        sealed_refs["cluster"],
        schema=CLUSTER_MANIFEST_SCHEMA,
        label="Cluster test manifest",
    )
    pair_tests = _sealed_pair_identities(pair_path, pair_manifest)
    cluster_tests = _sealed_cluster_identities(cluster_path, cluster_manifest)
    pair_test_signatures = {
        name: {signature_id for pair in identities for signature_id in pair} for name, identities in pair_tests.items()
    }

    plan_datasets: list[dict[str, Any]] = []
    common_roles = {"papers", "signatures", "specter_embeddings"}
    for spec in source["datasets"]:
        if not isinstance(spec, Mapping) or set(spec) != {"name", "split_mode", "files"}:
            raise ValueError("Each training dataset must contain exactly name, split_mode, files")
        name = str(spec["name"])
        split_mode = spec["split_mode"]
        if split_mode == "random_blocks":
            roles = common_roles | {"clusters"}
        elif split_mode == "fixed_pairs":
            roles = common_roles | {"train_pairs", "val_pairs"}
        else:
            raise ValueError(f"Training dataset {name!r} has unknown split_mode {split_mode!r}")
        files = _resolved_dataset_files(source_manifest_path, spec, sorted(roles))

        if split_mode == "fixed_pairs":
            train_pairs = _csv_pair_identities(files["train_pairs"], context=f"{name!r} train pairs")
            val_pairs = _csv_pair_identities(files["val_pairs"], context=f"{name!r} validation pairs")
            test_pairs = pair_tests.get(name)
            if test_pairs is None:
                raise ValueError(f"Fixed-pair dataset {name!r} is absent from the sealed pair test manifest")
            for left_name, left, right_name, right in (
                ("train", train_pairs, "validation", val_pairs),
                ("train", train_pairs, "test", test_pairs),
                ("validation", val_pairs, "test", test_pairs),
            ):
                overlap = left & right
                if overlap:
                    raise ValueError(
                        f"Dataset {name!r} {left_name}/{right_name} unordered pairs overlap: "
                        f"count={len(overlap)} sample={sorted(overlap)[:5]}"
                    )
        else:
            training_signatures = _signature_ids(files["signatures"], context=f"training signatures {name!r}")
            clustered_signatures = _cluster_signature_ids(files["clusters"], context=f"training clusters {name!r}")
            if training_signatures != clustered_signatures:
                raise ValueError(f"Training signatures and clusters disagree for dataset {name!r}")
            pair_signatures = pair_test_signatures.get(name)
            if pair_signatures is None:
                raise ValueError(f"Random-block dataset {name!r} is absent from the sealed pair test manifest")
            cluster_signatures = cluster_tests.get(name)
            if cluster_signatures is None:
                raise ValueError(f"Random-block dataset {name!r} is absent from the sealed cluster test manifest")
            test_signatures = pair_signatures | cluster_signatures
            overlap = training_signatures & test_signatures
            if overlap:
                raise ValueError(
                    f"Random-block dataset {name!r} contains sealed test signatures: "
                    f"count={len(overlap)} sample={sorted(overlap)[:5]}"
                )

        plan_datasets.append(
            {
                "name": name,
                "split_mode": split_mode,
                "files": {
                    role: {"path": str(files[role]), "sha256": str(spec["files"][role]["sha256"])}
                    for role in sorted(roles)
                },
            }
        )

    plan = {
        "schema_version": TRAINING_PLAN_SCHEMA,
        "source_manifest_sha256": args.expected_manifest_sha256,
        "datasets": plan_datasets,
        "sealed_test_manifests": {
            "pairwise": _sealed_manifest_binding(pair_manifest, pair_sha256),
            "cluster": _sealed_manifest_binding(cluster_manifest, cluster_sha256),
        },
    }
    output_path = Path(args.output_plan).resolve()
    _write_fresh_json(output_path, plan)
    return {**plan, "plan_sha256": _sha256_file(output_path)}


def _validated_evaluation_bindings(
    args: argparse.Namespace,
    *,
    manifest_schema: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Validate release-spec, population, and model identities before model loading."""

    spec_path = Path(args.release_spec).resolve()
    release_spec = _verified_release_spec(spec_path, args.expected_release_spec_sha256)
    manifest_path = Path(args.manifest).resolve()
    manifest = _verified_manifest(manifest_path, args.expected_manifest_sha256, manifest_schema)
    population_binding = {
        PAIR_MANIFEST_SCHEMA: ("pairwise_manifest_sha256", "pairwise_datasets"),
        CLUSTER_MANIFEST_SCHEMA: ("cluster_manifest_sha256", "cluster_datasets"),
    }[manifest_schema]
    digest_field, datasets_field = population_binding
    populations = release_spec["populations"]
    if populations[digest_field] != args.expected_manifest_sha256:
        raise ValueError(f"Release spec populations.{digest_field} does not match --expected-manifest-sha256")
    manifest_datasets = sorted(spec["name"] for spec in manifest["datasets"])
    if populations[datasets_field] != manifest_datasets:
        raise ValueError(f"Release spec populations.{datasets_field} does not match evaluation manifest datasets")
    model_path = _verified_complete_model_path(Path(args.model), args.expected_model_manifest_sha256)
    return model_path, manifest_path, manifest


def _validated_probability_vector(values: np.ndarray, *, label: str) -> np.ndarray:
    """Return a flat finite probability vector bounded to the unit interval."""

    probabilities = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{label} probabilities must all be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(f"{label} probabilities must all be in [0, 1]")
    return probabilities


def _validated_unit_interval_metrics(metrics: Mapping[str, Any], *, label: str) -> None:
    """Reject empty counts and non-finite or out-of-range metric values."""

    for name, value in metrics.items():
        if name in {"rows", "signature_count"}:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(f"{label} {name} must be a positive integer, got {value!r}")
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise RuntimeError(f"{label} {name} must be finite and in [0, 1], got {value!r}")


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
    _validated_probability_vector(probabilities, label="averaged")
    return metrics, probabilities


def _load_sealed_anddata(
    name: str,
    files: Mapping[str, Path],
    *,
    mode: str,
    name_counts_index: NameCountsIndex,
    name_tuples: frozenset[tuple[str, str]],
    random_seed: int = 1111,
) -> ANDData:
    """Load release JSON data with the explicitly selected global artifacts."""

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


def _load_release_artifacts(
    args: argparse.Namespace,
) -> ProductionArtifactAuthority:
    """Open the one external and two packaged global artifacts once."""

    authority = ProductionArtifactAuthority(
        name_counts_index=NameCountsIndex.open(Path(args.name_counts_index_root)),
        name_tuples=load_packaged_name_tuple_artifact(),
        orcid_prefix_counts=load_canonical_orcid_prefix_counts(_PACKAGE_DATA_DIR),
    )
    if authority.orcid_prefix_counts.name_tuples_sha256 != authority.name_tuples.data_sha256:
        raise ValueError("ORCID prefix counts were generated from a different name-tuple artifact")
    return authority


def _pairwise_predictions(
    pairs: list[tuple[str, str, int | float]],
    dataset: ANDData,
    clusterer: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    """Score one frozen pair population with one already-loaded model."""

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


def evaluate_pairs(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one complete bundle on one sealed pair population."""

    output_path = Path(args.output_report)
    if output_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")
    model_path, manifest_path, manifest = _validated_evaluation_bindings(
        args,
        manifest_schema=PAIR_MANIFEST_SCHEMA,
    )
    prepared: list[tuple[str, dict[str, Path], list[tuple[str, str, int | float]]]] = []
    roles = ("signatures", "papers", "specter_embeddings", "pairs")
    for spec in manifest["datasets"]:
        files = _resolved_dataset_files(manifest_path, spec, roles)
        name = str(spec["name"])
        raw_pairs = _read_json(files["pairs"])
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError(f"Pair file for {name!r} must contain a nonempty JSON list")
        pairs: list[tuple[str, str, int | float]] = []
        for row in raw_pairs:
            if not isinstance(row, Mapping):
                raise ValueError(f"Pair row for {name!r} must be an object")
            left = str(row["signature_id_1"])
            right = str(row["signature_id_2"])
            label = row.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
                raise ValueError(f"Pair label for {name!r} must be the JSON integer 0 or 1")
            pairs.append((left, right, label))
        prepared.append((name, files, pairs))

    artifacts = _load_release_artifacts(args)
    artifact_hashes = artifacts.hashes
    clusterer = load_production_model(model_path, expected_artifact_hashes=artifact_hashes)
    dataset_reports: dict[str, Any] = {}
    all_labels: list[np.ndarray] = []
    all_main: list[np.ndarray] = []
    all_nameless: list[np.ndarray] = []
    for name, files, pairs in prepared:
        dataset = _load_sealed_anddata(
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
        "release_spec_sha256": args.expected_release_spec_sha256,
        "model_manifest_sha256": args.expected_model_manifest_sha256,
        **artifact_hashes,
        "population_manifest_sha256": args.expected_manifest_sha256,
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
    for name in names:
        _validated_unit_interval_metrics(dataset_metrics[name], label=f"B3 dataset {name!r}")
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


def _training_config(bundle_dir: Path) -> dict[str, Any]:
    path = bundle_dir / "reproducibility" / "pairwise_training_config.json"
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("training_scope") != "production_full":
        raise ValueError("EPS calibration requires production_full pairwise training provenance")
    if not isinstance(payload.get("dataset_inputs"), Mapping) or not payload["dataset_inputs"]:
        raise ValueError("EPS calibration requires nonempty dataset_inputs")
    if isinstance(payload.get("data_random_seed"), bool) or not isinstance(payload["data_random_seed"], int):
        raise ValueError("EPS calibration data_random_seed must be an integer")
    return payload


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path contains the other."""

    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _calibration_preflight(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, dict[str, Any], str]:
    """Validate fresh paths, the frozen spec, and its source binding."""

    source_bundle = Path(args.source_bundle).resolve()
    spec_path = Path(args.spec).resolve()
    output_bundle = Path(args.output_bundle).resolve()
    output_report = Path(args.output_report).resolve()
    if not source_bundle.is_dir():
        raise FileNotFoundError(f"Calibration source bundle does not exist: {source_bundle}")
    if not spec_path.is_file():
        raise FileNotFoundError(f"Calibration spec does not exist: {spec_path}")
    if output_bundle.exists():
        raise FileExistsError(f"Calibration output bundle already exists: {output_bundle}")
    if output_report.exists():
        raise FileExistsError(f"Calibration output report already exists: {output_report}")
    if _paths_overlap(source_bundle, output_bundle):
        raise ValueError("Calibration source and output bundle paths must not overlap")
    if output_report.is_relative_to(source_bundle) or _paths_overlap(output_report, output_bundle):
        raise ValueError("Calibration output report must be outside the source and output bundles")

    spec = _verified_calibration_spec(spec_path, args.expected_spec_sha256)
    source_manifest_sha256 = _verified_sha256_file(
        source_bundle / "manifest.json",
        spec["source_manifest_sha256"],
        label="Calibration source manifest",
    )
    return source_bundle, output_bundle, output_report, spec, source_manifest_sha256


def _resolved_calibration_inputs(
    bundle_dir: Path,
    config: Mapping[str, Any],
) -> list[tuple[str, dict[str, Path]]]:
    """Resolve every clustered validation input and verify its frozen digest."""

    resolved_inputs: list[tuple[str, dict[str, Path]]] = []
    roles = {"clusters", "papers", "signatures", "specter_embeddings"}
    for name in sorted(config["dataset_inputs"]):
        dataset_spec = config["dataset_inputs"][name]
        if not isinstance(dataset_spec, Mapping) or set(dataset_spec) != {"files", "split_mode"}:
            raise ValueError(f"Calibration dataset {name!r} must contain exactly files and split_mode")
        split_mode = dataset_spec["split_mode"]
        if split_mode not in {"fixed_pairs", "random_blocks"}:
            raise ValueError(f"Calibration dataset {name!r} has unknown split_mode {split_mode!r}")
        if split_mode == "fixed_pairs":
            continue
        raw_files = dataset_spec["files"]
        if not isinstance(raw_files, Mapping) or set(raw_files) != roles:
            raise ValueError(f"Calibration dataset {name!r} must declare exact file roles {sorted(roles)}")
        files: dict[str, Path] = {}
        for role in sorted(roles):
            source = raw_files[role]
            if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
                raise ValueError(f"Calibration input {name}:{role} must contain exactly path and sha256")
            raw_path = source["path"]
            expected_sha256 = source["sha256"]
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"Calibration input {name}:{role} path must be a nonempty string")
            path = Path(raw_path)
            if not path.is_absolute():
                path = bundle_dir / path
            path = path.resolve()
            _verified_sha256_file(
                path,
                expected_sha256,
                label=f"Calibration input drift for {name}:{role}",
            )
            files[role] = path
        resolved_inputs.append((str(name), files))
    if not resolved_inputs:
        raise ValueError("EPS calibration found no clustered validation datasets")
    return resolved_inputs


def calibrate_eps(args: argparse.Namespace) -> dict[str, Any]:
    """Calibrate EPS from one frozen spec and always write a fresh bundle.

    Fresh outputs, the spec and source bindings, the source bundle, and all
    validation-input digests are checked before any distance matrix is built,
    so an operator error cannot surface only after hours of featurization.

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

    source_bundle, output_bundle, output_report, spec, source_manifest_sha256 = _calibration_preflight(args)
    artifacts = _load_release_artifacts(args)
    artifact_hashes = artifacts.hashes
    clusterer = _load_pairwise_staging_model(
        source_bundle,
        expected_artifact_hashes=artifact_hashes,
    )
    clusterer.n_jobs = args.n_jobs
    config = _training_config(source_bundle)
    resolved_inputs = _resolved_calibration_inputs(source_bundle, config)

    identities: dict[str, Any] = {}
    metrics_by_eps: dict[float, dict[str, Any]] = {eps: {} for eps in spec["eps_grid"]}
    original_eps = _calibration_unit_float(clusterer.cluster_model.eps, label="Source EPS")
    try:
        for name, files in resolved_inputs:
            dataset = _load_sealed_anddata(
                name,
                files,
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
                "digest": canonical_json_digest(val_blocks),
                "signature_count": sum(len(signatures) for signatures in val_blocks.values()),
            }
            for eps in spec["eps_grid"]:
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

    trials: list[dict[str, Any]] = []
    for eps in spec["eps_grid"]:
        trial = {"eps": eps, "datasets": metrics_by_eps[eps], **_aggregate_b3(metrics_by_eps[eps])}
        trials.append(trial)
    eligible = [
        trial
        for trial in trials
        if trial["signature_weighted"]["f1"] >= spec["floors"]["minimum_signature_weighted_f1"]
        and all(metrics["f1"] >= spec["floors"]["minimum_dataset_f1"] for metrics in trial["datasets"].values())
    ]
    if not eligible:
        raise RuntimeError("No EPS calibration trial met all frozen quality floors")
    selected = max(eligible, key=lambda trial: (trial["signature_weighted"]["f1"], -trial["eps"]))

    finalize_pairwise_eps(
        source_bundle_dir=source_bundle,
        output_bundle_dir=output_bundle,
        expected_manifest_sha256=source_manifest_sha256,
        expected_old_eps=original_eps,
        new_eps=selected["eps"],
        expected_artifact_hashes=artifact_hashes,
    )
    output_manifest_sha256 = _sha256_file(output_bundle / "manifest.json")
    report = {
        "schema_version": EPS_CALIBRATION_REPORT_SCHEMA,
        "calibration_spec_sha256": args.expected_spec_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "output_manifest_sha256": output_manifest_sha256,
        **artifact_hashes,
        "source_eps": original_eps,
        "selected_eps": selected["eps"],
        "validation_identities": identities,
        "trials": trials,
    }
    _write_fresh_json(output_report, report)
    return report


def evaluate_clusters(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one complete bundle on sealed cluster identities."""

    output_path = Path(args.output_report)
    if output_path.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_path}")
    model_path, manifest_path, manifest = _validated_evaluation_bindings(
        args,
        manifest_schema=CLUSTER_MANIFEST_SCHEMA,
    )
    roles = ("signatures", "papers", "specter_embeddings", "clusters", "blocks")
    prepared: list[tuple[str, dict[str, Path], dict[str, list[str]]]] = []
    for spec in manifest["datasets"]:
        files = _resolved_dataset_files(manifest_path, spec, roles)
        name = str(spec["name"])
        blocks = _read_json(files["blocks"])
        if not isinstance(blocks, dict) or not blocks:
            raise ValueError(f"Cluster blocks for {name!r} must be a nonempty object")
        if any(not isinstance(values, list) or not values for values in blocks.values()):
            raise ValueError(f"Cluster blocks for {name!r} must map block ids to nonempty signature lists")
        normalized_blocks = {str(key): [str(value) for value in values] for key, values in blocks.items()}
        flattened = [signature for values in normalized_blocks.values() for signature in values]
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"Cluster blocks for {name!r} contain duplicate signatures")
        prepared.append((name, files, normalized_blocks))

    artifacts = _load_release_artifacts(args)
    artifact_hashes = artifacts.hashes
    clusterer = load_production_model(model_path, expected_artifact_hashes=artifact_hashes)
    clusterer.n_jobs = args.n_jobs
    dataset_reports: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for name, files, normalized_blocks in prepared:
        dataset = _load_sealed_anddata(
            name,
            files,
            mode="train",
            name_counts_index=artifacts.name_counts_index,
            name_tuples=artifacts.name_tuples.pairs,
        )
        true_clusters = dataset.construct_cluster_to_signatures(normalized_blocks)
        predicted, _ = clusterer.predict(normalized_blocks, dataset)
        metrics[name] = _b3_report(true_clusters, predicted)
        dataset_reports[name] = {"metrics": metrics[name]}
    report = {
        "schema_version": "s2and_cluster_evaluation_report_v1",
        "release_spec_sha256": args.expected_release_spec_sha256,
        "model_manifest_sha256": args.expected_model_manifest_sha256,
        **artifact_hashes,
        "population_manifest_sha256": args.expected_manifest_sha256,
        "datasets": dataset_reports,
        **_aggregate_b3(metrics),
    }
    _write_fresh_json(output_path, report)
    return report


def _load_release_evidence(
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load every locally verified member of one digest-bound evidence manifest."""

    _verified_sha256_file(manifest_path, expected_manifest_sha256, label="Release-evidence manifest")
    members = validate_release_evidence_manifest(_read_json(manifest_path), manifest_path)
    manifest_root = manifest_path.resolve().parent
    evidence: dict[str, Any] = {}
    sha256: dict[str, str] = {}
    for role, member in members.items():
        relative_path = PurePosixPath(member["path"])
        path = (manifest_root / Path(*relative_path.parts)).resolve()
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Release evidence {role!r} must contain a JSON object")
        _reject_nonfinite_json(payload, label=f"Release evidence {role!r}")
        expected_schema = _RELEASE_EVIDENCE_SCHEMAS[role]
        schema_field = "schema" if role == "data_manifest" else "schema_version"
        observed_schema = payload.get(schema_field)
        if observed_schema != expected_schema:
            raise ValueError(
                f"Release evidence {role!r} schema mismatch: expected={expected_schema!r} observed={observed_schema!r}"
            )
        if role == "complete_model_manifest":
            linker_version = payload.get("incremental_linker_version")
            if not isinstance(linker_version, str) or not linker_version.strip():
                raise ValueError("Release evidence 'complete_model_manifest' must describe a complete bundle")
        evidence[role] = payload
        sha256[role] = member["sha256"]
    return evidence, sha256


def _evidence_field(evidence: Mapping[str, Any], role: str, *path: str) -> Any:
    """Read one required field from a fixed evidence report."""

    value: Any = evidence[role]
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Release evidence {role!r} is missing {'.'.join(path)!r}")
        value = value[part]
    return value


def _validate_evidence_bindings(
    release_spec: Mapping[str, Any],
    evidence: Mapping[str, Any],
    evidence_sha256: Mapping[str, str],
    release_spec_sha256: str,
) -> None:
    """Validate the identities needed to interpret the release evidence."""

    def require(binding_id: str, observed: Any, expected: Any) -> None:
        if observed != expected or isinstance(observed, bool) != isinstance(expected, bool):
            raise ValueError(f"Release evidence binding {binding_id!r} does not match")

    for role in ("pairwise_evaluation_report", "cluster_evaluation_report"):
        require(
            f"{role}.release_spec",
            _evidence_field(evidence, role, "release_spec_sha256"),
            release_spec_sha256,
        )

    for field in (
        "name_counts_manifest_sha256",
        "name_tuples_data_sha256",
        "orcid_prefix_counts_data_sha256",
        "orcid_prefix_counts_manifest_sha256",
    ):
        expected_artifact_sha256 = _evidence_field(evidence, "pairwise_evaluation_report", field)
        require(
            f"cluster_evaluation_report.{field}",
            _evidence_field(evidence, "cluster_evaluation_report", field),
            expected_artifact_sha256,
        )

    require(
        "data_manifest",
        evidence_sha256["data_manifest"],
        release_spec["inputs"]["data_manifest_sha256"],
    )
    complete_model_sha256 = evidence_sha256["complete_model_manifest"]
    for role in (
        "pairwise_evaluation_report",
        "cluster_evaluation_report",
        "linker_evaluation_report",
        "parity_evaluation_report",
        "performance_evaluation_report",
    ):
        require(
            f"{role}.complete_model",
            _evidence_field(evidence, role, "model_manifest_sha256"),
            complete_model_sha256,
        )

    digest_bindings = (
        (
            "pairwise_evaluation_report",
            "population_manifest_sha256",
            release_spec["populations"]["pairwise_manifest_sha256"],
        ),
        (
            "cluster_evaluation_report",
            "population_manifest_sha256",
            release_spec["populations"]["cluster_manifest_sha256"],
        ),
        (
            "performance_evaluation_report",
            "data_manifest_sha256",
            release_spec["inputs"]["performance_data_manifest_sha256"],
        ),
        (
            "performance_evaluation_report",
            "workload_sha256",
            release_spec["inputs"]["performance_workload_sha256"],
        ),
        (
            "subblocking_evaluation_report",
            "input_manifest_sha256",
            release_spec["inputs"]["subblocking_input_manifest_sha256"],
        ),
        (
            "parity_evaluation_report",
            "fixture_manifest_sha256",
            release_spec["inputs"]["parity_fixture_manifest_sha256"],
        ),
    )
    for role, field, expected in digest_bindings:
        require(f"{role}.{field}", _evidence_field(evidence, role, field), expected)

    parity_exact_match = _evidence_field(evidence, "parity_evaluation_report", "clusters_exact_match")
    if not isinstance(parity_exact_match, bool):
        raise ValueError("Release evidence parity clusters_exact_match must be boolean")

    expected_dataset_names = {
        "pairwise_evaluation_report": release_spec["populations"]["pairwise_datasets"],
        "cluster_evaluation_report": release_spec["populations"]["cluster_datasets"],
    }
    for role, expected_names in expected_dataset_names.items():
        datasets = _evidence_field(evidence, role, "datasets")
        if not isinstance(datasets, Mapping):
            raise ValueError(f"Release evidence {role!r} datasets must be an object")
        require(f"{role}.datasets", sorted(datasets), expected_names)

    query_predictions = _evidence_field(evidence, "linker_evaluation_report", "query_predictions")
    if (
        not isinstance(query_predictions, Mapping)
        or not is_lowercase_sha256(query_predictions.get("sha256"))
        or isinstance(query_predictions.get("bytes"), bool)
        or not isinstance(query_predictions.get("bytes"), int)
        or query_predictions["bytes"] < 0
    ):
        raise ValueError("Release evidence linker query_predictions must contain sha256 and nonnegative bytes")


def _linker_measurements(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the complete diagnostic linker measurement set."""

    observed = _evidence_field(evidence, "linker_evaluation_report", "observed_metrics")
    observed_keys = set(observed) if isinstance(observed, Mapping) else set()
    if not isinstance(observed, Mapping) or observed_keys != SUPPORTED_OFFICIAL_METRIC_KEYS:
        raise ValueError(
            "Release evidence linker metrics must contain the complete official metric set: "
            f"missing={sorted(SUPPORTED_OFFICIAL_METRIC_KEYS - observed_keys)} "
            f"extra={sorted(observed_keys - SUPPORTED_OFFICIAL_METRIC_KEYS)}"
        )
    measurements = dict(observed)
    for key in INTEGER_OFFICIAL_METRIC_KEYS:
        value = measurements[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Release evidence linker metric {key!r} must be a nonnegative integer")
    for key in FLOAT_OFFICIAL_METRIC_KEYS:
        value = _release_number(measurements[key], label=f"Release evidence linker metric {key}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Release evidence linker metric {key!r} must be in [0, 1]")
        measurements[key] = value

    training_rows = measurements["training_rows"]
    training_positive_rows = measurements["training_positive_rows"]
    queries = measurements["stratified_test_queries"]
    if training_rows <= 0 or training_positive_rows > training_rows:
        raise ValueError("Release evidence linker training counts are inconsistent")
    if queries <= 0 or any(
        measurements[key] > queries
        for key in (
            "stratified_test_errors",
            "stratified_test_false_abstain",
            "stratified_test_false_link",
            "stratified_test_wrong_candidate_link",
        )
    ):
        raise ValueError("Release evidence linker test counts are inconsistent")

    weights = measurements["weighted_average_error_weights"]
    expected_weight_keys = {
        "false_abstain_error_rate",
        "false_link_error_rate",
        "wrong_link_error_rate",
    }
    if not isinstance(weights, Mapping) or set(weights) != expected_weight_keys:
        raise ValueError("Release evidence linker weighted-average error weights have invalid fields")
    normalized_weights: dict[str, float] = {}
    for key in sorted(expected_weight_keys):
        value = _release_number(weights[key], label=f"Release evidence linker weight {key}")
        if value <= 0:
            raise ValueError(f"Release evidence linker weight {key!r} must be positive")
        normalized_weights[key] = value
    measurements["weighted_average_error_weights"] = normalized_weights
    return measurements


def _evidence_unit_metric(evidence: Mapping[str, Any], role: str, *path: str) -> float:
    value = _release_number(
        _evidence_field(evidence, role, *path),
        label=f"Release evidence metric {role}.{'.'.join(path)}",
    )
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Release evidence metric {role}.{'.'.join(path)} must be in [0, 1]")
    return value


def _release_measurements(
    release_spec: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the compact core and diagnostic measurement snapshot."""

    pairwise_metrics = ("auroc", "macro_f1")
    return {
        "cluster": {
            "datasets": {
                name: {
                    "f1": _evidence_unit_metric(
                        evidence,
                        "cluster_evaluation_report",
                        "datasets",
                        name,
                        "metrics",
                        "f1",
                    )
                }
                for name in release_spec["populations"]["cluster_datasets"]
            },
            "signature_weighted": {
                "f1": _evidence_unit_metric(
                    evidence,
                    "cluster_evaluation_report",
                    "signature_weighted",
                    "f1",
                )
            },
        },
        "linker": _linker_measurements(evidence),
        "pairwise": {
            "aggregate": {
                metric: _evidence_unit_metric(
                    evidence,
                    "pairwise_evaluation_report",
                    "aggregate",
                    metric,
                )
                for metric in pairwise_metrics
            },
            "datasets": {
                name: {
                    metric: _evidence_unit_metric(
                        evidence,
                        "pairwise_evaluation_report",
                        "datasets",
                        name,
                        "metrics",
                        metric,
                    )
                    for metric in pairwise_metrics
                }
                for name in release_spec["populations"]["pairwise_datasets"]
            },
        },
    }


def _drop_check(check_id: str, candidate: float, baseline: float, threshold: float) -> dict[str, Any]:
    observed = baseline - candidate
    return {
        "id": check_id,
        "candidate": candidate,
        "baseline": baseline,
        "observed_drop": observed,
        "threshold": threshold,
        "passed": observed <= threshold,
    }


def _maximum_check(check_id: str, candidate: float, threshold: float) -> dict[str, Any]:
    return {
        "id": check_id,
        "candidate": candidate,
        "threshold": threshold,
        "passed": candidate <= threshold,
    }


def _release_checks(release_spec: Mapping[str, Any], evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Apply the fixed release gates from a compact declarative table."""

    thresholds = release_spec["thresholds"]
    baselines = release_spec["baselines"]
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
    for dataset in release_spec["populations"]["pairwise_datasets"]:
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
            _evidence_unit_metric(evidence, *candidate_path),
            baseline,
            thresholds[threshold_key],
        )
        for check_id, candidate_path, baseline, threshold_key in drop_gates
    ]

    runtime = _release_number(
        _evidence_field(evidence, "performance_evaluation_report", "summary", "predict_seconds", "p50"),
        label="performance_evaluation_report runtime",
    )
    baseline_runtime = baselines["predict_seconds_p50"]
    if runtime <= 0 or baseline_runtime <= 0:
        raise ValueError("Release evidence runtime measurements must be positive")
    checks.append(
        {
            "id": "performance.runtime_ratio",
            "candidate": runtime,
            "baseline": baseline_runtime,
            "observed_ratio": runtime / baseline_runtime,
            "threshold": thresholds["runtime_max_ratio"],
            "passed": runtime / baseline_runtime <= thresholds["runtime_max_ratio"],
        }
    )

    peak_rss = _release_number(
        _evidence_field(evidence, "performance_evaluation_report", "summary", "peak_rss_gb", "max"),
        label="performance.peak_rss_absolute_gb",
    )
    if peak_rss <= 0:
        raise ValueError("Release evidence performance peak RSS must be positive")
    maximum_size = _evidence_field(
        evidence,
        "subblocking_evaluation_report",
        "rust",
        "partition",
        "max_subblock_size",
    )
    if isinstance(maximum_size, bool) or not isinstance(maximum_size, int) or maximum_size <= 0:
        raise ValueError("Release evidence subblocking maximum size must be a positive integer")
    checks.extend(
        (
            _maximum_check(
                "performance.peak_rss_absolute_gb",
                peak_rss,
                thresholds["peak_rss_absolute_max_gb"],
            ),
            _maximum_check(
                "subblocking.maximum_size",
                maximum_size,
                thresholds["subblocking_maximum_size"],
            ),
        )
    )

    component_recall = _evidence_unit_metric(
        evidence,
        "subblocking_evaluation_report",
        "rust",
        "component_preservation",
        "component_pair_recall",
    )
    parity_exact_match = _evidence_field(evidence, "parity_evaluation_report", "clusters_exact_match")
    if not isinstance(parity_exact_match, bool):
        raise ValueError("Release evidence parity clusters_exact_match must be boolean")
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
    """Validate all frozen evidence once and write the aggregate release decision."""

    output_path = Path(args.output_report)
    if output_path.exists():
        raise FileExistsError(f"Evaluation report output already exists: {output_path}")
    evidence_manifest_path = Path(args.evidence_manifest)
    evidence, evidence_sha256 = _load_release_evidence(
        evidence_manifest_path,
        args.expected_evidence_manifest_sha256,
    )
    release_spec = _validate_release_spec_payload(evidence["release_spec"])
    release_spec_sha256 = evidence_sha256["release_spec"]
    _validate_evidence_bindings(
        release_spec,
        evidence,
        evidence_sha256,
        release_spec_sha256,
    )
    measurements = _release_measurements(release_spec, evidence)
    checks = _release_checks(release_spec, evidence)
    passed = all(check["passed"] for check in checks)
    report = {
        "schema_version": EVALUATION_REPORT_SCHEMA,
        "release_spec_sha256": release_spec_sha256,
        "data_manifest_sha256": evidence_sha256["data_manifest"],
        "model_manifest_sha256": evidence_sha256["complete_model_manifest"],
        "evidence_manifest_sha256": args.expected_evidence_manifest_sha256,
        "release_identity": release_spec["release_identity"],
        "measurements": measurements,
        "checks": checks,
        "passed": passed,
    }
    _write_fresh_json(output_path, report)
    return report


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight-training-inputs",
        help="Validate training/test isolation and write a test-path-free training plan.",
    )
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--expected-manifest-sha256", required=True)
    preflight.add_argument("--output-plan", type=Path, required=True)
    preflight.set_defaults(handler=preflight_training_inputs)

    evaluation_report = commands.add_parser(
        "evaluate-release",
        help="Validate bound release evidence and write the one aggregate release decision.",
    )
    evaluation_report.add_argument("--evidence-manifest", type=Path, required=True)
    evaluation_report.add_argument("--expected-evidence-manifest-sha256", required=True)
    evaluation_report.add_argument("--output-report", type=Path, required=True)
    evaluation_report.set_defaults(handler=build_evaluation_report)

    calibrate = commands.add_parser(
        "calibrate-eps",
        help="Calibrate EPS from a frozen spec and write a fresh pairwise bundle and report.",
    )
    calibrate.add_argument("--source-bundle", type=Path, required=True)
    calibrate.add_argument("--spec", type=Path, required=True)
    calibrate.add_argument("--expected-spec-sha256", required=True)
    calibrate.add_argument("--output-bundle", type=Path, required=True)
    calibrate.add_argument("--output-report", type=Path, required=True)
    calibrate.add_argument("--name-counts-index-root", type=Path, required=True)
    calibrate.add_argument("--n-jobs", type=_positive_int, default=1)
    calibrate.add_argument("--total-ram-bytes", type=_positive_int)
    calibrate.set_defaults(handler=calibrate_eps)

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
        evaluator.add_argument("--model", type=Path, required=True, help="Complete production model bundle.")
        evaluator.add_argument("--expected-model-manifest-sha256", required=True)
        evaluator.add_argument("--release-spec", type=Path, required=True)
        evaluator.add_argument("--expected-release-spec-sha256", required=True)
        evaluator.add_argument("--manifest", type=Path, required=True)
        evaluator.add_argument("--expected-manifest-sha256", required=True)
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

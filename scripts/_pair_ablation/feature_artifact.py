"""Strict memory-mapped pair-feature artifacts for ablation runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from s2and.pairwise_training import pair_identity_digest
from scripts._pair_ablation.pair_sources import PAIR_COLUMNS, canonicalize_pairs
from scripts._pair_ablation.results import load_strict_json

FEATURE_STORE_SCHEMA_VERSION = "s2and_pair_ablation_feature_store_v1"
_OUTPUT_FILES = ("pairs.parquet", "main.npy", "nameless.npy", "labels.npy")
_MANIFEST_KEYS = {
    "artifact_identity_digest",
    "artifact_manifest_sha256",
    "domain",
    "main_feature_indices",
    "output_dtypes",
    "output_sha256",
    "output_shapes",
    "pair_digest",
    "rows",
    "schema_version",
    "nameless_feature_indices",
}


@dataclass(slots=True)
class DomainFeatureStore:
    """Verified memory-mapped pair features for one source domain."""

    domain: str
    pairs: pd.DataFrame
    main: np.ndarray
    nameless: np.ndarray
    labels: np.ndarray
    row_by_pair: dict[tuple[str, str], int]
    manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_binary_labels(values: Any, *, context: str) -> np.ndarray:
    """Validate original values are finite, scalar, exact binary labels."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{context} must be one-dimensional")
    if array.dtype.kind not in "buif":
        raise ValueError(f"{context} must contain numeric or Boolean labels")
    numeric = array.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all() or not bool(np.isin(numeric, (0.0, 1.0)).all()):
        raise ValueError(f"{context} must contain only exact finite binary labels")
    return numeric.astype(np.int8)


def _validate_matrix(values: Any, *, context: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.dtype.kind != "f":
        raise ValueError(f"{context} must be a two-dimensional floating-point matrix")
    if not matrix.flags.c_contiguous:
        matrix = np.ascontiguousarray(matrix)
    return matrix


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def write_feature_store(
    store_dir: Path,
    *,
    domain: str,
    pairs: pd.DataFrame,
    main: np.ndarray,
    nameless: np.ndarray,
    labels: np.ndarray,
    artifact_identity_digest: str,
    artifact_manifest_sha256: str,
    main_feature_indices: Sequence[int],
    nameless_feature_indices: Sequence[int],
) -> dict[str, Any]:
    """Validate and atomically persist one domain feature store."""

    canonical = canonicalize_pairs(pairs.loc[:, PAIR_COLUMNS])
    observed_domains = canonical["source_domain"].astype(str).unique().tolist()
    if observed_domains != [domain]:
        raise ValueError(f"feature-store domain mismatch: expected={domain!r}, observed={observed_domains}")
    main_matrix = _validate_matrix(main, context="main features")
    nameless_matrix = _validate_matrix(nameless, context="nameless features")
    target = validate_binary_labels(labels, context="feature labels")
    if len(canonical) != len(main_matrix) or len(canonical) != len(nameless_matrix) or len(canonical) != len(target):
        raise ValueError("feature-store arrays must have equal row counts")
    if main_matrix.shape[1] != len(main_feature_indices):
        raise ValueError("main feature count does not match main_feature_indices")
    if nameless_matrix.shape[1] != len(nameless_feature_indices):
        raise ValueError("nameless feature count does not match nameless_feature_indices")
    pair_labels = validate_binary_labels(canonical["label"].to_numpy(), context="pair labels")
    if not np.array_equal(pair_labels, target):
        raise ValueError("feature labels do not match canonical pair labels")
    store_dir.mkdir(parents=True, exist_ok=True)
    temporary_pairs = store_dir / "pairs.tmp.parquet"
    canonical.to_parquet(temporary_pairs, index=False)
    temporary_pairs.replace(store_dir / "pairs.parquet")
    _save_array_atomic(store_dir / "main.npy", main_matrix)
    _save_array_atomic(store_dir / "nameless.npy", nameless_matrix)
    _save_array_atomic(store_dir / "labels.npy", target)
    output_sha256 = {name: sha256_file(store_dir / name) for name in _OUTPUT_FILES}
    output_shapes = {
        "main.npy": list(main_matrix.shape),
        "nameless.npy": list(nameless_matrix.shape),
        "labels.npy": list(target.shape),
    }
    output_dtypes = {
        "main.npy": str(main_matrix.dtype),
        "nameless.npy": str(nameless_matrix.dtype),
        "labels.npy": str(target.dtype),
    }
    manifest = {
        "schema_version": FEATURE_STORE_SCHEMA_VERSION,
        "domain": domain,
        "rows": int(len(canonical)),
        "pair_digest": pair_identity_digest(canonical),
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "artifact_identity_digest": artifact_identity_digest,
        "output_sha256": output_sha256,
        "output_shapes": output_shapes,
        "output_dtypes": output_dtypes,
        "main_feature_indices": [int(value) for value in main_feature_indices],
        "nameless_feature_indices": [int(value) for value in nameless_feature_indices],
    }
    temporary_manifest = store_dir / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_manifest.replace(store_dir / "manifest.json")
    return manifest


def _require_manifest_shape(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value
    ):
        raise ValueError(f"{context} must be a list of non-negative dimensions")
    return tuple(value)


def load_feature_store(
    store_dir: Path,
    *,
    expected_domain: str,
    expected_pair_digest: str,
    expected_artifact_identity_digest: str,
    expected_main_feature_indices: Sequence[int],
    expected_nameless_feature_indices: Sequence[int],
) -> DomainFeatureStore:
    """Load one memory-mapped feature store after full manifest verification."""

    manifest_path = store_dir / "manifest.json"
    manifest = load_strict_json(manifest_path)
    observed_keys = set(manifest)
    if observed_keys != _MANIFEST_KEYS:
        raise ValueError(
            f"feature manifest schema mismatch: missing={sorted(_MANIFEST_KEYS - observed_keys)}, "
            f"extra={sorted(observed_keys - _MANIFEST_KEYS)}"
        )
    if manifest["schema_version"] != FEATURE_STORE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported feature-store schema: {manifest['schema_version']!r}")
    if manifest["domain"] != expected_domain or store_dir.name != expected_domain:
        raise ValueError("feature-store domain identity mismatch")
    if manifest["pair_digest"] != expected_pair_digest:
        raise ValueError("feature-store pair identity mismatch")
    if manifest["artifact_identity_digest"] != expected_artifact_identity_digest:
        raise ValueError("feature-store input-artifact identity mismatch")
    if manifest["main_feature_indices"] != [int(value) for value in expected_main_feature_indices]:
        raise ValueError("feature-store main feature indices mismatch")
    if manifest["nameless_feature_indices"] != [int(value) for value in expected_nameless_feature_indices]:
        raise ValueError("feature-store nameless feature indices mismatch")
    if not isinstance(manifest["output_sha256"], dict) or set(manifest["output_sha256"]) != set(_OUTPUT_FILES):
        raise ValueError("feature-store output hash manifest is malformed")
    for name in _OUTPUT_FILES:
        path = store_dir / name
        if not path.is_file() or sha256_file(path) != manifest["output_sha256"][name]:
            raise ValueError(f"feature-store output hash mismatch: {path}")
    pairs = canonicalize_pairs(pd.read_parquet(store_dir / "pairs.parquet").loc[:, PAIR_COLUMNS])
    observed_domains = pairs["source_domain"].astype(str).unique().tolist()
    if observed_domains != [expected_domain] or pair_identity_digest(pairs) != expected_pair_digest:
        raise ValueError("feature-store pairs do not match their declared domain/digest")
    main = np.load(store_dir / "main.npy", mmap_mode="r")
    nameless = np.load(store_dir / "nameless.npy", mmap_mode="r")
    labels = np.load(store_dir / "labels.npy", mmap_mode="r")
    arrays = {"main.npy": main, "nameless.npy": nameless, "labels.npy": labels}
    if not isinstance(manifest["output_shapes"], dict) or set(manifest["output_shapes"]) != set(arrays):
        raise ValueError("feature-store shape manifest is malformed")
    if not isinstance(manifest["output_dtypes"], dict) or set(manifest["output_dtypes"]) != set(arrays):
        raise ValueError("feature-store dtype manifest is malformed")
    for name, array in arrays.items():
        if array.shape != _require_manifest_shape(manifest["output_shapes"][name], f"output_shapes.{name}"):
            raise ValueError(f"feature-store shape mismatch: {name}")
        if str(array.dtype) != manifest["output_dtypes"][name]:
            raise ValueError(f"feature-store dtype mismatch: {name}")
    if main.ndim != 2 or main.dtype.kind != "f" or not main.flags.c_contiguous:
        raise ValueError("feature-store main matrix has invalid representation")
    if nameless.ndim != 2 or nameless.dtype.kind != "f" or not nameless.flags.c_contiguous:
        raise ValueError("feature-store nameless matrix has invalid representation")
    target = validate_binary_labels(labels, context="stored feature labels")
    if labels.dtype != np.dtype("int8") or labels.ndim != 1:
        raise ValueError("feature-store labels must be one-dimensional int8")
    rows = int(manifest["rows"])
    if rows != len(pairs) or rows != len(main) or rows != len(nameless) or rows != len(labels):
        raise ValueError("feature-store row counts do not reconcile")
    if main.shape[1] != len(expected_main_feature_indices) or nameless.shape[1] != len(
        expected_nameless_feature_indices
    ):
        raise ValueError("feature-store feature counts do not reconcile")
    if not np.array_equal(target, validate_binary_labels(pairs["label"].to_numpy(), context="pair labels")):
        raise ValueError("stored feature labels do not match pair labels")
    row_by_pair = {
        (str(pair1), str(pair2)): index
        for index, (_domain, _family, pair1, pair2, _label, _rule, _origin, _group) in enumerate(
            pairs.itertuples(index=False, name=None)
        )
    }
    if len(row_by_pair) != rows:
        raise ValueError("feature-store pairs are not unique by canonical pair key")
    return DomainFeatureStore(
        domain=expected_domain,
        pairs=pairs,
        main=main,
        nameless=nameless,
        labels=labels,
        row_by_pair=row_by_pair,
        manifest=manifest,
    )

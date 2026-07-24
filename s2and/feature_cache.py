"""Training-time cache for featurized train/validation/test splits.

The caller supplies a JSON-serializable ``source_key`` describing the dataset
inputs. This module adds the featurizer configuration and exact ordered pair
list, then stores one uncompressed NPZ file per split.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from s2and._atomic_io import exclusive_file_lock
from s2and.consts import DEFAULT_CHUNK_SIZE
from s2and.data import ANDData
from s2and.featurizer import (
    FeaturizationInfo,
    TupleOfArrays,
    many_pairs_featurize,
    resolve_training_pairs,
)

logger = logging.getLogger("s2and")

SNAPSHOT_SCHEMA_VERSION = 2

_PairList = list[tuple[str, str, int | float]]


def _hash_pair_list(pairs: _PairList) -> str:
    """Hash pairs in the same normalized form consumed by featurization."""

    digest = hashlib.sha256()
    for raw_id_1, raw_id_2, raw_label in pairs:
        id_1, id_2, label = str(raw_id_1), str(raw_id_2), float(raw_label)
        digest.update(f"{len(id_1)}:{id_1}{len(id_2)}:{id_2}|{label!r}\n".encode())
    return digest.hexdigest()


def _featurizer_identity(featurizer_info: FeaturizationInfo | None) -> dict[str, object] | None:
    if featurizer_info is None:
        return None
    return {
        "featurizer_version": int(featurizer_info.featurizer_version),
        "features_to_use": sorted(set(featurizer_info.features_to_use)),
    }


def _snapshot_key(
    *,
    source_key: Mapping[str, object],
    featurizer_info: FeaturizationInfo,
    nameless_featurizer_info: FeaturizationInfo | None,
    nan_value: float,
    pair_list_hash: str,
) -> str:
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": dict(source_key),
        "featurizer": _featurizer_identity(featurizer_info),
        "nameless_featurizer": _featurizer_identity(nameless_featurizer_info),
        "nan_value": repr(nan_value),
        "pairs_sha256": pair_list_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_snapshot_arrays(
    arrays: dict[str, np.ndarray],
    *,
    path: Path,
    expected_rows: int,
    expected_width: int,
    expected_nameless_width: int | None,
) -> TupleOfArrays:
    expected_members = {"X", "y"} | ({"nameless_X"} if expected_nameless_width is not None else set())
    if set(arrays) != expected_members:
        raise ValueError(f"Feature snapshot {path} has members {sorted(arrays)}, expected {sorted(expected_members)}")

    features, labels = arrays["X"], arrays["y"]
    nameless = arrays.get("nameless_X")
    for name in ("X", "y") + (("nameless_X",) if nameless is not None else ()):
        if arrays[name].dtype != np.float64:
            raise ValueError(f"Feature snapshot {path} member {name} has dtype {arrays[name].dtype}, expected float64")
    if features.shape != (expected_rows, expected_width):
        raise ValueError(
            f"Feature snapshot {path} X has shape {features.shape}, expected ({expected_rows}, {expected_width})"
        )
    if labels.shape != (expected_rows,):
        raise ValueError(f"Feature snapshot {path} y has shape {labels.shape}, expected ({expected_rows},)")
    if nameless is not None and nameless.shape != (expected_rows, expected_nameless_width):
        raise ValueError(
            f"Feature snapshot {path} nameless_X has shape {nameless.shape}, "
            f"expected ({expected_rows}, {expected_nameless_width})"
        )
    return features, labels, nameless


def _load_snapshot(
    path: Path,
    *,
    expected_rows: int,
    expected_width: int,
    expected_nameless_width: int | None,
) -> TupleOfArrays:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
    except (ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"Feature snapshot {path} is unreadable (corrupt or truncated); delete it and rerun: {exc}"
        ) from exc
    return _validate_snapshot_arrays(
        arrays,
        path=path,
        expected_rows=expected_rows,
        expected_width=expected_width,
        expected_nameless_width=expected_nameless_width,
    )


def _publish_snapshot(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    """Atomically publish ``path`` once without replacing an existing snapshot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as handle:
            tmp_path = Path(handle.name)
            # NumPy's stub cannot express this dynamically named array mapping.
            np.savez(handle, **arrays)  # ty: ignore[invalid-argument-type]
        with exclusive_file_lock(path.with_suffix(f"{path.suffix}.lock")):
            if path.exists():
                return False
            os.replace(tmp_path, path)
            tmp_path = None
            return True
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def cached_featurize(
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    *,
    source_key: Mapping[str, object],
    cache_dir: str | os.PathLike[str],
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nameless_featurizer_info: FeaturizationInfo | None = None,
    nan_value: float = np.nan,
    total_ram_bytes: int | None = None,
) -> tuple[TupleOfArrays, TupleOfArrays, TupleOfArrays]:
    """Featurize a training dataset, reusing content-addressed split snapshots.

    ``source_key`` must contain every dataset input that can affect features.
    The production training caller uses exact file and artifact SHA-256s plus
    the normalization version.
    """

    train_pairs, val_pairs, test_pairs = resolve_training_pairs(dataset)
    expected_width = len(featurizer_info.selected_feature_indices())
    expected_nameless_width = (
        len(nameless_featurizer_info.selected_feature_indices()) if nameless_featurizer_info is not None else None
    )

    results: list[TupleOfArrays] = []
    for split_name, split_pairs in (
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ):
        key = _snapshot_key(
            source_key=source_key,
            featurizer_info=featurizer_info,
            nameless_featurizer_info=nameless_featurizer_info,
            nan_value=nan_value,
            pair_list_hash=_hash_pair_list(split_pairs),
        )
        path = Path(cache_dir) / f"{split_name}_{key}.npz"
        expected_rows = len(split_pairs)
        if path.exists():
            logger.info("Feature snapshot hit split=%s path=%s", split_name, path)
            results.append(
                _load_snapshot(
                    path,
                    expected_rows=expected_rows,
                    expected_width=expected_width,
                    expected_nameless_width=expected_nameless_width,
                )
            )
            continue

        logger.info("Feature snapshot miss split=%s pairs=%d; computing", split_name, expected_rows)
        features, labels, nameless = many_pairs_featurize(
            split_pairs,
            dataset,
            featurizer_info,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            nameless_featurizer_info=nameless_featurizer_info,
            nan_value=nan_value,
            total_ram_bytes=total_ram_bytes,
        )
        arrays = {"X": features, "y": labels}
        if nameless is not None:
            arrays["nameless_X"] = nameless
        validated = _validate_snapshot_arrays(
            arrays,
            path=path,
            expected_rows=expected_rows,
            expected_width=expected_width,
            expected_nameless_width=expected_nameless_width,
        )
        if _publish_snapshot(path, arrays):
            logger.info("Feature snapshot published split=%s path=%s bytes=%d", split_name, path, path.stat().st_size)
            results.append(validated)
        else:
            logger.info("Feature snapshot filled by concurrent caller split=%s path=%s", split_name, path)
            results.append(
                _load_snapshot(
                    path,
                    expected_rows=expected_rows,
                    expected_width=expected_width,
                    expected_nameless_width=expected_nameless_width,
                )
            )

    train_result, val_result, test_result = results
    return train_result, val_result, test_result

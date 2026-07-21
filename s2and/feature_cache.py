"""Training-time featurized-split snapshot cache.

Caches the *output* of split featurization — one uncompressed NPZ file per
(train/val/test) split — keyed by a content hash of everything that determines
the matrices: the feature-input files, the name-counts binding, the featurizer
configuration, and the exact resolved pair lists. There is no pair-level
caching, no invalidation logic (a changed input is a different key), and no
production usage: production inference never touches this module.

Snapshot identity is fail-closed: if any feature input lacks verifiable
content identity, the owned build-and-cache operation raises instead of
caching blind.

Concurrency: snapshots are content-addressed and written once. Publication
happens under a short file lock, an existing snapshot is never overwritten,
and losing writers load the published winner before returning.
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
from typing import Any

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
from s2and.name_count_binding import NameCountsBinding
from s2and.text import compute_block

logger = logging.getLogger("s2and")

SNAPSHOT_SCHEMA_VERSION = 1

_PairList = list[tuple[str, str, int | float]]


def _require_sha256(value: object, *, source_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"Feature snapshot source {source_name!r} requires a lowercase SHA-256 digest")
    return value


def _dataset_feature_fingerprint(
    dataset: ANDData,
    source_content_sha256: Mapping[str, str],
    *,
    uses_name_counts: bool,
) -> dict[str, object]:
    """Return verifiable content identity for every feature input.

    Source identity is consumed from private one-shot ingestion state so the
    key describes the exact bytes materialized by the owned build-and-cache
    lifecycle.

    Args:
        dataset: The training dataset.
        source_content_sha256: Consumed hashes of the exact source bytes that
            were parsed while building ``dataset``.
        uses_name_counts: Whether either requested matrix includes name-count
            features.

    Returns:
        JSON-serializable mapping of feature-input identities.

    Raises:
        ValueError: If any feature input lacks verifiable content identity or
            the dataset uses an unsupported (unbound) configuration.
    """
    if not dataset.preprocess:
        raise ValueError("Feature snapshot caching supports only preprocess=True datasets")
    if dataset.compute_block_fn is not compute_block:
        raise ValueError("Feature snapshot caching supports only the default compute_block function")

    ingestion_hashes = dict(source_content_sha256)
    missing = sorted(name for name in ("signatures", "papers") if name not in ingestion_hashes)
    if missing:
        raise ValueError(
            "Feature snapshot caching requires file-backed dataset inputs with verifiable "
            f"content identity; not loaded from files: {missing}. Pass file paths to ANDData "
            "or disable the snapshot cache."
        )
    if dataset.specter_embeddings and "specter_embeddings" not in ingestion_hashes:
        raise ValueError(
            "Feature snapshot caching requires file-backed SPECTER embeddings; "
            "in-memory embeddings have no verifiable content identity."
        )
    source_identity: dict[str, object] = {
        name: _require_sha256(ingestion_hashes[name], source_name=name)
        for name in ("signatures", "papers", "specter_embeddings")
        if name in ingestion_hashes
    }

    provenance = dataset.name_counts_provenance if uses_name_counts else None
    name_counts_identity = (
        None
        if provenance is None
        else NameCountsBinding.from_provenance(provenance, context="feature snapshot cache").feature_contract_fields()
    )

    name_tuples_payload = json.dumps(
        sorted(dataset.name_tuples),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    name_tuples_digest = hashlib.sha256(name_tuples_payload).hexdigest()

    return {
        "source": source_identity,
        "name_counts": name_counts_identity,
        "name_tuples_sha256": name_tuples_digest,
        "normalization_version": dataset.normalization_version,
    }


def _hash_pair_list(pairs: _PairList) -> str:
    """Hash pairs in the normalized form featurization consumes: string IDs
    and float64 labels (``many_pairs_featurize`` applies the same coercions)."""
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
    dataset_fingerprint: dict[str, object],
    featurizer_info: FeaturizationInfo,
    nameless_featurizer_info: FeaturizationInfo | None,
    nan_value: float,
    pair_list_hash: str,
) -> str:
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "dataset": dataset_fingerprint,
        "featurizer": _featurizer_identity(featurizer_info),
        "nameless_featurizer": _featurizer_identity(nameless_featurizer_info),
        "nan_value": repr(nan_value),
        "pairs_sha256": pair_list_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_snapshot_arrays(
    arrays: dict[str, np.ndarray],
    *,
    path: Path,
    expected_key: str,
    expected_rows: int,
    expected_width: int,
    expected_nameless_width: int | None,
) -> TupleOfArrays:
    expected_members = {"key", "X", "y"} | ({"nameless_X"} if expected_nameless_width is not None else set())
    if set(arrays) != expected_members:
        raise ValueError(f"Feature snapshot {path} has members {sorted(arrays)}, expected {sorted(expected_members)}")
    embedded_key = arrays["key"]
    if embedded_key.shape != () or str(embedded_key.item()) != expected_key:
        raise ValueError(f"Feature snapshot {path} embedded key does not match the requested snapshot key")
    features, labels = arrays["X"], arrays["y"]
    nameless = arrays.get("nameless_X")
    for name in ("X", "y") + (("nameless_X",) if nameless is not None else ()):
        if arrays[name].dtype != np.float64:
            raise ValueError(f"Feature snapshot {path} member {name} has dtype {arrays[name].dtype}, expected float64")
    if features.ndim != 2 or features.shape[1] != expected_width:
        raise ValueError(f"Feature snapshot {path} X has shape {features.shape}, expected (rows, {expected_width})")
    if features.shape[0] != expected_rows:
        raise ValueError(f"Feature snapshot {path} X has {features.shape[0]} rows, expected {expected_rows}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(f"Feature snapshot {path} y has shape {labels.shape}, expected ({features.shape[0]},)")
    if nameless is not None and (nameless.ndim != 2 or nameless.shape != (features.shape[0], expected_nameless_width)):
        raise ValueError(
            f"Feature snapshot {path} nameless_X has shape {nameless.shape}, "
            f"expected ({features.shape[0]}, {expected_nameless_width})"
        )
    return features, labels, nameless


def _load_snapshot(
    path: Path,
    *,
    expected_key: str,
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
        expected_key=expected_key,
        expected_rows=expected_rows,
        expected_width=expected_width,
        expected_nameless_width=expected_nameless_width,
    )


def _publish_snapshot(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    """Write a snapshot once; never overwrite an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as handle:
            tmp_path = Path(handle.name)
            np.savez(handle, **arrays)
        with exclusive_file_lock(f"{path}.lock"):
            if path.exists():
                tmp_path.unlink()
                published = False
            else:
                os.replace(tmp_path, path)
                published = True
        tmp_path = None
        return published
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def build_and_cached_featurize(
    anddata_kwargs: Mapping[str, Any],
    featurizer_info: FeaturizationInfo,
    *,
    cache_dir: str | os.PathLike[str],
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nameless_featurizer_info: FeaturizationInfo | None = None,
    nan_value: float = np.nan,
    total_ram_bytes: int | None = None,
) -> tuple[ANDData, tuple[TupleOfArrays, TupleOfArrays, TupleOfArrays]]:
    """Build and immediately cache one fresh training dataset.

    ``ANDData`` construction is owned by this function. The dataset is not
    exposed until its one-shot provenance has been consumed and its split
    matrices have been loaded or computed, so mutable dataset state cannot be
    rebound to an existing key.

    Args:
        anddata_kwargs: Keyword arguments for a classic file-backed ``ANDData``.
            The private Arrow and source-capture arguments are owned outside
            this Python-only snapshot boundary and must not be passed.
        featurizer_info: Listing of feature groups to use.
        cache_dir: Snapshot root directory.
        n_jobs: Number of CPUs to use for computation on a miss.
        chunk_size: Chunk size for multiprocessing.
        nameless_featurizer_info: Featurization configuration for nameless
            features.
        nan_value: Value used to replace NaNs.
        total_ram_bytes: Optional explicit RAM input for memory budgeting.

    Returns:
        The newly built dataset and its train/validation/test arrays.

    Raises:
        ValueError: If the caller tries to control private construction state.
    """
    constructor_kwargs = dict(anddata_kwargs)
    if "_capture_feature_source_hashes" in constructor_kwargs:
        raise ValueError("Feature snapshot source capture is owned by build_and_cached_featurize")
    if "_validated_arrow_inputs" in constructor_kwargs:
        raise ValueError("Feature snapshot caching supports only classic file-backed Python training datasets")
    dataset = ANDData(**constructor_kwargs, _capture_feature_source_hashes=True)
    source_content_sha256 = dataset._consume_feature_source_sha256()
    splits = _cached_featurize_fresh(
        dataset,
        source_content_sha256,
        featurizer_info,
        cache_dir=cache_dir,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        nameless_featurizer_info=nameless_featurizer_info,
        nan_value=nan_value,
        total_ram_bytes=total_ram_bytes,
    )
    return dataset, splits


def _cached_featurize_fresh(
    dataset: ANDData,
    source_content_sha256: Mapping[str, str],
    featurizer_info: FeaturizationInfo,
    *,
    cache_dir: str | os.PathLike[str],
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nameless_featurizer_info: FeaturizationInfo | None = None,
    nan_value: float = np.nan,
    total_ram_bytes: int | None = None,
) -> tuple[TupleOfArrays, TupleOfArrays, TupleOfArrays]:
    """Featurize a freshly built training dataset with per-split snapshots.

    Resolves the train/val/test pair lists once (via
    :func:`s2and.featurizer.resolve_training_pairs`), hashes those exact lists
    into each split's snapshot key, and either loads the split's snapshot or
    computes it with :func:`s2and.featurizer.many_pairs_featurize` and
    publishes the result write-once.

    Args:
        dataset: A freshly built ``mode='train'`` dataset.
        source_content_sha256: Consumed hashes of the exact file bytes parsed.
        featurizer_info: Listing of feature groups to use.
        cache_dir: Snapshot root directory. Snapshots are stored as
            ``<cache_dir>/<split>_<key-prefix>.npz``.
        n_jobs: The number of cpus to use for computation on a miss.
        chunk_size: The chunk size for multiprocessing.
        nameless_featurizer_info: FeaturizationInfo for the nameless features.
        nan_value: The value to replace NaNs with.
        total_ram_bytes: Optional explicit RAM input for memory budgeting.

    Returns:
        Tuple of train/val/test ``(features, labels, nameless_features)``.

    Raises:
        ValueError: If any feature input lacks verifiable content identity, or
            an existing snapshot fails validation.
    """
    uses_name_counts = any(
        info is not None and "name_counts" in info.features_to_use
        for info in (featurizer_info, nameless_featurizer_info)
    )
    if uses_name_counts and dataset.name_counts_provenance is None:
        raise ValueError(
            "Feature snapshot caching requires verified name-count provenance when name_counts is selected"
        )

    dataset_fingerprint = _dataset_feature_fingerprint(
        dataset,
        source_content_sha256,
        uses_name_counts=uses_name_counts,
    )
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
            dataset_fingerprint=dataset_fingerprint,
            featurizer_info=featurizer_info,
            nameless_featurizer_info=nameless_featurizer_info,
            nan_value=nan_value,
            pair_list_hash=_hash_pair_list(split_pairs),
        )
        expected_rows = len(split_pairs)
        path = Path(cache_dir) / f"{split_name}_{key[:32]}.npz"
        if path.exists():
            logger.info("Feature snapshot hit split=%s path=%s", split_name, path)
            results.append(
                _load_snapshot(
                    path,
                    expected_key=key,
                    expected_rows=expected_rows,
                    expected_width=expected_width,
                    expected_nameless_width=expected_nameless_width,
                )
            )
            continue

        logger.info("Feature snapshot miss split=%s pairs=%d; computing", split_name, len(split_pairs))
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
        arrays = {"key": np.array(key), "X": features, "y": labels}
        if nameless is not None:
            arrays["nameless_X"] = nameless
        validated = _validate_snapshot_arrays(
            arrays,
            path=path,
            expected_key=key,
            expected_rows=expected_rows,
            expected_width=expected_width,
            expected_nameless_width=expected_nameless_width,
        )
        if _publish_snapshot(path, arrays):
            logger.info("Feature snapshot published split=%s path=%s bytes=%d", split_name, path, path.stat().st_size)
            results.append(validated)
        else:
            logger.info("Feature snapshot race lost split=%s path=%s; loading winner", split_name, path)
            results.append(
                _load_snapshot(
                    path,
                    expected_key=key,
                    expected_rows=expected_rows,
                    expected_width=expected_width,
                    expected_nameless_width=expected_nameless_width,
                )
            )

    train_result, val_result, test_result = results
    return train_result, val_result, test_result

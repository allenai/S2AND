"""Convert S2AND runtime inputs into direct-Rust Arrow artifacts.

The runtime bundle writer emits bounded Arrow IPC file-format tables plus the
current S2AND raw-planner batch-index sidecars (S2ABI003).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from s2and._atomic_io import exclusive_file_lock  # noqa: E402
from s2and._sha256 import sha256_file as _file_sha256  # noqa: E402
from s2and.arrow_inputs import (  # noqa: E402
    ARROW_COLLECTION_KIND,
    PUBLIC_DATA_KIND,
    ArrowDataset,
    build_arrow_artifact_manifest,
    read_arrow_collection_root,
    require_name_counts_index_artifact,
    write_arrow_artifact_manifest,
)
from s2and.arrow_provenance import write_conversion_provenance  # noqa: E402
from s2and.arrow_schema import validate_arrow_schema  # noqa: E402
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION  # noqa: E402

logger = logging.getLogger(__name__)


BENCHMARK_DATASETS = ("aminer", "arnetminer", "inspire", "kisti", "medline", "pubmed", "qian", "zbmath")
_ROOT_MANIFEST_LOCK_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class RuntimeDatasetSources:
    """Source files for one table-shaped runtime dataset."""

    dataset: str
    source_dir: Path
    signatures_path: Path
    papers_path: Path
    clusters_path: Path | None = None
    specter_path: Path | None = None
    specter2_path: Path | None = None
    train_pairs_path: Path | None = None
    val_pairs_path: Path | None = None
    test_pairs_path: Path | None = None


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as infile:
        return json.load(infile)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as tmp_file:
            tmp_file.write(encoded)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _resolve_manifest_path(path_value: Any, base_dir: Path | None) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _manifest_relative_path(path_value: Any, manifest_dir: Path) -> str:
    path = Path(str(path_value))
    try:
        return Path(os.path.relpath(str(path.resolve()), str(manifest_dir.resolve()))).as_posix()
    except ValueError:
        return path.as_posix()


def _root_child_manifest_path(root: Path, raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path must be a nonempty string")
    relative_path = Path(raw_path.replace("\\", "/"))
    if relative_path.is_absolute():
        raise ValueError(f"{label} path must be relative to its root")
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} manifest does not exist: {resolved}")
    return resolved


def _read_generic_root_manifest(
    root_manifest_path: Path,
    *,
    ignore_dataset: str | None = None,
) -> dict[str, Path]:
    datasets, _replays, release_version = read_arrow_collection_root(
        root_manifest_path,
        ignore_dataset=ignore_dataset,
    )
    if release_version is not None:
        raise ValueError(f"Generic Arrow conversion cannot modify a published root: {root_manifest_path}")
    return datasets


def _validate_existing_root_manifest(root_manifest_path: Path) -> None:
    if root_manifest_path.exists():
        _read_generic_root_manifest(root_manifest_path)


def _manifest_bindings(
    output_root: Path,
    paths: Mapping[str, str],
    *,
    label: str,
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, path_text in sorted(paths.items()):
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} keys must be nonempty strings")
        manifest_path = _root_child_manifest_path(output_root, path_text, label=f"{label}.{name}")
        bindings[name] = {
            "path": str(path_text).replace("\\", "/"),
            "sha256": _file_sha256(manifest_path),
        }
    return bindings


def _write_root_manifest(
    output_root: Path,
    *,
    dataset_manifests: Mapping[str, str],
    replay_bundles: Mapping[str, str] | None = None,
    release_version: str | None = None,
) -> dict[str, Any]:
    if release_version is not None and (not release_version or release_version.strip() != release_version):
        raise ValueError("release_version must be a nonempty trimmed string")
    if not dataset_manifests:
        raise ValueError("dataset_manifests must be nonempty")
    if replay_bundles and release_version is None:
        raise ValueError("Only a published public-data root may declare replay_bundles")
    payload: dict[str, Any] = {
        "kind": PUBLIC_DATA_KIND if release_version is not None else ARROW_COLLECTION_KIND,
        "format_version": PUBLIC_DATA_FORMAT_VERSION,
        "dataset_manifests": _manifest_bindings(
            output_root,
            dataset_manifests,
            label="dataset_manifests",
        ),
    }
    if release_version is not None:
        payload["release_version"] = release_version
    if replay_bundles:
        payload["replay_bundles"] = _manifest_bindings(
            output_root,
            replay_bundles,
            label="replay_bundles",
        )
    _replace_json(output_root / "manifest.json", payload)
    return payload


def _upsert_root_manifest(output_root: Path, *, dataset_name: str, dataset_dir: Path) -> None:
    root_manifest_path = output_root / "manifest.json"
    lock_path = root_manifest_path.with_suffix(root_manifest_path.suffix + ".lock")
    with exclusive_file_lock(lock_path, timeout_seconds=_ROOT_MANIFEST_LOCK_TIMEOUT_SECONDS):
        dataset_manifests: dict[str, str] = {}
        if root_manifest_path.exists():
            existing_datasets = _read_generic_root_manifest(
                root_manifest_path,
                ignore_dataset=dataset_name,
            )
            dataset_manifests = {
                name: _manifest_relative_path(path, output_root) for name, path in existing_datasets.items()
            }
        dataset_manifests[dataset_name] = _manifest_relative_path(
            dataset_dir / "manifest.json",
            output_root,
        )
        _write_root_manifest(
            output_root,
            dataset_manifests=dataset_manifests,
        )


def _mapping_by_id(rows: Any, *, id_key: str, label: str) -> dict[str, Mapping[str, Any]]:
    if isinstance(rows, Mapping):
        return {str(key): value for key, value in rows.items()}
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        raise TypeError(f"{label} must be a JSON object or list")
    mapped: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"{label} rows must be objects")
        row_id = row.get(id_key)
        if row_id is None:
            raise ValueError(f"{label} row is missing {id_key!r}")
        row_key = str(row_id)
        if row_key in mapped:
            raise ValueError(f"{label} contains duplicate {id_key}: {row_key!r}")
        mapped[row_key] = row
    return mapped


def join_canonical_benchmark_names(
    signatures: Mapping[str, Any],
    canonical_rows: Sequence[Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Replace benchmark name fields with canonical rows joined by signature ID."""

    fields = ("first", "middle", "last")
    for signature_id, signature in signatures.items():
        if signature["signature_id"] != signature_id:
            raise ValueError(f"benchmark signature key does not match signature_id: {signature_id!r}")

    canonical_by_id = _mapping_by_id(canonical_rows, id_key="signature_id", label="canonical name rows")

    missing = sorted(signatures.keys() - canonical_by_id.keys())
    extra = sorted(canonical_by_id.keys() - signatures.keys())
    if missing or extra:
        raise ValueError(
            "canonical name signature IDs must exactly match benchmark signatures: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    field_divergence_counts = {field: 0 for field in fields}
    changed_signature_count = 0
    joined: dict[str, dict[str, Any]] = {}
    for signature_id in sorted(signatures):
        signature = signatures[signature_id]
        author_info = dict(signature["author_info"])
        canonical = canonical_by_id[signature_id]
        changed = False
        for field in fields:
            if author_info.get(field) != canonical[field]:
                field_divergence_counts[field] += 1
                changed = True
            author_info[field] = canonical[field]
        changed_signature_count += int(changed)
        joined[signature_id] = {**signature, "author_info": author_info}

    return joined, {
        "rows": len(joined),
        "changed_signatures": changed_signature_count,
        "field_changes": field_divergence_counts,
    }


AUGMENTED_SOURCE_SEPARATOR = "___"
AUGMENTED_SIGNATURES_FILE = "augmented_signatures.json"
AUGMENTED_TRAINING_PAIR_FILES = ("train_pairs.csv", "val_pairs.csv")
AUGMENTED_TEST_PAIR_FILE = "test_pairs.csv"
AUGMENTED_TEST_PAIRS_OUTPUT = "test_pairs.json"


def _augmented_source(signature_id: str) -> tuple[str, str]:
    source, separator, source_signature_id = signature_id.partition(AUGMENTED_SOURCE_SEPARATOR)
    if not separator or not source or not source_signature_id:
        raise ValueError(f"augmented signature ID must look like <dataset>___<signature_id>: {signature_id!r}")
    return source, source_signature_id


def join_canonical_augmented_names(
    augmented_signatures: Mapping[str, Any],
    source_signatures: Mapping[str, Mapping[str, Any]],
    canonical_rows: Mapping[str, Any],
    *,
    drop_sources: frozenset[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Give augmented signatures their source datasets' canonical names.

    The augmented training set copies signatures from the benchmark datasets under IDs of
    the form ``<dataset>___<signature_id>``. Its generator changed only one name field: it
    shortened about half of the first names to their first character, and copied middle
    and last names unchanged. This join drops signatures from ``drop_sources``, checks
    that every remaining signature still matches its original benchmark signature, then
    replaces ``first``/``middle``/``last`` with the source dataset's canonical row,
    shortening the canonical first name wherever the augmented one was shortened.

    An original first name of one character cannot show whether it was shortened, so it
    is treated as unshortened and receives the full canonical first name; the report
    counts these rows.

    Args:
        augmented_signatures: Augmented signatures keyed by prefixed signature ID.
        source_signatures: Original (pre-join) benchmark signatures, keyed by dataset and
            then by the dataset's own signature ID.
        canonical_rows: Canonical name rows (``signature_id``, ``first``, ``middle``,
            ``last``) keyed by dataset, in the same format as the benchmark join.
        drop_sources: Source datasets whose augmented signatures are removed.

    Returns:
        The joined signatures and a report with kept/dropped counts, changed fields, and
        counts of re-shortened and one-character first names.

    Raises:
        ValueError: If an ID is malformed, a drop source matches no signature, a source
            dataset or canonical row is missing, an augmented signature no longer matches
            its original benchmark signature, or a canonical first name that must be
            shortened does not start with a letter or digit.
    """

    fields = ("first", "middle", "last")
    kept_by_source: dict[str, list[tuple[str, str]]] = {}
    dropped_by_source: dict[str, int] = {}
    for signature_id, signature in augmented_signatures.items():
        if signature.get("signature_id") != signature_id:
            raise ValueError(f"augmented signature key does not match signature_id: {signature_id!r}")
        source, source_signature_id = _augmented_source(signature_id)
        if source in drop_sources:
            dropped_by_source[source] = dropped_by_source.get(source, 0) + 1
        else:
            kept_by_source.setdefault(source, []).append((signature_id, source_signature_id))

    unmatched_drop_sources = sorted(drop_sources - dropped_by_source.keys())
    if unmatched_drop_sources:
        raise ValueError(f"drop sources match no augmented signatures: {unmatched_drop_sources}")
    missing_sources = sorted(
        source for source in kept_by_source if source not in source_signatures or source not in canonical_rows
    )
    if missing_sources:
        raise ValueError(f"augmented join needs source signatures and canonical names for: {missing_sources}")

    field_changes = {field: 0 for field in fields}
    changed_signatures = 0
    shortened_first_names = 0
    single_character_first_names = 0
    single_character_first_names_lengthened = 0
    mismatches: list[str] = []
    missing_rows: list[str] = []
    unshortenable: list[str] = []
    joined: dict[str, dict[str, Any]] = {}
    for source in sorted(kept_by_source):
        canonical_by_id = _mapping_by_id(
            canonical_rows[source], id_key="signature_id", label=f"{source} canonical rows"
        )
        originals = source_signatures[source]
        for signature_id, source_signature_id in sorted(kept_by_source[source]):
            original = originals.get(source_signature_id)
            canonical = canonical_by_id.get(source_signature_id)
            if original is None or canonical is None:
                missing_rows.append(signature_id)
                continue
            author_info = dict(augmented_signatures[signature_id]["author_info"])
            original_info = original["author_info"]
            original_first = original_info.get("first")
            augmented_first = author_info.get("first")
            shortened = bool(original_first) and len(original_first) > 1 and augmented_first == original_first[0]
            if (
                (augmented_first != original_first and not shortened)
                or author_info.get("middle") != original_info.get("middle")
                or author_info.get("last") != original_info.get("last")
            ):
                mismatches.append(signature_id)
                continue
            canonical_first = canonical["first"]
            if shortened and canonical_first and not canonical_first[0].isalnum():
                unshortenable.append(signature_id)
                continue
            if original_first and len(original_first) == 1:
                single_character_first_names += 1
                single_character_first_names_lengthened += int(bool(canonical_first) and len(canonical_first) > 1)
            new_names = {
                "first": canonical_first[0] if shortened and canonical_first else canonical_first,
                "middle": canonical["middle"],
                "last": canonical["last"],
            }
            changed = False
            for field in fields:
                if author_info.get(field) != new_names[field]:
                    field_changes[field] += 1
                    changed = True
                author_info[field] = new_names[field]
            changed_signatures += int(changed)
            shortened_first_names += int(shortened)
            joined[signature_id] = {**augmented_signatures[signature_id], "author_info": author_info}

    if missing_rows:
        raise ValueError(
            f"{len(missing_rows)} augmented signatures have no original signature or canonical row, "
            f"for example {missing_rows[:5]}"
        )
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} augmented signatures no longer match their original benchmark signatures "
            f"(first/middle/last), for example {mismatches[:5]}"
        )
    if unshortenable:
        raise ValueError(
            f"{len(unshortenable)} shortened augmented first names have canonical first names that do not start "
            f"with a letter or digit, for example {unshortenable[:5]}"
        )
    return joined, {
        "kept_signatures": {source: len(rows) for source, rows in sorted(kept_by_source.items())},
        "dropped_signatures": dict(sorted(dropped_by_source.items())),
        "changed_signatures": changed_signatures,
        "field_changes": field_changes,
        "shortened_first_names": shortened_first_names,
        "single_character_first_names": single_character_first_names,
        "single_character_first_names_lengthened": single_character_first_names_lengthened,
    }


def filter_augmented_pairs(
    pair_path: Path,
    *,
    kept_signature_ids: set[str],
    drop_sources: frozenset[str],
) -> tuple[list[tuple[str, str, str, int]], int]:
    """Drop pair rows that touch a dropped source dataset.

    Args:
        pair_path: Augmented fixed-pair CSV (``pair1``/``pairs1``/``signature_id_1`` style).
        kept_signature_ids: Signature IDs that remain after the join.
        drop_sources: Source datasets whose pairs are removed.

    Returns:
        Kept rows as ``(signature_id_1, signature_id_2, original_label, integer_label)``
        in file order, and the number of dropped rows.

    Raises:
        ValueError: If the file violates the fixed-pair contract, every row is dropped, or
            a kept row references a signature that is not in the joined signatures.
    """

    import pandas as pd

    from s2and.data import _map_fixed_pair_labels

    # Same reader and contract as ANDData fixed pairs; keep_default_na preserves raw labels.
    raw = pd.read_csv(pair_path, sep=",", dtype=str, keep_default_na=False)
    mapped = _map_fixed_pair_labels(raw, pair_path.name)
    left_column, right_column, label_column = list(mapped.columns)
    kept: list[tuple[str, str, str, int]] = []
    unknown: set[str] = set()
    for left, right, original_label, label in zip(
        mapped[left_column].tolist(),
        mapped[right_column].tolist(),
        raw[label_column].tolist(),
        mapped[label_column].tolist(),
        strict=True,
    ):
        if _augmented_source(left)[0] in drop_sources or _augmented_source(right)[0] in drop_sources:
            continue
        unknown.update(signature_id for signature_id in (left, right) if signature_id not in kept_signature_ids)
        kept.append((left, right, original_label, int(label)))
    if unknown:
        raise ValueError(f"{pair_path} references signatures missing from the joined set: {sorted(unknown)[:5]}")
    if not kept:
        raise ValueError(f"every pair in {pair_path} was dropped")
    return kept, len(raw) - len(kept)


def _altered_values(payload: Mapping[str, Any]) -> list[str]:
    values = payload.get("altered_cluster_signatures") or []
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("altered_cluster_signatures must be a list when present")
    return [str(value) for value in values]


def _require_groups_from_service_payload(value: Any) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        values = list(value.values())
        if all(
            not isinstance(item, Mapping) and (not isinstance(item, Sequence) or isinstance(item, str | bytes))
            for item in values
        ):
            groups_by_component: dict[str, list[str]] = {}
            for signature_id, component_key in value.items():
                groups_by_component.setdefault(str(component_key), []).append(str(signature_id))
            return list(groups_by_component.values())
        groups: list[list[str]] = []
        for members in values:
            if not isinstance(members, Sequence) or isinstance(members, str | bytes):
                raise TypeError("cluster_seeds.require must be either signature->cluster or cluster->signature-list")
            groups.append([str(signature_id) for signature_id in members])
        return groups
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        groups: list[list[str]] = []
        for item in value:
            if not isinstance(item, Sequence) or isinstance(item, str | bytes):
                raise TypeError("cluster_seeds.require list entries must be signature-id lists")
            groups.append([str(signature_id) for signature_id in item])
        return groups
    raise TypeError("cluster_seeds.require must be an object or list")


def _disallow_pairs_from_service_payload(value: Any) -> list[tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        pairs: list[tuple[str, str]] = []
        for left, right_values in value.items():
            if isinstance(right_values, Mapping):
                iterable_values = right_values.keys()
            elif isinstance(right_values, Sequence) and not isinstance(right_values, str | bytes):
                iterable_values = right_values
            else:
                raise TypeError("cluster_seeds.disallow object values must be signature-id lists or objects")
            pairs.extend((str(left), str(right)) for right in iterable_values)
        return pairs
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        pairs = []
        for item in value:
            if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) != 2:
                raise TypeError("cluster_seeds.disallow list entries must be pairs")
            pairs.append((str(item[0]), str(item[1])))
        return pairs
    raise TypeError("cluster_seeds.disallow must be an object or list")


def _cluster_seeds_payload(payload: Mapping[str, Any]) -> Any:
    cluster_seeds = payload.get("cluster_seeds")
    if not isinstance(cluster_seeds, Mapping) or not ({"require", "disallow"} & set(cluster_seeds)):
        return cluster_seeds
    unexpected_keys = sorted(set(cluster_seeds).difference({"require", "disallow"}))
    if unexpected_keys:
        raise ValueError(f"service-shaped cluster_seeds contains unsupported keys: {unexpected_keys}")

    legacy_cluster_seeds: dict[str, dict[str, str]] = {}

    def add_pair(left: str, right: str, constraint: str) -> None:
        existing = legacy_cluster_seeds.setdefault(left, {}).get(right)
        if existing is not None and existing != constraint:
            raise ValueError(
                f"cluster_seeds contains conflicting constraints for pair {(left, right)!r}: "
                f"{existing!r} and {constraint!r}"
            )
        legacy_cluster_seeds[left][right] = constraint

    for group in _require_groups_from_service_payload(cluster_seeds.get("require")):
        if not group:
            raise ValueError("cluster_seeds.require cannot contain an empty group")
        root = group[0]
        if len(group) == 1:
            add_pair(root, root, "require")
        else:
            for signature_id in group[1:]:
                add_pair(root, signature_id, "require")
    for left, right in _disallow_pairs_from_service_payload(cluster_seeds.get("disallow")):
        add_pair(left, right, "disallow")
    return legacy_cluster_seeds


def _specter_mapping(payload: Any) -> dict[str, np.ndarray]:
    if isinstance(payload, dict):
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in payload.items()}
    if isinstance(payload, tuple) and len(payload) == 2:
        matrix, keys = payload
        matrix_array = np.asarray(matrix, dtype=np.float32)
        return {str(key): np.asarray(matrix_array[index], dtype=np.float32) for index, key in enumerate(keys)}
    raise TypeError(f"Unsupported SPECTER payload type: {type(payload).__name__}")


def _write_specter_arrow(
    *,
    source_path: Path,
    output_path: Path,
    needed_paper_ids: set[str],
    overwrite: bool,
) -> dict[str, Any]:
    import pyarrow as pa

    if output_path.exists() and not overwrite:
        return {"path": str(output_path), "reused": True}

    with source_path.open("rb") as infile:
        specter_by_paper_id = _specter_mapping(pickle.load(infile))
    selected_items: list[tuple[str, np.ndarray]] = []
    empty_vector_count = 0
    for paper_id, vector in specter_by_paper_id.items():
        if str(paper_id) not in needed_paper_ids:
            continue
        if vector.size == 0:
            empty_vector_count += 1
            continue
        selected_items.append((paper_id, vector))
    if empty_vector_count:
        logger.warning(
            "Dropped %d SPECTER embeddings with zero-size vectors from %s",
            empty_vector_count,
            source_path,
        )
    selected_items.sort(key=lambda item: item[0])
    if not selected_items:
        raise ValueError(f"No SPECTER embeddings from {source_path} matched the dataset papers")

    dimension = int(selected_items[0][1].shape[0])
    for paper_id, vector in selected_items:
        if int(vector.shape[0]) != dimension:
            raise ValueError(
                f"SPECTER dimension mismatch in {source_path}: paper_id={paper_id!r} "
                f"expected={dimension} got={vector.shape[0]}"
            )

    matrix = np.vstack([vector for _paper_id, vector in selected_items]).astype(np.float32, copy=False)
    flat = pa.array(np.ravel(matrix), type=pa.float32())
    table = pa.table(
        {
            "paper_id": pa.array([paper_id for paper_id, _vector in selected_items], type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(flat, dimension),
        }
    )
    from s2and.incremental_linking.feature_block import RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS, write_arrow_ipc_table

    write_arrow_ipc_table(
        table,
        output_path,
        max_record_batch_rows=RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS["specter"],
    )
    return {
        "path": str(output_path),
        "reused": False,
        "row_count": int(table.num_rows),
        "dimension": dimension,
        "source_path": str(source_path),
        "dropped_empty_embedding_count": empty_vector_count,
    }


def _converter_identity(**options: Any) -> dict[str, Any]:
    """Record the converter revision and the options that shaped its output."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        git_head: str | None = git("rev-parse", "HEAD")
        git_dirty: bool | None = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("Converter Git metadata unavailable in %s: %s", _PROJECT_ROOT, exc)
        git_head = None
        git_dirty = None
    return {
        "script": "scripts/convert_to_arrow.py",
        "git_head": git_head,
        "git_dirty": git_dirty,
        "use_orcid_id": True,
        **options,
    }


def _source_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path}")
    return path


def _optional_source_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def benchmark_dataset_sources(source_root: Path, dataset: str) -> RuntimeDatasetSources:
    source_dir = source_root / dataset
    return RuntimeDatasetSources(
        dataset=dataset,
        source_dir=source_dir,
        signatures_path=_source_file(source_dir / f"{dataset}_signatures.json"),
        papers_path=_source_file(source_dir / f"{dataset}_papers.json"),
        clusters_path=_optional_source_file(source_dir / f"{dataset}_clusters.json"),
        specter_path=_optional_source_file(source_dir / f"{dataset}_specter.pickle"),
        specter2_path=_optional_source_file(source_dir / f"{dataset}_specter2.pkl"),
        train_pairs_path=_optional_source_file(source_dir / "train_pairs.csv"),
        val_pairs_path=_optional_source_file(source_dir / "val_pairs.csv"),
        test_pairs_path=_optional_source_file(source_dir / "test_pairs.csv"),
    )


def _validated_fixed_pair_labels(
    sources: RuntimeDatasetSources, signature_ids: set[str]
) -> dict[str, list[tuple[str, str, str]]]:
    """Validate a dataset's fixed-pair label CSVs before they are published.

    Pair-only benchmarks such as MEDLINE carry their labels as ``train_pairs.csv``,
    ``test_pairs.csv`` and an optional ``val_pairs.csv`` instead of clusters. Each file
    must satisfy the ``ANDData`` fixed-pair contract, reference only signatures in the
    dataset, and share no unordered pair with another split.

    Args:
        sources: Source files for the dataset being converted.
        signature_ids: Signature IDs present in the converted dataset.

    Returns:
        Rows with canonical column order keyed by manifest path key (``train_pairs``,
        ``val_pairs``, ``test_pairs``); empty when the dataset has no pair labels.
        Raw label spellings and repeated consistent pairs are preserved.

    Raises:
        ValueError: If only some required splits are present, or a file violates the
            fixed-pair contract.
    """

    import pandas as pd

    from s2and.data import _map_fixed_pair_labels

    candidates = {
        "train_pairs": sources.train_pairs_path,
        "val_pairs": sources.val_pairs_path,
        "test_pairs": sources.test_pairs_path,
    }
    present = {key: path for key, path in candidates.items() if path is not None}
    if not present:
        return {}
    if "train_pairs" not in present or "test_pairs" not in present:
        raise ValueError(
            f"Dataset {sources.dataset!r} fixed-pair labels require train_pairs.csv and test_pairs.csv; "
            f"found {sorted(present)}"
        )
    identities: dict[str, set[tuple[str, str]]] = {}
    rows_by_split: dict[str, list[tuple[str, str, str]]] = {}
    for key, path in present.items():
        csv_text = path.read_text(encoding="utf-8")
        raw = pd.read_csv(StringIO(csv_text), sep=",", dtype=str, keep_default_na=False)
        mapped_input = raw.copy()
        # Preserve ANDData's label inference (e.g. a numeric 1.0 column), while
        # retaining exact string IDs and the published label spelling.
        if "label" in raw:
            mapped_input["label"] = pd.read_csv(StringIO(csv_text), sep=",", usecols=lambda column: column == "label")[
                "label"
            ]
        frame = _map_fixed_pair_labels(mapped_input, key)
        labels: dict[tuple[str, str], int] = {}
        rows: list[tuple[str, str, str]] = []
        for (left, right, label), original_label in zip(
            frame.itertuples(index=False, name=None), raw["label"], strict=True
        ):
            if left == right:
                raise ValueError(f"{path} contains self-pair: {left!r}")
            pair = (left, right) if left < right else (right, left)
            if pair in labels and labels[pair] != int(label):
                raise ValueError(f"{path} contains conflicting labels for unordered pair: {pair!r}")
            labels[pair] = int(label)
            rows.append((left, right, original_label))
        unknown = sorted({signature_id for pair in labels for signature_id in pair} - signature_ids)
        if unknown:
            raise ValueError(f"{path} references signatures absent from dataset {sources.dataset!r}: {unknown[:5]}")
        identities[key] = set(labels)
        rows_by_split[key] = rows
    keys = sorted(identities)
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            overlap = identities[left] & identities[right]
            if overlap:
                raise ValueError(
                    f"Dataset {sources.dataset!r} fixed-pair splits {left!r} and {right!r} share "
                    f"{len(overlap)} unordered pairs, for example {sorted(overlap)[:3]}"
                )
    return rows_by_split


def _conversion_name_counts_index(
    root: Path, index_root: Path | None, *, skip: bool, context: str
) -> tuple[str | None, float]:
    """Check the shared index before preprocessing or writing dataset artifacts."""

    if skip:
        return None, 0.0
    start = time.perf_counter()
    path = require_name_counts_index_artifact(
        root / "name_counts_index" if index_root is None else index_root,
        context=context,
        producer_hint="run python -m scripts.production.counts.generate_name_counts first",
    )
    return path, time.perf_counter() - start


def linker_replay_dataset_sources(raw_root: Path, embeddings_root: Path, dataset: str) -> RuntimeDatasetSources:
    raw_dir = raw_root / dataset
    embeddings_dir = embeddings_root / dataset
    return RuntimeDatasetSources(
        dataset=dataset,
        source_dir=raw_dir,
        signatures_path=_source_file(raw_dir / "signatures.json"),
        papers_path=_source_file(raw_dir / "papers.json"),
        specter2_path=_source_file(embeddings_dir / "specter2.pkl"),
    )


def discover_benchmark_datasets(source_root: Path) -> list[str]:
    discovered: list[str] = []
    for dataset in BENCHMARK_DATASETS:
        source_dir = source_root / dataset
        if source_dir.exists() and (source_dir / f"{dataset}_signatures.json").exists():
            discovered.append(dataset)
    if discovered:
        return discovered
    return [
        child.name for child in sorted(source_root.iterdir()) if child.is_dir() and any(child.glob("*_signatures.json"))
    ]


def discover_linker_replay_datasets(raw_root: Path, embeddings_root: Path) -> list[str]:
    return [
        child.name
        for child in sorted(raw_root.iterdir())
        if child.is_dir()
        and (child / "signatures.json").exists()
        and (child / "papers.json").exists()
        and (embeddings_root / child.name / "specter2.pkl").exists()
    ]


def convert_service_json_to_arrow(
    *,
    input_json: Path,
    output_root: Path,
    dataset_name: str,
    name_counts_index_root: Path | None = None,
    n_jobs: int,
    overwrite: bool,
    skip_name_counts_index: bool,
    copy_source_json: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Write one service-shaped inference request as an Arrow dataset."""

    from s2and.data import ANDData
    from s2and.incremental_linking.feature_block import (
        write_arrow_ipc_table,
        write_raw_arrow_batch_lookup_indexes,
    )
    from scripts.arrow_conversion_helpers import write_raw_planner_arrow_from_anddata

    output_dir = output_root / dataset_name
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory already contains files for dataset {dataset_name!r}: {output_dir}. "
            "Use --overwrite to regenerate it."
        )
    _validate_existing_root_manifest(output_root / "manifest.json")
    name_counts_index_path, write_name_counts_index_seconds = _conversion_name_counts_index(
        output_root, name_counts_index_root, skip=skip_name_counts_index, context="service JSON conversion"
    )

    start = time.perf_counter()
    payload = _load_json(input_json)
    if not isinstance(payload, Mapping):
        raise TypeError("input JSON must contain an object")
    load_seconds = time.perf_counter() - start

    signatures = _mapping_by_id(payload.get("signatures"), id_key="signature_id", label="signatures")
    papers = _mapping_by_id(payload.get("papers"), id_key="paper_id", label="papers")
    altered = _altered_values(payload)
    specter_embeddings = payload.get("paper_embeddings")
    if specter_embeddings is None:
        specter_embeddings = payload.get("specter_embeddings")

    start = time.perf_counter()
    dataset = ANDData(
        signatures=signatures,
        papers=papers,
        name=dataset_name,
        mode="inference",
        clusters=None,
        specter_embeddings=specter_embeddings,
        cluster_seeds=_cluster_seeds_payload(payload),
        altered_cluster_signatures=altered,
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=1000,
        val_pairs_size=1000,
        test_pairs_size=1000,
        n_jobs=n_jobs,
        name_counts_index=None,
        preprocess=True,
        random_seed=42,
        name_tuples=None,
        use_orcid_id=True,
    )
    anddata_seconds = time.perf_counter() - start

    start = time.perf_counter()
    paths = write_raw_planner_arrow_from_anddata(
        dataset,
        output_dir,
        signature_ids=list(dataset.signatures),
        include_specter=specter_embeddings is not None,
        include_empty_cluster_seeds=True,
        overwrite=overwrite,
    )
    write_arrow_seconds = time.perf_counter() - start

    import pyarrow as pa

    altered_arrow_path = output_dir / "altered_cluster_signatures.arrow"
    if overwrite or not altered_arrow_path.exists():
        table = pa.table({"signature_id": pa.array(altered, type=pa.string())})
        write_arrow_ipc_table(table, altered_arrow_path)
    paths["altered_cluster_signatures"] = str(altered_arrow_path)

    if copy_source_json:
        source_paths = {
            "signatures_json": output_dir / "signatures.json",
            "papers_json": output_dir / "papers.json",
            "cluster_seeds_json": output_dir / "cluster_seeds.json",
        }
        if overwrite or not source_paths["signatures_json"].exists():
            _write_json(source_paths["signatures_json"], signatures)
        if overwrite or not source_paths["papers_json"].exists():
            _write_json(source_paths["papers_json"], papers)
        if overwrite or not source_paths["cluster_seeds_json"].exists():
            _write_json(source_paths["cluster_seeds_json"], payload.get("cluster_seeds") or {})
        paths.update({key: str(path) for key, path in source_paths.items()})

    start = time.perf_counter()
    paths, _raw_planner_index_metrics = write_raw_arrow_batch_lookup_indexes(
        paths,
        output_dir,
        overwrite=overwrite,
    )
    write_raw_planner_indexes_seconds = time.perf_counter() - start

    if name_counts_index_path is not None:
        paths["name_counts_index"] = name_counts_index_path

    manifest = build_arrow_artifact_manifest(paths, output_dir)
    validation_metrics: dict[str, Any] = {}
    if validate:
        validation_metrics = validate_arrow_dataset_manifest(
            manifest,
            require_embeddings=specter_embeddings is not None,
            require_name_counts_index=not skip_name_counts_index,
            base_dir=output_dir,
        )
    write_arrow_artifact_manifest(manifest, output_dir)
    _upsert_root_manifest(output_root, dataset_name=dataset_name, dataset_dir=output_dir)
    return {
        **manifest,
        "signature_count": len(dataset.signatures),
        "paper_count": len(dataset.papers),
        "timings_seconds": {
            "load_json_seconds": load_seconds,
            "anddata_seconds": anddata_seconds,
            "write_arrow_seconds": write_arrow_seconds,
            "write_raw_planner_indexes_seconds": write_raw_planner_indexes_seconds,
            "write_name_counts_index_seconds": write_name_counts_index_seconds,
        },
        "validation": validation_metrics,
    }


def convert_runtime_dataset_to_arrow(
    *,
    sources: RuntimeDatasetSources,
    output_dir: Path,
    root_manifest_dir: Path,
    name_counts_index_root: Path | None,
    n_jobs: int,
    overwrite: bool,
    skip_name_counts_index: bool,
    include_empty_cluster_seeds: bool = False,
    selected_embedding: str = "specter2",
    validate: bool = True,
) -> dict[str, Any]:
    """Write one benchmark or linker-replay dataset with one selected embedding table."""

    from s2and.data import ANDData
    from s2and.incremental_linking.feature_block import (
        write_raw_arrow_batch_lookup_indexes,
    )
    from scripts.arrow_conversion_helpers import write_raw_planner_arrow_from_anddata

    dataset_name = sources.dataset
    if selected_embedding not in {"specter", "specter2"}:
        raise ValueError(f"unsupported selected embedding: {selected_embedding!r}")
    embedding_source = sources.specter_path if selected_embedding == "specter" else sources.specter2_path
    if embedding_source is None or not embedding_source.is_file():
        raise FileNotFoundError(
            f"Dataset {dataset_name!r} has no {selected_embedding} source; "
            "canonical benchmark and production conversion requires SPECTER2"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory already contains files for dataset {dataset_name!r}: {output_dir}. "
            "Use --overwrite to regenerate it."
        )
    _validate_existing_root_manifest(root_manifest_dir / "manifest.json")
    name_counts_index_path, write_name_counts_index_seconds = _conversion_name_counts_index(
        root_manifest_dir, name_counts_index_root, skip=skip_name_counts_index, context="runtime dataset conversion"
    )
    converter = _converter_identity(
        selected_embedding=selected_embedding,
        include_empty_cluster_seeds=include_empty_cluster_seeds,
        name_counts_index="skipped" if skip_name_counts_index else "bound",
    )
    provenance_sources = {
        "signatures": sources.signatures_path,
        "papers": sources.papers_path,
        "specter_embeddings": embedding_source,
        **({} if sources.clusters_path is None else {"clusters": sources.clusters_path}),
    }
    source_hashes = {role: _file_sha256(path) for role, path in provenance_sources.items()}
    signatures = _load_json(sources.signatures_path)
    pair_rows = _validated_fixed_pair_labels(sources, set(signatures))

    start = time.perf_counter()
    dataset = ANDData(
        signatures=signatures,
        papers=str(sources.papers_path),
        name=dataset_name,
        mode="train" if sources.clusters_path is not None else "inference",
        specter_embeddings=None,
        clusters=str(sources.clusters_path) if sources.clusters_path is not None else None,
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=100000,
        val_pairs_size=10000,
        test_pairs_size=10000,
        n_jobs=n_jobs,
        name_counts_index=None,
        preprocess=True,
        random_seed=42,
        name_tuples=None,
        use_orcid_id=True,
    )
    anddata_seconds = time.perf_counter() - start

    start = time.perf_counter()
    paths = write_raw_planner_arrow_from_anddata(
        dataset,
        output_dir,
        signature_ids=list(dataset.signatures),
        include_specter=False,
        include_empty_cluster_seeds=include_empty_cluster_seeds,
        overwrite=overwrite,
    )
    write_common_seconds = time.perf_counter() - start

    if sources.clusters_path is not None:
        output_clusters_path = output_dir / f"{dataset_name}_clusters.json"
        if overwrite or not output_clusters_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sources.clusters_path, output_clusters_path)
        paths["clusters"] = str(output_clusters_path)

    for pair_key, rows in pair_rows.items():
        output_pair_path = output_dir / f"{pair_key}.csv"
        with output_pair_path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(["signature_id_1", "signature_id_2", "label"])
            writer.writerows(rows)
        paths[pair_key] = str(output_pair_path)
    if overwrite:
        for pair_key in {"train_pairs", "val_pairs", "test_pairs"} - pair_rows.keys():
            (output_dir / f"{pair_key}.csv").unlink(missing_ok=True)

    needed_paper_ids = {str(signature.paper_id) for signature in dataset.signatures.values()}
    embedding_path = output_dir / f"{selected_embedding}.arrow"
    _write_specter_arrow(
        source_path=embedding_source,
        output_path=embedding_path,
        needed_paper_ids=needed_paper_ids,
        overwrite=overwrite,
    )
    paths["specter"] = str(embedding_path)

    start = time.perf_counter()
    paths, _raw_planner_index_metrics = write_raw_arrow_batch_lookup_indexes(
        paths,
        output_dir,
        overwrite=overwrite,
    )
    write_raw_planner_indexes_seconds = time.perf_counter() - start
    if name_counts_index_path is not None:
        paths["name_counts_index"] = name_counts_index_path

    manifest = build_arrow_artifact_manifest(paths, output_dir)
    validation_metrics: dict[str, Any] = {}
    if validate:
        validation_metrics = validate_arrow_dataset_manifest(
            manifest,
            require_embeddings=True,
            require_name_counts_index=not skip_name_counts_index,
            base_dir=output_dir,
        )
    changed_sources = [role for role, path in provenance_sources.items() if _file_sha256(path) != source_hashes[role]]
    if changed_sources:
        raise ValueError(f"Source files changed during conversion: {changed_sources}; rerun from stable inputs")
    write_arrow_artifact_manifest(manifest, output_dir)
    provenance_path = write_conversion_provenance(
        output_dir,
        sources=provenance_sources,
        converter=converter,
    )
    recorded_sources = _load_json(provenance_path)["sources"]
    if any(recorded_sources[role]["sha256"] != digest for role, digest in source_hashes.items()):
        provenance_path.unlink()
        raise ValueError("Source files changed while writing conversion provenance; rerun from stable inputs")
    _upsert_root_manifest(root_manifest_dir, dataset_name=dataset_name, dataset_dir=output_dir)
    return {
        **manifest,
        "signature_count": len(dataset.signatures),
        "paper_count": len(dataset.papers),
        "timings_seconds": {
            "anddata_seconds": anddata_seconds,
            "write_common_seconds": write_common_seconds,
            "write_raw_planner_indexes_seconds": write_raw_planner_indexes_seconds,
            "write_name_counts_index_seconds": write_name_counts_index_seconds,
        },
        "validation": validation_metrics,
    }


def _read_arrow_table(path: str | Path) -> Any:
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all()


def _table_values(table: Any, column: str) -> list[Any]:
    if column not in table.column_names:
        raise ValueError(f"Arrow table is missing required column {column!r}")
    return table[column].to_pylist()


def _required_string_values(
    table: Any,
    column: str,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[str]:
    values = _table_values(table, column)
    out: list[str] = []
    for row_index, value in enumerate(values):
        if value is None:
            raise ValueError(f"{label} contains null value at row {row_index}")
        text = str(value)
        if not allow_empty and not text:
            raise ValueError(f"{label} contains empty value at row {row_index}")
        out.append(text)
    return out


def _duplicate_values(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def _ensure_unique(values: Sequence[Any], *, label: str) -> None:
    duplicates = _duplicate_values(values)
    if duplicates:
        raise ValueError(f"{label} contains duplicate ids: {duplicates[:10]}")


def _ensure_subset(values: Sequence[Any], allowed: set[str], *, label: str) -> None:
    missing = sorted({str(value) for value in values if str(value) not in allowed})
    if missing:
        raise ValueError(f"{label} contains ids missing from parent table: {missing[:10]}")


def validate_arrow_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    require_embeddings: bool,
    require_name_counts_index: bool,
    base_dir: Path | None = None,
    require_complete_embeddings: bool = False,
) -> dict[str, Any]:
    """Validate the generated Arrow tables and return compact audit metrics."""

    if not isinstance(manifest.get("paths"), Mapping):
        raise ValueError("manifest is missing paths")
    paths = {str(key): str(_resolve_manifest_path(value, base_dir)) for key, value in manifest["paths"].items()}
    required_paths = ["signatures", "papers", "paper_authors"]
    if require_embeddings:
        required_paths.append("specter")
    if require_name_counts_index:
        required_paths.append("name_counts_index")
    missing_paths = [key for key in required_paths if key not in paths or not Path(paths[key]).exists()]
    if missing_paths:
        raise FileNotFoundError(f"manifest is missing required path keys/files: {missing_paths}")

    signatures = _read_arrow_table(paths["signatures"])
    papers = _read_arrow_table(paths["papers"])
    paper_authors = _read_arrow_table(paths["paper_authors"])
    validate_arrow_schema(signatures.schema, table_name="signatures")
    validate_arrow_schema(papers.schema, table_name="papers")
    validate_arrow_schema(paper_authors.schema, table_name="paper_authors")
    signature_ids = _required_string_values(signatures, "signature_id", label="signatures.signature_id")
    signature_paper_ids = _required_string_values(signatures, "paper_id", label="signatures.paper_id")
    signature_author_positions = _table_values(signatures, "author_position")
    if any(position is None for position in signature_author_positions):
        raise ValueError("signatures.author_position contains null value")
    paper_ids = _required_string_values(papers, "paper_id", label="papers.paper_id")
    paper_author_paper_ids = _required_string_values(paper_authors, "paper_id", label="paper_authors.paper_id")
    _required_string_values(
        paper_authors,
        "author_name",
        label="paper_authors.author_name",
        allow_empty=True,
    )
    paper_author_positions = _table_values(paper_authors, "position")
    _ensure_unique(signature_ids, label="signatures.signature_id")
    _ensure_unique(paper_ids, label="papers.paper_id")
    paper_id_set = set(paper_ids)
    signature_id_set = set(signature_ids)
    _ensure_subset(signature_paper_ids, paper_id_set, label="signatures.paper_id")
    _ensure_subset(paper_author_paper_ids, paper_id_set, label="paper_authors.paper_id")
    _ensure_unique(
        [
            f"{paper_id}\x00{position}"
            for paper_id, position in zip(paper_author_paper_ids, paper_author_positions, strict=True)
        ],
        label="paper_authors.(paper_id,position)",
    )

    metrics: dict[str, Any] = {
        "signature_count": int(signatures.num_rows),
        "paper_count": int(papers.num_rows),
        "paper_author_count": int(paper_authors.num_rows),
        "required_paths_present": True,
    }
    if require_embeddings:
        specter = _read_arrow_table(paths["specter"])
        validate_arrow_schema(specter.schema, table_name="specter")
        specter_paper_ids = _required_string_values(specter, "paper_id", label="specter.paper_id")
        _ensure_unique(specter_paper_ids, label="specter.paper_id")
        missing_embeddings = sorted(set(signature_paper_ids) - set(specter_paper_ids))
        metrics["specter_count"] = int(specter.num_rows)
        metrics["missing_specter_paper_count"] = int(len(missing_embeddings))
        metrics["missing_specter_paper_examples"] = missing_embeddings[:10]
        if require_complete_embeddings and missing_embeddings:
            raise ValueError(
                "require_complete_embeddings=True but specter Arrow is missing embeddings for referenced "
                f"paper ids: {missing_embeddings[:10]}"
            )

    cluster_seed_path = paths.get("cluster_seeds")
    if cluster_seed_path is not None and Path(cluster_seed_path).exists():
        cluster_seeds = _read_arrow_table(cluster_seed_path)
        validate_arrow_schema(cluster_seeds.schema, table_name="cluster_seeds")
        seed_signature_ids = [str(value) for value in _table_values(cluster_seeds, "signature_id")]
        seed_cluster_ids = [str(value) for value in _table_values(cluster_seeds, "cluster_id")]
        _ensure_unique(seed_signature_ids, label="cluster_seeds.signature_id")
        _ensure_subset(seed_signature_ids, signature_id_set, label="cluster_seeds.signature_id")
        empty_cluster_ids = [cluster_id for cluster_id in seed_cluster_ids if not cluster_id]
        if empty_cluster_ids:
            raise ValueError("cluster_seeds.cluster_id contains empty values")
        metrics["cluster_seed_count"] = int(cluster_seeds.num_rows)
    else:
        seed_signature_ids = []

    disallow_path = paths.get("cluster_seed_disallows")
    if disallow_path is not None and Path(disallow_path).exists():
        disallows = _read_arrow_table(disallow_path)
        validate_arrow_schema(disallows.schema, table_name="cluster_seed_disallows")
        left_ids = [str(value) for value in _table_values(disallows, "signature_id_1")]
        right_ids = [str(value) for value in _table_values(disallows, "signature_id_2")]
        _ensure_subset(left_ids, signature_id_set, label="cluster_seed_disallows.signature_id_1")
        _ensure_subset(right_ids, signature_id_set, label="cluster_seed_disallows.signature_id_2")
        normalized_pairs: set[tuple[str, str]] = set()
        for left, right in zip(left_ids, right_ids, strict=True):
            if left == right:
                raise ValueError(f"cluster_seed_disallows contains self-pair: {left!r}")
            pair = (left, right) if left < right else (right, left)
            if pair in normalized_pairs:
                raise ValueError(f"cluster_seed_disallows contains duplicate undirected pair: {pair!r}")
            normalized_pairs.add(pair)
        metrics["cluster_seed_disallow_count"] = int(disallows.num_rows)

    altered_path = paths.get("altered_cluster_signatures")
    if altered_path is not None and Path(altered_path).exists():
        altered = _read_arrow_table(altered_path)
        validate_arrow_schema(altered.schema, table_name="altered_cluster_signatures")
        altered_signature_ids = [str(value) for value in _table_values(altered, "signature_id")]
        _ensure_unique(altered_signature_ids, label="altered_cluster_signatures.signature_id")
        _ensure_subset(altered_signature_ids, signature_id_set, label="altered_cluster_signatures.signature_id")
        if altered_signature_ids:
            _ensure_subset(
                altered_signature_ids,
                set(seed_signature_ids),
                label="altered_cluster_signatures.signature_id",
            )
        metrics["altered_cluster_signature_count"] = int(altered.num_rows)

    if require_name_counts_index:
        require_name_counts_index_artifact(
            paths["name_counts_index"],
            context="convert_to_arrow dataset validation",
            producer_hint="run python -m scripts.production.counts.generate_name_counts or rebuild the release bundle",
        )
        metrics["name_counts_index_present"] = True

    from s2and.incremental_linking.feature_block import (
        RAW_PLANNER_ARROW_KEY_COLUMNS,
        raw_planner_arrow_physical_layout,
        validate_arrow_batch_lookup_index,
    )

    physical_layout = raw_planner_arrow_physical_layout(paths)
    for table_name, raw_layout in physical_layout["tables"].items():
        table_key = str(table_name)
        index_key = str(raw_layout["batch_index_path_key"])
        if not bool(raw_layout["batch_index_present"]):
            raise FileNotFoundError(f"manifest is missing required raw-planner batch index: {index_key}")
        key_column = RAW_PLANNER_ARROW_KEY_COLUMNS[table_key]
        validate_arrow_batch_lookup_index(
            paths[table_key],
            paths[index_key],
            key_column=key_column,
            expected_row_count=int(raw_layout["row_count"]),
        )
    return metrics


def validate_arrow_dataset_dir(
    dataset_dir: Path,
    *,
    require_embeddings: bool,
    require_name_counts_index: bool,
    require_complete_embeddings: bool = False,
) -> dict[str, Any]:
    with ArrowDataset.open(
        dataset_dir,
        require_specter=require_embeddings,
        require_name_counts_index=require_name_counts_index,
    ):
        pass
    manifest = _load_json(dataset_dir / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise TypeError(f"dataset manifest must contain an object: {dataset_dir / 'manifest.json'}")
    return validate_arrow_dataset_manifest(
        manifest,
        require_embeddings=require_embeddings,
        require_name_counts_index=require_name_counts_index,
        require_complete_embeddings=require_complete_embeddings,
        base_dir=dataset_dir,
    )


def _print_report(report: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {
                "dataset": report["dataset"],
                "signature_count": report["signature_count"],
                "paper_count": report["paper_count"],
                "paths": report["paths"],
                "timings_seconds": report.get("timings_seconds", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_service_json(args: argparse.Namespace) -> None:
    dataset_name = str(args.dataset_name or args.input_json.stem)
    report = convert_service_json_to_arrow(
        input_json=args.input_json,
        output_root=args.output_root,
        dataset_name=dataset_name,
        name_counts_index_root=args.name_counts_index_root,
        n_jobs=int(args.n_jobs),
        overwrite=bool(args.overwrite),
        skip_name_counts_index=bool(args.skip_name_counts_index),
        copy_source_json=bool(args.copy_source_json),
        validate=not bool(args.skip_validation),
    )
    _print_report({**report, "dataset": dataset_name})


def _run_join_canonical_names(args: argparse.Namespace) -> None:
    signatures = _load_json(args.signatures)
    canonical_rows = _load_json(args.canonical_names)
    if not isinstance(signatures, Mapping):
        raise TypeError("benchmark signatures must be a JSON object keyed by signature_id")
    if not isinstance(canonical_rows, list):
        raise TypeError("canonical names must be a JSON list")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    joined, report = join_canonical_benchmark_names(signatures, canonical_rows)
    _write_json(args.output, joined)
    print(json.dumps({**report, "output": str(args.output)}, indent=2, sort_keys=True))


def _benchmark_signatures_path(source_root: Path, dataset: str) -> Path:
    for name in (f"{dataset}_signatures.json", "signatures.json"):
        path = source_root / dataset / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No {dataset}_signatures.json or signatures.json under {source_root / dataset}")


def _run_join_canonical_augmented_names(args: argparse.Namespace) -> None:
    augmented_dir: Path = args.augmented_dir
    output_dir: Path = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    drop_sources = frozenset(args.drop_sources or ())
    augmented_signatures = _load_json(augmented_dir / AUGMENTED_SIGNATURES_FILE)
    if not isinstance(augmented_signatures, Mapping):
        raise TypeError("augmented signatures must be a JSON object keyed by signature_id")
    pair_files = [*AUGMENTED_TRAINING_PAIR_FILES, AUGMENTED_TEST_PAIR_FILE]
    missing_pairs = [name for name in pair_files if not (augmented_dir / name).is_file()]
    if missing_pairs:
        raise FileNotFoundError(f"augmented directory is missing pair files: {missing_pairs}")

    sources = sorted({_augmented_source(str(signature_id))[0] for signature_id in augmented_signatures} - drop_sources)
    source_signatures: dict[str, Mapping[str, Any]] = {}
    canonical_rows: dict[str, Any] = {}
    for source in sources:
        signatures = _load_json(_benchmark_signatures_path(args.source_root, source))
        if not isinstance(signatures, Mapping):
            raise TypeError(f"{source} benchmark signatures must be a JSON object keyed by signature_id")
        rows = _load_json(args.canonical_names_root / f"{source}.json")
        if not isinstance(rows, list):
            raise TypeError(f"{source} canonical names must be a JSON list")
        source_signatures[source] = signatures
        canonical_rows[source] = rows

    joined, report = join_canonical_augmented_names(
        augmented_signatures,
        source_signatures,
        canonical_rows,
        drop_sources=drop_sources,
    )
    filtered = {}
    report["pairs"] = {}
    for name in pair_files:
        rows, dropped = filter_augmented_pairs(
            augmented_dir / name,
            kept_signature_ids=set(joined),
            drop_sources=drop_sources,
        )
        filtered[name] = rows
        report["pairs"][name] = {"kept": len(rows), "dropped": dropped}
    report = {**report, "drop_sources": sorted(drop_sources), "output_dir": str(output_dir)}

    # Publish all outputs together: build them in a sibling directory, then rename it.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        _write_json(staging_dir / AUGMENTED_SIGNATURES_FILE, joined)
        for name in AUGMENTED_TRAINING_PAIR_FILES:
            with (staging_dir / name).open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target, lineterminator="\n")
                writer.writerow(["signature_id_1", "signature_id_2", "label"])
                writer.writerows((left, right, original_label) for left, right, original_label, _ in filtered[name])
        _write_json(
            staging_dir / AUGMENTED_TEST_PAIRS_OUTPUT,
            [
                {"signature_id_1": left, "signature_id_2": right, "label": label}
                for left, right, _, label in filtered[AUGMENTED_TEST_PAIR_FILE]
            ],
        )
        _write_json(staging_dir / "join_report.json", report)
        staging_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


def _read_augmented_pair_csv(path: Path) -> tuple[list[str], list[list[str]], tuple[int, int]]:
    """Validate fixed-pair labels while retaining all CSV cell values and rows."""
    import pandas as pd

    from s2and.data import _map_fixed_pair_labels

    with path.open(encoding="utf-8", newline="") as source:
        csv_text = source.read()
    records = list(csv.reader(StringIO(csv_text)))
    if not records or len(set(records[0])) != len(records[0]):
        raise ValueError(f"{path} must have a unique fixed-pair CSV header")
    header, rows = records[0], records[1:]
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"{path} contains a row whose width differs from its header")
    mapped = _map_fixed_pair_labels(pd.read_csv(StringIO(csv_text)), path.name)
    return header, rows, (header.index(str(mapped.columns[0])), header.index(str(mapped.columns[1])))


def _run_filter_augmented_pairs(args: argparse.Namespace) -> None:
    """Write reviewed-input candidates without changing the draft release or sources."""
    from scripts.production.model.augmented_leakage import (
        PRODUCTION_AUGMENTED_SOURCES,
        training_members_from_sources,
    )
    from scripts.production.model.release_pairwise import (
        _block_signature_ids,
        _csv_pair_identities,
        _heldout_pair_identities,
        _signature_ids,
    )

    output_root: Path = args.output_root
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    if args.random_seed < 0:
        raise ValueError("--random-seed must be non-negative")
    release_path = args.release.resolve()
    source_paths: dict[str, Path] = {"release": release_path}
    source_hashes: dict[str, str] = {"release": _file_sha256(release_path)}
    draft = _load_json(release_path)

    def source(role: str, raw_path: Any) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{role} must name a source path in the draft release")
        path = _source_file(_resolve_manifest_path(raw_path, release_path.parent).resolve())
        source_paths[role] = path
        source_hashes[role] = _file_sha256(path)
        return path

    models = draft["model"]["datasets"]
    augmented = models["augmented"]
    train_path = source("model.augmented.train_pairs", augmented["train_pairs"])
    val_path = source("model.augmented.val_pairs", augmented["val_pairs"])
    val_bytes = val_path.read_bytes()
    test_path = source("evaluation.augmented.pairs", draft["evaluation"]["pairwise"]["augmented"]["pairs"])
    train_header, train_rows, train_indices = _read_augmented_pair_csv(train_path)
    _, val_rows, val_indices = _read_augmented_pair_csv(val_path)
    test_records = _load_json(test_path)
    _heldout_pair_identities(test_path, context="augmented evaluation pairs")

    def identity(signature_id: str) -> tuple[str, str]:
        name, original_id = _augmented_source(signature_id)
        if name not in PRODUCTION_AUGMENTED_SOURCES:
            raise ValueError(f"Unknown augmented source prefix {name!r} in {signature_id!r}")
        return name, original_id

    train_identities = [tuple(identity(row[index]) for index in train_indices) for row in train_rows]
    for row in val_rows:
        for index in val_indices:
            identity(row[index])
    test_identities = [(identity(row["signature_id_1"]), identity(row["signature_id_2"])) for row in test_records]
    required_sources = {name for row in train_identities for name, _ in row} - {"medline"}
    missing_sources = required_sources - models.keys()
    if missing_sources:
        raise ValueError(f"Draft model is missing augmented training source datasets: {sorted(missing_sources)}")

    training: dict[str, set[str]] = {}
    development: dict[str, set[str]] = {}
    for name, files in models.items():
        if name == "augmented":
            continue
        if "clusters" in files:
            model_files = {
                role: source(f"model.{name}.{role}", files[role]) for role in ("signatures", "papers", "clusters")
            }
            training[name] = training_members_from_sources(name, model_files, random_seed=args.random_seed)
            development[name] = _signature_ids(model_files["signatures"], context=f"model {name!r}")
        else:
            split_members = {
                split: {
                    sid
                    for pair in _csv_pair_identities(
                        source(f"model.{name}.{split}", files[split]), context=f"{name!r} {split}"
                    )
                    for sid in pair
                }
                for split in ("train_pairs", "val_pairs")
            }
            training[name] = split_members["train_pairs"]
            development[name] = split_members["train_pairs"] | split_members["val_pairs"]

    external_heldout: dict[str, set[str]] = {}
    for name, files in draft["evaluation"]["pairwise"].items():
        if name != "augmented":
            pairs = _heldout_pair_identities(
                source(f"evaluation.pairwise.{name}.pairs", files["pairs"]), context=f"{name!r} evaluation"
            )
            external_heldout.setdefault(name, set()).update(sid for pair in pairs for sid in pair)
    for name, files in draft["evaluation"].get("cluster", {}).items():
        external_heldout.setdefault(name, set()).update(
            _block_signature_ids(
                source(f"evaluation.cluster.{name}.blocks", files["blocks"]), context=f"{name!r} evaluation"
            )
        )

    def select(
        identities: Sequence[tuple[tuple[str, str], ...]], raw_labels: list[str], *, split: str
    ) -> tuple[list[int], dict[str, Any]]:
        kept: list[int] = []
        reasons: dict[str, int] = {}
        retained_labels: dict[str, int] = {}
        removed_labels: dict[str, int] = {}
        for index, row in enumerate(identities):
            row_reasons: set[str] = set()
            for name, original_id in row:
                if split == "train":
                    if name in training and original_id not in training[name]:
                        row_reasons.add(f"{name}:outside_training_split")
                    if original_id in external_heldout.get(name, set()):
                        row_reasons.add(f"{name}:external_heldout")
                elif original_id in development.get(name, set()):
                    row_reasons.add(f"{name}:model_development_population")
            if row_reasons:
                removed_labels[raw_labels[index]] = removed_labels.get(raw_labels[index], 0) + 1
                for reason in row_reasons:
                    reasons[reason] = reasons.get(reason, 0) + 1
            else:
                kept.append(index)
                retained_labels[raw_labels[index]] = retained_labels.get(raw_labels[index], 0) + 1
        return kept, {
            "input_rows": len(identities),
            "retained_rows": len(kept),
            "removed_rows": len(identities) - len(kept),
            "removed_rows_by_reason": dict(sorted(reasons.items())),
            "retained_rows_by_raw_label": dict(sorted(retained_labels.items())),
            "removed_rows_by_raw_label": dict(sorted(removed_labels.items())),
        }

    train_kept, train_report = select(
        train_identities, [row[train_header.index("label")] for row in train_rows], split="train"
    )
    test_kept, test_report = select(test_identities, [str(row["label"]) for row in test_records], split="test")
    report = {
        "random_seed": args.random_seed,
        "sources": {
            role: {"path": str(path), "sha256": source_hashes[role]} for role, path in sorted(source_paths.items())
        },
        "train": train_report,
        "validation": {"input_rows": len(val_rows), "retained_rows": len(val_rows), "removed_rows": 0},
        "test": test_report,
        "reason_counts_note": "A removed row can have multiple reasons; removed_rows counts each row once.",
    }
    if not train_kept or not test_kept:
        raise ValueError(
            "Filtering leaves no training or no held-out pairs. Review the model/evaluation population design; "
            f"do not bypass the leakage guards. Counts: {json.dumps(report, sort_keys=True)}"
        )
    changed = [role for role, path in source_paths.items() if _file_sha256(path) != source_hashes[role]]
    if changed:
        raise ValueError(f"Sources changed while filtering augmented pairs: {changed}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_root.name}.staging-", dir=output_root.parent) as staging_raw:
        staging = Path(staging_raw)
        with (staging / "train_pairs.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(train_header)
            writer.writerows(train_rows[index] for index in train_kept)
        (staging / "val_pairs.csv").write_bytes(val_bytes)
        kept_test_records = [test_records[index] for index in test_kept]
        with (staging / "test_pairs.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(["signature_id_1", "signature_id_2", "label"])
            writer.writerows((row["signature_id_1"], row["signature_id_2"], row["label"]) for row in kept_test_records)
        (staging / "test_pairs.json").write_text(json.dumps(kept_test_records, indent=2) + "\n", encoding="utf-8")
        report["outputs"] = {
            name: {"path": str(output_root.resolve() / name), "sha256": _file_sha256(staging / name)}
            for name in ("train_pairs.csv", "val_pairs.csv", "test_pairs.csv", "test_pairs.json")
        }
        _write_json(staging / "filter_report.json", report)
        staging.rename(output_root)
    print(json.dumps(report, indent=2, sort_keys=True))


def _selected_runtime_dataset_names(
    *,
    datasets: Sequence[str] | None,
    run_full: bool,
    discover: Callable[[], Sequence[str]],
    command: str,
) -> list[str]:
    if datasets is not None:
        return [str(dataset) for dataset in datasets]
    if run_full:
        return [str(dataset) for dataset in discover()]
    raise ValueError(f"{command} requires --datasets DATASET... for a bounded run or --run-full for full discovery")


def _run_benchmark(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_names = _selected_runtime_dataset_names(
        datasets=getattr(args, "datasets", None),
        run_full=bool(getattr(args, "run_full", False)),
        discover=lambda: discover_benchmark_datasets(args.source_root),
        command="benchmark",
    )
    if not dataset_names:
        raise ValueError(f"No benchmark datasets found under {args.source_root}")
    for dataset_name in dataset_names:
        start = time.perf_counter()
        report = convert_runtime_dataset_to_arrow(
            sources=benchmark_dataset_sources(args.source_root, dataset_name),
            output_dir=output_root / dataset_name,
            root_manifest_dir=output_root,
            name_counts_index_root=args.name_counts_index_root,
            n_jobs=int(args.n_jobs),
            overwrite=bool(args.overwrite),
            skip_name_counts_index=bool(args.skip_name_counts_index),
            selected_embedding="specter2",
            validate=not bool(args.skip_validation),
        )
        report["total_seconds"] = time.perf_counter() - start
        print(json.dumps({"dataset": dataset_name, "total_seconds": report["total_seconds"]}, sort_keys=True))
    print(json.dumps({"datasets": list(dataset_names)}, indent=2, sort_keys=True))


def _run_linker_replay(args: argparse.Namespace) -> None:
    output_root = args.output_root
    datasets_root = output_root / "datasets"
    datasets_root.mkdir(parents=True, exist_ok=True)
    dataset_names = _selected_runtime_dataset_names(
        datasets=getattr(args, "datasets", None),
        run_full=bool(getattr(args, "run_full", False)),
        discover=lambda: discover_linker_replay_datasets(args.raw_root, args.embeddings_root),
        command="linker-replay",
    )
    if not dataset_names:
        raise ValueError(f"No linker replay datasets found under {args.raw_root}")
    for dataset_name in dataset_names:
        start = time.perf_counter()
        report = convert_runtime_dataset_to_arrow(
            sources=linker_replay_dataset_sources(args.raw_root, args.embeddings_root, dataset_name),
            output_dir=datasets_root / dataset_name,
            root_manifest_dir=output_root,
            name_counts_index_root=args.name_counts_index_root,
            n_jobs=int(args.n_jobs),
            overwrite=bool(args.overwrite),
            skip_name_counts_index=bool(args.skip_name_counts_index),
            selected_embedding="specter2",
            validate=not bool(args.skip_validation),
        )
        report["total_seconds"] = time.perf_counter() - start
        print(json.dumps({"dataset": dataset_name, "total_seconds": report["total_seconds"]}, sort_keys=True))
    print(json.dumps({"datasets": list(dataset_names)}, indent=2, sort_keys=True))


def _run_validate(args: argparse.Namespace) -> None:
    metrics = validate_arrow_dataset_dir(
        args.dataset_dir,
        require_embeddings=bool(args.require_embeddings),
        require_name_counts_index=bool(args.require_name_counts_index),
        require_complete_embeddings=bool(args.require_complete_embeddings),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _add_common_runtime_args(parser: argparse.ArgumentParser, *, default_n_jobs: int) -> None:
    parser.add_argument(
        "--name-counts-index-root",
        type=Path,
        default=None,
        help="Manifest-backed name-count index directory (default: name_counts_index/ under the output root).",
    )
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-name-counts-index", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")


def _add_runtime_dataset_selection_args(parser: argparse.ArgumentParser) -> None:
    dataset_selection = parser.add_mutually_exclusive_group(required=True)
    dataset_selection.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Convert only the named datasets.",
    )
    dataset_selection.add_argument(
        "--run-full",
        action="store_true",
        help="Discover and convert every eligible dataset under the configured roots.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    service = subparsers.add_parser("service-json", help="Convert one service-shaped inference JSON payload.")
    service.add_argument("--input-json", type=Path, required=True)
    service.add_argument("--output-root", type=Path, required=True)
    service.add_argument("--dataset-name", default=None)
    service.add_argument("--copy-source-json", action="store_true")
    _add_common_runtime_args(service, default_n_jobs=4)
    service.set_defaults(func=_run_service_json)

    canonical_names = subparsers.add_parser(
        "join-canonical-names",
        help="Replace benchmark author name fields from canonical rows joined by signature ID.",
    )
    canonical_names.add_argument("--signatures", type=Path, required=True)
    canonical_names.add_argument("--canonical-names", type=Path, required=True)
    canonical_names.add_argument("--output", type=Path, required=True)
    canonical_names.set_defaults(func=_run_join_canonical_names)

    augmented_names = subparsers.add_parser(
        "join-canonical-augmented-names",
        help=(
            "Give the augmented training set its source datasets' canonical names, dropping signatures and pairs "
            "from --drop-sources."
        ),
    )
    augmented_names.add_argument(
        "--augmented-dir",
        type=Path,
        required=True,
        help="Directory with augmented_signatures.json, train_pairs.csv, val_pairs.csv, and test_pairs.csv.",
    )
    augmented_names.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Original (pre-join) benchmark root with <dataset>/<dataset>_signatures.json or signatures.json.",
    )
    augmented_names.add_argument(
        "--canonical-names-root",
        type=Path,
        required=True,
        help="Directory with one <dataset>.json canonical-name export per source dataset.",
    )
    augmented_names.add_argument(
        "--drop-sources",
        nargs="*",
        default=[],
        help="Source datasets whose augmented signatures and pairs are removed (v1.3: orcid).",
    )
    augmented_names.add_argument("--output-dir", type=Path, required=True, help="Fresh output directory.")
    augmented_names.set_defaults(func=_run_join_canonical_augmented_names)

    filtered_pairs = subparsers.add_parser(
        "filter-augmented-pairs", help="Filter augmented pairs against a draft release's frozen source populations."
    )
    filtered_pairs.add_argument("--release", type=Path, required=True)
    filtered_pairs.add_argument("--random-seed", type=int, required=True)
    filtered_pairs.add_argument(
        "--output-root", type=Path, required=True, help="Fresh output directory; never overwritten."
    )
    filtered_pairs.set_defaults(func=_run_filter_augmented_pairs)

    benchmark = subparsers.add_parser("benchmark", help="Convert benchmark dataset JSON/pickle files.")
    benchmark.add_argument("--source-root", type=Path, required=True)
    benchmark.add_argument("--output-root", type=Path, required=True)
    _add_runtime_dataset_selection_args(benchmark)
    _add_common_runtime_args(benchmark, default_n_jobs=20)
    benchmark.set_defaults(func=_run_benchmark)

    linker_replay = subparsers.add_parser("linker-replay", help="Convert linker replay raw JSON plus SPECTER2 files.")
    linker_replay.add_argument("--raw-root", type=Path, required=True)
    linker_replay.add_argument("--embeddings-root", type=Path, required=True)
    linker_replay.add_argument("--output-root", type=Path, required=True)
    _add_runtime_dataset_selection_args(linker_replay)
    _add_common_runtime_args(linker_replay, default_n_jobs=20)
    linker_replay.set_defaults(func=_run_linker_replay)

    validate = subparsers.add_parser("validate", help="Validate one generated Arrow dataset manifest.")
    validate.add_argument("--dataset-dir", type=Path, required=True)
    validate.add_argument("--require-embeddings", action="store_true")
    validate.add_argument("--require-name-counts-index", action="store_true")
    validate.add_argument(
        "--require-complete-embeddings",
        action="store_true",
        help="Fail if any referenced paper is missing from the embedding table.",
    )
    validate.set_defaults(func=_run_validate)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

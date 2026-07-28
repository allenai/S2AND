"""Convert S2AND runtime inputs into direct-Rust Arrow artifacts.

The runtime bundle writer emits bounded Arrow IPC file-format tables plus the
current S2AND raw-planner batch-index sidecars (S2ABI002).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
    )


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
    output_dir.mkdir(parents=True, exist_ok=True)

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

    write_name_counts_index_seconds = 0.0
    if not skip_name_counts_index:
        start = time.perf_counter()
        index_root = output_root if name_counts_index_root is None else name_counts_index_root
        name_counts_index_path = require_name_counts_index_artifact(
            Path(index_root) / "name_counts_index",
            context="service JSON conversion",
            producer_hint="run python -m scripts.production.counts.generate_name_counts first",
        )
        write_name_counts_index_seconds = time.perf_counter() - start
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
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    dataset = ANDData(
        signatures=str(sources.signatures_path),
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
    write_name_counts_index_seconds = 0.0
    if not skip_name_counts_index:
        start = time.perf_counter()
        index_root = root_manifest_dir if name_counts_index_root is None else name_counts_index_root
        name_counts_index_path = require_name_counts_index_artifact(
            Path(index_root) / "name_counts_index",
            context="runtime dataset conversion",
            producer_hint="run python -m scripts.production.counts.generate_name_counts first",
        )
        write_name_counts_index_seconds = time.perf_counter() - start
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
    write_arrow_artifact_manifest(manifest, output_dir)
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
    _print_report(report)


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
    reports = []
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
        reports.append(report)
        print(json.dumps({"dataset": dataset_name, "total_seconds": report["total_seconds"]}, sort_keys=True))
    print(json.dumps({"datasets": [report["dataset"] for report in reports]}, indent=2, sort_keys=True))


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
    reports = []
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
        reports.append(report)
        print(json.dumps({"dataset": dataset_name, "total_seconds": report["total_seconds"]}, sort_keys=True))
    print(json.dumps({"datasets": [report["dataset"] for report in reports]}, indent=2, sort_keys=True))


def _run_validate(args: argparse.Namespace) -> None:
    metrics = validate_arrow_dataset_dir(
        args.dataset_dir,
        require_embeddings=bool(args.require_embeddings),
        require_name_counts_index=bool(args.require_name_counts_index),
        require_complete_embeddings=bool(args.require_complete_embeddings),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _add_common_runtime_args(parser: argparse.ArgumentParser, *, default_n_jobs: int) -> None:
    parser.add_argument("--name-counts-index-root", type=Path, default=None)
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

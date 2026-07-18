"""Extract auditable signature pairs from the linker replay bundle.

The linker's labels are component-level decisions, not pair labels.  This
module therefore exposes two deliberately separate catalogs:

* ``strict`` contains only pairs whose labels can be independently derived
  from public gold clusters or an exact normalized ORCID match.
* ``linker_component_proxy`` contains bounded negative proxies expanded only
  from query/component relationships whose observed labels are all 0.  A
  relationship with any positive observation is excluded, and no component
  label is ever broadcast as a positive pair label.

Label and component parquet files are read only through paths declared in the
bundle's ``assets`` section.  This prevents stale files next to a bundle from
silently entering an ablation.  Arrow signature metadata is resolved only
through the bundle runtime contract and each dataset's declared manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

from s2and.text import normalize_orcid_compact

PAIR_COLUMNS = (
    "source_domain",
    "source_family",
    "pair1",
    "pair2",
    "label",
    "label_rule",
    "origin",
    "group_id",
)
PROVENANCE_COLUMNS = (
    "query_signature_id",
    "query_group_id",
    "candidate_component_key",
)
OUTPUT_COLUMNS = (*PAIR_COLUMNS, *PROVENANCE_COLUMNS)

PUBLIC_GOLD_LABEL_RULE = "linker_public_gold_cluster_relabel"
BIG_BLOCK_ORCID_LABEL_RULE = "linker_big_block_same_normalized_orcid"
LINKER_COMPONENT_PROXY_LABEL_RULE = "linker_component_proxy_label0"

_LABEL_COLUMNS = (
    "dataset",
    "query_group_id",
    "query_signature_id",
    "candidate_component_key",
    "label",
)
_MEMBER_COLUMNS = ("dataset", "candidate_component_key", "member_index", "signature_id")
_PAIR_KEY_COLUMNS = ("source_domain", "pair1", "pair2")


@dataclass(frozen=True)
class LinkerPairCatalog:
    """Strict and explicitly weak linker-derived pair catalogs."""

    strict: pd.DataFrame
    linker_component_proxy: pd.DataFrame


def linker_signature_input_paths(
    bundle_json_path: str | Path,
    *,
    public_datasets: Collection[str],
    big_block_datasets: Collection[str],
) -> dict[str, Path]:
    """Resolve Arrow files whose contents can affect linker pair extraction.

    Public domains consume their bundle Arrow signature table only when a
    referenced candidate key has the Rust ``block::component`` shape.  Big
    blocks additionally consume it for exact ORCID labels.  Both the dataset
    manifest and the signature table are returned so catalog resume identity
    is bound to the declaration as well as its resolved contents.
    """

    bundle_path = Path(bundle_json_path).resolve()
    bundle_root = bundle_path.parent
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    public = {str(dataset) for dataset in public_datasets}
    big = {str(dataset) for dataset in big_block_datasets}
    overlap = sorted(public & big)
    if overlap:
        raise ValueError(f"Datasets cannot be both public and big-block domains: {overlap}")
    selected = public | big
    labels = _read_declared_label_rows(payload, bundle_root, selected)

    required_datasets: set[str] = set()
    for dataset in sorted(selected):
        dataset_labels = labels.loc[labels["dataset"].eq(dataset)]
        if dataset_labels.empty:
            continue
        if dataset in big or _labels_require_block_metadata(dataset_labels):
            required_datasets.add(dataset)

    paths: dict[str, Path] = {}
    for dataset in sorted(required_datasets):
        manifest_path, signatures_path = _declared_signature_paths(payload, bundle_root, dataset)
        paths[f"dataset_manifest.{dataset}"] = manifest_path
        paths[f"signatures.{dataset}"] = signatures_path
    return paths


def extract_linker_pair_catalog(
    bundle_json_path: str | Path,
    *,
    public_gold_cluster_paths: Mapping[str, str | Path],
    big_block_datasets: Collection[str],
    proxy_negatives_per_query: int = 2,
    proxy_negatives_per_domain: int | None = 10_000,
    seed: int = 13,
) -> LinkerPairCatalog:
    """Extract strict and proxy pairs from a linker replay bundle.

    Args:
        bundle_json_path: Path to the replay bundle's ``bundle.json``.
        public_gold_cluster_paths: Public dataset name to external gold
            ``clusters.json`` path.  These dataset names define the public
            domains to expand and relabel.
        big_block_datasets: Linker big-block domains without complete gold
            clusters.  Their strict catalog contains positive pairs only when
            both signatures have the same non-null normalized ORCID.
        proxy_negatives_per_query: Maximum component proxy pairs per query
            signature when every observed label for the query/component
            relationship is 0.  Set to zero to disable the proxy catalog.
        proxy_negatives_per_domain: Maximum proxy pairs per domain after
            query-level capping, or ``None`` for no domain cap.
        seed: Stable hash seed used to choose bounded proxy pairs.

    Returns:
        A catalog with canonical pandas DataFrames.  Both frames contain
        ``PAIR_COLUMNS`` followed by explicit linker provenance columns.

    Raises:
        ValueError: If the bundle contract, labels, gold clusters, or final
            pair labels are inconsistent.
        KeyError: If a requested dataset lacks a declared component asset.
        FileNotFoundError: If a declared asset does not exist.
    """

    if proxy_negatives_per_query < 0:
        raise ValueError("proxy_negatives_per_query must be >= 0")
    if proxy_negatives_per_domain is not None and proxy_negatives_per_domain < 0:
        raise ValueError("proxy_negatives_per_domain must be >= 0 or None")

    bundle_path = Path(bundle_json_path).resolve()
    bundle_root = bundle_path.parent
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))

    public_datasets = {str(dataset) for dataset in public_gold_cluster_paths}
    big_datasets = {str(dataset) for dataset in big_block_datasets}
    overlap = sorted(public_datasets & big_datasets)
    if overlap:
        raise ValueError(f"Datasets cannot be both public and big-block domains: {overlap}")
    selected_datasets = public_datasets | big_datasets

    component_assets = _declared_component_assets(payload)
    missing_component_assets = sorted(selected_datasets - set(component_assets))
    if missing_component_assets:
        raise KeyError(f"No declared candidate-member asset for datasets: {missing_component_assets}")

    labels = _read_declared_label_rows(payload, bundle_root, selected_datasets)
    memberships = {
        dataset: _read_declared_members(
            bundle_root,
            component_assets[dataset],
            dataset=dataset,
        )
        for dataset in sorted(selected_datasets)
    }
    _validate_referenced_components(labels, memberships)

    signature_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in sorted(selected_datasets):
        dataset_labels = labels.loc[labels["dataset"].eq(dataset)]
        needs_block_metadata = _labels_require_block_metadata(dataset_labels)
        required_signature_columns: set[str] = set()
        if dataset in big_datasets and not dataset_labels.empty:
            required_signature_columns.add("author_orcid")
        if needs_block_metadata:
            required_signature_columns.add("author_block")
        if not required_signature_columns:
            continue
        metadata = _read_declared_signature_metadata(
            payload,
            bundle_root,
            dataset,
            _needed_signature_ids(dataset_labels, memberships[dataset]),
            required_columns=required_signature_columns,
        )
        signature_metadata[dataset] = metadata
        if needs_block_metadata:
            memberships[dataset] = _block_local_memberships(
                dataset_labels,
                memberships[dataset],
                metadata,
            )

    strict_rows: list[dict[str, Any]] = []
    for dataset in sorted(public_datasets):
        gold = _read_gold_signature_to_cluster(public_gold_cluster_paths[dataset])
        dataset_labels = labels.loc[labels["dataset"].eq(dataset)]
        strict_rows.extend(_public_gold_rows(dataset, dataset_labels, memberships[dataset], gold))

    for dataset in sorted(big_datasets):
        dataset_labels = labels.loc[labels["dataset"].eq(dataset)]
        needed_signature_ids = _needed_signature_ids(dataset_labels, memberships[dataset])
        metadata = signature_metadata.get(dataset, {})
        signature_orcids = {
            signature_id: normalize_orcid_compact(metadata[signature_id]["author_orcid"])
            for signature_id in needed_signature_ids
        }
        strict_rows.extend(_big_block_orcid_rows(dataset, dataset_labels, memberships[dataset], signature_orcids))

    strict = canonicalize_pair_rows(_rows_frame(strict_rows))
    strict_keys = {
        (str(row.source_domain), str(row.pair1), str(row.pair2))
        for row in strict[list(_PAIR_KEY_COLUMNS)].itertuples(index=False)
    }

    proxy_rows: list[dict[str, Any]] = []
    if proxy_negatives_per_query > 0 and (proxy_negatives_per_domain is None or proxy_negatives_per_domain > 0):
        for dataset in sorted(big_datasets):
            dataset_labels = labels.loc[labels["dataset"].eq(dataset)]
            proxy_rows.extend(
                _linker_component_proxy_rows(
                    dataset,
                    dataset_labels,
                    memberships[dataset],
                    excluded_pair_keys=strict_keys,
                    per_query_cap=proxy_negatives_per_query,
                    seed=seed,
                )
            )
    proxy = canonicalize_pair_rows(_rows_frame(proxy_rows))
    proxy = _apply_domain_cap(proxy, proxy_negatives_per_domain, seed=seed)

    return LinkerPairCatalog(strict=strict, linker_component_proxy=proxy)


def canonicalize_pair_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize pair orientation, deduplicate, and reject label conflicts.

    A pair key is namespaced by ``source_domain``.  Reversed duplicates are
    collapsed deterministically.  The function raises rather than choosing a
    winner if the same final key has both labels.
    """

    missing = sorted(set(PAIR_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"Pair rows are missing required columns: {missing}")
    if rows.empty:
        extras = [column for column in rows.columns if column not in PAIR_COLUMNS]
        return rows.loc[:, [*PAIR_COLUMNS, *extras]].reset_index(drop=True)

    out = rows.copy()
    required_text_columns = ("source_domain", "source_family", "pair1", "pair2", "label_rule", "origin", "group_id")
    if out[list(required_text_columns)].isna().any().any():
        raise ValueError("Required pair text columns cannot contain null values")
    for column in required_text_columns:
        out[column] = out[column].astype(str)
    if (out[list(required_text_columns)].apply(lambda column: column.str.strip().eq("")).any(axis=1)).any():
        raise ValueError("Required pair text columns cannot contain empty values")

    numeric_labels = pd.to_numeric(out["label"], errors="coerce")
    if numeric_labels.isna().any() or not numeric_labels.isin((0, 1)).all():
        raise ValueError("Pair labels must be exactly 0 or 1")
    out["label"] = numeric_labels.astype("int8")

    left = out["pair1"].copy()
    right = out["pair2"].copy()
    swap = left > right
    out.loc[swap, "pair1"] = right.loc[swap]
    out.loc[swap, "pair2"] = left.loc[swap]
    if out["pair1"].eq(out["pair2"]).any():
        examples = out.loc[out["pair1"].eq(out["pair2"]), list(_PAIR_KEY_COLUMNS)].head(5).to_dict("records")
        raise ValueError(f"Self-pairs are not valid training examples: {examples}")

    conflicting = out.groupby(list(_PAIR_KEY_COLUMNS), sort=False)["label"].nunique().gt(1)
    if conflicting.any():
        examples = [tuple(str(value) for value in key) for key in conflicting[conflicting].index.tolist()[:5]]
        raise ValueError(f"Conflicting final labels for canonical pair keys: {examples}")

    provenance_sort_columns = [
        column for column in ("label_rule", "origin", "group_id", *PROVENANCE_COLUMNS) if column in out.columns
    ]
    for column in provenance_sort_columns:
        out[column] = out[column].fillna("").astype(str)
    out = out.sort_values([*_PAIR_KEY_COLUMNS, *provenance_sort_columns], kind="stable")
    out = out.drop_duplicates(list(_PAIR_KEY_COLUMNS), keep="first")
    extras = [column for column in rows.columns if column not in PAIR_COLUMNS]
    return out.loc[:, [*PAIR_COLUMNS, *extras]].reset_index(drop=True)


def _declared_component_assets(payload: Mapping[str, Any]) -> dict[str, str]:
    try:
        datasets = payload["assets"]["candidate_members"]["datasets"]
    except (KeyError, TypeError) as exc:
        raise ValueError("bundle.json must declare assets.candidate_members.datasets") from exc
    if not isinstance(datasets, Mapping):
        raise ValueError("assets.candidate_members.datasets must be a mapping")
    return {str(dataset): str(path) for dataset, path in datasets.items()}


def _declared_label_assets(payload: Mapping[str, Any]) -> dict[str, str]:
    try:
        files = payload["assets"]["featureless_rows"]["files"]
    except (KeyError, TypeError) as exc:
        raise ValueError("bundle.json must declare assets.featureless_rows.files") from exc
    if not isinstance(files, Mapping):
        raise ValueError("assets.featureless_rows.files must be a mapping")
    return {str(key): str(path) for key, path in files.items()}


def _bundle_asset_path(bundle_root: Path, raw_path: str, *, context: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"{context} must be bundle-relative, got absolute path {raw_path!r}")
    resolved = (bundle_root / path).resolve()
    if not resolved.is_relative_to(bundle_root):
        raise ValueError(f"{context} escapes the bundle root: {raw_path!r}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Declared {context} does not exist: {resolved}")
    return resolved


def _read_declared_label_rows(
    payload: Mapping[str, Any],
    bundle_root: Path,
    selected_datasets: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for table_key, raw_path in sorted(_declared_label_assets(payload).items()):
        path = _bundle_asset_path(
            bundle_root,
            raw_path,
            context=f"featureless_rows.files[{table_key!r}]",
        )
        schema_names = set(pq.read_schema(path).names)
        missing = sorted(set(_LABEL_COLUMNS) - schema_names)
        if missing:
            raise ValueError(f"Declared linker label table {path} is missing columns: {missing}")
        frame = pd.read_parquet(path, columns=list(_LABEL_COLUMNS))
        frame["dataset"] = frame["dataset"].astype(str)
        frame = frame.loc[frame["dataset"].isin(selected_datasets)].copy()
        if frame.empty:
            continue
        if frame[list(_LABEL_COLUMNS[:-1])].isna().any().any():
            raise ValueError(f"Declared linker label table {path} has null identifier fields")
        numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
        if numeric_labels.isna().any() or not numeric_labels.isin((0, 1)).all():
            raise ValueError(f"Declared linker label table {path} has labels outside {{0, 1}}")
        frame["label"] = numeric_labels.astype("int8")
        for column in _LABEL_COLUMNS[:-1]:
            frame[column] = frame[column].astype(str)
        relative_path = path.relative_to(bundle_root).as_posix()
        frame["asset_origin"] = f"featureless_rows.files[{table_key}]={relative_path}"
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*_LABEL_COLUMNS, "asset_origin"])
    return pd.concat(frames, ignore_index=True)


def _read_declared_members(bundle_root: Path, raw_path: str, *, dataset: str) -> dict[str, tuple[str, ...]]:
    path = _bundle_asset_path(
        bundle_root,
        raw_path,
        context=f"candidate_members.datasets[{dataset!r}]",
    )
    schema_names = set(pq.read_schema(path).names)
    missing = sorted(set(_MEMBER_COLUMNS) - schema_names)
    if missing:
        raise ValueError(f"Declared candidate member table {path} is missing columns: {missing}")
    members = pd.read_parquet(path, columns=list(_MEMBER_COLUMNS))
    if members[list(_MEMBER_COLUMNS)].isna().any().any():
        raise ValueError(f"Declared candidate member table {path} has null required fields")
    members["dataset"] = members["dataset"].astype(str)
    unexpected_datasets = sorted(set(members["dataset"]) - {dataset})
    if unexpected_datasets:
        raise ValueError(
            f"Candidate member asset declared for {dataset!r} contains other datasets: {unexpected_datasets}"
        )
    members["candidate_component_key"] = members["candidate_component_key"].astype(str)
    members["signature_id"] = members["signature_id"].astype(str)
    member_indices = pd.to_numeric(members["member_index"], errors="coerce")
    if member_indices.isna().any():
        raise ValueError(f"Candidate member asset {path} has non-numeric member_index values")
    members["member_index"] = member_indices.astype("int64")
    members = members.sort_values(["candidate_component_key", "member_index", "signature_id"], kind="stable")
    members = members.drop_duplicates(["candidate_component_key", "signature_id"], keep="first")
    return {
        str(component): tuple(group["signature_id"].tolist())
        for component, group in members.groupby("candidate_component_key", sort=True)
    }


def _validate_referenced_components(
    labels: pd.DataFrame,
    memberships: Mapping[str, Mapping[str, Sequence[str]]],
) -> None:
    for dataset, dataset_labels in labels.groupby("dataset", sort=True):
        missing = sorted(set(dataset_labels["candidate_component_key"]) - set(memberships[str(dataset)]))
        if missing:
            raise KeyError(f"Linker labels for {dataset!r} reference undeclared component keys: {missing[:10]}")


def _read_gold_signature_to_cluster(path_value: str | Path) -> dict[str, str]:
    path = Path(path_value).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Gold cluster file must contain an object: {path}")
    signature_to_cluster: dict[str, str] = {}
    for raw_cluster_id, raw_cluster in payload.items():
        cluster_id = str(raw_cluster_id)
        if not isinstance(raw_cluster, Mapping) or not isinstance(raw_cluster.get("signature_ids"), list):
            raise ValueError(f"Gold cluster {cluster_id!r} in {path} must declare signature_ids as a list")
        for raw_signature_id in raw_cluster["signature_ids"]:
            signature_id = str(raw_signature_id)
            previous = signature_to_cluster.setdefault(signature_id, cluster_id)
            if previous != cluster_id:
                raise ValueError(
                    f"Gold signature {signature_id!r} belongs to both {previous!r} and {cluster_id!r} in {path}"
                )
    return signature_to_cluster


def _public_gold_rows(
    dataset: str,
    labels: pd.DataFrame,
    memberships: Mapping[str, Sequence[str]],
    gold: Mapping[str, str],
) -> list[dict[str, Any]]:
    referenced_ids = _needed_signature_ids(labels, memberships)
    missing_gold = sorted(referenced_ids - set(gold))
    if missing_gold:
        raise ValueError(
            f"Public linker domain {dataset!r} has signatures missing from gold clusters: {missing_gold[:10]}"
        )

    rows: list[dict[str, Any]] = []
    ordered_labels = labels.sort_values(
        ["query_signature_id", "candidate_component_key", "query_group_id", "asset_origin"], kind="stable"
    )
    source_columns = ("query_signature_id", "candidate_component_key", "query_group_id", "asset_origin")
    for query_value, component_value, query_group_value, origin_value in ordered_labels[
        list(source_columns)
    ].itertuples(index=False, name=None):
        query_id = str(query_value)
        query_cluster = gold[query_id]
        component_key = str(component_value)
        for member_id in memberships[component_key]:
            member_id = str(member_id)
            if member_id == query_id:
                continue
            rows.append(
                _pair_row(
                    dataset=dataset,
                    query_id=query_id,
                    member_id=member_id,
                    label=int(query_cluster == gold[member_id]),
                    label_rule=PUBLIC_GOLD_LABEL_RULE,
                    origin=str(origin_value),
                    query_group_id=str(query_group_value),
                    component_key=component_key,
                )
            )
    return rows


def _needed_signature_ids(labels: pd.DataFrame, memberships: Mapping[str, Sequence[str]]) -> set[str]:
    needed = set(labels["query_signature_id"].astype(str))
    for component in set(labels["candidate_component_key"].astype(str)):
        needed.update(str(signature_id) for signature_id in memberships[component])
    return needed


def _labels_require_block_metadata(labels: pd.DataFrame) -> bool:
    return bool(labels["candidate_component_key"].astype(str).str.contains("::", regex=False).any())


def _declared_signature_paths(
    payload: Mapping[str, Any],
    bundle_root: Path,
    dataset: str,
) -> tuple[Path, Path]:
    try:
        runtime_contract = payload["runtime_contract"]
    except (KeyError, TypeError) as exc:
        raise ValueError("bundle.json must declare runtime_contract.arrow_dataset_root") from exc
    if not isinstance(runtime_contract, Mapping):
        raise ValueError("bundle.json runtime_contract must be a mapping")
    arrow_root_raw = runtime_contract.get("arrow_dataset_root")
    if not isinstance(arrow_root_raw, str) or not arrow_root_raw.strip():
        raise ValueError("bundle.json must declare runtime_contract.arrow_dataset_root as a non-empty string")
    arrow_root = Path(arrow_root_raw)
    if arrow_root.is_absolute():
        raise ValueError("runtime_contract.arrow_dataset_root must be bundle-relative")
    resolved_arrow_root = (bundle_root / arrow_root).resolve()
    if not resolved_arrow_root.is_relative_to(bundle_root):
        raise ValueError("runtime_contract.arrow_dataset_root escapes the bundle root")
    dataset_root = (resolved_arrow_root / dataset).resolve()
    if not dataset_root.is_relative_to(resolved_arrow_root):
        raise ValueError(f"Arrow dataset name escapes runtime_contract.arrow_dataset_root: {dataset!r}")
    manifest_path = (dataset_root / "manifest.json").resolve()
    if not manifest_path.is_relative_to(dataset_root):
        raise ValueError(f"Arrow dataset manifest for {dataset!r} escapes the Arrow dataset root")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Arrow dataset manifest missing for {dataset!r}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("paths"), Mapping):
        raise ValueError(f"Arrow dataset manifest for {dataset!r} must declare paths as a mapping")
    signatures_raw = manifest["paths"].get("signatures")
    if not isinstance(signatures_raw, str) or not signatures_raw.strip():
        raise ValueError(f"Arrow dataset manifest for {dataset!r} must declare paths.signatures as a non-empty string")
    signatures_relative = Path(signatures_raw)
    if signatures_relative.is_absolute():
        raise ValueError(f"Arrow signatures path for {dataset!r} must be dataset-relative")
    signatures_path = (dataset_root / signatures_relative).resolve()
    if not signatures_path.is_relative_to(dataset_root):
        raise ValueError(f"Arrow signatures path for {dataset!r} escapes the Arrow dataset root")
    if not signatures_path.is_file():
        raise FileNotFoundError(f"Declared Arrow signature table does not exist for {dataset!r}: {signatures_path}")
    return manifest_path, signatures_path


def _read_declared_signature_metadata(
    payload: Mapping[str, Any],
    bundle_root: Path,
    dataset: str,
    needed_signature_ids: set[str],
    *,
    required_columns: Collection[str],
) -> dict[str, dict[str, Any]]:
    _manifest_path, signatures_path = _declared_signature_paths(payload, bundle_root, dataset)
    selected_columns = ("signature_id", *sorted(str(column) for column in required_columns))

    seen: set[str] = set()
    metadata: dict[str, dict[str, Any]] = {}
    with pa_ipc.open_file(signatures_path) as reader:
        missing = sorted(set(selected_columns) - set(reader.schema.names))
        if missing:
            raise ValueError(f"Arrow signature table {signatures_path} is missing columns: {missing}")
        for column in selected_columns:
            column_type = reader.schema.field(column).type
            if not (pa.types.is_string(column_type) or pa.types.is_large_string(column_type)):
                raise ValueError(
                    f"Arrow signature table {signatures_path} column {column!r} must use string or large_string "
                    f"type, got {column_type}"
                )
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index).select(list(selected_columns))
            rows = batch.to_pylist()
            for row in rows:
                raw_signature_id = row["signature_id"]
                if raw_signature_id is None:
                    raise ValueError(f"Arrow signature table {signatures_path} has a null signature_id")
                signature_id = str(raw_signature_id)
                if not signature_id:
                    raise ValueError(f"Arrow signature table {signatures_path} has an empty signature_id")
                if signature_id in seen:
                    raise ValueError(
                        f"Arrow signature table {signatures_path} has duplicate signature_id {signature_id!r}"
                    )
                seen.add(signature_id)
                if signature_id in needed_signature_ids:
                    metadata[signature_id] = {column: row[column] for column in selected_columns[1:]}
    missing_signatures = sorted(needed_signature_ids - seen)
    if missing_signatures:
        raise ValueError(f"Arrow signature table for {dataset!r} is missing referenced IDs: {missing_signatures[:10]}")
    return metadata


def _block_local_memberships(
    labels: pd.DataFrame,
    memberships: Mapping[str, Sequence[str]],
    signature_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    filtered_memberships = {
        component: tuple(str(value) for value in members) for component, members in memberships.items()
    }
    for component_key in sorted(set(labels["candidate_component_key"].astype(str))):
        if "::" not in component_key:
            continue
        block_key, _component = component_key.split("::", 1)
        raw_members = filtered_memberships[component_key]
        block_local = tuple(
            signature_id
            for signature_id in raw_members
            if signature_metadata[signature_id]["author_block"] is not None
            and str(signature_metadata[signature_id]["author_block"]) == block_key
        )
        filtered_memberships[component_key] = block_local or raw_members
    return filtered_memberships


def _big_block_orcid_rows(
    dataset: str,
    labels: pd.DataFrame,
    memberships: Mapping[str, Sequence[str]],
    signature_orcids: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    members_by_component_orcid: dict[tuple[str, str], list[str]] = {}
    referenced_components = set(labels["candidate_component_key"].astype(str))
    for component in referenced_components:
        member_ids = memberships[component]
        for member_id in member_ids:
            orcid = signature_orcids[str(member_id)]
            if orcid is not None:
                members_by_component_orcid.setdefault((str(component), orcid), []).append(str(member_id))

    rows: list[dict[str, Any]] = []
    ordered_labels = labels.sort_values(
        ["query_signature_id", "candidate_component_key", "query_group_id", "asset_origin"], kind="stable"
    )
    source_columns = ("query_signature_id", "candidate_component_key", "query_group_id", "asset_origin")
    for query_value, component_value, query_group_value, origin_value in ordered_labels[
        list(source_columns)
    ].itertuples(index=False, name=None):
        query_id = str(query_value)
        query_orcid = signature_orcids[query_id]
        if query_orcid is None:
            continue
        component_key = str(component_value)
        for member_id in members_by_component_orcid.get((component_key, query_orcid), ()):  # strict match only
            if member_id == query_id:
                continue
            rows.append(
                _pair_row(
                    dataset=dataset,
                    query_id=query_id,
                    member_id=member_id,
                    label=1,
                    label_rule=BIG_BLOCK_ORCID_LABEL_RULE,
                    origin=str(origin_value),
                    query_group_id=str(query_group_value),
                    component_key=component_key,
                )
            )
    return rows


def _linker_component_proxy_rows(
    dataset: str,
    labels: pd.DataFrame,
    memberships: Mapping[str, Sequence[str]],
    *,
    excluded_pair_keys: set[tuple[str, str, str]],
    per_query_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Expand negatives only for relationships with no positive observation.

    ``query_group_id`` identifies a feature view, so eligibility is decided at
    the view-independent dataset/query/component relationship boundary.  Any
    observed label 1 vetoes all label-0 views of that same relationship.
    """

    relationship_columns = ("dataset", "query_signature_id", "candidate_component_key")
    relationship_has_positive = labels.groupby(list(relationship_columns), sort=False)["label"].transform("max").eq(1)
    negative_labels = labels.loc[labels["label"].eq(0) & ~relationship_has_positive].copy()
    if negative_labels.empty:
        return []

    referenced_components = set(negative_labels["candidate_component_key"].astype(str))
    ranked_members = {
        component: tuple(
            sorted(
                (str(member_id) for member_id in memberships[component]),
                key=lambda member_id: (_stable_hash(seed, dataset, member_id), member_id),
            )
        )
        for component in referenced_components
    }

    rows: list[dict[str, Any]] = []
    for query_id, query_rows in negative_labels.groupby("query_signature_id", sort=True):
        query_id = str(query_id)
        component_sources = (
            query_rows.sort_values(["candidate_component_key", "query_group_id", "asset_origin"], kind="stable")
            .drop_duplicates("candidate_component_key", keep="first")
            .set_index("candidate_component_key")
        )
        candidates: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
        for component_key, source in component_sources.iterrows():
            component_key = str(component_key)
            eligible_from_component = 0
            for member_id in ranked_members[component_key]:
                if member_id == query_id:
                    continue
                pair1, pair2 = _canonical_pair(query_id, member_id)
                pair_key = (dataset, pair1, pair2)
                if pair_key in excluded_pair_keys:
                    continue
                rank = _stable_hash(seed, dataset, member_id)
                row = _pair_row(
                    dataset=dataset,
                    query_id=query_id,
                    member_id=member_id,
                    label=0,
                    label_rule=LINKER_COMPONENT_PROXY_LABEL_RULE,
                    origin=str(source["asset_origin"]),
                    query_group_id=str(source["query_group_id"]),
                    component_key=component_key,
                )
                candidate_key = (pair1, pair2)
                previous = candidates.get(candidate_key)
                candidate_value = (rank, row)
                if previous is None or _proxy_candidate_sort_key(candidate_value) < _proxy_candidate_sort_key(previous):
                    candidates[candidate_key] = candidate_value
                eligible_from_component += 1
                if eligible_from_component >= per_query_cap:
                    break

        selected = sorted(candidates.values(), key=_proxy_candidate_sort_key)[:per_query_cap]
        rows.extend(row for _, row in selected)
    return rows


def _proxy_candidate_sort_key(candidate: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
    rank, row = candidate
    return (
        rank,
        row["pair1"],
        row["pair2"],
        row["candidate_component_key"],
        row["origin"],
    )


def _apply_domain_cap(rows: pd.DataFrame, cap: int | None, *, seed: int) -> pd.DataFrame:
    if cap is None or rows.empty:
        return rows
    selected_frames: list[pd.DataFrame] = []
    for dataset, domain_rows in rows.groupby("source_domain", sort=True):
        domain_rows = domain_rows.copy()
        domain_rows["_cap_rank"] = [
            _stable_hash(seed, str(dataset), str(pair1), str(pair2))
            for pair1, pair2 in domain_rows[["pair1", "pair2"]].itertuples(index=False, name=None)
        ]
        domain_rows = domain_rows.sort_values(["_cap_rank", "pair1", "pair2"], kind="stable").head(cap)
        selected_frames.append(domain_rows.drop(columns="_cap_rank"))
    if not selected_frames:
        return _rows_frame([])
    return canonicalize_pair_rows(pd.concat(selected_frames, ignore_index=True))


def _pair_row(
    *,
    dataset: str,
    query_id: str,
    member_id: str,
    label: int,
    label_rule: str,
    origin: str,
    query_group_id: str,
    component_key: str,
) -> dict[str, Any]:
    pair1, pair2 = _canonical_pair(query_id, member_id)
    return {
        "source_domain": dataset,
        "source_family": "linker",
        "pair1": pair1,
        "pair2": pair2,
        "label": int(label),
        "label_rule": label_rule,
        "origin": origin,
        "group_id": f"{dataset}:{query_id}",
        "query_signature_id": query_id,
        "query_group_id": query_group_id,
        "candidate_component_key": component_key,
    }


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _stable_hash(seed: int, *parts: str) -> int:
    payload = "\x00".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def _rows_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))

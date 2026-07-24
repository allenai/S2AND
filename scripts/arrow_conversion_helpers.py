"""Direct Arrow conversion helpers for `ANDData`-like inputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from s2and.incremental_linking.feature_block_arrow import (
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    write_name_counts_index,
    write_raw_planner_arrow_tables,
)
from s2and.incremental_linking.feature_block_contract import (
    filter_cluster_seed_disallows_for_signature_subset,
)
from s2and.name_counts_manifest import NAME_COUNTS_PROVENANCE_SCHEMA_VERSION


def bounded_name_count_mappings_from_signature_payloads(
    signatures: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Count canonical name keys in one exact bounded signature payload."""

    from s2and.text import canonical_name_count_keys, canonicalize_name_parts

    counters: dict[str, Counter[str]] = {
        key: Counter() for key in ("first", "last", "first_last", "last_first_initial")
    }
    for signature in signatures.values():
        if not isinstance(signature, Mapping):
            raise TypeError("bounded signatures must contain object records")
        author_info = signature.get("author_info") or {}
        if not isinstance(author_info, Mapping):
            raise TypeError("bounded signature author_info must be an object")
        keys = canonical_name_count_keys(
            canonicalize_name_parts(
                author_info.get("first"),
                author_info.get("middle"),
                author_info.get("last"),
            )
        )
        for key, value in keys.items():
            if value is not None:
                counters[key][value] += 1
    return (
        dict(counters["first"]),
        dict(counters["last"]),
        dict(counters["first_last"]),
        dict(counters["last_first_initial"]),
    )


def write_bounded_name_counts_index(
    signatures: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[str, str]:
    """Write a canonical bounded name-count index and return its logical digest."""

    from s2and.consts import NORMALIZATION_VERSION

    mappings = bounded_name_count_mappings_from_signature_payloads(signatures)
    encoded_signatures = json.dumps(signatures, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    records_sha256 = hashlib.sha256(encoded_signatures).hexdigest()
    provenance = {
        "schema_version": NAME_COUNTS_PROVENANCE_SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "generation_id": f"bounded-{records_sha256[:16]}",
        "source_snapshot_id": f"bounded-json-{records_sha256[:16]}",
        "source_kind": "verification:bounded-json",
        "source_query_sha256": hashlib.sha256(b"bounded-name-counts-v1").hexdigest(),
        "selected_rows_sha256": records_sha256,
        "source_row_count": len(signatures),
    }
    index_path, _metrics = write_name_counts_index(
        output_dir,
        mappings,
        provenance,
    )
    logical_payload = json.dumps(
        {"mappings": mappings, "provenance": provenance},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return index_path, hashlib.sha256(logical_payload).hexdigest()


def write_raw_planner_arrow_from_anddata(
    dataset: Any,
    output_dir: str | Path,
    *,
    signature_ids: Sequence[Any] | None = None,
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    include_specter: bool = True,
    include_empty_cluster_seeds: bool = False,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> dict[str, str]:
    """Build typed Arrow tables from `ANDData` and write complete IPC files."""

    tables = raw_planner_arrow_tables_from_anddata(
        dataset,
        signature_ids=signature_ids,
        cluster_seeds_require=cluster_seeds_require,
        include_specter=include_specter,
    )
    return write_raw_planner_arrow_tables(
        tables,
        output_dir,
        include_empty_cluster_seeds=include_empty_cluster_seeds,
        max_record_batch_rows=max_record_batch_rows,
        overwrite=overwrite,
    )


def raw_planner_arrow_tables_from_anddata(
    dataset: Any,
    *,
    signature_ids: Sequence[Any] | None = None,
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    include_specter: bool = True,
) -> dict[str, pa.Table]:
    """Build the typed raw-planner Arrow tables from an `ANDData`-like object."""

    resolved_signature_ids = tuple(
        str(value) for value in (dataset.signatures.keys() if signature_ids is None else signature_ids)
    )
    if len(set(resolved_signature_ids)) != len(resolved_signature_ids):
        raise ValueError("signature_ids must be unique")
    signature_id_set = set(resolved_signature_ids)

    signature_columns: dict[str, list[Any]] = {
        "signature_id": [],
        "paper_id": [],
        "author_first": [],
        "author_middle": [],
        "author_last": [],
        "author_suffix": [],
        "author_affiliations": [],
        "author_orcid": [],
        "author_position": [],
        "author_block": [],
        "author_email": [],
        "source_author_ids": [],
    }
    paper_ids: list[str] = []
    seen_paper_ids: set[str] = set()
    for signature_id in resolved_signature_ids:
        if not signature_id:
            raise ValueError("signature_id must be non-empty")
        signature = dataset.signatures[signature_id]
        paper_id = str(signature.paper_id)
        if not paper_id:
            raise ValueError(f"paper_id must be non-empty for signature {signature_id!r}")
        signature_columns["signature_id"].append(signature_id)
        signature_columns["paper_id"].append(paper_id)
        signature_columns["author_first"].append(_optional_str(getattr(signature, "author_info_first", None)))
        signature_columns["author_middle"].append(_optional_str(getattr(signature, "author_info_middle", None)))
        signature_columns["author_last"].append(_optional_str(getattr(signature, "author_info_last", None)))
        signature_columns["author_suffix"].append(_optional_str(getattr(signature, "author_info_suffix", None)))
        signature_columns["author_affiliations"].append(
            list(
                _strict_string_tuple(
                    getattr(signature, "author_info_affiliations", None),
                    field_name="signatures.author_info_affiliations",
                )
            )
        )
        signature_columns["author_orcid"].append(_optional_str(getattr(signature, "author_info_orcid", None)))
        signature_columns["author_position"].append(
            _optional_int(
                getattr(signature, "author_info_position", None),
                field_name="signatures.author_info_position",
            )
        )
        signature_columns["author_block"].append(_optional_str(getattr(signature, "author_info_block", None)))
        signature_columns["author_email"].append(_optional_str(getattr(signature, "author_info_email", None)))
        signature_columns["source_author_ids"].append(
            list(
                _strict_string_tuple(
                    getattr(signature, "sourced_author_ids", None),
                    field_name="signatures.sourced_author_ids",
                    skip_none=True,
                )
            )
        )
        if paper_id not in seen_paper_ids:
            seen_paper_ids.add(paper_id)
            paper_ids.append(paper_id)

    papers_by_id = getattr(dataset, "papers", {})
    paper_columns: dict[str, list[Any]] = {
        "paper_id": [],
        "title": [],
        "abstract": [],
        "venue": [],
        "journal_name": [],
        "year": [],
        "predicted_language": [],
        "is_reliable": [],
        "language_reliability": [],
    }
    paper_author_columns: dict[str, list[Any]] = {
        "paper_id": [],
        "position": [],
        "author_name": [],
    }
    seen_paper_author_positions: set[tuple[str, int]] = set()
    for paper_id in paper_ids:
        paper = papers_by_id.get(paper_id)
        if paper is None:
            raise ValueError(f"ANDData papers are missing signature paper_id: {paper_id!r}")
        predicted_language, is_reliable, language_reliability = _paper_language_metadata(paper)
        paper_columns["paper_id"].append(paper_id)
        paper_columns["title"].append(_optional_str(getattr(paper, "title", None)))
        paper_columns["abstract"].append("Has Abstract" if bool(getattr(paper, "has_abstract", False)) else "")
        paper_columns["venue"].append(_optional_str(getattr(paper, "venue", None)))
        paper_columns["journal_name"].append(_optional_str(getattr(paper, "journal_name", None)))
        paper_columns["year"].append(_optional_int(getattr(paper, "year", None), field_name="papers.year"))
        paper_columns["predicted_language"].append(predicted_language)
        paper_columns["is_reliable"].append(is_reliable)
        paper_columns["language_reliability"].append(language_reliability)
        for index, author in enumerate(getattr(paper, "authors", None) or ()):
            position = _optional_int(getattr(author, "position", index), field_name="papers.authors.position")
            resolved_position = index if position is None else position
            key = (paper_id, resolved_position)
            if key in seen_paper_author_positions:
                raise ValueError(f"paper_authors contains duplicate (paper_id, position): {key!r}")
            seen_paper_author_positions.add(key)
            paper_author_columns["paper_id"].append(paper_id)
            paper_author_columns["position"].append(resolved_position)
            paper_author_columns["author_name"].append(str(getattr(author, "author_name", "") or ""))

    source_cluster_seeds = dict(
        getattr(dataset, "cluster_seeds_require", {}) if cluster_seeds_require is None else cluster_seeds_require
    )
    require_pairs: list[tuple[str, str]] = []
    seen_require_signature_ids: set[str] = set()
    for signature_id, component_id in source_cluster_seeds.items():
        signature_key = str(signature_id)
        if signature_key not in signature_id_set:
            continue
        component_key = str(component_id)
        if not signature_key:
            raise ValueError("cluster_seeds_require cannot contain empty signature_id values")
        if not component_key:
            raise ValueError(f"cluster_seeds_require cannot contain empty component_id values: {signature_key!r}")
        if signature_key in seen_require_signature_ids:
            raise ValueError(f"cluster_seeds_require contains duplicate signature_id: {signature_key!r}")
        seen_require_signature_ids.add(signature_key)
        require_pairs.append((signature_key, component_key))
    disallow_pairs = filter_cluster_seed_disallows_for_signature_subset(
        getattr(dataset, "cluster_seeds_disallow", set()),
        signature_id_set,
    )
    specter_paper_ids, specter_embeddings = _specter_from_anddata(
        dataset,
        paper_ids,
        include_specter=include_specter,
    )

    tables = {
        "signatures": pa.table(
            {
                "signature_id": pa.array(signature_columns["signature_id"], type=pa.string()),
                "paper_id": pa.array(signature_columns["paper_id"], type=pa.string()),
                "author_first": pa.array(signature_columns["author_first"], type=pa.string()),
                "author_middle": pa.array(signature_columns["author_middle"], type=pa.string()),
                "author_last": pa.array(signature_columns["author_last"], type=pa.string()),
                "author_suffix": pa.array(signature_columns["author_suffix"], type=pa.string()),
                "author_affiliations": pa.array(
                    signature_columns["author_affiliations"],
                    type=pa.list_(pa.string()),
                ),
                "author_orcid": pa.array(signature_columns["author_orcid"], type=pa.string()),
                "author_position": pa.array(signature_columns["author_position"], type=pa.int64()),
                "author_block": pa.array(signature_columns["author_block"], type=pa.string()),
                "author_email": pa.array(signature_columns["author_email"], type=pa.string()),
                "source_author_ids": pa.array(
                    signature_columns["source_author_ids"],
                    type=pa.list_(pa.string()),
                ),
            }
        ),
        "papers": pa.table(
            {
                "paper_id": pa.array(paper_columns["paper_id"], type=pa.string()),
                "title": pa.array(paper_columns["title"], type=pa.string()),
                "abstract": pa.array(paper_columns["abstract"], type=pa.string()),
                "venue": pa.array(paper_columns["venue"], type=pa.string()),
                "journal_name": pa.array(paper_columns["journal_name"], type=pa.string()),
                "year": pa.array(paper_columns["year"], type=pa.int64()),
                "predicted_language": pa.array(paper_columns["predicted_language"], type=pa.string()),
                "is_reliable": pa.array(paper_columns["is_reliable"], type=pa.bool_()),
                "language_reliability": pa.array(
                    paper_columns["language_reliability"],
                    type=pa.float64(),
                ),
            }
        ),
        "paper_authors": pa.table(
            {
                "paper_id": pa.array(paper_author_columns["paper_id"], type=pa.string()),
                "position": pa.array(paper_author_columns["position"], type=pa.int64()),
                "author_name": pa.array(paper_author_columns["author_name"], type=pa.string()),
            }
        ),
        "cluster_seeds": pa.table(
            {
                "signature_id": pa.array([signature_id for signature_id, _ in require_pairs], type=pa.string()),
                "cluster_id": pa.array([component_id for _, component_id in require_pairs], type=pa.string()),
            }
        ),
        "cluster_seed_disallows": pa.table(
            {
                "signature_id_1": pa.array([left for left, _ in disallow_pairs], type=pa.string()),
                "signature_id_2": pa.array([right for _, right in disallow_pairs], type=pa.string()),
            }
        ),
    }
    if specter_embeddings is not None:
        flat = pa.array(np.ravel(specter_embeddings), type=pa.float32())
        tables["specter"] = pa.table(
            {
                "paper_id": pa.array(specter_paper_ids, type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(flat, int(specter_embeddings.shape[1])),
            }
        )
    return tables


def _specter_from_anddata(
    dataset: Any,
    paper_ids: Sequence[str],
    *,
    include_specter: bool,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    if not include_specter:
        return (), None
    specter = getattr(dataset, "specter_embeddings", None)
    if specter is None:
        return (), None
    selected_paper_ids: list[str] = []
    vectors: list[np.ndarray] = []
    expected_dim: int | None = None
    for paper_id in paper_ids:
        vector = specter.get(paper_id)
        if vector is None:
            continue
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"SPECTER vector for paper_id={paper_id!r} must be 1D, got shape={array.shape}")
        if expected_dim is None:
            expected_dim = int(array.shape[0])
        elif int(array.shape[0]) != expected_dim:
            raise ValueError(
                "SPECTER vectors must have equal dimensions: "
                f"expected {expected_dim}, got {array.shape[0]} for paper_id={paper_id!r}"
            )
        selected_paper_ids.append(paper_id)
        vectors.append(array)
    if not vectors:
        return (), np.empty((0, 1), dtype=np.float32)
    return tuple(selected_paper_ids), np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)


def _strict_string_tuple(value: Any, *, field_name: str, skip_none: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        raise ValueError(f"{field_name} must be a sequence, not a scalar string")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence")
    items: list[str] = []
    for item in value:
        if item is None:
            if skip_none:
                continue
            raise ValueError(f"{field_name} cannot contain null values")
        items.append(str(item))
    return tuple(items)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer or null, got {value!r}") from exc


def _optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false"}:
            return False
        if normalized in {"1", "true"}:
            return True
    raise ValueError(f"{field_name} must be a boolean, 0/1, true/false, or null, got {value!r}")


def _optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float or null, got {value!r}") from exc


def _paper_language_metadata(paper: Any) -> tuple[str | None, bool | None, float | None]:
    predicted_language = _optional_str(getattr(paper, "predicted_language", None))
    is_reliable = _optional_bool(getattr(paper, "is_reliable", None), field_name="papers.is_reliable")
    language_reliability = _optional_float(
        getattr(paper, "language_reliability", None),
        field_name="papers.language_reliability",
    )
    if language_reliability is not None:
        if not np.isfinite(language_reliability):
            raise ValueError("papers.language_reliability must be finite")
        if not 0.0 <= language_reliability <= 1.0:
            raise ValueError("papers.language_reliability must be in [0.0, 1.0]")
        if is_reliable is False and language_reliability != 0.0:
            raise ValueError("papers.language_reliability must be 0.0 when papers.is_reliable is false")
    if predicted_language is None:
        if is_reliable is not None or language_reliability is not None:
            raise ValueError("papers.is_reliable and papers.language_reliability require papers.predicted_language")
    else:
        if not predicted_language.strip():
            raise ValueError("papers.predicted_language must be non-empty")
        if is_reliable is None:
            raise ValueError("papers.predicted_language requires papers.is_reliable")
        if language_reliability is None:
            raise ValueError("papers.predicted_language requires papers.language_reliability")
    return predicted_language, is_reliable, language_reliability

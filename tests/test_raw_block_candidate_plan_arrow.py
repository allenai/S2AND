from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import s2and.incremental_linking.retrieval as retrieval_module
from s2and.arrow_inputs import validate_arrow_publication_artifacts
from s2and.incremental_linking.feature_block import (
    write_arrow_batch_lookup_index,
    write_incremental_query_signatures_arrow,
    write_name_counts_index,
    write_raw_arrow_batch_lookup_indexes,
)
from s2and.incremental_linking.retrieval import (
    RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS,
    RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
    RawArrowPlanBundle,
    build_linker_retrieval_batch_from_raw_plan_bundle,
)
from s2and.incremental_linking.runtime import _seed_setup_from_component_members
from s2and.runtime import load_s2and_rust_extension
from tests.helpers import (
    tiny_name_counts_provenance,
    write_test_arrow_artifact_manifest,
)

s2and_rust = load_s2and_rust_extension()

pa = pytest.importorskip("pyarrow")

_FNV64_OFFSET = 14695981039346656037
_FNV64_PRIME = 1099511628211
_ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT = struct.Struct("<8sQQQQ")
_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT = struct.Struct("<QII")
_NAME_COUNTS_INDEX_HEADER_LEN = 32
_NAME_COUNTS_INDEX_RECORD_LEN = 40


def _indexed_pair_matrix(featurizer: Any, pairs: list[tuple[str, str]]) -> np.ndarray:
    signature_id_to_index = {str(signature_id): index for index, signature_id in enumerate(featurizer.signature_ids())}
    indexed_pairs = [(signature_id_to_index[left], signature_id_to_index[right]) for left, right in pairs]
    return np.asarray(featurizer.featurize_pairs_matrix_indexed(indexed_pairs, None, 1, np.nan))


def _minimal_raw_candidate_plan(**overrides: Any) -> dict[str, Any]:
    query_signature_ids = list(overrides.pop("query_signature_ids", ["q0"]))
    row_count = int(overrides.pop("row_count", 0))
    pair_count = int(overrides.pop("pair_count", 0))
    plan: dict[str, Any] = {
        "schema_version": RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
        "query_signature_ids": query_signature_ids,
        "query_views": ["full"] * len(query_signature_ids),
        "query_authors": ["Alice"] * len(query_signature_ids),
        "row_count": row_count,
        "pair_count": pair_count,
        "row_query_signature_indices": np.zeros(row_count, dtype=np.uint32),
        "row_component_keys": [f"c{index}" for index in range(row_count)],
        "retrieval_scores": np.zeros(row_count, dtype=np.float32),
        "retrieval_ranks": np.arange(1, row_count + 1, dtype=np.uint16),
        "pair_row_indices": np.zeros(pair_count, dtype=np.uint32),
        "left_signature_ids": [(query_signature_ids[0] if query_signature_ids else "")] * pair_count,
        "right_signature_ids": [f"s{index}" for index in range(pair_count)],
        "component_members": {},
        "telemetry": {},
    }
    for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        if dtype is object:
            plan[raw_key] = np.asarray([""] * row_count, dtype=object)
        else:
            plan[raw_key] = np.zeros(row_count, dtype=dtype)
    plan.update(overrides)
    return plan


def test_raw_arrow_plan_bundle_rejects_pair_left_id_that_disagrees_with_row_query() -> None:
    raw_plan = _minimal_raw_candidate_plan(
        query_signature_ids=["q0", "q1", "q2"],
        query_views=["full", "full", "full"],
        query_authors=["Alice", "Bob", "Carol"],
        row_count=1,
        pair_count=1,
        row_query_signature_indices=np.asarray([1], dtype=np.uint32),
        row_component_keys=["c1"],
        pair_row_indices=np.asarray([0], dtype=np.uint32),
        left_signature_ids=["q0"],
        right_signature_ids=["s1"],
        component_members={"c1": ["s1"]},
    )

    with pytest.raises(ValueError, match="left_signature_ids must match"):
        RawArrowPlanBundle.from_native_mapping(raw_plan)


def test_raw_arrow_plan_bundle_rejects_duplicate_query_signature_ids() -> None:
    raw_plan = _minimal_raw_candidate_plan(
        query_signature_ids=["q0", "q0"],
        query_views=["full", "full"],
        query_authors=["Alice", "Alice"],
    )

    with pytest.raises(ValueError, match="query_signature_ids must be unique"):
        RawArrowPlanBundle.from_native_mapping(raw_plan)


def test_raw_candidate_plan_seed_setup_rejects_duplicate_seed_signature() -> None:
    raw_plan = {"component_members": {"c1": ["s1", "s2"], "c2": ["s1"]}}

    with pytest.raises(ValueError, match="assigns signature_id 's1' to multiple components"):
        _seed_setup_from_component_members(raw_plan["component_members"])


def test_raw_candidate_plan_schema_requires_component_members() -> None:
    raw_plan = _minimal_raw_candidate_plan()
    raw_plan.pop("component_members")

    with pytest.raises(KeyError, match="component_members"):
        RawArrowPlanBundle.from_native_mapping(raw_plan)


def test_raw_candidate_plan_schema_rejects_non_mapping_component_members() -> None:
    raw_plan = _minimal_raw_candidate_plan(component_members=[])

    with pytest.raises(ValueError, match="component_members must be a mapping"):
        RawArrowPlanBundle.from_native_mapping(raw_plan)


def _fnv64_bytes(value: bytes) -> int:
    digest = _FNV64_OFFSET
    for byte in value:
        digest ^= byte
        digest = (digest * _FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    return digest


def _append_batch_index_record(
    index_path: str,
    *,
    key: str,
    batch_index: int,
    replace_existing: bool = False,
) -> None:
    path = Path(index_path)
    raw = path.read_bytes()
    magic, record_count, source_size, key_column_hash, source_fingerprint = (
        _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.unpack_from(raw, 0)
    )
    offset = _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.size
    record_size = _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.size
    records = [
        _ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.unpack_from(raw, offset + index * record_size)
        for index in range(record_count)
    ]
    key_hash = _fnv64_bytes(key.encode("utf-8"))
    if replace_existing:
        matching_indices = [index for index, record in enumerate(records) if record[0] == key_hash]
        assert len(matching_indices) == 1
        record_index = matching_indices[0]
        records[record_index] = (key_hash, int(batch_index), records[record_index][2])
    else:
        records.append((key_hash, int(batch_index), 0))
    records.sort()
    payload = bytearray(
        _ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT.pack(
            magic,
            len(records),
            source_size,
            key_column_hash,
            source_fingerprint,
        )
    )
    for record in records:
        payload.extend(_ARROW_BATCH_LOOKUP_INDEX_RECORD_STRUCT.pack(*record))
    path.write_bytes(payload)


def _write_ipc(path: Path, table: pa.Table) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def _write_ipc_batches(path: Path, table: pa.Table, *, batch_size: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            for batch in table.to_batches(max_chunksize=batch_size):
                writer.write_batch(batch)
    return str(path)


def _assert_raw_candidate_plans_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert set(left) == set(right)
    for key in sorted(set(left).difference({"telemetry"})):
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            left_array = np.asarray(left_value)
            right_array = np.asarray(right_value)
            if left_array.dtype.kind == "f" or right_array.dtype.kind == "f":
                np.testing.assert_allclose(left_array, right_array, rtol=1e-6, atol=1e-6, err_msg=key)
            else:
                np.testing.assert_array_equal(left_array, right_array, err_msg=key)
        else:
            assert left_value == right_value, key


def _write_tiny_name_counts_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    mappings = (
        {"alice": 10.0, "bob": 30.0},
        {"wang": 20.0, "jones": 40.0},
        {"alice wang": 5.0, "bob jones": 6.0},
        {"wang a": 8.0, "jones b": 9.0},
    )
    index_path, _metrics = write_name_counts_index(tmp_path, mappings, tiny_name_counts_provenance())
    return index_path


def _swap_first_two_name_count_records(index_root: str | Path, kind: str) -> None:
    record_path = _name_count_record_path(index_root, kind)
    payload = bytearray(record_path.read_bytes())
    first_start = _NAME_COUNTS_INDEX_HEADER_LEN
    second_start = first_start + _NAME_COUNTS_INDEX_RECORD_LEN
    third_start = second_start + _NAME_COUNTS_INDEX_RECORD_LEN
    first_record = bytes(payload[first_start:second_start])
    second_record = bytes(payload[second_start:third_start])
    payload[first_start:second_start] = second_record
    payload[second_start:third_start] = first_record
    record_path.write_bytes(payload)
    _refresh_name_count_file_manifest(index_root, kind, record_path)


def _corrupt_first_name_count_record_name_range(index_root: str | Path, kind: str) -> None:
    record_path = _name_count_record_path(index_root, kind)
    payload = bytearray(record_path.read_bytes())
    blob_len = struct.unpack_from("<Q", payload, 24)[0]
    first_start = _NAME_COUNTS_INDEX_HEADER_LEN
    struct.pack_into("<Q", payload, first_start + 16, blob_len)
    struct.pack_into("<I", payload, first_start + 24, 1)
    record_path.write_bytes(payload)
    _refresh_name_count_file_manifest(index_root, kind, record_path)


def _refresh_name_count_file_manifest(index_root: str | Path, kind: str, record_path: Path) -> None:
    manifest_path = Path(index_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][kind]["byte_count"] = record_path.stat().st_size
    manifest["files"][kind]["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _name_count_record_path(index_root: str | Path, kind: str) -> Path:
    index_path = Path(index_root)
    manifest = json.loads((index_path / "manifest.json").read_text(encoding="utf-8"))
    record_path = Path(manifest["files"][kind]["path"])
    if not record_path.is_absolute():
        record_path = index_path / record_path
    return record_path


def _base_arrow_paths(
    tmp_path: Path,
    *,
    with_indexes: bool = True,
    years: list[int] | None = None,
) -> dict[str, str]:
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array(
                [["AI Lab"], ["AI Lab"], ["Other Lab"]],
                type=pa.list_(pa.string()),
            ),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Different Topic"], type=pa.string()),
            "venue": pa.array(["NeurIPS", "NeurIPS", "ICML"], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array(years or [2020, 2020, 2010], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q", "p1", "p1", "p2", "p2"], type=pa.string()),
            "position": pa.array([0, 1, 0, 1, 0, 1], type=pa.int64()),
            "author_name": pa.array(
                ["Alice Wang", "Ann Smith", "Alice Wang", "Ann Smith", "Bob Jones", "Carl Doe"],
                type=pa.string(),
            ),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_match", "c_other"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }
    if not with_indexes:
        return paths
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    return indexed_paths


def _raw_candidate_plan_arrow(
    paths: dict[str, str],
    query_signature_ids: list[str],
    *,
    top_k: int = 25,
    query_view: str = "auto",
    orcid_enabled: bool = True,
    num_threads: int | None = None,
    max_exemplars: int = 4,
) -> dict[str, Any]:
    planner = _raw_candidate_planner_from_query_signatures(
        paths,
        query_signature_ids,
        top_k=top_k,
        query_view=query_view,
        orcid_enabled=orcid_enabled,
        num_threads=num_threads,
        max_exemplars=max_exemplars,
    )
    return planner.plan_query_signatures()


def _raw_candidate_planner_from_query_signatures(
    paths: dict[str, str],
    query_signature_ids: list[str],
    *,
    top_k: int = 25,
    query_view: str = "auto",
    orcid_enabled: bool = True,
    num_threads: int | None = None,
    max_exemplars: int = 4,
) -> Any:
    query_signatures_path = Path(paths["signatures"]).parent / "incremental_query_signatures.arrow"
    write_incremental_query_signatures_arrow(
        query_signatures_path,
        query_signature_ids,
        query_views=[query_view] * len(query_signature_ids),
    )
    return s2and_rust.RawBlockQueryCandidatePlanner.from_query_signatures(
        {**paths, "query_signatures": str(query_signatures_path)},
        top_k=top_k,
        orcid_enabled=orcid_enabled,
        num_threads=num_threads,
        max_exemplars=max_exemplars,
    )


def _raw_plan_for_base_paths(paths: dict[str, str]) -> dict[str, Any]:
    return _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )


def test_raw_arrow_candidate_planner_rejects_out_of_range_seed_year(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path, years=[2020, 2**63 - 1, 2010])

    with pytest.raises(ValueError, match="raw Arrow summary year is outside the supported i32 range"):
        _raw_candidate_planner_from_query_signatures(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_planner_matches_one_shot_plan(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)

    one_shot = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    planner = _raw_candidate_planner_from_query_signatures(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    planned = planner.plan(["q1"])

    _assert_raw_candidate_plans_equal(planned, one_shot)
    assert planner.build_telemetry()["query_signature_count"] == 1
    assert planner.build_telemetry()["planner_seed_state"] == 1
    assert planned["telemetry"]["planner_seed_state_reused"] == 1
    assert planned["telemetry"]["timings"]["read_cluster_seeds_secs"] == 0.0
    assert planned["telemetry"]["timings"]["read_name_counts_secs"] == 0.0


def test_raw_arrow_candidate_planner_ingests_query_signature_request_table(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    query_signatures_path = tmp_path / "incremental_query_signatures.arrow"
    write_incremental_query_signatures_arrow(
        query_signatures_path,
        ["q1"],
        query_views=["full"],
        query_authors=["Alice Wang"],
    )
    request_paths = {**paths, "query_signatures": str(query_signatures_path)}

    one_shot = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    planner = s2and_rust.RawBlockQueryCandidatePlanner.from_query_signatures(
        request_paths,
        top_k=2,
        orcid_enabled=False,
        num_threads=1,
    )
    planned = planner.plan_query_signatures()

    _assert_raw_candidate_plans_equal(planned, one_shot)
    assert planned["query_signature_ids"] == ["q1"]
    assert planned["query_views"] == ["full"]
    assert planner.build_telemetry()["query_signature_count"] == 1


def test_raw_arrow_candidate_planner_can_plan_bounded_auto_queries(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    query_signatures_path = tmp_path / "empty_incremental_query_signatures.arrow"
    write_incremental_query_signatures_arrow(query_signatures_path, [])
    request_paths = {**paths, "query_signatures": str(query_signatures_path)}

    strict_planner = s2and_rust.RawBlockQueryCandidatePlanner.from_query_signatures(
        request_paths,
        top_k=2,
        orcid_enabled=False,
        num_threads=1,
    )
    with pytest.raises(ValueError, match="outside the planner query set"):
        strict_planner.plan(["q1"])

    reusable_planner = s2and_rust.RawBlockQueryCandidatePlanner.from_auto_queries(
        paths,
        top_k=2,
        orcid_enabled=False,
        num_threads=1,
    )
    planned = reusable_planner.plan(["q1"])
    one_shot = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="auto",
        orcid_enabled=False,
        num_threads=1,
    )

    _assert_raw_candidate_plans_equal(planned, one_shot)
    assert reusable_planner.build_telemetry()["query_signature_count"] == 0
    with pytest.raises(RuntimeError, match="requires explicit plan"):
        reusable_planner.plan_query_signatures()


def test_reusable_raw_arrow_candidate_planner_owns_lookup_index_snapshots(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    planner = s2and_rust.RawBlockQueryCandidatePlanner.from_auto_queries(
        paths,
        top_k=2,
        orcid_enabled=False,
        num_threads=1,
    )
    for key in (
        "signatures_batch_index",
        "papers_batch_index",
        "paper_authors_batch_index",
    ):
        Path(paths[key]).unlink()

    first = planner.plan(["q1"])
    second = planner.plan(["q1"])

    _assert_raw_candidate_plans_equal(first, second)
    assert first["telemetry"]["planner_seed_state_reused"] == 1


@pytest.mark.parametrize(
    ("table_name", "expected_error"),
    [
        ("papers", r"signatures reference missing paper_id 'p1' in papers Arrow input"),
        ("paper_authors", r"paper_authors Arrow input is missing rows for paper_id 'p1'"),
    ],
)
def test_raw_arrow_candidate_planner_requires_seed_paper_metadata(
    tmp_path: Path,
    table_name: str,
    expected_error: str,
) -> None:
    paths = _base_arrow_paths(tmp_path, with_indexes=False)
    with pa.memory_map(paths[table_name], "r") as source:
        table = pa.ipc.open_file(source).read_all()
    keep_mask = pa.array([paper_id != "p1" for paper_id in table["paper_id"].to_pylist()])
    paths[table_name] = _write_ipc(
        tmp_path / f"{table_name}_without_seed_paper.arrow",
        table.filter(keep_mask),
    )
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    with pytest.raises(ValueError, match=expected_error):
        _raw_candidate_planner_from_query_signatures(
            indexed_paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


@pytest.mark.parametrize(
    ("table_name", "expected_error"),
    [
        ("papers", r"signatures reference missing paper_id 'p_q' in papers Arrow input"),
        ("paper_authors", r"paper_authors Arrow input is missing rows for paper_id 'p_q'"),
    ],
)
def test_raw_arrow_candidate_planner_requires_query_paper_metadata(
    tmp_path: Path,
    table_name: str,
    expected_error: str,
) -> None:
    paths = _base_arrow_paths(tmp_path, with_indexes=False)
    with pa.memory_map(paths[table_name], "r") as source:
        table = pa.ipc.open_file(source).read_all()
    keep_mask = pa.array([paper_id != "p_q" for paper_id in table["paper_id"].to_pylist()])
    paths[table_name] = _write_ipc(
        tmp_path / f"{table_name}_without_query_paper.arrow",
        table.filter(keep_mask),
    )
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    planner = _raw_candidate_planner_from_query_signatures(
        indexed_paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    with pytest.raises(ValueError, match=expected_error):
        planner.plan(["q1"])


def test_raw_arrow_candidate_planner_rejects_duplicate_query_signature_request_rows(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    query_signatures_path = tmp_path / "duplicate_incremental_query_signatures.arrow"
    _write_ipc(
        query_signatures_path,
        pa.table(
            {
                "signature_id": pa.array(["q1", "q1"], type=pa.string()),
                "query_view": pa.array(["full", "full"], type=pa.string()),
                "query_author": pa.array(["Alice Wang", "Alice Wang"], type=pa.string()),
            }
        ),
    )

    with pytest.raises(ValueError, match="duplicate signature_id"):
        s2and_rust.RawBlockQueryCandidatePlanner.from_query_signatures(
            {**paths, "query_signatures": str(query_signatures_path)},
            top_k=2,
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_planner_filters_batch_query_seed_overlap(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["cluster_seeds"] = _write_ipc(
        tmp_path / "cluster_seeds_with_query.arrow",
        pa.table(
            {
                "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
                "cluster_id": pa.array(["c_query", "c_match", "c_other"], type=pa.string()),
            }
        ),
    )

    one_shot = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    planner = _raw_candidate_planner_from_query_signatures(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    planned = planner.plan(["q1"])

    _assert_raw_candidate_plans_equal(planned, one_shot)
    assert "q1" not in planned["component_members"].get("c_query", [])


def test_raw_arrow_candidate_planner_rejects_multi_query_seed_overlap(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    planner = _raw_candidate_planner_from_query_signatures(
        paths,
        ["q1", "s1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    with pytest.raises(ValueError, match="singleton query windows"):
        planner.plan(["q1", "s1"])

    planned = planner.plan(["q1"])
    assert planned["row_component_keys"] == ["c_match", "c_other"]
    assert planned["right_signature_ids"] == ["s1", "s2"]


def test_raw_arrow_candidate_planner_requires_indexes(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path, with_indexes=False)

    with pytest.raises(ValueError, match="batch lookup index"):
        _raw_candidate_planner_from_query_signatures(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_filters_cluster_seed_disallows(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["cluster_seed_disallows"] = _write_ipc(
        tmp_path / "cluster_seed_disallows.arrow",
        pa.table(
            {
                "signature_id_1": pa.array(["q1"], type=pa.string()),
                "signature_id_2": pa.array(["s2"], type=pa.string()),
            }
        ),
    )

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["row_component_keys"] == ["c_match"]
    assert raw_plan["left_signature_ids"] == ["q1"]
    assert raw_plan["right_signature_ids"] == ["s1"]
    assert raw_plan["telemetry"]["cluster_seed_disallow_pair_count"] == 1
    assert raw_plan["telemetry"]["cluster_seed_disallowed_candidate_count"] == 1


def test_reusable_raw_arrow_planner_applies_dynamic_disallows_without_retaining_them(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    planner = s2and_rust.RawBlockQueryCandidatePlanner.from_auto_queries(
        paths,
        top_k=1,
        orcid_enabled=False,
        num_threads=1,
    )

    baseline = planner.plan(["q1"])
    excluded = planner.plan(["q1"], [("q1", "s1")])
    replanned = planner.plan(["q1"])

    assert baseline["row_component_keys"] == ["c_match"]
    assert excluded["row_component_keys"] == ["c_other"]
    assert excluded["telemetry"]["cluster_seed_disallow_pair_count"] == 1
    assert excluded["telemetry"]["cluster_seed_disallowed_candidate_count"] == 1
    assert replanned["row_component_keys"] == ["c_match"]
    assert replanned["telemetry"]["cluster_seed_disallow_pair_count"] == 0


def test_raw_arrow_candidate_plan_rejects_disallow_with_unknown_seed_endpoint(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["cluster_seed_disallows"] = _write_ipc(
        tmp_path / "cluster_seed_disallows.arrow",
        pa.table(
            {
                "signature_id_1": pa.array(["q1"], type=pa.string()),
                "signature_id_2": pa.array(["missing_seed"], type=pa.string()),
            }
        ),
    )

    with pytest.raises(ValueError, match="unknown seed endpoint"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_keeps_zero_specter_vectors(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["specter"] = _write_ipc(
        tmp_path / "specter.arrow",
        pa.table(
            {
                "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], type=pa.float32()),
                    2,
                ),
            }
        ),
    )
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["telemetry"]["specter_count"] == 3
    assert np.isfinite(np.asarray(raw_plan["specter_centroid_similarity"], dtype=np.float32)).all()


def test_rust_featurizer_from_arrow_paths_accepts_empty_specter_table(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path, with_indexes=False)
    paths["specter"] = _write_ipc(
        tmp_path / "specter.arrow",
        pa.table(
            {
                "paper_id": pa.array([], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([], type=pa.float32()), 1),
            }
        ),
    )
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    featurizer = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        ["q1", "s1"],
        set(),
        True,
        0.0,
        10000.0,
        1,
    )
    signature_index = {str(signature_id): index for index, signature_id in enumerate(featurizer.signature_ids())}
    matrix = np.asarray(
        featurizer.featurize_pairs_matrix_indexed(
            [(signature_index["q1"], signature_index["s1"])],
            None,
            1,
            np.nan,
        )
    )

    assert matrix.shape == (1, 33)
    assert np.isnan(matrix).any()


@pytest.mark.parametrize(
    ("name_tuples", "message"),
    [
        (None, "requires explicit name-tuple pairs"),
        ("filtered", "explicit collection of pairs"),
        ("aliases.txt", "explicit collection of pairs"),
    ],
)
def test_rust_featurizer_requires_python_loaded_name_tuple_pairs(
    tmp_path: Path,
    name_tuples: object,
    message: str,
) -> None:
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(_base_arrow_paths(tmp_path), tmp_path)

    with pytest.raises((TypeError, ValueError), match=message):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths,
            ["q1", "s1"],
            name_tuples,
            True,
            0.0,
            10000.0,
            1,
        )


def test_raw_arrow_candidate_plan_rejects_hidden_query_view(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)

    with pytest.raises(ValueError, match="unknown query_view"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="initial_only_no_specter",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_batch_indexes_bound_rows(tmp_path: Path) -> None:
    irrelevant_count = 24
    signature_ids = ["q1", "s1", "s2"] + [f"junk_sig_{index}" for index in range(irrelevant_count)]
    paper_ids = ["p_q", "p1", "p2"] + [f"junk_paper_{index}" for index in range(irrelevant_count)]
    signatures = pa.table(
        {
            "signature_id": pa.array(signature_ids, type=pa.string()),
            "paper_id": pa.array(paper_ids, type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"] + ["Noise"] * irrelevant_count, type=pa.string()),
            "author_middle": pa.array(["", "", ""] + [""] * irrelevant_count, type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"] + ["Ignored"] * irrelevant_count, type=pa.string()),
            "author_suffix": pa.array(["", "", ""] + [""] * irrelevant_count, type=pa.string()),
            "author_affiliations": pa.array(
                [["AI Lab"], ["AI Lab"], ["Other Lab"]] + [[] for _ in range(irrelevant_count)],
                type=pa.list_(pa.string()),
            ),
            "author_orcid": pa.array([None] * len(signature_ids), type=pa.string()),
            "author_position": pa.array([0] * len(signature_ids), type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(paper_ids, type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Different Topic"] + ["Noise"] * irrelevant_count),
            "venue": pa.array(["NeurIPS", "NeurIPS", "ICML"] + [""] * irrelevant_count),
            "journal_name": pa.array(["", "", ""] + [""] * irrelevant_count),
            "year": pa.array([2020, 2020, 2010] + [1990] * irrelevant_count, type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(paper_ids, type=pa.string()),
            "position": pa.array([0] * len(paper_ids), type=pa.int64()),
            "author_name": pa.array(["Alice Wang", "Alice Wang", "Bob Jones"] + ["Noise"] * irrelevant_count),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_match", "c_other"], type=pa.string()),
        }
    )
    specter = pa.table(
        {
            "paper_id": pa.array(paper_ids, type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(
                    [1.0, 0.0, 1.0, 0.0, 0.0, 1.0] + [0.0, 0.0] * irrelevant_count,
                    type=pa.float32(),
                ),
                2,
            ),
        }
    )
    batch_size = 1
    paths = {
        "signatures": _write_ipc_batches(tmp_path / "signatures.arrow", signatures, batch_size=batch_size),
        "papers": _write_ipc_batches(tmp_path / "papers.arrow", papers, batch_size=batch_size),
        "paper_authors": _write_ipc_batches(
            tmp_path / "paper_authors.arrow",
            paper_authors,
            batch_size=batch_size,
        ),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
        "specter": _write_ipc_batches(tmp_path / "specter.arrow", specter, batch_size=batch_size),
    }
    indexed_paths, index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    indexed_plan = _raw_candidate_plan_arrow(
        indexed_paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    telemetry = indexed_plan["telemetry"]
    assert telemetry["indexed_arrow_candidate_plan"] is True
    assert telemetry["signature_count"] == 3
    assert telemetry["paper_count"] == 3
    assert telemetry["paper_author_paper_count"] == 3
    assert telemetry["specter_count"] == 3
    assert telemetry["signature_rows_scanned"] == 1
    assert telemetry["paper_rows_scanned"] == 1
    assert telemetry["paper_author_rows_scanned"] == 1
    assert telemetry["specter_rows_scanned"] == 1
    timings = telemetry["timings"]
    assert isinstance(timings["drop_secs"], float)
    assert timings["drop_secs"] >= 0.0
    assert isinstance(timings["wall_secs"], float)
    assert timings["wall_secs"] >= timings["drop_secs"]
    assert index_metrics["signatures_batch_index"]["record_count"] == len(signature_ids)


def test_raw_arrow_candidate_plan_extra_hash_selected_batch_is_exact_filtered(tmp_path: Path) -> None:
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2", "bad"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2", None], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob", "Bad"], type=pa.string()),
            "author_middle": pa.array(["", "", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones", "Row"], type=pa.string()),
            "author_suffix": pa.array(["", "", "", ""], type=pa.string()),
            "author_affiliations": pa.array([["AI Lab"], ["AI Lab"], ["Other Lab"], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Different Topic"], type=pa.string()),
            "venue": pa.array(["NeurIPS", "NeurIPS", "ICML"], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array([2020, 2020, 2010], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q", "p1", "p2"], type=pa.string()),
            "position": pa.array([0, 1, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang", "Ann Smith", "Alice Wang", "Bob Jones"], type=pa.string()),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_match", "c_other"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc_batches(tmp_path / "signatures.arrow", signatures, batch_size=1),
        "papers": _write_ipc_batches(tmp_path / "papers.arrow", papers, batch_size=1),
        "paper_authors": _write_ipc_batches(tmp_path / "paper_authors.arrow", paper_authors, batch_size=1),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    _append_batch_index_record(indexed_paths["signatures_batch_index"], key="q1", batch_index=3)

    plan = _raw_candidate_plan_arrow(
        indexed_paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    assert plan["right_signature_ids"] == ["s1", "s2"]
    assert plan["telemetry"]["signature_count"] == 3
    assert plan["telemetry"]["signature_rows_scanned"] == 2


def test_raw_arrow_candidate_plan_rejects_out_of_range_batch_index(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    _append_batch_index_record(
        paths["signatures_batch_index"],
        key="s1",
        batch_index=999,
        replace_existing=True,
    )
    write_test_arrow_artifact_manifest(tmp_path, paths)

    with pytest.raises(ValueError, match="batch index 999 is out of bounds"):
        validate_arrow_publication_artifacts(
            paths,
            require_specter=False,
            require_name_counts_index=False,
        )

    with pytest.raises(ValueError, match="Cannot set batch to index 999"):
        _raw_candidate_planner_from_query_signatures(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_rust_featurizer_from_arrow_paths_empty_indexed_keep_set_skips_stale_validation(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    for key in ("signatures", "papers", "paper_authors"):
        with Path(paths[key]).open("ab") as outfile:
            outfile.write(b"\0")

    featurizer = s2and_rust.RustFeaturizer.from_arrow_paths(
        indexed_paths,
        [],
        set(),
        True,
        0.0,
        10000.0,
        1,
    )

    assert tuple(featurizer.signature_ids()) == ()


def test_raw_arrow_candidate_plan_rejects_stale_batch_index(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    with Path(paths["signatures"]).open("ab") as outfile:
        outfile.write(b"\0")

    with pytest.raises(ValueError, match="stale"):
        _raw_candidate_plan_arrow(
            indexed_paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_arrow_batch_lookup_index_accepts_transport_mtime_change(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    signatures_path = Path(paths["signatures"])
    stat = signatures_path.stat()
    os.utime(signatures_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    plan = _raw_candidate_plan_arrow(
        indexed_paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    assert plan["query_signature_ids"] == ["q1"]

    _index_path, reuse_metrics = write_arrow_batch_lookup_index(
        signatures_path,
        indexed_paths["signatures_batch_index"],
        key_column="signature_id",
        table_name="signatures",
        overwrite=False,
    )
    assert reuse_metrics["reused"] is True


def test_arrow_batch_lookup_index_rejects_wrong_key_column_reuse(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    bad_index_path, _metrics = write_arrow_batch_lookup_index(
        paths["signatures"],
        tmp_path / "signatures.bad_key_batch_index.bin",
        key_column="paper_id",
        table_name="signatures",
        overwrite=True,
    )
    indexed_paths = dict(paths)
    indexed_paths["signatures_batch_index"] = bad_index_path

    with pytest.raises(ValueError, match="different key column"):
        _raw_candidate_plan_arrow(
            indexed_paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_arrow_batch_lookup_index_rejects_same_size_same_mtime_sampled_source_change(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    filler_count = 30_000
    filler_ids = [f"x{index:013d}" for index in range(filler_count)]
    paths["signatures"] = _write_ipc_batches(
        tmp_path / "signatures.arrow",
        pa.table(
            {
                "signature_id": pa.array(["q1", "s1", "s2", *filler_ids], type=pa.string()),
                "paper_id": pa.array(
                    ["p_q", "p1", "p2", *[f"p_x{index}" for index in range(filler_count)]], type=pa.string()
                ),
                "author_first": pa.array(["Alice", "Alice", "Bob", *(["Filler"] * filler_count)], type=pa.string()),
                "author_middle": pa.array(["", "", "", *([""] * filler_count)], type=pa.string()),
                "author_last": pa.array(["Wang", "Wang", "Jones", *(["Person"] * filler_count)], type=pa.string()),
                "author_suffix": pa.array(["", "", "", *([""] * filler_count)], type=pa.string()),
                "author_affiliations": pa.array(
                    [["AI Lab"], ["AI Lab"], ["Other Lab"], *([[]] * filler_count)],
                    type=pa.list_(pa.string()),
                ),
                "author_orcid": pa.array([None, None, None, *([None] * filler_count)], type=pa.string()),
                "author_position": pa.array([0, 0, 0, *([0] * filler_count)], type=pa.int64()),
            }
        ),
        batch_size=1000,
    )
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    signatures_path = Path(paths["signatures"])
    stat = signatures_path.stat()
    payload = signatures_path.read_bytes()
    old_value = b"q1"
    new_value = b"qX"
    rewrite_offset = payload.index(old_value)
    assert len(old_value) == len(new_value)
    assert rewrite_offset < 65_536
    signatures_path.write_bytes(payload[:rewrite_offset] + new_value + payload[rewrite_offset + len(old_value) :])
    os.utime(signatures_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    with pytest.raises(ValueError, match="stale"):
        _raw_candidate_plan_arrow(
            indexed_paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_rust_featurizer_from_arrow_paths_deduplicates_unsorted_requested_ids(tmp_path: Path) -> None:
    unindexed_paths = _base_arrow_paths(tmp_path, with_indexes=False)

    with pytest.raises(ValueError, match="filtered full scan"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            unindexed_paths,
            ["q1", "s1"],
            set(),
            True,
            0.0,
            10000.0,
            1,
        )

    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(unindexed_paths, tmp_path)
    featurizer = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        ["q1", "s1", "q1", "s2", "s1"],
        set(),
        True,
        0.0,
        10000.0,
        1,
    )

    assert tuple(featurizer.signature_ids()) == ("q1", "s1", "s2")


def test_rust_featurizer_from_arrow_paths_rejects_null_author_position(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    with pa.memory_map(paths["signatures"], "r") as source:
        signatures = pa.ipc.open_file(source).read_all()
    position_index = signatures.schema.get_field_index("author_position")
    signatures = signatures.set_column(
        position_index,
        "author_position",
        pa.array([None, 0, 0], type=pa.int64()),
    )
    paths["signatures"] = _write_ipc(tmp_path / "signatures_with_null_position.arrow", signatures)
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    with pytest.raises(ValueError, match="author_position is null"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths,
            ["q1", "s1"],
            set(),
            True,
            0.0,
            10000.0,
            1,
        )


def test_rust_featurizer_from_arrow_paths_reuses_cached_language(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    with pa.memory_map(paths["papers"], "r") as source:
        papers = pa.ipc.open_file(source).read_all()
    papers = papers.append_column("predicted_language", pa.array(["en", "es", "fr"], type=pa.string()))
    papers = papers.append_column("is_reliable", pa.array([True, True, True], type=pa.bool_()))
    papers = papers.append_column("language_reliability", pa.array([0.91, 0.73, 0.64], type=pa.float64()))
    paths["papers"] = _write_ipc(tmp_path / "papers_with_language.arrow", papers)
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    featurizer = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        ["q1", "s1"],
        set(),
        True,
        0.0,
        10000.0,
        1,
    )

    assert tuple(featurizer.signature_ids()) == ("q1", "s1")
    features = _indexed_pair_matrix(featurizer, [("q1", "s1")])
    np.testing.assert_allclose(
        features[0, 18:21],
        [1.0, 0.0, 0.73],
        rtol=0.0,
        atol=1e-12,
        err_msg="english_count, same_language, and language_reliability_min must use cached Arrow values",
    )


def test_rust_featurizer_from_arrow_paths_uses_batch_indexes(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path, with_indexes=False)
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    indexed = s2and_rust.RustFeaturizer.from_arrow_paths(
        indexed_paths,
        ["q1", "s1"],
        set(),
        True,
        0.0,
        10000.0,
        1,
    )

    assert tuple(indexed.signature_ids()) == ("q1", "s1")
    assert _indexed_pair_matrix(indexed, [("q1", "s1")]).shape == (1, 33)

    with Path(paths["signatures"]).open("ab") as outfile:
        outfile.write(b"\0")
    with pytest.raises(ValueError, match="stale"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            indexed_paths,
            ["q1", "s1"],
            set(),
            True,
            0.0,
            10000.0,
            1,
        )


def test_raw_arrow_candidate_plan_orcid_override_returns_all_matches(tmp_path: Path) -> None:
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s_good", "s_middle", "s_year", "s_none"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p_good", "p_middle", "p_year", "p_none"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Alice", "Alice", "Alice"], type=pa.string()),
            "author_middle": pa.array(["Q", "Q", "Z", "Q", "Q"], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Wang", "Wang", "Wang"], type=pa.string()),
            "author_suffix": pa.array(["", "", "", "", ""], type=pa.string()),
            "author_affiliations": pa.array([[], [], [], [], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array(
                [
                    "https://orcid.org/0000\u20100002\u20101825\u20100097",
                    "0000\u20110002\u20111825\u20110097",
                    "ORCID: 0000000218250097",
                    "0000-0002-1825-0097",
                    None,
                ],
                type=pa.string(),
            ),
            "author_position": pa.array([0, 0, 0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_good", "p_middle", "p_year", "p_none"], type=pa.string()),
            "title": pa.array(["", "", "", "", ""], type=pa.string()),
            "venue": pa.array(["", "", "", "", ""], type=pa.string()),
            "journal_name": pa.array(["", "", "", "", ""], type=pa.string()),
            "year": pa.array([2024, 2024, 2024, 1900, 2024], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_good", "p_middle", "p_year", "p_none"], type=pa.string()),
            "position": pa.array([0, 0, 0, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang"] * 5, type=pa.string()),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s_good", "s_middle", "s_year", "s_none"], type=pa.string()),
            "cluster_id": pa.array(["c_good", "c_middle", "c_year", "c_none"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=1,
        query_view="full",
        orcid_enabled=True,
        num_threads=1,
    )

    assert set(plan["row_component_keys"]) == {"c_good", "c_middle", "c_year"}
    assert "c_none" not in plan["row_component_keys"]
    assert plan["row_orcid_match"].tolist() == [1, 1, 1]


def test_raw_arrow_candidate_plan_orcid_override_respects_seed_disallows(tmp_path: Path) -> None:
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s_good", "s_other"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p_good", "p_other"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Alice"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Wang"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([[], [], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array(
                ["0000-0002-1825-0097", "0000-0002-1825-0097", "0000-0002-1825-0097"],
                type=pa.string(),
            ),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_good", "p_other"], type=pa.string()),
            "title": pa.array(["", "", ""], type=pa.string()),
            "venue": pa.array(["", "", ""], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array([2024, 2024, 2024], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_good", "p_other"], type=pa.string()),
            "position": pa.array([0, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang"] * 3, type=pa.string()),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s_good", "s_other"], type=pa.string()),
            "cluster_id": pa.array(["c_good", "c_other"], type=pa.string()),
        }
    )
    disallows = pa.table(
        {
            "signature_id_1": pa.array(["q1"], type=pa.string()),
            "signature_id_2": pa.array(["s_good"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
        "cluster_seed_disallows": _write_ipc(tmp_path / "cluster_seed_disallows.arrow", disallows),
    }
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=1,
        query_view="full",
        orcid_enabled=True,
        num_threads=1,
    )

    # ORCID override now respects `cluster_seed_disallows`: the disallow
    # (q1, s_good) excludes the c_good component even when the query's ORCID
    # matches a seed in that component. Only c_other survives.
    assert set(plan["row_component_keys"]) == {"c_other"}
    assert set(plan["left_signature_ids"]) == {"q1"}
    assert set(plan["right_signature_ids"]) == {"s_other"}
    assert plan["row_orcid_match"].tolist() == [1]
    assert plan["telemetry"]["cluster_seed_disallowed_candidate_count"] == 1


def test_raw_arrow_candidate_plan_rejects_missing_query_position(
    tmp_path: Path,
) -> None:
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s_self", "s_real"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p_self", "p_real"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Alice"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Wang"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([[], [], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([None, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_self", "p_real"], type=pa.string()),
            "title": pa.array(["", "", ""], type=pa.string()),
            "venue": pa.array(["", "", ""], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array([2024, 2024, 2024], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q", "p_self", "p_self", "p_real", "p_real"], type=pa.string()),
            "position": pa.array([0, 1, 0, 1, 0, 1], type=pa.int64()),
            "author_name": pa.array(
                ["Alice Wang", "Ann Smith", "Alice Wang", "Alice Wang", "Alice Wang", "Ann Smith"],
                type=pa.string(),
            ),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s_self", "s_real"], type=pa.string()),
            "cluster_id": pa.array(["c_self", "c_real"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    with pytest.raises(ValueError, match="author_position is null"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_planner_rejects_missing_seed_position(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path, with_indexes=False)
    with pa.memory_map(paths["signatures"], "r") as source:
        signatures = pa.ipc.open_file(source).read_all()
    position_index = signatures.schema.get_field_index("author_position")
    signatures = signatures.set_column(
        position_index,
        "author_position",
        pa.array([0, None, 0], type=pa.int64()),
    )
    paths["signatures"] = _write_ipc(tmp_path / "signatures_with_null_seed_position.arrow", signatures)
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    with pytest.raises(ValueError, match="author_position is null"):
        _raw_candidate_plan_arrow(
            paths,
            ["q1"],
            top_k=2,
            query_view="full",
            orcid_enabled=False,
            num_threads=1,
        )


def test_raw_arrow_candidate_plan_matches_multi_query_auto_views_and_specter(tmp_path: Path) -> None:
    signatures = pa.table(
        {
            "signature_id": pa.array(["q_full", "q_initial", "s_full", "s_initial", "s_other"], type=pa.string()),
            "paper_id": pa.array(["p_qf", "p_qi", "p_full", "p_initial", "p_other"], type=pa.string()),
            "author_first": pa.array(["Alice", "A", "Alice", "A", "Carol"], type=pa.string()),
            "author_middle": pa.array(["B", "", "B", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Li", "Wang", "Li", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", "", "", ""], type=pa.string()),
            "author_affiliations": pa.array(
                [["AI Lab"], ["Robotics Center"], ["AI Lab"], ["Robotics Center"], []],
                type=pa.list_(pa.string()),
            ),
            "author_orcid": pa.array([None, None, None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_qf", "p_qi", "p_full", "p_initial", "p_other"], type=pa.string()),
            "title": pa.array(
                ["Graph Models", "Robot Planning", "Graph Models", "Robot Planning", ""],
                type=pa.string(),
            ),
            "venue": pa.array(["NeurIPS", "RSS", "NeurIPS", "RSS", ""], type=pa.string()),
            "journal_name": pa.array(["", "", "", "", ""], type=pa.string()),
            "year": pa.array([2020, 2022, 2020, 2022, None], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(
                ["p_qf", "p_qf", "p_qi", "p_qi", "p_full", "p_full", "p_initial", "p_initial", "p_other"],
                type=pa.string(),
            ),
            "position": pa.array([0, 1, 0, 1, 0, 1, 0, 1, 0], type=pa.int64()),
            "author_name": pa.array(
                [
                    "Alice Wang",
                    "Ann Smith",
                    "A Li",
                    "Ben Stone",
                    "Alice Wang",
                    "Ann Smith",
                    "A Li",
                    "Ben Stone",
                    "Carol Jones",
                ],
                type=pa.string(),
            ),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["s_full", "s_initial", "s_other"], type=pa.string()),
            "cluster_id": pa.array(["c_full", "c_initial", "c_other"], type=pa.string()),
        }
    )
    specter = pa.table(
        {
            "paper_id": pa.array(["p_qf", "p_qi", "p_full", "p_initial", "p_other"], type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(
                    [
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.2,
                        0.2,
                    ],
                    type=pa.float32(),
                ),
                2,
            ),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
        "specter": _write_ipc(tmp_path / "specter.arrow", specter),
    }
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q_full", "q_initial"],
        top_k=2,
        query_view="auto",
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["query_views"] == ["full", "initial_only"]
    assert raw_plan["left_signature_ids"] == ["q_full", "q_full", "q_initial", "q_initial"]
    assert raw_plan["right_signature_ids"] == ["s_full", "s_other", "s_initial", "s_other"]
    assert raw_plan["left_signature_ids"][0] is raw_plan["left_signature_ids"][1]
    assert raw_plan["left_signature_ids"][2] is raw_plan["left_signature_ids"][3]
    assert raw_plan["right_signature_ids"][1] is raw_plan["right_signature_ids"][3]


def test_raw_arrow_candidate_plan_excludes_query_seed_and_handles_missing_metadata(tmp_path: Path) -> None:
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array([None, None, None], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array([None, None, None], type=pa.string()),
            "author_affiliations": pa.array([None, None, []], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array([None, None, None], type=pa.string()),
            "venue": pa.array([None, None, None], type=pa.string()),
            "journal_name": pa.array([None, None, None], type=pa.string()),
            "year": pa.array([None, None, None], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "position": pa.array([0, 0, 0], type=pa.int64()),
            "author_name": pa.array(["Alice Wang", "Alice Wang", "Bob Jones"], type=pa.string()),
        }
    )
    cluster_seeds = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "cluster_id": pa.array(["c_self", "c_self", "c_other"], type=pa.string()),
        }
    )
    paths = {
        "signatures": _write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": _write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": _write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": _write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="auto",
        orcid_enabled=False,
        num_threads=1,
    )

    assert plan["telemetry"]["excluded_query_seed_count"] == 1
    assert plan["component_members"]["c_self"] == ["s1"]
    assert plan["seed_signature_ids"] == []
    assert plan["telemetry"]["payload_seed_signature_count"] == 0
    assert "seed_component_keys" not in plan
    assert "q1" not in plan["right_signature_ids"]
    assert set(plan["right_signature_ids"]) == {"s1", "s2"}
    assert plan["query_views"] == ["full"]
    np.testing.assert_array_equal(plan["row_query_has_coauthors"], np.zeros(int(plan["row_count"]), dtype=np.uint8))
    np.testing.assert_allclose(plan["coauthor_overlap"], np.zeros(int(plan["row_count"]), dtype=np.float32))

    narrow_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=1,
        query_view="auto",
        orcid_enabled=False,
        num_threads=1,
    )

    assert "c_self" in narrow_plan["component_members"]
    assert narrow_plan["component_members"]["c_other"] == ["s2"]


def test_raw_arrow_candidate_plan_accepts_missing_papers_year_column(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Different Topic"], type=pa.string()),
            "venue": pa.array(["NeurIPS", "NeurIPS", "ICML"], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
        }
    )
    paths["papers"] = _write_ipc(tmp_path / "papers_without_year.arrow", papers)
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    np.testing.assert_array_equal(raw_plan["row_query_year_missing"], np.ones(raw_plan["row_count"], dtype=np.uint8))
    np.testing.assert_array_equal(
        raw_plan["row_candidate_year_range_missing"],
        np.ones(raw_plan["row_count"], dtype=np.uint8),
    )


def test_raw_arrow_candidate_plan_bridge_maps_signature_ids_to_linker_indices(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    retrieval_batch = build_linker_retrieval_batch_from_raw_plan_bundle(
        RawArrowPlanBundle.from_native_mapping(raw_plan),
        signature_id_to_index={"q1": 7, "s1": 11, "s2": 13},
    )

    candidate_batch = retrieval_batch.candidate_batch
    assert cast(Any, candidate_batch.row_query_signature_indices).tolist() == [7, 7]
    assert candidate_batch.left_signature_indices.tolist() == [7, 7]
    assert candidate_batch.right_signature_indices.tolist() == [11, 13]
    assert candidate_batch.pair_row_indices.tolist() == [0, 1]
    assert candidate_batch.row_component_keys == ("c_match", "c_other")
    assert retrieval_batch.row_signals["query_view"].tolist() == ["full", "full"]
    np.testing.assert_array_equal(
        retrieval_batch.row_signals["retrieval_score"],
        cast(Any, candidate_batch.retrieval_scores),
    )
    assert "candidate_cluster_max_paper_author_count" in retrieval_batch.row_signals


def test_raw_arrow_plan_bundle_derives_signature_order_from_rust_plan(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    assert tuple(raw_plan) == (
        "schema_version",
        "row_count",
        "pair_count",
        "query_signature_ids",
        "query_views",
        "query_authors",
        "seed_signature_ids",
        "component_members",
        "left_signature_ids",
        "right_signature_ids",
        "pair_row_indices",
        "row_query_signature_indices",
        "row_component_keys",
        "retrieval_scores",
        "retrieval_ranks",
        *(raw_key for raw_key, _signal_key, _dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS),
        "telemetry",
    )
    expected_base_dtypes = {
        "pair_row_indices": np.uint32,
        "row_query_signature_indices": np.uint32,
        "retrieval_scores": np.float32,
        "retrieval_ranks": np.uint16,
    }
    for key, dtype in expected_base_dtypes.items():
        assert raw_plan[key].dtype == dtype
    for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        if dtype is object:
            assert isinstance(raw_plan[raw_key], list)
        else:
            expected_dtype = np.uint32 if raw_key in {"row_component_sizes", "row_named_signature_counts"} else dtype
            assert raw_plan[raw_key].dtype == expected_dtype
    bundle = RawArrowPlanBundle.from_native_mapping(raw_plan)

    assert bundle.row_count == raw_plan["row_count"]
    assert bundle.pair_count == raw_plan["pair_count"]
    assert bundle.signature_order.signature_ids == ("q1", "s1", "s2")
    assert bundle.signature_order.query_signature_ids == ("q1",)


def test_raw_arrow_plan_bundle_freezes_native_plan_values(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_plan = _minimal_raw_candidate_plan(
        row_count=1,
        pair_count=1,
        retrieval_scores=np.asarray([0.25], dtype=np.float32),
        retrieval_ranks=np.asarray([1], dtype=np.uint16),
        left_signature_ids=["q0"],
        right_signature_ids=["s0"],
        row_query_first_tokens=np.asarray(["Alice"], dtype=object),
        row_component_sizes=np.asarray([3.0], dtype=np.float32),
        component_members={"c0": ["s0"]},
        telemetry={"seed_signature_count": 1, "timings": {"total_secs": 0.5}},
    )
    bundle = RawArrowPlanBundle.from_native_mapping(raw_plan)

    bundle_arrays = (
        bundle.row_query_offsets,
        bundle.pair_row_indices,
        bundle.retrieval_scores,
        bundle.retrieval_ranks,
        *bundle.row_signals.values(),
    )
    assert all(not array.flags.writeable for array in bundle_arrays)
    with pytest.raises(ValueError, match="read-only"):
        raw_plan["retrieval_scores"][0] = 9.0
    with pytest.raises(ValueError, match="read-only"):
        raw_plan["row_query_first_tokens"][0] = "Z"

    raw_plan["query_signature_ids"][0] = "changed-query"
    raw_plan["query_views"][0] = "initial_only"
    raw_plan["query_authors"][0] = "Changed Author"
    raw_plan["row_component_keys"][0] = "changed-component"
    raw_plan["left_signature_ids"][0] = "changed-query"
    raw_plan["right_signature_ids"][0] = "changed-signature"
    raw_plan["component_members"]["c0"][0] = "changed-signature"
    raw_plan["telemetry"]["seed_signature_count"] = 99
    raw_plan["telemetry"]["timings"]["total_secs"] = 99.0

    def unexpected_conversion(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"internal bridge repeated raw-plan conversion: {args!r} {kwargs!r}")

    monkeypatch.setattr(retrieval_module, "_required_raw_plan_value", unexpected_conversion)
    monkeypatch.setattr(retrieval_module, "_raw_plan_array", unexpected_conversion)
    monkeypatch.setattr(retrieval_module, "_raw_plan_retrieval_ranks", unexpected_conversion)

    retrieval_batch = build_linker_retrieval_batch_from_raw_plan_bundle(
        bundle,
        signature_id_to_index={"q0": 7, "s0": 11},
    )

    candidate_batch = retrieval_batch.candidate_batch
    assert cast(Any, candidate_batch.row_query_signature_indices).tolist() == [7]
    assert candidate_batch.left_signature_indices.tolist() == [7]
    assert candidate_batch.right_signature_indices.tolist() == [11]
    assert candidate_batch.pair_row_indices.tolist() == [0]
    assert candidate_batch.row_component_keys == ("c0",)
    assert cast(Any, candidate_batch.retrieval_scores).tolist() == pytest.approx([0.25])
    assert cast(Any, candidate_batch.retrieval_ranks).tolist() == [1]
    assert retrieval_batch.row_signals["query_view"].tolist() == ["full"]
    assert retrieval_batch.row_signals["query_author"].tolist() == ["Alice"]
    assert retrieval_batch.row_signals["first_name_bucket"].tolist() == ["multi_letter_first"]
    assert retrieval_batch.row_signals["cluster_size"].tolist() == pytest.approx([3.0])
    assert bundle.query_signature_ids == ("q0",)
    assert bundle.row_component_keys == ("c0",)
    assert bundle.left_signature_ids == ("q0",)
    assert bundle.right_signature_ids == ("s0",)
    assert bundle.component_members["c0"] == ("s0",)
    assert bundle.telemetry is not None
    assert bundle.telemetry["seed_signature_count"] == 1
    assert bundle.telemetry["timings"]["total_secs"] == pytest.approx(0.5)


def test_raw_arrow_plan_bundle_adopts_native_numeric_arrays_without_copying() -> None:
    raw_plan = _minimal_raw_candidate_plan(
        row_count=1,
        pair_count=1,
        retrieval_scores=np.asarray([0.25], dtype=np.float32),
        retrieval_ranks=np.asarray([1], dtype=np.uint16),
        left_signature_ids=["q0"],
        right_signature_ids=["s0"],
        row_query_first_tokens=np.asarray(["Alice"], dtype=object),
        row_component_sizes=np.asarray([3.0], dtype=np.float32),
        component_members={"c0": ["s0"]},
    )
    source_arrays = {
        key: value for key, value in raw_plan.items() if isinstance(value, np.ndarray) and value.dtype != object
    }

    bundle = RawArrowPlanBundle.from_native_mapping(raw_plan)

    adopted_arrays = {
        "row_query_signature_indices": bundle.row_query_offsets,
        "pair_row_indices": bundle.pair_row_indices,
        "retrieval_scores": bundle.retrieval_scores,
        "retrieval_ranks": bundle.retrieval_ranks,
        **{
            raw_key: bundle.row_signals[signal_key]
            for raw_key, signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS
            if dtype is not object
        },
    }
    assert source_arrays.keys() <= adopted_arrays.keys()
    for key, source in source_arrays.items():
        adopted = adopted_arrays[key]
        assert np.shares_memory(adopted, source), key
        assert not source.flags.writeable, key
        assert not adopted.flags.writeable, key


def test_raw_arrow_labeled_candidate_plan_scores_frozen_rows_without_cluster_seeds(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")

    raw_plan = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1", "q1"],
        ["full", "full"],
        ["q1-full", "q1-full"],
        ["c_other", "c_match"],
        np.asarray([1, 2], dtype=np.uint16),
        {"c_match": ["s1"], "c_other": ["s2"]},
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["schema_version"] == "raw_arrow_labeled_candidate_plan_v1"
    assert raw_plan["row_component_keys"] == ["c_other", "c_match"]
    assert raw_plan["left_signature_ids"] == ["q1", "q1"]
    assert raw_plan["right_signature_ids"] == ["s2", "s1"]
    assert raw_plan["left_signature_ids"][0] is raw_plan["left_signature_ids"][1]
    q1_index = raw_plan["signature_ids"].index("q1")
    assert raw_plan["left_signature_ids"][0] is raw_plan["signature_ids"][q1_index]
    np.testing.assert_array_equal(raw_plan["pair_row_indices"], np.asarray([0, 1], dtype=np.uint32))
    assert raw_plan["retrieval_ranks"].tolist() == [2, 1]
    assert raw_plan["retrieval_scores"][1] > raw_plan["retrieval_scores"][0]
    assert raw_plan["query_views"] == ["full"]
    assert raw_plan["query_authors"] == [raw_plan["row_query_authors"][0]]
    assert raw_plan["row_query_views"] == ["full", "full"]
    assert "row_candidate_cluster_max_paper_author_count" in raw_plan
    assert raw_plan["telemetry"]["component_scope"] == "block-local"


def test_raw_arrow_labeled_candidate_plan_scores_use_all_components_for_global_df(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")
    component_members = {"c_match": ["s1"], "c_other": ["s2"]}

    one_row = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1"],
        ["full"],
        ["q1-full"],
        ["c_match"],
        np.asarray([1], dtype=np.uint16),
        component_members,
        orcid_enabled=False,
        num_threads=1,
    )
    two_rows = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1", "q1"],
        ["full", "full"],
        ["q1-full", "q1-full"],
        ["c_match", "c_other"],
        np.asarray([1, 2], dtype=np.uint16),
        component_members,
        orcid_enabled=False,
        num_threads=1,
    )

    assert one_row["telemetry"]["component_count"] == 2
    assert two_rows["telemetry"]["component_count"] == 2
    np.testing.assert_allclose(one_row["retrieval_scores"][0], two_rows["retrieval_scores"][0], rtol=1e-6, atol=1e-6)


def test_raw_arrow_candidate_plans_initial_view_keep_full_first_token(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    labeled_paths = dict(paths)
    labeled_paths.pop("cluster_seeds")

    raw_plan = s2and_rust.raw_arrow_labeled_candidate_plan(
        labeled_paths,
        ["q1"],
        ["initial_only"],
        ["q1-initial"],
        ["c_match"],
        np.asarray([1], dtype=np.uint16),
        {"c_match": ["s1"]},
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["row_query_views"] == ["initial_only"]
    assert raw_plan["query_views"] == ["initial_only"]
    assert raw_plan["query_authors"] == [raw_plan["row_query_authors"][0]]
    assert raw_plan["row_query_first_tokens"] == ["alice"]

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="initial_only",
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["query_views"] == ["initial_only"]
    assert raw_plan["row_query_first_tokens"] == ["alice", "alice"]


def test_raw_arrow_labeled_candidate_plan_applies_block_local_members(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([["AI Lab"], ["AI Lab"], ["Other Lab"]], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
            "author_block": pa.array(["block-a", "block-a", "block-b"], type=pa.string()),
        }
    )
    paths["signatures"] = _write_ipc(tmp_path / "signatures_with_blocks.arrow", signatures)
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    raw_plan = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1"],
        ["full"],
        ["q1-full"],
        ["block-a::c"],
        np.asarray([1], dtype=np.uint16),
        {"block-a::c": ["q1", "s1", "s2"]},
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["left_signature_ids"] == ["q1"]
    assert raw_plan["right_signature_ids"] == ["s1"]
    assert raw_plan["row_component_sizes"].tolist() == [1]


def test_raw_arrow_labeled_candidate_plan_drops_component_with_only_foreign_members(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths.pop("cluster_seeds")
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([[], [], []], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
            "author_block": pa.array(["block-a", "block-b", "block-b"], type=pa.string()),
        }
    )
    paths["signatures"] = _write_ipc(tmp_path / "signatures_with_foreign_component.arrow", signatures)
    paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)

    raw_plan = s2and_rust.raw_arrow_labeled_candidate_plan(
        paths,
        ["q1"],
        ["full"],
        ["q1-full"],
        ["block-a::foreign"],
        np.asarray([1], dtype=np.uint16),
        {"block-a::foreign": ["s1", "s2"]},
        orcid_enabled=False,
        num_threads=1,
    )

    assert raw_plan["left_signature_ids"] == []
    assert raw_plan["right_signature_ids"] == []
    assert raw_plan["row_component_sizes"].tolist() == [0]


def test_raw_arrow_candidate_plan_emits_native_row_signals_from_name_counts_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index", monkeypatch)

    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )

    np.testing.assert_allclose(
        raw_plan["row_last_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(20.0), 1.0 / np.sqrt(20.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        raw_plan["row_last_first_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        raw_plan["row_candidate_last_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(20.0), 1.0 / np.sqrt(40.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        raw_plan["row_candidate_last_first_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(5.0), 1.0 / np.sqrt(6.0)], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        raw_plan["row_first_prefix_x_last_first_name_count_min_rarity"],
        np.asarray([1.0 / np.sqrt(5.0), 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_rust_featurizer_from_arrow_paths_applies_cluster_seed_disallows(tmp_path: Path) -> None:
    paths = _base_arrow_paths(tmp_path)
    raw_plan = _raw_candidate_plan_arrow(
        paths,
        ["q1"],
        top_k=2,
        query_view="full",
        orcid_enabled=False,
        num_threads=1,
    )
    paths["cluster_seed_disallows"] = _write_ipc(
        tmp_path / "cluster_seed_disallows.arrow",
        pa.table(
            {
                "signature_id_1": pa.array(["q1"], type=pa.string()),
                "signature_id_2": pa.array(["s2"], type=pa.string()),
            }
        ),
    )
    signature_order = RawArrowPlanBundle.from_native_mapping(raw_plan).signature_order

    direct = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        list(signature_order.signature_ids),
        set(),
        True,
        0.0,
        10000.0,
        1,
    )
    pairs = [("q1", "s1"), ("q1", "s2")]

    assert tuple(direct.signature_ids()) == signature_order.signature_ids
    assert _indexed_pair_matrix(direct, pairs).shape == (2, 33)
    signature_index = {str(signature_id): index for index, signature_id in enumerate(direct.signature_ids())}
    assert direct.get_constraints_matrix_indexed([(signature_index["q1"], signature_index["s1"])]) == [None]
    assert direct.get_constraints_matrix_indexed([(signature_index["q1"], signature_index["s2"])]) == [10000.0]


def test_rust_featurizer_missing_name_counts_presence_is_consistent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _base_arrow_paths(tmp_path)
    signature_ids = ["q1", "s1", "s2"]

    from_arrow = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        signature_ids,
        set(),
        True,
        0.0,
        10000.0,
        1,
    )

    assert from_arrow.signature_name_counts_present() == [("q1", False), ("s1", False), ("s2", False)]

    paths_with_index = dict(paths)
    paths_with_index["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index_artifact", monkeypatch)
    with_name_counts = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths_with_index,
        signature_ids,
        set(),
        True,
        0.0,
        10000.0,
        1,
    )
    assert with_name_counts.signature_name_counts_present() == [("q1", True), ("s1", True), ("s2", True)]


def test_rust_featurizer_reuses_only_matching_planner_name_counts_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index_artifact", monkeypatch)
    planner = s2and_rust.RawBlockQueryCandidatePlanner.from_auto_queries(
        paths,
        top_k=2,
        orcid_enabled=False,
        num_threads=1,
    )
    name_counts_index = planner.name_counts_index()
    assert name_counts_index is not None
    assert name_counts_index.normalization_version == "canonical_v2"

    featurizer = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        ["q1", "s1", "s2"],
        set(),
        True,
        0.0,
        10000.0,
        1,
        name_counts_index,
    )
    assert featurizer.signature_name_counts_present() == [("q1", True), ("s1", True), ("s2", True)]

    paths_without_index = dict(paths)
    paths_without_index.pop("name_counts_index")
    with pytest.raises(ValueError, match=r"handle requires paths\['name_counts_index'\]"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths_without_index,
            ["q1", "s1", "s2"],
            set(),
            True,
            0.0,
            10000.0,
            1,
            name_counts_index,
        )

    manifest_path = Path(paths["name_counts_index"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "invalid-after-planner-build"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reused_snapshot = s2and_rust.RustFeaturizer.from_arrow_paths(
        paths,
        ["q1", "s1", "s2"],
        set(),
        True,
        0.0,
        10000.0,
        1,
        name_counts_index,
    )
    assert reused_snapshot.signature_name_counts_present() == [("q1", True), ("s1", True), ("s2", True)]
    with pytest.raises(ValueError, match="schema_version"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths,
            ["q1", "s1", "s2"],
            set(),
            True,
            0.0,
            10000.0,
            1,
        )


def test_rust_featurizer_from_arrow_paths_rejects_legacy_name_count_paths(tmp_path: Path) -> None:
    for legacy_key in ("name_counts", "name_counts_index_dir"):
        arrow_paths = _base_arrow_paths(tmp_path / legacy_key)
        arrow_paths[legacy_key] = _write_ipc(
            tmp_path / legacy_key / "legacy.arrow",
            pa.table({"count": [1.0]}),
        )

        with pytest.raises(ValueError, match="use name_counts_index"):
            s2and_rust.RustFeaturizer.from_arrow_paths(
                arrow_paths,
                ["q1", "s1", "s2"],
                set(),
                True,
                0.0,
                10000.0,
                1,
            )


def test_rust_featurizer_rejects_unsorted_name_counts_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index_artifact", monkeypatch)
    _swap_first_two_name_count_records(paths["name_counts_index"], "first")

    with pytest.raises(ValueError, match="not sorted"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths,
            ["q1", "s1", "s2"],
            set(),
            True,
            0.0,
            10000.0,
            1,
        )


def test_rust_featurizer_rejects_wrong_name_counts_index_schema_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index_artifact", monkeypatch)
    manifest_path = Path(paths["name_counts_index"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "unexpected"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths,
            ["q1", "s1", "s2"],
            set(),
            True,
            0.0,
            10000.0,
            1,
        )


def test_rust_featurizer_rejects_out_of_bounds_name_counts_index_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _base_arrow_paths(tmp_path)
    paths["name_counts_index"] = _write_tiny_name_counts_index(tmp_path / "index_artifact", monkeypatch)
    _corrupt_first_name_count_record_name_range(paths["name_counts_index"], "first")

    with pytest.raises(ValueError, match="outside blob length"):
        s2and_rust.RustFeaturizer.from_arrow_paths(
            paths,
            ["q1", "s1", "s2"],
            set(),
            True,
            0.0,
            10000.0,
            1,
        )

"""Tiny raw Arrow bundles and native planners shared across test domains."""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block import write_raw_arrow_batch_lookup_indexes
from s2and.incremental_linking.retrieval import RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS
from s2and.runtime import load_s2and_rust_extension
from tests.helpers import write_test_arrow_artifact_manifest


def write_ipc(path: Path, table: pa.Table) -> str:
    """Write a tiny IPC fixture without changing its schema or rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def base_arrow_paths(
    tmp_path: Path,
    *,
    with_indexes: bool = True,
    years: list[int] | None = None,
) -> dict[str, str]:
    """Build one matching and one unrelated seed for a single query."""
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
        "signatures": write_ipc(tmp_path / "signatures.arrow", signatures),
        "papers": write_ipc(tmp_path / "papers.arrow", papers),
        "paper_authors": write_ipc(tmp_path / "paper_authors.arrow", paper_authors),
        "cluster_seeds": write_ipc(tmp_path / "cluster_seeds.arrow", cluster_seeds),
    }
    if not with_indexes:
        return paths
    indexed_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(paths, tmp_path)
    return indexed_paths


def open_arrow_dataset(paths: dict[str, str]) -> ArrowDataset:
    """Bind a fixture's current files to a manifest and open its resources."""
    root = Path(paths["signatures"]).parent
    write_test_arrow_artifact_manifest(root, paths)
    return ArrowDataset.open(root)


def native_auto_planner(
    paths: dict[str, str],
    *,
    top_k: int,
    orcid_enabled: bool,
    num_threads: int | None,
) -> Any:
    """Create an automatic-query planner that retains its Arrow resources."""
    with open_arrow_dataset(paths) as arrow_dataset:
        return load_s2and_rust_extension().RawBlockQueryCandidatePlanner.from_auto_queries(
            arrow_dataset.native,
            paths["cluster_seeds"],
            top_k,
            cluster_seed_disallows_path=paths.get("cluster_seed_disallows"),
            orcid_enabled=orcid_enabled,
            num_threads=num_threads,
        )


def native_labeled_plan(paths: dict[str, str], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Build a native labeled plan while owning the fixture's Arrow handles."""
    with open_arrow_dataset(paths) as arrow_dataset:
        return load_s2and_rust_extension().raw_arrow_labeled_candidate_plan(arrow_dataset.native, *args, **kwargs)


def pair_position_keys(
    query_signature_ids: list[str], left_signature_ids: list[str], right_signature_ids: list[str]
) -> dict[str, Any]:
    """Plan keys for pairs given as ids: the pair signature order and each pair's positions in it."""
    order = list(dict.fromkeys([*query_signature_ids, *right_signature_ids]))
    position = {signature_id: index for index, signature_id in enumerate(order)}
    return {
        "pair_signature_ids": order,
        "left_pair_signature_indices": np.asarray([position[value] for value in left_signature_ids], dtype=np.uint32),
        "right_pair_signature_indices": np.asarray([position[value] for value in right_signature_ids], dtype=np.uint32),
    }


def plan_pair_ids(plan: Any) -> tuple[list[str], list[str]]:
    """Left and right signature ids of each pair of a planner plan, labeled plan or plan bundle."""
    if hasattr(plan, "signature_order"):
        order = plan.signature_order.signature_ids
        left, right = plan.left_pair_signature_indices, plan.right_pair_signature_indices
    elif "pair_signature_ids" in plan:
        order = plan["pair_signature_ids"]
        left, right = plan["left_pair_signature_indices"], plan["right_pair_signature_indices"]
    else:
        return list(plan["left_signature_ids"]), list(plan["right_signature_ids"])
    return [order[int(index)] for index in left], [order[int(index)] for index in right]


def minimal_raw_candidate_plan(**overrides: Any) -> dict[str, Any]:
    """Build a fresh empty raw-plan payload, with explicit row/pair overrides.

    Pairs may be given as left_signature_ids and right_signature_ids; they become
    the plan's pair signature order and positions.
    """
    left_signature_ids = overrides.pop("left_signature_ids", None)
    right_signature_ids = overrides.pop("right_signature_ids", None)
    query_signature_ids = list(overrides.pop("query_signature_ids", ["q0"]))
    row_count = int(overrides.pop("row_count", 0))
    pair_count = int(overrides.pop("pair_count", 0))
    plan: dict[str, Any] = {
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
        "component_members": {},
        "telemetry": {},
    }
    for raw_key, _signal_key, dtype in RAW_CANDIDATE_PLAN_ROW_SIGNAL_FIELDS:
        if dtype is object:
            plan[raw_key] = np.asarray([""] * row_count, dtype=object)
        else:
            plan[raw_key] = np.zeros(row_count, dtype=dtype)
    plan.update(overrides)
    if "pair_signature_ids" in plan:
        return plan
    queries = list(plan["query_signature_ids"])
    if left_signature_ids is None:
        left_signature_ids = [(queries[0] if queries else "")] * pair_count
    if right_signature_ids is None:
        right_signature_ids = [f"s{index}" for index in range(pair_count)]
    plan.update(pair_position_keys(queries, list(left_signature_ids), list(right_signature_ids)))
    return plan

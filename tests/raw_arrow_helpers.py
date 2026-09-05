"""Tiny raw Arrow bundles and native planners shared across test domains."""

from pathlib import Path
from typing import Any

import pyarrow as pa

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block import write_raw_arrow_batch_lookup_indexes
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

"""Predict author clusters with a complete S2AND production model.

Download the `inference` and `dataset` bundles into `s2and/data/` first; see docs/data.md.
The Arrow route runs in Rust through `Clusterer.predict_from_arrow` and clusters every block:

  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path s2and/data/production_model_v1.3 \
      --arrow-dataset s2and/data/benchmarks/qian \
      --n-jobs 4 --output-json qian_clusters.json

Incremental prediction links new signatures to existing author profiles through
`Clusterer.predict_incremental_from_arrow`. This example uses one block of released data.
Labeled signatures on papers before 2015 play the existing profiles; the block's other
signatures arrive as new papers:

  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path s2and/data/production_model_v1.3 \
      --arrow-dataset s2and/data/benchmarks/aminer \
      --incremental-block "j wang" \
      --seed-clusters s2and/data/benchmarks/aminer/aminer_clusters.json \
      --seed-before-year 2015 --n-jobs 4

The JSON route runs in Python through `Clusterer.predict` on an `ANDData` built from
`signatures.json`, `papers.json`, and a SPECTER2 embedding pickle:

  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path s2and/data/production_model_v1.3 \
      --json-dir path/to/records --specter-pickle path/to/specter2.pkl \
      --name-counts-index s2and/data/name_counts_index --n-jobs 4

The tutorial does not score the result.
`release_pairwise.py evaluate-heldout` computes the release metrics; see docs/reproduce.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from s2and.arrow_inputs import ArrowDataset  # noqa: E402


def _read_columns(arrow_dataset: ArrowDataset, table_name: str, columns: Sequence[str]) -> Any:
    """Read selected columns of one table of the dataset."""
    import pyarrow as pa

    with arrow_dataset.use() as lease, lease.open_file(table_name) as infile:
        with pa.PythonFile(infile, mode="r") as source:
            return pa.ipc.open_file(source).read_all().select(list(columns))


def arrow_blocks(arrow_dataset: ArrowDataset) -> dict[str, list[str]]:
    """Group the dataset's signature IDs by their S2 author block."""
    table = _read_columns(arrow_dataset, "signatures", ["signature_id", "author_block"])
    blocks: dict[str, list[str]] = defaultdict(list)
    signature_ids = table.column("signature_id").to_pylist()
    author_blocks = table.column("author_block").to_pylist()
    for signature_id, author_block in zip(signature_ids, author_blocks, strict=True):
        blocks[str(author_block)].append(str(signature_id))
    return dict(blocks)


def predict_arrow(
    clusterer: Any,
    dataset_dir: Path,
    *,
    batching_threshold: int | None,
    total_ram_bytes: int | None,
) -> dict[str, list[str]]:
    """Cluster every block of one Arrow dataset through the Rust route."""
    with ArrowDataset.open(dataset_dir, require_specter=True, require_name_counts_index=True) as arrow_dataset:
        clusters, _distances = clusterer.predict_from_arrow(
            arrow_blocks(arrow_dataset),
            arrow_dataset,
            batching_threshold=batching_threshold,
            total_ram_bytes=total_ram_bytes,
        )
    return clusters


def seed_profiles_before_year(
    arrow_dataset: ArrowDataset,
    signature_ids: Sequence[str],
    *,
    labeled_clusters: dict[str, Any],
    before_year: int,
) -> dict[str, str]:
    """Map each labeled signature on a paper before `before_year` to its labeled cluster.

    These signatures play the existing author profiles. The other signatures arrive as new papers.
    """
    signatures = _read_columns(arrow_dataset, "signatures", ["signature_id", "paper_id"])
    try:
        papers = _read_columns(arrow_dataset, "papers", ["paper_id", "year"])
    except KeyError as exc:
        # The public format makes papers.year optional; this example needs it to order the papers.
        raise ValueError("Seeding by year needs the optional papers.year column, which this dataset lacks") from exc
    signature_paper = dict(
        zip(
            map(str, signatures.column("signature_id").to_pylist()),
            map(str, signatures.column("paper_id").to_pylist()),
            strict=True,
        )
    )
    paper_year = dict(
        zip(map(str, papers.column("paper_id").to_pylist()), papers.column("year").to_pylist(), strict=True)
    )
    labels = {
        str(signature_id): str(cluster_id)
        for cluster_id, cluster in labeled_clusters.items()
        for signature_id in cluster["signature_ids"]
    }
    seeds = {}
    for signature_id in signature_ids:
        year = paper_year.get(signature_paper[signature_id])
        if signature_id in labels and year is not None and year < before_year:
            seeds[signature_id] = labels[signature_id]
    return seeds


def predict_incremental_arrow(
    clusterer: Any,
    dataset_dir: Path,
    *,
    block: str,
    seed_clusters: Path,
    seed_before_year: int,
    batching_threshold: int | None,
    total_ram_bytes: int | None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Link one block's newer signatures to profiles seeded from its older labeled signatures."""
    labeled_clusters = json.loads(seed_clusters.read_text(encoding="utf-8"))
    with ArrowDataset.open(dataset_dir, require_specter=True, require_name_counts_index=True) as arrow_dataset:
        blocks = arrow_blocks(arrow_dataset)
        if block not in blocks:
            raise ValueError(f"Block {block!r} is not in {dataset_dir}")
        seeds = seed_profiles_before_year(
            arrow_dataset,
            blocks[block],
            labeled_clusters=labeled_clusters,
            before_year=seed_before_year,
        )
        result = clusterer.predict_incremental_from_arrow(
            blocks[block],
            arrow_dataset,
            cluster_seeds_require=seeds,
            batching_threshold=batching_threshold,
            total_ram_bytes=total_ram_bytes,
        )
    return result["clusters"], seeds


def predict_json(
    clusterer: Any,
    json_dir: Path,
    *,
    specter_pickle: Path,
    name_counts_index: Path,
    n_jobs: int,
) -> dict[str, list[str]]:
    """Cluster every full block of one S2AND JSON dataset through the Python route."""
    from s2and.data import ANDData

    dataset = ANDData(
        signatures=str(json_dir / "signatures.json"),
        papers=str(json_dir / "papers.json"),
        name=json_dir.name,
        mode="inference",
        specter_embeddings=str(specter_pickle),
        name_counts_index=str(name_counts_index),
        n_jobs=n_jobs,
    )
    clusters, _distances = clusterer.predict(dataset.get_blocks(), dataset)
    return clusters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True, help="Complete production model directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--arrow-dataset", type=Path, help="Arrow dataset directory that holds manifest.json.")
    source.add_argument("--json-dir", type=Path, help="Directory with signatures.json and papers.json.")
    parser.add_argument("--specter-pickle", type=Path, help="SPECTER2 embedding pickle for --json-dir.")
    parser.add_argument("--name-counts-index", type=Path, help="Name-count index directory for --json-dir.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Threads for prediction (default: 4).")
    parser.add_argument(
        "--batching-threshold",
        type=int,
        default=None,
        help="Optional maximum subblock size for --arrow-dataset; larger blocks use native subblocking.",
    )
    parser.add_argument("--total-ram-bytes", type=int, default=None, help="Memory budget for the Arrow route.")
    parser.add_argument(
        "--incremental-block",
        default=None,
        help="Run incremental prediction on this block of --arrow-dataset instead of clustering every block.",
    )
    parser.add_argument(
        "--seed-clusters",
        type=Path,
        default=None,
        help="Labeled clusters JSON, such as benchmarks/aminer/aminer_clusters.json, for --incremental-block.",
    )
    parser.add_argument(
        "--seed-before-year",
        type=int,
        default=2015,
        help="Labeled signatures on papers before this year seed the existing profiles (default: 2015).",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="New file for the predicted clusters.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json_dir is not None and (args.specter_pickle is None or args.name_counts_index is None):
        parser.error("--json-dir requires --specter-pickle and --name-counts-index")
    # The Python route clusters full blocks only; reject subblocking before slow preprocessing starts.
    if args.json_dir is not None and args.batching_threshold is not None:
        parser.error("--batching-threshold requires --arrow-dataset; the --json-dir route clusters full blocks")
    if args.incremental_block is not None and (args.arrow_dataset is None or args.seed_clusters is None):
        parser.error("--incremental-block requires --arrow-dataset and --seed-clusters")
    if args.output_json is not None and args.output_json.exists():
        parser.error(f"--output-json already exists: {args.output_json}")

    # LightGBM and other OpenMP libraries read this variable when they first load.
    os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
    from s2and.production_model import load_production_model

    clusterer = load_production_model(args.model_path, n_jobs=args.n_jobs)
    if args.incremental_block is not None:
        clusters, seeds = predict_incremental_arrow(
            clusterer,
            args.arrow_dataset,
            block=args.incremental_block,
            seed_clusters=args.seed_clusters,
            seed_before_year=args.seed_before_year,
            batching_threshold=args.batching_threshold,
            total_ram_bytes=args.total_ram_bytes,
        )
        new_signatures = sum(len(members) for members in clusters.values()) - len(seeds)
        joined = sum(
            1
            for members in clusters.values()
            if any(str(member) in seeds for member in members)
            for member in members
            if str(member) not in seeds
        )
        print(f"Seeded {len(set(seeds.values()))} profiles with {len(seeds)} signatures.")
        print(f"Linked {new_signatures} new signatures: {joined} joined a profile, {new_signatures - joined} did not.")
    elif args.arrow_dataset is not None:
        clusters = predict_arrow(
            clusterer,
            args.arrow_dataset,
            batching_threshold=args.batching_threshold,
            total_ram_bytes=args.total_ram_bytes,
        )
    else:
        clusters = predict_json(
            clusterer,
            args.json_dir,
            specter_pickle=args.specter_pickle,
            name_counts_index=args.name_counts_index,
            n_jobs=args.n_jobs,
        )

    sizes = sorted((len(members) for members in clusters.values()), reverse=True)
    print(f"Predicted {len(sizes)} clusters for {sum(sizes)} signatures.")
    print(f"Largest cluster sizes: {sizes[:5]}")
    if args.output_json is not None:
        payload = {str(cluster_id): sorted(map(str, members)) for cluster_id, members in clusters.items()}
        args.output_json.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()

"""Predict author clusters with a complete S2AND production model.

Download the `inference` and `dataset` bundles into one folder first; see docs/reproduce.md.
The Arrow route runs in Rust through `Clusterer.predict_from_arrow`:

  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path s2and-1.3/production_model_v1.3 \
      --arrow-dataset s2and-1.3/benchmarks/qian \
      --n-jobs 4 --output-json qian_clusters.json

The JSON route runs in Python through `Clusterer.predict` on an `ANDData` built from
`signatures.json`, `papers.json`, and a SPECTER2 embedding pickle:

  uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
      --model-path s2and-1.3/production_model_v1.3 \
      --json-dir path/to/records --specter-pickle path/to/specter2.pkl \
      --name-counts-index s2and-1.3/name_counts_index --n-jobs 4

The tutorial predicts every block of one dataset and does not score the result.
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


def arrow_blocks(arrow_dataset: ArrowDataset) -> dict[str, list[str]]:
    """Group the dataset's signature IDs by their S2 author block."""
    import pyarrow as pa

    with arrow_dataset.use() as lease, lease.open_file("signatures") as infile:
        with pa.PythonFile(infile, mode="r") as source:
            table = pa.ipc.open_file(source).read_all().select(["signature_id", "author_block"])
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


def predict_json(
    clusterer: Any,
    json_dir: Path,
    *,
    specter_pickle: Path,
    name_counts_index: Path,
    n_jobs: int,
    batching_threshold: int | None,
) -> dict[str, list[str]]:
    """Cluster every block of one S2AND JSON dataset through the Python route."""
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
    clusters, _distances = clusterer.predict(dataset.get_blocks(), dataset, batching_threshold=batching_threshold)
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
        help="Optional maximum subblock size; larger blocks use the route's subblocking.",
    )
    parser.add_argument("--total-ram-bytes", type=int, default=None, help="Memory budget for the Arrow route.")
    parser.add_argument("--output-json", type=Path, default=None, help="New file for the predicted clusters.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json_dir is not None and (args.specter_pickle is None or args.name_counts_index is None):
        parser.error("--json-dir requires --specter-pickle and --name-counts-index")
    if args.output_json is not None and args.output_json.exists():
        parser.error(f"--output-json already exists: {args.output_json}")

    # LightGBM and other OpenMP libraries read this variable when they first load.
    os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
    from s2and.production_model import load_production_model

    clusterer = load_production_model(args.model_path, n_jobs=args.n_jobs)
    if args.arrow_dataset is not None:
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
            batching_threshold=args.batching_threshold,
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

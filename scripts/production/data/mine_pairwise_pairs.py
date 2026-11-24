"""Sample pairs from frozen pairwise_train blocks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from s2and.arrow_inputs import ArrowDataset
from scripts.production.data.block_roles import load_block_roles, normalize_block_key


def _real_signatures(root: Path, block: str) -> list[str]:
    with (
        ArrowDataset.open(root) as dataset,
        dataset.use() as lease,
        lease.open_file("signatures") as infile,
        pa.PythonFile(infile, mode="r") as source,
    ):
        rows = ipc.open_file(source).read_all().select(["signature_id", "author_block"]).to_pydict()
    real = set()
    for signature, raw_block in zip(rows["signature_id"], rows["author_block"], strict=True):
        if not signature or normalize_block_key(raw_block) != block or signature in real:
            raise ValueError(f"Expected unique signatures from block {block!r} in {root}")
        if not signature.startswith("visibleq:"):
            real.add(signature)
    return sorted(real)


def mine_pairs(
    roles_path: Path,
    blocks: Mapping[str, Path],
    output_path: Path,
    *,
    per_block: int,
    seed: int = 20261123,
) -> int:
    """Write a bounded uniform sample with per-block inclusion probabilities."""
    if per_block <= 0:
        raise ValueError("per_block must be positive")
    roles = load_block_roles(roles_path)
    normalized = {normalize_block_key(block): root for block, root in blocks.items()}
    if not normalized or len(normalized) != len(blocks):
        raise ValueError("Expected at least one block and no duplicate normalized keys")
    for block in normalized:
        if roles.role_for(block) != "pairwise_train":
            raise ValueError(f"Block {block!r} is outside pairwise_train")

    rng = random.Random(seed)
    rows: list[tuple[str, str, str, float]] = []
    for block, root in sorted(normalized.items()):
        signatures = _real_signatures(root, block)
        population = math.comb(len(signatures), 2)
        count = min(population, per_block)
        for index in sorted(rng.sample(range(population), count)):
            # Invert the triangular index without enumerating the pair population.
            right = (1 + math.isqrt(1 + 8 * index)) // 2
            left = index - math.comb(right, 2)
            rows.append((block, signatures[left], signatures[right], count / population))

    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block_key", "signature_id_1", "signature_id_2", "inclusion_probability"])
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    """Sample frozen pairwise training blocks from Arrow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-role-assignments", type=Path, required=True)
    parser.add_argument("--blocks-json", type=Path, required=True, help="Map block keys to Arrow directories.")
    parser.add_argument("--pairs-per-block", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = json.loads(args.blocks_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or any(not isinstance(path, str) or not path for path in raw.values()):
        raise ValueError("Expected a JSON object mapping block keys to Arrow directories")
    blocks = {block: args.blocks_json.parent / path for block, path in raw.items()}
    count = mine_pairs(args.block_role_assignments, blocks, args.output, per_block=args.pairs_per_block, seed=args.seed)
    print(json.dumps({"blocks": len(blocks), "pairs": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

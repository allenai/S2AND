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

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.text import normalize_orcid
from scripts.production.data.block_roles import load_block_roles, normalize_block_key


def _real_signatures(root: Path, block: str, orcid_home_roles: Mapping[str, str]) -> tuple[list[str], int]:
    with (
        ArrowDataset.open(root) as dataset,
        dataset.use() as lease,
        lease.open_file("signatures") as infile,
        pa.PythonFile(infile, mode="r") as source,
    ):
        reader = ipc.open_file(source)
        columns = ["signature_id", "author_block"]
        if "author_orcid" in reader.schema.names:
            columns.append("author_orcid")
        signatures: list[str] = []
        seen: set[str] = set()
        excluded = 0
        for batch_index in range(reader.num_record_batches):
            for row in reader.get_batch(batch_index).select(columns).to_pylist():
                signature = row["signature_id"]
                if not signature or normalize_block_key(row["author_block"]) != block or signature in seen:
                    raise ValueError(f"Expected unique signatures from block {block!r} in {root}")
                seen.add(signature)
                if signature.startswith("visibleq:"):
                    continue
                orcid = normalize_orcid(row.get("author_orcid") or "")
                if orcid and orcid_home_roles.get(orcid, "pairwise_train") != "pairwise_train":
                    excluded += 1
                else:
                    signatures.append(signature)
    return sorted(signatures), excluded


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

    homes = {
        orcid: roles.orcid_home_role.get(orcid, roles.block_to_role[block])
        for block, orcids in roles.block_orcids.items()
        for orcid in orcids
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    rng = random.Random(seed)
    counts = {}
    rows: list[tuple[str, str, str, float]] = []
    for block, root in sorted(normalized.items()):
        signatures, excluded = _real_signatures(root, block, homes)
        population = math.comb(len(signatures), 2)
        count = min(population, per_block)
        counts[block] = {
            "eligible_signatures": len(signatures),
            "excluded_non_home_signatures": excluded,
            "eligible_pairs": population,
            "sampled_pairs": count,
        }
        for index in sorted(rng.sample(range(population), count)):
            # Invert the triangular index without enumerating the pair population.
            right = (1 + math.isqrt(1 + 8 * index)) // 2
            left = index - math.comb(right, 2)
            rows.append((block, signatures[left], signatures[right], count / population))

    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["block_key", "signature_id_1", "signature_id_2", "inclusion_probability"])
        writer.writerows(rows)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "block_role_assignments_sha256": roles.sha256,
                "seed": seed,
                "role": "pairwise_train",
                "pairs_per_block": per_block,
                "output_sha256": sha256_file(output_path),
                "blocks": counts,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
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

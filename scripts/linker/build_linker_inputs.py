"""Build held-out request lists and ORCID-hidden 1.4 assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from s2and._atomic_io import exclusive_file_lock
from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_tables import read_cluster_seeds_arrow
from s2and.production_model import load_production_model
from scripts.common.arrow_roots import _replace_json, read_json_object, resolved_manifest_artifact_path
from scripts.common.block_roles import load_block_roles, normalize_block_key
from scripts.common.orcid_references import read_external_reference_identity_evidence
from scripts.eps.sweep_eps_on_linking_gold import _hash_directory, _write_orcid_free_dataset
from scripts.intake.requests_to_datasets import extract_request_tables, request_dataset_name
from scripts.linker.build_linker_task_tables import TASK_ROLES, _read_signatures

BATCHING_THRESHOLD = 15_000


def _request_queries(bundle: Path) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tables = extract_request_tables(bundle, Path(tmp), ("signatures", "cluster_seeds"))
        seeded = read_cluster_seeds_arrow(Path(tables["cluster_seeds"]))
        with pa.memory_map(tables["signatures"], "r") as source:
            reader = ipc.open_file(source)
            return [
                sid
                for index in range(reader.num_record_batches)
                for sid in reader.get_batch(index).column("signature_id").to_pylist()
                if sid not in seeded
            ]


def build_linker_inputs(
    roles_path: Path,
    blocks_path: Path,
    output_dir: Path,
    model_path: Path,
    *,
    queries_per_block: int = 300,
    seed: int = 20261123,
    n_jobs: int = 1,
) -> dict[str, dict[str, int]]:
    """Build or resume inputs for the supplied frozen linker and mining blocks."""
    if min(queries_per_block, n_jobs) < 1:
        raise ValueError("Expected positive query and thread counts")
    roles = load_block_roles(roles_path)
    raw = read_json_object(blocks_path)
    blocks = {normalize_block_key(block): spec for block, spec in raw.items()}
    if not blocks or len(blocks) != len(raw):
        raise ValueError("Supply unique frozen blocks")
    model_digest = _hash_directory(model_path)[2] if model_path.is_dir() else sha256_file(model_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "inputs.json"
    model = None
    counts: dict[str, dict[str, int]] = {}
    for block, spec in sorted(blocks.items()):
        role = roles.role_for(block)
        if role not in (*TASK_ROLES, "pairwise_train"):
            raise ValueError(f"{block!r} cannot supply linker or mining inputs from role {role!r}")
        name = block.replace(" ", "_")
        arrow_dir = (blocks_path.parent / spec["arrow_dir"]).resolve()
        label_path = (blocks_path.parent / spec["labels"]).resolve()
        request = (
            (blocks_path.parent / spec["incremental_request"]).resolve() if "incremental_request" in spec else None
        )
        if request and (role != "gate_thresh" or request_dataset_name(request) != name):
            raise ValueError(f"Invalid incremental request for {block!r}: {request}")

        target = output_dir / name
        assigned = target / ("assignments.parquet" if role == "pairwise_train" else "seeds.parquet")
        query_path = target / "queries.json"
        expected_files = [assigned] if role == "pairwise_train" else [assigned, query_path]
        with (
            exclusive_file_lock(output_dir / f".{name}.lock"),
            ArrowDataset.open(arrow_dir, require_specter=True, require_name_counts_index=True) as arrow,
        ):
            arrow_manifest = arrow_dir / "manifest.json"
            index_path = resolved_manifest_artifact_path(
                arrow_manifest, read_json_object(arrow_manifest)["paths"]["name_counts_index"], key="name_counts_index"
            )
            binding = {
                "generation": arrow.generation_id,
                "index_sha256": sha256_file(index_path / "manifest.json"),
                "model_sha256": model_digest,
                "roles_sha256": roles.sha256,
                "labels_sha256": sha256_file(label_path),
                "request_sha256": sha256_file(request) if request is not None else None,
                "queries_per_block": queries_per_block,
                "seed": seed,
            }
            record: dict[str, Any]
            if target.exists():
                record = read_json_object(target / "manifest.json")
                if record["binding"] != binding or record["files"] != {
                    path.name: sha256_file(path) for path in expected_files
                }:
                    raise ValueError(f"Existing linker inputs have changed: {target}")
            else:
                signatures = _read_signatures(arrow, block)
                identities, scored = read_external_reference_identity_evidence(label_path, require_scored=True)
                if missing := identities.keys() - signatures.keys():
                    raise ValueError(f"{block!r} labels contain unknown signatures: {sorted(missing)[:5]}")
                queries = []
                if role in TASK_ROLES:
                    if request:
                        queries = _request_queries(request)
                    else:
                        queries = sorted(
                            (sid for sid, orcid in scored.items() if roles.scores(block, orcid)),
                            key=lambda sid: (hashlib.sha256(f"{seed}:{block}:{sid}".encode()).digest(), sid),
                        )[:queries_per_block]
                    if not queries or len(set(queries)) != len(queries) or set(queries) - signatures.keys():
                        raise ValueError(f"{block!r} needs unique, real query signatures from its full dataset")
                remaining = sorted(signatures.keys() - set(queries))
                if not remaining:
                    raise ValueError(f"No seed signatures remain in {block!r}")
                if model is None:
                    model = load_production_model(model_path, n_jobs=n_jobs)
                free = _write_orcid_free_dataset(arrow_dir, output_dir / "orcid_free_arrow" / name)
                with ArrowDataset.open(free, require_specter=True, require_name_counts_index=True) as hidden:
                    clusters, _ = model.predict_from_arrow(
                        {block: remaining}, hidden, batching_threshold=BATCHING_THRESHOLD
                    )
                component_key = "cluster_id" if role == "pairwise_train" else "component_key"
                rows = [(sid, str(cid)) for cid, members in clusters.items() for sid in members]
                if sorted(sid for sid, _ in rows) != remaining:
                    raise ValueError(f"Prediction coverage differs from the seed set for {block!r}")
                staging = Path(tempfile.mkdtemp(dir=output_dir, prefix=f".{name}."))
                pq.write_table(
                    pa.table({"signature_id": [sid for sid, _ in rows], component_key: [cid for _, cid in rows]}),
                    staging / assigned.name,
                )
                if role in TASK_ROLES:
                    _replace_json(staging / query_path.name, queries)
                record = {
                    "binding": binding,
                    "files": {path.name: sha256_file(staging / path.name) for path in expected_files},
                    "counts": {"queries": len(queries), "assigned_signatures": len(rows), "components": len(clusters)},
                }
                _replace_json(staging / "manifest.json", record)
                staging.rename(target)
        counts[block] = record["counts"]
        if role in TASK_ROLES:
            with exclusive_file_lock(output_dir / ".inputs.lock"):
                entries = read_json_object(inputs_path) if inputs_path.is_file() else {}
                entries[block] = {
                    "arrow_dir": str(arrow_dir),
                    "labels": str(label_path),
                    "seeds": str(assigned),
                    "queries": str(query_path),
                }
                _replace_json(inputs_path, entries)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Build task inputs or full 1.4 assignments for pairwise training blocks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-role-assignments", required=True, type=Path)
    parser.add_argument(
        "--blocks-json",
        required=True,
        type=Path,
        help="Map blocks to arrow_dir, labels and optional incremental_request.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--queries-per-block", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    counts = build_linker_inputs(
        args.block_role_assignments,
        args.blocks_json,
        args.output_dir,
        args.model_path,
        queries_per_block=args.queries_per_block,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

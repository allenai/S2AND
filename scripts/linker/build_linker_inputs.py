"""Build held-out request lists and ORCID-hidden 1.4 assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from s2and._atomic_io import exclusive_file_lock
from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_tables import read_cluster_seeds_arrow
from s2and.incremental_linking_training.data_loading import load_clusterer
from s2and.production_bundle_contract import PAIRWISE_ONLY_BUNDLE_KIND, infer_bundle_kind
from s2and.production_model import load_production_model
from s2and.subblocking import _make_subblocks_with_telemetry_arrow_rust
from scripts.common.arrow_roots import _replace_json, read_json_object, resolved_manifest_artifact_path
from scripts.common.block_roles import BLOCK_BINDINGS_FIELD, ROLE_SHA256_FIELD, load_block_roles, normalize_block_key
from scripts.common.identity_ledger import IdentityLedger, MentionEvidence, load_identity_ledger
from scripts.common.linker_seeds import read_seed_components
from scripts.common.stage0_invariants import read_fit_pool_queries
from scripts.eps.sweep_eps_on_linking_gold import _hash_directory, _write_orcid_free_dataset
from scripts.intake.prepare_identity_labels import canonical_gold_block_filter
from scripts.intake.requests_to_datasets import REQUEST_SUFFIX, extract_request_tables, request_dataset_name
from scripts.linker.build_linker_task_tables import TASK_ROLES, _read_signatures

BATCHING_THRESHOLD = 15_000


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _arrival_requests(
    arrivals: Path,
    block: str,
    mentions: Mapping[str, MentionEvidence],
    eligible: Collection[str],
    cutoff: datetime,
    count: int,
) -> tuple[list[str], dict[str, Any]]:
    required = ["cluster_block_key", "corpus_paper_id", "position", "inserted", "updated", "clusterer"]
    source = ds.dataset(arrivals, format="parquet")
    if missing := set(required) - set(source.schema.names):
        raise ValueError(f"Arrival table is missing columns: {sorted(missing)}")
    if not pa.types.is_timestamp(source.schema.field("inserted").type):
        raise ValueError("The arrival table inserted column must be a timestamp")
    paper_type = source.schema.field("corpus_paper_id").type
    position_type = source.schema.field("position").type
    if not (
        pa.types.is_integer(paper_type) or pa.types.is_string(paper_type) or pa.types.is_large_string(paper_type)
    ) or not pa.types.is_integer(position_type):
        raise ValueError(f"Invalid arrival key types in {block!r}: {paper_type}, {position_type}")
    by_position = {}
    for batch in source.to_batches(columns=required, filter=canonical_gold_block_filter("cluster_block_key", {block})):
        for row in batch.to_pylist():
            if normalize_block_key(row["cluster_block_key"]) != block:
                continue
            paper_id = str(row["corpus_paper_id"])
            key = paper_id, row["position"]
            if re.fullmatch(r"0|[1-9][0-9]*", paper_id) is None or key[1] is None or key[1] < 0 or key in by_position:
                raise ValueError(f"Invalid or duplicate arrival key in {block!r}: {key!r}")
            by_position[key] = (row["inserted"], row["clusterer"])
    candidates = []
    missing_timestamp = before_cutoff = ineligible = 0
    for sid, mention in mentions.items():
        key = mention["corpus_paper_id"], mention["position"]
        inserted, clusterer = by_position.get(key, (None, None))
        if inserted is None:
            missing_timestamp += 1
        elif _utc(inserted) < cutoff:
            before_cutoff += 1
        elif sid not in eligible:
            ineligible += 1
        else:
            candidates.append((_utc(inserted), sid, clusterer))
    selected = sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True)[:count]
    if not selected:
        raise ValueError(f"No eligible arrival requests for {block!r} since {cutoff.isoformat()}")
    stats = {
        "eligible_arrivals": len(candidates),
        "missing_inserted": missing_timestamp,
        "inserted_before_since": before_cutoff,
        "ineligible_after_since": ineligible,
        "inserted_min": selected[-1][0].isoformat(),
        "inserted_max": selected[0][0].isoformat(),
        "clusterer_counts": dict(Counter(row[2] or "unknown" for row in selected)),
    }
    return [sid for _, sid, _ in selected], stats


def _production_seeds(
    bundle: Path, block: str, mentions: Mapping[str, MentionEvidence], queries: Collection[str]
) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        tables = extract_request_tables(bundle, Path(tmp), ("signatures", "cluster_seeds"))
        components = read_cluster_seeds_arrow(Path(tables["cluster_seeds"]))
        incremental = {}
        with pa.memory_map(tables["signatures"], "r") as source:
            reader = ipc.open_file(source)
            for index in range(reader.num_record_batches):
                rows = reader.get_batch(index).select(["signature_id", "paper_id", "author_position", "author_block"])
                for row in rows.to_pylist():
                    sid = row["signature_id"]
                    if normalize_block_key(row["author_block"]) != block or sid in incremental:
                        raise ValueError(f"Invalid incremental bundle signature for {block!r}: {sid!r}")
                    incremental[sid] = (row["paper_id"], row["author_position"])
    if components.keys() - incremental.keys() or any(
        incremental.get(sid) != (mention["corpus_paper_id"], mention["position"]) for sid, mention in mentions.items()
    ):
        raise ValueError(f"Incremental bundle does not cover the snapshot signatures for {block!r}")
    excluded = set(queries)
    return {sid: component for sid, component in components.items() if sid in mentions and sid not in excluded}


def build_linker_inputs(
    ledger: IdentityLedger,
    blocks: Collection[str],
    output_dir: Path,
    model_path: Path,
    *,
    incremental_requests: Mapping[str, Path] | None = None,
    arrivals: Path | None = None,
    requests_per_block: int | None = None,
    since: datetime | None = None,
    batching_threshold: int = BATCHING_THRESHOLD,
    queries_per_block: int = 300,
    seed: int = 20261123,
    n_jobs: int = 1,
) -> dict[str, dict[str, Any]]:
    """Build or resume inputs for the supplied frozen linker and mining blocks."""
    if min(queries_per_block, n_jobs, batching_threshold) < 1:
        raise ValueError("Expected positive query and thread counts")
    roles = ledger.roles
    requested = {normalize_block_key(block) for block in blocks}
    if not requested or len(blocks) != len(requested):
        raise ValueError("Supply unique frozen blocks")
    for block in sorted(requested):
        if roles.role_for(block) not in TASK_ROLES:
            raise ValueError(f"{block!r} cannot supply linker or mining inputs")
    directories = ledger.evidence.dataset_dirs(requested)
    incremental_requests = incremental_requests or {}
    model_digest = _hash_directory(model_path)[2] if model_path.is_dir() else sha256_file(model_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "inputs.json"
    model = None
    pairwise_only = False
    counts: dict[str, dict[str, Any]] = {}
    if arrivals is not None and (requests_per_block is None or requests_per_block < 1 or since is None):
        raise ValueError("Arrival requests require a positive --requests-per-block and --since")
    arrivals_sha256 = sha256_file(arrivals) if arrivals is not None else None
    for block, arrow_dir in sorted(directories.items()):
        role = roles.role_for(block)
        name = block.replace(" ", "_")
        request = incremental_requests.get(block)
        use_arrivals = role == "gate_thresh"
        if request and (not use_arrivals or request_dataset_name(request) != name):
            raise ValueError(f"Invalid incremental request for {block!r}: {request}")
        if use_arrivals and (arrivals is None or request is None or requests_per_block is None or since is None):
            raise ValueError("gate_thresh requires --arrivals, --requests-per-block, --since and an incremental bundle")

        target = output_dir / name
        seed_path = target / "seeds.parquet"
        query_path = target / "queries.json"
        expected_files = [seed_path, query_path]
        with (
            exclusive_file_lock(output_dir / f".{name}.lock"),
            ArrowDataset.open(arrow_dir, require_specter=True, require_name_counts_index=True) as arrow,
        ):
            arrow_manifest = arrow_dir / "manifest.json"
            index_path = resolved_manifest_artifact_path(
                arrow_manifest, read_json_object(arrow_manifest)["paths"]["name_counts_index"], key="name_counts_index"
            )
            common_binding = {
                "index_sha256": sha256_file(index_path / "manifest.json"),
                "model_sha256": model_digest,
                ROLE_SHA256_FIELD: roles.sha256,
            }
            if common_binding["index_sha256"] != ledger.evidence.name_counts_index_sha256:
                raise ValueError(f"Linker input name-counts index does not match intake for {block!r}")
            with exclusive_file_lock(output_dir / ".inputs.lock"):
                path = output_dir / "inputs_binding.json"
                if not path.is_file():
                    _replace_json(path, common_binding)
                elif read_json_object(path) != common_binding:
                    raise ValueError(f"Existing linker inputs differ in model, index, or roles: {output_dir}")
            binding = {
                "generation": arrow.generation_id,
                **common_binding,
                BLOCK_BINDINGS_FIELD: roles.bindings_for({block}),
                "request_sha256": sha256_file(request) if request is not None else None,
                "queries_per_block": queries_per_block,
                "seed": seed,
                "batching_threshold": batching_threshold,
                "arrivals_sha256": arrivals_sha256 if use_arrivals else None,
                "requests_per_block": requests_per_block if use_arrivals else None,
                "since": _utc(since).isoformat() if use_arrivals and since is not None else None,
            }
            record: dict[str, Any]
            if target.exists():
                record = read_json_object(target / "manifest.json")
                if record["binding"] != binding or record["files"] != {
                    path.name: sha256_file(path) for path in expected_files
                }:
                    raise ValueError(f"Existing linker inputs have changed: {target}")
                if role == "fit_pool":
                    frozen_queries = read_fit_pool_queries(query_path, block, record)
                    if overlap := frozen_queries & read_seed_components(seed_path).keys():
                        raise ValueError(
                            f"Query signatures in assignments for {block!r}: {sorted(overlap)[:5]}. "
                            "Rebuild linker inputs."
                        )
            else:
                signatures = _read_signatures(arrow, block)
                view = ledger.for_block(block, dataset_dir=arrow_dir)
                view.require_signatures(signatures.keys())
                decisions = {sid: view.eligibility(sid) for sid in signatures}
                eligible = {sid for sid, decision in decisions.items() if decision.fit_ok}
                scored = {sid for sid, decision in decisions.items() if decision.score_orcid is not None}
                request_stats: dict[str, Any] = {}
                if use_arrivals:
                    assert arrivals is not None and since is not None
                    assert requests_per_block is not None
                    queries, request_stats = _arrival_requests(
                        arrivals, block, view.rows, eligible, _utc(since), requests_per_block
                    )
                else:
                    # Keep enough scored mentions to label seeds and mine pairs.
                    limit = min(queries_per_block, len(scored) // 2)
                    queries = sorted(
                        scored, key=lambda sid: (hashlib.sha256(f"{seed}:{block}:{sid}".encode()).digest(), sid)
                    )[:limit]
                query_set = set(queries)
                if (
                    (not queries and role != "fit_pool")
                    or len(query_set) != len(queries)
                    or query_set - signatures.keys()
                ):
                    raise ValueError(f"{block!r} needs unique, real query signatures from its full dataset")
                remaining = sorted((signatures.keys() if role == "gate_thresh" else eligible) - query_set)
                if role == "fit_pool" and not eligible:
                    rows = []
                    components = 0
                    request_stats["empty_reason"] = "no_eligible_signatures"
                elif not remaining:
                    raise ValueError(f"No seed signatures remain in {block!r}")
                elif use_arrivals:
                    assert request is not None
                    production = _production_seeds(request, block, view.rows, queries)
                    if not production:
                        raise ValueError(f"No production seeds remain for {block!r}")
                    rows = sorted(production.items())
                    components = len(set(production.values()))
                    request_stats["unseeded_remaining"] = len(remaining) - len(production)
                else:
                    if model is None:
                        pairwise_only = (
                            model_path.is_dir()
                            and infer_bundle_kind(read_json_object(model_path / "manifest.json")["sha256"])
                            == PAIRWISE_ONLY_BUNDLE_KIND
                        )
                        if pairwise_only:
                            model = load_clusterer(model_path, n_jobs=n_jobs)
                        else:
                            model = load_production_model(model_path, n_jobs=n_jobs)
                    free = _write_orcid_free_dataset(arrow_dir, output_dir / "orcid_free_arrow" / name)
                    with ArrowDataset.open(free, require_specter=True, require_name_counts_index=True) as hidden:
                        prediction_blocks = {block: remaining}
                        if pairwise_only and len(remaining) > batching_threshold:
                            # The pairwise stage has no linker for attachment between subblocks.
                            prediction_blocks, _, _ = _make_subblocks_with_telemetry_arrow_rust(
                                hidden,
                                remaining,
                                maximum_size=batching_threshold,
                                first_k_letter_counts_sorted=model._subblocking_prefix_counts(),
                                graph_subblocking_config=model._subblocking_graph_config(),
                                graph_subblocking_random_seed=int(model.random_state or 0),
                                use_orcid_subblocking=False,
                            )
                        clusters, _ = model.predict_from_arrow(
                            prediction_blocks, hidden, batching_threshold=batching_threshold
                        )
                    rows = [(sid, str(cid)) for cid, members in clusters.items() for sid in members]
                    if sorted(sid for sid, _ in rows) != remaining:
                        raise ValueError(f"Prediction coverage differs from the requested signatures for {block!r}")
                    components = len({cid for _, cid in rows})
                staging = Path(tempfile.mkdtemp(dir=output_dir, prefix=f".{name}."))
                pq.write_table(
                    pa.table(
                        {
                            "signature_id": pa.array([sid for sid, _ in rows], type=pa.string()),
                            "component_key": pa.array([cid for _, cid in rows], type=pa.string()),
                        }
                    ),
                    staging / seed_path.name,
                )
                _replace_json(staging / query_path.name, queries)
                record = {
                    "binding": binding,
                    "files": {path.name: sha256_file(staging / path.name) for path in expected_files},
                    "counts": {
                        "queries": len(queries),
                        "assigned_signatures": len(rows),
                        "components": components,
                        **request_stats,
                    },
                }
                _replace_json(staging / "manifest.json", record)
                staging.rename(target)
        counts[block] = record["counts"]
        with exclusive_file_lock(output_dir / ".inputs.lock"):
            entries = read_json_object(inputs_path) if inputs_path.is_file() else {}
            entries[block] = {
                "input_manifest": str(target / "manifest.json"),
                "seeds": str(seed_path),
                "queries": str(query_path),
            }
            _replace_json(inputs_path, entries)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Build frozen seed partitions and query lists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-evidence", type=Path, required=True)
    parser.add_argument("--block-role-assignments", required=True, type=Path)
    parser.add_argument("--fit-pool-only", action="store_true", help="Build the fitting blocks for mining.")
    parser.add_argument(
        "--incremental-requests", nargs="+", type=Path, default=[], help="Gate request bundles or directories."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--arrivals", type=Path, help="Athena paper_authors arrival Parquet.")
    parser.add_argument("--requests-per-block", type=int)
    parser.add_argument("--since", type=datetime.fromisoformat, help="Inclusive UTC cutoff for arrival requests.")
    parser.add_argument("--batching-threshold", type=int, default=BATCHING_THRESHOLD)
    parser.add_argument("--queries-per-block", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    roles = load_block_roles(args.block_role_assignments)
    ledger = load_identity_ledger(args.identity_evidence, roles)
    required = {
        block
        for block, role in roles.block_to_role.items()
        if role == "fit_pool" or (role == "gate_thresh" and not args.fit_pool_only)
    }
    requests = {}
    for source in args.incremental_requests:
        for path in sorted(source.glob(f"*{REQUEST_SUFFIX}")) if source.is_dir() else (source,):
            block = normalize_block_key(request_dataset_name(path))
            if roles.block_to_role.get(block) != "gate_thresh" or block not in required or block in requests:
                raise ValueError(f"Unexpected or duplicate gate request bundle: {path}")
            requests[block] = path
    counts = build_linker_inputs(
        ledger,
        required,
        args.output_dir,
        args.model_path,
        incremental_requests=requests,
        arrivals=args.arrivals,
        requests_per_block=args.requests_per_block,
        since=args.since,
        batching_threshold=args.batching_threshold,
        queries_per_block=args.queries_per_block,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from s2and.production_model import load_production_model
from scripts.common.arrow_roots import _replace_json, read_json_object, resolved_manifest_artifact_path
from scripts.common.block_roles import BLOCK_BINDINGS_FIELD, load_block_roles, normalize_block_key
from scripts.common.identity_ledger import MentionEvidence, ledger_for_roles
from scripts.common.orcid_references import read_external_reference_identity_evidence
from scripts.eps.sweep_eps_on_linking_gold import _hash_directory, _write_orcid_free_dataset
from scripts.intake.prepare_identity_labels import canonical_gold_block_filter
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
    roles_path: Path,
    blocks_path: Path,
    output_dir: Path,
    model_path: Path,
    *,
    identity_evidence: Path | None = None,
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
    roles = load_block_roles(roles_path)
    ledger = ledger_for_roles(identity_evidence, roles)
    raw = read_json_object(blocks_path)
    blocks = {normalize_block_key(block): spec for block, spec in raw.items()}
    if not blocks or len(blocks) != len(raw):
        raise ValueError("Supply unique frozen blocks")
    model_digest = _hash_directory(model_path)[2] if model_path.is_dir() else sha256_file(model_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = output_dir / "inputs.json"
    model = None
    counts: dict[str, dict[str, Any]] = {}
    if arrivals is not None and (requests_per_block is None or requests_per_block < 1 or since is None):
        raise ValueError("Arrival requests require a positive --requests-per-block and --since")
    arrivals_sha256 = sha256_file(arrivals) if arrivals is not None else None
    for block, spec in sorted(blocks.items()):
        role = roles.role_for(block)
        if role not in (*TASK_ROLES, "pairwise_train"):
            raise ValueError(f"{block!r} cannot supply linker or mining inputs from role {role!r}")
        name = block.replace(" ", "_")
        arrow_dir = (blocks_path.parent / spec["arrow_dir"]).resolve()
        label_path = (blocks_path.parent / spec["labels"]).resolve() if ledger is None else None
        request = (
            (blocks_path.parent / spec["incremental_request"]).resolve() if "incremental_request" in spec else None
        )
        if request and (role != "gate_thresh" or request_dataset_name(request) != name):
            raise ValueError(f"Invalid incremental request for {block!r}: {request}")

        use_arrivals = ledger is not None and role == "gate_thresh"
        if use_arrivals and (arrivals is None or request is None or requests_per_block is None or since is None):
            raise ValueError("gate_thresh requires --arrivals, --requests-per-block, --since and an incremental bundle")

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
            common_binding = {
                "index_sha256": sha256_file(index_path / "manifest.json"),
                "model_sha256": model_digest,
                **({"roles_sha256": roles.sha256} if ledger is None else {}),
            }
            if ledger is not None and common_binding["index_sha256"] != ledger.evidence.name_counts_index_sha256:
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
                **({BLOCK_BINDINGS_FIELD: roles.bindings_for({block})} if ledger is not None else {}),
                "labels_sha256": sha256_file(label_path) if label_path is not None else None,
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
            else:
                signatures = _read_signatures(arrow, block, include_orcid=ledger is None)
                if ledger is None:
                    assert label_path is not None
                    identities, scored = read_external_reference_identity_evidence(label_path, require_scored=True)
                    if missing := identities.keys() - signatures.keys():
                        raise ValueError(f"{block!r} labels contain unknown signatures: {sorted(missing)[:5]}")
                    eligible = signatures.keys()
                    scored = {sid: orcid for sid, orcid in scored.items() if roles.scores(block, orcid)}
                else:
                    view = ledger.for_block(block, dataset_dir=arrow_dir)
                    view.require_signatures(signatures.keys())
                    decisions = {sid: view.eligibility(sid) for sid in signatures}
                    eligible = {sid for sid, decision in decisions.items() if decision.fit_ok}
                    scored = {
                        sid: decision.score_orcid
                        for sid, decision in decisions.items()
                        if decision.score_orcid is not None
                    }
                queries = []
                request_stats: dict[str, Any] = {}
                if role in TASK_ROLES:
                    if use_arrivals:
                        assert ledger is not None and arrivals is not None and since is not None
                        assert requests_per_block is not None
                        queries, request_stats = _arrival_requests(
                            arrivals, block, view.rows, eligible, _utc(since), requests_per_block
                        )
                    elif request:
                        queries = _request_queries(request)
                    else:
                        queries = sorted(
                            scored, key=lambda sid: (hashlib.sha256(f"{seed}:{block}:{sid}".encode()).digest(), sid)
                        )[:queries_per_block]
                    if (
                        (not queries and not use_arrivals)
                        or len(set(queries)) != len(queries)
                        or set(queries) - signatures.keys()
                    ):
                        raise ValueError(f"{block!r} needs unique, real query signatures from its full dataset")
                remaining = sorted((signatures.keys() if role == "gate_thresh" else eligible) - set(queries))
                if not remaining:
                    raise ValueError(f"No seed signatures remain in {block!r}")
                component_key = "cluster_id" if role == "pairwise_train" else "component_key"
                if use_arrivals:
                    assert request is not None and ledger is not None
                    production = _production_seeds(request, block, view.rows, queries)
                    if not production:
                        raise ValueError(f"No production seeds remain for {block!r}")
                    rows = sorted(production.items())
                    components = len(set(production.values()))
                    request_stats["unseeded_remaining"] = len(remaining) - len(production)
                else:
                    if model is None:
                        model = load_production_model(model_path, n_jobs=n_jobs)
                    free = _write_orcid_free_dataset(arrow_dir, output_dir / "orcid_free_arrow" / name)
                    with ArrowDataset.open(free, require_specter=True, require_name_counts_index=True) as hidden:
                        clusters, _ = model.predict_from_arrow(
                            {block: remaining}, hidden, batching_threshold=batching_threshold
                        )
                    rows = [(sid, str(cid)) for cid, members in clusters.items() for sid in members]
                    if sorted(sid for sid, _ in rows) != remaining:
                        raise ValueError(f"Prediction coverage differs from the seed set for {block!r}")
                    components = len(clusters)
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
        if role in TASK_ROLES:
            with exclusive_file_lock(output_dir / ".inputs.lock"):
                entries = read_json_object(inputs_path) if inputs_path.is_file() else {}
                entries[block] = {
                    "arrow_dir": str(arrow_dir),
                    **({"labels": str(label_path)} if label_path is not None else {}),
                    "input_manifest": str(target / "manifest.json"),
                    "seeds": str(assigned),
                    "queries": str(query_path),
                }
                _replace_json(inputs_path, entries)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Build task inputs or full 1.4 assignments for pairwise training blocks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-evidence", type=Path)
    parser.add_argument("--block-role-assignments", required=True, type=Path)
    parser.add_argument(
        "--blocks-json",
        required=True,
        type=Path,
        help="Map blocks to arrow_dir, labels and optional incremental_request.",
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
    counts = build_linker_inputs(
        args.block_role_assignments,
        args.blocks_json,
        args.output_dir,
        args.model_path,
        identity_evidence=args.identity_evidence,
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

"""Freeze native top-25 linker windows for the three development roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and import __version__, feature_port  # noqa: E402
from s2and._sha256 import sha256_file  # noqa: E402
from s2and.arrow_inputs import ArrowDataset  # noqa: E402
from s2and.incremental_linking.contracts import DEFAULT_RETRIEVAL_TOP_K  # noqa: E402
from s2and.incremental_linking.feature_block_arrow import temporary_cluster_seed_sidecars  # noqa: E402
from s2and.text import normalize_orcid  # noqa: E402
from scripts.common.block_roles import BlockRoles, load_block_roles, normalize_block_key  # noqa: E402
from scripts.common.distance_policy import DISTANCE_POLICY_VERSION  # noqa: E402
from scripts.common.orcid_references import (  # noqa: E402
    EXCLUDED_ORCID_GROUPS,
    read_external_reference_identity_evidence,
)

TASK_ROLES = ("linker_train", "gate_fit", "gate_thresh")
TASK_MANIFEST = "linker_tasks.json"
_PLAN_BATCH_SIZE = 128
_TASK_COLUMNS = (
    "dataset",
    "query_group_id",
    "query_signature_id",
    "query_view",
    "query_author",
    "query_author_middle",
    "candidate_component_key",
    "retrieval_rank",
    "retrieval_score",
    "label",
    "candidate_gold_scored",
    "gold_safe_target",
    "query_gold_scored",
    "whole_author_holdout",
)


def read_seed_components(path: Path) -> dict[str, str]:
    """Read one pre-attachment component assignment per seed signature."""
    rows = pq.read_table(path, columns=["signature_id", "component_key"]).to_pylist()
    components = {row["signature_id"]: row["component_key"] for row in rows}
    if len(components) != len(rows) or any(
        not isinstance(signature, str) or not signature or not isinstance(component, str) or not component
        for signature, component in components.items()
    ):
        raise ValueError(f"Invalid or duplicate seed assignments in {path}")
    return components


def _read_signatures(arrow: ArrowDataset, block: str) -> dict[str, tuple[str, str | None]]:
    signatures: dict[str, tuple[str, str | None]] = {}
    with arrow.use() as lease, lease.open_file("signatures") as infile, pa.PythonFile(infile, mode="r") as source:
        reader = ipc.open_file(source)
        columns = ["signature_id", "author_block", "author_middle", "author_orcid"]
        columns = [name for name in columns if name in reader.schema.names]
        for index in range(reader.num_record_batches):
            for row in reader.get_batch(index).select(columns).to_pylist():
                sid = row["signature_id"]
                if (
                    sid.startswith("visibleq:")
                    or normalize_block_key(row["author_block"]) != block
                    or sid in signatures
                ):
                    raise ValueError(f"Expected unique real signatures from {block!r}, found {sid!r}")
                orcid = normalize_orcid(row.get("author_orcid") or "")
                signatures[sid] = (row.get("author_middle") or "", orcid)
    return signatures


def _freeze_block_rows(
    arrow: ArrowDataset,
    *,
    block: str,
    block_roles: BlockRoles,
    seeds: Mapping[str, str],
    query_ids: Sequence[str],
    labels: Mapping[str, str],
    identities: Mapping[str, str],
    holdout_fraction: float,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    role = block_roles.role_for(block)
    signatures = _read_signatures(arrow, block)
    if missing := (set(seeds) | set(query_ids) | set(identities)) - signatures.keys():
        raise ValueError(f"{block!r} inputs reference missing signatures: {sorted(missing)[:5]}")
    if len(set(query_ids)) != len(query_ids) or set(query_ids) & seeds.keys():
        raise ValueError(f"Queries in {block!r} must be unique and absent from the pre-attachment seeds")
    scored = {
        sid: orcid
        for sid, orcid in labels.items()
        if orcid not in EXCLUDED_ORCID_GROUPS and block_roles.scores(block, orcid)
    }
    queries = [query for query in query_ids if query in scored or role == "gate_thresh"]
    gold_by_component: dict[str, set[str]] = defaultdict(set)
    components_by_orcid: dict[str, set[str]] = defaultdict(set)
    identities_by_component: dict[str, set[str]] = defaultdict(set)
    representatives: dict[str, str] = {}
    for sid, component in seeds.items():
        representatives.setdefault(component, sid)
        if sid in scored:
            gold_by_component[component].add(scored[sid])
        # Keep masked and deposited identities for exclusion and seed purity checks.
        for orcid in {identities.get(sid), signatures[sid][1]}:
            if orcid is not None:
                components_by_orcid[orcid].add(component)
                identities_by_component[component].add(orcid)
    safe_components: dict[str, set[str]] = defaultdict(set)
    safe_targets: set[str] = set()
    for component, gold in gold_by_component.items():
        if len(gold) == 1 and gold == identities_by_component[component]:
            safe_targets.add(component)
            safe_components[next(iter(gold))].add(component)
    eligible_authors = {scored[q] for q in queries if q in scored and scored[q] in safe_components}
    ordered_authors = sorted(
        eligible_authors, key=lambda orcid: hashlib.sha256(f"{seed}:{block}:{orcid}".encode()).hexdigest()
    )
    heldout_authors = set(ordered_authors[: round(holdout_fraction * len(ordered_authors))])
    counts: Counter[str] = Counter(short_windows=0)
    rows = []
    if queries:
        with temporary_cluster_seed_sidecars(seeds, prefix="linker_tasks_") as paths:
            planner = feature_port._require_rust_runtime().RawBlockQueryCandidatePlanner.from_auto_queries(
                arrow.native,
                paths["cluster_seeds"],
                top_k=DEFAULT_RETRIEVAL_TOP_K,
                orcid_enabled=False,
                num_threads=n_jobs,
                prefetch_signature_ids=queries,
            )
            try:
                for start in range(0, len(queries), _PLAN_BATCH_SIZE):
                    batch = queries[start : start + _PLAN_BATCH_SIZE]
                    removed = [
                        components_by_orcid[scored[query]] if scored.get(query) in heldout_authors else set()
                        for query in batch
                    ]
                    disallows = [
                        (query, representatives[component])
                        for query, excluded in zip(batch, removed, strict=True)
                        for component in sorted(excluded)
                    ]
                    plan = planner.plan(
                        batch, additional_cluster_seed_disallows=disallows, include_component_members=False
                    )
                    if plan["query_signature_ids"] != batch:
                        raise ValueError(f"Native planner changed query order in {block!r}")
                    windows: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
                    for index, rank, component, score in zip(
                        plan["row_query_signature_indices"],
                        plan["retrieval_ranks"],
                        plan["row_component_keys"],
                        plan["retrieval_scores"],
                        strict=True,
                    ):
                        windows[index].append((int(rank), component, float(score)))
                    for index, query in enumerate(batch):
                        expected = min(DEFAULT_RETRIEVAL_TOP_K, len(representatives) - len(removed[index]))
                        window = sorted(windows[index])
                        size = len(window)
                        candidate_keys = {component for _, component, _ in window}
                        if (
                            [rank for rank, _, _ in window] != list(range(1, size + 1))
                            or size > expected
                            or len(candidate_keys) != size
                            or candidate_keys - representatives.keys()
                            or candidate_keys & removed[index]
                        ):
                            raise ValueError(f"Incomplete or invalid top-25 window for {block!r} query {query!r}")
                        view = plan["query_views"][index]
                        query_orcid = scored.get(query)
                        natural_safe = safe_components.get(query_orcid or "", set())
                        is_heldout = query_orcid in heldout_authors
                        counts[view] += 1
                        counts["empty_windows"] += size == 0
                        counts["short_windows"] += 0 < size < DEFAULT_RETRIEVAL_TOP_K
                        counts["full_windows"] += size == DEFAULT_RETRIEVAL_TOP_K
                        counts["whole_author_holdout"] += is_heldout
                        counts["unscored_queries"] += query_orcid is None
                        for rank, component, score in window:
                            rows.append(
                                (
                                    block,
                                    json.dumps([block, query], ensure_ascii=False),
                                    query,
                                    view,
                                    plan["query_authors"][index],
                                    signatures[query][0],
                                    component,
                                    rank,
                                    score,
                                    int(component in natural_safe and not is_heldout),
                                    query_orcid is not None and component in safe_targets,
                                    bool(natural_safe) and not is_heldout,
                                    query_orcid is not None,
                                    is_heldout,
                                )
                            )
            finally:
                planner.release()
    frame = pd.DataFrame(rows, columns=pd.Index(_TASK_COLUMNS))
    return frame, {"rows": len(frame), "requested_queries": len(query_ids), "planned_queries": len(queries), **counts}


def build_linker_task_tables(
    roles_path: Path,
    inputs_path: Path,
    output_dir: Path,
    *,
    holdout_fraction: float = 1 / 3,
    seed: int = 20261123,
    n_jobs: int = 1,
    blocks: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build and audit split candidate tables from frozen roles and request inputs."""
    if not math.isfinite(holdout_fraction) or not 0 <= holdout_fraction <= 1 or n_jobs < 1:
        raise ValueError("Expected a holdout fraction in [0, 1] and at least one planner thread")
    roles = load_block_roles(roles_path)
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    by_block = {normalize_block_key(key): spec for key, spec in inputs.items()}
    if len(inputs) != len(by_block):
        raise ValueError("Duplicate normalized block keys in linker inputs")
    required = {block for block, role in roles.block_to_role.items() if role in TASK_ROLES}
    if blocks is not None:
        requested = {normalize_block_key(block) for block in blocks}
        if not requested or len(requested) != len(blocks) or requested - required:
            raise ValueError("Select unique frozen linker task blocks")
        by_block = {block: spec for block, spec in by_block.items() if block in requested}
        required = requested
    if set(by_block) != required:
        raise ValueError(
            f"Linker inputs must match frozen task roles: missing={sorted(required - by_block.keys())}, "
            f"unexpected={sorted(by_block.keys() - required)}"
        )
    tables: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=False)
    for role in TASK_ROLES:
        role_blocks = sorted(key for key in by_block if roles.role_for(key) == role)
        if blocks is not None and not role_blocks:
            continue
        block_records = {}
        role_path = output_dir / f"{role}.parquet"
        writer = None
        try:
            for block in role_blocks:
                spec = by_block[block]
                arrow_dir, seed_path, label_path, query_path = (
                    (inputs_path.parent / str(spec[field])).resolve()
                    for field in ("arrow_dir", "seeds", "labels", "queries")
                )
                query_ids = json.loads(query_path.read_text(encoding="utf-8"))
                if not isinstance(query_ids, list) or any(
                    not isinstance(query, str) or not query for query in query_ids
                ):
                    raise ValueError(f"Expected real request signature IDs in {query_path}")
                identities, labels = read_external_reference_identity_evidence(label_path, require_scored=True)
                with ArrowDataset.open(arrow_dir, require_specter=True, require_name_counts_index=True) as arrow:
                    frame, counts = _freeze_block_rows(
                        arrow,
                        block=block,
                        block_roles=roles,
                        seeds=read_seed_components(seed_path),
                        query_ids=query_ids,
                        labels=labels,
                        identities=identities,
                        holdout_fraction=0 if role == "gate_thresh" else holdout_fraction,
                        seed=seed,
                        n_jobs=n_jobs,
                    )
                    block_records[block] = {
                        "arrow_dir": str(arrow_dir),
                        "arrow_generation_id": arrow.generation_id,
                        "seeds": str(seed_path),
                        "seeds_sha256": sha256_file(seed_path),
                        "counts": counts,
                    }
                if not frame.empty:
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(role_path, table.schema)
                    writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if not role_path.is_file():
            raise ValueError(f"No candidate rows in frozen role {role!r}")
        tables[role] = {"path": role_path.name, "sha256": sha256_file(role_path), "blocks": block_records}
    manifest = {
        "block_role_assignments_sha256": roles.sha256,
        "distance_policy_version": DISTANCE_POLICY_VERSION,
        "s2and_version": __version__,
        "retrieval_top_k": DEFAULT_RETRIEVAL_TOP_K,
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "tables": tables,
    }
    (output_dir / TASK_MANIFEST).write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Build frozen linker tables from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-role-assignments", required=True, type=Path)
    parser.add_argument("--block-inputs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--blocks", nargs="+", help="Build only these frozen task blocks.")
    parser.add_argument("--holdout-fraction", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    manifest = build_linker_task_tables(
        args.block_role_assignments,
        args.block_inputs,
        args.output_dir,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        n_jobs=args.n_jobs,
        blocks=args.blocks,
    )
    print(
        json.dumps(
            {
                role: sum(b["counts"]["rows"] for b in spec["blocks"].values())
                for role, spec in manifest["tables"].items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

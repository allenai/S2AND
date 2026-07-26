from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# When this file is executed directly, ensure `scripts/` is importable so
# `import _rust_suite.*` works.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _rust_suite.common import (  # type: ignore  # noqa: E402
    PROJECT_ROOT,
    ProcessTreeRSSMonitor,
    build_run_metadata,
    collect_rust_extension_identity,
    get_result_markers,
)

from s2and.arrow_inputs import (  # noqa: E402
    MissingArrowArtifactError,
    ValidatedArrowInputs,
    validate_arrow_prediction_artifacts,
)

DEFAULT_ARROW_ROOT = PROJECT_ROOT / "s2and" / "data" / "s2and_and_big_blocks_linker_dataset_20260525"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scratch" / "promoted_incremental_arrow_profile"
REPORT_SCHEMA = "s2and_performance_evaluation_report_v1"

RESULT_JSON_START, RESULT_JSON_END = get_result_markers("profile")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _require_sha256(path: Path, expected: str, *, label: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected} observed={observed}")
    return observed


def _write_fresh_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Report output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        staging.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArrowSignatureRow:
    signature_id: str
    paper_id: str
    author_block: str
    author_first: str
    author_middle: str
    author_last: str
    author_orcid: str | None


@dataclass(frozen=True)
class ProfileWorkload:
    target_block: str
    block_signature_count: int
    seed_signature_to_cluster: dict[str, str]
    query_signature_ids: list[str]

    @property
    def block_signatures(self) -> list[str]:
        return [*self.seed_signature_to_cluster.keys(), *self.query_signature_ids]


def _workload_sha256(workload: ProfileWorkload, args: argparse.Namespace, seed_source: str) -> str:
    payload = {
        "batching_threshold": int(args.batching_threshold),
        "dataset": str(args.dataset),
        "n_jobs": int(args.n_jobs),
        "query_signature_ids": workload.query_signature_ids,
        "runs": int(args.runs),
        "seed_signature_to_cluster": workload.seed_signature_to_cluster,
        "seed_source": seed_source,
        "target_block": workload.target_block,
        "total_ram_bytes": int(args.total_ram_bytes),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _resolve_arrow_dataset_root(arrow_root: Path, dataset: str) -> Path:
    candidates = [
        arrow_root / dataset,
        arrow_root / "datasets" / dataset,
    ]
    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            return candidate
    checked = ", ".join(str(candidate / "manifest.json") for candidate in candidates)
    raise FileNotFoundError(f"Missing Arrow manifest for dataset={dataset!r}; checked {checked}")


def _resolve_manifest_path(dataset_root: Path, value: Any) -> str:
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [dataset_root / raw, PROJECT_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0])


def _resolve_arrow_dataset_paths(arrow_root: Path, dataset: str) -> ValidatedArrowInputs:
    dataset_root = _resolve_arrow_dataset_root(arrow_root, dataset)
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_paths = manifest.get("paths")
    if not isinstance(manifest_paths, dict):
        raise ValueError(f"Arrow manifest is missing object paths: {manifest_path}")

    paths: dict[str, str] = {}
    for key in (
        "signatures",
        "papers",
        "paper_authors",
        "specter",
        "name_counts_index",
        "signatures_batch_index",
        "papers_batch_index",
        "paper_authors_batch_index",
        "specter_batch_index",
        "cluster_seed_disallows",
        "altered_cluster_signatures",
    ):
        value = manifest_paths.get(key)
        if value is not None:
            paths[key] = _resolve_manifest_path(dataset_root, value)
    clusters_path = dataset_root / f"{dataset}_clusters.json"
    if clusters_path.exists():
        paths["clusters"] = str(clusters_path.resolve())
    try:
        validated = validate_arrow_prediction_artifacts(
            paths,
            require_specter=True,
            require_name_counts_index=True,
            context=f"promoted incremental Arrow profile dataset {dataset}",
            producer_hint=(
                "use the canonical s2and_and_big_blocks_linker_dataset_20260525 bundle "
                "with manifest-declared name_counts_index and raw-planner batch indexes"
            ),
        )
    except MissingArrowArtifactError as exc:
        raise FileNotFoundError(str(exc)) from exc
    return validated


def _read_signature_rows(signatures_path: Path) -> list[ArrowSignatureRow]:
    import pyarrow as pa

    columns = [
        "signature_id",
        "paper_id",
        "author_block",
        "author_first",
        "author_middle",
        "author_last",
    ]
    with pa.memory_map(str(signatures_path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    columns.extend(name for name in ("author_orcid",) if name in table.column_names)
    table = table.select(columns)
    rows: list[ArrowSignatureRow] = []
    for row in table.to_pylist():
        rows.append(
            ArrowSignatureRow(
                signature_id=str(row["signature_id"]),
                paper_id=str(row["paper_id"]),
                author_block=str(row["author_block"] or ""),
                author_first=str(row["author_first"] or ""),
                author_middle=str(row["author_middle"] or ""),
                author_last=str(row["author_last"] or ""),
                author_orcid=None if row.get("author_orcid") is None else str(row["author_orcid"]),
            )
        )
    return rows


def _block_dict(rows: list[ArrowSignatureRow]) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        blocks[row.author_block].append(row.signature_id)
    return dict(blocks)


def _read_signature_to_cluster_id(clusters_path: Path) -> dict[str, str]:
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    signature_to_cluster: dict[str, str] = {}
    for cluster_id, cluster_info in clusters.items():
        for signature_id in cluster_info["signature_ids"]:
            signature_to_cluster[str(signature_id)] = str(cluster_id)
    return signature_to_cluster


def _synthetic_signature_to_cluster_id(
    blocks: dict[str, list[str]],
    *,
    max_seed_clusters: int,
) -> dict[str, str]:
    signature_to_cluster: dict[str, str] = {}
    for block_key, block_signatures in blocks.items():
        seed_limit = len(block_signatures) if max_seed_clusters <= 0 else min(len(block_signatures), max_seed_clusters)
        for seed_index, signature_id in enumerate(block_signatures[:seed_limit]):
            signature_to_cluster[str(signature_id)] = f"synthetic:{block_key}:{seed_index}"
    return signature_to_cluster


def _select_workload(
    *,
    blocks: dict[str, list[str]],
    signature_to_cluster_id: dict[str, str],
    target_block: str,
    query_limit: int,
    max_seed_clusters: int,
) -> ProfileWorkload:
    if target_block:
        if target_block not in blocks:
            raise ValueError(f"Requested target block {target_block!r} is not present")
        selected_block = target_block
    else:
        selected_block = max(blocks, key=lambda block_key: len(blocks[block_key]))

    block_signatures = [str(signature_id) for signature_id in blocks[selected_block]]
    seen_clusters: set[str] = set()
    seed_signature_to_cluster: dict[str, str] = {}
    for signature_id in block_signatures:
        cluster_id = signature_to_cluster_id.get(signature_id)
        if cluster_id is None or cluster_id in seen_clusters:
            continue
        seed_signature_to_cluster[signature_id] = cluster_id
        seen_clusters.add(cluster_id)
        if max_seed_clusters > 0 and len(seed_signature_to_cluster) >= max_seed_clusters:
            break

    if not seed_signature_to_cluster:
        raise ValueError(f"Selected block {selected_block!r} has no cluster-labeled seed signatures")

    seed_ids = set(seed_signature_to_cluster)
    query_signature_ids = [signature_id for signature_id in block_signatures if signature_id not in seed_ids]
    if query_limit > 0:
        query_signature_ids = query_signature_ids[:query_limit]
    if not query_signature_ids:
        raise ValueError(f"Selected block {selected_block!r} has no query signatures after seed/query selection")

    return ProfileWorkload(
        target_block=selected_block,
        block_signature_count=len(block_signatures),
        seed_signature_to_cluster=seed_signature_to_cluster,
        query_signature_ids=query_signature_ids,
    )


def _summarize_runs(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    predict_seconds = [float(run["predict_seconds"]) for run in per_run]
    telemetry_values = [dict(run.get("telemetry", {})) for run in per_run]

    def _numeric_telemetry_summary(key: str) -> dict[str, float] | None:
        values = [float(telemetry[key]) for telemetry in telemetry_values if key in telemetry]
        if not values:
            return None
        return {
            "min": min(values),
            "p50": statistics.median(values),
            "max": max(values),
        }

    memory_estimates = {
        key: summary
        for key, summary in {
            "memory_final_predicted_peak_delta_bytes": _numeric_telemetry_summary(
                "memory_final_predicted_peak_delta_bytes"
            ),
            "memory_final_predicted_peak_rss_bytes": _numeric_telemetry_summary(
                "memory_final_predicted_peak_rss_bytes"
            ),
            "candidate_row_count": _numeric_telemetry_summary("candidate_row_count"),
            "query_batch_count": _numeric_telemetry_summary("query_batch_count"),
        }.items()
        if summary is not None
    }
    return {
        "run_count": len(per_run),
        "predict_seconds": {
            "min": min(predict_seconds),
            "p50": statistics.median(predict_seconds),
            "max": max(predict_seconds),
        },
        "peak_rss_gb": {
            "max": max(float(run["peak_rss_gb"]) for run in per_run),
        },
        "memory_estimates": memory_estimates,
    }


def _set_runtime_env(n_jobs: int) -> dict[str, str | None]:
    prior = {
        "S2AND_BACKEND": os.environ.get("S2AND_BACKEND"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    os.environ["S2AND_BACKEND"] = "rust"
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(n_jobs)))
    return prior


def _restore_runtime_env(prior: dict[str, str | None]) -> None:
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.runs <= 0:
        raise ValueError("--runs must be > 0")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be > 0")
    if args.query_limit < 0:
        raise ValueError("--query-limit must be >= 0")
    if args.query_limit == 0 or args.query_limit > 400:
        if not args.full_run:
            raise ValueError("Refusing large profiling run without --full-run")

    arrow_root = Path(args.arrow_root)
    model_manifest = (Path(args.model_path) / "manifest.json").resolve()
    model_manifest_sha256 = _require_sha256(
        model_manifest,
        args.expected_model_manifest_sha256,
        label="Model manifest",
    )
    dataset_root = _resolve_arrow_dataset_root(arrow_root, args.dataset)
    data_manifest = (dataset_root / "manifest.json").resolve()
    data_manifest_sha256 = _require_sha256(
        data_manifest,
        args.expected_data_manifest_sha256,
        label="Data manifest",
    )
    arrow_paths = _resolve_arrow_dataset_paths(arrow_root, args.dataset)
    signature_rows = _read_signature_rows(Path(arrow_paths["signatures"]))
    blocks = _block_dict(signature_rows)
    clusters_path = arrow_paths.get("clusters")
    if clusters_path is None:
        if not args.synthetic_seeds_when_clusters_missing:
            raise FileNotFoundError(
                f"Missing eval clusters for {args.dataset}; pass --synthetic-seeds-when-clusters-missing "
                "to generate deterministic profiling-only seed clusters"
            )
        signature_to_cluster_id = _synthetic_signature_to_cluster_id(
            blocks,
            max_seed_clusters=int(args.max_seed_clusters),
        )
        seed_source = "synthetic"
    else:
        signature_to_cluster_id = _read_signature_to_cluster_id(Path(clusters_path))
        seed_source = "clusters"
    workload = _select_workload(
        blocks=blocks,
        signature_to_cluster_id=signature_to_cluster_id,
        target_block=str(args.target_block or ""),
        query_limit=int(args.query_limit),
        max_seed_clusters=int(args.max_seed_clusters),
    )
    workload_sha256 = _workload_sha256(workload, args, seed_source)
    if workload_sha256 != args.expected_workload_sha256:
        raise ValueError(
            f"Workload SHA-256 mismatch: expected={args.expected_workload_sha256} observed={workload_sha256}"
        )
    from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow
    from s2and.production_model import load_production_model

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clusterer = load_production_model(str(args.model_path))
    clusterer.n_jobs = int(args.n_jobs)

    prior_env = _set_runtime_env(int(args.n_jobs))
    per_run: list[dict[str, Any]] = []
    run_file_prefix = f"{os.getpid()}_{time.time_ns()}"
    try:
        for run_index in range(int(args.runs)):
            cluster_seeds_path = output_dir / f"cluster_seeds_{run_file_prefix}_run_{run_index}.arrow"
            write_cluster_seeds_arrow(cluster_seeds_path, workload.seed_signature_to_cluster)
            request_arrow_paths = arrow_paths.with_request_sidecars(
                {"cluster_seeds": cluster_seeds_path},
                required_keys=("cluster_seeds",),
                context=f"promoted incremental Arrow profile run {run_index}",
                producer_hint="write the selected workload seeds as cluster_seeds.arrow",
            )
            with ProcessTreeRSSMonitor() as monitor:
                start = time.perf_counter()
                result = clusterer.predict_incremental_from_arrow_paths(
                    workload.block_signatures,
                    request_arrow_paths,
                    prevent_new_incompatibilities=False,
                    batching_threshold=None if args.batching_threshold <= 0 else int(args.batching_threshold),
                    total_ram_bytes=None if args.total_ram_bytes <= 0 else int(args.total_ram_bytes),
                )
                elapsed = time.perf_counter() - start
            telemetry = dict(result.get("incremental_linker_telemetry", {}))
            if result.get("incremental_linker_query_view") != "raw_arrow":
                raise RuntimeError("Profile run did not use raw Arrow")
            if telemetry.get("arrow_promoted_incremental") != 1:
                raise RuntimeError("Profile run missed promoted Arrow incremental execution")
            clustered_signature_ids = {
                str(signature_id) for members in dict(result.get("clusters", {})).values() for signature_id in members
            }
            if clustered_signature_ids != set(workload.block_signatures):
                raise RuntimeError("Profile run lost or added signatures")
            per_run.append(
                {
                    "run_index": run_index,
                    "predict_seconds": float(elapsed),
                    "peak_rss_gb": float(monitor.peak_gb),
                    "cluster_count": len(result.get("clusters", {})),
                    "telemetry": telemetry,
                }
            )
    finally:
        _restore_runtime_env(prior_env)

    payload = {
        "schema_version": REPORT_SCHEMA,
        "data_manifest_sha256": data_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "workload_sha256": workload_sha256,
        "runner": "promoted_incremental_arrow_profile",
        "canonical_arrow_root": str(DEFAULT_ARROW_ROOT),
        "arrow_root": str(arrow_root),
        "dataset": str(args.dataset),
        "target_block": workload.target_block,
        "block_signature_count": workload.block_signature_count,
        "profile_signature_count": len(workload.block_signatures),
        "seed_signature_count": len(workload.seed_signature_to_cluster),
        "seed_source": seed_source,
        "query_signature_count": len(workload.query_signature_ids),
        "runs": per_run,
        "summary": _summarize_runs(per_run),
        "rust_extension": collect_rust_extension_identity(require_release=bool(args.require_rust_release)),
        "run_metadata": build_run_metadata(script_path=Path(__file__).resolve(), project_root=PROJECT_ROOT),
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile promoted Rust/Arrow incremental linking without JSON or ANDData runners."
    )
    parser.add_argument("--arrow-root", type=Path, default=DEFAULT_ARROW_ROOT)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-data-manifest-sha256", required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Complete native production bundle path.")
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--expected-workload-sha256", required=True)
    parser.add_argument("--target-block", default="")
    parser.add_argument("--query-limit", type=int, default=25)
    parser.add_argument("--max-seed-clusters", type=int, default=0)
    parser.add_argument(
        "--synthetic-seeds-when-clusters-missing",
        action="store_true",
        help="Use deterministic profiling-only seed clusters when the Arrow bundle has no eval clusters JSON.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--batching-threshold", type=int, default=0)
    parser.add_argument("--total-ram-bytes", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-json", type=Path, required=True)
    parser.add_argument("--require-rust-release", action="store_true")
    parser.add_argument("--full-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_json.exists():
        raise FileExistsError(f"Report output already exists: {args.write_json}")
    payload = run(args)
    _write_fresh_json(args.write_json, payload)
    print(RESULT_JSON_START)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(RESULT_JSON_END)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

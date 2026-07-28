"""Produce the promoted incremental Arrow performance release report."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s2and.arrow_inputs import ArrowDataset  # noqa: E402

REPORT_SCHEMA = "s2and_performance_evaluation_report_v1"
RUNNER = "promoted_incremental_arrow_profile"
MAX_BOUNDED_QUERIES = 400
MAX_BOUNDED_SEED_CLUSTERS = 400
_WORKLOAD_KEYS = {
    "dataset",
    "target_block",
    "query_limit",
    "max_seed_clusters",
    "seed_source",
    "runs",
    "n_jobs",
    "batching_threshold",
    "total_ram_bytes",
    "synthetic_seeds_when_clusters_missing",
}


@dataclass(frozen=True)
class ProfileWorkload:
    """One bounded block split into seed and query signatures."""

    target_block: str
    block_signature_count: int
    seed_signature_to_cluster: dict[str, str]
    query_signature_ids: list[str]

    @property
    def block_signatures(self) -> list[str]:
        return [*self.seed_signature_to_cluster, *self.query_signature_ids]


class ProcessTreeRSSMonitor:
    """Sample peak RSS for this process and its children."""

    def __init__(self, psutil_module: Any, interval_seconds: float = 0.05):
        self._psutil = psutil_module
        self._process = psutil_module.Process()
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_bytes = 0

    def _sample(self) -> None:
        try:
            processes = [self._process, *self._process.children(recursive=True)]
        except self._psutil.Error:
            processes = [self._process]
        total_bytes = 0
        for process in processes:
            try:
                total_bytes += int(process.memory_info().rss)
            except self._psutil.Error:
                continue
        self._peak_bytes = max(self._peak_bytes, total_bytes)

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def __enter__(self) -> ProcessTreeRSSMonitor:
        self._sample()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample()

    @property
    def peak_gb(self) -> float:
        return self._peak_bytes / (1024**3)


def _require_psutil() -> Any:
    try:
        import psutil
    except ModuleNotFoundError as exc:
        raise RuntimeError("Process-tree RSS profiling requires psutil; run with `uv run --with psutil ...`") from exc
    return psutil


def _rust_extension_identity(require_release: bool) -> dict[str, Any]:
    from s2and.runtime import load_s2and_rust_extension

    extension = load_s2and_rust_extension()
    module_path = Path(str(extension.__file__)).resolve()
    build_info = dict(extension.get_build_info())
    debug_assertions = build_info.get("debug_assertions")
    if require_release and debug_assertions is not False:
        raise RuntimeError(
            "Rust release build required; rebuild with `uv run maturin develop -m s2and_rust/Cargo.toml --release`"
        )
    with module_path.open("rb") as binary:
        sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
    return {
        "available": True,
        "module_name": str(extension.__name__),
        "module_version": str(getattr(extension, "__version__", "unknown")),
        "module_file": str(module_path),
        "binary": {
            "path": str(module_path),
            "sha256": sha256,
            "size_bytes": module_path.stat().st_size,
        },
        "build_info": build_info,
        "debug_assertions": debug_assertions,
    }


def _run_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain")
    return {
        "generated_at_utc": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat(),
        "argv": sys.argv,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
    }


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


def _resolve_dataset_root(arrow_root: Path, dataset: str) -> Path:
    for candidate in (arrow_root / dataset, arrow_root / "datasets" / dataset):
        if (candidate / "manifest.json").exists():
            return candidate
    raise FileNotFoundError(f"Missing Arrow manifest for dataset={dataset!r} under {arrow_root}")


def read_signature_blocks(source_file: BinaryIO) -> dict[str, list[str]]:
    """Read the only two signature columns needed to select a profile block."""

    import pyarrow as pa

    with pa.PythonFile(source_file, mode="r") as source:
        rows = pa.ipc.open_file(source).read_all().select(["signature_id", "author_block"]).to_pylist()
    blocks: dict[str, list[str]] = {}
    for row in rows:
        blocks.setdefault(str(row["author_block"] or ""), []).append(str(row["signature_id"]))
    return blocks


def _read_signature_to_cluster_id(clusters_path: Path) -> dict[str, str]:
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    return {
        str(signature_id): str(cluster_id)
        for cluster_id, cluster in clusters.items()
        for signature_id in cluster["signature_ids"]
    }


def _synthetic_clusters(
    blocks: dict[str, list[str]],
    max_seed_clusters: int,
) -> dict[str, str]:
    return {
        signature_id: f"synthetic:{block_key}:{index}"
        for block_key, signatures in blocks.items()
        for index, signature_id in enumerate(signatures if max_seed_clusters == 0 else signatures[:max_seed_clusters])
    }


def select_profile_workload(
    *,
    blocks: dict[str, list[str]],
    signature_to_cluster_id: dict[str, str],
    target_block: str,
    query_limit: int,
    max_seed_clusters: int,
) -> ProfileWorkload:
    """Choose a deterministic target block, one seed per cluster, and queries."""

    if target_block and target_block not in blocks:
        raise ValueError(f"Requested target block {target_block!r} is not present")
    selected_block = target_block or max(blocks, key=lambda key: len(blocks[key]))
    block_signatures = [str(signature_id) for signature_id in blocks[selected_block]]
    seen_clusters: set[str] = set()
    seeds: dict[str, str] = {}
    for signature_id in block_signatures:
        cluster_id = signature_to_cluster_id.get(signature_id)
        if cluster_id is None or cluster_id in seen_clusters:
            continue
        seeds[signature_id] = cluster_id
        seen_clusters.add(cluster_id)
        if max_seed_clusters and len(seeds) >= max_seed_clusters:
            break
    if not seeds:
        raise ValueError(f"Selected block {selected_block!r} has no cluster-labeled seed signatures")

    queries = [signature_id for signature_id in block_signatures if signature_id not in seeds]
    if query_limit:
        queries = queries[:query_limit]
    if not queries:
        raise ValueError(f"Selected block {selected_block!r} has no query signatures after seed/query selection")
    return ProfileWorkload(selected_block, len(block_signatures), seeds, queries)


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def summary(values: list[float]) -> dict[str, float]:
        return {"min": min(values), "p50": statistics.median(values), "max": max(values)}

    memory_estimates: dict[str, dict[str, float]] = {}
    for key in (
        "memory_final_predicted_peak_delta_bytes",
        "memory_final_predicted_peak_rss_bytes",
        "candidate_row_count",
        "query_batch_count",
    ):
        values = [float(run["telemetry"][key]) for run in runs if key in run["telemetry"]]
        if values:
            memory_estimates[key] = summary(values)
    return {
        "run_count": len(runs),
        "predict_seconds": summary([float(run["predict_seconds"]) for run in runs]),
        "peak_rss_gb": {"max": max(float(run["peak_rss_gb"]) for run in runs)},
        "memory_estimates": memory_estimates,
    }


def _performance_args(args: argparse.Namespace) -> tuple[argparse.Namespace, dict[str, Any]]:
    payload = json.loads(Path(args.evaluation_plan).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "pairwise",
        "cluster",
        "performance",
        "baselines",
        "gates",
    }:
        raise ValueError("Invalid evaluation plan")
    performance = payload["performance"]
    if not isinstance(performance, Mapping) or set(performance) != {"arrow_root", "workload"}:
        raise ValueError("Invalid evaluation performance plan")
    arrow_root = performance["arrow_root"]
    workload = performance["workload"]
    if not isinstance(arrow_root, str) or not Path(arrow_root).is_absolute():
        raise ValueError("Evaluation performance arrow_root must be absolute")
    if not isinstance(workload, Mapping) or set(workload) != _WORKLOAD_KEYS:
        raise ValueError(f"Evaluation performance workload must contain exactly {sorted(_WORKLOAD_KEYS)}")

    integer_keys = ("query_limit", "max_seed_clusters", "runs", "n_jobs")
    optional_integer_keys = ("batching_threshold", "total_ram_bytes")
    if (
        not isinstance(workload["dataset"], str)
        or not workload["dataset"]
        or not isinstance(workload["target_block"], str)
        or workload["seed_source"] not in {"clusters", "synthetic"}
        or not isinstance(workload["synthetic_seeds_when_clusters_missing"], bool)
        or any(isinstance(workload[key], bool) or not isinstance(workload[key], int) for key in integer_keys)
        or any(
            value is not None and (isinstance(value, bool) or not isinstance(value, int))
            for value in (workload[key] for key in optional_integer_keys)
        )
    ):
        raise ValueError("Invalid evaluation performance workload")

    execution = argparse.Namespace(
        **vars(args),
        arrow_root=Path(arrow_root),
        **{key: (0 if workload[key] is None else workload[key]) for key in _WORKLOAD_KEYS - {"seed_source"}},
    )
    return execution, dict(workload)


def _validate_args(args: argparse.Namespace) -> None:
    if args.runs <= 0:
        raise ValueError("evaluation performance runs must be > 0")
    if args.n_jobs <= 0:
        raise ValueError("evaluation performance n_jobs must be > 0")
    if args.query_limit < 0:
        raise ValueError("evaluation performance query_limit must be >= 0")
    if args.max_seed_clusters < 0:
        raise ValueError("evaluation performance max_seed_clusters must be >= 0")
    if not args.full_run and not 1 <= args.query_limit <= MAX_BOUNDED_QUERIES:
        raise ValueError(
            f"evaluation performance query_limit must be 1..{MAX_BOUNDED_QUERIES} unless --full-run is set"
        )
    if not args.full_run and not 1 <= args.max_seed_clusters <= MAX_BOUNDED_SEED_CLUSTERS:
        raise ValueError(
            f"evaluation performance max_seed_clusters must be 1..{MAX_BOUNDED_SEED_CLUSTERS} unless --full-run is set"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the profile and return its release-report payload."""

    args, planned_workload = _performance_args(args)
    _validate_args(args)
    psutil_module = _require_psutil()
    rust_extension = _rust_extension_identity(bool(args.require_rust_release))
    arrow_root = Path(args.arrow_root)
    dataset_root = _resolve_dataset_root(arrow_root, args.dataset)
    profile_runs: list[dict[str, Any]] = []
    with ArrowDataset.open(
        dataset_root,
        require_specter=True,
        require_name_counts_index=True,
    ) as arrow_dataset:
        clusters_path = dataset_root / f"{args.dataset}_clusters.json"
        if not clusters_path.is_file():
            if not args.synthetic_seeds_when_clusters_missing:
                raise FileNotFoundError(
                    f"Missing eval clusters for {args.dataset}; enable synthetic seeds in the evaluation plan"
                )
            clusters_path = None
        with arrow_dataset.use() as lease, lease.open_file("signatures") as signatures_file:
            blocks = read_signature_blocks(signatures_file)
        if clusters_path is None:
            signature_to_cluster_id = _synthetic_clusters(blocks, int(args.max_seed_clusters))
            seed_source = "synthetic"
        else:
            signature_to_cluster_id = _read_signature_to_cluster_id(clusters_path)
            seed_source = "clusters"
        workload = select_profile_workload(
            blocks=blocks,
            signature_to_cluster_id=signature_to_cluster_id,
            target_block=str(args.target_block or ""),
            query_limit=int(args.query_limit),
            max_seed_clusters=int(args.max_seed_clusters),
        )

        from s2and.production_model import load_production_model

        clusterer = load_production_model(str(args.model_path))
        clusterer.n_jobs = int(args.n_jobs)
        env_keys = ("S2AND_BACKEND", "OMP_NUM_THREADS")
        prior_env = {key: os.environ.get(key) for key in env_keys}
        os.environ.update(S2AND_BACKEND="rust", OMP_NUM_THREADS=str(args.n_jobs))
        try:
            for run_index in range(int(args.runs)):
                with ProcessTreeRSSMonitor(psutil_module) as monitor:
                    start = time.perf_counter()
                    result = clusterer.predict_incremental_from_arrow(
                        workload.block_signatures,
                        arrow_dataset,
                        prevent_new_incompatibilities=False,
                        batching_threshold=None if args.batching_threshold <= 0 else int(args.batching_threshold),
                        total_ram_bytes=None if args.total_ram_bytes <= 0 else int(args.total_ram_bytes),
                        cluster_seeds_require=workload.seed_signature_to_cluster,
                    )
                    elapsed = time.perf_counter() - start
                telemetry = dict(result.get("incremental_linker_telemetry", {}))
                if result.get("incremental_linker_query_view") != "raw_arrow":
                    raise RuntimeError("Profile run did not use raw Arrow")
                if telemetry.get("arrow_promoted_incremental") != 1:
                    raise RuntimeError("Profile run missed promoted Arrow incremental execution")
                clustered_ids = {
                    str(signature_id)
                    for members in dict(result.get("clusters", {})).values()
                    for signature_id in members
                }
                if clustered_ids != set(workload.block_signatures):
                    raise RuntimeError("Profile run lost or added signatures")
                profile_runs.append(
                    {
                        "run_index": run_index,
                        "predict_seconds": elapsed,
                        "peak_rss_gb": monitor.peak_gb,
                        "cluster_count": len(result.get("clusters", {})),
                        "telemetry": telemetry,
                    }
                )
        finally:
            for key, value in prior_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    workload_report = {
        "dataset": str(args.dataset),
        "target_block": str(args.target_block or ""),
        "query_limit": int(args.query_limit),
        "max_seed_clusters": int(args.max_seed_clusters),
        "seed_source": seed_source,
        "runs": int(args.runs),
        "n_jobs": int(args.n_jobs),
        "batching_threshold": None if args.batching_threshold <= 0 else int(args.batching_threshold),
        "total_ram_bytes": None if args.total_ram_bytes <= 0 else int(args.total_ram_bytes),
        "synthetic_seeds_when_clusters_missing": bool(args.synthetic_seeds_when_clusters_missing),
    }
    if workload_report != planned_workload:
        raise RuntimeError("Observed performance workload does not match the evaluation plan")
    return {
        "schema_version": REPORT_SCHEMA,
        "runner": RUNNER,
        "workload": workload_report,
        "arrow_root": str(arrow_root),
        "dataset": str(args.dataset),
        "target_block": workload.target_block,
        "block_signature_count": workload.block_signature_count,
        "profile_signature_count": len(workload.block_signatures),
        "seed_signature_count": len(workload.seed_signature_to_cluster),
        "seed_source": seed_source,
        "query_signature_count": len(workload.query_signature_ids),
        "runs": profile_runs,
        "summary": _summarize_runs(profile_runs),
        "rust_extension": rust_extension,
        "run_metadata": _run_metadata(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
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
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts._rust_suite.common import (
        ProcessTreeRSSMonitor,
        build_run_metadata,
        collect_rust_extension_identity,
        compute_rss_growth_fraction,
    )
else:
    try:
        from _rust_suite.common import (
            ProcessTreeRSSMonitor,
            build_run_metadata,
            collect_rust_extension_identity,
            compute_rss_growth_fraction,
        )
    except ModuleNotFoundError:
        _SCRIPTS_DIR = Path(__file__).resolve().parents[1]
        if str(_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_DIR))
        from _rust_suite.common import (
            ProcessTreeRSSMonitor,
            build_run_metadata,
            collect_rust_extension_identity,
            compute_rss_growth_fraction,
        )


def _import_rust_module() -> Any:
    try:
        import s2and_rust
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"Failed to import s2and_rust: {exc}") from exc
    rust_featurizer = getattr(s2and_rust, "RustFeaturizer", None)
    if rust_featurizer is None:
        raise RuntimeError("s2and_rust.RustFeaturizer is unavailable")
    return s2and_rust


def _dataset_paths(dataset_name: str) -> dict[str, str | None]:
    dataset = dataset_name.strip().lower()
    from s2and.consts import PROJECT_ROOT_PATH

    project_root = Path(PROJECT_ROOT_PATH)
    if dataset in {"dummy", "qian"}:
        dataset_root = project_root / "tests" / dataset
        signatures = dataset_root / "signatures.json"
        papers = dataset_root / "papers.json"
        clusters = dataset_root / "clusters.json"
        cluster_seeds = dataset_root / "cluster_seeds.json"
        return {
            "dataset_name": dataset,
            "signatures": str(signatures),
            "papers": str(papers),
            "clusters": str(clusters) if clusters.exists() else None,
            "cluster_seeds": str(cluster_seeds) if cluster_seeds.exists() else None,
            "specter": None,
        }

    dataset_root = project_root / "data" / dataset
    signatures = dataset_root / f"{dataset}_signatures.json"
    papers = dataset_root / f"{dataset}_papers.json"
    clusters = dataset_root / f"{dataset}_clusters.json"
    cluster_seeds = dataset_root / f"{dataset}_cluster_seeds.json"
    specter = dataset_root / f"{dataset}_specter.pickle"
    return {
        "dataset_name": dataset,
        "signatures": str(signatures),
        "papers": str(papers),
        "clusters": str(clusters) if clusters.exists() else None,
        "cluster_seeds": str(cluster_seeds) if cluster_seeds.exists() else None,
        "specter": str(specter) if specter.exists() else None,
    }


def _validate_paths(paths: dict[str, str | None]) -> None:
    for required_key in ("signatures", "papers"):
        path = paths.get(required_key)
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset path for {required_key}: {path}")
    clusters_path = paths.get("clusters")
    if clusters_path is not None and not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Configured clusters path does not exist: {clusters_path}")
    cluster_seeds_path = paths.get("cluster_seeds")
    if cluster_seeds_path is not None and not os.path.exists(cluster_seeds_path):
        raise FileNotFoundError(f"Configured cluster seeds path does not exist: {cluster_seeds_path}")
    specter_path = paths.get("specter")
    if specter_path is not None and not os.path.exists(specter_path):
        raise FileNotFoundError(f"Configured specter path does not exist: {specter_path}")


def _build_dataset(
    *,
    paths: dict[str, str | None],
    compute_reference_features: bool,
    preprocess: bool,
    num_threads: int,
    use_specter: bool,
) -> Any:
    from s2and.data import ANDData

    signatures_path = paths["signatures"]
    papers_path = paths["papers"]
    if signatures_path is None or papers_path is None:
        raise ValueError("Missing required signatures/papers paths")

    return ANDData(
        signatures=signatures_path,
        papers=papers_path,
        name=f"{paths['dataset_name']}_stress",
        mode="train",
        specter_embeddings=paths["specter"] if use_specter else None,
        clusters=paths["clusters"],
        cluster_seeds=paths["cluster_seeds"],
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=1000,
        val_pairs_size=1000,
        test_pairs_size=1000,
        n_jobs=max(1, int(num_threads)),
        load_name_counts=False,
        preprocess=bool(preprocess),
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=False,
        compute_reference_features=bool(compute_reference_features),
    )


def _build_from_json_paths(
    *,
    s2and_rust_module: Any,
    paths: dict[str, str | None],
    compute_reference_features: bool,
    preprocess: bool,
    num_threads: int,
    use_specter: bool,
) -> Any:
    return s2and_rust_module.RustFeaturizer.from_json_paths(
        paths["signatures"],
        paths["papers"],
        paths["clusters"],
        paths["cluster_seeds"],
        paths["specter"] if use_specter else None,
        None,  # name_tuples_path
        None,  # name_counts_path
        bool(preprocess),
        bool(compute_reference_features),
        0.0,
        10000.0,
        max(1, int(num_threads)),
    )


def _build_from_dataset(
    *,
    s2and_rust_module: Any,
    paths: dict[str, str | None],
    compute_reference_features: bool,
    preprocess: bool,
    num_threads: int,
    use_specter: bool,
) -> tuple[Any, Any]:
    dataset = _build_dataset(
        paths=paths,
        compute_reference_features=compute_reference_features,
        preprocess=preprocess,
        num_threads=num_threads,
        use_specter=use_specter,
    )
    featurizer = s2and_rust_module.RustFeaturizer.from_dataset(
        dataset,
        0.0,
        10000.0,
        max(1, int(num_threads)),
    )
    return featurizer, dataset


def run_rebuild_stress(
    *,
    dataset: str,
    build_path: str,
    repeats: int,
    num_threads: int,
    compute_reference_features: bool = False,
    preprocess: bool = True,
    use_specter: bool = False,
    rss_sample_ms: int = 50,
    rss_growth_max_fraction: float | None = None,
    require_rust_release: bool = False,
    write_json: str | None = None,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if build_path not in {"from_json_paths", "from_dataset"}:
        raise ValueError("build_path must be one of: from_json_paths, from_dataset")
    if int(rss_sample_ms) <= 0:
        raise ValueError("rss_sample_ms must be positive")

    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    os.environ.setdefault("S2AND_BACKEND", "rust")
    paths = _dataset_paths(dataset)
    _validate_paths(paths)
    s2and_rust_module = _import_rust_module()
    rust_extension_identity = collect_rust_extension_identity(
        require_release=bool(require_rust_release),
        fail_if_unavailable=True,
    )

    iterations: list[dict[str, Any]] = []
    success_count = 0
    rss_peak_gb_by_iteration: list[float] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    sample_interval_seconds = max(0.001, float(rss_sample_ms) / 1000.0)
    print(
        "Starting rebuild stress: "
        f"dataset={paths['dataset_name']} build_path={build_path} repeats={repeats} "
        f"num_threads={num_threads} preprocess={preprocess} use_specter={use_specter} "
        f"rss_sample_ms={rss_sample_ms}"
    )

    for iteration in range(1, repeats + 1):
        status = "ok"
        error_payload = None
        with ProcessTreeRSSMonitor(interval_seconds=sample_interval_seconds) as rss_monitor:
            start = time.perf_counter()
            featurizer = None
            dataset_obj = None
            try:
                if build_path == "from_json_paths":
                    featurizer = _build_from_json_paths(
                        s2and_rust_module=s2and_rust_module,
                        paths=paths,
                        compute_reference_features=compute_reference_features,
                        preprocess=preprocess,
                        num_threads=num_threads,
                        use_specter=use_specter,
                    )
                else:
                    featurizer, dataset_obj = _build_from_dataset(
                        s2and_rust_module=s2and_rust_module,
                        paths=paths,
                        compute_reference_features=compute_reference_features,
                        preprocess=preprocess,
                        num_threads=num_threads,
                        use_specter=use_specter,
                    )
                del featurizer
                del dataset_obj
                success_count += 1
            except Exception as exc:  # pragma: no cover - exercised in integration environments
                status = "error"
                error_payload = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            finally:
                gc.collect()
            elapsed = time.perf_counter() - start
            rss_peak_gb = float(round(rss_monitor.peak_gb, 6))

        rss_peak_gb_by_iteration.append(rss_peak_gb)
        iteration_payload = {
            "iteration": int(iteration),
            "status": status,
            "elapsed_seconds": float(round(elapsed, 6)),
            "rss_peak_gb": rss_peak_gb,
            "error": error_payload,
        }
        iterations.append(iteration_payload)

        if error_payload is None:
            print(f"[{iteration}/{repeats}] status=ok elapsed_seconds={elapsed:.3f} rss_peak_gb={rss_peak_gb:.3f}")
        else:
            print(
                f"[{iteration}/{repeats}] status=error elapsed_seconds={elapsed:.3f} "
                f"rss_peak_gb={rss_peak_gb:.3f} "
                f"error_type={error_payload['type']} error_message={error_payload['message']}"
            )

    rss_growth_fraction = compute_rss_growth_fraction(rss_peak_gb_by_iteration)
    rss_growth_gate_pass = True
    if rss_growth_max_fraction is not None and rss_growth_fraction is not None:
        rss_growth_gate_pass = rss_growth_fraction <= float(rss_growth_max_fraction)

    result: dict[str, Any] = {
        "dataset": str(paths["dataset_name"]),
        "build_path": str(build_path),
        "repeats": int(repeats),
        "num_threads": int(max(1, int(num_threads))),
        "compute_reference_features": bool(compute_reference_features),
        "preprocess": bool(preprocess),
        "use_specter": bool(use_specter),
        "rss_sample_ms": int(rss_sample_ms),
        "rss_growth_max_fraction": (None if rss_growth_max_fraction is None else float(rss_growth_max_fraction)),
        "rss_peak_gb_by_iteration": rss_peak_gb_by_iteration,
        "rss_growth_fraction": rss_growth_fraction,
        "rss_growth_gate_pass": bool(rss_growth_gate_pass),
        "started_utc": started,
        "success_count": int(success_count),
        "failure_count": int(repeats - success_count),
        "iterations": iterations,
        "rust_extension_identity": rust_extension_identity,
        "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
    }

    if write_json:
        output_path = Path(write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote stress artifact: {output_path}")

    if not rss_growth_gate_pass:
        assert rss_growth_max_fraction is not None
        raise RuntimeError(
            "rss_growth_fraction exceeded threshold: " f"{rss_growth_fraction} > {float(rss_growth_max_fraction)}"
        )

    return result


def _rss_growth_fraction(rss_peak_gb_by_iteration: list[float]) -> float | None:
    return compute_rss_growth_fraction(rss_peak_gb_by_iteration)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatedly build/drop RustFeaturizer using from_json_paths or from_dataset "
            "to stress lifecycle robustness."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. dummy, qian, aminer)")
    parser.add_argument(
        "--build-path",
        choices=("from_json_paths", "from_dataset"),
        default="from_json_paths",
        help="Rust build path to stress.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of rebuild iterations.")
    parser.add_argument("--num-threads", type=int, default=1, help="Thread count passed to RustFeaturizer build.")
    parser.add_argument(
        "--compute-reference-features",
        action="store_true",
        help="Build featurizer with reference features enabled.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable preprocess at ingest time (for toggle-matrix triage).",
    )
    parser.add_argument(
        "--use-specter",
        action="store_true",
        help="Load and pass specter embeddings when available.",
    )
    parser.add_argument(
        "--rss-sample-ms",
        type=int,
        default=50,
        help="RSS sampler interval in milliseconds.",
    )
    parser.add_argument(
        "--rss-growth-max-fraction",
        type=float,
        default=None,
        help="Optional max allowed RSS growth fraction ((last-first)/first).",
    )
    parser.add_argument(
        "--require-rust-release",
        type=int,
        choices=[0, 1],
        default=0,
        help="Fail if loaded Rust extension build has debug_assertions enabled.",
    )
    parser.add_argument("--write-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_rebuild_stress(
        dataset=args.dataset,
        build_path=args.build_path,
        repeats=args.repeats,
        num_threads=args.num_threads,
        compute_reference_features=bool(args.compute_reference_features),
        preprocess=not bool(args.no_preprocess),
        use_specter=bool(args.use_specter),
        rss_sample_ms=int(args.rss_sample_ms),
        rss_growth_max_fraction=args.rss_growth_max_fraction,
        require_rust_release=bool(args.require_rust_release),
        write_json=args.write_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

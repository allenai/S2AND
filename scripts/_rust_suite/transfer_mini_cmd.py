"""
Mini transfer-experiment profiler.

Reproduces the Rust-hot code paths of transfer_experiment_internal.py
at reduced scale so a full Python-vs-Rust A/B completes in ~30 min
instead of ~10 hours.

Knobs turned down vs the real command:
  - 3 datasets instead of 10  (default: kisti, inspire, inventors_s2and)
  - 10k train pairs instead of 100k
  - 5 hyperopt iterations instead of 50

What's preserved:
  - Multi-dataset featurizer cache thrashing  (S2AND_RUST_FEATURIZER_MAX_INMEM=1)
  - Union model: train pairwise + nameless, fit clusterer on combined val blocks
  - cluster_eval on a target dataset  (fused predict path with Rust featurization)
  - Stage-level timing and peak-RSS monitoring

Usage:
  # A/B two configs as isolated subprocesses (recommended):
  uv run --no-project python scripts/rust_suite.py transfer-mini --mode compare \\
      --datasets kisti inspire inventors_s2and --target inventors_s2and --n-jobs 8 \\
      --write-json scratch/profile_transfer_mini.json

  # single backend (foreground, for debugging):
  uv run --no-project python scripts/rust_suite.py transfer-mini --mode single \\
      --backend rust --datasets kisti inspire inventors_s2and --target inventors_s2and --n-jobs 8

Diagnostic toggles:
  --rust-cleanup-boundary {0,1}
  --force-python-paper-preprocess {0,1}
"""

import argparse
import gc
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

from _rust_suite.common import (
    ProcessTreeRSSMonitor,
    build_run_metadata,
    canonical_sha256,
    collect_rust_extension_identity,
    extract_marked_json_payload,
)

RESULT_JSON_START = "===S2AND_PROFILE_RESULT_START==="
RESULT_JSON_END = "===S2AND_PROFILE_RESULT_END==="
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUST_FORCE_PYTHON_PAPER_PREPROCESS_ENV = "S2AND_RUST_FORCE_PYTHON_PAPER_PREPROCESS"

# Match transfer_experiment_internal.py defaults
SPECTER_SUFFIX = "_specter.pickle"
BLOCK_TYPE = "s2"
PREPROCESS = True
PAIRWISE_ONLY_DATASETS = {"medline", "augmented"}
WORKLOAD_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "datasets": ["kisti"],
        "target": "kisti",
        "n_jobs": 2,
        "n_train_pairs": 300,
        "n_iter": 1,
    },
    "full": {
        "datasets": ["kisti", "arnetminer", "zbmath"],
        "target": "kisti",
        "n_jobs": 4,
        "n_train_pairs": 10000,
        "n_iter": 5,
    },
}


def _build_workload(
    *,
    datasets: list[str],
    target: str,
    n_jobs: int,
    n_train_pairs: int,
    n_iter: int,
    random_seed: int,
    train_pairs_size_mode: str,
) -> dict[str, Any]:
    return {
        "datasets": list(datasets),
        "target": str(target),
        "n_jobs": int(n_jobs),
        "n_train_pairs": int(n_train_pairs),
        "n_iter": int(n_iter),
        "random_seed": int(random_seed),
        "train_pairs_size_mode": str(train_pairs_size_mode),
    }


def _workload_id(workload: dict[str, Any]) -> str:
    return canonical_sha256(workload)


def _resolve_workload(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    preset_defaults = WORKLOAD_PRESETS[str(args.preset)]
    preset_datasets = list(cast(list[str], preset_defaults["datasets"]))
    datasets = list(args.datasets) if args.datasets is not None else preset_datasets
    target = str(args.target) if args.target is not None else str(preset_defaults["target"])
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else int(cast(int, preset_defaults["n_jobs"]))
    n_train_pairs = (
        int(args.n_train_pairs) if args.n_train_pairs is not None else int(cast(int, preset_defaults["n_train_pairs"]))
    )
    n_iter = int(args.n_iter) if args.n_iter is not None else int(cast(int, preset_defaults["n_iter"]))
    random_seed = int(args.random_seed) if args.random_seed is not None else 1
    train_pairs_size_mode = str(args.train_pairs_size_mode) if args.train_pairs_size_mode is not None else "scaled"
    workload = _build_workload(
        datasets=datasets,
        target=target,
        n_jobs=n_jobs,
        n_train_pairs=n_train_pairs,
        n_iter=n_iter,
        random_seed=random_seed,
        train_pairs_size_mode=train_pairs_size_mode,
    )
    return workload, _workload_id(workload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_data_dir() -> str:
    project_root = str(PROJECT_ROOT)
    config_path = os.path.join(project_root, "data", "path_config.json")
    with open(config_path) as f:
        config = json.load(f)
    internal = config.get("internal_data_dir", "")
    if internal and os.path.exists(internal):
        return internal
    return os.path.join(project_root, "data")


def _fmt(seconds: float) -> str:
    return f"{seconds:.3f}"


def _snapshot_stage_rss(stage_rss: dict[str, float], monitor: ProcessTreeRSSMonitor, key: str) -> None:
    stage_rss[key] = round(monitor.sample_gb(), 3)


def _effective_train_pairs_size(n_train_pairs: int, mode: str) -> int:
    if mode == "exact_internal":
        # Matches transfer_experiment_internal.py behavior exactly.
        return max(int(n_train_pairs), 100000)
    if mode == "scaled":
        return max(1, int(n_train_pairs))
    raise ValueError(f"Unknown train_pairs_size_mode={mode!r}")


def _trial_duration_seconds(book_time: Any, refresh_time: Any) -> float | None:
    if book_time is None or refresh_time is None:
        return None
    try:
        maybe_delta = refresh_time - book_time
    except TypeError:
        return None
    if hasattr(maybe_delta, "total_seconds"):
        seconds = float(maybe_delta.total_seconds())
    elif isinstance(maybe_delta, int | float):
        seconds = float(maybe_delta)
    else:
        return None
    if seconds < 0:
        return None
    return seconds


def _normalize_hyperopt_trial_vals(values: Any) -> dict[str, list[Any]]:
    if not isinstance(values, dict):
        return {}
    normalized: dict[str, list[Any]] = {}
    for key, raw_value in sorted(values.items(), key=lambda item: str(item[0])):
        if isinstance(raw_value, list):
            normalized[str(key)] = [
                int(value) if hasattr(value, "is_integer") and value.is_integer() else value for value in raw_value
            ]
    return normalized


def _summarize_hyperopt_trials(trials_obj: Any) -> dict[str, Any]:
    trials = getattr(trials_obj, "trials", None)
    if not isinstance(trials, list):
        return {"available": False}

    losses: list[float] = []
    durations: list[float] = []
    trial_param_hashes: list[str] = []

    for trial in trials:
        if not isinstance(trial, dict):
            continue
        result_payload = trial.get("result")
        if isinstance(result_payload, dict):
            loss_value = result_payload.get("loss")
            if isinstance(loss_value, int | float) and not isinstance(loss_value, bool):
                losses.append(float(loss_value))
        duration_seconds = _trial_duration_seconds(trial.get("book_time"), trial.get("refresh_time"))
        if duration_seconds is not None:
            durations.append(duration_seconds)
        misc_payload = trial.get("misc")
        trial_vals = _normalize_hyperopt_trial_vals(misc_payload.get("vals") if isinstance(misc_payload, dict) else {})
        if len(trial_vals) > 0:
            trial_param_hashes.append(canonical_sha256(trial_vals))

    summary: dict[str, Any] = {
        "available": True,
        "n_trials": int(len(trials)),
    }
    if len(losses) > 0:
        summary["loss_min"] = round(min(losses), 6)
        summary["loss_max"] = round(max(losses), 6)
        summary["loss_mean"] = round(sum(losses) / len(losses), 6)
    if len(durations) > 0:
        summary["trial_seconds_total"] = round(sum(durations), 6)
        summary["trial_seconds_mean"] = round(sum(durations) / len(durations), 6)
        summary["trial_seconds"] = [round(value, 6) for value in durations]
    if len(trial_param_hashes) > 0:
        summary["trial_param_hashes"] = trial_param_hashes
    return summary


def _lgbm_fit_summary(modeler: Any) -> dict[str, Any]:
    classifier = getattr(modeler, "classifier", None)
    if classifier is None:
        return {}
    summary: dict[str, Any] = {}
    best_iteration = getattr(classifier, "best_iteration_", None)
    if isinstance(best_iteration, int):
        summary["best_iteration"] = int(best_iteration)
    n_estimators_fit = getattr(classifier, "n_estimators_", None)
    if isinstance(n_estimators_fit, int):
        summary["n_estimators_fit"] = int(n_estimators_fit)
    else:
        n_estimators = getattr(classifier, "n_estimators", None)
        if isinstance(n_estimators, int):
            summary["n_estimators"] = int(n_estimators)
    return summary


def _resolve_dataset_file(
    data_dir: str,
    dataset_name: str,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    dataset_root = os.path.join(data_dir, dataset_name)
    for relative_path in candidates:
        path = os.path.join(dataset_root, relative_path)
        if os.path.exists(path):
            return path
    if required:
        raise FileNotFoundError(
            f"Missing required dataset file for {dataset_name}. Tried: "
            + ", ".join(os.path.join(dataset_root, c) for c in candidates)
        )
    return None


def _build_anddata_kwargs(
    *,
    data_dir: str,
    dataset_name: str,
    n_jobs: int,
    random_seed: int,
    n_train_pairs: int,
    n_val_test_size: int,
    name_counts: dict[str, Any],
    train_pairs_size_mode: str,
) -> dict[str, Any]:
    signatures_path = _resolve_dataset_file(
        data_dir,
        dataset_name,
        [f"{dataset_name}_signatures.json", "signatures.json"],
    )
    papers_path = _resolve_dataset_file(
        data_dir,
        dataset_name,
        [f"{dataset_name}_papers.json", "papers.json"],
    )
    specter_path = _resolve_dataset_file(
        data_dir,
        dataset_name,
        [f"{dataset_name}{SPECTER_SUFFIX}", "specter.pickle", f"{dataset_name}_specter2.pkl", "specter2.pkl"],
    )

    if dataset_name in PAIRWISE_ONLY_DATASETS:
        clusters_path = None
        train_pairs_path = _resolve_dataset_file(data_dir, dataset_name, ["train_pairs.csv"])
        val_pairs_path = _resolve_dataset_file(data_dir, dataset_name, ["val_pairs.csv"], required=False)
        test_pairs_path = _resolve_dataset_file(data_dir, dataset_name, ["test_pairs.csv"])
    else:
        clusters_path = _resolve_dataset_file(
            data_dir,
            dataset_name,
            [f"{dataset_name}_clusters.json", "clusters.json"],
        )
        train_pairs_path = None
        val_pairs_path = None
        test_pairs_path = None

    return {
        "signatures": signatures_path,
        "papers": papers_path,
        "name": dataset_name,
        "mode": "train",
        "specter_embeddings": specter_path,
        "clusters": clusters_path,
        "block_type": BLOCK_TYPE,
        "train_pairs": train_pairs_path,
        "val_pairs": val_pairs_path,
        "test_pairs": test_pairs_path,
        "train_pairs_size": _effective_train_pairs_size(n_train_pairs, train_pairs_size_mode),
        "val_pairs_size": n_val_test_size,
        "test_pairs_size": n_val_test_size,
        "n_jobs": n_jobs,
        "load_name_counts": name_counts,
        "preprocess": PREPROCESS,
        "random_seed": random_seed,
        "name_tuples": "filtered",
    }


# ---------------------------------------------------------------------------
# Core single-process run
# ---------------------------------------------------------------------------
def _single_run(
    backend: str,
    dataset_names: list[str],
    target_name: str,
    n_jobs: int,
    n_train_pairs: int,
    n_iter: int,
    random_seed: int,
    train_pairs_size_mode: str,
    run_label: str,
    workload: dict[str, Any],
    workload_id: str,
    require_rust_release: bool,
    rust_cleanup_boundary: bool,
    force_python_paper_preprocess: bool,
) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))
    os.environ["S2AND_BACKEND"] = backend
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    # Match the internal script: limit featurizer residency
    os.environ["S2AND_RUST_FEATURIZER_MAX_INMEM"] = "1"
    # Improve RSS peak capture inside Rust batch loops unless explicitly disabled.
    os.environ.setdefault("S2AND_RUST_BATCH_RSS_SAMPLER_MS", "5")
    if force_python_paper_preprocess:
        os.environ[RUST_FORCE_PYTHON_PAPER_PREPROCESS_ENV] = "1"
    else:
        os.environ.pop(RUST_FORCE_PYTHON_PAPER_PREPROCESS_ENV, None)

    rust_extension_identity: dict[str, Any] | None = None
    if backend == "rust":
        rust_extension_identity = collect_rust_extension_identity(
            require_release=bool(require_rust_release),
            fail_if_unavailable=False,
        )

    import numpy as np
    from hyperopt import hp

    from s2and.consts import DEFAULT_CHUNK_SIZE, FEATURIZER_VERSION, NAME_COUNTS_PATH
    from s2and.data import ANDData
    from s2and.eval import cluster_eval
    from s2and.featurizer import FeaturizationInfo, featurize
    from s2and.file_cache import cached_path
    from s2and.model import Clusterer, FastCluster, PairwiseModeler

    DATA_DIR = _resolve_data_dir()
    N_VAL_TEST_SIZE = 10000

    FEATURES_TO_USE = [
        "name_similarity",
        "affiliation_similarity",
        "email_similarity",
        "coauthor_similarity",
        "venue_similarity",
        "year_diff",
        "title_similarity",
        # "reference_features",  # matches internal script
        "misc_features",
        "name_counts",
        "embedding_similarity",
        "journal_similarity",
        "advanced_name_similarity",
    ]
    NAMELESS_FEATURES_TO_USE = [
        f for f in FEATURES_TO_USE if f not in {"name_similarity", "advanced_name_similarity", "name_counts"}
    ]

    FEATURIZER_INFO = FeaturizationInfo(features_to_use=FEATURES_TO_USE, featurizer_version=FEATURIZER_VERSION)
    NAMELESS_FEATURIZER_INFO = FeaturizationInfo(
        features_to_use=NAMELESS_FEATURES_TO_USE, featurizer_version=FEATURIZER_VERSION
    )
    MONOTONE_CONSTRAINTS = FEATURIZER_INFO.lightgbm_monotone_constraints
    NAMELESS_MONOTONE_CONSTRAINTS = NAMELESS_FEATURIZER_INFO.lightgbm_monotone_constraints

    stage_timings: dict[str, Any] = {}
    stage_rss_gb: dict[str, float] = {}
    total_start = time.perf_counter()

    with ProcessTreeRSSMonitor(interval_seconds=0.05) as monitor:
        # ----- load name counts -----
        t0 = time.perf_counter()
        with open(cached_path(NAME_COUNTS_PATH), "rb") as f:
            first_dict, last_dict, first_last_dict, last_first_initial_dict = pickle.load(f)
        name_counts = {
            "first_dict": first_dict,
            "last_dict": last_dict,
            "first_last_dict": first_last_dict,
            "last_first_initial_dict": last_first_initial_dict,
        }
        stage_timings["name_counts_load_seconds"] = round(time.perf_counter() - t0, 3)
        _snapshot_stage_rss(stage_rss_gb, monitor, "name_counts_load")

        # ----- per-dataset: load + featurize -----
        datasets: dict[str, dict[str, Any]] = {}
        per_dataset_timings: dict[str, dict[str, Any]] = {}

        for dataset_name in dataset_names:
            dt: dict[str, Any] = {}
            print(f"  [{run_label}] Loading {dataset_name}...")

            t0 = time.perf_counter()
            anddata_kwargs = _build_anddata_kwargs(
                data_dir=DATA_DIR,
                dataset_name=dataset_name,
                n_jobs=n_jobs,
                random_seed=random_seed,
                n_train_pairs=n_train_pairs,
                n_val_test_size=N_VAL_TEST_SIZE,
                name_counts=name_counts,
                train_pairs_size_mode=train_pairs_size_mode,
            )
            anddata = ANDData(**anddata_kwargs)
            dt["anddata_build_seconds"] = round(time.perf_counter() - t0, 3)
            dt["anddata_train_pairs_size"] = int(anddata_kwargs["train_pairs_size"])
            dt["rss_after_anddata_build_gb"] = round(monitor.sample_gb(), 3)
            print(f"  [{run_label}] {dataset_name} built in {_fmt(dt['anddata_build_seconds'])}s")

            t0 = time.perf_counter()
            train, val, test = featurize(
                anddata,
                FEATURIZER_INFO,
                n_jobs=n_jobs,
                use_cache=False,
                chunk_size=DEFAULT_CHUNK_SIZE,
                nameless_featurizer_info=NAMELESS_FEATURIZER_INFO,
                nan_value=np.nan,
            )
            dt["featurize_seconds"] = round(time.perf_counter() - t0, 3)
            dt["rss_after_featurize_gb"] = round(monitor.sample_gb(), 3)
            print(f"  [{run_label}] {dataset_name} featurized in {_fmt(dt['featurize_seconds'])}s")

            X_train, y_train, nameless_X_train = train
            # Downsample if needed (match internal script behavior)
            if len(y_train) > n_train_pairs:
                np.random.seed(random_seed)
                subset = np.random.choice(len(y_train), size=n_train_pairs, replace=False)
                X_train = X_train[subset, :]
                if nameless_X_train is not None:
                    nameless_X_train = nameless_X_train[subset, :]
                y_train = y_train[subset]
            X_val, y_val, nameless_X_val = val
            assert test is not None
            X_test, y_test, nameless_X_test = test

            dt["n_train_pairs"] = int(len(y_train))
            dt["n_val_pairs"] = int(len(y_val))
            dt["n_test_pairs"] = int(len(y_test))
            rust_build_count: int | None = None
            if backend == "rust":
                try:
                    from s2and import feature_port as _feature_port

                    rust_build_count = int(_feature_port._rust_featurizer_build_count(anddata))
                except Exception:
                    rust_build_count = None
            dt["rust_featurizer_build_count"] = rust_build_count

            datasets[dataset_name] = {
                "anddata": anddata,
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
                "nameless_X_train": nameless_X_train,
                "nameless_X_val": nameless_X_val,
                "nameless_X_test": nameless_X_test,
                "name": dataset_name,
            }
            per_dataset_timings[dataset_name] = dt

        stage_timings["per_dataset"] = per_dataset_timings
        _snapshot_stage_rss(stage_rss_gb, monitor, "per_dataset_complete")
        stage_timings["rust_cleanup_boundary_enabled"] = bool(backend == "rust" and rust_cleanup_boundary)
        if backend == "rust" and rust_cleanup_boundary:
            try:
                from s2and import feature_port as _feature_port

                evicted = int(_feature_port.clear_rust_featurizer_cache())
                print(f"  [{run_label}] Cleared Rust featurizer cache entries: {evicted}")
                stage_timings["rust_cleanup_evicted_entries"] = evicted
            except Exception as exc:
                print(f"  [{run_label}] Rust featurizer cleanup skipped: {exc!r}")
                stage_timings["rust_cleanup_evicted_entries"] = None
            gc_collected = int(gc.collect())
            stage_timings["rust_cleanup_gc_collected"] = gc_collected
            _snapshot_stage_rss(stage_rss_gb, monitor, "post_rust_cleanup")
            print(
                f"  [{run_label}] Post-Rust cleanup RSS snapshot: " f"{stage_rss_gb.get('post_rust_cleanup', 'n/a')} GB"
            )

        # ----- union pairwise model -----
        print(f"  [{run_label}] Training union pairwise model...")
        t0 = time.perf_counter()
        X_train_union = np.vstack([datasets[d]["X_train"] for d in dataset_names])
        y_train_union = np.hstack([datasets[d]["y_train"] for d in dataset_names])
        X_val_union = np.vstack([datasets[d]["X_val"] for d in dataset_names if d not in {"augmented"}])
        y_val_union = np.hstack([datasets[d]["y_val"] for d in dataset_names if d not in {"augmented"}])
        nameless_X_train_union = np.vstack([datasets[d]["nameless_X_train"] for d in dataset_names])
        nameless_X_val_union = np.vstack(
            [datasets[d]["nameless_X_val"] for d in dataset_names if d not in {"augmented"}]
        )

        union_classifier = PairwiseModeler(
            n_iter=n_iter,
            monotone_constraints=MONOTONE_CONSTRAINTS,
            random_state=random_seed,
        )
        union_classifier.fit(X_train_union, y_train_union, X_val_union, y_val_union)
        stage_timings["union_pairwise_fit_seconds"] = round(time.perf_counter() - t0, 3)
        stage_timings["union_pairwise_hyperopt"] = _summarize_hyperopt_trials(union_classifier.hyperopt_trials_store)
        stage_timings["union_pairwise_lgbm"] = _lgbm_fit_summary(union_classifier)
        _snapshot_stage_rss(stage_rss_gb, monitor, "union_pairwise_fit")
        print(f"  [{run_label}] Union pairwise fit in {_fmt(stage_timings['union_pairwise_fit_seconds'])}s")

        t0 = time.perf_counter()
        nameless_union_classifier = PairwiseModeler(
            n_iter=n_iter,
            monotone_constraints=NAMELESS_MONOTONE_CONSTRAINTS,
            random_state=random_seed,
        )
        nameless_union_classifier.fit(nameless_X_train_union, y_train_union, nameless_X_val_union, y_val_union)
        stage_timings["union_nameless_pairwise_fit_seconds"] = round(time.perf_counter() - t0, 3)
        stage_timings["union_nameless_pairwise_hyperopt"] = _summarize_hyperopt_trials(
            nameless_union_classifier.hyperopt_trials_store
        )
        stage_timings["union_nameless_pairwise_lgbm"] = _lgbm_fit_summary(nameless_union_classifier)
        _snapshot_stage_rss(stage_rss_gb, monitor, "union_nameless_pairwise_fit")
        print(
            f"  [{run_label}] Union nameless pairwise fit in "
            f"{_fmt(stage_timings['union_nameless_pairwise_fit_seconds'])}s"
        )

        # ----- union clusterer.fit -----
        # This is the critical Rust path: make_distance_matrices for each dataset's val blocks,
        # then 5 hyperopt iterations (cheap, reuses cached dists).
        print(f"  [{run_label}] Fitting union clusterer (n_iter={n_iter})...")
        anddatas = [datasets[d]["anddata"] for d in dataset_names if d not in PAIRWISE_ONLY_DATASETS]
        if len(anddatas) == 0:
            raise ValueError(
                "No non-pairwise datasets available for union clusterer.fit. "
                "At least one dataset must not be in PAIRWISE_ONLY_DATASETS."
            )
        cluster_search_space: dict[str, Any] = {"eps": hp.uniform("choice", 0, 1)}

        union_clusterer = Clusterer(
            FEATURIZER_INFO,
            union_classifier.classifier,
            cluster_model=FastCluster(linkage="average"),
            search_space=cluster_search_space,
            n_jobs=n_jobs,
            use_cache=False,
            nameless_classifier=nameless_union_classifier.classifier,
            nameless_featurizer_info=NAMELESS_FEATURIZER_INFO,
            random_state=random_seed,
            use_default_constraints_as_supervision=True,
            n_iter=n_iter,
        )

        distance_matrix_seconds = 0.0
        distance_matrix_calls = 0
        original_make_distance_matrices = union_clusterer.make_distance_matrices

        def _timed_make_distance_matrices(*args: Any, **kwargs: Any):
            nonlocal distance_matrix_seconds, distance_matrix_calls
            _t = time.perf_counter()
            result = original_make_distance_matrices(*args, **kwargs)
            distance_matrix_seconds += time.perf_counter() - _t
            distance_matrix_calls += 1
            return result

        union_clusterer.make_distance_matrices = _timed_make_distance_matrices  # type: ignore[method-assign]
        t0 = time.perf_counter()
        try:
            union_clusterer.fit(anddatas)
        finally:
            union_clusterer.make_distance_matrices = original_make_distance_matrices  # type: ignore[method-assign]
        union_clusterer_fit_seconds = time.perf_counter() - t0
        stage_timings["union_clusterer_fit_seconds"] = round(union_clusterer_fit_seconds, 3)
        stage_timings["union_distance_matrix_seconds"] = round(distance_matrix_seconds, 3)
        stage_timings["union_hyperopt_seconds"] = round(
            max(0.0, union_clusterer_fit_seconds - distance_matrix_seconds),
            3,
        )
        stage_timings["union_distance_matrix_calls"] = int(distance_matrix_calls)
        stage_timings["union_clusterer_hyperopt"] = _summarize_hyperopt_trials(union_clusterer.hyperopt_trials_store)
        _snapshot_stage_rss(stage_rss_gb, monitor, "union_clusterer_fit")
        print(f"  [{run_label}] Union clusterer fit in {_fmt(stage_timings['union_clusterer_fit_seconds'])}s")
        stage_timings["union_clusterer_best_params"] = union_clusterer.best_params

        # ----- cluster_eval on target -----
        # This is the fused predict path: featurize test blocks from scratch + score.
        if target_name in PAIRWISE_ONLY_DATASETS:
            raise ValueError(
                f"target={target_name!r} is pairwise-only. Choose a clustered target dataset for cluster_eval."
            )
        target_anddata = datasets[target_name]["anddata"]
        print(f"  [{run_label}] Running cluster_eval on {target_name} test split...")

        import s2and.model as s2and_model_module

        original_predict = union_clusterer.predict
        original_many_pairs_featurize = s2and_model_module.many_pairs_featurize

        model_predict_seconds = 0.0
        model_predict_calls = 0
        pair_featurize_seconds = 0.0
        pair_featurize_calls = 0

        def _timed_predict(*args: Any, **kwargs: Any):
            nonlocal model_predict_seconds, model_predict_calls
            _t = time.perf_counter()
            result = original_predict(*args, **kwargs)
            model_predict_seconds += time.perf_counter() - _t
            model_predict_calls += 1
            return result

        def _timed_many_pairs_featurize(*args: Any, **kwargs: Any):
            nonlocal pair_featurize_seconds, pair_featurize_calls
            _t = time.perf_counter()
            result = original_many_pairs_featurize(*args, **kwargs)
            pair_featurize_seconds += time.perf_counter() - _t
            pair_featurize_calls += 1
            return result

        union_clusterer.predict = _timed_predict  # type: ignore[method-assign]
        s2and_model_module.many_pairs_featurize = cast(Any, _timed_many_pairs_featurize)
        t0 = time.perf_counter()
        try:
            cluster_metrics, _ = cluster_eval(target_anddata, union_clusterer, split="test")
        finally:
            union_clusterer.predict = original_predict  # type: ignore[method-assign]
            s2and_model_module.many_pairs_featurize = original_many_pairs_featurize
        cluster_eval_seconds = time.perf_counter() - t0
        stage_timings["cluster_eval_seconds"] = round(cluster_eval_seconds, 3)
        stage_timings["cluster_eval_model_predict_seconds"] = round(model_predict_seconds, 3)
        stage_timings["cluster_eval_model_predict_calls"] = int(model_predict_calls)
        stage_timings["cluster_eval_pair_featurize_seconds"] = round(pair_featurize_seconds, 3)
        stage_timings["cluster_eval_pair_featurize_calls"] = int(pair_featurize_calls)
        _snapshot_stage_rss(stage_rss_gb, monitor, "cluster_eval")
        print(f"  [{run_label}] cluster_eval in {_fmt(stage_timings['cluster_eval_seconds'])}s")

    total_seconds = time.perf_counter() - total_start

    def _triplet(key: str):
        v = cluster_metrics.get(key, (float("nan"),) * 3)
        if isinstance(v, list | tuple) and len(v) >= 3:
            return [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]
        return [float("nan")] * 3

    return {
        "run_label": run_label,
        "backend": backend,
        # Stage-level override knobs were removed; backend is now uniform.
        "constraints_backend": backend if backend == "rust" else "",
        "pair_featurization_backend": backend if backend == "rust" else "",
        "datasets": dataset_names,
        "target": target_name,
        "n_jobs": n_jobs,
        "n_train_pairs": n_train_pairs,
        "train_pairs_size_mode": train_pairs_size_mode,
        "n_iter": n_iter,
        "random_seed": random_seed,
        "workload": workload,
        "workload_id": workload_id,
        "total_seconds": round(total_seconds, 3),
        "peak_rss_gb": round(monitor.peak_gb, 3),
        "b3": _triplet("B3 (P, R, F1)"),
        "cluster": _triplet("Cluster (P, R F1)"),
        "cluster_macro": _triplet("Cluster Macro (P, R, F1)"),
        "stage_timings": stage_timings,
        "stage_rss_gb": stage_rss_gb,
        "diagnostics": {
            "rust_cleanup_boundary": bool(rust_cleanup_boundary),
            "force_python_paper_preprocess": bool(force_python_paper_preprocess),
        },
        "rust_extension_identity": rust_extension_identity,
        "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
    }


# ---------------------------------------------------------------------------
# Subprocess wrapper for process isolation
# ---------------------------------------------------------------------------
def _run_subprocess(
    *,
    backend: str,
    datasets: list[str],
    target: str,
    n_jobs: int,
    n_train_pairs: int,
    n_iter: int,
    random_seed: int,
    train_pairs_size_mode: str,
    require_rust_release: int,
    run_label: str = "",
    rust_cleanup_boundary: int = 1,
    force_python_paper_preprocess: int = 0,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "single",
        "--backend",
        backend,
        "--datasets",
        *datasets,
        "--target",
        target,
        "--n-jobs",
        str(n_jobs),
        "--n-train-pairs",
        str(n_train_pairs),
        "--train-pairs-size-mode",
        train_pairs_size_mode,
        "--n-iter",
        str(n_iter),
        "--random-seed",
        str(random_seed),
        "--require-rust-release",
        str(int(require_rust_release)),
        "--rust-cleanup-boundary",
        str(int(rust_cleanup_boundary)),
        "--force-python-paper-preprocess",
        str(int(force_python_paper_preprocess)),
        "--run-label",
        run_label or backend,
    ]

    print(f"\n{'='*60}")
    print(f"Starting subprocess: {run_label or backend}")
    print(f"{'='*60}")
    env = dict(os.environ)
    # Improve log/progress visibility for long-running child processes.
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    assert process.stdout is not None
    output_lines: list[str] = []
    for line in process.stdout:
        output_lines.append(line)
        print(line, end="")
    process.wait()
    output_text = "".join(output_lines)
    if process.returncode != 0:
        print(f"[{run_label}] FAILED with return code {process.returncode}")
        print(f"[{run_label}] output tail:\n{output_text[-3000:]}")
        raise RuntimeError(f"Subprocess {run_label} failed")
    return extract_marked_json_payload(
        output_text,
        RESULT_JSON_START,
        RESULT_JSON_END,
        error_tail_chars=2000,
    )


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------
def _compare(args: argparse.Namespace, workload: dict[str, Any], workload_id: str) -> None:
    common = dict(workload)

    configs = [
        {"backend": "python", "run_label": "python"},
        {"backend": "rust", "run_label": "rust"},
    ]

    results = []
    for config in configs:
        result = _run_subprocess(
            backend=config["backend"],
            datasets=common["datasets"],
            target=common["target"],
            n_jobs=common["n_jobs"],
            n_train_pairs=common["n_train_pairs"],
            n_iter=common["n_iter"],
            random_seed=common["random_seed"],
            train_pairs_size_mode=common["train_pairs_size_mode"],
            require_rust_release=int(args.require_rust_release),
            rust_cleanup_boundary=int(args.rust_cleanup_boundary),
            force_python_paper_preprocess=int(args.force_python_paper_preprocess),
            run_label=config["run_label"],
        )
        results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Config':<25} {'Total(s)':>10} {'RSS(GB)':>10} {'B3 F1':>8} {'Cl F1':>8} {'ClM F1':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        b3_f1 = r["b3"][2]
        cl_f1 = r["cluster"][2]
        clm_f1 = r["cluster_macro"][2]
        print(
            f"{r['run_label']:<25} {r['total_seconds']:>10.1f} {r['peak_rss_gb']:>10.3f} "
            f"{b3_f1:>8.4f} {cl_f1:>8.4f} {clm_f1:>8.4f}"
        )
    print()

    # Stage breakdown
    print("Stage breakdown (seconds):")
    stages = [
        "name_counts_load_seconds",
        "union_pairwise_fit_seconds",
        "union_nameless_pairwise_fit_seconds",
        "union_clusterer_fit_seconds",
        "union_distance_matrix_seconds",
        "union_hyperopt_seconds",
        "cluster_eval_model_predict_seconds",
        "cluster_eval_pair_featurize_seconds",
        "cluster_eval_seconds",
    ]
    header2 = f"{'Stage':<40}" + "".join(f"{r['run_label']:>20}" for r in results)
    print(header2)
    print("-" * len(header2))
    for stage in stages:
        vals = [str(r["stage_timings"].get(stage, "n/a")) for r in results]
        print(f"{stage:<40}" + "".join(f"{v:>20}" for v in vals))

    print()
    print("Stage RSS snapshots (GB):")
    rss_stages = [
        "name_counts_load",
        "per_dataset_complete",
        "post_rust_cleanup",
        "union_pairwise_fit",
        "union_nameless_pairwise_fit",
        "union_clusterer_fit",
        "cluster_eval",
    ]
    header3 = f"{'Stage':<40}" + "".join(f"{r['run_label']:>20}" for r in results)
    print(header3)
    print("-" * len(header3))
    for stage in rss_stages:
        vals = [str(r.get("stage_rss_gb", {}).get(stage, "n/a")) for r in results]
        print(f"{stage:<40}" + "".join(f"{v:>20}" for v in vals))

    # Per-dataset breakdown
    print()
    print("Per-dataset breakdown (seconds):")
    for ds in common["datasets"]:
        print(f"  {ds}:")
        for r in results:
            dt = r["stage_timings"].get("per_dataset", {}).get(ds, {})
            build = dt.get("anddata_build_seconds", "?")
            feat = dt.get("featurize_seconds", "?")
            n_tr = dt.get("n_train_pairs", "?")
            train_pairs_size = dt.get("anddata_train_pairs_size", "?")
            rust_build_count = dt.get("rust_featurizer_build_count", "?")
            print(
                f"    {r['run_label']:<22} build={build}  featurize={feat}  "
                f"train_pairs={n_tr}  anddata_train_pairs_size={train_pairs_size}  "
                f"rust_featurizer_build_count={rust_build_count}"
            )

    if args.write_json:
        out_path = Path(args.write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "preset": args.preset,
                    "workload": workload,
                    "workload_id": workload_id,
                    "results": results,
                    "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"\nJSON written to: {out_path}")


def _extract_run_result(payload: dict[str, Any], run_label: str) -> dict[str, Any]:
    maybe_results = payload.get("results")
    if isinstance(maybe_results, list):
        for item in maybe_results:
            if isinstance(item, dict) and str(item.get("run_label", "")) == run_label:
                return item
        if len(maybe_results) > 0 and isinstance(maybe_results[0], dict):
            return maybe_results[0]
    if "backend" in payload and "total_seconds" in payload:
        return payload
    raise RuntimeError("Could not find a benchmark run result in JSON payload")


def _numeric_stage_timings(stage_timings: Any) -> dict[str, float]:
    if not isinstance(stage_timings, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in stage_timings.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            out[str(key)] = float(value)
    return out


def _optional_fraction_delta(current: float, baseline: float) -> float | None:
    if baseline <= 0:
        return None
    return (current - baseline) / baseline


def _gate(args: argparse.Namespace) -> None:
    baseline_path = Path(args.baseline_json)
    current_path = Path(args.current_json)
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    current_payload = json.loads(current_path.read_text(encoding="utf-8"))

    baseline_run = _extract_run_result(baseline_payload, str(args.gate_run_label))
    current_run = _extract_run_result(current_payload, str(args.gate_run_label))

    baseline_workload_id = str(baseline_run.get("workload_id") or baseline_payload.get("workload_id") or "")
    current_workload_id = str(current_run.get("workload_id") or current_payload.get("workload_id") or "")
    if baseline_workload_id != current_workload_id:
        raise RuntimeError(
            "Workload mismatch between baseline and current artifacts. "
            f"baseline_workload_id={baseline_workload_id!r} current_workload_id={current_workload_id!r}"
        )

    baseline_runtime = float(baseline_run["total_seconds"])
    current_runtime = float(current_run["total_seconds"])
    baseline_peak_rss = float(baseline_run["peak_rss_gb"])
    current_peak_rss = float(current_run["peak_rss_gb"])
    baseline_b3_f1 = float(baseline_run["b3"][2])
    current_b3_f1 = float(current_run["b3"][2])

    runtime_delta_fraction = _optional_fraction_delta(current_runtime, baseline_runtime)
    peak_rss_delta_fraction = _optional_fraction_delta(current_peak_rss, baseline_peak_rss)
    b3_f1_drop = baseline_b3_f1 - current_b3_f1

    baseline_stage_timings = _numeric_stage_timings(baseline_run.get("stage_timings", {}))
    current_stage_timings = _numeric_stage_timings(current_run.get("stage_timings", {}))
    min_stage_seconds_for_gate = float(args.min_stage_seconds_for_gate)
    shared_stages = sorted(set(baseline_stage_timings).intersection(current_stage_timings))
    stage_deltas: dict[str, dict[str, float | None]] = {}
    skipped_stage_regression_checks: list[str] = []
    violations: list[str] = []
    for stage in shared_stages:
        baseline_value = baseline_stage_timings[stage]
        current_value = current_stage_timings[stage]
        delta_value = current_value - baseline_value
        delta_fraction = _optional_fraction_delta(current_value, baseline_value)
        stage_regression_check_skipped = baseline_value < min_stage_seconds_for_gate
        stage_deltas[stage] = {
            "baseline_seconds": baseline_value,
            "current_seconds": current_value,
            "delta_seconds": delta_value,
            "delta_fraction": delta_fraction,
            "regression_check_skipped": stage_regression_check_skipped,
        }
        if stage_regression_check_skipped:
            skipped_stage_regression_checks.append(stage)
            continue
        if delta_fraction is not None and delta_fraction > float(args.max_stage_regression_fraction):
            violations.append(
                f"stage {stage!r} regression {delta_fraction:.6f} exceeds "
                f"{float(args.max_stage_regression_fraction):.6f}"
            )

    if runtime_delta_fraction is not None and runtime_delta_fraction > float(args.max_runtime_regression_fraction):
        violations.append(
            f"runtime regression {runtime_delta_fraction:.6f} exceeds "
            f"{float(args.max_runtime_regression_fraction):.6f}"
        )
    if peak_rss_delta_fraction is not None and peak_rss_delta_fraction > float(args.max_peak_rss_regression_fraction):
        violations.append(
            f"peak RSS regression {peak_rss_delta_fraction:.6f} exceeds "
            f"{float(args.max_peak_rss_regression_fraction):.6f}"
        )
    if b3_f1_drop > float(args.max_b3_f1_drop):
        violations.append(f"B3 F1 drop {b3_f1_drop:.6f} exceeds {float(args.max_b3_f1_drop):.6f}")

    report: dict[str, Any] = {
        "mode": "gate",
        "run_label": str(args.gate_run_label),
        "baseline_json": str(baseline_path),
        "current_json": str(current_path),
        "workload": current_run.get("workload") or current_payload.get("workload"),
        "workload_id": current_workload_id,
        "baseline": {
            "total_seconds": baseline_runtime,
            "peak_rss_gb": baseline_peak_rss,
            "b3_f1": baseline_b3_f1,
        },
        "current": {
            "total_seconds": current_runtime,
            "peak_rss_gb": current_peak_rss,
            "b3_f1": current_b3_f1,
        },
        "delta": {
            "runtime_seconds": current_runtime - baseline_runtime,
            "runtime_fraction": runtime_delta_fraction,
            "peak_rss_gb": current_peak_rss - baseline_peak_rss,
            "peak_rss_fraction": peak_rss_delta_fraction,
            "b3_f1_delta": current_b3_f1 - baseline_b3_f1,
            "b3_f1_drop": b3_f1_drop,
        },
        "stage_deltas": stage_deltas,
        "thresholds": {
            "max_runtime_regression_fraction": float(args.max_runtime_regression_fraction),
            "max_peak_rss_regression_fraction": float(args.max_peak_rss_regression_fraction),
            "max_b3_f1_drop": float(args.max_b3_f1_drop),
            "max_stage_regression_fraction": float(args.max_stage_regression_fraction),
            "min_stage_seconds_for_gate": min_stage_seconds_for_gate,
        },
        "skipped_stage_regression_checks": skipped_stage_regression_checks,
        "violations": violations,
        "pass": len(violations) == 0,
        "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
    }

    print("Gate summary:")
    print(f"1. workload_id: {current_workload_id}")
    print(f"2. runtime delta fraction: {runtime_delta_fraction}")
    print(f"3. peak RSS delta fraction: {peak_rss_delta_fraction}")
    print(f"4. B3 F1 drop: {b3_f1_drop}")
    print(
        f"5. skipped stage checks (<{min_stage_seconds_for_gate:.3f}s baseline): {len(skipped_stage_regression_checks)}"
    )
    print(f"6. violations: {len(violations)}")
    for violation in violations:
        print(f"   - {violation}")

    if args.write_json:
        out_path = Path(args.write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Gate JSON written to: {out_path}")

    if violations:
        raise RuntimeError("Gate checks failed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Mini transfer experiment profiler for Rust A/B testing.")
    parser.add_argument("--mode", choices=["compare", "single", "gate"], default="compare")
    parser.add_argument("--preset", choices=sorted(WORKLOAD_PRESETS), default="smoke")
    parser.add_argument("--backend", choices=["python", "rust"], default="rust")
    parser.add_argument("--datasets", nargs="+", default=None, help="Override workload preset datasets.")
    parser.add_argument("--target", default=None, help="Override workload preset target dataset.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override workload preset n_jobs.")
    parser.add_argument("--n-train-pairs", type=int, default=None, help="Override workload preset n_train_pairs.")
    parser.add_argument(
        "--train-pairs-size-mode",
        choices=["scaled", "exact_internal"],
        default=None,
        help=(
            "How to set ANDData train_pairs_size. "
            "'scaled' uses --n-train-pairs directly; "
            "'exact_internal' matches transfer_experiment_internal.py (max(n_train_pairs, 100000))."
        ),
    )
    parser.add_argument("--n-iter", type=int, default=None, help="Override workload preset hyperopt iterations.")
    parser.add_argument("--random-seed", type=int, default=None, help="Override workload preset random seed.")
    parser.add_argument("--require-rust-release", type=int, choices=[0, 1], default=0)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--rust-cleanup-boundary", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--force-python-paper-preprocess",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Diagnostic override: when set to 1, force Python paper preprocessing by "
            f"setting {RUST_FORCE_PYTHON_PAPER_PREPROCESS_ENV}=1 in the child process."
        ),
    )
    parser.add_argument("--write-json", default="", help="Output JSON path (compare or single mode).")
    parser.add_argument("--baseline-json", default="", help="Required for --mode gate.")
    parser.add_argument("--current-json", default="", help="Required for --mode gate.")
    parser.add_argument("--gate-run-label", default="rust", help="Result run_label to compare in gate mode.")
    parser.add_argument("--max-runtime-regression-fraction", type=float, default=0.05)
    parser.add_argument("--max-peak-rss-regression-fraction", type=float, default=0.05)
    parser.add_argument("--max-b3-f1-drop", type=float, default=0.001)
    parser.add_argument("--max-stage-regression-fraction", type=float, default=0.1)
    parser.add_argument(
        "--min-stage-seconds-for-gate",
        type=float,
        default=10.0,
        help=(
            "Skip per-stage regression checks when baseline stage time is below this threshold. "
            "Overall runtime/RSS/B3 gates still apply."
        ),
    )
    args = parser.parse_args()

    if args.mode == "gate":
        if not args.baseline_json or not args.current_json:
            raise ValueError("--baseline-json and --current-json are required for --mode gate")
        _gate(args)
        return

    workload, workload_id = _resolve_workload(args)
    if workload["target"] not in workload["datasets"]:
        raise ValueError(
            f"--target must be included in --datasets. got target={workload['target']!r} "
            f"datasets={workload['datasets']!r}"
        )

    if args.mode == "single":
        result = _single_run(
            backend=args.backend,
            dataset_names=workload["datasets"],
            target_name=workload["target"],
            n_jobs=workload["n_jobs"],
            n_train_pairs=workload["n_train_pairs"],
            train_pairs_size_mode=workload["train_pairs_size_mode"],
            n_iter=workload["n_iter"],
            random_seed=workload["random_seed"],
            run_label=args.run_label or args.backend,
            workload=workload,
            workload_id=workload_id,
            require_rust_release=bool(args.require_rust_release),
            rust_cleanup_boundary=bool(args.rust_cleanup_boundary),
            force_python_paper_preprocess=bool(args.force_python_paper_preprocess),
        )
        if args.write_json:
            out_path = Path(args.write_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "preset": args.preset,
                        "workload": workload,
                        "workload_id": workload_id,
                        "results": [result],
                        "run_metadata": build_run_metadata(script_path=Path(__file__).resolve()),
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            print(f"\nJSON written to: {out_path}")
        print(RESULT_JSON_START)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(RESULT_JSON_END)
    else:
        _compare(args, workload, workload_id)


if __name__ == "__main__":
    main()

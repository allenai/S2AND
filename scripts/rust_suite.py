from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INTERNAL_DIR = Path(__file__).resolve().parent / "_rust_suite"

RESULT_JSON_START = "===S2AND_PROFILE_RESULT_START==="
RESULT_JSON_END = "===S2AND_PROFILE_RESULT_END==="

_MODULE_FILES = {
    "compare": "compare_cmd",
    "transfer_mini": "transfer_mini_cmd",
    "prod_inference": "prod_inference_cmd",
    "largest_block": "largest_block_cmd",
    "big_block_incremental": "big_block_incremental_cmd",
    "featurizer_reuse": "featurizer_reuse_cmd",
    "stress_rebuild": "stress_rebuild_cmd",
    "calibrate_phase_a": "calibrate_phase_a_cmd",
    "calibrate_rust_batch": "calibrate_rust_batch_cmd",
    "measure_counter_data": "measure_counter_data_cmd",
}

_MODULE_CACHE: dict[str, ModuleType] = {}
_ACTIVE_CANONICAL_ARGV: list[str] | None = None
_ACTIVE_CANONICAL_COMMAND: str | None = None


def _run_git_command(project_root: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _build_run_metadata() -> dict[str, Any]:
    env_keys = (
        "S2AND_BACKEND",
        "S2AND_SKIP_FASTTEXT",
        "S2AND_RUST_FEATURIZER_MAX_INMEM",
        "S2AND_RUST_BATCH_RSS_SAMPLER_MS",
        "S2AND_NORMALIZATION_VERSION",
        "S2AND_RUST_NAME_COUNTS_JSON",
        "PYTHONHASHSEED",
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
    )
    env_snapshot = {key: os.environ[key] for key in env_keys if key in os.environ}
    git_status = _run_git_command(PROJECT_ROOT, ["status", "--porcelain"])
    canonical_argv = _ACTIVE_CANONICAL_ARGV if _ACTIVE_CANONICAL_ARGV is not None else list(sys.argv)
    return {
        "generated_at_utc": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat(),
        "script": str(Path(__file__).resolve()),
        "argv": list(canonical_argv),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "git_commit": _run_git_command(PROJECT_ROOT, ["rev-parse", "HEAD"]),
        "git_branch": _run_git_command(PROJECT_ROOT, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": (None if git_status is None else bool(git_status)),
        "env": env_snapshot,
    }


class ProcessTreeRSSMonitor:
    """Monitor peak RSS across current process and all child workers."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self.peak_rss_bytes = 0

    def _tree_rss_bytes(self) -> int:
        rss_total = 0
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.Error):
            pass
        for proc in processes:
            try:
                rss_total += int(proc.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return rss_total

    def sample_rss_bytes(self) -> int:
        rss = self._tree_rss_bytes()
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        return rss

    def sample_gb(self) -> float:
        return self.sample_rss_bytes() / (1024**3)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_rss_bytes()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self.peak_rss_bytes = self.sample_rss_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> ProcessTreeRSSMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def peak_gb(self) -> float:
        return self.peak_rss_bytes / (1024**3)


class RSSMonitor:
    """Monitor peak RSS for current process only."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self.peak_rss_bytes = 0

    def _rss_bytes(self) -> int:
        return int(self._process.memory_info().rss)

    def sample_rss_bytes(self) -> int:
        rss = self._rss_bytes()
        if rss > self.peak_rss_bytes:
            self.peak_rss_bytes = rss
        return rss

    def sample_gb(self) -> float:
        return self.sample_rss_bytes() / (1024**3)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_rss_bytes()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self.peak_rss_bytes = self.sample_rss_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __enter__(self) -> RSSMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def peak_gb(self) -> float:
        return self.peak_rss_bytes / (1024**3)


def _patch_internal_module(module_key: str, module: ModuleType) -> None:
    if hasattr(module, "_build_run_metadata"):
        module._build_run_metadata = _build_run_metadata
    if hasattr(module, "ProcessTreeRSSMonitor"):
        module.ProcessTreeRSSMonitor = ProcessTreeRSSMonitor
    if hasattr(module, "RSSMonitor"):
        module.RSSMonitor = RSSMonitor

    if module_key == "transfer_mini":

        def _resolve_data_dir_from_repo_root() -> str:
            config_path = PROJECT_ROOT / "data" / "path_config.json"
            with config_path.open("r", encoding="utf-8") as infile:
                config = json.load(infile)
            internal = str(config.get("internal_data_dir", "")).strip()
            if internal and Path(internal).exists():
                return internal
            return str(PROJECT_ROOT / "data")

        module._resolve_data_dir = _resolve_data_dir_from_repo_root

    if module_key == "largest_block":
        module.PROJECT_ROOT = PROJECT_ROOT
        module.DATA_DIR = PROJECT_ROOT / "data"
        module.DEFAULT_MODEL_PATH = str((PROJECT_ROOT / "data" / "production_model_v1.1.pickle").resolve())


def _load_internal_module(module_key: str) -> ModuleType:
    cached = _MODULE_CACHE.get(module_key)
    if cached is not None:
        return cached

    file_stem = _MODULE_FILES[module_key]
    module_path = _INTERNAL_DIR / f"{file_stem}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing internal rust-suite module: {module_path}")

    spec = importlib.util.spec_from_file_location(f"rust_suite_internal_{file_stem}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load internal module spec: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _patch_internal_module(module_key, module)
    _MODULE_CACHE[module_key] = module
    return module


def _extract_marked_json_payload(stdout_text: str, start_marker: str, end_marker: str) -> dict[str, Any]:
    start = stdout_text.find(start_marker)
    end = stdout_text.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Failed to parse result JSON markers from subprocess output")
    payload_text = stdout_text[start + len(start_marker) : end].strip()
    return json.loads(payload_text)


def _run_marked_subprocess(cmd: list[str], start_marker: str, end_marker: str) -> dict[str, Any]:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return _extract_marked_json_payload(completed.stdout, start_marker, end_marker)


# ---------------------------------------------------------------------------
# Helper exports used by tests
# ---------------------------------------------------------------------------


def _language_feature_indices(feature_names: list[str]) -> list[int]:
    return _load_internal_module("compare")._language_feature_indices(feature_names)


def _compute_feature_parity(
    python_features,
    rust_features,
    feature_names: list[str],
    *,
    non_language_rtol: float,
    non_language_atol: float,
    language_max_mismatch_fraction: float,
):
    return _load_internal_module("compare")._compute_feature_parity(
        python_features,
        rust_features,
        feature_names,
        non_language_rtol=non_language_rtol,
        non_language_atol=non_language_atol,
        language_max_mismatch_fraction=language_max_mismatch_fraction,
    )


def _load_dataset_inputs(
    dataset: str,
    limit: int | None,
    project_root: str,
    *,
    force_paths: bool = False,
):
    return _load_internal_module("compare")._load_dataset_inputs(
        dataset,
        limit,
        project_root,
        force_paths=force_paths,
    )


def _effective_train_pairs_size(n_train_pairs: int, mode: str) -> int:
    return _load_internal_module("transfer_mini")._effective_train_pairs_size(n_train_pairs, mode)


def _resolve_dataset_file(
    data_dir: str,
    dataset_name: str,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    return _load_internal_module("transfer_mini")._resolve_dataset_file(
        data_dir,
        dataset_name,
        candidates,
        required=required,
    )


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
    return _load_internal_module("transfer_mini")._build_anddata_kwargs(
        data_dir=data_dir,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        random_seed=random_seed,
        n_train_pairs=n_train_pairs,
        n_val_test_size=n_val_test_size,
        name_counts=name_counts,
        train_pairs_size_mode=train_pairs_size_mode,
    )


def _build_data_paths(project_root: str, dataset_name: str, data_root: str, specter_file: str) -> dict[str, str]:
    return _load_internal_module("prod_inference")._build_data_paths(
        project_root,
        dataset_name,
        data_root,
        specter_file,
    )


def _single_run(
    backend: str,
    dataset_name: str,
    n_jobs: int,
    profile_output_path: str,
    model_path: str = os.path.join("data", "production_model_v1.1.pickle"),
    data_root: str = os.path.join("data", "s2and_mini"),
    specter_file: str = "",
    rust_warm_featurizer_before_predict: int = 0,
    run_label: str | None = None,
) -> dict[str, Any]:
    return _load_internal_module("prod_inference")._single_run(
        backend=backend,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        profile_output_path=profile_output_path,
        model_path=model_path,
        data_root=data_root,
        specter_file=specter_file,
        rust_warm_featurizer_before_predict=rust_warm_featurizer_before_predict,
        run_label=run_label,
    )


def _run_single_subprocess(
    script_path: Path,
    backend: str,
    dataset_name: str,
    n_jobs: int,
    profile_output_path: str,
    model_path: str = os.path.join("data", "production_model_v1.1.pickle"),
    data_root: str = os.path.join("data", "s2and_mini"),
    specter_file: str = "",
    rust_warm_featurizer_before_predict: int = 0,
    single_write_json: str = "",
    run_label: str = "",
) -> dict[str, Any]:
    script_path_resolved = Path(script_path)
    if script_path_resolved.name == Path(__file__).name:
        cmd = [
            sys.executable,
            str(script_path_resolved),
            "prod-inference",
            "--mode",
            "single",
            "--backend",
            backend,
            "--dataset-name",
            dataset_name,
            "--n-jobs",
            str(n_jobs),
            "--profile-output-path",
            profile_output_path,
            "--model-path",
            model_path,
            "--data-root",
            data_root,
        ]
        if specter_file:
            cmd.extend(["--specter-file", specter_file])
        if rust_warm_featurizer_before_predict in {0, 1}:
            cmd.extend(["--rust-warm-featurizer-before-predict", str(int(rust_warm_featurizer_before_predict))])
        if single_write_json:
            cmd.extend(["--single-write-json", single_write_json])
        if run_label:
            cmd.extend(["--run-label", run_label])
        return _run_marked_subprocess(cmd, RESULT_JSON_START, RESULT_JSON_END)

    return _load_internal_module("prod_inference")._run_single_subprocess(
        script_path=script_path,
        backend=backend,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        profile_output_path=profile_output_path,
        model_path=model_path,
        data_root=data_root,
        specter_file=specter_file,
        rust_warm_featurizer_before_predict=rust_warm_featurizer_before_predict,
        single_write_json=single_write_json,
        run_label=run_label,
    )


def _cluster_membership_digest(cluster_to_signatures: dict[str, list[str]]) -> str:
    return _load_internal_module("largest_block")._cluster_membership_digest(cluster_to_signatures)


def _signature_to_cluster_fingerprint_map(cluster_to_signatures: dict[str, list[str]]) -> dict[str, str]:
    return _load_internal_module("largest_block")._signature_to_cluster_fingerprint_map(cluster_to_signatures)


def _pairwise_precision_recall_fscore_with_singleton_fix(
    true_clusters: dict[str, list[str]],
    pred_clusters: dict[str, list[str]],
) -> tuple[float, float, float]:
    return _load_internal_module("largest_block")._pairwise_precision_recall_fscore_with_singleton_fix(
        true_clusters,
        pred_clusters,
    )


def _effective_seed_cluster_count(seed_signature_count: int, requested_seed_clusters: int) -> int:
    return _load_internal_module("big_block_incremental")._effective_seed_cluster_count(
        seed_signature_count,
        requested_seed_clusters,
    )


def _build_cluster_seeds(seed_signature_ids: list[str], seed_cluster_count: int) -> dict[str, dict[str, str]]:
    return _load_internal_module("big_block_incremental")._build_cluster_seeds(seed_signature_ids, seed_cluster_count)


def _paper_has_block_safe_author_names(paper_payload: dict[str, Any]) -> bool:
    return _load_internal_module("big_block_incremental")._paper_has_block_safe_author_names(paper_payload)


def _validate_args(args: argparse.Namespace) -> None:
    return _load_internal_module("big_block_incremental")._validate_args(args)


def run_rebuild_stress(**kwargs: Any) -> dict[str, Any]:
    return _load_internal_module("stress_rebuild").run_rebuild_stress(**kwargs)


_COMMANDS = {
    "compare": {
        "module": "compare",
        "help": "Python vs Rust feature/runtime parity workflow.",
        "main_kind": "noargv",
    },
    "transfer-mini": {
        "module": "transfer_mini",
        "help": "Mini transfer benchmark workflow.",
        "main_kind": "noargv",
    },
    "prod-inference": {
        "module": "prod_inference",
        "help": "Production-model inference profiling workflow.",
        "main_kind": "noargv",
    },
    "largest-block": {
        "module": "largest_block",
        "help": "Largest-block compare/single profiling workflow.",
        "main_kind": "noargv",
    },
    "big-block-incremental": {
        "module": "big_block_incremental",
        "help": "Big-block incremental baseline/phase-split workflow.",
        "main_kind": "noargv",
    },
    "featurizer-reuse": {
        "module": "featurizer_reuse",
        "help": "Rust featurizer reuse microbenchmark.",
        "main_kind": "noargv",
    },
    "stress-rebuild": {
        "module": "stress_rebuild",
        "help": "Repeated Rust featurizer rebuild stress workflow.",
        "main_kind": "noargv",
    },
    "calibrate-phase-a": {
        "module": "calibrate_phase_a",
        "help": "Calibrate phase-A accumulator entry bytes from logs.",
        "main_kind": "argv",
    },
    "calibrate-rust-batch": {
        "module": "calibrate_rust_batch",
        "help": "Calibrate Rust batch overhead bytes from logs.",
        "main_kind": "argv",
    },
    "measure-counter-data": {
        "module": "measure_counter_data",
        "help": "Measure CounterData memory impact workflow.",
        "main_kind": "noargv",
    },
}


def _build_cli_parser() -> argparse.ArgumentParser:
    command_lines = [f"  - {name}: {spec['help']}" for name, spec in _COMMANDS.items()]
    command_help = "\n".join(command_lines)
    parser = argparse.ArgumentParser(
        description=(
            "Canonical Rust test/benchmark/stress/calibration CLI for S2AND.\n\n" "Commands:\n" f"{command_help}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=sorted(_COMMANDS.keys()))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def _dispatch(command: str, forwarded_args: list[str]) -> int:
    command_spec = _COMMANDS[command]
    module = _load_internal_module(command_spec["module"])

    global _ACTIVE_CANONICAL_ARGV, _ACTIVE_CANONICAL_COMMAND
    previous_argv = list(sys.argv)
    _ACTIVE_CANONICAL_ARGV = [str(Path(__file__).resolve()), command, *forwarded_args]
    _ACTIVE_CANONICAL_COMMAND = command

    try:
        if command_spec["main_kind"] == "argv":
            return int(module.main(forwarded_args))

        sys.argv = [f"{Path(__file__).resolve()} {command}", *forwarded_args]
        module.main()
        return 0
    finally:
        sys.argv = previous_argv
        _ACTIVE_CANONICAL_ARGV = None
        _ACTIVE_CANONICAL_COMMAND = None


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    parsed = parser.parse_args(argv)
    forwarded_args = list(parsed.args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    return _dispatch(parsed.command, forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())

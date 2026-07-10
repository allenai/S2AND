from __future__ import annotations

import argparse
import importlib
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DATA_ROOT = PROJECT_ROOT / "s2and" / "data"
_SCRIPTS_DIR = Path(__file__).resolve().parent

# Ensure `scripts/_rust_suite` is importable even when this file is loaded via
# `importlib.util.spec_from_file_location` in tests (sys.path won't include scripts/).
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

if TYPE_CHECKING:
    from scripts._rust_suite.common import ProcessTreeRSSMonitor as CommonProcessTreeRSSMonitor
    from scripts._rust_suite.common import RSSMonitor as CommonRSSMonitor
    from scripts._rust_suite.common import build_run_metadata as common_build_run_metadata
    from scripts._rust_suite.common import extract_marked_json_payload as common_extract_marked_json_payload
    from scripts._rust_suite.common import get_result_markers as common_get_result_markers
else:
    from _rust_suite.common import ProcessTreeRSSMonitor as CommonProcessTreeRSSMonitor
    from _rust_suite.common import RSSMonitor as CommonRSSMonitor
    from _rust_suite.common import build_run_metadata as common_build_run_metadata
    from _rust_suite.common import extract_marked_json_payload as common_extract_marked_json_payload
    from _rust_suite.common import get_result_markers as common_get_result_markers

RESULT_JSON_START, RESULT_JSON_END = common_get_result_markers("profile")

_MODULE_IMPORTS = {
    "compare": "_rust_suite.compare_cmd",
    "transfer_mini": "_rust_suite.transfer_mini_cmd",
    "prod_inference": "_rust_suite.prod_inference_cmd",
    "largest_block": "_rust_suite.largest_block_cmd",
    "promoted_incremental_arrow_profile": "_rust_suite.promoted_incremental_arrow_profile_cmd",
    "featurizer_reuse": "_rust_suite.featurizer_reuse_cmd",
    "stress_rebuild": "_rust_suite.stress_rebuild_cmd",
    "calibrate_phase_a": "_rust_suite.calibrate_phase_a_cmd",
    "calibrate_rust_batch": "_rust_suite.calibrate_rust_batch_cmd",
    "summarize_memory_telemetry": "_rust_suite.summarize_memory_telemetry_cmd",
}

_MODULE_CACHE: dict[str, ModuleType] = {}
_ACTIVE_CANONICAL_ARGV: list[str] | None = None
_ACTIVE_LOG_FILE: str | None = None
_ACTIVE_MEMORY_TELEMETRY_JSONL: str | None = None


def _build_run_metadata() -> dict[str, Any]:
    canonical_argv = _ACTIVE_CANONICAL_ARGV if _ACTIVE_CANONICAL_ARGV is not None else list(sys.argv)
    return common_build_run_metadata(
        script_path=Path(__file__).resolve(),
        argv=list(canonical_argv),
        project_root=PROJECT_ROOT,
    )


def _configure_file_logging(log_file: str | None) -> logging.FileHandler | None:
    if not log_file:
        return None
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path = log_path.resolve()

    logger = logging.getLogger("s2and")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved_log_path:
            return None

    handler = logging.FileHandler(resolved_log_path, encoding="utf-8")
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return handler


def _configure_memory_telemetry_jsonl(path: str | None) -> None:
    from s2and import memory_budget

    if not path:
        memory_budget.configure_memory_telemetry_jsonl(None)
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memory_budget.configure_memory_telemetry_jsonl(output_path)


def _active_global_cli_args() -> list[str]:
    args: list[str] = []
    if _ACTIVE_LOG_FILE:
        args.extend(["--log-file", _ACTIVE_LOG_FILE])
    if _ACTIVE_MEMORY_TELEMETRY_JSONL:
        args.extend(["--memory-telemetry-jsonl", _ACTIVE_MEMORY_TELEMETRY_JSONL])
    return args


# Preserve historical test-facing helper exports while using shared implementations.
ProcessTreeRSSMonitor = CommonProcessTreeRSSMonitor
RSSMonitor = CommonRSSMonitor


def _load_internal_module(module_key: str) -> ModuleType:
    cached = _MODULE_CACHE.get(module_key)
    if cached is not None:
        return cached

    module_path = _MODULE_IMPORTS[module_key]
    module = importlib.import_module(module_path)
    _MODULE_CACHE[module_key] = module
    return module


def _run_marked_subprocess(cmd: list[str], start_marker: str, end_marker: str) -> dict[str, Any]:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return common_extract_marked_json_payload(completed.stdout, start_marker, end_marker)


def _single_run(
    backend: str,
    dataset_name: str,
    n_jobs: int,
    profile_output_path: str,
    model_path: str,
    data_root: str = str(PACKAGE_DATA_ROOT / "s2and_mini"),
    arrow_data_root: str = str(PACKAGE_DATA_ROOT),
    specter_file: str = "",
    specter_suffix: str = "_specter2.pkl",
    rust_warm_featurizer_before_predict: int = 0,
    run_label: str | None = None,
    input_format: str = "json",
) -> dict[str, Any]:
    result = _load_internal_module("prod_inference")._single_run(
        backend=backend,
        dataset_name=dataset_name,
        n_jobs=n_jobs,
        profile_output_path=profile_output_path,
        model_path=model_path,
        data_root=data_root,
        arrow_data_root=arrow_data_root,
        specter_file=specter_file,
        specter_suffix=specter_suffix,
        rust_warm_featurizer_before_predict=rust_warm_featurizer_before_predict,
        run_label=run_label,
        input_format=input_format,
    )
    # Ensure metadata points to the canonical CLI entrypoint (this file), even if
    # internal modules are invoked directly.
    if isinstance(result, dict):
        result["run_metadata"] = _build_run_metadata()
    return result


def _run_single_subprocess(
    script_path: Path,
    backend: str,
    dataset_name: str,
    n_jobs: int,
    profile_output_path: str,
    model_path: str,
    data_root: str = str(PACKAGE_DATA_ROOT / "s2and_mini"),
    arrow_data_root: str = str(PACKAGE_DATA_ROOT),
    specter_file: str = "",
    specter_suffix: str = "_specter2.pkl",
    rust_warm_featurizer_before_predict: int = 0,
    single_write_json: str = "",
    run_label: str = "",
    input_format: str = "json",
) -> dict[str, Any]:
    script_path_resolved = Path(script_path)
    if script_path_resolved.name == Path(__file__).name:
        cmd = [
            sys.executable,
            str(script_path_resolved),
            *_active_global_cli_args(),
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
            "--arrow-data-root",
            arrow_data_root,
            "--input-format",
            input_format,
            "--specter-suffix",
            specter_suffix,
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
        arrow_data_root=arrow_data_root,
        specter_file=specter_file,
        specter_suffix=specter_suffix,
        rust_warm_featurizer_before_predict=rust_warm_featurizer_before_predict,
        single_write_json=single_write_json,
        run_label=run_label,
        input_format=input_format,
    )


# ---------------------------------------------------------------------------
# Helper exports used by tests
# ---------------------------------------------------------------------------

_PROXY_EXPORTS: dict[str, tuple[str, str]] = {
    "_language_feature_indices": ("compare", "_language_feature_indices"),
    "_compute_feature_parity": ("compare", "_compute_feature_parity"),
    "_load_dataset_inputs": ("compare", "_load_dataset_inputs"),
    "_effective_train_pairs_size": ("transfer_mini", "_effective_train_pairs_size"),
    "_build_workload": ("transfer_mini", "_build_workload"),
    "_workload_id": ("transfer_mini", "_workload_id"),
    "_resolve_dataset_file": ("transfer_mini", "_resolve_dataset_file"),
    "_build_anddata_kwargs": ("transfer_mini", "_build_anddata_kwargs"),
    "_build_data_paths": ("prod_inference", "_build_data_paths"),
    "_cluster_membership_digest": ("largest_block", "_cluster_membership_digest"),
    "_signature_to_cluster_fingerprint_map": ("largest_block", "_signature_to_cluster_fingerprint_map"),
    "_pairwise_precision_recall_fscore_with_singleton_fix": (
        "largest_block",
        "_pairwise_precision_recall_fscore_with_singleton_fix",
    ),
    "run_rebuild_stress": ("stress_rebuild", "run_rebuild_stress"),
    "_rss_growth_fraction": ("stress_rebuild", "_rss_growth_fraction"),
}


def __getattr__(name: str) -> Any:
    target = _PROXY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_key, attribute_name = target
    return getattr(_load_internal_module(module_key), attribute_name)


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
    "promoted-incremental-arrow-profile": {
        "module": "promoted_incremental_arrow_profile",
        "help": "Arrow-only promoted incremental linker profiling workflow.",
        "main_kind": "argv",
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
        "help": "Calibrate phase-A accumulator entry bytes from memory telemetry JSONL.",
        "main_kind": "argv",
    },
    "calibrate-rust-batch": {
        "module": "calibrate_rust_batch",
        "help": "Calibrate Rust batch overhead bytes from memory telemetry JSONL.",
        "main_kind": "argv",
    },
    "summarize-memory-telemetry": {
        "module": "summarize_memory_telemetry",
        "help": "Summarize memory prediction error ratios from memory telemetry JSONL.",
        "main_kind": "argv",
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
    parser.add_argument(
        "--log-file",
        default=None,
        help="Append s2and logger output to this file. Must appear before the command name.",
    )
    parser.add_argument(
        "--memory-telemetry-jsonl",
        default=None,
        help="Append structured memory telemetry JSONL to this file. Must appear before the command name.",
    )
    return parser


def _dispatch(command: str, forwarded_args: list[str], *, global_args: list[str] | None = None) -> int:
    command_spec = _COMMANDS[command]
    module = _load_internal_module(command_spec["module"])

    global _ACTIVE_CANONICAL_ARGV
    previous_argv = list(sys.argv)
    _ACTIVE_CANONICAL_ARGV = [str(Path(__file__).resolve()), *(global_args or []), command, *forwarded_args]

    try:
        if command_spec["main_kind"] == "argv":
            return int(module.main(forwarded_args))

        sys.argv = [f"{Path(__file__).resolve()} {command}", *forwarded_args]
        module.main()
        return 0
    finally:
        sys.argv = previous_argv
        _ACTIVE_CANONICAL_ARGV = None


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    parsed = parser.parse_args(argv)
    forwarded_args = list(parsed.args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    global_args: list[str] = []
    if parsed.log_file:
        global_args.extend(["--log-file", parsed.log_file])
    if parsed.memory_telemetry_jsonl:
        global_args.extend(["--memory-telemetry-jsonl", parsed.memory_telemetry_jsonl])

    global _ACTIVE_LOG_FILE, _ACTIVE_MEMORY_TELEMETRY_JSONL
    previous_log_file = _ACTIVE_LOG_FILE
    previous_memory_telemetry = _ACTIVE_MEMORY_TELEMETRY_JSONL
    from s2and import memory_budget

    previous_memory_telemetry_path = memory_budget.memory_telemetry_jsonl_path()
    _ACTIVE_LOG_FILE = parsed.log_file
    _ACTIVE_MEMORY_TELEMETRY_JSONL = parsed.memory_telemetry_jsonl
    file_handler = _configure_file_logging(parsed.log_file)
    _configure_memory_telemetry_jsonl(parsed.memory_telemetry_jsonl)
    try:
        return _dispatch(parsed.command, forwarded_args, global_args=global_args)
    finally:
        if file_handler is not None:
            logger = logging.getLogger("s2and")
            logger.removeHandler(file_handler)
            file_handler.close()
        _ACTIVE_LOG_FILE = previous_log_file
        _ACTIVE_MEMORY_TELEMETRY_JSONL = previous_memory_telemetry
        memory_budget.configure_memory_telemetry_jsonl(previous_memory_telemetry_path)


if __name__ == "__main__":
    raise SystemExit(main())

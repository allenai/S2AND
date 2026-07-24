from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType

_SCRIPTS_DIR = Path(__file__).resolve().parent

# Ensure `scripts/_rust_suite` is importable when this module is imported as
# `scripts.rust_suite` instead of executed as a script.
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_MODULE_IMPORTS = {
    "compare": "_rust_suite.compare_cmd",
    "transfer_mini": "_rust_suite.transfer_mini_cmd",
    "largest_block": "_rust_suite.largest_block_cmd",
    "promoted_incremental_arrow_profile": "_rust_suite.promoted_incremental_arrow_profile_cmd",
    "featurizer_reuse": "_rust_suite.featurizer_reuse_cmd",
    "stress_rebuild": "_rust_suite.stress_rebuild_cmd",
    "calibrate_phase_a": "_rust_suite.calibrate_phase_a_cmd",
    "calibrate_rust_batch": "_rust_suite.calibrate_rust_batch_cmd",
    "summarize_memory_telemetry": "_rust_suite.summarize_memory_telemetry_cmd",
}


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


def _load_internal_module(module_key: str) -> ModuleType:
    return importlib.import_module(_MODULE_IMPORTS[module_key])


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
        description=(f"Canonical Rust test/benchmark/stress/calibration CLI for S2AND.\n\nCommands:\n{command_help}"),
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


def _dispatch(command: str, forwarded_args: list[str]) -> int:
    command_spec = _COMMANDS[command]
    module = _load_internal_module(command_spec["module"])

    previous_argv = list(sys.argv)
    try:
        if command_spec["main_kind"] == "argv":
            return int(module.main(forwarded_args))

        sys.argv = [f"{Path(__file__).resolve()} {command}", *forwarded_args]
        module.main()
        return 0
    finally:
        sys.argv = previous_argv


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    parsed = parser.parse_args(argv)
    forwarded_args = list(parsed.args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    from s2and import memory_budget

    previous_memory_telemetry_path = memory_budget.memory_telemetry_jsonl_path()
    file_handler = _configure_file_logging(parsed.log_file)
    _configure_memory_telemetry_jsonl(parsed.memory_telemetry_jsonl)
    try:
        return _dispatch(parsed.command, forwarded_args)
    finally:
        if file_handler is not None:
            logger = logging.getLogger("s2and")
            logger.removeHandler(file_handler)
            file_handler.close()
        memory_budget.configure_memory_telemetry_jsonl(previous_memory_telemetry_path)


if __name__ == "__main__":
    raise SystemExit(main())

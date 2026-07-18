"""Strict run and cross-run comparison identities for pair ablations."""

from __future__ import annotations

import importlib
import importlib.machinery
import platform
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

from scripts._pair_ablation.results import load_strict_json, recipe_id_for, strict_json_digest

RUN_MANIFEST_SCHEMA_VERSION = "s2and_pair_ablation_run_manifest_v3"
COMPARISON_IDENTITY_SCHEMA_VERSION = "s2and_pair_ablation_comparison_identity_v2"

THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
)
RUNTIME_VERSION_KEYS = (
    "python",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "lightgbm",
    "fastcluster",
    "pyarrow",
)
RUN_MANIFEST_KEYS = frozenset(
    {
        "adapter",
        "comparison_identity",
        "config",
        "donor_model_sha256",
        "featurizer_version",
        "git",
        "implementation_sha256",
        "input_identity",
        "recipes",
        "run_id",
        "runtime_versions",
        "rust_extension_sha256",
        "rust_version",
        "schema_version",
        "thread_environment",
        "warning",
    }
)
_RUN_MANIFEST_BASE_KEYS = RUN_MANIFEST_KEYS.difference({"comparison_identity", "run_id"})
_COMPARISON_IDENTITY_KEYS = {"schema_version", "sha256"}
_GIT_KEYS = {"branch", "commit", "diff_binary_sha256", "status_short"}
_DONOR_MODEL_KEYS = {"main", "nameless"}
_RECIPE_ENTRY_KEYS = {"recipe", "recipe_id"}
_HEX_CHARACTERS = frozenset("0123456789abcdef")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str] | frozenset[str], context: str) -> None:
    missing = sorted(expected.difference(value))
    extra = sorted(set(value).difference(expected))
    if missing or extra:
        raise ValueError(f"{context} schema mismatch: missing={missing}, extra={extra}")


def _require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value).issubset(_HEX_CHARACTERS):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_runtime_versions() -> dict[str, str]:
    """Return the exact runtime package versions bound into every run."""

    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": str(numpy.__version__),
        "pandas": str(pandas.__version__),
        "scipy": str(scipy.__version__),
        "sklearn": str(sklearn.__version__),
        "lightgbm": distribution_version("lightgbm"),
        "fastcluster": distribution_version("fastcluster"),
        "pyarrow": distribution_version("pyarrow"),
    }


def rust_extension_binary_sha256() -> str:
    """Hash the loaded native Rust extension binary, not its Python shim."""

    package = importlib.import_module("s2and_rust")
    prefix = f"{package.__name__}."
    candidates: set[Path] = set()
    for name, module in tuple(sys.modules.items()):
        if name != package.__name__ and not name.startswith(prefix):
            continue
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        if any(raw_path.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
            candidates.add(Path(raw_path).resolve())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one loaded s2and_rust extension binary, observed={sorted(candidates)}")
    binary = next(iter(candidates))
    if not binary.is_file():
        raise FileNotFoundError(f"Loaded s2and_rust extension binary is missing: {binary}")
    return _sha256_file(binary)


def _validate_config(config: Any) -> dict[str, Any]:
    value = _require_object(config, "run manifest config")
    for key in ("training_seed", "arm_names"):
        if key not in value:
            raise ValueError(f"run manifest config is missing {key!r}")
    training_seed = value["training_seed"]
    if not isinstance(training_seed, int) or isinstance(training_seed, bool) or training_seed < 0:
        raise ValueError("run manifest config.training_seed must be a non-negative integer")
    arm_names = value["arm_names"]
    if (
        not isinstance(arm_names, Sequence)
        or isinstance(arm_names, str | bytes)
        or not arm_names
        or any(not isinstance(arm, str) or not arm.strip() for arm in arm_names)
        or len(arm_names) != len(set(arm_names))
    ):
        raise ValueError("run manifest config.arm_names must be a non-empty unique sequence of strings")
    return value


def _validate_recipes(recipes: Any, *, arm_names: Sequence[str]) -> None:
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("run manifest recipes must be a non-empty list")
    observed_arms: list[str] = []
    observed_ids: set[str] = set()
    for index, raw_entry in enumerate(recipes):
        entry = _require_object(raw_entry, f"run manifest recipes[{index}]")
        _require_exact_keys(entry, _RECIPE_ENTRY_KEYS, f"run manifest recipes[{index}]")
        recipe = _require_object(entry["recipe"], f"run manifest recipes[{index}].recipe")
        expected_id = recipe_id_for(recipe)
        if entry["recipe_id"] != expected_id:
            raise ValueError(f"run manifest recipes[{index}].recipe_id does not match recipe content")
        if expected_id in observed_ids:
            raise ValueError(f"run manifest contains duplicate recipe_id {expected_id!r}")
        observed_ids.add(expected_id)
        observed_arms.append(str(recipe["arm"]))
    if observed_arms != list(arm_names):
        raise ValueError(
            "run manifest recipe arms do not match config.arm_names: "
            f"expected={list(arm_names)}, observed={observed_arms}"
        )


def _validate_digest_map(value: Any, *, expected_keys: set[str] | None, context: str) -> None:
    mapping = _require_object(value, context)
    if not mapping:
        raise ValueError(f"{context} must not be empty")
    if expected_keys is not None:
        _require_exact_keys(mapping, expected_keys, context)
    for key, digest in mapping.items():
        _require_text(key, f"{context} key")
        _require_digest(digest, f"{context}.{key}")


def _validate_run_fields(payload: Mapping[str, Any]) -> None:
    if payload["schema_version"] != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported run-manifest schema version: {payload['schema_version']!r}")
    config = _validate_config(payload["config"])
    _validate_recipes(payload["recipes"], arm_names=config["arm_names"])
    _require_text(payload["adapter"], "run manifest adapter")
    _require_text(payload["warning"], "run manifest warning")
    if not isinstance(payload["featurizer_version"], int | str) or isinstance(payload["featurizer_version"], bool):
        raise ValueError("run manifest featurizer_version must be an integer or string")
    if isinstance(payload["featurizer_version"], str) and not payload["featurizer_version"].strip():
        raise ValueError("run manifest featurizer_version must not be blank")
    _require_text(payload["rust_version"], "run manifest rust_version")
    _require_digest(payload["rust_extension_sha256"], "run manifest rust_extension_sha256")
    _validate_digest_map(
        payload["donor_model_sha256"],
        expected_keys=_DONOR_MODEL_KEYS,
        context="run manifest donor_model_sha256",
    )
    _validate_digest_map(
        payload["implementation_sha256"],
        expected_keys=None,
        context="run manifest implementation_sha256",
    )
    if not _require_object(payload["input_identity"], "run manifest input_identity"):
        raise ValueError("run manifest input_identity must not be empty")

    git = _require_object(payload["git"], "run manifest git")
    _require_exact_keys(git, _GIT_KEYS, "run manifest git")
    _require_text(git["commit"], "run manifest git.commit")
    _require_digest(git["diff_binary_sha256"], "run manifest git.diff_binary_sha256")
    for key in ("branch", "status_short"):
        if not isinstance(git[key], str):
            raise ValueError(f"run manifest git.{key} must be a string")

    thread_environment = _require_object(payload["thread_environment"], "run manifest thread_environment")
    _require_exact_keys(thread_environment, set(THREAD_ENVIRONMENT_KEYS), "run manifest thread_environment")
    for key, value in thread_environment.items():
        if value is not None and not isinstance(value, str):
            raise ValueError(f"run manifest thread_environment.{key} must be a string or null")

    runtime_versions = _require_object(payload["runtime_versions"], "run manifest runtime_versions")
    _require_exact_keys(runtime_versions, set(RUNTIME_VERSION_KEYS), "run manifest runtime_versions")
    for key, value in runtime_versions.items():
        _require_text(value, f"run manifest runtime_versions.{key}")


def _comparison_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    config = dict(_validate_config(payload["config"]))
    del config["training_seed"]
    del config["arm_names"]
    normalized_run = {
        key: value for key, value in payload.items() if key not in {"comparison_identity", "recipes", "run_id"}
    }
    normalized_run["config"] = config
    normalized = {
        "schema_version": COMPARISON_IDENTITY_SCHEMA_VERSION,
        "normalized_run_manifest": normalized_run,
    }
    return {
        "schema_version": COMPARISON_IDENTITY_SCHEMA_VERSION,
        "sha256": strict_json_digest(normalized),
    }


def build_run_manifest(base_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build and validate a run manifest with derived comparison and run IDs."""

    payload = dict(base_payload)
    _require_exact_keys(payload, _RUN_MANIFEST_BASE_KEYS, "run manifest base payload")
    _validate_run_fields(payload)
    payload["comparison_identity"] = _comparison_identity(payload)
    payload["run_id"] = strict_json_digest(payload)
    return validate_run_manifest(payload)


def validate_run_manifest(raw_payload: Any) -> dict[str, Any]:
    """Strictly validate a run manifest and both of its derived identities."""

    payload = _require_object(raw_payload, "run manifest")
    _require_exact_keys(payload, RUN_MANIFEST_KEYS, "run manifest")
    _validate_run_fields(payload)
    comparison = _require_object(payload["comparison_identity"], "run manifest comparison_identity")
    _require_exact_keys(comparison, _COMPARISON_IDENTITY_KEYS, "run manifest comparison_identity")
    if comparison["schema_version"] != COMPARISON_IDENTITY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported comparison-identity schema version: {comparison['schema_version']!r}")
    _require_digest(comparison["sha256"], "run manifest comparison_identity.sha256")
    expected_comparison = _comparison_identity(payload)
    if comparison != expected_comparison:
        raise ValueError("run manifest comparison_identity does not match normalized run content")
    _require_digest(payload["run_id"], "run manifest run_id")
    expected_run_id = strict_json_digest({key: value for key, value in payload.items() if key != "run_id"})
    if payload["run_id"] != expected_run_id:
        raise ValueError("run manifest run_id does not match run content")
    return payload


def load_run_manifest(path: Path) -> dict[str, Any]:
    """Load and strictly validate one run manifest."""

    return validate_run_manifest(load_strict_json(path))


__all__ = [
    "COMPARISON_IDENTITY_SCHEMA_VERSION",
    "RUNTIME_VERSION_KEYS",
    "RUN_MANIFEST_KEYS",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "THREAD_ENVIRONMENT_KEYS",
    "build_run_manifest",
    "current_runtime_versions",
    "load_run_manifest",
    "rust_extension_binary_sha256",
    "validate_run_manifest",
]

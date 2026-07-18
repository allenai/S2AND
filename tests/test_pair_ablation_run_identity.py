from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from scripts._pair_ablation.results import recipe_id_for, strict_json_digest
from scripts._pair_ablation.run_identity import (
    COMPARISON_IDENTITY_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    RUNTIME_VERSION_KEYS,
    THREAD_ENVIRONMENT_KEYS,
    build_run_manifest,
    current_runtime_versions,
    rust_extension_binary_sha256,
    validate_run_manifest,
)


def _digest(label: str) -> str:
    return strict_json_digest({"label": label})


def _recipe(arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "assembly_version": "exact_budget_v1",
        "auxiliary_sources": [] if arm == "uniform_100k" else ["fixture_auxiliary"],
        "balancing": "none" if arm == "uniform_100k" else "explicit_auxiliary_sources",
        "base_sampler": "uniform_100k",
        "budget_policy": "exact_uniform_after_lodo",
        "complexity_rank": 0 if arm == "uniform_100k" else 1,
        "fixed_budget": arm != "uniform_100k",
        "source_caps": {"linker_pairs_per_domain": 10_000},
    }


def _base_payload(*, training_seed: int = 1111, arms: tuple[str, ...] = ("uniform_100k",)) -> dict[str, Any]:
    recipes = [_recipe(arm) for arm in arms]
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "adapter": "practice_only_legacy_arrow_rust_v1",
        "config": {
            "training_seed": training_seed,
            "evaluation_seed": 1111,
            "arm_names": list(arms),
            "uniform_pairs_per_domain": 100_000,
            "name_pairs_per_domain": 10_000,
            "balanced_medium_pairs_per_domain": 50_000,
            "balanced_pool_pairs_per_domain": 100_000,
            "linker_pairs_per_domain": 10_000,
            "eval_pairs_per_domain": 20_000,
            "threshold_pairs_per_domain": 5_000,
        },
        "donor_model_sha256": {"main": _digest("donor-main"), "nameless": _digest("donor-nameless")},
        "featurizer_version": 5,
        "git": {
            "branch": "fixture",
            "commit": "fixture-commit",
            "diff_binary_sha256": _digest("git-diff"),
            "status_short": "",
        },
        "implementation_sha256": {"scripts/run.py": _digest("implementation")},
        "input_identity": {"catalog_files": {"fixture": {"sha256": _digest("input")}}},
        "recipes": [{"recipe_id": recipe_id_for(recipe), "recipe": recipe} for recipe in recipes],
        "runtime_versions": {
            "python": "3.11.13",
            "numpy": "2.3.1",
            "pandas": "2.3.1",
            "scipy": "1.16.0",
            "sklearn": "1.7.0",
            "lightgbm": "4.6.0",
            "fastcluster": "1.3.0",
            "pyarrow": "21.0.0",
        },
        "rust_extension_sha256": _digest("rust-extension"),
        "rust_version": "0.60.0",
        "thread_environment": {key: None for key in THREAD_ENVIRONMENT_KEYS},
        "warning": "fixture warning",
    }


def test_comparison_identity_allows_only_training_seed_and_arm_recipe_subset() -> None:
    full = build_run_manifest(_base_payload(training_seed=1111, arms=("uniform_100k", "balanced_pairwise")))
    finalist_only = build_run_manifest(_base_payload(training_seed=2222, arms=("uniform_100k",)))

    assert full["run_id"] != finalist_only["run_id"]
    assert full["comparison_identity"] == finalist_only["comparison_identity"]


def test_v3_manifest_and_v2_comparison_reject_old_nested_provenance_shapes() -> None:
    manifest = build_run_manifest(_base_payload())
    missing_diff = copy.deepcopy(manifest)
    del missing_diff["git"]["diff_binary_sha256"]
    missing_runtime = copy.deepcopy(manifest)
    del missing_runtime["runtime_versions"]["fastcluster"]

    assert RUN_MANIFEST_SCHEMA_VERSION == "s2and_pair_ablation_run_manifest_v3"
    assert COMPARISON_IDENTITY_SCHEMA_VERSION == "s2and_pair_ablation_comparison_identity_v2"
    with pytest.raises(ValueError, match="run manifest git schema mismatch"):
        validate_run_manifest(missing_diff)
    with pytest.raises(ValueError, match="runtime_versions schema mismatch"):
        validate_run_manifest(missing_runtime)


def _set_nested(*keys: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        target = payload
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return mutate


@pytest.mark.parametrize(
    "mutate",
    [
        _set_nested("config", "evaluation_seed", value=2222),
        _set_nested("config", "linker_pairs_per_domain", value=50_000),
        _set_nested("config", "balanced_pool_pairs_per_domain", value=120_000),
        _set_nested("implementation_sha256", "scripts/run.py", value=_digest("changed-code")),
        _set_nested("input_identity", "catalog_files", "fixture", "sha256", value=_digest("changed-input")),
        _set_nested("donor_model_sha256", "main", value=_digest("changed-donor")),
        _set_nested("git", "commit", value="changed-commit"),
        _set_nested("git", "diff_binary_sha256", value=_digest("changed-git-diff")),
        _set_nested("featurizer_version", value=6),
        _set_nested("rust_version", value="0.61.0"),
        _set_nested("rust_extension_sha256", value=_digest("changed-rust-binary")),
        _set_nested("adapter", value="changed-adapter"),
        _set_nested("thread_environment", "RAYON_NUM_THREADS", value="8"),
        _set_nested("runtime_versions", "numpy", value="9.9.9"),
        _set_nested("runtime_versions", "pyarrow", value="99.0.0"),
        _set_nested("warning", value="changed-warning"),
    ],
)
def test_comparison_identity_rejects_behavior_or_provenance_drift(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    reference = build_run_manifest(_base_payload())
    changed_payload = copy.deepcopy(_base_payload())
    mutate(changed_payload)
    changed = build_run_manifest(changed_payload)

    assert changed["comparison_identity"] != reference["comparison_identity"]


def test_run_manifest_validation_recomputes_comparison_identity_and_run_id() -> None:
    manifest = build_run_manifest(_base_payload())
    changed_content = copy.deepcopy(manifest)
    changed_content["runtime_versions"]["numpy"] = "9.9.9"

    with pytest.raises(ValueError, match="comparison_identity"):
        validate_run_manifest(changed_content)

    changed_run_id = copy.deepcopy(manifest)
    changed_run_id["run_id"] = _digest("forged-run-id")
    with pytest.raises(ValueError, match="run_id does not match"):
        validate_run_manifest(changed_run_id)

    changed_comparison = copy.deepcopy(manifest)
    changed_comparison["comparison_identity"]["sha256"] = _digest("forged-comparison")
    changed_comparison["run_id"] = strict_json_digest(
        {key: value for key, value in changed_comparison.items() if key != "run_id"}
    )
    with pytest.raises(ValueError, match="comparison_identity"):
        validate_run_manifest(changed_comparison)


def test_runtime_and_native_extension_versions_are_complete() -> None:
    versions = current_runtime_versions()
    extension_digest = rust_extension_binary_sha256()

    assert set(versions) == set(RUNTIME_VERSION_KEYS)
    assert all(isinstance(value, str) and value for value in versions.values())
    assert len(extension_digest) == 64
    assert set(extension_digest) <= set("0123456789abcdef")

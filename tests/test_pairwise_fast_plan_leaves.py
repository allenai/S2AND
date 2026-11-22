"""Every trained LightGBM model stays inside native fast-plan tree sizes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pytest
from hyperopt.pyll import stochastic

from s2and.incremental_linking_training.classic import _build_classic_classifier
from s2and.model import PairwiseModeler
from s2and.model_pairwise import MAX_FAST_PLAN_LEAVES
from scripts.production.model import train_pairwise
from tests.helpers import pairwise_training_args

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_leaf_cap_matches_native_lockstep_limit() -> None:
    source = (REPO_ROOT / "s2and_rust/src/lightgbm_booster/fast.rs").read_text(encoding="utf-8")
    match = re.search(r"const MAX_LOCKSTEP_LEAVES: usize = (\d+);", source)
    assert match is not None
    assert int(match.group(1)) == MAX_FAST_PLAN_LEAVES
    space = PairwiseModeler(n_iter=1).search_space
    rng = np.random.default_rng(0)
    samples = [int(stochastic.sample(space["num_leaves"], rng=rng)) for _ in range(128)]
    assert max(samples) <= MAX_FAST_PLAN_LEAVES
    assert min(samples) >= 7
    # The cap is reachable, so the search still covers the full fast range.
    assert max(samples) > MAX_FAST_PLAN_LEAVES // 2


def test_shipped_incumbent_is_within_leaf_cap() -> None:
    path = REPO_ROOT / "scripts/production/model/pairwise_incumbent_h13.json"
    assert set(train_pairwise._load_incumbent_parameters(path)) == {"main", "nameless"}


@pytest.mark.parametrize("num_leaves", [MAX_FAST_PLAN_LEAVES + 1, 1, "99", None])
def test_incumbent_outside_leaf_cap_is_rejected(tmp_path: Path, num_leaves: object) -> None:
    parameters = {"learning_rate": 0.05}
    if num_leaves is not None:
        parameters["num_leaves"] = num_leaves
    path = tmp_path / "incumbent.json"
    path.write_text(json.dumps({"nameless": parameters}), encoding="utf-8")
    with pytest.raises(SystemExit, match="num_leaves"):
        train_pairwise._load_incumbent_parameters(path)


def test_linker_target_params_are_within_leaf_cap() -> None:
    target = json.loads(
        (REPO_ROOT / "tests/fixtures/incremental_linker_training_target.json").read_text(encoding="utf-8")
    )
    classifier = _build_classic_classifier(dict(target["params"]))
    assert classifier.get_params()["num_leaves"] <= MAX_FAST_PLAN_LEAVES


@pytest.mark.parametrize("num_leaves", [MAX_FAST_PLAN_LEAVES + 1, 1, 64.0])
def test_linker_classifier_outside_leaf_cap_is_rejected(num_leaves: object) -> None:
    with pytest.raises(ValueError, match="num_leaves"):
        _build_classic_classifier({"num_leaves": num_leaves})


def test_fixed_parameters_fit_both_heads_without_search(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    arrays: dict[str, Path] = {}
    for role, rows in (("X_train", 200), ("X_val", 80)):
        features = rng.normal(size=(rows, 3))
        arrays[role] = tmp_path / f"{role}.npy"
        np.save(arrays[role], features)
        labels = (features[:, 0] + 0.1 * rng.normal(size=rows) > 0).astype(np.int8)
        arrays["y_" + role.split("_")[1]] = tmp_path / f"y_{role}.npy"
        np.save(arrays["y_" + role.split("_")[1]], labels)
    # The file cannot override the run's thread budget or the featurizer's constraints.
    fixed = {
        "num_leaves": 7,
        "n_estimators": 5,
        "learning_rate": 0.1,
        "min_child_samples": 5,
        "n_jobs": 8,
        "monotone_constraints": "0,0,0",
    }
    args = argparse.Namespace(n_iter=50, n_jobs=1)

    # Featurizer info supplies the constraints as a comma-separated string.
    model, auc, selection = train_pairwise._fit_pairwise_model(
        args, arrays, feature_prefix="", monotone_constraints="1,0,0", feature_names=["a", "b", "c"], fixed=fixed
    )

    assert selection == {"selected_source": "fixed", "fixed_validation_roc_auc": auc}
    # The summary records the applied values, not the file's overridden ones.
    assert model.best_params == {**fixed, "n_jobs": 1, "monotone_constraints": "1,0,0"}
    assert model.hyperopt_trials_store == {}
    assert model.classifier is not None
    params = model.classifier.get_params()
    assert (params["num_leaves"], params["n_estimators"], params["n_jobs"]) == (7, 5, 1)
    # The search's own estimator defaults stay in force, so a fixed refit repeats the chosen recipe.
    assert (params["tree_learner"], params["monotone_constraints"], params["monotone_constraints_method"]) == (
        "data",
        "1,0,0",
        "advanced",
    )
    # The booster applied the string as three per-feature constraints.
    assert "[monotone_constraints: 1,0,0]" in model.classifier.booster_.model_to_string()
    assert auc > 0.9


@pytest.mark.parametrize(
    "payload", [{"main": {"num_leaves": 7}}, {"main": {"num_leaves": 7}, "nameless": {"num_leaves": 300}}]
)
def test_fixed_parameters_need_both_heads_within_leaf_cap(tmp_path: Path, payload: dict[str, object]) -> None:
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="--fixed-parameters"):
        train_pairwise._load_fixed_parameters(path)


def test_fixed_parameters_exclude_incumbents_before_data_loads(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed.json"
    fixed.write_text(json.dumps({"main": {"num_leaves": 7}, "nameless": {"num_leaves": 7}}), encoding="utf-8")
    args = pairwise_training_args(tmp_path, fixed_parameters=fixed, incumbent_parameters=fixed)
    with pytest.raises(SystemExit, match="do not combine"):
        train_pairwise._preflight_pairwise(args)

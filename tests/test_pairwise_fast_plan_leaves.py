"""Every trained LightGBM model stays inside native fast-plan tree sizes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from hyperopt.pyll import stochastic

from s2and.incremental_linking_training.classic import _build_classic_classifier
from s2and.model import PairwiseModeler
from s2and.model_pairwise import MAX_FAST_PLAN_LEAVES
from scripts.production.model import train_pairwise

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_leaf_cap_matches_native_lockstep_limit() -> None:
    source = (REPO_ROOT / "s2and_rust/src/lightgbm_booster/fast.rs").read_text(encoding="utf-8")
    match = re.search(r"const MAX_LOCKSTEP_LEAVES: usize = (\d+);", source)
    assert match is not None
    assert int(match.group(1)) == MAX_FAST_PLAN_LEAVES


def test_default_search_space_never_exceeds_leaf_cap() -> None:
    space = PairwiseModeler(n_iter=1).search_space
    samples = [int(stochastic.sample(space["num_leaves"])) for _ in range(5_000)]
    assert max(samples) <= MAX_FAST_PLAN_LEAVES
    assert min(samples) >= 7
    # The cap is reachable, so the search still covers the full fast range.
    assert max(samples) > MAX_FAST_PLAN_LEAVES // 2


def test_shipped_incumbent_is_within_leaf_cap() -> None:
    path = REPO_ROOT / "scripts/production/model/pairwise_incumbent_h13.json"
    assert set(train_pairwise._load_incumbent_parameters(path)) == {"main", "nameless"}


@pytest.mark.parametrize("num_leaves", [MAX_FAST_PLAN_LEAVES + 1, 387, 1, "99", None])
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


@pytest.mark.parametrize("num_leaves", [MAX_FAST_PLAN_LEAVES + 1, 1, 64.0, True])
def test_linker_classifier_outside_leaf_cap_is_rejected(num_leaves: object) -> None:
    with pytest.raises(ValueError, match="num_leaves"):
        _build_classic_classifier({"num_leaves": num_leaves})

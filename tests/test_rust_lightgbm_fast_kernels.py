"""Bit-exact parity for the Rust booster's fast QuickScorer and lockstep kernels.

``RustLightGBMBooster`` builds a fast plan when a model fits its guards:
QuickScorer when every tree has at most 32 leaves, lockstep traversal when every
tree has at most 256. Single-tree constants and stumps retain the baseline, as
does every unsupported model. The public
API does not say which kernel ran, so ``_expected_kernel`` mirrors the Rust
dispatch rules from the model text. That lets each test check that its model
really exercises the kernel it claims to cover.

Every public scoring method is compared bit-for-bit against two oracles:
- LightGBM itself, via ``predict(raw_score=True/False)`` on the same float64
  or float32 input;
- a *fallback twin*: the same model text with one extra Zero-missing tree whose
  leaves are both ``0.0``. Zero-missing splits force the baseline, and adding
  ``+0.0`` to a margin that starts at ``+0.0`` never changes its bits, so the
  twin is a platform-independent baseline oracle for probabilities too.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pytest

from s2and.runtime import load_s2and_rust_extension

RustLightGBMBooster: Any = load_s2and_rust_extension().RustLightGBMBooster

PRODUCTION_MAIN_MODEL = Path(__file__).with_name("fixtures") / "rust_lightgbm" / "production_main.lgb"
K_ZERO_THRESHOLD = float(np.float32(1e-35))
THREAD_COUNTS = (1, 4)
NUM_FEATURES = 6
MONOTONE_CONSTRAINTS = [1, -1, 0, 0, 1, 0]
ROW_COUNTS = (1, 7, 8, 9, 15, 16, 17, 255, 256, 257, 1001)

# One Zero-missing split whose leaves are both +0.0; appended last.
_FALLBACK_TREE = (
    "Tree={index}\nnum_leaves=2\nnum_cat=0\nsplit_feature=0\nsplit_gain=0\nthreshold=0\n"
    "decision_type=4\nleft_child=-1\nright_child=-2\nleaf_value=0 0\nis_linear=0\nshrinkage=1\n\n"
)

SHARED_CHILD_MODEL = """tree
version=v4
num_class=1
num_tree_per_iteration=1
max_feature_idx=1
objective=binary sigmoid:1
Tree=0
num_leaves=3
num_cat=0
is_linear=0
split_feature=0 1
threshold=0.5 0.5
decision_type=0 0
left_child=1 -1
right_child=1 -2
leaf_value=10 20 30
end of trees
"""


def _tree_blocks(model_text: str) -> list[dict[str, str]]:
    """Return each tree's ``key=value`` fields from a LightGBM model text."""

    trees: list[dict[str, str]] = []
    for line in model_text.split("end of trees", 1)[0].splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key == "Tree":
            trees.append({})
        elif trees:
            trees[-1][key] = value
    return trees


def _expected_kernel(model_text: str) -> str:
    """Mirror the Rust fast-plan dispatch for normal LightGBM output.

    Normal LightGBM trees are proper trees, so the topology guard is not
    repeated here.
    """

    num_features = int(model_text.split("max_feature_idx=", 1)[1].split()[0]) + 1
    thresholds: dict[str, set[float]] = {}
    max_leaves = 1
    trees = _tree_blocks(model_text)
    for tree in trees:
        max_leaves = max(max_leaves, int(tree["num_leaves"]))
        if tree["num_leaves"] == "1":
            continue
        decision_types = [int(value) for value in tree["decision_type"].split()]
        split_thresholds = [float(value) for value in tree["threshold"].split()]
        if any((value >> 2) & 3 == 1 for value in decision_types) or any(map(math.isnan, split_thresholds)):
            return "baseline"
        for feature, threshold in zip(tree["split_feature"].split(), split_thresholds, strict=True):
            thresholds.setdefault(feature, set()).add(threshold + 0.0)
    if num_features > 256 or max_leaves > 256 or any(len(values) > 255 for values in thresholds.values()):
        return "baseline"
    if len(trees) == 1 and max_leaves <= 2:
        return "baseline"
    return "quickscorer" if max_leaves <= 32 else "lockstep"


def _fallback_twin(model_text: str) -> Any:
    """Load the same model plus one zero-valued Zero-missing tree (baseline route)."""

    header, trees = model_text.split("\nTree=0\n", 1)
    header = "\n".join(line for line in header.splitlines() if not line.startswith("tree_sizes="))
    trees, footer = trees.split("end of trees", 1)
    tree_count = len(_tree_blocks(model_text))
    twin_text = f"{header}\nTree=0\n{trees}{_FALLBACK_TREE.format(index=tree_count)}end of trees{footer}"
    assert _expected_kernel(twin_text) == "baseline"
    return RustLightGBMBooster.from_string(twin_text)


def _train_booster(num_leaves: int, seed: int) -> lgb.Booster:
    """Train a small monotone-constrained booster on data with NaNs."""

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(4000, NUM_FEATURES))
    logits = features @ np.array([1.5, -1.0, 0.7, -0.4, 0.9, 0.2]) + 0.5 * features[:, 2] * features[:, 3]
    labels = (logits + rng.normal(scale=0.7, size=len(features)) > 0).astype(np.int64)
    features[rng.random(size=features.shape) < 0.15] = np.nan
    params = {
        "objective": "binary",
        "num_leaves": num_leaves,
        "min_child_samples": 3,
        "learning_rate": 0.2,
        "monotone_constraints": MONOTONE_CONSTRAINTS,
        "seed": seed,
        "verbose": -1,
    }
    return lgb.train(params, lgb.Dataset(features, label=labels, params={"verbose": -1}), num_boost_round=25)


def _adversarial_rows(model_text: str, num_features: int, row_count: int, seed: int) -> np.ndarray:
    """Mix thresholds, their float64/float32 neighbours, edge values and random values."""

    thresholds: dict[int, list[float]] = {}
    for tree in _tree_blocks(model_text):
        if tree["num_leaves"] != "1":
            for feature, threshold in zip(tree["split_feature"].split(), tree["threshold"].split(), strict=True):
                thresholds.setdefault(int(feature), []).append(float(threshold))
    tiny = np.float32(K_ZERO_THRESHOLD)
    specials = np.array(
        [
            0.0,
            -0.0,
            K_ZERO_THRESHOLD,
            -K_ZERO_THRESHOLD,
            np.nextafter(K_ZERO_THRESHOLD, np.inf),
            np.nextafter(K_ZERO_THRESHOLD, 0.0),
            np.nextafter(-K_ZERO_THRESHOLD, -np.inf),
            np.nextafter(-K_ZERO_THRESHOLD, 0.0),
            0.5 * K_ZERO_THRESHOLD,
            -0.5 * K_ZERO_THRESHOLD,
            float(np.nextafter(tiny, np.float32(np.inf))),
            float(np.nextafter(tiny, np.float32(0.0))),
            np.inf,
            -np.inf,
            np.nan,
            -np.nan,
            np.array(0x7FF0000000000001, dtype=np.uint64).view(np.float64),
            5e-324,
            -5e-324,
            float(np.float32(1e-45)),
            np.finfo(np.float64).tiny,
            1.0,
            -1.0,
        ]
    )
    rng = np.random.default_rng(seed)
    rows = rng.normal(scale=2.0, size=(row_count, num_features))
    for feature in range(num_features):
        choice = rng.integers(0, 10, size=row_count)
        feature_thresholds = np.array(thresholds.get(feature, [0.0]))
        picked = feature_thresholds[rng.integers(0, len(feature_thresholds), size=row_count)]
        narrowed = picked.astype(np.float32)
        candidates = {
            0: picked,
            1: np.nextafter(picked, np.inf),
            2: np.nextafter(picked, -np.inf),
            3: np.nextafter(narrowed, np.float32(np.inf)).astype(np.float64),
            4: np.nextafter(narrowed, np.float32(-np.inf)).astype(np.float64),
            5: specials[rng.integers(0, len(specials), size=row_count)],
            6: specials[rng.integers(0, len(specials), size=row_count)],
        }
        for value, column in candidates.items():
            rows[choice == value, feature] = column[choice == value]
    return rows


def _to_float32(rows: np.ndarray) -> np.ndarray:
    """Narrow like the caller would; overflow to infinity and NaN payloads are intended."""

    with np.errstate(over="ignore", invalid="ignore"):
        return np.ascontiguousarray(rows, dtype=np.float32)


def _lightgbm_predict(booster: lgb.Booster, rows: np.ndarray, *, raw_score: bool) -> np.ndarray:
    """LightGBM's dense binary predictions as a float64 vector."""

    return np.asarray(booster.predict(rows, raw_score=raw_score), dtype=np.float64)


def _assert_same_bits(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    actual = np.asarray(actual)
    assert actual.dtype == np.float64 and expected.dtype == np.float64, label
    assert actual.shape == expected.shape, label
    mismatched = np.flatnonzero(actual.view(np.uint64) != expected.view(np.uint64))
    assert mismatched.size == 0, f"{label}: {mismatched.size} rows differ, first {mismatched[:5].tolist()}"


def _assert_bit_identical(lgb_booster: lgb.Booster, rust: Any, twin: Any, rows: np.ndarray, label: str) -> None:
    """Every public scoring method matches LightGBM and the fallback twin bit-for-bit."""

    rows = np.ascontiguousarray(rows, dtype=np.float64)
    rows_f32 = _to_float32(rows)
    expected = {
        "predict_raw": _lightgbm_predict(lgb_booster, rows, raw_score=True),
        "predict_proba_positive": _lightgbm_predict(lgb_booster, rows, raw_score=False),
        "predict_raw_f32": _lightgbm_predict(lgb_booster, rows_f32, raw_score=True),
        "predict_proba_positive_f32": _lightgbm_predict(lgb_booster, rows_f32, raw_score=False),
    }
    for method, lightgbm_scores in expected.items():
        inputs = rows_f32 if method.endswith("_f32") else rows
        baseline = getattr(twin, method)(inputs, num_threads=1)
        _assert_same_bits(baseline, lightgbm_scores, f"{label} {method} twin vs LightGBM")
        for threads in THREAD_COUNTS:
            scores = getattr(rust, method)(inputs, num_threads=threads)
            _assert_same_bits(scores, lightgbm_scores, f"{label} {method} threads={threads}")


@pytest.fixture(scope="module", params=[(24, "quickscorer"), (64, "lockstep")], ids=["quickscorer", "lockstep"])
def trained_case(request: pytest.FixtureRequest) -> tuple[lgb.Booster, str]:
    """A trained booster sized for one kernel and its model text."""

    num_leaves, kernel = request.param
    booster = _train_booster(num_leaves, seed=num_leaves)
    model_text = booster.model_to_string()
    assert _expected_kernel(model_text) == kernel
    leaf_counts = {int(tree["num_leaves"]) for tree in _tree_blocks(model_text)}
    assert max(leaf_counts) > 32 if kernel == "lockstep" else max(leaf_counts) <= 32
    return booster, model_text


def test_trained_models_match_lightgbm_on_adversarial_and_random_rows(trained_case: tuple[lgb.Booster, str]) -> None:
    booster, model_text = trained_case
    rust = RustLightGBMBooster.from_string(model_text)
    twin = _fallback_twin(model_text)
    decision_types = {int(value) for tree in _tree_blocks(model_text) for value in tree["decision_type"].split()}
    # NaN-missing splits with both default directions.
    assert {8, 10} <= decision_types, decision_types

    adversarial = _adversarial_rows(model_text, NUM_FEATURES, 3000, seed=7)
    _assert_bit_identical(booster, rust, twin, adversarial, "adversarial")
    rng = np.random.default_rng(11)
    random_rows = rng.normal(scale=1.5, size=(2000, NUM_FEATURES))
    random_rows[rng.random(size=random_rows.shape) < 0.1] = np.nan
    _assert_bit_identical(booster, rust, twin, random_rows, "random")


def test_tile_boundaries_empty_and_noncontiguous_inputs(trained_case: tuple[lgb.Booster, str]) -> None:
    booster, model_text = trained_case
    rust = RustLightGBMBooster.from_string(model_text)
    twin = _fallback_twin(model_text)
    rows = _adversarial_rows(model_text, NUM_FEATURES, max(ROW_COUNTS), seed=3)
    for row_count in ROW_COUNTS:
        # Tails of QuickScorer's 16-row passes, lockstep's 8-row groups and 256-row tiles.
        _assert_bit_identical(booster, rust, twin, rows[:row_count], f"rows={row_count}")

    for method in ("predict_raw", "predict_proba_positive", "predict_raw_f32", "predict_proba_positive_f32"):
        dtype = np.float32 if method.endswith("_f32") else np.float64
        empty = getattr(rust, method)(np.empty((0, NUM_FEATURES), dtype=dtype))
        assert empty.shape == (0,) and empty.dtype == np.float64, method
        with pytest.raises(ValueError, match=f"features must have {NUM_FEATURES} columns, got {NUM_FEATURES - 1}"):
            getattr(rust, method)(np.zeros((3, NUM_FEATURES - 1), dtype=dtype))

    for base, raw_method, proba_method in (
        (rows, "predict_raw", "predict_proba_positive"),
        (_to_float32(rows), "predict_raw_f32", "predict_proba_positive_f32"),
    ):
        wide = np.ascontiguousarray(np.column_stack([base, base[:, :2]]))
        views = {
            "fortran": np.asfortranarray(base),
            "row-strided": base[::3],
            "column-sliced": wide[:, :NUM_FEATURES],
        }
        for name, view in views.items():
            assert not view.flags.c_contiguous, name
            contiguous = np.ascontiguousarray(view)
            for threads in THREAD_COUNTS:
                label = f"{name} {raw_method} threads={threads}"
                _assert_same_bits(
                    getattr(rust, raw_method)(view, num_threads=threads),
                    _lightgbm_predict(booster, contiguous, raw_score=True),
                    label,
                )
                _assert_same_bits(
                    getattr(rust, proba_method)(view, num_threads=threads),
                    _lightgbm_predict(booster, contiguous, raw_score=False),
                    label,
                )

    default_threads = rust.predict_raw(rows)
    _assert_same_bits(default_threads, rust.predict_raw(rows, num_threads=1), "default num_threads")
    for duplicate in (copy.copy(rust), copy.deepcopy(rust)):
        _assert_same_bits(duplicate.predict_raw(rows, num_threads=4), default_threads, "copied booster")


def test_production_main_fixture_matches_lightgbm() -> None:
    """The v1.21-shaped production main model (30 leaves per tree) takes QuickScorer."""

    model_text = PRODUCTION_MAIN_MODEL.read_text(encoding="utf-8")
    assert _expected_kernel(model_text) == "quickscorer"
    booster = lgb.Booster(model_str=model_text)
    rust = RustLightGBMBooster(str(PRODUCTION_MAIN_MODEL))
    twin = _fallback_twin(model_text)
    num_features = booster.num_feature()
    rows = _adversarial_rows(model_text, num_features, 2048, seed=20260924)
    _assert_bit_identical(booster, rust, twin, rows, "production_main")


@pytest.mark.parametrize("constant", [True, False], ids=["constant", "stump"])
def test_single_tree_models_use_baseline_without_changing_scores(constant: bool) -> None:
    rng = np.random.default_rng(392)
    features = rng.normal(size=(256, NUM_FEATURES))
    labels = np.zeros(len(features), dtype=np.int64) if constant else (features[:, -1] > 0).astype(np.int64)
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 2, "min_child_samples": 2, "verbose": -1, "num_threads": 1},
        lgb.Dataset(features, label=labels),
        num_boost_round=1,
    )
    model_text = booster.model_to_string()
    trees = _tree_blocks(model_text)
    assert len(trees) == 1 and int(trees[0]["num_leaves"]) == (1 if constant else 2)
    assert _expected_kernel(model_text) == "baseline"
    rows = _adversarial_rows(model_text, NUM_FEATURES, 257, seed=92)
    _assert_bit_identical(
        booster,
        RustLightGBMBooster.from_string(model_text),
        _fallback_twin(model_text),
        rows,
        "single-tree baseline",
    )


def test_shared_child_model_scores_through_baseline() -> None:
    """A parser-accepted shared-child graph must not reach QuickScorer's tree-only masks."""

    rust = RustLightGBMBooster.from_string(SHARED_CHILD_MODEL)
    rows = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    expected = np.array([20.0, 20.0, 10.0, 10.0])
    for threads in THREAD_COUNTS:
        _assert_same_bits(rust.predict_raw(rows, num_threads=threads), expected, f"f64 threads={threads}")
        _assert_same_bits(
            rust.predict_raw_f32(rows.astype(np.float32), num_threads=threads),
            expected,
            f"f32 threads={threads}",
        )
        probabilities = rust.predict_proba_positive(rows, num_threads=threads)
        _assert_same_bits(
            rust.predict_proba_positive_f32(rows.astype(np.float32), num_threads=threads),
            probabilities,
            f"proba threads={threads}",
        )
        np.testing.assert_allclose(probabilities, 1.0 / (1.0 + np.exp(-expected)), rtol=0.0, atol=1e-15)

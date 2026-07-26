"""Parity tests for the pure-Rust LightGBM evaluator (RustLightGBMBooster).

The parity bar is deliberately stronger than the bundle fixture tolerance:
raw scores (sum of leaf values) must be BIT-EXACT against Python lightgbm,
because both sides accumulate the same doubles in the same tree order. Any
residual probability difference is then provably confined to the final
sigmoid's exp() call, which is asserted to 1e-12 (bundle fixtures use 1e-10).

Coverage targets the three spots where a homegrown evaluator can silently
diverge:
1. decision_type bit decoding (categorical bit, default_left bit, missing type)
2. missing-value semantics (NaN vs None vs Zero, the |v| <= (double)1e-35f
   zero window, NaN->0.0 conversion on non-NaN-missing splits)
3. the raw-score -> probability sigmoid, including non-default sigmoid params

Grids inject the exact split thresholds harvested from each model so the
`fval <= threshold` boundary itself is exercised, not just values around it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pytest

from tests.helpers import import_s2and_rust

HAS_RUST, s2and_rust = import_s2and_rust()
if not HAS_RUST:
    raise pytest.skip.Exception("s2and_rust extension is unavailable", allow_module_level=True)
assert s2and_rust is not None and not isinstance(s2and_rust, Exception)
RustLightGBMBooster: Any = s2and_rust.RustLightGBMBooster

SOURCE_BUNDLE_DIR = Path(__file__).resolve().parents[1] / "s2and" / "data" / "production_model_v1.21"
SOURCE_BUNDLE_BOOSTER_RELPATHS = [
    "pairwise/main.lgb",
    "pairwise/nameless.lgb",
    "incremental_linker/booster.lgb",
]

# LightGBM's kZeroThreshold is the float literal 1e-35f widened to double.
K_ZERO_THRESHOLD = float(np.float32(1e-35))

PROBA_ATOL = 1e-12


def _model_split_thresholds(model_text: str) -> np.ndarray:
    """Harvest every numerical split threshold from a .lgb model text."""
    values: list[float] = []
    for match in re.finditer(r"^threshold=(.*)$", model_text, flags=re.MULTILINE):
        values.extend(float(token) for token in match.group(1).split())
    return np.asarray(values, dtype=np.float64)


def _special_value_pool(model_text: str, rng: np.random.Generator) -> np.ndarray:
    """Adversarial cell values: zero-window boundaries, NaN, signed zeros, infs,
    plus exact split thresholds (and their float neighbors) from the model."""
    boundary = np.asarray(
        [
            np.nan,
            0.0,
            -0.0,
            K_ZERO_THRESHOLD,
            -K_ZERO_THRESHOLD,
            np.nextafter(K_ZERO_THRESHOLD, 0.0),
            np.nextafter(K_ZERO_THRESHOLD, 1.0),
            np.nextafter(-K_ZERO_THRESHOLD, 0.0),
            np.nextafter(-K_ZERO_THRESHOLD, -1.0),
            1e-36,
            -1e-36,
            1e-34,
            -1e-34,
            np.inf,
            -np.inf,
            1.0,
            -1.0,
        ],
        dtype=np.float64,
    )
    thresholds = _model_split_thresholds(model_text)
    if thresholds.size > 400:
        thresholds = rng.choice(thresholds, size=400, replace=False)
    neighbors = np.concatenate([np.nextafter(thresholds, np.inf), np.nextafter(thresholds, -np.inf)])
    return np.concatenate([boundary, thresholds, neighbors])


def _mixed_matrix(
    model_text: str,
    num_features: int,
    rng: np.random.Generator,
    n_rows: int = 4096,
    special_fraction: float = 0.35,
) -> np.ndarray:
    """Random matrix mixing multi-scale uniforms with adversarial specials."""
    scales = rng.choice([1.0, 1e2, 1e4, 1e6], size=(n_rows, num_features))
    matrix = rng.uniform(0.0, 1.0, size=(n_rows, num_features)) * scales
    negate = rng.random(size=matrix.shape) < 0.25
    matrix[negate] *= -1.0

    pool = _special_value_pool(model_text, rng)
    special_mask = rng.random(size=matrix.shape) < special_fraction
    matrix[special_mask] = rng.choice(pool, size=int(special_mask.sum()))

    matrix[0, :] = np.nan
    matrix[1, :] = 0.0
    matrix[2, :] = np.inf
    matrix[3, :] = -np.inf
    return matrix


def _assert_parity(lgb_booster: lgb.Booster, rust_booster: Any, features: np.ndarray) -> None:
    features = np.ascontiguousarray(features, dtype=np.float64)
    assert rust_booster.num_features() == lgb_booster.num_feature()

    raw_python = np.asarray(lgb_booster.predict(features, raw_score=True), dtype=np.float64)
    raw_rust = np.asarray(rust_booster.predict_raw(features), dtype=np.float64)
    assert raw_python.shape == raw_rust.shape
    assert raw_python.tobytes() == raw_rust.tobytes(), (
        f"raw scores not bit-exact: max abs diff {np.max(np.abs(raw_python - raw_rust))}"
    )

    proba_python = np.asarray(lgb_booster.predict(features), dtype=np.float64)
    proba_rust = np.asarray(rust_booster.predict_proba_positive(features), dtype=np.float64)
    np.testing.assert_allclose(proba_rust, proba_python, rtol=0.0, atol=PROBA_ATOL)

    # Thread count must not change a single bit (parallelism is over rows only).
    raw_threaded = np.asarray(rust_booster.predict_raw(features, num_threads=4), dtype=np.float64)
    assert raw_rust.tobytes() == raw_threaded.tobytes()


def _assert_float32_matches_prior_widening(rust_booster: Any, features: np.ndarray) -> None:
    """The optimized f32 traversal must reproduce the retired f32->f64 path bit-for-bit."""
    with np.errstate(over="ignore"):
        features_f32 = np.ascontiguousarray(features, dtype=np.float32)
    widened = np.ascontiguousarray(features_f32, dtype=np.float64)

    raw_widened = np.asarray(rust_booster.predict_raw(widened), dtype=np.float64)
    raw_f32 = np.asarray(rust_booster.predict_raw_f32(features_f32), dtype=np.float64)
    assert raw_f32.tobytes() == raw_widened.tobytes()

    proba_widened = np.asarray(rust_booster.predict_proba_positive(widened), dtype=np.float64)
    proba_f32 = np.asarray(rust_booster.predict_proba_positive_f32(features_f32), dtype=np.float64)
    assert proba_f32.tobytes() == proba_widened.tobytes()

    proba_threaded = np.asarray(
        rust_booster.predict_proba_positive_f32(features_f32, num_threads=4),
        dtype=np.float64,
    )
    assert proba_f32.tobytes() == proba_threaded.tobytes()


def _train_booster(
    rng: np.random.Generator,
    *,
    params: dict[str, Any] | None = None,
    inject_nans: bool = False,
    inject_zeros: bool = False,
    num_boost_round: int = 30,
    n_rows: int = 3000,
    n_features: int = 6,
) -> lgb.Booster:
    features = rng.normal(size=(n_rows, n_features))
    logits = features @ rng.normal(size=n_features) + 0.25 * features[:, 0] * features[:, 1]
    labels = (logits + rng.normal(scale=0.5, size=n_rows) > 0).astype(np.int64)
    if inject_nans:
        nan_mask = rng.random(size=features.shape) < 0.15
        features[nan_mask] = np.nan
    if inject_zeros:
        zero_mask = rng.random(size=features.shape) < 0.15
        features[zero_mask] = 0.0
    full_params: dict[str, Any] = {
        "objective": "binary",
        "num_leaves": 15,
        "min_child_samples": 5,
        "learning_rate": 0.3,
        "seed": 0,
        "verbose": -1,
        "verbosity": -1,
    }
    full_params.update(params or {})
    train_set = lgb.Dataset(features, label=labels, params={"verbose": -1})
    return lgb.train(full_params, train_set, num_boost_round=num_boost_round)


def _rust_from_booster(lgb_booster: lgb.Booster) -> Any:
    return RustLightGBMBooster.from_string(lgb_booster.model_to_string())


@pytest.mark.parametrize(
    ("filename", "params", "inject_nans", "inject_zeros", "missing_key"),
    [
        ("main.lgb", {}, True, False, "missing_nan"),
        ("nameless.lgb", {"zero_as_missing": True}, False, True, "missing_zero"),
    ],
)
def test_written_pairwise_boosters_reload_with_python_rust_parity(
    tmp_path: Path,
    filename: str,
    params: dict[str, Any],
    inject_nans: bool,
    inject_zeros: bool,
    missing_key: str,
) -> None:
    """Synthetic main and nameless boosters retain missing/default behavior after reload."""

    rng = np.random.default_rng(20260726)
    trained = _train_booster(
        rng,
        params=params,
        inject_nans=inject_nans,
        inject_zeros=inject_zeros,
    )
    model_path = tmp_path / filename
    trained.save_model(model_path)

    python_booster = lgb.Booster(model_file=str(model_path))
    rust_booster = RustLightGBMBooster(str(model_path))
    summary = rust_booster.decision_type_summary()
    assert summary[missing_key] > 0
    assert 0 < summary["default_left"] < summary["num_splits"]

    features = _mixed_matrix(
        model_path.read_text(encoding="utf-8"),
        python_booster.num_feature(),
        rng,
    )
    _assert_parity(python_booster, rust_booster, features)


# ---------------------------------------------------------------------------
# Explicit historical source-bundle boosters: bit-exact raw scores, fixture agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relpath", SOURCE_BUNDLE_BOOSTER_RELPATHS)
def test_source_bundle_booster_bit_exact_parity(relpath: str) -> None:
    model_path = SOURCE_BUNDLE_DIR / relpath
    if not model_path.exists():
        raise pytest.skip.Exception(f"source bundle booster missing: {model_path}")
    lgb_booster = lgb.Booster(model_file=str(model_path))
    rust_booster = RustLightGBMBooster(str(model_path))
    assert rust_booster.num_trees() == lgb_booster.num_trees()
    assert rust_booster.objective_name() == "binary"

    rng = np.random.default_rng(20260707)
    model_text = model_path.read_text(encoding="utf-8")
    features = _mixed_matrix(model_text, lgb_booster.num_feature(), rng)
    _assert_parity(lgb_booster, rust_booster, features)
    _assert_float32_matches_prior_widening(rust_booster, features)


@pytest.mark.parametrize(
    "model_relpath, fixture_relpath",
    [
        ("pairwise/main.lgb", "pairwise/main_prediction_fixture.json"),
        ("pairwise/nameless.lgb", "pairwise/nameless_prediction_fixture.json"),
    ],
)
def test_source_bundle_fixture_probabilities(model_relpath: str, fixture_relpath: str) -> None:
    import json

    model_path = SOURCE_BUNDLE_DIR / model_relpath
    fixture_path = SOURCE_BUNDLE_DIR / fixture_relpath
    if not model_path.exists() or not fixture_path.exists():
        raise pytest.skip.Exception(f"source bundle artifacts missing under {SOURCE_BUNDLE_DIR}")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    features = np.ascontiguousarray(fixture["features"], dtype=np.float64)
    expected = np.asarray(fixture["expected_probabilities"], dtype=np.float64)

    rust_booster = RustLightGBMBooster(str(model_path))
    positive = np.asarray(rust_booster.predict_proba_positive(features), dtype=np.float64)
    observed = np.column_stack((1.0 - positive, positive))
    rtol = float(fixture.get("rtol", 1e-10))
    atol = float(fixture.get("atol", 1e-10))
    np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)

    _assert_parity(lgb.Booster(model_file=str(model_path)), rust_booster, features)


def test_source_bundle_decision_type_coverage() -> None:
    """The source bundle models must exercise both default directions and
    more than one missing type, so the parity tests above are non-vacuous."""
    model_path = SOURCE_BUNDLE_DIR / "pairwise/main.lgb"
    if not model_path.exists():
        raise pytest.skip.Exception(f"source bundle booster missing: {model_path}")
    summary = RustLightGBMBooster(str(model_path)).decision_type_summary()
    assert summary["num_splits"] > 0
    assert 0 < summary["default_left"] < summary["num_splits"]
    assert summary["missing_none"] > 0
    assert summary["missing_nan"] > 0


# ---------------------------------------------------------------------------
# Synthetic boosters: one per missing-type semantics, plus sigmoid variant
# ---------------------------------------------------------------------------


def test_missing_type_nan_semantics() -> None:
    rng = np.random.default_rng(1)
    lgb_booster = _train_booster(rng, inject_nans=True)
    rust_booster = _rust_from_booster(lgb_booster)
    summary = rust_booster.decision_type_summary()
    assert summary["missing_nan"] > 0, "model must contain NaN-missing splits for coverage"

    features = _mixed_matrix(lgb_booster.model_to_string(), lgb_booster.num_feature(), rng)
    _assert_parity(lgb_booster, rust_booster, features)
    _assert_float32_matches_prior_widening(rust_booster, features)


def test_missing_type_none_semantics() -> None:
    """use_missing=False writes missing_type=None; predict-time NaN becomes 0.0."""
    rng = np.random.default_rng(2)
    lgb_booster = _train_booster(rng, params={"use_missing": False})
    rust_booster = _rust_from_booster(lgb_booster)
    summary = rust_booster.decision_type_summary()
    assert summary["missing_none"] == summary["num_splits"] > 0

    features = _mixed_matrix(lgb_booster.model_to_string(), lgb_booster.num_feature(), rng)
    _assert_parity(lgb_booster, rust_booster, features)
    _assert_float32_matches_prior_widening(rust_booster, features)

    # NaN rows and zero rows must be indistinguishable under None-missing
    # semantics (NaN -> 0.0 conversion), pinning the conversion explicitly.
    nan_rows = np.full((8, lgb_booster.num_feature()), np.nan)
    zero_rows = np.zeros_like(nan_rows)
    nan_raw = np.asarray(rust_booster.predict_raw(nan_rows))
    zero_raw = np.asarray(rust_booster.predict_raw(zero_rows))
    assert nan_raw.tobytes() == zero_raw.tobytes()


def test_missing_type_zero_semantics() -> None:
    """zero_as_missing=True writes missing_type=Zero; the |v| <= 1e-35f window
    and NaN->0.0->default routing are the trickiest branch in the evaluator."""
    rng = np.random.default_rng(3)
    lgb_booster = _train_booster(rng, params={"zero_as_missing": True}, inject_zeros=True)
    rust_booster = _rust_from_booster(lgb_booster)
    summary = rust_booster.decision_type_summary()
    assert summary["missing_zero"] > 0, "model must contain Zero-missing splits for coverage"

    features = _mixed_matrix(
        lgb_booster.model_to_string(),
        lgb_booster.num_feature(),
        rng,
        special_fraction=0.5,
    )
    _assert_parity(lgb_booster, rust_booster, features)
    _assert_float32_matches_prior_widening(rust_booster, features)


def test_default_direction_coverage_across_synthetic_models() -> None:
    """Both default_left directions must appear somewhere in the synthetic set."""
    rng = np.random.default_rng(4)
    totals = {"default_left": 0, "num_splits": 0}
    for seed, params, inject_nans, inject_zeros in [
        (10, None, True, False),
        (11, {"zero_as_missing": True}, False, True),
        (12, {"use_missing": False}, False, False),
    ]:
        booster = _train_booster(
            np.random.default_rng(seed),
            params=params,
            inject_nans=inject_nans,
            inject_zeros=inject_zeros,
        )
        summary = _rust_from_booster(booster).decision_type_summary()
        totals["default_left"] += summary["default_left"]
        totals["num_splits"] += summary["num_splits"]
    assert 0 < totals["default_left"] < totals["num_splits"]
    del rng


def test_non_default_sigmoid_parameter() -> None:
    rng = np.random.default_rng(5)
    lgb_booster = _train_booster(rng, params={"sigmoid": 2.5}, inject_nans=True)
    rust_booster = _rust_from_booster(lgb_booster)
    assert rust_booster.sigmoid() == 2.5

    features = _mixed_matrix(lgb_booster.model_to_string(), lgb_booster.num_feature(), rng)
    _assert_parity(lgb_booster, rust_booster, features)

    # The probability path is exactly sigmoid(raw): verify against numpy so a
    # future change can't silently reorder the transform.
    raw = np.asarray(rust_booster.predict_raw(features))
    proba = np.asarray(rust_booster.predict_proba_positive(features))
    np.testing.assert_allclose(proba, 1.0 / (1.0 + np.exp(-2.5 * raw)), rtol=0.0, atol=1e-15)


def test_single_leaf_trees() -> None:
    """min_child_samples larger than the dataset forces constant (1-leaf) trees."""
    rng = np.random.default_rng(6)
    lgb_booster = _train_booster(rng, params={"min_child_samples": 10_000_000}, num_boost_round=5)
    rust_booster = _rust_from_booster(lgb_booster)
    features = _mixed_matrix(lgb_booster.model_to_string(), lgb_booster.num_feature(), rng, n_rows=64)
    _assert_parity(lgb_booster, rust_booster, features)


# ---------------------------------------------------------------------------
# Unsupported model shapes must be rejected at load, not scored wrongly
# ---------------------------------------------------------------------------


def test_categorical_model_rejected() -> None:
    rng = np.random.default_rng(7)
    n_rows = 2000
    categories = rng.integers(0, 8, size=n_rows)
    noise = rng.normal(size=(n_rows, 2))
    features = np.column_stack([categories.astype(np.float64), noise])
    labels = (categories % 2 == 0).astype(np.int64)
    train_set = lgb.Dataset(features, label=labels, categorical_feature=[0], params={"verbose": -1})
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 7, "min_data_per_group": 1, "verbose": -1, "seed": 0},
        train_set,
        num_boost_round=5,
    )
    assert "num_cat=1" in booster.model_to_string(), "training must actually produce a categorical split"
    with pytest.raises(ValueError, match="categorical"):
        RustLightGBMBooster.from_string(booster.model_to_string())


def test_linear_tree_model_rejected() -> None:
    rng = np.random.default_rng(8)
    lgb_booster = _train_booster(rng, params={"linear_tree": True}, num_boost_round=5)
    with pytest.raises(ValueError, match="linear"):
        RustLightGBMBooster.from_string(lgb_booster.model_to_string())


def test_multiclass_model_rejected() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(600, 4))
    labels = rng.integers(0, 3, size=600)
    train_set = lgb.Dataset(features, label=labels, params={"verbose": -1})
    booster = lgb.train(
        {"objective": "multiclass", "num_class": 3, "verbose": -1, "seed": 0},
        train_set,
        num_boost_round=3,
    )
    with pytest.raises(ValueError, match="binary"):
        RustLightGBMBooster.from_string(booster.model_to_string())


def test_regression_model_rejected() -> None:
    rng = np.random.default_rng(10)
    features = rng.normal(size=(600, 4))
    target = rng.normal(size=600)
    train_set = lgb.Dataset(features, label=target, params={"verbose": -1})
    booster = lgb.train({"objective": "regression", "verbose": -1, "seed": 0}, train_set, num_boost_round=3)
    with pytest.raises(ValueError, match="binary"):
        RustLightGBMBooster.from_string(booster.model_to_string())


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_input_shape_and_layout_handling() -> None:
    rng = np.random.default_rng(11)
    lgb_booster = _train_booster(rng, inject_nans=True, num_boost_round=10)
    rust_booster = _rust_from_booster(lgb_booster)
    features = np.ascontiguousarray(rng.normal(size=(32, lgb_booster.num_feature())))

    with pytest.raises(ValueError, match="columns"):
        rust_booster.predict_raw(features[:, :-1])

    empty = np.empty((0, lgb_booster.num_feature()), dtype=np.float64)
    assert np.asarray(rust_booster.predict_raw(empty)).shape == (0,)

    fortran = np.asfortranarray(features)
    c_raw = np.asarray(rust_booster.predict_raw(features))
    f_raw = np.asarray(rust_booster.predict_raw(fortran))
    assert c_raw.tobytes() == f_raw.tobytes()

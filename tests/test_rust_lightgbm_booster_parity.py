"""Parity tests for the pure-Rust LightGBM evaluator (RustLightGBMBooster).

Covered raw-score cases must be bit-exact against the serialized Python
LightGBM booster because both evaluators accumulate the same doubles in tree
order. This is a regression bar over deterministic adversaries, not a claim of
exhaustive equivalence for every model LightGBM can produce. Probability
agreement is asserted to 1e-12 (bundle fixtures use 1e-10).

Coverage targets the three spots where a homegrown evaluator can silently
diverge:
1. decision_type bit decoding (categorical bit, default_left bit, missing type)
2. missing-value semantics (NaN vs None vs Zero, the |v| <= (double)1e-35f
   zero window, NaN->0.0 conversion on non-NaN-missing splits)
3. the raw-score -> probability sigmoid, including non-default sigmoid params

Synthetic and writer-gate grids construct ancestor-compatible rows for every
numerical split, assert that at least one row reaches its target node, and place
each reachable threshold neighbor in that split's feature. Repeated-feature
ancestors can make a descendant boundary unreachable; those nodes receive an
explicit path witness instead. The much larger historical boosters use a
hard-bounded sample stratified by missing behavior and threshold-neighbor
semantics. All grids exercise missing-value sentinels per split feature instead
of relying on random cell placement.
"""

from __future__ import annotations

from dataclasses import dataclass
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

REALISTIC_FIXTURE_DIR = Path(__file__).with_name("fixtures") / "rust_lightgbm"
REALISTIC_MODEL_PATH = REALISTIC_FIXTURE_DIR / "production_main.lgb"
REALISTIC_PROBABILITY_FIXTURE_PATH = REALISTIC_FIXTURE_DIR / "production_main_prediction_fixture.json"

# LightGBM's kZeroThreshold is the float literal 1e-35f widened to double.
K_ZERO_THRESHOLD = float(np.float32(1e-35))

PROBA_ATOL = 1e-12
PARITY_NUM_THREADS = 4
REALISTIC_FIXTURE_MAX_ROWS = 4096

ONE_SPLIT_INTERIOR_ZERO_MODEL = """\
tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=0
objective=binary sigmoid:1
feature_names=f0
feature_infos=[-2:2]

Tree=0
num_leaves=2
num_cat=0
split_feature=0
split_gain=80
threshold=-5.0000000900125474e-36
decision_type=2
left_child=-1
right_child=-2
leaf_value=-0.2 0.2
leaf_weight=10 10
leaf_count=40 40
internal_value=0
internal_weight=20
internal_count=80
is_linear=0
shrinkage=0.1


end of trees
"""


@dataclass(frozen=True)
class _NumericalSplit:
    """One numerical node and the branch constraints needed to reach it."""

    tree_index: int
    node_index: int
    feature: int
    threshold: float
    default_left: bool
    missing_type: str
    ancestors: tuple[tuple[_NumericalSplit, bool], ...]


def _model_numerical_splits(
    booster: lgb.Booster,
) -> list[_NumericalSplit]:
    """Read numerical splits and paths from LightGBM's public model dump."""

    splits: list[_NumericalSplit] = []

    def visit(
        tree_index: int,
        node: dict[str, Any],
        ancestors: tuple[tuple[_NumericalSplit, bool], ...],
    ) -> None:
        if "split_index" not in node:
            return
        assert node["decision_type"] == "<="
        split = _NumericalSplit(
            tree_index,
            int(node["split_index"]),
            int(node["split_feature"]),
            float(node["threshold"]),
            bool(node["default_left"]),
            str(node["missing_type"]),
            ancestors,
        )
        splits.append(split)
        visit(tree_index, node["left_child"], (*ancestors, (split, True)))
        visit(tree_index, node["right_child"], (*ancestors, (split, False)))

    for tree_index, tree in enumerate(booster.dump_model()["tree_info"]):
        visit(tree_index, tree["tree_structure"], ())
    return splits


def _split_goes_left(split: _NumericalSplit, value: float) -> bool:
    """Apply LightGBM's dense-adapter and numerical-decision semantics."""

    if not np.isnan(value) and -K_ZERO_THRESHOLD <= value <= K_ZERO_THRESHOLD:
        value = 0.0
    if np.isnan(value) and split.missing_type != "NaN":
        value = 0.0
    takes_default = (split.missing_type == "Zero" and -K_ZERO_THRESHOLD <= value <= K_ZERO_THRESHOLD) or (
        split.missing_type == "NaN" and np.isnan(value)
    )
    if takes_default:
        return split.default_left
    return value <= split.threshold


def _select_adversarial_splits(
    splits: list[_NumericalSplit],
    max_splits: int | None,
) -> list[_NumericalSplit]:
    """Select a hard-bounded, semantically stratified split sample."""

    if max_splits is None or len(splits) <= max_splits:
        return splits
    if max_splits <= 0:
        raise ValueError("max_splits must be positive")

    strata: dict[tuple[Any, ...], list[int]] = {}
    for index, split in enumerate(splits):
        strata.setdefault(_split_stratum(split), []).append(index)
    if len(strata) > max_splits:
        raise ValueError(f"{max_splits=} cannot represent all {len(strata)} split strata")

    per_stratum = max_splits // len(strata)
    selected_indices = {
        indices[int(position)]
        for indices in strata.values()
        for position in np.linspace(0, len(indices) - 1, num=min(len(indices), per_stratum), dtype=int)
    }
    remaining = max_splits - len(selected_indices)
    unselected = [index for index in range(len(splits)) if index not in selected_indices]
    selected_indices.update(
        unselected[int(position)]
        for position in np.linspace(0, len(unselected) - 1, num=min(len(unselected), remaining), dtype=int)
    )
    selected = [split for index, split in enumerate(splits) if index in selected_indices]
    assert len(selected) <= max_splits
    assert {_split_stratum(split) for split in selected} == set(strata)
    return selected


def _threshold_neighbors(split: _NumericalSplit) -> tuple[float, float, float]:
    """Return the three values surrounding one serialized threshold."""

    return (
        np.nextafter(split.threshold, -np.inf),
        split.threshold,
        np.nextafter(split.threshold, np.inf),
    )


def _threshold_regime(threshold: float) -> str:
    """Classify thresholds around LightGBM's float32 zero window."""

    if threshold < -K_ZERO_THRESHOLD:
        return "below_zero_window"
    if threshold == -K_ZERO_THRESHOLD:
        return "negative_boundary"
    if threshold < 0.0:
        return "negative_interior"
    if threshold == 0.0:
        return "zero"
    if threshold < K_ZERO_THRESHOLD:
        return "positive_interior"
    if threshold == K_ZERO_THRESHOLD:
        return "positive_boundary"
    return "above_zero_window"


def _reachable_threshold_neighbors(split: _NumericalSplit) -> tuple[bool, bool, bool]:
    """Identify boundary values compatible with same-feature ancestors."""

    constraints = [constraint for constraint in split.ancestors if constraint[0].feature == split.feature]
    return tuple(
        all(_split_goes_left(ancestor, value) == take_left for ancestor, take_left in constraints)
        for value in _threshold_neighbors(split)
    )


def _split_stratum(split: _NumericalSplit) -> tuple[str, bool, str, tuple[bool, ...]]:
    """Describe the split semantics that an adversarial sample must retain."""

    return (
        split.missing_type,
        split.default_left,
        _threshold_regime(split.threshold),
        _reachable_threshold_neighbors(split),
    )


def _candidate_path_values(
    constraints: list[tuple[_NumericalSplit, bool]],
) -> list[float]:
    """Return deterministic candidates spanning every ancestor boundary."""

    candidates: list[float] = [
        0.0,
        -1.0,
        1.0,
        -np.inf,
        np.inf,
        np.nan,
        -K_ZERO_THRESHOLD,
        K_ZERO_THRESHOLD,
    ]
    candidates.extend(
        value
        for split, _ in constraints
        for value in (
            np.nextafter(split.threshold, -np.inf),
            split.threshold,
            np.nextafter(split.threshold, np.inf),
        )
    )
    return list(dict.fromkeys(candidates))


def _path_constrained_row(
    target: _NumericalSplit,
    target_value: float | None,
    num_features: int,
) -> np.ndarray | None:
    """Construct a verified target row, or None for an unreachable fixed value."""

    by_feature: dict[int, list[tuple[_NumericalSplit, bool]]] = {}
    for constraint in target.ancestors:
        by_feature.setdefault(constraint[0].feature, []).append(constraint)
    by_feature.setdefault(target.feature, [])

    row = np.zeros(num_features, dtype=np.float64)
    for feature, constraints in by_feature.items():
        if feature == target.feature:
            candidates = [target_value] if target_value is not None else _candidate_path_values(constraints)
        else:
            candidates = _candidate_path_values(constraints)
        value = next(
            (
                candidate
                for candidate in candidates
                if all(_split_goes_left(split, candidate) == take_left for split, take_left in constraints)
            ),
            None,
        )
        if value is None and feature == target.feature and target_value is not None:
            return None
        assert value is not None, (
            f"no value reaches tree {target.tree_index} node {target.node_index} through feature {feature}"
        )
        row[feature] = value

    assert all(_split_goes_left(split, row[split.feature]) == take_left for split, take_left in target.ancestors)
    return row


def _adversarial_matrix(
    booster: lgb.Booster,
    num_features: int,
    rng: np.random.Generator,
    n_random_rows: int = 1024,
    max_split_boundaries: int | None = None,
) -> np.ndarray:
    """Mix broad random rows with deterministic split-feature adversaries."""

    scales = rng.choice([1.0, 1e2, 1e4, 1e6], size=(n_random_rows, num_features))
    matrix = rng.uniform(0.0, 1.0, size=(n_random_rows, num_features)) * scales
    negate = rng.random(size=matrix.shape) < 0.25
    matrix[negate] *= -1.0

    matrix[0, :] = np.nan
    matrix[1, :] = 0.0
    matrix[2, :] = np.inf
    matrix[3, :] = -np.inf
    matrix[4, ::2] = np.nan
    matrix[4, 1::2] = 0.0

    missing_values = (
        np.nan,
        0.0,
        -0.0,
        K_ZERO_THRESHOLD,
        -K_ZERO_THRESHOLD,
        np.nextafter(K_ZERO_THRESHOLD, 0.0),
        np.nextafter(K_ZERO_THRESHOLD, np.inf),
        np.nextafter(-K_ZERO_THRESHOLD, 0.0),
        np.nextafter(-K_ZERO_THRESHOLD, -np.inf),
        1e-36,
        -1e-36,
        1e-34,
        -1e-34,
        np.inf,
        -np.inf,
    )
    splits = _model_numerical_splits(booster)
    split_features = sorted({split.feature for split in splits})
    selected_splits = _select_adversarial_splits(splits, max_split_boundaries)

    threshold_rows: list[np.ndarray] = []
    for split in selected_splits:
        split_rows: list[np.ndarray] = []
        reachable_neighbors = _reachable_threshold_neighbors(split)
        for value, reachable in zip(_threshold_neighbors(split), reachable_neighbors, strict=True):
            row = _path_constrained_row(split, value, num_features)
            assert (row is not None) == reachable
            if row is not None:
                split_rows.append(row)
        if not split_rows:
            witness = _path_constrained_row(split, None, num_features)
            assert witness is not None
            split_rows.append(witness)
        threshold_rows.extend(split_rows)

    missing_rows: list[np.ndarray] = []
    for feature in split_features:
        for value in missing_values:
            row = np.full(num_features, 0.5, dtype=np.float64)
            row[feature] = value
            missing_rows.append(row)

    deterministic_rows = threshold_rows + missing_rows
    if not deterministic_rows:
        return matrix
    return np.vstack((matrix, np.stack(deterministic_rows)))


def _assert_parity(
    lgb_booster: lgb.Booster,
    rust_booster: Any,
    features: np.ndarray,
    *,
    case_id: str = "",
) -> None:
    features = np.ascontiguousarray(features, dtype=np.float64)
    assert rust_booster.num_features() == lgb_booster.num_feature(), case_id

    raw_python = np.asarray(
        lgb_booster.predict(features, raw_score=True, num_threads=PARITY_NUM_THREADS),
        dtype=np.float64,
    )
    raw_rust = np.asarray(rust_booster.predict_raw(features, num_threads=PARITY_NUM_THREADS), dtype=np.float64)
    assert raw_python.shape == raw_rust.shape, case_id
    assert raw_python.tobytes() == raw_rust.tobytes(), (
        f"{case_id}: raw scores not bit-exact: max abs diff {np.max(np.abs(raw_python - raw_rust))}"
    )

    proba_python = np.asarray(lgb_booster.predict(features, num_threads=PARITY_NUM_THREADS), dtype=np.float64)
    proba_rust = np.asarray(
        rust_booster.predict_proba_positive(features, num_threads=PARITY_NUM_THREADS),
        dtype=np.float64,
    )
    np.testing.assert_allclose(proba_rust, proba_python, rtol=0.0, atol=PROBA_ATOL, err_msg=case_id)


def _assert_float32_matches_prior_widening(rust_booster: Any, features: np.ndarray) -> None:
    """The optimized f32 traversal must reproduce the retired f32->f64 path bit-for-bit."""
    with np.errstate(over="ignore"):
        features_f32 = np.ascontiguousarray(features, dtype=np.float32)
    widened = np.ascontiguousarray(features_f32, dtype=np.float64)

    raw_widened = np.asarray(
        rust_booster.predict_raw(widened, num_threads=PARITY_NUM_THREADS),
        dtype=np.float64,
    )
    raw_f32 = np.asarray(
        rust_booster.predict_raw_f32(features_f32, num_threads=PARITY_NUM_THREADS),
        dtype=np.float64,
    )
    assert raw_f32.tobytes() == raw_widened.tobytes()

    proba_widened = np.asarray(
        rust_booster.predict_proba_positive(widened, num_threads=PARITY_NUM_THREADS),
        dtype=np.float64,
    )
    proba_f32 = np.asarray(
        rust_booster.predict_proba_positive_f32(features_f32, num_threads=PARITY_NUM_THREADS),
        dtype=np.float64,
    )
    assert proba_f32.tobytes() == proba_widened.tobytes()


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


def test_written_pairwise_boosters_reload_with_python_rust_parity(tmp_path: Path) -> None:
    """Synthetic main and nameless boosters retain missing/default behavior after reload."""

    cases = (
        ("main", "main.lgb", {}, True, False, "missing_nan"),
        ("nameless", "nameless.lgb", {"zero_as_missing": True}, True, True, "missing_zero"),
    )
    for case_id, filename, params, inject_nans, inject_zeros, missing_key in cases:
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
        assert summary[missing_key] > 0, case_id
        assert 0 < summary["default_left"] < summary["num_splits"], case_id

        features = _adversarial_matrix(
            python_booster,
            python_booster.num_feature(),
            rng,
        )
        _assert_parity(python_booster, rust_booster, features, case_id=case_id)


def test_negative_zero_sentinel_split_matches_lightgbm() -> None:
    """Pin LightGBM's strict equality behavior at a serialized ``-1e-35f`` split."""

    rng = np.random.default_rng(8)
    lgb_booster = _train_booster(rng, inject_nans=True)
    splits = _model_numerical_splits(lgb_booster)
    sentinel_features = {split.feature for split in splits if split.threshold == -K_ZERO_THRESHOLD}
    assert sentinel_features, "fixture must contain a serialized lower-zero-sentinel split"

    rows: list[np.ndarray] = []
    for feature in sorted(sentinel_features):
        for value in (
            np.nextafter(-K_ZERO_THRESHOLD, -np.inf),
            -K_ZERO_THRESHOLD,
            np.nextafter(-K_ZERO_THRESHOLD, np.inf),
        ):
            row = np.zeros(lgb_booster.num_feature(), dtype=np.float64)
            row[feature] = value
            rows.append(row)

    rust_booster = _rust_from_booster(lgb_booster)
    features = np.stack(rows)
    _assert_parity(lgb_booster, rust_booster, features)

    features_f32 = np.ascontiguousarray(features, dtype=np.float32)
    raw_python_f32 = np.asarray(lgb_booster.predict(features_f32, raw_score=True), dtype=np.float64)
    raw_rust_f32 = np.asarray(rust_booster.predict_raw_f32(features_f32), dtype=np.float64)
    assert raw_python_f32.tobytes() == raw_rust_f32.tobytes()


def test_dense_zero_window_precedes_one_split_missing_semantics() -> None:
    """Dense values inside LightGBM's zero window become zero for every missing type."""

    features = np.asarray(
        [
            [np.nextafter(-K_ZERO_THRESHOLD, -np.inf)],
            [-0.75 * K_ZERO_THRESHOLD],
            [-0.5 * K_ZERO_THRESHOLD],
            [-0.25 * K_ZERO_THRESHOLD],
            [0.0],
            [0.75 * K_ZERO_THRESHOLD],
            [np.nextafter(K_ZERO_THRESHOLD, np.inf)],
            [np.nan],
        ],
        dtype=np.float64,
    )
    features_f32 = np.ascontiguousarray(features, dtype=np.float32)
    for decision_type in (0, 2, 4, 6, 8, 10):
        case_id = f"decision-type-{decision_type}"
        model_text = ONE_SPLIT_INTERIOR_ZERO_MODEL.replace(
            "decision_type=2",
            f"decision_type={decision_type}",
        )
        python_booster = lgb.Booster(model_str=model_text)
        rust_booster = RustLightGBMBooster.from_string(model_text)
        _assert_parity(python_booster, rust_booster, features, case_id=case_id)

        raw_python_f32 = np.asarray(python_booster.predict(features_f32, raw_score=True), dtype=np.float64)
        raw_rust_f32 = np.asarray(rust_booster.predict_raw_f32(features_f32), dtype=np.float64)
        assert raw_python_f32.tobytes() == raw_rust_f32.tobytes(), case_id


# ---------------------------------------------------------------------------
# One realistic production-shaped booster: bit-exact raw scores and fixture agreement
# ---------------------------------------------------------------------------


def test_realistic_booster_bit_exact_parity() -> None:
    assert REALISTIC_MODEL_PATH.is_file()
    lgb_booster = lgb.Booster(model_file=str(REALISTIC_MODEL_PATH))
    rust_booster = RustLightGBMBooster(str(REALISTIC_MODEL_PATH))
    assert rust_booster.num_trees() == lgb_booster.num_trees()
    assert rust_booster.objective_name() == "binary"

    rng = np.random.default_rng(20260707)
    features = _adversarial_matrix(
        lgb_booster,
        lgb_booster.num_feature(),
        rng,
        max_split_boundaries=512,
    )
    assert features.shape[0] <= REALISTIC_FIXTURE_MAX_ROWS
    _assert_parity(lgb_booster, rust_booster, features)
    _assert_float32_matches_prior_widening(rust_booster, features)


def test_realistic_booster_fixture_probabilities() -> None:
    import json

    assert REALISTIC_MODEL_PATH.is_file()
    assert REALISTIC_PROBABILITY_FIXTURE_PATH.is_file()
    fixture = json.loads(REALISTIC_PROBABILITY_FIXTURE_PATH.read_text(encoding="utf-8"))
    features = np.ascontiguousarray(fixture["features"], dtype=np.float64)
    expected = np.asarray(fixture["expected_probabilities"], dtype=np.float64)

    rust_booster = RustLightGBMBooster(str(REALISTIC_MODEL_PATH))
    positive = np.asarray(rust_booster.predict_proba_positive(features), dtype=np.float64)
    observed = np.column_stack((1.0 - positive, positive))
    rtol = float(fixture.get("rtol", 1e-10))
    atol = float(fixture.get("atol", 1e-10))
    np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)

    _assert_parity(lgb.Booster(model_file=str(REALISTIC_MODEL_PATH)), rust_booster, features)


def test_realistic_booster_decision_type_coverage() -> None:
    """The fixture covers both default directions and two missing types."""

    assert REALISTIC_MODEL_PATH.is_file()
    summary = RustLightGBMBooster(str(REALISTIC_MODEL_PATH)).decision_type_summary()
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

    features = _adversarial_matrix(lgb_booster, lgb_booster.num_feature(), rng)
    _assert_parity(lgb_booster, rust_booster, features)
    _assert_float32_matches_prior_widening(rust_booster, features)


def test_missing_type_none_semantics() -> None:
    """use_missing=False writes missing_type=None; predict-time NaN becomes 0.0."""
    rng = np.random.default_rng(2)
    lgb_booster = _train_booster(rng, params={"use_missing": False})
    rust_booster = _rust_from_booster(lgb_booster)
    summary = rust_booster.decision_type_summary()
    assert summary["missing_none"] == summary["num_splits"] > 0

    features = _adversarial_matrix(lgb_booster, lgb_booster.num_feature(), rng)
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
    lgb_booster = _train_booster(
        rng,
        params={"zero_as_missing": True},
        inject_nans=True,
        inject_zeros=True,
    )
    rust_booster = _rust_from_booster(lgb_booster)
    summary = rust_booster.decision_type_summary()
    assert summary["missing_zero"] > 0, "model must contain Zero-missing splits for coverage"

    features = _adversarial_matrix(
        lgb_booster,
        lgb_booster.num_feature(),
        rng,
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

    features = _adversarial_matrix(lgb_booster, lgb_booster.num_feature(), rng)
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
    features = _adversarial_matrix(
        lgb_booster,
        lgb_booster.num_feature(),
        rng,
        n_random_rows=64,
    )
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


def test_input_shape_layout_and_thread_identity() -> None:
    rng = np.random.default_rng(11)
    lgb_booster = _train_booster(rng, inject_nans=True, num_boost_round=10)
    rust_booster = _rust_from_booster(lgb_booster)
    features = np.ascontiguousarray(rng.normal(size=(32, lgb_booster.num_feature())))

    with pytest.raises(ValueError, match="columns"):
        rust_booster.predict_raw(features[:, :-1])

    empty = np.empty((0, lgb_booster.num_feature()), dtype=np.float64)
    assert np.asarray(rust_booster.predict_raw(empty)).shape == (0,)

    one_thread = np.asarray(rust_booster.predict_raw(features, num_threads=1))
    four_threads = np.asarray(rust_booster.predict_raw(features, num_threads=PARITY_NUM_THREADS))
    assert one_thread.tobytes() == four_threads.tobytes()

    fortran = np.asfortranarray(features)
    fortran_raw = np.asarray(rust_booster.predict_raw(fortran, num_threads=PARITY_NUM_THREADS))
    assert four_threads.tobytes() == fortran_raw.tobytes()

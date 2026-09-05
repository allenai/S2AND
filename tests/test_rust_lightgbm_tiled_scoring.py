"""Verify tiled scoring preserves scalar score bits across tile boundaries."""

from pathlib import Path

import numpy as np
import pytest

from s2and.runtime import load_s2and_rust_extension

RustLightGBMBooster = load_s2and_rust_extension().RustLightGBMBooster
ONE_SPLIT_INTERIOR_ZERO_MODEL = (
    Path(__file__).with_name("fixtures") / "rust_lightgbm" / "interior_zero.lgb"
).read_text(encoding="utf-8")


@pytest.mark.parametrize("row_count", [0, 1, 63, 64, 65, 127, 128, 129, 1001])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("probability", [False, True])
def test_tiled_scores_match_single_rows_exactly(row_count, dtype, probability):
    """Exercise parallel tiles, their partial tails, and noncontiguous inputs."""
    booster = RustLightGBMBooster.from_string(ONE_SPLIT_INTERIOR_ZERO_MODEL)
    tiny = dtype(1e-35)
    values = np.asarray(
        [
            np.nan,
            -np.inf,
            -tiny,
            np.nextafter(-tiny, dtype(-np.inf)),
            -0.0,
            0.0,
            np.nextafter(tiny, dtype(np.inf)),
            tiny,
            np.inf,
        ],
        dtype=dtype,
    )
    rows = np.resize(values, row_count * 2).reshape(row_count, 2)[:, :1]
    method_name = "predict_proba_positive" if probability else "predict_raw"
    if dtype == np.float32:
        method_name += "_f32"
    predict = getattr(booster, method_name)
    expected = np.asarray(
        [predict(row.reshape(1, 1), num_threads=1)[0] for row in rows],
        dtype=np.float64,
    )
    actual = predict(rows, num_threads=10)
    assert actual.tobytes() == expected.tobytes()

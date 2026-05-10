"""Shared retrieval scoring policy mirrored from Rust when available."""

from __future__ import annotations

from typing import Any

try:
    import s2and_rust
except ImportError:  # pragma: no cover - Rust extension optional for pure-Python helpers
    s2and_rust = None  # type: ignore[assignment]

PYTHON_RETRIEVAL_FEATURE_ORDER = ("centroid", "coauthor", "affiliation", "middle", "first_name")
PYTHON_DEFAULT_HYBRID_CENTROID_POLICY_NAME = "h_wang_any_input_v2"
PYTHON_DEFAULT_HYBRID_CENTROID_WEIGHTS = (0.527232, 0.223412, 0.146909, 0.009439, 0.093007)
PYTHON_DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS = (0.520012, 0.220264, 0.109278, 0.150447, 0.0)
PYTHON_DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS = (0.40, 0.23, 0.12, 0.05, 0.07)
PYTHON_RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE = -0.25
PYTHON_RETRIEVAL_YEAR_SCORE_DECAY_YEARS = 15.0
PYTHON_RETRIEVAL_YEAR_SCORE_RANGE_GAP = 10
PYTHON_RETRIEVAL_YEAR_SCORE_RANGE_PENALTY = 0.15
PYTHON_RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP = 35


def _rust_tuple(name: str, python_value: tuple[Any, ...]) -> tuple[Any, ...]:
    if s2and_rust is None or not hasattr(s2and_rust, name):
        return python_value
    return tuple(getattr(s2and_rust, name))


def _rust_float(name: str, python_value: float) -> float:
    if s2and_rust is None or not hasattr(s2and_rust, name):
        return python_value
    return float(getattr(s2and_rust, name))


def _rust_int(name: str, python_value: int) -> int:
    if s2and_rust is None or not hasattr(s2and_rust, name):
        return python_value
    return int(getattr(s2and_rust, name))


def _rust_str(name: str, python_value: str) -> str:
    if s2and_rust is None or not hasattr(s2and_rust, name):
        return python_value
    return str(getattr(s2and_rust, name))


HYBRID_FEATURE_ORDER = tuple(
    str(value) for value in _rust_tuple("RETRIEVAL_FEATURE_ORDER", PYTHON_RETRIEVAL_FEATURE_ORDER)
)
DEFAULT_HYBRID_CENTROID_WEIGHTS = tuple(
    float(value) for value in _rust_tuple("DEFAULT_HYBRID_CENTROID_WEIGHTS", PYTHON_DEFAULT_HYBRID_CENTROID_WEIGHTS)
)
DEFAULT_HYBRID_CENTROID_POLICY_NAME = _rust_str(
    "DEFAULT_HYBRID_CENTROID_POLICY_NAME",
    PYTHON_DEFAULT_HYBRID_CENTROID_POLICY_NAME,
)
DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS = tuple(
    float(value)
    for value in _rust_tuple(
        "DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS",
        PYTHON_DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS,
    )
)
DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS = tuple(
    float(value) for value in _rust_tuple("DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS", PYTHON_DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS)
)
RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE = _rust_float(
    "RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE",
    PYTHON_RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE,
)
RETRIEVAL_YEAR_SCORE_DECAY_YEARS = _rust_float(
    "RETRIEVAL_YEAR_SCORE_DECAY_YEARS",
    PYTHON_RETRIEVAL_YEAR_SCORE_DECAY_YEARS,
)
RETRIEVAL_YEAR_SCORE_RANGE_GAP = _rust_int(
    "RETRIEVAL_YEAR_SCORE_RANGE_GAP",
    PYTHON_RETRIEVAL_YEAR_SCORE_RANGE_GAP,
)
RETRIEVAL_YEAR_SCORE_RANGE_PENALTY = _rust_float(
    "RETRIEVAL_YEAR_SCORE_RANGE_PENALTY",
    PYTHON_RETRIEVAL_YEAR_SCORE_RANGE_PENALTY,
)
RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP = _rust_int(
    "RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP",
    PYTHON_RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP,
)

from __future__ import annotations

import warnings

_SKLEARN_FEATURE_NAME_WARNING_PATTERNS = (
    r"X does not have valid feature names, but .* was fitted with feature names",
    r"X has feature names, but .* was fitted without feature names",
)


def suppress_sklearn_feature_name_warnings() -> None:
    for pattern in _SKLEARN_FEATURE_NAME_WARNING_PATTERNS:
        warnings.filterwarnings(
            "ignore",
            message=pattern,
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )

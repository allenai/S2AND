"""Fail-hard import and ABI smoke for an installed ``s2and_rust`` wheel."""

from __future__ import annotations

import s2and_rust


def main() -> None:
    """Validate APIs needed by production scoring without loading model data."""

    build_info = s2and_rust.get_build_info()
    required_planner_methods = {
        "from_query_signatures",
        "from_auto_queries",
        "plan",
        "plan_query_signatures",
        "name_counts_index",
        "build_telemetry",
    }
    observed_planner_methods = set(build_info.get("raw_arrow_query_signature_planner_methods", ()))
    missing_planner_methods = sorted(required_planner_methods - observed_planner_methods)
    if missing_planner_methods:
        raise RuntimeError(f"s2and_rust raw planner ABI is missing methods: {missing_planner_methods}")
    if not callable(getattr(s2and_rust.RustFeaturizer, "from_arrow_paths", None)):
        raise RuntimeError("s2and_rust.RustFeaturizer.from_arrow_paths is unavailable")
    for method_name in ("from_auto_queries", "plan", "name_counts_index"):
        if not callable(getattr(s2and_rust.RawBlockQueryCandidatePlanner, method_name, None)):
            raise RuntimeError(f"s2and_rust.RawBlockQueryCandidatePlanner.{method_name} is unavailable")
    if not callable(getattr(s2and_rust.RustLightGBMBooster, "predict_proba_positive_f32", None)):
        raise RuntimeError("s2and_rust.RustLightGBMBooster.predict_proba_positive_f32 is unavailable")
    print(f"Validated installed s2and_rust {s2and_rust.__version__} production ABI")


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import math
import os
import sys
from importlib.machinery import PathFinder
from typing import Any

from s2and.data import ANDData


def equalish(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-3) -> bool:
    if math.isnan(float(a)) and math.isnan(float(b)):
        return True
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def import_s2and_rust(
    *,
    required_method: str | None = None,
    prefer_site_packages: bool = False,
) -> tuple[bool, Any | Exception | None]:
    def _has_required_api(module: Any) -> bool:
        rust_featurizer = getattr(module, "RustFeaturizer", None)
        if rust_featurizer is None:
            return False
        method_name = required_method or "from_dataset"
        return hasattr(rust_featurizer, method_name)

    try:
        import s2and_rust

        if _has_required_api(s2and_rust):
            return True, s2and_rust
        raise AttributeError("s2and_rust imported, but required RustFeaturizer API is unavailable")
    except Exception as err:
        if not prefer_site_packages:
            return False, err

        try:
            sys.modules.pop("s2and_rust", None)
            sys.modules.pop("s2and_rust.s2and_rust", None)
            sys.modules.pop("s2and_rust._s2and_rust", None)
            site_paths = [path for path in sys.path if "site-packages" in path]
            spec = PathFinder.find_spec("s2and_rust", site_paths)
            if spec is None or spec.loader is None:
                raise err
            module = importlib.util.module_from_spec(spec)
            sys.modules["s2and_rust"] = module
            spec.loader.exec_module(module)
            if not _has_required_api(module):
                raise AttributeError("s2and_rust imported from site-packages, but required API is unavailable")
            return True, module
        except Exception as fallback_err:
            return False, fallback_err


def build_dummy_dataset(
    name: str,
    *,
    mode: str = "train",
    load_name_counts: bool = False,
    compute_reference_features: bool = False,
    n_jobs: int = 1,
) -> ANDData:
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        mode=mode,
        load_name_counts=load_name_counts,
        preprocess=True,
        n_jobs=n_jobs,
        compute_reference_features=compute_reference_features,
    )

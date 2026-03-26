from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from importlib.machinery import PathFinder
from typing import Any

import scripts.eval_cluster_retrieval as retrieval
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


def build_query_features(
    *,
    first: str = "a",
    middle_initials: frozenset[str] = frozenset(),
    year: int | None = None,
    orcid: str | None = None,
    specter: Any | None = None,
    has_coauthors: bool = False,
    has_affiliations: bool = False,
    has_full_first: bool = False,
    has_middle: bool = False,
) -> retrieval.QueryFeatures:
    """Build a compact `QueryFeatures` fixture for retrieval tests."""

    return retrieval.QueryFeatures(
        first=first,
        middle="",
        first_initial=first[:1] if first else "",
        middle_initials=middle_initials,
        coauthor_blocks=frozenset({"a smith"}) if has_coauthors else frozenset(),
        affiliation_terms=frozenset({"lab"}) if has_affiliations else frozenset(),
        venue_terms=frozenset(),
        year=year,
        orcid=orcid,
        specter=specter,
        has_specter=specter is not None,
        has_coauthors=has_coauthors,
        has_affiliations=has_affiliations,
        has_full_first=has_full_first,
        has_middle=has_middle,
    )


def build_cluster_summary(
    *,
    component_key: str,
    size: int = 1,
    first_name_counts: Counter[str] | None = None,
    middle_initial_counts: Counter[str] | None = None,
    coauthor_counts: Counter[str] | None = None,
    affiliation_counts: Counter[str] | None = None,
    venue_counts: Counter[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    year_mean: float | None = None,
    orcid_values: frozenset[str] = frozenset(),
    specter_centroid: Any | None = None,
    exemplar_vectors: list[Any] | None = None,
) -> retrieval.ClusterSummary:
    """Build a compact `ClusterSummary` fixture for retrieval tests."""

    return retrieval.ClusterSummary(
        component_key=component_key,
        cluster_id=component_key,
        block_key="b",
        size=size,
        first_name_counts=first_name_counts or Counter(),
        middle_initial_counts=middle_initial_counts or Counter(),
        coauthor_counts=coauthor_counts or Counter(),
        affiliation_counts=affiliation_counts or Counter(),
        venue_counts=venue_counts or Counter(),
        year_values=[],
        year_min=year_min,
        year_max=year_max,
        year_mean=year_mean,
        orcid_values=orcid_values,
        specter_centroid=specter_centroid,
        exemplar_vectors=[] if exemplar_vectors is None else exemplar_vectors,
    )

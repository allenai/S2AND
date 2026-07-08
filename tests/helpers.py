from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from importlib.machinery import PathFinder
from typing import Any

from s2and.data import ANDData
from s2and.incremental_linking.query_adapter import ClusterSummary, QueryFeatures
from s2and.runtime import detect_rust_runtime_capabilities


def tiny_name_counts() -> dict[str, dict[str, int]]:
    """Return a small deterministic name-count artifact for dummy tests."""

    return {
        "first_dict": {
            "abdul": 10,
            "alexander": 20,
            "dr": 30,
        },
        "last_dict": {
            "sattar": 40,
            "konovalov": 50,
        },
        "first_last_dict": {
            "abdul sattar": 60,
            "alexander konovalov": 70,
            "dr sattar": 80,
        },
        "last_first_initial_dict": {
            "sattar a": 90,
            "sattar d": 100,
            "konovalov a": 110,
        },
    }


def tiny_name_counts_tuple() -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Return tiny name counts in the tuple shape used by the cached loader."""

    counts = tiny_name_counts()
    return (
        counts["first_dict"],
        counts["last_dict"],
        counts["first_last_dict"],
        counts["last_first_initial_dict"],
    )


def patch_tiny_name_counts_loader(monkeypatch: Any) -> None:
    """Patch the production name-count loader to avoid huge fixture generation."""

    import s2and.data as data_module

    monkeypatch.setattr(data_module, "_load_name_counts_cached", tiny_name_counts_tuple)


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
        method_name = required_method or "from_arrow_paths"
        if not hasattr(rust_featurizer, method_name):
            return False
        capabilities = detect_rust_runtime_capabilities(extension_module=module)
        return capabilities.core_runtime_available

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


def attach_arrow_featurizer_bundle(
    dataset: ANDData,
    bundle_dir: Any,
    *,
    name_counts: str = "tiny",
) -> dict[str, str]:
    """Write a temporary Arrow bundle from an in-memory ``ANDData`` and attach it.

    Rust featurizers are built exclusively through ``from_arrow_paths``, so
    tests that exercise the Rust featurizer on a hand-built ``ANDData`` must
    spool it to an Arrow bundle first. Writes signatures/papers/paper_authors
    (+ specter when the dataset carries embeddings) with the same writers
    production conversion uses, adds the raw-planner batch indexes and a
    ``name_counts_index`` sidecar (``"tiny"`` -> ``tiny_name_counts()``,
    ``"empty"`` -> empty lookups so every name-count query misses), then
    attaches the validated paths for featurization and prediction.
    """

    from unittest import mock

    import s2and.data as data_module
    from s2and.arrow_training import attach_training_arrow_featurizer_paths
    from s2and.incremental_linking.feature_block_arrow import (
        write_name_counts_index,
        write_raw_arrow_batch_lookup_indexes,
    )
    from scripts.arrow_conversion_helpers import write_feature_block_arrow_from_anddata

    if name_counts == "tiny":
        loader = tiny_name_counts_tuple
    elif name_counts == "empty":
        loader = lambda: ({}, {}, {}, {})  # noqa: E731
    else:
        raise ValueError(f"name_counts must be 'tiny' or 'empty', got {name_counts!r}")

    include_specter = bool(getattr(dataset, "specter_embeddings", None))
    arrow_paths = write_feature_block_arrow_from_anddata(
        dataset,
        bundle_dir,
        include_specter=include_specter,
    )
    arrow_paths, _index_metrics = write_raw_arrow_batch_lookup_indexes(arrow_paths, bundle_dir)
    with mock.patch.object(data_module, "_load_name_counts_cached", loader):
        name_counts_index_path, _name_counts_metrics = write_name_counts_index(bundle_dir, overwrite=True)
    arrow_paths["name_counts_index"] = name_counts_index_path
    return attach_training_arrow_featurizer_paths(dataset, arrow_paths)


def build_dummy_dataset(
    name: str,
    *,
    mode: str = "train",
    load_name_counts: bool | dict[str, dict[str, int]] = False,
    n_jobs: int = 1,
) -> ANDData:
    resolved_name_counts = tiny_name_counts() if load_name_counts is True else load_name_counts
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        mode=mode,
        load_name_counts=resolved_name_counts,
        preprocess=True,
        n_jobs=n_jobs,
    )


def build_query_features(
    *,
    first: str = "a",
    middle_initials: frozenset[str] = frozenset(),
    year: int | None = None,
    orcid: str | None = None,
    specter: Any | None = None,
    coauthor_blocks: frozenset[str] | None = None,
    affiliation_terms: frozenset[str] | None = None,
    venue_terms: frozenset[str] | None = None,
    has_coauthors: bool = False,
    has_affiliations: bool = False,
    has_full_first: bool = False,
    has_middle: bool = False,
    title_terms: frozenset[str] = frozenset(),
    name_counts: Any | None = None,
    paper_author_count: int = 0,
    paper_author_names: frozenset[str] = frozenset(),
    author_position: int | None = None,
    local10_author_names: frozenset[str] = frozenset(),
    signature_id: str = "",
) -> QueryFeatures:
    """Build a compact `QueryFeatures` fixture for retrieval tests."""

    return QueryFeatures(
        first=first,
        middle="",
        first_initial=first[:1] if first else "",
        middle_initials=middle_initials,
        coauthor_blocks=(
            coauthor_blocks
            if coauthor_blocks is not None
            else (frozenset({"a smith"}) if has_coauthors else frozenset())
        ),
        affiliation_terms=(
            affiliation_terms
            if affiliation_terms is not None
            else (frozenset({"lab"}) if has_affiliations else frozenset())
        ),
        venue_terms=venue_terms if venue_terms is not None else frozenset(),
        year=year,
        orcid=orcid,
        specter=specter,
        has_specter=specter is not None,
        has_coauthors=has_coauthors,
        has_affiliations=has_affiliations,
        has_full_first=has_full_first,
        has_middle=has_middle,
        title_terms=title_terms,
        name_counts=name_counts,
        paper_author_count=paper_author_count,
        paper_author_names=paper_author_names,
        author_position=author_position,
        local10_author_names=local10_author_names,
        signature_id=signature_id,
    )


def build_cluster_summary(
    *,
    component_key: str,
    size: int = 1,
    first_name_counts: Counter[str] | None = None,
    middle_initial_counts: Counter[str] | None = None,
    coauthor_counts: Counter[str] | None = None,
    non_mega_coauthor_counts: Counter[str] | None = None,
    affiliation_counts: Counter[str] | None = None,
    venue_counts: Counter[str] | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    year_mean: float | None = None,
    orcid_values: frozenset[str] = frozenset(),
    specter_centroid: Any | None = None,
    exemplar_vectors: list[Any] | None = None,
    title_counts: Counter[str] | None = None,
    name_counts_values: tuple[Any, ...] = (),
    max_paper_author_count: int = 0,
    member_paper_author_names: tuple[frozenset[str], ...] = (),
    member_paper_author_counts: tuple[int, ...] = (),
    member_author_positions: tuple[int | None, ...] = (),
    member_local10_author_names: tuple[frozenset[str], ...] = (),
    member_signature_ids: tuple[str, ...] = (),
    member_title_terms: tuple[frozenset[str], ...] = (),
) -> ClusterSummary:
    """Build a compact `ClusterSummary` fixture for retrieval tests."""

    return ClusterSummary(
        component_key=component_key,
        cluster_id=component_key,
        block_key="b",
        size=size,
        first_name_counts=first_name_counts or Counter(),
        middle_initial_counts=middle_initial_counts or Counter(),
        coauthor_counts=coauthor_counts or Counter(),
        non_mega_coauthor_counts=(
            non_mega_coauthor_counts if non_mega_coauthor_counts is not None else coauthor_counts or Counter()
        ),
        affiliation_counts=affiliation_counts or Counter(),
        venue_counts=venue_counts or Counter(),
        year_values=[],
        year_min=year_min,
        year_max=year_max,
        year_mean=year_mean,
        orcid_values=orcid_values,
        specter_centroid=specter_centroid,
        exemplar_vectors=[] if exemplar_vectors is None else exemplar_vectors,
        title_counts=title_counts or Counter(),
        name_counts_values=name_counts_values,
        max_paper_author_count=max_paper_author_count,
        member_paper_author_names=member_paper_author_names,
        member_paper_author_counts=member_paper_author_counts,
        member_author_positions=member_author_positions,
        member_local10_author_names=member_local10_author_names,
        member_signature_ids=member_signature_ids,
        member_title_terms=member_title_terms,
    )

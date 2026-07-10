from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from s2and.data import ANDData
from s2and.incremental_linking.query_adapter import ClusterSummary, QueryFeatures
from s2and.name_counts_index import NameCountsIndex
from s2and.runtime import REQUIRED_RUST_EXTENSION_VERSION, load_s2and_rust_extension


def write_test_arrow_artifact_manifest(bundle_dir: Any, paths: dict[str, str]) -> Path:
    """Write the canonical manifest required by production Arrow boundaries."""

    from s2and.arrow_inputs import _build_arrow_artifact_generation
    from s2and.consts import NORMALIZATION_VERSION

    root = Path(bundle_dir)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "normalization_version": NORMALIZATION_VERSION,
                "paths": dict(paths),
                "artifact_generation": _build_arrow_artifact_generation(paths, root),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return manifest_path


def write_minimal_arrow_prediction_bundle(
    bundle_dir: Any,
    *,
    include_specter: bool = False,
) -> dict[str, str]:
    """Write tiny valid Arrow tables, batch indexes, and a bound manifest."""

    import pyarrow as pa

    from s2and.arrow_inputs import RAW_PLANNER_ARROW_BATCH_INDEX_KEYS, RAW_PLANNER_ARROW_KEY_COLUMNS
    from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_arrow_ipc_table

    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    signature_ids = [str(index) for index in range(10)] + ["q1", "q2", "seed1"]
    tables = {
        "signatures": pa.table({"signature_id": pa.array(signature_ids, type=pa.string())}),
        "papers": pa.table({"paper_id": pa.array(["p0"], type=pa.string())}),
        "paper_authors": pa.table({"paper_id": pa.array(["p0"], type=pa.string())}),
    }
    if include_specter:
        tables["specter"] = pa.table({"paper_id": pa.array(["p0"], type=pa.string())})

    paths: dict[str, str] = {}
    for table_name, table in tables.items():
        arrow_path = root / f"{table_name}.arrow"
        paths[table_name] = write_arrow_ipc_table(table, arrow_path)
        index_key = RAW_PLANNER_ARROW_BATCH_INDEX_KEYS[table_name]
        index_path = root / f"{table_name}.{index_key}.bin"
        paths[index_key], _metrics = write_arrow_batch_lookup_index(
            arrow_path,
            index_path,
            key_column=RAW_PLANNER_ARROW_KEY_COLUMNS[table_name],
            table_name=table_name,
        )
    write_test_arrow_artifact_manifest(root, paths)
    return paths


def tiny_name_counts_provenance() -> dict[str, Any]:
    """Return explicit provenance for synthetic in-memory name counts."""

    return {
        "schema_version": "name_counts_provenance_v1",
        "normalization_version": "canonical_v2",
        "generation_id": "test-tiny-name-counts",
        "source_snapshot_id": "test-fixture",
        "source_kind": "fixture:test",
        "source_query_sha256": "1" * 64,
        "selected_rows_sha256": "2" * 64,
        "selected_row_count": 1,
        "source_row_count": 1,
        "pickle_sha256": "0" * 64,
    }


def tiny_name_counts() -> dict[str, Any]:
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
        "provenance": tiny_name_counts_provenance(),
    }


def tiny_name_counts_tuple() -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Return tiny mappings for offline index-writer tests."""

    counts = tiny_name_counts()
    return (
        counts["first_dict"],
        counts["last_dict"],
        counts["first_last_dict"],
        counts["last_first_initial_dict"],
    )


_TINY_NAME_COUNTS_ROOT = Path(tempfile.mkdtemp(prefix="s2and-test-name-counts-"))
_TINY_NAME_COUNTS_LOCK = threading.Lock()
_TINY_NAME_COUNTS_PATH: str | None = None


def _cleanup_tiny_name_counts() -> None:
    shutil.rmtree(_TINY_NAME_COUNTS_ROOT, ignore_errors=True)


atexit.register(_cleanup_tiny_name_counts)


def tiny_name_counts_index_path() -> str:
    """Build and share one manifest-backed tiny index for Python tests."""

    global _TINY_NAME_COUNTS_PATH
    if _TINY_NAME_COUNTS_PATH is not None:
        return _TINY_NAME_COUNTS_PATH
    with _TINY_NAME_COUNTS_LOCK:
        if _TINY_NAME_COUNTS_PATH is None:
            from s2and.incremental_linking.feature_block_arrow import write_name_counts_index

            _TINY_NAME_COUNTS_PATH, _metrics = write_name_counts_index(
                _TINY_NAME_COUNTS_ROOT,
                tiny_name_counts_tuple(),
                tiny_name_counts_provenance(),
                overwrite=True,
            )
    return _TINY_NAME_COUNTS_PATH


def tiny_name_counts_index() -> NameCountsIndex:
    """Return the shared verified native handle used by Python tests."""

    return NameCountsIndex.open(tiny_name_counts_index_path())


def equalish(a: float, b: float, rel_tol: float = 0.0, abs_tol: float = 1e-6) -> bool:
    if math.isnan(float(a)) and math.isnan(float(b)):
        return True
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def import_s2and_rust(
    *,
    required_method: str | None = None,
    required_module_attrs: tuple[str, ...] = (),
) -> tuple[bool, Any | Exception | None]:
    require_rust = os.environ.get("S2AND_TEST_REQUIRE_RUST", "").strip().lower() in {"1", "true", "yes", "on"}

    def _has_required_api(module: Any) -> bool:
        for attr_name in required_module_attrs:
            if not hasattr(module, attr_name):
                return False
        rust_featurizer = getattr(module, "RustFeaturizer", None)
        if rust_featurizer is None:
            return False
        method_name = required_method or "from_arrow_paths"
        if not hasattr(rust_featurizer, method_name):
            return False
        return getattr(module, "__version__", None) == REQUIRED_RUST_EXTENSION_VERSION

    try:
        s2and_rust = load_s2and_rust_extension()
        if _has_required_api(s2and_rust):
            return True, s2and_rust
        raise AttributeError("s2and_rust imported, but required Rust runtime API is unavailable")
    except Exception as err:
        if require_rust:
            raise RuntimeError("Rust-enabled tests require a working s2and_rust runtime") from err
        return False, err


def build_arrow_training_dataset(
    dataset: ANDData,
    bundle_dir: Any,
    *,
    name_counts: str = "tiny",
) -> ANDData:
    """Round-trip an ``ANDData`` through the fixed Rust-training constructor.

    Rust featurizers are built exclusively through ``from_arrow_paths``, so
    tests that exercise the Rust featurizer on a hand-built ``ANDData`` must
    spool it to an Arrow bundle first. Writes signatures/papers/paper_authors
    (+ specter when the dataset carries embeddings) with the same writers
    production conversion uses, adds the raw-planner batch indexes and a
    ``name_counts_index`` sidecar (``"tiny"`` -> ``tiny_name_counts()``,
    ``"empty"`` -> empty lookups so every name-count query misses), then
    returns a new fully initialized Rust-backed dataset.
    """

    from s2and.arrow_training import build_training_anddata_from_arrow
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
    name_counts_index_path, _name_counts_metrics = write_name_counts_index(
        bundle_dir,
        loader(),
        tiny_name_counts_provenance(),
        overwrite=True,
    )
    arrow_paths["name_counts_index"] = name_counts_index_path
    write_test_arrow_artifact_manifest(bundle_dir, arrow_paths)
    from s2and.consts import NORMALIZATION_VERSION

    signature_to_cluster_id = getattr(dataset, "signature_to_cluster_id", None) or {}
    members_by_cluster: dict[str, list[str]] = {}
    for signature_id in dataset.signatures:
        cluster_id = str(signature_to_cluster_id.get(signature_id, f"singleton_{signature_id}"))
        members_by_cluster.setdefault(cluster_id, []).append(str(signature_id))
    clusters = {
        cluster_id: {
            "cluster_id": cluster_id,
            "signature_ids": signature_ids,
            "model_version": -1,
        }
        for cluster_id, signature_ids in members_by_cluster.items()
    }
    name_tuples = getattr(dataset, "name_tuples", "filtered")
    if isinstance(name_tuples, frozenset):
        name_tuples = set(name_tuples)
    arrow_dataset = build_training_anddata_from_arrow(
        arrow_paths,
        f"{dataset.name}_arrow",
        expected_normalization_version=NORMALIZATION_VERSION,
        clusters=clusters,
        block_type=str(getattr(dataset, "block_type", "s2")),
        train_pairs_size=int(getattr(dataset, "train_pairs_size", 30_000)),
        val_pairs_size=int(getattr(dataset, "val_pairs_size", 5_000)),
        test_pairs_size=int(getattr(dataset, "test_pairs_size", 5_000)),
        random_seed=int(getattr(dataset, "random_seed", 1111)),
        n_jobs=int(getattr(dataset, "n_jobs", 1)),
        preprocess=bool(getattr(dataset, "preprocess", True)),
        name_tuples=name_tuples,
    )
    # Parity tests intentionally exercise Python reference methods on the same
    # object. Keep that already-materialized reference view; Rust consumes only
    # the immutable Arrow generation created above.
    arrow_dataset.signatures = dict(dataset.signatures)
    arrow_dataset.papers = dict(dataset.papers)
    arrow_dataset.specter_embeddings = dict(getattr(dataset, "specter_embeddings", {}) or {})
    arrow_dataset.signature_to_block = dict(getattr(dataset, "signature_to_block", {}))
    arrow_dataset.cluster_seeds_require = dict(getattr(dataset, "cluster_seeds_require", {}))
    arrow_dataset.cluster_seeds_disallow = set(getattr(dataset, "cluster_seeds_disallow", set()))
    return arrow_dataset


def build_dummy_dataset(
    name: str,
    *,
    mode: str = "train",
    name_counts_index: bool | str | os.PathLike[str] | NameCountsIndex | None = None,
    n_jobs: int = 1,
) -> ANDData:
    resolved_name_counts_index = tiny_name_counts_index() if name_counts_index is True else name_counts_index
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        mode=mode,
        name_counts_index=resolved_name_counts_index,
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

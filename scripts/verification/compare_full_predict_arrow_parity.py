"""Compare incumbent full predict against direct Arrow/Rust full predict.

This is the reusable gate for the complete Arrow inference schema. It builds a
bounded incumbent ``ANDData`` first, writes Arrow IPC tables and current
raw-planner batch-index sidecars from the same bounded payload, builds
``RustFeaturizer.from_arrow_paths(...)``, and compares features, constraints,
distances, and clusters.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as infile:
        return json.load(infile)


def _resolve_fixture_path(fixture_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else fixture_dir / path


def _fixture_meta_path(meta: dict[str, Any], fixture_dir: Path, key: str) -> Path:
    path_value = meta.get("paths", {}).get(key)
    if path_value is None:
        raise KeyError(f"fixture meta is missing paths.{key}")
    return _resolve_fixture_path(fixture_dir, path_value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _select_block_signature_ids(signatures: dict[str, Any], block_name: str, block_size: int) -> list[str]:
    selected = []
    for signature_id, payload in signatures.items():
        author_info = payload.get("author_info", {}) if isinstance(payload, dict) else {}
        if str(author_info.get("block", "")) == block_name:
            selected.append(str(signature_id))
    selected.sort()
    if not selected:
        raise ValueError(f"No signatures found for block {block_name!r}")
    return selected[: min(int(block_size), len(selected))]


def _filter_payloads(
    signatures: dict[str, Any],
    papers: dict[str, Any],
    specter_embeddings: Any,
    selected_signature_ids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    keep_signature_ids = set(selected_signature_ids)
    filtered_signatures = {
        signature_id: payload for signature_id, payload in signatures.items() if str(signature_id) in keep_signature_ids
    }
    needed_paper_ids = {
        str(payload.get("paper_id", payload.get("paperId")))
        for payload in filtered_signatures.values()
        if isinstance(payload, dict) and payload.get("paper_id", payload.get("paperId")) is not None
    }
    filtered_papers = {paper_id: payload for paper_id, payload in papers.items() if str(paper_id) in needed_paper_ids}
    if isinstance(specter_embeddings, dict):
        filtered_specter = {
            paper_id: vector for paper_id, vector in specter_embeddings.items() if str(paper_id) in needed_paper_ids
        }
    elif isinstance(specter_embeddings, tuple) and len(specter_embeddings) == 2:
        matrix, paper_ids = specter_embeddings
        keep_offsets = [offset for offset, paper_id in enumerate(paper_ids) if str(paper_id) in needed_paper_ids]
        filtered_specter = (
            np.asarray(matrix, dtype=np.float32)[np.asarray(keep_offsets, dtype=np.int64)],
            [paper_ids[offset] for offset in keep_offsets],
        )
    else:
        filtered_specter = specter_embeddings
    return filtered_signatures, filtered_papers, filtered_specter


def _load_cluster_seeds_require(
    meta: dict[str, Any],
    fixture_dir: Path,
    selected_signature_ids: Sequence[str],
    *,
    enabled: bool,
) -> dict[str, str]:
    if not enabled:
        return {}
    path = meta.get("paths", {}).get("cluster_seeds_require")
    if path is None:
        raise ValueError("Requested --use-cluster-seeds but fixture meta has no paths.cluster_seeds_require")
    selected_ids = {str(signature_id) for signature_id in selected_signature_ids}
    payload = _load_json(_resolve_fixture_path(fixture_dir, path))
    if not isinstance(payload, dict):
        raise TypeError(f"cluster_seeds_require must be a JSON object, got {type(payload).__name__}")
    return {
        str(signature_id): str(component_id)
        for signature_id, component_id in payload.items()
        if str(signature_id) in selected_ids
    }


def _numeric_report(left: np.ndarray, right: np.ndarray, *, treat_nan_as_mismatch: bool) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return {
            "shape_match": False,
            "left_shape": tuple(int(value) for value in left_array.shape),
            "right_shape": tuple(int(value) for value in right_array.shape),
        }
    left_nan = np.isnan(left_array)
    right_nan = np.isnan(right_array)
    comparable_mask = ~(left_nan | right_nan)
    diff = np.abs(left_array[comparable_mask] - right_array[comparable_mask])
    return {
        "shape_match": True,
        "nan_mismatch_count": int(np.count_nonzero(left_nan != right_nan)) if treat_nan_as_mismatch else 0,
        "max_absdiff": float(diff.max()) if diff.size else 0.0,
        "nonzero_absdiff_count": int(np.count_nonzero(diff > 0.0)),
        "allclose_equal_nan": bool(np.allclose(left_array, right_array, rtol=0.0, atol=0.0, equal_nan=True)),
    }


def _upper_triangle_index_pairs(signature_indices: Sequence[int]) -> list[tuple[int, int]]:
    return [
        (int(signature_indices[left]), int(signature_indices[right]))
        for left in range(len(signature_indices))
        for right in range(left + 1, len(signature_indices))
    ]


def _block_signature_indices(featurizer: Any, signature_ids: Sequence[str]) -> list[int]:
    signature_index_by_id = {str(signature_id): index for index, signature_id in enumerate(featurizer.signature_ids())}
    return [int(signature_index_by_id[str(signature_id)]) for signature_id in signature_ids]


def _constraint_values_equal(left: list[Any], right: list[Any]) -> bool:
    if len(left) != len(right):
        return False
    for left_value, right_value in zip(left, right, strict=True):
        if left_value is None or right_value is None:
            if left_value is not None or right_value is not None:
                return False
        elif not np.allclose(
            np.asarray([float(left_value)]),
            np.asarray([float(right_value)]),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            return False
    return True


def _constraint_report(
    incumbent_featurizer: Any,
    arrow_featurizer: Any,
    signature_ids: Sequence[str],
    *,
    n_jobs: int,
) -> dict[str, Any]:
    from s2and.rust_calls import get_constraints_block_upper_triangle_indexed_rust

    incumbent_indices = _block_signature_indices(incumbent_featurizer, signature_ids)
    arrow_indices = _block_signature_indices(arrow_featurizer, signature_ids)
    inc_left, inc_right, inc_values = get_constraints_block_upper_triangle_indexed_rust(
        None,
        incumbent_indices,
        start_offset=0,
        max_pairs=None,
        dont_merge_cluster_seeds=True,
        incremental_dont_use_cluster_seeds=False,
        num_threads=n_jobs,
        featurizer=incumbent_featurizer,
    )
    arrow_left, arrow_right, arrow_values = get_constraints_block_upper_triangle_indexed_rust(
        None,
        arrow_indices,
        start_offset=0,
        max_pairs=None,
        dont_merge_cluster_seeds=True,
        incremental_dont_use_cluster_seeds=False,
        num_threads=n_jobs,
        featurizer=arrow_featurizer,
    )
    value_mismatch_count = 0
    if len(inc_values) == len(arrow_values):
        for left_value, right_value in zip(inc_values, arrow_values, strict=True):
            if left_value is None or right_value is None:
                value_mismatch_count += int(left_value is not None or right_value is not None)
            else:
                value_mismatch_count += int(
                    not np.allclose(
                        np.asarray([float(left_value)]),
                        np.asarray([float(right_value)]),
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
                )
    else:
        value_mismatch_count = abs(len(inc_values) - len(arrow_values))
    return {
        "left_indices_equal": inc_left == arrow_left,
        "right_indices_equal": inc_right == arrow_right,
        "value_count": int(len(inc_values)),
        "values_equal": _constraint_values_equal(inc_values, arrow_values),
        "value_mismatch_count": int(value_mismatch_count),
    }


def _feature_constraint_report(
    incumbent_featurizer: Any,
    arrow_featurizer: Any,
    signature_ids: Sequence[str],
    *,
    n_jobs: int,
) -> dict[str, Any]:
    incumbent_pairs = _upper_triangle_index_pairs(_block_signature_indices(incumbent_featurizer, signature_ids))
    arrow_pairs = _upper_triangle_index_pairs(_block_signature_indices(arrow_featurizer, signature_ids))
    incumbent_features = np.asarray(
        incumbent_featurizer.featurize_pairs_matrix_indexed(incumbent_pairs, None, n_jobs, np.nan),
        dtype=np.float64,
    )
    arrow_features = np.asarray(
        arrow_featurizer.featurize_pairs_matrix_indexed(arrow_pairs, None, n_jobs, np.nan),
        dtype=np.float64,
    )
    return {
        "pair_count": int(len(incumbent_pairs)),
        "feature_matrix": _numeric_report(incumbent_features, arrow_features, treat_nan_as_mismatch=True),
        "constraints": _constraint_report(
            incumbent_featurizer,
            arrow_featurizer,
            signature_ids,
            n_jobs=n_jobs,
        ),
    }


def _assert_exact(report: dict[str, Any]) -> None:
    for block_key, comparison in report["distance_comparison"].items():
        if not comparison.get("allclose_equal_nan", False):
            raise AssertionError(f"distance mismatch for block {block_key}: {comparison}")
    feature_comparison = report.get("feature_constraint_comparison")
    if feature_comparison is not None:
        feature_matrix = feature_comparison["feature_matrix"]
        constraints = feature_comparison["constraints"]
        if not feature_matrix.get("allclose_equal_nan", False) or feature_matrix.get("nan_mismatch_count") != 0:
            raise AssertionError(f"feature matrix mismatch: {feature_matrix}")
        if not constraints.get("left_indices_equal", False) or not constraints.get("right_indices_equal", False):
            raise AssertionError(f"constraint index mismatch: {constraints}")
        if not constraints.get("values_equal", False):
            raise AssertionError(f"constraint mismatch: {constraints}")
    if not report.get("clusters_exact_match", False):
        raise AssertionError("cluster outputs differ")


def _cluster_partition(clusters: Mapping[Any, Sequence[Any]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(str(signature_id) for signature_id in members) for members in clusters.values())


def _write_raw_planner_indexes_and_layout(
    arrow_paths: dict[str, str],
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    """Add current raw-planner sidecars and physical-layout metrics."""

    from s2and.incremental_linking.feature_block import (
        raw_planner_arrow_physical_layout,
        write_raw_arrow_batch_lookup_indexes,
    )

    indexed_paths, index_metrics = write_raw_arrow_batch_lookup_indexes(arrow_paths, output_dir)
    return indexed_paths, index_metrics, raw_planner_arrow_physical_layout(indexed_paths)


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("S2AND_BACKEND", "rust")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.n_jobs))

    from s2and.consts import NORMALIZATION_VERSION
    from s2and.data import ANDData
    from s2and.feature_port import (
        _get_rust_featurizer,
        build_rust_featurizer_from_arrow_paths,
        clear_rust_featurizer_cache,
    )
    from s2and.incremental_linking.feature_block import (
        write_name_counts_arrow,
        write_name_counts_index,
    )
    from s2and.production_model import load_production_model
    from scripts.arrow_conversion_helpers import write_feature_block_arrow_from_anddata

    timings: dict[str, float] = {}
    meta = _load_json(args.fixture_dir / "meta.json")
    block_name = str(meta["block"])

    start = time.perf_counter()
    signatures = _load_json(_fixture_meta_path(meta, args.fixture_dir, "signatures"))
    selected_signature_ids = _select_block_signature_ids(signatures, block_name, int(args.block_size))
    timings["load_signatures_and_select_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    papers = _load_json(_fixture_meta_path(meta, args.fixture_dir, "papers"))
    if args.no_specter:
        specter_embeddings = None
    else:
        with open(_fixture_meta_path(meta, args.fixture_dir, "specter"), "rb") as infile:
            specter_embeddings = pickle.load(infile)
    filtered_signatures, filtered_papers, filtered_specter = _filter_payloads(
        signatures,
        papers,
        specter_embeddings,
        selected_signature_ids,
    )
    del signatures, papers, specter_embeddings
    timings["load_filter_papers_specter_seconds"] = time.perf_counter() - start

    cluster_seeds_require = _load_cluster_seeds_require(
        meta,
        args.fixture_dir,
        selected_signature_ids,
        enabled=bool(args.use_cluster_seeds),
    )

    start = time.perf_counter()
    dataset = ANDData(
        signatures=filtered_signatures,
        papers=filtered_papers,
        name=f"{meta['dataset']}_complete_arrow_full_predict_parity_{args.block_size}",
        mode="inference",
        specter_embeddings=filtered_specter,
        clusters=None,
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        n_jobs=int(args.n_jobs),
        load_name_counts=True,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
    )
    if cluster_seeds_require:
        dataset.cluster_seeds_require = cluster_seeds_require
        dataset.cluster_seeds_disallow = set()
    timings["anddata_subset_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    arrow_paths = write_feature_block_arrow_from_anddata(
        dataset,
        args.output_dir,
        signature_ids=selected_signature_ids,
        include_specter=not args.no_specter,
    )
    timings["write_complete_arrow_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    arrow_paths, raw_planner_index_metrics, physical_layout = _write_raw_planner_indexes_and_layout(
        arrow_paths,
        args.output_dir,
    )
    timings["write_raw_planner_indexes_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    name_artifact_dir = args.name_artifact_dir or args.output_dir
    name_counts_artifact_dir = name_artifact_dir
    name_counts_arrow_path, name_counts_arrow_metrics = write_name_counts_arrow(name_counts_artifact_dir)
    arrow_paths["name_counts"] = name_counts_arrow_path
    timings["write_name_counts_arrow_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    name_counts_index_path, name_counts_index_metrics = write_name_counts_index(name_counts_artifact_dir)
    arrow_paths["name_counts_index"] = name_counts_index_path
    timings["write_name_counts_index_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    clusterer = load_production_model(args.model_path)
    clusterer.n_jobs = int(args.n_jobs)
    clusterer.use_cache = False
    timings["load_model_seconds"] = time.perf_counter() - start

    block_dict = {block_name: selected_signature_ids}
    clear_rust_featurizer_cache()
    start = time.perf_counter()
    incumbent_dists = clusterer.make_distance_matrices(
        block_dict,
        dataset,
        partial_supervision={},
        incremental_dont_use_cluster_seeds=False,
        disable_tqdm=True,
    )
    timings["incumbent_make_dists_seconds"] = time.perf_counter() - start
    incumbent_featurizer = _get_rust_featurizer(dataset) if args.compare_features else None

    clear_rust_featurizer_cache()
    start = time.perf_counter()
    arrow_featurizer = build_rust_featurizer_from_arrow_paths(
        arrow_paths,
        expected_normalization_version=NORMALIZATION_VERSION,
        signature_ids=selected_signature_ids,
        name_tuples=getattr(dataset, "name_tuples", "filtered"),
        load_name_counts=True,
        preprocess=True,
        num_threads=int(args.n_jobs),
    )
    timings["arrow_featurizer_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    arrow_dists = clusterer.make_distance_matrices_from_rust_featurizer(
        block_dict,
        arrow_featurizer,
        partial_supervision={},
        incremental_dont_use_cluster_seeds=False,
        total_ram_bytes=int(args.total_ram_bytes),
    )
    timings["arrow_make_dists_seconds"] = time.perf_counter() - start

    feature_constraint_comparison: dict[str, Any] | None = None
    if args.compare_features:
        if incumbent_featurizer is None:
            raise RuntimeError("missing incumbent featurizer for feature comparison")
        start = time.perf_counter()
        feature_constraint_comparison = _feature_constraint_report(
            incumbent_featurizer,
            arrow_featurizer,
            selected_signature_ids,
            n_jobs=int(args.n_jobs),
        )
        timings["feature_constraint_compare_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    incumbent_clusters, _ = clusterer.predict(
        block_dict,
        dataset,
        dists=incumbent_dists,
        partial_supervision={},
        use_s2_clusters=False,
    )
    timings["incumbent_cluster_from_dists_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    arrow_clusters, _ = clusterer.predict_from_rust_featurizer(
        block_dict,
        arrow_featurizer,
        dists=arrow_dists,
        partial_supervision={},
    )
    timings["arrow_cluster_from_dists_seconds"] = time.perf_counter() - start

    distance_comparison = {
        block_key: _numeric_report(
            incumbent_dists[block_key],
            arrow_dists[block_key],
            treat_nan_as_mismatch=False,
        )
        for block_key in block_dict
    }
    incumbent_partition = _cluster_partition(incumbent_clusters)
    arrow_partition = _cluster_partition(arrow_clusters)
    clusters_exact_match = incumbent_partition == arrow_partition
    report = {
        "fixture_dir": str(args.fixture_dir),
        "output_dir": str(args.output_dir),
        "dataset": str(meta["dataset"]),
        "block": block_name,
        "block_size": int(len(selected_signature_ids)),
        "pair_count": int(len(selected_signature_ids) * (len(selected_signature_ids) - 1) // 2),
        "n_jobs": int(args.n_jobs),
        "total_ram_bytes": int(args.total_ram_bytes),
        "include_specter": not bool(args.no_specter),
        "use_cluster_seeds": bool(args.use_cluster_seeds),
        "cluster_seeds_require_count": int(len(cluster_seeds_require)),
        "arrow_paths": arrow_paths,
        "physical_layout": physical_layout,
        "raw_planner_batch_indexes": raw_planner_index_metrics,
        "name_counts_arrow_metrics": name_counts_arrow_metrics,
        "name_counts_index_metrics": name_counts_index_metrics,
        "timings_seconds": {key: float(value) for key, value in timings.items()},
        "distance_comparison": distance_comparison,
        "feature_constraint_comparison": feature_constraint_comparison,
        "clusters_exact_match": clusters_exact_match,
        "incumbent_cluster_count": int(len(incumbent_clusters)),
        "arrow_cluster_count": int(len(arrow_clusters)),
        "incumbent_cluster_sizes_top10": sorted((len(v) for v in incumbent_clusters.values()), reverse=True)[:10],
        "arrow_cluster_sizes_top10": sorted((len(v) for v in arrow_clusters.values()), reverse=True)[:10],
    }
    if not clusters_exact_match:
        report["incumbent_clusters"] = _jsonable(incumbent_clusters)
        report["arrow_clusters"] = _jsonable(arrow_clusters)
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--name-artifact-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path("s2and/data/production_model_v1.21"))
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--total-ram-bytes", type=int, default=1_000_000_000_000)
    parser.add_argument(
        "--compare-features",
        dest="compare_features",
        action="store_true",
        default=True,
        help="Compare feature matrices and constraints. This is the default parity gate.",
    )
    parser.add_argument(
        "--no-compare-features",
        dest="compare_features",
        action="store_false",
        help="Only compare distances and clusters.",
    )
    parser.add_argument("--use-cluster-seeds", action="store_true")
    parser.add_argument("--no-specter", action="store_true")
    parser.add_argument("--allow-mismatch", action="store_true")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    report = run(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.allow_mismatch:
        _assert_exact(report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"incumbent_clusters", "arrow_clusters"}}, indent=2))


if __name__ == "__main__":
    main()

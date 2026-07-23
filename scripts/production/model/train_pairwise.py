"""Train the pairwise half of a native production model bundle.

This replaces the historical pickle dump flow. The output directory is a
pairwise-only ``production_model_vX.Y`` bundle stage. Run
``train_linker_and_finalize.py`` next with this stage as input to publish the
complete model into a separate, previously nonexistent directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from hyperopt import hp
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.consts import FEATURIZER_VERSION, NAME_COUNTS_INDEX_PATH, PROJECT_ROOT_PATH  # noqa: E402
from s2and.data import ANDData  # noqa: E402
from s2and.feature_cache import cached_featurize  # noqa: E402
from s2and.featurizer import (  # noqa: E402
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_NAMELESS_FEATURE_GROUPS,
    FeaturizationInfo,
    featurize,
)
from s2and.model import Clusterer, FastCluster, PairwiseModeler  # noqa: E402
from s2and.name_tuple_artifact import load_packaged_name_tuple_artifact  # noqa: E402
from s2and.production_bundle import write_pairwise_production_bundle  # noqa: E402
from s2and.production_model import canonical_artifact_hashes  # noqa: E402

logger = logging.getLogger("s2and")

DEFAULT_SPECTER_SUFFIX = "_specter2.pkl"
DEFAULT_SIGNATURES_SUFFIX = "_signatures.json"
DEFAULT_SOURCE_DATASET_NAMES = ("aminer", "arnetminer", "inspire", "kisti", "orcid", "pubmed", "qian", "zbmath")
PAIRWISE_ONLY_DATASETS = frozenset({"medline", "augmented"})
DEFAULT_TRAIN_PAIRS_SIZE = 100_000
DEFAULT_VAL_TEST_SIZE = 10_000
DEFAULT_N_ITER = 50
DEFAULT_N_JOBS = 25
DEFAULT_CHUNK_SIZE = 100


def _sha256_file(path: str | os.PathLike[str]) -> str:
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _feature_source_hashes(
    *,
    signatures_path: str | os.PathLike[str],
    papers_path: str | os.PathLike[str],
    specter_path: str | os.PathLike[str],
) -> dict[str, str]:
    return {
        "signatures_sha256": _sha256_file(signatures_path),
        "papers_sha256": _sha256_file(papers_path),
        "specter_embeddings_sha256": _sha256_file(specter_path),
    }


def _feature_snapshot_source_key(
    dataset: ANDData,
    *,
    source_hashes: Mapping[str, str],
    name_tuples_data_sha256: str,
) -> dict[str, object]:
    name_counts_provenance = dataset.name_counts_provenance
    if name_counts_provenance is None:
        raise RuntimeError(f"Production training dataset {dataset.name!r} has no name-count manifest")
    return {
        **source_hashes,
        "name_tuples_data_sha256": name_tuples_data_sha256,
        "name_counts_manifest_sha256": name_counts_provenance["manifest_sha256"],
        "normalization_version": dataset.normalization_version,
    }


def _build_cacheable_anddata(
    anddata_kwargs: dict[str, Any],
    *,
    name_tuples_data_sha256: str,
) -> tuple[ANDData, dict[str, object]]:
    """Load a dataset only if its cache-keyed source files remain unchanged."""

    source_paths = {
        "signatures_path": anddata_kwargs["signatures"],
        "papers_path": anddata_kwargs["papers"],
        "specter_path": anddata_kwargs["specter_embeddings"],
    }
    before = _feature_source_hashes(**source_paths)
    dataset = ANDData(**anddata_kwargs)
    after = _feature_source_hashes(**source_paths)
    if before != after:
        changed = ", ".join(key.removesuffix("_sha256") for key in before if before[key] != after[key])
        raise RuntimeError(f"Production training source files changed while loading ANDData: {changed}")
    return dataset, _feature_snapshot_source_key(
        dataset,
        source_hashes=before,
        name_tuples_data_sha256=name_tuples_data_sha256,
    )


def _search_space() -> dict[str, Any]:
    return {
        "eps": hp.uniform("eps", 0, 1),
        "linkage": hp.choice("linkage", ["average"]),
    }


def _dataset_names(*, include_augmented: bool, selected_datasets: list[str] | None = None) -> list[str]:
    names = list(selected_datasets or DEFAULT_SOURCE_DATASET_NAMES)
    if include_augmented:
        if selected_datasets is not None:
            return names
        names.append("augmented")
    return names


def _resolve_dataset_file(data_dir: Path, dataset_name: str, *candidates: str) -> Path:
    dataset_dir = data_dir / dataset_name
    for candidate in candidates:
        path = dataset_dir / candidate
        if path.exists():
            return path
    joined = ", ".join(str(dataset_dir / candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find any dataset file for {dataset_name}: {joined}")


def _optional_dataset_file(data_dir: Path, dataset_name: str, *candidates: str) -> Path | None:
    dataset_dir = data_dir / dataset_name
    for candidate in candidates:
        path = dataset_dir / candidate
        if path.exists():
            return path
    return None


def _dataset_pair_paths(data_dir: Path, dataset_name: str) -> tuple[str | None, str | None, str | None, str | None]:
    if dataset_name not in PAIRWISE_ONLY_DATASETS:
        clusters_path = _resolve_dataset_file(
            data_dir,
            dataset_name,
            f"{dataset_name}_clusters.json",
            "clusters.json",
        )
        return (
            str(clusters_path),
            None,
            None,
            None,
        )
    train_pairs_path = _resolve_dataset_file(data_dir, dataset_name, "train_pairs.csv")
    val_pairs_path = _optional_dataset_file(data_dir, dataset_name, "val_pairs.csv")
    test_pairs_path = _resolve_dataset_file(data_dir, dataset_name, "test_pairs.csv")
    return (
        None,
        str(train_pairs_path),
        str(val_pairs_path) if val_pairs_path is not None else None,
        str(test_pairs_path),
    )


def _training_config(args: argparse.Namespace, dataset_names: list[str]) -> dict[str, Any]:
    return {
        "chunk_size": int(args.chunk_size),
        "data_dir": str(Path(args.data_dir)),
        "features_to_use": list(DEFAULT_FEATURE_GROUPS),
        "featurizer_version": int(FEATURIZER_VERSION),
        "include_augmented": bool(args.include_augmented),
        "n_iter": int(args.n_iter),
        "n_jobs": int(args.n_jobs),
        "nameless_features_to_use": list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        "production_version": str(args.production_version),
        "source_dataset_names": dataset_names,
        "specter_suffix": str(args.specter_suffix),
        "signatures_suffix": str(args.signatures_suffix),
        "train_pairs_size": int(args.train_pairs_size),
        "val_test_size": int(args.val_test_size),
    }


def train_pairwise_bundle(args: argparse.Namespace) -> dict[str, Any]:
    """Train pairwise models and write the pairwise production bundle stage."""

    if not args.run_full:
        raise SystemExit("pairwise production training is unbounded; pass --run-full explicitly")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir or data_dir / f"production_model_v{args.production_version}")
    if output_dir.exists():
        raise SystemExit(f"--output-dir must name a new directory: {output_dir}")

    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.n_jobs)))

    canonical_name_tuples = load_packaged_name_tuple_artifact()
    artifact_hashes = canonical_artifact_hashes()
    featurizer_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )
    nameless_featurizer_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS),
        featurizer_version=FEATURIZER_VERSION,
    )
    if args.negative_one_for_nan:
        monotone_constraints = None
        nameless_monotone_constraints = None
        nan_value = -1.0
    else:
        monotone_constraints = featurizer_info.lightgbm_monotone_constraints
        nameless_monotone_constraints = nameless_featurizer_info.lightgbm_monotone_constraints
        nan_value = np.nan

    started = time.perf_counter()
    dataset_names = _dataset_names(
        include_augmented=bool(args.include_augmented),
        selected_datasets=[str(dataset) for dataset in args.datasets] if args.datasets else None,
    )
    datasets: dict[str, dict[str, Any]] = {}
    for dataset_name in tqdm(dataset_names, desc="Processing datasets and fitting base models"):
        logger.info("processing dataset %s", dataset_name)
        clusters_path, train_pairs_path, val_pairs_path, test_pairs_path = _dataset_pair_paths(data_dir, dataset_name)
        signatures_path = str(
            _resolve_dataset_file(
                data_dir,
                dataset_name,
                f"{dataset_name}{args.signatures_suffix}",
                args.signatures_suffix.lstrip("_"),
                "signatures.json",
            )
        )
        papers_path = str(
            _resolve_dataset_file(
                data_dir,
                dataset_name,
                f"{dataset_name}_papers.json",
                "papers.json",
            )
        )
        specter_path = str(
            _resolve_dataset_file(
                data_dir,
                dataset_name,
                f"{dataset_name}{args.specter_suffix}",
                args.specter_suffix.lstrip("_"),
                "specter.pickle",
            )
        )

        anddata_kwargs: dict[str, Any] = {
            "signatures": signatures_path,
            "papers": papers_path,
            "name": dataset_name,
            "mode": "train",
            "specter_embeddings": specter_path,
            "clusters": clusters_path,
            "train_pairs": train_pairs_path,
            "val_pairs": val_pairs_path,
            "test_pairs": test_pairs_path,
            "train_pairs_size": int(args.train_pairs_size),
            "val_pairs_size": int(args.val_test_size),
            "test_pairs_size": int(args.val_test_size),
            "name_counts_index": NAME_COUNTS_INDEX_PATH,
            "preprocess": True,
        }

        source_key = None
        if args.feature_cache_dir is None:
            anddata = ANDData(**anddata_kwargs)
        else:
            anddata, source_key = _build_cacheable_anddata(
                anddata_kwargs,
                name_tuples_data_sha256=canonical_name_tuples.data_sha256,
            )
        if anddata.name_tuples != canonical_name_tuples.pairs:
            raise ValueError(f"Production training dataset {dataset_name!r} does not use the canonical name tuples")

        if source_key is None:
            train, val, test = featurize(
                anddata,
                featurizer_info,
                n_jobs=int(args.n_jobs),
                chunk_size=int(args.chunk_size),
                nameless_featurizer_info=nameless_featurizer_info,
                nan_value=nan_value,
            )
        else:
            train, val, test = cached_featurize(
                anddata,
                featurizer_info,
                source_key=source_key,
                cache_dir=args.feature_cache_dir,
                n_jobs=int(args.n_jobs),
                chunk_size=int(args.chunk_size),
                nameless_featurizer_info=nameless_featurizer_info,
                nan_value=nan_value,
            )

        if train is None or val is None or test is None:
            raise RuntimeError(f"Expected train/val/test features for {dataset_name}")
        X_train, y_train, nameless_X_train = train
        X_val, y_val, nameless_X_val = val
        X_test, y_test, nameless_X_test = test
        datasets[dataset_name] = {
            "anddata": anddata,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "nameless_X_train": nameless_X_train,
            "nameless_X_val": nameless_X_val,
            "nameless_X_test": nameless_X_test,
        }

    anddatas = [
        datasets[dataset_name]["anddata"]
        for dataset_name in dataset_names
        if dataset_name not in PAIRWISE_ONLY_DATASETS
    ]
    X_train = np.vstack([datasets[dataset_name]["X_train"] for dataset_name in dataset_names])
    y_train = np.hstack([datasets[dataset_name]["y_train"] for dataset_name in dataset_names])
    validation_dataset_names = [dataset_name for dataset_name in dataset_names if dataset_name != "augmented"]
    X_val = np.vstack([datasets[dataset_name]["X_val"] for dataset_name in validation_dataset_names])
    y_val = np.hstack([datasets[dataset_name]["y_val"] for dataset_name in validation_dataset_names])
    nameless_X_train = np.vstack([datasets[dataset_name]["nameless_X_train"] for dataset_name in dataset_names])
    nameless_X_val = np.vstack([datasets[dataset_name]["nameless_X_val"] for dataset_name in validation_dataset_names])

    logger.info("fitting pairwise model")
    union_classifier = PairwiseModeler(
        n_iter=int(args.n_iter),
        n_jobs=int(args.n_jobs),
        monotone_constraints=monotone_constraints,
    )
    union_classifier.fit(X_train, y_train, X_val, y_val)

    logger.info("fitting nameless pairwise model")
    nameless_union_classifier = PairwiseModeler(
        n_iter=int(args.n_iter),
        n_jobs=int(args.n_jobs),
        monotone_constraints=nameless_monotone_constraints,
    )
    nameless_union_classifier.fit(nameless_X_train, y_train, nameless_X_val, y_val)

    logger.info("fitting clustering threshold")
    union_clusterer = Clusterer(
        featurizer_info,
        union_classifier.classifier,
        cluster_model=FastCluster(),
        search_space=_search_space(),
        n_iter=int(args.cluster_n_iter),
        n_jobs=int(args.n_jobs),
        nameless_classifier=nameless_union_classifier.classifier,
        nameless_featurizer_info=nameless_featurizer_info,
    )
    union_clusterer.feature_contract.update(artifact_hashes)
    union_clusterer.fit(anddatas)
    best_params = union_clusterer.best_params
    if best_params is None:
        raise RuntimeError("Clusterer fitting did not produce best clustering parameters.")

    training_summary = {
        "best_clustering_params": dict(best_params),
        "elapsed_seconds": round(float(time.perf_counter() - started), 3),
        "main_train_rows": int(X_train.shape[0]),
        "main_val_rows": int(X_val.shape[0]),
        "nameless_train_rows": int(nameless_X_train.shape[0]),
        "nameless_val_rows": int(nameless_X_val.shape[0]),
        "output_dir": str(output_dir),
    }
    bundle_summary = write_pairwise_production_bundle(
        union_clusterer,
        output_dir,
        bundle_version=str(args.production_version),
        source_model_version=str(args.production_version),
        pairwise_training_config=_training_config(args, dataset_names),
        pairwise_training_summary=training_summary,
    )
    result = {
        "bundle_dir": str(bundle_summary.bundle_dir),
        "bundle_status": bundle_summary.bundle_status,
        "bundle_version": bundle_summary.bundle_version,
        "manifest_path": str(bundle_summary.manifest_path),
        "training_summary": training_summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-version", required=True, help="Version suffix for production_model_vX.Y.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path(PROJECT_ROOT_PATH) / "data")
    parser.add_argument("--specter-suffix", default=DEFAULT_SPECTER_SUFFIX)
    parser.add_argument("--signatures-suffix", default=DEFAULT_SIGNATURES_SUFFIX)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--cluster-n-iter", type=int, default=25)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--train-pairs-size", type=int, default=DEFAULT_TRAIN_PAIRS_SIZE)
    parser.add_argument("--val-test-size", type=int, default=DEFAULT_VAL_TEST_SIZE)
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names for smoke tests.")
    parser.add_argument("--include-augmented", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Optional featurized-split snapshot cache directory for repeated training experiments. "
        "Prefer a local unsynced directory; snapshots are content-addressed NPZ files.",
    )
    parser.add_argument("--negative-one-for-nan", action="store_true")
    parser.add_argument("--run-full", action="store_true", help="Explicitly allow full production pairwise training.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.DEBUG)
    parser = build_parser()
    train_pairwise_bundle(parser.parse_args(argv))


if __name__ == "__main__":
    main()

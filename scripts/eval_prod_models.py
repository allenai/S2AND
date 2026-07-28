# mypy: ignore-errors

"""
Evaluate current SPECTER2 production bundles and embedding retraining experiments.

Current production evaluation is SPECTER2-only. SPECTER1 production belongs to
S2AND v1.21 and earlier; SPECTER1 remains available here only as an explicit
``--train --specter-suffixes _specter.pickle`` research comparison.

================================================================================
Which bundle the numbers below refer to
================================================================================

The expected B3 numbers in this docstring are all measured against the
**s2and-mini** bundle (`--dataset mini`). Mini is a curated subset of the full
S2AND benchmark; Ai2 employees can find it at
`s3://ai2-s2-research/s2and/s2and-mini/`.

The mini bundle shares signature content with the full benchmark bundle but
re-keys signature ids (mini ids are the full ids prefixed with `<dataset>_`).
For three datasets (arnetminer, pubmed, qian) the mini signature set is
identical to the full signature set, so the docstring numbers also reproduce
on `--dataset full`. For the other three (inspire, kisti, zbmath) mini is a
strict subset of full, so `--dataset full` evaluates a different (larger,
harder) signature set and reports different B3:

    Dataset      mini sigs      full sigs    mini == full?
    arnetminer       7,144          7,144    yes
    pubmed           2,871          2,871    yes
    qian             6,542          6,542    yes
    inspire          9,305        536,564    no (mini is a 1.7% subset)
    kisti           37,779         40,383    no (mini is a 94% subset)
    zbmath           5,327         15,181    no (mini is a 35% subset)

If you ran `--dataset full --use-arrow` and saw inspire/kisti/zbmath drift by
~0.4-1.2 F1 from the numbers below, that is the bundle difference, not a
regression. To reproduce the docstring numbers exactly, run `--dataset mini`.

================================================================================
With retraining (random seed 42, dataset=mini)
================================================================================

Performance with SPECTERv1 data, on arnetminer (B3): (0.922, 0.985, 0.952)
Performance with SPECTERv2 data, on arnetminer (B3): (0.93, 0.988, 0.958)

Performance with SPECTERv1 data, on inspire (B3): (0.958, 0.974, 0.966)
Performance with SPECTERv2 data, on inspire (B3): (0.995, 0.959, 0.977)

Performance with SPECTERv1 data, on kisti (B3): (0.951, 0.971, 0.961)
Performance with SPECTERv2 data, on kisti (B3): (0.946, 0.98, 0.963)

Performance with SPECTERv1 data, on pubmed (B3): (0.849, 0.988, 0.913)
Performance with SPECTERv2 data, on pubmed (B3): (0.86, 0.988, 0.92)

Performance with SPECTERv1 data, on qian (B3): (0.936, 0.943, 0.94)
Performance with SPECTERv2 data, on qian (B3): (0.95, 0.964, 0.957)

Performance with SPECTERv1 data, on zbmath (B3): (0.966, 0.984, 0.975)
Performance with SPECTERv2 data, on zbmath (B3): (0.975, 0.991, 0.983)

================================================================================
Without retraining (production model artifacts, dataset=mini) — HISTORICAL
================================================================================

SPECTER2 numbers verified 2026-05-21 against the ANDData/Python backend; all
six datasets reproduce bit-identically on 2026-05-28 by passing the evaluated
bundle explicitly with `--specter2-model-path`.

These numbers predate the split contract: they were measured with an
evaluator-owned seed 42 against a bundle trained under seed 1111, so their
"test" blocks overlapped that bundle's training split at the ~80% base rate.
Bundle evaluation now derives its split seed from the bundle's recorded
``data_random_seed`` (the trainer's actual held-out split) and rejects an
explicit ``--seed``, so freshly measured numbers are genuinely held out and
will not match the values below.

Performance with SPECTERv2 data, on arnetminer (B3): (0.946, 0.982, 0.963)

Performance with SPECTERv2 data, on inspire (B3): (0.998, 0.927, 0.961)

Performance with SPECTERv2 data, on kisti (B3): (0.96, 0.96, 0.96)

Performance with SPECTERv2 data, on pubmed (B3): (1.0, 0.892, 0.943)

Performance with SPECTERv2 data, on qian (B3): (0.978, 0.964, 0.971)

Performance with SPECTERv2 data, on zbmath (B3): (0.961, 0.992, 0.976)

================================================================================
Full-bundle Arrow numbers (no retraining, SPECTER2, --dataset full --use-arrow)
================================================================================

For reference, when evaluating the full bundle through the Arrow + Rust
production path (`--dataset full --use-arrow --specter2-model-path
/path/to/bundle`, evaluator seed 42, pre-split-contract), measured 2026-05-28:

Performance on arnetminer (B3): (0.946, 0.982, 0.963)    # matches mini
Performance on inspire    (B3): (0.983, 0.932, 0.957)    # mini ⊂ full
Performance on kisti      (B3): (0.942, 0.968, 0.955)    # mini ⊂ full
Performance on pubmed     (B3): (1.0,   0.892, 0.943)    # matches mini
Performance on qian       (B3): (0.978, 0.964, 0.971)    # matches mini
Performance on zbmath     (B3): (0.945, 0.985, 0.964)    # mini ⊂ full

================================================================================
Usage
================================================================================

    # Evaluate one explicit SPECTER2 model on inventors_s2and
    uv run python scripts/eval_prod_models.py --dataset inventors_s2and \
        --specter2-model-path /path/to/production_model_bundle

    # Evaluate a bundle on mini via the ANDData/Python backend (split seed
    # comes from the bundle's recorded data_random_seed)
    S2AND_BACKEND=python uv run python scripts/eval_prod_models.py \
        --dataset mini --no-arrow \
        --specter2-model-path /path/to/production_model_bundle

    # Evaluate the full released benchmark via Arrow + Rust production path
    uv run python scripts/eval_prod_models.py --dataset full --use-arrow \
        --specter2-model-path /path/to/production_model_bundle

    # Retrain SPECTER2 from scratch instead of using a production model
    uv run python scripts/eval_prod_models.py --train

    # Opt into a historical SPECTER1-vs-SPECTER2 retraining comparison
    uv run python scripts/eval_prod_models.py --train \
        --specter-suffixes _specter.pickle _specter2.pkl

    # Override seed / n_jobs (--seed is --train only; bundle evaluation always
    # uses the bundle's recorded data_random_seed)
    uv run python scripts/eval_prod_models.py --train --seed 42 --n_jobs 8
"""

import argparse
import contextlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np

from s2and.arrow_inputs import ArrowDataset, read_arrow_collection_root

TRAIN_MODE_ANDDATA_CURRENT = "anddata-current"
TRAIN_MODE_ANDDATA_PYTHON = "anddata-python"
TRAIN_MODE_ARROW_RUST = "arrow-rust"
TRAIN_MODE_CHOICES = (
    TRAIN_MODE_ANDDATA_CURRENT,
    TRAIN_MODE_ANDDATA_PYTHON,
    TRAIN_MODE_ARROW_RUST,
)
TRAIN_MODE_COMPARISON = (
    TRAIN_MODE_ANDDATA_PYTHON,
    TRAIN_MODE_ARROW_RUST,
)
SPECTER1_SUFFIX = "_specter.pickle"
SPECTER2_SUFFIX = "_specter2.pkl"
SPECTER_SUFFIXES = (SPECTER1_SUFFIX, SPECTER2_SUFFIX)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SPECTER2 production bundles or retrain embedding variants")
    parser.add_argument(
        "--dataset",
        choices=["inventors_s2and", "mini", "full"],
        default="inventors_s2and",
        help="Which dataset(s) to evaluate on (default: inventors_s2and)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for --train research runs (default: 42). Production-bundle "
            "evaluation derives its split seed from the bundle's recorded "
            "data_random_seed and rejects an explicit --seed."
        ),
    )
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs (default: 4)")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset names to evaluate, e.g. --datasets zbmath qian.",
    )
    parser.add_argument(
        "--specter-suffixes",
        nargs="*",
        choices=SPECTER_SUFFIXES,
        default=None,
        help=(
            "Embedding suffixes to evaluate. Defaults to SPECTER2. "
            "SPECTER1 is accepted only with --train for historical research comparisons."
        ),
    )
    parser.add_argument(
        "--specter2-model-path",
        type=Path,
        default=None,
        help="Explicit model bundle for _specter2.pkl production evaluation.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain models from scratch instead of loading explicit model bundles.",
    )
    parser.add_argument(
        "--train-modes",
        nargs="*",
        choices=TRAIN_MODE_CHOICES,
        default=None,
        help=(
            "Training backend modes to run when --train is set. The default preserves the historical "
            "ANDData training behavior. Use --compare-train-modes for the qian parity harness."
        ),
    )
    parser.add_argument(
        "--compare-train-modes",
        action="store_true",
        help=("Run the qian-only pairwise training parity harness: ANDData/Python and retained Arrow/Rust."),
    )
    parser.add_argument(
        "--use-arrow",
        action="store_true",
        help=(
            "Force production-model evaluation through direct Arrow/Rust predict_from_arrow. "
            "Arrow is used automatically for supported evals when complete artifacts exist. Not supported with --train."
        ),
    )
    parser.add_argument(
        "--no-arrow",
        action="store_true",
        help="Disable automatic Arrow/Rust evaluation even when Arrow artifacts exist.",
    )
    parser.add_argument(
        "--arrow-data-root",
        type=Path,
        default=None,
        help="Explicit Arrow data root for evaluation or Arrow/Rust training.",
    )
    parser.add_argument(
        "--json-data-root",
        type=Path,
        default=None,
        help="Explicit JSON/pickle dataset root for evaluation or ANDData training.",
    )
    parser.add_argument(
        "--name-counts-index-root",
        type=Path,
        default=None,
        help="Explicit manifest-backed name-count index used by JSON/ANDData evaluation and training.",
    )
    parser.add_argument(
        "--name-tuples-path",
        type=Path,
        default=None,
        help="Explicit canonical name-tuple data file used by JSON/ANDData evaluation and training.",
    )
    parser.add_argument("--train-pairs-size", type=int, default=100000)
    parser.add_argument("--val-pairs-size", type=int, default=10000)
    parser.add_argument("--test-pairs-size", type=int, default=10000)
    parser.add_argument("--pairwise-n-iter", type=int, default=25)
    parser.add_argument("--cluster-n-iter", type=int, default=25)
    parser.add_argument(
        "--fixed-lightgbm-params",
        action="store_true",
        help="Disable pairwise LightGBM hyperopt and fit the default deterministic LightGBM parameters.",
    )
    parser.add_argument(
        "--fixed-cluster-eps",
        type=float,
        default=None,
        help="Disable cluster hyperopt and use this fixed FastCluster eps value.",
    )
    return parser


def _resolve_requested_datasets(
    default_datasets: list[str],
    requested_datasets: list[str] | None,
    dataset_label: str,
) -> list[str]:
    if not requested_datasets:
        return list(default_datasets)
    requested = [str(dataset_name) for dataset_name in requested_datasets]
    unknown_datasets = sorted(set(requested) - set(default_datasets))
    if unknown_datasets:
        raise ValueError(f"Unknown dataset(s) for --dataset {dataset_label}: {unknown_datasets}")
    return requested


def _resolve_requested_specter_suffixes(default_suffixes: list[str], requested_suffixes: list[str] | None) -> list[str]:
    if not requested_suffixes:
        return list(default_suffixes)
    return [str(suffix) for suffix in requested_suffixes]


def _resolve_requested_train_modes(
    requested_modes: Sequence[str] | None,
    *,
    compare_train_modes: bool,
) -> list[str]:
    if compare_train_modes:
        if requested_modes:
            raise ValueError("Pass either --compare-train-modes or --train-modes, not both")
        return list(TRAIN_MODE_COMPARISON)
    if not requested_modes:
        return [TRAIN_MODE_ANDDATA_CURRENT]
    return [str(mode) for mode in requested_modes]


def _validate_train_mode_scope(train_modes: Sequence[str], datasets: Sequence[str]) -> None:
    if any(mode != TRAIN_MODE_ANDDATA_CURRENT for mode in train_modes) and list(datasets) != ["qian"]:
        raise ValueError("Non-default training modes are currently qian-only; pass --datasets qian")


def _assert_training_mode_metrics_identical(
    results: Mapping[tuple[str, str], Sequence[Mapping[str, tuple]]],
    *,
    specter_suffixes_to_check: Sequence[str],
    train_modes: Sequence[str],
    datasets: Sequence[str],
) -> None:
    if len(train_modes) <= 1:
        return
    baseline_mode = str(train_modes[0])
    for specter_suffix in specter_suffixes_to_check:
        baseline_metrics = results[(str(specter_suffix), baseline_mode)]
        for train_mode in train_modes[1:]:
            observed_metrics = results[(str(specter_suffix), str(train_mode))]
            for dataset_index, dataset_name in enumerate(datasets):
                expected = baseline_metrics[dataset_index]
                observed = observed_metrics[dataset_index]
                if set(expected) != set(observed):
                    raise AssertionError(
                        f"Training mode metrics keys differ for dataset={dataset_name} "
                        f"specter_suffix={specter_suffix} mode={train_mode}: "
                        f"expected={sorted(expected)} observed={sorted(observed)}"
                    )
                for metric_name, expected_value in expected.items():
                    observed_value = observed[metric_name]
                    if not np.allclose(
                        np.asarray(expected_value, dtype=np.float64),
                        np.asarray(observed_value, dtype=np.float64),
                        equal_nan=True,
                        atol=0.0,
                        rtol=0.0,
                    ):
                        raise AssertionError(
                            "Training mode metrics differ for "
                            f"dataset={dataset_name} specter_suffix={specter_suffix} "
                            f"metric={metric_name} baseline_mode={baseline_mode} mode={train_mode}: "
                            f"expected={expected_value} observed={observed_value}"
                        )


@contextlib.contextmanager
def _temporary_s2and_backend(backend: str | None):
    previous = os.environ.get("S2AND_BACKEND")
    if backend is not None:
        os.environ["S2AND_BACKEND"] = backend
    try:
        yield
    finally:
        if backend is not None:
            if previous is None:
                os.environ.pop("S2AND_BACKEND", None)
            else:
                os.environ["S2AND_BACKEND"] = previous


def _backend_for_train_mode(train_mode: str) -> str | None:
    if train_mode == TRAIN_MODE_ANDDATA_PYTHON:
        return "python"
    return None


def _supports_arrow_eval(dataset_label: str) -> bool:
    return dataset_label in {"mini", "full"}


def _should_use_arrow_eval(
    *,
    force_arrow: bool,
    no_arrow: bool,
    arrow_available: bool,
) -> bool:
    if force_arrow:
        return True
    return bool(arrow_available and not no_arrow)


specter_suffixes = [SPECTER2_SUFFIX]


def resolve_dataset_file(data_root: str, dataset_name: str, preferred_name: str, fallback_name: str) -> str:
    """Try preferred filename, then fallback, raising FileNotFoundError if neither exists."""
    preferred_path = os.path.join(data_root, dataset_name, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path
    fallback_path = os.path.join(data_root, dataset_name, fallback_name)
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Missing dataset file. Tried '{preferred_path}' and '{fallback_path}'.")


def resolve_arrow_dataset_root(arrow_root: str, dataset_name: str) -> str:
    """Resolve a dataset declared by one explicit Arrow collection root."""

    root = Path(arrow_root)
    root_manifest = root / "manifest.json"
    if not root_manifest.is_file():
        raise FileNotFoundError(f"Arrow root manifest does not exist: {root_manifest}")
    dataset_manifests, _replay_bundles, _release_version = read_arrow_collection_root(root_manifest)
    manifest_path = dataset_manifests.get(dataset_name)
    if manifest_path is None:
        raise FileNotFoundError(f"Arrow root manifest does not declare dataset {dataset_name!r}: {root_manifest}")
    return str(manifest_path.parent)


def bundle_data_random_seed(model_path: Path) -> int:
    """Return the data-split seed the pairwise trainer recorded in a bundle.

    Production-bundle evaluation reuses the trainer's seed. This reproduces
    its split only while dataset bytes and ordering are identical; release
    evaluation still requires persisted split identities.

    Args:
        model_path: Production bundle directory (``--specter2-model-path``).

    Returns:
        The ``data_random_seed`` from the bundle's pairwise training config.

    Raises:
        FileNotFoundError: If the bundle records no pairwise training config.
        ValueError: If the config lacks an integer ``data_random_seed``.
    """
    config_path = Path(model_path) / "reproducibility" / "pairwise_training_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Production bundle records no training split seed (missing {config_path}); "
            "evaluation refuses to guess a split for a bundle trained under an unknown one"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = config.get("data_random_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Pairwise training config at {config_path} lacks an integer data_random_seed")
    return seed


def resolve_arrow_dataset(
    arrow_root: str,
    dataset_name: str,
    specter_suffix: str,
) -> ArrowDataset:
    """Open the exact immutable Arrow generation for one evaluation dataset."""

    dataset_root = resolve_arrow_dataset_root(arrow_root, dataset_name)
    manifest = json.loads((Path(dataset_root) / "manifest.json").read_text(encoding="utf-8"))
    expected_specter_name = "specter2.arrow" if specter_suffix == SPECTER2_SUFFIX else "specter.arrow"
    declared_specter = Path(str(manifest.get("paths", {}).get("specter", ""))).name
    if declared_specter != expected_specter_name:
        raise FileNotFoundError(
            f"Arrow dataset {dataset_name!r} declares {declared_specter!r}, not requested {expected_specter_name!r}"
        )
    return ArrowDataset.open(
        dataset_root,
        require_specter=True,
        require_name_counts_index=True,
    )


def arrow_datasets_available(arrow_root: str | None, datasets: list[str], specter_suffixes: list[str]) -> bool:
    return first_missing_arrow_dataset_error(arrow_root, datasets, specter_suffixes) is None


def first_missing_arrow_dataset_error(
    arrow_root: str | None,
    datasets: list[str],
    specter_suffixes: list[str],
) -> FileNotFoundError | None:
    if arrow_root is None:
        return FileNotFoundError("Missing Arrow data root")
    for dataset_name in datasets:
        for specter_suffix in specter_suffixes:
            try:
                with resolve_arrow_dataset(arrow_root, dataset_name, specter_suffix):
                    pass
            except (FileNotFoundError, ValueError) as exc:
                return FileNotFoundError(
                    f"Missing Arrow files for dataset={dataset_name!r}, specter_suffix={specter_suffix!r}: {exc}"
                )
    return None


def read_arrow_s2_blocks(arrow_dataset: ArrowDataset) -> dict[str, list[str]]:
    import pyarrow as pa

    with arrow_dataset.use() as lease, lease.open_file("signatures") as infile:
        with pa.PythonFile(infile, mode="r") as source:
            table = pa.ipc.open_file(source).read_all().select(["signature_id", "author_block"])
    block_dict: dict[str, list[str]] = defaultdict(list)
    signature_ids = table.column("signature_id").to_pylist()
    author_blocks = table.column("author_block").to_pylist()
    for signature_id, author_block in zip(signature_ids, author_blocks, strict=True):
        block_dict[str(author_block)].append(str(signature_id))
    return dict(block_dict)


def split_blocks_like_anddata(
    blocks_dict: dict[str, list[str]],
    *,
    random_seed: int,
    num_clusters_for_block_size: int = 1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    block_ids = []
    block_sizes = []
    # Match ANDData.split_blocks_helper exactly. This seeded stratified split is
    # order-sensitive; sorting changes pinned production-eval test sets.
    for block_id in blocks_dict:
        signatures = blocks_dict[block_id]
        block_ids.append(block_id)
        block_sizes.append(len(signatures))
    if len(block_ids) == 0:
        return {}, {}, {}
    y_group = (
        KMeans(n_clusters=num_clusters_for_block_size, random_state=random_seed, n_init=10)
        .fit(np.array(block_sizes).reshape(-1, 1))
        .labels_
    )
    train_blocks, val_test_blocks, _, val_test_length = train_test_split(
        block_ids,
        y_group,
        test_size=val_ratio + test_ratio,
        stratify=y_group,
        random_state=random_seed,
    )
    val_blocks, test_blocks = train_test_split(
        val_test_blocks,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=val_test_length,
        random_state=random_seed,
    )
    return (
        {block_id: blocks_dict[block_id] for block_id in train_blocks},
        {block_id: blocks_dict[block_id] for block_id in val_blocks},
        {block_id: blocks_dict[block_id] for block_id in test_blocks},
    )


def read_signature_to_cluster_id(clusters_path: str | Path) -> dict[str, str]:
    with open(clusters_path, encoding="utf-8") as infile:
        clusters = json.load(infile)
    signature_to_cluster_id = {}
    for cluster_id, cluster_info in clusters.items():
        for signature_id in cluster_info["signature_ids"]:
            signature_to_cluster_id[str(signature_id)] = str(cluster_id)
    return signature_to_cluster_id


def construct_cluster_to_signatures(
    signature_to_cluster_id: dict[str, str],
    block_dict: dict[str, list[str]],
) -> dict[str, list[str]]:
    cluster_to_signatures: dict[str, list[str]] = defaultdict(list)
    missing_signature_ids: list[str] = []
    for signatures in block_dict.values():
        for signature_id in signatures:
            signature_key = str(signature_id)
            cluster_id = signature_to_cluster_id.get(signature_key)
            if cluster_id is None:
                missing_signature_ids.append(signature_key)
                continue
            cluster_to_signatures[cluster_id].append(signature_key)
    if missing_signature_ids:
        raise ValueError(
            "clusters.json is missing cluster assignments for "
            f"{len(missing_signature_ids)} evaluated signature(s): {missing_signature_ids[:10]}"
        )
    return dict(cluster_to_signatures)


def _arrow_clusters_path(arrow_dataset: ArrowDataset) -> Path:
    path = arrow_dataset.root / f"{arrow_dataset.root.name}_clusters.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Arrow evaluation clusters: {path}")
    return path


def cluster_eval_arrow(
    arrow_dataset: ArrowDataset,
    clusterer,
    *,
    random_seed: int,
    n_jobs: int,
    split: str = "test",
    total_ram_bytes: int = 1_000_000_000_000,
    batching_threshold: int | None = None,
) -> tuple[dict[str, tuple], dict[str, tuple[float, float, float]]]:
    import numpy as np

    from s2and.eval import b3_precision_recall_fscore, pairwise_precision_recall_fscore

    train_block_dict, val_block_dict, test_block_dict = split_blocks_like_anddata(
        read_arrow_s2_blocks(arrow_dataset),
        random_seed=random_seed,
    )
    if split == "test":
        block_dict = test_block_dict
    elif split == "val":
        block_dict = val_block_dict
    elif split == "train":
        block_dict = train_block_dict
    else:
        raise ValueError("Split must be one of: train, val, test")
    signature_to_cluster_id = read_signature_to_cluster_id(_arrow_clusters_path(arrow_dataset))
    cluster_to_signatures = construct_cluster_to_signatures(signature_to_cluster_id, block_dict)
    pred_clusters, _ = clusterer.predict_from_arrow(
        block_dict,
        arrow_dataset,
        total_ram_bytes=total_ram_bytes,
        batching_threshold=batching_threshold,
        name_tuples=None,
    )
    (
        b3_p,
        b3_r,
        b3_f1,
        b3_metrics_per_signature,
        pred_bigger_ratios,
        true_bigger_ratios,
    ) = b3_precision_recall_fscore(cluster_to_signatures, pred_clusters)
    metrics: dict[str, tuple] = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}
    metrics["Cluster (P, R F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "clusters"
    )
    metrics["Cluster Macro (P, R, F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, block_dict, "cmacro"
    )

    def _mean_or_nan(xs):
        if len(xs) == 0:
            return float("nan")
        return float(np.round(np.mean(xs), 2))

    metrics["Pred bigger ratio (mean, count)"] = (_mean_or_nan(pred_bigger_ratios), len(pred_bigger_ratios))
    metrics["True bigger ratio (mean, count)"] = (_mean_or_nan(true_bigger_ratios), len(true_bigger_ratios))
    return metrics, b3_metrics_per_signature


@dataclass(frozen=True)
class PairwiseTrainingSplits:
    train_pairs: list[tuple[str, str, int | float]]
    val_pairs: list[tuple[str, str, int | float]]
    test_pairs: list[tuple[str, str, int | float]]
    train_block_dict: dict[str, list[str]]
    val_block_dict: dict[str, list[str]]
    test_block_dict: dict[str, list[str]]
    signature_to_cluster_id: dict[str, str]


def build_eval_anddata(
    *,
    data_root: str,
    dataset_name: str,
    name_counts_index_root: Path,
    name_tuples: frozenset[tuple[str, str]],
    specter_suffix: str,
    n_jobs: int,
    random_seed: int,
    train_pairs_size: int,
    val_pairs_size: int,
    test_pairs_size: int,
) -> Any:
    from s2and.data import ANDData

    return ANDData(
        signatures=resolve_dataset_file(data_root, dataset_name, f"{dataset_name}_signatures.json", "signatures.json"),
        papers=resolve_dataset_file(data_root, dataset_name, f"{dataset_name}_papers.json", "papers.json"),
        name=dataset_name,
        mode="train",
        specter_embeddings=resolve_dataset_file(
            data_root,
            dataset_name,
            f"{dataset_name}{specter_suffix}",
            specter_suffix.lstrip("_"),
        ),
        clusters=resolve_dataset_file(data_root, dataset_name, f"{dataset_name}_clusters.json", "clusters.json"),
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=train_pairs_size,
        val_pairs_size=val_pairs_size,
        test_pairs_size=test_pairs_size,
        n_jobs=n_jobs,
        name_counts_index=name_counts_index_root,
        preprocess=True,
        random_seed=random_seed,
        name_tuples=name_tuples,
    )


def pair_splits_from_anddata(anddata: Any) -> PairwiseTrainingSplits:
    train_block_dict, val_block_dict, test_block_dict = anddata.split_cluster_signatures()
    train_pairs, val_pairs, test_pairs = anddata.split_pairs(train_block_dict, val_block_dict, test_block_dict)
    return PairwiseTrainingSplits(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_block_dict=train_block_dict,
        val_block_dict=val_block_dict,
        test_block_dict=test_block_dict,
        signature_to_cluster_id={
            str(key): str(value) for key, value in (anddata.signature_to_cluster_id or {}).items()
        },
    )


def _sample_within_block_random_pairs(
    blocks: Mapping[str, Sequence[str]],
    signature_to_cluster_id: Mapping[str, str],
    *,
    sample_size: int,
    random_seed: int,
) -> list[tuple[str, str, int | float]]:
    from s2and.sampling import random_sampling

    possible: list[tuple[str, str, int | float]] = []
    for signatures in blocks.values():
        signature_ids = [str(signature_id) for signature_id in signatures]
        for index, left in enumerate(signature_ids):
            for right in signature_ids[index + 1 :]:
                possible.append((left, right, int(signature_to_cluster_id[left] == signature_to_cluster_id[right])))
    return random_sampling(possible, min(len(possible), int(sample_size)), int(random_seed))


def pair_splits_from_arrow_dataset(
    arrow_dataset: ArrowDataset,
    *,
    random_seed: int,
    train_pairs_size: int,
    val_pairs_size: int,
    test_pairs_size: int,
) -> PairwiseTrainingSplits:
    train_block_dict, val_block_dict, test_block_dict = split_blocks_like_anddata(
        read_arrow_s2_blocks(arrow_dataset),
        random_seed=random_seed,
    )
    signature_to_cluster_id = read_signature_to_cluster_id(_arrow_clusters_path(arrow_dataset))
    return PairwiseTrainingSplits(
        train_pairs=_sample_within_block_random_pairs(
            train_block_dict,
            signature_to_cluster_id,
            sample_size=train_pairs_size,
            random_seed=random_seed,
        ),
        val_pairs=_sample_within_block_random_pairs(
            val_block_dict,
            signature_to_cluster_id,
            sample_size=val_pairs_size,
            random_seed=random_seed,
        ),
        test_pairs=_sample_within_block_random_pairs(
            test_block_dict,
            signature_to_cluster_id,
            sample_size=test_pairs_size,
            random_seed=random_seed,
        ),
        train_block_dict=train_block_dict,
        val_block_dict=val_block_dict,
        test_block_dict=test_block_dict,
        signature_to_cluster_id=signature_to_cluster_id,
    )


def _feature_indices(featurizer_info: Any) -> list[int]:
    indices: set[int] = set()
    for feature_name in featurizer_info.features_to_use:
        indices.update(featurizer_info.feature_group_to_index[feature_name])
    return sorted(indices)


def _feature_tuple_from_rust_featurizer(
    rust_featurizer: Any,
    pairs: Sequence[tuple[str, str, int | float]],
    *,
    featurizer_info: Any,
    nameless_featurizer_info: Any | None,
    n_jobs: int,
    nan_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    selected_indices = _feature_indices(featurizer_info)
    nameless_indices = _feature_indices(nameless_featurizer_info) if nameless_featurizer_info is not None else []
    labels = np.asarray([float(pair[2]) for pair in pairs], dtype=np.float64)
    if not pairs:
        nameless_empty = (
            np.empty((0, len(nameless_indices)), dtype=np.float64) if nameless_featurizer_info is not None else None
        )
        return np.empty((0, len(selected_indices)), dtype=np.float64), labels, nameless_empty

    index_by_signature_id = {
        str(signature_id): index for index, signature_id in enumerate(rust_featurizer.signature_ids())
    }
    indexed_pairs = [(index_by_signature_id[str(left)], index_by_signature_id[str(right)]) for left, right, _ in pairs]
    union_indices = sorted(set(selected_indices) | set(nameless_indices))
    union_features = np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed(indexed_pairs, union_indices, int(n_jobs), nan_value),
        dtype=np.float64,
    )
    position_by_index = {feature_index: position for position, feature_index in enumerate(union_indices)}
    features = (
        union_features
        if selected_indices == union_indices
        else np.take(union_features, [position_by_index[index] for index in selected_indices], axis=1)
    )
    nameless_features = None
    if nameless_featurizer_info is not None:
        nameless_features = (
            union_features
            if nameless_indices == union_indices
            else np.take(union_features, [position_by_index[index] for index in nameless_indices], axis=1)
        )
    return features, labels, nameless_features


def arrow_training_feature_splits(
    arrow_dataset: ArrowDataset,
    splits: PairwiseTrainingSplits,
    *,
    featurizer_info: Any,
    nameless_featurizer_info: Any,
    n_jobs: int,
    nan_value: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray | None],
    tuple[np.ndarray, np.ndarray, np.ndarray | None],
    tuple[np.ndarray, np.ndarray, np.ndarray | None],
    Any,
]:
    from s2and import feature_port

    rust_featurizer = feature_port.build_rust_featurizer_from_arrow_dataset(
        arrow_dataset,
        name_tuples=None,
        num_threads=n_jobs,
    )
    return (
        _feature_tuple_from_rust_featurizer(
            rust_featurizer,
            splits.train_pairs,
            featurizer_info=featurizer_info,
            nameless_featurizer_info=nameless_featurizer_info,
            n_jobs=n_jobs,
            nan_value=nan_value,
        ),
        _feature_tuple_from_rust_featurizer(
            rust_featurizer,
            splits.val_pairs,
            featurizer_info=featurizer_info,
            nameless_featurizer_info=nameless_featurizer_info,
            n_jobs=n_jobs,
            nan_value=nan_value,
        ),
        _feature_tuple_from_rust_featurizer(
            rust_featurizer,
            splits.test_pairs,
            featurizer_info=featurizer_info,
            nameless_featurizer_info=nameless_featurizer_info,
            n_jobs=n_jobs,
            nan_value=nan_value,
        ),
        rust_featurizer,
    )


def build_pairwise_clusterer_from_features(
    train: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    val: tuple[np.ndarray, np.ndarray, np.ndarray | None],
    *,
    featurization_info: Any,
    nameless_featurization_info: Any,
    n_jobs: int,
    random_seed: int,
    pairwise_n_iter: int,
    cluster_n_iter: int,
    fixed_lightgbm_params: bool = False,
    fixed_cluster_eps: float | None = None,
) -> Any:
    from lightgbm import LGBMClassifier

    from s2and.model import Clusterer, PairwiseModeler
    from s2and.model_pairwise import FastCluster

    X_train, y_train, nameless_X_train = train
    X_val, y_val, nameless_X_val = val
    if nameless_X_train is None or nameless_X_val is None:
        raise RuntimeError("Nameless training features are required")
    pairwise_search_space = None
    pairwise_estimator = None
    pairwise_monotone_constraints = featurization_info.lightgbm_monotone_constraints
    nameless_pairwise_estimator = None
    nameless_pairwise_monotone_constraints = nameless_featurization_info.lightgbm_monotone_constraints
    if fixed_lightgbm_params:
        pairwise_search_space = {}
        fixed_params: dict[str, Any] = {
            "objective": "binary",
            "metric": "auc",
            "n_jobs": n_jobs,
            "verbose": -1,
            "tree_learner": "data",
            "random_state": random_seed,
        }
        pairwise_estimator_params = dict(fixed_params)
        if pairwise_monotone_constraints is not None:
            pairwise_estimator_params["monotone_constraints"] = pairwise_monotone_constraints
            pairwise_estimator_params["monotone_constraints_method"] = "advanced"
        pairwise_estimator = LGBMClassifier(**pairwise_estimator_params)
        nameless_estimator_params = dict(fixed_params)
        if nameless_pairwise_monotone_constraints is not None:
            nameless_estimator_params["monotone_constraints"] = nameless_pairwise_monotone_constraints
            nameless_estimator_params["monotone_constraints_method"] = "advanced"
        nameless_pairwise_estimator = LGBMClassifier(**nameless_estimator_params)
        pairwise_monotone_constraints = None
        nameless_pairwise_monotone_constraints = None

    pairwise_modeler = PairwiseModeler(
        n_iter=pairwise_n_iter,
        estimator=pairwise_estimator,
        search_space=pairwise_search_space,
        monotone_constraints=pairwise_monotone_constraints,
        random_state=random_seed,
    )
    pairwise_modeler.fit(X_train, y_train, X_val, y_val)

    nameless_pairwise_modeler = PairwiseModeler(
        n_iter=pairwise_n_iter,
        estimator=nameless_pairwise_estimator,
        search_space=pairwise_search_space,
        monotone_constraints=nameless_pairwise_monotone_constraints,
        random_state=random_seed,
    )
    nameless_pairwise_modeler.fit(nameless_X_train, y_train, nameless_X_val, y_val)
    fixed_cluster_model = None
    fixed_cluster_search_space = None
    if fixed_cluster_eps is not None:
        fixed_cluster_model = FastCluster(linkage="average", eps=float(fixed_cluster_eps))
        fixed_cluster_search_space = {}

    return Clusterer(
        featurization_info,
        pairwise_modeler.classifier,
        cluster_model=fixed_cluster_model,
        search_space=fixed_cluster_search_space,
        n_jobs=n_jobs,
        n_iter=cluster_n_iter,
        nameless_classifier=nameless_pairwise_modeler.classifier,
        nameless_featurizer_info=nameless_featurization_info,
        random_state=random_seed,
        use_default_constraints_as_supervision=False,
    )


def fit_clusterer_from_arrow_validation(
    clusterer: Any,
    splits: PairwiseTrainingSplits,
    rust_featurizer: Any,
    *,
    random_seed: int,
) -> Any:
    from hyperopt import Trials, fmin, space_eval, tpe

    from s2and.eval import b3_precision_recall_fscore
    from s2and.model_pairwise import intify

    val_block_dict = clusterer.filter_blocks(splits.val_block_dict, clusterer.val_blocks_size)
    val_cluster_to_signatures = construct_cluster_to_signatures(splits.signature_to_cluster_id, val_block_dict)
    val_dists = clusterer.make_distance_matrices_from_rust_featurizer(val_block_dict, rust_featurizer)
    weight = float(sum(len(signatures) for signatures in val_block_dict.values()))
    if weight <= 0:
        raise ValueError("Arrow validation split has no signatures after filtering")

    def obj(params):
        clusterer.set_params(params)
        pred_clusters, _ = clusterer.predict_from_rust_featurizer(
            val_block_dict,
            rust_featurizer,
            dists=val_dists,
        )
        _precision, _recall, f1, _per_signature, _pred_ratios, _true_ratios = b3_precision_recall_fscore(
            val_cluster_to_signatures,
            pred_clusters,
        )
        return -float(np.average([f1], weights=[weight]))

    clusterer.hyperopt_trials_store = Trials()
    _ = fmin(
        fn=obj,
        space=clusterer.search_space,
        algo=partial(tpe.suggest, n_startup_jobs=5),
        max_evals=clusterer.n_iter,
        trials=clusterer.hyperopt_trials_store,
        rstate=np.random.default_rng(random_seed),
    )
    best_params = space_eval(clusterer.search_space, clusterer.hyperopt_trials_store.argmin)
    clusterer.best_params = {key: intify(value) for key, value in best_params.items()}
    clusterer.set_params(clusterer.best_params)
    return clusterer


def apply_fixed_cluster_eps(clusterer: Any, fixed_cluster_eps: float | None) -> Any:
    if fixed_cluster_eps is None:
        return clusterer
    clusterer.best_params = {"eps": float(fixed_cluster_eps)}
    clusterer.set_params(clusterer.best_params)
    return clusterer


def main() -> None:
    from s2and.consts import DEFAULT_CHUNK_SIZE
    from s2and.eval import cluster_eval
    from s2and.featurizer import (
        DEFAULT_FEATURE_GROUPS,
        DEFAULT_NAMELESS_FEATURE_GROUPS,
        FeaturizationInfo,
        featurize,
    )
    from s2and.name_tuple_artifact import load_name_tuple_artifact
    from s2and.production_model import load_production_model

    args = _build_parser().parse_args()
    n_jobs = args.n_jobs
    train_flag = bool(args.train)
    if args.use_arrow and args.no_arrow:
        raise ValueError("Pass only one of --use-arrow or --no-arrow")
    if args.use_arrow and train_flag:
        raise ValueError("--use-arrow is for production-model evaluation and cannot be combined with --train")
    os.environ["OMP_NUM_THREADS"] = str(n_jobs)
    train_modes = (
        _resolve_requested_train_modes(args.train_modes, compare_train_modes=bool(args.compare_train_modes))
        if train_flag
        else ["production"]
    )

    data_original = None if args.json_data_root is None else str(args.json_data_root.resolve())
    arrow_data_root = None if args.arrow_data_root is None else str(args.arrow_data_root.resolve())
    if args.dataset == "mini":
        # aminer has too much variance; medline is pairwise only
        datasets = ["arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath"]
    elif args.dataset == "full":
        datasets = ["arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath"]
    else:
        if args.use_arrow:
            raise ValueError("--use-arrow currently supports --dataset mini and --dataset full only")
        datasets = ["inventors_s2and"]
    datasets = _resolve_requested_datasets(datasets, args.datasets, args.dataset)
    if train_flag:
        _validate_train_mode_scope(train_modes, datasets)
    active_specter_suffixes = _resolve_requested_specter_suffixes(specter_suffixes, args.specter_suffixes)
    if not train_flag and SPECTER1_SUFFIX in active_specter_suffixes:
        raise ValueError(
            "SPECTER1 production evaluation was removed; use S2AND v1.21 or earlier for the historical "
            "production model, or pass --train for a research retraining comparison"
        )
    if train_flag and args.specter2_model_path is not None:
        raise ValueError("Explicit production model paths cannot be combined with --train")
    if not train_flag and args.specter2_model_path is None:
        raise ValueError("Production evaluation requires an explicit model path via --specter2-model-path")
    if train_flag:
        random_seed = 42 if args.seed is None else int(args.seed)
    else:
        if args.seed is not None:
            raise ValueError(
                "--seed applies only to --train; production-bundle evaluation uses the bundle's "
                "recorded data_random_seed so its test split is the trainer's held-out split"
            )
        random_seed = bundle_data_random_seed(cast(Path, args.specter2_model_path))
    if train_flag:
        if any(train_mode != TRAIN_MODE_ARROW_RUST for train_mode in train_modes) and data_original is None:
            raise ValueError("ANDData training requires an explicit --json-data-root")
        if TRAIN_MODE_ARROW_RUST in train_modes and arrow_data_root is None:
            raise ValueError("Arrow/Rust training requires an explicit --arrow-data-root")
    elif args.use_arrow and arrow_data_root is None:
        raise ValueError("--use-arrow requires an explicit --arrow-data-root")
    elif args.no_arrow and data_original is None:
        raise ValueError("--no-arrow requires an explicit --json-data-root")
    elif arrow_data_root is None and data_original is None:
        raise ValueError("Production evaluation requires --arrow-data-root or --json-data-root")
    missing_arrow_error = (
        first_missing_arrow_dataset_error(arrow_data_root, datasets, active_specter_suffixes)
        if _supports_arrow_eval(args.dataset) and not train_flag
        else FileNotFoundError("Arrow eval is unavailable for this configuration")
    )
    arrow_available = _supports_arrow_eval(args.dataset) and not train_flag and missing_arrow_error is None
    if args.use_arrow and missing_arrow_error is not None:
        raise missing_arrow_error
    use_arrow = _should_use_arrow_eval(
        force_arrow=bool(args.use_arrow),
        no_arrow=bool(args.no_arrow),
        arrow_available=bool(arrow_available),
    )
    if not use_arrow and not train_flag and data_original is None:
        raise ValueError(
            "Arrow artifacts are unavailable; pass an explicit --json-data-root or repair --arrow-data-root"
        )
    uses_json_anddata = (train_flag and any(mode != TRAIN_MODE_ARROW_RUST for mode in train_modes)) or (
        not train_flag and not use_arrow
    )
    if uses_json_anddata and (args.name_counts_index_root is None or args.name_tuples_path is None):
        raise ValueError(
            "JSON/ANDData evaluation and training require explicit --name-counts-index-root and --name-tuples-path"
        )
    json_name_tuples = (
        load_name_tuple_artifact(args.name_tuples_path).pairs
        if uses_json_anddata and args.name_tuples_path is not None
        else None
    )

    seed_source = "--train seed" if train_flag else "bundle data_random_seed"
    print(
        f"Config: dataset={args.dataset}, seed={random_seed} ({seed_source}), n_jobs={n_jobs}, "
        f"train={train_flag}, use_arrow={use_arrow}"
    )
    print(f"Datasets: {datasets}")
    print(f"SPECTER suffixes: {active_specter_suffixes}")
    if train_flag:
        print(f"Train modes: {train_modes}")
        print(f"JSON data root: {data_original}")
    if use_arrow:
        print(f"Arrow data root: {arrow_data_root}")
    elif train_flag and TRAIN_MODE_ARROW_RUST in train_modes:
        print(f"Arrow data root: {arrow_data_root}")
    print()

    featurization_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_FEATURE_GROUPS),
    )
    nameless_featurization_info = FeaturizationInfo(
        features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS),
    )

    results: dict[tuple[str, str], list[dict[str, tuple]]] = {}
    for specter_suffix in active_specter_suffixes:
        for train_mode in train_modes:
            clusterer = None
            if not train_flag:
                model_path = cast(Path, args.specter2_model_path)
                if not model_path.exists():
                    raise FileNotFoundError(f"Missing explicit model artifact at {model_path}")
                print(f"=== specter_suffix: {specter_suffix}, model: {model_path} ===")
                clusterer = load_production_model(model_path)
                clusterer.n_jobs = n_jobs
            else:
                print(f"=== specter_suffix: {specter_suffix}, train_mode: {train_mode} ===")

            cluster_metrics_all = []
            for dataset_name in datasets:
                print(f"-- dataset: {dataset_name} --")
                if use_arrow:
                    if clusterer is None:
                        raise RuntimeError("Arrow evaluation requires a loaded production Clusterer")
                    if arrow_data_root is None:
                        raise RuntimeError("Arrow evaluation requires an explicit --arrow-data-root")
                    with resolve_arrow_dataset(arrow_data_root, dataset_name, specter_suffix) as arrow_dataset:
                        cluster_metrics, _b3_metrics_per_signature = cluster_eval_arrow(
                            arrow_dataset,
                            clusterer,
                            random_seed=random_seed,
                            n_jobs=n_jobs,
                        )
                    print(cluster_metrics)
                    cluster_metrics_all.append(cluster_metrics)
                    continue

                if train_flag and train_mode == TRAIN_MODE_ARROW_RUST:
                    if arrow_data_root is None:
                        raise RuntimeError("Arrow Rust training requires an explicit --arrow-data-root")
                    with resolve_arrow_dataset(arrow_data_root, dataset_name, specter_suffix) as arrow_dataset:
                        splits = pair_splits_from_arrow_dataset(
                            arrow_dataset,
                            random_seed=random_seed,
                            train_pairs_size=int(args.train_pairs_size),
                            val_pairs_size=int(args.val_pairs_size),
                            test_pairs_size=int(args.test_pairs_size),
                        )
                        train, val, _test, rust_featurizer = arrow_training_feature_splits(
                            arrow_dataset,
                            splits,
                            featurizer_info=featurization_info,
                            nameless_featurizer_info=nameless_featurization_info,
                            n_jobs=n_jobs,
                            nan_value=np.nan,
                        )
                        clusterer = build_pairwise_clusterer_from_features(
                            train,
                            val,
                            featurization_info=featurization_info,
                            nameless_featurization_info=nameless_featurization_info,
                            n_jobs=n_jobs,
                            random_seed=random_seed,
                            pairwise_n_iter=int(args.pairwise_n_iter),
                            cluster_n_iter=int(args.cluster_n_iter),
                            fixed_lightgbm_params=bool(args.fixed_lightgbm_params),
                            fixed_cluster_eps=args.fixed_cluster_eps,
                        )
                        if args.fixed_cluster_eps is None:
                            clusterer = fit_clusterer_from_arrow_validation(
                                clusterer,
                                splits,
                                rust_featurizer,
                                random_seed=random_seed,
                            )
                        else:
                            clusterer = apply_fixed_cluster_eps(clusterer, args.fixed_cluster_eps)
                        cluster_metrics, _b3_metrics_per_signature = cluster_eval_arrow(
                            arrow_dataset,
                            clusterer,
                            random_seed=random_seed,
                            n_jobs=n_jobs,
                        )
                    print(cluster_metrics)
                    cluster_metrics_all.append(cluster_metrics)
                    continue

                backend = _backend_for_train_mode(train_mode)
                if data_original is None:
                    raise RuntimeError("ANDData evaluation requires an explicit --json-data-root")
                with _temporary_s2and_backend(backend):
                    if args.name_counts_index_root is None or json_name_tuples is None:
                        raise RuntimeError("ANDData evaluation requires explicit canonical artifact paths")
                    anddata = build_eval_anddata(
                        data_root=data_original,
                        dataset_name=dataset_name,
                        name_counts_index_root=args.name_counts_index_root,
                        name_tuples=json_name_tuples,
                        specter_suffix=specter_suffix,
                        n_jobs=n_jobs,
                        random_seed=random_seed,
                        train_pairs_size=int(args.train_pairs_size),
                        val_pairs_size=int(args.val_pairs_size),
                        test_pairs_size=int(args.test_pairs_size),
                    )
                    train = None
                    val = None

                    if train_flag:
                        train, val, _test = featurize(
                            anddata,
                            featurization_info,
                            n_jobs=n_jobs,
                            chunk_size=DEFAULT_CHUNK_SIZE,
                            nameless_featurizer_info=nameless_featurization_info,
                            nan_value=np.nan,
                        )

                if train_flag:
                    if train is None or val is None:
                        raise RuntimeError("Training mode did not produce train/val features")
                    evaluation_anddata = anddata
                    clusterer = build_pairwise_clusterer_from_features(
                        cast(tuple[np.ndarray, np.ndarray, np.ndarray | None], train),
                        cast(tuple[np.ndarray, np.ndarray, np.ndarray | None], val),
                        featurization_info=featurization_info,
                        nameless_featurization_info=nameless_featurization_info,
                        n_jobs=n_jobs,
                        random_seed=random_seed,
                        pairwise_n_iter=int(args.pairwise_n_iter),
                        cluster_n_iter=int(args.cluster_n_iter),
                        fixed_lightgbm_params=bool(args.fixed_lightgbm_params),
                        fixed_cluster_eps=args.fixed_cluster_eps,
                    )
                    if args.fixed_cluster_eps is None:
                        with _temporary_s2and_backend("python"):
                            clusterer.fit(evaluation_anddata)
                    else:
                        clusterer = apply_fixed_cluster_eps(clusterer, args.fixed_cluster_eps)
                else:
                    evaluation_anddata = anddata

                if clusterer is None:
                    raise RuntimeError("Clusterer was not initialized. Check --train flag and model artifact path.")

                with _temporary_s2and_backend("python" if train_flag else None):
                    cluster_metrics, _b3_metrics_per_signature = cluster_eval(
                        evaluation_anddata,
                        clusterer,
                        split="test",
                    )
                print(cluster_metrics)
                cluster_metrics_all.append(cluster_metrics)

            results[(specter_suffix, train_mode)] = cluster_metrics_all
            b3s = [m["B3 (P, R, F1)"][-1] for m in cluster_metrics_all]
            print(f"B3 F1s: {b3s}, mean: {sum(b3s) / len(b3s):.3f}")
            print()

    if train_flag and len(train_modes) > 1:
        _assert_training_mode_metrics_identical(
            results,
            specter_suffixes_to_check=active_specter_suffixes,
            train_modes=train_modes,
            datasets=datasets,
        )
        print("Training mode parity check: passed")
        print()

    # summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for (specter_suffix, train_mode), metrics_by_dataset in results.items():
        for i, dataset_name in enumerate(datasets):
            print(
                f"Performance with {specter_suffix} data, mode={train_mode}, on {dataset_name} (B3): "
                f"{metrics_by_dataset[i]['B3 (P, R, F1)']}"
            )
        print()


if __name__ == "__main__":
    main()

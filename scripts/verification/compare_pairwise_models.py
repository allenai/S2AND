"""Compare two native pairwise packages on the same non-patent test pairs.

Use a fixed probability threshold of 0.5. Report deltas as with patents minus
without patents. EPS calibration and cluster prediction have no role here.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.arrow_inputs import read_arrow_collection_root  # noqa: E402
from s2and.feature_port import featurize_pairs_with_rust  # noqa: E402
from s2and.production_bundle_contract import infer_bundle_kind, require_bundle_state  # noqa: E402
from s2and.production_model import _load_pairwise_staging_model, load_production_model  # noqa: E402
from s2and.production_training_contract import load_packaged_artifact_authority  # noqa: E402
from scripts.production.model import release_pairwise as release  # noqa: E402
from scripts.production.model.unknown_signatures import (  # noqa: E402
    augmented_unknown_ids,
    exclude_unknown_pairs,
    load_unknown_signatures,
)

PATENT_DATASET = "inventors_s2and"
HEADS = ("main", "nameless", "production")
BlockKey = tuple[str, str]


@dataclass
class PairedScores:
    """Keep pair identities, labels, blocks, and both model scores together."""

    dataset: str
    pairs: list[tuple[str, str, int | float]]
    blocks: list[tuple[str, str]]
    source_blocks: list[tuple[BlockKey, BlockKey]]
    labels: np.ndarray
    with_patents: dict[str, np.ndarray]
    without_patents: dict[str, np.ndarray]
    removed_unknown_pairs: int


def _load_model(path: Path, artifact_hashes: Mapping[str, str], n_jobs: int) -> Any:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    kind = infer_bundle_kind(manifest["sha256"])
    _, eps_state = require_bundle_state(kind, manifest["eps_calibration"])
    if kind == "pairwise_only":
        model = _load_pairwise_staging_model(
            path, expected_eps_calibration=eps_state, expected_artifact_hashes=artifact_hashes
        )
    else:
        model = load_production_model(path, expected_artifact_hashes=artifact_hashes)
    model.n_jobs = n_jobs
    return model


def _feature_contract(with_patents: Any, without_patents: Any) -> dict[str, Any]:
    contract = {}
    for head, attribute in (("main", "featurizer_info"), ("nameless", "nameless_featurizer_info")):
        left = getattr(with_patents, attribute)
        right = getattr(without_patents, attribute)
        if left is None or right is None:
            raise ValueError("Both models require main and nameless features")
        names = left.get_feature_names()
        indices = left.selected_feature_indices()
        if names != right.get_feature_names() or indices != right.selected_feature_indices():
            raise ValueError(f"Models have different ordered {head} feature names or indices")
        contract[head] = {"names": names, "indices": indices}
    return contract


def _paired_predictions(
    pairs: Sequence[tuple[str, str, int | float]],
    featurizer: Any,
    with_patents: Any,
    without_patents: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    features, labels, nameless_features = featurize_pairs_with_rust(
        featurizer,
        pairs,
        with_patents.featurizer_info,
        nameless_featurizer_info=with_patents.nameless_featurizer_info,
        n_jobs=args.n_jobs,
        nan_value=np.nan,
        total_ram_bytes=args.total_ram_bytes,
    )
    expected = np.asarray([pair[2] for pair in pairs])
    if not np.array_equal(labels, expected):
        raise ValueError("Feature labels differ from the ordered test pair labels")
    if nameless_features is None:
        raise ValueError("Pair comparison requires nameless features")
    predictions = []
    for model in (with_patents, without_patents):
        if model.nameless_classifier is None:
            raise ValueError("Pair comparison requires the nameless classifier")
        main = model.classifier.predict_proba(features)[:, 1]
        nameless = model.nameless_classifier.predict_proba(nameless_features)[:, 1]
        _, production = release.pairwise_metrics(labels, main, nameless)
        predictions.append({"main": main, "nameless": nameless, "production": production})
    return np.asarray(labels, dtype=np.int8), predictions[0], predictions[1]


def _metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    if weights is None:
        baseline, _ = release.pairwise_metrics(labels, scores, scores)
        metrics = {key: float(value) for key, value in baseline.items() if key != "rows"}
    else:
        macro = precision_recall_fscore_support(
            labels, scores > 0.5, average="macro", zero_division=0, sample_weight=weights
        )
        metrics = {
            "auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
            "macro_precision": float(macro[0]),
            "macro_recall": float(macro[1]),
            "macro_f1": float(macro[2]),
        }
    binary = precision_recall_fscore_support(
        labels, scores > 0.5, average="binary", zero_division=0, sample_weight=weights
    )
    metrics.update(
        average_precision=float(average_precision_score(labels, scores, sample_weight=weights)),
        precision=float(binary[0]),
        recall=float(binary[1]),
        f1=float(binary[2]),
    )
    return metrics


def _block_components(edges: Sequence[tuple[BlockKey, BlockKey]]) -> list[BlockKey]:
    """Join cross-block pairs so each bootstrap unit contains all linked blocks."""
    parents: dict[BlockKey, BlockKey] = {}

    def find(key: BlockKey) -> BlockKey:
        parents.setdefault(key, key)
        root = key
        while parents[root] != root:
            root = parents[root]
        while parents[key] != key:
            parent = parents[key]
            parents[key] = root
            key = parent
        return root

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            small, large = sorted((left_root, right_root))
            parents[large] = small
    return [find(left) for left, _right in edges]


def _score_report(
    labels: np.ndarray,
    with_patents: Mapping[str, np.ndarray],
    without_patents: Mapping[str, np.ndarray],
    units: Sequence[BlockKey],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    heads = {}
    for head in HEADS:
        left, right = _metrics(labels, with_patents[head]), _metrics(labels, without_patents[head])
        heads[head] = {
            "with_patents": left,
            "without_patents": right,
            "delta_with_minus_without": {metric: left[metric] - right[metric] for metric in left},
        }
    unique_units = sorted(set(units))
    intervals: dict[str, Any] = {"resamples": resamples, "valid_resamples": 0, "units": len(unique_units)}
    if resamples and len(unique_units) >= 2:
        unit_index = {unit: index for index, unit in enumerate(unique_units)}
        pair_unit = np.asarray([unit_index[unit] for unit in units])
        deltas: dict[str, dict[str, list[float]]] = {
            head: {metric: [] for metric in heads[head]["with_patents"]} for head in HEADS
        }
        rng = np.random.default_rng(seed)
        for _ in range(resamples):
            sampled = rng.integers(len(unique_units), size=len(unique_units))
            weights = np.bincount(sampled, minlength=len(unique_units))[pair_unit]
            if not all(np.sum(weights[labels == label]) > 0 for label in (0, 1)):
                continue
            intervals["valid_resamples"] += 1
            for head in HEADS:
                if np.array_equal(with_patents[head], without_patents[head]):
                    delta = {metric: 0.0 for metric in deltas[head]}
                else:
                    left = _metrics(labels, with_patents[head], weights)
                    right = _metrics(labels, without_patents[head], weights)
                    delta = {metric: left[metric] - right[metric] for metric in left}
                for metric, value in delta.items():
                    deltas[head][metric].append(value)
        if intervals["valid_resamples"]:
            intervals["delta_percentile_95"] = {
                head: {metric: np.quantile(values, [0.025, 0.975]).tolist() for metric, values in metrics.items()}
                for head, metrics in deltas.items()
            }
    return {
        "rows": int(labels.size),
        "positive_rows": int(labels.sum()),
        "heads": heads,
        "paired_block_bootstrap": intervals,
    }


def _source_block(
    dataset: str, signature_id: str, block: str, benchmark_blocks: Mapping[str, Mapping[str, str]]
) -> BlockKey:
    if dataset != "augmented":
        return dataset, block
    origin, separator, original_id = signature_id.partition("___")
    if not separator or not origin or not original_id:
        raise ValueError(f"Invalid augmented source signature ID: {signature_id!r}")
    if origin == PATENT_DATASET:
        raise ValueError("Non-patent comparison cannot contain augmented patent signatures")
    return origin, benchmark_blocks.get(origin, {}).get(original_id, block)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    """Write matched pair scores and metrics from two validated native packages."""
    start = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Comparison output already exists: {output_dir}")
    inputs_path = Path(args.source_inputs).resolve()
    version, all_specs = release._heldout_input_specs(inputs_path)
    specs = {name: files for name, files in all_specs.items() if name != PATENT_DATASET}
    if not specs:
        raise ValueError("Comparison requires at least one non-patent test dataset")
    arrow_root = Path(args.arrow_root).resolve()
    arrow_datasets, _, _ = read_arrow_collection_root(arrow_root / "manifest.json")
    if set(arrow_datasets) != set(all_specs):
        raise ValueError("Arrow collection differs from the declared evaluation datasets")
    artifacts = load_packaged_artifact_authority(name_counts_index_root=Path(args.index_root))
    model_paths = {
        "with_patents": Path(args.model_with_patents).resolve(),
        "without_patents": Path(args.model_without_patents).resolve(),
    }
    models = {name: _load_model(path, artifacts.hashes, args.n_jobs) for name, path in model_paths.items()}
    if any(model.production_model_release_version != version for model in models.values()):
        raise ValueError("Test inputs and model release versions differ")
    feature_contract = _feature_contract(models["with_patents"], models["without_patents"])
    verified = {name: release._verified_files(files, label=f"comparison.{name}") for name, files in specs.items()}
    unknowns = {name: load_unknown_signatures(files) for name, files in verified.items()}
    for name, source in unknowns.items():
        if source.path is not None and source.path != verified[name].get("unknown_signatures"):
            raise ValueError(f"Unknown-signature sidecar lacks a declared hash for {name!r}")
    augmented_unknowns = augmented_unknown_ids({name: source.ids for name, source in unknowns.items()})
    benchmark_blocks = {}
    for name, files in verified.items():
        if "blocks" in files:
            blocks = release._normalized_blocks(files["blocks"], name=name)
            benchmark_blocks[name] = {signature: block for block, members in blocks.items() for signature in members}
    scored: list[PairedScores] = []
    for name, files in verified.items():
        pairs = release._labeled_pairs(files["pairs"], name=name)
        unknown_ids = augmented_unknowns if name == "augmented" else unknowns[name].ids
        selected = exclude_unknown_pairs(pairs, unknown_ids)
        if not selected:
            raise ValueError(f"Unknown exclusions remove all test pairs for {name!r}")
        dataset = release._load_anddata(name, files, mode="inference", name_tuples=artifacts.name_tuples.pairs)
        blocks = dataset.get_signatures_to_block()
        missing = {member for left, right, _ in selected for member in (left, right)} - blocks.keys()
        if missing:
            raise ValueError(f"Test pair signatures are absent from {name!r}: {sorted(missing)[:10]}")
        if name in benchmark_blocks and blocks != benchmark_blocks[name]:
            raise ValueError(f"Test blocks differ from the source signatures for {name!r}")
        if name != "augmented" and not unknown_ids <= set(blocks):
            raise ValueError(f"Unknown signatures fall outside the test source for {name!r}")
        pair_blocks = [(blocks[left], blocks[right]) for left, right, _ in selected]
        source_blocks = [
            (
                _source_block(name, left, blocks[left], benchmark_blocks),
                _source_block(name, right, blocks[right], benchmark_blocks),
            )
            for left, right, _ in selected
        ]
        with release._release_rust_featurizer(
            arrow_root,
            name,
            source_sha256=release._spec_sha256(specs[name]),
            dataset=dataset,
            signature_ids={member for left, right, _ in selected for member in (left, right)},
            artifacts=artifacts,
            n_jobs=args.n_jobs,
        ) as featurizer:
            labels, left_scores, right_scores = _paired_predictions(
                selected, featurizer, models["with_patents"], models["without_patents"], args
            )
        scored.append(
            PairedScores(
                name,
                selected,
                pair_blocks,
                source_blocks,
                labels,
                left_scores,
                right_scores,
                len(pairs) - len(selected),
            )
        )
        del dataset
        gc.collect()
        print(json.dumps({"event": "paired_dataset_complete", "dataset": name, "rows": len(selected)}), flush=True)
    all_edges = [edge for item in scored for edge in item.source_blocks]
    all_units = _block_components(all_edges)
    output_dir.mkdir(parents=True)
    dataset_reports = {}
    offset = 0
    for item in scored:
        units = all_units[offset : offset + len(item.pairs)]
        offset += len(item.pairs)
        dataset_report = _score_report(
            item.labels,
            item.with_patents,
            item.without_patents,
            units,
            resamples=0,
            seed=args.bootstrap_seed,
        )
        dataset_report["unknown_pairs_removed"] = item.removed_unknown_pairs
        dataset_report["cross_block_pairs"] = sum(left != right for left, right in item.source_blocks)
        data: dict[str, Any] = {
            "dataset": [item.dataset] * len(item.pairs),
            "signature_id_1": [pair[0] for pair in item.pairs],
            "signature_id_2": [pair[1] for pair in item.pairs],
            "label": item.labels,
            "block_id_1": [blocks[0] for blocks in item.blocks],
            "block_id_2": [blocks[1] for blocks in item.blocks],
            "source_block_1": [json.dumps(blocks[0]) for blocks in item.source_blocks],
            "source_block_2": [json.dumps(blocks[1]) for blocks in item.source_blocks],
            "bootstrap_unit": [json.dumps(unit) for unit in units],
        }
        for head in HEADS:
            data[f"with_patents_{head}"] = item.with_patents[head]
            data[f"without_patents_{head}"] = item.without_patents[head]
        score_path = output_dir / f"pairs_{item.dataset}.parquet"
        pq.write_table(pa.table(data), score_path, compression="zstd")
        dataset_report["pair_scores_file"] = score_path.name
        dataset_reports[item.dataset] = dataset_report
    labels = np.concatenate([item.labels for item in scored])
    aggregate_with = {head: np.concatenate([item.with_patents[head] for item in scored]) for head in HEADS}
    aggregate_without = {head: np.concatenate([item.without_patents[head] for item in scored]) for head in HEADS}
    aggregate = _score_report(
        labels,
        aggregate_with,
        aggregate_without,
        all_units,
        resamples=0,
        seed=args.bootstrap_seed,
    )
    report = {
        "models": {name: str(path) for name, path in model_paths.items()},
        "feature_contract": feature_contract,
        "excluded_datasets": [PATENT_DATASET],
        "metric_contract": {
            "threshold": {"operator": ">", "value": 0.5},
            "production_score": "arithmetic_mean_main_nameless",
            "delta": "with_patents_minus_without_patents",
            "aggregate": "all_pairs",
            "bootstrap_unit": "connected_source_blocks_across_all_pairs",
            "augmented_block_authority": "source_holdout_block_for_matched_copies_else_augmented_block",
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_interval": "paired_percentile_95",
            "uncertainty": "Blocks can remain correlated. Intervals describe this test cohort and omit fit variation.",
        },
        "datasets": dataset_reports,
        "aggregate": aggregate,
        "elapsed_seconds": time.perf_counter() - start,
    }
    release._reject_nonfinite_json(report, label="paired point metrics")
    release._write_fresh_json(output_dir / "point_metrics.json", report)
    lines = [
        "Deltas use with patents minus without patents.",
        "Metrics use the production arithmetic mean and the fixed threshold > 0.5.",
        "",
        "| Dataset | Pairs | AUROC with | AUROC without | Delta AUROC | Delta AP | Delta macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, population in [*dataset_reports.items(), ("All pairs", aggregate)]:
        head = population["heads"]["production"]
        delta = head["delta_with_minus_without"]
        lines.append(
            f"| {name} | {population['rows']} | {head['with_patents']['auroc']:.6f} | "
            f"{head['without_patents']['auroc']:.6f} | {delta['auroc']:+.6f} | "
            f"{delta['average_precision']:+.6f} | {delta['macro_f1']:+.6f} |"
        )
    (output_dir / "point_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps({"event": "paired_point_metrics_complete", "report": str(output_dir / "point_metrics.json")}),
        flush=True,
    )
    offset = 0
    for item in scored:
        units = all_units[offset : offset + len(item.pairs)]
        offset += len(item.pairs)
        dataset_reports[item.dataset]["paired_block_bootstrap"] = _score_report(
            item.labels,
            item.with_patents,
            item.without_patents,
            units,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )["paired_block_bootstrap"]
    report["aggregate"]["paired_block_bootstrap"] = _score_report(
        labels,
        aggregate_with,
        aggregate_without,
        all_units,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )["paired_block_bootstrap"]
    report["elapsed_seconds"] = time.perf_counter() - start
    release._reject_nonfinite_json(report, label="paired comparison")
    release._write_fresh_json(output_dir / "comparison.json", report)
    return report


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "model-with-patents",
        "model-without-patents",
        "source-inputs",
        "arrow-root",
        "index-root",
        "output-dir",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--n-jobs", type=release._positive_int, default=1)
    parser.add_argument("--total-ram-bytes", type=release._positive_int)
    parser.add_argument("--bootstrap-resamples", type=_nonnegative_int, default=500)
    parser.add_argument("--bootstrap-seed", type=_nonnegative_int, default=13)
    return parser


def main(argv: list[str] | None = None) -> None:
    report = compare(build_parser().parse_args(argv))
    print(json.dumps({"aggregate": report["aggregate"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

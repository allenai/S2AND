"""Exercise installed S2AND's bulk and promoted incremental Arrow entrypoints."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pyarrow as pa

from s2and._sha256 import sha256_file as _sha256_file
from s2and.arrow_inputs import (
    INFERENCE_ARROW_BUNDLE_SCHEMA_VERSION,
    ArrowDataset,
    build_arrow_artifact_manifest,
    write_arrow_artifact_manifest,
)
from s2and.consts import FEATURIZER_VERSION, NAME_COUNTS_INDEX_PATH, NORMALIZATION_VERSION
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.incremental_linking.feature_block_arrow import write_raw_arrow_batch_lookup_indexes
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from s2and.model import Clusterer, FastCluster
from s2and.production_bundle import finalize_production_bundle, write_pairwise_production_bundle
from s2and.production_model import canonical_artifact_hashes, load_production_model, pairwise_bundle_binding
from s2and.runtime import build_runtime_context

RELEASE_DATA_MANIFEST_SCHEMA = INFERENCE_ARROW_BUNDLE_SCHEMA_VERSION


def _require_sha256(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected_sha256} observed={observed_sha256}")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _release_dataset_root(data_root: Path, dataset: str) -> Path:
    root_manifest = _read_json_object(data_root / "manifest.json", label="Release data manifest")
    if root_manifest.get("schema") != RELEASE_DATA_MANIFEST_SCHEMA:
        raise ValueError(
            "Release data manifest schema mismatch: "
            f"expected={RELEASE_DATA_MANIFEST_SCHEMA!r} observed={root_manifest.get('schema')!r}"
        )
    entries = [entry for entry in root_manifest["dataset_manifests"] if str(entry["dataset"]) == dataset]
    if len(entries) != 1:
        raise ValueError(f"Release data manifest must contain exactly one dataset entry for {dataset!r}")

    entry = entries[0]
    dataset_manifest_path = (data_root / entry["manifest_path"]).resolve()
    _require_sha256(
        dataset_manifest_path,
        str(entry["manifest_sha256"]),
        label=f"Release dataset {dataset!r} manifest",
    )
    return dataset_manifest_path.parent


def _fit_binary_classifier(width: int, *, seed: int) -> lgb.LGBMClassifier:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(32, width))
    labels = np.asarray([0, 1] * 16, dtype=np.int8)
    classifier = lgb.LGBMClassifier(
        objective="binary",
        verbosity=-1,
        n_jobs=1,
        learning_rate=0.2,
        num_leaves=3,
        min_child_samples=1,
        min_data_in_bin=1,
        force_col_wise=True,
        n_estimators=4,
        random_state=seed,
    )
    classifier.fit(matrix, labels)
    return classifier


def _write_ipc(path: Path, table: pa.Table) -> str:
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return str(path)


def _write_arrow_request(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    signatures = pa.table(
        {
            "signature_id": pa.array(["q1", "s1", "s2"], type=pa.string()),
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "author_first": pa.array(["Alice", "Alice", "Bob"], type=pa.string()),
            "author_middle": pa.array(["", "", ""], type=pa.string()),
            "author_last": pa.array(["Wang", "Wang", "Jones"], type=pa.string()),
            "author_suffix": pa.array(["", "", ""], type=pa.string()),
            "author_affiliations": pa.array([["AI Lab"], ["AI Lab"], ["Other Lab"]], type=pa.list_(pa.string())),
            "author_orcid": pa.array([None, None, None], type=pa.string()),
            "author_position": pa.array([0, 0, 0], type=pa.int64()),
        }
    )
    papers = pa.table(
        {
            "paper_id": pa.array(["p_q", "p1", "p2"], type=pa.string()),
            "title": pa.array(["Graph Models", "Graph Models", "Different Topic"], type=pa.string()),
            "venue": pa.array(["NeurIPS", "NeurIPS", "ICML"], type=pa.string()),
            "journal_name": pa.array(["", "", ""], type=pa.string()),
            "year": pa.array([2020, 2020, 2010], type=pa.int64()),
        }
    )
    paper_authors = pa.table(
        {
            "paper_id": pa.array(["p_q", "p_q", "p1", "p1", "p2", "p2"], type=pa.string()),
            "position": pa.array([0, 1, 0, 1, 0, 1], type=pa.int64()),
            "author_name": pa.array(
                ["Alice Wang", "Ann Smith", "Alice Wang", "Ann Smith", "Bob Jones", "Carl Doe"],
                type=pa.string(),
            ),
        }
    )
    paths = {
        "signatures": _write_ipc(root / "signatures.arrow", signatures),
        "papers": _write_ipc(root / "papers.arrow", papers),
        "paper_authors": _write_ipc(root / "paper_authors.arrow", paper_authors),
    }
    paths, _ = write_raw_arrow_batch_lookup_indexes(paths, root)
    manifest = build_arrow_artifact_manifest(paths, root)
    manifest_path = write_arrow_artifact_manifest(manifest, root)
    paths["manifest"] = str(manifest_path)
    return paths


def _write_synthetic_bundle(root: Path) -> Path:
    featurizer_info = FeaturizationInfo(["year_diff"], featurizer_version=FEATURIZER_VERSION)
    clusterer = Clusterer(
        featurizer_info,
        _fit_binary_classifier(1, seed=101),
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
        nameless_classifier=_fit_binary_classifier(1, seed=102),
        nameless_featurizer_info=featurizer_info,
        batch_size=32,
    )
    clusterer.feature_contract = {
        "normalization_version": NORMALIZATION_VERSION,
        **canonical_artifact_hashes(),
    }
    clusterer.best_params = {"eps": 0.5, "linkage": "average"}
    pairwise_bundle_dir = root / "pairwise_stage" / "production_model_v0.0"
    write_pairwise_production_bundle(
        clusterer,
        pairwise_bundle_dir,
        bundle_version="0.0",
    )

    feature_count = len(promoted_linker_feature_columns())
    linker = _fit_binary_classifier(feature_count, seed=103)
    gate_config = logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        bias=np.asarray([0.0, 0.0, 10.0], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="installed_release_smoke",
    )
    linker_dir = root / "incremental_linker"
    save_incremental_linking_artifact(
        linker,
        linker_dir,
        gate_config=gate_config,
        target_spec={},
        pairwise_bundle_binding=pairwise_bundle_binding(pairwise_bundle_dir),
    )
    target_json = root / "incremental_linker_training_target.json"
    target_json.write_text("{}\n", encoding="utf-8")
    bundle_dir = root / "production_model_v0.0"
    finalize_production_bundle(
        pairwise_bundle_dir=pairwise_bundle_dir,
        output_bundle_dir=bundle_dir,
        incremental_linker_artifact_dir=linker_dir,
        target_json=target_json,
    )
    return bundle_dir


def run_smoke(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    bundle_dir = _write_synthetic_bundle(root)
    arrow_root = root / "arrow"
    _write_arrow_request(arrow_root)
    clusterer = load_production_model(bundle_dir)
    clusterer.n_jobs = 1
    with ArrowDataset.open(arrow_root) as arrow_dataset:
        bulk_summary = _bulk_smoke_summary(
            clusterer,
            arrow_dataset,
            ["q1", "s1", "s2"],
            label="installed bulk smoke",
        )
        result = clusterer.predict_incremental_from_arrow(
            ["q1", "s1", "s2"],
            arrow_dataset,
            prevent_new_incompatibilities=False,
            batching_threshold=1,
            runtime_context=build_runtime_context("installed_incremental_arrow_smoke", backend="rust"),
            total_ram_bytes=1_000_000_000,
            name_tuples=set(),
            cluster_seeds_require={"s1": "c_match", "s2": "c_other"},
        )
    return bulk_summary | _smoke_result_summary(result, {"q1", "s1", "s2"}, label="installed incremental smoke")


def _bulk_smoke_summary(
    clusterer: Clusterer,
    arrow_dataset: ArrowDataset,
    signature_ids: list[str],
    *,
    label: str,
) -> dict[str, int]:
    clusters, _ = clusterer.predict_from_arrow(
        {"smoke": signature_ids},
        arrow_dataset,
        runtime_context=build_runtime_context(label, backend="rust"),
        total_ram_bytes=1_000_000_000,
    )
    clustered_ids = [str(signature_id) for members in clusters.values() for signature_id in members]
    if len(clustered_ids) != len(signature_ids) or set(clustered_ids) != set(signature_ids):
        raise RuntimeError(f"{label} returned an invalid partition: {clusters}")
    pair_count = clusterer._last_arrow_predict_telemetry["rust_make_dists_pair_count"]
    if pair_count != 3:
        raise RuntimeError(f"{label} did not score all three pairs: {pair_count}")
    return {
        "bulk_signature_count": len(clustered_ids),
        "bulk_pair_count": 3,
    }


def _smoke_result_summary(
    result: dict[str, Any],
    expected_signature_ids: set[str],
    *,
    label: str,
) -> dict[str, Any]:
    telemetry = dict(result["incremental_linker_telemetry"])
    if result.get("incremental_linker_query_view") != "raw_arrow":
        raise RuntimeError(f"{label} did not use raw Arrow: {result}")
    if telemetry.get("arrow_promoted_incremental") != 1:
        raise RuntimeError(f"{label} missed promoted Arrow runtime: {telemetry}")
    clustered_signature_ids = [
        str(signature_id) for members in dict(result["clusters"]).values() for signature_id in members
    ]
    if (
        len(clustered_signature_ids) != len(expected_signature_ids)
        or set(clustered_signature_ids) != expected_signature_ids
    ):
        raise RuntimeError(f"{label} lost signatures: {result['clusters']}")
    return {
        "arrow_promoted_incremental": telemetry["arrow_promoted_incremental"],
        "cluster_count": len(result["clusters"]),
        "query_view": result["incremental_linker_query_view"],
        "signature_count": len(clustered_signature_ids),
    }


def run_release_candidate_smoke(
    *,
    model_dir: Path,
    data_root: Path,
    dataset: str,
    signature_ids: Iterable[str],
) -> dict[str, Any]:
    """Exercise exact downloaded release artifacts through installed Rust paths."""

    model_dir = Path(model_dir).resolve()
    data_root = Path(data_root).resolve()
    selected_signature_ids = [str(signature_id) for signature_id in signature_ids]
    if len(selected_signature_ids) != 3 or len(set(selected_signature_ids)) != 3:
        raise ValueError("Release-candidate smoke requires exactly three distinct signature IDs")

    expected_name_counts_index = (data_root / "name_counts_index").resolve()
    configured_name_counts_index = Path(NAME_COUNTS_INDEX_PATH).resolve()
    if configured_name_counts_index != expected_name_counts_index:
        raise ValueError(
            "Configured NAME_COUNTS_INDEX_PATH does not select the downloaded release data: "
            f"configured={configured_name_counts_index} expected={expected_name_counts_index}"
        )

    clusterer = load_production_model(model_dir)
    clusterer.n_jobs = 1
    with ArrowDataset.open(
        _release_dataset_root(data_root, dataset),
        require_name_counts_index=True,
    ) as arrow_dataset:
        bound_name_counts = arrow_dataset.name_counts_index
        if bound_name_counts is None or Path(bound_name_counts.path).resolve() != configured_name_counts_index:
            raise ValueError(f"Release dataset {dataset!r} does not bind the configured name-count index")
        bulk_summary = _bulk_smoke_summary(
            clusterer,
            arrow_dataset,
            selected_signature_ids,
            label="release-candidate bulk smoke",
        )
        result = clusterer.predict_incremental_from_arrow(
            selected_signature_ids,
            arrow_dataset,
            prevent_new_incompatibilities=False,
            batching_threshold=1,
            runtime_context=build_runtime_context("installed_release_candidate_smoke", backend="rust"),
            total_ram_bytes=1_000_000_000,
            cluster_seeds_require={
                selected_signature_ids[0]: "smoke-seed-1",
                selected_signature_ids[1]: "smoke-seed-2",
            },
        )
    summary = _smoke_result_summary(
        result,
        set(selected_signature_ids),
        label="release-candidate smoke",
    )
    return (
        bulk_summary
        | summary
        | {
            "configured_name_counts_index": str(configured_name_counts_index),
            "dataset": dataset,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", dest="synthetic_work_dir", type=Path, default=None)
    commands = parser.add_subparsers(dest="command")
    release = commands.add_parser(
        "release-candidate",
        help="Exercise an already-downloaded complete model and data root.",
    )
    release.add_argument("--model-dir", type=Path, required=True)
    release.add_argument("--data-root", type=Path, required=True)
    release.add_argument("--dataset", required=True)
    release.add_argument("--signature-ids", nargs=3, required=True, metavar=("SEED_1", "SEED_2", "QUERY"))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "release-candidate":
        summary = run_release_candidate_smoke(
            model_dir=args.model_dir,
            data_root=args.data_root,
            dataset=args.dataset,
            signature_ids=args.signature_ids,
        )
    elif args.synthetic_work_dir is not None:
        summary = run_smoke(args.synthetic_work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="s2and_installed_incremental_smoke_") as temp_dir:
            summary = run_smoke(Path(temp_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

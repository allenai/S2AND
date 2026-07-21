"""Exercise installed S2AND's public promoted incremental Arrow entrypoint."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pyarrow as pa

from s2and.arrow_inputs import build_arrow_artifact_manifest, write_arrow_artifact_manifest
from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import save_incremental_linking_artifact
from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow
from s2and.incremental_linking.feature_block_arrow import write_raw_arrow_batch_lookup_indexes
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from s2and.model import Clusterer, FastCluster
from s2and.production_bundle import finalize_production_bundle, write_pairwise_production_bundle
from s2and.production_model import canonical_artifact_hashes, load_production_model, pairwise_bundle_binding
from s2and.runtime import build_runtime_context


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
    cluster_seeds_path = root / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, {"s1": "c_match", "s2": "c_other"})
    paths["cluster_seeds"] = str(cluster_seeds_path)
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
        source_model_version="0.0",
    )

    feature_count = len(promoted_linker_feature_columns())
    linker = _fit_binary_classifier(feature_count, seed=103)
    fixture = np.zeros((2, feature_count), dtype=np.float32)
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
        prediction_fixture_matrix=fixture,
        gate_config=gate_config,
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
        bundle_version="0.0",
        pairwise_model_version="0.0",
        incremental_linker_version="0.0",
    )
    return bundle_dir


def run_smoke(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    bundle_dir = _write_synthetic_bundle(root)
    arrow_paths = _write_arrow_request(root / "arrow")
    clusterer = load_production_model(bundle_dir)
    clusterer.n_jobs = 1
    result = clusterer.predict_incremental_from_arrow_paths(
        ["q1", "s1", "s2"],
        arrow_paths,
        prevent_new_incompatibilities=False,
        batching_threshold=1,
        runtime_context=build_runtime_context("installed_incremental_arrow_smoke", backend="rust"),
        total_ram_bytes=1_000_000_000,
        name_tuples=set(),
    )
    telemetry = dict(result["incremental_linker_telemetry"])
    if result.get("incremental_linker_query_view") != "raw_arrow":
        raise RuntimeError(f"installed incremental smoke did not use raw Arrow: {result}")
    if telemetry.get("arrow_promoted_incremental") != 1:
        raise RuntimeError(f"installed incremental smoke missed promoted Arrow runtime: {telemetry}")
    clustered_signature_ids = {
        str(signature_id) for members in dict(result["clusters"]).values() for signature_id in members
    }
    if clustered_signature_ids != {"q1", "s1", "s2"}:
        raise RuntimeError(f"installed incremental smoke lost signatures: {result['clusters']}")
    return {
        "arrow_promoted_incremental": telemetry["arrow_promoted_incremental"],
        "cluster_count": len(result["clusters"]),
        "query_view": result["incremental_linker_query_view"],
        "signature_count": len(clustered_signature_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.work_dir is not None:
        summary = run_smoke(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="s2and_installed_incremental_smoke_") as temp_dir:
            summary = run_smoke(Path(temp_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_provenance import verify_conversion_provenance
from s2and.data import ANDData
from s2and.incremental_linking_training.classic import fit_classic
from s2and.name_counts_index import NameCountsIndex
from s2and.name_tuple_artifact import load_name_tuple_artifact
from s2and.orcid_prefix_counts import load_canonical_orcid_prefix_counts
from s2and.production_training_contract import ModelDataset, ProductionArtifactAuthority
from scripts.production.model import train_pairwise
from tests.helpers import pairwise_training_args
from tests.pairwise_release_helpers import (
    convert_release_arrow,
    source_sha256,
    write_pairwise_artifacts,
    write_pairwise_dataset,
)
from tests.training_helpers import write_classic_tiny_bundle


def test_fit_classic_fits_real_booster_and_logistic_gate(tmp_path: Path) -> None:
    fitted = fit_classic(
        write_classic_tiny_bundle(tmp_path / "classic-bundle"),
        n_jobs=1,
    )

    assert fitted.model.booster_.num_trees() > 0
    assert fitted.model.n_features_in_ == 1
    assert fitted.training_summary["rows"] == 22
    assert fitted.training_summary["rows_removed_above_retrieval_top_k"] == 12
    assert fitted.gate_config["training_summary"]["calibration_queries"] == 4
    assert fitted.gate_config["feature_names"]
    gate_weights = np.asarray(fitted.gate_config["weights"], dtype=np.float64)
    assert gate_weights.shape == (len(fitted.gate_config["feature_names"]), 3)
    assert np.isfinite(gate_weights).all()
    assert np.any(gate_weights != 0)


def test_train_pairwise_bundle_runs_real_featurization_and_model_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    dataset_files = write_pairwise_dataset(data_dir)
    name_tuples_path, name_counts_path, orcid_counts_path = write_pairwise_artifacts(tmp_path)
    arrow_dir = convert_release_arrow(data_dir, "qian", tmp_path / "arrow", name_counts_path)
    matrix_work_dir = tmp_path / "matrix-work"
    matrix_work_dir.mkdir()
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_artifact_authority",
        lambda **_kwargs: ProductionArtifactAuthority(
            name_counts_index=NameCountsIndex.open(name_counts_path),
            name_tuples=load_name_tuple_artifact(name_tuples_path),
            orcid_prefix_counts=load_canonical_orcid_prefix_counts(orcid_counts_path),
        ),
    )
    args = pairwise_training_args(
        tmp_path,
        output_dir=tmp_path / "production_model_v9.9",
        train_pairs_size=24,
        validation_pairs_size=6,
        matrix_work_dir=matrix_work_dir,
        total_ram_bytes=1_000_000_000,
        name_counts_index_root=name_counts_path,
    )
    monkeypatch.setattr(
        train_pairwise,
        "_preflight_pairwise",
        lambda _args: train_pairwise.PairwisePreflightPlan(
            output_dir=args.output_dir,
            release_version="9.9",
            dataset_names=("qian",),
            datasets={
                "qian": ModelDataset(
                    files=dataset_files,
                    sha256={role: sha256_file(path) for role, path in dataset_files.items()},
                )
            },
            model_plan_sha256="0" * 64,
            arrow_dirs={"qian": arrow_dir},
            arrow_provenance_sha256={
                "qian": verify_conversion_provenance(arrow_dir, expected_source_sha256=source_sha256(dataset_files))
            },
            matrix_work_dir=matrix_work_dir,
            matrix_work_free_bytes=1_000_000_000,
            total_ram_bytes=args.total_ram_bytes,
        ),
    )
    monkeypatch.setattr(
        train_pairwise.Clusterer,
        "fit",
        lambda *_args, **_kwargs: pytest.fail("Stage 3 must not run clustering calibration"),
    )

    real_anddata = train_pairwise.ANDData

    def checked_anddata(**kwargs):
        assert kwargs["test_pairs"] is None
        # Selection only: the Arrow conversion supplies name counts to the Rust featurizer.
        assert kwargs["name_counts_index"] is None and kwargs["preprocess"] is False
        assert kwargs["name_tuples"] == load_name_tuple_artifact(name_tuples_path).pairs
        assert not {"train_ratio", "val_ratio", "test_ratio"} & set(kwargs)
        return real_anddata(**kwargs)

    monkeypatch.setattr(train_pairwise, "ANDData", checked_anddata)
    result = train_pairwise.train_pairwise_bundle(args)

    summary = result["training_summary"]
    assert summary["main_train_rows"] == 24
    assert summary["main_val_rows"] == 6
    assert summary["nameless_train_rows"] == 24
    assert summary["main_pairwise_best_params"]
    assert summary["nameless_pairwise_best_params"]
    from s2and.production_training_contract import block_membership_sha256, frozen_test_blocks

    signatures = json.loads(dataset_files["signatures"].read_text(encoding="utf-8"))
    blocks: dict[str, list[str]] = {}
    for signature_id, signature in signatures.items():
        blocks.setdefault(signature["author_info"]["block"], []).append(signature_id)
    _, _, expected_test = ANDData.split_blocks_helper(
        SimpleNamespace(
            random_seed=args.random_seed,
            num_clusters_for_block_size=1,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        ),
        blocks,
    )
    record = summary["cluster_test_splits"]["qian"]
    assert record["block_membership_sha256"] == block_membership_sha256(blocks)
    assert frozen_test_blocks(dict(reversed(list(blocks.items()))), record) == expected_test
    assert "best_clustering_params" not in summary
    assert 0.0 <= summary["main_validation_roc_auc"] <= 1.0
    assert 0.0 <= summary["nameless_validation_roc_auc"] <= 1.0
    assert result["eps_calibration"] == "pending"
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["eps_calibration"] == "pending"
    training_config = json.loads(
        (args.output_dir / "reproducibility" / "pairwise_training_config.json").read_text(encoding="utf-8")
    )
    assert training_config["featurization_backend"] == "rust_from_provenance_bound_arrow"
    assert training_config["dataset_inputs"]["qian"]["arrow_dir"] == str(arrow_dir)
    assert training_config["dataset_inputs"]["qian"]["arrow_conversion_provenance_sha256"] == sha256_file(
        arrow_dir / "conversion_provenance.json"
    )
    config = json.loads((args.output_dir / "clusterer.json").read_text(encoding="utf-8"))
    assert config["cluster_model"] == {"eps": 0.5, "linkage": "average"}
    assert list(matrix_work_dir.iterdir()) == []
    assert args.output_dir.is_dir()

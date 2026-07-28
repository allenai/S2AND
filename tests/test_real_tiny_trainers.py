from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.incremental_linking_training.classic import OfficialBundle, fit_classic
from s2and.name_counts_index import NameCountsIndex
from s2and.name_tuple_artifact import load_name_tuple_artifact
from s2and.orcid_prefix_counts import load_canonical_orcid_prefix_counts
from s2and.production_training_contract import ModelDataset, ProductionArtifactAuthority
from scripts.production.counts.generate_orcid_name_prefix_counts import write_publication
from scripts.production.generate_canonical_name_tuples import regenerate
from scripts.production.model import train_pairwise
from tests.helpers import pairwise_training_args

_PAIRWISE_LAST_NAMES = (
    "Adams",
    "Baker",
    "Clark",
    "Davis",
    "Evans",
    "Foster",
    "Green",
    "Harris",
    "Irwin",
    "Jones",
)


def _classic_candidate_rows(
    query_specs: list[tuple[str, str, str, bool]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_index, (query_id, base_group_id, dataset, has_positive) in enumerate(query_specs):
        for rank in (1, 2, 30):
            label = int((has_positive and rank == 1) or (not has_positive and rank == 30))
            rows.append(
                {
                    "query_group_id": query_id,
                    "base_group_id": base_group_id,
                    "dataset": dataset,
                    "query_view": "full",
                    "query_first_token": "alex",
                    "candidate_component_key": f"{query_id}:candidate:{rank}",
                    "retrieval_rank": rank,
                    "label": label,
                    "tiny_score": float(label * 2 + query_index / 10 - rank / 100),
                }
            )
    return pd.DataFrame(rows)


def _write_classic_tiny_bundle(root: Path) -> OfficialBundle:
    root.mkdir()
    train_rows = _classic_candidate_rows(
        [
            (
                "fit-negative" if index == 0 else f"train-{index}",
                "fit-negative-base" if index == 0 else f"train-base-{index}",
                "train",
                index % 2 == 0,
            )
            for index in range(12)
        ]
    )
    gate_rows = _classic_candidate_rows(
        [
            ("fit-negative", "fit-negative-base", "a_khan", False),
            ("fit-positive", "fit-positive-base", "a_khan", True),
        ]
    )
    hwang_rows = _classic_candidate_rows(
        [
            ("check-negative", "check-negative-base", "h_wang", False),
            ("check-positive", "check-positive-base", "h_wang", True),
        ]
    )
    s2and_rows = _classic_candidate_rows(
        [
            ("test-negative", "test-negative-base", "s2and", False),
            ("test-positive", "test-positive-base", "s2and", True),
        ]
    )
    tables = {
        "train.csv.gz": train_rows,
        "gate.csv.gz": gate_rows,
        "hwang.csv.gz": hwang_rows,
        "s2and.csv.gz": s2and_rows,
    }
    for filename, frame in tables.items():
        frame.to_csv(root / filename, index=False, compression="gzip")

    pd.DataFrame(
        [
            {
                "query_group_id": query_id,
                "source_key": source_key,
                "split": split,
                "base_group_id": base_group_id,
            }
            for query_id, base_group_id, source_key, split in (
                ("fit-negative", "fit-negative-base", "a_khan_eval", "calibration_fit"),
                ("fit-positive", "fit-positive-base", "a_khan_eval", "calibration_fit"),
                ("check-negative", "check-negative-base", "hwang_eval", "calibration_check"),
                ("check-positive", "check-positive-base", "hwang_eval", "calibration_check"),
                ("test-negative", "test-negative-base", "s2and_eval", "test"),
                ("test-positive", "test-positive-base", "s2and_eval", "test"),
            )
        ]
    ).to_csv(root / "assignments.csv", index=False)
    pd.DataFrame({"base_group_id": ["fit-negative-base", "fit-positive-base"]}).to_csv(
        root / "internal_eval_base_groups.csv",
        index=False,
    )

    return OfficialBundle(
        root=root,
        bundle_name="tiny-real-classic",
        assets={},
        models={
            "classic": {
                "feature_columns": ["tiny_score"],
                "retrieval_top_k": 2,
                "best_params": {
                    "learning_rate": 0.1,
                    "max_depth": 2,
                    "min_child_samples": 1,
                    "n_estimators": 5,
                    "num_leaves": 4,
                },
                "train_path": "train.csv.gz",
                "classic_gate_source_path": "gate.csv.gz",
                "classic_gate_internal_eval_base_groups_path": "internal_eval_base_groups.csv",
                "s2and_eval_path": "s2and.csv.gz",
                "hwang_eval_path": "hwang.csv.gz",
                "promoted_stratified_gate": {
                    "calibration_splits": ["calibration_fit", "calibration_check"],
                    "test_split": "test",
                },
                "stratified_eval_test_split": {
                    "assignments_path": "assignments.csv",
                    "split_order": ["calibration_fit", "calibration_check", "test"],
                    "test_split": "test",
                },
            }
        },
        expected_metrics={},
    )


def test_fit_classic_fits_real_booster_and_logistic_gate(tmp_path: Path) -> None:
    fitted = fit_classic(
        _write_classic_tiny_bundle(tmp_path / "classic-bundle"),
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


def _write_pairwise_dataset(data_dir: Path) -> dict[str, Path]:
    dataset_dir = data_dir / "qian"
    dataset_dir.mkdir(parents=True)
    signatures: dict[str, dict[str, Any]] = {}
    papers: dict[str, dict[str, Any]] = {}
    clusters: dict[str, dict[str, Any]] = {}
    embeddings: dict[str, np.ndarray] = {}

    for block_index, last_name in enumerate(_PAIRWISE_LAST_NAMES):
        block = f"a {last_name.lower()}"
        for author_index in range(4):
            signature_id = f"signature-{block_index}-{author_index}"
            paper_id = f"paper-{block_index}-{author_index}"
            cluster_index = author_index // 2
            first_name = "Alex" if cluster_index == 0 else "Avery"
            affiliation = f"Institute {cluster_index}"
            coauthor = f"Coauthor {cluster_index}"
            signatures[signature_id] = {
                "signature_id": signature_id,
                "paper_id": paper_id,
                "author_info": {
                    "position": 0,
                    "block": block,
                    "first": first_name,
                    "middle": None,
                    "last": last_name,
                    "suffix": None,
                    "email": f"{first_name.lower()}@example.org",
                    "affiliations": [affiliation],
                },
            }
            papers[paper_id] = {
                "paper_id": paper_id,
                "title": f"Topic {cluster_index} study {block_index}",
                "abstract": f"Deterministic abstract for topic {cluster_index}.",
                "authors": [
                    {"author_name": f"{first_name} {last_name}", "position": 0},
                    {"author_name": coauthor, "position": 1},
                ],
                "venue": f"Venue {cluster_index}",
                "journal_name": f"Journal {cluster_index}",
                "year": 2010 + block_index,
            }
            vector = np.zeros(768, dtype=np.float32)
            vector[cluster_index] = 1.0
            vector[10 + block_index] = 0.1
            vector[100 + author_index] = 0.01
            embeddings[paper_id] = vector
            cluster_id = f"cluster-{block_index}-{cluster_index}"
            clusters.setdefault(
                cluster_id,
                {
                    "cluster_id": cluster_id,
                    "signature_ids": [],
                    "model_version": -1,
                },
            )["signature_ids"].append(signature_id)

    for filename, payload in (
        ("qian_signatures.json", signatures),
        ("qian_papers.json", papers),
        ("qian_clusters.json", clusters),
    ):
        (dataset_dir / filename).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    with (dataset_dir / "qian_specter2.pkl").open("wb") as output:
        pickle.dump(embeddings, output, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "clusters": dataset_dir / "qian_clusters.json",
        "papers": dataset_dir / "qian_papers.json",
        "signatures": dataset_dir / "qian_signatures.json",
        "specter_embeddings": dataset_dir / "qian_specter2.pkl",
    }


def _write_pairwise_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    tuple_source = tmp_path / "raw_name_tuples.txt"
    tuple_source.write_text("William,Bill\n", encoding="utf-8")
    tuple_path = tmp_path / "canonical_name_tuples.txt"
    regenerate(str(tuple_source), str(tuple_path))
    name_tuples = load_name_tuple_artifact(tuple_path)

    first_counts = {"alex": 20, "avery": 20}
    last_counts = {last_name.lower(): 4 for last_name in _PAIRWISE_LAST_NAMES}
    first_last_counts = {
        f"{first} {last_name.lower()}": 2 for first in first_counts for last_name in _PAIRWISE_LAST_NAMES
    }
    last_first_initial_counts = {f"{last_name.lower()} a": 4 for last_name in _PAIRWISE_LAST_NAMES}
    name_counts_parent = tmp_path / "name-counts"
    name_counts_path, _ = write_name_counts_index(
        name_counts_parent,
        (
            first_counts,
            last_counts,
            first_last_counts,
            last_first_initial_counts,
        ),
    )

    orcid_counts_path = tmp_path / "orcid-prefix-counts"
    write_publication(
        {},
        output_dir=orcid_counts_path,
        name_tuples=name_tuples,
    )
    return tuple_path, Path(name_counts_path), orcid_counts_path


def test_train_pairwise_bundle_runs_real_featurization_and_model_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    dataset_files = _write_pairwise_dataset(data_dir)
    name_tuples_path, name_counts_path, orcid_counts_path = _write_pairwise_artifacts(tmp_path)
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
            datasets={"qian": ModelDataset(files=dataset_files)},
            model_plan_sha256="0" * 64,
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

    result = train_pairwise.train_pairwise_bundle(args)

    summary = result["training_summary"]
    assert summary["main_train_rows"] == 24
    assert summary["main_val_rows"] == 6
    assert summary["nameless_train_rows"] == 24
    assert summary["main_pairwise_best_params"]
    assert summary["nameless_pairwise_best_params"]
    assert "best_clustering_params" not in summary
    assert 0.0 <= summary["main_validation_roc_auc"] <= 1.0
    assert 0.0 <= summary["nameless_validation_roc_auc"] <= 1.0
    assert result["eps_calibration"] == "pending"
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["eps_calibration"] == "pending"
    config = json.loads((args.output_dir / "clusterer.json").read_text(encoding="utf-8"))
    assert config["cluster_model"] == {"eps": 0.5, "linkage": "average"}
    assert list(matrix_work_dir.iterdir()) == []
    assert args.output_dir.is_dir()

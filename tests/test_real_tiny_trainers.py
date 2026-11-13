from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from s2and._sha256 import sha256_file
from s2and.data import ANDData
from s2and.incremental_linking_training.classic import fit_classic
from s2and.name_counts_index import NameCountsIndex
from s2and.name_tuple_artifact import load_name_tuple_artifact
from s2and.orcid_prefix_counts import load_canonical_orcid_prefix_counts
from s2and.production_training_contract import ModelDataset, ProductionArtifactAuthority
from scripts.production.model import train_pairwise
from tests.helpers import pairwise_training_args
from tests.pairwise_release_helpers import convert_release_arrow, write_pairwise_artifacts, write_pairwise_dataset
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


def _arrow_plan_dataset(
    data_dir: Path,
    name: str,
    arrow_root: Path,
    name_counts_path: Path,
) -> tuple[ModelDataset, dict[str, Path]]:
    """Convert one tiny benchmark, hold out its last block, and return the plan entry.

    The returned JSON files hold only the development population, so tests can select
    expected pairs independently of the Arrow loader.
    """

    files = write_pairwise_dataset(data_dir, name)
    arrow_dir = convert_release_arrow(data_dir, name, arrow_root, name_counts_path)
    signatures = json.loads(files["signatures"].read_text(encoding="utf-8"))
    clusters = json.loads(files["clusters"].read_text(encoding="utf-8"))
    blocks: dict[str, list[str]] = {}
    for signature_id, signature in signatures.items():
        blocks.setdefault(signature["author_info"]["block"], []).append(signature_id)
    heldout_block = list(blocks)[-1]
    heldout = set(blocks[heldout_block])
    heldout_dir = arrow_root.parent / "heldout" / name
    heldout_dir.mkdir(parents=True)
    plan_files = {
        "arrow": arrow_dir / "manifest.json",
        "clusters": arrow_dir / f"{name}_clusters.json",
        "heldout_blocks": heldout_dir / "test_blocks.json",
        "heldout_pairs": heldout_dir / "test_pairs.json",
    }
    plan_files["heldout_blocks"].write_text(json.dumps({heldout_block: blocks[heldout_block]}), encoding="utf-8")
    first, second = blocks[heldout_block][:2]
    plan_files["heldout_pairs"].write_text(
        json.dumps([{"signature_id_1": first, "signature_id_2": second, "label": 1}]), encoding="utf-8"
    )
    development = data_dir / "development" / name
    development.mkdir(parents=True)
    development_files = {
        "signatures": development / f"{name}_signatures.json",
        "clusters": development / f"{name}_clusters.json",
        "papers": files["papers"],
    }
    development_files["signatures"].write_text(
        json.dumps({key: value for key, value in signatures.items() if key not in heldout}), encoding="utf-8"
    )
    development_files["clusters"].write_text(
        json.dumps(
            {
                key: {**value, "signature_ids": [sid for sid in value["signature_ids"] if sid not in heldout]}
                for key, value in clusters.items()
                if any(sid not in heldout for sid in value["signature_ids"])
            }
        ),
        encoding="utf-8",
    )
    return (
        ModelDataset(files=plan_files, sha256={role: sha256_file(path) for role, path in plan_files.items()}),
        development_files,
    )


def _json_selection(files: dict[str, Path], name: str, args: Any) -> ANDData:
    """Select pairs from JSON development records, independently of the Arrow loader."""

    return ANDData(
        signatures=str(files["signatures"]),
        papers=str(files["papers"]),
        clusters=str(files["clusters"]),
        name=name,
        mode="train",
        preprocess=False,
        random_seed=args.random_seed,
        name_counts_index=None,
        train_pairs_size=args.train_pairs_size,
        val_pairs_size=args.validation_pairs_size,
    )


def _preflight_plan(args: Any, datasets: dict[str, ModelDataset], work_dir: Path) -> Any:
    return train_pairwise.PairwisePreflightPlan(
        output_dir=args.output_dir,
        release_version="9.9",
        dataset_names=tuple(name for name in datasets if name != "augmented"),
        datasets=datasets,
        model_plan_sha256="0" * 64,
        model_plan_content_sha256="1" * 64,
        arrow_dirs={name: dataset.files["arrow"].parent for name, dataset in datasets.items() if name != "augmented"},
        matrix_work_dir=work_dir,
        matrix_work_free_bytes=1_000_000_000,
        total_ram_bytes=args.total_ram_bytes,
    )


@pytest.mark.parametrize(
    "augmented_population",
    ["train", "test", "omitted", "changed-arrow", "unknown-validation"],
)
def test_train_pairwise_bundle_runs_real_featurization_and_model_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    augmented_population: str,
) -> None:
    name_tuples_path, name_counts_path, orcid_counts_path = write_pairwise_artifacts(tmp_path)
    matrix_work_dir = tmp_path / "matrix-work"
    matrix_work_dir.mkdir()
    authority = ProductionArtifactAuthority(
        name_counts_index=NameCountsIndex.open(name_counts_path),
        name_tuples=load_name_tuple_artifact(name_tuples_path),
        orcid_prefix_counts=load_canonical_orcid_prefix_counts(orcid_counts_path),
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
    plan_dataset, development_files = _arrow_plan_dataset(
        tmp_path / "data", "qian", tmp_path / "dataset" / "benchmarks", name_counts_path
    )
    selection = _json_selection(development_files, "qian", args)
    train, val, test = selection.split_cluster_signatures()
    unknown_ids = frozenset()
    if augmented_population.startswith("unknown-"):
        unknown_ids = frozenset({next(iter(train.values()))[1], next(iter(val.values()))[0]})
        # The plan declares the list beside the benchmark labels.
        sidecar = plan_dataset.files["arrow"].parent / "restored_unknown_signature_ids_20261009.json"
        sidecar.write_text(json.dumps(sorted(unknown_ids)), encoding="utf-8")
        files = {**plan_dataset.files, "unknown_signatures": sidecar}
        plan_dataset = ModelDataset(files=files, sha256={role: sha256_file(path) for role, path in files.items()})
    safe = next(iter(train.values()))[0]
    target = {
        "train": next(iter(train.values()))[1],
        "test": next(iter(test.values()))[0],
        "omitted": "not-in-model-plan",
        "changed-arrow": next(iter(train.values()))[1],
        "unknown-validation": next(iter(val.values()))[0],
    }[augmented_population]
    from s2and.featurizer import resolve_selection_pairs

    expected_pairs = resolve_selection_pairs(selection)
    expected_rows = [
        sum(left not in unknown_ids and right not in unknown_ids for left, right, _ in pairs)
        for pairs in expected_pairs
    ]
    real_featurizer = train_pairwise.build_verified_rust_featurizer
    built_members = []

    def checked_featurizer(*args, **kwargs):
        members = kwargs["signature_ids"]
        assert not members & unknown_ids
        built_members.append(members)
        return real_featurizer(*args, **kwargs)

    monkeypatch.setattr(train_pairwise, "build_verified_rust_featurizer", checked_featurizer)
    augmented_pairs = tmp_path / "augmented_train.csv"
    augmented_pairs.write_text(
        f"signature_id_1,signature_id_2,label\nqian___{safe},qian___{target},1\n", encoding="utf-8"
    )
    datasets = {
        "qian": plan_dataset,
        "augmented": ModelDataset(
            files={"train_pairs": augmented_pairs}, sha256={"train_pairs": sha256_file(augmented_pairs)}
        ),
    }
    monkeypatch.setattr(
        train_pairwise, "_preflight_pairwise", lambda _args: _preflight_plan(args, datasets, matrix_work_dir)
    )

    def load_authority(**_kwargs: Any) -> ProductionArtifactAuthority:
        if augmented_population == "changed-arrow":
            # The swap happens after plan verification and after the early leakage check.
            manifest = plan_dataset.files["arrow"]
            manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return authority

    monkeypatch.setattr(train_pairwise, "load_packaged_artifact_authority", load_authority)
    monkeypatch.setattr(
        train_pairwise.Clusterer,
        "fit",
        lambda *_args, **_kwargs: pytest.fail("Stage 3 must not run clustering calibration"),
    )
    real_loader = train_pairwise.load_development_dataset

    def checked_loader(*loader_args: Any, **kwargs: Any) -> ANDData:
        assert kwargs["name_tuples"] == load_name_tuple_artifact(name_tuples_path).pairs
        assert kwargs["train_pairs_size"] == args.train_pairs_size
        assert kwargs["val_pairs_size"] == args.validation_pairs_size
        return real_loader(*loader_args, **kwargs)

    monkeypatch.setattr(train_pairwise, "load_development_dataset", checked_loader)
    if augmented_population in {"test", "omitted", "changed-arrow"}:
        monkeypatch.setattr(
            train_pairwise,
            "_featurize_selection",
            lambda *_args, **_kwargs: pytest.fail("unsafe inputs reached featurization"),
        )
        message = (
            "changed after model-plan verification"
            if augmented_population == "changed-arrow"
            else "outside its training split"
        )
        with pytest.raises(ValueError, match=message):
            train_pairwise.train_pairwise_bundle(args)
        assert not args.output_dir.exists()
        assert list(matrix_work_dir.iterdir()) == []
        return
    result = train_pairwise.train_pairwise_bundle(args)

    summary = result["training_summary"]
    assert summary["main_train_rows"] == expected_rows[0]
    assert summary["main_val_rows"] == expected_rows[1]
    assert summary["nameless_train_rows"] == expected_rows[0]
    assert built_members
    assert summary["main_pairwise_best_params"]
    assert summary["nameless_pairwise_best_params"]
    from s2and.production_training_contract import block_membership_sha256, frozen_test_blocks

    signatures = json.loads(development_files["signatures"].read_text(encoding="utf-8"))
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
    from lightgbm import Booster

    # Saved boosters name their features, as the released 1.3 boosters do, not Column_N.
    for head in ("main", "nameless"):
        booster = Booster(model_file=str(args.output_dir / "pairwise" / f"{head}.lgb"))
        assert booster.feature_name() == manifest["feature_names"][head]
    training_config = json.loads(
        (args.output_dir / "reproducibility" / "pairwise_training_config.json").read_text(encoding="utf-8")
    )
    assert training_config["featurization_backend"] == "rust_from_model_plan_arrow"
    assert training_config["unknown_signature_inputs"]["qian"]["signature_count"] == len(unknown_ids)
    assert training_config["dataset_inputs"]["qian"]["arrow_dir"] == str(plan_dataset.files["arrow"].parent)
    config = json.loads((args.output_dir / "clusterer.json").read_text(encoding="utf-8"))
    assert config["cluster_model"] == {"eps": 0.5, "linkage": "average"}
    assert list(matrix_work_dir.iterdir()) == []
    assert args.output_dir.is_dir()


def test_all_augmented_sources_are_checked_before_any_dataset_featurization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, name_counts_path, _ = write_pairwise_artifacts(tmp_path)
    args = pairwise_training_args(tmp_path)
    datasets: dict[str, ModelDataset] = {}
    members = {}
    for name in ("qian", "kisti"):
        datasets[name], development_files = _arrow_plan_dataset(
            tmp_path / "data", name, tmp_path / "dataset" / "benchmarks", name_counts_path
        )
        train, val, _ = _json_selection(development_files, name, args).split_cluster_signatures()
        members[name] = next(iter((train if name == "qian" else val).values()))[0]
    pair_path = tmp_path / "augmented_train.csv"
    pair_path.write_text(
        f"signature_id_1,signature_id_2,label\nqian___{members['qian']},kisti___{members['kisti']},0\n",
        encoding="utf-8",
    )
    datasets["augmented"] = ModelDataset(
        files={"train_pairs": pair_path}, sha256={"train_pairs": sha256_file(pair_path)}
    )
    work_dir = tmp_path / "matrix-work"
    work_dir.mkdir()
    plan = _preflight_plan(args, datasets, work_dir)
    monkeypatch.setattr(train_pairwise, "_preflight_pairwise", lambda _args: plan)
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_artifact_authority",
        lambda **_kwargs: pytest.fail("all leakage checks must precede runtime artifact loading"),
    )
    monkeypatch.setattr(
        train_pairwise,
        "_featurize_selection",
        lambda *_args, **_kwargs: pytest.fail("unsafe data reached featurization"),
    )
    with pytest.raises(ValueError, match="'kisti' signatures outside its training split"):
        train_pairwise.train_pairwise_bundle(args)
    assert not plan.output_dir.exists()
    assert list(work_dir.iterdir()) == []


@pytest.mark.parametrize("source_split", ["train", "validation"])
def test_augmented_validation_copies_must_come_from_the_source_validation_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_split: str,
) -> None:
    _, name_counts_path, _ = write_pairwise_artifacts(tmp_path)
    args = pairwise_training_args(tmp_path, output_dir=tmp_path / "production_model_v9.9")
    plan_dataset, development_files = _arrow_plan_dataset(
        tmp_path / "data", "qian", tmp_path / "dataset" / "benchmarks", name_counts_path
    )
    train, val, _test = _json_selection(development_files, "qian", args).split_cluster_signatures()
    train_ids = [signature for members in train.values() for signature in members]
    val_ids = [signature for members in val.values() for signature in members]
    augmented_files = {"train_pairs": tmp_path / "augmented_train.csv", "val_pairs": tmp_path / "augmented_val.csv"}
    augmented_files["train_pairs"].write_text(
        f"signature_id_1,signature_id_2,label\nqian___{train_ids[0]},qian___{train_ids[1]},1\n", encoding="utf-8"
    )
    validation_source = train_ids if source_split == "train" else val_ids
    augmented_files["val_pairs"].write_text(
        f"signature_id_1,signature_id_2,label\nqian___{val_ids[0]},qian___{validation_source[-1]},1\n",
        encoding="utf-8",
    )
    datasets = {
        "qian": plan_dataset,
        "augmented": ModelDataset(
            files=augmented_files,
            sha256={role: sha256_file(path) for role, path in augmented_files.items()},
        ),
    }
    monkeypatch.setattr(train_pairwise, "_preflight_pairwise", lambda _args: _preflight_plan(args, datasets, tmp_path))

    class GuardPassed(Exception):
        pass

    def stop_after_guard(**_kwargs: object) -> None:
        raise GuardPassed

    monkeypatch.setattr(train_pairwise, "load_packaged_artifact_authority", stop_after_guard)
    if source_split == "train":
        with pytest.raises(ValueError, match="Augmented validation contains 'qian' signatures outside its validation"):
            train_pairwise.train_pairwise_bundle(args)
    else:
        with pytest.raises(GuardPassed):
            train_pairwise.train_pairwise_bundle(args)
    assert not args.output_dir.exists()

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2and import production_training_contract
from s2and._sha256 import sha256_file
from scripts.production.model import train_pairwise
from tests.helpers import pairwise_training_args as _args


def _artifact_authority(
    *,
    tuple_hash: str = "a" * 64,
) -> train_pairwise.ProductionArtifactAuthority:
    return train_pairwise.ProductionArtifactAuthority(
        name_counts_index=SimpleNamespace(
            manifest_sha256="d" * 64,
        ),
        name_tuples=SimpleNamespace(data_sha256="a" * 64, pairs=frozenset()),
        orcid_prefix_counts=SimpleNamespace(
            name_tuples_sha256=tuple_hash,
            data_sha256="b" * 64,
        ),
    )


def _write_model_plan(tmp_path: Path) -> Path:
    input_root = tmp_path / "planned_inputs"
    input_root.mkdir()
    datasets = {}
    for name in (*train_pairwise.DEFAULT_SOURCE_DATASET_NAMES, "augmented"):
        roles = {"papers", "signatures", "specter_embeddings"}
        roles |= {"train_pairs", "val_pairs"} if name == "augmented" else {"clusters"}
        files = {}
        for role in sorted(roles):
            path = input_root / f"{name}_{role}"
            if role in {"train_pairs", "val_pairs"}:
                path.write_text("signature_id_1,signature_id_2,label\na,b,1\n", encoding="utf-8")
            else:
                path.write_text(f"{name}:{role}\n", encoding="utf-8")
            files[role] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        datasets[name] = files
    plan = {
        "release_version": "9.9",
        "datasets": datasets,
        "eps": {
            "grid": [0.3, 0.5, 0.7],
            "minimum_dataset_f1": 0.0,
            "minimum_signature_weighted_f1": 0.0,
        },
    }
    path = tmp_path / "model_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_import_preserves_backend_environment() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for case_id, backend in (("unset", None), ("python", "python")):
        env = os.environ.copy()
        if backend is None:
            env.pop("S2AND_BACKEND", None)
        else:
            env["S2AND_BACKEND"] = backend
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, os; import scripts.production.model.train_pairwise; "
                "print(json.dumps({'backend': os.environ.get('S2AND_BACKEND')}))",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, f"{case_id}: {completed.stderr}"
        assert json.loads(completed.stdout.strip().splitlines()[-1])["backend"] == backend, case_id


def test_preflight_loads_test_free_direct_path_plan(tmp_path: Path) -> None:
    plan_path = _write_model_plan(tmp_path)
    args = _args(
        tmp_path,
        model_plan=plan_path,
    )

    plan = train_pairwise._preflight_pairwise(args)
    config = train_pairwise._training_config(args, plan, artifact_hashes={})

    assert plan.dataset_names == (*train_pairwise.DEFAULT_SOURCE_DATASET_NAMES, "augmented")
    assert plan.release_version == "9.9"
    assert plan.model_plan_sha256 == sha256_file(plan_path)
    assert plan.datasets["aminer"].split_mode == "random_blocks"
    assert plan.datasets["augmented"].split_mode == "fixed_pairs"
    assert config["model_plan_sha256"] == plan.model_plan_sha256
    assert "release_version" not in config
    assert all(
        isinstance(path, str) for dataset in config["dataset_inputs"].values() for path in dataset["files"].values()
    )
    assert all("test_pairs" not in spec["files"] for spec in config["dataset_inputs"].values())
    assert list(plan.matrix_work_dir.iterdir()) == []


def test_training_rejects_source_mutation_before_loading_artifacts_or_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _write_model_plan(tmp_path)
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    changed_file = Path(plan_payload["datasets"]["aminer"]["clusters"]["path"])
    changed_file.write_text("mutated after preflight\n", encoding="utf-8")
    args = _args(tmp_path, model_plan=plan_path)
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_artifact_authority",
        lambda **_kwargs: pytest.fail("artifact loading must not run"),
    )
    monkeypatch.setattr(train_pairwise, "ANDData", lambda **_kwargs: pytest.fail("ANDData must not load"))

    with pytest.raises(ValueError):
        train_pairwise.train_pairwise_bundle(args)


def test_training_reaches_anddata_without_any_test_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _write_model_plan(tmp_path)
    args = _args(
        tmp_path,
        model_plan=plan_path,
    )

    class BoundaryReached(Exception):
        pass

    def inspect_anddata(**kwargs: object) -> None:
        assert kwargs["test_pairs"] is None
        assert kwargs["name_counts_index"] is authority.name_counts_index
        assert kwargs["name_tuples"] == frozenset()
        assert not {"train_ratio", "val_ratio", "test_ratio"} & set(kwargs)
        raise BoundaryReached

    authority = _artifact_authority()
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_artifact_authority",
        lambda **_kwargs: authority,
    )
    monkeypatch.setattr(train_pairwise, "_canonical_training_artifact_hashes", lambda _authority: {})
    monkeypatch.setattr(train_pairwise, "ANDData", inspect_anddata)

    with pytest.raises(BoundaryReached):
        train_pairwise.train_pairwise_bundle(args)


def test_augmented_dataset_contract_is_checked_before_featurization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = _write_model_plan(tmp_path)
    args = _args(tmp_path, model_plan=plan_path)
    full_plan = train_pairwise._preflight_pairwise(args)
    augmented_plan = train_pairwise.PairwisePreflightPlan(
        output_dir=full_plan.output_dir,
        release_version=full_plan.release_version,
        dataset_names=("augmented",),
        datasets={"augmented": full_plan.datasets["augmented"]},
        model_plan_sha256=full_plan.model_plan_sha256,
        matrix_work_dir=full_plan.matrix_work_dir,
        matrix_work_free_bytes=full_plan.matrix_work_free_bytes,
        total_ram_bytes=full_plan.total_ram_bytes,
    )
    authority = _artifact_authority()
    fake_dataset = SimpleNamespace(
        name_counts_manifest_sha256="e" * 64,
        name_tuples=authority.name_tuples.pairs,
    )
    monkeypatch.setattr(train_pairwise, "_preflight_pairwise", lambda _args: augmented_plan)
    monkeypatch.setattr(train_pairwise, "load_packaged_artifact_authority", lambda **_kwargs: authority)
    monkeypatch.setattr(train_pairwise, "ANDData", lambda **_kwargs: fake_dataset)
    monkeypatch.setattr(
        train_pairwise,
        "_featurize_selection",
        lambda *_args, **_kwargs: pytest.fail("featurization ran before dataset contract validation"),
    )

    with pytest.raises(ValueError, match=r"dataset 'augmented'.*name-count manifest mismatch"):
        train_pairwise.train_pairwise_bundle(args)


def test_preflight_rejects_unsafe_launch(tmp_path: Path) -> None:
    cases = (
        ("n-jobs", {"n_jobs": 0}, "--n-jobs must be positive"),
        ("total-ram", {"total_ram_bytes": 0}, "--total-ram-bytes must be positive"),
        ("run-full", {"run_full": False}, "requires --run-full"),
    )
    for case_id, change, message in cases:
        case_root = tmp_path / case_id
        case_root.mkdir()
        plan_path = _write_model_plan(case_root)
        args = _args(
            case_root,
            model_plan=plan_path,
        )
        for field, value in change.items():
            setattr(args, field, value)

        with pytest.raises(SystemExit, match=message):
            train_pairwise._preflight_pairwise(args)


def test_preflight_rejects_existing_output_and_nonempty_scratch(tmp_path: Path) -> None:
    plan_path = _write_model_plan(tmp_path)
    args = _args(
        tmp_path,
        model_plan=plan_path,
    )
    args.output_dir.mkdir()
    with pytest.raises(SystemExit, match="must name a new directory"):
        train_pairwise._preflight_pairwise(args)

    args.output_dir.rmdir()
    (args.matrix_work_dir / "stale.npy").write_bytes(b"stale")
    with pytest.raises(SystemExit, match="--matrix-work-dir must be empty"):
        train_pairwise._preflight_pairwise(args)


def test_artifact_authority_uses_packaged_runtime_data(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: dict[str, Path] = {}

    def open_orcid(root: Path) -> SimpleNamespace:
        opened["orcid"] = root
        return SimpleNamespace(
            name_tuples_sha256="a" * 64,
            data_sha256="b" * 64,
        )

    def open_name_counts(path: Path) -> SimpleNamespace:
        opened["name_counts"] = path
        return SimpleNamespace(
            manifest_sha256="d" * 64,
        )

    monkeypatch.setattr(
        production_training_contract.NameCountsIndex,
        "open",
        staticmethod(open_name_counts),
    )
    monkeypatch.setattr(
        production_training_contract,
        "load_packaged_name_tuple_artifact",
        lambda: SimpleNamespace(data_sha256="a" * 64),
    )
    monkeypatch.setattr(production_training_contract, "load_canonical_orcid_prefix_counts", open_orcid)

    authority = production_training_contract.load_packaged_artifact_authority(
        name_counts_index_root=Path("name-counts"),
    )

    assert train_pairwise._canonical_training_artifact_hashes(_artifact_authority()) == authority.hashes
    assert opened == {
        "name_counts": Path("name-counts"),
        "orcid": Path(production_training_contract._PACKAGE_DATA_DIR),
    }


def test_artifact_validation_rejects_mismatched_name_tuples() -> None:
    with pytest.raises(RuntimeError, match="different canonical name-tuple"):
        train_pairwise._canonical_training_artifact_hashes(_artifact_authority(tuple_hash="x" * 64))


def test_staging_keeps_only_train_validation_and_checks_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = (
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([1], dtype=np.int8),
        np.asarray([[2.0, 3.0]], dtype=np.float32),
    )
    paths = train_pairwise._stage_dataset_features(
        tmp_path,
        "toy",
        train=split,
        val=split,
    )
    union_path = train_pairwise._concatenate_staged_arrays(
        tmp_path / "union.npy",
        [paths["X_train"], paths["X_val"]],
    )
    np.testing.assert_array_equal(
        train_pairwise._load_staged_array(union_path),
        np.vstack([split[0], split[0]]),
    )

    monkeypatch.setattr(train_pairwise.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="Insufficient disk"):
        train_pairwise._stage_array(tmp_path / "too_large.npy", np.ones(10))


def test_selected_validation_roc_auc_must_be_finite() -> None:
    for value in (np.nan, np.inf):
        with pytest.raises(RuntimeError, match="must be finite"):
            train_pairwise._finite_validation_roc_auc(value)

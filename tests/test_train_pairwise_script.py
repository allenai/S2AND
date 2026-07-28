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


@pytest.mark.parametrize("backend", [None, "python"])
def test_import_preserves_backend_environment(backend: str | None) -> None:
    repo_root = Path(__file__).resolve().parents[1]
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
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1])["backend"] == backend


def test_parser_has_one_explicit_release_mode() -> None:
    common = [
        "--production-version",
        "9.9",
        "--output-dir",
        "production_model_v9.9",
        "--matrix-work-dir",
        "matrices",
        "--name-counts-index-root",
        "name-counts-index",
        "--model-plan",
        "model-plan.json",
    ]
    parser = train_pairwise.build_parser()
    parsed = parser.parse_args([*common, "--run-full"])

    assert parsed.run_full is True
    with pytest.raises(SystemExit):
        parser.parse_args(common)


def test_preflight_loads_test_free_direct_path_plan(tmp_path: Path) -> None:
    plan_path = _write_model_plan(tmp_path)
    args = _args(
        tmp_path,
        model_plan=plan_path,
    )

    plan = train_pairwise._preflight_pairwise(args)
    config = train_pairwise._training_config(args, plan, artifact_hashes={})

    assert plan.dataset_names == (*train_pairwise.DEFAULT_SOURCE_DATASET_NAMES, "augmented")
    assert plan.model_plan_sha256 == sha256_file(plan_path)
    assert plan.datasets["aminer"].split_mode == "random_blocks"
    assert plan.datasets["augmented"].split_mode == "fixed_pairs"
    assert config["model_plan_sha256"] == plan.model_plan_sha256
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
        assert kwargs["name_counts_index"] == args.name_counts_index_root
        assert kwargs["name_tuples"] == frozenset()
        assert not {"train_ratio", "val_ratio", "test_ratio"} & set(kwargs)
        raise BoundaryReached

    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_artifact_authority",
        lambda **_kwargs: _artifact_authority(),
    )
    monkeypatch.setattr(train_pairwise, "_canonical_training_artifact_hashes", lambda _authority: {})
    monkeypatch.setattr(train_pairwise, "ANDData", inspect_anddata)

    with pytest.raises(BoundaryReached):
        train_pairwise.train_pairwise_bundle(args)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n_jobs": 0}, "--n-jobs must be positive"),
        ({"total_ram_bytes": 0}, "--total-ram-bytes must be positive"),
        ({"production_version": " 9.9 "}, "no surrounding whitespace"),
        ({"run_full": False}, "requires --run-full"),
    ],
)
def test_preflight_rejects_unsafe_launch(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    plan_path = _write_model_plan(tmp_path)
    args = _args(
        tmp_path,
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


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_selected_validation_roc_auc_must_be_finite(value: float) -> None:
    with pytest.raises(RuntimeError, match="must be finite"):
        train_pairwise._finite_validation_roc_auc(value)

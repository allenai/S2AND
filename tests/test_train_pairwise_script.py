from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s2and import production_training_contract
from scripts.production.model import train_pairwise
from tests.helpers import pairwise_training_args as _args


def _artifact_authority(
    *,
    orcid_source: str = "redshift:test",
    tuple_hash: str = "a" * 64,
    name_source: str = "redshift:test",
) -> train_pairwise.ProductionArtifactAuthority:
    return train_pairwise.ProductionArtifactAuthority(
        name_counts_index=SimpleNamespace(
            manifest_sha256="d" * 64,
            source_provenance={"source_kind": name_source},
        ),
        name_tuples=SimpleNamespace(data_sha256="a" * 64, pairs=frozenset()),
        orcid_prefix_counts=SimpleNamespace(
            source_kind=orcid_source,
            name_tuples_sha256=tuple_hash,
            data_sha256="b" * 64,
            manifest_sha256="c" * 64,
        ),
    )


def _write_training_plan(tmp_path: Path) -> tuple[Path, str]:
    input_root = tmp_path / "planned_inputs"
    input_root.mkdir()
    datasets = []
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
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        datasets.append(
            {
                "name": name,
                "split_mode": "fixed_pairs" if name == "augmented" else "random_blocks",
                "files": files,
            }
        )
    plan = {
        "schema_version": train_pairwise.TRAINING_PLAN_SCHEMA,
        "source_manifest_sha256": "a" * 64,
        "datasets": datasets,
        "sealed_test_manifests": {
            "pairwise": {"manifest_sha256": "b" * 64, "members": {"test": {"pairs": "c" * 64}}},
            "cluster": {"manifest_sha256": "d" * 64, "members": {"test": {"blocks": "e" * 64}}},
        },
    }
    path = tmp_path / "training_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, sha256(path.read_bytes()).hexdigest()


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
        "--training-plan",
        "training-plan.json",
        "--expected-training-plan-sha256",
        "a" * 64,
    ]
    parser = train_pairwise.build_parser()
    parsed = parser.parse_args([*common, "--run-full"])

    assert parsed.run_full is True
    with pytest.raises(SystemExit):
        parser.parse_args(common)


def test_preflight_loads_test_free_digest_bound_plan(tmp_path: Path) -> None:
    plan_path, plan_sha256 = _write_training_plan(tmp_path)
    args = _args(
        tmp_path,
        training_plan=plan_path,
        expected_training_plan_sha256=plan_sha256,
    )

    plan = train_pairwise._preflight_pairwise(args)
    config = train_pairwise._training_config(args, plan, artifact_hashes={})

    assert plan.dataset_names == (*train_pairwise.DEFAULT_SOURCE_DATASET_NAMES, "augmented")
    assert config["pairwise_inputs_manifest_sha256"] == "a" * 64
    assert config["sealed_test_manifests"]["pairwise"]["manifest_sha256"] == "b" * 64
    assert '"path"' not in json.dumps(config["sealed_test_manifests"])
    assert all("test_pairs" not in spec["files"] for spec in config["dataset_inputs"].values())
    assert list(plan.matrix_work_dir.iterdir()) == []


def test_preflight_requires_exact_training_plan_digest(tmp_path: Path) -> None:
    plan_path, _ = _write_training_plan(tmp_path)
    args = _args(
        tmp_path,
        training_plan=plan_path,
        expected_training_plan_sha256="0" * 64,
    )

    with pytest.raises(SystemExit, match="Training-plan SHA-256 mismatch"):
        train_pairwise._preflight_pairwise(args)


def test_training_reaches_anddata_without_any_test_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan_sha256 = _write_training_plan(tmp_path)
    args = _args(
        tmp_path,
        training_plan=plan_path,
        expected_training_plan_sha256=plan_sha256,
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
    plan_path, plan_sha256 = _write_training_plan(tmp_path)
    args = _args(
        tmp_path,
        training_plan=plan_path,
        expected_training_plan_sha256=plan_sha256,
    )
    for field, value in change.items():
        setattr(args, field, value)

    with pytest.raises(SystemExit, match=message):
        train_pairwise._preflight_pairwise(args)


def test_preflight_rejects_existing_output_and_nonempty_scratch(tmp_path: Path) -> None:
    plan_path, plan_sha256 = _write_training_plan(tmp_path)
    args = _args(
        tmp_path,
        training_plan=plan_path,
        expected_training_plan_sha256=plan_sha256,
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
            manifest_sha256="c" * 64,
        )

    def open_name_counts(path: Path) -> SimpleNamespace:
        opened["name_counts"] = path
        return SimpleNamespace(
            manifest_sha256="d" * 64,
            source_provenance={"source_kind": "redshift:snapshot"},
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


@pytest.mark.parametrize(
    ("orcid_source", "tuple_hash", "name_source", "message"),
    [
        ("fixture:test", "a" * 64, "redshift:test", "warehouse-generated ORCID"),
        ("redshift:test", "x" * 64, "redshift:test", "different canonical name-tuple"),
        ("redshift:test", "a" * 64, "fixture:test", "warehouse-generated name-count"),
    ],
)
def test_artifact_validation_rejects_nonproduction_inputs(
    orcid_source: str,
    tuple_hash: str,
    name_source: str,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        train_pairwise._canonical_training_artifact_hashes(
            _artifact_authority(
                orcid_source=orcid_source,
                tuple_hash=tuple_hash,
                name_source=name_source,
            )
        )


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


def test_source_mutation_is_detected_after_preflight(tmp_path: Path) -> None:
    plan_path, plan_sha256 = _write_training_plan(tmp_path)
    plan = train_pairwise._preflight_pairwise(
        _args(
            tmp_path,
            training_plan=plan_path,
            expected_training_plan_sha256=plan_sha256,
        )
    )
    plan.datasets["aminer"].files["papers"].write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="aminer:papers"):
        train_pairwise._assert_sources_unchanged(plan)

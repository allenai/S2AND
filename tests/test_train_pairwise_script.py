from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from scripts.production.model import train_pairwise


def _write_qian_inputs(data_dir: Path) -> dict[str, Path]:
    dataset_dir = data_dir / "qian"
    dataset_dir.mkdir(parents=True)
    paths = {
        "clusters": dataset_dir / "qian_clusters.json",
        "signatures": dataset_dir / "qian_signatures.json",
        "papers": dataset_dir / "qian_papers.json",
        "specter_embeddings": dataset_dir / "qian_specter2.pkl",
    }
    for role, path in paths.items():
        path.write_bytes(f"{role}\n".encode())
    return paths


def _args(tmp_path: Path, *, data_dir: Path | None = None, output_dir: Path | None = None) -> Namespace:
    matrix_work_dir = tmp_path / "matrix_work"
    matrix_work_dir.mkdir(exist_ok=True)
    return Namespace(
        run_full=False,
        preflight_only=False,
        data_dir=data_dir or tmp_path / "data",
        output_dir=output_dir or tmp_path / "production_model_v9.9",
        production_version="9.9",
        n_iter=1,
        cluster_n_iter=1,
        n_jobs=1,
        chunk_size=8,
        train_pairs_size=4,
        val_test_size=2,
        random_seed=1111,
        datasets=["qian"],
        signatures_suffix=train_pairwise.DEFAULT_SIGNATURES_SUFFIX,
        specter_suffix=train_pairwise.DEFAULT_SPECTER_SUFFIX,
        feature_cache_dir=None,
        matrix_work_dir=matrix_work_dir,
        total_ram_bytes=1_000_000_000_000,
        training_plan=None,
        expected_training_plan_sha256=None,
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
            files[role] = {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
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


def test_parser_requires_paths_and_has_three_unambiguous_modes() -> None:
    parser = train_pairwise.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--production-version", "9.9", "--run-full"])

    common = [
        "--production-version",
        "9.9",
        "--data-dir",
        "data",
        "--output-dir",
        "production_model_v9.9",
        "--matrix-work-dir",
        "matrices",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--run-full", "--preflight-only"])
    for removed_option in (
        "--negative-one-for-nan",
        "--smoke-only",
        "--min-free-disk-bytes",
        "--no-include-augmented",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([*common, removed_option])


def test_preflight_hashes_inputs_and_derives_smoke_scope(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    paths = _write_qian_inputs(data_dir)
    args = _args(tmp_path, data_dir=data_dir)
    args.random_seed = 0

    plan = train_pairwise._preflight_pairwise(args)

    assert plan.dataset_names == ("qian",)
    assert plan.datasets["qian"].files == {role: path.resolve() for role, path in paths.items()}
    assert plan.datasets["qian"].sha256 == {role: sha256(path.read_bytes()).hexdigest() for role, path in paths.items()}
    assert plan.matrix_work_free_bytes > 0
    assert list(plan.matrix_work_dir.iterdir()) == []


def test_preflight_rejects_nonempty_matrix_work_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_qian_inputs(data_dir)
    args = _args(tmp_path, data_dir=data_dir)
    (args.matrix_work_dir / "stale.npy").write_bytes(b"stale")

    with pytest.raises(SystemExit, match="--matrix-work-dir must be empty"):
        train_pairwise._preflight_pairwise(args)


def test_preflight_records_measured_matrix_work_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _write_qian_inputs(data_dir)
    args = _args(tmp_path, data_dir=data_dir)
    monkeypatch.setattr(train_pairwise.shutil, "disk_usage", lambda _path: SimpleNamespace(free=99))

    plan = train_pairwise._preflight_pairwise(args)
    config = train_pairwise._training_config(args, plan, artifact_hashes={})

    assert plan.matrix_work_free_bytes == 99
    assert config["matrix_work_storage"] == {
        "path": str(args.matrix_work_dir.resolve()),
        "measured_free_bytes": 99,
    }


def test_preflight_probes_matrix_work_writability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _write_qian_inputs(data_dir)
    args = _args(tmp_path, data_dir=data_dir)

    def reject_write(**_kwargs: object) -> None:
        raise PermissionError("read-only fixture")

    monkeypatch.setattr(train_pairwise.tempfile, "NamedTemporaryFile", reject_write)

    with pytest.raises(SystemExit, match="--matrix-work-dir is not writable"):
        train_pairwise._preflight_pairwise(args)


def test_preflight_rejects_feature_cache_at_output_path(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_qian_inputs(data_dir)
    args = _args(tmp_path, data_dir=data_dir)
    args.feature_cache_dir = args.output_dir

    with pytest.raises(SystemExit, match="must differ from --output-dir"):
        train_pairwise._preflight_pairwise(args)


def test_full_preflight_uses_test_free_plan_without_data_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan_sha256 = _write_training_plan(tmp_path)
    args = _args(tmp_path)
    args.datasets = None
    args.run_full = True
    args.data_dir = tmp_path / "must-not-be-discovered"
    args.training_plan = plan_path
    args.expected_training_plan_sha256 = plan_sha256
    monkeypatch.setattr(train_pairwise, "_resolve_dataset_inputs", lambda **_kwargs: pytest.fail("data discovered"))

    plan = train_pairwise._preflight_pairwise(args)

    assert plan.dataset_names == (*train_pairwise.DEFAULT_SOURCE_DATASET_NAMES, "augmented")
    config = train_pairwise._training_config(args, plan, artifact_hashes={})  # noqa: SLF001
    assert config["pairwise_inputs_manifest_sha256"] == "a" * 64
    assert config["sealed_test_manifests"]["pairwise"]["manifest_sha256"] == "b" * 64
    assert '"path"' not in json.dumps(config["sealed_test_manifests"])
    assert all("test_pairs" not in spec["files"] for spec in config["dataset_inputs"].values())


def test_full_preflight_requires_digest_bound_training_plan(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.datasets = None
    args.preflight_only = True

    with pytest.raises(SystemExit, match="--training-plan and --expected-training-plan-sha256"):
        train_pairwise._preflight_pairwise(args)


def test_training_plan_rejects_wrong_digest(tmp_path: Path) -> None:
    plan_path, _ = _write_training_plan(tmp_path)

    with pytest.raises(SystemExit, match="Training-plan SHA-256 mismatch"):
        train_pairwise._verified_training_plan(plan_path, "0" * 64)  # noqa: SLF001


def test_full_training_reaches_anddata_without_any_test_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan_sha256 = _write_training_plan(tmp_path)
    args = _args(tmp_path)
    args.datasets = None
    args.run_full = True
    args.data_dir = tmp_path / "must-not-be-discovered"
    args.training_plan = plan_path
    args.expected_training_plan_sha256 = plan_sha256

    class BoundaryReached(Exception):
        pass

    def inspect_anddata(**kwargs: object) -> None:
        assert kwargs["test_pairs"] is None
        assert not {"train_ratio", "val_ratio", "test_ratio"} & set(kwargs)
        assert all("test" not in str(value) for key, value in kwargs.items() if key.endswith("_pairs"))
        raise BoundaryReached

    monkeypatch.setattr(train_pairwise, "_resolve_dataset_inputs", lambda **_kwargs: pytest.fail("data discovered"))
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_name_tuple_artifact",
        lambda: SimpleNamespace(data_sha256="a" * 64),
    )
    monkeypatch.setattr(train_pairwise, "_canonical_training_artifact_hashes", lambda _sha: {})
    monkeypatch.setattr(train_pairwise, "ANDData", inspect_anddata)

    with pytest.raises(BoundaryReached):
        train_pairwise.train_pairwise_bundle(args)


def test_release_input_resolution_never_opens_fixed_test_pairs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "augmented"
    dataset_dir.mkdir(parents=True)
    for filename in (
        "augmented_signatures.json",
        "augmented_papers.json",
        "augmented_specter2.pkl",
        "train_pairs.csv",
        "val_pairs.csv",
    ):
        (dataset_dir / filename).write_text("fixture\n", encoding="utf-8")

    plan = train_pairwise._resolve_dataset_inputs(  # noqa: SLF001
        data_dir=tmp_path,
        dataset_name="augmented",
        signatures_suffix=train_pairwise.DEFAULT_SIGNATURES_SUFFIX,
        specter_suffix=train_pairwise.DEFAULT_SPECTER_SUFFIX,
        include_test_pairs=False,
    )

    assert "test_pairs" not in plan.files


def test_release_staging_contains_only_train_and_validation(tmp_path: Path) -> None:
    split = (
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([1], dtype=np.int8),
        np.asarray([[2.0]], dtype=np.float32),
    )

    paths = train_pairwise._stage_dataset_features(  # noqa: SLF001
        tmp_path,
        "toy",
        train=split,
        val=split,
    )

    assert set(paths) == {"X_train", "y_train", "nameless_X_train", "X_val", "y_val", "nameless_X_val"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"datasets": ["qian", "qian"]}, "duplicate names"),
        ({"datasets": ["unknown"]}, "unknown names"),
        ({"n_jobs": 0}, "--n-jobs must be positive"),
        ({"total_ram_bytes": 0}, "--total-ram-bytes must be positive"),
        ({"production_version": " 9.9 "}, "no surrounding whitespace"),
        ({"run_full": True}, "--run-full forbids --datasets"),
        ({"datasets": None}, "full production training requires --run-full"),
    ],
)
def test_preflight_rejects_invalid_or_ambiguous_launch(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    data_dir = tmp_path / "data"
    _write_qian_inputs(data_dir)
    args = _args(tmp_path, data_dir=data_dir)
    for field, value in change.items():
        setattr(args, field, value)

    with pytest.raises(SystemExit, match=message):
        train_pairwise._preflight_pairwise(args)


def test_preflight_rejects_output_before_artifact_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _write_qian_inputs(data_dir)
    output_dir = tmp_path / "production_model_v9.9"
    output_dir.mkdir()
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_name_tuple_artifact",
        lambda: pytest.fail("artifact loading started"),
    )

    with pytest.raises(SystemExit, match="must name a new directory"):
        train_pairwise.train_pairwise_bundle(_args(tmp_path, data_dir=data_dir, output_dir=output_dir))


def test_preflight_rejects_version_mismatch_and_missing_later_dataset_before_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _write_qian_inputs(data_dir)
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_name_tuple_artifact",
        lambda: pytest.fail("artifact loading started"),
    )
    mismatch = _args(tmp_path, data_dir=data_dir, output_dir=tmp_path / "production_model_v8.8")
    with pytest.raises(SystemExit, match="basename and --production-version disagree"):
        train_pairwise.train_pairwise_bundle(mismatch)

    missing = _args(tmp_path, data_dir=data_dir)
    missing.datasets = ["qian", "pubmed"]
    with pytest.raises(FileNotFoundError, match="pubmed"):
        train_pairwise.train_pairwise_bundle(missing)


def test_artifact_validation_uses_canonical_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_pairwise,
        "load_canonical_orcid_prefix_counts",
        lambda _root: SimpleNamespace(
            source_kind="redshift:snapshot",
            name_tuples_sha256="a" * 64,
            data_sha256="b" * 64,
            manifest_sha256="c" * 64,
        ),
    )
    monkeypatch.setattr(
        train_pairwise.NameCountsIndex,
        "open",
        staticmethod(
            lambda _path: SimpleNamespace(
                manifest_sha256="d" * 64,
                source_provenance={"source_kind": "redshift:snapshot"},
            )
        ),
    )

    assert train_pairwise._canonical_training_artifact_hashes("a" * 64) == {
        "name_tuples_data_sha256": "a" * 64,
        "name_counts_manifest_sha256": "d" * 64,
        "orcid_prefix_counts_data_sha256": "b" * 64,
        "orcid_prefix_counts_manifest_sha256": "c" * 64,
    }


@pytest.mark.parametrize(
    ("orcid_source", "tuple_hash", "name_source", "message"),
    [
        ("fixture:test", "a" * 64, "redshift:test", "warehouse-generated ORCID"),
        ("redshift:test", "x" * 64, "redshift:test", "different canonical name-tuple"),
        ("redshift:test", "a" * 64, "fixture:test", "warehouse-generated name-count"),
    ],
)
def test_artifact_validation_rejects_nonproduction_or_mismatched_inputs(
    monkeypatch: pytest.MonkeyPatch,
    orcid_source: str,
    tuple_hash: str,
    name_source: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        train_pairwise,
        "load_canonical_orcid_prefix_counts",
        lambda _root: SimpleNamespace(
            source_kind=orcid_source,
            name_tuples_sha256=tuple_hash,
            data_sha256="b" * 64,
            manifest_sha256="c" * 64,
        ),
    )
    monkeypatch.setattr(
        train_pairwise.NameCountsIndex,
        "open",
        staticmethod(
            lambda _path: SimpleNamespace(
                manifest_sha256="d" * 64,
                source_provenance={"source_kind": name_source},
            )
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        train_pairwise._canonical_training_artifact_hashes("a" * 64)


def test_preflight_only_has_no_output_or_dataset_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    _write_qian_inputs(data_dir)
    output_dir = tmp_path / "absent" / "production_model_v9.9"
    args = _args(tmp_path, data_dir=data_dir, output_dir=output_dir)
    args.preflight_only = True
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_name_tuple_artifact",
        lambda: SimpleNamespace(data_sha256="a" * 64),
    )
    monkeypatch.setattr(
        train_pairwise,
        "_canonical_training_artifact_hashes",
        lambda _sha: {"name_tuples_data_sha256": "a" * 64},
    )
    monkeypatch.setattr(train_pairwise, "ANDData", lambda **_kwargs: pytest.fail("ANDData loaded"))

    result = train_pairwise.train_pairwise_bundle(args)

    assert result["mode"] == "preflight"
    assert result["training_config"]["training_scope"] == "smoke_subset"
    assert result["training_config"]["cluster_n_iter"] == args.cluster_n_iter
    assert not output_dir.exists()
    assert not output_dir.parent.exists()


def test_staged_union_preserves_order_and_checks_actual_disk_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    second = np.asarray([[5.0, 6.0]], dtype=np.float64)
    first_path = train_pairwise._stage_array(tmp_path / "first.npy", first)
    second_path = train_pairwise._stage_array(tmp_path / "second.npy", second)
    union_path = train_pairwise._concatenate_staged_arrays(tmp_path / "union.npy", [first_path, second_path])
    np.testing.assert_array_equal(train_pairwise._load_staged_array(union_path), np.vstack([first, second]))

    monkeypatch.setattr(train_pairwise.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="Insufficient disk"):
        train_pairwise._stage_array(tmp_path / "too_large.npy", np.ones(10))


def test_source_mutation_is_detected_after_featurization(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    paths = _write_qian_inputs(data_dir)
    plan = train_pairwise._preflight_pairwise(_args(tmp_path, data_dir=data_dir))
    paths["papers"].write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="qian:papers"):
        train_pairwise._assert_sources_unchanged(plan)


def test_smoke_pairwise_metrics_average_both_models() -> None:
    metrics = train_pairwise._smoke_pairwise_metrics(
        labels=np.asarray([0, 1]),
        main_probabilities=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        nameless_probabilities=np.asarray([[0.8, 0.2], [0.2, 0.8]]),
    )
    assert metrics["rows"] == 2
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_smoke_pairwise_metrics_match_the_release_evaluator_schema() -> None:
    """Smoke evidence must exercise the sealed release report schema, not a parallel one."""

    from scripts.production.model.release_pairwise import pairwise_metrics

    labels = np.asarray([0, 1, 1, 0])
    main = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9], [0.6, 0.4]])
    nameless = np.asarray([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.9, 0.1]])

    smoke = train_pairwise._smoke_pairwise_metrics(
        labels=labels,
        main_probabilities=main,
        nameless_probabilities=nameless,
    )
    release, _ = pairwise_metrics(labels, main[:, 1], nameless[:, 1])

    # The smoke report is the release report plus one diagnostic-only field.
    assert set(smoke) - set(release) == {"average_precision"}
    assert set(release) - set(smoke) == set()
    for key, value in release.items():
        assert smoke[key] == pytest.approx(value), key


def test_subset_result_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        train_pairwise,
        "write_pairwise_production_bundle",
        lambda *_args, **_kwargs: pytest.fail("smoke run published a bundle"),
    )

    result = train_pairwise._publish_result(
        args,
        train_pairwise.PairwisePreflightPlan(
            output_dir=args.output_dir,
            dataset_names=("qian",),
            datasets={},
            feature_cache_dir=None,
            matrix_work_dir=args.matrix_work_dir,
            matrix_work_free_bytes=1_000_000,
            total_ram_bytes=args.total_ram_bytes,
            source_manifest_sha256=None,
            sealed_test_manifests={},
        ),
        cast(train_pairwise.Clusterer, SimpleNamespace()),
        {"training_scope": "smoke_subset"},
        {"smoke_pairwise_test_metrics": {"qian": {"auroc": 1.0}}},
    )

    assert result["mode"] == "smoke"
    assert result["bundle_status"] == "not_published"
    assert result["training_config"]["training_scope"] == "smoke_subset"
    assert result["training_summary"]["smoke_pairwise_test_metrics"]["qian"]["auroc"] == pytest.approx(1.0)
    assert not args.output_dir.exists()

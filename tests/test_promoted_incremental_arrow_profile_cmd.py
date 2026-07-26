from __future__ import annotations

import hashlib
import json

import pytest

from scripts._rust_suite import promoted_incremental_arrow_profile_cmd as cmd


def _bound_args(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_manifest = model_dir / "manifest.json"
    model_manifest.write_text("{}\n", encoding="utf-8")
    arrow_root = tmp_path / "arrow"
    dataset_root = arrow_root / "dummy"
    dataset_root.mkdir(parents=True)
    data_manifest = dataset_root / "manifest.json"
    data_manifest.write_text("{}\n", encoding="utf-8")
    return cmd.parse_args(
        [
            "--arrow-root",
            str(arrow_root),
            "--dataset",
            "dummy",
            "--expected-data-manifest-sha256",
            hashlib.sha256(data_manifest.read_bytes()).hexdigest(),
            "--model-path",
            str(model_dir),
            "--expected-model-manifest-sha256",
            hashlib.sha256(model_manifest.read_bytes()).hexdigest(),
            "--expected-workload-sha256",
            "0" * 64,
            "--write-json",
            str(tmp_path / "report.json"),
        ]
    )


def test_select_workload_uses_largest_block_and_stable_seed_queries() -> None:
    workload = cmd._select_workload(
        blocks={
            "small": ["x"],
            "large": ["a", "b", "c", "d", "e"],
        },
        signature_to_cluster_id={
            "a": "cluster-1",
            "b": "cluster-1",
            "c": "cluster-2",
            "d": "cluster-3",
            "e": "cluster-3",
        },
        target_block="",
        query_limit=2,
        max_seed_clusters=2,
    )

    assert workload.target_block == "large"
    assert workload.block_signature_count == 5
    assert workload.seed_signature_to_cluster == {"a": "cluster-1", "c": "cluster-2"}
    assert workload.query_signature_ids == ["b", "d"]
    assert workload.block_signatures == ["a", "c", "b", "d"]


@pytest.mark.parametrize(
    ("query_limit", "message"),
    [("0", "--full-run"), ("-1", "--query-limit must be >= 0")],
)
def test_run_rejects_unbounded_or_negative_query_limit_without_full_run(query_limit: str, message: str) -> None:
    args = cmd.parse_args(
        [
            "--dataset",
            "dummy",
            "--expected-data-manifest-sha256",
            "1" * 64,
            "--model-path",
            "model",
            "--expected-model-manifest-sha256",
            "2" * 64,
            "--expected-workload-sha256",
            "3" * 64,
            "--write-json",
            "report.json",
            "--query-limit",
            query_limit,
        ]
    )

    with pytest.raises(ValueError, match=message):
        cmd.run(args)


def test_profile_main_writes_fresh_report(tmp_path, monkeypatch) -> None:
    report_path = tmp_path / "report.json"
    payload = {"schema_version": cmd.REPORT_SCHEMA, "summary": {"run_count": 1}}
    monkeypatch.setattr(cmd, "run", lambda _args: payload)

    exit_code = cmd.main(
        [
            "--dataset",
            "dummy",
            "--expected-data-manifest-sha256",
            "1" * 64,
            "--model-path",
            "model",
            "--expected-model-manifest-sha256",
            "2" * 64,
            "--expected-workload-sha256",
            "3" * 64,
            "--write-json",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    ("wrong_binding", "message"),
    [
        ("model", "Model manifest SHA-256 mismatch"),
        ("data", "Data manifest SHA-256 mismatch"),
    ],
)
def test_profile_wrong_manifest_digest_fails_before_data_loading(
    tmp_path,
    monkeypatch,
    wrong_binding,
    message,
) -> None:
    args = _bound_args(tmp_path)
    setattr(args, f"expected_{wrong_binding}_manifest_sha256", "f" * 64)
    monkeypatch.setattr(
        cmd,
        "_resolve_arrow_dataset_paths",
        lambda *_args, **_kwargs: pytest.fail("data loading must not run"),
    )

    with pytest.raises(ValueError, match=message):
        cmd.run(args)


def test_profile_wrong_workload_digest_fails_before_model_load(tmp_path, monkeypatch) -> None:
    from s2and import production_model

    args = _bound_args(tmp_path)
    monkeypatch.setattr(
        cmd, "_resolve_arrow_dataset_paths", lambda *_args, **_kwargs: {"signatures": "x", "clusters": "y"}
    )
    monkeypatch.setattr(cmd, "_read_signature_rows", lambda _path: [])
    monkeypatch.setattr(cmd, "_block_dict", lambda _rows: {"block": ["seed", "query"]})
    monkeypatch.setattr(cmd, "_read_signature_to_cluster_id", lambda _path: {"seed": "cluster"})
    monkeypatch.setattr(
        production_model,
        "load_production_model",
        lambda _path: pytest.fail("model loading must not run"),
    )

    with pytest.raises(ValueError, match="Workload SHA-256 mismatch"):
        cmd.run(args)

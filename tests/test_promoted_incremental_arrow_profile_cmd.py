from __future__ import annotations

from pathlib import Path

import pytest

from scripts import rust_suite
from scripts._rust_suite import promoted_incremental_arrow_profile_cmd as cmd


def test_promoted_incremental_arrow_profile_is_canonical_command() -> None:
    parsed = rust_suite._build_cli_parser().parse_args(["promoted-incremental-arrow-profile"])
    assert parsed.command == "promoted-incremental-arrow-profile"

    with pytest.raises(SystemExit):
        rust_suite._build_cli_parser().parse_args(["big-block-incremental"])


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


def test_run_refuses_unbounded_query_batch_without_full_run() -> None:
    args = cmd.parse_args(["--dataset", "dummy", "--model-path", "model", "--query-limit", "0"])

    with pytest.raises(ValueError, match="--full-run"):
        cmd.run(args)


def test_run_refuses_negative_query_limit_without_full_run() -> None:
    args = cmd.parse_args(["--dataset", "dummy", "--model-path", "model", "--query-limit", "-1"])

    with pytest.raises(ValueError, match="--query-limit must be >= 0"):
        cmd.run(args)


def test_run_uses_direct_arrow_api_and_forwards_batching_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from s2and import production_model
    from s2and.incremental_linking import feature_block

    captured: dict[str, object] = {}

    class FakeClusterer:
        n_jobs = 1

        def predict_incremental_from_arrow_paths(self, block_signatures, arrow_paths, **kwargs):
            captured["block_signatures"] = block_signatures
            captured["arrow_paths"] = arrow_paths
            captured["batching_threshold"] = kwargs["batching_threshold"]
            return {"clusters": {"cluster-1": list(block_signatures)}, "incremental_linker_telemetry": {}}

    class FakeMonitor:
        peak_gb = 0.0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    arrow_paths = cmd.ValidatedArrowInputs._from_verified(
        paths={"signatures": str(tmp_path / "signatures.arrow"), "clusters": str(tmp_path / "clusters.json")},
        generation_id="test-generation",
        normalization_version="test-normalization",
    )
    rows = [
        cmd.ArrowSignatureRow("seed", "p1", "block", "A", "", "Name", None),
        cmd.ArrowSignatureRow("query", "p2", "block", "A", "", "Name", None),
    ]

    monkeypatch.setattr(cmd, "_resolve_arrow_dataset_paths", lambda *_args: arrow_paths)
    monkeypatch.setattr(cmd, "_read_signature_rows", lambda _path: rows)
    monkeypatch.setattr(cmd, "_read_signature_to_cluster_id", lambda _path: {"seed": "c1", "query": "c1"})
    monkeypatch.setattr(cmd, "ProcessTreeRSSMonitor", FakeMonitor)
    monkeypatch.setattr(cmd, "collect_rust_extension_identity", lambda **_kwargs: {})
    monkeypatch.setattr(cmd, "build_run_metadata", lambda **_kwargs: {})
    monkeypatch.setattr(cmd, "_set_runtime_env", lambda _n_jobs: {})
    monkeypatch.setattr(cmd, "_restore_runtime_env", lambda _prior: None)
    monkeypatch.setattr(production_model, "load_production_model", lambda _path: FakeClusterer())
    monkeypatch.setattr(
        feature_block,
        "write_cluster_seeds_arrow",
        lambda path, _seeds: Path(path).write_bytes(b"cluster seeds"),
    )

    args = cmd.parse_args(
        [
            "--arrow-root",
            str(tmp_path),
            "--dataset",
            "dummy",
            "--model-path",
            str(tmp_path / "model"),
            "--query-limit",
            "1",
            "--batching-threshold",
            "7",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    cmd.run(args)

    assert captured["block_signatures"] == ["seed", "query"]
    assert captured["batching_threshold"] == 7
    captured_arrow_paths = captured["arrow_paths"]
    assert isinstance(captured_arrow_paths, cmd.ValidatedArrowInputs)
    assert "cluster_seeds" in captured_arrow_paths

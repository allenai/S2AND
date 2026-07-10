from __future__ import annotations

import pytest

from scripts import rust_suite
from scripts._rust_suite import promoted_incremental_arrow_profile_cmd as cmd


def test_promoted_incremental_arrow_profile_is_canonical_command() -> None:
    assert "promoted-incremental-arrow-profile" in rust_suite._COMMANDS  # noqa: SLF001
    assert "big-block-incremental" not in rust_suite._COMMANDS  # noqa: SLF001


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

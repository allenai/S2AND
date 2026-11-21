"""Query author evidence stays exact across reusable planner calls."""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from s2and.feature_port import _require_rust_runtime
from tests.raw_arrow_helpers import base_arrow_paths, open_arrow_dataset, write_ipc


def _assert_same_plan(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Compare emitted values and ordering, excluding execution telemetry."""
    assert actual.keys() == expected.keys()
    for key in actual.keys() - {"telemetry"}:
        if isinstance(actual[key], np.ndarray):
            left, right = actual[key], expected[key]
            assert left.dtype == right.dtype, key
            assert left.shape == right.shape, key
            if left.dtype.hasobject:
                np.testing.assert_array_equal(left, right, err_msg=key)
            else:
                assert left.tobytes() == right.tobytes(), key
        else:
            assert actual[key] == expected[key], key


@pytest.mark.parametrize("num_threads", [1, 2])
def test_query_author_evidence_matches_fresh_plans_when_roles_change(tmp_path: Path, num_threads: int) -> None:
    """A seed paper can first need query-only evidence, then enter a cached summary."""
    paths = base_arrow_paths(tmp_path)
    runtime = _require_rust_runtime()
    names = {
        "q1": {"alice wang", "ann smith"},
        "s1": {"alice wang", "ann smith"},
        "s2": {"bob jones", "carl doe"},
    }
    local_names = {"q1": {"ann smith"}, "s1": {"ann smith"}, "s2": {"carl doe"}}
    members = {"c_match": "s1", "c_other": "s2"}
    calls = [
        (["q1"], []),
        (["s2"], []),
        (["q1"], [("q1", "s1")]),
        (["s2"], []),
        (["s1"], []),
        (["q1"], [("q1", "s1"), ("q1", "s2")]),
        (["q1"], []),
    ]
    with open_arrow_dataset(paths) as arrow:

        def make_planner() -> Any:
            return runtime.RawBlockQueryCandidatePlanner.from_auto_queries(
                arrow.native, paths["cluster_seeds"], top_k=1, num_threads=num_threads
            )

        planner = make_planner()
        empty_calls = 0
        for queries, disallows in calls:
            actual = dict(planner.plan(queries, disallows))
            fresh = make_planner()
            expected = dict(fresh.plan(queries, disallows))
            _assert_same_plan(actual, expected)
            empty_calls += int(actual["row_count"] == 0)
            for row, (query_index, component_key) in enumerate(
                zip(actual["row_query_signature_indices"], actual["row_component_keys"], strict=True)
            ):
                query = queries[int(query_index)]
                member = members[component_key]
                overlap = len(names[query] & names[member])
                local_overlap = len(local_names[query] & local_names[member])
                assert actual["row_paper_author_list_max_jaccard"][row] == overlap / len(names[query] | names[member])
                assert actual["row_paper_author_list_max_containment"][row] == overlap / 2
                assert actual["row_paper_author_list_max_overlap_count"][row] == overlap
                assert actual["row_local_author_window10_jaccard_max"][row] == local_overlap
                assert actual["row_local_author_window10_overlap_count_max"][row] == local_overlap
                assert actual["row_best_author_count_log_absdiff"][row] == 0
        assert empty_calls == 1


def test_query_author_tables_are_unneeded_without_candidates(tmp_path: Path) -> None:
    """Exercise the native planner's empty-seed route across repeated query calls."""
    paths = base_arrow_paths(tmp_path)
    paths["cluster_seeds"] = write_ipc(
        tmp_path / "cluster_seeds.arrow",
        pa.table({"signature_id": pa.array([], type=pa.string()), "cluster_id": pa.array([], type=pa.string())}),
    )
    with open_arrow_dataset(paths) as arrow:
        planner = _require_rust_runtime().RawBlockQueryCandidatePlanner.from_auto_queries(
            arrow.native, paths["cluster_seeds"], top_k=1, num_threads=2
        )
        for queries in (["q1"], ["s1", "s2"], ["s2", "q1", "s1"]):
            payload = dict(planner.plan(queries))
            assert payload["query_signature_ids"] == queries
            assert payload["row_count"] == payload["pair_count"] == 0
            assert payload["row_paper_author_list_max_jaccard"].size == 0

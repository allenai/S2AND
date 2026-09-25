"""Raw Arrow linker row signals on papers with thousands of authors.

The planner and the labeled-row route read each paper's authors once per call
and intersect interned name ids. These tests pin the paper author-list evidence
to an independent Python reference and pin both routes to stay fast on
2,000-author papers, where per-signature normalization and string-set
intersections scaled with authors squared per paper. Time bounds are process
CPU seconds, so machine load does not trip them.
"""

from __future__ import annotations

import math
import string
import time
from functools import cache
from typing import Any

import numpy as np
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.feature_port import _require_rust_runtime
from s2and.incremental_linking.feature_block_arrow import temporary_cluster_seed_sidecars
from s2and.text import normalize_text
from tests.helpers import write_arrow_tables_bundle

PAPER_COUNT = 3
AUTHORS_PER_PAPER = 2000
PAPER_OFFSET = 300
EVIDENCE_COLUMNS = (
    "row_paper_author_list_max_jaccard",
    "row_paper_author_list_max_containment",
    "row_paper_author_list_max_overlap_count",
    "row_local_author_window10_jaccard_max",
    "row_local_author_window10_overlap_count_max",
    "row_best_author_count_log_absdiff",
)


def _surname(person: int) -> str:
    letters = []
    for _ in range(3):
        person, digit = divmod(person, 26)
        letters.append(string.ascii_lowercase[digit])
    return "Q" + "".join(letters)


def _paper_people(paper: int) -> list[int]:
    """People at positions 0.. of `paper`; consecutive papers share most of them."""

    return list(range(paper * PAPER_OFFSET, paper * PAPER_OFFSET + AUTHORS_PER_PAPER))


@pytest.fixture(scope="module")
def giant_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    import pyarrow as pa

    signature_rows: list[tuple[str, str, int, int]] = []
    author_rows: list[tuple[str, int, str]] = []
    for paper in range(PAPER_COUNT):
        paper_id = f"giant-{paper}"
        for position, person in enumerate(_paper_people(paper)):
            author_rows.append((paper_id, position, f"Anna {_surname(person)}"))
            signature_rows.append((f"{paper_id}-{position}", paper_id, position, person))
    signature_ids = [row[0] for row in signature_rows]
    lasts = [_surname(row[3]) for row in signature_rows]
    root = tmp_path_factory.mktemp("giant_authors") / "arrow"
    write_arrow_tables_bundle(
        root,
        {
            "signatures": pa.table(
                {
                    "signature_id": pa.array(signature_ids, type=pa.string()),
                    "paper_id": pa.array([row[1] for row in signature_rows], type=pa.string()),
                    "author_first": pa.array(["Anna"] * len(signature_rows), type=pa.string()),
                    "author_middle": pa.array([""] * len(signature_rows), type=pa.string()),
                    "author_last": pa.array(lasts, type=pa.string()),
                    "author_suffix": pa.array([""] * len(signature_rows), type=pa.string()),
                    "author_affiliations": pa.array([[]] * len(signature_rows), type=pa.list_(pa.string())),
                    "author_position": pa.array([row[2] for row in signature_rows], type=pa.int64()),
                    "author_block": pa.array([f"a {last.lower()}" for last in lasts], type=pa.string()),
                }
            ),
            "papers": pa.table(
                {
                    "paper_id": pa.array([f"giant-{paper}" for paper in range(PAPER_COUNT)], type=pa.string()),
                    "title": pa.array(["Observation of a new boson"] * PAPER_COUNT, type=pa.string()),
                    "venue": pa.array([""] * PAPER_COUNT, type=pa.string()),
                    "journal_name": pa.array([""] * PAPER_COUNT, type=pa.string()),
                }
            ),
            "paper_authors": pa.table(
                {
                    "paper_id": pa.array([row[0] for row in author_rows], type=pa.string()),
                    "position": pa.array([row[1] for row in author_rows], type=pa.int64()),
                    "author_name": pa.array([row[2] for row in author_rows], type=pa.string()),
                }
            ),
        },
    )
    components: dict[str, list[str]] = {}
    for signature_id, _paper_id, _position, person in signature_rows:
        components.setdefault(f"person-{person}", []).append(signature_id)
    signatures = {row[0]: (row[1], row[2], row[3]) for row in signature_rows}
    queries = signature_ids[::10]
    return {"root": root, "components": components, "signatures": signatures, "queries": queries}


@cache
def _paper_names(paper_id: str) -> tuple[str, ...]:
    return tuple(normalize_text(f"Anna {_surname(person)}") for person in _paper_people(int(paper_id.split("-")[1])))


@cache
def _paper_name_set(paper_id: str) -> frozenset[str]:
    return frozenset(_paper_names(paper_id))


@cache
def _local_names(paper_id: str, position: int) -> frozenset[str]:
    names = _paper_names(paper_id)
    window = range(max(0, position - 10), min(len(names), position + 11))
    return frozenset(names[other] for other in window if other != position)


def _round_six(value: float) -> np.float32:
    return np.float32(np.round(value * 1_000_000.0) / 1_000_000.0)


def _reference_evidence(
    query: str,
    members: list[str],
    signatures: dict[str, tuple[str, int, int]],
) -> list[np.float32]:
    """Python reference of the paper author-list evidence of one candidate row."""

    query_paper, query_position, _ = signatures[query]
    query_set = _paper_name_set(query_paper)
    query_local = _local_names(query_paper, query_position)
    jaccard = containment = overlap = local_jaccard = local_overlap = 0.0
    log_absdiff = math.inf
    for member in members:
        member_paper, member_position, _ = signatures[member]
        member_set = _paper_name_set(member_paper)
        intersection = len(query_set & member_set)
        jaccard = max(jaccard, intersection / len(query_set | member_set))
        containment = max(containment, intersection / min(len(query_set), len(member_set)))
        overlap = max(overlap, float(intersection))
        author_counts = (len(_paper_names(query_paper)), len(_paper_names(member_paper)))
        log_absdiff = min(log_absdiff, abs(math.log1p(author_counts[0]) - math.log1p(author_counts[1])))
        if member == query:
            continue
        member_local = _local_names(member_paper, member_position)
        local_intersection = len(query_local & member_local)
        local_union = len(query_local | member_local)
        if local_union:
            local_jaccard = max(local_jaccard, local_intersection / local_union)
        local_overlap = max(local_overlap, float(local_intersection))
    return [
        _round_six(value)
        for value in (jaccard, containment, overlap, local_jaccard, local_overlap, 0.0 if not members else log_absdiff)
    ]


def _assert_evidence_matches(
    payload: dict[str, Any],
    row_queries: list[str],
    row_members: list[list[str]],
    signatures: dict[str, tuple[str, int, int]],
    fractional_column: str,
) -> None:
    observed = np.column_stack([np.asarray(payload[column], dtype=np.float32) for column in EVIDENCE_COLUMNS])
    expected = np.asarray(
        [
            _reference_evidence(query, members, signatures)
            for query, members in zip(row_queries, row_members, strict=True)
        ],
        dtype=np.float32,
    )
    assert observed.shape == expected.shape
    assert np.array_equal(observed.view(np.uint32), expected.view(np.uint32))
    fractional = observed[:, EVIDENCE_COLUMNS.index(fractional_column)]
    assert ((fractional > 0) & (fractional < 1)).any(), f"{fractional_column} takes fractional values"


def test_labeled_rows_on_giant_papers_match_reference_and_stay_fast(giant_dataset: dict[str, Any]) -> None:
    """Every query is a member of its first candidate, so each row rebuilds a residual component summary."""

    signatures = giant_dataset["signatures"]
    components = giant_dataset["components"]
    row_queries: list[str] = []
    row_keys: list[str] = []
    ranks: list[int] = []
    for query in giant_dataset["queries"]:
        person = signatures[query][2]
        for rank, key in enumerate((f"person-{person}", f"person-{person + 1}"), start=1):
            if key in components:
                row_queries.append(query)
                row_keys.append(key)
                ranks.append(rank)
    with ArrowDataset.open(giant_dataset["root"]) as arrow:
        started = time.process_time()
        payload = dict(
            _require_rust_runtime().raw_arrow_labeled_candidate_plan(
                arrow.native,
                row_queries,
                ["full"] * len(row_queries),
                [f"{query}|full" for query in row_queries],
                row_keys,
                ranks,
                components,
                orcid_enabled=True,
                num_threads=2,
                max_exemplars=4,
            )
        )
        elapsed = time.process_time() - started
    row_members = [
        [member for member in components[key] if member != query]
        for query, key in zip(row_queries, row_keys, strict=True)
    ]
    _assert_evidence_matches(payload, row_queries, row_members, signatures, "row_paper_author_list_max_jaccard")
    assert elapsed < 4.0, f"labeled candidate plan used {elapsed:.2f} CPU s on {PAPER_COUNT} giant papers"


def test_planner_rows_on_giant_papers_match_reference_and_stay_fast(giant_dataset: dict[str, Any]) -> None:
    signatures = giant_dataset["signatures"]
    # Retrieval scores every component against each query's ~2,000 coauthor
    # terms, so this route uses fewer, larger components and fewer queries.
    queries = giant_dataset["queries"][::10]
    query_set = set(queries)
    components: dict[str, list[str]] = {}
    for signature_id, (_paper_id, _position, person) in signatures.items():
        if signature_id not in query_set:
            components.setdefault(f"group-{person // 20}", []).append(signature_id)
    seeds = {member: key for key, members in components.items() for member in members}
    with ArrowDataset.open(giant_dataset["root"]) as arrow:
        with temporary_cluster_seed_sidecars(seeds, prefix="s2and_giant_authors_") as sidecars:
            started = time.process_time()
            planner = _require_rust_runtime().RawBlockQueryCandidatePlanner.from_auto_queries(
                arrow.native,
                sidecars["cluster_seeds"],
                top_k=5,
                cluster_seed_disallows_path=None,
                orcid_enabled=True,
                num_threads=2,
                max_exemplars=4,
            )
            payload = dict(planner.plan(queries))
            del planner
            elapsed = time.process_time() - started
    plan_queries = [str(query) for query in payload["query_signature_ids"]]
    row_queries = [plan_queries[int(index)] for index in payload["row_query_signature_indices"]]
    row_members = [components[str(key)] for key in payload["row_component_keys"]]
    assert len(row_queries) == 5 * len(queries)
    # Every candidate has a member on the query's paper, so only the local window is fractional.
    _assert_evidence_matches(payload, row_queries, row_members, signatures, "row_local_author_window10_jaccard_max")
    assert elapsed < 3.0, f"raw planner build and plan used {elapsed:.2f} CPU s on {PAPER_COUNT} giant papers"


def test_heavy_queries_score_many_components_quickly_and_match_direct_scores(giant_dataset: dict[str, Any]) -> None:
    """Heavy queries against one component per person: the planner's indexed retrieval scores equal the
    labeled route's per-component scores, and scoring ~2,000 coauthor terms against ~2,500 components stays fast.
    """

    signatures = giant_dataset["signatures"]
    queries = giant_dataset["queries"][::10]
    query_set = set(queries)
    components: dict[str, list[str]] = {}
    for signature_id, (_paper_id, _position, person) in signatures.items():
        if signature_id not in query_set:
            components.setdefault(f"person-{person:05d}", []).append(signature_id)
    seeds = {member: key for key, members in components.items() for member in members}
    runtime = _require_rust_runtime()
    with ArrowDataset.open(giant_dataset["root"]) as arrow:
        with temporary_cluster_seed_sidecars(seeds, prefix="s2and_heavy_queries_") as sidecars:
            planner = runtime.RawBlockQueryCandidatePlanner.from_auto_queries(
                arrow.native,
                sidecars["cluster_seeds"],
                top_k=5,
                cluster_seed_disallows_path=None,
                orcid_enabled=True,
                num_threads=2,
                max_exemplars=4,
            )
            started = time.process_time()
            payload = dict(planner.plan(queries))
            elapsed = time.process_time() - started
            del planner
        plan_queries = [str(query) for query in payload["query_signature_ids"]]
        row_queries = [plan_queries[int(index)] for index in payload["row_query_signature_indices"]]
        row_keys = [str(key) for key in payload["row_component_keys"]]
        labeled = dict(
            runtime.raw_arrow_labeled_candidate_plan(
                arrow.native,
                row_queries,
                ["auto"] * len(row_queries),
                [f"{query}|auto" for query in row_queries],
                row_keys,
                [int(rank) for rank in payload["retrieval_ranks"]],
                components,
                orcid_enabled=True,
                num_threads=2,
                max_exemplars=4,
            )
        )
    planner_scores = np.asarray(payload["retrieval_scores"], dtype=np.float32).astype(np.float64)
    rounded = np.round(planner_scores * 1_000_000.0) / 1_000_000.0
    assert [str(key) for key in labeled["row_component_keys"]] == row_keys
    assert np.array_equal(
        rounded.astype(np.float32).view(np.uint32),
        np.asarray(labeled["retrieval_scores"], dtype=np.float32).view(np.uint32),
    )
    assert len(row_keys) == 5 * len(queries)
    assert elapsed < 3.0, f"{len(queries)} heavy queries over {len(components)} components used {elapsed:.2f} CPU s"

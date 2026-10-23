from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_arrow_ipc_table
from s2and.subblocking import (
    GraphSubblockingConfig,
    _pack_component_ids_greedy,
    _projection_neighbor_edge_scores,
    _prune_edge_scores,
    _score_candidate_edge,
    _sorted_subblock_merge_candidates,
    cluster_with_graph_fallback,
    make_dataset_graph_subblocking_cluster_fn,
    make_subblocks_with_telemetry,
)
from tests.helpers import write_test_arrow_artifact_manifest


def _open_graph_dataset(paths: dict[str, Any], tmp_path) -> ArrowDataset:
    import pyarrow as pa

    with pa.memory_map(str(paths["signatures"]), "r") as source:
        signatures = pa.ipc.open_file(source).read_all()
    for column_name, value in (
        ("author_last", ""),
        ("author_suffix", ""),
        ("author_block", "h test"),
    ):
        if column_name not in signatures.column_names:
            signatures = signatures.append_column(
                column_name,
                pa.array([value] * signatures.num_rows, type=pa.string()),
            )
    signatures_path = tmp_path / "signatures.complete.arrow"
    write_arrow_ipc_table(signatures, signatures_path)
    paths["signatures"] = signatures_path
    paper_ids = {str(value) for value in signatures["paper_id"].to_pylist() if value is not None and str(value)}
    papers_path = tmp_path / "papers.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(sorted(paper_ids) or ["p1"], type=pa.string()),
                "title": pa.array([""] * max(1, len(paper_ids)), type=pa.string()),
                "venue": pa.array([""] * max(1, len(paper_ids)), type=pa.string()),
                "journal_name": pa.array([""] * max(1, len(paper_ids)), type=pa.string()),
            }
        ),
        papers_path,
    )
    paths["papers"] = papers_path
    index_specs = (
        ("signatures", "signatures_batch_index", "signature_id"),
        ("papers", "papers_batch_index", "paper_id"),
        ("paper_authors", "paper_authors_batch_index", "paper_id"),
        ("specter", "specter_batch_index", "paper_id"),
    )
    for table_key, index_key, key_column in index_specs:
        index_path = tmp_path / f"{table_key}.{index_key}.bin"
        write_arrow_batch_lookup_index(paths[table_key], index_path, key_column=key_column, table_name=table_key)
        paths[index_key] = index_path
    write_test_arrow_artifact_manifest(tmp_path, {key: str(path) for key, path in paths.items()})
    return ArrowDataset.open(tmp_path, require_specter=True)


def test_sorted_subblock_merge_candidates_handles_edge_cases() -> None:
    output = {
        "alex": ["a1", "a2"],
        "alex|middle=w": ["a3"],
        "alexander": ["a4"],
        "a|middle=wei": ["s1"],
        "a|middle=weijun": ["s2"],
        "a|middle=li": ["s3", "s4"],
        "b": ["skip"],
        "bo": ["b1"],
        "carol": ["c1", "c2", "c3"],
        "david": ["d1", "d2", "d3", "d4"],
    }
    counts = {
        "alex": {"bo": 7},
        "li": {"wei": 11},
    }

    observed = _sorted_subblock_merge_candidates(output, maximum_size=5, first_k_letter_counts_sorted=counts)

    assert (("alex", "bo"), 7) in observed
    assert (("a|middle=wei", "a|middle=li"), 11) in observed
    assert (("carol", "david"), 0) not in observed
    assert any(pair == ("a|middle=wei", "a|middle=weijun") for pair, _score in observed)


def test_sorted_subblock_merge_candidates_middle_prefix_score_is_order_invariant() -> None:
    def score_for(output):
        candidates = _sorted_subblock_merge_candidates(output, maximum_size=3, first_k_letter_counts_sorted={})
        assert len(candidates) == 1
        return candidates[0][1]

    short_first = {
        "alex|middle=jo": ["s1"],
        "alex|middle=john": ["s2"],
    }
    long_first = {
        "alex|middle=john": ["s2"],
        "alex|middle=jo": ["s1"],
    }

    assert score_for(short_first) == score_for(long_first) == 1e10 + 2


def test_sorted_subblock_merge_candidates_orcid_counts_are_order_invariant() -> None:
    counts = {"li": {"wei": 7}}
    forward = _sorted_subblock_merge_candidates(
        {"a|middle=wei": ["s1"], "a|middle=li": ["s2"]},
        maximum_size=3,
        first_k_letter_counts_sorted=counts,
    )
    reverse = _sorted_subblock_merge_candidates(
        {"a|middle=li": ["s2"], "a|middle=wei": ["s1"]},
        maximum_size=3,
        first_k_letter_counts_sorted=counts,
    )

    assert [(frozenset(pair), score) for pair, score in forward] == [
        (frozenset(pair), score) for pair, score in reverse
    ]
    assert forward[0][1] == 7


def test_sorted_subblock_merge_candidates_keeps_exact_maximum_size_pair() -> None:
    output = {
        "alex|middle=a": ["s1", "s2"],
        "alex|middle=b": ["s3", "s4", "s5"],
    }

    assert _sorted_subblock_merge_candidates(output, maximum_size=5, first_k_letter_counts_sorted={}) == [
        (("alex|middle=a", "alex|middle=b"), 1e10)
    ]


@pytest.mark.parametrize("row_count", [1, 6], ids=["singleton-no-self-edges", "overlapping-neighborhoods"])
def test_projection_neighbor_edge_scores_match_slow_reference(row_count: int) -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.96, 0.20, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.95, 0.30],
            [0.0, 0.0, 1.0],
            [0.20, 0.0, 0.98],
        ],
        dtype=np.float32,
    )
    matrix = matrix[:row_count]
    matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    evidences = [
        SimpleNamespace(coauthor_blocks=frozenset({"a"}), affiliation_keys=frozenset({"lab1"})),
        SimpleNamespace(coauthor_blocks=frozenset({"a"}), affiliation_keys=frozenset({"lab2"})),
        SimpleNamespace(coauthor_blocks=frozenset({"b"}), affiliation_keys=frozenset({"lab2"})),
        SimpleNamespace(coauthor_blocks=frozenset({"b"}), affiliation_keys=frozenset({"lab3"})),
        SimpleNamespace(coauthor_blocks=frozenset({"c"}), affiliation_keys=frozenset({"lab3"})),
        SimpleNamespace(coauthor_blocks=frozenset({"c"}), affiliation_keys=frozenset({"lab1"})),
    ]
    evidences = evidences[:row_count]
    config = GraphSubblockingConfig(
        projection_count=4,
        projection_window=3,
        min_edge_score=0.25,
        max_candidate_edges=10_000,
    )
    rng = np.random.default_rng(11)
    projection_vectors = rng.standard_normal((matrix.shape[1], int(config.projection_count)), dtype=np.float32)
    projection_norms = np.linalg.norm(projection_vectors, axis=0)
    projection_vectors[:, projection_norms > 0] /= projection_norms[projection_norms > 0]
    projection_scores = matrix @ projection_vectors
    expected: dict[tuple[int, int], float] = {}
    for projection_index in range(projection_scores.shape[1]):
        order = np.argsort(projection_scores[:, projection_index], kind="mergesort")
        for position, left_index_raw in enumerate(order):
            stop = min(len(order), position + int(config.projection_window) + 1)
            for right_index_raw in order[position + 1 : stop]:
                _score_candidate_edge(
                    expected,
                    left_index=int(left_index_raw),
                    right_index=int(right_index_raw),
                    matrix=matrix,
                    evidences=cast(Any, evidences),
                    config=config,
                )

    observed = _projection_neighbor_edge_scores(matrix, cast(Any, evidences), config, seed=11)

    assert set(observed) == set(expected)
    for edge, expected_score in expected.items():
        assert observed[edge] == pytest.approx(expected_score)


def test_prune_edge_scores_tie_breaker_is_independent_of_insertion_order() -> None:
    forward = {(0, 3): 1.0, (0, 1): 1.0, (0, 2): 1.0}
    reverse = dict(reversed(list(forward.items())))

    _prune_edge_scores(forward, 2)
    _prune_edge_scores(reverse, 2)

    assert tuple(forward) == ((0, 1), (0, 2))
    assert tuple(reverse) == tuple(forward)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("edge-greedy", [[0, 2], [1, 3]]), ("aggregate-greedy", [[0, 3], [1, 2]])],
)
def test_component_packing_uses_selected_affinity_within_capacity(strategy, expected) -> None:
    # A and B cannot share a bin. C favors A's best edge (.8), but B's total (1.1).
    components = [["a0", "a1"], ["b0", "b1"], ["c"], ["d"]]
    edges = {(0, 1): 100.0, (0, 4): 0.8, (1, 4): 0.1, (2, 4): 0.5, (3, 4): 0.6}
    bins = _pack_component_ids_greedy(
        components,
        edges,
        root_by_index=[0, 0, 2, 2, 4, 5],
        component_id_by_root={0: 0, 2: 1, 4: 2, 5: 3},
        target_subblock_size=3,
        config=GraphSubblockingConfig(component_pack_strategy=strategy),
    )

    assert bins == expected
    assert [sum(len(components[index]) for index in bucket) for bucket in bins] == [3, 3]


def test_graph_subblocking_packs_micro_components_before_legacy_merge() -> None:
    signature_ids = [f"s{index}" for index in range(6)]
    signatures = {
        signature_id: SimpleNamespace(
            signature_id=signature_id,
            paper_id=f"p{index}",
            author_info_first="hui",
            author_info_middle="",
            author_info_first_normalized="hui",
            author_info_middle_normalized="",
            author_info_affiliations=(),
            author_info_affiliations_n_grams=None,
            author_info_coauthor_blocks=(),
            author_info_coauthors=None,
            author_info_orcid=None,
            author_info_position=0,
        )
        for index, signature_id in enumerate(signature_ids)
    }
    dataset = SimpleNamespace(
        signatures=signatures,
        papers={},
        specter_embeddings={f"p{index}": np.zeros(2, dtype=np.float32) for index in range(6)},
        random_seed=13,
    )
    fallback = make_dataset_graph_subblocking_cluster_fn(
        config=GraphSubblockingConfig(
            neighbor_mode="projection",
            projection_count=2,
            projection_window=2,
            min_edge_score=999.0,
            pack_components=True,
        )
    )

    subblocks, telemetry = make_subblocks_with_telemetry(
        signature_ids,
        dataset,
        maximum_size=2,
        first_k_letter_counts_sorted={},
        specter_cluster_fn=fallback,
    )

    assert telemetry["specter_invocation_count"] == 1
    assert telemetry["pre_merge_specter_labeled_subblock_count"] == 3
    assert telemetry["final_specter_labeled_subblock_count"] == 3
    assert max(len(values) for values in subblocks.values()) == 2
    assert fallback.stats[0]["raw_component_count"] == 6
    assert fallback.stats[0]["packed_component_count"] == 3


def test_graph_subblocking_uses_raw_paper_coauthors_when_signature_blocks_are_missing() -> None:
    signature_ids = ["s1", "s2", "s3", "s4"]
    signatures = {
        signature_id: SimpleNamespace(
            signature_id=signature_id,
            paper_id=f"p{index}",
            author_info_affiliations=(),
            author_info_affiliations_n_grams=None,
            author_info_coauthor_blocks=None,
            author_info_coauthors=None,
            author_info_position=0,
        )
        for index, signature_id in enumerate(signature_ids)
    }
    papers = {
        "p0": SimpleNamespace(
            authors=[
                SimpleNamespace(author_name="Hui Wang", position=0),
                SimpleNamespace(author_name="Ada Lovelace", position=1),
            ]
        ),
        "p1": SimpleNamespace(
            authors=[
                SimpleNamespace(author_name="Hui Wang", position=0),
                SimpleNamespace(author_name="Grace Hopper", position=1),
            ]
        ),
        "p2": SimpleNamespace(
            authors=[
                SimpleNamespace(author_name="Hui Wang", position=0),
                SimpleNamespace(author_name="Ada Lovelace", position=1),
            ]
        ),
        "p3": SimpleNamespace(
            authors=[
                SimpleNamespace(author_name="Hui Wang", position=0),
                SimpleNamespace(author_name="Grace Hopper", position=1),
            ]
        ),
    }
    dataset = SimpleNamespace(
        signatures=signatures,
        papers=papers,
        specter_embeddings={},
        random_seed=0,
    )
    hidden_stats: list[dict[str, object]] = []
    dataset._graph_subblocking_stats = hidden_stats

    subblocks = cluster_with_graph_fallback(
        signature_ids,
        dataset,
        target_subblock_size=2,
        config=GraphSubblockingConfig(
            neighbor_mode="exact",
            neighbors=3,
            min_edge_score=0.5,
            specter_weight=0.0,
            coauthor_weight=1.0,
            affiliation_weight=0.0,
            pack_components=False,
        ),
    )

    assert {frozenset(values) for values in subblocks.values()} == {
        frozenset({"s1", "s3"}),
        frozenset({"s2", "s4"}),
    }
    assert hidden_stats == []

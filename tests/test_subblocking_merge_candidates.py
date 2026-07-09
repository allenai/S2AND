from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from s2and.arrow_inputs import MissingArrowArtifactError
from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_arrow_ipc_table
from s2and.subblocking import (
    GraphSubblockingConfig,
    _projection_neighbor_edge_scores,
    _prune_edge_scores,
    _score_candidate_edge,
    _sorted_subblock_merge_candidates,
    cluster_with_graph_fallback,
    make_arrow_graph_subblocking_cluster_fn,
    make_dataset_graph_subblocking_cluster_fn,
    make_subblocks_with_telemetry,
)
from s2and.text import same_prefix_tokens


def _legacy_sorted_subblock_merge_candidates(output, maximum_size, first_k_letter_counts_sorted):
    small_enough_keys = [k for k, v in output.items() if len(v) < maximum_size]
    small_enough_pairs_counts = []
    for pair in list(combinations(small_enough_keys, 2)):
        if len(output[pair[0]]) + len(output[pair[1]]) <= maximum_size:
            pair_0_split = pair[0].split("|")
            pair_1_split = pair[1].split("|")

            first_name_1 = pair_0_split[0]
            first_name_2 = pair_1_split[0]

            if len(pair_0_split) > 1:
                middle_name_1 = pair_0_split[1].split("=")[1]
            else:
                middle_name_1 = None

            if len(pair_1_split) > 1:
                middle_name_2 = pair_1_split[1].split("=")[1]
            else:
                middle_name_2 = None

            if len(first_name_1) > 1 and len(first_name_2) > 1:
                name_for_splits_1 = first_name_1
                name_for_splits_2 = first_name_2
            elif (
                len(first_name_1) == 1
                and len(first_name_2) == 1
                and middle_name_1 is not None
                and middle_name_2 is not None
            ):
                name_for_splits_1 = middle_name_1
                name_for_splits_2 = middle_name_2
            else:
                continue

            if name_for_splits_1 == name_for_splits_2:
                if middle_name_1 is not None and middle_name_2 is not None:
                    score = 0
                    for i in range(1, len(middle_name_1)):
                        if middle_name_1[:i] == middle_name_2[:i]:
                            score = i
                else:
                    score = 0
                small_enough_pairs_counts.append((pair, 1e10 + score))
            elif same_prefix_tokens(name_for_splits_1, name_for_splits_2):
                score = min(len(name_for_splits_1), len(name_for_splits_2))
                small_enough_pairs_counts.append((pair, 1e5 + score))
            else:
                # canonical_v2: prefix-count lookups use the full canonical name
                # string (the legacy first-token reduction is retired).
                lookup_1 = name_for_splits_1
                lookup_2 = name_for_splits_2
                if lookup_1 in first_k_letter_counts_sorted and lookup_2 in first_k_letter_counts_sorted[lookup_1]:
                    small_enough_pairs_counts.append((pair, first_k_letter_counts_sorted[lookup_1][lookup_2]))

    return sorted(small_enough_pairs_counts, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)


def _write_arrow_ipc_batches(path, batches) -> None:
    pa = pytest.importorskip("pyarrow")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, batches[0].schema) as writer:
            for batch in batches:
                writer.write_batch(batch)


def _add_graph_batch_indexes(paths: dict[str, Any], tmp_path) -> dict[str, Any]:
    specter_key = "specter2" if "specter2" in paths and "specter" not in paths else "specter"
    index_specs = (
        ("signatures", "signatures_batch_index", "signature_id"),
        ("paper_authors", "paper_authors_batch_index", "paper_id"),
        (specter_key, f"{specter_key}_batch_index", "paper_id"),
    )
    for table_key, index_key, key_column in index_specs:
        index_path = tmp_path / f"{table_key}.{index_key}.bin"
        write_arrow_batch_lookup_index(paths[table_key], index_path, key_column=key_column, table_name=table_key)
        paths[index_key] = index_path
    return paths


def test_sorted_subblock_merge_candidates_matches_legacy_edge_cases() -> None:
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
        "bo": {"alex": 99},
        "wei": {"li": 5},
        "li": {"wei": 11},
    }

    observed = _sorted_subblock_merge_candidates(output, maximum_size=5, first_k_letter_counts_sorted=counts)
    expected = _legacy_sorted_subblock_merge_candidates(output, maximum_size=5, first_k_letter_counts_sorted=counts)

    assert observed == expected
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


def test_sorted_subblock_merge_candidates_keeps_exact_maximum_size_pair() -> None:
    output = {
        "alex|middle=a": ["s1", "s2"],
        "alex|middle=b": ["s3", "s4", "s5"],
    }

    assert _sorted_subblock_merge_candidates(output, maximum_size=5, first_k_letter_counts_sorted={}) == [
        (("alex|middle=a", "alex|middle=b"), 1e10)
    ]


def test_projection_neighbor_edge_scores_match_slow_reference() -> None:
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
    matrix /= np.linalg.norm(matrix, axis=1)[:, None]
    evidences = [
        SimpleNamespace(coauthor_blocks=frozenset({"a"}), affiliation_keys=frozenset({"lab1"})),
        SimpleNamespace(coauthor_blocks=frozenset({"a"}), affiliation_keys=frozenset({"lab2"})),
        SimpleNamespace(coauthor_blocks=frozenset({"b"}), affiliation_keys=frozenset({"lab2"})),
        SimpleNamespace(coauthor_blocks=frozenset({"b"}), affiliation_keys=frozenset({"lab3"})),
        SimpleNamespace(coauthor_blocks=frozenset({"c"}), affiliation_keys=frozenset({"lab3"})),
        SimpleNamespace(coauthor_blocks=frozenset({"c"}), affiliation_keys=frozenset({"lab1"})),
    ]
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


def test_sorted_subblock_merge_candidates_matches_legacy_many_keys() -> None:
    output = {}
    counts = {}
    for index in range(80):
        if index % 3 == 0:
            key = f"al{index}"
        elif index % 3 == 1:
            key = f"a|middle=mi{index}"
        else:
            key = f"bo {index}"
        output[key] = [f"s{index}_{j}" for j in range(index % 4 + 1)]
    for left in range(0, 80, 5):
        # Keyed by the FULL canonical name strings of the "bo {index}" keys so
        # the prefix-count fallback branch is exercised under canonical lookups.
        counts.setdefault(f"al{left}", {})[f"bo {left + 2}"] = left + 1

    assert _sorted_subblock_merge_candidates(output, 7, counts) == _legacy_sorted_subblock_merge_candidates(
        output,
        7,
        counts,
    )


def test_arrow_graph_subblocking_fallback_loads_arrow_evidence_and_packs_components(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"

    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1", "s2", "s3", "s4"], type=pa.string()),
                "paper_id": pa.array(["p1", "p2", "p3", "p4"], type=pa.string()),
                "author_first": pa.array(["hui", "hui", "hui", "hui"], type=pa.string()),
                "author_middle": pa.array(["", "", "", ""], type=pa.string()),
                "author_affiliations": pa.array(
                    [
                        ["Department of Artificial Intelligence, Example University"],
                        ["Department of Artificial Intelligence, Example University"],
                        ["Department of Robotics, Other University"],
                        ["Department of Robotics, Other University"],
                    ]
                ),
                "author_orcid": pa.array([None, None, None, None], type=pa.string()),
                "author_position": pa.array([0, 0, 0, 0], type=pa.int64()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"], type=pa.string()),
                "position": pa.array([0, 1, 0, 1, 0, 1, 0, 1], type=pa.int64()),
                "author_name": pa.array(
                    [
                        "Hui Wang",
                        "Ada Lovelace",
                        "Hui Wang",
                        "Ada Lovelace",
                        "Hui Wang",
                        "Grace Hopper",
                        "Hui Wang",
                        "Grace Hopper",
                    ],
                    type=pa.string(),
                ),
            }
        ),
        paper_authors_path,
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=np.float32,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2", "p3", "p4"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.ravel(embeddings), type=pa.float32()),
                    2,
                ),
            }
        ),
        specter_path,
    )

    paths = _add_graph_batch_indexes(
        {"signatures": signatures_path, "paper_authors": paper_authors_path, "specter": specter_path},
        tmp_path,
    )

    fallback = make_arrow_graph_subblocking_cluster_fn(
        paths,
        ["s1", "s2", "s3", "s4"],
        config=GraphSubblockingConfig(
            neighbor_mode="exact",
            neighbors=1,
            min_edge_score=0.8,
            pack_components=True,
            component_pack_strategy="edge-greedy",
        ),
        random_seed=7,
    )

    signature_ids = (signature_id for signature_id in ["s1", "s2", "s3", "s4"])
    subblocks = fallback(signature_ids, object(), target_subblock_size=2)

    assert {frozenset(values) for values in subblocks.values()} == {frozenset({"s1", "s2"}), frozenset({"s3", "s4"})}
    assert max(len(values) for values in subblocks.values()) <= 2
    assert fallback.load_seconds >= 0.0
    assert fallback.load_metrics["signatures_rows_loaded"] == 4
    assert fallback.load_metrics["paper_authors_rows_loaded"] == 8
    assert fallback.load_metrics["specter_rows_loaded"] == 4
    assert fallback.stats[0]["raw_component_count"] == 2
    assert fallback.stats[0]["packed_component_count"] == 2
    assert fallback._dataset is not None
    s1_affiliation_keys = fallback._dataset.signatures["s1"].author_info_affiliations_n_grams
    assert s1_affiliation_keys is not None
    assert "artificial intelligence" in s1_affiliation_keys


def test_arrow_graph_subblocking_rejects_null_author_position(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"

    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1"], type=pa.string()),
                "paper_id": pa.array(["p1"], type=pa.string()),
                "author_first": pa.array(["hui"], type=pa.string()),
                "author_middle": pa.array([""], type=pa.string()),
                "author_affiliations": pa.array([["lab"]], type=pa.list_(pa.string())),
                "author_orcid": pa.array([None], type=pa.string()),
                "author_position": pa.array([None], type=pa.int64()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
                "author_name": pa.array(["Hui Wang"], type=pa.string()),
            }
        ),
        paper_authors_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 0.0], type=pa.float32()), 2),
            }
        ),
        specter_path,
    )
    paths = _add_graph_batch_indexes(
        {"signatures": signatures_path, "paper_authors": paper_authors_path, "specter": specter_path},
        tmp_path,
    )
    fallback = make_arrow_graph_subblocking_cluster_fn(paths, ["s1"])

    with pytest.raises(ValueError, match="null author_position"):
        fallback(["s1"], object(), target_subblock_size=2)


def test_arrow_graph_subblocking_rejects_null_affiliation_items(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"

    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1"], type=pa.string()),
                "paper_id": pa.array(["p1"], type=pa.string()),
                "author_first": pa.array(["hui"], type=pa.string()),
                "author_middle": pa.array([""], type=pa.string()),
                "author_affiliations": pa.array([["lab", None]], type=pa.list_(pa.string())),
                "author_orcid": pa.array([None], type=pa.string()),
                "author_position": pa.array([0], type=pa.int64()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
                "author_name": pa.array(["Hui Wang"], type=pa.string()),
            }
        ),
        paper_authors_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 0.0], type=pa.float32()), 2),
            }
        ),
        specter_path,
    )
    paths = _add_graph_batch_indexes(
        {"signatures": signatures_path, "paper_authors": paper_authors_path, "specter": specter_path},
        tmp_path,
    )
    fallback = make_arrow_graph_subblocking_cluster_fn(paths, ["s1"])

    with pytest.raises(ValueError, match="cannot contain null values"):
        fallback(["s1"], object(), target_subblock_size=2)


@pytest.mark.parametrize("paper_id", [None, ""])
def test_arrow_graph_subblocking_rejects_null_or_empty_paper_id(tmp_path, paper_id) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"

    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1"], type=pa.string()),
                "paper_id": pa.array([paper_id], type=pa.string()),
                "author_first": pa.array(["hui"], type=pa.string()),
                "author_middle": pa.array([""], type=pa.string()),
                "author_affiliations": pa.array([["lab"]], type=pa.list_(pa.string())),
                "author_orcid": pa.array([None], type=pa.string()),
                "author_position": pa.array([0], type=pa.int64()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
                "author_name": pa.array(["Hui Wang"], type=pa.string()),
            }
        ),
        paper_authors_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 0.0], type=pa.float32()), 2),
            }
        ),
        specter_path,
    )
    paths = _add_graph_batch_indexes(
        {"signatures": signatures_path, "paper_authors": paper_authors_path, "specter": specter_path},
        tmp_path,
    )
    fallback = make_arrow_graph_subblocking_cluster_fn(paths, ["s1"])

    with pytest.raises(ValueError, match="null/empty paper_id"):
        fallback(["s1"], object(), target_subblock_size=2)


def test_arrow_graph_subblocking_prepare_limits_loaded_evidence_to_fallback_union(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"

    _write_arrow_ipc_batches(
        signatures_path,
        [
            pa.record_batch(
                {
                    "signature_id": pa.array(["s1", "s2"], type=pa.string()),
                    "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                    "author_first": pa.array(["hui", "hui"], type=pa.string()),
                    "author_middle": pa.array(["", ""], type=pa.string()),
                    "author_affiliations": pa.array([["lab a"], ["lab a"]]),
                    "author_orcid": pa.array([None, None], type=pa.string()),
                    "author_position": pa.array([0, 0], type=pa.int64()),
                }
            ),
            pa.record_batch(
                {
                    "signature_id": pa.array(["s3", "s4"], type=pa.string()),
                    "paper_id": pa.array(["p3", "p4"], type=pa.string()),
                    "author_first": pa.array(["hui", "hui"], type=pa.string()),
                    "author_middle": pa.array(["", ""], type=pa.string()),
                    "author_affiliations": pa.array([["lab b"], ["lab b"]]),
                    "author_orcid": pa.array([None, None], type=pa.string()),
                    "author_position": pa.array([0, 0], type=pa.int64()),
                }
            ),
        ],
    )
    _write_arrow_ipc_batches(
        paper_authors_path,
        [
            pa.record_batch(
                {
                    "paper_id": pa.array(["p1", "p1", "p2", "p2"], type=pa.string()),
                    "position": pa.array([0, 1, 0, 1], type=pa.int64()),
                    "author_name": pa.array(
                        ["Hui Wang", "Ada Lovelace", "Hui Wang", "Ada Lovelace"],
                        type=pa.string(),
                    ),
                }
            ),
            pa.record_batch(
                {
                    "paper_id": pa.array(["p3", "p3", "p4", "p4"], type=pa.string()),
                    "position": pa.array([0, 1, 0, 1], type=pa.int64()),
                    "author_name": pa.array(
                        ["Hui Wang", "Grace Hopper", "Hui Wang", "Grace Hopper"],
                        type=pa.string(),
                    ),
                }
            ),
        ],
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ],
        dtype=np.float32,
    )
    _write_arrow_ipc_batches(
        specter_path,
        [
            pa.record_batch(
                {
                    "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                    "embedding": pa.FixedSizeListArray.from_arrays(
                        pa.array(np.ravel(embeddings[:2]), type=pa.float32()),
                        2,
                    ),
                }
            ),
            pa.record_batch(
                {
                    "paper_id": pa.array(["p3", "p4"], type=pa.string()),
                    "embedding": pa.FixedSizeListArray.from_arrays(
                        pa.array(np.ravel(embeddings[2:]), type=pa.float32()),
                        2,
                    ),
                }
            ),
        ],
    )

    paths = _add_graph_batch_indexes(
        {"signatures": signatures_path, "paper_authors": paper_authors_path, "specter": specter_path},
        tmp_path,
    )
    fallback = make_arrow_graph_subblocking_cluster_fn(
        paths,
        ["s1", "s2", "s3", "s4"],
        config=GraphSubblockingConfig(neighbor_mode="exact", neighbors=1, min_edge_score=0.8),
        random_seed=7,
    )

    fallback.prepare([["s1", "s2"]])
    subblocks = fallback(["s1", "s2"], object(), target_subblock_size=2)

    assert {frozenset(values) for values in subblocks.values()} == {frozenset({"s1", "s2"})}
    assert fallback.load_metrics["prepared_signature_count"] == 2
    assert fallback.load_metrics["signatures_record_batches_scanned"] == 1
    assert fallback.load_metrics["signatures_rows_scanned"] == 2
    assert fallback.load_metrics["signatures_rows_loaded"] == 2
    assert fallback.load_metrics["paper_authors_record_batches_scanned"] == 1
    assert fallback.load_metrics["paper_authors_rows_scanned"] == 4
    assert fallback.load_metrics["paper_authors_rows_loaded"] == 4
    assert fallback.load_metrics["specter_record_batches_scanned"] == 1
    assert fallback.load_metrics["specter_rows_scanned"] == 2
    assert fallback.load_metrics["specter_rows_loaded"] == 2


def test_arrow_graph_subblocking_requires_batch_indexes(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1"], type=pa.string()),
                "paper_id": pa.array(["p1"], type=pa.string()),
                "author_first": pa.array(["hui"], type=pa.string()),
                "author_middle": pa.array([""], type=pa.string()),
                "author_affiliations": pa.array([["lab"]]),
                "author_orcid": pa.array([None], type=pa.string()),
                "author_position": pa.array([0], type=pa.int64()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "position": pa.array([0], type=pa.int64()),
                "author_name": pa.array(["Hui Wang"], type=pa.string()),
            }
        ),
        paper_authors_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 0.0], type=pa.float32()), 2),
            }
        ),
        specter_path,
    )
    fallback = make_arrow_graph_subblocking_cluster_fn(
        {"signatures": signatures_path, "paper_authors": paper_authors_path, "specter": specter_path},
        ["s1"],
    )

    with pytest.raises(MissingArrowArtifactError, match="signatures_batch_index"):
        fallback.prepare([["s1"]])


def test_arrow_graph_subblocking_accepts_specter2_with_matching_batch_index(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter2_path = tmp_path / "specter2.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1", "s2"], type=pa.string()),
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "author_first": pa.array(["hui", "hui"], type=pa.string()),
                "author_middle": pa.array(["", ""], type=pa.string()),
                "author_affiliations": pa.array([["lab"], ["lab"]], type=pa.list_(pa.string())),
                "author_orcid": pa.array([None, None], type=pa.string()),
                "author_position": pa.array([0, 0], type=pa.int64()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "position": pa.array([0, 0], type=pa.int64()),
                "author_name": pa.array(["Hui Wang", "Hui Wang"], type=pa.string()),
            }
        ),
        paper_authors_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(pa.array([1.0, 0.0, 0.99, 0.01], type=pa.float32()), 2),
            }
        ),
        specter2_path,
    )
    paths = _add_graph_batch_indexes(
        {"signatures": signatures_path, "paper_authors": paper_authors_path, "specter2": specter2_path},
        tmp_path,
    )
    assert "specter2_batch_index" in paths
    assert "specter_batch_index" not in paths

    fallback = make_arrow_graph_subblocking_cluster_fn(
        paths,
        ["s1", "s2"],
        config=GraphSubblockingConfig(neighbor_mode="exact", neighbors=1, min_edge_score=0.8),
        random_seed=7,
    )

    subblocks = fallback(["s1", "s2"], object(), target_subblock_size=2)

    assert {frozenset(values) for values in subblocks.values()} == {frozenset({"s1", "s2"})}
    assert fallback.load_metrics["specter_rows_loaded"] == 2


def test_arrow_graph_subblocking_tolerates_sparse_evidence_and_reports_load_metrics(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")

    signatures_path = tmp_path / "signatures.arrow"
    paper_authors_path = tmp_path / "paper_authors.arrow"
    specter_path = tmp_path / "specter.arrow"

    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["s1", "s2", "s3", "s4"], type=pa.string()),
                "paper_id": pa.array(["p1", "p2", "p3", "p4"], type=pa.string()),
                "author_first": pa.array(["hui", "hui", "hui", "hui"], type=pa.string()),
                "author_middle": pa.array(["", "", "", ""], type=pa.string()),
                "author_affiliations": pa.array(
                    [None, [], ["lab b"], None],
                    type=pa.list_(pa.string()),
                ),
                "author_orcid": pa.array([None, None, None, None], type=pa.string()),
                "author_position": pa.array([0, 0, 0, 0], type=pa.int64()),
            }
        ),
        signatures_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p1", "p2", "p2", "p3", "p4"], type=pa.string()),
                "position": pa.array([0, 1, 0, 1, 0, 0], type=pa.int64()),
                "author_name": pa.array(
                    ["Hui Wang", "Ada Lovelace", "Hui Wang", "Grace Hopper", "Hui Wang", "Grace Hopper"],
                    type=pa.string(),
                ),
            }
        ),
        paper_authors_path,
    )
    write_arrow_ipc_table(
        pa.table(
            {
                "paper_id": pa.array(["p1", "p2"], type=pa.string()),
                "embedding": pa.FixedSizeListArray.from_arrays(
                    pa.array(np.zeros(4, dtype=np.float32), type=pa.float32()),
                    2,
                ),
            }
        ),
        specter_path,
    )

    paths = _add_graph_batch_indexes(
        {"signatures": signatures_path, "paper_authors": paper_authors_path, "specter": specter_path},
        tmp_path,
    )

    fallback = make_arrow_graph_subblocking_cluster_fn(
        paths,
        ["s1", "s2", "s3", "s4"],
        config=GraphSubblockingConfig(
            neighbor_mode="projection",
            projection_count=2,
            projection_window=2,
            min_edge_score=999.0,
            pack_components=True,
        ),
        random_seed=7,
    )

    subblocks = fallback(["s1", "s2", "s3", "s4"], object(), target_subblock_size=2)

    assert sorted(len(values) for values in subblocks.values()) == [2, 2]
    assert fallback.load_metrics == {
        "signatures_record_batches_scanned": 1,
        "signatures_rows_scanned": 4,
        "signatures_rows_loaded": 4,
        "paper_authors_record_batches_scanned": 1,
        "paper_authors_rows_scanned": 6,
        "paper_authors_rows_loaded": 6,
        "specter_record_batches_scanned": 1,
        "specter_rows_scanned": 2,
        "specter_rows_loaded": 2,
    }
    assert fallback.stats[0]["candidate_edge_count"] == 0
    assert fallback.stats[0]["raw_component_count"] == 4
    assert fallback.stats[0]["packed_component_count"] == 2


def test_graph_subblocking_packs_micro_components_before_legacy_merge() -> None:
    signature_ids = [f"s{index}" for index in range(6)]
    signatures = {
        signature_id: SimpleNamespace(
            signature_id=signature_id,
            paper_id=f"p{index}",
            author_info_first="hui",
            author_info_middle="",
            author_info_first_normalized_without_apostrophe="hui",
            author_info_middle_normalized_without_apostrophe="",
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
        specter_embeddings={f"p{index}": np.zeros(2, dtype=np.float32) for index in range(4)},
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

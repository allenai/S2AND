from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.incremental_linking.feature_block_arrow as feature_block_arrow_module
import s2and.runtime as runtime_module
from s2and.arrow_inputs import ArrowDataset
from s2and.data import ANDData, Author, NameCounts
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import (
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    IncrementalQuerySignatureRequest,
    arrow_ipc_physical_layout,
    normalize_cluster_seed_disallow_pairs,
    raw_planner_arrow_physical_layout,
    read_altered_cluster_signatures_arrow,
    read_cluster_seed_disallows_arrow,
    read_cluster_seeds_arrow,
    read_incremental_query_signatures_arrow,
    temporary_cluster_seed_sidecars,
    write_altered_cluster_signatures_arrow,
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_incremental_query_signatures_arrow,
    write_name_counts_index,
    write_raw_arrow_batch_lookup_indexes,
    write_raw_planner_arrow_tables,
)
from s2and.incremental_linking.features import LinkerFeatureMatrix
from s2and.incremental_linking.retrieval import (
    RawArrowPlanBundle,
    build_linker_retrieval_batch_from_raw_plan_bundle,
)
from s2and.incremental_linking.runtime import (
    CandidateBatchPairwiseModelResult,
    LinkOrAbstainCompactResult,
    LinkOrAbstainProductionResult,
    LinkOrAbstainRetrievedCandidatesResult,
    _predict_incremental_link_or_abstain_from_preplanned_raw_arrow,
)
from s2and.model import Clusterer
from scripts.arrow_conversion_helpers import (
    raw_planner_arrow_tables_from_anddata,
    write_raw_planner_arrow_from_anddata,
)
from tests.helpers import (
    tiny_name_counts_tuple,
    write_minimal_arrow_prediction_bundle,
    write_test_arrow_artifact_manifest,
)


def _raw_test_clusterer(
    *,
    n_jobs: int = 1,
    suppress_orcid: bool = False,
    features_to_use: list[str] | None = None,
) -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=features_to_use or []),
        classifier=None,
        n_jobs=n_jobs,
        suppress_orcid=suppress_orcid,
    )


def _raw_test_artifact(*, retrieval_top_k: int = 25) -> Any:
    return SimpleNamespace(retrieval_top_k=retrieval_top_k)


def _signature_payload(
    signature_id: str,
    paper_id: str,
    *,
    first: str,
    last: str,
    position: int,
    orcid: str | None = None,
) -> dict[str, Any]:
    author_info: dict[str, Any] = {
        "first": first,
        "middle": "",
        "last": last,
        "suffix": "",
        "affiliations": ["Analytical Engine Lab"],
        "email": "",
        "position": position,
        "block": f"{first[:1].lower()} {last.lower()}",
        "source_ids": [orcid] if orcid else [],
    }
    if orcid is not None:
        author_info["source_id_source"] = "ORCID"
    return {
        "signature_id": signature_id,
        "paper_id": paper_id,
        "author_info": author_info,
        "sourced_author_ids": [f"source-{signature_id}"],
    }


def _paper_payload(paper_id: str, *, title: str, year: int, authors: list[str]) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": "",
        "venue": "Royal Society",
        "journal_name": "",
        "year": year,
        "references": [],
        "authors": [{"position": index, "author_name": name} for index, name in enumerate(authors)],
    }


def _tiny_anddata() -> ANDData:
    dataset = ANDData(
        signatures={
            "q": _signature_payload(
                "q",
                "p_q",
                first="Ada",
                last="Lovelace",
                position=0,
                orcid="0000-0000-0000-0001",
            ),
            "s1": _signature_payload("s1", "p1", first="Ada", last="Lovelace", position=0),
            "s2": _signature_payload("s2", "p2", first="Grace", last="Hopper", position=0),
        },
        papers={
            "p_q": _paper_payload("p_q", title="Notes", year=1843, authors=["Ada Lovelace", "Charles Babbage"]),
            "p1": _paper_payload("p1", title="Notes", year=1843, authors=["Ada Lovelace", "Charles Babbage"]),
            "p2": _paper_payload("p2", title="Compiler", year=1952, authors=["Grace Hopper"]),
        },
        name="tiny_feature_block",
        mode="inference",
        name_counts_index=None,
        preprocess=False,
        name_tuples=set(),
    )
    dataset.cluster_seeds_require = {"s1": "c_ada", "s2": "c_grace"}
    dataset.cluster_seeds_disallow = {("q", "s2")}
    dataset.signatures["q"] = dataset.signatures["q"]._replace(
        author_info_name_counts=NameCounts(first=10.0, last=20.0, first_last=5.0, last_first_initial=8.0)
    )
    dataset.specter_embeddings = {
        "p_q": np.asarray([1.0, 0.0], dtype=np.float32),
        "p1": np.asarray([1.0, 0.1], dtype=np.float32),
    }
    return dataset


def _raw_plan() -> dict[str, Any]:
    row_count = 2
    pair_count = 3
    plan: dict[str, Any] = {
        "row_count": row_count,
        "pair_count": pair_count,
        "query_signature_ids": ["q"],
        "query_views": ["full"],
        "row_query_signature_indices": np.asarray([0, 0], dtype=np.uint32),
        "left_signature_ids": ["q", "q", "q"],
        "right_signature_ids": ["s1", "s2", "s3"],
        "pair_row_indices": np.asarray([0, 1, 1], dtype=np.uint32),
        "row_component_keys": ["c_ada", "c_other"],
        "retrieval_scores": np.asarray([0.9, 0.2], dtype=np.float32),
        "retrieval_ranks": np.asarray([1, 2], dtype=np.uint16),
        "row_component_sizes": np.asarray([1, 2], dtype=np.float32),
        "row_named_signature_counts": np.asarray([1, 2], dtype=np.float32),
        "row_dominant_first_names": np.asarray(["ada", "grace"], dtype=object),
        "row_candidate_year_min": np.asarray([1843, 1952], dtype=np.int32),
        "row_candidate_year_max": np.asarray([1843, 1952], dtype=np.int32),
        "row_candidate_year_range_missing": np.asarray([0, 0], dtype=np.uint8),
        "row_query_first_tokens": np.asarray(["ada", "ada"], dtype=object),
        "row_query_years": np.asarray([1843, 1843], dtype=np.int32),
        "row_query_year_missing": np.asarray([0, 0], dtype=np.uint8),
        "row_query_has_affiliations": np.asarray([1, 1], dtype=np.uint8),
        "row_query_has_coauthors": np.asarray([1, 1], dtype=np.uint8),
        "row_orcid_match": np.asarray([0, 0], dtype=np.uint8),
        "middle_initial_compatibility": np.asarray([1, 0], dtype=np.float32),
        "affiliation_overlap": np.asarray([1, 0], dtype=np.float32),
        "coauthor_overlap": np.asarray([1, 0], dtype=np.float32),
        "venue_overlap": np.asarray([1, 0], dtype=np.float32),
        "year_compatibility": np.asarray([1, 0], dtype=np.float32),
        "title_overlap": np.asarray([1, 0], dtype=np.float32),
        "specter_centroid_similarity": np.asarray([1, 0], dtype=np.float32),
        "specter_exemplar_similarity": np.asarray([1, 0], dtype=np.float32),
        "row_last_name_count_min_rarity": np.asarray([0.1, 0.2], dtype=np.float32),
        "row_candidate_last_name_count_min_rarity": np.asarray([0.1, 0.2], dtype=np.float32),
        "row_candidate_last_first_name_count_min_rarity": np.asarray([0.1, 0.2], dtype=np.float32),
        "row_last_first_name_count_min_rarity": np.asarray([0.1, 0.2], dtype=np.float32),
        "row_first_prefix_x_last_first_name_count_min_rarity": np.asarray([0.1, 0.0], dtype=np.float32),
        "row_candidate_cluster_max_paper_author_count": np.asarray([2, 2], dtype=np.float32),
        "row_paper_author_list_max_jaccard": np.asarray([1, 0.2], dtype=np.float32),
        "row_paper_author_list_max_containment": np.asarray([1, 0.5], dtype=np.float32),
        "row_paper_author_list_max_overlap_count": np.asarray([2, 1], dtype=np.float32),
        "row_local_author_window10_jaccard_max": np.asarray([1, 0], dtype=np.float32),
        "row_local_author_window10_overlap_count_max": np.asarray([1, 0], dtype=np.float32),
        "row_best_author_count_log_absdiff": np.asarray([0, 0], dtype=np.float32),
        "query_authors": ["Ada Lovelace"],
        "component_members": {"c_ada": ["s1"], "c_other": ["s2", "s3"]},
        "telemetry": {"timings": {}},
    }
    return plan


def _open_test_arrow_dataset(tmp_path: Path) -> ArrowDataset:
    write_minimal_arrow_prediction_bundle(tmp_path)
    return ArrowDataset.open(tmp_path)


def _write_raw_planner_arrow_paths(tmp_path: Path) -> dict[str, str]:
    dataset = _tiny_anddata()
    dataset.signatures["s3"] = dataset.signatures["s2"]._replace(
        paper_id="p3",
        author_info_position=1,
        sourced_author_ids=("source-s3",),
    )
    dataset.papers["p2"] = dataset.papers["p2"]._replace(
        authors=[
            Author(author_name="Grace Hopper", position=0),
            Author(author_name="Jane Doe", position=1),
        ]
    )
    dataset.papers["p3"] = dataset.papers["p2"]._replace(
        paper_id="p3",
        authors=[
            Author(author_name="Jane Doe", position=0),
            Author(author_name="Grace Hopper", position=1),
        ],
    )
    dataset.cluster_seeds_require = {"s1": "c_ada", "s2": "c_other", "s3": "c_other"}
    tables = raw_planner_arrow_tables_from_anddata(
        dataset,
        include_specter=False,
    )
    return write_raw_planner_arrow_tables(tables, tmp_path, include_empty_cluster_seeds=True)


def _with_fake_batch_indexes(arrow_paths: dict[str, str], tmp_path: Path) -> dict[str, str]:
    indexed, _metrics = write_raw_arrow_batch_lookup_indexes(arrow_paths, tmp_path)
    write_test_arrow_artifact_manifest(tmp_path, indexed)
    return indexed


def _with_query_signatures(
    arrow_paths: dict[str, str],
    tmp_path: Path,
    *,
    query_signature_ids: tuple[str, ...] = ("q",),
    query_view: str = "full",
    query_author: str = "Ada Lovelace",
) -> dict[str, str]:
    request_paths = dict(arrow_paths)
    query_signatures_path = tmp_path / "incremental_query_signatures.arrow"
    write_incremental_query_signatures_arrow(
        query_signatures_path,
        query_signature_ids,
        query_views=[query_view] * len(query_signature_ids),
        query_authors=[query_author] * len(query_signature_ids),
    )
    request_paths["query_signatures"] = str(query_signatures_path)
    return request_paths


def test_raw_planner_arrow_tables_from_anddata_builds_requested_mini_contract() -> None:
    dataset = _tiny_anddata()
    dataset.cluster_seeds_disallow = set()
    tables = raw_planner_arrow_tables_from_anddata(
        dataset,
        signature_ids=["q", "s1"],
    )

    assert tables["signatures"]["signature_id"].to_pylist() == ["q", "s1"]
    assert tables["cluster_seeds"].to_pydict() == {
        "signature_id": ["s1"],
        "cluster_id": ["c_ada"],
    }
    assert tables["cluster_seed_disallows"].num_rows == 0
    assert tables["papers"]["paper_id"].to_pylist() == ["p_q", "p1"]
    assert tables["paper_authors"].to_pylist() == [
        {"paper_id": "p_q", "position": 0, "author_name": "ada lovelace"},
        {"paper_id": "p_q", "position": 1, "author_name": "charles babbage"},
        {"paper_id": "p1", "position": 0, "author_name": "ada lovelace"},
        {"paper_id": "p1", "position": 1, "author_name": "charles babbage"},
    ]
    assert tables["signatures"]["author_orcid"].to_pylist()[0] == "0000-0000-0000-0001"
    assert tables["specter"]["paper_id"].to_pylist() == ["p_q", "p1"]
    np.testing.assert_allclose(tables["specter"]["embedding"].to_pylist(), [[1.0, 0.0], [1.0, 0.1]])


def test_raw_planner_arrow_tables_from_anddata_preserves_blank_paper_author_rows() -> None:
    dataset = _tiny_anddata()
    dataset.papers["p_q"] = dataset.papers["p_q"]._replace(
        authors=[Author(author_name="", position=0), Author(author_name="   ", position=1)]
    )

    tables = raw_planner_arrow_tables_from_anddata(dataset, signature_ids=["q"])

    assert tables["paper_authors"].to_pylist() == [
        {"paper_id": "p_q", "position": 0, "author_name": ""},
        {"paper_id": "p_q", "position": 1, "author_name": "   "},
    ]


def test_raw_planner_arrow_tables_parse_false_language_reliability() -> None:
    dataset = _tiny_anddata()
    dataset.papers["p_q"] = dataset.papers["p_q"]._replace(
        predicted_language="en",
        is_reliable=cast(Any, "false"),
        language_reliability=0.0,
    )

    tables = raw_planner_arrow_tables_from_anddata(
        dataset,
        signature_ids=["q"],
    )

    assert tables["papers"]["is_reliable"].to_pylist() == [False]


@pytest.mark.parametrize(
    ("predicted_language", "is_reliable", "language_reliability", "message"),
    (
        ("en", True, np.nan, "must be finite"),
        ("en", True, -0.01, "must be in [0.0, 1.0]"),
        ("en", False, 0.25, "must be 0.0 when"),
        ("en", None, 0.75, "predicted_language requires papers.is_reliable"),
        ("en", True, None, "predicted_language requires papers.language_reliability"),
        (None, True, 0.75, "require papers.predicted_language"),
        (" \t", True, 0.75, "predicted_language must be non-empty"),
    ),
)
def test_raw_planner_arrow_tables_validate_language_metadata(
    predicted_language: str | None,
    is_reliable: bool | None,
    language_reliability: float | None,
    message: str,
) -> None:
    dataset = _tiny_anddata()
    dataset.papers["p_q"] = dataset.papers["p_q"]._replace(
        predicted_language=predicted_language,
        is_reliable=is_reliable,
        language_reliability=language_reliability,
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        raw_planner_arrow_tables_from_anddata(
            dataset,
            signature_ids=["q"],
        )


def test_raw_planner_arrow_tables_from_anddata_rejects_signature_missing_paper() -> None:
    dataset = _tiny_anddata()
    del dataset.papers["p1"]

    with pytest.raises(ValueError, match="missing signature paper_id"):
        raw_planner_arrow_tables_from_anddata(
            dataset,
            signature_ids=["q", "s1"],
        )


def test_raw_planner_arrow_tables_keep_all_null_optional_columns_typed() -> None:
    import pyarrow as pa

    tables = raw_planner_arrow_tables_from_anddata(
        _tiny_anddata(),
        signature_ids=["q"],
        include_specter=False,
    )

    assert tables["signatures"].schema.field("author_suffix").type == pa.string()
    assert tables["signatures"].schema.field("author_email").type == pa.string()
    assert tables["papers"].schema.field("predicted_language").type == pa.string()
    assert tables["papers"].schema.field("is_reliable").type == pa.bool_()
    assert tables["papers"].schema.field("language_reliability").type == pa.float64()
    assert tables["cluster_seeds"].schema.field("signature_id").type == pa.string()
    assert tables["cluster_seed_disallows"].schema.field("signature_id_1").type == pa.string()


def test_write_raw_planner_arrow_from_anddata_skips_empty_seed_table(tmp_path: Path) -> None:
    import pyarrow as pa

    dataset = _tiny_anddata()
    dataset.cluster_seeds_disallow = set()
    paths = write_raw_planner_arrow_from_anddata(
        dataset,
        tmp_path,
        signature_ids=["q"],
        include_specter=False,
    )

    assert set(paths) == {"signatures", "papers", "paper_authors"}
    with pa.memory_map(paths["signatures"], "r") as source:
        signatures = pa.ipc.open_file(source).read_all()
    assert "name_count_first" not in signatures.column_names
    assert signatures.to_pydict()["signature_id"] == ["q"]


def test_incremental_query_signatures_arrow_round_trips_typed_rows(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "incremental_query_signatures.arrow"

    write_incremental_query_signatures_arrow(
        path,
        ["q1", "q2"],
        query_views=["full", "initial_only"],
        query_authors=["Ada Lovelace", "Grace Hopper"],
    )

    assert read_incremental_query_signatures_arrow(path) == (
        IncrementalQuerySignatureRequest(
            signature_id="q1",
            query_view="full",
            query_author="Ada Lovelace",
        ),
        IncrementalQuerySignatureRequest(
            signature_id="q2",
            query_view="initial_only",
            query_author="Grace Hopper",
        ),
    )
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    assert table.schema.field("signature_id").type == pa.string()
    assert table.schema.field("query_view").type == pa.string()
    assert table.schema.field("query_author").type == pa.string()
    assert table.to_pydict() == {
        "signature_id": ["q1", "q2"],
        "query_view": ["full", "initial_only"],
        "query_author": ["Ada Lovelace", "Grace Hopper"],
    }


def test_incremental_query_signatures_arrow_keeps_empty_table_typed(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "empty_incremental_query_signatures.arrow"

    write_incremental_query_signatures_arrow(path, [])

    assert read_incremental_query_signatures_arrow(path) == ()
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    assert table.num_rows == 0
    assert table.schema.field("signature_id").type == pa.string()
    assert table.schema.field("query_view").type == pa.string()
    assert table.schema.field("query_author").type == pa.string()


def test_incremental_query_signatures_arrow_rejects_duplicate_rows(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "duplicate_incremental_query_signatures.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["q1", "q1"], type=pa.string()),
                "query_view": pa.array(["full", "initial_only"], type=pa.string()),
                "query_author": pa.array(["Ada Lovelace", "Ada Lovelace"], type=pa.string()),
            }
        ),
        path,
    )

    with pytest.raises(ValueError, match="duplicate signature_id"):
        read_incremental_query_signatures_arrow(path)


def test_incremental_query_signatures_arrow_rejects_unknown_query_view(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "unknown_query_view_incremental_query_signatures.arrow"

    with pytest.raises(ValueError, match="unknown query_view"):
        write_incremental_query_signatures_arrow(path, ["q1"], query_views=["profile"])

    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["q1"], type=pa.string()),
                "query_view": pa.array(["profile"], type=pa.string()),
                "query_author": pa.array(["Ada Lovelace"], type=pa.string()),
            }
        ),
        path,
    )

    with pytest.raises(ValueError, match="unknown query_view"):
        read_incremental_query_signatures_arrow(path)


def test_incremental_query_signatures_arrow_rejects_null_request_values(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "null_incremental_query_signatures.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(["q1"], type=pa.string()),
                "query_view": pa.array(["full"], type=pa.string()),
                "query_author": pa.array([None], type=pa.string()),
            }
        ),
        path,
    )

    with pytest.raises(ValueError, match="null query_author"):
        read_incremental_query_signatures_arrow(path)


def test_arrow_readers_reject_integer_id_columns(tmp_path: Path) -> None:
    import pyarrow as pa

    cases = (
        (
            "query-signature-id",
            "incremental_query_signatures.arrow",
            {"signature_id": [1], "query_view": ["full"], "query_author": ["Ada Lovelace"]},
            read_incremental_query_signatures_arrow,
            "signature_id",
        ),
        (
            "disallow-signature-id",
            "cluster_seed_disallows.arrow",
            {"signature_id_1": [1], "signature_id_2": ["s2"]},
            read_cluster_seed_disallows_arrow,
            "signature_id_1",
        ),
        (
            "seed-signature-id",
            "cluster_seeds.arrow",
            {"signature_id": [1], "cluster_id": ["c1"]},
            read_cluster_seeds_arrow,
            "signature_id",
        ),
    )
    for case_id, filename, columns, reader, bad_column in cases:
        path = tmp_path / filename
        write_arrow_ipc_table(pa.table(columns), path)

        try:
            reader(path)
        except ValueError as error:
            assert f"column '{bad_column}' expected string" in str(error), f"{case_id}: {error}"
        else:
            raise AssertionError(f"{case_id}: integer ID column was accepted")


def test_altered_cluster_signatures_arrow_round_trips_and_rejects_duplicates(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "altered_cluster_signatures.arrow"

    write_altered_cluster_signatures_arrow(path, ["seed0", "seed2"])

    assert read_altered_cluster_signatures_arrow(path) == ("seed0", "seed2")

    duplicate_path = tmp_path / "duplicate_altered_cluster_signatures.arrow"
    write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["seed0", "seed0"], type=pa.string())}),
        duplicate_path,
    )

    with pytest.raises(ValueError, match="duplicate signature_id"):
        read_altered_cluster_signatures_arrow(duplicate_path)


def test_temporary_cluster_seed_sidecars_clean_up_tmpdir() -> None:

    with temporary_cluster_seed_sidecars(
        {"s1": "c1"},
        prefix="test-arrow-sidecars-",
        cluster_seeds_disallow=[("s1", "s2")],
    ) as sidecars:
        cluster_seed_path = Path(sidecars["cluster_seeds"])
        disallow_path = Path(sidecars["cluster_seed_disallows"])
        temp_dir = cluster_seed_path.parent

        assert cluster_seed_path.exists()
        assert disallow_path.exists()
        assert temp_dir.exists()

    assert not temp_dir.exists()
    assert not cluster_seed_path.exists()
    assert not disallow_path.exists()


def test_read_cluster_seeds_arrow_rejects_duplicate_signature_rows(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "cluster_seeds.arrow"
    table = pa.table(
        {
            "signature_id": pa.array(["s1", "s1"], type=pa.string()),
            "cluster_id": pa.array(["c1", "c1"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(table, path)

    with pytest.raises(ValueError, match="duplicate signature_id"):
        read_cluster_seeds_arrow(path)


def test_write_arrow_ipc_table_writes_bounded_record_batches(tmp_path: Path) -> None:
    import pyarrow as pa

    table = pa.table({"signature_id": pa.array([str(index) for index in range(5)], type=pa.string())})
    path = write_arrow_ipc_table(table, tmp_path / "signatures.arrow", max_record_batch_rows=2)

    assert arrow_ipc_physical_layout(path) == {
        "row_count": 5,
        "record_batch_count": 3,
        "actual_max_batch_rows": 2,
    }


def test_raw_planner_index_rejects_unbounded_large_batch(tmp_path: Path) -> None:
    import pyarrow as pa

    table = pa.table({"signature_id": pa.array([str(index) for index in range(5)], type=pa.string())})
    path = write_arrow_ipc_table(table, tmp_path / "signatures.arrow")

    with pytest.raises(ValueError, match="exceeding the raw-planner limit of 2"):
        write_raw_arrow_batch_lookup_indexes(
            {"signatures": path},
            tmp_path,
            max_record_batch_rows={"signatures": 2},
        )


def test_raw_planner_index_metadata_uses_stem_qualified_sidecar(tmp_path: Path) -> None:
    import pyarrow as pa

    table = pa.table({"signature_id": pa.array([str(index) for index in range(5)], type=pa.string())})
    path = write_arrow_ipc_table(table, tmp_path / "signatures.arrow", max_record_batch_rows=2)
    indexed_paths, index_metrics = write_raw_arrow_batch_lookup_indexes(
        {"signatures": path},
        tmp_path,
        max_record_batch_rows={"signatures": 2},
    )
    layout = raw_planner_arrow_physical_layout(indexed_paths, max_record_batch_rows={"signatures": 2})

    assert Path(indexed_paths["signatures_batch_index"]).name == "signatures.signatures_batch_index.bin"
    assert index_metrics["signatures_batch_index"]["record_batch_count"] == 3
    assert index_metrics["signatures_batch_index"]["actual_max_batch_rows"] == 2
    assert layout["tables"]["signatures"]["batch_index_present"] is True
    assert layout["tables"]["signatures"]["max_record_batch_rows"] == 2
    assert RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS["signatures"] == 16_384


def test_raw_planner_index_omits_none_optional_paths(tmp_path: Path) -> None:
    import pyarrow as pa

    table = pa.table({"signature_id": pa.array(["s1"], type=pa.string())})
    path = write_arrow_ipc_table(table, tmp_path / "signatures.arrow")
    indexed_paths, index_metrics = write_raw_arrow_batch_lookup_indexes(
        {
            "signatures": path,
            "specter": None,
        },
        tmp_path,
    )

    assert indexed_paths["signatures"] == path
    assert "signatures_batch_index" in indexed_paths
    assert "specter" not in indexed_paths
    assert "specter_batch_index" not in index_metrics


def test_raw_planner_index_rejects_null_lookup_keys(tmp_path: Path) -> None:
    import pyarrow as pa

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", None], type=pa.string())}),
        tmp_path / "signatures.arrow",
    )

    with pytest.raises(ValueError, match="null values in key column"):
        write_raw_arrow_batch_lookup_indexes({"signatures": path}, tmp_path)


def test_raw_planner_index_rejects_stale_python_reuse(tmp_path: Path) -> None:
    import pyarrow as pa

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
    )
    write_raw_arrow_batch_lookup_indexes({"signatures": path}, tmp_path)
    write_arrow_ipc_table(
        pa.table({"signature_id": pa.array([f"s{index}" for index in range(200)], type=pa.string())}),
        path,
    )

    with pytest.raises(ValueError, match="stale"):
        write_raw_arrow_batch_lookup_indexes({"signatures": path}, tmp_path, overwrite=False)


def test_raw_planner_index_reuse_metrics_match_fresh_schema(tmp_path: Path) -> None:
    import pyarrow as pa

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"

    _, fresh_metrics = write_arrow_batch_lookup_index(
        path,
        index_path,
        key_column="signature_id",
        table_name="signatures",
        max_record_batch_rows=1,
        overwrite=True,
    )
    _, reused_metrics = write_arrow_batch_lookup_index(
        path,
        index_path,
        key_column="signature_id",
        table_name="signatures",
        max_record_batch_rows=1,
        overwrite=False,
    )

    assert set(reused_metrics) == set(fresh_metrics)
    assert reused_metrics == {**fresh_metrics, "reused": True}


def test_raw_planner_index_reuse_rejects_record_count_mismatch(tmp_path: Path) -> None:
    import pyarrow as pa

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", "s2", "s3"], type=pa.string())}),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=2,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(
        path,
        index_path,
        key_column="signature_id",
        table_name="signatures",
        max_record_batch_rows=2,
        overwrite=True,
    )

    header_struct = feature_block_arrow_module._ARROW_BATCH_LOOKUP_INDEX_HEADER_STRUCT  # noqa: SLF001
    with index_path.open("r+b") as index_file:
        values = list(header_struct.unpack(index_file.read(header_struct.size)))
        values[1] = 2
        index_file.seek(0)
        index_file.write(header_struct.pack(*values))

    with pytest.raises(ValueError, match="row count mismatch"):
        write_arrow_batch_lookup_index(
            path,
            index_path,
            key_column="signature_id",
            table_name="signatures",
            max_record_batch_rows=2,
            overwrite=False,
        )


def test_write_name_counts_index(tmp_path: Path) -> None:
    mappings = ({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7})

    index_path, index_metrics = write_name_counts_index(tmp_path, mappings)

    assert "reused" not in index_metrics
    assert index_metrics["row_count"] == 4
    assert index_metrics["first_count"] == 1
    manifest = json.loads((Path(index_path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "s2and_name_counts"
    assert manifest["format_version"] == 1
    assert set(manifest["files"]["first"]) == {"byte_count", "sha256"}
    with pytest.raises(FileExistsError, match="target already exists"):
        write_name_counts_index(tmp_path, mappings)


@pytest.mark.skipif(os.name != "nt", reason="Windows prevents unlinking open sort-run files")
def test_name_count_index_merge_failure_closes_sort_runs_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "first.bin"

    def fail_mid_merge(
        path: Path,
        records: Any,
        *,
        record_count: int,
    ) -> int:
        del path, record_count
        next(iter(records))
        raise RuntimeError("simulated merge failure")

    monkeypatch.setattr(feature_block_arrow_module, "_write_sorted_name_count_records", fail_mid_merge)

    with pytest.raises(RuntimeError, match="simulated merge failure"):
        feature_block_arrow_module._write_name_count_index_file(  # noqa: SLF001
            output_path,
            "first",
            {"ada": 1, "grace": 2},
            max_records_in_memory=1,
        )

    assert list(tmp_path.glob(f".{output_path.name}.run.*")) == []


def test_raw_planner_arrow_tables_from_anddata_filters_one_sided_disallow_pair() -> None:
    tables = raw_planner_arrow_tables_from_anddata(
        _tiny_anddata(),
        signature_ids=["q", "s1"],
    )

    assert tables["cluster_seed_disallows"].num_rows == 0


def test_disallow_contract_rejects_out_of_set_pair() -> None:
    with pytest.raises(ValueError, match="missing from signature set"):
        normalize_cluster_seed_disallow_pairs(
            [("q", "outside")],
            valid_signature_ids=["q"],
        )


def test_raw_planner_arrow_tables_reject_scalar_sequence_fields() -> None:
    dataset = _tiny_anddata()
    dataset.signatures["q"] = dataset.signatures["q"]._replace(
        author_info_affiliations=cast(Any, "Lab"),
    )

    with pytest.raises(ValueError, match="signatures.author_info_affiliations"):
        raw_planner_arrow_tables_from_anddata(
            dataset,
            signature_ids=["q"],
        )


def test_raw_candidate_plan_bridge_accepts_feature_block_signature_order() -> None:
    order = RawArrowPlanBundle.from_native_mapping(_raw_plan()).signature_order

    retrieval_batch = build_linker_retrieval_batch_from_raw_plan_bundle(
        RawArrowPlanBundle.from_native_mapping(_raw_plan()),
        feature_block_signature_order=order,
    )

    candidate_batch = retrieval_batch.candidate_batch
    assert cast(Any, candidate_batch.row_query_signature_indices).tolist() == [0, 0]
    assert candidate_batch.left_signature_indices.tolist() == [0, 0, 0]
    assert candidate_batch.right_signature_indices.tolist() == [1, 2, 3]
    assert candidate_batch.row_component_keys == ("c_ada", "c_other")
    np.testing.assert_allclose(retrieval_batch.row_signals["retrieval_score"], [0.9, 0.2])


def test_raw_candidate_plan_bridge_reports_missing_signature_id() -> None:
    with pytest.raises(KeyError, match="right_signature_ids contains signature_id not present"):
        build_linker_retrieval_batch_from_raw_plan_bundle(
            RawArrowPlanBundle.from_native_mapping(_raw_plan()),
            signature_id_to_index={"q": 0, "s1": 1, "s2": 2},
        )


def test_raw_planner_arrow_tables_reject_duplicate_paper_author_positions() -> None:
    dataset = _tiny_anddata()
    dataset.papers["p1"] = dataset.papers["p1"]._replace(
        authors=[
            Author(author_name="Alice Smith", position=0),
            Author(author_name="A. Smith", position=0),
        ]
    )

    with pytest.raises(ValueError, match=r"duplicate \(paper_id, position\)"):
        raw_planner_arrow_tables_from_anddata(
            dataset,
            signature_ids=["s1"],
        )


def test_arrow_validation_and_planner_share_one_native_name_count_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_arrow_paths = _write_raw_planner_arrow_paths(tmp_path)
    mappings = tiny_name_counts_tuple()
    name_counts_index_path, _metrics = write_name_counts_index(
        tmp_path,
        mappings,
    )
    base_arrow_paths["name_counts_index"] = name_counts_index_path
    arrow_paths = _with_query_signatures(
        _with_fake_batch_indexes(base_arrow_paths, tmp_path),
        tmp_path,
        query_author="ada lovelace",
    )
    real_extension = runtime_module.load_s2and_rust_extension()
    native_open_calls = 0

    class CountingNativeNameCountsIndex:
        @staticmethod
        def open(path_arg: str):
            nonlocal native_open_calls
            native_open_calls += 1
            return real_extension.NameCountsIndex.open(path_arg)

    monkeypatch.setattr(
        runtime_module,
        "load_s2and_rust_extension",
        lambda: SimpleNamespace(
            NameCountsIndex=CountingNativeNameCountsIndex,
            _ArrowDataset=real_extension._ArrowDataset,
        ),
    )

    with ArrowDataset.open(
        tmp_path,
        require_specter=False,
        require_name_counts_index=True,
    ) as arrow_dataset:
        retained_index = arrow_dataset.name_counts_index
        assert retained_index is not None
        retained_native = arrow_dataset.native_name_counts_index
        assert retained_native is not None

        labeled_plan = real_extension.raw_arrow_labeled_candidate_plan(
            arrow_dataset.native,
            ["q"],
            ["full"],
            ["group-q"],
            ["c_ada"],
            [1],
            {"c_ada": ["s1"]},
            orcid_enabled=False,
            num_threads=1,
            max_exemplars=4,
        )
        assert labeled_plan["telemetry"]["reused_name_counts_index"] is True
        assert native_open_calls == 1

        planner = real_extension.RawBlockQueryCandidatePlanner.from_query_signatures(
            arrow_dataset.native,
            arrow_paths["query_signatures"],
            arrow_paths["cluster_seeds"],
            2,
            orcid_enabled=False,
            num_threads=1,
            max_exemplars=4,
        )
        plan = planner.plan_query_signatures()

        assert int(plan["row_count"]) > 0
        assert native_open_calls == 1
        assert planner.build_telemetry()["reused_name_counts_index"] is True
        assert planner.name_counts_index().name_counts_manifest_sha256 == retained_index.manifest_sha256
        featurizer = build_rust_featurizer_from_arrow_dataset(
            arrow_dataset,
            signature_ids=("s1",),
            name_tuples=set(),
            preprocess=True,
            num_threads=1,
        )
        assert featurizer.name_counts_manifest_sha256 == retained_native.name_counts_manifest_sha256
        assert native_open_calls == 1


def test_raw_arrow_partial_supervision_require_unknown_seed_rejected(tmp_path: Path) -> None:
    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3"]

    raw_plan = _raw_plan()
    raw_plan["component_members"] = {}

    with _open_test_arrow_dataset(tmp_path) as arrow_dataset:
        with pytest.raises(ValueError, match="partial_supervision_require_unknown_seed_signature"):
            _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
                _raw_test_clusterer(),
                _raw_test_artifact(),
                arrow_dataset=arrow_dataset,
                query_signature_ids=["q"],
                raw_plan_bundle=RawArrowPlanBundle.from_native_mapping(raw_plan),
                rust_featurizer=FakeFeaturizer(),
                partial_supervision={("q", "s1"): 0},
            )


def test_raw_arrow_scoring_requires_featurizer_with_provided_raw_plan(tmp_path: Path) -> None:
    with _open_test_arrow_dataset(tmp_path) as arrow_dataset:
        with pytest.raises(ValueError, match="preplanned raw Arrow scoring requires rust_featurizer"):
            _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
                _raw_test_clusterer(),
                _raw_test_artifact(),
                arrow_dataset=arrow_dataset,
                query_signature_ids=["q"],
                raw_plan_bundle=RawArrowPlanBundle.from_native_mapping(_raw_plan()),
                rust_featurizer=None,
            )


@pytest.mark.parametrize(
    ("component_members", "expected_seed_count", "expected_component_count"),
    [
        ({"c_ada": ["s1"], "c_other": ["s2", "s3"]}, 3, 2),
        ({"c_ada": ["s1", "s1"], "c_other": ["s2", "s3"], "empty": []}, 3, 2),
        ({"empty": []}, 7, 0),
    ],
)
def test_preplanned_raw_arrow_scoring_uses_provided_plan_and_featurizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    component_members: dict[str, list[str]],
    expected_seed_count: int,
    expected_component_count: int,
) -> None:
    captured: dict[str, Any] = {}

    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3"]

    def fake_from_retrieval(**kwargs: Any) -> LinkOrAbstainProductionResult:
        retrieval_batch = kwargs["retrieval_batch"]
        captured["cluster_seeds_require"] = kwargs["cluster_seeds_require"]
        captured["featurizer"] = kwargs["featurizer"]
        captured["retrieval_left_indices"] = retrieval_batch.candidate_batch.left_signature_indices.tolist()
        captured["retrieval_right_indices"] = retrieval_batch.candidate_batch.right_signature_indices.tolist()
        captured["queries"] = kwargs["queries"]
        return LinkOrAbstainProductionResult(
            feature_matrix=LinkerFeatureMatrix(
                matrix=np.empty((2, 0), dtype=np.float32),
                feature_columns=(),
                candidate_batch=retrieval_batch.candidate_batch,
            ),
            compact_result=LinkOrAbstainCompactResult(
                probabilities=np.asarray([0.8, 0.2], dtype=np.float32),
                decisions=(),
            ),
            telemetry={"pairwise_feature_seconds": 0.5, "constraint_api_mode": "rust_index_arrays"},
            retrieval_batch=retrieval_batch,
            pairwise_model_result=CandidateBatchPairwiseModelResult(
                row_signals={},
                pairwise_stats=cast(Any, None),
                telemetry={},
            ),
            linked_signature_clusters={"q": "c_ada"},
        )

    monkeypatch.setattr(
        "s2and.incremental_linking.runtime._predict_incremental_link_or_abstain_production_from_retrieval_private",
        lambda *args, **kwargs: fake_from_retrieval(**kwargs),
    )

    fake_featurizer = FakeFeaturizer()
    raw_plan = _raw_plan()
    raw_plan["component_members"] = component_members
    raw_plan["telemetry"]["seed_signature_count"] = 7
    with _open_test_arrow_dataset(tmp_path) as arrow_dataset:
        result = _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
            _raw_test_clusterer(),
            _raw_test_artifact(),
            arrow_dataset=arrow_dataset,
            query_signature_ids=["q"],
            raw_plan_bundle=RawArrowPlanBundle.from_native_mapping(raw_plan),
            rust_featurizer=fake_featurizer,
            top_k=2,
            n_jobs=1,
        )

    assert captured["cluster_seeds_require"] == {
        signature_id: component for component, members in component_members.items() for signature_id in members
    }
    assert captured["featurizer"] is fake_featurizer
    assert captured["retrieval_left_indices"] == [0, 0, 0]
    assert captured["retrieval_right_indices"] == [1, 2, 3]
    assert captured["queries"][0].query_author == "Ada Lovelace"
    assert result.linked_signature_clusters == {"q": "c_ada"}
    assert result.telemetry["raw_arrow_retrieval_seconds"] == 0.0
    assert result.telemetry["raw_arrow_signature_count"] == 4
    assert result.telemetry["raw_arrow_plan_signature_count"] == 4
    assert result.telemetry["seed_signature_count"] == expected_seed_count
    assert result.telemetry["raw_arrow_seed_signature_count"] == expected_seed_count
    assert result.telemetry["seed_component_count"] == expected_component_count
    assert result.telemetry["raw_arrow_seed_component_count"] == expected_component_count
    assert isinstance(result.telemetry["raw_arrow_featurizer_seconds"], float)


def test_preplanned_raw_arrow_scoring_rejects_mismatched_raw_plan_query_ids(tmp_path: Path) -> None:
    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3"]

    with _open_test_arrow_dataset(tmp_path) as arrow_dataset:
        with pytest.raises(ValueError, match="must exactly match requested query_signature_ids"):
            _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
                _raw_test_clusterer(),
                _raw_test_artifact(),
                arrow_dataset=arrow_dataset,
                query_signature_ids=["s1"],
                raw_plan_bundle=RawArrowPlanBundle.from_native_mapping(_raw_plan()),
                rust_featurizer=FakeFeaturizer(),
                top_k=2,
                n_jobs=1,
            )


def test_from_retrieval_skips_pair_id_build_when_partial_supervision_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import s2and.incremental_linking.runtime as runtime_module

    order = RawArrowPlanBundle.from_native_mapping(_raw_plan()).signature_order
    retrieval_batch = build_linker_retrieval_batch_from_raw_plan_bundle(
        RawArrowPlanBundle.from_native_mapping(_raw_plan()),
        feature_block_signature_order=order,
    )

    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return list(order.signature_ids)

    def fail_candidate_pair_ids(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("pair ids should not be materialized when partial_supervision is empty")

    def fake_pairwise_model(*args: Any, **_kwargs: Any) -> CandidateBatchPairwiseModelResult:
        candidate_batch = args[1]
        return CandidateBatchPairwiseModelResult(
            row_signals={
                "paper_author_list_max_overlap_count": np.zeros(candidate_batch.row_count, dtype=np.float32),
            },
            pairwise_stats=cast(Any, None),
            telemetry={"feature_seconds": 0.0},
        )

    def fake_retrieved_candidates(*args: Any, **kwargs: Any) -> LinkOrAbstainRetrievedCandidatesResult:
        current_retrieval_batch = args[1]
        return LinkOrAbstainRetrievedCandidatesResult(
            feature_matrix=LinkerFeatureMatrix(
                matrix=np.empty((2, 0), dtype=np.float32),
                feature_columns=(),
                candidate_batch=current_retrieval_batch.candidate_batch,
            ),
            compact_result=LinkOrAbstainCompactResult(
                probabilities=np.asarray([0.8, 0.2], dtype=np.float32),
                decisions=(),
            ),
            telemetry={"candidate_row_count": current_retrieval_batch.candidate_batch.row_count},
        )

    monkeypatch.setattr(runtime_module, "_candidate_pair_ids", fail_candidate_pair_ids)
    monkeypatch.setattr(
        runtime_module,
        "compute_candidate_batch_pairwise_model_and_aggregate_stats",
        fake_pairwise_model,
    )
    monkeypatch.setattr(runtime_module, "_production_query_author_row_signals", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runtime_module,
        "_predict_incremental_link_or_abstain_retrieved_candidates",
        fake_retrieved_candidates,
    )

    result = runtime_module._predict_incremental_link_or_abstain_production_from_retrieval_private(
        Clusterer(
            featurizer_info=FeaturizationInfo(features_to_use=[]),
            classifier=None,
            n_jobs=1,
            use_default_constraints_as_supervision=False,
        ),
        _raw_test_artifact(),
        featurizer=FakeFeaturizer(),
        retrieval_batch=retrieval_batch,
        queries=[object()],
        query_signature_ids=["q"],
        partial_supervision=None,
        cluster_seeds_require={"s1": "c_ada", "s2": "c_other", "s3": "c_other"},
        n_jobs=1,
        total_ram_bytes=None,
        retrieval_top_k=2,
    )

    assert result.telemetry["partial_supervision_pair_count"] == 0
    assert result.telemetry["candidate_row_count"] == 2

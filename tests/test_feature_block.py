from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import s2and.incremental_linking.feature_block_arrow as feature_block_arrow_module
from s2and.arrow_inputs import MissingArrowArtifactError, ValidatedArrowInputs
from s2and.consts import NORMALIZATION_VERSION
from s2and.data import ANDData, NameCounts
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import (
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    FeatureBlock,
    FeatureBlockPaper,
    FeatureBlockPaperAuthor,
    FeatureBlockSignature,
    FeatureBlockSignatureOrder,
    IncrementalQuerySignatureRequest,
    arrow_ipc_physical_layout,
    feature_block_for_signature_order,
    feature_block_signature_order_from_raw_candidate_plan,
    raw_planner_arrow_physical_layout,
    read_altered_cluster_signatures_arrow,
    read_cluster_seed_disallows_arrow,
    read_cluster_seeds_arrow,
    read_incremental_query_signatures_arrow,
    temporary_arrow_paths_with_cluster_seeds,
    write_altered_cluster_signatures_arrow,
    write_arrow_batch_lookup_index,
    write_arrow_ipc_table,
    write_feature_block_arrow_tables,
    write_incremental_query_signatures_arrow,
    write_name_counts_index,
    write_raw_arrow_batch_lookup_indexes,
)
from s2and.incremental_linking.feature_block_arrow import feature_block_from_arrow_paths
from s2and.incremental_linking.features import LinkerFeatureMatrix
from s2and.incremental_linking.retrieval import (
    RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
    RawArrowPlanBundle,
    build_linker_retrieval_batch_from_raw_plan_bundle,
)
from s2and.incremental_linking.runtime import (
    CandidateBatchPairwiseModelResult,
    LinkOrAbstainCompactResult,
    LinkOrAbstainProductionResult,
    LinkOrAbstainRetrievedCandidatesResult,
    _predict_incremental_link_or_abstain_from_preplanned_raw_arrow,
    predict_incremental_link_or_abstain_from_raw_arrow_paths,
)
from s2and.model import Clusterer
from scripts.arrow_conversion_helpers import feature_block_from_anddata, write_feature_block_arrow_from_anddata
from tests.helpers import tiny_name_counts_provenance, tiny_name_counts_tuple, write_test_arrow_artifact_manifest


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
    return SimpleNamespace(metadata=SimpleNamespace(retrieval_top_k=retrieval_top_k))


def test_feature_block_paper_is_reliable_parses_false_string() -> None:
    paper = FeatureBlockPaper(
        paper_id="p1",
        title="",
        abstract="",
        venue="",
        journal_name="",
        year=None,
        is_reliable=cast(Any, "false"),
    )

    assert paper.is_reliable is False


@pytest.mark.parametrize(
    ("is_reliable", "language_reliability", "match"),
    [
        (True, np.nan, "must be finite"),
        (True, np.inf, "must be finite"),
        (True, -0.01, r"must be in \[0.0, 1.0\]"),
        (True, 1.01, r"must be in \[0.0, 1.0\]"),
        (False, 0.25, "must be 0.0 when"),
    ],
)
def test_feature_block_paper_rejects_malformed_language_reliability(
    is_reliable: bool,
    language_reliability: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        FeatureBlockPaper(
            paper_id="p1",
            title="A title",
            abstract=None,
            venue=None,
            journal_name=None,
            year=2024,
            predicted_language="en",
            is_reliable=is_reliable,
            language_reliability=language_reliability,
        )


@pytest.mark.parametrize(
    ("is_reliable", "language_reliability", "match"),
    [
        (None, 0.75, "predicted_language requires FeatureBlockPaper.is_reliable"),
        (True, None, "predicted_language requires FeatureBlockPaper.language_reliability"),
    ],
)
def test_feature_block_paper_requires_complete_stored_language_detection(
    is_reliable: bool | None,
    language_reliability: float | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        FeatureBlockPaper(
            paper_id="p1",
            title="A title",
            abstract=None,
            venue=None,
            journal_name=None,
            year=2024,
            predicted_language="en",
            is_reliable=is_reliable,
            language_reliability=language_reliability,
        )


def test_feature_block_paper_accepts_language_reliability_boundaries() -> None:
    reliable = FeatureBlockPaper(
        paper_id="p1",
        title="A title",
        abstract=None,
        venue=None,
        journal_name=None,
        year=2024,
        predicted_language="en",
        is_reliable=True,
        language_reliability=1.0,
    )
    unreliable = FeatureBlockPaper(
        paper_id="p2",
        title="Un titre",
        abstract=None,
        venue=None,
        journal_name=None,
        year=2024,
        predicted_language="fr",
        is_reliable=False,
        language_reliability=0.0,
    )

    assert reliable.language_reliability == 1.0
    assert unreliable.language_reliability == 0.0


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
        "schema_version": RAW_CANDIDATE_PLAN_SCHEMA_VERSION,
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


def _validated_empty_arrow_inputs() -> ValidatedArrowInputs:
    return ValidatedArrowInputs._from_verified(
        paths={},
        generation_id="test-generation",
        normalization_version=NORMALIZATION_VERSION,
    )


def _feature_block_for_plan(
    *,
    cluster_seeds_disallow: tuple[tuple[str, str], ...] = (("q", "s2"),),
    specter_embeddings: np.ndarray | None = None,
    specter_paper_ids: tuple[str, ...] = (),
) -> FeatureBlock:
    return FeatureBlock(
        signatures=(
            FeatureBlockSignature(
                signature_id="q",
                paper_id="p_q",
                author_first="Ada",
                author_middle="",
                author_last="Lovelace",
                author_suffix="",
                author_affiliations=("Analytical Engine Lab",),
                author_orcid="0000-0000-0000-0001",
                author_position=0,
                author_block="a lovelace",
                author_email="",
                source_author_ids=("source-q",),
            ),
            FeatureBlockSignature(
                signature_id="s1",
                paper_id="p1",
                author_first="Ada",
                author_middle="",
                author_last="Lovelace",
                author_suffix="",
                author_affiliations=("Analytical Engine Lab",),
                author_orcid=None,
                author_position=0,
                author_block="a lovelace",
                author_email="",
                source_author_ids=("source-s1",),
            ),
            FeatureBlockSignature(
                signature_id="s2",
                paper_id="p2",
                author_first="Grace",
                author_middle="",
                author_last="Hopper",
                author_suffix="",
                author_affiliations=("Analytical Engine Lab",),
                author_orcid=None,
                author_position=0,
                author_block="g hopper",
                author_email="",
                source_author_ids=("source-s2",),
            ),
            FeatureBlockSignature(
                signature_id="s3",
                paper_id="p3",
                author_first="Grace",
                author_middle="",
                author_last="Hopper",
                author_suffix="",
                author_affiliations=("Analytical Engine Lab",),
                author_orcid=None,
                author_position=1,
                author_block="g hopper",
                author_email="",
                source_author_ids=("source-s3",),
            ),
        ),
        papers=(
            FeatureBlockPaper(
                paper_id="p_q",
                title="Notes",
                abstract="",
                venue="Royal Society",
                journal_name="",
                year=1843,
            ),
            FeatureBlockPaper(
                paper_id="p1",
                title="Notes",
                abstract="",
                venue="Royal Society",
                journal_name="",
                year=1843,
            ),
            FeatureBlockPaper(
                paper_id="p2",
                title="Compiler",
                abstract="",
                venue="Royal Society",
                journal_name="",
                year=1952,
            ),
            FeatureBlockPaper(
                paper_id="p3",
                title="Compiler",
                abstract="",
                venue="Royal Society",
                journal_name="",
                year=1952,
            ),
        ),
        paper_authors=(
            FeatureBlockPaperAuthor(paper_id="p_q", position=0, author_name="Ada Lovelace"),
            FeatureBlockPaperAuthor(paper_id="p_q", position=1, author_name="Charles Babbage"),
            FeatureBlockPaperAuthor(paper_id="p1", position=0, author_name="Ada Lovelace"),
            FeatureBlockPaperAuthor(paper_id="p1", position=1, author_name="Charles Babbage"),
            FeatureBlockPaperAuthor(paper_id="p2", position=0, author_name="Grace Hopper"),
            FeatureBlockPaperAuthor(paper_id="p2", position=1, author_name="Jane Doe"),
            FeatureBlockPaperAuthor(paper_id="p3", position=0, author_name="Jane Doe"),
            FeatureBlockPaperAuthor(paper_id="p3", position=1, author_name="Grace Hopper"),
        ),
        cluster_seeds_require=(("s1", "c_ada"), ("s2", "c_other"), ("s3", "c_other")),
        cluster_seeds_disallow=cluster_seeds_disallow,
        query_signature_ids=("q",),
        specter_paper_ids=specter_paper_ids,
        specter_embeddings=specter_embeddings,
    )


def _write_feature_block_arrow_paths(tmp_path: Path) -> dict[str, str]:
    feature_block = _feature_block_for_plan()
    return write_feature_block_arrow_tables(feature_block, tmp_path, include_empty_cluster_seeds=True)


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


def _strict_signature_arrow_table(**overrides: Any) -> Any:
    pa = pytest.importorskip("pyarrow")
    row_count = len(overrides.get("signature_id", ["q", "s1", "s2", "s3"]))
    data = {
        "signature_id": pa.array(["q", "s1", "s2", "s3"], type=pa.string()),
        "paper_id": pa.array(["p_q", "p1", "p2", "p3"], type=pa.string()),
        "author_first": pa.array(["Ada", "Ada", "Grace", "Grace"], type=pa.string()),
        "author_middle": pa.array(["", "", "", ""], type=pa.string()),
        "author_last": pa.array(["Lovelace", "Lovelace", "Hopper", "Hopper"], type=pa.string()),
        "author_suffix": pa.array(["", "", "", ""], type=pa.string()),
        "author_affiliations": pa.array([[], [], [], []], type=pa.list_(pa.string())),
        "author_orcid": pa.array(["0000-0000-0000-0001", None, None, None], type=pa.string()),
        "author_position": pa.array([0, 0, 0, 1], type=pa.int64()),
    }
    for key, value in overrides.items():
        if not hasattr(value, "type"):
            raise TypeError(f"override {key!r} must be a pyarrow Array")
        if len(value) != row_count:
            raise ValueError(f"override {key!r} length {len(value)} does not match row count {row_count}")
        data[key] = value
    return pa.table(data)


def test_feature_block_from_anddata_builds_requested_mini_contract() -> None:
    dataset = _tiny_anddata()
    dataset.cluster_seeds_disallow = set()
    feature_block = feature_block_from_anddata(
        dataset,
        signature_ids=["q", "s1"],
        query_signature_ids=["q"],
    )

    assert feature_block.signature_ids == ("q", "s1")
    assert feature_block.query_signature_ids == ("q",)
    assert feature_block.signature_id_to_index == {"q": 0, "s1": 1}
    assert feature_block.cluster_seeds_require == (("s1", "c_ada"),)
    assert feature_block.cluster_seeds_disallow == ()
    assert feature_block.seed_component_members == {"c_ada": ("s1",)}
    assert [paper.paper_id for paper in feature_block.papers] == ["p_q", "p1"]
    assert [(row.paper_id, row.position, row.author_name) for row in feature_block.paper_authors] == [
        ("p_q", 0, "ada lovelace"),
        ("p_q", 1, "charles babbage"),
        ("p1", 0, "ada lovelace"),
        ("p1", 1, "charles babbage"),
    ]
    assert feature_block.signatures[0].author_orcid == "0000-0000-0000-0001"
    assert feature_block.specter_paper_ids == ("p_q", "p1")
    assert feature_block.specter_embeddings is not None
    np.testing.assert_allclose(feature_block.specter_embeddings, [[1.0, 0.0], [1.0, 0.1]])


def test_feature_block_from_anddata_does_not_infer_reliability_from_boolean() -> None:
    dataset = _tiny_anddata()
    dataset.papers["p_q"] = dataset.papers["p_q"]._replace(
        predicted_language="en",
        is_reliable=True,
        language_reliability=None,
    )

    with pytest.raises(ValueError, match="predicted_language requires FeatureBlockPaper.language_reliability"):
        feature_block_from_anddata(
            dataset,
            signature_ids=["q"],
            query_signature_ids=["q"],
        )


def test_feature_block_from_anddata_rejects_signature_missing_paper() -> None:
    dataset = _tiny_anddata()
    del dataset.papers["p1"]

    with pytest.raises(ValueError, match="missing signature paper_id"):
        feature_block_from_anddata(
            dataset,
            signature_ids=["q", "s1"],
            query_signature_ids=["q"],
        )


def test_feature_block_to_arrow_tables_matches_raw_schema() -> None:
    pa = pytest.importorskip("pyarrow")

    dataset = _tiny_anddata()
    dataset.cluster_seeds_disallow = set()
    tables = feature_block_from_anddata(dataset, signature_ids=["q", "s1"], query_signature_ids=["q"]).to_arrow_tables()

    assert set(tables) == {
        "signatures",
        "papers",
        "paper_authors",
        "cluster_seeds",
        "cluster_seed_disallows",
        "specter",
    }
    assert tables["signatures"].column_names == [
        "signature_id",
        "paper_id",
        "author_first",
        "author_middle",
        "author_last",
        "author_suffix",
        "author_affiliations",
        "author_orcid",
        "author_position",
        "author_block",
        "author_email",
        "source_author_ids",
    ]
    assert tables["signatures"].schema.field("author_suffix").type == pa.string()
    assert tables["papers"].schema.field("abstract").type == pa.string()
    assert tables["papers"].schema.field("predicted_language").type == pa.string()
    assert tables["papers"].schema.field("is_reliable").type == pa.bool_()
    assert tables["papers"].schema.field("language_reliability").type == pa.float64()
    assert tables["cluster_seeds"].to_pydict() == {"signature_id": ["s1"], "cluster_id": ["c_ada"]}
    assert tables["cluster_seed_disallows"].to_pydict() == {"signature_id_1": [], "signature_id_2": []}


def test_feature_block_to_arrow_tables_keeps_all_null_optional_columns_typed() -> None:
    pa = pytest.importorskip("pyarrow")

    tables = FeatureBlock(
        signatures=(
            FeatureBlockSignature(
                signature_id="q",
                paper_id="p_q",
                author_first="Ada",
                author_middle=None,
                author_last="Lovelace",
                author_suffix=None,
                author_affiliations=(),
                author_orcid=None,
                author_position=0,
            ),
        ),
        papers=(
            FeatureBlockPaper(
                paper_id="p_q",
                title="",
                abstract=None,
                venue=None,
                journal_name=None,
                year=1843,
            ),
        ),
        paper_authors=(FeatureBlockPaperAuthor(paper_id="p_q", position=0, author_name="Ada Lovelace"),),
    ).to_arrow_tables()

    assert tables["signatures"].schema.field("author_suffix").type == pa.string()
    assert tables["signatures"].schema.field("author_email").type == pa.string()
    assert tables["papers"].schema.field("predicted_language").type == pa.string()
    assert tables["papers"].schema.field("is_reliable").type == pa.bool_()
    assert tables["papers"].schema.field("language_reliability").type == pa.float64()
    assert tables["cluster_seeds"].schema.field("signature_id").type == pa.string()
    assert tables["cluster_seed_disallows"].schema.field("signature_id_1").type == pa.string()


def test_write_feature_block_arrow_from_anddata_skips_empty_seed_table(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    dataset = _tiny_anddata()
    dataset.cluster_seeds_disallow = set()
    paths = write_feature_block_arrow_from_anddata(
        dataset,
        tmp_path,
        signature_ids=["q"],
        query_signature_ids=["q"],
        include_specter=False,
    )

    assert set(paths) == {"signatures", "papers", "paper_authors"}
    with pa.memory_map(paths["signatures"], "r") as source:
        signatures = pa.ipc.open_file(source).read_all()
    assert "name_count_first" not in signatures.column_names
    assert signatures.to_pydict()["signature_id"] == ["q"]


def test_incremental_query_signatures_arrow_round_trips_typed_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
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
    pa = pytest.importorskip("pyarrow")
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
    pa = pytest.importorskip("pyarrow")
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
    pa = pytest.importorskip("pyarrow")
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
    pa = pytest.importorskip("pyarrow")
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


def test_incremental_query_signatures_arrow_rejects_integer_id_columns(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "integer_incremental_query_signatures.arrow"
    write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array([1], type=pa.int64()),
                "query_view": pa.array(["full"], type=pa.string()),
                "query_author": pa.array(["Ada Lovelace"], type=pa.string()),
            }
        ),
        path,
    )

    with pytest.raises(ValueError, match="column 'signature_id' expected string"):
        read_incremental_query_signatures_arrow(path)


def test_feature_block_from_arrow_paths_reads_cluster_seed_disallows(tmp_path: Path) -> None:
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)

    feature_block = feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())

    assert feature_block.cluster_seeds_disallow == (("q", "s2"),)


def test_feature_block_from_arrow_paths_accepts_missing_orcid_column(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    signatures_path = Path(arrow_paths["signatures"])
    with pa.memory_map(str(signatures_path), "r") as source:
        signatures = pa.ipc.open_file(source).read_all().drop(["author_orcid"])
        signatures = pa.Table.from_pylist(signatures.to_pylist(), schema=signatures.schema)
    write_arrow_ipc_table(signatures, signatures_path)

    feature_block = feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())

    assert all(signature.author_orcid is None for signature in feature_block.signatures)


def test_feature_block_from_arrow_paths_filters_one_sided_disallow_pair(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    one_sided = pa.table(
        {
            "signature_id_1": pa.array(["q"], type=pa.string()),
            "signature_id_2": pa.array(["unused"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(one_sided, Path(arrow_paths["cluster_seed_disallows"]))

    feature_block = feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())

    assert feature_block.cluster_seeds_disallow == ()


def test_feature_block_from_arrow_paths_filters_valid_out_of_scope_disallow_pairs(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    out_of_scope = pa.table(
        {
            "signature_id_1": pa.array(["unused1", "q"], type=pa.string()),
            "signature_id_2": pa.array(["unused2", "s2"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(out_of_scope, Path(arrow_paths["cluster_seed_disallows"]))

    feature_block = feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())

    assert feature_block.cluster_seeds_disallow == (("q", "s2"),)


def test_feature_block_from_arrow_paths_rejects_in_scope_self_disallow_pair(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    self_pair = pa.table(
        {
            "signature_id_1": pa.array(["q"], type=pa.string()),
            "signature_id_2": pa.array(["q"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(self_pair, Path(arrow_paths["cluster_seed_disallows"]))

    with pytest.raises(ValueError, match="self-pair"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())


def test_read_cluster_seed_disallows_arrow_rejects_integer_id_columns(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "cluster_seed_disallows.arrow"
    table = pa.table(
        {
            "signature_id_1": pa.array([1], type=pa.int64()),
            "signature_id_2": pa.array(["s2"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(table, path)

    with pytest.raises(ValueError, match="column 'signature_id_1' expected string"):
        read_cluster_seed_disallows_arrow(path)


def test_altered_cluster_signatures_arrow_round_trips_and_rejects_duplicates(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
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


@pytest.mark.parametrize(
    ("bad_path", "match"),
    [
        (None, "papers.*None"),
        (" ", "papers.*empty"),
        (".", "papers.*current directory"),
    ],
)
def test_temporary_arrow_paths_with_cluster_seeds_rejects_invalid_paths(
    tmp_path: Path,
    bad_path: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        with temporary_arrow_paths_with_cluster_seeds(
            {
                "signatures": tmp_path / "signatures.arrow",
                "papers": bad_path,
            },
            {},
            prefix="test-arrow-paths-",
        ):
            raise AssertionError("invalid paths should fail before yielding")


def test_temporary_arrow_paths_with_cluster_seeds_cleans_up_tmpdir(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    with temporary_arrow_paths_with_cluster_seeds(
        {
            "signatures": tmp_path / "signatures.arrow",
            "papers": tmp_path / "papers.arrow",
        },
        {"s1": "c1"},
        prefix="test-arrow-paths-",
        cluster_seeds_disallow=[("s1", "s2")],
    ) as paths:
        cluster_seed_path = Path(paths["cluster_seeds"])
        disallow_path = Path(paths["cluster_seed_disallows"])
        temp_dir = cluster_seed_path.parent

        assert paths["signatures"] == str(tmp_path / "signatures.arrow")
        assert paths["papers"] == str(tmp_path / "papers.arrow")
        assert cluster_seed_path.exists()
        assert disallow_path.exists()
        assert temp_dir.exists()

    assert not temp_dir.exists()
    assert not cluster_seed_path.exists()
    assert not disallow_path.exists()


def test_temporary_arrow_paths_with_cluster_seeds_rewrites_stale_empty_seed_path(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    stale_seed_path = tmp_path / "missing_cluster_seeds.arrow"

    with temporary_arrow_paths_with_cluster_seeds(
        {
            "signatures": tmp_path / "signatures.arrow",
            "papers": tmp_path / "papers.arrow",
            "cluster_seeds": stale_seed_path,
        },
        {},
        prefix="test-arrow-paths-",
        reuse_existing_cluster_seeds_when_empty=True,
        cluster_seeds_disallow=[("s1", "s2")],
    ) as paths:
        cluster_seed_path = Path(paths["cluster_seeds"])
        assert cluster_seed_path.exists()
        assert cluster_seed_path != stale_seed_path


def test_read_cluster_seeds_arrow_rejects_duplicate_signature_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
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


def test_read_cluster_seeds_arrow_rejects_conflicting_duplicate_signature_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "cluster_seeds.arrow"
    table = pa.table(
        {
            "signature_id": pa.array(["s1", "s1"], type=pa.string()),
            "cluster_id": pa.array(["c1", "c2"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(table, path)

    with pytest.raises(ValueError, match="duplicate signature_id"):
        read_cluster_seeds_arrow(path)


def test_read_cluster_seeds_arrow_rejects_integer_id_columns(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    path = tmp_path / "cluster_seeds.arrow"
    table = pa.table(
        {
            "signature_id": pa.array([1], type=pa.int64()),
            "cluster_id": pa.array(["c1"], type=pa.string()),
        }
    )
    write_arrow_ipc_table(table, path)

    with pytest.raises(ValueError, match="column 'signature_id' expected string"):
        read_cluster_seeds_arrow(path)


def test_feature_block_from_arrow_paths_rejects_duplicate_signature_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    duplicate_signatures = _strict_signature_arrow_table(
        signature_id=pa.array(["q", "q", "s2", "s3"], type=pa.string()),
        paper_id=pa.array(["p_q", "p_q", "p2", "p3"], type=pa.string()),
    )
    write_arrow_ipc_table(duplicate_signatures, Path(arrow_paths["signatures"]))

    with pytest.raises(ValueError, match="duplicate signature_id"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())


def test_feature_block_from_arrow_paths_does_not_infer_reliability_from_boolean(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    papers_path = Path(arrow_paths["papers"])
    with pa.memory_map(str(papers_path), "r") as source:
        papers = pa.ipc.open_file(source).read_all()
        paper_rows = papers.to_pylist()
        paper_schema = papers.schema
    papers = pa.Table.from_pylist(paper_rows, schema=paper_schema)
    row_count = papers.num_rows
    predicted_languages = [None] * row_count
    reliable_flags = [None] * row_count
    predicted_languages[0] = "en"
    reliable_flags[0] = True
    predicted_index = papers.schema.get_field_index("predicted_language")
    reliable_index = papers.schema.get_field_index("is_reliable")
    reliability_index = papers.schema.get_field_index("language_reliability")
    papers = papers.set_column(
        predicted_index,
        "predicted_language",
        pa.array(predicted_languages, type=pa.string()),
    )
    papers = papers.set_column(
        reliable_index,
        "is_reliable",
        pa.array(reliable_flags, type=pa.bool_()),
    )
    papers = papers.remove_column(reliability_index)
    write_arrow_ipc_table(papers, papers_path)

    with pytest.raises(ValueError, match="predicted_language requires FeatureBlockPaper.language_reliability"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())


def test_feature_block_from_arrow_paths_validates_raw_plan_schema(tmp_path: Path) -> None:
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    raw_plan = _raw_plan()
    del raw_plan["schema_version"]

    with pytest.raises(KeyError, match="schema_version"):
        feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=raw_plan)


def test_filter_arrow_table_by_values_rejects_post_cast_duplicates() -> None:
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    table = pa.table({"paper_id": pa.array(["p1", "p2"], type=pa.string())})

    with pytest.raises(ValueError, match="paper_id filter values are not one-to-one"):
        feature_block_arrow_module._filter_arrow_table_by_values(pa, pc, table, "paper_id", ["p1", "p1"])  # noqa: SLF001


def test_name_counts_generation_cleanup_skips_unpublished_generation(tmp_path: Path) -> None:
    index_path, _metrics = write_name_counts_index(
        tmp_path,
        tiny_name_counts_tuple(),
        tiny_name_counts_provenance(),
        overwrite=True,
    )
    index_dir = Path(index_path)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    current = index_dir / Path(manifest["files"]["first"]["path"]).parent
    generations_dir = index_dir / "generations"
    unpublished = generations_dir / "gen-unpublished"
    old_published = generations_dir / "gen-old"
    unpublished.mkdir()
    old_published.mkdir()
    (old_published / ".published").write_text("", encoding="utf-8")

    feature_block_arrow_module.cleanup_stale_name_counts_generations(index_dir)

    assert current.exists()
    assert unpublished.exists()
    assert not old_published.exists()


def test_feature_block_from_arrow_paths_accepts_null_string_list_values(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    arrow_paths = _write_feature_block_arrow_paths(tmp_path)
    with pa.memory_map(arrow_paths["signatures"], "r") as source:
        signatures = pa.ipc.open_file(source).read_all()
        signature_rows = signatures.to_pylist()
        signature_schema = signatures.schema
    signature_rows[0]["source_author_ids"] = None
    signatures = pa.Table.from_pylist(signature_rows, schema=signature_schema)
    write_arrow_ipc_table(signatures, Path(arrow_paths["signatures"]))

    feature_block = feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan())

    assert feature_block.signatures[0].source_author_ids == ()


def test_feature_block_from_arrow_paths_reads_specter_when_requested(tmp_path: Path) -> None:
    feature_block = _feature_block_for_plan(
        specter_paper_ids=("p_q", "p1"),
        specter_embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    arrow_paths = write_feature_block_arrow_tables(feature_block, tmp_path, include_empty_cluster_seeds=True)

    materialized = feature_block_from_arrow_paths(arrow_paths, raw_candidate_plan=_raw_plan(), include_specter=True)

    assert materialized.specter_paper_ids == ("p_q", "p1")
    assert materialized.specter_embeddings is not None
    np.testing.assert_allclose(materialized.specter_embeddings, [[1.0, 0.0], [0.5, 0.5]])


def test_write_arrow_ipc_table_writes_bounded_record_batches(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    table = pa.table({"signature_id": pa.array([str(index) for index in range(5)], type=pa.string())})
    path = write_arrow_ipc_table(table, tmp_path / "signatures.arrow", max_record_batch_rows=2)

    assert arrow_ipc_physical_layout(path) == {
        "row_count": 5,
        "record_batch_count": 3,
        "actual_max_batch_rows": 2,
    }


def test_raw_planner_index_rejects_unbounded_large_batch(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    table = pa.table({"signature_id": pa.array([str(index) for index in range(5)], type=pa.string())})
    path = write_arrow_ipc_table(table, tmp_path / "signatures.arrow")

    with pytest.raises(ValueError, match="exceeding the raw-planner limit of 2"):
        write_raw_arrow_batch_lookup_indexes(
            {"signatures": path},
            tmp_path,
            max_record_batch_rows={"signatures": 2},
        )


def test_raw_planner_index_metadata_uses_stem_qualified_sidecar(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

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
    assert layout["schema"] == "s2and_arrow_physical_v1"
    assert layout["tables"]["signatures"]["batch_index_present"] is True
    assert layout["tables"]["signatures"]["max_record_batch_rows"] == 2
    assert RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS["signatures"] == 16_384


def test_raw_planner_index_omits_none_optional_paths(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

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
    pa = pytest.importorskip("pyarrow")

    path = write_arrow_ipc_table(
        pa.table({"signature_id": pa.array(["s1", None], type=pa.string())}),
        tmp_path / "signatures.arrow",
    )

    with pytest.raises(ValueError, match="null values in key column"):
        write_raw_arrow_batch_lookup_indexes({"signatures": path}, tmp_path)


def test_raw_planner_index_rejects_stale_python_reuse(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
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


def test_raw_planner_index_rejects_same_size_sampled_rewrite_python_reuse(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")

    signature_ids = [f"key{index:013d}" for index in range(30_000)]
    path = write_arrow_ipc_table(
        pa.table(
            {
                "signature_id": pa.array(signature_ids, type=pa.string()),
                "payload": pa.array(["x" * 8] * len(signature_ids), type=pa.string()),
            }
        ),
        tmp_path / "signatures.arrow",
        max_record_batch_rows=1000,
    )
    index_path = tmp_path / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(path, index_path, key_column="signature_id", table_name="signatures")
    payload = Path(path).read_bytes()
    old_value = signature_ids[0].encode()
    new_value = b"new0000000000000"
    rewrite_offset = payload.index(old_value)
    assert len(old_value) == len(new_value)
    assert rewrite_offset < 65_536
    Path(path).write_bytes(payload[:rewrite_offset] + new_value + payload[rewrite_offset + len(old_value) :])

    with pytest.raises(ValueError, match="stale"):
        write_arrow_batch_lookup_index(
            path,
            index_path,
            key_column="signature_id",
            table_name="signatures",
            overwrite=False,
        )


def test_raw_planner_index_reuse_metrics_match_fresh_schema(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
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
    pa = pytest.importorskip("pyarrow")
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
    provenance = tiny_name_counts_provenance()

    index_path, index_metrics = write_name_counts_index(tmp_path, mappings, provenance)

    assert index_metrics["reused"] is False
    assert index_metrics["row_count"] == 4
    assert index_metrics["first_count"] == 1
    manifest = json.loads((Path(index_path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "name_counts_index_v1"
    assert manifest["exact_string_verification"] is True
    assert manifest["files"]["first"]["path"].startswith("generations/")
    assert write_name_counts_index(tmp_path, mappings, provenance)[1] == {"reused": True}


def test_write_name_counts_index_rebuilds_fingerprintless_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_mappings = ({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7})
    first_provenance = tiny_name_counts_provenance()
    index_path, _metrics = write_name_counts_index(tmp_path, first_mappings, first_provenance)
    manifest_path = Path(index_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_fingerprint = manifest.pop("fingerprint")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    second_mappings = ({"grace": 11}, {"hopper": 13}, {"grace hopper": 17}, {"hopper g": 19})
    second_provenance = {**tiny_name_counts_provenance(), "generation_id": "test-grace-hopper"}

    _index_path, metrics = write_name_counts_index(tmp_path, second_mappings, second_provenance, overwrite=False)

    rebuilt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert metrics["reused"] is False
    assert rebuilt_manifest["fingerprint"] != original_fingerprint


def test_write_name_counts_index_rebuilds_manifest_with_missing_schema_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mappings = ({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7})
    provenance = tiny_name_counts_provenance()
    index_path, _metrics = write_name_counts_index(tmp_path, mappings, provenance)
    manifest_path = Path(index_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("schema_version")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    _index_path, metrics = write_name_counts_index(tmp_path, mappings, provenance, overwrite=False)

    rebuilt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert metrics["reused"] is False
    assert rebuilt_manifest["schema_version"] == "name_counts_index_v1"


def test_write_name_counts_index_failed_overwrite_keeps_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_mappings = ({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7})
    first_provenance = tiny_name_counts_provenance()
    index_path, _metrics = write_name_counts_index(tmp_path, first_mappings, first_provenance)
    manifest_path = Path(index_path) / "manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    original_first_path = Path(index_path) / json.loads(original_manifest)["files"]["first"]["path"]

    real_write_file = feature_block_arrow_module._write_name_count_index_file  # noqa: SLF001

    def fail_after_first_file(path: Path, kind: str, mapping: Any, **kwargs: Any) -> dict[str, int]:
        if kind == "last":
            raise RuntimeError("simulated index write failure")
        return real_write_file(path, kind, mapping, **kwargs)

    monkeypatch.setattr(feature_block_arrow_module, "_write_name_count_index_file", fail_after_first_file)
    second_mappings = ({"alan": 11}, {"turing": 13}, {"alan turing": 17}, {"turing a": 19})
    second_provenance = {**tiny_name_counts_provenance(), "generation_id": "test-alan-turing"}

    with pytest.raises(RuntimeError, match="simulated index write failure"):
        write_name_counts_index(tmp_path, second_mappings, second_provenance, overwrite=True)

    assert manifest_path.read_text(encoding="utf-8") == original_manifest
    assert original_first_path.exists()


def test_write_name_counts_index_published_marker_failure_keeps_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_mappings = ({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7})
    first_provenance = tiny_name_counts_provenance()
    index_path, _metrics = write_name_counts_index(tmp_path, first_mappings, first_provenance)
    manifest_path = Path(index_path) / "manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    original_first_path = Path(index_path) / json.loads(original_manifest)["files"]["first"]["path"]

    real_open = Path.open

    def fail_published_write(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == ".published":
            raise RuntimeError("simulated published marker failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_published_write)
    second_mappings = ({"alan": 11}, {"turing": 13}, {"alan turing": 17}, {"turing a": 19})
    second_provenance = {**tiny_name_counts_provenance(), "generation_id": "test-alan-turing"}

    with pytest.raises(RuntimeError, match="simulated published marker failure"):
        write_name_counts_index(tmp_path, second_mappings, second_provenance, overwrite=True)

    assert manifest_path.read_text(encoding="utf-8") == original_manifest
    assert original_first_path.exists()


def test_feature_block_from_anddata_filters_one_sided_disallow_pair() -> None:
    feature_block = feature_block_from_anddata(
        _tiny_anddata(),
        signature_ids=["q", "s1"],
        query_signature_ids=["q"],
    )

    assert feature_block.cluster_seeds_disallow == ()


def test_feature_block_contract_rejects_out_of_block_disallow_pair() -> None:
    with pytest.raises(ValueError, match="missing from FeatureBlock"):
        FeatureBlock(
            signatures=(
                FeatureBlockSignature(
                    signature_id="q",
                    paper_id="p_q",
                    author_first="Ada",
                    author_middle=None,
                    author_last="Lovelace",
                    author_suffix=None,
                    author_affiliations=(),
                    author_orcid=None,
                    author_position=0,
                ),
            ),
            papers=(
                FeatureBlockPaper(
                    paper_id="p_q",
                    title="Notes",
                    abstract=None,
                    venue=None,
                    journal_name=None,
                    year=1843,
                ),
            ),
            paper_authors=(FeatureBlockPaperAuthor(paper_id="p_q", position=0, author_name="Ada Lovelace"),),
            cluster_seeds_disallow=(("q", "outside"),),
        )


def test_feature_block_signature_rejects_scalar_sequence_fields() -> None:
    with pytest.raises(ValueError, match="FeatureBlockSignature.author_affiliations"):
        FeatureBlockSignature(
            signature_id="q",
            paper_id="p_q",
            author_first="Ada",
            author_middle=None,
            author_last="Lovelace",
            author_suffix=None,
            author_affiliations=cast(Any, "Lab"),
            author_orcid=None,
            author_position=0,
        )


def test_feature_block_for_signature_order_rejects_missing_plan_signature() -> None:
    dataset = _tiny_anddata()
    dataset.cluster_seeds_disallow = set()
    feature_block = feature_block_from_anddata(
        dataset,
        signature_ids=["q", "s1"],
        query_signature_ids=["q"],
    )
    order = feature_block_signature_order_from_raw_candidate_plan(_raw_plan())

    with pytest.raises(ValueError, match="missing raw-plan signatures"):
        feature_block_for_signature_order(feature_block, order)


def test_feature_block_for_signature_order_filters_cross_subset_disallow_pair() -> None:
    feature_block = feature_block_from_anddata(
        _tiny_anddata(),
        query_signature_ids=["q"],
    )
    order = FeatureBlockSignatureOrder(signature_ids=("q", "s1"), query_signature_ids=("q",))

    mini = feature_block_for_signature_order(feature_block, order)

    assert mini.cluster_seeds_disallow == ()


def test_feature_block_for_signature_order_keeps_specter_aligned_to_papers() -> None:
    feature_block = FeatureBlock(
        signatures=(
            FeatureBlockSignature(
                signature_id="s1",
                paper_id="p1",
                author_first="Ada",
                author_middle=None,
                author_last="Lovelace",
                author_suffix=None,
                author_affiliations=(),
                author_orcid=None,
                author_position=0,
            ),
            FeatureBlockSignature(
                signature_id="s2",
                paper_id="p2",
                author_first="Grace",
                author_middle=None,
                author_last="Hopper",
                author_suffix=None,
                author_affiliations=(),
                author_orcid=None,
                author_position=0,
            ),
        ),
        papers=(
            FeatureBlockPaper(
                paper_id="p1",
                title="Paper 1",
                abstract=None,
                venue=None,
                journal_name=None,
                year=2020,
            ),
            FeatureBlockPaper(
                paper_id="p2",
                title="Paper 2",
                abstract=None,
                venue=None,
                journal_name=None,
                year=2021,
            ),
        ),
        paper_authors=(
            FeatureBlockPaperAuthor(paper_id="p1", position=0, author_name="Ada Lovelace"),
            FeatureBlockPaperAuthor(paper_id="p2", position=0, author_name="Grace Hopper"),
        ),
        specter_paper_ids=("p2", "p1"),
        specter_embeddings=np.asarray([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    order = FeatureBlockSignatureOrder(signature_ids=("s1", "s2"), query_signature_ids=("s1",))

    mini = feature_block_for_signature_order(feature_block, order)

    assert tuple(paper.paper_id for paper in mini.papers) == ("p1", "p2")
    assert mini.specter_paper_ids == ("p1", "p2")
    assert mini.specter_embeddings is not None
    np.testing.assert_allclose(mini.specter_embeddings, [[1.0, 0.0], [2.0, 0.0]])


def test_feature_block_signature_order_from_raw_candidate_plan_queries_first() -> None:
    order = feature_block_signature_order_from_raw_candidate_plan(_raw_plan())

    assert order == FeatureBlockSignatureOrder(signature_ids=("q", "s1", "s2", "s3"), query_signature_ids=("q",))
    assert order.signature_id_to_index == {"q": 0, "s1": 1, "s2": 2, "s3": 3}


def test_feature_block_signature_order_rejects_empty_raw_plan() -> None:
    with pytest.raises(ValueError, match="at least one signature"):
        feature_block_signature_order_from_raw_candidate_plan(
            {
                "query_signature_ids": [],
                "left_signature_ids": [],
                "right_signature_ids": [],
            }
        )


def test_raw_candidate_plan_bridge_accepts_feature_block_signature_order() -> None:
    order = feature_block_signature_order_from_raw_candidate_plan(_raw_plan())

    retrieval_batch = build_linker_retrieval_batch_from_raw_plan_bundle(
        RawArrowPlanBundle.from_mapping(_raw_plan()),
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
            RawArrowPlanBundle.from_mapping(_raw_plan()),
            signature_id_to_index={"q": 0, "s1": 1, "s2": 2},
        )


def test_feature_block_rejects_duplicate_paper_author_positions() -> None:
    with pytest.raises(ValueError, match=r"duplicate \(paper_id, position\)"):
        FeatureBlock(
            signatures=(
                FeatureBlockSignature(
                    signature_id="s1",
                    paper_id="p1",
                    author_first="Alice",
                    author_middle=None,
                    author_last="Smith",
                    author_suffix=None,
                    author_affiliations=(),
                    author_orcid=None,
                    author_position=0,
                ),
            ),
            papers=(
                FeatureBlockPaper(
                    paper_id="p1",
                    title="One",
                    abstract="",
                    venue="",
                    journal_name="",
                    year=2020,
                ),
            ),
            paper_authors=(
                FeatureBlockPaperAuthor(paper_id="p1", position=0, author_name="Alice Smith"),
                FeatureBlockPaperAuthor(paper_id="p1", position=0, author_name="A. Smith"),
            ),
        )


def test_feature_block_rejects_empty_paper_author_name() -> None:
    with pytest.raises(ValueError, match="author_name must be non-empty"):
        FeatureBlockPaperAuthor(paper_id="p1", position=0, author_name="")


def test_raw_arrow_scoring_wrapper_uses_direct_arrow_featurizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrow_paths = _with_query_signatures(
        _with_fake_batch_indexes(_write_feature_block_arrow_paths(tmp_path), tmp_path),
        tmp_path,
    )
    captured: dict[str, Any] = {}

    class FakePlanner:
        def __init__(
            self,
            paths_arg: dict[str, str],
            query_signature_ids: list[str],
            **kwargs: Any,
        ) -> None:
            captured["retrieval_paths"] = paths_arg
            captured["retrieval_query_signature_ids"] = tuple(query_signature_ids)
            captured["retrieval_kwargs"] = kwargs
            self._query_signature_ids = list(query_signature_ids)

        @classmethod
        def from_query_signatures(cls, paths_arg: dict[str, str], **kwargs: Any) -> FakePlanner:
            rows = read_incremental_query_signatures_arrow(Path(paths_arg["query_signatures"]))
            return cls(paths_arg, [row.signature_id for row in rows], **kwargs)

        def plan_query_signatures(self) -> dict[str, Any]:
            return self.plan(self._query_signature_ids)

        def plan(self, query_signature_ids: list[str], **kwargs: Any) -> dict[str, Any]:
            captured["retrieval_plan_query_signature_ids"] = tuple(query_signature_ids)
            captured["retrieval_plan_kwargs"] = kwargs
            plan = _raw_plan()
            plan["telemetry"]["cluster_count"] = 99
            return plan

        def build_telemetry(self) -> dict[str, Any]:
            return {"timings": {}}

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3", "extra"]

    def fake_build_rust_featurizer_from_arrow_paths(paths_arg: Any, **kwargs: Any) -> FakeFeaturizer:
        captured["featurizer_paths"] = paths_arg
        captured["featurizer_signature_ids"] = tuple(kwargs["signature_ids"])
        return FakeFeaturizer()

    def fake_from_retrieval(**kwargs: Any) -> LinkOrAbstainProductionResult:
        retrieval_batch = kwargs["retrieval_batch"]
        captured["retrieval_left_indices"] = retrieval_batch.candidate_batch.left_signature_indices.tolist()
        captured["retrieval_right_indices"] = retrieval_batch.candidate_batch.right_signature_indices.tolist()
        captured["retrieval_query_author"] = retrieval_batch.row_signals["query_author"].tolist()
        captured["extra_row_signal_builder"] = kwargs["extra_row_signal_builder"]
        captured["seed_setup"] = kwargs["seed_setup"]
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

    monkeypatch.setattr("s2and.incremental_linking.runtime.feature_port._require_rust_runtime", lambda: FakeRustModule)
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port.build_rust_featurizer_from_arrow_paths",
        fake_build_rust_featurizer_from_arrow_paths,
    )
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime._predict_incremental_link_or_abstain_production_from_retrieval_private",
        lambda *args, **kwargs: fake_from_retrieval(**kwargs),
    )

    result = predict_incremental_link_or_abstain_from_raw_arrow_paths(
        _raw_test_clusterer(),
        _raw_test_artifact(),
        arrow_paths=arrow_paths,
        top_k=2,
        n_jobs=1,
        load_name_counts=False,
        name_tuples=set(),
    )

    assert captured["retrieval_query_signature_ids"] == ("q",)
    assert "query_signatures" in captured["retrieval_paths"]
    assert captured["retrieval_kwargs"]["top_k"] == 2
    assert captured["retrieval_plan_query_signature_ids"] == ("q",)
    assert captured["retrieval_plan_kwargs"] == {}
    assert captured["featurizer_signature_ids"] == ("q", "s1", "s2", "s3")
    assert captured["retrieval_left_indices"] == [0, 0, 0]
    assert captured["retrieval_right_indices"] == [1, 2, 3]
    assert captured["retrieval_query_author"] == ["Ada Lovelace", "Ada Lovelace"]
    assert captured["extra_row_signal_builder"] is None
    queries = captured["queries"]
    assert isinstance(queries, tuple)
    assert queries[0].query_author == "Ada Lovelace"
    assert captured["seed_setup"][0] == {"s1": "c_ada", "s2": "c_other", "s3": "c_other"}
    assert result.linked_signature_clusters == {"q": "c_ada"}
    assert result.telemetry["raw_arrow_signature_count"] == 5
    assert result.telemetry["raw_arrow_plan_signature_count"] == 4
    assert result.telemetry["raw_arrow_seed_signature_count"] == 3
    assert result.telemetry["seed_component_count"] == 2
    assert result.telemetry["raw_arrow_seed_component_count"] == 2
    assert result.telemetry["raw_arrow_plan_cluster_count"] == 99
    assert "raw_arrow_retrieval_seconds" in result.telemetry
    retrieval_paths_without_query_request = dict(captured["retrieval_paths"])
    retrieval_paths_without_query_request.pop("query_signatures")
    assert isinstance(captured["featurizer_paths"], ValidatedArrowInputs)
    assert retrieval_paths_without_query_request == dict(captured["featurizer_paths"])


def test_raw_arrow_rejects_candidate_plan_without_component_members() -> None:
    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3"]

    raw_plan = _raw_plan()
    raw_plan.pop("component_members")
    raw_plan["telemetry"]["cluster_count"] = 99

    with pytest.raises(KeyError, match="component_members"):
        _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
            _raw_test_clusterer(),
            _raw_test_artifact(),
            arrow_paths=_validated_empty_arrow_inputs(),
            query_signature_ids=["q"],
            raw_plan_bundle=RawArrowPlanBundle.from_mapping(raw_plan),
            rust_featurizer=FakeFeaturizer(),
            runtime_context=SimpleNamespace(operation="raw-arrow-test", run_id="raw-arrow-test"),
        )


def test_raw_arrow_scoring_uses_provided_query_signatures_arrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrow_paths = _with_fake_batch_indexes(_write_feature_block_arrow_paths(tmp_path), tmp_path)
    query_signatures_path = tmp_path / "incremental_query_signatures.arrow"
    write_incremental_query_signatures_arrow(
        query_signatures_path,
        ["q"],
        query_views=["full"],
        query_authors=["Ada Lovelace"],
    )
    arrow_paths["query_signatures"] = str(query_signatures_path)
    captured: dict[str, Any] = {}

    class StopAfterPlanner(RuntimeError):
        pass

    class FakePlanner:
        def __init__(self, paths_arg: dict[str, str], query_signature_ids: list[str], **_kwargs: Any) -> None:
            captured["retrieval_paths"] = dict(paths_arg)
            self._query_signature_ids = list(query_signature_ids)

        @classmethod
        def from_query_signatures(cls, paths_arg: dict[str, str], **kwargs: Any) -> FakePlanner:
            rows = read_incremental_query_signatures_arrow(Path(paths_arg["query_signatures"]))
            return cls(paths_arg, [row.signature_id for row in rows], **kwargs)

        def plan_query_signatures(self) -> dict[str, Any]:
            captured["planned_query_signature_ids"] = tuple(self._query_signature_ids)
            return _raw_plan()

        def build_telemetry(self) -> dict[str, Any]:
            return {"timings": {}}

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

    def stop_before_featurizer_build(paths_arg: Any, **_kwargs: Any) -> Any:
        captured["featurizer_paths"] = dict(paths_arg)
        raise StopAfterPlanner("captured planner boundary")

    monkeypatch.setattr("s2and.incremental_linking.runtime.feature_port._require_rust_runtime", lambda: FakeRustModule)
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port.build_rust_featurizer_from_arrow_paths",
        stop_before_featurizer_build,
    )

    with pytest.raises(StopAfterPlanner):
        predict_incremental_link_or_abstain_from_raw_arrow_paths(
            _raw_test_clusterer(),
            _raw_test_artifact(),
            arrow_paths=arrow_paths,
            top_k=2,
            n_jobs=1,
            load_name_counts=False,
            name_tuples=set(),
        )

    assert captured["planned_query_signature_ids"] == ("q",)
    assert captured["retrieval_paths"]["query_signatures"] == str(query_signatures_path)
    assert "query_signatures" not in captured["featurizer_paths"]


def test_raw_arrow_scoring_requires_query_signatures_before_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrow_paths = _with_fake_batch_indexes(_write_feature_block_arrow_paths(tmp_path), tmp_path)

    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port._require_rust_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("planner should not be loaded before validation")),
    )

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        predict_incremental_link_or_abstain_from_raw_arrow_paths(
            _raw_test_clusterer(),
            _raw_test_artifact(),
            arrow_paths=arrow_paths,
            top_k=2,
            n_jobs=1,
            load_name_counts=False,
            name_tuples=set(),
        )

    assert exc_info.value.context == "Raw Arrow scoring"
    assert exc_info.value.missing_keys == ("query_signatures",)


def test_raw_arrow_scoring_requires_planner_artifacts_before_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrow_paths = _with_query_signatures(
        _with_fake_batch_indexes(_write_feature_block_arrow_paths(tmp_path), tmp_path),
        tmp_path,
    )
    del arrow_paths["cluster_seeds"]

    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port._require_rust_runtime",
        lambda: (_ for _ in ()).throw(AssertionError("planner should not be loaded before validation")),
    )

    with pytest.raises(MissingArrowArtifactError) as exc_info:
        predict_incremental_link_or_abstain_from_raw_arrow_paths(
            _raw_test_clusterer(),
            _raw_test_artifact(),
            arrow_paths=arrow_paths,
            top_k=2,
            n_jobs=1,
            load_name_counts=False,
            name_tuples=set(),
        )

    assert exc_info.value.context == "Raw Arrow scoring"
    assert exc_info.value.missing_keys == ("cluster_seeds",)


def test_raw_arrow_partial_supervision_require_unknown_seed_rejected() -> None:
    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3"]

    raw_plan = _raw_plan()
    raw_plan["component_members"] = {}

    with pytest.raises(ValueError, match="partial_supervision_require_unknown_seed_signature"):
        _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
            _raw_test_clusterer(),
            _raw_test_artifact(),
            arrow_paths=_validated_empty_arrow_inputs(),
            query_signature_ids=["q"],
            raw_plan_bundle=RawArrowPlanBundle.from_mapping(raw_plan),
            rust_featurizer=FakeFeaturizer(),
            partial_supervision={("q", "s1"): 0},
        )


def test_raw_arrow_scoring_requires_featurizer_with_provided_raw_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_build_rust_featurizer_from_arrow_paths(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("provided raw candidate plans must not trigger a new Arrow featurizer scan")

    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port.build_rust_featurizer_from_arrow_paths",
        fail_build_rust_featurizer_from_arrow_paths,
    )

    with pytest.raises(ValueError, match="preplanned raw Arrow scoring requires rust_featurizer"):
        _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
            _raw_test_clusterer(),
            _raw_test_artifact(),
            arrow_paths=_validated_empty_arrow_inputs(),
            query_signature_ids=["q"],
            raw_plan_bundle=RawArrowPlanBundle.from_mapping(_raw_plan()),
            rust_featurizer=None,
        )


def test_preplanned_raw_arrow_scoring_uses_provided_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3", "window-only"]

    def fail_build_rust_featurizer_from_arrow_paths(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preplanned raw Arrow scoring should reuse the supplied featurizer")

    def fake_from_retrieval(**kwargs: Any) -> LinkOrAbstainProductionResult:
        retrieval_batch = kwargs["retrieval_batch"]
        captured["dataset"] = kwargs["dataset"]
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
        "s2and.incremental_linking.runtime.feature_port.build_rust_featurizer_from_arrow_paths",
        fail_build_rust_featurizer_from_arrow_paths,
    )
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime._predict_incremental_link_or_abstain_production_from_retrieval_private",
        lambda *args, **kwargs: fake_from_retrieval(**kwargs),
    )

    result = _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
        _raw_test_clusterer(),
        _raw_test_artifact(),
        arrow_paths=_validated_empty_arrow_inputs(),
        query_signature_ids=["q"],
        raw_plan_bundle=RawArrowPlanBundle.from_mapping(_raw_plan()),
        rust_featurizer=FakeFeaturizer(),
        top_k=2,
        n_jobs=1,
    )

    assert captured["dataset"] is None
    assert captured["retrieval_left_indices"] == [0, 0, 0]
    assert captured["retrieval_right_indices"] == [1, 2, 3]
    assert captured["queries"][0].query_author == "Ada Lovelace"
    assert result.linked_signature_clusters == {"q": "c_ada"}
    assert result.telemetry["raw_arrow_featurizer_reused"] == 1
    assert result.telemetry["raw_arrow_retrieval_seconds"] == 0.0


@pytest.mark.parametrize(
    ("suppress_orcid", "orcid_enabled_arg", "expected_orcid_enabled"),
    (
        (False, None, True),
        (True, None, False),
        (True, True, True),
        (False, False, False),
    ),
)
def test_raw_arrow_scoring_resolves_orcid_enabled_from_clusterer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suppress_orcid: bool,
    orcid_enabled_arg: bool | None,
    expected_orcid_enabled: bool,
) -> None:
    captured: dict[str, Any] = {}

    class StopAfterRetrieval(RuntimeError):
        pass

    class FakePlanner:
        def __init__(
            self,
            _paths_arg: dict[str, str],
            _query_signature_ids: list[str],
            **kwargs: Any,
        ) -> None:
            captured["orcid_enabled"] = kwargs["orcid_enabled"]
            self._query_signature_ids = list(_query_signature_ids)

        @classmethod
        def from_query_signatures(cls, paths_arg: dict[str, str], **kwargs: Any) -> FakePlanner:
            rows = read_incremental_query_signatures_arrow(Path(paths_arg["query_signatures"]))
            return cls(paths_arg, [row.signature_id for row in rows], **kwargs)

        def plan_query_signatures(self) -> dict[str, Any]:
            return self.plan(self._query_signature_ids)

        def plan(self, _query_signature_ids: list[str], **_kwargs: Any) -> dict[str, Any]:
            return _raw_plan()

        def build_telemetry(self) -> dict[str, Any]:
            return {"timings": {}}

    class FakeRustModule:
        RawBlockQueryCandidatePlanner = FakePlanner

    def stop_before_featurizer_build(*_args: Any, **_kwargs: Any) -> Any:
        raise StopAfterRetrieval("captured retrieval kwargs")

    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.require_arrow_name_counts_index_for_clusterer",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("s2and.incremental_linking.runtime.feature_port._require_rust_runtime", lambda: FakeRustModule)
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port.build_rust_featurizer_from_arrow_paths",
        stop_before_featurizer_build,
    )

    kwargs: dict[str, Any] = {}
    if orcid_enabled_arg is not None:
        kwargs["orcid_enabled"] = orcid_enabled_arg

    with pytest.raises(StopAfterRetrieval):
        predict_incremental_link_or_abstain_from_raw_arrow_paths(
            _raw_test_clusterer(suppress_orcid=suppress_orcid),
            _raw_test_artifact(),
            arrow_paths=_with_query_signatures(
                _with_fake_batch_indexes(_write_feature_block_arrow_paths(tmp_path), tmp_path),
                tmp_path,
            ),
            top_k=2,
            n_jobs=1,
            load_name_counts=False,
            name_tuples=set(),
            **kwargs,
        )

    assert captured["orcid_enabled"] is expected_orcid_enabled


def test_preplanned_raw_arrow_scoring_uses_provided_rust_featurizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3", "window-only"]

    def fail_build_rust_featurizer_from_arrow_paths(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("prebuilt raw Arrow featurizer should be reused")

    def fake_from_retrieval(**kwargs: Any) -> LinkOrAbstainProductionResult:
        retrieval_batch = kwargs["retrieval_batch"]
        captured["featurizer"] = kwargs["featurizer"]
        captured["retrieval_left_indices"] = retrieval_batch.candidate_batch.left_signature_indices.tolist()
        captured["retrieval_right_indices"] = retrieval_batch.candidate_batch.right_signature_indices.tolist()
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

    fake_featurizer = FakeFeaturizer()
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime.feature_port.build_rust_featurizer_from_arrow_paths",
        fail_build_rust_featurizer_from_arrow_paths,
    )
    monkeypatch.setattr(
        "s2and.incremental_linking.runtime._predict_incremental_link_or_abstain_production_from_retrieval_private",
        lambda *args, **kwargs: fake_from_retrieval(**kwargs),
    )

    result = _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
        _raw_test_clusterer(),
        _raw_test_artifact(),
        arrow_paths=_validated_empty_arrow_inputs(),
        query_signature_ids=["q"],
        raw_plan_bundle=RawArrowPlanBundle.from_mapping(_raw_plan()),
        rust_featurizer=fake_featurizer,
        top_k=2,
        n_jobs=1,
    )

    assert captured["featurizer"] is fake_featurizer
    assert captured["retrieval_left_indices"] == [0, 0, 0]
    assert captured["retrieval_right_indices"] == [1, 2, 3]
    assert result.telemetry["raw_arrow_featurizer_reused"] == 1
    assert result.telemetry["raw_arrow_signature_count"] == 5
    assert result.telemetry["raw_arrow_plan_signature_count"] == 5
    assert isinstance(result.telemetry["raw_arrow_featurizer_seconds"], float)


def test_preplanned_raw_arrow_scoring_rejects_mismatched_raw_plan_query_ids() -> None:
    class FakeFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["q", "s1", "s2", "s3"]

    with pytest.raises(ValueError, match="must exactly match requested query_signature_ids"):
        _predict_incremental_link_or_abstain_from_preplanned_raw_arrow(
            _raw_test_clusterer(),
            _raw_test_artifact(),
            arrow_paths=_validated_empty_arrow_inputs(),
            query_signature_ids=["s1"],
            raw_plan_bundle=RawArrowPlanBundle.from_mapping(_raw_plan()),
            rust_featurizer=FakeFeaturizer(),
            top_k=2,
            n_jobs=1,
        )


def test_from_retrieval_skips_pair_id_build_when_partial_supervision_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import s2and.incremental_linking.runtime as runtime_module

    order = feature_block_signature_order_from_raw_candidate_plan(_raw_plan())
    retrieval_batch = build_linker_retrieval_batch_from_raw_plan_bundle(
        RawArrowPlanBundle.from_mapping(_raw_plan()),
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
        dataset=None,
        featurizer=FakeFeaturizer(),
        retrieval_batch=retrieval_batch,
        queries=[object()],
        query_signature_ids=["q"],
        partial_supervision=None,
        seed_setup=(
            {"s1": "c_ada", "s2": "c_other", "s3": "c_other"},
            {"c_ada": "c_ada", "c_other": "c_other"},
            {"c_ada": ["s1"], "c_other": ["s2", "s3"]},
        ),
        n_jobs=1,
        total_ram_bytes=None,
        retrieval_top_k=2,
    )

    assert result.telemetry["partial_supervision_pair_count"] == 0
    assert result.telemetry["candidate_row_count"] == 2


def test_raw_candidate_plan_bridge_rejects_missing_schema_version() -> None:
    raw_plan = _raw_plan()
    del raw_plan["schema_version"]

    with pytest.raises(KeyError, match="schema_version"):
        build_linker_retrieval_batch_from_raw_plan_bundle(
            RawArrowPlanBundle.from_mapping(raw_plan),
            feature_block_signature_order=feature_block_signature_order_from_raw_candidate_plan(_raw_plan()),
        )

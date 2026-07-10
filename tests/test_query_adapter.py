from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

import s2and.incremental_linking.query_adapter as query_adapter_module
from s2and.incremental_linking.query_adapter import (
    build_cluster_summary,
    extract_query_features,
    mask_query_features,
    query_view_for_features,
    raw_paper_evidence_features,
)
from s2and.incremental_linking_training.query_support import counter_query_overlap, title_overlap
from tests.helpers import build_dummy_dataset, build_query_features


def _dataset_arg(dataset: object) -> Any:
    return cast(Any, dataset)


def _signature(paper_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        paper_id=paper_id,
        author_info_first="Alice",
        author_info_middle="",
        author_info_first_normalized_without_apostrophe="alice",
        author_info_middle_normalized_without_apostrophe="",
        author_info_position=0,
        author_info_coauthor_blocks=[],
        author_info_coauthors=[],
        author_info_affiliations_n_grams={},
        author_info_affiliations=[],
        author_info_orcid=None,
        author_info_name_counts=None,
    )


def test_title_and_venue_terms_keep_single_character_tokens() -> None:
    dataset = SimpleNamespace(
        signatures={"q": _signature("pq"), "c": _signature("pc")},
        papers={
            "pq": SimpleNamespace(title="A M Study", venue="Series A", journal_name=None, year=2020),
            "pc": SimpleNamespace(title="A Different Study", venue="A", journal_name=None, year=2021),
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    query = extract_query_features(_dataset_arg(dataset), "q", feature_cache=feature_cache)
    summary = build_cluster_summary(
        _dataset_arg(dataset),
        cluster_id="cluster",
        component_key="component",
        signature_ids=("c",),
        max_exemplars=4,
        feature_cache=feature_cache,
        orcid_enabled=False,
        block_key="block",
    )

    assert query.title_terms == frozenset({"a", "m", "study"})
    assert query.venue_terms == frozenset({"series", "a"})
    assert summary.title_counts["a"] == 1
    assert summary.venue_counts["a"] == 1
    assert title_overlap(query, summary) == pytest.approx(2.0 / 3.0)
    assert counter_query_overlap(query.venue_terms, summary.venue_counts, summary.size) == pytest.approx(0.5)


def test_title_terms_preserve_identifying_digits() -> None:
    dataset = SimpleNamespace(
        signatures={"q": _signature("pq")},
        papers={"pq": SimpleNamespace(title="Part 1: Co3O4", venue=None, journal_name=None, year=2020)},
        specter_embeddings=None,
    )

    query = extract_query_features(_dataset_arg(dataset), "q")

    assert query.title_terms == frozenset({"part", "1", "co3o4"})


def test_signature_query_author_ignores_raw_full_name_and_uses_canonical_fields() -> None:
    signature = SimpleNamespace(
        author_info_full_name="Ada B. Lovelace, PhD",
        author_info_first="Dr.",
        author_info_middle="B-2",
        author_info_last="Lovelace",
        author_info_suffix="PhD",
    )

    assert query_adapter_module._signature_query_author(signature) == "b lovelace phd"

    signature.author_info_first_normalized_without_apostrophe = ""
    signature.author_info_middle_normalized_without_apostrophe = ""
    signature.author_info_last_normalized = "lovelace"
    signature.author_info_suffix_normalized = "phd"
    assert query_adapter_module._signature_query_author(signature) == "lovelace phd"


def test_signature_coauthor_blocks_uses_precomputed_blocks_without_position() -> None:
    signature = _signature("paper")
    signature.author_info_position = None
    signature.author_info_coauthor_blocks = ["ada", "", None]
    dataset = SimpleNamespace(papers={})

    assert query_adapter_module._signature_coauthor_blocks(  # noqa: SLF001
        signature,
        _dataset_arg(dataset),
    ) == frozenset({"ada"})


def test_signature_coauthor_blocks_uses_explicit_coauthors_without_position() -> None:
    signature = _signature("paper")
    signature.author_info_position = None
    signature.author_info_coauthor_blocks = None
    signature.author_info_coauthors = ["Ada Lovelace", "", None]
    dataset = SimpleNamespace(papers={})

    assert query_adapter_module._signature_coauthor_blocks(  # noqa: SLF001
        signature,
        _dataset_arg(dataset),
    ) == frozenset({"a lovelace"})


def test_signature_coauthor_blocks_tolerates_null_paper_author_position() -> None:
    signature = _signature("paper")
    signature.author_info_coauthor_blocks = None
    signature.author_info_coauthors = None
    dataset = SimpleNamespace(
        papers={
            "paper": SimpleNamespace(
                authors=[
                    SimpleNamespace(position=0, author_name="Alice Query"),
                    SimpleNamespace(position=None, author_name="Grace Hopper"),
                ],
            )
        }
    )

    assert query_adapter_module._signature_coauthor_blocks(  # noqa: SLF001
        signature,
        _dataset_arg(dataset),
    ) == frozenset({"g hopper"})


def test_mask_query_features_keeps_orcid_only_when_enabled() -> None:
    base = build_query_features(
        first="alice",
        middle_initials=frozenset({"q"}),
        orcid="0000-0001",
        has_full_first=True,
        has_middle=True,
        has_coauthors=True,
        has_affiliations=True,
    )

    full_without_orcid = mask_query_features(base, "full", orcid_enabled=False)
    full_with_orcid = mask_query_features(base, "full", orcid_enabled=True)
    initial_with_orcid = mask_query_features(base, "initial_only", orcid_enabled=True)

    assert full_without_orcid.orcid is None
    assert full_with_orcid.orcid == "0000-0001"
    assert full_with_orcid.middle_initials == frozenset({"q"})
    assert initial_with_orcid.first == "a"
    assert initial_with_orcid.orcid == "0000-0001"
    assert initial_with_orcid.middle_initials == frozenset()


def test_query_and_summary_orcids_are_canonicalized_and_empty_values_ignored() -> None:
    dataset = SimpleNamespace(
        signatures={
            "q_blank": SimpleNamespace(**{**_signature("p_q_blank").__dict__, "author_info_orcid": "   "}),
            "q_trim": SimpleNamespace(
                **{**_signature("p_q_trim").__dict__, "author_info_orcid": "ORCID: 000000021825009x"}
            ),
            "seed_blank": SimpleNamespace(**{**_signature("p_seed_blank").__dict__, "author_info_orcid": "   "}),
            "seed_trim": SimpleNamespace(
                **{
                    **_signature("p_seed_trim").__dict__,
                    "author_info_orcid": " https://orcid.org/0000-0002-1825-0097 ",
                }
            ),
        },
        papers={
            "p_q_blank": SimpleNamespace(title="Blank Query", venue=None, journal_name=None, year=2020, authors=[]),
            "p_q_trim": SimpleNamespace(title="Trim Query", venue=None, journal_name=None, year=2020, authors=[]),
            "p_seed_blank": SimpleNamespace(title="Blank Seed", venue=None, journal_name=None, year=2021, authors=[]),
            "p_seed_trim": SimpleNamespace(title="Trim Seed", venue=None, journal_name=None, year=2021, authors=[]),
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    assert (
        extract_query_features(_dataset_arg(dataset), "q_blank", feature_cache=feature_cache, orcid_enabled=True).orcid
        is None
    )
    assert (
        extract_query_features(_dataset_arg(dataset), "q_trim", feature_cache=feature_cache, orcid_enabled=True).orcid
        == "0000-0002-1825-009X"
    )

    summary = build_cluster_summary(
        _dataset_arg(dataset),
        cluster_id="cluster",
        component_key="component",
        signature_ids=("seed_blank", "seed_trim"),
        max_exemplars=4,
        feature_cache=feature_cache,
        orcid_enabled=True,
        block_key="block",
    )
    assert summary.orcid_values == frozenset({"0000-0002-1825-0097"})


def test_orcid_enabled_false_suppresses_populated_orcid_field() -> None:
    # work_plan section-2 #229 invariant: the per-request orcid_enabled flag is
    # authoritative, NOT ORCID field presence. A populated, valid
    # author_info_orcid must be suppressed at both the query-feature and
    # cluster-summary layers when orcid_enabled=False, matching Rust
    # (raw_arrow_features.rs gates the orcid_hash on orcid_enabled). This guards
    # against re-introducing a field-presence-only gate on the Python side,
    # which was the (now-resolved) divergence risk in work_plan item #229.
    dataset = SimpleNamespace(
        signatures={
            "q_trim": SimpleNamespace(
                **{**_signature("p_q_trim").__dict__, "author_info_orcid": "0000-0002-1825-0097"}
            ),
            "seed_trim": SimpleNamespace(
                **{**_signature("p_seed_trim").__dict__, "author_info_orcid": "0000-0002-1825-0097"}
            ),
        },
        papers={
            "p_q_trim": SimpleNamespace(title="Trim Query", venue=None, journal_name=None, year=2020, authors=[]),
            "p_seed_trim": SimpleNamespace(title="Trim Seed", venue=None, journal_name=None, year=2021, authors=[]),
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    # Query feature: ORCID suppressed despite a populated, valid field.
    assert (
        extract_query_features(_dataset_arg(dataset), "q_trim", feature_cache=feature_cache, orcid_enabled=False).orcid
        is None
    )

    # Cluster summary: no ORCID values collected despite a populated seed field.
    summary = build_cluster_summary(
        _dataset_arg(dataset),
        cluster_id="cluster",
        component_key="component",
        signature_ids=("seed_trim",),
        max_exemplars=4,
        feature_cache=feature_cache,
        orcid_enabled=False,
        block_key="block",
    )
    assert summary.orcid_values == frozenset()


def test_query_view_for_features_uses_full_only_for_full_first_name() -> None:
    assert query_view_for_features(build_query_features(first="alice", has_full_first=True)) == "full"
    assert query_view_for_features(build_query_features(first="a", has_full_first=False)) == "initial_only"


def test_build_incremental_linker_inputs_resolves_auto_and_per_query_views(monkeypatch) -> None:
    dataset = build_dummy_dataset("dummy_query_view_resolution")

    def fake_build_retriever(summaries, *, include_exemplars=True):
        del include_exemplars
        return SimpleNamespace(
            retriever=object(),
            summary_by_component={str(summary.component_key): summary for summary in summaries},
        )

    monkeypatch.setattr(query_adapter_module, "build_rust_hybrid_centroid_retriever", fake_build_retriever)

    auto_inputs = query_adapter_module.build_incremental_linker_inputs(
        dataset=_dataset_arg(dataset),
        query_signature_ids=["5", "8"],
        cluster_seeds_require={"3": "seed", "4": "seed"},
        query_view=None,
    )
    assert auto_inputs.query_views == ("full", "full")
    assert auto_inputs.query_view_by_signature_id == {"5": "full", "8": "full"}
    assert auto_inputs.query_by_signature_id["5"].first == "alexander"
    assert auto_inputs.query_by_signature_id["5"].has_full_first is True

    explicit_inputs = query_adapter_module.build_incremental_linker_inputs(
        dataset=_dataset_arg(dataset),
        query_signature_ids=["5", "8"],
        cluster_seeds_require={"3": "seed", "4": "seed"},
        query_view=("full", "initial_only"),
    )
    assert explicit_inputs.query_views == ("full", "initial_only")
    assert explicit_inputs.query_by_signature_id["5"].first == "alexander"
    assert explicit_inputs.query_by_signature_id["8"].first == "a"
    assert explicit_inputs.query_by_signature_id["8"].has_full_first is False


def test_build_incremental_linker_inputs_threads_orcid_enabled_to_queries_and_summaries(monkeypatch) -> None:
    dataset = SimpleNamespace(
        signatures={
            "q": SimpleNamespace(**{**_signature("p_q").__dict__, "author_info_orcid": "ORCID: 000000021825009X"}),
            "seed_a": SimpleNamespace(
                **{**_signature("p_seed_a").__dict__, "author_info_orcid": "https://orcid.org/0000-0002-1825-0097"}
            ),
            "seed_b": _signature("p_seed_b"),
        },
        papers={
            "p_q": SimpleNamespace(title="Query Paper", venue=None, journal_name=None, year=2020, authors=[]),
            "p_seed_a": SimpleNamespace(title="Seed Paper A", venue=None, journal_name=None, year=2020, authors=[]),
            "p_seed_b": SimpleNamespace(title="Seed Paper B", venue=None, journal_name=None, year=2021, authors=[]),
        },
        specter_embeddings=None,
    )

    def fake_build_retriever(summaries, *, include_exemplars=True):
        del include_exemplars
        return SimpleNamespace(
            retriever=object(),
            summary_by_component={str(summary.component_key): summary for summary in summaries},
        )

    monkeypatch.setattr(query_adapter_module, "build_rust_hybrid_centroid_retriever", fake_build_retriever)

    disabled_inputs = query_adapter_module.build_incremental_linker_inputs(
        dataset=_dataset_arg(dataset),
        query_signature_ids=["q"],
        cluster_seeds_require={"seed_a": "seed", "seed_b": "seed"},
        query_view="full",
        orcid_enabled=False,
    )
    assert disabled_inputs.query_by_signature_id["q"].orcid is None
    assert disabled_inputs.summary_by_component["seed"].orcid_values == frozenset()

    enabled_inputs = query_adapter_module.build_incremental_linker_inputs(
        dataset=_dataset_arg(dataset),
        query_signature_ids=["q"],
        cluster_seeds_require={"seed_a": "seed", "seed_b": "seed"},
        query_view="full",
        orcid_enabled=True,
    )
    assert enabled_inputs.query_by_signature_id["q"].orcid == "0000-0002-1825-009X"
    assert enabled_inputs.summary_by_component["seed"].orcid_values == frozenset({"0000-0002-1825-0097"})


def test_cluster_summary_tracks_non_mega_coauthors_separately() -> None:
    small_signature = SimpleNamespace(
        **{
            **_signature("p_small").__dict__,
            "author_info_coauthor_blocks": ["shared coauthor", "small only"],
        }
    )
    mega_signature = SimpleNamespace(
        **{
            **_signature("p_mega").__dict__,
            "author_info_coauthor_blocks": ["shared coauthor", "mega only"],
        }
    )
    dataset = SimpleNamespace(
        signatures={"small": small_signature, "mega": mega_signature},
        papers={
            "p_small": SimpleNamespace(
                title="Small Paper",
                venue=None,
                journal_name=None,
                year=2020,
                authors=[SimpleNamespace(author_name="Alice"), SimpleNamespace(author_name="Bob")],
            ),
            "p_mega": SimpleNamespace(
                title="Mega Paper",
                venue=None,
                journal_name=None,
                year=2021,
                authors=[SimpleNamespace(author_name=f"Author {index}") for index in range(50)],
            ),
        },
        specter_embeddings=None,
    )

    summary = build_cluster_summary(
        _dataset_arg(dataset),
        cluster_id="cluster",
        component_key="component",
        signature_ids=("small", "mega"),
        max_exemplars=4,
        feature_cache={},
        orcid_enabled=False,
        block_key="block",
    )

    assert summary.max_paper_author_count == 50
    assert summary.coauthor_counts["shared coauthor"] == 2
    assert summary.coauthor_counts["mega only"] == 1
    assert summary.non_mega_coauthor_counts["shared coauthor"] == 1
    assert summary.non_mega_coauthor_counts["small only"] == 1
    assert "mega only" not in summary.non_mega_coauthor_counts


def test_raw_paper_evidence_features_use_author_lists_and_local_windows() -> None:
    dataset = SimpleNamespace(
        signatures={
            "q": _signature("pq"),
            "c_match": SimpleNamespace(**{**_signature("pc_match").__dict__, "author_info_position": 1}),
            "c_other": SimpleNamespace(**{**_signature("pc_other").__dict__, "author_info_position": 0}),
        },
        papers={
            "pq": SimpleNamespace(
                title="Shared Collaboration Result",
                venue=None,
                journal_name=None,
                year=2020,
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Bob Jones", position=1),
                    SimpleNamespace(author_name="Carol Lee", position=2),
                ],
            ),
            "pc_match": SimpleNamespace(
                title="Shared Collaboration Result",
                venue=None,
                journal_name=None,
                year=2020,
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Bob Jones", position=1),
                    SimpleNamespace(author_name="Carol Lee", position=2),
                ],
            ),
            "pc_other": SimpleNamespace(
                title="Different Topic",
                venue=None,
                journal_name=None,
                year=2021,
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Dana Kim", position=1),
                ],
            ),
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    query = extract_query_features(_dataset_arg(dataset), "q", feature_cache=feature_cache)
    summary = build_cluster_summary(
        _dataset_arg(dataset),
        cluster_id="cluster",
        component_key="component",
        signature_ids=("c_other", "c_match"),
        max_exemplars=4,
        feature_cache=feature_cache,
        orcid_enabled=False,
        block_key="block",
    )

    features = raw_paper_evidence_features(query, summary)

    assert query.paper_author_names == frozenset({"alice smith", "bob jones", "carol lee"})
    assert query.author_position == 0
    assert query.local10_author_names == frozenset({"bob jones", "carol lee"})
    assert features["paper_author_list_max_jaccard"] == pytest.approx(1.0)
    assert features["paper_author_list_max_containment"] == pytest.approx(1.0)
    assert features["paper_author_list_max_overlap_count"] == pytest.approx(3.0)
    assert features["local_author_window10_jaccard_max"] == pytest.approx(1.0 / 3.0)
    assert features["local_author_window10_overlap_count_max"] == pytest.approx(1.0)
    assert features["best_author_count_log_absdiff"] == pytest.approx(0.0)


def test_local10_evidence_ignores_query_signature_member() -> None:
    dataset = SimpleNamespace(
        signatures={"q": _signature("pq")},
        papers={
            "pq": SimpleNamespace(
                title="Shared Collaboration Result",
                venue=None,
                journal_name=None,
                year=2020,
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Bob Jones", position=1),
                    SimpleNamespace(author_name="Carol Lee", position=2),
                ],
            ),
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    query = extract_query_features(_dataset_arg(dataset), "q", feature_cache=feature_cache)
    summary = build_cluster_summary(
        _dataset_arg(dataset),
        cluster_id="cluster",
        component_key="component",
        signature_ids=("q",),
        max_exemplars=4,
        feature_cache=feature_cache,
        orcid_enabled=False,
        block_key="block",
    )

    features = raw_paper_evidence_features(query, summary)

    assert features["paper_author_list_max_jaccard"] == pytest.approx(1.0)
    assert features["paper_author_list_max_overlap_count"] == pytest.approx(3.0)
    assert features["best_author_count_log_absdiff"] == pytest.approx(0.0)
    assert features["local_author_window10_jaccard_max"] == pytest.approx(0.0)
    assert features["local_author_window10_overlap_count_max"] == pytest.approx(0.0)

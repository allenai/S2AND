from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest

import s2and.incremental_linking.query_adapter as query_adapter_module
from s2and.incremental_linking.query_adapter import (
    build_cluster_summary,
    extract_query_features,
    mask_query_features,
    query_view_for_features,
    raw_paper_evidence_features,
)
from tests.helpers import build_dummy_dataset, build_query_features

_summary = partial(
    build_cluster_summary, cluster_id="cluster", component_key="component", max_exemplars=4, block_key="block"
)


def _paper(title="", *, venue=None, journal_name=None, year=2020, authors=()):
    """Construct only the raw paper fields used by query adaptation."""
    return SimpleNamespace(title=title, venue=venue, journal_name=journal_name, year=year, authors=authors)


def _signature(paper_id: str, **overrides: Any) -> SimpleNamespace:
    signature = SimpleNamespace(
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
    signature.__dict__.update(overrides)
    return signature


def test_title_and_venue_terms_keep_single_character_tokens() -> None:
    dataset = SimpleNamespace(
        signatures={"q": _signature("pq"), "c": _signature("pc")},
        papers={
            "pq": _paper(title="A M Study Part 1 Co3O4", venue="Series A"),
            "pc": _paper(title="A Different Study", venue="A", year=2021),
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    query = extract_query_features(dataset, "q", feature_cache=feature_cache)
    summary = _summary(dataset, signature_ids=("c",), feature_cache=feature_cache)

    assert query.title_terms == frozenset({"a", "m", "study", "part", "1", "co3o4"})
    assert query.venue_terms == frozenset({"series", "a"})
    assert summary.title_counts["a"] == 1
    assert summary.venue_counts["a"] == 1


def test_classic_query_and_summary_match_arrow_year_missingness() -> None:
    for case_id, year, expected in (
        ("none", None, None),
        ("zero", 0, None),
        ("present", 2020, 2020),
    ):
        dataset = SimpleNamespace(
            signatures={"q": _signature("pq"), "seed": _signature("ps")},
            papers={"pq": _paper(title="", year=year, authors=[]), "ps": _paper(title="", year=year, authors=[])},
            specter_embeddings=None,
        )

        query = extract_query_features(dataset, "q")
        summary = _summary(dataset, signature_ids=("seed",), max_exemplars=0, feature_cache={})

        assert query.year == expected, case_id
        assert summary.year_values == ([] if expected is None else [expected]), case_id
        assert summary.year_min == expected, case_id
        assert summary.year_max == expected, case_id
        assert summary.year_mean == expected, case_id


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
        dataset,
    ) == frozenset({"ada"})


def test_signature_coauthor_blocks_uses_explicit_coauthors_without_position() -> None:
    signature = _signature("paper")
    signature.author_info_position = None
    signature.author_info_coauthor_blocks = None
    signature.author_info_coauthors = ["Ada Lovelace", "", None]
    dataset = SimpleNamespace(papers={})

    assert query_adapter_module._signature_coauthor_blocks(  # noqa: SLF001
        signature,
        dataset,
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
        dataset,
    ) == frozenset({"g hopper"})


def test_query_features_count_blank_author_rows_without_modern_name_evidence() -> None:
    signature = _signature("paper")
    signature.author_info_coauthor_blocks = None
    signature.author_info_coauthors = None
    dataset = SimpleNamespace(
        signatures={"q": signature},
        papers={
            "paper": _paper(
                title="Paper",
                authors=[
                    SimpleNamespace(position=0, author_name="Alice Smith"),
                    SimpleNamespace(position=1, author_name=""),
                    SimpleNamespace(position=2, author_name="   "),
                    SimpleNamespace(position=3, author_name="Bob Jones"),
                ],
            )
        },
        specter_embeddings=None,
    )

    features = extract_query_features(dataset, "q")

    assert features.paper_author_count == 4
    assert features.paper_author_names == frozenset({"alice smith", "bob jones"})
    assert features.local10_author_names == frozenset({"bob jones"})
    assert features.coauthor_blocks == frozenset({"b jones"})


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


def test_build_incremental_linker_inputs_resolves_auto_and_per_query_views() -> None:
    dataset = build_dummy_dataset("dummy_query_view_resolution")

    auto_inputs = query_adapter_module.build_incremental_linker_inputs(
        dataset=dataset,
        query_signature_ids=["5", "8"],
        cluster_seeds_require={"3": "seed", "4": "seed"},
        query_view=None,
    )
    assert auto_inputs.query_views == ("full", "full")
    assert auto_inputs.query_view_by_signature_id == {"5": "full", "8": "full"}
    assert auto_inputs.query_by_signature_id["5"].first == "alexander"
    assert auto_inputs.query_by_signature_id["5"].has_full_first is True

    explicit_inputs = query_adapter_module.build_incremental_linker_inputs(
        dataset=dataset,
        query_signature_ids=["5", "8"],
        cluster_seeds_require={"3": "seed", "4": "seed"},
        query_view=("full", "initial_only"),
    )
    assert explicit_inputs.query_views == ("full", "initial_only")
    assert explicit_inputs.query_by_signature_id["5"].first == "alexander"
    assert explicit_inputs.query_by_signature_id["8"].first == "a"
    assert explicit_inputs.query_by_signature_id["8"].has_full_first is False
    assert query_view_for_features(auto_inputs.query_by_signature_id["5"]) == "full"
    assert query_view_for_features(explicit_inputs.query_by_signature_id["8"]) == "initial_only"


def test_build_incremental_linker_inputs_canonicalizes_and_gates_orcids():
    dataset = SimpleNamespace(
        signatures={
            "q": _signature("pq", author_info_orcid="ORCID: 000000021825009x"),
            "blank": _signature("pb", author_info_orcid="   "),
            "seed_a": _signature("ps", author_info_orcid=" https://orcid.org/0000-0002-1825-0097 "),
            "seed_b": _signature("pb", author_info_orcid="   "),
        },
        papers={key: _paper() for key in ("pq", "pb", "ps")},
        specter_embeddings=None,
    )
    for enabled in (False, True):
        inputs = query_adapter_module.build_incremental_linker_inputs(
            dataset=dataset,
            query_signature_ids=["q", "blank"],
            cluster_seeds_require={"seed_a": "seed", "seed_b": "seed"},
            query_view="full",
            orcid_enabled=enabled,
        )
        assert inputs.query_by_signature_id["q"].orcid == ("0000-0002-1825-009X" if enabled else None)
        assert inputs.query_by_signature_id["blank"].orcid is None
        assert inputs.summary_by_component["seed"].orcid_values == (
            frozenset({"0000-0002-1825-0097"}) if enabled else frozenset()
        )


def test_cluster_summary_tracks_non_mega_coauthors_separately() -> None:
    small_signature = _signature("p_small", author_info_coauthor_blocks=["shared coauthor", "small only"])
    mega_signature = _signature("p_mega", author_info_coauthor_blocks=["shared coauthor", "mega only"])
    dataset = SimpleNamespace(
        signatures={"small": small_signature, "mega": mega_signature},
        papers={
            "p_small": _paper(
                title="Small Paper", authors=[SimpleNamespace(author_name="Alice"), SimpleNamespace(author_name="Bob")]
            ),
            "p_mega": _paper(
                title="Mega Paper",
                year=2021,
                authors=[SimpleNamespace(author_name=f"Author {index}") for index in range(50)],
            ),
        },
        specter_embeddings=None,
    )

    summary = _summary(dataset, signature_ids=("small", "mega"), feature_cache={})

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
            "c_match": _signature("pc_match", author_info_position=1),
            "c_other": _signature("pc_other", author_info_position=0),
        },
        papers={
            "pq": _paper(
                title="Shared Collaboration Result",
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Bob Jones", position=1),
                    SimpleNamespace(author_name="Carol Lee", position=2),
                ],
            ),
            "pc_match": _paper(
                title="Shared Collaboration Result",
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Bob Jones", position=1),
                    SimpleNamespace(author_name="Carol Lee", position=2),
                ],
            ),
            "pc_other": _paper(
                title="Different Topic",
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

    query = extract_query_features(dataset, "q", feature_cache=feature_cache)
    summary = _summary(dataset, signature_ids=("c_other", "c_match"), feature_cache=feature_cache)

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
            "pq": _paper(
                title="Shared Collaboration Result",
                authors=[
                    SimpleNamespace(author_name="Alice Smith", position=0),
                    SimpleNamespace(author_name="Bob Jones", position=1),
                    SimpleNamespace(author_name="Carol Lee", position=2),
                ],
            )
        },
        specter_embeddings=None,
    )
    feature_cache = {}

    query = extract_query_features(dataset, "q", feature_cache=feature_cache)
    summary = _summary(dataset, signature_ids=("q",), feature_cache=feature_cache)

    features = raw_paper_evidence_features(query, summary)

    assert features["paper_author_list_max_jaccard"] == pytest.approx(1.0)
    assert features["paper_author_list_max_overlap_count"] == pytest.approx(3.0)
    assert features["best_author_count_log_absdiff"] == pytest.approx(0.0)
    assert features["local_author_window10_jaccard_max"] == pytest.approx(0.0)
    assert features["local_author_window10_overlap_count_max"] == pytest.approx(0.0)

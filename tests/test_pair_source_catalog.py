from __future__ import annotations

import random
from itertools import combinations

import pandas as pd
import pytest

from scripts._pair_ablation.pair_sources import (
    PAIR_COLUMNS,
    PairRecord,
    build_gold_cluster_lookup,
    canonicalize_pairs,
    cap_pairs_per_domain,
    exclude_source_domains,
    load_historical_augmented_pairs,
    load_medline_pairs,
    sample_name_challenge_pairs,
    sample_pair_ranks,
    sample_within_blocks_anchor_uniform,
    sample_within_blocks_balanced,
    sample_within_blocks_uniform,
    unrank_pair,
)


def _record(
    *,
    domain: str,
    pair1: str,
    pair2: str,
    label: int | str,
    origin: str,
    group_id: str = "q1",
) -> PairRecord:
    return PairRecord(
        source_domain=domain,
        source_family="fixture",
        pair1=pair1,
        pair2=pair2,
        label=label,  # type: ignore[arg-type]
        label_rule="fixture_label",
        origin=origin,
        group_id=group_id,
    )


def test_pair_record_has_exact_schema_and_canonical_orientation() -> None:
    frame = canonicalize_pairs([_record(domain="medline", pair1="z", pair2="a", label="YES", origin="fixture:1")])

    assert tuple(frame.columns) == PAIR_COLUMNS
    assert frame.to_dict(orient="records") == [
        {
            "source_domain": "medline",
            "source_family": "fixture",
            "pair1": "a",
            "pair2": "z",
            "label": 1,
            "label_rule": "fixture_label",
            "origin": "fixture:1",
            "group_id": "q1",
        }
    ]
    assert str(frame["label"].dtype) == "int8"


def test_pair_record_rejects_missing_metadata_and_unknown_labels() -> None:
    with pytest.raises(ValueError, match="source_domain must be non-empty"):
        _record(domain=float("nan"), pair1="a", pair2="b", label=1, origin="fixture")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="label must be one of"):
        _record(domain="d", pair1="a", pair2="b", label="MAYBE", origin="fixture")


def test_load_medline_pairs_maps_all_supported_labels(tmp_path) -> None:
    (tmp_path / "train_pairs.csv").write_text(
        "pair1,pair2,label\n" "z,a,YES\n" "b,c,0\n",
        encoding="utf-8",
    )
    (tmp_path / "test_pairs.csv").write_text(
        "pair1,pair2,label\n" "d,e,NO\n" "g,f,1\n",
        encoding="utf-8",
    )

    frame = load_medline_pairs(tmp_path)

    assert frame[["pair1", "pair2", "label"]].to_records(index=False).tolist() == [
        ("a", "z", 1),
        ("b", "c", 0),
        ("d", "e", 0),
        ("f", "g", 1),
    ]
    assert set(frame["source_domain"]) == {"medline"}
    assert set(frame["source_family"]) == {"pairwise_only"}
    assert frame["group_id"].tolist() == [
        "medline:train:0",
        "medline:train:1",
        "medline:test:0",
        "medline:test:1",
    ]


def test_load_historical_augmented_pairs_handles_test_typo_and_strips_prefixes(tmp_path) -> None:
    (tmp_path / "test_pairs.csv").write_text(
        "pairs1,pair2,label\n" "aminer___sig-z,aminer___sig-a,NO\n" "qian___2,qian___1,YES\n",
        encoding="utf-8",
    )

    frame = load_historical_augmented_pairs(tmp_path, splits=("test",))

    assert frame[["source_domain", "pair1", "pair2", "label"]].to_records(index=False).tolist() == [
        ("aminer", "sig-a", "sig-z", 0),
        ("qian", "1", "2", 1),
    ]
    assert not frame["pair1"].str.contains("___", regex=False).any()
    assert not frame["pair2"].str.contains("___", regex=False).any()


def test_augmented_loader_rejects_cross_domain_pair(tmp_path) -> None:
    path = tmp_path / "test_pairs.csv"
    path.write_text(
        "pairs1,pair2,label\n" "aminer___sig-1,qian___sig-2,YES\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different domains"):
        load_historical_augmented_pairs(tmp_path, splits=("test",))


def test_augmented_loader_requires_prefix_on_both_endpoints(tmp_path) -> None:
    path = tmp_path / "test_pairs.csv"
    path.write_text(
        "pairs1,pair2,label\n" "aminer___sig-1,sig-2,YES\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing a dataset___ prefix"):
        load_historical_augmented_pairs(tmp_path, splits=("test",))


def test_canonicalize_pairs_deduplicates_unordered_pairs_and_rejects_conflicts() -> None:
    duplicate = canonicalize_pairs(
        [
            _record(domain="d", pair1="a", pair2="b", label=1, origin="first"),
            _record(domain="d", pair1="b", pair2="a", label="1", origin="second"),
        ]
    )
    assert len(duplicate) == 1
    assert duplicate.iloc[0]["origin"] == "first"

    with pytest.raises(ValueError, match="Conflicting labels.*first.*opposite"):
        canonicalize_pairs(
            [
                _record(domain="d", pair1="a", pair2="b", label=1, origin="first"),
                _record(domain="d", pair1="b", pair2="a", label=0, origin="opposite"),
            ]
        )


def test_fold_exclusion_uses_source_domain_not_source_family() -> None:
    frame = canonicalize_pairs(
        [
            _record(domain="h_wang", pair1="1", pair2="2", label=1, origin="big-block"),
            _record(domain="medline", pair1="1", pair2="2", label=0, origin="fixed"),
            _record(domain="qian", pair1="1", pair2="2", label=1, origin="cluster"),
        ]
    )

    filtered = exclude_source_domains(frame, {"h_wang", "qian"})

    assert filtered["source_domain"].tolist() == ["medline"]
    assert "h_wang" not in set(filtered["source_domain"])


def test_pair_and_query_uniform_domain_caps_are_deterministic() -> None:
    records = []
    for index in range(8):
        records.append(
            _record(
                domain="d1",
                pair1=f"a{index}",
                pair2=f"b{index}",
                label=index % 2,
                origin=f"d1:{index}",
                group_id="large" if index < 6 else f"small-{index}",
            )
        )
    for index in range(4):
        records.append(
            _record(
                domain="d2",
                pair1=f"c{index}",
                pair2=f"d{index}",
                label=index % 2,
                origin=f"d2:{index}",
                group_id=f"query-{index}",
            )
        )
    frame = canonicalize_pairs(records)

    first = cap_pairs_per_domain(frame, 3, random_seed=17, sampling="pair_uniform")
    second = cap_pairs_per_domain(frame.sample(frac=1, random_state=3), 3, random_seed=17, sampling="pair_uniform")
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("source_domain").size().to_dict() == {"d1": 3, "d2": 3}

    query_sample = cap_pairs_per_domain(frame, {"d1": 3, "d2": 2}, random_seed=17, sampling="query_uniform")
    d1_groups = set(query_sample.loc[query_sample["source_domain"] == "d1", "group_id"])
    assert d1_groups == {"large", "small-6", "small-7"}


def test_gold_cluster_lookup_supports_canonical_json_shape_and_rejects_overlap() -> None:
    lookup = build_gold_cluster_lookup(
        {
            "c1": {"cluster_id": "c1", "signature_ids": ["s1", "s2"], "model_version": -1},
            "c2": {"cluster_id": "c2", "signature_ids": ["s3"], "model_version": -1},
        }
    )
    assert lookup == {"s1": "c1", "s2": "c1", "s3": "c2"}

    with pytest.raises(ValueError, match="multiple gold clusters"):
        build_gold_cluster_lookup({"c1": ["s1"], "c2": ["s1"]})


@pytest.mark.parametrize("block_size", range(2, 10))
def test_unrank_pair_matches_nested_loop_enumeration(block_size: int) -> None:
    enumerated = list(combinations(range(block_size), 2))

    assert [unrank_pair(block_size, rank) for rank in range(len(enumerated))] == enumerated


def test_sample_pair_ranks_is_bounded_and_matches_random_sample() -> None:
    assert sample_pair_ranks(5, 100, 42) == random.Random(42).sample(range(5), 5)
    assert sample_pair_ranks(0, 100, 42) == []
    with pytest.raises(ValueError, match="sample_size"):
        sample_pair_ranks(5, -1, 42)


def test_virtual_within_block_sampler_matches_enumerated_random_sample() -> None:
    blocks = {
        "block-b": ["z", "a", "m"],
        "block-a": ["x", "c", "q", "p"],
        "singleton": ["only"],
    }
    clusters = {
        "z": "c1",
        "a": "c1",
        "m": "c2",
        "x": "c3",
        "c": "c4",
        "q": "c3",
        "p": "c5",
        "only": "c6",
    }
    universe = []
    for block_id, signatures in blocks.items():
        for pair1, pair2 in combinations(signatures, 2):
            pair1, pair2 = sorted((pair1, pair2))
            universe.append((block_id, pair1, pair2, int(clusters[pair1] == clusters[pair2])))
    expected = random.Random(91).sample(universe, 7)

    sampled = sample_within_blocks_uniform(
        blocks,
        clusters,
        7,
        random_seed=91,
        source_domain="fixture",
    )
    actual = [
        (group_id, pair1, pair2, int(label))
        for pair1, pair2, label, group_id in sampled[["pair1", "pair2", "label", "group_id"]].itertuples(
            index=False,
            name=None,
        )
    ]

    assert actual == expected
    assert sampled["label_rule"].tolist() == [
        "same_gold_cluster" if label else "different_gold_cluster" for _, _, _, label in expected
    ]


def test_virtual_sampler_rejects_missing_labels_and_duplicate_block_membership() -> None:
    with pytest.raises(ValueError, match="missing a gold cluster"):
        sample_within_blocks_uniform(
            {"b": ["s1", "s2"]},
            {"s1": "c1"},
            1,
            random_seed=1,
            source_domain="d",
        )
    with pytest.raises(ValueError, match="appears more than once"):
        sample_within_blocks_uniform(
            {"b1": ["s1", "s2"], "b2": ["s1", "s3"]},
            {"s1": "c1", "s2": "c1", "s3": "c2"},
            1,
            random_seed=1,
            source_domain="d",
        )


def test_anchor_uniform_sampler_is_deterministic_complete_and_naturally_labeled() -> None:
    blocks = {
        "large": ["s0", "s1", "s2", "s3", "s4"],
        "pair": ["t0", "t1"],
        "singleton": ["u0"],
    }
    clusters = {
        "s0": "a",
        "s1": "a",
        "s2": "b",
        "s3": "c",
        "s4": "c",
        "t0": "d",
        "t1": "e",
        "u0": "f",
    }

    first = sample_within_blocks_anchor_uniform(
        blocks,
        clusters,
        100,
        random_seed=37,
        source_domain="fixture",
    )
    second = sample_within_blocks_anchor_uniform(
        blocks,
        clusters,
        100,
        random_seed=37,
        source_domain="fixture",
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 11
    assert not first.duplicated(["pair1", "pair2"]).any()
    actual_pairs = {
        (str(group_id), str(pair1), str(pair2))
        for group_id, pair1, pair2 in first[["group_id", "pair1", "pair2"]].itertuples(index=False, name=None)
    }
    assert actual_pairs == {
        (block_id, *sorted(pair)) for block_id, signatures in blocks.items() for pair in combinations(signatures, 2)
    }
    for pair1, pair2, label in first[["pair1", "pair2", "label"]].itertuples(index=False, name=None):
        assert int(label) == int(clusters[str(pair1)] == clusters[str(pair2)])


def test_anchor_uniform_sampler_weights_anchors_before_quadratic_block_size() -> None:
    blocks = {"large": [f"l{i}" for i in range(20)]}
    blocks.update({f"small-{i}": [f"a{i}", f"b{i}"] for i in range(10)})
    clusters = {signature: signature for signatures in blocks.values() for signature in signatures}

    small_pair_count = 0
    for seed in range(40):
        sampled = sample_within_blocks_anchor_uniform(
            blocks,
            clusters,
            10,
            random_seed=seed,
            source_domain="fixture",
        )
        assert len(sampled) == 10
        small_pair_count += int((sampled["group_id"] != "large").sum())

    # Small blocks contain only 5% of the pair universe but half the anchors.
    assert small_pair_count >= 100


def test_anchor_uniform_sampler_validates_size_and_handles_empty_universe() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        sample_within_blocks_anchor_uniform(
            {"b": ["s1", "s2"]},
            {"s1": "c", "s2": "c"},
            -1,
            random_seed=1,
            source_domain="fixture",
        )

    empty = sample_within_blocks_anchor_uniform(
        {"singleton": ["s1"]},
        {"s1": "c"},
        10,
        random_seed=1,
        source_domain="fixture",
    )
    assert empty.empty
    assert tuple(empty.columns) == PAIR_COLUMNS


def test_balanced_sampler_fills_available_class_quotas_without_enumerating_pairs() -> None:
    blocks = {
        "b1": ["a1", "b1", "a2", "c1", "b2"],
        "b2": ["d1", "d2", "e1"],
    }
    clusters = {
        "a1": "a",
        "a2": "a",
        "b1": "b",
        "b2": "b",
        "c1": "c",
        "d1": "d",
        "d2": "d",
        "e1": "e",
    }

    first = sample_within_blocks_balanced(
        blocks,
        clusters,
        positive_size=3,
        negative_size=5,
        random_seed=19,
        source_domain="fixture",
    )
    second = sample_within_blocks_balanced(
        blocks,
        clusters,
        positive_size=3,
        negative_size=5,
        random_seed=19,
        source_domain="fixture",
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["label"].value_counts().to_dict() == {0: 5, 1: 3}
    assert not first.duplicated(["pair1", "pair2"]).any()
    for pair1, pair2, label in first[["pair1", "pair2", "label"]].itertuples(index=False, name=None):
        assert int(label) == int(clusters[str(pair1)] == clusters[str(pair2)])


def test_balanced_sampler_returns_every_available_pair_when_quotas_exceed_universe() -> None:
    blocks = {"b": ["a1", "b1", "a2", "c1", "b2"]}
    clusters = {"a1": "a", "a2": "a", "b1": "b", "b2": "b", "c1": "c"}

    sampled = sample_within_blocks_balanced(
        blocks,
        clusters,
        positive_size=100,
        negative_size=100,
        random_seed=7,
        source_domain="fixture",
    )

    expected = {(*sorted(pair), int(clusters[pair[0]] == clusters[pair[1]])) for pair in combinations(blocks["b"], 2)}
    actual = {
        (str(pair1), str(pair2), int(label))
        for pair1, pair2, label in sampled[["pair1", "pair2", "label"]].itertuples(index=False, name=None)
    }
    assert actual == expected
    assert sampled["label"].value_counts().to_dict() == {0: 8, 1: 2}


def test_balanced_sampler_validates_class_quotas() -> None:
    with pytest.raises(ValueError, match="positive_size and negative_size"):
        sample_within_blocks_balanced(
            {"b": ["s1", "s2"]},
            {"s1": "c", "s2": "c"},
            positive_size=-1,
            negative_size=0,
            random_seed=1,
            source_domain="fixture",
        )


def test_name_challenge_sampler_emits_only_requested_hard_strata() -> None:
    blocks = {"smith": ["s1", "s2", "s3", "s4", "s5", "s6"]}
    clusters = {"s1": "a", "s2": "a", "s3": "b", "s4": "b", "s5": "c", "s6": "d"}
    names = {
        "s1": "Alice Smith",
        "s2": "A. Smith",
        "s3": "Alice Smith",
        "s4": "Alice Smith",
        "s5": "Alice Smith",
        "s6": "Bob Smith",
    }

    first = sample_name_challenge_pairs(
        blocks,
        clusters,
        names,
        positive_size=4,
        negative_size=4,
        random_seed=11,
        source_domain="fixture",
    )
    second = sample_name_challenge_pairs(
        blocks,
        clusters,
        names,
        positive_size=4,
        negative_size=4,
        random_seed=11,
        source_domain="fixture",
    )
    pd.testing.assert_frame_equal(first, second)

    normalized_names = {signature: name.casefold() for signature, name in names.items()}
    for pair1, pair2, label, label_rule in first[["pair1", "pair2", "label", "label_rule"]].itertuples(
        index=False,
        name=None,
    ):
        if label == 1:
            assert clusters[pair1] == clusters[pair2]
            assert normalized_names[pair1] != normalized_names[pair2]
            assert label_rule == "different_name_same_gold_cluster"
        else:
            assert clusters[pair1] != clusters[pair2]
            assert normalized_names[pair1] == normalized_names[pair2]
            assert label_rule == "same_name_different_gold_cluster"

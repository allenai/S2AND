"""Direct regression coverage for applying incremental seed-link decisions."""

from copy import deepcopy

import pytest

from s2and.incremental_linking.seed_assignment import apply_seed_links


def test_assignment_preserves_order_and_coalesces_output_cluster_ids():
    result = apply_seed_links(
        unassigned_signature_ids=["q2", "abstain", "q1", "new"],
        linked_signature_to_cluster={"q1": 10, "q2": "1", "new": 9},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["s2", "s1"], "1": ["s3"], 2: ["s4"], 3: []},
        prevent_new_incompatibilities=False,
        first_names={},
        name_tuples=set(),
    )

    assert result.clusters == {"1": ["s2", "s1", "s3", "q2", "q1"], "2": ["s4"], "9": ["new"]}
    assert list(result.clusters) == ["1", "2", "9"]
    assert result.residual_signature_ids == ["abstain"]
    assert result.rejected_signature_ids == []


def test_incompatible_links_are_distinguished_from_abstentions():
    result = apply_seed_links(
        unassigned_signature_ids=["absent", "bad2", "good", "bad1"],
        linked_signature_to_cluster={"bad1": 10, "good": 10, "bad2": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["seed"]},
        prevent_new_incompatibilities=True,
        first_names={"seed": "alice", "good": "ali", "bad1": "bob", "bad2": "charles"},
        name_tuples=set(),
    )

    assert result.clusters == {"1": ["seed", "good"]}
    assert result.residual_signature_ids == ["absent", "bad2", "bad1"]
    assert result.rejected_signature_ids == ["bad2", "bad1"]


def test_forced_link_skips_the_final_name_check():
    # A require constraint already decided the forced link; the name check must not undo it.
    inputs = dict(
        unassigned_signature_ids=["forced", "free"],
        linked_signature_to_cluster={"forced": 10, "free": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["seed"]},
        prevent_new_incompatibilities=True,
        first_names={"seed": "alice", "forced": "bob", "free": "bob"},
        name_tuples=set(),
    )
    result = apply_seed_links(**inputs, forced_signature_ids={"forced"})

    assert result.clusters == {"1": ["seed", "forced"]}
    assert result.residual_signature_ids == ["free"]
    assert result.rejected_signature_ids == ["free"]


@pytest.mark.parametrize("query_name", ["rob", "robert", "r", "", "bob"])
def test_prefix_alias_and_unknown_names_are_compatible(query_name):
    result = apply_seed_links(
        unassigned_signature_ids=["query"],
        linked_signature_to_cluster={"query": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["seed"]},
        prevent_new_incompatibilities=True,
        first_names={"seed": "robert", "query": query_name},
        name_tuples={("bob", "robert")},
    )

    assert result.clusters == {"1": ["seed", "query"]}
    assert result.residual_signature_ids == []
    assert result.rejected_signature_ids == []


def test_any_full_name_in_compatibility_group_can_match():
    result = apply_seed_links(
        unassigned_signature_ids=["query"],
        linked_signature_to_cluster={"query": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["alice", "bob", "initial"]},
        prevent_new_incompatibilities=True,
        first_names={"alice": "alice", "bob": "bob", "initial": "z", "query": "bob"},
        name_tuples=set(),
    )

    assert result.clusters == {"1": ["alice", "bob", "initial", "query"]}
    assert result.residual_signature_ids == []


@pytest.mark.parametrize(
    ("split_seeds", "accepted"),
    [({10: ["split_seed"]}, True), ({10: []}, True), ({99: ["split_seed"]}, False), (None, False)],
)
def test_split_membership_takes_precedence_including_empty_group(split_seeds, accepted):
    result = apply_seed_links(
        unassigned_signature_ids=["query"],
        linked_signature_to_cluster={"query": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["original_seed"]},
        split_cluster_seeds_require_inverse=split_seeds,
        prevent_new_incompatibilities=True,
        first_names={"original_seed": "alice", "split_seed": "bob", "query": "bob"},
        name_tuples=set(),
    )

    assert result.clusters == {"1": ["original_seed", "query"] if accepted else ["original_seed"]}
    assert result.residual_signature_ids == ([] if accepted else ["query"])
    assert result.rejected_signature_ids == ([] if accepted else ["query"])


@pytest.mark.parametrize("seed_name", ["r", ""])
def test_initial_only_or_unknown_seed_group_does_not_read_query_name(seed_name):
    result = apply_seed_links(
        unassigned_signature_ids=["query"],
        linked_signature_to_cluster={"query": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["seed"]},
        prevent_new_incompatibilities=True,
        first_names={"seed": seed_name},
        name_tuples=set(),
    )

    assert result.clusters == {"1": ["seed", "query"]}
    assert result.residual_signature_ids == []


def test_unremapped_links_do_not_read_names_even_with_other_remappings():
    result = apply_seed_links(
        unassigned_signature_ids=["query", "abstain"],
        linked_signature_to_cluster={"query": 1},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["seed"]},
        prevent_new_incompatibilities=True,
        first_names={},
        name_tuples=set(),
    )

    assert result.clusters == {"1": ["seed", "query"]}
    assert result.residual_signature_ids == ["abstain"]
    assert result.rejected_signature_ids == []


def test_assignment_does_not_mutate_or_alias_inputs():
    inputs = {
        "unassigned_signature_ids": ["good", "bad", "abstain"],
        "linked_signature_to_cluster": {"good": 10, "bad": 10},
        "recluster_map": {10: 1},
        "cluster_seeds_require_inverse": {1: ["seed"]},
        "split_cluster_seeds_require_inverse": {10: ["seed"]},
        "prevent_new_incompatibilities": True,
        "first_names": {"seed": "alice", "good": "alice", "bad": "bob"},
        "name_tuples": set(),
    }
    original = deepcopy(inputs)

    result = apply_seed_links(**inputs)

    assert inputs == original
    result.clusters["1"].append("output_only")
    result.residual_signature_ids.append("output_only")
    result.rejected_signature_ids.append("output_only")
    assert inputs == original
    assert result.residual_signature_ids == ["bad", "abstain", "output_only"]
    assert result.rejected_signature_ids == ["bad", "output_only"]


@pytest.mark.parametrize(
    ("query_first", "query_cjk", "accepted"),
    [
        # 重郷 shares the prefix "zhong" with 重喜 but is a different whole CJK name.
        ("zhong", "Hzhongxiang", False),
        ("zhong", "Hzhongxi", True),
        # Hangul and Han forms compare by the first-name rule.
        ("zhong", "Gzhongxiang", True),
    ],
)
def test_restored_profile_name_check_applies_the_cjk_rule(query_first, query_cjk, accepted):
    result = apply_seed_links(
        unassigned_signature_ids=["query"],
        linked_signature_to_cluster={"query": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["seed"]},
        prevent_new_incompatibilities=True,
        first_names={"seed": "zhong", "query": query_first},
        name_tuples=set(),
        cjk_given_names={"seed": "Hzhongxi", "query": query_cjk},
    )

    assert (result.clusters == {"1": ["seed", "query"]}) is accepted


@pytest.mark.parametrize(
    ("seed", "query", "accepted"),
    [
        # A joined-initials reading that spells the other name exactly, on either side.
        (("Kristan", "Allan"), ("Ka", ""), True),
        (("Ka", ""), ("Kristan", "Allan"), True),
        (("Marco Aurelio Lisboa", ""), ("Mal", ""), True),
        (("Mohammad", "Arfan"), ("MArfan", ""), True),
        # Other names keep the first-name rule.
        (("Kristian", ""), ("Ka", ""), False),
    ],
)
def test_restored_profile_name_check_applies_the_joined_initials_reading(seed, query, accepted):
    from s2and.text import canonicalize_name_parts, given_name_for_constraint

    names = {"seed": seed, "query": query}
    parts = {key: canonicalize_name_parts(first, middle, "Leite") for key, (first, middle) in names.items()}
    result = apply_seed_links(
        unassigned_signature_ids=["query"],
        linked_signature_to_cluster={"query": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["seed"]},
        prevent_new_incompatibilities=True,
        first_names={key: value.first for key, value in parts.items()},
        name_tuples=set(),
        cjk_given_names={key: given_name_for_constraint(*value) for key, value in names.items()},
        name_parts=parts.__getitem__,
    )

    assert (result.clusters == {"1": ["seed", "query"]}) is accepted


def test_restored_profile_name_check_keeps_one_letter_cjk_first_syllables():
    # 阿明 has canonical first "a"; the CJK rule still vetoes 阿強.
    result = apply_seed_links(
        unassigned_signature_ids=["query"],
        linked_signature_to_cluster={"query": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["seed"]},
        prevent_new_incompatibilities=True,
        first_names={"seed": "a", "query": "a"},
        name_tuples=set(),
        cjk_given_names={"seed": "Haming", "query": "Haqiang"},
    )

    assert result.rejected_signature_ids == ["query"]

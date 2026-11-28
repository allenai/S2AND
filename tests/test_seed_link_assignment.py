"""Direct regression coverage for applying incremental seed-link decisions."""

from copy import deepcopy

from s2and.incremental_linking.seed_assignment import apply_seed_links
from s2and.text import canonicalize_name_parts, given_name_for_constraint


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
    assert result.residual_signature_ids == ["abstain"]
    assert result.rejected_signature_ids == []


def test_assignment_keeps_forced_and_unremapped_links_and_owns_results():
    queries = ["absent", "bad2", "forced", "unremapped", "bad1"]
    links = {"bad1": 10, "forced": 10, "bad2": 10, "unremapped": 1}
    restored: dict[int | str, int | str] = {10: 1}
    seed_members: dict[int | str, list[str]] = {1: ["seed"]}
    first_names = {"seed": "alice", "forced": "bob", "bad1": "bob", "bad2": "charles"}
    forced = {"forced"}
    inputs = (queries, links, restored, seed_members, first_names, forced)
    original = deepcopy(inputs)

    result = apply_seed_links(
        unassigned_signature_ids=queries,
        linked_signature_to_cluster=links,
        recluster_map=restored,
        cluster_seeds_require_inverse=seed_members,
        prevent_new_incompatibilities=True,
        first_names=first_names,
        name_tuples=set(),
        forced_signature_ids=forced,
    )

    assert inputs == original
    assert result.clusters == {"1": ["seed", "forced", "unremapped"]}
    assert result.residual_signature_ids == ["absent", "bad2", "bad1"]
    assert result.rejected_signature_ids == ["bad2", "bad1"]
    result.clusters["1"].append("output_only")
    result.residual_signature_ids.append("output_only")
    result.rejected_signature_ids.append("output_only")
    assert inputs == original


def test_prefix_alias_initial_and_unknown_query_names_are_compatible():
    queries = {"prefix": "rob", "initial": "r", "unknown": "", "alias": "bob"}
    result = apply_seed_links(
        unassigned_signature_ids=list(queries),
        linked_signature_to_cluster=dict.fromkeys(queries, 10),
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["alice", "robert", "initial_seed"]},
        prevent_new_incompatibilities=True,
        first_names={"alice": "alice", "robert": "robert", "initial_seed": "z", **queries},
        name_tuples={("bob", "robert")},
    )

    assert result.clusters == {"1": ["alice", "robert", "initial_seed", *queries]}
    assert result.residual_signature_ids == []
    assert result.rejected_signature_ids == []


def test_split_membership_takes_precedence_including_empty_group():
    result = apply_seed_links(
        unassigned_signature_ids=["split", "empty", "fallback"],
        linked_signature_to_cluster={"split": 10, "empty": 11, "fallback": 12},
        recluster_map={10: 1, 11: 1, 12: 1},
        cluster_seeds_require_inverse={1: ["original_seed"]},
        split_cluster_seeds_require_inverse={10: ["split_seed"], 11: []},
        prevent_new_incompatibilities=True,
        first_names={"original_seed": "alice", "split_seed": "bob", "split": "bob", "fallback": "bob"},
        name_tuples=set(),
    )

    assert result.clusters == {"1": ["original_seed", "split", "empty"]}
    assert result.residual_signature_ids == ["fallback"]
    assert result.rejected_signature_ids == ["fallback"]


def test_initial_only_seed_group_does_not_read_query_name():
    result = apply_seed_links(
        unassigned_signature_ids=["query"],
        linked_signature_to_cluster={"query": 10},
        recluster_map={10: 1},
        cluster_seeds_require_inverse={1: ["seed"]},
        prevent_new_incompatibilities=True,
        first_names={"seed": "r"},
        name_tuples=set(),
    )

    assert result.clusters == {"1": ["seed", "query"]}
    assert result.residual_signature_ids == []


def test_restored_profile_name_check_applies_the_cjk_rule():
    queries = {
        "different_han": "Hzhongxiang",
        "same_han": "Hzhongxi",
        "hangul": "Gzhongxiang",
        "different_one_letter": "Haqiang",
    }
    result = apply_seed_links(
        unassigned_signature_ids=list(queries),
        linked_signature_to_cluster={"different_han": 10, "same_han": 10, "hangul": 10, "different_one_letter": 20},
        recluster_map={10: 1, 20: 2},
        cluster_seeds_require_inverse={1: ["seed"], 2: ["one_letter_seed"]},
        prevent_new_incompatibilities=True,
        first_names={
            "seed": "zhong",
            "one_letter_seed": "a",
            "different_han": "zhong",
            "same_han": "zhong",
            "hangul": "zhong",
            "different_one_letter": "a",
        },
        name_tuples=set(),
        cjk_given_names={"seed": "Hzhongxi", "one_letter_seed": "Haming", **queries},
    )

    assert result.clusters == {"1": ["seed", "same_han", "hangul"], "2": ["one_letter_seed"]}
    assert result.rejected_signature_ids == ["different_han", "different_one_letter"]


def test_restored_profile_name_check_applies_the_joined_initials_reading():
    names = {
        "full_seed": ("Kristan", "Allan"),
        "short_seed": ("Ka", ""),
        "expansion": ("Kristan", "Allan"),
        "joined": ("Ka", ""),
        "incompatible": ("Kristian", ""),
        "three_initials_seed": ("Marco Aurelio Lisboa", ""),
        "middle_initials_seed": ("Jip", ""),
        "full_middle_seed": ("Mohammad", "Arfan"),
        "three_initials": ("Mal", ""),
        "middle_initials": ("Johanna", "IP"),
        "full_middle": ("MArfan", ""),
    }
    parts = {key: canonicalize_name_parts(first, middle, "Leite") for key, (first, middle) in names.items()}
    result = apply_seed_links(
        unassigned_signature_ids=[
            "expansion",
            "joined",
            "incompatible",
            "three_initials",
            "middle_initials",
            "full_middle",
        ],
        linked_signature_to_cluster={
            "expansion": 10,
            "joined": 20,
            "incompatible": 10,
            "three_initials": 30,
            "middle_initials": 40,
            "full_middle": 50,
        },
        recluster_map={10: 1, 20: 2, 30: 3, 40: 4, 50: 5},
        cluster_seeds_require_inverse={
            1: ["short_seed"],
            2: ["full_seed"],
            3: ["three_initials_seed"],
            4: ["middle_initials_seed"],
            5: ["full_middle_seed"],
        },
        prevent_new_incompatibilities=True,
        first_names={key: value.first for key, value in parts.items()},
        name_tuples=set(),
        cjk_given_names={key: given_name_for_constraint(*value) for key, value in names.items()},
        name_parts=parts.__getitem__,
    )

    assert result.clusters == {
        "1": ["short_seed", "expansion"],
        "2": ["full_seed", "joined"],
        "3": ["three_initials_seed", "three_initials"],
        "4": ["middle_initials_seed", "middle_initials"],
        "5": ["full_middle_seed", "full_middle"],
    }
    assert result.rejected_signature_ids == ["incompatible"]

from __future__ import annotations

import pytest

from scripts.production.data.assign_block_roles import (
    LEGACY_DEV_TEST_BLOCKS,
    PERMANENT_EXCLUDED_BLOCKS,
    BlockSpec,
    assign_block_roles,
    verify_role_assignments,
)


def _sample_blocks() -> list[BlockSpec]:
    blocks = [
        BlockSpec("f_matsen", 1200, frozenset({"o_matsen"}), "small"),
        BlockSpec("a_khan", 25000, frozenset({"o_khan_1", "o_shared_dev"}), "large"),
        # Shares o_shared_dev with a_khan -> must follow a_khan into dev_test!
        BlockSpec("b_linked_to_khan", 5000, frozenset({"o_shared_dev", "o_b1"}), "medium"),
        # Pair of non-dev blocks sharing o_cross -> must land in the same pool!
        BlockSpec("c_pair_1", 4000, frozenset({"o_cross", "o_c1"}), "medium"),
        BlockSpec("c_pair_2", 3500, frozenset({"o_cross", "o_c2"}), "medium"),
    ]
    for i in range(18):
        stratum = ("large", "medium", "small")[i % 3]
        blocks.append(BlockSpec(f"blk_{i:02d}", 1000 + i * 500, frozenset({f"o_blk_{i}"}), stratum))
    return blocks


def test_assign_block_roles_closes_orcids_excludes_f_matsen_and_pins_dev_test() -> None:
    blocks = _sample_blocks()
    targets = {
        "final_test": 4,
        "eps_cal": 4,
        "gate_thresh": 2,
        "gate_fit": 2,
        "linker_train": 3,
        "pairwise_val": 2,
        "pairwise_train": 3,
    }
    res1 = assign_block_roles(blocks, targets, seed=42, extra_dev_test_blocks={"blk_00"})
    res2 = assign_block_roles(blocks, targets, seed=42, extra_dev_test_blocks={"blk_00"})
    assert res1 == res2

    mapping = res1["block_to_role"]
    for excluded in PERMANENT_EXCLUDED_BLOCKS:
        assert excluded not in mapping
    assert res1["excluded_blocks"] == ["f_matsen"]

    # Legacy giant and its ORCID-linked neighbor must both be in dev_test.
    assert mapping["a_khan"] == "dev_test"
    assert mapping["b_linked_to_khan"] == "dev_test"
    assert mapping["blk_00"] == "dev_test"
    # Cross-block ORCID pair must receive the exact same role.
    assert mapping["c_pair_1"] == mapping["c_pair_2"]
    assert res1["verification"]["cross_role_orcid_overlap"] == 0
    assert ["c_pair_1", "c_pair_2"] in res1["cross_block_components"]


def test_assign_block_roles_rejects_conflicting_pins_across_shared_orcid() -> None:
    blocks = [
        BlockSpec("a_khan", 10000, frozenset({"o_shared"})),
        BlockSpec("new_block", 5000, frozenset({"o_shared"})),
    ]
    with pytest.raises(ValueError, match="conflicting pinned roles"):
        assign_block_roles(
            blocks,
            {"final_test": 1},
            pinned_roles={"new_block": "final_test"},
        )


def test_verify_role_assignments_catches_orcid_leak_and_legacy_violation() -> None:
    legacy = sorted(LEGACY_DEV_TEST_BLOCKS)[0]
    blocks = [
        BlockSpec("x1", 100, frozenset({"o1"})),
        BlockSpec("x2", 100, frozenset({"o1"})),
        BlockSpec(legacy, 100, frozenset({"o2"})),
    ]
    with pytest.raises(ValueError, match="ORCID overlap"):
        verify_role_assignments(blocks, {"x1": "pairwise_train", "x2": "final_test", legacy: "dev_test"})
    with pytest.raises(ValueError, match="must be assigned to 'dev_test'"):
        verify_role_assignments(blocks, {"x1": "pairwise_train", "x2": "pairwise_train", legacy: "final_test"})

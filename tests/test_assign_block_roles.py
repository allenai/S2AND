from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.production.data.assign_block_roles import (
    LEGACY_DEV_TEST_BLOCKS,
    PERMANENT_EXCLUDED_BLOCKS,
    BlockSpec,
    assign_block_roles,
    main,
    verify_role_assignments,
)


def _sample_blocks() -> list[BlockSpec]:
    valid_a = "0000-0002-1694-233X"
    valid_b = "0000-0001-5000-0007"
    excluded_shared = "0000-0002-1825-0097"  # In EXCLUDED_ORCID_GROUPS -> must NOT link blocks!
    blocks = [
        # Real space-separated gold keys!
        BlockSpec("f matsen", 1200, frozenset({valid_a}), "small", "holdout"),
        BlockSpec("a khan", 25000, frozenset({valid_a, excluded_shared}), "large", "development"),
        BlockSpec("s feldman", 8000, frozenset(), "medium", "development"),
        # Shares valid_a with a khan (via HTTPS URL form) -> must follow a khan into dev_test!
        BlockSpec("b linked", 5000, frozenset({f"https://orcid.org/{valid_a}"}), "medium", "development"),
        # Pair of non-dev blocks sharing valid_b -> must land in the same pool!
        BlockSpec("c pair 1", 4000, frozenset({valid_b, excluded_shared}), "medium", "development"),
        BlockSpec("c pair 2", 3500, frozenset({valid_b}), "medium", "development"),
    ]
    for i in range(20):
        stratum = ("large", "medium", "small")[i % 3]
        split = "holdout" if i < 6 else "development"
        blocks.append(BlockSpec(f"blk {i:02d}", 1000 + i * 500, frozenset(), stratum, split))
    return blocks


def test_assign_block_roles_normalizes_keys_excludes_f_matsen_and_handles_overflow() -> None:
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
    res1 = assign_block_roles(
        blocks,
        targets,
        seed=42,
        extra_dev_test_blocks={"blk_06"},
        enforce_source_splits=True,
    )
    res2 = assign_block_roles(
        list(reversed(blocks)),
        targets,
        seed=42,
        extra_dev_test_blocks={"blk_06"},
        enforce_source_splits=True,
    )
    assert res1 == res2

    mapping = res1["block_to_role"]
    for excluded in PERMANENT_EXCLUDED_BLOCKS:
        assert excluded not in mapping
    assert res1["excluded_blocks"] == ["f matsen"]

    # Legacy giants (space-separated) including s feldman and ORCID-linked b linked all go to dev_test.
    assert mapping["a khan"] == "dev_test"
    assert mapping["s feldman"] == "dev_test"
    assert mapping["b linked"] == "dev_test"
    assert mapping["blk 06"] == "dev_test"
    # Cross-block ORCID pair must receive the exact same role (and NOT be dragged to dev_test by excluded_shared!).
    assert mapping["c pair 1"] == mapping["c pair 2"]
    assert mapping["c pair 1"] != "dev_test"
    # final_test receives ONLY holdout blocks and does NOT absorb overflow.
    assert [len(res1["roles"][role]) for role in ("final_test_a", "final_test_b")] == [2, 2]
    assert len(res1["unassigned"]) > 0
    assert res1["verification"]["cross_role_orcid_overlap"] == 0


def test_assign_block_roles_balances_strata_without_aliasing() -> None:
    # Regression test (Astra #13 & Fable #2): 4 A-stratum + 4 B-stratum blocks across 2 equal roles
    # must give 2 A + 2 B to each role (never 4 A to one and 4 B to the other!).
    blocks = [
        *[BlockSpec(f"a_{i}", 1000, frozenset(), "A") for i in range(4)],
        *[BlockSpec(f"b_{i}", 1000, frozenset(), "B") for i in range(4)],
    ]
    res = assign_block_roles(blocks, {"eps_cal": 4, "gate_fit": 4}, seed=7)
    for role in ("eps_cal", "gate_fit"):
        assigned = res["roles"][role]
        assert len(assigned) == 4
        assert sum(k.startswith("a ") for k in assigned) == 2
        assert sum(k.startswith("b ") for k in assigned) == 2


def test_home_pool_mode_avoids_mega_component_percolation() -> None:
    shared_orcid = "0000-0002-1694-233X"
    blocks = [
        BlockSpec("a khan", 20000, frozenset({shared_orcid}), "large"),
        BlockSpec("blk 1", 10000, frozenset({shared_orcid}), "large"),
        BlockSpec("blk 2", 10000, frozenset({shared_orcid}), "large"),
    ]
    res = assign_block_roles(blocks, {"eps_cal": 2}, seed=1, closure_mode="home_pool")
    assert res["block_to_role"]["a khan"] == "dev_test"
    assert res["block_to_role"]["blk 1"] == "eps_cal"
    assert res["block_to_role"]["blk 2"] == "eps_cal"
    assert shared_orcid in res["orcid_home_role"]
    assert res["verification"]["raw_cross_role_orcid_overlap"] == 1
    assert res["verification"]["cross_role_orcid_overlap"] == 0


def test_assign_block_roles_rejects_conflicting_pins_and_verify_catches_leaks() -> None:
    shared = "0000-0002-1694-233X"
    blocks = [
        BlockSpec("a khan", 10000, frozenset({shared})),
        BlockSpec("new block", 5000, frozenset({shared})),
    ]
    with pytest.raises(ValueError, match="conflicting pinned roles"):
        assign_block_roles(blocks, {"final_test": 1}, pinned_roles={"new_block": "final_test_a"})

    legacy = sorted(LEGACY_DEV_TEST_BLOCKS)[0]
    verify_blocks = [
        BlockSpec("x1", 100, frozenset({shared})),
        BlockSpec("x2", 100, frozenset({shared})),
        BlockSpec(legacy, 100, frozenset()),
    ]
    with pytest.raises(ValueError, match="ORCID overlap"):
        verify_role_assignments(verify_blocks, {"x1": "pairwise_train", "x2": "final_test_a", legacy: "dev_test"})
    with pytest.raises(ValueError, match="must be assigned to 'dev_test'"):
        verify_role_assignments(verify_blocks, {"x1": "pairwise_train", "x2": "pairwise_train", legacy: "final_test_a"})


def test_stratified_holdout_reserves_rare_giants_across_both_halves() -> None:
    blocks = [
        *[BlockSpec(f"small_{i}", 200, stratum="a_small", source_split="holdout") for i in range(1000)],
        *[BlockSpec(f"giant_{i}", 20_000, stratum="z_giant", source_split="holdout") for i in range(40)],
    ]
    result = assign_block_roles(blocks, {"final_test": 20}, enforce_source_splits=True)
    counts = result["verification"]["role_block_counts_by_stratum"]
    assert sum(counts[role]["z_giant"] for role in ("final_test_a", "final_test_b")) == 1
    assert [len(result["roles"][role]) for role in ("final_test_a", "final_test_b")] == [10, 10]
    assert len(result["unassigned"]) == 1020


def test_assign_roles_parquet_strata_and_required_source_split(tmp_path: Path) -> None:
    blocks = tmp_path / "blocks.parquet"
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"eps_cal": 2, "final_test": 2}), encoding="utf-8")
    records = [
        {
            "block_key": f"a_{i}",
            "signature_count": 16000,
            "orcids": [],
            "script": "Latn",
            "region": "Iberian",
            "split": split,
        }
        for i, split in enumerate(("development", "development", "holdout", "holdout"))
    ]
    pq.write_table(pa.Table.from_pylist(records), blocks)
    args = [
        "--blocks",
        str(blocks),
        "--targets-json",
        str(targets),
        "--output-dir",
        str(tmp_path / "out"),
        "--closure-mode",
        "home_pool",
        "--enforce-source-splits",
    ]
    assert main(args) == 0
    result = json.loads((tmp_path / "out" / "assignments.json").read_text(encoding="utf-8"))
    assert result["roles"]["eps_cal"] == ["a 0", "a 1"]
    assert result["verification"]["role_block_counts_by_stratum"]["final_test_a"] == {">15000/Latn/Iberian": 1}

    records[-1]["split"] = None
    pq.write_table(pa.Table.from_pylist(records), blocks)
    with pytest.raises(ValueError, match="missing or invalid source_split"):
        main([*args, "--overwrite"])

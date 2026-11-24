from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.common.block_roles import load_block_roles
from scripts.intake.assign_block_roles import (
    LEGACY_DEV_TEST_BLOCKS,
    PERMANENT_EXCLUDED_BLOCKS,
    BlockSpec,
    assign_block_roles,
    load_blocks,
    main,
    verify_role_assignments,
)

pytestmark = pytest.mark.tooling


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
        split = "holdout" if i < 8 else "development"
        blocks.append(BlockSpec(f"blk {i:02d}", 1000 + i * 500, frozenset(), stratum, split))
    return blocks


def test_assign_block_roles_normalizes_keys_excludes_f_matsen_and_handles_overflow() -> None:
    blocks = _sample_blocks()
    targets = {
        "final_test": 6,
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
        extra_dev_test_blocks={"blk_08"},
        enforce_source_splits=True,
    )
    res2 = assign_block_roles(
        list(reversed(blocks)),
        targets,
        seed=42,
        extra_dev_test_blocks={"blk_08"},
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
    assert mapping["blk 08"] == "dev_test"
    # Cross-block ORCID pair must receive the exact same role (and NOT be dragged to dev_test by excluded_shared!).
    assert mapping["c pair 1"] == mapping["c pair 2"]
    assert mapping["c pair 1"] != "dev_test"
    # final_test receives ONLY holdout blocks and does NOT absorb overflow.
    assert [len(res1["roles"][role]) for role in ("final_test_a", "final_test_b")] == [3, 3]
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
    for role in ("final_test_a", "final_test_b"):
        assert counts[role]["z_giant"] >= 1
        assert counts[role]["a_small"] >= 1
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
        main([*args, "--overwrite", "--pinned-roles-json", str(tmp_path / "out" / "assignments.json")])


def test_free_development_role_cannot_consume_holdout() -> None:
    result = assign_block_roles(
        [BlockSpec(key, 20, source_split="holdout") for key in ("red", "blue", "green")],
        {"final_test": 2, "dev_test": 1},
        enforce_source_splits=True,
        closure_mode="home_pool",
    )
    assert result["roles"]["dev_test"] == []
    assert result["shortfalls"] == {"dev_test": 1}


def test_home_role_prefers_evaluation_and_preserves_pinned_homes(tmp_path: Path) -> None:
    shared, tie = "0000-0002-1694-233X", "0000-0002-9079-593X"
    blocks = [
        BlockSpec("t giant", 30000, frozenset({shared})),
        BlockSpec("p one", 50, frozenset({shared})),
        BlockSpec("p two", 50, frozenset({shared})),
        BlockSpec("d fresh", 9000, frozenset({tie})),
        BlockSpec("p three", 50, frozenset({tie})),
    ]
    pins = {
        "t giant": "final_test_a",
        "p one": "pairwise_train",
        "p two": "pairwise_train",
        "d fresh": "dev_test",
        "p three": "pairwise_train",
    }
    first = assign_block_roles(blocks, {"pairwise_train": 3}, pinned_roles=pins, closure_mode="home_pool")
    assert first["orcid_home_role"] == {shared: "final_test_a", tie: "dev_test"}
    # A new evaluation role would win by count without the prior home pin.
    blocks.extend(BlockSpec(key, 50, frozenset({shared})) for key in ("b one", "b two"))
    result = assign_block_roles(
        blocks,
        {"final_test_b": 2},
        pinned_roles={**pins, "b one": "final_test_b", "b two": "final_test_b"},
        pinned_home_roles=first["orcid_home_role"],
        closure_mode="home_pool",
    )
    assert result["orcid_home_role"] == first["orcid_home_role"]


def test_home_role_uses_mention_counts_within_same_priority() -> None:
    orcid = "0000-0002-1694-233X"
    blocks = [
        BlockSpec("one", 20, frozenset({orcid}), orcid_mention_counts={orcid: 3}),
        BlockSpec("two", 20, frozenset({orcid})),
        BlockSpec("three", 20, frozenset({orcid})),
    ]
    result = assign_block_roles(
        blocks,
        {"pairwise_train": 1, "eps_cal": 2},
        pinned_roles={"one": "pairwise_train", "two": "eps_cal", "three": "eps_cal"},
        closure_mode="home_pool",
    )
    assert result["orcid_home_role"] == {orcid: "pairwise_train"}


def test_examined_orcid_scores_outside_the_sealed_half() -> None:
    identity = "0000-0002-1694-233X"
    result = assign_block_roles(
        [
            BlockSpec("examined", 1, frozenset({identity}), orcid_mention_counts={identity: 1}),
            BlockSpec("sealed", 100, frozenset({identity}), orcid_mention_counts={identity: 100}),
        ],
        {"dev_test": 1, "final_test_a": 1},
        closure_mode="home_pool",
        pinned_roles={"examined": "dev_test", "sealed": "final_test_a"},
        pinned_home_roles={identity: "final_test_a"},
    )
    assert result["orcid_home_role"] == {identity: "dev_test"}


def test_load_blocks_accepts_parquet_map_mention_counts(tmp_path: Path) -> None:
    path = tmp_path / "blocks.parquet"
    identity = "0000-0002-1694-233X"
    table = pa.table(
        {
            "block_key": ["a ross"],
            "signature_count": [2],
            "orcids": [[identity]],
            "orcid_mention_counts": pa.array([[(identity, 2)]], type=pa.map_(pa.string(), pa.int64())),
        }
    )
    pq.write_table(table, path)
    assert load_blocks(path)[0].orcid_mention_counts == {identity: 2}


def test_frozen_block_roles_reject_unlisted_orcid(tmp_path: Path) -> None:
    orcid = "0000-0002-1694-233X"
    path = tmp_path / "roles.json"
    result = assign_block_roles(
        [BlockSpec("one", 20, frozenset({orcid})), BlockSpec("two", 20)],
        {"final_test_a": 1, "pairwise_train": 1},
        pinned_roles={"one": "final_test_a", "two": "pairwise_train"},
    )
    path.write_text(json.dumps(result), encoding="utf-8")
    roles = load_block_roles(path)
    assert roles.scores("one", orcid)
    for block, identity in (("one", "0000-0001-5000-0007"), ("two", orcid)):
        with pytest.raises(ValueError, match="ORCID .* absent .* block"):
            roles.scores(block, identity)


@pytest.mark.parametrize("check", ["implicit_home", "exclusion", "missing_pin", "extra_pin", "plain_pin"])
def test_refreeze_preserves_homes_and_exclusions(tmp_path: Path, check: str) -> None:
    implicit, shared = "0000-0002-1694-233X", "0000-0002-9079-593X"
    records = [
        {"block_key": "training", "orcids": [implicit, shared], "split": "development"},
        {"block_key": "old holdout", "orcids": [shared], "split": "holdout"},
        {"block_key": "excluded", "orcids": [], "split": "development"},
    ]
    blocks, targets = tmp_path / "blocks.json", tmp_path / "targets.json"

    def freeze(output: str, role_targets: dict[str, int], *extra: str) -> Path:
        blocks.write_text(json.dumps([dict(row, signature_count=20) for row in records]), encoding="utf-8")
        targets.write_text(json.dumps(role_targets), encoding="utf-8")
        assert (
            main(
                [
                    "--blocks",
                    str(blocks),
                    "--targets-json",
                    str(targets),
                    "--output-dir",
                    str(tmp_path / output),
                    "--closure-mode",
                    "home_pool",
                    "--enforce-source-splits",
                    *extra,
                ]
            )
            == 0
        )
        return tmp_path / output / "assignments.json"

    first = freeze("first", {"pairwise_train": 1, "final_test_a": 1}, "--exclude-blocks", "excluded", "absent")
    assert set(json.loads(first.read_text(encoding="utf-8"))["excluded_blocks"]) == {"excluded", "absent"}
    prior = load_block_roles(first)
    assert prior.orcid_home_role == {shared: "final_test_a"}
    records.append({"block_key": "new holdout", "orcids": [implicit, shared], "split": "holdout"})
    args = ("--pinned-roles-json", str(first))
    if check == "missing_pin":
        records.pop(1)
        with pytest.raises(ValueError, match="old holdout"):
            freeze("second", {"pairwise_train": 2, "final_test_b": 1}, *args)
        return

    if check == "plain_pin":
        pins = tmp_path / "pins.json"
        pins.write_text(json.dumps(prior.block_to_role), encoding="utf-8")
        with pytest.raises(ValueError, match="Use prior assignments.json"):
            freeze("second", {"pairwise_train": 2, "final_test_b": 1}, "--pinned-roles-json", str(pins))
        return
    if check == "extra_pin":
        pins = tmp_path / "pins.json"
        pins.write_text(json.dumps({"new_holdout": "final_test_b"}), encoding="utf-8")
        args += ("--extra-pins-json", str(pins))
    second = freeze("second", {"pairwise_train": 2, "final_test_b": 1}, *args)
    current = load_block_roles(second)
    if check == "implicit_home":
        assert current.orcid_home_role == {shared: "final_test_a", implicit: "pairwise_train"}
        assert current.scores("training", implicit)
        assert not current.scores("new holdout", implicit)
        assert not current.scores("new holdout", shared)
    else:
        assert json.loads(second.read_text(encoding="utf-8"))["excluded_blocks"] == ["absent", "excluded"]
        assert "excluded" not in current.block_to_role
    assert all(current.role_for(block) == role for block, role in prior.block_to_role.items())


@pytest.mark.parametrize("giants", [9, 1])
def test_small_development_roles_reserve_giants(giants: int) -> None:
    blocks = [
        *[BlockSpec(f"small {i}", 100, stratum="small", source_split="development") for i in range(400)],
        *[BlockSpec(f"giant {i}", 20_000, stratum="giant", source_split="development") for i in range(giants)],
    ]
    roles = ("eps_cal", "gate_fit", "gate_thresh", "dev_test")
    result = assign_block_roles(blocks, dict.fromkeys(roles, 10), enforce_source_splits=True)
    counts = result["verification"]["role_block_counts_by_stratum"]
    assert all(len(result["roles"][role]) == 10 for role in roles)
    assert sum(counts[role]["giant"] > 0 for role in roles) == min(giants, len(roles))
    assert result["stratum_shortfalls"] == {role: ["giant"] for role in roles if not counts[role]["giant"]}


def test_single_block_strata_fit_one_half_each() -> None:
    blocks = [
        BlockSpec("small", 100, stratum="small", source_split="holdout"),
        BlockSpec("giant", 20_000, stratum="giant", source_split="holdout"),
    ]
    result = assign_block_roles(blocks, {"final_test": 2}, enforce_source_splits=True)
    assert set(result["roles"]["final_test_a"] + result["roles"]["final_test_b"]) == {"small", "giant"}
    assert all(len(result["stratum_shortfalls"][role]) == 1 for role in ("final_test_a", "final_test_b"))


def test_sealed_target_must_cover_each_stratum() -> None:
    blocks = [
        BlockSpec(f"{stratum} {i}", 100, stratum=stratum, source_split="holdout")
        for stratum, count in (("small", 100), ("medium", 10), ("giant", 2))
        for i in range(count)
    ]
    with pytest.raises(ValueError, match="Sealed targets.*one block per stratum"):
        assign_block_roles(blocks, {"final_test": 4}, enforce_source_splits=True)


@pytest.mark.parametrize("count_key", ["https://orcid.org/0000-0002-1694-233x", "0000-0001-5000-0007"])
def test_load_blocks_normalizes_mention_counts(tmp_path: Path, count_key: str) -> None:
    orcid = "0000-0002-1694-233X"
    path = tmp_path / "blocks.json"
    path.write_text(
        json.dumps(
            [
                {
                    "block_key": "train",
                    "signature_count": 100,
                    "orcids": [orcid],
                    "orcid_mention_counts": {count_key: 50},
                },
                {"block_key": "tune", "signature_count": 100, "orcids": [orcid, orcid]},
            ]
        ),
        encoding="utf-8",
    )
    if count_key != "https://orcid.org/0000-0002-1694-233x":
        with pytest.raises(ValueError, match="Missing ORCID mention counts"):
            load_blocks(path)
        return
    result = assign_block_roles(
        load_blocks(path),
        {"pairwise_train": 1, "eps_cal": 1},
        pinned_roles={"train": "pairwise_train", "tune": "eps_cal"},
        closure_mode="home_pool",
    )
    assert result["orcid_home_role"] == {orcid: "pairwise_train"}


def test_sealed_halves_fill_their_last_quota_from_the_same_stratum() -> None:
    blocks = [
        BlockSpec(f"{stratum} {index}", 100, stratum=stratum, source_split="holdout")
        for stratum in ("small", "medium")
        for index in range(5)
    ]
    result = assign_block_roles(blocks, {"final_test": 6}, enforce_source_splits=True)
    counts = result["verification"]["role_block_counts_by_stratum"]
    assert sorted(counts["final_test_a"].values()) == [1, 2]
    assert counts["final_test_a"] == counts["final_test_b"]


@pytest.mark.parametrize("require_orcid_lists", [False, True])
def test_missing_orcid_list_obeys_input_requirement(tmp_path: Path, require_orcid_lists: bool) -> None:
    path = tmp_path / "blocks.json"
    path.write_text(json.dumps([{"block_key": "a ross", "signature_count": 3}]), encoding="utf-8")
    if require_orcid_lists:
        with pytest.raises(ValueError, match="Expected an ORCID list"):
            load_blocks(path, require_orcid_lists=True)
    else:
        (block,) = load_blocks(path, require_orcid_lists=False)
        assert not block.orcids


def test_implicit_mention_counts_normalize_repeated_orcids(tmp_path: Path) -> None:
    orcid = "0000-0002-1694-233X"
    url = f"https://orcid.org/{orcid}"
    path = tmp_path / "blocks.json"
    path.write_text(
        json.dumps([{"block_key": "a ross", "signature_count": 3, "orcids": [orcid, url, ""]}]),
        encoding="utf-8",
    )
    (block,) = load_blocks(path, require_orcid_lists=True)
    assert block.orcid_mention_counts == {orcid: 2}

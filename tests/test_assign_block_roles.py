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
        # Shares valid_a with a khan via HTTPS URL form.
        BlockSpec("b linked", 5000, frozenset({f"https://orcid.org/{valid_a}"}), "medium", "development"),
        # Pair of development blocks sharing valid_b.
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
        "fit_pool": 8,
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

    # Legacy giants and explicit pins go to dev_test.
    assert mapping["a khan"] == "dev_test"
    assert mapping["s feldman"] == "dev_test"
    assert mapping.get("b linked") != "dev_test"
    assert mapping["blk 08"] == "dev_test"
    # final_test receives ONLY holdout blocks and does NOT absorb overflow.
    assert len(res1["roles"]["final_test"]) == 6
    assert len(res1["unassigned"]) > 0
    assert res1["verification"]["cross_role_orcid_overlap"] == 0


def test_assign_block_roles_balances_strata_without_aliasing() -> None:
    # Regression test (Astra #13 & Fable #2): 4 A-stratum + 4 B-stratum blocks across 2 equal roles
    # must give 2 A + 2 B to each role (never 4 A to one and 4 B to the other!).
    blocks = [
        *[BlockSpec(f"a_{i}", 1000, frozenset(), "A") for i in range(4)],
        *[BlockSpec(f"b_{i}", 1000, frozenset(), "B") for i in range(4)],
    ]
    res = assign_block_roles(blocks, {"eps_cal": 4, "gate_thresh": 4}, seed=7)
    for role in ("eps_cal", "gate_thresh"):
        assigned = res["roles"][role]
        assert len(assigned) == 4
        assert sum(k.startswith("a ") for k in assigned) == 2
        assert sum(k.startswith("b ") for k in assigned) == 2


@pytest.mark.parametrize(
    "old_role", ["pairwise_train", "pairwise_val", "linker_train", "gate_fit", "final_test_a", "final_test_b"]
)
def test_rejects_removed_roles(tmp_path: Path, old_role: str) -> None:
    blocks = [BlockSpec("a ross", 32)]
    with pytest.raises(ValueError, match=old_role):
        assign_block_roles(blocks, {old_role: 1})
    with pytest.raises(ValueError, match=old_role):
        assign_block_roles(blocks, {"fit_pool": 1}, pinned_roles={"a ross": old_role})
    manifest = assign_block_roles(blocks, {"fit_pool": 1})
    manifest["roles"][old_role] = manifest["roles"].pop("fit_pool")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    error = "gate_fit role. Reassign blocks" if old_role == "gate_fit" else rf"Unknown block role '{old_role}'"
    with pytest.raises(ValueError, match=error):
        load_block_roles(path)


def test_pairwise_validation_blocks_are_stratified_and_order_independent(tmp_path: Path) -> None:
    blocks = [BlockSpec(f"{origin} {i}", 500, stratum=f"<=1000|{origin}") for origin in "ab" for i in range(10)]
    result = assign_block_roles(blocks, {"fit_pool": 20}, pairwise_validation_fraction=0.2)
    validation = result["pairwise_validation_blocks"]
    assert [sum(key.startswith(f"{origin} ") for key in validation) for origin in "ab"] == [2, 2]
    assert assign_block_roles(blocks[::-1], {"fit_pool": 20}, pairwise_validation_fraction=0.2) == result
    assert len(assign_block_roles(blocks, {"fit_pool": 20})["pairwise_validation_blocks"]) == 3
    assert assign_block_roles(blocks[:1], {"fit_pool": 1})["pairwise_validation_blocks"] == []
    path = tmp_path / "roles.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert load_block_roles(path).pairwise_validation_blocks == set(validation)
    path.write_text(json.dumps({**result, "pairwise_validation_blocks": ["c missing"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="pairwise_validation_blocks"):
        load_block_roles(path)


def test_gate_training_blocks_are_stratified_frozen_and_can_overlap_validation(tmp_path: Path) -> None:
    blocks = [BlockSpec(f"{origin} {i}", 500, stratum=f"<=1000|{origin}") for origin in "ab" for i in range(14)]
    targets = {"fit_pool": 24, "gate_thresh": 4}
    result = assign_block_roles(blocks, targets)
    marked = result["gate_training_blocks"]
    assert len(set(marked)) == len(marked) == 20
    assert set(marked) <= set(result["roles"]["fit_pool"])
    assert [sum(key.startswith(f"{origin} ") for key in marked) for origin in "ab"] == [10, 10]
    assert marked == assign_block_roles(blocks[::-1], targets)["gate_training_blocks"]
    all_fit = assign_block_roles(blocks, targets, gate_training_block_count=24)
    assert set(all_fit["pairwise_validation_blocks"]) <= set(all_fit["gate_training_blocks"])
    path = tmp_path / "assignments.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    roles = load_block_roles(path)
    assert roles.gate_training_blocks == set(marked)
    path.write_text(json.dumps({**result, "gate_training_blocks": marked[:-1]}), encoding="utf-8")
    assert load_block_roles(path).sha256 != roles.sha256


def test_giant_caps_include_pins_and_fit_pool_pins_cannot_override_dev_test() -> None:
    blocks = [
        BlockSpec("a khan", 15_001, stratum="giant"),
        BlockSpec("a fit", 15_001, stratum="giant"),
        BlockSpec("a spare", 15_001, stratum="giant"),
        BlockSpec("a small", 15_000, stratum="small"),
        BlockSpec("a other", 500, stratum="small"),
    ]
    result = assign_block_roles(
        blocks, {"fit_pool": 2, "eps_cal": 1}, fit_pool_blocks=["A_FIT"], giant_caps={"fit_pool": 1, "eps_cal": 0}
    )
    assert result["roles"]["dev_test"] == ["a khan"]
    assert set(result["roles"]["fit_pool"]) & {"a fit", "a spare"} == {"a fit"}
    assert len(result["roles"]["fit_pool"]) == 2
    assert result["roles"]["eps_cal"][0] in {"a small", "a other"}
    for key in ("A_KHAN", "a small"):
        with pytest.raises(ValueError, match="both fit_pool and dev_test"):
            assign_block_roles(blocks, {"fit_pool": 2}, fit_pool_blocks=[key], extra_dev_test_blocks=["a small"])
    with pytest.raises(ValueError, match="exceed the giant cap"):
        assign_block_roles(blocks, {"fit_pool": 2}, giant_caps={"dev_test": 0})


def test_sealed_role_respects_giant_cap() -> None:
    blocks = [
        *[BlockSpec(f"s {i}", 15_000, stratum="small", source_split="holdout") for i in range(8)],
        *[BlockSpec(f"g {i}", 15_001, stratum="giant", source_split="holdout") for i in range(8)],
    ]
    result = assign_block_roles(blocks, {"final_test": 8}, giant_caps={"final_test": 1}, enforce_source_splits=True)
    assert len(result["roles"]["final_test"]) == 8
    assert result["verification"]["role_block_counts_by_stratum"]["final_test"]["giant"] == 1


def test_block_roles_do_not_follow_shared_orcids() -> None:
    shared_orcid = "0000-0002-1694-233X"
    blocks = [
        BlockSpec("a khan", 20000, frozenset({shared_orcid}), "large"),
        BlockSpec("blk 1", 10000, frozenset({shared_orcid}), "large"),
        BlockSpec("blk 2", 10000, frozenset({shared_orcid}), "large"),
    ]
    res = assign_block_roles(blocks, {"eps_cal": 2}, seed=1)
    assert res["block_to_role"]["a khan"] == "dev_test"
    assert res["block_to_role"]["blk 1"] == "eps_cal"
    assert res["block_to_role"]["blk 2"] == "eps_cal"
    assert res["verification"]["raw_cross_role_orcid_overlap"] == 1
    assert res["verification"]["cross_role_orcid_overlap"] == 0


def test_verify_rejects_misassigned_legacy_blocks() -> None:
    shared = "0000-0002-1694-233X"
    legacy = sorted(LEGACY_DEV_TEST_BLOCKS)[0]
    verify_blocks = [
        BlockSpec("x1", 100, frozenset({shared})),
        BlockSpec("x2", 100, frozenset({shared})),
        BlockSpec(legacy, 100, frozenset()),
    ]
    with pytest.raises(ValueError, match="must be assigned to 'dev_test'"):
        verify_role_assignments(verify_blocks, {"x1": "fit_pool", "x2": "fit_pool", legacy: "final_test"})


def test_stratified_holdout_reserves_rare_giants() -> None:
    blocks = [
        *[BlockSpec(f"small_{i}", 200, stratum="a_small", source_split="holdout") for i in range(1000)],
        *[BlockSpec(f"giant_{i}", 20_000, stratum="z_giant", source_split="holdout") for i in range(40)],
    ]
    result = assign_block_roles(blocks, {"final_test": 20}, enforce_source_splits=True)
    counts = result["verification"]["role_block_counts_by_stratum"]
    assert counts["final_test"]["z_giant"] >= 1
    assert counts["final_test"]["a_small"] >= 1
    assert len(result["roles"]["final_test"]) == 20
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
    result = assign_block_roles(load_blocks(blocks), {"eps_cal": 2, "final_test": 2}, enforce_source_splits=True)
    assert result["roles"]["eps_cal"] == ["a 0", "a 1"]
    assert result["verification"]["role_block_counts_by_stratum"]["final_test"] == {">15000/Latn/Iberian": 2}
    records[-1]["split"] = None
    pq.write_table(pa.Table.from_pylist(records), blocks)
    with pytest.raises(ValueError, match="missing or invalid source_split"):
        assign_block_roles(load_blocks(blocks), {"eps_cal": 2, "final_test": 2}, enforce_source_splits=True)


def test_free_development_role_cannot_consume_holdout() -> None:
    result = assign_block_roles(
        [BlockSpec(key, 20, source_split="holdout") for key in ("red", "blue", "green")],
        {"final_test": 2, "dev_test": 1},
        enforce_source_splits=True,
    )
    assert result["roles"]["dev_test"] == []
    assert result["shortfalls"] == {"dev_test": 1}


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
        {"final_test": 1, "fit_pool": 1},
        pinned_roles={"one": "final_test", "two": "fit_pool"},
    )
    path.write_text(json.dumps(result), encoding="utf-8")
    roles = load_block_roles(path)
    assert roles.scores("one", orcid)
    for block, identity in (("one", "0000-0001-5000-0007"), ("two", orcid)):
        with pytest.raises(ValueError, match="ORCID .* absent .* block"):
            roles.scores(block, identity)


@pytest.mark.parametrize("giants", [9, 1])
def test_small_development_roles_reserve_giants(giants: int) -> None:
    blocks = [
        *[BlockSpec(f"small {i}", 100, stratum="small", source_split="development") for i in range(400)],
        *[BlockSpec(f"giant {i}", 20_000, stratum="giant", source_split="development") for i in range(giants)],
    ]
    roles = ("eps_cal", "fit_pool", "gate_thresh", "dev_test")
    result = assign_block_roles(blocks, dict.fromkeys(roles, 10), enforce_source_splits=True)
    counts = result["verification"]["role_block_counts_by_stratum"]
    assert all(len(result["roles"][role]) == 10 for role in roles)
    assert sum(counts[role]["giant"] > 0 for role in roles) == min(giants, len(roles))
    assert result["stratum_shortfalls"] == {role: ["giant"] for role in roles if not counts[role]["giant"]}


def test_sealed_role_covers_single_block_strata() -> None:
    blocks = [
        BlockSpec("small", 100, stratum="small", source_split="holdout"),
        BlockSpec("giant", 20_000, stratum="giant", source_split="holdout"),
    ]
    result = assign_block_roles(blocks, {"final_test": 2}, enforce_source_splits=True)
    assert set(result["roles"]["final_test"]) == {"small", "giant"}
    assert not result["stratum_shortfalls"]


@pytest.mark.parametrize("count_key", ["https://orcid.org/0000-0002-1694-233x", "0000-0001-5000-0007"])
def test_load_blocks_normalizes_mention_counts(tmp_path: Path, count_key: str) -> None:
    orcid = "0000-0002-1694-233X"
    path = tmp_path / "blocks.parquet"
    pq.write_table(
        pa.table(
            {
                "block_key": ["train", "tune"],
                "signature_count": [100, 100],
                "orcids": [[orcid], [orcid, orcid]],
                "orcid_mention_counts": pa.array([[(count_key, 50)], None], type=pa.map_(pa.string(), pa.int64())),
            }
        ),
        path,
    )
    if count_key != "https://orcid.org/0000-0002-1694-233x":
        with pytest.raises(ValueError, match="Missing ORCID mention counts"):
            load_blocks(path)
        return
    result = assign_block_roles(
        load_blocks(path),
        {"fit_pool": 1, "eps_cal": 1},
        pinned_roles={"train": "fit_pool", "tune": "eps_cal"},
    )
    assert result["verification"]["raw_cross_role_orcid_overlap"] == 1


def test_load_blocks_requires_orcid_list(tmp_path: Path) -> None:
    path = tmp_path / "blocks.parquet"
    pq.write_table(pa.table({"block_key": ["a ross"], "signature_count": [3]}), path)
    with pytest.raises(ValueError, match="Expected an ORCID list"):
        load_blocks(path)


def test_implicit_mention_counts_normalize_repeated_orcids(tmp_path: Path) -> None:
    orcid = "0000-0002-1694-233X"
    url = f"https://orcid.org/{orcid}"
    path = tmp_path / "blocks.parquet"
    pq.write_table(pa.table({"block_key": ["a ross"], "signature_count": [3], "orcids": [[orcid, url, ""]]}), path)
    (block,) = load_blocks(path)
    assert block.orcid_mention_counts == {orcid: 2}

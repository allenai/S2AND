"""Exercise the frozen mining handoff with valid hashes and invalid content."""

import csv
import json
from pathlib import Path

import pytest

from s2and._sha256 import sha256_file
from scripts.common.stage0_invariants import require_fit_pool_coverage, require_intake_index, require_pair_conservation
from scripts.pairwise.mine_pairwise_pairs import mine_pairs
from scripts.pairwise.training_pairs import FitPair, JudgePair, TrainValPairs, training_pairs
from tests.stage0_world import Stage0World, build_stage0_world

pytestmark = pytest.mark.tooling


def _load(world: Stage0World, *, name_counts_sha256: str | None = None) -> TrainValPairs:
    return training_pairs(
        world.roles,
        world.ledger,
        world.mined,
        name_counts_sha256=name_counts_sha256 or sha256_file(world.name_counts / "manifest.json"),
    )


def test_handoff_preserves_exact_train_validation_and_judge_pairs(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path)
    result = _load(world)
    train, val = result.blocks["a ross"], result.blocks["b crane"]
    assert train.fit == [
        FitPair("a ross", "10-0", "11-0", "missed_link", 1, "train"),
        FitPair("a ross", "10-0", "12-0", "false_merge", 0, "train"),
        FitPair("a ross", "10-0", "13-0", "random", 0, "train"),
        FitPair("a ross", "11-0", "12-0", "random", 0, "train"),
        FitPair("a ross", "11-0", "13-0", "random", 0, "train"),
        FitPair("a ross", "12-0", "13-0", "missed_link", 1, "train"),
    ]
    assert val.fit == [FitPair("b crane", "20-0", "21-0", "random", 1, "val")]
    assert train.judge == [JudgePair("a ross", "10-0", "14-0"), JudgePair("a ross", "12-0", "14-0")]
    assert val.judge == []
    require_fit_pool_coverage(world.roles, result.blocks.keys())
    manifest = json.loads(world.manifest.read_text(encoding="utf-8"))
    require_intake_index(world.ledger.evidence.name_counts_index_sha256, manifest["index_sha256"], context="Miner")
    require_pair_conservation(
        "a ross",
        "train",
        manifest["blocks"]["a ross"]["strata"],
        {
            ("train", "false_merge"): [("10-0", "12-0")],
            ("train", "missed_link"): [("10-0", "11-0"), ("12-0", "13-0")],
            ("train", "random"): [("10-0", "13-0"), ("11-0", "12-0"), ("11-0", "13-0")],
            ("train", "merged_unlabeled"): [("10-0", "14-0"), ("12-0", "14-0")],
        },
    )


@pytest.mark.parametrize(
    ("hazard", "error"),
    [
        ("missing_fit_pool_block", r"missing=\['b crane'\]"),
        ("foreign_index", "pairwise assignments name-counts index does not match intake"),
        ("judge_as_fitted", r"Fitted pair requires two ledger scores in 'a ross' stratum 'random'.*'14-0'"),
        ("fitted_as_judge", r"Judge-only pair has two ledger scores in 'a ross' stratum 'merged_unlabeled'"),
        ("lost_ledger_score", r"Fitted pair requires two ledger scores in 'b crane' stratum 'random'.*'21-0'"),
    ],
)
def test_handoff_rejects_rebound_hazards(tmp_path: Path, hazard: str, error: str) -> None:
    world = build_stage0_world(tmp_path, **{hazard: True})
    manifest = json.loads(world.manifest.read_text(encoding="utf-8"))
    assert manifest["output_sha256"] == sha256_file(world.mined)
    assert manifest["block_role_assignments_sha256"] == world.roles.sha256
    assert manifest["block_bindings"] == world.roles.bindings_for(world.inputs.keys())
    with pytest.raises(ValueError, match=error):
        _load(world)
    if hazard in {"missing_fit_pool_block", "foreign_index"}:
        output = tmp_path / "rejected.csv"
        with pytest.raises(ValueError, match=error):
            mine_pairs(world.roles_path, world.inputs, output, identity_evidence=world.evidence_path)
        assert not output.exists()
        assert not output.with_suffix(".csv.manifest.json").exists()


def test_trainer_checks_its_own_index(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path, foreign_index=True)
    manifest = json.loads(world.manifest.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Pairwise trainer name-counts index does not match intake"):
        _load(world, name_counts_sha256=manifest["index_sha256"])


def test_handoff_accepts_an_explicit_empty_block(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path, empty_block=True)
    result = _load(world)
    empty = result.blocks["e vale"]
    assert empty.fold == "train" and not empty.fit and not empty.judge
    strata = json.loads(world.manifest.read_text(encoding="utf-8"))["blocks"]["e vale"]["strata"]
    assert {name: counts["sampled_pairs"] for name, counts in strata.items()} == {
        "false_merge": 0,
        "missed_link": 0,
        "random": 0,
        "merged_unlabeled": 0,
    }


def test_sample_budget_is_conserved_in_the_manifest(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path)
    mined = tmp_path / "capped.csv"
    assert (
        mine_pairs(
            world.roles_path,
            world.inputs,
            mined,
            identity_evidence=world.evidence_path,
            per_block=2,
            per_block_floor=0,
            verified_pair_fraction=1,
        )
        == 5
    )
    strata = json.loads(mined.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))["blocks"]["a ross"][
        "strata"
    ]
    for stratum, expected in {
        "false_merge": (1, 1, 0, None),
        "missed_link": (2, 1, 1, "sampling_budget"),
        "random": (3, 0, 3, "sampling_budget"),
        "merged_unlabeled": (2, 2, 0, None),
    }.items():
        assert (
            tuple(strata[stratum][key] for key in ("frame_pairs", "sampled_pairs", "dropped_pairs", "drop_reason"))
            == expected
        )
    training_pairs(
        world.roles,
        world.ledger,
        mined,
        name_counts_sha256=sha256_file(world.name_counts / "manifest.json"),
    )


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("missing_row", r"pair count differs .*'b crane' stratum 'random' fold 'val'"),
        ("changed_ids", r"pair IDs differ .*'a ross' stratum 'merged_unlabeled' fold 'train'"),
        ("moved_stratum", r"pair count differs .*'a ross' stratum 'false_merge' fold 'train'"),
        ("duplicate", r"Duplicate mined pair .*'a ross' stratum 'random'"),
        ("foreign_block", r"outside fit_pool: 'c lane'"),
        ("wrong_fold", r"Mined fold for 'b crane' is 'train', expected 'val'"),
        ("unbalanced_frame", r"frame counts do not reconcile .*'a ross' stratum 'random' fold 'train'"),
    ],
)
def test_handoff_reconciles_pairs_and_manifest(tmp_path: Path, change: str, error: str) -> None:
    world = build_stage0_world(tmp_path)
    manifest = json.loads(world.manifest.read_text(encoding="utf-8"))
    with world.mined.open(encoding="utf-8", newline="") as handle:
        columns, *rows = csv.reader(handle)
    if change == "missing_row":
        rows = [row for row in rows if row[0] != "b crane"]
    elif change == "changed_ids":
        next(row for row in rows if row[1:3] == ["10-0", "14-0"])[1] = "11-0"
    elif change == "moved_stratum":
        next(row for row in rows if row[1:3] == ["10-0", "12-0"])[3] = "random"
    elif change == "duplicate":
        rows.append(["a ross", "10-0", "12-0", "random", "1"])
    elif change == "foreign_block":
        rows.append(["c lane", "30-0", "31-0", "random", "1"])
        with pytest.raises(ValueError, match="outside fit_pool"):
            mine_pairs(
                world.roles_path,
                world.inputs | {"c lane": world.inputs["a ross"]},
                tmp_path / "foreign.csv",
                identity_evidence=world.evidence_path,
            )
    elif change == "wrong_fold":
        manifest["blocks"]["b crane"]["fold"] = "train"
    elif change == "unbalanced_frame":
        manifest["blocks"]["a ross"]["strata"]["random"]["frame_pairs"] += 1
    with world.mined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    manifest["output_sha256"] = sha256_file(world.mined)
    world.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        _load(world)

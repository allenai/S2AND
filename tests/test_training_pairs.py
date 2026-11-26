"""Exercise the frozen mining handoff with valid hashes and invalid content."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from s2and._sha256 import sha256_file
from scripts.common.block_roles import load_block_roles
from scripts.common.identity_ledger import ledger_for_roles
from scripts.common.stage0_invariants import require_fit_pool_coverage, require_intake_index
from scripts.pairwise.mine_pairwise_pairs import mine_pairs
from scripts.pairwise.training_pairs import FitPair, JudgePair, TrainValPairs, training_pairs
from tests.stage0_world import Stage0World, build_stage0_world

pytestmark = pytest.mark.tooling


def _load(
    world: Stage0World, *, name_counts_sha256: str | None = None, min_mined_validation_pairs: int = 1
) -> TrainValPairs:
    return training_pairs(
        world.roles,
        world.ledger,
        world.mined,
        name_counts_sha256=name_counts_sha256 or sha256_file(world.name_counts / "manifest.json"),
        min_mined_validation_pairs=min_mined_validation_pairs,
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


@pytest.mark.parametrize(
    ("hazard", "error"),
    [
        ("missing_fit_pool_block", r"missing=\['b crane'\]"),
        ("foreign_index", "pairwise assignments name-counts index does not match intake"),
        ("judge_as_fitted", r"Mined pair selection differs .*'a ross'.*'14-0'.*'random'.*'merged_unlabeled'"),
        ("fitted_as_judge", r"Mined pair selection differs .*'a ross'.*'13-0'.*'merged_unlabeled'.*'random'"),
        ("lost_ledger_score", r"Mined pair selection differs .*'b crane'.*'21-0'.*'random'.*'merged_unlabeled'"),
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
    world = build_stage0_world(tmp_path)
    with pytest.raises(ValueError, match="Pairwise trainer name-counts index does not match intake"):
        _load(world, name_counts_sha256="b" * 64)


@pytest.mark.parametrize(
    ("hazard", "detail"),
    [
        ("lost_validation", r"'b crane' fold 'val'.*observed None, expected \('random', 1\.0\)"),
        ("empty_frame", r"'b crane' fold 'val'.*observed None, expected \('random', 1\.0\)"),
        ("swapped_strata", r"'a ross' fold 'train'.*observed \('false_merge'.*expected \('missed_link'"),
        ("stale_frame", r"'e vale' fold 'train'.*observed None, expected \('missed_link', 1\.0\)"),
        ("zeroed_judge_frame", r"'e vale' fold 'train'.*observed None, expected \('merged_unlabeled', 1\.0\)"),
    ],
)
def test_handoff_replays_the_frozen_sample(tmp_path: Path, hazard: str, detail: str) -> None:
    world = build_stage0_world(tmp_path, sampling_hazard=hazard)
    manifest = json.loads(world.manifest.read_text(encoding="utf-8"))
    assert manifest["output_sha256"] == sha256_file(world.mined)
    assert manifest["block_bindings"] == world.roles.bindings_for(world.inputs.keys())
    for block, entry in world.inputs.items():
        assert manifest["blocks"][block]["assignments_sha256"] == sha256_file(entry.assignments)
    with pytest.raises(ValueError, match="Mined pair selection differs from frozen inputs in " + detail):
        _load(world)


def test_judge_budget_is_independent_of_zero_fitted_cap(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path, sampling_hazard="judge_zero_cap")
    empty = _load(world).blocks["e vale"]
    assert not empty.fit
    assert empty.judge == [JudgePair("e vale", "40-0", "41-0")]
    mined = tmp_path / "zero_fit_cap.csv"
    mine_pairs(
        world.roles_path,
        world.inputs,
        mined,
        identity_evidence=world.evidence_path,
        per_block_floor=0,
        verified_pair_fraction=0,
    )
    report = json.loads(mined.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
    assert report["blocks"]["b crane"]["strata"]["random"]["drops"] == {"zero_cap": 1}


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


def test_all_ineligible_fit_block_reaches_mining(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path, all_ineligible_block=True)
    assignment = world.inputs["e vale"].assignments
    receipt = json.loads(assignment.with_name("manifest.json").read_text(encoding="utf-8"))
    assert receipt["counts"] == {
        "assigned_signatures": 0,
        "components": 0,
        "queries": 0,
        "empty_reason": "no_eligible_signatures",
    }
    assert pq.read_table(assignment).num_rows == pq.read_table(assignment.with_name("seeds.parquet")).num_rows == 0
    assert json.loads(assignment.with_name("queries.json").read_text(encoding="utf-8")) == []
    assert receipt["files"][assignment.name] == sha256_file(assignment)
    empty = _load(world).blocks["e vale"]
    assert not empty.fit and not empty.judge


def test_empty_fit_pool_fails_at_both_boundaries(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path)
    payload = json.loads(world.roles_path.read_text(encoding="utf-8"))
    payload["roles"]["fit_pool"] = []
    payload["block_to_role"] = {"c lane": "gate_fit"}
    payload["pairwise_validation_blocks"] = []
    world.roles_path.write_text(json.dumps(payload), encoding="utf-8")
    roles = load_block_roles(world.roles_path)
    ledger = ledger_for_roles(world.evidence_path, roles)
    assert ledger is not None
    world.mined.write_text(world.mined.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    world.manifest.write_text(
        json.dumps(
            {
                "role": "fit_pool",
                "block_role_assignments_sha256": roles.sha256,
                "block_bindings": {},
                "blocks": {},
                "output_sha256": sha256_file(world.mined),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Expected at least one frozen fit_pool block"):
        _load(replace(world, roles=roles, ledger=ledger))
    with pytest.raises(ValueError, match="Expected at least one frozen fit_pool block"):
        mine_pairs(world.roles_path, {}, tmp_path / "rejected.csv", identity_evidence=world.evidence_path)


@pytest.mark.parametrize("validation_blocks", [(), ("e vale",)])
def test_validation_requires_a_fitted_mined_pair(tmp_path: Path, validation_blocks: tuple[str, ...]) -> None:
    world = build_stage0_world(tmp_path, empty_block=True, validation_blocks=validation_blocks)
    with pytest.raises(ValueError, match="Expected at least 1 fitted mined validation pairs, found 0"):
        _load(world)


def test_validation_minimum_can_be_raised(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path)
    with pytest.raises(ValueError, match="Expected at least 2 fitted mined validation pairs, found 1"):
        _load(world, min_mined_validation_pairs=2)


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
        "false_merge": (1, 1, {}),
        "missed_link": (2, 1, {"sampling_budget": 1}),
        "random": (3, 0, {"cross_stratum_duplicate": 3, "sampling_budget": 3}),
        "merged_unlabeled": (2, 2, {}),
    }.items():
        assert tuple(strata[stratum][key] for key in ("frame_pairs", "sampled_pairs", "drops")) == expected
        counts = strata[stratum]
        assert counts["eligible_pairs"] == counts["sampled_pairs"] + sum(counts["drops"].values())
    training_pairs(
        world.roles,
        world.ledger,
        mined,
        name_counts_sha256=sha256_file(world.name_counts / "manifest.json"),
    )


def test_identity_cap_drops_are_counted(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path)
    mined = tmp_path / "identity_cap.csv"
    mine_pairs(world.roles_path, world.inputs, mined, identity_evidence=world.evidence_path, per_identity=1)
    report = json.loads(mined.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
    random = report["blocks"]["a ross"]["strata"]["random"]
    assert random["frame_pairs"] == 1
    assert random["drops"] == {"cross_stratum_duplicate": 3, "identity_cap": 2}


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("changed_ids", r"pair selection differs .*'a ross' fold 'train'.*'merged_unlabeled'"),
        ("moved_stratum", r"pair selection differs .*'a ross' fold 'train'.*'false_merge'"),
        ("duplicate", r"Duplicate mined pair .*'a ross' stratum 'random'"),
        ("foreign_block", r"outside fit_pool: 'c lane'"),
        ("wrong_fold", r"Mined fold for 'b crane' is 'train', expected 'val'"),
        ("unbalanced_frame", r"strata report differs .*'a ross' fold 'train'"),
        ("missing_columns", r"Invalid mined pair CSV columns"),
        ("extra_column_value", r"Invalid mined pair CSV row 2"),
        ("probability", r"pair selection differs .*'a ross' fold 'train'.*999\.0"),
    ],
)
def test_handoff_reconciles_pairs_and_manifest(tmp_path: Path, change: str, error: str) -> None:
    world = build_stage0_world(tmp_path)
    manifest = json.loads(world.manifest.read_text(encoding="utf-8"))
    with world.mined.open(encoding="utf-8", newline="") as handle:
        columns, *rows = csv.reader(handle)
    if change == "changed_ids":
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
    elif change == "missing_columns":
        columns = columns[:4]
    elif change == "extra_column_value":
        rows[0].append("unheaded")
    elif change == "probability":
        rows[0][-1] = "999"
    with world.mined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    manifest["output_sha256"] = sha256_file(world.mined)
    world.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        _load(world)

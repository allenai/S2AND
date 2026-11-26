"""Exercise the frozen mining handoff with valid hashes and invalid content."""

import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from s2and._sha256 import sha256_file
from scripts.common.block_roles import load_block_roles
from scripts.common.identity_ledger import IdentityLedger, ledger_for_roles
from scripts.pairwise.mine_pairwise_pairs import SamplingPolicy, mine_pairs
from scripts.pairwise.training_pairs import FitPair, TrainValPairs, training_pairs
from tests.stage0_world import Stage0World, build_stage0_world

pytestmark = pytest.mark.tooling


def _load(
    world: Stage0World, *, name_counts_sha256: str | None = None, min_mined_validation_pairs: int = 1
) -> TrainValPairs:
    return training_pairs(
        world.ledger,
        world.inputs_path,
        policy=SamplingPolicy(),
        name_counts_sha256=name_counts_sha256 or sha256_file(world.name_counts / "manifest.json"),
        min_mined_validation_pairs=min_mined_validation_pairs,
    )


def test_handoff_preserves_folds_and_excludes_frozen_queries(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path, with_query=True, empty_block=True)
    result = _load(world)
    train, val, empty = (result.blocks[block] for block in ("a ross", "b crane", "e vale"))
    assert train.fit == [
        FitPair("a ross", "10-0", "11-0", "missed_link", 1, "train"),
        FitPair("a ross", "10-0", "12-0", "false_merge", 0, "train"),
        FitPair("a ross", "11-0", "12-0", "random", 0, "train"),
    ]
    assert train.judge_count == 2
    assert val.fit == [FitPair("b crane", "20-0", "21-0", "random", 1, "val")]
    assert val.judge_count == 0
    assert empty.fold == "train" and not empty.fit and not empty.judge_count
    mined = tmp_path / "mined.csv"
    mine_pairs(world.roles_path, world.inputs, mined, identity_evidence=world.evidence_path)
    manifest = json.loads(mined.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
    counts = manifest["blocks"]["a ross"]
    assert counts["excluded_query_signatures"] == 1
    with pytest.raises(ValueError, match="Expected at least 2 fitted mined validation pairs, found 1"):
        _load(world, min_mined_validation_pairs=2)
    with pytest.raises(ValueError, match="Pairwise trainer name-counts index does not match intake"):
        _load(world, name_counts_sha256="b" * 64)


@pytest.mark.parametrize(
    ("hazard", "error"),
    [
        ("missing_fit_pool_block", r"missing=\['b crane'\]"),
        ("foreign_index", "pairwise assignments name-counts index does not match intake"),
        ("lost_ledger_score", "Expected at least 1 fitted mined validation pairs, found 0"),
    ],
)
def test_handoff_rejects_rebound_hazards(tmp_path: Path, hazard: str, error: str) -> None:
    world = build_stage0_world(tmp_path, **{hazard: True})
    with pytest.raises(ValueError, match=error):
        _load(world)
    if hazard in {"missing_fit_pool_block", "foreign_index"}:
        output = tmp_path / "rejected.csv"
        with pytest.raises(ValueError, match=error):
            mine_pairs(world.roles_path, world.inputs, output, identity_evidence=world.evidence_path)
        assert not output.exists()
        assert not output.with_suffix(".csv.manifest.json").exists()


@pytest.mark.parametrize("digest", [None, "b" * 64])
def test_miner_and_trainer_require_one_assignment_model(tmp_path: Path, digest: str | None) -> None:
    world = build_stage0_world(tmp_path)
    assignments = world.inputs["b crane"].assignments
    receipt_path = assignments.with_name("manifest.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if digest is None:
        receipt["binding"].pop("model_sha256")
    else:
        receipt["binding"]["model_sha256"] = digest
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="'b crane' pairwise assignments lack a shared model_sha256"):
        _load(world)
    with pytest.raises(ValueError, match="'b crane' pairwise assignments lack a shared model_sha256"):
        mine_pairs(world.roles_path, world.inputs, tmp_path / "rejected.csv", identity_evidence=world.evidence_path)


def test_trainer_rejects_a_direct_ledger_with_foreign_evidence(tmp_path: Path) -> None:
    first, second = (tmp_path / name for name in ("first", "second"))
    first.mkdir()
    second.mkdir()
    world, foreign = build_stage0_world(first), build_stage0_world(second)
    ledger = IdentityLedger(foreign.ledger.evidence, world.roles)
    with pytest.raises(ValueError, match="Frozen roles do not match the identity evidence and intake snapshot"):
        _load(replace(world, ledger=ledger))


def test_sampling_rejects_queries_in_assignments_with_a_matching_digest(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path, sampling_hazard="query_in_assignments")
    error = r"Query signatures in assignments for 'a ross': \['13-0'\]"
    with pytest.raises(ValueError, match=error):
        _load(world)
    rejected = tmp_path / "rejected.csv"
    with pytest.raises(ValueError, match=error):
        mine_pairs(world.roles_path, world.inputs, rejected, identity_evidence=world.evidence_path)
    assert not rejected.exists()


def test_judge_budget_is_independent_of_zero_fitted_cap(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path, sampling_hazard="judge_zero_cap")
    empty = _load(world).blocks["e vale"]
    assert not empty.fit
    assert empty.judge_count == 1
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


def test_all_ineligible_fit_block_reaches_mining(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path, all_ineligible_block=True)
    assignment = world.inputs["e vale"].assignments
    assert pq.read_table(assignment).num_rows == 0
    assert json.loads(assignment.with_name("queries.json").read_text(encoding="utf-8")) == []
    mine_pairs(world.roles_path, world.inputs, tmp_path / "mined.csv", identity_evidence=world.evidence_path)
    empty = _load(world).blocks["e vale"]
    assert not empty.fit and not empty.judge_count


def test_empty_fit_pool_fails_at_both_boundaries(tmp_path: Path) -> None:
    world = build_stage0_world(tmp_path)
    payload = json.loads(world.roles_path.read_text(encoding="utf-8"))
    payload["roles"]["fit_pool"] = []
    payload["block_to_role"] = {"c lane": "gate_thresh"}
    payload["pairwise_validation_blocks"] = []
    payload["gate_training_blocks"] = []
    world.roles_path.write_text(json.dumps(payload), encoding="utf-8")
    roles = load_block_roles(world.roles_path)
    ledger = ledger_for_roles(world.evidence_path, roles)
    assert ledger is not None
    world.inputs_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected at least one frozen fit_pool block"):
        _load(replace(world, roles=roles, ledger=ledger))
    with pytest.raises(ValueError, match="Expected at least one frozen fit_pool block"):
        mine_pairs(world.roles_path, {}, tmp_path / "rejected.csv", identity_evidence=world.evidence_path)


@pytest.mark.parametrize("validation_blocks", [(), ("e vale",)])
def test_validation_requires_a_fitted_mined_pair(tmp_path: Path, validation_blocks: tuple[str, ...]) -> None:
    world = build_stage0_world(tmp_path, empty_block=True, validation_blocks=validation_blocks)
    with pytest.raises(ValueError, match="Expected at least 1 fitted mined validation pairs, found 0"):
        _load(world)


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
        world.ledger,
        world.inputs_path,
        policy=SamplingPolicy(pairs_per_block=2, pairs_per_block_floor=0, verified_pair_fraction=1),
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

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pandas as pd
import pytest

from s2and.arrow_inputs import ArrowDataset
from scripts.intake.assign_block_roles import BlockSpec, assign_block_roles
from tests.test_build_linker_task_tables import _ORCIDS, _OUTSIDE_HOME, _block_inputs

pytestmark = pytest.mark.tooling


def test_role_b3_masks_shared_orcids_and_rejects_other_roles(tmp_path: Path) -> None:
    from scripts.eps import orcid_b3

    block = "a ross"
    spec = _block_inputs(tmp_path / "block", block, _ORCIDS[0])
    roles_path = tmp_path / "roles.json"
    result = assign_block_roles(
        [
            BlockSpec(block, 32, frozenset({_ORCIDS[0], _OUTSIDE_HOME})),
            BlockSpec("sealed", 2, frozenset({_OUTSIDE_HOME})),
        ],
        {"eps_cal": 1, "final_test_a": 1},
        pinned_roles={block: "eps_cal", "sealed": "final_test_a"},
        closure_mode="block_only",
    )
    roles_path.write_text(json.dumps(result), encoding="utf-8")
    assignments = dict.fromkeys(("q_full", "q_initial", "s00", "s01", "s02", "s03", "q_outside_home"), "merged")
    with ArrowDataset.open(spec["arrow_dir"]) as arrow:
        report = orcid_b3.score_block_role(
            arrow,
            tmp_path,
            block,
            roles_path=roles_path,
            role="eps_cal",
            assignments=assignments,
            eps=0.38,
        )
        assert report["reference"]["excluded_non_home_role_records"] == 2
        assert report["with_singletons"]["scored_signatures"] == 5
        assert report["with_singletons"]["f1"] == 1.0
        with pytest.raises(ValueError, match="is outside role"):
            orcid_b3.score_block_role(
                arrow,
                tmp_path,
                block,
                roles_path=roles_path,
                role="final_test_a",
                assignments=assignments,
                eps=0.38,
            )


def _mining_input(tmp_path: Path, block: str, *, deposited_orcids: bool = True):
    from scripts.pairwise.mine_pairwise_pairs import BlockInput

    spec = _block_inputs(tmp_path / "block", block, _ORCIDS[0], deposited_orcids=deposited_orcids)
    assignments = tmp_path / "assignments.parquet"
    signatures = ["q_initial", "q_full", "q_outside_home", *(f"s{i:02}" for i in range(29))]
    clusters = ["mixed", "split", "mixed", "mixed", "mixed", "mixed", "mixed", "mixed", *signatures[8:]]
    pd.DataFrame({"signature_id": signatures, "cluster_id": clusters}).to_parquet(assignments)
    labels = Path(spec["labels"])
    frame = pd.read_parquet(labels)
    frame.loc[frame["signature_id"].isin(["q_outside_home", "s03"]), "scored"] = False
    frame.loc[frame["signature_id"].eq("s01"), "orcid"] = _ORCIDS[1]
    frame.to_parquet(labels, index=False)
    return BlockInput(Path(spec["arrow_dir"]), labels, assignments)


def test_pair_miner_cli_resolves_relative_block_directories(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.pairwise.mine_pairwise_pairs import main

    block = "a ross"
    spec = _mining_input(tmp_path, block)
    roles_path, blocks_path, output = (tmp_path / name for name in ("roles.json", "blocks.json", "nested/pairs.csv"))
    roles = assign_block_roles([BlockSpec(block, 32, frozenset({*_ORCIDS, _OUTSIDE_HOME}))], {"fit_pool": 1})
    roles_path.write_text(json.dumps(roles), encoding="utf-8")
    relative = {
        field: getattr(spec, field).relative_to(tmp_path).as_posix() for field in ("arrow_dir", "labels", "assignments")
    }
    blocks_path.write_text(json.dumps({block: relative}), encoding="utf-8")
    assert (
        main(
            [
                "--block-role-assignments",
                str(roles_path),
                "--blocks-json",
                str(blocks_path),
                "--pairs-per-block",
                "6",
                "--verified-pair-fraction",
                "1",
                "--pairs-per-block-floor",
                "0",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    rows = pd.read_csv(output)
    assert json.loads(capsys.readouterr().out) == {"blocks": 1, "pairs": len(rows)}
    assert list(rows["block_key"].unique()) == [block]
    assert rows["stratum"].ne("merged_unlabeled").sum() == 6
    assert rows["stratum"].eq("merged_unlabeled").sum() <= 6
    assert rows["inclusion_probability"].between(0, 1, inclusive="right").all()
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
    assert manifest["pairs_per_block"] == 6
    assert manifest["verified_pair_fraction"] == 1
    assert manifest["pairs_per_block_floor"] == 0


def test_role_b3_rejects_partial_reference_assignments(tmp_path: Path) -> None:
    from scripts.eps.orcid_b3 import score_block_role

    block = "a ross"
    spec = _block_inputs(tmp_path / "block", block, _ORCIDS[0])
    roles_path = tmp_path / "roles.json"
    roles_path.write_text(
        json.dumps(
            assign_block_roles(
                [BlockSpec(block, 32, frozenset({_ORCIDS[0], _OUTSIDE_HOME}))],
                {"dev_test": 1},
            )
        ),
        encoding="utf-8",
    )
    with ArrowDataset.open(spec["arrow_dir"]) as arrow:
        with pytest.raises(ValueError, match="Unmatched external ORCID labels.*1"):
            score_block_role(
                arrow,
                tmp_path,
                block,
                roles_path=roles_path,
                role="dev_test",
                assignments={"q_full": "one"},
                eps=0.38,
                external_labels={"q_full": _ORCIDS[0], "absent": _ORCIDS[0]},
            )
        with pytest.raises(ValueError, match="Incomplete ORCID reference coverage.*missing 6"):
            score_block_role(
                arrow,
                tmp_path,
                block,
                roles_path=roles_path,
                role="dev_test",
                assignments={"q_full": "merged"},
                eps=0.38,
            )


@pytest.mark.parametrize("deposited_orcids", [True, False])
def test_pair_miner_excludes_non_home_before_sampling(tmp_path: Path, deposited_orcids: bool) -> None:
    from s2and._sha256 import sha256_file
    from scripts.pairwise.mine_pairwise_pairs import mine_pairs

    block = "a ross"
    spec = _mining_input(tmp_path, block, deposited_orcids=deposited_orcids)
    roles_path = tmp_path / "roles.json"
    roles_path.write_text(
        json.dumps(
            assign_block_roles(
                [
                    BlockSpec(block, 32, frozenset({*_ORCIDS, _OUTSIDE_HOME})),
                    BlockSpec("sealed", 2, frozenset({_OUTSIDE_HOME})),
                ],
                {"fit_pool": 1, "final_test_a": 1},
                pinned_roles={block: "fit_pool", "sealed": "final_test_a"},
                closure_mode="block_only",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pairs.csv"
    mined = mine_pairs(roles_path, {block: spec}, output, per_block=500, per_identity=2, seed=42)
    rows = pd.read_csv(output)
    assert mined == len(rows)
    assert not set(rows["signature_id_1"]) & {"q_outside_home", "s03"}
    assert not set(rows["signature_id_2"]) & {"q_outside_home", "s03"}
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
    info = manifest["blocks"][block]
    assert manifest["block_role_assignments_sha256"] == sha256_file(roles_path)
    assert manifest["output_sha256"] == sha256_file(output)
    assert info["labels_sha256"] == sha256_file(spec.labels)
    assert info["assignments_sha256"] == sha256_file(spec.assignments)
    assert info["eligible_signatures"] == 30
    assert info["excluded_non_home_signatures"] == 2
    assert info["verified_signatures"] == 4
    assert info["verified_within_block_pairs"] == 6
    assert info["pair_cap"] == 6
    strata = info["strata"]
    fitted = [name for name in strata if name != "merged_unlabeled"]
    # Judge-only pairs have their own budget and never displace ledger-labeled pairs.
    assert sum(strata[name]["sampled_pairs"] for name in fitted) == min(
        6, sum(strata[name]["frame_pairs"] for name in fitted)
    )
    assert strata["merged_unlabeled"]["sampled_pairs"] == min(6, strata["merged_unlabeled"]["frame_pairs"])
    assert strata["merged_unlabeled"]["sampled_pairs"] > 0
    assert mined <= 12
    random = rows.loc[rows["stratum"].eq("random")]
    author = {"q_full", "q_initial", "s00"}
    assert (random["signature_id_1"].isin(author) | random["signature_id_2"].isin(author)).sum() <= 2


def test_role_b3_scores_singleton_only_reference(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.eps.orcid_b3 import main

    block = "a ross"
    identity = _ORCIDS[0]
    spec = _block_inputs(tmp_path / "block", block, identity)
    roles, assignments, labels = (tmp_path / name for name in ("roles.json", "assignments.json", "labels.parquet"))
    roles.write_text(
        json.dumps(assign_block_roles([BlockSpec(block, 32, frozenset({identity}))], {"dev_test": 1})),
        encoding="utf-8",
    )
    assignments.write_text(json.dumps({"q_full": "c"}), encoding="utf-8")
    pd.DataFrame({"signature_id": ["q_full"], "orcid": [identity]}).to_parquet(labels, index=False)
    assert (
        main(
            [
                "--block-role-assignments",
                str(roles),
                "--role",
                "dev_test",
                "--block",
                block,
                "--arrow-dir",
                str(spec["arrow_dir"]),
                "--snapshot-root",
                str(tmp_path),
                "--assignments-json",
                str(assignments),
                "--external-labels",
                str(labels),
                "--eps",
                "0.38",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["reference"]["reference_source"] == "external_labels"
    assert result["with_singletons"]["scored_signatures"] == 1
    assert result["with_singletons"]["f1"] == 1.0
    assert result["repeated"] is None


@pytest.mark.parametrize(
    ("entry_point", "option"),
    [
        ("scripts.eps.orcid_b3", "--assignments-json"),
        ("scripts/linker/build_linker_task_tables.py", "--block-inputs"),
        ("scripts.intake.check_v14_training_overlap", "--training-parquet"),
        ("scripts.pairwise.mine_pairwise_pairs", "--pairs-per-block"),
    ],
    ids=["orcid-b3", "build-linker-task-tables", "check-v14-training-overlap", "mine-pairwise-pairs"],
)
def test_release_tool_entry_point_help(tmp_path: Path, entry_point: str, option: str) -> None:
    root = Path(__file__).resolve().parents[1]
    if entry_point.endswith(".py"):
        # Bypass the editable install's .pth so the script must add the checkout.
        command = [sys.executable, "-S", str(root / entry_point), "--help"]
        env = os.environ | {"PYTHONPATH": sysconfig.get_path("purelib")}
        cwd = tmp_path
    else:
        command = [sys.executable, "-m", entry_point, "--help"]
        env = os.environ.copy()
        cwd = root
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert option in result.stdout

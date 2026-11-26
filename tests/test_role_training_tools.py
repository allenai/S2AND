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
        {"eps_cal": 1, "final_test": 1},
        pinned_roles={block: "eps_cal", "sealed": "final_test"},
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
                role="final_test",
                assignments=assignments,
                eps=0.38,
            )


def test_pair_miner_cli_resolves_relative_block_directories(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.pairwise.mine_pairwise_pairs import main
    from tests.stage0_world import build_stage0_world

    world = build_stage0_world(tmp_path)
    inputs_path, output = tmp_path / "inputs.json", tmp_path / "nested" / "pairs.csv"
    inputs_path.write_text(
        json.dumps(
            {
                block: {"seeds": spec.assignments.relative_to(tmp_path).as_posix()}
                for block, spec in world.inputs.items()
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--block-role-assignments",
                str(world.roles_path),
                "--identity-evidence",
                str(world.evidence_path),
                "--block-inputs",
                str(inputs_path),
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
    assert json.loads(capsys.readouterr().out) == {"blocks": 2, "pairs": len(rows)}
    assert set(rows["block_key"]) == set(world.inputs)
    assert 0 < rows["stratum"].ne("merged_unlabeled").sum() <= 12
    assert rows["stratum"].eq("merged_unlabeled").sum() <= 12
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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2and.arrow_inputs import ArrowDataset
from scripts.production.data.assign_block_roles import BlockSpec, assign_block_roles
from tests.test_build_linker_task_tables import _ORCIDS, _OUTSIDE_HOME, _block_inputs


def test_role_b3_masks_shared_orcids_and_rejects_other_roles(tmp_path: Path) -> None:
    from scripts.eps_sweep import orcid_b3

    block = "a ross"
    spec = _block_inputs(tmp_path / "block", block, _ORCIDS[0])
    roles_path = tmp_path / "roles.json"
    result = assign_block_roles(
        [
            BlockSpec(block, 32, frozenset({_ORCIDS[0], _OUTSIDE_HOME})),
            BlockSpec("sealed", 2, frozenset({_OUTSIDE_HOME})),
        ],
        {"dev_test": 1, "final_test_a": 1},
        pinned_roles={block: "dev_test", "sealed": "final_test_a"},
        closure_mode="home_pool",
        pinned_home_roles={_OUTSIDE_HOME: "final_test_a"},
    )
    roles_path.write_text(json.dumps(result), encoding="utf-8")
    assignments = dict.fromkeys(("q_full", "q_initial", "s00", "s01", "s02", "s03", "q_outside_home"), "merged")
    with ArrowDataset.open(spec["arrow_dir"]) as arrow:
        report = orcid_b3.score_block_role(
            arrow,
            tmp_path,
            block,
            roles_path=roles_path,
            role="dev_test",
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


def test_pair_miner_rejects_sealed_blocks_before_writing(tmp_path: Path) -> None:
    from scripts.production.data import mine_pairwise_pairs

    block = "a ross"
    spec = _block_inputs(tmp_path / "block", block, _ORCIDS[0])
    roles_path = tmp_path / "roles.json"
    frozen = assign_block_roles(
        [BlockSpec(block, 32), BlockSpec("sealed", 5)],
        {"pairwise_train": 1, "final_test_a": 1},
        pinned_roles={block: "pairwise_train", "sealed": "final_test_a"},
    )
    roles_path.write_text(json.dumps(frozen), encoding="utf-8")
    output = tmp_path / "pairs.csv"
    with pytest.raises(ValueError, match="outside pairwise_train"):
        mine_pairwise_pairs.mine_pairs(
            roles_path, {block: Path(spec["arrow_dir"]), "sealed": tmp_path}, output, per_block=6
        )
    assert not output.exists()
    assert mine_pairwise_pairs.mine_pairs(roles_path, {block: Path(spec["arrow_dir"])}, output, per_block=6) == 6
    rows = pd.read_csv(output)
    assert list(rows["block_key"].unique()) == [block]
    assert len(rows[["signature_id_1", "signature_id_2"]].drop_duplicates()) == 6
    assert (rows["signature_id_1"] < rows["signature_id_2"]).all()
    np.testing.assert_allclose(rows["inclusion_probability"], 6 / (32 * 31 / 2))


def test_role_b3_rejects_partial_reference_assignments(tmp_path: Path) -> None:
    from scripts.eps_sweep.orcid_b3 import score_block_role

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


@pytest.mark.parametrize("check", ["sampling_frame", "manifest"])
def test_pair_miner_excludes_non_home_before_sampling(tmp_path: Path, check: str) -> None:
    from s2and._sha256 import sha256_file
    from scripts.production.data.mine_pairwise_pairs import mine_pairs

    block = "a ross"
    spec = _block_inputs(tmp_path / "block", block, _ORCIDS[0])
    roles_path = tmp_path / "roles.json"
    roles_path.write_text(
        json.dumps(
            assign_block_roles(
                [
                    BlockSpec(block, 32, frozenset({_ORCIDS[0], _OUTSIDE_HOME})),
                    BlockSpec("sealed", 2, frozenset({_OUTSIDE_HOME})),
                ],
                {"pairwise_train": 1, "final_test_a": 1},
                pinned_roles={block: "pairwise_train", "sealed": "final_test_a"},
                closure_mode="home_pool",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "pairs.csv"
    mined = mine_pairs(roles_path, {block: Path(spec["arrow_dir"])}, output, per_block=500, seed=42)
    if check == "sampling_frame":
        assert mined == 435
        rows = pd.read_csv(output)
        assert not set(rows["signature_id_1"]) & {"q_outside_home", "s03"}
        assert not set(rows["signature_id_2"]) & {"q_outside_home", "s03"}
        assert rows["inclusion_probability"].eq(1.0).all()
        small = tmp_path / "sample.csv"
        assert mine_pairs(roles_path, {block: Path(spec["arrow_dir"])}, small, per_block=6, seed=42) == 6
        np.testing.assert_allclose(pd.read_csv(small)["inclusion_probability"], 6 / 435)
    else:
        manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))
        assert manifest["seed"] == 42
        assert manifest["block_role_assignments_sha256"] == sha256_file(roles_path)
        assert manifest["output_sha256"] == sha256_file(output)
        assert manifest["blocks"][block] == {
            "eligible_signatures": 30,
            "excluded_non_home_signatures": 2,
            "eligible_pairs": 435,
            "sampled_pairs": 435,
        }


def test_role_b3_scores_singleton_only_reference(tmp_path: Path) -> None:
    from scripts.eps_sweep.orcid_b3 import score_block_role

    block = "a ross"
    identity = _ORCIDS[0]
    spec = _block_inputs(tmp_path / "block", block, identity)
    roles = tmp_path / "roles.json"
    roles.write_text(json.dumps(assign_block_roles([BlockSpec(block, 32, frozenset({identity}))], {"dev_test": 1})))
    with ArrowDataset.open(spec["arrow_dir"]) as arrow:
        result = score_block_role(
            arrow,
            tmp_path,
            block,
            roles_path=roles,
            role="dev_test",
            assignments={"q_full": "c"},
            eps=0.38,
            external_labels={"q_full": identity},
        )
    assert result["with_singletons"]["scored_signatures"] == 1
    assert result["with_singletons"]["f1"] == 1.0
    assert result["repeated"] is None

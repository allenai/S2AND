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

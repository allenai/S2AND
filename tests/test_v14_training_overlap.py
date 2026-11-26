from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.intake.assign_block_roles import BlockSpec, assign_block_roles
from tests.helpers import write_arrow_tables_bundle

pytestmark = pytest.mark.tooling


@pytest.mark.parametrize("orcid_source", [None, "manifest", "arrow"])
@pytest.mark.parametrize("signature_hit", [False, True])
def test_training_overlap_cli_checks_all_sealed_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], signature_hit: bool, orcid_source: str | None
) -> None:
    from scripts.intake.check_v14_training_overlap import main

    orcid_hit = orcid_source is not None
    expected_overlap = signature_hit or orcid_hit

    a_orcid, b_orcid = "0000-0002-1694-233X", "0000-0001-5000-0007"
    arrow_orcid = "0000-0003-0011-7637"
    training_orcid = {"manifest": b_orcid, "arrow": arrow_orcid}.get(orcid_source or "")
    roles_path, blocks_path, train_path = (tmp_path / name for name in ("roles.json", "blocks.json", "train.parquet"))
    roles_path.write_text(
        json.dumps(
            assign_block_roles(
                [
                    BlockSpec("a ross", 1, frozenset({a_orcid}), source_split="holdout"),
                    BlockSpec("b ross", 1, frozenset({b_orcid}), source_split="holdout"),
                ],
                {"final_test": 2},
                pinned_roles={"a ross": "final_test", "b ross": "final_test"},
                enforce_source_splits=True,
            )
        ),
        encoding="utf-8",
    )
    for block, sid in (("a ross", "a-001"), ("b ross", "b-002")):
        tables = {
            "signatures": pa.table(
                {
                    "signature_id": [sid],
                    "paper_id": [sid],
                    "author_first": ["Alice"],
                    "author_middle": [""],
                    "author_last": ["Ross"],
                    "author_suffix": [""],
                    "author_position": [0],
                    "author_block": [block],
                    "author_orcid": pa.array([arrow_orcid if block == "b ross" else None], type=pa.string()),
                    "author_affiliations": pa.array([[]], type=pa.list_(pa.string())),
                }
            ),
            "papers": pa.table({"paper_id": [sid], "title": ["Graph Models"], "venue": [""], "journal_name": [""]}),
            "paper_authors": pa.table({"paper_id": [sid], "position": [0], "author_name": ["Alice Ross"]}),
        }
        write_arrow_tables_bundle(tmp_path / block, tables)
    blocks_path.write_text(json.dumps({"a_ross": "a ross", "b_ross": "b ross"}), encoding="utf-8")
    pd.DataFrame(
        {
            "signature_id": ["train-1", "train-2", "train-3"],
            "paper_id": ["a-001", "a-001", "train"] if signature_hit else ["p1", "p2", "p3"],
            "position": [0, 0, 0],
            "orcid": [None, None, f"https://orcid.org/{training_orcid}" if training_orcid else None],
        }
    ).to_parquet(train_path, index=False)
    assert main(
        [
            "--block-role-assignments",
            str(roles_path),
            "--blocks-json",
            str(blocks_path),
            "--training-parquet",
            str(train_path),
            *(["--allow-no-training-orcids"] if not training_orcid else []),
        ]
    ) == int(expected_overlap)
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] == (not expected_overlap)
    assert report["training_shards"][0]["rows"] == 3
    assert report["distinct_training_orcids"] == int(orcid_hit)
    assert report["sealed_roles"] == {
        "final_test": {
            "signatures": 2,
            "orcids": 3,
            "signature_overlap": 0,
            "paper_position_overlap": int(signature_hit),
            "orcid_overlap": int(orcid_hit),
        },
    }


def test_build_inventory_uses_giant_and_private_training_labels(tmp_path: Path) -> None:
    from scripts.intake.build_v14_training_inventory import build_training_inventory

    gold, extra, out = (tmp_path / name for name in ("gold.parquet", "extra.parquet", "out.parquet"))
    a, b = "0000-0002-1694-233X", "0000-0001-5000-0007"
    pq.write_table(
        pa.table(
            {
                "cluster_block_key": ["a khan", "not a giant"],
                "corpus_paper_id": [31, 32],
                "position": [2, 0],
                "orcid": [a, b],
            }
        ),
        gold,
    )
    pq.write_table(pa.table({"signature_id": ["42-5"], "author_orcid": [f"https://orcid.org/{b}"]}), extra)
    assert build_training_inventory(gold, [extra], out) == {str(gold): 1, str(extra): 1}
    assert pq.read_table(out).to_pylist() == [
        {"signature_id": "31-2", "orcid": a, "paper_id": "31", "position": 2},
        {"signature_id": "42-5", "orcid": b, "paper_id": None, "position": None},
    ]

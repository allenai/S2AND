from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s2and.arrow_tables import write_name_counts_index
from scripts.intake.assign_block_roles import freeze_identity_roles
from scripts.intake.build_freeze_input import build_freeze_input
from scripts.intake.build_identity_evidence import build_identity_evidence
from scripts.intake.requests_to_datasets import convert_requests
from tests.helpers import request_bundle

pytestmark = pytest.mark.tooling


@pytest.mark.parametrize(
    ("orcid_source", "signature_hit"),
    [(None, False), ("disjoint", False), ("disjoint", True), ("manifest", False), ("arrow", False), ("arrow", True)],
)
def test_training_overlap_cli_checks_all_sealed_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], signature_hit: bool, orcid_source: str | None
) -> None:
    from scripts.intake.check_v14_training_overlap import main

    orcid_hit = orcid_source in {"manifest", "arrow"}
    expected_pass = orcid_source == "disjoint" and not signature_hit

    a_orcid, b_orcid = "0000-0002-1694-233X", "0000-0001-5000-0007"
    arrow_orcid = "0000-0003-0011-7637"
    training_orcid = {"manifest": b_orcid, "arrow": arrow_orcid, "disjoint": "0000-0001-5109-3700"}.get(
        orcid_source or ""
    )
    roles_path, train_path = (tmp_path / name for name in ("roles.json", "train.parquet"))
    snapshot = tmp_path / "snapshot"
    index, _ = write_name_counts_index(tmp_path, ({}, {}, {}, {}))
    convert_requests(
        [
            request_bundle(tmp_path, "a ross", [("101", None)]),
            request_bundle(tmp_path, "b ross", [("102", arrow_orcid)]),
        ],
        snapshot,
        Path(index),
        allowed_papers=frozenset({"101", "102"}),
        abstract_allowed_papers=frozenset(),
    )
    labels = tmp_path / "gold.parquet"
    pq.write_table(
        pa.table(
            {
                "cluster_block_key": ["a ross", "b ross"],
                "corpus_paper_id": [101, 102],
                "position": [0, 0],
                "orcid": [a_orcid, b_orcid],
                "split": ["holdout", "holdout"],
                "scored": [True, True],
            }
        ),
        labels,
    )
    evidence = tmp_path / "evidence" / "manifest.json"
    build_identity_evidence(snapshot, labels, evidence.parent)
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"cluster_block_key": ["a ross", "b ross"], "split": ["holdout", "holdout"]}), source)
    blocks = tmp_path / "freeze.parquet"
    build_freeze_input(evidence, source, blocks)
    roles_path.write_text(json.dumps(freeze_identity_roles(blocks, evidence, {"final_test": 2})), encoding="utf-8")
    pd.DataFrame(
        {
            "signature_id": ["train-1", "train-2", "train-3"],
            "paper_id": ["101", "101", "train"] if signature_hit else ["p1", "p2", "p3"],
            "position": [0, 0, 0],
            "orcid": [None, None, f"https://orcid.org/{training_orcid}" if training_orcid else None],
        }
    ).to_parquet(train_path, index=False)
    assert main(
        [
            "--block-role-assignments",
            str(roles_path),
            "--identity-evidence",
            str(evidence),
            "--training-parquet",
            str(train_path),
        ]
    ) == int(not expected_pass)
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] == expected_pass
    assert report["training_shards"][0]["rows"] == 3
    assert report["distinct_training_orcids"] == int(training_orcid is not None)
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

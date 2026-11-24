from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from scripts.production.data.assign_block_roles import BlockSpec, assign_block_roles
from scripts.production.data.block_roles import load_block_roles
from scripts.production.data.prepare_identity_labels import (
    names_are_compatible,
    normalize_block_key,
    prepare_identity_labels,
)


def _write_signatures_arrow(directory: Path, rows: list[dict]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    path = directory / "signatures.arrow"
    with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return path


def test_identity_label_join_scores_shared_orcid_only_in_home_role(tmp_path: Path) -> None:
    shared = "0000-0002-1694-233X"
    assignments = assign_block_roles(
        [BlockSpec("a ross", 1, frozenset({shared})), BlockSpec("b ross", 1, frozenset({shared}))],
        {"eps_cal": 1, "final_test_a": 1},
        pinned_roles={"a ross": "eps_cal", "b ross": "final_test_a"},
        closure_mode="home_pool",
    )
    path = tmp_path / "assignments.json"
    path.write_text(json.dumps(assignments), encoding="utf-8")
    roles = load_block_roles(path)
    gold = tmp_path / "gold.parquet"
    pd.DataFrame(
        {
            "paper_id": ["1", "2"],
            "position": [0, 0],
            "orcid": [shared, shared],
            "block_key": ["a ross", "b ross"],
            "scored": [True, True],
        }
    ).to_parquet(gold, index=False)
    for idx, block in enumerate(("a_ross", "b_ross"), 1):
        directory = tmp_path / block
        arrow_path = _write_signatures_arrow(
            directory,
            [
                {
                    "signature_id": f"s{idx}",
                    "paper_id": str(idx),
                    "author_position": 0,
                    "author_block": block,
                    "author_orcid": shared,
                }
            ],
        )
        output = directory / "labels.parquet"
        audit = prepare_identity_labels(directory, gold, output, block_roles=roles)
        assert audit["non_home_verified_rows"] == (1 if idx == 1 else 0)
        labels = pd.read_parquet(output)
        assert labels["signature_id"].tolist() == [f"s{idx}"]
        assert labels["scored"].tolist() == [idx == 2]
        assert pa.ipc.open_file(str(arrow_path)).read_all()["author_orcid"].to_pylist() == [shared]


def test_normalize_block_key_and_name_compatibility() -> None:
    assert normalize_block_key("  A_Khan ") == "a khan"
    assert normalize_block_key("a   khan") == "a khan"
    assert names_are_compatible("Alice", "M", "Smith", "A", None, "Smith")
    assert names_are_compatible("A", None, "Smith", "Alice", "Marie", "Smith")
    assert names_are_compatible("Jean-Luc", None, "Ou Yang", "Jean", None, "Ouyang")
    assert names_are_compatible("Hans", None, "Mueller", "Hans", None, "Muller")
    assert not names_are_compatible("Alice", "M", "Smith", "Alice", "K", "Smith")
    assert not names_are_compatible("Alice", None, "Smith", "Bob", None, "Smith")
    assert not names_are_compatible("Alice", None, "Smith", "Alice", None, "Jones")


def test_prepare_identity_labels_enforces_all_verification_guards(tmp_path: Path) -> None:
    arrow_dir = tmp_path / "a_smith"
    _write_signatures_arrow(
        arrow_dir,
        [
            {
                "signature_id": "s_ok1",
                "paper_id": "101",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_ok2",
                "paper_id": "102",
                "author_position": 1,
                "author_block": "a smith",
                "author_first": "A",
                "author_middle": "M",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_unscored",
                "paper_id": "103",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_null_scored",
                "paper_id": "1031",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_wrong_split",
                "paper_id": "1032",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_wrong_block",
                "paper_id": "104",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_wrong_name",
                "paper_id": "105",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_dup_block_1",
                "paper_id": "106",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_dup_block_2",
                "paper_id": "106",
                "author_position": 2,
                "author_block": "a smith",
                "author_first": "Alan",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_excluded_orcid",
                "paper_id": "107",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_reviewed_err",
                "paper_id": "108",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_conflict",
                "paper_id": "109",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_frac_pos",
                "paper_id": "110",
                "author_position": 1,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
        ],
    )

    valid_orcid = "0000-0002-1694-233X"
    other_orcid = "0000-0001-5000-0007"
    gold_path = tmp_path / "gold.parquet"
    # Test int64 corpus_paper_id (matching real orcid-blocks-v2), nullable scored, split, and float position!
    pd.DataFrame(
        [
            {
                "corpus_paper_id": 101,
                "position": 0.0,
                "orcid": f"https://orcid.org/{valid_orcid}",
                "scored": True,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": 102,
                "position": 1.0,
                "orcid": valid_orcid,
                "scored": True,
                "split": "development",
                "cluster_block_key": "a smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": 103,
                "position": 0.0,
                "orcid": valid_orcid,
                "scored": False,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            # Retain the identity and leave a null scored flag ineligible.
            {
                "corpus_paper_id": 1031,
                "position": 0.0,
                "orcid": valid_orcid,
                "scored": None,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            # Wrong split must be rejected when expected_split="development".
            {
                "corpus_paper_id": 1032,
                "position": 0.0,
                "orcid": valid_orcid,
                "scored": True,
                "split": "holdout",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": 104,
                "position": 0.0,
                "orcid": valid_orcid,
                "scored": True,
                "split": "development",
                "cluster_block_key": "b_jones",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": 105,
                "position": 0.0,
                "orcid": valid_orcid,
                "scored": True,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Bob",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": 106,
                "position": 0.0,
                "orcid": valid_orcid,
                "scored": True,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": 107,
                "position": 0.0,
                "orcid": "0000-0002-1825-0097",
                "scored": True,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": 108,
                "position": 0.0,
                "orcid": other_orcid,
                "scored": True,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": 109,
                "position": 0.0,
                "orcid": valid_orcid,
                "scored": True,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": 109,
                "position": 0.0,
                "orcid": other_orcid,
                "scored": True,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            # Fractional position 1.9 must NOT truncate to 1!
            {
                "corpus_paper_id": 110,
                "position": 1.9,
                "orcid": valid_orcid,
                "scored": True,
                "split": "development",
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
        ]
    ).to_parquet(gold_path, index=False)

    out_parquet = tmp_path / "verified.parquet"
    audit_json = tmp_path / "audit.json"
    audit = prepare_identity_labels(
        arrow_dir,
        gold_path,
        out_parquet,
        audit_json_path=audit_json,
        error_signature_ids={"s_reviewed_err"},
        expected_split="development",
    )

    verified = pd.read_parquet(out_parquet)
    assert verified.set_index("signature_id")["scored"].to_dict() == {
        "s_ok1": True,
        "s_ok2": True,
        "s_unscored": False,
        "s_null_scored": False,
    }
    verified = verified.loc[verified["scored"]]
    assert verified["orcid"].tolist() == [valid_orcid, valid_orcid]
    assert verified["split"].tolist() == ["development", "development"]
    assert audit["dropped_wrong_split"] == 1
    assert audit["gold_unscored_verified_rows"] == 2
    assert audit["dropped_invalid_position"] == 1
    # s_wrong_block ('b_jones') is excluded at the Parquet block-key pushdown level before scanning!
    assert "s_wrong_block" not in set(verified["signature_id"])
    assert audit["dropped_block_key_mismatch"] == 0
    assert audit["dropped_name_mismatch"] == 1
    assert audit["dropped_ambiguous_paper_block"] == 1
    assert audit["dropped_excluded_orcid_group"] == 1
    assert audit["dropped_reviewed_error_signature"] == 1
    assert audit["dropped_conflicting_gold_rows"] == 2
    assert audit["scored_rows"] == 2
    assert audit["repeated_orcids"] == 1
    assert audit["verification_mode"] == "position_block_and_name"
    assert json.loads(audit_json.read_text(encoding="utf-8"))["verified_rows"] == 4


def test_duplicate_verified_labels_keep_any_scored_evidence(tmp_path: Path) -> None:
    orcid = "0000-0002-1694-233X"
    arrow = tmp_path / "a_ross"
    _write_signatures_arrow(
        arrow,
        [
            {
                "signature_id": "s1",
                "paper_id": "1",
                "author_position": 0,
                "author_block": "a ross",
                "author_orcid": orcid,
            }
        ],
    )
    gold = tmp_path / "gold.parquet"
    row = {"paper_id": "1", "position": 0, "orcid": orcid, "block_key": "a ross"}
    pd.DataFrame([{**row, "scored": False}, {**row, "scored": True}]).to_parquet(gold)
    output = tmp_path / "labels.parquet"
    prepare_identity_labels(arrow, gold, output)
    assert pd.read_parquet(output)[["signature_id", "scored"]].values.tolist() == [["s1", True]]

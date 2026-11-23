from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

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


def test_normalize_block_key_and_name_compatibility() -> None:
    assert normalize_block_key("  A_Khan ") == "a khan"
    assert normalize_block_key("a   khan") == "a khan"
    assert names_are_compatible("Alice", "M", "Smith", "A", None, "Smith")
    assert names_are_compatible("A", None, "Smith", "Alice", "Marie", "Smith")
    assert not names_are_compatible("Alice", None, "Smith", "Bob", None, "Smith")
    assert not names_are_compatible("Alice", None, "Smith", "Alice", None, "Jones")


def test_prepare_identity_labels_enforces_all_verification_guards(tmp_path: Path) -> None:
    arrow_dir = tmp_path / "a_smith"
    _write_signatures_arrow(
        arrow_dir,
        [
            {
                "signature_id": "s_ok1",
                "paper_id": "p1",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_ok2",
                "paper_id": "p2",
                "author_position": 1,
                "author_block": "a smith",
                "author_first": "A",
                "author_middle": "M",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_unscored",
                "paper_id": "p3",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_wrong_block",
                "paper_id": "p4",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_wrong_name",
                "paper_id": "p5",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_dup_block_1",
                "paper_id": "p6",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_dup_block_2",
                "paper_id": "p6",
                "author_position": 2,
                "author_block": "a smith",
                "author_first": "Alan",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_excluded_orcid",
                "paper_id": "p7",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_reviewed_err",
                "paper_id": "p8",
                "author_position": 0,
                "author_block": "a smith",
                "author_first": "Alice",
                "author_middle": "",
                "author_last": "Smith",
            },
            {
                "signature_id": "s_conflict",
                "paper_id": "p9",
                "author_position": 0,
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
    pd.DataFrame(
        [
            # Two valid matches for same ORCID (repeated identity).
            {
                "corpus_paper_id": "p1",
                "position": 0,
                "orcid": f"https://orcid.org/{valid_orcid}",
                "scored": True,
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": "p2",
                "position": 1,
                "orcid": valid_orcid,
                "scored": True,
                "cluster_block_key": "a smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            # Unscored row dropped.
            {
                "corpus_paper_id": "p3",
                "position": 0,
                "orcid": valid_orcid,
                "scored": False,
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            # Block mismatch dropped.
            {
                "corpus_paper_id": "p4",
                "position": 0,
                "orcid": valid_orcid,
                "scored": True,
                "cluster_block_key": "b_jones",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            # Name mismatch dropped.
            {
                "corpus_paper_id": "p5",
                "position": 0,
                "orcid": valid_orcid,
                "scored": True,
                "cluster_block_key": "a_smith",
                "author_first": "Bob",
                "author_last": "Smith",
            },
            # Ambiguous two-author same-block paper dropped.
            {
                "corpus_paper_id": "p6",
                "position": 0,
                "orcid": valid_orcid,
                "scored": True,
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            # Shared/excluded ORCID group dropped.
            {
                "corpus_paper_id": "p7",
                "position": 0,
                "orcid": "0000-0002-1825-0097",
                "scored": True,
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            # Reviewed wrong-ORCID signature dropped.
            {
                "corpus_paper_id": "p8",
                "position": 0,
                "orcid": other_orcid,
                "scored": True,
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            # Two conflicting gold ORCIDs for the same (paper, position) dropped.
            {
                "corpus_paper_id": "p9",
                "position": 0,
                "orcid": valid_orcid,
                "scored": True,
                "cluster_block_key": "a_smith",
                "author_first": "Alice",
                "author_last": "Smith",
            },
            {
                "corpus_paper_id": "p9",
                "position": 0,
                "orcid": other_orcid,
                "scored": True,
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
    )

    verified = pd.read_parquet(out_parquet)
    assert verified["signature_id"].tolist() == ["s_ok1", "s_ok2"]
    assert verified["orcid"].tolist() == [valid_orcid, valid_orcid]
    assert audit["dropped_unscored"] == 1
    assert audit["dropped_block_key_mismatch"] == 1
    assert audit["dropped_name_mismatch"] == 1
    assert audit["dropped_ambiguous_paper_block"] == 1
    assert audit["dropped_excluded_orcid_group"] == 1
    assert audit["dropped_reviewed_error_signature"] == 1
    assert audit["dropped_conflicting_gold_rows"] == 2
    assert audit["verified_rows"] == 2
    assert audit["repeated_orcids"] == 1
    assert json.loads(audit_json.read_text(encoding="utf-8"))["verified_rows"] == 2

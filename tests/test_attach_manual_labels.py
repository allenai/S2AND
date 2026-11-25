from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.intake import build_identity_evidence
from tests.test_identity_ledger import FIXTURE_UNMATCHED_LIMIT
from tests.test_identity_ledger import evidence_inputs as evidence_inputs

pytestmark = pytest.mark.tooling


@pytest.mark.parametrize(
    ("authors", "expected"),
    [
        ([(2, "s2", "A.", "O’Neil"), (4, "s4", "Anne", "O'Neil")], ("exact", "s2")),
        ([(2, "s2", "Ana", "Smith"), (4, "s4", "O’Neil", "Ana")], ("shifted", "s4")),
        ([(1, "s1", "Ana", "O'Neil"), (4, "s4", "Anne", "O’Neil")], ("ambiguous", None)),
        ([(1, "s1", "Beth", "O’Neil"), (4, "s4", "Ana", "Smith")], ("orphaned", None)),
        ([], ("orphaned", None)),
    ],
)
def test_resolve_anchor_status(authors: list[tuple[int, str, str, str]], expected: tuple[str, str | None]) -> None:
    from scripts.intake.attach_manual_labels import _author_name_keys, _resolve_anchor

    names = {(position, signature): _author_name_keys(first, last) for position, signature, first, last in authors}
    assert _resolve_anchor(2, "Ána", "O'Neil", names, {}) == expected


def test_unexported_author_prevents_guessing_a_shift() -> None:
    from scripts.intake.attach_manual_labels import _author_name_keys, _resolve_anchor

    names = {(3, "s3"): _author_name_keys("A.", "Lovelace")}
    assert _resolve_anchor(1, "Ada", "Lovelace", names, {2: {"ada m lovelace"}}) == ("ambiguous", None)
    assert _resolve_anchor(1, "Ada", "Lovelace", names, {1: {"lovelace ada"}}) == ("orphaned", None)


def test_attach_manual_labels_writes_gated_rows_and_provenance(
    tmp_path: Path, evidence_inputs: tuple[Path, Path]
) -> None:
    from s2and._sha256 import sha256_file
    from scripts.intake.attach_manual_labels import attach_manual_labels

    snapshot, gold = evidence_inputs
    evidence = tmp_path / "evidence" / "manifest.json"
    build_identity_evidence.build_identity_evidence(
        snapshot, gold, evidence.parent, max_unmatched_gold=FIXTURE_UNMATCHED_LIMIT
    )
    anchors, pairs, changes = (tmp_path / f"{name}.parquet" for name in ("anchors", "pairs", "changes"))
    pq.write_table(
        pa.table(
            {
                "corpus_paper_id": [1, 2, 3, 100],
                "position": [0, 1, 0, 0],
                "first": ["Ada"] * 4,
                "last": ["Lovelace"] * 4,
                "anchor_status": ["anchored"] * 4,
            }
        ),
        anchors,
    )
    pq.write_table(
        pa.table(
            {
                "source_id": ["review"] * 4,
                "id_space": ["corpus"] * 4,
                "usable": [True, True, True, False],
                "corpus_paper_id_a": [1] * 4,
                "position_a": [0] * 4,
                "corpus_paper_id_b": [3, 2, 100, 3],
                "position_b": [0, 1, 0, 0],
            }
        ),
        pairs,
    )
    pq.write_table(
        pa.table(
            {
                "source_id": ["review"] * 3,
                "id_space": ["corpus"] * 3,
                "usable": [True] * 3,
                "member_signature_ids": [["1-0", "3-0"], ["1-0", "2-1"], ["1-0", "100-0"]],
            }
        ),
        changes,
    )
    output = tmp_path / "attached"
    report = attach_manual_labels(anchors, pairs, changes, snapshot, evidence, output)
    remapped_pairs = pq.read_table(output / "pair_labels.parquet")
    remapped_changes = pq.read_table(output / "cluster_changes.parquet")
    assert remapped_pairs["status"].to_pylist() == ["exact", "shifted", "orphaned", "exact"]
    assert remapped_pairs["snapshot_signature_id_a"].to_pylist() == ["1-0", "1-0", None, None]
    assert remapped_pairs["snapshot_signature_id_b"].to_pylist() == ["3-0", "2-0", None, None]
    assert remapped_changes["snapshot_member_signature_ids"].to_pylist() == [["1-0", "3-0"], ["1-0", "2-0"], None]
    assert report["pair_labels"]["by_source"]["review"] == {"exact": 2, "shifted": 1, "ambiguous": 0, "orphaned": 1}
    assert report["pair_labels"]["attached"] == 2
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    assert all(
        sha256_file(path) in manifest for path in (anchors, pairs, changes, evidence, snapshot / "manifest.json")
    )

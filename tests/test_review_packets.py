from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.production.data.review_packets import (
    ORCID_REDACTED_TOKEN,
    build_blind_linker_packets,
    build_blind_pair_packets,
    validate_blind_packet_records,
)


def _paper(idx: int, *, email: bool = False, text_orcid: bool = False) -> dict:
    return {
        "signature_id": f"visibleq:s{idx}",
        "paper_id": f"visiblep:p{idx}",
        "title": f"Paper {idx}" + (" contact a@b.edu" if email else ""),
        "abstract": f"Abstract {idx}" + (" ORCID 000000021694233X" if text_orcid else ""),
        "venue": "NeurIPS",
        "year": 2020 + (idx % 5),
        "author_name": f"A. Author{idx}",
        # S2AND papers.json stores author dicts with author_id / s2_author_ids -> must flatten to names only!
        "authors": [{"author_name": f"A. Author{idx}", "author_id": "9999", "decision": "same author"}],
        "affiliations": ["Univ"],
        "coauthors": ["Coauthor X"],
        "author_orcid": "0000-0002-1694-233X",
        "cluster_id": f"cluster_{idx}",
        "cluster_summary": "LEAK",
        "score": 0.99,
    }


def test_build_blind_pair_packets_strips_leaks_isolates_key_and_stratifies_controls(tmp_path: Path) -> None:
    items = [
        {
            "source_id": f"src_{i}",
            "block_key": "a_smith",
            "paper_a": _paper(2 * i, email=True, text_orcid=True),
            "paper_b": _paper(2 * i + 1),
        }
        for i in range(12)
    ]
    controls = [
        {
            "source_id": f"ctrl_{i}",
            "block_key": "a_smith",
            "paper_a": _paper(100 + 2 * i),
            "paper_b": _paper(100 + 2 * i + 1),
            "metadata": {"gold_label": i % 2},
        }
        for i in range(6)
    ]
    out1 = tmp_path / "pair_packets_pass1"
    key1 = tmp_path / "private" / "pass1_key.json"
    summary1 = build_blind_pair_packets(
        items=items,
        control_items=controls,
        output_dir=out1,
        answer_key_path=key1,
        seed=101,
        control_seed=999,
        control_fraction=0.2,
        max_items_per_shard=5,
    )
    assert summary1["total_items"] == 15
    assert summary1["control_items"] == 3
    assert summary1["shard_count"] == 3
    # Answer key is stored strictly outside output_dir.
    assert not (out1 / "answer_key.json").exists()
    assert key1.is_file()

    # Every emitted blind shard passes strict leak validation with zero IDs/ORCIDs/author_ids/emails.
    all_records = []
    for rel in summary1["shards"]:
        raw_text = (out1 / rel).read_text(encoding="utf-8")
        assert "0000-0002-1694-233X" not in raw_text
        assert "000000021694233X" not in raw_text
        assert ORCID_REDACTED_TOKEN in raw_text
        assert "visibleq:" not in raw_text
        assert "author_id" not in raw_text
        assert "cluster_summary" not in raw_text
        assert "a@b.edu" not in raw_text
        for line in raw_text.splitlines():
            rec = json.loads(line)
            assert rec["paper_a"]["authors"] == [rec["paper_a"]["author_name"]]
            all_records.append(rec)
    assert validate_blind_packet_records(all_records, packet_type="pair") == []

    # Controls are stratified evenly: 1 control per each of the 3 shards!
    key1_data = json.loads(key1.read_text(encoding="utf-8"))
    by_id = {row["item_id"]: row for row in key1_data["items"]}
    for rel in summary1["shards"]:
        shard_ids = [json.loads(line)["item_id"] for line in (out1 / rel).read_text(encoding="utf-8").splitlines()]
        assert sum(by_id[iid]["is_control"] for iid in shard_ids) == 1

    # Second pass with new order seed + same control_seed freezes exact control set!
    out2 = tmp_path / "pair_packets_pass2"
    key2 = tmp_path / "private" / "pass2_key.json"
    build_blind_pair_packets(
        items=items,
        control_items=controls,
        output_dir=out2,
        answer_key_path=key2,
        seed=202,
        control_seed=999,
        control_fraction=0.2,
        max_items_per_shard=5,
    )
    key2_data = json.loads(key2.read_text(encoding="utf-8"))
    ctrl_ids_1 = {row["source_id"] for row in key1_data["items"] if row["is_control"]}
    ctrl_ids_2 = {row["source_id"] for row in key2_data["items"] if row["is_control"]}
    assert ctrl_ids_1 == ctrl_ids_2

    # Refuses reusing existing output_dir or putting answer_key inside output_dir.
    with pytest.raises(FileExistsError):
        build_blind_pair_packets(items=items, output_dir=out1)
    with pytest.raises(ValueError, match="outside the blind packet directory"):
        build_blind_pair_packets(items=items, output_dir=tmp_path / "bad", answer_key_path=tmp_path / "bad" / "k.json")


def test_build_blind_linker_packets_and_validator_catches_nested_leaks(tmp_path: Path) -> None:
    queries = [
        {
            "source_id": f"q_{i}",
            "block_key": "b_lee",
            "query": _paper(i),
            "candidates": [
                {"candidate_key": f"cand_{i}_a", "exemplars": [_paper(50 + i)]},
                {"candidate_key": f"cand_{i}_b", "exemplars": [_paper(60 + i)]},
            ],
        }
        for i in range(6)
    ]
    out = tmp_path / "linker_packets"
    summary = build_blind_linker_packets(queries=queries, output_dir=out, seed=7, max_items_per_shard=4)
    assert summary["shard_count"] == 2
    assert Path(summary["answer_key_path"]).is_file()

    records = [
        json.loads(line) for rel in summary["shards"] for line in (out / rel).read_text(encoding="utf-8").splitlines()
    ]
    assert validate_blind_packet_records(records, packet_type="linker") == []
    assert all([c["candidate_slot"] for c in rec["candidates"]] == ["C1", "C2"] for rec in records)

    # Validator catches nested author dicts, compact ORCIDs, forbidden keys, and plaintext emails.
    leaky = [
        {
            "item_id": "pair_1",
            "block_key": "a_smith",
            "paper_a": {
                "title": "Compact 000000021694233X",
                "authors": [{"decision": "same author", "identity": "0000000218250097"}],
            },
            "paper_b": {"title": "Contact foo@bar.edu"},
            "model_score": 0.9,
        }
    ]
    violations = validate_blind_packet_records(leaky, packet_type="pair")
    assert any("disallowed top-level keys" in v for v in violations)
    assert any("must be a list of plain strings" in v for v in violations)
    assert any("ORCID pattern" in v for v in violations)
    assert any("plaintext email" in v for v in violations)
    assert any("forbidden key" in v for v in violations)

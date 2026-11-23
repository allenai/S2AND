from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.production.data.review_packets import (
    build_blind_linker_packets,
    build_blind_pair_packets,
    validate_blind_packet_records,
)


def _paper(idx: int, *, email: bool = False) -> dict:
    return {
        "signature_id": f"s{idx}",
        "paper_id": f"p{idx}",
        "title": f"Paper {idx}" + (" contact a@b.edu" if email else ""),
        "abstract": f"Abstract {idx}",
        "venue": "NeurIPS",
        "year": 2020 + (idx % 5),
        "author_name": f"A. Author{idx}",
        "affiliations": ["Univ"],
        "coauthors": ["Coauthor X"],
        # Deliberately include forbidden source fields that sanitizer must strip!
        "author_orcid": "0000-0002-1694-233X",
        "cluster_id": f"cluster_{idx}",
        "cluster_summary": "LEAK",
        "score": 0.99,
    }


def test_build_blind_pair_packets_strips_leaks_randomizes_and_injects_controls(tmp_path: Path) -> None:
    items = [
        {
            "source_id": f"src_{i}",
            "block_key": "a_smith",
            "paper_a": _paper(2 * i, email=True),
            "paper_b": _paper(2 * i + 1),
        }
        for i in range(10)
    ]
    controls = [
        {
            "source_id": f"ctrl_{i}",
            "block_key": "a_smith",
            "paper_a": _paper(100 + 2 * i),
            "paper_b": _paper(100 + 2 * i + 1),
            "metadata": {"gold_label": i % 2},
        }
        for i in range(4)
    ]
    out = tmp_path / "pair_packets"
    summary = build_blind_pair_packets(
        items=items,
        control_items=controls,
        output_dir=out,
        seed=123,
        control_fraction=0.15,
        max_items_per_shard=5,
    )
    assert summary["total_items"] == 12
    assert summary["control_items"] == 2
    assert summary["shard_count"] == 3

    # Every emitted blind shard must pass strict leak validation and contain no ORCIDs/scores/clusters.
    all_records = []
    for rel in summary["shards"]:
        raw_text = (out / rel).read_text(encoding="utf-8")
        assert "0000-0002-1694-233X" not in raw_text
        assert "cluster_summary" not in raw_text
        assert "a@b.edu" not in raw_text
        for line in raw_text.splitlines():
            all_records.append(json.loads(line))
    assert len(all_records) == 12
    assert validate_blind_packet_records(all_records, packet_type="pair") == []

    answer_key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    assert answer_key["control_items"] == 2
    assert any(row["sides_swapped"] for row in answer_key["items"])
    assert any(not row["sides_swapped"] for row in answer_key["items"])


def test_build_blind_linker_packets_anonymizes_candidates_and_validator_catches_leaks(tmp_path: Path) -> None:
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

    records = [
        json.loads(line) for rel in summary["shards"] for line in (out / rel).read_text(encoding="utf-8").splitlines()
    ]
    assert validate_blind_packet_records(records, packet_type="linker") == []
    assert all([c["candidate_slot"] for c in rec["candidates"]] == ["C1", "C2"] for rec in records)

    # Validator must catch forbidden keys, ORCID strings, and plaintext emails.
    leaky = [
        {
            "item_id": "pair_1",
            "block_key": "a_smith",
            "paper_a": {"title": "ORCID 0000-0002-1694-233X", "cluster_summary": "bad"},
            "paper_b": {"title": "Contact foo@bar.edu"},
            "model_score": 0.9,
        }
    ]
    violations = validate_blind_packet_records(leaky, packet_type="pair")
    assert any("disallowed top-level keys" in v for v in violations)
    assert any("ORCID pattern" in v for v in violations)
    assert any("plaintext email" in v for v in violations)
    assert any("forbidden key" in v for v in violations)

    with pytest.raises(ValueError, match="Single record exceeds"):
        build_blind_pair_packets(
            items=[{"source_id": "s", "block_key": "a", "paper_a": _paper(1), "paper_b": _paper(2)}],
            output_dir=tmp_path / "tiny_shard",
            max_bytes_per_shard=20,
        )

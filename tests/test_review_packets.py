from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.production.data.assign_block_roles import BlockSpec, assign_block_roles
from scripts.production.data.block_roles import load_block_roles
from scripts.production.data.review_packets import (
    ORCID_REDACTED_TOKEN,
    build_blind_linker_packets,
    build_blind_pair_packets,
    main,
    sanitize_paper_item,
    score_judges,
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


@pytest.mark.parametrize("year", [1601, " 1800 "])
def test_blind_review_hides_corpus_placeholder_years(year: int | str) -> None:
    record = {"item_id": "test", "block_key": "a smith", "paper_a": {"year": year}, "paper_b": {"year": 1801}}
    assert validate_blind_packet_records([record], packet_type="pair")
    record["paper_a"] = sanitize_paper_item(record["paper_a"])
    assert record["paper_a"] == {"year": None}
    assert sanitize_paper_item(record["paper_b"]) == {"year": 1801}
    assert validate_blind_packet_records([record], packet_type="pair") == []


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


def test_review_controls_require_home_role_and_keep_other_mentions_blind(tmp_path: Path) -> None:
    shared = "0000-0002-1694-233X"
    local = "0000-0001-5000-0007"
    assignments = assign_block_roles(
        [BlockSpec("a smith", 5, frozenset({shared, local})), BlockSpec("b smith", 2, frozenset({shared}))],
        {"gate_thresh": 1, "final_test_a": 1},
        pinned_roles={"a smith": "gate_thresh", "b smith": "final_test_a"},
        closure_mode="home_pool",
    )
    path = tmp_path / "roles.json"
    path.write_text(json.dumps(assignments), encoding="utf-8")
    roles = load_block_roles(path)
    shared_paper = _paper(1)
    gold_paper = {**_paper(2), "author_orcid": local}
    item = {"source_id": "natural", "block_key": "a smith", "paper_a": shared_paper, "paper_b": gold_paper}
    controls = [
        {**item, "source_id": "outside_home"},
        {**item, "source_id": "inside_home", "paper_a": gold_paper, "metadata": {"gold_label": 1}},
        {**item, "source_id": "unscored", "paper_a": {**gold_paper, "scored": False}},
        {**item, "source_id": "sealed", "block_key": "b smith", "paper_b": shared_paper},
    ]
    result = build_blind_pair_packets(
        items=[item],
        control_items=controls,
        output_dir=tmp_path / "packets",
        block_roles=roles,
        allowed_roles={"gate_thresh"},
        control_fraction=0.5,
    )
    key = json.loads(Path(result["answer_key_path"]).read_text(encoding="utf-8"))
    assert result["excluded_controls"] == 3
    assert {row["source_id"] for row in key["items"]} == {"natural", "inside_home"}
    with pytest.raises(ValueError, match="allowed packet roles"):
        build_blind_pair_packets(
            items=[{**item, "block_key": "b smith"}],
            output_dir=tmp_path / "rejected",
            block_roles=roles,
            allowed_roles={"gate_thresh"},
        )


def test_score_reports_controls_decided_share_and_directional_skew(tmp_path: Path) -> None:
    def make_item(source_id: str, *, label: int | None = None) -> dict:
        return {
            "source_id": source_id,
            "block_key": "a lee",
            "paper_a": _paper(1),
            "paper_b": _paper(2),
            "metadata": {"stratum": "giant", "gold_label": label},
        }

    regular = [make_item(source) for source in ("first", "second", "third")]
    controls = [make_item(source, label=label) for source, label in (("true_same", 1), ("true_different", 0))]
    verdicts_by_family = {
        "sol": {
            "first": "same",
            "second": "different",
            "third": "insufficient_evidence",
            "true_same": "same",
            "true_different": "same",
        },
        "claude": {"first": "same", "second": "different", "true_same": "different", "true_different": "different"},
    }
    arguments = ["score"]
    for idx, (family, verdicts) in enumerate(verdicts_by_family.items()):
        report = build_blind_pair_packets(
            items=regular,
            control_items=controls,
            output_dir=tmp_path / family,
            control_fraction=0.5,
            seed=100 + idx,
            control_seed=10,
            max_items_per_shard=3,
        )
        answer_key = Path(report["answer_key_path"])
        items = json.loads(answer_key.read_text(encoding="utf-8"))["items"]
        path = tmp_path / f"{family}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"item_id": row["item_id"], "verdict": verdicts[row["source_id"]]}) + "\n"
                for row in items
                if row["source_id"] in verdicts
            ),
            encoding="utf-8",
        )
        arguments.extend(["--judge", family, str(path), str(answer_key)])
    result_path = tmp_path / "score.json"
    assert main([*arguments, "--output-json", str(result_path)]) == 0
    summary = json.loads(result_path.read_text(encoding="utf-8"))
    for judge, skew, missing in zip(summary["judges"], (1, -1), (0, 1), strict=True):
        assert len(judge["by_shard"]) == 2
        assert sorted(report["control_accuracy"] for report in judge["by_shard"].values()) == [0, 1]
        assert judge["by_stratum"]["giant"]["decided_share"] == 0.8
        assert judge["by_stratum"]["giant"]["false_same_minus_false_different"] == skew
        assert judge["by_stratum"]["giant"]["missing"] == missing
    agreement = summary["cross_family_agreement"][0]
    assert agreement["both_decided_share"] == 0.8
    assert agreement["agreement"] == 0.5
    assert agreement["cohen_kappa"] == 0.2


def test_score_aligns_linker_slots_across_packet_orders(tmp_path: Path) -> None:
    item = {
        "source_id": "real_query",
        "block_key": "a smith",
        "query": _paper(1),
        "candidates": [
            {"candidate_key": "a", "exemplars": [_paper(2)]},
            {"candidate_key": "b", "exemplars": [_paper(3)]},
        ],
    }
    judges = []
    orders = []
    for family, seed in (("sol", 1), ("claude", 5)):
        summary = build_blind_linker_packets(queries=[item], output_dir=tmp_path / family, seed=seed)
        key_path = Path(summary["answer_key_path"])
        (record,) = json.loads(key_path.read_text(encoding="utf-8"))["items"]
        slots = record["slot_to_candidate_key"]
        orders.append(slots)
        verdicts = {
            "item_id": record["item_id"],
            "candidate_verdicts": {
                slot: "same" if candidate == "a" else "different" for slot, candidate in slots.items()
            },
        }
        verdicts_path = tmp_path / f"{family}.jsonl"
        verdicts_path.write_text(json.dumps(verdicts) + "\n", encoding="utf-8")
        judges.append((family, verdicts_path, key_path))
    assert orders[0] != orders[1]
    assert score_judges(judges)["cross_family_agreement"][0]["agreement"] == 1.0
    with pytest.raises(ValueError, match="distinct judge families"):
        score_judges([judges[0], judges[0]])

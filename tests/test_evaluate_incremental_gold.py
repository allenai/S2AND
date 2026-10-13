from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and import memory_budget
from s2and import production_model as production_model_module
from scripts.verification import evaluate_incremental_gold as gold
from scripts.verification import smoke_installed_incremental_arrow as smoke


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, single_author: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "snapshot"
    members = {"q1": "p_q", "s1": "p1"} if single_author else {"q1": "p_q", "s1": "p1", "s2": "p2"}
    source = root / "corrected_sources" / "smoke"
    _write_json(
        source / "smoke_signatures.json",
        {
            signature: {"paper_id": paper, "author_info": {"block": "block", "first": "Real", "last": "Name"}}
            for signature, paper in members.items()
        },
    )
    clusters = {"author_a": {"signature_ids": ["q1", "s1"]}}
    if not single_author:
        clusters["author_b"] = {"signature_ids": ["s2"]}
    _write_json(source / "smoke_clusters.json", clusters)
    _write_json(source / "restored_unknown_signature_ids_20261009.json", [])
    smoke._write_arrow_request(root / "benchmarks" / "smoke")  # noqa: SLF001
    mask = tmp_path / "mask.json"
    _write_json(
        mask,
        {
            "datasets": {
                "smoke": {
                    "block_ids": ["block"],
                    "known_query_signature_ids": list(members),
                    "unknown_signature_ids": [],
                }
            }
        },
    )
    return root, mask


def test_prepare_keeps_pure_profiles_and_excludes_complete_query_paper(tmp_path: Path) -> None:
    root, mask = _fixture(tmp_path)
    source = root / "corrected_sources" / "smoke"
    signatures = json.loads((source / "smoke_signatures.json").read_text(encoding="utf-8"))
    signatures["duplicate"] = {"paper_id": "p_q", "author_info": {"block": "block"}}
    signatures["unknown"] = {"paper_id": "unknown_paper", "author_info": {"block": "block"}}
    _write_json(source / "smoke_signatures.json", signatures)
    _write_json(
        source / "smoke_clusters.json",
        {
            "author_a": {"signature_ids": ["q1", "s1", "duplicate"]},
            "author_b": {"signature_ids": ["s2", "unknown"]},
        },
    )
    _write_json(source / "restored_unknown_signature_ids_20261009.json", ["unknown"])
    payload = gold.prepare_cases(snapshot=root, eligible_mask=mask, max_blocks=1, max_cases=4)
    assert payload == gold.prepare_cases(snapshot=root, eligible_mask=mask, max_blocks=1, max_cases=4)
    block = payload["blocks"][0]
    assert block["profiles"] == {"author_a": ["duplicate", "q1", "s1"], "author_b": ["s2"]}
    assert payload["counts"]["smoke"]["excluded_unknown_signatures"] == 1
    assert payload["counts"]["smoke"]["cases_positive"] == 1
    assert payload["counts"]["smoke"]["cases_author_absent"] == 2
    for case in payload["cases"]:
        seeds = gold.case_seeds(block, case)
        query_paper = block["signature_papers"][case["query_signature_id"]]
        assert all(block["signature_papers"][member] != query_paper for member in seeds)
        assert set(seeds) <= set(block["signature_papers"])
        assert "unknown" not in seeds
        if case["context"] == "positive":
            assert case["query_author_id"] in seeds.values()
        else:
            assert case["query_author_id"] not in seeds.values()


def test_prepare_counts_empty_profiles_and_rejects_pool_limits(tmp_path: Path) -> None:
    root, mask = _fixture(tmp_path, single_author=True)
    payload = gold.prepare_cases(snapshot=root, eligible_mask=mask, max_blocks=1, max_cases=1)
    assert len(payload["cases"]) == 1
    assert payload["counts"]["smoke"]["empty_profiles_author_absent"] == 1
    root, mask = _fixture(tmp_path)
    with pytest.raises(ValueError, match="case limit"):
        gold.prepare_cases(snapshot=root, eligible_mask=mask, max_blocks=1, max_cases=2)
    value = json.loads(mask.read_text(encoding="utf-8"))
    value["datasets"]["smoke"]["block_ids"].append("other")
    _write_json(mask, value)
    with pytest.raises(ValueError, match="block limit"):
        gold.prepare_cases(snapshot=root, eligible_mask=mask, max_blocks=1, max_cases=4)


def test_prepare_rejects_known_author_across_selected_blocks(tmp_path: Path) -> None:
    root, mask = _fixture(tmp_path)
    source = root / "corrected_sources" / "smoke" / "smoke_signatures.json"
    signatures = json.loads(source.read_text(encoding="utf-8"))
    signatures["s1"]["author_info"]["block"] = "second_block"
    _write_json(source, signatures)
    eligible = json.loads(mask.read_text(encoding="utf-8"))
    eligible["datasets"]["smoke"]["block_ids"].append("second_block")
    _write_json(mask, eligible)
    with pytest.raises(ValueError, match="Known gold author spans selected blocks: smoke/author_a"):
        gold.prepare_cases(snapshot=root, eligible_mask=mask, max_blocks=2, max_cases=4)


@pytest.mark.parametrize(
    ("clusters", "context", "outcome"),
    [
        ({"a": ["q", "a"], "b": ["b"]}, "positive", "correct_link"),
        ({"q": ["q"], "a": ["a"], "b": ["b"]}, "positive", "false_abstain"),
        ({"a": ["a"], "b": ["q", "b"]}, "positive", "wrong_candidate"),
        ({"q": ["q"], "a": ["a"], "b": ["b"]}, "author_absent", "correct_abstain"),
        ({"a": ["a"], "b": ["q", "b"]}, "author_absent", "false_link"),
    ],
)
def test_identity_classification(clusters, context, outcome) -> None:
    assert gold.classify_query(clusters, "q", {"a": "author_a", "b": "author_b"}, "author_a", context)[0] == outcome


def test_identity_classification_rejects_malformed_partition() -> None:
    with pytest.raises(ValueError, match="exact request"):
        gold.classify_query({"a": ["q", "a"], "b": ["q", "b"]}, "q", {"a": "A", "b": "B"}, "A", "positive")


def test_identity_classification_rejects_split_profile_away_from_query() -> None:
    with pytest.raises(ValueError, match="Seed profile splits across result clusters: B"):
        gold.classify_query(
            {"a": ["q", "a"], "b1": ["b1"], "b2": ["b2"]}, "q", {"a": "A", "b1": "B", "b2": "B"}, "A", "positive"
        )


def test_identity_classification_rejects_merged_profiles_away_from_query() -> None:
    with pytest.raises(ValueError, match="Seed profiles merge in result cluster"):
        gold.classify_query(
            {"a": ["q", "a"], "other": ["b", "c"]}, "q", {"a": "A", "b": "B", "c": "C"}, "A", "positive"
        )


def test_two_cases_use_serialized_complete_model_and_actual_arrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, mask = _fixture(tmp_path)
    payload = gold.prepare_cases(snapshot=root, eligible_mask=mask, max_blocks=1, max_cases=3)
    payload["cases"] = [case for case in payload["cases"] if case["query_author_id"] == "author_a"]
    cases = tmp_path / "cases.json"
    _write_json(cases, payload)
    hashes = {"name_tuples_data_sha256": "1" * 64, "orcid_prefix_counts_data_sha256": "2" * 64}
    monkeypatch.setattr(smoke, "canonical_artifact_hashes", lambda: dict(hashes))
    monkeypatch.setattr(production_model_module, "canonical_artifact_hashes", lambda: dict(hashes))
    monkeypatch.setattr(memory_budget, "current_rss_bytes_best_effort", lambda _total: (100_000_000, "test"))
    model = smoke._write_synthetic_bundle(tmp_path / "model")  # noqa: SLF001
    summary = gold.evaluate_cases(
        snapshot=root,
        cases_path=cases,
        model_dir=model,
        output_dir=tmp_path / "evaluation",
        n_jobs=1,
        total_ram_bytes=2_000_000_000,
    )
    assert summary["datasets"]["smoke"]["cases"] == 2
    assert summary["datasets"]["smoke"]["positive"] == 1
    assert summary["datasets"]["smoke"]["author_absent"] == 1
    rows = [json.loads(line) for line in (tmp_path / "evaluation" / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["profile_sizes"] for row in rows)
    assert all(
        row["outcome"] in {"correct_link", "correct_abstain", "false_abstain", "false_link", "wrong_candidate"}
        for row in rows
    )
    source = root / "corrected_sources" / "smoke" / "smoke_clusters.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Frozen source differs"):
        gold.evaluate_cases(
            snapshot=root,
            cases_path=cases,
            model_dir=model,
            output_dir=tmp_path / "bad",
            n_jobs=1,
            total_ram_bytes=2_000_000_000,
        )

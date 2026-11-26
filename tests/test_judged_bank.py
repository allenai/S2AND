"""Exercise the blind package and versioned bank command paths."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.common.identity_ledger import IdentityLedger
from scripts.judging import judged_bank
from tests.test_review_packets import frozen_snapshot as frozen_snapshot
from tests.test_v15_stage_zero import gate_packet_inputs as gate_packet_inputs

pytestmark = pytest.mark.tooling


@pytest.mark.parametrize(
    ("candidate", "evidence", "expected"),
    [
        (("a", "b", "c", "d", "e"), ("a", "b", "c"), True),
        (("a", "c"), ("a", "c"), False),
        (("a", "b", "c", "d"), ("a", "c", "d"), False),
        (("a", "b", "c"), ("c",), True),
    ],
)
def test_reuse_requires_containment_and_strict_majority(
    candidate: tuple[str, ...], evidence: tuple[str, ...], expected: bool
) -> None:
    assert judged_bank.covers(("a", "b"), candidate, evidence) is expected


def _capture(path: Path, rows: list[tuple[int, list[int], list[int]]], *, block: str = "a smith") -> Path:
    def sid(number: int) -> str:
        return f"visibleq:s{number}" if block == "a smith" else f"{number}-0"

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "block": block,
                    "query_signature_id": sid(query),
                    "shown_group_signature_ids": [sid(i) for i in shown],
                    "target_signature_ids": [sid(i) for i in shown + extra],
                    "labeled_signature_ids": [sid(i) for i in extra],
                    "stratum": "sample",
                    "target_component": "component-private",
                    "score": 0.888888,
                    "cutoff": 0.72123,
                    "arm": "arm-private",
                }
                for query, shown, extra in rows
            ],
        ),
        path,
    )
    return path


def _spec(path: Path, package: str, quota: int, controls: dict[str, int] | None = None) -> Path:
    path.write_text(
        json.dumps({"package": package, "seed": 42, "quotas": {"sample": quota}, "controls": controls or {}}),
        encoding="utf-8",
    )
    return path


def _votes(path: Path, verdicts: list[str]) -> Path:
    path.write_text(
        "".join(
            json.dumps({"item": item, "verdict": verdict, "confidence": "high"}) + "\n"
            for item, verdict in enumerate(verdicts, 1)
        ),
        encoding="utf-8",
    )
    return path


def test_package_merge_and_reuse_round_trip(tmp_path: Path, frozen_snapshot: Path) -> None:
    bank = tmp_path / "bank"
    captured = _capture(tmp_path / "first.parquet", [(1, [2, 3], [])])
    spec = _spec(tmp_path / "first.json", "first", 1)
    key = tmp_path / "private" / "first.json"
    output = tmp_path / "blind"
    assert (
        judged_bank.main(
            [
                "build-package",
                "--bank",
                str(bank),
                "--captured",
                str(captured),
                "--strata",
                str(spec),
                "--snapshot",
                str(frozen_snapshot),
                "--output-dir",
                str(output),
                "--key",
                str(key),
            ]
        )
        == 0
    )
    body = (output / "package.md").read_text(encoding="utf-8")
    assert {path.name for path in output.iterdir()} == {"package.md"}
    assert all(text in body for text in ("Paper 1", "Paper 2", "Paper 3", "## Item 1"))
    assert not any(
        text in body
        for text in (
            "0.888888",
            "0.72123",
            "visibleq:",
            "visiblep:",
            "component-private",
            "arm-private",
            "a@b.edu",
            "000000021694233X",
            "target_signature_ids",
            "first.json",
        )
    )
    first = _votes(tmp_path / "astra.jsonl", ["same"])
    second = _votes(tmp_path / "fable.jsonl", ["cannot_tell"])
    assert (
        judged_bank.main(["merge-verdicts", "--bank", str(bank), "--key", str(key), "--judge", "astra", str(first)])
        == 0
    )
    original = judged_bank.latest_version(bank)
    invalid = _votes(tmp_path / "invalid.jsonl", ["same", "different"])
    with pytest.raises(ValueError, match="unknown or repeated item"):
        judged_bank.merge_verdicts(bank, key, [("fable", second), ("bad", invalid)])
    assert judged_bank.latest_version(bank) == original
    version = judged_bank.merge_verdicts(bank, key, [("fable", second)])
    assert judged_bank.merge_verdicts(bank, key, [("fable", second)]) == version
    with pytest.raises(ValueError, match="already frozen"):
        judged_bank.merge_verdicts(bank, key, [("fable", first)])
    rows = pq.read_table(version / "judgments.parquet").to_pylist()
    assert [(row["judge"], row["verdict"], row["confidence"]) for row in rows] == [
        ("astra", "same", "high"),
        ("fable", "cannot_tell", "high"),
    ]
    assert (version / "frozen" / f"{rows[1]['verdict_sha256']}.jsonl").read_bytes() == second.read_bytes()
    [pair] = pq.read_table(version / "pairs.parquet").to_pylist()
    assert pair["consensus"] == [
        {"package": "first", "item": 1, "judge_a": "astra", "judge_b": "fable", "verdict": "undecided"}
    ]

    captured = _capture(tmp_path / "next.parquet", [(1, [3, 2], [4]), (1, [2, 5], []), (7, [8], [])])
    key = tmp_path / "private" / "next.json"
    result = judged_bank.build_package(
        captured=captured,
        strata=_spec(tmp_path / "next.json", "next", 1),
        bank=bank,
        output_dir=tmp_path / "next",
        key_path=key,
        snapshot=frozen_snapshot,
    )
    assert result == {"sampled": 1, "reused": 1, "items": 1}
    next_key = json.loads(key.read_text(encoding="utf-8"))
    assert next_key["strata"]["sample"] == {"reused": 1, "eligible": 2, "sampled": 1}
    assert next_key["reused"][0]["bank_pair_ids"] == [pair["pair_id"]]
    assert next_key["items"]["1"]["pair_id"] != pair["pair_id"]
    assert next_key["items"]["1"]["inclusion_probability"] == 0.5

    # A historical repeat keeps both package occurrences under the same pair ID.
    later = json.loads(key.read_text(encoding="utf-8"))
    later["items"]["1"] = next_key["reused"][0]
    key.write_text(json.dumps(later), encoding="utf-8")
    later_version = judged_bank.merge_verdicts(bank, key, [("astra", first)])
    [pair] = pq.read_table(later_version / "pairs.parquet").to_pylist()
    assert {(row["package"], row["item"]) for row in pair["occurrences"]} == {("first", 1), ("next", 1)}


def test_planted_controls_use_ledger_and_cached_given_names(
    tmp_path: Path, gate_packet_inputs: tuple[str, IdentityLedger, Path]
) -> None:
    block, ledger, _ = gate_packet_inputs
    labels = ledger.for_block(block).scored()
    candidate = _capture(tmp_path / "controls.parquet", [(11, [14], []), (11, [13], []), (14, [13], [])], block=block)
    cache = tmp_path / "orcid"
    cache.mkdir()
    names = {labels["11-0"]: " ALEX ", labels["13-0"]: "alex"}
    for orcid, name in names.items():
        (cache / f"{orcid}.json").write_text(
            json.dumps({"status": 200, "person": {"name": {"given-names": {"value": name}}}}), encoding="utf-8"
        )
    spec = _spec(tmp_path / "spec.json", "controls", 0, {"same": 1, "different": 1})
    key = tmp_path / "key.json"
    options = dict(
        captured=candidate,
        strata=spec,
        bank=tmp_path / "bank",
        output_dir=tmp_path / "blind",
        key_path=key,
        ledger=ledger,
        orcid_cache=cache,
    )
    with pytest.raises(ValueError, match="Insufficient ledger-pure"):
        judged_bank.build_package(**options)
    different_path = cache / f"{labels['13-0']}.json"
    different_path.write_text(different_path.read_text(encoding="utf-8").replace("alex", "Alice"), encoding="utf-8")
    assert judged_bank.build_package(**options)["items"] == 2
    items = json.loads(key.read_text(encoding="utf-8"))["items"]
    assert {row["gold_verdict"] for row in items.values() if row["control"]} == {"same", "different"}
    votes = [items[str(i)]["gold_verdict"] for i in (1, 2)]
    first = _votes(tmp_path / "first.jsonl", votes)
    second = _votes(tmp_path / "second.jsonl", ["same", "same"])
    judged_bank.merge_verdicts(options["bank"], key, [("first", first), ("second", second)])
    checks = ["judge-check", "--bank", str(options["bank"]), "--package", "controls", "--judge-same", "1"]
    assert judged_bank.main([*checks, "--judge-different", "1"]) == 1
    assert judged_bank.main([*checks, "--judge-different", "1", "--judge", "first"]) == 0
    assert judged_bank.main([*checks, "--consensus-different", "1"]) == 1

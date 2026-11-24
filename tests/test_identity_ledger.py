import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s2and.arrow_tables import write_name_counts_index
from tests.helpers import write_arrow_tables_bundle

pytestmark = pytest.mark.tooling
GOLD = "0000-0002-1694-233X"
DEPOSITED = "0000-0001-5109-3700"


def request_bundle(root: Path, block: str, rows: list[tuple[str, str | None]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = block.replace(" ", "_")
    first, last = block.split()
    papers = [paper for paper, _ in rows]
    author = f"{first} {last}"
    directory = root / name
    write_arrow_tables_bundle(
        directory,
        {
            "signatures": pa.table(
                {
                    "signature_id": [f"{paper}-0" for paper in papers],
                    "paper_id": papers,
                    "author_position": [0] * len(papers),
                    "author_first": [first] * len(papers),
                    "author_middle": [""] * len(papers),
                    "author_last": [last] * len(papers),
                    "author_suffix": [""] * len(papers),
                    "author_affiliations": pa.array([[]] * len(papers), type=pa.list_(pa.string())),
                    "author_block": [block] * len(papers),
                    "author_orcid": pa.array([orcid for _, orcid in rows], type=pa.string()),
                }
            ),
            "papers": pa.table(
                {
                    "paper_id": papers,
                    "title": ["Study"] * len(papers),
                    "venue": [""] * len(papers),
                    "journal_name": [""] * len(papers),
                }
            ),
            "paper_authors": pa.table(
                {"paper_id": papers, "position": [0] * len(papers), "author_name": [author] * len(papers)}
            ),
            "specter": pa.table(
                {"paper_id": papers, "embedding": pa.array([[1.0, 0.0]] * len(papers), type=pa.list_(pa.float32(), 2))}
            ),
        },
    )
    path = root / f"{name}.arrow.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for arrow in directory.glob("*.arrow"):
            archive.add(arrow, arcname=arrow.name)
    return path


@pytest.fixture
def evidence_inputs(tmp_path: Path) -> tuple[Path, Path]:
    from scripts.intake.requests_to_datasets import convert_requests

    first = request_bundle(tmp_path, "a lovelace", [("1", DEPOSITED), ("2", DEPOSITED), ("3", None), ("4", None)])
    second = request_bundle(tmp_path, "g hopper", [("5", None)])
    index, _ = write_name_counts_index(tmp_path, ({}, {}, {}, {}))
    snapshot = tmp_path / "snapshot"
    convert_requests(
        [first, second], snapshot, Path(index), allowed_papers=frozenset("12345"), abstract_allowed_papers=frozenset()
    )
    gold = tmp_path / "gold.parquet"
    pq.write_table(
        pa.table(
            {
                "cluster_block_key": ["A_Lovelace", "a lovelace", "g hopper", "a lovelace"],
                "corpus_paper_id": [1, 3, 5, 4],
                "position": [0, 0, 0, 0],
                "orcid": [GOLD] * 4,
                "split": ["development", "development", "holdout", "development"],
                "scored": [True, False, True, True],
                "author_last": ["Lovelace", "Lovelace", "Hopper", "Wrong"],
            }
        ),
        gold,
    )
    return snapshot, gold


def test_evidence_keeps_all_verified_sources_and_flags(evidence_inputs: tuple[Path, Path], tmp_path: Path) -> None:
    from scripts.common.identity_ledger import load_identity_evidence
    from scripts.intake.build_identity_evidence import build_identity_evidence

    snapshot, gold = evidence_inputs
    output = tmp_path / "evidence"
    result = build_identity_evidence(snapshot, gold, output, {"g hopper": ["5-0"]})
    ledger = load_identity_evidence(output / "manifest.json")
    rows = ledger.read_block("a_lovelace")
    assert rows["1-0"]["gold"] == [{"orcid": GOLD, "scored": True, "split": "development"}]
    assert rows["1-0"]["evidence_state"] == "conflict"
    assert rows["2-0"]["deposited_orcid"] == DEPOSITED
    assert rows["2-0"]["gold"] == []
    assert rows["3-0"]["gold"] == [{"orcid": GOLD, "scored": False, "split": "development"}]
    assert rows["4-0"]["evidence_state"] == "none"
    assert result["audit"]["a lovelace"]["dropped_name_mismatch"] == 1
    assert ledger.read_block("g hopper")["5-0"]["reviewed_error"]
    partition = output / json.loads((output / "manifest.json").read_text())["blocks"]["a lovelace"]["path"]
    partition.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        ledger.read_block("a lovelace")


def test_freeze_input_counts_both_sources_once(evidence_inputs: tuple[Path, Path], tmp_path: Path) -> None:
    from scripts.intake.build_freeze_input import build_freeze_input
    from scripts.intake.build_identity_evidence import build_identity_evidence

    snapshot, gold = evidence_inputs
    evidence = tmp_path / "freeze-evidence"
    build_identity_evidence(snapshot, gold, evidence)
    source = tmp_path / "blocks.parquet"
    pq.write_table(
        pa.table({"cluster_block_key": ["g hopper", "a lovelace"], "split": ["holdout", "development"]}), source
    )
    output = tmp_path / "freeze-input.parquet"
    build_freeze_input(evidence / "manifest.json", source, output, strata={"a_lovelace": "tiny"})
    first, second = pq.read_table(output).to_pylist()
    assert first["signature_count"] == 4
    assert first["stratum"] == "tiny"
    assert dict(first["orcid_mention_counts"]) == {DEPOSITED: 2, GOLD: 2}
    assert second["source_split"] == "holdout"


def test_freeze_masks_shared_identities_and_reserves_both_halves(
    evidence_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    from scripts.common.block_roles import load_block_roles
    from scripts.intake.assign_block_roles import BlockSpec, assign_block_roles, freeze_identity_roles
    from scripts.intake.build_freeze_input import build_freeze_input
    from scripts.intake.build_identity_evidence import build_identity_evidence

    snapshot, gold = evidence_inputs
    evidence = tmp_path / "evidence"
    build_identity_evidence(snapshot, gold, evidence)
    source = tmp_path / "blocks.parquet"
    pq.write_table(
        pa.table({"cluster_block_key": ["g hopper", "a lovelace"], "split": ["holdout", "development"]}), source
    )
    blocks = tmp_path / "freeze-input.parquet"
    build_freeze_input(evidence / "manifest.json", source, blocks)
    targets = {"pairwise_train": 1, "final_test_a": 1}
    frozen = freeze_identity_roles(blocks, evidence / "manifest.json", targets)
    manifest = tmp_path / "assignments.json"
    manifest.write_text(json.dumps(frozen))
    roles = load_block_roles(manifest)
    assert not roles.scores("a lovelace", GOLD)
    assert not roles.scores("g hopper", GOLD)
    assert roles.scores("a lovelace", DEPOSITED)
    assert "orcid_home_role" not in frozen
    assert frozen["verification"]["cross_role_mentions_by_role"]["pairwise_train"] == 2
    with pytest.raises(ValueError, match="new exported blocks"):
        freeze_identity_roles(
            blocks,
            evidence / "manifest.json",
            targets,
            prior_path=manifest,
            extra_pins={"a lovelace": "pairwise_train"},
        )
    allocation = assign_block_roles(
        [
            BlockSpec("a block", 1, stratum="singleton"),
            BlockSpec("b block", 1, stratum="paired"),
            BlockSpec("c block", 1, stratum="paired"),
        ],
        {"final_test": 2},
        closure_mode="block_only",
    )
    assert allocation["unassigned"] == ["a block"]


def test_eligibility_blocks_conflicting_and_cross_role_mentions(
    evidence_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    from scripts.common.block_roles import load_block_roles
    from scripts.common.identity_ledger import ledger_for_roles
    from scripts.intake.assign_block_roles import freeze_identity_roles
    from scripts.intake.build_freeze_input import build_freeze_input
    from scripts.intake.build_identity_evidence import build_identity_evidence

    snapshot, gold = evidence_inputs
    output = tmp_path / "evidence"
    build_identity_evidence(snapshot, gold, output)
    evidence = output / "manifest.json"
    source = tmp_path / "blocks.parquet"
    pq.write_table(
        pa.table({"cluster_block_key": ["a lovelace", "g hopper"], "split": ["development", "holdout"]}), source
    )
    blocks = tmp_path / "freeze.parquet"
    build_freeze_input(evidence, source, blocks)
    frozen = tmp_path / "roles.json"
    frozen.write_text(json.dumps(freeze_identity_roles(blocks, evidence, {"pairwise_train": 1, "final_test_a": 1})))
    ledger = ledger_for_roles(evidence, load_block_roles(frozen))
    assert ledger is not None
    block = ledger.for_block("a lovelace")
    assert not block.eligibility("1-0").fit_ok
    assert not block.eligibility("3-0").fit_ok
    for sid in ("2-0", "4-0"):
        assert block.eligibility(sid).fit_ok
        assert block.eligibility(sid).score_orcid is None
    assert block.component_label(["1-0", "2-0"]) is None
    assert not ledger.for_block("g hopper").scored()
    with pytest.raises(ValueError, match="No identity evidence"):
        block.eligibility("missing")


def test_miner_uses_ledger_endpoints_and_rejects_stale_roles(
    evidence_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    import csv

    from s2and._sha256 import sha256_file
    from scripts.common.block_roles import load_block_roles
    from scripts.common.identity_ledger import ledger_for_roles
    from scripts.intake.assign_block_roles import freeze_identity_roles
    from scripts.intake.build_freeze_input import build_freeze_input
    from scripts.intake.build_identity_evidence import build_identity_evidence
    from scripts.pairwise.mine_pairwise_pairs import BlockInput, mine_pairs

    snapshot, gold = evidence_inputs
    directory = tmp_path / "identity"
    build_identity_evidence(snapshot, gold, directory)
    evidence = directory / "manifest.json"
    source = tmp_path / "blocks.parquet"
    pq.write_table(
        pa.table({"cluster_block_key": ["a lovelace", "g hopper"], "split": ["development", "holdout"]}), source
    )
    blocks = tmp_path / "freeze-input.parquet"
    build_freeze_input(evidence, source, blocks)
    role_path = tmp_path / "roles.json"
    role_path.write_text(json.dumps(freeze_identity_roles(blocks, evidence, {"pairwise_train": 1, "final_test_a": 1})))
    ledger = ledger_for_roles(evidence, load_block_roles(role_path))
    assert ledger is not None
    assignment_root = tmp_path / "assignments"
    assignment_root.mkdir()
    assignments = assignment_root / "assignments.parquet"
    pq.write_table(pa.table({"signature_id": ["2-0", "4-0"], "cluster_id": ["a", "b"]}), assignments)
    manifest = assignment_root / "manifest.json"
    record = {"binding": ledger.bindings, "files": {assignments.name: sha256_file(assignments)}}
    manifest.write_text(json.dumps(record))
    specs = {"a lovelace": BlockInput(snapshot / "datasets" / "a_lovelace", None, assignments)}
    output = tmp_path / "pairs.csv"
    assert mine_pairs(role_path, specs, output, per_block=1, identity_evidence=evidence) == 1
    with output.open(newline="", encoding="utf-8") as source_file:
        (row,) = csv.DictReader(source_file)
    assert {row["signature_id_1"], row["signature_id_2"]} == {"2-0", "4-0"}
    record["binding"]["block_role_assignments_sha256"] = "old"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not match"):
        mine_pairs(role_path, specs, tmp_path / "stale.csv", per_block=1, identity_evidence=evidence)


def test_gate_arrival_order_uses_inserted_and_production_components(
    evidence_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    from datetime import UTC, datetime

    from scripts.common.identity_ledger import load_identity_evidence
    from scripts.intake.build_identity_evidence import build_identity_evidence
    from scripts.linker.build_linker_inputs import _arrival_requests, _production_seeds

    snapshot, gold = evidence_inputs
    output = tmp_path / "identity"
    build_identity_evidence(snapshot, gold, output)
    mentions = load_identity_evidence(output / "manifest.json").read_block("a lovelace")
    arrivals = tmp_path / "arrivals.parquet"
    pq.write_table(
        pa.table(
            {
                "cluster_block_key": ["a lovelace"] * 5,
                "corpus_paper_id": [1, 2, 3, 4, 10],
                "position": [0] * 5,
                "inserted": [
                    datetime(2026, 10, 19),
                    datetime(2026, 11, 21),
                    datetime(2026, 11, 20),
                    None,
                    datetime(2026, 11, 22),
                ],
                "updated": [datetime(2026, 11, 23)] * 5,
                "clusterer": ["AltAdIncremental", "AltAdFullBlock", "AltAdIncremental", "AltAdIncremental", None],
            }
        ),
        arrivals,
    )
    queries, counts = _arrival_requests(
        arrivals, "a lovelace", mentions, mentions.keys() - {"1-0"}, datetime(2026, 10, 20, tzinfo=UTC), 1
    )
    assert queries == ["2-0"]
    assert counts["missing_inserted"] == counts["inserted_before_since"] == 1
    assert counts["clusterer_counts"] == {"AltAdFullBlock": 1}

    raw = tmp_path / "a_lovelace"
    seeds = raw / "cluster_seeds.arrow"
    table = pa.table(
        {
            "signature_id": ["1-0", "2-0", "3-0", "4-0"],
            "cluster_id": ["production a", "production a", "production b", "production a"],
        }
    )
    with pa.ipc.new_file(str(seeds), table.schema) as writer:
        writer.write_table(table)
    bundle = tmp_path / "incremental" / "a_lovelace.arrow.tar.gz"
    bundle.parent.mkdir()
    with tarfile.open(bundle, "w:gz") as archive:
        for member in (raw / "signatures.arrow", seeds):
            archive.add(member, arcname=member.name)
    assert _production_seeds(bundle, "a lovelace", mentions, queries) == {
        "1-0": "production a",
        "3-0": "production b",
        "4-0": "production a",
    }

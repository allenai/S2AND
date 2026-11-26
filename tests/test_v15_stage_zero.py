"""Exercise the stage-zero command paths on five production-shaped blocks."""

from __future__ import annotations

import csv
import json
import tarfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_tables import write_cluster_seeds_arrow, write_name_counts_index
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.incremental_linking.feature_block import write_arrow_ipc_table
from s2and.production_model import load_production_model
from scripts.common.block_roles import BLOCK_BINDINGS_FIELD, ROLE_SHA256_FIELD, load_block_roles
from scripts.common.identity_ledger import IdentityLedger, ledger_for_roles, load_identity_evidence
from scripts.eps import apply_reviewed_eps, orcid_b3, sweep_eps_on_linking_gold
from scripts.intake import (
    assign_block_roles,
    build_freeze_input,
    build_identity_evidence,
    build_v14_training_inventory,
    check_v14_training_overlap,
    requests_to_datasets,
)
from scripts.judging import review_packets
from scripts.linker import build_linker_inputs, build_linker_task_tables
from scripts.linker.blind_gate_metrics import _apply_blind_gate_labels
from scripts.linker.train_linker_and_finalize import _validate_frozen_task_bindings
from scripts.pairwise import mine_pairwise_pairs
from tests.helpers import request_bundle
from tests.test_giant_eps_release import _complete_model
from tests.test_identity_ledger import DEPOSITED, GOLD, HOLDOUT

pytestmark = pytest.mark.tooling


def _run(module: ModuleType, *args: str | Path) -> None:
    assert module.main(list(map(str, args))) in (0, None)


def _build_freeze_input(evidence: Path, gold_blocks: Path, output: Path) -> None:
    origins = output.with_suffix(".origins.csv")
    origins.write_text("surname,broad\n", encoding="utf-8")
    _run(
        build_freeze_input,
        "--identity-evidence",
        evidence,
        "--gold-blocks",
        gold_blocks,
        "--surname-origin-csv",
        origins,
        "--output",
        output,
    )


def _json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parquet(path: Path, **columns: list) -> Path:
    pq.write_table(pa.table(columns), path)
    return path


@pytest.fixture
def gate_packet_inputs(tmp_path: Path) -> tuple[str, IdentityLedger, Path]:
    block = "g schneider"
    papers = [(str(i), {25: HOLDOUT, 32: GOLD}.get(i)) for i in (11, 13, 14, 21, 22, 23, 24, 25, 31, 32)]
    bundle = request_bundle(tmp_path / "raw", block, papers)
    raw = bundle.parent / "g_schneider"
    with pa.ipc.open_file(raw / "papers.arrow") as reader:
        table = reader.read_all()
    table = table.set_column(
        table.schema.get_field_index("title"), "title", pa.array([f"Study {pid}" for pid, _ in papers])
    )
    table = table.set_column(
        table.schema.get_field_index("venue"), "venue", pa.array([f"Venue {pid}" for pid, _ in papers])
    ).append_column("year", pa.array([2000 + int(pid) for pid, _ in papers]))
    write_arrow_ipc_table(table, raw / "papers.arrow")
    with pa.ipc.open_file(raw / "paper_authors.arrow") as reader:
        authors = reader.read_all()
    authors = authors.set_column(
        authors.schema.get_field_index("author_name"),
        "author_name",
        pa.array([f"G Schneider {pid}" for pid, _ in papers]),
    )
    write_arrow_ipc_table(authors, raw / "paper_authors.arrow")
    with tarfile.open(bundle, "w:gz") as archive:
        for path in raw.glob("*.arrow"):
            archive.add(path, arcname=path.name)
    fit = "a lovelace"
    fit_bundle = request_bundle(tmp_path / "raw", fit, [("1", None), ("2", None), ("3", None)])
    index, _ = write_name_counts_index(tmp_path, ({}, {}, {}, {}))
    snapshot = tmp_path / "snapshot"
    allowed = _json(tmp_path / "allowed.json", [paper for paper, _ in papers] + ["1", "2", "3"])
    _run(
        requests_to_datasets,
        bundle,
        fit_bundle,
        "--output-root",
        snapshot,
        "--name-counts-index",
        Path(index),
        "--allowed-papers",
        allowed,
        "--abstract-allowed-papers",
        allowed,
    )
    gold = _parquet(
        tmp_path / "gold.parquet",
        cluster_block_key=[block] * 4 + [fit] * 3,
        corpus_paper_id=[11, 13, 14, 31, 1, 2, 3],
        position=[0] * 7,
        orcid=[HOLDOUT, GOLD, HOLDOUT, HOLDOUT, "0000-0001-5000-0007", DEPOSITED, "0000-0001-5000-0007"],
        split=["development"] * 7,
        scored=[True] * 7,
    )
    evidence = tmp_path / "evidence" / "manifest.json"
    _run(build_identity_evidence, "--snapshot", snapshot, "--gold-parquet", gold, "--output-dir", evidence.parent)
    gold_blocks = _parquet(tmp_path / "gold-blocks.parquet", cluster_block_key=[block, fit], split=["development"] * 2)
    blocks = tmp_path / "blocks.parquet"
    _build_freeze_input(evidence, gold_blocks, blocks)
    freeze = tmp_path / "freeze"
    _run(
        assign_block_roles,
        "--identity-evidence",
        evidence,
        "--blocks",
        blocks,
        "--output-dir",
        freeze,
        "--targets-json",
        _json(tmp_path / "targets.json", {"gate_thresh": 1, "fit_pool": 1}),
        "--extra-pins-json",
        _json(tmp_path / "pins.json", {block: "gate_thresh", fit: "fit_pool"}),
        "--gate-training-block-count",
        "1",
    )
    roles = load_block_roles(freeze / "assignments.json")
    assert roles.gate_training_blocks == {fit}
    ledger = ledger_for_roles(evidence, roles)
    assert ledger is not None
    seeds = _parquet(
        tmp_path / "seeds.parquet",
        signature_id=["21-0", "22-0", "23-0", "24-0", "25-0", "11-0", "31-0", "32-0"],
        component_key=["other"] * 6 + ["mixed"] * 2,
    )
    arrow_dir = snapshot / "datasets" / "g_schneider"
    with ArrowDataset.open(arrow_dir) as arrow:
        spec = {
            "arrow_dir": str(arrow_dir),
            "arrow_generation_id": arrow.generation_id,
            "seeds": str(seeds),
            "seeds_sha256": sha256_file(seeds),
        }
    task = _json(
        tmp_path / "tasks.json",
        {
            "tables": {"gate_thresh": {"blocks": {block: spec}}},
            BLOCK_BINDINGS_FIELD: roles.bindings_for({block}),
            ROLE_SHA256_FIELD: roles.sha256,
        },
    )
    return block, ledger, task


def test_gate_controls_show_verified_identity(
    tmp_path: Path, gate_packet_inputs: tuple[str, IdentityLedger, Path]
) -> None:
    block, ledger, task = gate_packet_inputs
    view = ledger.for_block(block)
    natural = {
        "block_key": block,
        "source_id": "natural",
        "query": {"signature_id": "13-0", "title": "Study 13"},
        "candidates": [{"candidate_key": "other"}],
    }
    controls = [
        {
            **natural,
            "source_id": f"control-{i}",
            "query": {"signature_id": "14-0" if i % 2 else "13-0", "title": "Study"},
        }
        for i in range(10)
    ]
    controls.append({**controls[0], "source_id": "mixed", "candidates": [{"candidate_key": "mixed"}]})
    key = tmp_path / "key.json"
    summary = review_packets.build_blind_linker_packets(
        queries=[natural],
        control_queries=controls,
        block_roles=ledger.roles,
        ledger=ledger,
        allowed_roles={"gate_thresh"},
        linker_task_manifest=task,
        output_dir=tmp_path / "packets",
        answer_key_path=key,
    )
    assert summary["control_items"] == 10
    assert summary["excluded_controls"] == 1
    assert view.eligibility("25-0").score_orcid is None
    for item, line in zip(
        json.loads(key.read_text())["items"],
        next((tmp_path / "packets/packets").glob("*.jsonl")).read_text().splitlines(),
        strict=True,
    ):
        exemplars = json.loads(line)["candidates"][0]["exemplars"]
        expected = [25, 11, 21, 22] if item["is_control"] else [21, 22, 23, 24]
        assert [paper["title"] for paper in exemplars] == [f"Study {pid}" for pid in expected]

    resolved, _ = review_packets.resolve_gate_exemplars(controls[:1], ledger.roles, task)
    assert not review_packets._scored_controls(
        resolved,
        ledger.roles,
        packet_type="linker",
        allowed_roles={"gate_thresh"},
        ledger=ledger,
    )


def test_gate_control_exemplars_check_the_frozen_dataset(
    gate_packet_inputs: tuple[str, IdentityLedger, Path],
) -> None:
    block, ledger, task = gate_packet_inputs
    manifest = json.loads(task.read_text(encoding="utf-8"))
    spec = manifest["tables"]["gate_thresh"]["blocks"][block]
    datasets = ledger.evidence.snapshot_root / "datasets"
    foreign_dir = next(path for path in datasets.iterdir() if path != Path(spec["arrow_dir"]))
    with ArrowDataset.open(foreign_dir) as foreign:
        spec.update(arrow_dir=str(foreign_dir), arrow_generation_id=foreign.generation_id)
    _json(task, manifest)
    items = [{"block_key": block, "candidates": [{"candidate_key": "other"}]}]
    with pytest.raises(ValueError, match="Identity evidence was built for a different dataset generation"):
        review_packets.resolve_gate_exemplars(items, ledger.roles, task, control_ledger=ledger)


def test_gate_query_metadata_follows_frozen_signature(
    tmp_path: Path, gate_packet_inputs: tuple[str, IdentityLedger, Path]
) -> None:
    block, ledger, task = gate_packet_inputs
    item = {
        "block_key": block,
        "source_id": "swapped-query",
        "query": {
            "signature_id": "13-0",
            "paper_id": "11",
            "title": "Study 11",
            "venue": "Venue 11",
            "year": 2011,
            "authors": ["G Schneider 11"],
            "coauthors": ["Caller input"],
            "abstract": "Caller input",
        },
        "candidates": [{"candidate_key": "other"}],
    }
    packets = tmp_path / "packets"
    review_packets.build_blind_linker_packets(
        queries=[item],
        block_roles=ledger.roles,
        ledger=ledger,
        allowed_roles={"gate_thresh"},
        linker_task_manifest=task,
        output_dir=packets,
        control_fraction=0,
    )
    query = json.loads(next((packets / "packets").glob("*.jsonl")).read_text())["query"]
    assert query == {
        "title": "Study 13",
        "venue": "Venue 13",
        "year": 2013,
        "authors": ["G Schneider 13"],
        "author_name": "g schneider",
        "affiliations": [],
    }


def test_mixed_role_packet_resolves_fit_controls_from_frozen_papers(
    tmp_path: Path, gate_packet_inputs: tuple[str, IdentityLedger, Path]
) -> None:
    gate, ledger, task = gate_packet_inputs
    natural = {
        "block_key": gate,
        "source_id": "gate",
        "query": {"signature_id": "13-0"},
        "candidates": [{"candidate_key": "other"}],
    }
    controls = [
        {
            "block_key": "a lovelace",
            "source_id": f"fit-{index}",
            "query": {"signature_id": "3-0" if index % 2 else "2-0", "title": "Caller query"},
            "candidates": [
                {
                    "candidate_key": "fit",
                    "member_signature_ids": ["1-0"],
                    "exemplars": [{"signature_id": "1-0", "title": "Caller exemplar"}],
                }
            ],
        }
        for index in range(10)
    ]
    output = tmp_path / "mixed"
    review_packets.build_blind_linker_packets(
        queries=[natural],
        control_queries=controls,
        block_roles=ledger.roles,
        ledger=ledger,
        allowed_roles={"gate_thresh", "fit_pool"},
        linker_task_manifest=task,
        output_dir=output,
    )
    records = [
        json.loads(line) for path in (output / "packets").glob("*.jsonl") for line in path.read_text().splitlines()
    ]
    fit = [record for record in records if record["block_key"] == "a lovelace"]
    assert len(fit) == 10
    for record in fit:
        assert record["query"]["title"] == "Study"
        assert record["candidates"][0]["exemplars"][0]["title"] == "Study"


def test_gate_scorer_rejects_old_votes_for_swapped_query_display(
    tmp_path: Path, gate_packet_inputs: tuple[str, IdentityLedger, Path]
) -> None:
    block, ledger, task = gate_packet_inputs
    output = tmp_path / "packet"
    report = review_packets.build_blind_linker_packets(
        queries=[
            {
                "block_key": block,
                "source_id": "swapped-query",
                "query": {"signature_id": "13-0", "title": "Study 11"},
                "candidates": [{"candidate_key": "other"}],
            }
        ],
        block_roles=ledger.roles,
        ledger=ledger,
        allowed_roles={"gate_thresh"},
        linker_task_manifest=task,
        output_dir=output,
        control_fraction=0,
    )
    key = Path(report["answer_key_path"])
    display = json.loads((output / report["shards"][0]).read_text(encoding="utf-8"))
    assert display["query"]["title"] == "Study 13"
    wrong_display = {**display, "query": {**display["query"], "title": "Study 11"}}
    votes = {"item_id": display["item_id"], "candidate_verdicts": {"C1": "same"}}
    judges = [(family, tmp_path / f"{family}.jsonl", key) for family in ("one", "two")]
    for payload in (votes, {**votes, "display_sha256": canonical_json_digest(wrong_display)}):
        for _, path, _ in judges:
            _json(path, payload)
        with pytest.raises(ValueError, match="display hash"):
            review_packets.score_judges(judges, ledger=ledger)
    for _, path, _ in judges:
        _json(path, {**votes, "display_sha256": canonical_json_digest(display)})
    assert review_packets.score_judges(judges, ledger=ledger)["gate_thresh_labels"] == [
        {"dataset": block, "query_signature_id": "13-0", "candidate_component_key": "other", "verdict": "same"}
    ]


@pytest.mark.parametrize("control", [False, True])
def test_gate_builder_rejects_query_hidden_in_candidate_members(
    tmp_path: Path, gate_packet_inputs: tuple[str, IdentityLedger, Path], control: bool
) -> None:
    block, ledger, task = gate_packet_inputs
    natural = {
        "block_key": block,
        "source_id": "natural",
        "query": {"signature_id": "13-0"},
        "candidates": [{"candidate_key": "other"}],
    }
    overlap = {**natural, "source_id": "overlap", "query": {"signature_id": "25-0"}}
    with pytest.raises(ValueError, match="query.*member of candidate"):
        review_packets.build_blind_linker_packets(
            queries=[natural if control else overlap],
            control_queries=[overlap] if control else (),
            block_roles=ledger.roles,
            ledger=ledger,
            allowed_roles={"gate_thresh"},
            linker_task_manifest=task,
            output_dir=tmp_path / "packet",
            control_fraction=0,
        )


def test_score_gate_key_after_unrelated_control_change(
    tmp_path: Path, gate_packet_inputs: tuple[str, IdentityLedger, Path]
) -> None:
    block, ledger, task = gate_packet_inputs
    control_block = "a lovelace"
    fit_block = "p dirac"
    holder = "e noether"
    key = tmp_path / "key.json"
    review_packets.build_blind_linker_packets(
        queries=[
            {
                "block_key": block,
                "source_id": "gate",
                "query": {"signature_id": "13-0"},
                "candidates": [{"candidate_key": "other"}],
            }
        ],
        block_roles=ledger.roles,
        ledger=ledger,
        allowed_roles={"gate_thresh"},
        linker_task_manifest=task,
        output_dir=tmp_path / "packets",
        answer_key_path=key,
        control_fraction=0,
    )
    roles = replace(
        ledger.roles,
        block_to_role={**ledger.roles.block_to_role, fit_block: "fit_pool"},
        source_splits={**ledger.roles.source_splits, fit_block: "development"},
        evidence_blocks={**ledger.roles.evidence_blocks, fit_block: "d" * 64, control_block: "a" * 64},
        block_orcids={**ledger.roles.block_orcids, fit_block: frozenset(), control_block: frozenset({DEPOSITED})},
        identity_roles={**ledger.roles.identity_roles, DEPOSITED: frozenset({"fit_pool"})},
    )
    payload = json.loads(key.read_text())
    gate_item = payload["items"][0]
    control = {
        **gate_item,
        "item_id": "control",
        "source_id": "control",
        "block_key": control_block,
        "is_control": True,
        "metadata": {"gold_candidate_keys": []},
    }
    payload["items"].append(control)
    payload[BLOCK_BINDINGS_FIELD] = roles.bindings_for({block, control_block})
    _json(key, payload)
    judges = []
    for family in ("one", "two"):
        verdicts = tmp_path / f"{family}.jsonl"
        verdicts.write_text(
            "".join(
                json.dumps(
                    {
                        "item_id": row["item_id"],
                        "display_sha256": row["display_sha256"],
                        "candidate_verdicts": {"C1": "different"},
                    }
                )
                + "\n"
                for row in (gate_item, control)
            ),
            encoding="utf-8",
        )
        judges.append((family, verdicts, key))
    before = review_packets.score_judges(judges, ledger=replace(ledger, roles=roles))
    assert before["judges"][0]["overall"]["control_requests"] == 1
    assert len(before["gate_thresh_labels"]) == 1
    changed = replace(roles, evidence_blocks={**roles.evidence_blocks, control_block: "b" * 64})
    after = review_packets.score_judges(judges, ledger=replace(ledger, roles=changed))
    assert after["gate_thresh_labels"] == before["gate_thresh_labels"]
    assert after["judges"][0]["overall"]["control_requests"] == 0
    assert after[BLOCK_BINDINGS_FIELD] == roles.bindings_for({block})

    fit_key = _json(
        tmp_path / "fit-key.json",
        {
            "packet_type": "linker",
            "shards": payload["shards"],
            "items": [{**gate_item, "block_key": fit_block, "source_id": "fit"}, control],
            BLOCK_BINDINGS_FIELD: roles.bindings_for({fit_block, control_block}),
        },
    )
    fit_judges = []
    for family, original, _ in judges:
        path = tmp_path / f"{family}-fit.jsonl"
        path.write_bytes(original.read_bytes())
        fit_judges.append((family, path, fit_key))
    combined = review_packets.score_judges([*judges, *fit_judges], ledger=replace(ledger, roles=changed))
    assert combined["gate_thresh_labels"] == before["gate_thresh_labels"]
    assert [judge["overall"]["control_requests"] for judge in combined["judges"]] == [0, 0, 0, 0]
    assert combined[BLOCK_BINDINGS_FIELD] == roles.bindings_for({block, fit_block})
    changed_fit = replace(changed, evidence_blocks={**changed.evidence_blocks, fit_block: "e" * 64})
    with pytest.raises(ValueError, match="judging key block provenance"):
        review_packets.score_judges([*judges, *fit_judges], ledger=replace(ledger, roles=changed_fit))

    relevant = replace(
        changed,
        block_to_role={**changed.block_to_role, holder: "gate_thresh"},
        source_splits={**changed.source_splits, holder: "development"},
        evidence_blocks={**changed.evidence_blocks, holder: "c" * 64},
        block_orcids={**changed.block_orcids, holder: frozenset({HOLDOUT})},
        gate_block_orcids={**changed.gate_block_orcids, holder: frozenset({HOLDOUT})},
    )
    assert relevant.bindings_for({block}) == changed.bindings_for({block})
    with pytest.raises(ValueError, match="gate judging key block provenance"):
        review_packets.score_judges(judges, ledger=replace(ledger, roles=relevant))


def _v14_source(tmp_path: Path, index: Path, allowed_abstracts: Path) -> Path:
    root = tmp_path / "v14"
    export = request_bundle(tmp_path / "v14-exports", "h wang", [("801", None), ("802", None), ("803", None)])
    allowed = _json(tmp_path / "v14-allowed.json", ["801", "802", "803"])
    _run(
        requests_to_datasets,
        export,
        "--output-root",
        root,
        "--name-counts-index",
        index,
        "--allowed-papers",
        allowed,
        "--abstract-allowed-papers",
        allowed_abstracts,
    )
    labels = _parquet(
        root / "train.parquet", dataset=["h_wang"], query_signature_id=["801-0"], candidate_component_key=["selected"]
    )
    members = _parquet(
        root / "members.parquet", candidate_component_key=["selected"] * 2, signature_id=["802-0", "803-0"]
    )
    _json(
        root / "bundle.json",
        {
            "assets": {
                "candidate_members": {"datasets": {"h_wang": members.name}},
                "featureless_rows": {"files": {"train_path": labels.name}},
            }
        },
    )
    return root


def test_stage_zero_commands_use_one_ledger_and_production_requests(tmp_path: Path) -> None:
    eps_orcid = "0000-0001-5000-0007"
    other_orcid = "0000-0002-9079-593X"
    gate_a = "0000-0002-1825-0855"
    gate_b = "0000-0003-0011-7637"
    unscored = "0000-0002-5719-1149"
    excluded = "0000-0002-1825-0097"
    paper_ids = {
        "a lovelace": [(str(i), {101: DEPOSITED, 102: DEPOSITED, 108: excluded}.get(i)) for i in range(101, 109)],
        "b ross": [(str(i), None) for i in range(201, 206)],
        "d hopper": [("301", None)],
        "e turing": [("401", None)],
        "g schneider": [(str(i), {501: gate_a, 502: GOLD, 503: excluded}.get(i)) for i in range(500, 533)],
    }
    gold_rows = [
        ("a lovelace", 101, GOLD, "development", True),
        ("a lovelace", 103, GOLD, "development", True),
        ("a lovelace", 104, unscored, "development", False),
        ("a lovelace", 106, unscored, "development", True),
        ("a lovelace", 107, unscored, "development", True),
        ("b ross", 201, eps_orcid, "development", True),
        ("b ross", 202, eps_orcid, "development", True),
        ("b ross", 203, eps_orcid, "development", False),
        ("d hopper", 301, GOLD, "holdout", True),
        ("e turing", 401, other_orcid, "holdout", True),
        ("g schneider", 500, gate_a, "development", True),
        ("g schneider", 501, gate_b, "development", True),
        ("g schneider", 532, gate_a, "development", True),
        ("f aba", 601, "0000-0002-7589-9587", "development", False),
        ("z last", 701, other_orcid, "holdout", False),
        ("h wang", 801, other_orcid, "development", True),
    ]
    # Give the inventory its own identity, disjoint from sealed gold.
    gold_rows[-1] = ("h wang", 801, "0000-0002-7589-9587", "development", True)
    labels = _parquet(
        tmp_path / "gold.parquet",
        cluster_block_key=[row[0] for row in gold_rows],
        corpus_paper_id=[row[1] for row in gold_rows],
        orcid=[row[2] for row in gold_rows],
        split=[row[3] for row in gold_rows],
        scored=[row[4] for row in gold_rows],
        position=[0] * len(gold_rows),
    )
    source_blocks = _parquet(
        tmp_path / "gold-blocks.parquet",
        cluster_block_key=[*paper_ids, "f aba", "z last"],
        split=["development", "development", "holdout", "holdout", "development", "development", "holdout"],
    )
    exports = {block: request_bundle(tmp_path / "full", block, ids) for block, ids in paper_ids.items()}
    index_text, _ = write_name_counts_index(tmp_path, ({}, {}, {}, {}))
    index = Path(index_text)
    model = _complete_model(tmp_path)
    permissions = _json(tmp_path / "allowed.json", [paper for ids in paper_ids.values() for paper, _ in ids])
    abstract_permissions = _json(tmp_path / "abstracts.json", [])
    reviewed = _json(tmp_path / "errors.json", {"a lovelace": ["105-0"]})
    snapshot = tmp_path / "snapshot"
    targets = _json(tmp_path / "targets.json", {"fit_pool": 1, "eps_cal": 1, "final_test": 2})
    pins = _json(
        tmp_path / "pins.json",
        {"a lovelace": "fit_pool", "b ross": "eps_cal", "d hopper": "final_test", "e turing": "final_test"},
    )
    gate = "g schneider"
    raw_gate = exports[gate].parent / "g_schneider"
    inc = tmp_path / "incremental"
    inc.mkdir()
    inc_export = inc / exports[gate].name
    seed_path = inc / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(seed_path, {f"{i}-0": "mixed" if i < 502 else f"seed-{i}" for i in range(500, 533)})
    with tarfile.open(inc_export, "w:gz") as archive:
        archive.add(raw_gate / "signatures.arrow", arcname="signatures.arrow")
        archive.add(seed_path, arcname=seed_path.name)
    arrivals = _parquet(
        tmp_path / "arrivals.parquet",
        cluster_block_key=[gate] * 4,
        corpus_paper_id=[529, 530, 531, 532],
        position=[0] * 4,
        inserted=[datetime(2026, 10, 19), None, datetime(2026, 11, 20), datetime(2026, 11, 22)],
        updated=[datetime(2026, 11, 23)] * 4,
        clusterer=["AltAdIncremental"] * 3 + ["AltAdFullBlock"],
    )

    def intake_and_freeze(
        name: str, selected: list[Path], role_pins: Path, prior: Path | None = None
    ) -> tuple[Path, Path]:
        _run(
            requests_to_datasets,
            *selected,
            "--output-root",
            snapshot,
            "--name-counts-index",
            index,
            "--allowed-papers",
            permissions,
            "--abstract-allowed-papers",
            abstract_permissions,
        )
        evidence_dir = tmp_path / "evidence"
        _run(
            build_identity_evidence,
            "--snapshot",
            snapshot,
            "--gold-parquet",
            labels,
            "--output-dir",
            evidence_dir,
            "--reviewed-errors-json",
            reviewed,
            *(["--append"] if prior else []),
        )
        evidence = evidence_dir / "manifest.json"
        blocks = tmp_path / f"{name}-blocks.parquet"
        _build_freeze_input(evidence, source_blocks, blocks)
        output = tmp_path / name
        prior_args = ["--pinned-roles-json", prior] if prior else []
        _run(
            assign_block_roles,
            "--blocks",
            blocks,
            "--identity-evidence",
            evidence,
            "--targets-json",
            targets,
            "--output-dir",
            output,
            "--extra-pins-json",
            role_pins,
            *prior_args,
        )
        return evidence, output / "assignments.json"

    _, prior = intake_and_freeze("first", [exports[block] for block in exports if block != gate], pins)
    gate_pin = _json(tmp_path / "gate-pin.json", {gate: "gate_thresh"})
    evidence, role_path = intake_and_freeze(
        "next", [inc_export.parent.parent / "full" / inc_export.name], gate_pin, prior
    )
    old_roles, roles = load_block_roles(prior), load_block_roles(role_path)
    assert roles.block_to_role == {**old_roles.block_to_role, gate: "gate_thresh"}
    fresh_dir = tmp_path / "fresh"
    with pytest.raises(ValueError, match="requires the master assignment"):
        _run(
            assign_block_roles,
            "--fresh",
            "--blocks",
            tmp_path / "next-blocks.parquet",
            "--identity-evidence",
            evidence,
            "--output-dir",
            fresh_dir,
        )
    _run(
        assign_block_roles,
        "--fresh",
        "--blocks",
        tmp_path / "next-blocks.parquet",
        "--identity-evidence",
        evidence,
        "--output-dir",
        fresh_dir,
        "--pinned-roles-json",
        role_path,
    )
    assert (fresh_dir / "assignments.json").read_bytes() == role_path.read_bytes()
    # Owner decision 2026-11-25: a formatting-only rewrite keeps the assignment's identity.
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(json.loads(role_path.read_text("utf-8")), indent=4) + "\n\n", encoding="utf-8")
    assert load_block_roles(reformatted).sha256 == roles.sha256
    ledger_for_roles(evidence, load_block_roles(reformatted))
    altered = tmp_path / "altered.json"
    payload = json.loads(role_path.read_text("utf-8"))
    payload["seed"] = payload.get("seed", 0) + 1
    altered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded master block role assignment"):
        ledger_for_roles(evidence, load_block_roles(altered))
    binding = ["--identity-evidence", evidence, "--block-role-assignments", role_path]
    ledger = ledger_for_roles(evidence, roles)
    assert ledger is not None
    assert ledger.for_block(gate).component_label(["500-0", "501-0"]) is None
    assert not ledger.for_block("d hopper").scored()

    arrow_dirs = {block: str(snapshot / "datasets" / block.replace(" ", "_")) for block in exports}
    v14 = _v14_source(tmp_path, index, abstract_permissions)
    inventory = tmp_path / "inventory.parquet"
    _run(build_v14_training_inventory, "--gold-parquet", labels, "--linker-source-root", v14, "--output", inventory)
    assert any(row["signature_id"] == "802-0" and row["orcid"] is None for row in pq.read_table(inventory).to_pylist())
    _run(check_v14_training_overlap, *binding, "--training-parquet", inventory)
    inputs = tmp_path / "linker-inputs"
    _run(
        build_linker_inputs,
        *binding,
        "--incremental-requests",
        inc_export,
        "--output-dir",
        inputs,
        "--model-path",
        model,
        "--batching-threshold",
        "3",
        "--arrivals",
        arrivals,
        "--since",
        "2026-10-20",
        "--requests-per-block",
        "2",
        "--queries-per-block",
        "300",
    )
    built = json.loads((inputs / "inputs.json").read_text())
    fit_inputs = built["a lovelace"]
    fit_assignments = {
        row["signature_id"]: row["component_key"] for row in pq.read_table(fit_inputs["seeds"]).to_pylist()
    }
    fit_queries = set(json.loads(Path(fit_inputs["queries"]).read_text(encoding="utf-8")))
    assert fit_queries == {"106-0"}
    assert fit_assignments.keys() == {"102-0", "104-0", "107-0"}
    assert build_linker_task_tables.read_seed_components(Path(fit_inputs["seeds"])) == fit_assignments
    assert json.loads(Path(built[gate]["queries"]).read_text()) == ["532-0", "531-0"]
    seed_map = build_linker_task_tables.read_seed_components(Path(built[gate]["seeds"]))
    assert seed_map["500-0"] == seed_map["501-0"] == "mixed"
    assert seed_map.keys().isdisjoint({"531-0", "532-0"})
    tasks = tmp_path / "tasks"
    _run(
        build_linker_task_tables,
        *binding,
        "--block-inputs",
        inputs / "inputs.json",
        "--output-dir",
        tasks,
        "--blocks",
        gate,
    )
    rows = pq.read_table(tasks / "gate_thresh.parquet").to_pylist()
    assert len(rows) == 50
    assert {row["query_signature_id"] for row in rows} == {"531-0", "532-0"}
    assert all(
        row["label"] is None and not row["candidate_gold_scored"] and not row["query_gold_scored"] for row in rows
    )
    genuine = next(row for row in rows if row["candidate_component_key"] == "mixed")
    candidate = genuine["candidate_component_key"]
    query = genuine["query_signature_id"]
    exemplar = "501-0"
    assert not ledger.for_block(gate).eligibility(exemplar).fit_ok
    assert seed_map[exemplar] == candidate
    vetoed_components = {"seed-502", "seed-503"}
    other = next(
        row["candidate_component_key"]
        for row in rows
        if row["query_signature_id"] == query and row["candidate_component_key"] not in vetoed_components | {candidate}
    )
    other_exemplar = next(sid for sid, component in seed_map.items() if component == other)
    blind_items = _json(
        tmp_path / "blind-items.json",
        [
            {
                "source_id": "request",
                "block_key": gate,
                "query": {"signature_id": query, "paper_id": query.split("-")[0], "title": "Study"},
                "candidates": [
                    {
                        "candidate_key": candidate,
                        "exemplars": [
                            {"signature_id": query, "paper_id": query.split("-")[0], "title": "Mislabeled exemplar"}
                        ],
                    },
                    {
                        "candidate_key": other,
                        "exemplars": [
                            {"signature_id": other_exemplar, "paper_id": other_exemplar.split("-")[0], "title": "Study"}
                        ],
                    },
                ],
            }
        ],
    )
    natural_item = json.loads(blind_items.read_text())[0]
    for sid in ("502-0", "503-0"):
        vetoed_key = seed_map[sid]
        bad_candidate = {"candidate_key": vetoed_key, "member_signature_ids": [], "exemplars": []}
        with pytest.raises(ValueError, match="vetoed identity"):
            review_packets.build_blind_linker_packets(
                queries=[{**natural_item, "candidates": [bad_candidate]}],
                ledger=ledger,
                block_roles=roles,
                allowed_roles={"gate_thresh"},
                linker_task_manifest=tasks / build_linker_task_tables.TASK_MANIFEST,
                output_dir=tmp_path / "vetoed-packet",
            )
    blind_key_path = tmp_path / "gate-key.json"
    _run(
        review_packets,
        "build",
        *binding,
        "--allowed-role",
        "gate_thresh",
        "--packet-type",
        "linker",
        "--items-json",
        blind_items,
        "--linker-task-manifest",
        tasks / build_linker_task_tables.TASK_MANIFEST,
        "--output-dir",
        tmp_path / "gate-packets",
        "--answer-key-path",
        blind_key_path,
        "--control-fraction",
        "0",
    )
    packet = json.loads(next((tmp_path / "gate-packets/packets").glob("*.jsonl")).read_text())
    assert all(paper["title"] != "Mislabeled exemplar" for c in packet["candidates"] for paper in c["exemplars"])
    answer_key = json.loads(blind_key_path.read_text())
    (item,) = answer_key["items"]
    verdicts = {
        slot: "same" if key == candidate else "different" for slot, key in item["slot_to_candidate_key"].items()
    }
    audit_slots = {slot: key for slot, key in item["slot_to_candidate_key"].items() if key == other}
    audit_key = _json(
        tmp_path / "audit-key.json", {**answer_key, "items": [{**item, "slot_to_candidate_key": audit_slots}]}
    )
    blind_judges = []
    display_hash = canonical_json_digest(packet)
    for family in ("sol", "claude"):
        verdict_path = _json(
            tmp_path / f"gate-{family}.jsonl",
            {"item_id": item["item_id"], "display_sha256": display_hash, "candidate_verdicts": verdicts},
        )
        blind_judges.extend(["--judge", family, verdict_path, blind_key_path])
    audit_verdict = _json(
        tmp_path / "gate-audit.jsonl",
        {
            "item_id": item["item_id"],
            "display_sha256": display_hash,
            "candidate_verdicts": {slot: verdicts[slot] for slot in audit_slots},
        },
    )
    blind_judges.extend(["--judge", "audit", audit_verdict, audit_key])
    judged_path = tmp_path / "gate-judgments.json"
    _run(review_packets, "score", *binding, *blind_judges, "--output-json", judged_path)
    report = json.loads(judged_path.read_text())
    gate_table = pq.read_table(tasks / "gate_thresh.parquet").to_pandas()
    task_manifest = json.loads((tasks / build_linker_task_tables.TASK_MANIFEST).read_text())
    gate_blocks = task_manifest["tables"]["gate_thresh"]["blocks"]
    index_digest = ledger.evidence.name_counts_index_sha256
    _validate_frozen_task_bindings(task_manifest, roles, index_sha256=index_digest)
    fitting_manifest = build_linker_task_tables.build_linker_task_tables(
        role_path, inputs / "inputs.json", tmp_path / "fit-tasks", blocks=["a lovelace"], identity_evidence=evidence
    )
    _validate_frozen_task_bindings(fitting_manifest, roles, task_role="fit_pool", pairwise_model_path=model)
    wrong_model = tmp_path / "wrong-pairwise-model"
    wrong_model.mkdir()
    (wrong_model / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen seeds use a different pairwise model"):
        _validate_frozen_task_bindings(fitting_manifest, roles, task_role="fit_pool", pairwise_model_path=wrong_model)
    for field, error in (
        ("index_sha256", "Linker inputs use a different name-counts index"),
        ("model_sha256", "Input model, index, or role digest mismatch"),
    ):
        rebound = {**task_manifest, "linker_input_binding": {**task_manifest["linker_input_binding"], field: "0" * 64}}
        with pytest.raises(ValueError, match=error):
            _validate_frozen_task_bindings(rebound, roles, index_sha256=index_digest)
    stale_seed_report = {**report, "gate_seed_sha256": {gate: "0" * 64}}
    with pytest.raises(ValueError, match="frozen seed assignments"):
        _apply_blind_gate_labels(gate_table, stale_seed_report, roles, gate_blocks)
    scored = _apply_blind_gate_labels(gate_table, report, roles, gate_blocks)
    assert scored["label"].sum() == 1
    assert scored["candidate_gold_scored"].sum() == 2
    assert report["gate_thresh_audit_coverage"]["audit"]["planned_share"] == 0.5
    assert scored["query_gold_scored"].sum() == 0
    assert scored.loc[scored["label"].eq(1), "candidate_gold_mixed"].all()
    vetoed = gate_table.loc[gate_table["candidate_component_key"].isin(vetoed_components)]
    assert set(vetoed["candidate_component_key"]) == vetoed_components
    assert gate_table["candidate_has_vetoed_identity"].equals(
        gate_table["candidate_component_key"].isin(vetoed_components)
    )
    vetoed_votes = vetoed[["dataset", "query_signature_id", "candidate_component_key"]].assign(verdict="same")
    unsafe_report = {**report, "gate_thresh_labels": [*report["gate_thresh_labels"], *vetoed_votes.to_dict("records")]}
    assert _apply_blind_gate_labels(gate_table, unsafe_report, roles, gate_blocks).equals(scored)
    with pytest.raises(ValueError, match="candidate identity vetoes"):
        _apply_blind_gate_labels(gate_table.drop(columns="candidate_has_vetoed_identity"), report, roles, gate_blocks)
    report["gate_thresh_labels"][0]["candidate_component_key"] = "outside-window"
    with pytest.raises(ValueError, match="outside the frozen windows"):
        _apply_blind_gate_labels(gate_table, report, roles, gate_blocks)

    pairs = tmp_path / "pairs.csv"
    _run(
        mine_pairwise_pairs,
        *binding,
        "--block-inputs",
        inputs / "inputs.json",
        "--pairs-per-block",
        "10",
        "--output",
        pairs,
    )
    with pairs.open(encoding="utf-8", newline="") as handle:
        mined = list(csv.DictReader(handle))
    assert not mined
    fit_counts = json.loads(pairs.with_suffix(".csv.manifest.json").read_text(encoding="utf-8"))["blocks"]["a lovelace"]
    assert fit_counts["eligible_signatures"] == 3
    assert fit_counts["excluded_query_signatures"] == 1
    assert fit_counts["excluded_non_home_signatures"] == 4
    assert fit_counts["ineligible_signatures"] == 5

    paper = {"paper_id": "201", "signature_id": "201-0", "name": "B Ross", "title": "Study"}
    items = _json(
        tmp_path / "items.json",
        [
            {
                "source_id": "comparison",
                "block_key": "b ross",
                "paper_a": paper,
                "paper_b": {**paper, "signature_id": "202-0"},
            }
        ],
    )
    controls = _json(
        tmp_path / "controls.json",
        [
            {
                "source_id": "unscored",
                "block_key": "b ross",
                "paper_a": paper,
                "paper_b": {**paper, "signature_id": "203-0"},
            }
        ],
    )
    review = tmp_path / "review"
    answer_key = tmp_path / "key.json"
    _run(
        review_packets,
        "build",
        *binding,
        "--allowed-role",
        "eps_cal",
        "--packet-type",
        "pair",
        "--items-json",
        items,
        "--controls-json",
        controls,
        "--output-dir",
        review,
        "--answer-key-path",
        answer_key,
        "--control-fraction",
        "0",
    )
    key = json.loads(answer_key.read_text())
    assert {row["source_id"] for row in key["items"]} == {"comparison"}
    judge_paths = []
    for family in ("sol", "claude"):
        verdict = tmp_path / f"{family}.jsonl"
        _json(
            verdict,
            {
                "item_id": key["items"][0]["item_id"],
                "display_sha256": key["items"][0]["display_sha256"],
                "verdict": "same",
            },
        )
        judge_paths.extend(["--judge", family, verdict, answer_key])
    _run(review_packets, "score", *binding, *judge_paths, "--output-json", tmp_path / "judging.json")
    judging = json.loads((tmp_path / "judging.json").read_text())
    assert judging.get("block_bindings") == roles.bindings_for({"b ross"})
    mixed_report = tmp_path / "gate-and-pair-judgments.json"
    _run(review_packets, "score", *binding, *blind_judges, *judge_paths, "--output-json", mixed_report)
    judged = json.loads(mixed_report.read_text())
    assert judged["block_bindings"] == roles.bindings_for({gate, "b ross"})
    assert _apply_blind_gate_labels(gate_table, judged, roles, gate_blocks).equals(scored)
    changed_pair_role = replace(roles, evidence_blocks={**roles.evidence_blocks, "b ross": "0" * 64})
    assert _apply_blind_gate_labels(gate_table, judged, changed_pair_role, gate_blocks).equals(scored)

    # A same-role block may hold an identity named by the judged candidates.
    candidate_id = judged["gate_thresh_candidate_orcids"][gate][0]
    shared = "same role block"
    holders = replace(
        roles,
        block_to_role={**roles.block_to_role, shared: "gate_thresh"},
        block_orcids={**roles.block_orcids, shared: frozenset({candidate_id})},
        gate_block_orcids={**roles.gate_block_orcids, shared: frozenset({candidate_id})},
        source_splits={**roles.source_splits, shared: "development"},
        evidence_blocks={**roles.evidence_blocks, shared: "1" * 64},
    )
    with pytest.raises(ValueError, match="block provenance"):
        _apply_blind_gate_labels(gate_table, judged, holders, gate_blocks)
    bound_shared = {**judged, "gate_thresh_block_bindings": holders.bindings_for({gate, shared})}
    assert _apply_blind_gate_labels(gate_table, bound_shared, holders, gate_blocks).equals(scored)
    changed_shared = replace(holders, evidence_blocks={**holders.evidence_blocks, shared: "0" * 64})
    with pytest.raises(ValueError, match="block provenance"):
        _apply_blind_gate_labels(gate_table, bound_shared, changed_shared, gate_blocks)

    eps_root = tmp_path / "eps"
    eps_output = eps_root / "b_ross"
    _run(
        sweep_eps_on_linking_gold,
        *binding,
        "--dataset",
        "b_ross",
        "--arrow-root",
        snapshot,
        "--model-path",
        model,
        "--output-dir",
        eps_output,
        "--batching-threshold",
        "3",
        "--n-jobs",
        "1",
        "--route",
        "production",
        "--allow-all-reference-identities",
        "--compute-missing-dists",
        "--allow-full-run",
        "--eps-values",
        "0.38",
    )
    assert set(pq.read_table(eps_output / "orcid_reference.parquet")["signature_id"].to_pylist()) == {"201-0", "202-0"}
    b3_assignments = _json(tmp_path / "prediction.json", {"201-0": "cluster", "202-0": "cluster"})
    _run(
        orcid_b3,
        *binding,
        "--role",
        "eps_cal",
        "--block",
        "b ross",
        "--arrow-dir",
        Path(arrow_dirs["b ross"]),
        "--snapshot-root",
        snapshot,
        "--eps",
        "0.38",
        "--assignments-json",
        b3_assignments,
    )
    reviewed_model = tmp_path / "reviewed-model"
    report = tmp_path / "eps-review.json"
    _run(
        apply_reviewed_eps,
        *binding,
        "--source-model",
        model,
        "--output-model",
        reviewed_model,
        "--sweep-root",
        eps_root,
        "--datasets",
        "b_ross",
        "--expected-batching-threshold",
        "3",
        "--eps",
        "0.38",
        "--precision-f1-tolerance",
        "0.001",
        "--review-note",
        "Synthetic review",
        "--output-report",
        report,
    )
    assert load_production_model(reviewed_model).cluster_model.eps == 0.38
    assert json.loads(report.read_text())["identity_evidence_sha256"] == sha256_file(evidence)
    for duplicate in ("b_ross", "B Ross"):
        with pytest.raises(ValueError, match="EPS review datasets must be unique"):
            apply_reviewed_eps.apply_reviewed_eps(
                model,
                tmp_path / "duplicate-model",
                eps_root,
                ["b_ross", duplicate],
                0.38,
                "Synthetic review",
                tmp_path / "duplicate-report.json",
                expected_batching_threshold=3,
                precision_f1_tolerance=0.01,
                block_role_assignments=role_path,
                identity_evidence=evidence,
            )
    assert not (tmp_path / "duplicate-model").exists()
    assert not (tmp_path / "duplicate-report.json").exists()

    old_evidence = load_identity_evidence(evidence)
    filenames = [
        *(evidence.parent / entry["path"] for entry in old_evidence.blocks.values()),
        *(
            Path(directory) / filename
            for directory in arrow_dirs.values()
            for filename in ("manifest.json", "request_source.json")
        ),
        tasks / build_linker_task_tables.TASK_MANIFEST,
        tasks / "gate_thresh.parquet",
        blind_key_path,
        judged_path,
        inputs / "g_schneider" / "manifest.json",
        eps_output / "plan.json",
    ]
    old_files = {path: sha256_file(path) for path in filenames}
    added = [
        request_bundle(tmp_path / "full", "f aba", [("601", None)]),
        request_bundle(tmp_path / "full", "z last", [("701", None)]),
    ]
    extended = _json(tmp_path / "extended.json", [*json.loads(permissions.read_text()), "601", "701"])
    _run(
        requests_to_datasets,
        *added,
        "--output-root",
        snapshot,
        "--name-counts-index",
        index,
        "--allowed-papers",
        extended,
        "--abstract-allowed-papers",
        abstract_permissions,
    )
    _run(
        build_identity_evidence,
        "--snapshot",
        snapshot,
        "--gold-parquet",
        labels,
        "--output-dir",
        evidence.parent,
        "--append",
    )
    appended = load_identity_evidence(evidence)
    assert set(appended.blocks) - old_evidence.blocks.keys() == {"f aba", "z last"}
    assert all(appended.blocks[block] == entry for block, entry in old_evidence.blocks.items())
    assert appended.blocks["f aba"]["permission_sha256"] != old_evidence.blocks[gate]["permission_sha256"]
    readded_blocks = tmp_path / "appended-blocks.parquet"
    _build_freeze_input(evidence, source_blocks, readded_blocks)
    common = ["--blocks", readded_blocks, "--identity-evidence", evidence, "--targets-json", targets]
    with pytest.raises(ValueError, match="prior.*--fresh"):
        _run(assign_block_roles, *common, "--output-dir", tmp_path / "unbound-freeze")
    _run(assign_block_roles, *common, "--pinned-roles-json", role_path, "--output-dir", tmp_path / "appended-freeze")
    appended_path = tmp_path / "appended-freeze" / "assignments.json"
    newer_roles = load_block_roles(appended_path)
    assert newer_roles.block_to_role == roles.block_to_role
    assert newer_roles.bindings_for({"d hopper", "e turing"}) == roles.bindings_for({"d hopper", "e turing"})
    assert newer_roles.identity_roles[other_orcid] == {"final_test"}
    with pytest.raises(ValueError, match="absent from the frozen role assignments"):
        newer_roles.bindings_for({"z last"})
    assert {path: sha256_file(path) for path in filenames} == old_files
    with pytest.raises(ValueError, match="master block role assignment"):
        _validate_frozen_task_bindings(task_manifest, newer_roles)
    foreign_roles = replace(
        newer_roles, identity_roles={**newer_roles.identity_roles, gate_a: frozenset({"gate_thresh", "final_test"})}
    )
    with pytest.raises(ValueError, match="block provenance"):
        _validate_frozen_task_bindings({**task_manifest, ROLE_SHA256_FIELD: foreign_roles.sha256}, foreign_roles)
    with pytest.raises(ValueError, match="block provenance"):
        _apply_blind_gate_labels(gate_table, json.loads(judged_path.read_text()), foreign_roles, gate_blocks)
    assert _apply_blind_gate_labels(gate_table, json.loads(judged_path.read_text()), newer_roles, gate_blocks).equals(
        scored
    )
    newer_judged = tmp_path / "gate-judgments-after-append.json"
    _run(
        review_packets,
        "score",
        "--identity-evidence",
        evidence,
        "--block-role-assignments",
        appended_path,
        *blind_judges,
        "--output-json",
        newer_judged,
    )
    assert newer_judged.read_bytes() == judged_path.read_bytes()
    raised_targets = _json(
        tmp_path / "raised-targets.json",
        {"fit_pool": 2, "eps_cal": 1, "final_test": 3},
    )
    new_pins = _json(tmp_path / "new-pins.json", {"z last": "final_test"})
    raised = tmp_path / "raised-freeze"
    _run(
        assign_block_roles,
        "--blocks",
        readded_blocks,
        "--identity-evidence",
        evidence,
        "--targets-json",
        raised_targets,
        "--pinned-roles-json",
        appended_path,
        "--extra-pins-json",
        new_pins,
        "--output-dir",
        raised,
    )
    assert load_block_roles(raised / "assignments.json").block_to_role == {
        **roles.block_to_role,
        "f aba": "fit_pool",
        "z last": "final_test",
    }
    assert load_block_roles(raised / "assignments.json").pairwise_validation_blocks == {"f aba"}
    for directory in (snapshot, evidence.parent):
        receipt = directory / ".role_freeze.json"
        receipt.rename(receipt.with_suffix(".bak"))
        with pytest.raises(ValueError, match="prior.*--fresh"):
            _run(assign_block_roles, *common, "--output-dir", tmp_path / "missing-receipt-freeze")

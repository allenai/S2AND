"""Exercise the stage-zero command paths on five production-shaped blocks."""

from __future__ import annotations

import csv
import json
import tarfile
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_tables import write_cluster_seeds_arrow, write_name_counts_index
from s2and.production_model import load_production_model
from scripts.common.block_roles import load_block_roles
from scripts.common.identity_ledger import ledger_for_roles
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
from scripts.linker.train_linker_and_finalize import _apply_blind_gate_labels
from scripts.pairwise import mine_pairwise_pairs
from tests.test_giant_eps_release import _complete_model
from tests.test_identity_ledger import DEPOSITED, GOLD, request_bundle

pytestmark = pytest.mark.tooling


def _run(module: ModuleType, *args: str | Path) -> None:
    assert module.main(list(map(str, args))) in (0, None)


def _json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _parquet(path: Path, **columns: list) -> Path:
    pq.write_table(pa.table(columns), path)
    return path


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
        "g schneider": [(str(i), None) for i in range(500, 533)],
    }
    gold_rows = [
        ("a lovelace", 101, GOLD, "development", True),
        ("a lovelace", 103, GOLD, "development", True),
        ("a lovelace", 104, unscored, "development", False),
        ("b ross", 201, eps_orcid, "development", True),
        ("b ross", 202, eps_orcid, "development", True),
        ("b ross", 203, eps_orcid, "development", False),
        ("d hopper", 301, GOLD, "holdout", True),
        ("e turing", 401, other_orcid, "holdout", True),
        ("g schneider", 500, gate_a, "development", True),
        ("g schneider", 501, gate_b, "development", True),
        ("g schneider", 532, gate_a, "development", True),
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
        cluster_block_key=list(paper_ids),
        split=["development", "development", "holdout", "holdout", "development"],
    )
    exports = {block: request_bundle(tmp_path / "full", block, ids) for block, ids in paper_ids.items()}
    index_text, _ = write_name_counts_index(tmp_path, ({}, {}, {}, {}))
    index = Path(index_text)
    model = _complete_model(tmp_path)
    permissions = _json(tmp_path / "allowed.json", [paper for ids in paper_ids.values() for paper, _ in ids])
    abstract_permissions = _json(tmp_path / "abstracts.json", [])
    reviewed = _json(tmp_path / "errors.json", {"a lovelace": ["105-0"]})
    snapshot = tmp_path / "snapshot"
    targets = _json(tmp_path / "targets.json", {"pairwise_train": 1, "eps_cal": 1, "final_test": 2})
    pins = _json(
        tmp_path / "pins.json",
        {"a lovelace": "pairwise_train", "b ross": "eps_cal", "d hopper": "final_test_a", "e turing": "final_test_b"},
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
        evidence_dir = tmp_path / f"{name}-evidence"
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
        )
        evidence = evidence_dir / "manifest.json"
        blocks = tmp_path / f"{name}-blocks.parquet"
        _run(build_freeze_input, "--identity-evidence", evidence, "--gold-blocks", source_blocks, "--output", blocks)
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
    sealed = _json(tmp_path / "sealed.json", {key: arrow_dirs[key] for key in ("d hopper", "e turing")})
    _run(check_v14_training_overlap, *binding, "--blocks-json", sealed, "--training-parquet", inventory)

    linker_specs = _json(
        tmp_path / "linker-specs.json",
        {
            gate: {"arrow_dir": arrow_dirs[gate], "incremental_request": str(inc_export)},
            "a lovelace": {"arrow_dir": arrow_dirs["a lovelace"]},
        },
    )
    inputs = tmp_path / "linker-inputs"
    _run(
        build_linker_inputs,
        *binding,
        "--blocks-json",
        linker_specs,
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
    )
    built = json.loads((inputs / "inputs.json").read_text())
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
    genuine = next(row for row in rows if row["candidate_identity_eligible"])
    candidate = genuine["candidate_component_key"]
    query = genuine["query_signature_id"]
    exemplar = next(sid for sid, component in seed_map.items() if component == candidate)
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
                        "exemplars": [{"signature_id": exemplar, "paper_id": exemplar.split("-")[0], "title": "Study"}],
                    }
                ],
            }
        ],
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
        "--output-dir",
        tmp_path / "gate-packets",
        "--answer-key-path",
        blind_key_path,
        "--control-fraction",
        "0",
    )
    (item,) = json.loads(blind_key_path.read_text())["items"]
    (slot,) = item["slot_to_candidate_key"]
    blind_judges = []
    for family in ("sol", "claude"):
        verdict_path = _json(
            tmp_path / f"gate-{family}.jsonl", {"item_id": item["item_id"], "candidate_verdicts": {slot: "same"}}
        )
        blind_judges.extend(["--judge", family, verdict_path, blind_key_path])
    judged_path = tmp_path / "gate-judgments.json"
    _run(review_packets, "score", *binding, *blind_judges, "--output-json", judged_path)
    report = json.loads(judged_path.read_text())
    gate_table = pq.read_table(tasks / "gate_thresh.parquet").to_pandas()
    scored = _apply_blind_gate_labels(gate_table, report, roles)
    assert scored["label"].sum() == scored["candidate_gold_scored"].sum() == 1
    assert scored["query_gold_scored"].sum() == 0
    report["gate_thresh_labels"][0]["candidate_component_key"] = "outside-window"
    with pytest.raises(ValueError, match="outside the frozen windows"):
        _apply_blind_gate_labels(gate_table, report, roles)

    pairs = tmp_path / "pairs.csv"
    assignments = inputs / "a_lovelace" / "assignments.parquet"
    hashed = set(pq.read_table(assignments)["signature_id"].to_pylist())
    assert hashed == {"102-0", "104-0", "106-0", "107-0"}
    miner = _json(
        tmp_path / "miner.json",
        {"a lovelace": {"arrow_dir": arrow_dirs["a lovelace"], "assignments": str(assignments)}},
    )
    _run(mine_pairwise_pairs, *binding, "--blocks-json", miner, "--pairs-per-block", "10", "--output", pairs)
    with pairs.open(encoding="utf-8", newline="") as handle:
        mined = list(csv.DictReader(handle))
    assert mined and all({row["signature_id_1"], row["signature_id_2"]} <= hashed for row in mined)

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
        _json(verdict, {"item_id": key["items"][0]["item_id"], "verdict": "same"})
        judge_paths.extend(["--judge", family, verdict, answer_key])
    _run(review_packets, "score", *binding, *judge_paths, "--output-json", tmp_path / "judging.json")

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
        "0.01",
        "--review-note",
        "Synthetic review",
        "--output-report",
        report,
    )
    assert load_production_model(reviewed_model).cluster_model.eps == 0.38
    assert json.loads(report.read_text())["identity_evidence_sha256"] == sha256_file(evidence)

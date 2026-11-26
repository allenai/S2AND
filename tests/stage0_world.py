"""Generate one tiny frozen mining handoff and its planted hazards."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from s2and.arrow_tables import write_name_counts_index
from scripts.common.block_roles import BLOCK_BINDINGS_FIELD, ROLE_SHA256_FIELD, BlockRoles, load_block_roles
from scripts.common.identity_ledger import IdentityLedger, ledger_for_roles
from scripts.intake.assign_block_roles import freeze_identity_roles
from scripts.intake.build_freeze_input import build_freeze_input
from scripts.intake.build_identity_evidence import build_identity_evidence
from scripts.intake.requests_to_datasets import convert_requests
from scripts.linker.build_linker_inputs import build_linker_inputs
from scripts.pairwise.mine_pairwise_pairs import BlockInput, mine_pairs
from tests.helpers import request_bundle
from tests.pairwise_release_helpers import write_pairwise_artifacts


@dataclass(frozen=True)
class Stage0World:
    roles_path: Path
    evidence_path: Path
    name_counts: Path
    inputs: dict[str, BlockInput]
    mined: Path
    manifest: Path
    roles: BlockRoles
    ledger: IdentityLedger


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_stage0_world(
    root: Path,
    *,
    missing_fit_pool_block: bool = False,
    foreign_index: bool = False,
    judge_as_fitted: bool = False,
    fitted_as_judge: bool = False,
    lost_ledger_score: bool = False,
    empty_block: bool = False,
    all_ineligible_block: bool = False,
    validation_blocks: tuple[str, ...] = ("b crane",),
    sampling_hazard: str | None = None,
) -> Stage0World:
    """Build a real intake, freeze, and mined sample, with optional rebound hazards."""
    _, name_counts, _ = write_pairwise_artifacts(root)
    snapshot = root / "snapshot"
    papers = {"a ross": [10, 11, 12, 13, 14], "b crane": [20, 21], "c lane": [30]}
    clusters = {"a ross": ["x", "y", "x", "z", "x"], "b crane": ["x", "x"]}
    if empty_block or all_ineligible_block:
        papers["e vale"] = [40]
        clusters["e vale"] = ["x"]
    if sampling_hazard in {"stale_frame", "judge_zero_cap", "zeroed_judge_frame"}:
        papers["e vale"] = [40, 41]
        clusters["e vale"] = ["x", "y" if sampling_hazard == "stale_frame" else "x"]
    convert_requests(
        [request_bundle(root / "raw", block, [(str(paper), None) for paper in ids]) for block, ids in papers.items()],
        snapshot,
        name_counts,
        allowed_papers=frozenset(str(paper) for ids in papers.values() for paper in ids),
        abstract_allowed_papers=frozenset(),
    )
    first, second, gate = "0000-0002-1694-233X", "0000-0001-5109-3700", "0000-0002-9079-593X"
    gold_rows = [
        {
            "cluster_block_key": block,
            "corpus_paper_id": paper,
            "position": 0,
            "orcid": orcid,
            "scored": True,
            "split": "development",
        }
        for block, labels in {
            "a ross": {10: first, 11: first, 12: second, 13: second},
            "b crane": {20: first, 21: first},
            "c lane": {30: gate},
        }.items()
        for paper, orcid in labels.items()
    ]
    source = root / "source.parquet"
    pq.write_table(pa.table({"cluster_block_key": list(papers), "split": ["development"] * len(papers)}), source)
    roles_path = root / "roles.json"

    def freeze(directory: Path) -> tuple[Path, BlockRoles]:
        gold = root / "gold.parquet"
        pq.write_table(pa.Table.from_pylist(gold_rows), gold)
        evidence = directory / "manifest.json"
        build_identity_evidence(
            snapshot, gold, directory, reviewed_errors={"e vale": ["40-0"]} if all_ineligible_block else None
        )
        freeze_input = directory / "freeze.parquet"
        build_freeze_input(evidence, source, freeze_input)
        frozen = freeze_identity_roles(
            freeze_input, evidence, {"fit_pool": len(clusters), "gate_fit": 1}, extra_pins={"c lane": "gate_fit"}
        )
        frozen["pairwise_validation_blocks"] = list(validation_blocks)
        _write_json(roles_path, frozen)
        return evidence, load_block_roles(roles_path)

    evidence, roles = freeze(root / "evidence")
    index = sha256_file(name_counts / "manifest.json")
    model = root / "model"
    model.write_bytes(b"fixture model")
    model_digest = sha256_file(model) if all_ineligible_block else "a" * 64
    inputs = {}
    for block, cluster_ids in clusters.items():
        name = block.replace(" ", "_")
        if all_ineligible_block and block == "e vale":
            arrow_dir = snapshot / "datasets" / name
            linker_input = root / "linker-input.json"
            _write_json(linker_input, {block: {"arrow_dir": str(arrow_dir)}})
            output = root / "linker-inputs"
            build_linker_inputs(roles_path, linker_input, output, model, identity_evidence=evidence)
            inputs[block] = BlockInput(arrow_dir, None, output / name / "assignments.parquet")
            continue
        assignments = root / name / "assignments.parquet"
        assignments.parent.mkdir()
        pq.write_table(
            pa.table({"signature_id": [f"{paper}-0" for paper in papers[block]], "cluster_id": cluster_ids}),
            assignments,
        )
        inputs[block] = BlockInput(snapshot / "datasets" / name, None, assignments)

    def write_receipts() -> dict[str, Any]:
        bindings = {}
        for block, spec in inputs.items():
            if all_ineligible_block and block == "e vale":
                continue
            bindings[block] = {
                ROLE_SHA256_FIELD: roles.sha256,
                BLOCK_BINDINGS_FIELD: roles.bindings_for({block}),
                "model_sha256": model_digest,
                "index_sha256": index,
            }
            _write_json(
                spec.assignments.parent / "manifest.json",
                {
                    "binding": bindings[block],
                    "files": {spec.assignments.name: sha256_file(spec.assignments)},
                },
            )
        return bindings

    write_receipts()
    mined = root / "mined.csv"
    mine_pairs(roles_path, inputs, mined, identity_evidence=evidence)
    manifest_path = mined.with_suffix(".csv.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with mined.open(encoding="utf-8", newline="") as handle:
        columns, *rows = csv.reader(handle)

    if lost_ledger_score:
        next(row for row in gold_rows if row["corpus_paper_id"] == 21)["scored"] = False
    if sampling_hazard == "stale_frame":
        gold_rows.extend(
            gold_rows[0] | {"cluster_block_key": "e vale", "corpus_paper_id": paper, "orcid": "0000-0003-3151-6105"}
            for paper in papers["e vale"]
        )
    ledger_changed = lost_ledger_score or sampling_hazard == "stale_frame"
    if ledger_changed:
        evidence, roles = freeze(root / "updated_evidence")
    if foreign_index:
        foreign, _ = write_name_counts_index(root / "foreign", ({"foreign": 1}, {}, {}, {}))
        index = sha256_file(Path(foreign) / "manifest.json")
    if ledger_changed or foreign_index:
        manifest[ROLE_SHA256_FIELD] = roles.sha256
        manifest[BLOCK_BINDINGS_FIELD] = roles.bindings_for(inputs.keys())
        manifest["index_sha256"] = index
        for block, binding in write_receipts().items():
            manifest["blocks"][block]["assignment_binding"] = binding
    if missing_fit_pool_block:
        inputs.pop("b crane")
        manifest["blocks"].pop("b crane")
        manifest[BLOCK_BINDINGS_FIELD] = roles.bindings_for(inputs.keys())
        rows = [row for row in rows if row[0] != "b crane"]
    if judge_as_fitted or fitted_as_judge:
        left, right, target = ("10-0", "14-0", "random") if judge_as_fitted else ("10-0", "13-0", "merged_unlabeled")
        next(row for row in rows if row[1:3] == [left, right])[3] = target
        for stratum in ("random", "merged_unlabeled"):
            ids = sorted((a, b) for block, a, b, tag, _ in rows if block == "a ross" and tag == stratum)
            counts = manifest["blocks"]["a ross"]["strata"][stratum]
            counts.update(frame_pairs=len(ids), sampled_pairs=len(ids), reporting_population_pairs=len(ids))
    if sampling_hazard in {"lost_validation", "empty_frame", "zeroed_judge_frame"}:
        block, stratum = (
            ("e vale", "merged_unlabeled") if sampling_hazard == "zeroed_judge_frame" else ("b crane", "random")
        )
        rows = [row for row in rows if (row[0], row[3]) != (block, stratum)]
        counts = manifest["blocks"][block]["strata"][stratum]
        counts.update(sampled_pairs=0, dropped_pairs=1, inclusion_probability=0)
        if sampling_hazard == "lost_validation":
            counts["drops"] = {"sampling_budget": 1}
        else:
            counts.update(frame_pairs=0, reporting_population_pairs=0, drops={"zero_cap": 1})
    if sampling_hazard == "swapped_strata":
        for endpoints, stratum in ((["10-0", "11-0"], "false_merge"), (["10-0", "12-0"], "missed_link")):
            next(row for row in rows if row[1:3] == endpoints)[3] = stratum
    changed_rows = (
        missing_fit_pool_block
        or judge_as_fitted
        or fitted_as_judge
        or sampling_hazard in {"lost_validation", "empty_frame", "swapped_strata", "zeroed_judge_frame"}
    )
    if changed_rows:
        with mined.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        manifest["output_sha256"] = sha256_file(mined)
    if changed_rows or foreign_index or ledger_changed:
        _write_json(manifest_path, manifest)

    ledger = ledger_for_roles(evidence, roles)
    assert ledger is not None
    return Stage0World(roles_path, evidence, name_counts, inputs, mined, manifest_path, roles, ledger)

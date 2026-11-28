"""Generate one tiny frozen mining handoff and its planted hazards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from s2and.arrow_tables import write_name_counts_index
from scripts.common.block_roles import BLOCK_BINDINGS_FIELD, ROLE_SHA256_FIELD, BlockRoles, load_block_roles
from scripts.common.identity_ledger import IdentityLedger, load_identity_ledger
from scripts.intake.assign_block_roles import freeze_identity_roles
from scripts.intake.build_freeze_input import build_freeze_input
from scripts.intake.build_identity_evidence import build_identity_evidence
from scripts.intake.requests_to_datasets import convert_requests
from scripts.linker.build_linker_inputs import build_linker_inputs
from scripts.pairwise.mine_pairwise_pairs import BlockInput
from tests.helpers import request_bundle
from tests.pairwise_release_helpers import write_pairwise_artifacts


@dataclass(frozen=True)
class Stage0World:
    roles_path: Path
    evidence_path: Path
    name_counts: Path
    inputs: dict[str, BlockInput]
    inputs_path: Path
    roles: BlockRoles
    ledger: IdentityLedger


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_stage0_world(
    root: Path,
    *,
    missing_fit_pool_block: bool = False,
    foreign_index: bool = False,
    lost_ledger_score: bool = False,
    empty_block: bool = False,
    all_ineligible_block: bool = False,
    validation_blocks: tuple[str, ...] = ("b crane",),
    sampling_hazard: str | None = None,
    with_query: bool = False,
) -> Stage0World:
    """Build a real intake and frozen assignments with optional handoff hazards."""
    _, name_counts, _ = write_pairwise_artifacts(root)
    snapshot = root / "snapshot"
    papers = {"a ross": [10, 11, 12, 13, 14], "b crane": [20, 21], "c lane": [30]}
    clusters = {"a ross": ["x", "y", "x", "z", "x"], "b crane": ["x", "x"]}
    if empty_block or all_ineligible_block:
        papers["e vale"] = [40]
        clusters["e vale"] = ["x"]
    if sampling_hazard == "judge_zero_cap":
        papers["e vale"] = [40, 41]
        clusters["e vale"] = ["x", "x"]
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
            freeze_input, evidence, {"fit_pool": len(clusters), "gate_thresh": 1}, extra_pins={"c lane": "gate_thresh"}
        )
        frozen["pairwise_validation_blocks"] = list(validation_blocks)
        _write_json(roles_path, frozen)
        return evidence, load_block_roles(roles_path)

    evidence, roles = freeze(root / "evidence")
    index = sha256_file(name_counts / "manifest.json")
    model = root / "model"
    model.write_bytes(b"fixture model")
    model_digest = sha256_file(model) if all_ineligible_block else "a" * 64
    query = "13-0" if with_query or sampling_hazard == "query_in_assignments" else None
    inputs = {}
    for block, cluster_ids in clusters.items():
        name = block.replace(" ", "_")
        if all_ineligible_block and block == "e vale":
            arrow_dir = snapshot / "datasets" / name
            output = root / "linker-inputs"
            build_linker_inputs(load_identity_ledger(evidence, roles), [block], output, model)
            inputs[block] = BlockInput(arrow_dir, output / name / "seeds.parquet")
            continue
        assignments = root / name / "seeds.parquet"
        assignments.parent.mkdir()
        queries = [query] if query is not None and block == "a ross" else []
        assignments.with_name("queries.json").write_text(json.dumps(queries), encoding="utf-8")
        rows = [
            (sid, cid)
            for paper, cid in zip(papers[block], cluster_ids, strict=True)
            if (sid := f"{paper}-0") not in queries
        ]
        pq.write_table(
            pa.table({"signature_id": [sid for sid, _ in rows], "component_key": [cid for _, cid in rows]}),
            assignments,
        )
        inputs[block] = BlockInput(snapshot / "datasets" / name, assignments)

    def write_receipts() -> None:
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
                    "files": {
                        path.name: sha256_file(path)
                        for path in (spec.assignments, spec.assignments.with_name("queries.json"))
                    },
                },
            )

    write_receipts()
    if sampling_hazard == "query_in_assignments":
        assignments = inputs["a ross"].assignments
        table = pq.read_table(assignments)
        pq.write_table(
            pa.concat_tables([table, pa.table({"signature_id": [query], "component_key": ["z"]})]), assignments
        )
        write_receipts()
    if lost_ledger_score:
        next(row for row in gold_rows if row["corpus_paper_id"] == 21)["scored"] = False
        evidence, roles = freeze(root / "updated_evidence")
        write_receipts()
    if foreign_index:
        foreign, _ = write_name_counts_index(root / "foreign", ({"foreign": 1}, {}, {}, {}))
        index = sha256_file(Path(foreign) / "manifest.json")
        write_receipts()
    if missing_fit_pool_block:
        inputs.pop("b crane")
    inputs_path = root / "inputs.json"
    _write_json(inputs_path, {block: {"seeds": str(spec.assignments)} for block, spec in inputs.items()})

    ledger = load_identity_ledger(evidence, roles)
    return Stage0World(roles_path, evidence, name_counts, inputs, inputs_path, roles, ledger)

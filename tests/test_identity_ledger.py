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
HOLDOUT = "0000-0003-3808-2083"


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
                "orcid": [GOLD, GOLD, GOLD, DEPOSITED],
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
    assert rows["4-0"]["rejected_gold"][0]["orcid"] == DEPOSITED
    assert rows["4-0"]["gold"] == []
    assert result["audit"]["a lovelace"]["dropped_name_mismatch"] == 1
    assert ledger.read_block("g hopper")["5-0"]["reviewed_error"]
    partition = output / json.loads((output / "manifest.json").read_text())["blocks"]["a lovelace"]["path"]
    partition.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest mismatch"):
        ledger.read_block("a lovelace")


def test_rejected_gold_vetoes_a_foreign_role(evidence_inputs: tuple[Path, Path], tmp_path: Path) -> None:
    from scripts.common.block_roles import load_block_roles
    from scripts.common.identity_ledger import ledger_for_roles
    from scripts.intake.assign_block_roles import freeze_identity_roles
    from scripts.intake.build_freeze_input import build_freeze_input
    from scripts.intake.build_identity_evidence import build_identity_evidence

    snapshot, gold = evidence_inputs
    labels = pq.read_table(gold)
    labels = labels.set_column(3, "orcid", pa.array([GOLD, GOLD, HOLDOUT, DEPOSITED]))
    foreign = dict(labels.to_pylist()[2], cluster_block_key="l a", corpus_paper_id=4, author_last="Lovelace")
    pq.write_table(pa.concat_tables([labels, pa.Table.from_pylist([foreign], schema=labels.schema)]), gold)
    evidence = tmp_path / "evidence"
    audit = build_identity_evidence(snapshot, gold, evidence)["audit"]["a lovelace"]
    source = tmp_path / "gold-blocks.parquet"
    pq.write_table(
        pa.table({"cluster_block_key": ["a lovelace", "g hopper"], "split": ["development", "holdout"]}), source
    )
    blocks = tmp_path / "blocks.parquet"
    build_freeze_input(evidence / "manifest.json", source, blocks)
    frozen = tmp_path / "roles.json"
    roles = freeze_identity_roles(blocks, evidence / "manifest.json", {"pairwise_train": 1, "final_test_a": 1})
    frozen.write_text(json.dumps(roles), encoding="utf-8")
    ledger = ledger_for_roles(evidence / "manifest.json", load_block_roles(frozen))
    assert ledger is not None
    rejected = ledger.for_block("a lovelace").eligibility("4-0")
    assert not rejected.fit_ok
    assert rejected.score_orcid is None
    assert not rejected.identities
    assert ledger.for_block("g hopper").scored() == {"5-0": HOLDOUT}
    assert audit["dropped_name_order_rotation"] == 1


def test_ambiguous_gold_position_is_never_a_mining_endpoint() -> None:
    import pandas as pd

    from scripts.common.block_roles import BlockRoles
    from scripts.common.identity_ledger import BlockEligibility
    from scripts.intake.build_identity_evidence import _block_rows
    from scripts.intake.prepare_identity_labels import SignatureJoin

    block, role, paper = "c klein", "pairwise_train", "173993366"
    rows = [
        {
            "signature_id": f"{paper}-{pos}",
            "paper_id": paper,
            "author_position": pos,
            "author_block": block,
            "author_first": first,
            "author_last": "Klein",
        }
        for pos, first in ((1, "Carol"), (2, "Charles"))
    ]
    label = {
        "cluster_block_key": block,
        "corpus_paper_id": int(paper),
        "position": 2,
        "orcid": GOLD,
        "scored": True,
        "split": "development",
    }
    evidence, audit = _block_rows(SignatureJoin(pd.DataFrame(rows)), block, [label], ())
    roles = BlockRoles({block: role}, "", {block: frozenset({GOLD})}, {GOLD: frozenset({role})})
    view = BlockEligibility(block, role, "development", {row["signature_id"]: row for row in evidence}, roles)
    decision = view.eligibility(f"{paper}-2")
    assert not decision.identities
    assert decision.score_orcid is None
    assert not decision.fit_ok
    other = view.eligibility(f"{paper}-1")
    assert not other.fit_ok
    assert other.score_orcid is None
    assert not other.identities
    assert audit["dropped_ambiguous_paper_block"] == 1


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


@pytest.mark.parametrize(
    ("gold_block", "gold_position", "reason"),
    [
        ("a lovelace", 1, "dropped_position_shift"),
        ("g hopper", 0, "dropped_block_key_mismatch"),
        ("l ada", 0, "dropped_name_order_rotation"),
    ],
)
def test_rejected_gold_never_supplies_an_identity_or_label(gold_block: str, gold_position: int, reason: str) -> None:
    import pandas as pd

    from scripts.common.block_roles import BlockRoles
    from scripts.common.identity_ledger import BlockEligibility
    from scripts.intake.build_freeze_input import mention_counts
    from scripts.intake.build_identity_evidence import _block_rows
    from scripts.intake.prepare_identity_labels import SignatureJoin

    block, role = "a lovelace", "linker_train"
    signatures = pd.DataFrame(
        {
            "signature_id": ["1-0", "2-0"],
            "paper_id": ["1", "2"],
            "author_position": [0, 0],
            "author_block": [block, block],
            "author_first": ["Ada", "Ada"],
            "author_last": ["Lovelace", "Lovelace"],
        }
    )
    gold = {"cluster_block_key": block, "position": 0, "orcid": GOLD, "scored": True, "split": "development"}
    rows, audit = _block_rows(
        SignatureJoin(signatures),
        block,
        [
            dict(gold, corpus_paper_id=1),
            dict(gold, corpus_paper_id=1, cluster_block_key=gold_block, position=gold_position),
            dict(gold, corpus_paper_id=2),
            dict(gold, corpus_paper_id=1, cluster_block_key=gold_block, position=gold_position, orcid=DEPOSITED),
        ],
        (),
    )
    assert audit[reason] == 2
    roles = BlockRoles({block: role}, "", {block: frozenset({GOLD})}, {GOLD: frozenset({role})})
    view = BlockEligibility(block, role, "development", {row["signature_id"]: row for row in rows}, roles)
    assert mention_counts(view.rows) == {GOLD: 2}
    rejected = view.eligibility("1-0")
    assert not rejected.fit_ok
    assert rejected.score_orcid is None
    assert rejected.identities == {GOLD}
    fitting_seeds = [sid for sid in view.rows if view.eligibility(sid).fit_ok]
    assert fitting_seeds == ["2-0"]
    assert view.component_label(fitting_seeds) == GOLD


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
    with pytest.raises(ValueError, match="new or previously unassigned exported blocks"):
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
    assert block.eligibility("2-0").fit_ok
    assert not block.eligibility("4-0").fit_ok
    assert block.eligibility("4-0").score_orcid is None
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
    labels = pq.read_table(gold)
    pq.write_table(labels.set_column(3, "orcid", pa.array([GOLD, GOLD, HOLDOUT, DEPOSITED])), gold)
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
    pq.write_table(pa.table({"signature_id": ["2-0", "3-0"], "cluster_id": ["a", "b"]}), assignments)
    manifest = assignment_root / "manifest.json"
    record = {
        "binding": {
            "block_bindings": ledger.roles.bindings_for({"a lovelace"}),
            "model_sha256": "a" * 64,
            "index_sha256": "b" * 64,
        },
        "files": {assignments.name: sha256_file(assignments)},
    }
    manifest.write_text(json.dumps(record))
    specs = {"a lovelace": BlockInput(snapshot / "datasets" / "a_lovelace", None, assignments)}
    output = tmp_path / "pairs.csv"
    assert mine_pairs(role_path, specs, output, per_block=1, identity_evidence=evidence) == 1
    with output.open(newline="", encoding="utf-8") as source_file:
        (row,) = csv.DictReader(source_file)
    assert {row["signature_id_1"], row["signature_id_2"]} == {"2-0", "3-0"}
    record["binding"]["block_bindings"]["a lovelace"]["role_sha256"] = "old"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not match"):
        mine_pairs(role_path, specs, tmp_path / "stale.csv", per_block=1, identity_evidence=evidence)

    pq.write_table(pa.table({"cluster_block_key": ["g hopper", "a lovelace"], "split": ["development"] * 2}), source)
    both = tmp_path / "both-blocks.parquet"
    build_freeze_input(evidence, source, both)
    role_path.write_text(json.dumps(freeze_identity_roles(both, evidence, {"pairwise_train": 2})))
    ledger = ledger_for_roles(evidence, load_block_roles(role_path))
    assert ledger is not None
    pq.write_table(pa.table({"signature_id": ["2-0", "3-0"], "cluster_id": ["a", "b"]}), assignments)
    other = tmp_path / "other" / assignments.name
    other.parent.mkdir()
    pq.write_table(pa.table({"signature_id": ["5-0"], "cluster_id": ["a"]}), other)
    for block, model_sha, path in (("a lovelace", "a" * 64, assignments), ("g hopper", "c" * 64, other)):
        (path.parent / "manifest.json").write_text(
            json.dumps(
                {
                    "binding": {
                        "block_bindings": ledger.roles.bindings_for({block}),
                        "model_sha256": model_sha,
                        "index_sha256": "b" * 64,
                    },
                    "files": {path.name: sha256_file(path)},
                }
            )
        )
    specs["g hopper"] = BlockInput(snapshot / "datasets" / "g_hopper", None, other)
    with pytest.raises(ValueError, match="model_sha256"):
        mine_pairs(role_path, specs, tmp_path / "mixed-model.csv", per_block=1, identity_evidence=evidence)


def test_evidence_rejects_unmatched_gold_and_stale_exclusions(
    evidence_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.common import identity_ledger, orcid_references
    from scripts.intake.build_identity_evidence import build_identity_evidence

    snapshot, gold = evidence_inputs
    original = pq.read_table(gold)
    output = tmp_path / "identity"
    missing = original.slice(0, 1).set_column(1, "corpus_paper_id", pa.array([999]))
    pq.write_table(pa.concat_tables([original, missing]), gold)
    with pytest.raises(ValueError, match="a lovelace.*unmatched gold"):
        build_identity_evidence(snapshot, gold, output)
    assert not (output / "manifest.json").exists()
    pq.write_table(missing, gold)
    with pytest.raises(ValueError, match="No gold rows matched"):
        build_identity_evidence(snapshot, gold, output, max_unmatched_gold=1)
    pq.write_table(original, gold)
    build_identity_evidence(snapshot, gold, output)
    monkeypatch.setattr(orcid_references, "EXCLUDED_ORCID_GROUPS", frozenset({GOLD}))
    with pytest.raises(ValueError, match="excluded ORCID groups"):
        identity_ledger.load_identity_evidence(output / "manifest.json")


@pytest.mark.parametrize("check", ["load", "train", "subset", "empty"])
def test_frozen_trainer_requires_current_exclusions_and_complete_roles(
    check: str, evidence_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.common import orcid_references
    from scripts.common.block_roles import load_block_roles
    from scripts.intake.assign_block_roles import freeze_identity_roles
    from scripts.intake.build_freeze_input import build_freeze_input
    from scripts.intake.build_identity_evidence import build_identity_evidence
    from scripts.linker.train_linker_and_finalize import _validate_frozen_task_bindings

    snapshot, gold = evidence_inputs
    evidence = tmp_path / "evidence" / "manifest.json"
    build_identity_evidence(snapshot, gold, evidence.parent)
    source = tmp_path / "gold-blocks.parquet"
    pq.write_table(pa.table({"cluster_block_key": ["a lovelace", "g hopper"], "split": ["development"] * 2}), source)
    blocks = tmp_path / "blocks.parquet"
    build_freeze_input(evidence, source, blocks)
    path = tmp_path / "assignments.json"
    path.write_text(json.dumps(freeze_identity_roles(blocks, evidence, {"linker_train": 2})), encoding="utf-8")
    roles = load_block_roles(path)
    manifest = {
        "tables": {"linker_train": {"blocks": dict.fromkeys(roles.block_to_role)}},
        "block_bindings": roles.bindings_for(roles.block_to_role),
    }
    _validate_frozen_task_bindings(manifest, roles)
    if check in {"load", "train"}:
        exclusions = orcid_references.EXCLUDED_ORCID_GROUPS | {HOLDOUT}
        monkeypatch.setattr(orcid_references, "EXCLUDED_ORCID_GROUPS", exclusions)
        with pytest.raises(ValueError, match="excluded ORCID groups"):
            if check == "load":
                load_block_roles(path)
            else:
                _validate_frozen_task_bindings(manifest, roles)
    else:
        subset = {"a lovelace": {}} if check == "subset" else {}
        with pytest.raises(ValueError, match="Missing.*g hopper"):
            _validate_frozen_task_bindings(
                {"tables": {"linker_train": {"blocks": subset}}, "block_bindings": roles.bindings_for(subset)}, roles
            )


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

    with pytest.raises(ValueError, match="No.*requests.*a lovelace"):
        _arrival_requests(arrivals, "a lovelace", mentions, {"1-0"}, datetime(2026, 10, 20, tzinfo=UTC), 1)
    floats = pq.read_table(arrivals)
    index = floats.schema.get_field_index("corpus_paper_id")
    pq.write_table(floats.set_column(index, "corpus_paper_id", floats.column(index).cast(pa.float64())), arrivals)
    with pytest.raises(ValueError, match="arrival.*key"):
        _arrival_requests(arrivals, "a lovelace", mentions, mentions.keys(), datetime(2026, 10, 20, tzinfo=UTC), 1)

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


def test_v14_inventory_includes_null_orcid_candidate_members(tmp_path: Path) -> None:
    from scripts.intake.build_v14_training_inventory import build_training_inventory
    from scripts.intake.requests_to_datasets import convert_requests

    export = request_bundle(
        tmp_path / "raw",
        "h wang",
        [("10", GOLD), ("20", None), ("30", None), ("40", None), ("50", None), ("193154881", None)],
    )
    index, _ = write_name_counts_index(tmp_path, ({}, {}, {}, {}))
    root = tmp_path / "linker"
    convert_requests(
        [export],
        root,
        Path(index),
        allowed_papers=frozenset({"10", "20", "30", "40", "50", "193154881"}),
        abstract_allowed_papers=frozenset(),
    )
    train = root / "train.parquet"
    pq.write_table(
        pa.table({"dataset": ["h_wang"], "query_signature_id": ["10-0"], "candidate_component_key": ["used"]}), train
    )
    members = root / "members.parquet"
    pq.write_table(
        pa.table(
            {
                "candidate_component_key": ["used", "used", "unused", "cal", "classic"],
                "signature_id": ["20-0", "30-0", "40-0", "193154881-0", "50-0"],
            }
        ),
        members,
    )
    tables = {"train_path": train.name}
    for key, components, groups in (
        ("classic_gate_source_path", ["classic"], ["classic-query"]),
        ("hwang_eval_path", ["cal", "unused"], ["cal-query", "test-query"]),
        ("s2and_eval_path", ["unused"], ["bench-query"]),
    ):
        tables[key] = f"{key}.parquet"
        pq.write_table(
            pa.table(
                {
                    "dataset": ["h_wang"] * len(components),
                    "query_signature_id": ["10-0"] * len(components),
                    "query_group_id": groups,
                    "candidate_component_key": components,
                }
            ),
            root / tables[key],
        )
    (root / "splits.csv").write_text(
        "query_group_id,base_group_id,source_key,split\n"
        "cal-query,cal,hwang_eval,calibration_check\n"
        "test-query,test,hwang_eval,test\n",
        encoding="utf-8",
    )
    (root / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_name": "fixture",
                "expected_metrics": {},
                "models": {
                    "classic": {
                        **tables,
                        "stratified_eval_test_split": {"assignments_path": "splits.csv"},
                        "promoted_stratified_gate": {"calibration_splits": ["calibration_fit", "calibration_check"]},
                    }
                },
                "assets": {
                    "featureless_rows": {"files": tables},
                    "candidate_members": {"datasets": {"h_wang": members.name}},
                },
            }
        )
    )
    gold = tmp_path / "gold.parquet"
    pq.write_table(
        pa.table({"cluster_block_key": [" A_KHAN "], "corpus_paper_id": [99], "position": [0], "orcid": [DEPOSITED]}),
        gold,
    )
    output = tmp_path / "inventory.parquet"
    build_training_inventory(gold, [], output, linker_sources=[root])
    observed = {row["signature_id"]: row["orcid"] for row in pq.read_table(output).to_pylist()}
    assert observed == {"10-0": GOLD, "20-0": None, "30-0": None, "50-0": None, "193154881-0": None, "99-0": DEPOSITED}

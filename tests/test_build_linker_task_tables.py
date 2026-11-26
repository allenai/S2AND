from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s2and import __version__
from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import write_name_counts_index
from s2and.incremental_linking.features import promoted_linker_feature_columns
from scripts.common.block_roles import load_block_roles
from scripts.common.distance_policy import DISTANCE_POLICY_VERSION
from scripts.common.orcid_references import read_external_reference_identity_evidence
from scripts.intake.assign_block_roles import BlockSpec, assign_block_roles
from scripts.intake.prepare_identity_labels import prepare_identity_labels
from scripts.linker.build_linker_task_tables import (
    TASK_MANIFEST,
    TASK_ROLES,
    _freeze_block_rows,
    build_linker_task_tables,
    read_seed_components,
)
from scripts.linker.train_linker_and_finalize import _materialize_frozen_linker_tasks
from tests.helpers import write_arrow_tables_bundle, write_test_arrow_artifact_manifest
from tests.stage0_world import build_stage0_world

pytestmark = pytest.mark.tooling

_OUTSIDE_HOME = "0000-0002-1694-233X"
_ORCIDS = ("0000-0001-5000-0007", "0000-0002-9079-593X", "0000-0002-1825-0855")


class _Pairwise:
    def predict_proba_positive(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 0.8)


def _block_inputs(
    root: Path,
    block: str,
    orcid: str,
    *,
    recent_seeds: int = 29,
    deposited_orcids: bool = True,
    query_year: int = 2020,
    recent_seed_year: int = 2020,
) -> dict[str, str]:
    root.mkdir()
    arrow_dir = root / "arrow"
    n_seeds = 29
    query_ids = ["q_initial", "q_full", "q_outside_home"]
    seed_ids = [f"s{i:02}" for i in range(n_seeds)]
    signatures = query_ids + seed_ids
    size = len(signatures)
    first = ["A", *(["Alice"] * (size - 1))]
    matching = [True] * 6 + [False] * (size - 6)
    tables = {
        "signatures": pa.table(
            {
                "signature_id": signatures,
                "paper_id": signatures,
                "author_first": first,
                "author_middle": ["Marie"] * size,
                "author_last": ["Ross"] * size,
                "author_suffix": [""] * size,
                "author_position": pa.array([0] * size, type=pa.int64()),
                "author_block": [block] * size,
                "author_affiliations": pa.array([["Lab"]] * size, type=pa.list_(pa.string())),
                "author_orcid": pa.array(
                    [
                        orcid,
                        orcid,
                        _OUTSIDE_HOME,
                        orcid,
                        orcid,
                        orcid,
                        _OUTSIDE_HOME,
                        *([None] * (size - 7)),
                    ]
                    if deposited_orcids
                    else [None] * size,
                    type=pa.string(),
                ),
            }
        ),
        "papers": pa.table(
            {
                "paper_id": signatures,
                "title": ["Graph models" if hit else "Genetics" for hit in matching],
                "venue": ["Conf"] * size,
                "journal_name": [""] * size,
                "year": pa.array(
                    [query_year] * len(query_ids)
                    + [recent_seed_year] * recent_seeds
                    + [1900] * (n_seeds - recent_seeds),
                    type=pa.int64(),
                ),
            }
        ),
        "paper_authors": pa.table(
            {
                "paper_id": signatures,
                "position": pa.array([0] * size, type=pa.int64()),
                "author_name": [f"{name} Marie Ross" for name in first],
            }
        ),
        "specter": pa.table(
            {
                "paper_id": signatures,
                "embedding": pa.array(
                    [[1.0, 0.0] if hit else [0.0, 1.0] for hit in matching], type=pa.list_(pa.float32(), 2)
                ),
            }
        ),
    }
    paths = write_arrow_tables_bundle(arrow_dir, tables)
    paths["name_counts_index"], _ = write_name_counts_index(
        arrow_dir / "name_counts_index", ({"alice": 10}, {"ross": 30}, {"alice ross": 5}, {"ross a": 8})
    )
    write_test_arrow_artifact_manifest(arrow_dir, paths)
    seeds_path, labels_path, query_path = (root / name for name in ("seeds.parquet", "labels.parquet", "queries.json"))
    pd.DataFrame({"signature_id": seed_ids, "component_key": [f"c{i:02}" for i in range(n_seeds)]}).to_parquet(
        seeds_path
    )
    pd.DataFrame(
        {
            "signature_id": query_ids + ["s00", "s01", "s03"],
            "orcid": [orcid, orcid, _OUTSIDE_HOME, orcid, orcid, _OUTSIDE_HOME],
            "scored": True,
        }
    ).to_parquet(labels_path)
    query_path.write_text(json.dumps(query_ids), encoding="utf-8")
    return {
        "arrow_dir": str(arrow_dir),
        "seeds": str(seeds_path),
        "labels": str(labels_path),
        "queries": str(query_path),
    }


@pytest.mark.parametrize(
    ("hazard", "error"),
    [
        ("missing_seed", r"Fitting seed coverage differs in 'a ross': missing=\['11-0'\]"),
        ("unscored_query", r"Unscored fitting queries in 'a ross': \['14-0'\]"),
    ],
)
def test_fitting_linker_inputs_reconcile_queries_and_seeds(tmp_path: Path, hazard: str, error: str) -> None:
    world = build_stage0_world(tmp_path)
    block = "a ross"
    spec = world.inputs[block]
    directory = spec.assignments.parent
    receipt_path = directory / "manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    common = {key: value for key, value in receipt["binding"].items() if key != "block_bindings"}
    binding_path = directory / "inputs_binding.json"
    binding_path.write_text(json.dumps(common), encoding="utf-8")
    seeds, queries = directory / "seeds.parquet", directory / "queries.json"
    inputs_path = directory / "inputs.json"
    inputs_path.write_text(
        json.dumps(
            {
                block: {
                    "arrow_dir": str(spec.arrow_dir),
                    "input_manifest": receipt_path.name,
                    "seeds": seeds.name,
                    "queries": queries.name,
                }
            }
        ),
        encoding="utf-8",
    )

    def write_inputs(query_ids: list[str], omitted: set[str]) -> None:
        queries.write_text(json.dumps(query_ids), encoding="utf-8")
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"signature_id": row["signature_id"], "component_key": row["cluster_id"]}
                    for row in pq.read_table(spec.assignments).to_pylist()
                    if row["signature_id"] not in omitted
                ]
            ),
            seeds,
        )
        receipt["files"].update({path.name: sha256_file(path) for path in (seeds, queries)})
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    write_inputs(["10-0"], {"10-0"})
    complete = tmp_path / "complete"
    build_linker_task_tables(
        world.roles_path,
        inputs_path,
        complete,
        blocks=[block],
        identity_evidence=world.evidence_path,
        holdout_fraction=0,
    )
    assert pq.read_table(complete / "fit_pool.parquet")["label"].to_pylist().count(1) == 1
    if hazard == "missing_seed":
        write_inputs(["10-0"], {"10-0", "11-0"})
    else:
        write_inputs(["10-0", "14-0"], {"10-0", "14-0"})
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match=error):
        build_linker_task_tables(
            world.roles_path,
            inputs_path,
            rejected,
            blocks=[block],
            identity_evidence=world.evidence_path,
            holdout_fraction=0,
        )
    assert not (rejected / TASK_MANIFEST).exists()


def test_frozen_linker_tables_retrieve_after_author_exclusion_and_feed_trainer(tmp_path: Path) -> None:
    blocks = dict(zip(TASK_ROLES, ("a ross", "b ross", "c ross"), strict=True))
    orcids = dict(zip(TASK_ROLES, _ORCIDS, strict=True))
    inputs = {block: _block_inputs(tmp_path / role, block, orcids[role]) for role, block in blocks.items()}
    specs = [BlockSpec(block, 32, frozenset({orcids[role], _OUTSIDE_HOME})) for role, block in blocks.items()]
    specs.append(BlockSpec("sealed", 2, frozenset({_OUTSIDE_HOME})))
    frozen = assign_block_roles(
        specs,
        dict.fromkeys(TASK_ROLES, 1),
        pinned_roles={**{key: role for role, key in blocks.items()}, "sealed": "final_test_a"},
        closure_mode="block_only",
    )
    roles_path, inputs_path, output = tmp_path / "roles.json", tmp_path / "inputs.json", tmp_path / "tasks"
    roles_path.write_text(json.dumps(frozen), encoding="utf-8")
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
    subset = build_linker_task_tables(roles_path, inputs_path, tmp_path / "subset", blocks=["B_ROSS"])
    assert set(subset["tables"]) == {"gate_fit"}
    assert set(subset["tables"]["gate_fit"]["blocks"]) == {"b ross"}
    manifest = build_linker_task_tables(roles_path, inputs_path, output, holdout_fraction=1)
    assert manifest["distance_policy_version"] == DISTANCE_POLICY_VERSION
    assert manifest["s2and_version"] == __version__
    assert set(manifest["tables"]) == set(TASK_ROLES)
    assert manifest["tables"]["gate_fit"]["blocks"]["b ross"]["counts"]["full_windows"] == 2
    for role in ("fit_pool", "gate_fit"):
        rows = pd.read_parquet(output / f"{role}.parquet")
        assert rows.groupby("query_signature_id").size().to_dict() == {"q_full": 25, "q_initial": 25}
        assert rows["whole_author_holdout"].all()
        assert not rows["label"].any()
        assert not set(rows["candidate_component_key"]) & {"c00", "c01", "c02"}
        assert set(rows["query_author_middle"]) == {"Marie"}
        assert rows.groupby("query_signature_id")["query_view"].first().to_dict() == {
            "q_full": "full",
            "q_initial": "initial_only",
        }
    natural = pd.read_parquet(output / "gate_thresh.parquet")
    assert not natural["whole_author_holdout"].any()
    assert int(natural.loc[natural["query_signature_id"].eq("q_full"), "label"].sum()) == 2
    assert not natural.loc[natural["query_signature_id"].eq("q_outside_home"), "candidate_gold_scored"].any()

    clusterer = SimpleNamespace(
        classifier=_Pairwise(),
        featurizer_info=FeaturizationInfo(["year_diff"]),
        nameless_classifier=None,
        nameless_featurizer_info=None,
    )
    roles = load_block_roles(roles_path)
    feature_rows, summaries = _materialize_frozen_linker_tasks(
        output / TASK_MANIFEST,
        roles,
        target_features=promoted_linker_feature_columns(),
        name_tuples=frozenset(),
        clusterer=clusterer,
        n_jobs=1,
        total_ram_bytes=1_000_000_000,
        max_query_groups=None,
    )
    assert set(feature_rows) == set(TASK_ROLES)
    assert sum(s["rows"] for s in summaries) == 175
    for role, rows in feature_rows.items():
        frozen_rows = pd.read_parquet(output / f"{role}.parquet")
        assert rows["candidate_component_key"].tolist() == frozen_rows["candidate_component_key"].tolist()
        np.testing.assert_array_equal(rows["retrieval_rank"], frozen_rows["retrieval_rank"])
        np.testing.assert_allclose(rows["retrieval_score"], frozen_rows["retrieval_score"])
        assert set(promoted_linker_feature_columns()) <= set(rows)
    manifest_path = output / TASK_MANIFEST
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_text(
        json.dumps({**manifest, "distance_policy_version": DISTANCE_POLICY_VERSION - 1}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="different distance policy"):
        _materialize_frozen_linker_tasks(
            manifest_path,
            roles,
            target_features=(),
            name_tuples=frozenset(),
            clusterer=clusterer,
            n_jobs=1,
            total_ram_bytes=1_000_000_000,
            max_query_groups=None,
        )
    manifest_path.write_bytes(original_manifest)
    (output / "fit_pool.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen linker table changed"):
        _materialize_frozen_linker_tasks(
            output / TASK_MANIFEST,
            roles,
            target_features=(),
            name_tuples=frozenset(),
            clusterer=clusterer,
            n_jobs=1,
            total_ram_bytes=1_000_000_000,
            max_query_groups=None,
        )


def test_pairwise_validation_blocks_feed_ranker_and_can_be_enabled_for_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.linker.build_linker_task_tables import TASK_BLOCK_ROLES

    specs = [
        BlockSpec(key, 32, frozenset({identity, _OUTSIDE_HOME}))
        for key, identity in zip(("a ross", "b ross"), _ORCIDS[:2], strict=True)
    ]
    frozen = assign_block_roles(specs, {"fit_pool": 2})
    (block,) = frozen["pairwise_validation_blocks"]
    identity = next(iter(next(spec.orcids for spec in specs if spec.block_key == block) - {_OUTSIDE_HOME}))
    roles_path = tmp_path / "roles.json"
    roles_path.write_text(json.dumps(frozen), encoding="utf-8")
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(json.dumps({block: _block_inputs(tmp_path / "block", block, identity)}), encoding="utf-8")
    ranker = build_linker_task_tables(roles_path, inputs_path, tmp_path / "ranker", blocks=[block])
    assert set(ranker["tables"]) == {"fit_pool"}
    assert set(ranker["tables"]["fit_pool"]["blocks"]) == {block}
    monkeypatch.setitem(TASK_BLOCK_ROLES, "gate_fit", frozenset({"gate_fit", "fit_pool"}))
    shared = build_linker_task_tables(roles_path, inputs_path, tmp_path / "shared", blocks=[block])
    assert set(shared["tables"]) == {"fit_pool", "gate_fit"}
    assert set(shared["tables"]["gate_fit"]["blocks"]) == {block}


@pytest.mark.parametrize("recent_seeds", [1, 3])
def test_frozen_builder_keeps_native_short_windows(tmp_path: Path, recent_seeds: int) -> None:
    block = "a ross"
    spec = _block_inputs(tmp_path / "block", block, _ORCIDS[0], recent_seeds=recent_seeds)
    rows, counts = _freeze_fixture(spec, block, holdout_fraction=0)
    assert rows["retrieval_rank"].tolist() == list(range(1, recent_seeds + 1))
    assert rows["candidate_component_key"].tolist() == [f"c{i:02}" for i in range(recent_seeds)]
    assert counts["short_windows"] == 1


def test_seed_identity_evidence_survives_scoring_masks(tmp_path: Path) -> None:
    block = "a ross"
    spec = _block_inputs(tmp_path / "block", block, _ORCIDS[0])
    seeds = pd.read_parquet(spec["seeds"])
    seeds.loc[seeds["signature_id"].eq("s03"), "component_key"] = "c00"
    seeds.to_parquet(spec["seeds"], index=False)
    natural, _ = _freeze_fixture(spec, block, holdout_fraction=0)
    mixed = natural.loc[natural["candidate_component_key"].eq("c00")].iloc[0]
    assert not mixed["candidate_gold_scored"]
    assert mixed["label"] == 0

    labels = pd.read_parquet(spec["labels"]).assign(scored=True)
    labels.loc[len(labels)] = ["s04", _ORCIDS[0], False]
    labels.to_parquet(spec["labels"], index=False)
    heldout, _ = _freeze_fixture(spec, block, holdout_fraction=1)
    assert heldout["whole_author_holdout"].all()
    assert "c04" not in set(heldout["candidate_component_key"])
    assert not heldout["candidate_gold_scored"].any()


def _freeze_fixture(
    spec: dict[str, str], block: str, *, holdout_fraction: float
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(spec["labels"]).parent / "roles.json"
    roles = assign_block_roles(
        [
            BlockSpec(block, 32, frozenset({_ORCIDS[0], _OUTSIDE_HOME})),
            BlockSpec("sealed", 2, frozenset({_OUTSIDE_HOME})),
        ],
        {"gate_fit": 1},
        pinned_roles={block: "gate_fit", "sealed": "final_test_a"},
        closure_mode="block_only",
    )
    path.write_text(json.dumps(roles), encoding="utf-8")
    identities, labels = read_external_reference_identity_evidence(Path(spec["labels"]))
    with ArrowDataset.open(spec["arrow_dir"], require_specter=True, require_name_counts_index=True) as arrow:
        return _freeze_block_rows(
            arrow,
            block=block,
            block_roles=load_block_roles(path),
            seeds=read_seed_components(Path(spec["seeds"])),
            query_ids=["q_full"],
            labels=labels,
            identities=identities,
            holdout_fraction=holdout_fraction,
            seed=42,
            n_jobs=1,
        )


@pytest.mark.parametrize("check", ["mixed_seed", "heldout_fragment", "missing_scored"])
def test_joined_identity_evidence_reaches_task_builder(tmp_path: Path, check: str) -> None:
    blocks = dict(zip(TASK_ROLES, ("a ross", "b ross", "c ross"), strict=True))
    orcids = dict(zip(TASK_ROLES, _ORCIDS, strict=True))
    inputs = {
        block: _block_inputs(tmp_path / role, block, orcids[role], deposited_orcids=False)
        for role, block in blocks.items()
    }
    frozen = assign_block_roles(
        [
            *[BlockSpec(block, 32, frozenset({orcids[role], _OUTSIDE_HOME})) for role, block in blocks.items()],
            BlockSpec("sealed", 2, frozenset({_OUTSIDE_HOME})),
        ],
        dict.fromkeys(TASK_ROLES, 1),
        pinned_roles={**{block: role for role, block in blocks.items()}, "sealed": "final_test_a"},
        closure_mode="block_only",
    )
    roles_path, inputs_path = tmp_path / "roles.json", tmp_path / "inputs.json"
    roles_path.write_text(json.dumps(frozen), encoding="utf-8")
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
    roles = load_block_roles(roles_path)
    for role, block in blocks.items():
        spec = inputs[block]
        label_path = Path(spec["labels"])
        gold = pd.read_parquet(label_path).rename(columns={"signature_id": "paper_id"}).assign(position=0)
        gold.loc[len(gold)] = ["s04", orcids[role], False, 0]
        gold_path = label_path.with_name("gold.parquet")
        gold.to_parquet(gold_path, index=False)
        prepare_identity_labels(Path(spec["arrow_dir"]), gold_path, label_path, block_roles=roles)

    spec = inputs[blocks["gate_thresh"]]
    seeds = pd.read_parquet(spec["seeds"])
    seeds.loc[seeds["signature_id"].eq("s03"), "component_key"] = "c00"
    seeds.to_parquet(spec["seeds"], index=False)
    output = tmp_path / "tasks"
    if check == "missing_scored":
        label_path = Path(inputs[blocks["fit_pool"]]["labels"])
        pd.read_parquet(label_path).drop(columns="scored", errors="ignore").to_parquet(label_path, index=False)
        with pytest.raises(ValueError, match="missing required columns.*scored"):
            build_linker_task_tables(roles_path, inputs_path, output, holdout_fraction=1)
        return
    build_linker_task_tables(roles_path, inputs_path, output, holdout_fraction=1)
    if check == "mixed_seed":
        natural = pd.read_parquet(output / "gate_thresh.parquet")
        mixed = natural.loc[natural["candidate_component_key"].eq("c00")]
        assert len(mixed) > 0
        assert not mixed["candidate_gold_scored"].any()
        assert not mixed["label"].any()
    else:
        heldout = pd.read_parquet(output / "gate_fit.parquet")
        assert heldout["whole_author_holdout"].all()
        assert not set(heldout["candidate_component_key"]) & {"c00", "c01", "c04"}

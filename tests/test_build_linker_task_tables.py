from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import write_name_counts_index
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.logistic_gate import (
    build_logistic_gate_matrix,
    default_logistic_gate_feature_names,
    feature_values_from_candidate_frame,
)
from s2and.incremental_linking.policy import NAME_COUNTS_MANIFEST_SHA256_FIELD
from scripts.common.block_roles import BlockRoles
from scripts.common.distance_policy import DISTANCE_POLICY_VERSION
from scripts.common.identity_ledger import BlockEligibility, MentionEvidence
from scripts.linker import build_linker_inputs
from scripts.linker.build_linker_task_tables import (
    TASK_MANIFEST,
    _freeze_block_rows,
    _read_signatures,
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
                [row for row in pq.read_table(spec.assignments).to_pylist() if row["signature_id"] not in omitted]
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


def test_frozen_linker_tables_retrieve_after_author_exclusion_and_feed_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_stage0_world(tmp_path)

    def predict(blocks: dict[str, list[str]], *_args: object, **_kwargs: object):
        return {sid: [sid] for signatures in blocks.values() for sid in signatures}, None

    monkeypatch.setattr(
        build_linker_inputs, "load_production_model", lambda *_a, **_kw: SimpleNamespace(predict_from_arrow=predict)
    )
    input_dir, output = tmp_path / "inputs", tmp_path / "tasks"
    build_linker_inputs.build_linker_inputs(
        world.ledger, world.inputs.keys(), input_dir, tmp_path / "model", queries_per_block=1
    )
    manifest = build_linker_task_tables(
        world.roles_path,
        input_dir / "inputs.json",
        output,
        blocks=list(world.inputs),
        holdout_fraction=1,
        identity_evidence=world.evidence_path,
    )
    assert set(manifest["tables"]) == {"fit_pool"}
    assert set(manifest["tables"]["fit_pool"]["blocks"]) == set(world.inputs)
    rows = pd.read_parquet(output / "fit_pool.parquet")
    assert not rows.empty
    assert rows["whole_author_holdout"].all()
    assert not rows["label"].any()

    clusterer = SimpleNamespace(
        classifier=_Pairwise(),
        featurizer_info=FeaturizationInfo(["year_diff"]),
        nameless_classifier=None,
        nameless_featurizer_info=None,
        feature_contract={NAME_COUNTS_MANIFEST_SHA256_FIELD: world.ledger.evidence.name_counts_index_sha256},
    )
    roles = world.roles
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
    assert set(feature_rows) == {"fit_pool"}
    assert sum(s["rows"] for s in summaries) == len(rows)
    for role, rows in feature_rows.items():
        frozen_rows = pd.read_parquet(output / f"{role}.parquet")
        assert rows["candidate_component_key"].tolist() == frozen_rows["candidate_component_key"].tolist()
        np.testing.assert_array_equal(rows["retrieval_rank"], frozen_rows["retrieval_rank"])
        np.testing.assert_allclose(rows["retrieval_score"], frozen_rows["retrieval_score"])
        assert set(promoted_linker_feature_columns()) <= set(rows)
    fit_rows = feature_rows["fit_pool"]
    gate_columns = default_logistic_gate_feature_names()
    matrix, choices = build_logistic_gate_matrix(
        gate_columns,
        query_indices=fit_rows["query_group_id"].to_numpy(),
        probabilities=np.full(len(fit_rows), 0.8),
        feature_values=feature_values_from_candidate_frame(fit_rows),
        retrieval_ranks=fit_rows["retrieval_rank"].to_numpy(),
        component_keys=fit_rows["candidate_component_key"].to_numpy(),
    )
    bucket_column = gate_columns.index("first_name_bucket_single_letter_first")
    np.testing.assert_array_equal(
        matrix[:, bucket_column], fit_rows.iloc[choices.best_rows]["query_view"].eq("initial_only")
    )
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
    seeds.loc[seeds["signature_id"].eq("s02"), "component_key"] = "c00"
    seeds.to_parquet(spec["seeds"], index=False)
    labels = pd.read_parquet(spec["labels"])
    labels.loc[len(labels)] = ["s02", _ORCIDS[1], True]
    labels.to_parquet(spec["labels"], index=False)
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
    identities = frozenset({*_ORCIDS, _OUTSIDE_HOME})
    roles = BlockRoles(
        {block: "fit_pool", "sealed": "final_test"},
        "",
        {block: identities, "sealed": frozenset({_OUTSIDE_HOME})},
        identity_roles={
            identity: frozenset({"fit_pool", "final_test"} if identity == _OUTSIDE_HOME else {"fit_pool"})
            for identity in identities
        },
    )
    labels = {row.signature_id: row for row in pd.read_parquet(spec["labels"]).itertuples(index=False)}
    with ArrowDataset.open(spec["arrow_dir"], require_specter=True, require_name_counts_index=True) as arrow:
        rows: dict[str, MentionEvidence] = {}
        for sid in _read_signatures(arrow, block):
            label = labels.get(sid)
            rows[sid] = {
                "block_key": block,
                "signature_id": sid,
                "corpus_paper_id": sid,
                "position": 0,
                "gold": [{"orcid": label.orcid, "scored": label.scored, "split": "development"}] if label else [],
                "deposited_orcid": None,
                "excluded": False,
                "reviewed_error": sid == "q_initial",
                "evidence_state": "single" if label else "none",
            }
        eligibility = BlockEligibility(block, "fit_pool", "development", rows, roles)
        return _freeze_block_rows(
            arrow,
            seeds={
                sid: component
                for sid, component in read_seed_components(Path(spec["seeds"])).items()
                if eligibility.eligibility(sid).fit_ok
            },
            query_ids=["q_full"],
            eligibility=eligibility,
            holdout_fraction=holdout_fraction,
            seed=42,
            n_jobs=1,
        )

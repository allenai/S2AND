from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import write_name_counts_index
from s2and.incremental_linking.features import promoted_linker_feature_columns
from scripts.production.data.assign_block_roles import BlockSpec, assign_block_roles
from scripts.production.data.block_roles import load_block_roles
from scripts.production.data.build_linker_task_tables import TASK_MANIFEST, TASK_ROLES, build_linker_task_tables
from scripts.production.model.train_linker_and_finalize import _materialize_frozen_linker_tasks
from tests.helpers import write_arrow_tables_bundle, write_test_arrow_artifact_manifest

_OUTSIDE_HOME = "0000-0002-1694-233X"
_ORCIDS = ("0000-0001-5000-0007", "0000-0002-9079-593X", "0000-0002-1825-0855")


class _Pairwise:
    def predict_proba_positive(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 0.8)


def _block_inputs(root: Path, block: str, orcid: str) -> dict[str, str]:
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
                "author_orcid": [
                    orcid,
                    orcid,
                    _OUTSIDE_HOME,
                    orcid,
                    orcid,
                    orcid,
                    _OUTSIDE_HOME,
                    *([None] * (size - 7)),
                ],
            }
        ),
        "papers": pa.table(
            {
                "paper_id": signatures,
                "title": ["Graph models" if hit else "Genetics" for hit in matching],
                "venue": ["Conf"] * size,
                "journal_name": [""] * size,
                "year": pa.array([2020] * size, type=pa.int64()),
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
        }
    ).to_parquet(labels_path)
    query_path.write_text(json.dumps(query_ids), encoding="utf-8")
    return {
        "arrow_dir": str(arrow_dir),
        "seeds": str(seeds_path),
        "labels": str(labels_path),
        "queries": str(query_path),
    }


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
        closure_mode="home_pool",
    )
    roles_path, inputs_path, output = tmp_path / "roles.json", tmp_path / "inputs.json", tmp_path / "tasks"
    roles_path.write_text(json.dumps(frozen), encoding="utf-8")
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
    manifest = build_linker_task_tables(roles_path, inputs_path, output, holdout_fraction=1)
    assert set(manifest["tables"]) == set(TASK_ROLES)
    assert manifest["tables"]["gate_fit"]["blocks"]["b ross"]["counts"]["full_windows"] == 2
    for role in ("linker_train", "gate_fit"):
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
    (output / "linker_train.parquet").write_bytes(b"changed")
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

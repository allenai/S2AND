"""Verify linker materializer catalogs against native candidate plans."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from s2and import feature_port
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking import runtime as linker_runtime
from s2and.incremental_linking.feature_block import write_raw_arrow_batch_lookup_indexes
from s2and.incremental_linking.retrieval import RawArrowPlanBundle, build_linker_retrieval_batch_from_raw_plan_bundle
from scripts.production.model import train_linker_and_finalize as trainer
from scripts.production.model.initial_query_sources import derived_query_id
from tests.raw_arrow_helpers import base_arrow_paths, open_arrow_dataset, write_ipc


def _context_paths(root: Path) -> dict[str, str]:
    paths = base_arrow_paths(root)
    for key in ("signatures", "papers", "paper_authors"):
        with pa.OSFile(paths[key], "rb") as source:
            table = pa.ipc.open_file(source).read_all()
        rows = table.to_pylist()
        if key == "signatures":
            rows[2].update(author_first="Alice", author_last="Wang", author_affiliations=["AI Lab"])
        elif key == "papers":
            rows[2].update(title="Graph Models", venue="NeurIPS", year=2020)
        else:
            rows[4]["author_name"] = "Alice Wang"
            rows[5]["author_name"] = "Dan Lee"
            rows.insert(2, {"paper_id": "p_q", "position": 2, "author_name": "Dan Lee"})
        paths[key] = write_ipc(root / f"context_{key}.arrow", pa.Table.from_pylist(rows, schema=table.schema))
    paths, _metrics = write_raw_arrow_batch_lookup_indexes(paths, root)
    return paths


class _ConstantPairwise:
    def predict_proba_positive(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 0.8)


def test_materializer_ranks_match_the_native_labeled_plan(tmp_path: Path) -> None:
    """Duplicate and mixed profiles keep the ranks of the native labeled plan."""
    source = tmp_path / "source"
    paths = _context_paths(source / "datasets" / "toy")
    (source / "splits").mkdir()
    (source / "labels").mkdir()
    members = {
        "a": ("s1",),
        "b": ("s2",),
        "absent_a": ("s1",),
        "mixed": ("s1", "s2"),
    }
    rows = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "query_signature_id": "q1",
                "query_group_id": group,
                "base_group_id": "q1",
                "query_view": "full",
                "query_author": "alice wang",
                "query_first_token": "alice",
                "candidate_component_key": component,
                "retrieval_rank": rank,
                "label": int(component == "a"),
            }
            for group, component, rank in (
                ("present", "a", 1),
                ("absent", "absent_a", 1),
                ("mixed", "mixed", 1),
                ("present", "b", 2),
            )
        ]
    )
    rows.to_parquet(source / "labels" / "train.parquet", index=False)
    pd.DataFrame(
        [
            {"candidate_component_key": key, "member_index": index, "signature_id": signature}
            for key, signatures in members.items()
            for index, signature in enumerate(signatures)
        ]
    ).to_parquet(source / "members.parquet", index=False)
    (source / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_name": "separate_contexts",
                "assets": {
                    "featureless_rows": {"files": {"train_path": "labels/train.parquet"}},
                    "candidate_members": {"datasets": {"toy": "members.parquet"}},
                },
                "models": {"classic": {"retrieval_top_k": 2}},
                "expected_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    feature_names = list(trainer.promoted_linker_feature_columns())
    runtime = feature_port._require_rust_runtime()  # noqa: SLF001
    with open_arrow_dataset(paths) as arrow:
        global_plan = runtime.raw_arrow_labeled_candidate_plan(
            arrow.native,
            rows["query_signature_id"].tolist(),
            rows["query_view"].tolist(),
            rows["query_group_id"].tolist(),
            rows["candidate_component_key"].tolist(),
            rows["retrieval_rank"].tolist(),
            members,
            num_threads=1,
        )
        output = tmp_path / "output"
        trainer._materialize_arrow_rust_feature_bundle(  # noqa: SLF001
            source_bundle=trainer.load_bundle(source),
            output_bundle_root=output,
            target={
                "features": feature_names,
                "feature_count": len(feature_names),
                "params": {"n_estimators": 1},
                "metrics": {},
            },
            name_tuples=frozenset(),
            clusterer=SimpleNamespace(
                classifier=_ConstantPairwise(),
                featurizer_info=FeaturizationInfo(["year_diff"]),
                nameless_classifier=None,
                nameless_featurizer_info=None,
                use_default_constraints_as_supervision=True,
            ),
            n_jobs=1,
            total_ram_bytes=1_000_000_000,
            table_keys=("train_path",),
            max_exemplars=1,
            pairwise_model_nan_value=np.nan,
            pairwise_aggregate_nan_value=0.0,
            arrow_datasets={"toy": arrow},
        )
    materialized = pd.read_parquet(output / "features_corrected" / "train.parquet")
    pd.testing.assert_frame_equal(
        materialized[list(rows.columns)],
        rows.assign(retrieval_rank=np.asarray(global_plan["retrieval_ranks"], dtype=np.int64)),
    )
    np.testing.assert_array_equal(
        materialized["retrieval_rank"],
        global_plan["retrieval_ranks"],
    )


@pytest.mark.parametrize("explicit_disallow", [False, True])
def test_initial_clone_matches_public_overlay_and_keeps_origin_constraints(
    tmp_path: Path, explicit_disallow: bool
) -> None:
    """A cloned query must not use its full source identity as profile evidence."""
    source = tmp_path / "source"
    clone = derived_query_id("visibleq:", "q1", "p_q")
    clone_paper = derived_query_id("visiblep:", "q1", "p_q")
    paths = _context_paths(source / "datasets" / "toy")
    for key in ("signatures", "papers", "paper_authors"):
        with pa.OSFile(paths[key], "rb") as file:
            table = pa.ipc.open_file(file).read_all()
        records = table.to_pylist()
        if key == "signatures":
            records.append({**records[0], "signature_id": clone, "paper_id": clone_paper, "author_first": "A"})
        elif key == "papers":
            records.append({**records[0], "paper_id": clone_paper})
        else:
            records.extend(
                {
                    **record,
                    "paper_id": clone_paper,
                    "author_name": "A Wang" if record["position"] == 0 else record["author_name"],
                }
                for record in list(records)
                if record["paper_id"] == "p_q"
            )
        paths[key] = write_ipc(
            source / "datasets" / "toy" / f"projected_{key}.arrow", pa.Table.from_pylist(records, schema=table.schema)
        )
    paths, _metrics = write_raw_arrow_batch_lookup_indexes(paths, source / "datasets" / "toy")
    mapping_dir = source / "incremental_patent_tasks"
    mapping_dir.mkdir()
    (mapping_dir / "initial_query_source_map.jsonl").write_text(
        json.dumps(
            {
                "dataset": "toy",
                "source_query_signature_id": "q1",
                "initial_query_signature_id": clone,
                "source_paper_id": "p_q",
                "initial_paper_id": clone_paper,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    members = {"a": ("q1", "s1"), "b": ("s2",), "self": ("q1",)}
    pd.DataFrame(
        [
            {"candidate_component_key": key, "member_index": index, "signature_id": member}
            for key, values in members.items()
            for index, member in enumerate(values)
        ]
    ).to_parquet(source / "members.parquet", index=False)
    (source / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_name": "projected",
                "assets": {"candidate_members": {"datasets": {"toy": "members.parquet"}}},
                "models": {},
                "expected_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    bundle = trainer.load_bundle(source)
    rows = pd.DataFrame(
        [
            {
                "dataset": "toy",
                "query_signature_id": clone,
                "query_group_id": "initial_case",
                "query_view": "initial_only",
                "candidate_component_key": key,
                "retrieval_rank": rank,
                "label": int(key != "b"),
            }
            for rank, key in enumerate(members, start=1)
        ]
    )
    clusterer = SimpleNamespace(
        classifier=_ConstantPairwise(),
        featurizer_info=FeaturizationInfo(["year_diff"]),
        nameless_classifier=None,
        nameless_featurizer_info=None,
        use_default_constraints_as_supervision=True,
    )
    columns = trainer.promoted_linker_feature_columns()
    (source / "labels").mkdir()
    (source / "splits").mkdir()
    unprojected = rows.loc[rows["candidate_component_key"].ne("self")].assign(
        query_signature_id="q1", query_group_id="unprojected_case"
    )
    pd.concat([rows, unprojected], ignore_index=True).to_parquet(source / "labels" / "train.parquet", index=False)
    bundle.assets["featureless_rows"] = {"files": {"train_path": "labels/train.parquet"}}
    bundle.models["classic"] = {"retrieval_top_k": 2}
    (source / "bundle.json").write_text(
        json.dumps(
            {
                "bundle_name": bundle.bundle_name,
                "assets": bundle.assets,
                "models": bundle.models,
                "expected_metrics": {},
            }
        ),
        encoding="utf-8",
    )
    with open_arrow_dataset(paths) as arrow:
        clean, _summary = trainer._clean_arrow_rust_structural_rows(  # noqa: SLF001
            source_bundle=bundle,
            table_key="train_path",
            rows=rows,
            component_membership_cache={},
            arrow_datasets={"toy": arrow},
        )
        assert clean["candidate_component_key"].tolist() == ["a", "b"]
        context = trainer._build_arrow_rust_dataset_context(
            source_bundle=bundle, dataset_name="toy", arrow_dataset=arrow
        )  # noqa: SLF001
        context.cluster_seeds_require = {"q1": "source_seed", "s1": "source_seed", "s2": "other_seed"}
        context.cluster_seeds_disallow = frozenset({("q1", "s2")}) if explicit_disallow else frozenset()
        features, summary, selected = trainer._materialize_arrow_rust_dataset_rows(  # noqa: SLF001
            context=context,
            rows=clean,
            target_features=columns,
            name_tuples=frozenset(),
            clusterer=clusterer,
            n_jobs=1,
            total_ram_bytes=1_000_000_000,
            max_exemplars=1,
            pairwise_model_nan_value=np.nan,
            pairwise_aggregate_nan_value=0.0,
        )
        assert selected.tolist() == [0, 1]
        assert summary["constraint_seed_bypass_pair_count"] == 2
        assert features["cluster_size_log"][0] == pytest.approx(np.log1p(1))
        output = tmp_path / "projected_output"
        trainer._materialize_arrow_rust_feature_bundle(  # noqa: SLF001
            source_bundle=bundle,
            output_bundle_root=output,
            target={
                "features": list(columns),
                "feature_count": len(columns),
                "params": {"n_estimators": 1},
                "metrics": {},
            },
            name_tuples=frozenset(),
            clusterer=clusterer,
            n_jobs=1,
            total_ram_bytes=1_000_000_000,
            table_keys=("train_path",),
            max_exemplars=1,
            pairwise_model_nan_value=np.nan,
            pairwise_aggregate_nan_value=0.0,
            arrow_datasets={"toy": arrow},
        )
        projected = pd.read_parquet(output / "features_corrected" / "train.parquet")
        assert projected["query_group_id"].tolist() == [
            "initial_case",
            "initial_case",
            "unprojected_case",
            "unprojected_case",
        ]
        assert projected["cluster_size_log"].tolist() == pytest.approx([np.log1p(1)] * 4)

        # open_trusted rejects request-local seed tables; this check passes its own seeds below.
        overlay_paths = {
            key: value
            for key, value in paths.items()
            if key not in {"cluster_seeds", "cluster_seed_disallows", "altered_cluster_signatures"}
        }
        for key in ("signatures", "paper_authors"):
            with pa.OSFile(paths[key], "rb") as file:
                table = pa.ipc.open_file(file).read_all()
            records = table.to_pylist()
            if key == "signatures":
                records[0]["author_first"] = "A"
            else:
                records[0]["author_name"] = "A Wang"
            overlay_paths[key] = write_ipc(
                source / "datasets" / "toy" / f"overlay_{key}.arrow", pa.Table.from_pylist(records, schema=table.schema)
            )
        overlay_paths, _metrics = write_raw_arrow_batch_lookup_indexes(overlay_paths, tmp_path / "overlay_indexes")
        with trainer.ArrowDataset.open_trusted(overlay_paths) as overlay:
            seeds = write_ipc(
                tmp_path / "public_seeds.arrow", pa.table({"signature_id": ["s1", "s2"], "cluster_id": ["a", "b"]})
            )
            plan = (
                feature_port._require_rust_runtime()
                .RawBlockQueryCandidatePlanner.from_auto_queries(
                    overlay.native, seeds, top_k=2, orcid_enabled=False, num_threads=1, max_exemplars=1
                )
                .plan(["q1"])
            )  # noqa: SLF001
            featurizer = feature_port.build_rust_featurizer_from_arrow_dataset(
                overlay,
                signature_ids=tuple(plan["pair_signature_ids"]),
                name_tuples=frozenset(),
                num_threads=1,
            )
            featurizer = featurizer.with_cluster_seeds(
                {"s1": "a", "s2": "b"}, {("q1", "s2")} if explicit_disallow else set()
            )
            signatures = tuple(featurizer.signature_ids())
            order = [list(plan["row_component_keys"]).index(key) for key in ("a", "b")]
            retrieval = build_linker_retrieval_batch_from_raw_plan_bundle(
                RawArrowPlanBundle.from_native_mapping(plan),
                signature_id_to_index={str(value): index for index, value in enumerate(signatures)},
            )
            batch, signals = retrieval.candidate_batch, retrieval.row_signals
            labels, _counts = linker_runtime._resolve_candidate_batch_pair_labels_rust(
                candidate_batch=batch,
                signature_ids_by_index=signatures,
                partial_supervision={},
                use_default_constraints_as_supervision=True,
                dont_merge_cluster_seeds=True,
                n_jobs=1,
                featurizer=featurizer,
            )  # noqa: SLF001
            pairwise = linker_runtime.compute_candidate_batch_pairwise_model_and_aggregate_stats(
                None,
                batch,
                classifier=clusterer.classifier,
                featurizer_info=clusterer.featurizer_info,
                pair_labels=labels,
                n_jobs=1,
                total_ram_bytes=1_000_000_000,
                featurizer=featurizer,
            )
            public_features, _telemetry = linker_runtime._featureize_linker_candidates_with_telemetry(
                dataset=None,
                candidate_batch=batch,
                row_signals={**signals, **pairwise.row_signals},
                feature_columns=columns,
                pairwise_stats=pairwise.pairwise_stats,
                n_jobs=1,
                total_ram_bytes=1_000_000_000,
                featurizer=featurizer,
            )  # noqa: SLF001
            np.testing.assert_array_equal(
                np.column_stack([features[column] for column in columns]), public_features.matrix[order]
            )
    assert features["min_distance"][1] == pytest.approx(LARGE_DISTANCE if explicit_disallow else 0.2)

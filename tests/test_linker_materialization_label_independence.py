"""Keep training and evaluation linker features independent of target labels."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from s2and import feature_port
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking import runtime
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.linker_pairwise import LinkerCandidateBatch
from s2and.runtime import build_runtime_context
from scripts.production.model import train_linker_and_finalize as materializer
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset

_MEMORY_BUDGET = 100_000_000


class ConstantPairwise:
    """Isolate feature-generation differences with a fixed, valid predictor."""

    def predict_proba_positive(self, features: np.ndarray) -> np.ndarray:
        """Return the same probability for every unresolved signature pair."""
        return np.full(len(features), 0.8)


@pytest.mark.parametrize(
    ("candidate_first", "seed_holdout", "explicit_disallow"),
    [
        ("Abdul", False, False),
        ("Alice", False, False),
        ("Abdul", True, False),
        ("Alice", True, False),
        ("Abdul", True, True),
    ],
    ids=["compatible", "name-conflict", "seed-holdout", "seed-name-conflict", "explicit-disallow"],
)
def test_materialized_features_ignore_labels_and_match_runtime(
    tmp_path: Path, candidate_first: str, seed_holdout: bool, explicit_disallow: bool
) -> None:
    """Real planner, constraints, and all 53 features agree after label flips/removal."""
    source = build_dummy_dataset("label_independence", mode="train")
    source.signatures["1"] = source.signatures["1"]._replace(author_info_first=candidate_first)
    training = build_arrow_training_dataset(source, tmp_path / "arrow")
    arrow = training.arrow_dataset
    assert arrow is not None
    context = materializer.ArrowRustDatasetContext(
        dataset_name="label_independence",
        row_component_scope="block-local",
        pairwise_component_scope="block-local",
        runtime_context=build_runtime_context("joint_safe_link_arrow_rust_featureization", backend="rust"),
        arrow_dataset=arrow,
        component_members={"component": ("1",), "other": ("2",)},
        cluster_seeds_require={"0": "seed", "1": "seed", "2": "different"} if seed_holdout else {},
        cluster_seeds_disallow=frozenset({("0", "1")}) if explicit_disallow else frozenset(),
        max_block_component_size=1,
    )
    clusterer = SimpleNamespace(
        classifier=ConstantPairwise(),
        featurizer_info=FeaturizationInfo(["year_diff"]),
        nameless_classifier=None,
        nameless_featurizer_info=None,
        use_default_constraints_as_supervision=True,
    )
    columns = promoted_linker_feature_columns()
    rows = pd.DataFrame(
        [
            {
                "query_signature_id": "0",
                "query_view": "full",
                "query_group_id": "q",
                "candidate_component_key": "component",
                "retrieval_rank": 1,
            },
            {
                "query_signature_id": "0",
                "query_view": "full",
                "query_group_id": "q",
                "candidate_component_key": "other",
                "retrieval_rank": 2,
            },
        ]
    )
    try:
        results = []
        for label in (None, [1, 0], [0, 1]):
            candidate_rows = rows if label is None else rows.assign(label=label)
            features, summary, selected = materializer._materialize_arrow_rust_dataset_rows(
                context=context,
                rows=candidate_rows,
                target_features=columns,
                name_tuples=frozenset(),
                clusterer=clusterer,
                n_jobs=1,
                total_ram_bytes=_MEMORY_BUDGET,
                max_exemplars=1,
                pairwise_model_nan_value=np.nan,
                pairwise_aggregate_nan_value=0.0,
            )
            np.testing.assert_array_equal(selected, [0, 1])
            assert summary["pair_operation_count"] == 2
            assert summary["fused_pairwise_pairs"] == 2
            assert summary["constraint_seed_bypass_pair_count"] == 2 * int(seed_holdout)
            results.append(np.column_stack([features[column] for column in columns]))
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[0], results[2])

        # The production inference resolver never sees target labels. Run its
        # real feature assembly on the same native-planned candidate identities.
        plan = feature_port._require_rust_runtime().raw_arrow_labeled_candidate_plan(
            arrow.native,
            ["0", "0"],
            ["full", "full"],
            ["q", "q"],
            ["component", "other"],
            [1, 2],
            context.component_members,
            orcid_enabled=False,
            num_threads=1,
            max_exemplars=1,
        )
        assert plan["row_count"] == 2
        assert list(plan["left_signature_ids"]) == ["0", "0"]
        assert list(plan["right_signature_ids"]) == ["1", "2"]
        featurizer = feature_port.build_rust_featurizer_from_arrow_dataset(
            arrow,
            signature_ids=tuple(plan["signature_ids"]),
            name_tuples=frozenset(),
            num_threads=1,
        )
        # A held-out query arrives without its own input seed membership, while
        # explicit disallows remain request evidence at inference.
        featurizer = featurizer.with_cluster_seeds(
            {key: value for key, value in context.cluster_seeds_require.items() if key != "0"},
            context.cluster_seeds_disallow,
        )
        signature_ids = tuple(featurizer.signature_ids())
        batch, signals = materializer._arrow_labeled_plan_to_batch_and_row_signals(
            plan=plan,
            rows=rows,
            signature_id_to_index={str(value): index for index, value in enumerate(signature_ids)},
            row_group_ids=(0, 0),
        )
        labels, _ = runtime._resolve_candidate_batch_pair_labels_rust(
            candidate_batch=batch,
            signature_ids_by_index=signature_ids,
            partial_supervision={},
            use_default_constraints_as_supervision=True,
            dont_merge_cluster_seeds=True,
            n_jobs=1,
            featurizer=featurizer,
        )
        pairwise = runtime.compute_candidate_batch_pairwise_model_and_aggregate_stats(
            None,
            batch,
            classifier=clusterer.classifier,
            featurizer_info=clusterer.featurizer_info,
            pair_labels=labels,
            n_jobs=1,
            total_ram_bytes=_MEMORY_BUDGET,
            featurizer=featurizer,
        )
        runtime_features, _ = runtime._featureize_linker_candidates_with_telemetry(
            dataset=None,
            candidate_batch=batch,
            row_signals={**signals, **pairwise.row_signals},
            feature_columns=columns,
            pairwise_stats=pairwise.pairwise_stats,
            n_jobs=1,
            total_ram_bytes=_MEMORY_BUDGET,
            featurizer=featurizer,
        )
        np.testing.assert_array_equal(results[0], runtime_features.matrix)
        expected_distance = 0.2 if candidate_first == "Abdul" and not explicit_disallow else LARGE_DISTANCE
        assert results[0][0, columns.index("min_distance")] == pytest.approx(expected_distance)
        assert results[0][1, columns.index("min_distance")] == pytest.approx(0.2)
    finally:
        arrow.close()


@pytest.mark.parametrize("constraint", ["require", "name-conflict", "explicit-disallow"])
def test_leave_one_out_preserves_real_name_and_explicit_disallow_constraints(tmp_path: Path, constraint: str) -> None:
    """Seed holdout removes derived require evidence without clearing hard negatives."""
    source = build_dummy_dataset("seed_holdout", mode="train")
    source.cluster_seeds_require = {"0": "seed", "1": "seed"}
    source.cluster_seeds_disallow = {("0", "1")} if constraint == "explicit-disallow" else set()
    if constraint == "name-conflict":
        source.signatures["1"] = source.signatures["1"]._replace(author_info_first="Alice")
    training = build_arrow_training_dataset(source, tmp_path / "arrow")
    arrow = training.arrow_dataset
    assert arrow is not None
    try:
        featurizer = feature_port.build_rust_featurizer_from_arrow_dataset(arrow, name_tuples=frozenset())
        featurizer = featurizer.with_cluster_seeds(source.cluster_seeds_require, source.cluster_seeds_disallow)
        indices = {str(value): index for index, value in enumerate(featurizer.signature_ids())}
        batch = LinkerCandidateBatch(
            row_count=1,
            left_signature_indices=np.asarray([indices["0"]], dtype=np.uint32),
            right_signature_indices=np.asarray([indices["1"]], dtype=np.uint32),
            pair_row_indices=np.asarray([0], dtype=np.uint32),
            labels=np.asarray([1], dtype=np.int8),
        )
        labels, summary = materializer._resolve_arrow_rust_pair_labels(
            clusterer=SimpleNamespace(use_default_constraints_as_supervision=True),
            batch=batch,
            featurizer=featurizer,
            n_jobs=1,
            pair_seed_bypass=np.asarray([True]),
        )
        if constraint == "require":
            assert np.isnan(labels[0])
        else:
            assert labels[0] == LARGE_DISTANCE - LARGE_INTEGER
        assert summary["constraint_seed_bypass_pair_count"] == 1
    finally:
        arrow.close()

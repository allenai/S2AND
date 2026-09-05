"""Deterministic training inputs for classic linker and holdout tests."""

from pathlib import Path
from typing import Any

import pandas as pd

from s2and.incremental_linking_training.classic import OfficialBundle


def _classic_candidate_rows(
    query_specs: list[tuple[str, str, str, bool]],
) -> pd.DataFrame:
    """Create labeled candidates both inside and outside the retrieval cutoff."""
    rows: list[dict[str, Any]] = []
    for query_index, (query_id, base_group_id, dataset, has_positive) in enumerate(query_specs):
        for rank in (1, 2, 30):
            label = int((has_positive and rank == 1) or (not has_positive and rank == 30))
            rows.append(
                {
                    "query_group_id": query_id,
                    "base_group_id": base_group_id,
                    "dataset": dataset,
                    "query_view": "full",
                    "query_first_token": "alex",
                    "candidate_component_key": f"{query_id}:candidate:{rank}",
                    "retrieval_rank": rank,
                    "label": label,
                    "tiny_score": float(label * 2 + query_index / 10 - rank / 100),
                }
            )
    return pd.DataFrame(rows)


def write_classic_tiny_bundle(root: Path) -> OfficialBundle:
    """Write bounded training, calibration, and holdout populations on disk."""
    root.mkdir()
    train_rows = _classic_candidate_rows(
        [
            (
                "fit-negative" if index == 0 else f"train-{index}",
                "fit-negative-base" if index == 0 else f"train-base-{index}",
                "train",
                index % 2 == 0,
            )
            for index in range(12)
        ]
    )
    gate_rows = _classic_candidate_rows(
        [
            ("fit-negative", "fit-negative-base", "a_khan", False),
            ("fit-positive", "fit-positive-base", "a_khan", True),
        ]
    )
    hwang_rows = _classic_candidate_rows(
        [
            ("check-negative", "check-negative-base", "h_wang", False),
            ("check-positive", "check-positive-base", "h_wang", True),
        ]
    )
    s2and_rows = _classic_candidate_rows(
        [
            ("test-negative", "test-negative-base", "s2and", False),
            ("test-positive", "test-positive-base", "s2and", True),
        ]
    )
    tables = {
        "train.csv.gz": train_rows,
        "gate.csv.gz": gate_rows,
        "hwang.csv.gz": hwang_rows,
        "s2and.csv.gz": s2and_rows,
    }
    for filename, frame in tables.items():
        frame.to_csv(root / filename, index=False, compression="gzip")

    pd.DataFrame(
        [
            {
                "query_group_id": query_id,
                "source_key": source_key,
                "split": split,
                "base_group_id": base_group_id,
            }
            for query_id, base_group_id, source_key, split in (
                ("fit-negative", "fit-negative-base", "a_khan_eval", "calibration_fit"),
                ("fit-positive", "fit-positive-base", "a_khan_eval", "calibration_fit"),
                ("check-negative", "check-negative-base", "hwang_eval", "calibration_check"),
                ("check-positive", "check-positive-base", "hwang_eval", "calibration_check"),
                ("test-negative", "test-negative-base", "s2and_eval", "test"),
                ("test-positive", "test-positive-base", "s2and_eval", "test"),
            )
        ]
    ).to_csv(root / "assignments.csv", index=False)
    pd.DataFrame({"base_group_id": ["fit-negative-base", "fit-positive-base"]}).to_csv(
        root / "internal_eval_base_groups.csv",
        index=False,
    )

    return OfficialBundle(
        root=root,
        bundle_name="tiny-real-classic",
        assets={},
        models={
            "classic": {
                "feature_columns": ["tiny_score"],
                "retrieval_top_k": 2,
                "best_params": {
                    "learning_rate": 0.1,
                    "max_depth": 2,
                    "min_child_samples": 1,
                    "n_estimators": 5,
                    "num_leaves": 4,
                },
                "train_path": "train.csv.gz",
                "classic_gate_source_path": "gate.csv.gz",
                "classic_gate_internal_eval_base_groups_path": "internal_eval_base_groups.csv",
                "s2and_eval_path": "s2and.csv.gz",
                "hwang_eval_path": "hwang.csv.gz",
                "promoted_stratified_gate": {
                    "calibration_splits": ["calibration_fit", "calibration_check"],
                    "test_split": "test",
                },
                "stratified_eval_test_split": {
                    "assignments_path": "assignments.csv",
                    "split_order": ["calibration_fit", "calibration_check", "test"],
                    "test_split": "test",
                },
            }
        },
        expected_metrics={},
    )

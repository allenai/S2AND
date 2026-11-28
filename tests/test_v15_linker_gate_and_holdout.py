from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s2and.incremental_linking_training.gate import _apply_logistic_gate_predictions
from s2and.incremental_linking_training.ranker import _score_query_choices
from scripts.common.block_roles import load_block_roles
from scripts.eps import apply_reviewed_eps as apply_eps
from scripts.eps.orcid_b3 import REFERENCE_POLICY_ALL_IDENTITIES
from scripts.linker import train_linker_and_finalize as promoted_train
from scripts.linker.blind_gate_metrics import summarize_blind_gate_predictions, with_query_gold_scored

pytestmark = pytest.mark.tooling


def test_smoke_mode_cap_per_source_and_block_disjoint_eps_review_guard(tmp_path: Path) -> None:
    assignments_path = tmp_path / "assignments.csv"
    # 3 splits x 10 rows = 30 rows for one source_key, with string "True"/"False" booleans!
    pd.DataFrame(
        {
            "query_group_id": [f"q{i:02d}" for i in range(30)],
            "source_key": ["s2and_eval"] * 30,
            "split": ["calibration_fit"] * 10 + ["calibration_check"] * 10 + ["test"] * 10,
            "has_positive_candidate": ["True" if i % 2 == 0 else "False" for i in range(30)],
        }
    ).to_csv(assignments_path, index=False)
    by_source = promoted_train._subset_smoke_assignments_file(assignments_path, max_query_groups=6)  # noqa: SLF001
    # Total per source_key is capped at 6 (2 per split), NOT 18!
    assert len(by_source["s2and_eval"]) == 6
    reloaded = pd.read_csv(assignments_path)
    for _split, grp in reloaded.groupby("split"):
        assert len(grp) == 2
        flags = promoted_train._parse_bool_flag(grp["has_positive_candidate"])  # noqa: SLF001
        assert int(flags.sum()) == 1
        assert int((~flags).sum()) == 1

    # Verify apply_reviewed_eps block-role manifest verifier accepts clean eps_cal and rejects bad manifests.
    role_manifest = tmp_path / "assignments.json"
    role_manifest.write_text(
        json.dumps(
            {
                "roles": {"eps_cal": ["a_smith", "b_lee"], "final_test": ["c_park"]},
                "pairwise_validation_blocks": [],
                "gate_training_blocks": [],
                "verification": {"cross_role_orcid_overlap": 0},
                "block_orcids": {"a smith": [], "b lee": [], "c park": []},
            }
        ),
        encoding="utf-8",
    )
    eps_cal = apply_eps._eps_cal_blocks(load_block_roles(role_manifest))  # noqa: SLF001
    assert eps_cal == frozenset({"a smith", "b lee"})

    # Rejects block-disjoint policy when no role manifest is provided.
    with pytest.raises(ValueError, match="requires --block-role-assignments"):
        apply_eps._require_test_separated_reference(  # noqa: SLF001
            tmp_path,
            "a_smith",
            {"reference": {"reference_policy": REFERENCE_POLICY_ALL_IDENTITIES}, "arrow_root": str(tmp_path)},
            lambda *_a: frozenset(),
            None,  # type: ignore[arg-type]
            verified_eps_cal_blocks=None,
        )


def test_fixed_ranker_load_checks_bundle_and_requested_hash(tmp_path: Path) -> None:
    from s2and.production_bundle_contract import COMPLETE_MANIFEST_FILES
    from tests.test_promoted_linker_training_cli import _fixed_ranker_fixture

    _, bundle, digest = _fixed_ranker_fixture(tmp_path)
    path = bundle / COMPLETE_MANIFEST_FILES["incremental_linker_booster"]
    loaded, source, contract = promoted_train._load_fixed_ranker(bundle, digest)
    assert source == path
    assert loaded.num_feature() == len(promoted_train.promoted_linker_feature_columns())
    assert contract == 1
    with pytest.raises(ValueError, match="ranker.*SHA-256"):
        promoted_train._load_fixed_ranker(bundle, "0" * 64)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch.*booster"):
        promoted_train._load_fixed_ranker(bundle, digest)


def test_blind_gate_metrics_require_a_fully_resolved_window() -> None:
    rows = pd.DataFrame(
        {
            "query_group_id": ["partial", "partial", "positive", "positive", "negative", "negative"],
            "dataset": ["gate"] * 6,
            "query_view": ["full"] * 6,
            "candidate_component_key": ["first", "second"] * 3,
            "retrieval_rank": [1, 2] * 3,
            "label": [0, 0, 0, 1, 0, 0],
            "candidate_gold_scored": [True, False, True, True, True, True],
            "query_gold_scored": [False, False, True, True, True, True],
        }
    )
    choices = _score_query_choices(
        rows, np.array([0.5, 0.4] * 3), query_id_column="query_group_id", include_margin=True
    )
    choices = with_query_gold_scored(choices, rows)
    predictions = _apply_logistic_gate_predictions(choices, np.array([False, False, False]))
    metrics = summarize_blind_gate_predictions(predictions)
    assert predictions["chosen_candidate_gold_scored"].all()
    assert predictions["query_gold_scored"].tolist() == [False, True, True]
    assert metrics["n_unscored_queries"] == 1
    assert metrics["n_queries"] == 2
    assert metrics["n_negative_queries"] == 1
    assert metrics["false_abstain"] == 1
    assert metrics["accuracy"] == 0.5

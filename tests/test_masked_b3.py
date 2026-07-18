from __future__ import annotations

import pytest

from scripts._pair_ablation.evaluation import build_block_linkages
from scripts._pair_ablation.masked_b3 import (
    masked_b3_for_threshold,
    masked_component_ceiling,
    select_masked_orcid_components,
)


def test_masked_component_selection_is_deterministic_and_budgeted() -> None:
    components = {
        "component_a": ["a1", "a2", "x1", "x2"],
        "component_a_split": ["a3", "x3"],
        "component_b": ["b1", "b2", "x4"],
        "component_c": ["c1", "c2"],
    }
    orcids = {
        "a1": "A",
        "a2": "A",
        "a3": "A",
        "b1": "B",
        "b2": "B",
        "c1": "C",
        "c2": "C",
        "x1": "unique-x1",
    }

    first = select_masked_orcid_components(
        dataset="toy",
        components=components,
        orcid_by_signature=orcids,
        pair_budget=7,
        max_block_size=3,
        random_seed=17,
    )
    second = select_masked_orcid_components(
        dataset="toy",
        components=components,
        orcid_by_signature=orcids,
        pair_budget=7,
        max_block_size=3,
        random_seed=17,
    )

    assert first.plan.identity_payload() == second.plan.identity_payload()
    assert first.target_gold == second.target_gold
    assert first.stats == second.stats
    assert first.stats["selected_pair_count"] <= 7
    assert first.stats["repeated_orcid_group_count"] == 3
    assert first.stats["eligible_fragmented_orcid_group_count"] == 1
    selected_orcids = set(first.target_gold.values())
    for orcid in selected_orcids:
        assert {signature_id for signature_id, value in orcids.items() if value == orcid}.issubset(first.target_gold)
    planned_signatures = {signature_id for block in first.plan.blocks for signature_id in block.signatures}
    assert set(first.target_gold).issubset(planned_signatures)


def test_masked_b3_projects_out_distractors_and_penalizes_component_splits() -> None:
    blocks = {
        "left": ["a1", "a2", "distractor"],
        "right": ["a3", "b1", "b2"],
    }
    distances = {
        "left": [0.1, 0.9, 0.9],
        "right": [0.9, 0.9, 0.1],
    }
    linkages = build_block_linkages(blocks, distances)
    target_gold = {
        "a1": "A",
        "a2": "A",
        "a3": "A",
        "b1": "B",
        "b2": "B",
    }

    precision, recall, f1 = masked_b3_for_threshold(
        linkages,
        target_gold,
        0.5,
        dataset_prefix="toy",
    )

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.733)
    assert f1 == pytest.approx(0.846)


def test_masked_component_ceiling_is_perfect_within_but_not_across_components() -> None:
    selection = select_masked_orcid_components(
        dataset="toy",
        components={
            "left": ["a1", "a2", "x1"],
            "right": ["a3", "b1", "b2"],
        },
        orcid_by_signature={
            "a1": "A",
            "a2": "A",
            "a3": "A",
            "b1": "B",
            "b2": "B",
        },
        pair_budget=6,
        max_block_size=3,
        random_seed=1,
    )

    precision, recall, f1 = masked_component_ceiling(
        selection.plan,
        selection.target_gold,
        dataset_prefix="toy",
    )

    assert precision == pytest.approx(1.0)
    assert recall < 1.0
    assert f1 < 1.0

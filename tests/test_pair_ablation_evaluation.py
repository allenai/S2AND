from __future__ import annotations

import numpy as np
import pytest

from scripts._pair_ablation.evaluation import (
    GoldBlockData,
    b3_for_threshold,
    build_b3_evaluation_plans,
    build_block_linkages,
    predicted_clusters_at_threshold,
    select_blocks_with_pair_budget,
    split_blocks_like_anddata,
    tune_b3_threshold,
)


def _gold_data() -> GoldBlockData:
    blocks = {f"b{index}": [f"s{index}", f"t{index}"] for index in range(20)}
    cluster_by_signature = {
        signature_id: f"c{index}" for index, signatures in enumerate(blocks.values()) for signature_id in signatures
    }
    full_names = {signature_id: signature_id for signature_id in cluster_by_signature}
    return GoldBlockData("d", blocks, cluster_by_signature, full_names)


def test_split_blocks_like_anddata_is_disjoint_and_deterministic() -> None:
    blocks = {f"b{index}": [f"s{index}", f"t{index}"] for index in range(20)}
    first = split_blocks_like_anddata(blocks, random_seed=1111)
    second = split_blocks_like_anddata(blocks, random_seed=1111)
    assert first == second
    assert [len(part) for part in first] == [16, 2, 2]
    keys = [set(part) for part in first]
    assert not keys[0] & keys[1]
    assert not keys[0] & keys[2]
    assert not keys[1] & keys[2]


def test_select_blocks_with_pair_budget_is_deterministic_and_whole_block() -> None:
    blocks = {"large": list("abcdef"), "small_a": ["g", "h"], "small_b": ["i", "j"]}
    selected = select_blocks_with_pair_budget(blocks, pair_budget=2, random_seed=7)
    assert selected == select_blocks_with_pair_budget(blocks, pair_budget=2, random_seed=7)
    assert set(selected).issubset(blocks)
    assert sum(len(values) * (len(values) - 1) // 2 for values in selected.values()) <= 2


def test_b3_evaluation_plans_are_deterministic_disjoint_and_content_addressed() -> None:
    gold = _gold_data()
    first = build_b3_evaluation_plans(
        {"d": gold},
        evaluation_seed=1111,
        threshold_pairs_per_domain=2,
        b3_scope="test",
    )["d"]
    second = build_b3_evaluation_plans(
        {"d": gold},
        evaluation_seed=1111,
        threshold_pairs_per_domain=2,
        b3_scope="test",
    )["d"]

    assert first == second
    assert first.calibration.plan_digest == second.calibration.plan_digest
    assert first.heldout.plan_digest == second.heldout.plan_digest
    calibration_ids = {signature_id for block in first.calibration.blocks for signature_id in block.signatures}
    heldout_ids = {signature_id for block in first.heldout.blocks for signature_id in block.signatures}
    assert calibration_ids
    assert heldout_ids
    assert calibration_ids.isdisjoint(heldout_ids)
    assert first.calibration.role == "calibration"
    assert first.heldout.role == "heldout_test"

    changed_seed = build_b3_evaluation_plans(
        {"d": gold},
        evaluation_seed=2222,
        threshold_pairs_per_domain=2,
        b3_scope="test",
    )["d"]
    full_scope = build_b3_evaluation_plans(
        {"d": gold},
        evaluation_seed=1111,
        threshold_pairs_per_domain=2,
        b3_scope="full",
    )["d"]
    assert changed_seed.calibration.plan_digest != first.calibration.plan_digest
    assert full_scope.heldout.plan_digest != first.heldout.plan_digest


def test_b3_evaluation_plan_digest_binds_gold_assignments() -> None:
    gold = _gold_data()
    original = build_b3_evaluation_plans(
        {"d": gold},
        evaluation_seed=1111,
        threshold_pairs_per_domain=2,
        b3_scope="test",
    )["d"].heldout
    changed_clusters = dict(gold.cluster_by_signature)
    changed_signature = original.blocks[0].signatures[0]
    changed_clusters[changed_signature] = "different-cluster"
    changed_gold = GoldBlockData("d", gold.blocks, changed_clusters, gold.full_name_by_signature)
    changed = build_b3_evaluation_plans(
        {"d": changed_gold},
        evaluation_seed=1111,
        threshold_pairs_per_domain=2,
        b3_scope="test",
    )["d"].heldout
    assert changed.plan_digest != original.plan_digest


def test_build_linkages_validates_condensed_shape() -> None:
    with pytest.raises(ValueError, match="distance shape mismatch"):
        build_block_linkages({"b": ["a", "b", "c"]}, {"b": np.asarray([0.1, 0.2])})


def test_threshold_scan_selects_separating_cut() -> None:
    blocks = {"d": {"b": ["a", "b", "c"]}}
    # scipy condensed order: (a,b), (a,c), (b,c)
    linkages = {"d": build_block_linkages(blocks["d"], {"b": np.asarray([0.1, 0.9, 0.9])})}
    gold = {"d": {"a": "x", "b": "x", "c": "y"}}
    threshold, metrics = tune_b3_threshold(linkages, blocks, gold, [0.05, 0.2, 0.95])
    assert threshold == 0.2
    assert metrics["f1"] == 1.0
    assert b3_for_threshold(linkages, blocks, gold, threshold) == (1.0, 1.0, 1.0)
    predicted = predicted_clusters_at_threshold(linkages["d"], threshold, dataset_prefix="d")
    assert sorted(sorted(values) for values in predicted.values()) == [
        [("d", "a"), ("d", "b")],
        [("d", "c")],
    ]


def test_b3_namespaces_local_signature_ids_across_datasets() -> None:
    """A local ID shared by datasets must still represent two B-cubed items."""

    # Keep dataset_one last so the pre-fix raw-ID reverse lookup deterministically
    # overwrote dataset_two's truth and returned (0.5, 1.0, 0.667).
    blocks = {
        "dataset_two": {"b": ["shared", "local"]},
        "dataset_one": {"b": ["shared", "local"]},
    }
    linkages = {
        dataset: build_block_linkages(domain_blocks, {"b": np.asarray([0.1])})
        for dataset, domain_blocks in blocks.items()
    }
    gold = {
        "dataset_two": {"shared": "same", "local": "same"},
        "dataset_one": {"shared": "first", "local": "second"},
    }

    assert b3_for_threshold(linkages, blocks, gold, 0.2) == (0.75, 1.0, 0.857)

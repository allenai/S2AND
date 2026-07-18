from __future__ import annotations

from pathlib import Path

import pytest

from scripts._pair_ablation.modeling import (
    BIG_LINKER_SOURCE_DOMAINS,
    LINKER_BIG_POSITIVE_FAMILY,
    LINKER_PROXY_NEGATIVE_FAMILY,
)
from scripts._pair_ablation.ranking import LoadedFold
from scripts.analyze_additive_linker_dose_ablation import (
    _arm_identity,
    _validate_additive_audit,
    plan_next_doses,
)


def _score(source_set: str, dose: int, passed: bool) -> dict[str, object]:
    return {
        "additive_linker": {
            "source_set": source_set,
            "dose_per_source": dose,
            "b3_safety_eligible": passed,
        }
    }


def _loaded(payload: dict[str, object], *, digest: str) -> LoadedFold:
    return LoadedFold(
        path=Path("result.json"),
        sha256=digest,
        payload=payload,
        run_manifest_path=Path("run_manifest.json"),
        run_manifest_sha256="b" * 64,
        comparison_identity_sha256="c" * 64,
    )


def _audit_fixture() -> tuple[LoadedFold, LoadedFold]:
    base_digest = "d" * 64
    baseline = _loaded(
        {
            "arm": "uniform_100k",
            "training_seed": 1111,
            "held_out_domain": "pubmed",
            "evaluation_seed": 1111,
            "evaluation_pair_digest": "e" * 64,
            "b3_evaluation_digest": "f" * 64,
            "training_pair_digest": base_digest,
            "training_rows": 600,
            "pairwise": {"oracle_kind": "gold_cluster_pairs"},
        },
        digest="a" * 64,
    )
    source_counts = [
        {
            "source_domain": "qian",
            "source_family": "gold_cluster_uniform",
            "rows": 600,
            "positives": 100,
            "negatives": 500,
        }
    ]
    domain_audits = []
    for domain in BIG_LINKER_SOURCE_DOMAINS:
        source_counts.extend(
            [
                {
                    "source_domain": domain,
                    "source_family": LINKER_BIG_POSITIVE_FAMILY,
                    "rows": 2,
                    "positives": 2,
                    "negatives": 0,
                },
                {
                    "source_domain": domain,
                    "source_family": LINKER_PROXY_NEGATIVE_FAMILY,
                    "rows": 2,
                    "positives": 0,
                    "negatives": 2,
                },
            ]
        )
        domain_audits.append(
            {
                "source_domain": domain,
                "selected_per_class": 2,
                "selected_rows": 4,
            }
        )
    candidate_digest = "1" * 64
    candidate = _loaded(
        {
            "arm": "uniform_100k_plus_linker_big7_2500",
            "training_seed": 1111,
            "held_out_domain": "pubmed",
            "evaluation_seed": 1111,
            "evaluation_pair_digest": "e" * 64,
            "b3_evaluation_digest": "f" * 64,
            "training_pair_digest": candidate_digest,
            "training_rows": 628,
            "training_source_counts": source_counts,
            "pair_recipe_assembly": {
                "mode": "additive_linker",
                "base_pair_digest": base_digest,
                "training_pair_digest": candidate_digest,
                "base_rows_after_lodo": 600,
                "linker_source_set": "big7",
                "linker_cap_per_domain": 2_500,
                "linker_source_domains": list(BIG_LINKER_SOURCE_DOMAINS),
                "linker_eligible_domains_after_lodo": list(BIG_LINKER_SOURCE_DOMAINS),
                "linker_selected_rows": 28,
                "held_out_rows": 0,
                "final_rows": 628,
                "linker_domains": domain_audits,
                "frozen_baseline_result_sha256": "a" * 64,
                "frozen_baseline_run_manifest_sha256": "b" * 64,
            },
        },
        digest="2" * 64,
    )
    return candidate, baseline


def test_plan_requests_complete_initial_sweep_first() -> None:
    plan = plan_next_doses([])

    assert not plan["complete"]
    assert {(row["source_set"], row["dose_per_source"]) for row in plan["next_arms"]} == {
        (source_set, dose)
        for source_set in ("all13", "big7")
        for dose in (2_500, 5_000, 10_000)
    }


def test_plan_expands_to_25k_only_when_10k_passes() -> None:
    scores = [
        *(_score("all13", dose, True) for dose in (2_500, 5_000, 10_000)),
        *(_score("big7", dose, dose < 10_000) for dose in (2_500, 5_000, 10_000)),
    ]

    plan = plan_next_doses(scores)

    requests = {(row["source_set"], row["dose_per_source"], row["reason"]) for row in plan["next_arms"]}
    assert ("all13", 25_000, "ten_thousand_passed") in requests
    assert all(not (source_set == "all13" and dose == 50_000) for source_set, dose, _reason in requests)
    assert ("big7", 7_500, "refine_first_pass_fail_bracket") in requests
    assert all(not (source_set == "big7" and dose >= 25_000) for source_set, dose, _reason in requests)


def test_plan_requests_50k_only_after_25k_passes() -> None:
    scores = [
        *(_score(source_set, dose, True) for source_set in ("all13", "big7") for dose in (2_500, 5_000, 10_000)),
        _score("all13", 25_000, True),
        _score("big7", 25_000, False),
    ]

    plan = plan_next_doses(scores)

    requests = {(row["source_set"], row["dose_per_source"], row["reason"]) for row in plan["next_arms"]}
    assert ("all13", 50_000, "twenty_five_thousand_passed") in requests
    assert ("big7", 17_500, "refine_first_pass_fail_bracket") in requests
    assert all(not (source_set == "big7" and dose == 50_000) for source_set, dose, _reason in requests)


def test_plan_stops_when_brackets_are_at_most_500() -> None:
    scores = [
        *(_score("all13", dose, True) for dose in (2_500, 5_000, 10_000, 25_000)),
        _score("all13", 50_000, False),
        _score("all13", 49_500, True),
        _score("big7", 2_500, False),
        _score("big7", 5_000, False),
        _score("big7", 10_000, False),
        _score("big7", 2_000, True),
    ]

    plan = plan_next_doses(scores)

    assert plan["complete"]
    assert plan["next_arms"] == []
    assert plan["boundaries"]["all13"]["largest_passing_dose"] == 49_500
    assert plan["boundaries"]["all13"]["smallest_failing_dose"] == 50_000
    assert plan["boundaries"]["big7"]["largest_passing_dose"] == 2_000
    assert plan["boundaries"]["big7"]["smallest_failing_dose"] == 2_500


def test_plan_accepts_50k_when_every_tested_dose_passes() -> None:
    scores = [
        *(_score(source_set, dose, True) for source_set in ("all13", "big7") for dose in (2_500, 5_000, 10_000)),
        *(_score(source_set, dose, True) for source_set in ("all13", "big7") for dose in (25_000, 50_000)),
    ]

    plan = plan_next_doses(scores)

    assert plan["complete"]
    assert plan["boundaries"]["all13"]["largest_passing_dose"] == 50_000
    assert plan["boundaries"]["big7"]["largest_passing_dose"] == 50_000


@pytest.mark.parametrize(
    "arm",
    (
        "uniform_100k",
        "uniform_100k_plus_linker_big7_625",
        "uniform_100k_plus_linker_unknown_2500",
    ),
)
def test_arm_identity_rejects_non_additive_or_odd_recipes(arm: str) -> None:
    with pytest.raises(ValueError):
        _arm_identity(arm)


def test_additive_audit_binds_training_digest_and_actual_class_balance() -> None:
    candidate, baseline = _audit_fixture()

    linker_rows, _share, domain_audits = _validate_additive_audit(
        candidate,
        baseline,
        source_set="big7",
        dose=2_500,
    )

    assert linker_rows == 28
    assert all(record["actual_positives"] == record["actual_negatives"] == 2 for record in domain_audits)

    candidate.payload["pair_recipe_assembly"]["training_pair_digest"] = "9" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="training digest"):
        _validate_additive_audit(candidate, baseline, source_set="big7", dose=2_500)


def test_additive_audit_rejects_claimed_balance_that_source_counts_disprove() -> None:
    candidate, baseline = _audit_fixture()
    source_counts = candidate.payload["training_source_counts"]  # type: ignore[assignment]
    positive = next(
        record
        for record in source_counts  # type: ignore[union-attr]
        if record["source_domain"] == "a_khan" and record["source_family"] == LINKER_BIG_POSITIVE_FAMILY
    )
    negative = next(
        record
        for record in source_counts  # type: ignore[union-attr]
        if record["source_domain"] == "a_khan" and record["source_family"] == LINKER_PROXY_NEGATIVE_FAMILY
    )
    positive.update(rows=3, positives=3)
    negative.update(rows=1, negatives=1)

    with pytest.raises(ValueError, match="balanced audit quotas"):
        _validate_additive_audit(candidate, baseline, source_set="big7", dose=2_500)

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts._pair_ablation.study import BASELINE_NAME, GOLD_DOMAINS, AdditiveDose
from scripts.analyze_additive_linker_dose_ablation import (
    MIN_MEAN_DELTA_B3,
    MIN_Q05_DELTA_B3,
    MIN_WORST_DOMAIN_DELTA_B3,
    analyze,
    evaluate_gates,
)

CANDIDATE = AdditiveDose("big7", 2500).name
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _row(*, seed: int, arm: str, domain: str, b3_f1: float | None) -> dict[str, object]:
    baseline_digest = DIGEST_B
    return {
        "training_seed": seed,
        "arm": arm,
        "held_out_domain": domain,
        "study_digest": DIGEST_C,
        "prepared_digest": DIGEST_A,
        "training_pair_digest": baseline_digest if arm == BASELINE_NAME else "c" * 64,
        "baseline_pair_digest": baseline_digest,
        "evaluation_digest": f"eval-{seed}-{domain}",
        "b3_f1": b3_f1,
    }


def _write_grid(
    root: Path,
    *,
    seeds: tuple[int, ...] = (11, 22),
    domains: tuple[str, ...] = (*GOLD_DOMAINS, "a_khan"),
    candidate_deltas: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    deltas = candidate_deltas or {domain: (0.29 if domain == "a_khan" else 0.001) for domain in domains}
    rows = []
    for seed in seeds:
        for domain in domains:
            baseline_f1 = 0.7 if domain in GOLD_DOMAINS else None
            candidate_f1 = None if baseline_f1 is None else baseline_f1 + deltas[domain]
            baseline = _row(seed=seed, arm=BASELINE_NAME, domain=domain, b3_f1=baseline_f1)
            candidate = _row(
                seed=seed,
                arm=CANDIDATE,
                domain=domain,
                b3_f1=candidate_f1,
            )
            rows.extend((baseline, candidate))
    for row in rows:
        path = root / str(row["training_seed"]) / str(row["arm"]) / f"{row['held_out_domain']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row), encoding="utf-8")
    return rows


def test_analysis_is_deterministic_and_excludes_proxy_b3(tmp_path: Path) -> None:
    _write_grid(tmp_path)

    first = analyze([tmp_path], bootstrap_samples=500, bootstrap_seed=7)
    (tmp_path / "analysis.json").write_text(json.dumps(first), encoding="utf-8")
    second = analyze([tmp_path], bootstrap_samples=500, bootstrap_seed=7)

    assert first == second
    score = first["arms"][0]
    assert score["gold_domains"] == list(GOLD_DOMAINS)
    assert score["mean_delta_b3"] == pytest.approx(0.001)
    assert "a_khan" not in score["domain_mean_delta_b3"]
    assert score["coverage_complete"] is True and score["passed"] is True

    subset = tmp_path / "subset"
    _write_grid(subset, domains=("aminer", "pubmed"))
    subset_score = analyze([subset], bootstrap_samples=10)["arms"][0]
    assert subset_score["coverage_complete"] is False and subset_score["passed"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate result cell"),
        ("missing", "grid mismatch"),
        ("prepared", "different prepared inputs"),
        ("study", "different study recipe"),
        ("baseline_digest", "wrong baseline pairs"),
        ("baseline_seed_drift", "changed across seeds"),
        ("evaluation", "different evaluation inputs"),
    ],
)
def test_analysis_rejects_unpaired_grids(tmp_path: Path, mutation: str, message: str) -> None:
    rows = _write_grid(tmp_path, seeds=(11, 22), domains=("aminer", "pubmed"))
    candidate_path = next(tmp_path.rglob(f"{CANDIDATE}/*.json"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        duplicate = tmp_path / "duplicate" / str(candidate["training_seed"]) / CANDIDATE / "aminer.json"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_text(json.dumps(candidate), encoding="utf-8")
    elif mutation == "missing":
        candidate_path.unlink()
    elif mutation == "prepared":
        candidate["prepared_digest"] = "d" * 64
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    elif mutation == "study":
        candidate["study_digest"] = "d" * 64
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    elif mutation == "baseline_digest":
        candidate["baseline_pair_digest"] = "d" * 64
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    elif mutation == "baseline_seed_drift":
        baseline_path = tmp_path / "22" / BASELINE_NAME / "aminer.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline["training_pair_digest"] = baseline["baseline_pair_digest"] = "d" * 64
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    else:
        candidate["evaluation_digest"] = "different"
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        analyze([tmp_path], bootstrap_samples=10)

    assert rows


def test_all_three_gates_are_inclusive_and_fail_below_boundary() -> None:
    thresholds = (MIN_MEAN_DELTA_B3, MIN_Q05_DELTA_B3, MIN_WORST_DOMAIN_DELTA_B3)
    boundary = evaluate_gates(*thresholds)

    assert list(boundary) == ["mean_delta_b3", "q05_delta_b3", "worst_domain_delta_b3"]
    assert all(gate["passed"] for gate in boundary.values())

    for index, name in enumerate(boundary):
        values = list(thresholds)
        values[index] = float(np.nextafter(values[index], -np.inf))
        gates = evaluate_gates(*values)
        assert gates[name]["passed"] is False
        assert sum(not gate["passed"] for gate in gates.values()) == 1

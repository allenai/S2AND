from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts._pair_ablation.ranking import (
    ALL_DOMAINS,
    B3_DOMAINS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PAIR_DOMAINS,
    RANKING_INPUT_SCHEMA_VERSION,
    concentration_diagnostics,
    make_bootstrap_draws,
    rank_manifest,
    score_candidate,
    select_recommendation,
    two_way_paired_bootstrap,
    write_ranking_outputs,
)
from scripts._pair_ablation.results import (
    FOLD_RESULT_SCHEMA_VERSION,
    FoldResultExpectation,
    load_strict_json,
    recipe_id_for,
    write_fold_result,
)
from scripts._pair_ablation.run_identity import (
    RUN_MANIFEST_SCHEMA_VERSION,
    THREAD_ENVIRONMENT_KEYS,
    build_run_manifest,
)
from scripts.combine_pair_ablation_ranking_inputs import combine_ranking_inputs


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recipe(
    arm: str,
    *,
    fixed_budget: bool,
    complexity_rank: int,
    dose: int,
) -> dict[str, object]:
    return {
        "arm": arm,
        "assembly_version": "exact_budget_v1",
        "auxiliary_sources": [] if arm == "uniform_100k" else ["fixture_auxiliary"],
        "balancing": "none" if arm == "uniform_100k" else "domain_label_balanced",
        "base_sampler": "uniform_100k",
        "budget_policy": "exact_after_lodo" if fixed_budget else "additive",
        "complexity_rank": complexity_rank,
        "fixed_budget": fixed_budget,
        "source_caps": {"fixture_dose": dose},
    }


@dataclass(frozen=True, slots=True)
class CandidateFixture:
    arm: str
    gold_delta_auprc: float = 0.002
    gold_delta_auroc: float = 0.001
    b3_delta: float = 0.0
    proxy_delta_auprc: float = 0.0
    proxy_delta_auroc: float = 0.0
    fixed_budget: bool = True
    complexity_rank: int = 1
    dose: int = 10_000
    training_rows: int = 100
    evaluation_drift_domain: str | None = None
    run_identity_drift: str | None = None


def _oracle(domain: str) -> str:
    if domain in B3_DOMAINS:
        return "gold_cluster_pairs"
    if domain == "medline":
        return "fixed_pair_labels"
    return "linker_component_proxy"


def _payload(
    expectation: FoldResultExpectation,
    *,
    training_rows: int,
    auprc: float,
    auroc: float,
    b3_f1: float | None,
) -> dict[str, Any]:
    positives = training_rows // 2
    negatives = training_rows - positives
    audit = None
    if expectation.requires_recipe_audit:
        audit = {
            "base_filler_rows": training_rows // 2,
            "final_rows": training_rows,
            "held_out_rows": 0,
            "selection_sha256": expectation.training_pair_digest,
            "target_rows": training_rows,
        }
    b3 = None
    if b3_f1 is not None:
        b3 = {
            "f1": b3_f1,
            "heldout_blocks": 3,
            "heldout_pairs": 9,
            "heldout_signatures": 7,
            "precision": b3_f1,
            "recall": b3_f1,
            "scope": "test",
            "scoring_backend": "rust_pair_features_and_rust_lightgbm",
            "threshold": 0.5,
            "threshold_calibration": {"f1": 0.8, "precision": 0.8, "recall": 0.8},
        }
    return {
        "arm": expectation.arm,
        "b3": b3,
        "b3_cache_digests": [_digest("cache")] if b3 is not None else [],
        "b3_evaluation_digest": expectation.b3_evaluation_digest,
        "elapsed_seconds": 1.0,
        "evaluation_pair_digest": expectation.evaluation_pair_digest,
        "evaluation_seed": expectation.evaluation_seed,
        "held_out_domain": expectation.held_out_domain,
        "model_cache_hit": False,
        "models": {
            "main": {"model_path": "main.lgb", "model_sha256": _digest("main")},
            "nameless": {"model_path": "nameless.lgb", "model_sha256": _digest("nameless")},
        },
        "pair_recipe_assembly": audit,
        "pairwise": {
            "auprc": auprc,
            "auroc": auroc,
            "negatives": 6,
            "oracle_kind": expectation.oracle_kind,
            "positives": 4,
            "prevalence": 0.4,
            "rows": 10,
        },
        "recipe": expectation.recipe,
        "recipe_id": recipe_id_for(expectation.recipe),
        "run_id": expectation.run_id,
        "schema_version": FOLD_RESULT_SCHEMA_VERSION,
        "source_families": list(expectation.source_families),
        "training_negatives": negatives,
        "training_pair_digest": expectation.training_pair_digest,
        "training_positives": positives,
        "training_rows": training_rows,
        "training_seed": expectation.training_seed,
        "training_source_counts": [
            {
                "negatives": negatives,
                "positives": positives,
                "rows": training_rows,
                "source_domain": "fixture_training_domain",
                "source_family": "gold_cluster_uniform",
            }
        ],
    }


def _manifest_entry(
    path: Path,
    expectation: FoldResultExpectation,
    *,
    root: Path,
    run_manifest_path: Path,
) -> dict[str, Any]:
    expected = asdict(expectation)
    expected["source_families"] = list(expectation.source_families)
    return {
        "expected": expected,
        "path": path.relative_to(root).as_posix(),
        "result_sha256": _sha256_file(path),
        "run_manifest_path": run_manifest_path.relative_to(root).as_posix(),
        "run_manifest_sha256": _sha256_file(run_manifest_path),
    }


def _write_combiner_manifest(tmp_path: Path, name: str, arms: list[str]) -> Path:
    path = tmp_path / f"{name}.json"
    entries = [
        {
            "expected": {"arm": arm},
            "path": str((tmp_path / name / f"{arm}.json").resolve()),
            "result_sha256": _digest(f"{name}:{arm}"),
            "run_manifest_path": str((tmp_path / name / "run_manifest.json").resolve()),
            "run_manifest_sha256": _digest(f"{name}:run-manifest"),
        }
        for arm in arms
    ]
    path.write_text(
        json.dumps({"folds": entries, "schema_version": RANKING_INPUT_SCHEMA_VERSION}),
        encoding="utf-8",
    )
    return path


def _run_manifest(recipe: dict[str, Any], *, training_seed: int, drift: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "adapter": "practice_only_legacy_arrow_rust_v1",
        "config": {
            "training_seed": training_seed,
            "evaluation_seed": 1111,
            "arm_names": [recipe["arm"]],
            "uniform_pairs_per_domain": 100_000,
            "name_pairs_per_domain": 10_000,
            "balanced_medium_pairs_per_domain": 50_000,
            "balanced_pool_pairs_per_domain": 100_000,
            "linker_pairs_per_domain": 10_000,
            "eval_pairs_per_domain": 20_000,
            "threshold_pairs_per_domain": 5_000,
        },
        "donor_model_sha256": {"main": _digest("donor-main"), "nameless": _digest("donor-nameless")},
        "featurizer_version": 5,
        "git": {
            "branch": "fixture",
            "commit": "fixture-commit",
            "diff_binary_sha256": _digest("git-diff"),
            "status_short": "",
        },
        "implementation_sha256": {"scripts/run.py": _digest("implementation")},
        "input_identity": {"catalog_files": {"fixture": {"sha256": _digest("input")}}},
        "recipes": [{"recipe_id": recipe_id_for(recipe), "recipe": recipe}],
        "runtime_versions": {
            "python": "3.11.13",
            "numpy": "2.3.1",
            "pandas": "2.3.1",
            "scipy": "1.16.0",
            "sklearn": "1.7.0",
            "lightgbm": "4.6.0",
            "fastcluster": "1.3.0",
            "pyarrow": "21.0.0",
        },
        "rust_extension_sha256": _digest("rust-extension"),
        "rust_version": "0.60.0",
        "thread_environment": {key: None for key in THREAD_ENVIRONMENT_KEYS},
        "warning": "fixture warning",
    }
    if drift == "implementation":
        payload["implementation_sha256"]["scripts/run.py"] = _digest("drifted-implementation")
    return build_run_manifest(payload)


def _write_study(
    tmp_path: Path,
    candidates: list[CandidateFixture],
    *,
    reverse_entries: bool = False,
) -> Path:
    seeds = (1111, 2222, 3333)
    baseline = CandidateFixture(
        arm="uniform_100k",
        gold_delta_auprc=0,
        gold_delta_auroc=0,
        b3_delta=0,
        fixed_budget=False,
        complexity_rank=0,
        dose=100_000,
        training_rows=100,
    )
    entries = []
    for candidate in (baseline, *candidates):
        recipe = _recipe(
            candidate.arm,
            fixed_budget=candidate.fixed_budget,
            complexity_rank=candidate.complexity_rank,
            dose=candidate.dose,
        )
        recipe_id = recipe_id_for(recipe)
        for seed in seeds:
            run_manifest = _run_manifest(
                recipe,
                training_seed=seed,
                drift=candidate.run_identity_drift,
            )
            run_manifest_path = tmp_path / "runs" / recipe_id / str(seed) / "run_manifest.json"
            run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            run_manifest_path.write_text(
                json.dumps(run_manifest, allow_nan=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            run_id = run_manifest["run_id"]
            for domain in ALL_DOMAINS:
                evaluation_digest = _digest(f"evaluation:{domain}")
                if candidate.evaluation_drift_domain == domain and seed == seeds[-1]:
                    evaluation_digest = _digest(f"drift:{recipe_id}:{domain}")
                b3_digest = _digest(f"b3:{domain}") if domain in B3_DOMAINS else None
                expectation = FoldResultExpectation(
                    run_id=run_id,
                    arm=candidate.arm,
                    source_families=("gold_cluster_uniform",),
                    held_out_domain=domain,
                    training_seed=seed,
                    evaluation_seed=1111,
                    recipe=recipe,
                    training_pair_digest=_digest(f"training:{recipe_id}:{seed}:{domain}"),
                    evaluation_pair_digest=evaluation_digest,
                    b3_evaluation_digest=b3_digest,
                    oracle_kind=_oracle(domain),
                    b3_scope="test" if domain in B3_DOMAINS else None,
                    requires_recipe_audit=candidate.fixed_budget,
                )
                gold_domain = domain in PAIR_DOMAINS
                auprc = 0.7 + (candidate.gold_delta_auprc if gold_domain else candidate.proxy_delta_auprc)
                auroc = 0.8 + (candidate.gold_delta_auroc if gold_domain else candidate.proxy_delta_auroc)
                b3_f1 = 0.8 + candidate.b3_delta if domain in B3_DOMAINS else None
                result_path = tmp_path / "folds" / recipe_id / str(seed) / f"{domain}.json"
                write_fold_result(
                    result_path,
                    _payload(
                        expectation,
                        training_rows=candidate.training_rows,
                        auprc=auprc,
                        auroc=auroc,
                        b3_f1=b3_f1,
                    ),
                    expected=expectation,
                )
                entries.append(
                    _manifest_entry(
                        result_path,
                        expectation,
                        root=tmp_path,
                        run_manifest_path=run_manifest_path,
                    )
                )
    if reverse_entries:
        entries.reverse()
    manifest = {"folds": entries, "schema_version": RANKING_INPUT_SCHEMA_VERSION}
    path = tmp_path / "ranking_input.json"
    path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _replace_declared_run_manifest(
    ranking_input_path: Path,
    *,
    entry_index: int,
    run_manifest: dict[str, Any],
) -> None:
    ranking_input = json.loads(ranking_input_path.read_text(encoding="utf-8"))
    declared_path = ranking_input["folds"][entry_index]["run_manifest_path"]
    run_manifest_path = ranking_input_path.parent / declared_path
    run_manifest_path.write_text(
        json.dumps(run_manifest, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    run_manifest_sha256 = _sha256_file(run_manifest_path)
    for entry in ranking_input["folds"]:
        if entry["run_manifest_path"] == declared_path:
            entry["run_manifest_sha256"] = run_manifest_sha256
    ranking_input_path.write_text(
        json.dumps(ranking_input, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_combiner_filters_initial_full_seed_and_finalist_only_seeds(tmp_path: Path) -> None:
    finalists = ["uniform_100k", "balanced_linker"]
    paths = [
        _write_combiner_manifest(
            tmp_path,
            "seed_1111",
            ["uniform_100k", "balanced", "pairwise", "balanced_linker", "pairwise_linker"],
        ),
        _write_combiner_manifest(tmp_path, "seed_2222", finalists),
        _write_combiner_manifest(tmp_path, "seed_3333", finalists),
    ]

    combined = combine_ranking_inputs(paths, arms=finalists)

    assert [entry["expected"]["arm"] for entry in combined["folds"]] == finalists * 3
    assert all(Path(entry["path"]).is_absolute() for entry in combined["folds"])
    assert all(Path(entry["run_manifest_path"]).is_absolute() for entry in combined["folds"])
    assert combined["schema_version"] == RANKING_INPUT_SCHEMA_VERSION


@pytest.mark.parametrize("arms", [[], ["uniform_100k", "uniform_100k"], ["uniform_100k", " "]])
def test_combiner_rejects_invalid_requested_arms(tmp_path: Path, arms: list[str]) -> None:
    manifest = _write_combiner_manifest(tmp_path, "seed_1111", ["uniform_100k"])

    with pytest.raises(ValueError, match="Requested arms"):
        combine_ranking_inputs([manifest], arms=arms)


def test_combiner_requires_every_requested_arm_in_every_manifest(tmp_path: Path) -> None:
    full = _write_combiner_manifest(tmp_path, "seed_1111", ["uniform_100k", "balanced_linker"])
    incomplete = _write_combiner_manifest(tmp_path, "seed_2222", ["uniform_100k"])

    with pytest.raises(ValueError, match=r"missing requested arms \['balanced_linker'\].*seed_2222"):
        combine_ranking_inputs([full, incomplete], arms=["uniform_100k", "balanced_linker"])


def test_combiner_validates_expected_arm_before_filtering(tmp_path: Path) -> None:
    manifest = _write_combiner_manifest(tmp_path, "seed_1111", ["uniform_100k", "excluded"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["folds"][1]["expected"]["arm"] = " "
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="malformed expected.arm"):
        combine_ranking_inputs([manifest], arms=["uniform_100k"])


def _direct_score(
    *,
    delta_auprc: np.ndarray | None = None,
    delta_auroc: np.ndarray | None = None,
    delta_b3: np.ndarray | None = None,
    proxy_delta_auprc: float = 0.0,
    proxy_delta_auroc: float = 0.0,
) -> dict[str, Any]:
    seeds = (1, 2, 3)
    recipe = _recipe("candidate", fixed_budget=True, complexity_rank=1, dose=10_000)
    return score_candidate(
        recipe_id=recipe_id_for(recipe),
        arm="candidate",
        recipe=recipe,
        seeds=seeds,
        delta_auprc=np.full((3, 8), 0.002) if delta_auprc is None else delta_auprc,
        delta_auroc=np.full((3, 8), 0.001) if delta_auroc is None else delta_auroc,
        delta_b3=np.zeros((3, 7)) if delta_b3 is None else delta_b3,
        absolute_b3=np.full((3, 7), 0.8),
        proxy_delta_auprc=np.full((3, 7), proxy_delta_auprc),
        proxy_delta_auroc=np.full((3, 7), proxy_delta_auroc),
        training_rows=np.full((3, 15), 100),
        draws=make_bootstrap_draws(3),
    )


def test_ranker_selects_candidate_and_writes_all_outputs(tmp_path: Path) -> None:
    manifest = _write_study(tmp_path, [CandidateFixture("balanced_pairwise")])

    artifacts = rank_manifest(manifest)
    output = tmp_path / "ranking"
    write_ranking_outputs(output, artifacts)

    assert artifacts.ranking["decision"]["decision"] == "replace_baseline"
    assert artifacts.ranking["decision"]["recommended_arm"] == "balanced_pairwise"
    assert artifacts.ranking["policy"]["bootstrap_replicates"] == 50_000
    assert artifacts.ranking["policy"]["bootstrap_seed"] == BOOTSTRAP_SEED
    assert artifacts.input_manifest["comparison_identity"]["sha256"]
    assert len(artifacts.input_manifest["run_manifests"]) == 6
    expected_files = {
        "bootstrap_summary.csv",
        "concentration.json",
        "domain_summary.csv",
        "input_manifest.json",
        "paired_deltas.csv",
        "ranking.csv",
        "ranking.json",
        "report.md",
    }
    assert {path.name for path in output.iterdir()} == expected_files
    load_strict_json(output / "ranking.json")
    with (output / "paired_deltas.csv").open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3 * 15


def test_ranker_rejects_incomplete_recipe_seed_domain_grid(tmp_path: Path) -> None:
    manifest = _write_study(tmp_path, [CandidateFixture("candidate")])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["folds"].pop()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Incomplete recipe/seed/15-domain grid"):
        rank_manifest(manifest)


def test_ranker_rejects_evaluation_identity_drift(tmp_path: Path) -> None:
    manifest = _write_study(
        tmp_path,
        [CandidateFixture("candidate", evaluation_drift_domain="qian")],
    )

    with pytest.raises(ValueError, match="Evaluation identity mismatch|changes across seeds"):
        rank_manifest(manifest)


def test_ranker_rejects_result_content_not_bound_by_manifest(tmp_path: Path) -> None:
    manifest = _write_study(tmp_path, [CandidateFixture("candidate")])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    result_path = tmp_path / payload["folds"][0]["path"]
    result_path.write_text(result_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rank_manifest(manifest)


def test_ranker_rejects_run_manifest_content_not_bound_by_manifest(tmp_path: Path) -> None:
    manifest = _write_study(tmp_path, [CandidateFixture("candidate")])
    ranking_input = json.loads(manifest.read_text(encoding="utf-8"))
    run_manifest_path = tmp_path / ranking_input["folds"][0]["run_manifest_path"]
    run_manifest_path.write_text(run_manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Run manifest SHA-256 mismatch"):
        rank_manifest(manifest)


def test_ranker_strictly_recomputes_declared_run_manifest_identity(tmp_path: Path) -> None:
    manifest = _write_study(tmp_path, [CandidateFixture("candidate")])
    ranking_input = json.loads(manifest.read_text(encoding="utf-8"))
    run_manifest_path = tmp_path / ranking_input["folds"][0]["run_manifest_path"]
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["runtime_versions"]["numpy"] = "9.9.9"
    _replace_declared_run_manifest(manifest, entry_index=0, run_manifest=run_manifest)

    with pytest.raises(ValueError, match="comparison_identity"):
        rank_manifest(manifest)


def test_ranker_requires_fold_result_run_id_to_match_run_manifest(tmp_path: Path) -> None:
    manifest = _write_study(tmp_path, [CandidateFixture("candidate")])
    ranking_input = json.loads(manifest.read_text(encoding="utf-8"))
    expected = ranking_input["folds"][0]["expected"]
    replacement = _run_manifest(expected["recipe"], training_seed=9999)
    _replace_declared_run_manifest(manifest, entry_index=0, run_manifest=replacement)

    with pytest.raises(ValueError, match="Fold result run_id does not match"):
        rank_manifest(manifest)


def test_ranker_rejects_cross_run_comparison_identity_drift(tmp_path: Path) -> None:
    manifest = _write_study(
        tmp_path,
        [CandidateFixture("candidate", run_identity_drift="implementation")],
    )

    with pytest.raises(ValueError, match="share one comparison identity"):
        rank_manifest(manifest)


def test_two_way_bootstrap_is_deterministic_and_uses_both_axes() -> None:
    draws = make_bootstrap_draws(3)
    values = np.arange(24, dtype=np.float64).reshape(3, 8)

    first = two_way_paired_bootstrap(
        values,
        seed_indices=draws.seed_indices,
        domain_indices=draws.pair_domain_indices,
    )
    second = two_way_paired_bootstrap(
        values,
        seed_indices=draws.seed_indices,
        domain_indices=draws.pair_domain_indices,
    )
    expected_first = values[
        draws.seed_indices[0, :, None],
        draws.pair_domain_indices[0, None, :],
    ].mean()

    np.testing.assert_array_equal(first, second)
    assert len(first) == BOOTSTRAP_REPLICATES
    assert first[0] == expected_first


def test_manifest_order_does_not_change_decision_or_bootstrap(tmp_path: Path) -> None:
    forward_root = tmp_path / "forward"
    reverse_root = tmp_path / "reverse"
    forward = rank_manifest(_write_study(forward_root, [CandidateFixture("candidate")]))
    reverse = rank_manifest(_write_study(reverse_root, [CandidateFixture("candidate")], reverse_entries=True))

    assert forward.ranking["decision"] == reverse.ranking["decision"]
    assert forward.ranking["candidates"][0]["bootstrap"] == reverse.ranking["candidates"][0]["bootstrap"]


def test_gate_boundaries_are_inclusive_but_replacement_requires_positive_q05() -> None:
    exact_b3 = _direct_score(delta_b3=np.full((3, 7), -0.002))
    below_q05_b3 = _direct_score(delta_b3=np.full((3, 7), -0.002001))
    below_worst_ap = np.full((3, 8), 0.005)
    below_worst_ap[:, 0] = -0.010001
    bad_ap = _direct_score(delta_auprc=below_worst_ap)
    below_worst_auc = np.full((3, 8), 0.005)
    below_worst_auc[:, 0] = -0.010001
    bad_auc = _direct_score(delta_auroc=below_worst_auc)
    below_worst_b3 = np.zeros((3, 7))
    below_worst_b3[:, 0] = -0.010001
    bad_b3 = _direct_score(delta_b3=below_worst_b3)
    zero_ap = _direct_score(delta_auprc=np.zeros((3, 8)))

    assert exact_b3["gates"]["q05_delta_b3"]["passed"] is True
    assert below_q05_b3["gates"]["q05_delta_b3"]["passed"] is False
    assert bad_ap["gates"]["worst_domain_delta_auprc"]["passed"] is False
    assert bad_auc["gates"]["worst_domain_delta_auroc"]["passed"] is False
    assert bad_b3["gates"]["worst_domain_delta_b3"]["passed"] is False
    assert zero_ap["replacement_evidence"]["passed"] is False
    assert zero_ap["replacement_eligible"] is False


def test_positive_seed_and_domain_gates_use_frozen_counts() -> None:
    two_positive_seeds = np.vstack((np.full(8, 0.002), np.full(8, 0.002), np.full(8, -0.001)))
    one_positive_seed = np.vstack((np.full(8, 0.002), np.full(8, -0.001), np.full(8, -0.001)))
    five_positive_domains = np.zeros((3, 8))
    five_positive_domains[:, :5] = 0.002
    four_positive_domains = np.zeros((3, 8))
    four_positive_domains[:, :4] = 0.002

    assert _direct_score(delta_auprc=two_positive_seeds)["gates"]["positive_seed_fraction"]["passed"]
    assert not _direct_score(delta_auprc=one_positive_seed)["gates"]["positive_seed_fraction"]["passed"]
    assert _direct_score(delta_auprc=five_positive_domains)["gates"]["positive_auprc_domains"]["passed"]
    assert not _direct_score(delta_auprc=four_positive_domains)["gates"]["positive_auprc_domains"]["passed"]


def _selection_score(
    recipe_id: str,
    *,
    q05: float,
    fixed_budget: bool,
    rows: float,
    manual_review: bool = False,
) -> dict[str, Any]:
    return {
        "arm": recipe_id,
        "bootstrap": {"delta_auprc": {"q05": q05}},
        "complexity_rank": 1,
        "fixed_budget": fixed_budget,
        "manual_review_required": manual_review,
        "mean_delta_auprc": q05,
        "mean_delta_auroc": 0.001,
        "mean_delta_b3": 0.0,
        "mean_training_rows": rows,
        "recipe_id": recipe_id,
        "replacement_eligible": True,
        "worst_absolute_domain_mean_b3": 0.8,
    }


def test_practical_tie_prefers_fixed_budget_but_larger_gap_preserves_primary() -> None:
    simple = _selection_score("simple", q05=0.0010, fixed_budget=True, rows=100)
    close_complex = _selection_score("complex", q05=0.0014, fixed_budget=False, rows=120)
    far_complex = _selection_score("complex", q05=0.0016, fixed_budget=False, rows=120)

    tied = select_recommendation([simple, close_complex], baseline_recipe_id="baseline")
    separated = select_recommendation([simple, far_complex], baseline_recipe_id="baseline")

    assert tied["recommended_recipe_id"] == "simple"
    assert separated["recommended_recipe_id"] == "complex"


def test_proxy_loss_is_manual_review_not_rank_credit() -> None:
    exact_boundary = _direct_score(proxy_delta_auprc=-0.05)
    below_boundary = _direct_score(proxy_delta_auprc=-0.050001)
    selected = select_recommendation([below_boundary], baseline_recipe_id="baseline")

    assert exact_boundary["manual_review_required"] is False
    assert below_boundary["manual_review_required"] is True
    assert selected["decision"] == "manual_review_required"
    assert selected["recommended_recipe_id"] == "baseline"
    assert selected["provisional_recipe_id"] == below_boundary["recipe_id"]


def test_concentration_reports_top_shares_and_leave_one_out() -> None:
    values = np.asarray([0.4, 0.3, 0.2, 0.1, -0.1, 0.0, 0.0, 0.0])

    diagnostics = concentration_diagnostics(values)

    assert diagnostics["top1_domain"] == PAIR_DOMAINS[0]
    assert diagnostics["top1_share"] == pytest.approx(0.4)
    assert diagnostics["top2_share"] == pytest.approx(0.7)
    assert diagnostics["leave_one_domain_out"][PAIR_DOMAINS[0]] == pytest.approx(0.5 / 7)


def test_same_arm_with_different_recipe_ids_remains_two_candidates(tmp_path: Path) -> None:
    manifest = _write_study(
        tmp_path,
        [
            CandidateFixture("balanced_linker", gold_delta_auprc=0.002, dose=10_000),
            CandidateFixture("balanced_linker", gold_delta_auprc=0.003, dose=50_000),
        ],
    )

    artifacts = rank_manifest(manifest)

    assert len(artifacts.ranking["candidates"]) == 2
    assert len({row["recipe_id"] for row in artifacts.ranking["candidates"]}) == 2


def test_no_eligible_candidate_retains_uniform_baseline() -> None:
    score = _selection_score("candidate", q05=0.001, fixed_budget=True, rows=100)
    score = copy.deepcopy(score)
    score["replacement_eligible"] = False

    decision = select_recommendation([score], baseline_recipe_id="baseline")

    assert decision["decision"] == "retain_baseline_no_eligible_replacement"
    assert decision["recommended_arm"] == "uniform_100k"

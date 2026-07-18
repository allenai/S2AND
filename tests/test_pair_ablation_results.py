from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts._pair_ablation.results import (
    FOLD_RESULT_SCHEMA_VERSION,
    FoldResultExpectation,
    load_fold_result,
    recipe_id_for,
    validate_fold_result,
    write_fold_result,
)

RUN_ID = "a" * 64
TRAINING_DIGEST = "b" * 64
EVALUATION_DIGEST = "c" * 64
B3_DIGEST = "d" * 64
CACHE_DIGEST = "e" * 64


def _recipe(*, fixed_budget: bool = False) -> dict[str, object]:
    return {
        "arm": "uniform_100k",
        "assembly_version": "exact_budget_v1",
        "auxiliary_sources": [],
        "balancing": "none",
        "base_sampler": "uniform_100k",
        "budget_policy": "uniform_after_lodo" if fixed_budget else "additive",
        "complexity_rank": 0,
        "fixed_budget": fixed_budget,
        "source_caps": {"uniform_pairs_per_domain": 100_000},
    }


def _expectation(*, public: bool = True, fixed_budget: bool = False) -> FoldResultExpectation:
    return FoldResultExpectation(
        run_id=RUN_ID,
        arm="uniform_100k",
        source_families=("gold_cluster_uniform",),
        held_out_domain="aminer" if public else "medline",
        training_seed=1111,
        evaluation_seed=1111,
        recipe=_recipe(fixed_budget=fixed_budget),
        training_pair_digest=TRAINING_DIGEST,
        evaluation_pair_digest=EVALUATION_DIGEST,
        b3_evaluation_digest=B3_DIGEST if public else None,
        oracle_kind="gold_cluster_pairs" if public else "fixed_pair_labels",
        b3_scope="test" if public else None,
        requires_recipe_audit=fixed_budget,
    )


def _payload(*, public: bool = True, fixed_budget: bool = False) -> dict[str, object]:
    expectation = _expectation(public=public, fixed_budget=fixed_budget)
    recipe = expectation.recipe
    audit = None
    if fixed_budget:
        audit = {
            "target_rows": 10,
            "final_rows": 10,
            "held_out_rows": 0,
            "base_filler_rows": 7,
            "selection_sha256": TRAINING_DIGEST,
        }
    b3 = None
    if public:
        b3 = {
            "scope": "test",
            "threshold": 0.49,
            "threshold_calibration": {"precision": 0.9, "recall": 0.8, "f1": 0.847},
            "heldout_blocks": 3,
            "heldout_signatures": 7,
            "heldout_pairs": 9,
            "precision": 0.8,
            "recall": 0.75,
            "f1": 0.774,
            "scoring_backend": "rust_pair_features_and_rust_lightgbm",
        }
    return {
        "schema_version": FOLD_RESULT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "recipe_id": recipe_id_for(recipe),
        "recipe": recipe,
        "arm": expectation.arm,
        "source_families": list(expectation.source_families),
        "held_out_domain": expectation.held_out_domain,
        "training_seed": expectation.training_seed,
        "evaluation_seed": expectation.evaluation_seed,
        "evaluation_pair_digest": expectation.evaluation_pair_digest,
        "b3_evaluation_digest": expectation.b3_evaluation_digest,
        "training_pair_digest": expectation.training_pair_digest,
        "model_cache_hit": False,
        "training_rows": 10,
        "training_positives": 4,
        "training_negatives": 6,
        "training_source_counts": [
            {
                "source_domain": "qian",
                "source_family": "gold_cluster_uniform",
                "rows": 10,
                "positives": 4,
                "negatives": 6,
            }
        ],
        "pair_recipe_assembly": audit,
        "pairwise": {
            "oracle_kind": expectation.oracle_kind,
            "rows": 10,
            "positives": 4,
            "negatives": 6,
            "prevalence": 0.4,
            "auroc": 0.8,
            "auprc": 0.7,
        },
        "b3": b3,
        "b3_cache_digests": [CACHE_DIGEST] if public else [],
        "models": {
            "main": {"model_path": "main.lgb", "model_sha256": "1" * 64},
            "nameless": {"model_path": "nameless.lgb", "model_sha256": "2" * 64},
        },
        "elapsed_seconds": 2.5,
    }


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("fixed_budget", [False, True])
def test_fold_result_round_trip(tmp_path: Path, public: bool, fixed_budget: bool) -> None:
    expected = _expectation(public=public, fixed_budget=fixed_budget)
    payload = _payload(public=public, fixed_budget=fixed_budget)
    path = tmp_path / "result.json"

    write_fold_result(path, payload, expected=expected)

    assert load_fold_result(path, expected=expected) == payload


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(schema_version="bad"), "schema version"),
        (lambda value: value.update(arm="copied_arm"), "arm mismatch"),
        (lambda value: value.update(held_out_domain="qian"), "held_out_domain mismatch"),
        (lambda value: value["pairwise"].update(rows=9), "pairwise counts"),
        (lambda value: value["pairwise"].update(prevalence=0.5), "prevalence"),
        (lambda value: value.update(training_rows=9), "training counts"),
        (
            lambda value: value["training_source_counts"][0].update(source_domain="aminer"),
            "held-out source domain",
        ),
        (lambda value: value["pairwise"].update(auprc=float("nan")), "pairwise.auprc"),
        (lambda value: value.update(unexpected=True), "schema mismatch"),
        (lambda value: value.update(recipe_id="f" * 64), "recipe identity"),
        (lambda value: value.update(b3=None), "B3 must be present"),
        (lambda value: value.update(b3_cache_digests=[]), "b3_cache_digests"),
    ],
)
def test_fold_result_rejects_malformed_payload(mutation: object, match: str) -> None:
    payload = copy.deepcopy(_payload())
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        validate_fold_result(payload, expected=_expectation())


def test_fold_result_rejects_b3_on_pair_only_domain() -> None:
    payload = _payload(public=False)
    payload["b3"] = copy.deepcopy(_payload(public=True)["b3"])
    payload["b3_cache_digests"] = [CACHE_DIGEST]

    with pytest.raises(ValueError, match="B3 must be absent"):
        validate_fold_result(payload, expected=_expectation(public=False))


def test_fold_result_requires_exact_budget_audit() -> None:
    payload = _payload(fixed_budget=True)
    payload["pair_recipe_assembly"] = None

    with pytest.raises(ValueError, match="pair_recipe_assembly is required"):
        validate_fold_result(payload, expected=_expectation(fixed_budget=True))


def test_fold_result_accepts_required_additive_recipe_audit() -> None:
    expected = replace(_expectation(), requires_recipe_audit=True)
    payload = _payload()
    payload["pair_recipe_assembly"] = {
        "target_rows": 10,
        "final_rows": 10,
        "held_out_rows": 0,
        "base_filler_rows": 0,
        "selection_sha256": TRAINING_DIGEST,
        "mode": "additive_linker",
        "base_rows_after_lodo": 8,
        "linker_selected_rows": 2,
    }

    assert validate_fold_result(payload, expected=expected) == payload


def test_fold_result_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"run_id":"a","run_id":"b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        load_fold_result(path, expected=_expectation())


def test_writer_rejects_non_json_values(tmp_path: Path) -> None:
    payload = _payload()
    payload["models"]["main"]["unsupported"] = Path("main.lgb")  # type: ignore[index]

    with pytest.raises(TypeError):
        write_fold_result(tmp_path / "result.json", payload, expected=_expectation())


def test_recipe_id_is_seed_independent_but_dose_sensitive() -> None:
    recipe = _recipe()
    same = copy.deepcopy(recipe)
    different = copy.deepcopy(recipe)
    different["source_caps"]["uniform_pairs_per_domain"] = 50_000  # type: ignore[index]

    assert recipe_id_for(recipe) == recipe_id_for(same)
    assert recipe_id_for(recipe) != recipe_id_for(different)


def test_strict_writer_never_serializes_nan(tmp_path: Path) -> None:
    payload = _payload()
    payload["elapsed_seconds"] = float("nan")

    with pytest.raises(ValueError, match="elapsed_seconds"):
        write_fold_result(tmp_path / "result.json", payload, expected=_expectation())


def test_written_json_is_standard_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_fold_result(path, _payload(), expected=_expectation())

    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == FOLD_RESULT_SCHEMA_VERSION

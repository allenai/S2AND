from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from scripts._pair_ablation.prepared import B3Evaluation, PairEvaluation, PreparedStudy
from scripts._pair_ablation.study import (
    BASELINE,
    AdditiveDose,
    validate_catalog,
)
from scripts.run_pair_source_ablation import _arms, _b3_at_threshold, _build_linkages, run_ablation


def _donor(path: Path, width: int) -> None:
    rng = np.random.default_rng(width)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 10)
    features = rng.normal(size=(20, width)).astype(np.float32)
    features[:, 0] += labels * 2
    model = LGBMClassifier(
        n_estimators=3,
        learning_rate=0.1,
        num_leaves=4,
        max_depth=3,
        min_child_samples=1,
        min_child_weight=1e-3,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.2,
        min_split_gain=0.0,
        verbosity=-1,
        random_state=7,
        n_jobs=1,
    ).fit(features, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def _pairs(domain: str, family: str, count: int) -> list[tuple[str, str, str, str, int]]:
    return [(domain, family, f"{domain}:a{index:03}", f"{domain}:b{index:03}", index % 2) for index in range(count)]


def _pair_evaluation(rng: np.random.Generator) -> PairEvaluation:
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 6)
    main = rng.normal(size=(len(labels), 33)).astype(np.float32)
    nameless = rng.normal(size=(len(labels), 17)).astype(np.float32)
    main[:, 0] += labels * 2
    nameless[:, 0] += labels * 2
    return PairEvaluation(main, nameless, labels)


def _b3_evaluation(rng: np.random.Generator, domain: str) -> B3Evaluation:
    main = rng.normal(size=(6, 33)).astype(np.float32)
    nameless = rng.normal(size=(6, 17)).astype(np.float32)
    staged = np.full(6, np.nan)
    staged[0] = -90_000.0  # LARGE_DISTANCE after S2AND removes its staging offset.
    return B3Evaluation(
        main=main,
        nameless=nameless,
        staged_labels=staged,
        pair_offsets=np.asarray([0, 6], dtype=np.int64),
        signature_offsets=np.asarray([0, 4], dtype=np.int64),
        signature_ids=np.asarray([f"{domain}:s{index}" for index in range(4)]),
        gold_cluster_ids=np.asarray(["gold-a", "gold-a", "gold-b", "gold-b"]),
    )


def _prepared(tmp_path: Path) -> PreparedStudy:
    rows = [
        *_pairs("pubmed", "base", 20),
        *_pairs("qian", "base", 20),
        *_pairs("h_wang", "linker", 8),
    ]
    catalog = validate_catalog(
        pd.DataFrame(
            rows,
            columns=("source_domain", "source_family", "pair1", "pair2", "label"),
        )
    )
    rng = np.random.default_rng(41)
    labels = catalog["label"].to_numpy()
    main = rng.normal(size=(len(catalog), 33)).astype(np.float32)
    nameless = rng.normal(size=(len(catalog), 17)).astype(np.float32)
    main[:, 0] += labels * 2
    nameless[:, 0] += labels * 2
    return PreparedStudy(
        root=tmp_path / "prepared",
        catalog=catalog,
        training_main=main,
        training_nameless=nameless,
        evaluation={"pubmed": _pair_evaluation(rng), "qian": _pair_evaluation(rng)},
        b3={
            "pubmed": _b3_evaluation(rng, "pubmed"),
            "qian": _b3_evaluation(rng, "qian"),
        },
        prepared_digest="a" * 64,
    )


def test_fresh_runner_preserves_base_and_writes_flat_paired_results(tmp_path) -> None:
    donors = tmp_path / "donors"
    _donor(donors / "main.lgb", 33)
    _donor(donors / "nameless.lgb", 17)
    output = tmp_path / "run"
    results = run_ablation(
        _prepared(tmp_path),
        donor_model_dir=donors,
        output_dir=output,
        domains=("pubmed",),
        arms=(BASELINE, AdditiveDose("big7", 4)),
        training_seed=17,
        n_jobs=1,
        estimator_scale=1.0,
        total_ram_bytes=1024**3,
    )

    baseline, candidate = results
    assert len(baseline["study_digest"]) == 64
    assert candidate["study_digest"] == baseline["study_digest"]
    assert candidate["baseline_pair_digest"] == baseline["training_pair_digest"]
    assert candidate["baseline_rows"] == baseline["training_rows"]
    assert candidate["training_rows"] == baseline["training_rows"] + 4
    assert candidate["linker_rows"] == 4
    assert baseline["b3_f1"] is not None and candidate["b3_f1"] is not None
    for result in results:
        result_path = output / "results" / "17" / result["arm"] / "pubmed.json"
        assert result_path.is_file()
        assert (output / "models" / "17" / result["arm"] / "pubmed" / "main.lgb").is_file()

    with pytest.raises(FileExistsError, match="already exists"):
        run_ablation(
            _prepared(tmp_path),
            donor_model_dir=donors,
            output_dir=output,
            domains=("pubmed",),
            arms=(BASELINE,),
            training_seed=17,
            n_jobs=1,
        )


def test_cli_arm_expansion_is_small_and_strict() -> None:
    arms = _arms([4], ["big7"])
    assert arms == (BASELINE, AdditiveDose("big7", 4))
    assert _arms([], []) == (BASELINE,)
    with pytest.raises(ValueError, match="requires"):
        _arms([], ["big7"])
    with pytest.raises(ValueError, match="positive even"):
        _arms([3], ["big7"])


def test_b3_rows_use_scipy_condensed_distance_order() -> None:
    evaluation = _b3_evaluation(np.random.default_rng(3), "pubmed")
    distances = np.asarray([0.1, 0.9, 0.9, 0.9, 0.9, 0.1])
    metrics = _b3_at_threshold(
        {"pubmed": evaluation},
        {"pubmed": _build_linkages(evaluation, distances)},
        0.5,
    )
    assert metrics == pytest.approx((1.0, 1.0, 1.0))

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import lightgbm as lgb
import numpy as np
import pytest
from lightgbm import LGBMClassifier

from s2and.featurizer import FeaturizationInfo
from scripts._pair_ablation.modeling import (
    averaged_positive_probability,
    donor_lightgbm_params,
    pairwise_metrics,
    train_pairwise_models,
)


def _info(constraints: str) -> FeaturizationInfo:
    return cast(FeaturizationInfo, SimpleNamespace(lightgbm_monotone_constraints=constraints))


def _write_donor(path, features: np.ndarray, labels: np.ndarray) -> None:
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


def test_fixed_pairwise_models_train_save_and_average_predictions(tmp_path) -> None:
    rng = np.random.default_rng(11)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 20)
    main = rng.normal(size=(40, 3)).astype(np.float32)
    nameless = rng.normal(size=(40, 2)).astype(np.float32)
    main[:, 0] += labels * 2
    nameless[:, 0] += labels * 2
    main[0, 2] = np.nan
    nameless[1, 1] = np.nan
    donors = tmp_path / "donors"
    _write_donor(donors / "main.lgb", main, labels)
    _write_donor(donors / "nameless.lgb", nameless, labels)

    models = train_pairwise_models(
        main,
        nameless,
        labels,
        main_featurizer_info=_info("0,0,0"),
        nameless_featurizer_info=_info("0,0"),
        donor_model_dir=donors,
        output_dir=tmp_path / "trained",
        n_jobs=1,
        random_seed=17,
        estimator_scale=1.0,
    )

    assert models.main_path == tmp_path / "trained" / "main.lgb"
    assert models.nameless_path == tmp_path / "trained" / "nameless.lgb"
    assert models.main_path.is_file() and models.nameless_path.is_file()
    observed = averaged_positive_probability(models, main, nameless)
    expected = (models.main.predict_proba_positive(main) + models.nameless.predict_proba_positive(nameless)) / 2
    np.testing.assert_allclose(observed, expected)
    assert np.all((observed >= 0) & (observed <= 1))


def test_donor_parameters_are_fixed_and_scaled(tmp_path) -> None:
    features = np.asarray([[0.0], [1.0], [0.1], [0.9]], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    path = tmp_path / "donor.lgb"
    _write_donor(path, features, labels)

    donor = lgb.Booster(model_file=str(path)).params
    translated = donor_lightgbm_params(path, estimator_scale=0.5)

    assert translated["n_estimators"] == max(2, round(int(donor["num_iterations"]) * 0.5))
    assert translated["learning_rate"] == float(donor["learning_rate"])
    with pytest.raises(ValueError, match="positive"):
        donor_lightgbm_params(path, estimator_scale=0)


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0.0, 0.5]),
        np.asarray([0.0, np.nan]),
        np.asarray([0.0, np.inf]),
        np.asarray(["0", "1"]),
        np.asarray([False, True]),
        np.asarray([1, 1]),
    ],
)
def test_training_rejects_invalid_binary_labels_before_loading_donors(tmp_path, labels: np.ndarray) -> None:
    features = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="labels"):
        train_pairwise_models(
            features,
            features,
            labels,
            main_featurizer_info=_info("0"),
            nameless_featurizer_info=_info("0"),
            donor_model_dir=tmp_path / "missing",
            output_dir=tmp_path / "output",
            n_jobs=1,
            random_seed=1,
        )


def test_training_rejects_infinity_before_loading_donors(tmp_path) -> None:
    labels = np.asarray([0, 1])
    infinite_features = np.asarray([[np.inf], [1.0]], dtype=np.float32)
    info = _info("0")

    with pytest.raises(ValueError, match="infinite"):
        train_pairwise_models(
            infinite_features,
            infinite_features,
            labels,
            main_featurizer_info=info,
            nameless_featurizer_info=info,
            donor_model_dir=tmp_path / "missing",
            output_dir=tmp_path / "output",
            n_jobs=1,
            random_seed=1,
        )


def test_pairwise_metrics_reports_scores_and_validates_inputs() -> None:
    metrics = pairwise_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.2, 0.8, 0.9]),
        oracle_kind="gold",
    )
    assert metrics == {
        "oracle_kind": "gold",
        "rows": 4,
        "positives": 2,
        "negatives": 2,
        "prevalence": 0.5,
        "auroc": 1.0,
        "auprc": 1.0,
    }
    with pytest.raises(ValueError, match="both classes"):
        pairwise_metrics(np.asarray([1, 1]), np.asarray([0.7, 0.8]), oracle_kind="proxy")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pairwise_metrics(np.asarray([0, 1]), np.asarray([0.2, 1.1]), oracle_kind="gold")

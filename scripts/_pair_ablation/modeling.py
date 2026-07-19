"""Fit and score the two fixed pairwise models used by the ablation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from s2and.featurizer import FeaturizationInfo
from s2and.model_pairwise import PairwiseModeler
from s2and.production_model import NativeLightGBMBinaryClassifier


def donor_lightgbm_params(model_path: str | Path, *, estimator_scale: float = 1.0) -> dict[str, Any]:
    """Translate a released LightGBM booster into fixed training parameters."""

    if estimator_scale <= 0:
        raise ValueError("estimator_scale must be positive")
    params = lgb.Booster(model_file=str(model_path)).params
    return {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": max(2, round(int(params["num_iterations"]) * estimator_scale)),
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "max_depth": int(params["max_depth"]),
        "min_child_samples": int(params["min_data_in_leaf"]),
        "min_child_weight": float(params["min_sum_hessian_in_leaf"]),
        "subsample": float(params["bagging_fraction"]),
        "subsample_freq": int(params["bagging_freq"]),
        "colsample_bytree": float(params["feature_fraction"]),
        "reg_alpha": float(params["lambda_l1"]),
        "reg_lambda": float(params["lambda_l2"]),
        "min_split_gain": float(params["min_gain_to_split"]),
        "monotone_penalty": float(params.get("monotone_penalty", 0.0)),
    }


@dataclass(frozen=True, slots=True)
class TrainedPairwiseModels:
    """The saved main and nameless models and their native scorers."""

    main_path: Path
    nameless_path: Path
    main: NativeLightGBMBinaryClassifier
    nameless: NativeLightGBMBinaryClassifier


def _feature_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty-width feature matrix")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must contain numeric features")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real-valued features")
    matrix = np.asarray(raw, dtype=np.float32 if raw.dtype == np.float32 else np.float64, order="C")
    if bool(np.isinf(matrix).any()):
        raise ValueError(f"{name} must not contain infinite values")
    return matrix


def _binary_labels(values: np.ndarray, *, require_both: bool) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError("labels must be a numeric vector of integral values 0 or 1")
    numeric = np.asarray(raw, dtype=np.float64)
    if (
        not bool(np.isfinite(numeric).all())
        or not bool(np.equal(numeric, np.floor(numeric)).all())
        or not bool(np.isin(numeric, (0.0, 1.0)).all())
    ):
        raise ValueError("labels must be a numeric vector of integral values 0 or 1")
    labels = numeric.astype(np.int8)
    if require_both and set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both classes")
    return labels


def _fit_one(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    featurizer_info: FeaturizationInfo,
    donor_path: Path,
    output_path: Path,
    n_jobs: int,
    random_seed: int,
    estimator_scale: float,
) -> NativeLightGBMBinaryClassifier:
    params = donor_lightgbm_params(donor_path, estimator_scale=estimator_scale)
    params.update(
        {
            "n_jobs": n_jobs,
            "random_state": random_seed,
            "tree_learner": "data",
            "verbosity": -1,
            "monotone_constraints": featurizer_info.lightgbm_monotone_constraints,
            "monotone_constraints_method": "advanced",
        }
    )
    # PairwiseModeler remains the S2AND training boundary. An empty search space
    # makes this ablation a fixed-model comparison instead of a tuning study.
    modeler = PairwiseModeler(
        estimator=LGBMClassifier(**params),
        search_space={},
        n_iter=0,
        n_jobs=n_jobs,
        random_state=random_seed,
    )
    modeler.fit(features, labels, features, labels)
    if modeler.classifier is None:
        raise RuntimeError("PairwiseModeler did not produce a classifier")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modeler.classifier.booster_.save_model(str(output_path))
    return NativeLightGBMBinaryClassifier(output_path, n_jobs=n_jobs, n_features=features.shape[1])


def train_pairwise_models(
    main_features: np.ndarray,
    nameless_features: np.ndarray,
    labels: np.ndarray,
    *,
    main_featurizer_info: FeaturizationInfo,
    nameless_featurizer_info: FeaturizationInfo,
    donor_model_dir: str | Path,
    output_dir: str | Path,
    n_jobs: int,
    random_seed: int,
    estimator_scale: float = 1.0,
) -> TrainedPairwiseModels:
    """Fit fixed main and nameless models and save them under ``output_dir``."""

    main = _feature_matrix(main_features, name="main_features")
    nameless = _feature_matrix(nameless_features, name="nameless_features")
    target = _binary_labels(labels, require_both=True)
    if len(main) != len(nameless) or len(main) != len(target):
        raise ValueError("training matrices and labels must have equal row counts")
    if n_jobs <= 0:
        raise ValueError("n_jobs must be positive")

    donor_root = Path(donor_model_dir)
    output_root = Path(output_dir)
    main_path = output_root / "main.lgb"
    nameless_path = output_root / "nameless.lgb"
    main_model = _fit_one(
        main,
        target,
        featurizer_info=main_featurizer_info,
        donor_path=donor_root / "main.lgb",
        output_path=main_path,
        n_jobs=n_jobs,
        random_seed=random_seed,
        estimator_scale=estimator_scale,
    )
    nameless_model = _fit_one(
        nameless,
        target,
        featurizer_info=nameless_featurizer_info,
        donor_path=donor_root / "nameless.lgb",
        output_path=nameless_path,
        n_jobs=n_jobs,
        random_seed=random_seed,
        estimator_scale=estimator_scale,
    )
    return TrainedPairwiseModels(main_path, nameless_path, main_model, nameless_model)


def averaged_positive_probability(
    models: TrainedPairwiseModels,
    main_features: np.ndarray,
    nameless_features: np.ndarray,
) -> np.ndarray:
    """Average main and nameless same-cluster probabilities."""

    main = _feature_matrix(main_features, name="main_features")
    nameless = _feature_matrix(nameless_features, name="nameless_features")
    if len(main) != len(nameless):
        raise ValueError("evaluation matrices must have equal row counts")
    return (models.main.predict_proba_positive(main) + models.nameless.predict_proba_positive(nameless)) / 2.0


def pairwise_metrics(
    labels: np.ndarray,
    positive_probability: np.ndarray,
    *,
    oracle_kind: str,
) -> dict[str, float | int | str]:
    """Compute AUROC and AUPRC for same-cluster probability."""

    target = _binary_labels(labels, require_both=True)
    probability = np.asarray(positive_probability)
    if (
        probability.ndim != 1
        or len(target) != len(probability)
        or np.issubdtype(probability.dtype, np.bool_)
        or not np.issubdtype(probability.dtype, np.number)
        or np.issubdtype(probability.dtype, np.complexfloating)
    ):
        raise ValueError("probabilities must be a numeric vector aligned with labels")
    probability = np.asarray(probability, dtype=np.float64)
    if not bool(np.isfinite(probability).all()) or bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("probabilities must be finite values in [0, 1]")
    positives = int(target.sum())
    return {
        "oracle_kind": str(oracle_kind),
        "rows": int(len(target)),
        "positives": positives,
        "negatives": int(len(target) - positives),
        "prevalence": float(positives / len(target)),
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
    }

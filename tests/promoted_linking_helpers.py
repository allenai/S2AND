from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import lightgbm as lgb
import numpy as np

from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.features import promoted_linker_feature_columns
from s2and.incremental_linking.logistic_gate import logistic_gate_config
from s2and.model import Clusterer, FastCluster, _selected_feature_indices
from s2and.production_bundle import write_pairwise_production_bundle
from s2and.production_bundle_contract import CALIBRATED_EPS_CALIBRATION, EpsCalibration


def synthetic_pairwise_bundle_binding() -> dict[str, object]:
    """Return a structurally valid pairwise binding for isolated linker tests."""

    return {
        "ordered_feature_contract_digest": "1" * 64,
        "main_booster_sha256": "2" * 64,
        "nameless_booster_sha256": "3" * 64,
    }


def tiny_binary_booster(width: int, *, seed: int) -> lgb.LGBMClassifier:
    """Fit a deterministic binary booster for production-bundle tests."""

    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(32, width))
    labels = np.asarray([0, 1] * 16, dtype=np.int8)
    classifier = lgb.LGBMClassifier(
        objective="binary",
        verbosity=-1,
        n_jobs=1,
        learning_rate=0.2,
        num_leaves=3,
        min_child_samples=1,
        min_data_in_bin=1,
        force_col_wise=True,
        n_estimators=4,
        random_state=seed,
    )
    classifier.fit(matrix, labels)
    return classifier


def write_synthetic_pairwise_bundle(
    bundle_dir: Path,
    *,
    artifact_hashes: Mapping[str, str],
    release_version: str,
    eps_calibration: EpsCalibration = CALIBRATED_EPS_CALIBRATION,
) -> Clusterer:
    """Write a tiny native pairwise stage and return its source clusterer."""

    main_info = FeaturizationInfo(["name_similarity"])
    nameless_info = FeaturizationInfo(["year_diff"])
    clusterer = Clusterer(
        main_info,
        tiny_binary_booster(len(_selected_feature_indices(main_info)), seed=101),
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
        nameless_classifier=tiny_binary_booster(len(_selected_feature_indices(nameless_info)), seed=102),
        nameless_featurizer_info=nameless_info,
        batch_size=100,
    )
    clusterer.feature_contract = dict(artifact_hashes)
    clusterer.best_params = {"eps": 0.5, "linkage": "average"}
    write_pairwise_production_bundle(
        clusterer,
        bundle_dir,
        release_version=release_version,
        eps_calibration=eps_calibration,
    )
    return clusterer


def build_tiny_promoted_booster() -> tuple[lgb.Booster, np.ndarray]:
    """Build a deterministic booster and prediction fixture for promoted-linker tests."""

    columns = promoted_linker_feature_columns()
    matrix = np.zeros((8, len(columns)), dtype=np.float32)
    matrix[:, columns.index("min_distance")] = np.linspace(1.0, 0.0, len(matrix), dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    dataset = lgb.Dataset(matrix, label=labels, free_raw_data=False)
    booster = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_threads": 1,
            "learning_rate": 0.3,
            "num_leaves": 3,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "force_col_wise": True,
        },
        dataset,
        num_boost_round=6,
    )
    return booster, matrix[:3]


def tiny_logistic_gate_config(link: bool = True) -> dict[str, object]:
    """Build a deterministic gate with an explicit link preference."""
    return logistic_gate_config(
        feature_names=("chosen_probability",),
        weights=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        bias=np.asarray([0.0, 0.0, 10.0 if link else -10.0], dtype=np.float64),
        missing_values=np.asarray([0.0], dtype=np.float64),
        calibration_mode="test",
    )

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from s2and.consts import FEATURIZER_VERSION, NORMALIZATION_VERSION
from s2and.incremental_linking.features import promoted_linker_feature_columns


def synthetic_pairwise_bundle_binding() -> dict[str, object]:
    """Return a structurally valid pairwise binding for isolated linker tests."""

    return {
        "normalization_version": NORMALIZATION_VERSION,
        "featurizer_version": FEATURIZER_VERSION,
        "ordered_feature_contract_digest": "1" * 64,
        "main_booster_sha256": "2" * 64,
        "nameless_booster_sha256": "3" * 64,
    }


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

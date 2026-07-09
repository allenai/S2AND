"""last_first_initial count-key semantics (canonical_v2: initial-char only).

The pre-cutover "legacy_full_first_token" semantics was retired with the
canonical_v2 migration (D8); these tests pin that the token is rejected
everywhere and that the initial-char behavior is stable across prediction.
"""

import numpy as np
import pytest

from s2and.data import (
    NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
    ANDData,
)
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer

RETIRED_LEGACY_SEMANTICS = "legacy_full_first_token"


class _ConstantClassifier:
    def predict_proba(self, X, **_kwargs):
        array = np.asarray(X)
        rows = int(array.shape[0]) if array.ndim > 1 else int(array.size)
        return np.tile(np.asarray([[0.5, 0.5]], dtype=np.float64), (rows, 1))


def _name_count_tables() -> dict[str, dict[str, int]]:
    return {
        "first_dict": {"abdul": 7, "alexander": 8},
        "last_dict": {"sattar": 9, "konovalov": 10},
        "first_last_dict": {"abdul sattar": 11, "alexander konovalov": 12},
        "last_first_initial_dict": {
            "sattar a": 13,
            "sattar abdul": 41,
            "konovalov a": 14,
            "konovalov alexander": 42,
        },
    }


def _build_clusterer(*, featurizer_version: int) -> Clusterer:
    featurizer_info = FeaturizationInfo(features_to_use=["name_counts"], featurizer_version=featurizer_version)
    return Clusterer(featurizer_info=featurizer_info, classifier=_ConstantClassifier(), n_jobs=1, use_cache=False)


def _dummy_dataset(name: str, **kwargs) -> ANDData:
    return ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name=name,
        mode="inference",
        load_name_counts=_name_count_tables(),
        preprocess=True,
        n_jobs=1,
        **kwargs,
    )


def test_initial_char_semantics_is_default_and_stable():
    dataset = _dummy_dataset("dummy_name_count_semantics_default")
    baseline = dataset.signatures["1"].author_info_name_counts
    assert baseline is not None
    assert baseline.last_first_initial == 13
    assert dataset.name_counts_last_first_initial_semantics == NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR

    # Re-setting the only valid value is a no-op.
    assert dataset.set_name_counts_last_first_initial_semantics(NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR) is False


def test_retired_legacy_semantics_token_is_rejected():
    dataset = _dummy_dataset("dummy_name_count_semantics_rejection")
    with pytest.raises(ValueError, match="canonical_v2 retired"):
        dataset.set_name_counts_last_first_initial_semantics(RETIRED_LEGACY_SEMANTICS)

    with pytest.raises(ValueError, match="canonical_v2 retired"):
        _dummy_dataset(
            "dummy_name_count_semantics_ctor_rejection",
            name_counts_last_first_initial_semantics=RETIRED_LEGACY_SEMANTICS,
        )


def test_inference_prediction_keeps_initial_char_semantics():
    dataset = _dummy_dataset(
        "dummy_inference_name_count_semantics_gate",
        name_counts_last_first_initial_semantics=NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
    )
    clusterer = _build_clusterer(featurizer_version=1)

    before = dataset.signatures["1"].author_info_name_counts
    assert before is not None
    assert before.last_first_initial == 13

    clusterer.predict_helper({"block": ["1"]}, dataset, use_s2_clusters=True)

    after = dataset.signatures["1"].author_info_name_counts
    assert after is not None
    assert after.last_first_initial == 13
    assert dataset.name_counts_last_first_initial_semantics == NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR

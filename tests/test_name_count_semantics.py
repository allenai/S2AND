import numpy as np
import pytest

import s2and.feature_port as feature_port
from s2and.data import (
    NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
    NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY,
    ANDData,
)
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer


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


def test_set_name_counts_semantics_recomputes_signature_counts():
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name="dummy_name_count_semantics_switch",
        mode="inference",
        load_name_counts=_name_count_tables(),
        preprocess=True,
        n_jobs=1,
    )
    baseline = dataset.signatures["1"].author_info_name_counts
    assert baseline is not None
    assert baseline.last_first_initial == 13
    assert dataset.name_counts_last_first_initial_semantics == NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR

    changed = dataset.set_name_counts_last_first_initial_semantics(NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY)
    assert changed is True
    after = dataset.signatures["1"].author_info_name_counts
    assert after is not None
    assert after.last_first_initial == 41
    assert dataset.name_counts_last_first_initial_semantics == NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY


def test_inference_prediction_uses_initial_char_semantics_for_old_model():
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name="dummy_inference_name_count_semantics_gate",
        mode="inference",
        load_name_counts=_name_count_tables(),
        preprocess=True,
        n_jobs=1,
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


def test_train_prediction_uses_initial_char_model_semantics_for_old_model():
    dataset = ANDData(
        "tests/dummy/signatures.json",
        "tests/dummy/papers.json",
        clusters="tests/dummy/clusters.json",
        name="dummy_train_name_count_semantics_gate",
        mode="train",
        load_name_counts=_name_count_tables(),
        preprocess=True,
        n_jobs=1,
        name_counts_last_first_initial_semantics=NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY,
    )
    clusterer = _build_clusterer(featurizer_version=1)

    before = dataset.signatures["1"].author_info_name_counts
    assert before is not None
    assert before.last_first_initial == 41

    clusterer.predict_helper({"block": ["1"]}, dataset, use_s2_clusters=True)

    after = dataset.signatures["1"].author_info_name_counts
    assert after is not None
    assert after.last_first_initial == 13
    assert dataset.name_counts_last_first_initial_semantics == NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR


def test_set_name_counts_semantics_logs_and_reraises_cache_evict_failure(monkeypatch, caplog):
    dataset = ANDData(
        signatures={},
        papers={},
        name="semantics_cache_evict_failure_dataset",
        mode="inference",
        load_name_counts=False,
        preprocess=False,
    )

    def _raise_runtime_error(_dataset):
        raise RuntimeError("eviction boom")

    monkeypatch.setattr(feature_port, "evict_rust_featurizer", _raise_runtime_error)

    with caplog.at_level("ERROR", logger="s2and"), pytest.raises(RuntimeError, match="eviction boom"):
        dataset.set_name_counts_last_first_initial_semantics(NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY)

    logs = "\n".join(caplog.messages)
    assert "Failed to evict Rust featurizer cache during name-count semantics refresh" in logs
    assert "semantics_cache_evict_failure_dataset" in logs
    assert "mode=inference" in logs
    assert dataset.runtime_context.run_id in logs

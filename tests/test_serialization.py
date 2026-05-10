import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import LabelEncoder

from s2and.serialization import load_pickle_with_verified_label_encoder_compat


class LegacyLabelEncoderCarrier:
    def __init__(
        self,
        classes: np.ndarray,
        *,
        encoded_classes: np.ndarray | None = None,
        estimator_name: str = "LabelEncoder",
    ) -> None:
        self._classes = classes
        self._le = LabelEncoder()
        self._le.classes_ = classes if encoded_classes is None else encoded_classes
        self._warning_estimator_name = estimator_name

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
        warnings.warn(
            InconsistentVersionWarning(
                estimator_name=self._warning_estimator_name,
                current_sklearn_version=sklearn_version,
                original_sklearn_version="0.23.2",
            ),
            stacklevel=2,
        )


class _DummyFeaturizerInfo:
    def __init__(self, featurizer_version: int):
        self.featurizer_version = int(featurizer_version)


class LegacyClustererWithoutFeatureContract:
    def __init__(self, featurizer_version: int):
        self.featurizer_info = _DummyFeaturizerInfo(featurizer_version)


class LegacyClustererWithFeatureContract:
    def __init__(self, featurizer_version: int, semantics: str):
        self.featurizer_info = _DummyFeaturizerInfo(featurizer_version)
        self.feature_contract = {"name_counts_last_first_initial_semantics": semantics}


def _dump_pickle(path: Path, obj) -> None:
    with path.open("wb") as pickle_file:
        pickle.dump(obj, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def test_load_pickle_suppresses_verified_label_encoder_warning(tmp_path):
    safe_object = {"clusterer": LegacyLabelEncoderCarrier(classes=np.array([0.0, 1.0]))}
    pickle_path = tmp_path / "safe_model.pkl"
    _dump_pickle(pickle_path, safe_object)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_pickle_with_verified_label_encoder_compat(pickle_path)

    inconsistent_warnings = [w for w in caught if isinstance(w.message, InconsistentVersionWarning)]
    assert len(inconsistent_warnings) == 0
    assert np.array_equal(loaded["clusterer"]._classes, loaded["clusterer"]._le.classes_)


def test_load_pickle_replays_warning_when_mapping_does_not_match(tmp_path):
    unsafe_object = {
        "clusterer": LegacyLabelEncoderCarrier(
            classes=np.array([0.0, 1.0]),
            encoded_classes=np.array([1.0, 2.0]),
        )
    }
    pickle_path = tmp_path / "unsafe_model.pkl"
    _dump_pickle(pickle_path, unsafe_object)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_pickle_with_verified_label_encoder_compat(pickle_path)

    inconsistent_warnings = [w for w in caught if isinstance(w.message, InconsistentVersionWarning)]
    assert len(inconsistent_warnings) == 1
    warning_message = inconsistent_warnings[0].message
    assert isinstance(warning_message, InconsistentVersionWarning)
    assert warning_message.estimator_name == "LabelEncoder"


def test_load_pickle_replays_non_label_encoder_inconsistent_warning(tmp_path):
    non_label_warning_object = {
        "clusterer": LegacyLabelEncoderCarrier(
            classes=np.array([0.0, 1.0]),
            estimator_name="RandomForestClassifier",
        )
    }
    pickle_path = tmp_path / "non_label_warning_model.pkl"
    _dump_pickle(pickle_path, non_label_warning_object)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_pickle_with_verified_label_encoder_compat(pickle_path)

    inconsistent_warnings = [w for w in caught if isinstance(w.message, InconsistentVersionWarning)]
    assert len(inconsistent_warnings) == 1
    warning_message = inconsistent_warnings[0].message
    assert isinstance(warning_message, InconsistentVersionWarning)
    assert warning_message.estimator_name == "RandomForestClassifier"


def test_load_pickle_attaches_initial_char_name_count_feature_contract_for_legacy_model(tmp_path):
    payload = {"clusterer": LegacyClustererWithoutFeatureContract(featurizer_version=1)}
    pickle_path = tmp_path / "legacy_clusterer.pkl"
    _dump_pickle(pickle_path, payload)

    loaded = load_pickle_with_verified_label_encoder_compat(pickle_path)
    contract = loaded["clusterer"].feature_contract
    assert contract["name_counts_last_first_initial_semantics"] == "initial_char"


def test_load_pickle_preserves_existing_name_count_feature_contract(tmp_path):
    payload = {
        "clusterer": LegacyClustererWithFeatureContract(
            featurizer_version=1,
            semantics="initial_char",
        )
    }
    pickle_path = tmp_path / "contract_clusterer.pkl"
    _dump_pickle(pickle_path, payload)

    loaded = load_pickle_with_verified_label_encoder_compat(pickle_path)
    contract = loaded["clusterer"].feature_contract
    assert contract["name_counts_last_first_initial_semantics"] == "initial_char"

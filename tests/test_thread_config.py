import pytest
from lightgbm import LGBMClassifier

from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.thread_config import resolve_n_jobs


def test_clusterer_n_jobs_propagates_to_lightgbm() -> None:
    classifier = LGBMClassifier(verbosity=-1)
    nameless_classifier = LGBMClassifier(verbosity=-1)
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=classifier,
        nameless_classifier=nameless_classifier,
        n_jobs=2,
        use_default_constraints_as_supervision=False,
    )

    assert clusterer.n_jobs == 2
    assert clusterer.classifier is not None
    assert clusterer.nameless_classifier is not None
    assert clusterer.classifier.get_params().get("n_jobs") == 2
    assert clusterer.nameless_classifier.get_params().get("n_jobs") == 2

    clusterer.n_jobs = 7
    assert clusterer.n_jobs == 7
    assert clusterer.classifier.get_params().get("n_jobs") == 7
    assert clusterer.nameless_classifier.get_params().get("n_jobs") == 7


def test_clusterer_n_jobs_minus_one_uses_all_cores(monkeypatch) -> None:
    monkeypatch.setattr("s2and.thread_config.os.cpu_count", lambda: 6)
    classifier = LGBMClassifier(verbosity=-1)
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=classifier,
        n_jobs=-1,
        use_default_constraints_as_supervision=False,
    )

    assert resolve_n_jobs(-1) == 6
    assert clusterer.n_jobs == 6
    assert clusterer.classifier.get_params().get("n_jobs") == 6


def test_resolve_n_jobs_handles_defaults_offsets_and_invalid_values(monkeypatch) -> None:
    monkeypatch.setattr("s2and.thread_config.os.cpu_count", lambda: 6)

    assert resolve_n_jobs(None) == 1
    assert resolve_n_jobs(-2) == 5

    with pytest.raises(ValueError, match="must not be zero"):
        resolve_n_jobs(0)

    for invalid in (True, "2"):
        with pytest.raises(TypeError, match="must be an int or None"):
            resolve_n_jobs(invalid)  # type: ignore[arg-type]

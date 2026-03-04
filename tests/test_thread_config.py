import lightgbm as lgb

from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer


def test_clusterer_n_jobs_propagates_to_lightgbm() -> None:
    classifier = lgb.LGBMClassifier(verbosity=-1)
    nameless_classifier = lgb.LGBMClassifier(verbosity=-1)
    clusterer = Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=classifier,
        nameless_classifier=nameless_classifier,
        n_jobs=2,
        use_cache=False,
        use_default_constraints_as_supervision=False,
    )

    assert clusterer.n_jobs == 2
    assert clusterer.classifier.get_params().get("n_jobs") == 2
    assert clusterer.nameless_classifier.get_params().get("n_jobs") == 2

    clusterer.n_jobs = 7
    assert clusterer.n_jobs == 7
    assert clusterer.classifier.get_params().get("n_jobs") == 7
    assert clusterer.nameless_classifier.get_params().get("n_jobs") == 7

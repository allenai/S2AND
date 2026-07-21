from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from fastcluster import linkage
from hyperopt import Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from lightgbm import Booster, LGBMClassifier
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import roc_auc_score


def lightgbm_booster(model: Any) -> Booster:
    """Return the fitted booster behind an S2AND or LightGBM model wrapper."""

    if isinstance(model, Booster):
        return model
    inner = getattr(model, "classifier", None)
    if inner is not None and inner is not model:
        return lightgbm_booster(inner)
    for attribute in ("booster_", "_Booster"):
        booster = getattr(model, attribute, None)
        if isinstance(booster, Booster):
            return booster
    raise TypeError(f"Expected a fitted LightGBM model, got {type(model)!r}")


def _validated_classifier_features(
    classifier: Any,
    features: Any,
    *,
    feature_names: Sequence[str] | None = None,
) -> Any:
    """Return classifier input after validating its fitted feature schema."""

    values = np.asarray(features)
    if values.ndim != 2:
        raise ValueError(f"Classifier features must be 2D, got shape={values.shape}")

    expected_count = getattr(classifier, "n_features_in_", None)
    if expected_count is not None and int(expected_count) != values.shape[1]:
        raise ValueError(
            f"Classifier feature count does not match fitted schema: {values.shape[1]} != {int(expected_count)}"
        )

    supplied_names = tuple(str(name) for name in feature_names) if feature_names is not None else None
    if supplied_names is not None and len(supplied_names) != values.shape[1]:
        raise ValueError(
            f"Classifier feature names do not match matrix width: {len(supplied_names)} != {values.shape[1]}"
        )

    fitted_names_raw = getattr(classifier, "feature_names_in_", None)
    if fitted_names_raw is None:
        return values
    fitted_names = tuple(str(name) for name in fitted_names_raw)
    actual_names = supplied_names
    if actual_names is None:
        columns = getattr(features, "columns", None)
        if columns is not None:
            actual_names = tuple(str(name) for name in columns)
    if actual_names is not None and actual_names != fitted_names:
        raise ValueError(f"Classifier feature names do not match fitted schema: {actual_names!r} != {fitted_names!r}")
    if getattr(features, "columns", None) is not None:
        return features

    import pandas as pd

    return pd.DataFrame(values, columns=pd.Index(fitted_names), copy=False)


def predict_pairwise_class0(classifier: Any, features: np.ndarray) -> np.ndarray:
    """Predict class-0 probabilities with native positive-probability fast path support."""

    predict_proba_positive = getattr(classifier, "predict_proba_positive", None)
    raw_features = np.asarray(features)
    features_2d = np.asarray(
        raw_features,
        dtype=np.float32 if callable(predict_proba_positive) and raw_features.dtype == np.float32 else np.float64,
        order="C",
    )
    if features_2d.size == 0:
        return np.asarray([], dtype=np.float64)

    if callable(predict_proba_positive):
        return 1.0 - np.asarray(predict_proba_positive(features_2d), dtype=np.float64).reshape(-1)

    probabilities = classifier.predict_proba(_validated_classifier_features(classifier, features_2d))
    return np.asarray(probabilities, dtype=np.float64)[:, 0]


class PairwiseModeler:
    """
    Wrapper to learn the pairwise model + hyperparameter optimization

    Parameters
    ----------
    estimator: sklearn compatible classifier
        A binary classifier with fit/predict interface.
        Defaults to LGBMClassifier if not specified. Will be cloned.
    search_space: Dict:
            A hyperopt search space for hyperparam optimization.
            Defaults to an appropriate LGBMClassifier space if not specified.
    monotone_constraints: string
            Monotonic constraints for lightbm only.
            Defaults to None and is not used.
    n_iter: int
        Number of iterations for hyperparam optimization.
    n_jobs: int
        Parallelization for the classifier.
        Note: the hyperopt is serial, but can be made semi-parallel with batch search.
    random_state: int
        Random state for classifier and hyperopt.
    """

    def __init__(
        self,
        estimator: Any | None = None,
        search_space: dict[str, Any] | None = None,
        monotone_constraints: str | None = None,
        n_iter: int = 50,
        n_jobs: int = 16,  # for the model, not the hyperopt
        random_state: int = 42,
    ):
        if estimator is None:
            self.estimator = LGBMClassifier(
                objective="binary",
                metric="auc",  # lightgbm doesn't do F1 directly
                n_jobs=n_jobs,
                verbose=-1,
                tree_learner="data",
                random_state=random_state,
            )
        else:
            self.estimator = clone(estimator)

        if search_space is None:
            self.search_space = {
                "learning_rate": hp.loguniform("learning_rate", -7, 0),
                "num_leaves": scope.int(hp.qloguniform("num_leaves", 2, 7, 1)),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
                "subsample": hp.uniform("subsample", 0.5, 1),
                "min_child_samples": scope.int(hp.qloguniform("min_child_samples", 3, 9, 1)),
                "min_child_weight": hp.loguniform("min_child_weight", -16, 5),
                "reg_alpha": hp.loguniform("reg_alpha", -16, 2),
                "reg_lambda": hp.loguniform("reg_lambda", -16, 2),
                "n_estimators": scope.int(hp.quniform("n_estimators", 1000, 2500, 1)),
                "max_depth": scope.int(hp.quniform("max_depth", 1, 100, 1)),
                "min_split_gain": hp.uniform("min_split_gain", 0, 2),
            }
        else:
            self.search_space = search_space

        self.monotone_constraints = monotone_constraints
        if self.monotone_constraints is not None and isinstance(self.estimator, LGBMClassifier):
            self.estimator.set_params(monotone_constraints=self.monotone_constraints)
            self.estimator.set_params(monotone_constraints_method="advanced")
            self.search_space["monotone_penalty"] = hp.uniform("monotone_penalty", 0, 5)

        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params: dict | None = None
        self.hyperopt_trials_store: Trials | dict[Any, Any] | None = None
        self.classifier: Any | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Trials | dict[Any, Any]:
        """
        Fits the classifier

        Parameters
        ----------
        X_train: np.ndarray
            feature matrix for the training set
        y_train: np.ndarray
            labels for the training set
        X_val: np.ndarray
            feature matrix for the validation set
        y_val: np.ndarray
            labels for the validation set

        Returns
        -------
        Trials: the Trials object from hyperparameter optimization
        """
        if len(self.search_space) > 0:

            def obj(params):
                params = {k: intify(v) for k, v in params.items()}
                self.estimator.set_params(**params)
                self.estimator.fit(X_train, y_train)
                y_pred_proba = np.asarray(self.estimator.predict_proba(X_val), dtype=np.float64)[:, 1]
                return -roc_auc_score(y_val, y_pred_proba)

            self.hyperopt_trials_store = Trials()
            _ = fmin(
                fn=obj,
                space=self.search_space,
                algo=tpe.suggest,
                max_evals=self.n_iter,
                trials=self.hyperopt_trials_store,
                rstate=np.random.default_rng(self.random_state),
            )
            assert isinstance(self.hyperopt_trials_store, Trials)
            best_params = space_eval(self.search_space, self.hyperopt_trials_store.argmin)
            self.best_params = {k: intify(v) for k, v in best_params.items()}
            self.estimator.set_params(**self.best_params)
        else:
            self.best_params = {}
            self.hyperopt_trials_store = {}

        # refitting but only on training data so as not to leak anything
        self.classifier = self.estimator.fit(X_train, y_train)

        assert self.hyperopt_trials_store is not None
        return self.hyperopt_trials_store

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.classifier is not None, "You need to call fit first"
        return self.classifier.predict_proba(_validated_classifier_features(self.classifier, X))


def intify(x):
    """Hyperopt is bad at ints..."""
    if hasattr(x, "is_integer") and x.is_integer():
        return int(x)
    else:
        return x


class FastCluster(TransformerMixin, BaseEstimator):
    """
    A scikit-learn wrapper for fastcluster.
    Inputs:
        linkage: string (default="average")
            Agglomerative linkage method. Defaults to "average".
            Must be one of "'complete', 'average', 'single,
            'weighted', 'ward', 'centroid', 'median'."
        eps: float (default=0.5)
            Cutoff used to determine number of clusters.
        preserve_input: bool (default=True)
            Whether to preserve the X input or modify in place.
            Defaults to False, which modifies in place.
        input_as_observation_matrix: bool (default=False)
            If True, the input to fit/transform must be a 2-D array
            of observation vectors (N by d). If False input to fit/transform
            must be a 1-D condensed distance matrix, then it must be a
            (N choose 2) sized vector, where N is the number
            of original observations paired in the distance matrix, and
            d is the dimensionality of the vector space.

    Note: FastCluster does *not* support two-dimensional distance matrices
    as input. They *must* be flattened. For more details, please see:
    https://cran.r-project.org/web/packages/fastcluster/vignettes/fastcluster.pdf
    """

    def __init__(
        self,
        linkage: str = "average",
        eps: float = 0.5,
        preserve_input: bool = True,
        input_as_observation_matrix: bool = False,
    ):
        if linkage not in {
            "complete",
            "average",
            "weighted",
            "ward",
            "centroid",
            "median",
            "single",
        }:
            raise ValueError(
                "The 'linkage' parameter has to be one of: "
                + "'single', complete', 'average', 'weighted', 'ward', 'centroid', 'median'."
            )

        self.linkage = linkage
        self.eps = eps
        self.preserve_input = preserve_input
        self.input_as_observation_matrix = input_as_observation_matrix
        self.labels_ = None

    def fit(self, X: np.ndarray) -> FastCluster:
        """
        Fit the estimator on input data. The results are stored in self.labels_.
        Parameters
        ----------
        X: np.array
            The input may be either a 1-D condensed distance matrix
            or a 2-D array of observation vectors. If X is a 1-D condensed distance
            matrix, then it must be (N choose 2) sized vector, where N is the number
            of original observations paired in the distance matrix. If X is 2-D
            then the flag `input_as_observation_matrix` must be set to True in init.
        Returns
        -------
        self
        """
        X = np.asarray(X)
        if len(X.shape) == 1 and self.input_as_observation_matrix:
            raise ValueError(
                "Input to fit is one-dimensional, but input_as_observation_matrix flag is set to True. "
                "If you intended to pass in an observation matrix, it must be 2-D (N x feature_dimension)."
            )
        elif len(X.shape) == 2 and not self.input_as_observation_matrix:
            raise ValueError(
                "Input to fit is two-dimensional, but input_as_observation_matrix flag is set to False. "
                "If you intended to pass in a distance matrix, it must be flattened (1-D)."
            )
        elif len(X.shape) > 2:
            raise ValueError("The input to fit can only be one-dimensional or two-dimensional.")
        Z = linkage(X, self.linkage, preserve_input=self.preserve_input)
        self.labels_ = fcluster(Z, t=self.eps, criterion="distance")
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params: Any,
    ) -> np.ndarray:
        """
        Fit the estimator on input data, and returns results.
        Parameters
        ----------
        X: np.array
            The input may be either a 1-D condensed distance matrix
            or a 2-D array of observation vectors. If X is a 1-D condensed distance
            matrix, then it must be (N choose 2) sized vector, where N is the number
            of original observations paired in the distance matrix.
        Returns
        -------
        np.array: A N-length array of clustering labels.
        """
        del y, fit_params
        self.fit(X)
        return self.labels_  # type: ignore

    def transform(self, X: np.ndarray):
        raise NotImplementedError("FastCluster has no inductive mode. Use 'fit' or 'fit_transform' instead.")

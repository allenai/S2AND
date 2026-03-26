from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
from fastcluster import linkage
from hyperopt import Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from lightgbm import LGBMClassifier
from scipy.cluster.hierarchy import fcluster
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import roc_auc_score

from s2and.warnings_utils import suppress_sklearn_feature_name_warnings


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
        X_train: np.ndarray[Any, Any] | None | Any,
        y_train: np.ndarray[Any, Any] | None | Any,
        X_val: np.ndarray[Any, Any] | None | Any,
        y_val: np.ndarray[Any, Any] | None | Any,
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
                y_pred_proba = self.estimator.predict_proba(X_val)[:, 1]
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
        with warnings.catch_warnings():
            suppress_sklearn_feature_name_warnings()
            return self.classifier.predict_proba(X)


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

    # ---- new: robust get_params ----
    def get_params(self, deep=True):
        """
        Return params but gracefully handle the case where an instance
        (e.g., loaded from an old pickle) is missing attributes.
        """
        params = {}
        sig = inspect.signature(self.__class__.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # prefer the runtime attribute if present, otherwise the __init__ default
            if hasattr(self, name):
                params[name] = getattr(self, name)
            else:
                params[name] = param.default if param.default is not inspect.Parameter.empty else None

        if deep:
            # sklearn convention: include nested estimator params with __ separator
            for key, val in list(params.items()):
                if hasattr(val, "get_params"):
                    for subk, subv in val.get_params(deep=True).items():
                        params[f"{key}__{subk}"] = subv
        return params

    # ---- new: ensure defaults after unpickling ----
    def __setstate__(self, state):
        """
        Called on unpickle. Populate any missing ctor attrs with their defaults.
        """
        self.__dict__.update(state)
        sig = inspect.signature(self.__class__.__init__)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if not hasattr(self, name):
                default = param.default if param.default is not inspect.Parameter.empty else None
                setattr(self, name, default)

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


class VotingClassifier:
    """
    Stripped-down version of VotingClassifier that uses prefit estimators

    Parameters
    ----------
    estimators: List[sklearn classifier]
        A list of sklearn classifiers that support predict_proba.
    voting: string
        Type of voting.
        Defaults to "hard", can also be "soft".
        "soft" means "take the highest average probability class" and
        "hard" means "take the class that the plurality of the models pick"
    weights: List or np.array
        Weights for each estimator.
    """

    def __init__(self, estimators, voting="soft", weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        predictions : array-like, shape = [n_samples]
            Predicted class labels.
        """
        if self.voting == "soft":
            predictions = np.argmax(self.predict_proba(X), axis=1)
        elif self.voting == "hard":
            predictions = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=self._predict(X).astype("int"),
            )
        else:
            raise ValueError("Voting type must be one of 'soft' or 'hard'")
        return predictions

    def _collect_probas(self, X):
        """Collect results from clf.predict calls."""
        with warnings.catch_warnings():
            suppress_sklearn_feature_name_warnings()
            return np.asarray([clf.predict_proba(X) for clf in self.estimators])

    def predict_proba(self, X):
        """
        Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        if self.voting == "hard":
            raise AttributeError(f"predict_proba is not available when voting={self.voting!r}")
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        """
        Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        if self.voting == "soft":
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        with warnings.catch_warnings():
            suppress_sklearn_feature_name_warnings()
            return np.asarray([clf.predict(X) for clf in self.estimators]).T

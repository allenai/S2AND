from __future__ import annotations

import inspect
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster

from s2and.consts import HARD_LINKAGE_DISTANCE, LARGE_DISTANCE

# Prediction needs none of scikit-learn, LightGBM, or pandas. Training code imports them where it uses
# them, because together they add about 1.5 s to every new prediction process.
if TYPE_CHECKING:
    from hyperopt import Trials
    from lightgbm import Booster

# Giant blocks can hold hard negatives in most pairs; scan in bounded chunks so
# the temporary mask stays small, not one byte or index per pair.
_HARD_NEGATIVE_SCAN_CHUNK = 1 << 22


def _hard_negative_chunks(distances: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield each condensed chunk with its exact hard-negative sentinel mask."""
    for start in range(0, len(distances), _HARD_NEGATIVE_SCAN_CHUNK):
        chunk = distances[start : start + _HARD_NEGATIVE_SCAN_CHUNK]
        yield chunk, chunk == LARGE_DISTANCE


def constrained_linkage(distances: np.ndarray, method: str, *, preserve_input: bool) -> np.ndarray:
    """Build a linkage tree in which hard negatives never share a cluster below EPS.

    Every pair at exactly ``LARGE_DISTANCE`` is a hard negative. Linkage gets
    ``HARD_LINKAGE_DISTANCE`` for these pairs, because average linkage can
    dilute a finite sentinel below EPS inside a large merge. Other distances,
    including soft distances above ``LARGE_DISTANCE``, pass through unchanged.

    Args:
        distances: Condensed distance matrix.
        method: fastcluster linkage method.
        preserve_input: Keep the caller's array unchanged. When False, the
            caller gives up the array, and a writable float64 array is
            modified in place.

    Returns:
        The fastcluster linkage matrix.
    """
    condensed = np.asarray(distances, dtype=np.float64)
    if not any(mask.any() for _, mask in _hard_negative_chunks(condensed)):
        return linkage(condensed, method, preserve_input=preserve_input)
    shared = isinstance(distances, np.ndarray) and np.may_share_memory(condensed, distances)
    if shared and (preserve_input or not condensed.flags.writeable):
        condensed = condensed.copy()
    for chunk, mask in _hard_negative_chunks(condensed):
        chunk[mask] = HARD_LINKAGE_DISTANCE
    # condensed is now owned by this call, or released by the caller.
    return linkage(condensed, method, preserve_input=False)


def lightgbm_booster(model: Any) -> Booster:
    """Return the fitted booster behind an S2AND or LightGBM model wrapper."""

    from lightgbm import Booster

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


def native_pairwise_scorers(classifier: Any, nameless_classifier: Any | None) -> tuple[Any, Any | None] | None:
    """Return the Rust boosters of native classifiers, for fused featurize-and-score passes.

    Fused scoring reproduces `predict_pairwise_class0` for native boosters only;
    other classifiers keep the feature-matrix path. Returns None when either
    classifier is not native.
    """

    if str(getattr(classifier, "prediction_backend", "python")) != "rust_lightgbm":
        return None
    if nameless_classifier is None:
        return classifier._rust_scorer, None
    if str(getattr(nameless_classifier, "prediction_backend", "python")) != "rust_lightgbm":
        return None
    return classifier._rust_scorer, nameless_classifier._rust_scorer


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
        from hyperopt import hp
        from hyperopt.pyll import scope
        from lightgbm import LGBMClassifier
        from sklearn.base import clone

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
        *,
        feature_names: Sequence[str] | None = None,
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
        feature_names: optional sequence of str
            column names for the final fit of a LightGBM estimator. Its saved
            booster then names features instead of using Column_N; the trees
            are unchanged.

        Returns
        -------
        Trials: the Trials object from hyperparameter optimization
        """
        if len(self.search_space) > 0:
            from hyperopt import Trials, fmin, space_eval, tpe
            from sklearn.metrics import roc_auc_score

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
        # Only this fit is kept, so only it needs names; the trials score unnamed arrays.
        fit_kwargs = {} if feature_names is None else {"feature_name": [str(name) for name in feature_names]}
        self.classifier = self.estimator.fit(X_train, y_train, **fit_kwargs)

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


class FastCluster:
    """
    A scikit-learn compatible wrapper for fastcluster.

    It implements the parameter API that ``sklearn.base.clone`` uses, without
    importing scikit-learn, so prediction processes start faster.
    Inputs:
        linkage: string (default="average")
            Agglomerative linkage method. Defaults to "average".
            Must be one of "'complete', 'average', 'single,
            'weighted', 'ward', 'centroid', 'median'."
        eps: float (default=0.5)
            Cutoff used to determine number of clusters.
        preserve_input: bool (default=True)
            Whether to preserve the X input or modify in place.
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

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Return the constructor parameter names, sorted as scikit-learn sorts them."""
        parameters = inspect.signature(cls.__init__).parameters.values()
        return sorted(parameter.name for parameter in parameters if parameter.name != "self")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the constructor parameters, as ``sklearn.base.BaseEstimator.get_params`` does.

        Args:
            deep: Accepted for scikit-learn compatibility; no parameter is an estimator.
        """
        del deep
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params: Any) -> FastCluster:
        """Set constructor parameters, as ``sklearn.base.BaseEstimator.set_params`` does.

        Raises:
            ValueError: A name is not a constructor parameter.
        """
        valid = self._get_param_names()
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for estimator {self}. Valid parameters are: {valid!r}.")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        defaults = {
            parameter.name: parameter.default
            for parameter in inspect.signature(type(self).__init__).parameters.values()
            if parameter.name != "self"
        }
        changed = ", ".join(
            f"{name}={value!r}" for name, value in self.get_params().items() if value != defaults.get(name)
        )
        return f"{type(self).__name__}({changed})"

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
        self.labels_ = fcluster(self.linkage_matrix(X), t=self.eps, criterion="distance")
        return self

    def linkage_matrix(self, X: np.ndarray) -> np.ndarray:
        """Return the linkage of `X` that `fit` cuts at `eps`.

        Parameters
        ----------
        X: np.array
            The input that `fit` takes.

        Returns
        -------
        np.ndarray: the SciPy linkage matrix
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
        if self.input_as_observation_matrix:
            return linkage(X, self.linkage, preserve_input=self.preserve_input)
        return constrained_linkage(X, self.linkage, preserve_input=self.preserve_input)

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


def clone_cluster_model(cluster_model: Any) -> Any:
    """Return an unfitted copy of a clustering estimator with the same parameters.

    A FastCluster copies itself, so prediction does not import scikit-learn.
    Other estimators use ``sklearn.base.clone``.
    """
    if isinstance(cluster_model, FastCluster):
        return type(cluster_model)(**cluster_model.get_params())
    from sklearn.base import clone

    return clone(cluster_model)

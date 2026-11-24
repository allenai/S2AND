"""Configure clustering and dispatch public prediction requests."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from s2and.arrow_inputs import ArrowDataset
from s2and.corrections.name_conflict_repair import DEFAULT_MAX_AVERAGE_DISTANCE
from s2and.corrections.profile_exclusions import (
    evict_profile_exclusion_violations,
    normalize_profile_exclusions,
    validate_profile_exclusions,
)
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block import normalize_cluster_seed_disallow_pairs
from s2and.model_pairwise import FastCluster, clone_cluster_model, intify
from s2and.prediction_state import PredictionState
from s2and.runtime import RuntimeContext, build_runtime_context
from s2and.subblocking import GraphSubblockingConfig
from s2and.supervision import normalize_partial_supervision, validate_seed_group_separations
from s2and.thread_config import resolve_n_jobs

from . import bulk, classic, constraints, distances, incremental
from ._shared import _clusterer_with_prediction_params, _gc_paused, logger
from .constraints import (
    DEFAULT_ORCID_NAME_WAIVER,
    _cluster_seeds_require_from_rust_featurizer,
    _normalize_cluster_seeds_require,
    _RustFeaturizerPredictDataset,
)

if TYPE_CHECKING:
    from hyperopt import Trials


def _propagate_n_jobs(estimator: Any, n_jobs: int) -> None:
    """Best-effort propagation of `n_jobs` into estimators/wrappers.

    Keeps S2AND's `Clusterer.n_jobs` as the single knob for both Rust `num_threads` and
    Python model inference thread pools (LightGBM/OpenMP, sklearn estimators, etc.).
    """
    if estimator is None:
        return

    inner = getattr(estimator, "classifier", None)
    if inner is not None and inner is not estimator:
        _propagate_n_jobs(inner, n_jobs)

    for attr in ("estimators", "estimators_"):
        children = getattr(estimator, attr, None)
        if not isinstance(children, list | tuple):
            continue
        for child in children:
            if isinstance(child, tuple) and len(child) == 2:
                _propagate_n_jobs(child[1], n_jobs)
            else:
                _propagate_n_jobs(child, n_jobs)

    resolved_n_jobs = resolve_n_jobs(n_jobs)
    set_params = getattr(estimator, "set_params", None)
    get_params = getattr(estimator, "get_params", None)
    # Serial estimators expose set_params but reject n_jobs, including fitted forest trees.
    if callable(set_params) and (not callable(get_params) or "n_jobs" in get_params(deep=False)):
        try:
            set_params(n_jobs=resolved_n_jobs)
        except TypeError as exc:
            logger.debug(
                "Skipping set_params n_jobs propagation for estimator=%s: %s",
                type(estimator).__name__,
                exc,
            )
        except Exception:
            logger.debug(
                "Unexpected error while propagating n_jobs via set_params for estimator=%s",
                type(estimator).__name__,
                exc_info=True,
            )
            raise

    if hasattr(estimator, "n_jobs"):
        try:
            estimator.n_jobs = resolved_n_jobs
        except (AttributeError, TypeError) as exc:
            logger.debug(
                "Skipping n_jobs attribute propagation for estimator=%s: %s",
                type(estimator).__name__,
                exc,
            )
        except Exception:
            logger.debug(
                "Unexpected error while propagating n_jobs via attribute assignment for estimator=%s",
                type(estimator).__name__,
                exc_info=True,
            )
            raise


class Clusterer:
    """
    A wrapper for learning a clusterer

    Args:
        featurizer_info: FeaturizationInfo
            Featurization information
        classifier: sklearn compatible model
            Classifier which uses pairwise features to make a distance matrix
        val_blocks_size: int
            How many blocks to use during hyperparam optimization.
            Defaults to None, which uses all of them.
        cluster_model: sklearn compatible model
            Clusterer model
            Defaults to None, which uses FastCluster with average linking.
        search_space: Dict
            Search space for the hyperpamater optimization.
            Defaults to None; fit creates the FastCluster calibration space.
        n_iter: int
            Number of hyperparameter evaluations
        n_jobs: int
            Parallelize each clusterer this many ways
        use_default_constraints_as_supervision: bool
            Whether to use the default constraints when constructing the distance matrices.
            These are high precision and can save a lot of compute/time.
        random_state: int
            Random state
        nameless_classifier: sklearn compatible model
            A second classifier which uses pairwise features excluding all name information, and
            whose predictions are averaged with the main classifier. Won't be used if None
        nameless_featurizer_info: FeaturizationInfo
            The FeaturizationInfo for the second classifier. Won't be used if None
        dont_merge_cluster_seeds: bool
            whether to enforce "disallow" constraints for signatures in different required seed clusters
        batch_size: int
            batch size for featurization, lower means less memory, but slower
        suppress_orcid: bool
            Whether default constraint resolution should ignore same-ORCID must-link constraints.
    """

    __module__ = "s2and.model"

    # Average model distance below which name-conflict repair merges two
    # clusters; None turns the repair off. A class attribute, so models loaded
    # from older pickles get it too. See s2and.name_conflict_repair.
    name_conflict_repair_max_distance: float | None = DEFAULT_MAX_AVERAGE_DISTANCE

    # Whether a block's clusters are split, after name-conflict repair, at empty stretches of
    # more than YEAR_DIFF_CAP years that no close pair, ORCID or seed group bridges. It needs
    # the cluster model's close-pair distance and is independent of the repair. A class
    # attribute, like the repair distance. See s2and.year_gap_split and s2and.block_corrections.
    year_gap_split: bool = True

    # Join same-ORCID clusters with enough evidence after bulk linkage.
    # Previously exported model bundles use this package default.
    orcid_join: bool = True

    # Block distances drop a name conflict that an ORCID's own mentions disprove.
    # A mention with an ORCID stops forbidding a mention without one when its name
    # conflicts with other mentions of its ORCID, the name without an ORCID does not
    # conflict with at least two of those, and it conflicts with at most the second
    # value's share of that ORCID's mentions with a full first name, of which the
    # ORCID has at least the first value. The pair then gets its model distance.
    # None turns this off. A class attribute, like the repair distance.
    orcid_name_waiver: tuple[int, float] | None = DEFAULT_ORCID_NAME_WAIVER

    def __init__(
        self,
        featurizer_info: FeaturizationInfo,
        classifier: Any,
        val_blocks_size: int | None = None,
        cluster_model: Any | None = None,
        search_space: dict[str, Any] | None = None,
        n_iter: int = 25,
        n_jobs: int = 16,
        *,
        use_default_constraints_as_supervision: bool = True,
        random_state: int = 42,
        nameless_classifier: Any | None = None,
        nameless_featurizer_info: FeaturizationInfo | None = None,
        dont_merge_cluster_seeds: bool = True,
        batch_size: int = 1000000,
        suppress_orcid: bool = False,
    ):
        self.featurizer_info = featurizer_info
        self.nameless_featurizer_info = nameless_featurizer_info
        self.classifier = copy.deepcopy(classifier)
        self.nameless_classifier = copy.deepcopy(nameless_classifier)
        self.val_blocks_size = val_blocks_size
        self.n_iter = n_iter
        self._n_jobs = 1
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_default_constraints_as_supervision = use_default_constraints_as_supervision
        self.dont_merge_cluster_seeds = dont_merge_cluster_seeds
        self.suppress_orcid = suppress_orcid
        if cluster_model is None:
            self.cluster_model = FastCluster(linkage="average")
        else:
            self.cluster_model = copy.deepcopy(cluster_model)

        self.search_space = search_space

        self.feature_contract: dict[str, str] = {}
        self.hyperopt_trials_store: Trials | list[Trials] | None = None
        self.best_params: dict[Any, Any] | None = None
        self.batch_size = batch_size
        self.incremental_linker_artifact_dir: Path | None = None
        self.incremental_linker_artifact: Any | None = None
        self.production_model_bundle_dir: Path | None = None
        self.production_model_release_version: str | None = None
        self.subblocking_graph_config = GraphSubblockingConfig()

    @property
    def n_jobs(self) -> int:
        return int(getattr(self, "_n_jobs", 1))

    @n_jobs.setter
    def n_jobs(self, value: int) -> None:
        n_jobs = resolve_n_jobs(value)
        self._n_jobs = n_jobs
        _propagate_n_jobs(getattr(self, "classifier", None), n_jobs)
        _propagate_n_jobs(getattr(self, "nameless_classifier", None), n_jobs)

    @staticmethod
    def filter_blocks(block_dict: dict[str, list[str]], num_to_keep: int | None = None) -> dict[str, list[str]]:
        """
        Filter out blocks of size 1, as they are not useful or train/val

        Parameters
        ----------
        block_dict: Dict
            the block dictionary
        num_to_keep: int
            the number of blocks to keep, keeps all if None

        Returns
        -------
        either the loaded json, or the passed in object
        """
        out_dict = {}
        count = 0
        for block_key, signatures in block_dict.items():
            if len(signatures) > 1:
                out_dict[block_key] = signatures
                count += 1
                if num_to_keep is not None and count == num_to_keep:
                    return out_dict
        return out_dict

    _resolve_constraint_batch = constraints._resolve_constraint_batch

    _flush_completed_block = distances._flush_completed_block

    distance_matrix_helper = distances.distance_matrix_helper

    _yield_non_fused_chunks = distances._yield_non_fused_chunks

    _distance_matrix_chunk_helper_rust = distances._distance_matrix_chunk_helper_rust

    _predict_distance_matrix_chunk = distances._predict_distance_matrix_chunk

    _iter_rust_predicted_distance_matrix_chunks = distances._iter_rust_predicted_distance_matrix_chunks

    _iter_python_predicted_distance_matrix_batches = distances._iter_python_predicted_distance_matrix_batches

    _featurize_predict_write_batches = distances._featurize_predict_write_batches

    make_distance_matrices = distances.make_distance_matrices

    make_distance_matrices_from_rust_featurizer = distances.make_distance_matrices_from_rust_featurizer

    _make_distance_matrices_from_verified_rust_featurizer = (
        distances._make_distance_matrices_from_verified_rust_featurizer
    )

    def predict_from_rust_featurizer(
        self,
        block_dict: dict[str, list[str]],
        rust_featurizer: object,
        dists: dict[str, np.ndarray] | None = None,
        cluster_model_params: dict[str, Any] | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
        cluster_seeds_require: Mapping[str, int | str] | None = None,
        cluster_seeds_disallow: set[tuple[str, str]] | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
        """Predict full blocks from an already-built Rust featurizer.

        Explicit seed arguments on this lower-level bridge are distance
        overrides and remain active when default constraints are disabled.
        """

        if dists is None and self.featurizer_info.legacy_paper_years:
            rust_featurizer = cast(Any, rust_featurizer).with_legacy_paper_years()
        partial_supervision = normalize_partial_supervision(partial_supervision)
        if (
            partial_supervision
            and cluster_seeds_require
            and not incremental_dont_use_cluster_seeds
            and getattr(self, "dont_merge_cluster_seeds", True)
        ):
            validate_seed_group_separations(
                cluster_seeds_require,
                partial_supervision,
                {str(signature): key for key, signatures in block_dict.items() for signature in signatures},
            )
        return Clusterer._predict_from_rust_featurizer(
            self,
            block_dict=block_dict,
            rust_featurizer=rust_featurizer,
            dists=dists,
            cluster_model_params=cluster_model_params,
            partial_supervision=partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            cluster_seeds_require=cluster_seeds_require,
            cluster_seeds_disallow=cluster_seeds_disallow,
        )

    def _predict_from_rust_featurizer(
        self,
        block_dict: dict[str, list[str]],
        rust_featurizer: object,
        dists: dict[str, np.ndarray] | None = None,
        cluster_model_params: dict[str, Any] | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
        cluster_seeds_require: Mapping[str, int | str] | None = None,
        cluster_seeds_disallow: set[tuple[str, str]] | None = None,
        prediction_state: PredictionState | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
        """Predict full blocks from an already-built Rust featurizer."""

        if any(len(signatures) > 1 for signatures in block_dict.values()):
            self = _clusterer_with_prediction_params(self, cluster_model_params)
        if prediction_state is None:
            prediction_state = PredictionState()

        if runtime_context is None:
            runtime_context = build_runtime_context("cluster_predict_rust_featurizer", backend="rust")
        if partial_supervision is None:
            partial_supervision = {}
        resolved_cluster_seeds_require: dict[str, int | str]
        if cluster_seeds_require is None:
            resolved_cluster_seeds_require = _cluster_seeds_require_from_rust_featurizer(rust_featurizer)
        else:
            resolved_cluster_seeds_require = {
                str(signature_id): component_id for signature_id, component_id in cluster_seeds_require.items()
            }
        resolved_cluster_seeds_disallow = set(normalize_cluster_seed_disallow_pairs(cluster_seeds_disallow or ()))
        explicit_cluster_seeds_require = (
            None
            if cluster_seeds_require is None
            else {str(signature_id): component_id for signature_id, component_id in cluster_seeds_require.items()}
        )
        if dists is not None and resolved_cluster_seeds_disallow:
            raise ValueError(
                "cluster_seeds_disallow cannot be used with precomputed dists because disallow pairs "
                "would not be injected into the distance matrix"
            )
        if dists is not None and resolved_cluster_seeds_require and not incremental_dont_use_cluster_seeds:
            raise ValueError(
                "cluster_seeds_require cannot be used with precomputed dists because require pairs "
                "would not be injected into the distance matrix"
            )
        effective_cluster_seeds_require = {} if incremental_dont_use_cluster_seeds else resolved_cluster_seeds_require
        proxy_dataset = _RustFeaturizerPredictDataset(
            cluster_seeds_require=effective_cluster_seeds_require,
            cluster_seeds_disallow=resolved_cluster_seeds_disallow,
            signatures={},
            rust_featurizer=rust_featurizer,
            repair_cluster_seeds_require=resolved_cluster_seeds_require,
        )
        return Clusterer._predict_from_prepared_rust_featurizer(
            self,
            block_dict,
            rust_featurizer,
            proxy_dataset=proxy_dataset,
            dists=dists,
            cluster_model_params=cluster_model_params,
            partial_supervision=partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            explicit_cluster_seeds_require=explicit_cluster_seeds_require,
            resolved_cluster_seeds_disallow=resolved_cluster_seeds_disallow,
            prediction_state=prediction_state,
        )

    _predict_from_prepared_rust_featurizer = bulk._predict_from_prepared_rust_featurizer

    _build_arrow_subblocked_block_dict = bulk._build_arrow_subblocked_block_dict

    _predict_subblocked_single_letter_groups_from_arrow = bulk._predict_subblocked_single_letter_groups_from_arrow

    _predict_from_arrow = bulk._predict_from_arrow

    @_gc_paused
    def predict_from_arrow(
        self,
        block_dict: dict[str, list[str]],
        arrow_dataset: ArrowDataset,
        dists: dict[str, np.ndarray] | None = None,
        cluster_model_params: dict[str, Any] | None = None,
        partial_supervision: dict[tuple[str, str], int | float] | None = None,
        incremental_dont_use_cluster_seeds: bool = False,
        runtime_context: RuntimeContext | None = None,
        total_ram_bytes: int | None = None,
        batching_threshold: int | None = None,
        load_name_counts: bool | None = None,
        name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
        cluster_seeds_require: Mapping[Any, Any] | None = None,
        cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
        altered_cluster_signatures: Sequence[Any] | None = None,
        profile_exclusions: Iterable[tuple[Any, Any]] | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
        """Predict blocks through one open Arrow dataset.

        Args:
            block_dict: Mapping from block keys to signature IDs. Integer IDs
                are accepted and normalized to strings, including output members
                and caller supervision keys, on every Arrow bulk route.
            arrow_dataset: Open immutable Arrow dataset.
            batching_threshold: Maximum native subblock size. ``None`` keeps
                full-block prediction; a positive value routes oversized blocks
                through the Arrow-native Rust subblocker.
            cluster_seeds_require: Current request-local required seed components.
            altered_cluster_signatures: Accepted but ignored in bulk prediction.
                Temporary altered-profile splits apply only to incremental linking;
                bulk prediction uses the supplied required seed groups unchanged.
            profile_exclusions: ``(anchor_signature_id, excluded_signature_id)``
                pairs. The excluded mention never ends in the anchor's cluster;
                it is evicted from that cluster rather than vetoing its merges.

        Returns:
            Predicted clusters and optional precomputed distance matrices."""

        block_dict = {key: [str(signature_id) for signature_id in members] for key, members in block_dict.items()}
        partial_supervision = normalize_partial_supervision(partial_supervision, stringify_ids=True)
        # Attachment passes send large seed maps without caller distances; skip the copy then.
        if (
            partial_supervision
            and cluster_seeds_require is not None
            and not incremental_dont_use_cluster_seeds
            and getattr(self, "dont_merge_cluster_seeds", True)
        ):
            validate_seed_group_separations(
                _normalize_cluster_seeds_require(cluster_seeds_require),
                partial_supervision,
                {signature: key for key, signatures in block_dict.items() for signature in signatures},
            )
        exclusions = normalize_profile_exclusions(profile_exclusions)
        prediction_state: PredictionState | None = None
        if exclusions:
            if dists is not None:
                raise ValueError("profile_exclusions cannot be used with precomputed dists")
            validate_profile_exclusions(
                exclusions,
                {} if cluster_seeds_require is None else _normalize_cluster_seeds_require(cluster_seeds_require),
            )
            prediction_state = PredictionState(profile_exclusions=exclusions)
        clusters, matrices = Clusterer._predict_from_arrow_request(
            self,
            block_dict=block_dict,
            arrow_dataset=arrow_dataset,
            dists=dists,
            cluster_model_params=cluster_model_params,
            partial_supervision=partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            batching_threshold=batching_threshold,
            load_name_counts=load_name_counts,
            name_tuples=name_tuples,
            cluster_seeds_require=cluster_seeds_require,
            cluster_seeds_disallow=cluster_seeds_disallow,
            prediction_state=prediction_state,
        )
        if not exclusions:
            return clusters, matrices
        # Attachment and reunite steps run after block clustering; recheck the rule.
        clusters, _moved = evict_profile_exclusion_violations(
            clusters,
            exclusions,
            None if cluster_seeds_require is None else _normalize_cluster_seeds_require(cluster_seeds_require),
        )
        return clusters, matrices

    _predict_from_arrow_request = bulk._predict_from_arrow_request

    fit = classic.fit

    def set_params(self, params: dict[str, Any] | None, clone_flag: bool = False):
        """
        Sets params on the cluster model

        Parameters
        ----------
        params: Dict
            the params to set
        clone_flag: bool
            whether to return a clone of the cluster model
        """
        if params is None:
            params = {}
        else:
            params = {k: intify(v) for k, v in params.items()}
        if clone_flag:
            cluster_model = clone_cluster_model(self.cluster_model)
            cluster_model.set_params(**params)
            return cluster_model
        else:
            self.cluster_model.set_params(**params)

    _subblocking_graph_config = bulk._subblocking_graph_config

    _subblocking_prefix_counts = bulk._subblocking_prefix_counts

    _partition_subblocked_first_name_groups = bulk._partition_subblocked_first_name_groups

    _predict_subblocked_multiple_letter_groups = classic._predict_subblocked_multiple_letter_groups

    _reunite_no_middle_dead_ends = bulk._reunite_no_middle_dead_ends

    predict = classic.predict

    _cluster_one_block = bulk._cluster_one_block

    _finish_block_labels = bulk._finish_block_labels

    _enforced_cluster_seeds_require = bulk._enforced_cluster_seeds_require

    _repair_seed_groups = bulk._repair_seed_groups

    @staticmethod
    def _seeds_in_sight(dataset: ANDData, prediction_state: PredictionState | None) -> Mapping[str, Any] | None:
        """Return every required seed group in sight, whether or not block labels enforce it."""
        groups: dict[str, Any] = dict(dataset.cluster_seeds_require)
        groups.update(getattr(dataset, "repair_cluster_seeds_require", None) or {})
        if prediction_state is not None:
            groups.update(prediction_state.cluster_seeds_require)
        return groups or None

    predict_eps_grid_from_rust_featurizer = classic.predict_eps_grid_from_rust_featurizer

    predict_eps_grid = classic.predict_eps_grid

    _cluster_one_block_with_logging = bulk._cluster_one_block_with_logging

    predict_helper = classic.predict_helper

    # Keep the original boundary identity when callers override predict_helper.
    _default_predict_helper = predict_helper

    _predict_with_normalized_supervision = classic._predict_with_normalized_supervision

    _build_incremental_seed_setup = incremental._build_incremental_seed_setup

    _make_residual_clusterer = incremental._make_residual_clusterer

    predict_incremental_from_arrow = incremental.predict_incremental_from_arrow


# classic.py reads Clusterer at run time (for example _default_predict_helper in predict).
# Bind it here because classic.py cannot import clusterer.py without a cycle.
classic.Clusterer = Clusterer

# Preserve the original signatures and pickle paths of the class methods.
for _member in vars(Clusterer).values():
    if isinstance(_member, staticmethod):
        _functions = (_member.__func__,)
    elif isinstance(_member, property):
        _functions = (_member.fget, _member.fset)
    else:
        _functions = (_member,)
    for _method in _functions:
        if isinstance(_method, FunctionType) and _method.__module__.startswith("s2and.model."):
            _method.__module__ = Clusterer.__module__
            _method.__qualname__ = f"{Clusterer.__qualname__}.{_method.__name__}"
            _method.__annotations__.pop("self", None)
            if hasattr(_method, "__wrapped__"):
                _method.__wrapped__.__module__ = _method.__module__
                _method.__wrapped__.__qualname__ = _method.__qualname__
                _method.__wrapped__.__annotations__.pop("self", None)

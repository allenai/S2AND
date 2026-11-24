"""Fit and predict with classic ANDData inputs."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from s2and.corrections.block_corrections import average_linkage_cut, corrections_copy_codes, corrections_from_dataset
from s2and.corrections.profile_exclusions import (
    evict_profile_exclusion_violations,
    normalize_profile_exclusions,
    validate_profile_exclusions,
)
from s2and.corrections.seed_merging import seed_disallow_adjacency
from s2and.data import ANDData, _resolve_signature_splits
from s2and.incremental_linking.policy import (
    NAME_COUNTS_MANIFEST_SHA256_FIELD,
    clusterer_uses_name_count_features,
    require_dataset_name_counts_binding_for_clusterer,
    require_name_counts_manifest_sha256,
)
from s2and.metrics import b3_precision_recall_fscore
from s2and.model_pairwise import FastCluster, clone_cluster_model, intify
from s2and.prediction_state import PredictionState
from s2and.runtime import RuntimeContext, build_runtime_context, dataset_stage_uses_rust, stage_uses_rust
from s2and.supervision import normalize_partial_supervision, validate_seed_group_separations
from s2and.thread_config import resolve_n_jobs

from .constraints import (
    _cluster_seeds_require_from_rust_featurizer,
    _profile_exclusion_kwargs,
    _RustFeaturizerPredictDataset,
)
from .distances import (
    _compute_predict_batch_chunk_plan,
    _count_selected_features,
    _ensure_lightgbm_fitted,
    _guard_predict_block_matrix_allocation,
)
from .incremental import (
    _clusterer_with_prediction_params,
    _dataset_altered_cluster_signatures,
    _prediction_cluster_model_params,
    logger,
)

if TYPE_CHECKING:
    from .clusterer import Clusterer


def _validate_python_seed_separations(
    clusterer: Any,
    block_dict: Mapping[str, Sequence[str]],
    cluster_seeds_require: Mapping[str, int | str],
    partial_supervision: Mapping[tuple[str, str], int | float],
    incremental_dont_use_cluster_seeds: bool,
) -> None:
    """Reject caller distances that would join two seed groups the Python route keeps apart.

    The Python constraint separates seed groups only with default constraints on.
    """
    if (
        partial_supervision
        and cluster_seeds_require
        and not incremental_dont_use_cluster_seeds
        and getattr(clusterer, "dont_merge_cluster_seeds", True)
        and getattr(clusterer, "use_default_constraints_as_supervision", True)
    ):
        validate_seed_group_separations(
            cluster_seeds_require,
            partial_supervision,
            {signature: key for key, signatures in block_dict.items() for signature in signatures},
        )


def _prediction_state_from_dataset(dataset: ANDData) -> PredictionState:
    """Snapshot prediction inputs without changing the dataset's collections."""
    return PredictionState(
        cluster_seeds_require=dict(dataset.cluster_seeds_require),
        cluster_seeds_disallow=set(dataset.cluster_seeds_disallow),
        altered_cluster_signatures=list(_dataset_altered_cluster_signatures(dataset)),
    )


def fit(
    self: Clusterer,
    datasets: ANDData | list[ANDData],
    val_dists_precomputed: dict[str, dict[str, np.ndarray]] | None = None,
    metric_for_hyperopt: str = "b3",
) -> Clusterer:
    """
    Fits the clusterer

    Parameters
    ----------
    datasets: List[ANDData]
        the list of datasets to use for validations
    val_dists_precomputed: Dict
        precomputed distance matrices
    metric_for_hyperopt: string
        the metric to use for hyperparamter optimization

    Returns
    -------
    Clusterer: a fit clusterer, also sets the best params
    """
    from hyperopt import Trials, fmin, space_eval, tpe

    from s2and.calibration import default_cluster_search_space

    assert metric_for_hyperopt in {"b3", "ratio"}
    logger.info("Fitting clusterer")
    if isinstance(datasets, ANDData):
        datasets = [datasets]
    if len(datasets) == 0:
        raise ValueError("Clusterer.fit requires at least one dataset")

    if self.search_space is None:
        self.search_space = default_cluster_search_space()

    contract = getattr(self, "feature_contract", None)
    if not isinstance(contract, dict):
        contract = {}
    if clusterer_uses_name_count_features(self):
        count_bindings = {
            require_name_counts_manifest_sha256(
                getattr(dataset, "name_counts_manifest_sha256", None),
                context=f"Clusterer.fit dataset={getattr(dataset, 'name', '<unnamed>')}",
            )
            for dataset in datasets
        }
        if len(count_bindings) != 1:
            raise ValueError(
                "Clusterer.fit requires one verified name-count generation; "
                f"observed={sorted(repr(binding) for binding in count_bindings)}"
            )
        contract[NAME_COUNTS_MANIFEST_SHA256_FIELD] = next(iter(count_bindings))
    self.feature_contract = contract

    val_block_dict_list = []
    val_cluster_to_signatures_list = []
    val_dists_list = []
    val_datasets_list: list[ANDData] = []
    weights: list[float] = []
    for dataset in datasets:
        train_block_dict, val_block_dict, _ = _resolve_signature_splits(dataset)
        # incremental setting uses all the signatures in train and val
        # block-wise split uses only validation set for building the clustering model
        if dataset.unit_of_data_split == "time" or dataset.unit_of_data_split == "signatures":
            for block_key, signatures in train_block_dict.items():
                if block_key in val_block_dict:
                    val_block_dict[block_key].extend(signatures)

        # we don't need val blocks with only a single element
        val_block_dict = self.filter_blocks(val_block_dict, self.val_blocks_size)
        val_block_dict_list.append(val_block_dict)

        # block ground truth labels: cluster_to_signatures
        val_cluster_to_signatures = dataset.construct_cluster_to_signatures(val_block_dict)
        val_cluster_to_signatures_list.append(val_cluster_to_signatures)

        if val_dists_precomputed is None:
            val_dists = self.make_distance_matrices(val_block_dict, dataset)
        else:
            val_dists = val_dists_precomputed[dataset.name]
        val_dists_list.append(val_dists)
        val_datasets_list.append(dataset)

        # weights for weighted F1 average: total # of signatures in dataset
        weights.append(np.sum([len(i) for i in val_block_dict.values()]))

    def obj(params):
        self.set_params(params)
        f1s = []
        ratios = []
        for val_dataset, val_block_dict, val_cluster_to_signatures, val_dists in zip(
            val_datasets_list, val_block_dict_list, val_cluster_to_signatures_list, val_dists_list, strict=True
        ):
            pred_clusters, _ = self.predict(
                val_block_dict,
                dataset=val_dataset,
                dists=val_dists,
            )
            (
                _,
                _,
                f1,
                _,
                pred_bigger_ratios,
                true_bigger_ratios,
            ) = b3_precision_recall_fscore(val_cluster_to_signatures, pred_clusters)
            ratios.append(np.mean(pred_bigger_ratios + true_bigger_ratios))
            f1s.append(f1)
        if metric_for_hyperopt == "ratio":
            return np.average(ratios, weights=weights)
        elif metric_for_hyperopt == "b3":
            # minimize means we need to negate
            return -np.average(f1s, weights=weights)

    self.hyperopt_trials_store = Trials()
    _ = fmin(
        fn=obj,
        space=self.search_space,
        algo=partial(tpe.suggest, n_startup_jobs=5),
        max_evals=self.n_iter,
        trials=self.hyperopt_trials_store,
        rstate=np.random.default_rng(self.random_state),
    )
    # hyperopt has some problems with hp.choice so we need to do this:
    assert isinstance(self.hyperopt_trials_store, Trials)
    best_params = space_eval(self.search_space, self.hyperopt_trials_store.argmin)
    self.best_params = {k: intify(v) for k, v in best_params.items()}
    self.set_params(self.best_params)

    logger.info("Clusterer fit")
    return self


def _predict_subblocked_multiple_letter_groups(
    self: Clusterer,
    block_dict_multiple_letter: dict[str, list[str]],
    *,
    alert_flag: bool,
    dataset: ANDData,
    cluster_model_params: dict[str, Any] | None,
    partial_supervision: dict[tuple[str, str], int | float],
    use_s2_clusters: bool,
    incremental_dont_use_cluster_seeds: bool,
    runtime_context: RuntimeContext,
    total_ram_bytes: int | None,
    rust_featurizer: object | None = None,
    cluster_seeds_disallow: set[tuple[str, str]] | None = None,
    explicit_cluster_seeds_require: dict[str, int | str] | None = None,
    prediction_state: PredictionState | None = None,
) -> dict[str, list[str]]:
    pred_clusters: dict[str, list[str]] = {}
    if len(block_dict_multiple_letter) == 0:
        return pred_clusters

    if alert_flag:
        logger.info("Note! There are no subblocks with multiple letter first names")
        logger.info("Running predict on subblocks with single letter first names")
    else:
        logger.info("Running predict on subblocks with multiple letter first names")

    if rust_featurizer is not None:
        ordered_blocks = {
            block_key: block_dict_multiple_letter[block_key] for block_key in sorted(block_dict_multiple_letter)
        }
        start = time.time()
        if not isinstance(dataset, _RustFeaturizerPredictDataset):
            raise TypeError("Rust subblock prediction requires prepared Rust featurizer metadata")
        pred_clusters_intermediate, _ = self._predict_from_prepared_rust_featurizer(
            ordered_blocks,
            rust_featurizer,
            proxy_dataset=dataset,
            dists=None,
            cluster_model_params=cluster_model_params,
            partial_supervision=partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            explicit_cluster_seeds_require=explicit_cluster_seeds_require,
            resolved_cluster_seeds_disallow=set(cluster_seeds_disallow or ()),
            prediction_state=prediction_state,
        )
        logger.info("Finished Rust subblocks in %.3fs", time.time() - start)
        return pred_clusters_intermediate

    predict_times: dict[str, float] = {}
    predict_helper = self.predict_helper
    if getattr(predict_helper, "__func__", None) is Clusterer._default_predict_helper:
        predict_helper = self._predict_with_normalized_supervision
    for block_key in sorted(block_dict_multiple_letter):
        block_signatures = block_dict_multiple_letter[block_key]
        logger.info(f"Working on subblock {block_key}")
        start = time.time()
        pred_clusters_intermediate, _ = predict_helper(
            {block_key: block_signatures},
            dataset,
            dists=None,
            cluster_model_params=cluster_model_params,
            partial_supervision=partial_supervision,
            use_s2_clusters=use_s2_clusters,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            total_ram_bytes=total_ram_bytes,
            prediction_state=prediction_state,
        )
        end = time.time()
        predict_times[block_key] = end - start
        pred_clusters.update(pred_clusters_intermediate)
    logger.info(f"Finished, here's how long each took: {predict_times}")
    return pred_clusters


def predict(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    dataset: ANDData,
    dists: dict[str, np.ndarray] | None = None,
    cluster_model_params: dict[str, Any] | None = None,
    partial_supervision: dict[tuple[str, str], int | float] | None = None,
    use_s2_clusters: bool = False,
    incremental_dont_use_cluster_seeds: bool = False,
    batching_threshold: int | None = None,
    runtime_context: RuntimeContext | None = None,
    total_ram_bytes: int | None = None,
    profile_exclusions: Iterable[tuple[Any, Any]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
    """
    Predicts clusters

    Parameters
    ----------
    block_dict: Dict
        the block dict to predict clusters from
    dataset: ANDData
        the dataset
    dists: Dict
        (optional) precomputed distance matrices
    cluster_model_params: Dict
        Request-local clustering parameters, including incremental subblock phases.
    partial_supervision: Dict
        the dictionary of partial supervision provided with this dataset/these blocks
    use_s2_clusters: bool
        whether to "predict" using the clusters from Semantic Scholar's old system
    incremental_dont_use_cluster_seeds: bool
        whether to ignore dataset cluster seeds while resolving constraints in incremental flows
    batching_threshold: int | None
        Retired for ANDData. Use predict_from_arrow for subblocked prediction.
    total_ram_bytes: Optional[int]
        Optional explicit RAM budget for exact/non-incremental predict paths. When set, predict_helper
        uses it for pair-batch sizing and fails fast before allocating a block distance matrix that
        would exceed the usable budget.
    profile_exclusions: Iterable[tuple[Any, Any]] | None
        ``(anchor_signature_id, excluded_signature_id)`` pairs. The excluded mention never ends
        in the anchor's cluster; it is evicted from that cluster rather than vetoing its merges.

    Returns
    -------
    Dict: the predicted clusters
    Optional[Dict]: the predicted distance matrices. This is None when
    distances are built and clustered in the fused one-block-at-a-time path.
    """

    if batching_threshold is not None:
        raise ValueError("Classic subblocked prediction is retired; use predict_from_arrow with batching_threshold")
    partial_supervision = normalize_partial_supervision(partial_supervision)
    if not use_s2_clusters and any(len(signatures) > 1 for signatures in block_dict.values()):
        self = _clusterer_with_prediction_params(self, cluster_model_params)
    if runtime_context is None:
        runtime_context = build_runtime_context("cluster_predict", backend="python")
    elif stage_uses_rust(runtime_context):
        raise ValueError("Clusterer.predict operates on ANDData through Python; use predict_from_arrow for Rust")

    prediction_state = _prediction_state_from_dataset(dataset)
    _validate_python_seed_separations(
        self,
        block_dict,
        prediction_state.cluster_seeds_require,
        partial_supervision,
        incremental_dont_use_cluster_seeds,
    )
    exclusions = normalize_profile_exclusions(profile_exclusions)
    if exclusions:
        if dists is not None or use_s2_clusters:
            raise ValueError("profile_exclusions cannot be used with precomputed dists or use_s2_clusters")
        validate_profile_exclusions(exclusions, prediction_state.cluster_seeds_require)
        prediction_state.profile_exclusions = exclusions

    # normal mode - everything goes through full block clustering
    logger.info("Running predict on full blocks - no subblocking")
    start = time.time()
    predict_helper = self.predict_helper
    if getattr(predict_helper, "__func__", None) is Clusterer._default_predict_helper:
        predict_helper = self._predict_with_normalized_supervision
    pred_clusters, dists = predict_helper(
        block_dict,
        dataset,
        dists=dists,
        cluster_model_params=cluster_model_params,
        partial_supervision=partial_supervision,
        use_s2_clusters=use_s2_clusters,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        runtime_context=runtime_context,
        total_ram_bytes=total_ram_bytes,
        prediction_state=prediction_state,
    )
    end = time.time()
    total_predict_time = end - start
    logger.info(f"Finished predict on full blocks. Time taken: {total_predict_time}")

    if exclusions:
        pred_clusters, _moved = evict_profile_exclusion_violations(
            pred_clusters, exclusions, prediction_state.cluster_seeds_require
        )
    return dict(pred_clusters), dists


def predict_eps_grid_from_rust_featurizer(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    rust_featurizer: object,
    dists: Mapping[str, np.ndarray],
    eps_grid: Sequence[float],
    *,
    small_block_signatures: frozenset[str] | None = None,
) -> tuple[list[str], dict[float, np.ndarray]]:
    """Cluster every block at each EPS from precomputed distances and a Rust featurizer.

    The clusters at an EPS equal those of
    ``predict_from_rust_featurizer(block_dict, rust_featurizer)`` with
    ``cluster_model.eps`` set to it, when ``dists`` are the distances that call
    builds. See ``predict_eps_grid``.

    Args:
        block_dict: Blocks to cluster.
        rust_featurizer: Featurizer that holds every signature of the blocks. Name-conflict
            repair reads its first-name and ORCID facts.
        dists: Condensed distance matrix of every block with two or more signatures. A
            mapping may load each matrix on first access.
        eps_grid: EPS values to cut at.
        small_block_signatures: Signatures whose original surname block is small. Pass it
            when ``block_dict`` holds subblocks, which are smaller than their block;
            ``predict_from_arrow`` decides this before subblocking.

    Returns:
        The signatures in block order, and per EPS one cluster id per signature.

    Raises:
        ValueError: If the featurizer carries required cluster seeds. Precomputed
            distances do not hold them, as in ``predict_from_rust_featurizer``.
    """
    if _cluster_seeds_require_from_rust_featurizer(rust_featurizer):
        raise ValueError(
            "cluster_seeds_require cannot be used with precomputed dists because require pairs "
            "would not be injected into the distance matrix"
        )
    proxy_dataset = _RustFeaturizerPredictDataset(
        cluster_seeds_require={},
        cluster_seeds_disallow=set(),
        signatures={},
        rust_featurizer=rust_featurizer,
    )
    return self.predict_eps_grid(
        block_dict, cast(ANDData, proxy_dataset), dists, eps_grid, small_block_signatures=small_block_signatures
    )


def predict_eps_grid(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    dataset: ANDData,
    dists: Mapping[str, np.ndarray],
    eps_grid: Sequence[float],
    *,
    small_block_signatures: frozenset[str] | None = None,
) -> tuple[list[str], dict[float, np.ndarray]]:
    """Cluster every block at each EPS from one linkage per block.

    Only the FastCluster cut depends on EPS, so each block is linked once and
    cut at every EPS. The clusters at an EPS equal those of
    ``predict(block_dict, dataset)`` with ``cluster_model.eps`` set to it, when
    ``dists`` are the distances that call builds.

    Name-conflict repair runs here, unlike in ``predict(..., dists=dists)``. Pass only
    matrices built from this dataset's rules and seeds. An impossible pair that comes
    from supervision of the caller's own is read as a name conflict when the names
    also conflict.

    Args:
        block_dict: Blocks to cluster.
        dataset: Dataset whose cluster seeds constrain the clusters.
        dists: Condensed distance matrix of every block with two or more signatures.
            Each block's matrix is read once, so a mapping may load it on access.
        eps_grid: EPS values to cut at.
        small_block_signatures: Signatures whose original surname block is small, when
            ``block_dict`` holds subblocks; None lets the dataset or the blocks decide.

    Returns:
        The signatures in block order, and per EPS one cluster id per
        signature. Ids are unique across blocks.
    """
    if (
        not isinstance(self.cluster_model, FastCluster)
        or self.cluster_model.input_as_observation_matrix
        or _prediction_cluster_model_params(self, None) is not None
    ):
        raise ValueError("predict_eps_grid needs a FastCluster over condensed distances and no prediction params")
    from scipy.cluster.hierarchy import fcluster

    try:
        # fcluster validates the linkage on every call, about 0.15 ms; the
        # cut itself is this C routine, so later EPS call it directly.
        from scipy.cluster._hierarchy import cluster_dist
    except ImportError:
        cluster_dist = None

    prediction_state = _prediction_state_from_dataset(dataset)
    disallow_adjacency = seed_disallow_adjacency(
        prediction_state.cluster_seeds_disallow, normalize_partial_supervision(None)
    )
    corrections = corrections_from_dataset(self, block_dict, dataset, small_block_signatures=small_block_signatures)
    seed_groups = self._repair_seed_groups(dataset, prediction_state)
    split_seed_groups = self._seeds_in_sight(dataset, prediction_state)
    signatures: list[str] = []
    ids_by_eps: dict[float, list[np.ndarray]] = {float(eps): [] for eps in eps_grid}
    next_id = 0
    for block_key, block_signatures in block_dict.items():
        # predict requires a matrix for every block, even a single signature's.
        if block_key not in dists:
            raise KeyError(f"Missing precomputed distance matrix for block '{block_key}'")
        signatures.extend(block_signatures)
        if len(block_signatures) < 2:
            for ids in ids_by_eps.values():
                ids.append(np.arange(next_id, next_id + len(block_signatures), dtype=np.int64))
            next_id += len(block_signatures)
            continue
        cluster_model = clone_cluster_model(self.cluster_model)
        # predict clusters reused matrices with preserve_input=True; so does this.
        cluster_model.set_params(preserve_input=True)
        block_distances = dists[block_key]
        linkage_matrix = np.ascontiguousarray(cluster_model.linkage_matrix(block_distances), dtype=np.float64)
        block_width = len(block_signatures)
        # The matrix is preserved, so one copy of its codes serves the corrections at every EPS.
        block_codes = (
            None
            if corrections is None
            else corrections.codes_for(block_signatures, np.asarray(block_distances), resolve_n_jobs(self.n_jobs))
        )
        del block_distances
        for eps_index, (eps, ids) in enumerate(ids_by_eps.items()):
            if eps_index == 0 or cluster_dist is None:
                cut = fcluster(linkage_matrix, t=eps, criterion="distance")
            else:
                cut = np.zeros(block_width, dtype="i")
                cluster_dist(linkage_matrix, cut, float(eps), block_width)
            labels = self._finish_block_labels(
                block_signatures,
                cut,
                dataset,
                disallow_adjacency,
                prediction_state=prediction_state,
            )
            if corrections is not None:
                labels = corrections.apply(
                    block_signatures,
                    labels,
                    block_codes,
                    cut_height=average_linkage_cut(self.cluster_model, {"eps": float(eps)}),
                    forbidden=disallow_adjacency,
                    repair_seed_groups=seed_groups,
                    split_seed_groups=split_seed_groups,
                    block_key=block_key,
                )
            _, block_ids = np.unique(np.asarray(labels), return_inverse=True)
            ids.append(next_id + block_ids.astype(np.int64))
        next_id += block_width
    return signatures, {
        eps: np.concatenate(ids) if ids else np.zeros(0, dtype=np.int64) for eps, ids in ids_by_eps.items()
    }


def predict_helper(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    dataset: ANDData,
    dists: dict[str, np.ndarray] | None = None,
    cluster_model_params: dict[str, Any] | None = None,
    partial_supervision: dict[tuple[str, str], int | float] | None = None,
    use_s2_clusters: bool = False,
    incremental_dont_use_cluster_seeds: bool = False,
    runtime_context: RuntimeContext | None = None,
    total_ram_bytes: int | None = None,
    prediction_state: PredictionState | None = None,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
    """
    Predicts clusters

    Parameters
    ----------
    block_dict: Dict
        the block dict to predict clusters from
    dataset: ANDData
        the dataset
    dists: Dict
        (optional) precomputed distance matrices
    cluster_model_params: Dict
        params to set on the cluster model
    partial_supervision: Dict
        the dictionary of partial supervision provided with this dataset/these blocks
    use_s2_clusters: bool
        whether to "predict" using the clusters from Semantic Scholar's old system
    incremental_dont_use_cluster_seeds: bool
        whether to ignore dataset cluster seeds while resolving constraints in incremental flows
    total_ram_bytes: Optional[int]
        Optional explicit RAM budget for pair-batch sizing and exact block-matrix allocation checks.

    Returns
    -------
    Dict: the predicted clusters
    Optional[Dict]: the predicted distance matrices. This is None when
    distances are built and clustered in the fused one-block-at-a-time path.
    """
    partial_supervision = normalize_partial_supervision(partial_supervision)
    _validate_python_seed_separations(
        self,
        block_dict,
        dataset.cluster_seeds_require if prediction_state is None else prediction_state.cluster_seeds_require,
        partial_supervision,
        incremental_dont_use_cluster_seeds,
    )
    return self._predict_with_normalized_supervision(
        block_dict,
        dataset,
        dists=dists,
        cluster_model_params=cluster_model_params,
        partial_supervision=partial_supervision,
        use_s2_clusters=use_s2_clusters,
        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        runtime_context=runtime_context,
        total_ram_bytes=total_ram_bytes,
        prediction_state=prediction_state,
    )


def _predict_with_normalized_supervision(
    self: Clusterer,
    block_dict: dict[str, list[str]],
    dataset: ANDData,
    *,
    partial_supervision: dict[tuple[str, str], int | float],
    dists: dict[str, np.ndarray] | None = None,
    cluster_model_params: dict[str, Any] | None = None,
    use_s2_clusters: bool = False,
    incremental_dont_use_cluster_seeds: bool = False,
    runtime_context: RuntimeContext | None = None,
    total_ram_bytes: int | None = None,
    prediction_state: PredictionState | None = None,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray] | None]:
    """Predict using caller distances already validated at a public boundary."""
    if not use_s2_clusters and any(len(signatures) > 1 for signatures in block_dict.values()):
        self = _clusterer_with_prediction_params(self, cluster_model_params)
    cluster_model_params = _prediction_cluster_model_params(self, cluster_model_params)
    if runtime_context is None:
        runtime_context = build_runtime_context("cluster_predict", backend="python")

    if dists is not None and partial_supervision:
        raise ValueError(
            "partial_supervision cannot be used with precomputed dists because override pairs "
            "would not be injected into the distance matrix"
        )
    if dists is None and not use_s2_clusters:
        require_dataset_name_counts_binding_for_clusterer(
            self,
            dataset,
            context="Clusterer.predict_helper",
        )

    pred_clusters = defaultdict(list)

    if use_s2_clusters:
        for _, signature_list in block_dict.items():
            for _signature in signature_list:
                s2_cluster_key = dataset.signatures[_signature].author_id
                pred_clusters[s2_cluster_key].append(_signature)

        return dict(pred_clusters), dists

    cluster_seeds_disallow = (
        dataset.cluster_seeds_disallow if prediction_state is None else prediction_state.cluster_seeds_disallow
    )
    disallow_adjacency = seed_disallow_adjacency(cluster_seeds_disallow, partial_supervision)
    # A precomputed matrix can hold impossible pairs from the caller's own supervision, and
    # nothing here can tell those from name conflicts. Only distances built in this call are repaired.
    corrections = None if dists is not None else corrections_from_dataset(self, block_dict, dataset)

    effective_cluster_model_params = cluster_model_params
    fastcluster_fused_dtype = np.float16
    if isinstance(self.cluster_model, FastCluster):
        fastcluster_params: dict[str, Any] = dict(cluster_model_params or {})
        # Reused matrices should stay immutable by default; single-use fused path favors lower peak memory.
        if "preserve_input" not in fastcluster_params:
            fastcluster_params["preserve_input"] = bool(dists is not None)
        effective_cluster_model_params = fastcluster_params
        if dists is None:
            fastcluster_fused_dtype = np.float64

    if dists is not None:
        # precomputed dists (hyperopt path) — cluster from existing matrices
        for block_key in block_dict.keys():
            if block_key not in dists:
                raise KeyError(f"Missing precomputed distance matrix for block '{block_key}'")
            labels = self._cluster_one_block_with_logging(
                block_dict[block_key],
                dists[block_key],
                effective_cluster_model_params,
                dataset,
                disallow_adjacency,
                block_key=block_key,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                prediction_state=prediction_state,
                corrections=corrections,
                **_profile_exclusion_kwargs(prediction_state),
            )
            for signature, label in zip(block_dict[block_key], labels, strict=True):
                pred_clusters[block_key + "_" + str(label)].append(signature)
        return dict(pred_clusters), dists

    # fused path: build one block's matrix, cluster it, free it, repeat
    _ensure_lightgbm_fitted(self.classifier)
    _ensure_lightgbm_fitted(self.nameless_classifier)

    prev_block_key = ""
    pairwise_proba: np.ndarray | None = None
    seen_block_keys: set[str] = set()
    num_pairs = sum(len(sigs) * (len(sigs) - 1) // 2 for sigs in block_dict.values())
    logger.info("Predict helper: total_pairs=%d", num_pairs)
    model_predict_seconds = 0.0
    selected_count = _count_selected_features(self.featurizer_info)
    nameless_count = (
        _count_selected_features(self.nameless_featurizer_info) if self.nameless_featurizer_info is not None else 0
    )
    batch_chunk_plan = _compute_predict_batch_chunk_plan(
        self.featurizer_info.number_of_features,
        selected_feature_count=selected_count,
        nameless_feature_count=nameless_count,
        total_pairs=num_pairs,
        total_ram_bytes=total_ram_bytes,
    )
    pair_chunk_size = max(
        1,
        min(
            int(self.batch_size),
            int(batch_chunk_plan.chunk_pairs) if batch_chunk_plan is not None else int(self.batch_size),
        ),
    )
    if batch_chunk_plan is not None:
        logger.info(
            "Predict memory chunking: total_pairs=%d chunk_pairs=%d total_ram=%d total_ram_source=%s "
            "rss=%d rss_source=%s available=%d effective_available_fraction=%.3f stage_budget_bytes=%d",
            num_pairs,
            pair_chunk_size,
            int(batch_chunk_plan.total_ram_bytes),
            str(batch_chunk_plan.total_ram_source),
            int(batch_chunk_plan.current_rss_bytes),
            str(batch_chunk_plan.current_rss_source),
            int(batch_chunk_plan.available_bytes),
            float(batch_chunk_plan.effective_available_fraction),
            int(batch_chunk_plan.stage_budget_bytes),
        )
    use_rust_blockwise = dataset_stage_uses_rust(runtime_context, dataset)
    if use_rust_blockwise:
        for prediction_chunk in self._iter_rust_predicted_distance_matrix_chunks(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            pair_chunk_size=pair_chunk_size,
            total_ram_bytes=total_ram_bytes,
            prediction_state=prediction_state,
        ):
            chunk = prediction_chunk.chunk
            block_key = chunk.block_key
            if block_key != prev_block_key:
                self._flush_completed_block(
                    block_key=prev_block_key,
                    pairwise_proba=pairwise_proba,
                    block_dict=block_dict,
                    effective_cluster_model_params=effective_cluster_model_params,
                    dataset=dataset,
                    disallow_adjacency=disallow_adjacency,
                    pred_clusters=pred_clusters,
                    incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    prediction_state=prediction_state,
                    corrections=corrections,
                )
                pairwise_proba = None

                seen_block_keys.add(block_key)
                _guard_predict_block_matrix_allocation(
                    block_key=block_key,
                    block_size=int(chunk.block_size),
                    uses_fastcluster=isinstance(self.cluster_model, FastCluster),
                    fastcluster_fused_dtype=fastcluster_fused_dtype,
                    total_ram_bytes=total_ram_bytes,
                    copies_codes=corrections_copy_codes(self),
                )
                if isinstance(self.cluster_model, FastCluster):
                    pairwise_proba = np.zeros(
                        chunk.block_size * (chunk.block_size - 1) // 2,
                        dtype=fastcluster_fused_dtype,
                    )
                else:
                    pairwise_proba = np.zeros((chunk.block_size, chunk.block_size), dtype=np.float16)

            batch_predictions = prediction_chunk.predictions
            model_predict_seconds += prediction_chunk.batch_seconds
            assert pairwise_proba is not None
            if isinstance(self.cluster_model, FastCluster):
                start = int(chunk.start_offset)
                end = start + int(len(chunk.labels))
                pairwise_proba[start:end] = np.asarray(batch_predictions, dtype=pairwise_proba.dtype)
            else:
                pairwise_proba[chunk.index_i, chunk.index_j] = np.asarray(
                    batch_predictions,
                    dtype=pairwise_proba.dtype,
                )
            prev_block_key = block_key

        self._flush_completed_block(
            block_key=prev_block_key,
            pairwise_proba=pairwise_proba,
            block_dict=block_dict,
            effective_cluster_model_params=effective_cluster_model_params,
            dataset=dataset,
            disallow_adjacency=disallow_adjacency,
            pred_clusters=pred_clusters,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            prediction_state=prediction_state,
            corrections=corrections,
        )
    else:
        block_pair_index = 0

        def _on_block_start(block_key: str) -> None:
            nonlocal pairwise_proba, block_pair_index
            pairwise_proba = None
            seen_block_keys.add(block_key)
            block_size = len(block_dict[block_key])
            _guard_predict_block_matrix_allocation(
                block_key=block_key,
                block_size=block_size,
                uses_fastcluster=isinstance(self.cluster_model, FastCluster),
                fastcluster_fused_dtype=fastcluster_fused_dtype,
                total_ram_bytes=total_ram_bytes,
                copies_codes=corrections_copy_codes(self),
            )
            if isinstance(self.cluster_model, FastCluster):
                pairwise_proba = np.zeros(
                    block_size * (block_size - 1) // 2,
                    dtype=fastcluster_fused_dtype,
                )
            else:
                pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
            block_pair_index = 0

        def _write_prediction(
            _block_key: str,
            index_pair: tuple[int, int],
            prediction: float,
        ) -> None:
            nonlocal block_pair_index
            assert pairwise_proba is not None
            if isinstance(self.cluster_model, FastCluster):
                pairwise_proba[block_pair_index] = prediction
            else:
                i, j = index_pair
                pairwise_proba[i, j] = prediction
            block_pair_index += 1

        def _post_block_callback(block_key: str) -> None:
            self._flush_completed_block(
                block_key=block_key,
                pairwise_proba=pairwise_proba,
                block_dict=block_dict,
                effective_cluster_model_params=effective_cluster_model_params,
                dataset=dataset,
                disallow_adjacency=disallow_adjacency,
                pred_clusters=pred_clusters,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                prediction_state=prediction_state,
                corrections=corrections,
            )

        model_predict_seconds += self._featurize_predict_write_batches(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            runtime_context=runtime_context,
            num_pairs=num_pairs,
            write_prediction=_write_prediction,
            on_block_start=_on_block_start,
            post_block_callback=_post_block_callback,
            pair_chunk_size=pair_chunk_size,
            total_ram_bytes=total_ram_bytes,
            prediction_state=prediction_state,
        )

    # handle singleton blocks (0 or 1 signature — never appeared in generator)
    for block_key in block_dict.keys():
        if block_key not in seen_block_keys:
            for signature in block_dict[block_key]:
                pred_clusters[block_key + "_0"].append(signature)

    logger.info(
        "Telemetry stage: stage=model_predict_total seconds=%.3f blocks=%d",
        model_predict_seconds,
        len(block_dict),
    )
    return dict(pred_clusters), None

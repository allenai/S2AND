from __future__ import annotations

from s2and.eval import b3_precision_recall_fscore
from s2and.featurizer import FeaturizationInfo, many_pairs_featurize
from s2and.data import ANDData
from s2and.consts import LARGE_INTEGER, DEFAULT_CHUNK_SIZE

from typing import Dict, Optional, Any, Union, List, Tuple
from collections import defaultdict
import warnings
from functools import partial
from tqdm import tqdm
import logging
import copy

import numpy as np
from scipy.cluster.hierarchy import fcluster
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll import scope
from fastcluster import linkage

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import EfficiencyWarning

logger = logging.getLogger("s2and")


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
            Defaults to None, which uses a space appropriate to FastCluster.
        n_iter: int
            Number of hyperparameter evaluations
        n_jobs: int
            Parallelize each clusterer this many ways
        use_cache: bool
            Whether to use the cache when making distance matrices
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
            this flag controls whether to use cluster seeds to enforce "dont merge"
            as well as "must merge" constraints
        batch_size: int
            batch size for featurization, lower means less memory, but slower
    """

    def __init__(
        self,
        featurizer_info: FeaturizationInfo,
        classifier: Any,
        val_blocks_size: Optional[int] = None,
        cluster_model: Optional[Any] = None,
        search_space: Optional[Dict[str, Any]] = None,
        n_iter: int = 25,
        n_jobs: int = 16,
        use_cache: bool = False,
        use_default_constraints_as_supervision: bool = True,
        random_state: int = 42,
        nameless_classifier: Optional[Any] = None,
        nameless_featurizer_info: Optional[FeaturizationInfo] = None,
        dont_merge_cluster_seeds: bool = True,
        batch_size: int = 1000000,
    ):
        self.featurizer_info = featurizer_info
        self.nameless_featurizer_info = nameless_featurizer_info
        self.classifier = copy.deepcopy(classifier)
        self.nameless_classifier = copy.deepcopy(nameless_classifier)
        self.val_blocks_size = val_blocks_size
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_cache = use_cache
        self.use_default_constraints_as_supervision = use_default_constraints_as_supervision
        self.dont_merge_cluster_seeds = dont_merge_cluster_seeds
        if cluster_model is None:
            self.cluster_model = FastCluster(linkage="average")
        else:
            self.cluster_model = copy.deepcopy(cluster_model)

        if search_space is None:
            self.search_space = {"eps": hp.uniform("eps", 0, 1)}
        else:
            self.search_space = search_space

        self.hyperopt_trials_store: Optional[Union[Trials, List[Trials]]] = None
        self.best_params: Optional[Dict[Any, Any]] = None
        self.batch_size = batch_size

    @staticmethod
    def filter_blocks(block_dict: Dict[str, List[str]], num_to_keep: Optional[int] = None) -> Dict[str, List[str]]:
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
        # blocks with only 1 element are useless for train/val
        # and we can only keep as many as is specified
        out_dict = {}
        count = 0
        for block_key, signatures in block_dict.items():
            if len(signatures) > 1:
                out_dict[block_key] = signatures
                count += 1
                # early stopping if we have enough
                if num_to_keep is not None and count == num_to_keep:
                    return out_dict
        return out_dict

    def distance_matrix_helper(
        self,
        block_dict: Dict,
        dataset: ANDData,
        partial_supervision: Dict[Tuple[str, str], Union[int, float]],
        incremental_dont_use_cluster_seeds: bool = False,
    ):
        """
        Helper generator function to yield one pair for batch featurization on the fly

        Parameters
        ----------
        block_dict: Dict
            the block dictionary
        dataset: ANDData
            the dataset
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        incremental_dont_use_cluster_seeds: bool
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

        Returns
        -------
        yields pairs of ((sig id 1, sig id 2, label), index pair into the distance matrix, block key)
        """
        for block_key, signatures in block_dict.items():
            for i, j in zip(*np.triu_indices(len(signatures), k=1)):
                # subtracting LARGE_INTEGER so many_pairs_featurize knows not to make features
                label = np.nan
                if (signatures[i], signatures[j]) in partial_supervision:
                    label = partial_supervision[(signatures[i], signatures[j])] - LARGE_INTEGER
                elif (signatures[j], signatures[i]) in partial_supervision:
                    label = partial_supervision[(signatures[j], signatures[i])] - LARGE_INTEGER
                elif self.use_default_constraints_as_supervision:
                    value = dataset.get_constraint(
                        signatures[i],
                        signatures[j],
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                        incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
                    )
                    if value is not None:
                        label = value - LARGE_INTEGER

                yield ((signatures[i], signatures[j], label), (i, j), block_key)

    def make_distance_matrices(
        self,
        block_dict: Dict[str, List[str]],
        dataset: ANDData,
        partial_supervision: Dict[Tuple[str, str], Union[int, float]] = {},
        disable_tqdm: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
    ) -> Dict[str, np.array]:
        """
        Creates the distance matrices for the input blocks.
        Note: This function is much more complicated than it needs to be in an
        effort to reduce its memory footprint

        Parameters
        ----------
        block_dict: Dict
            the block dictionary to make distances for
        dataset: ANDData
            the dataset
        partial_supervision: Dict
            the dictionary of partial supervision provided with this dataset/these blocks
        disable_tqdm: bool
            whether to turn off the tqdm progress bars in this function
        incremental_dont_use_cluster_seeds: bool
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

        Returns
        -------
        Dict: the distance matrix dictionary, keyed by block key
        """
        logger.info(f"Making {len(block_dict)} distance matrices")
        logger.info("Initializing pairwise_probas")
        # initialize pairwise_probas with correctly size arrays
        pairwise_probas = {}
        for block_key, signatures in block_dict.items():
            block_size = len(signatures)
            if isinstance(self.cluster_model, FastCluster):
                # flattened pdist style
                pairwise_proba = np.zeros(int(block_size * (block_size - 1) / 2), dtype=np.float16)
            else:
                pairwise_proba = np.zeros((block_size, block_size), dtype=np.float16)
            pairwise_probas[block_key] = pairwise_proba

        logger.info("Pairwise probas initialized, starting making all pairs")

        # featurize and predict in batches
        helper_output = self.distance_matrix_helper(
            block_dict,
            dataset,
            partial_supervision,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
        )

        prev_block_key = ""
        batch_num = 0
        while True:
            logger.info(f"Featurizing batch {batch_num}")
            count = 0
            pairs = []
            indices = []
            blocks = []
            logger.info("Getting constraints")
            # iterate over a batch_size number of pairs
            for item in helper_output:
                pairs.append(item[0])
                indices.append(item[1])
                blocks.append(item[2])
                count += 1
                if count == self.batch_size:
                    break

            if len(pairs) == 0:
                break

            batch_features, _, batch_nameless_features = many_pairs_featurize(
                pairs,
                dataset,
                self.featurizer_info,
                self.n_jobs,
                use_cache=self.use_cache,
                chunk_size=DEFAULT_CHUNK_SIZE,
                nameless_featurizer_info=self.nameless_featurizer_info,
            )
            # get predictions where there isn't partial supervision
            # and fill the rest with partial supervision
            # undoing the offset by LARGE_INTEGER from above
            logger.info("Making predict flags")
            batch_labels = np.array([i[2] for i in pairs])
            predict_flag = np.isnan(batch_labels)
            not_predict_flag = ~predict_flag
            batch_predictions = np.zeros(len(batch_features))
            # index 0 is p(not the same)
            logger.info("Pairwise classification")
            if np.any(predict_flag):
                if self.nameless_classifier is not None:
                    batch_predictions[predict_flag] = (
                        self.classifier.predict_proba(batch_features[predict_flag, :])[:, 0]
                        + self.nameless_classifier.predict_proba(
                            batch_nameless_features[predict_flag, :]  # type: ignore
                        )[:, 0]
                    ) / 2
                else:
                    batch_predictions[predict_flag] = self.classifier.predict_proba(batch_features[predict_flag, :])[
                        :, 0
                    ]
            if np.any(not_predict_flag):
                batch_predictions[not_predict_flag] = batch_labels[not_predict_flag] + LARGE_INTEGER

            logger.info("Starting to make matrices")
            for within_batch_index, (prediction, signature_pair) in tqdm(
                enumerate(zip(batch_predictions, pairs)),
                total=len(batch_predictions),
                desc="Writing matrices",
                disable=disable_tqdm,
            ):
                block_key = blocks[within_batch_index]
                if block_key != prev_block_key:
                    block_key_start_index = blocks.index(block_key) + (batch_num * self.batch_size)
                    pairwise_proba = pairwise_probas[block_key]

                if isinstance(self.cluster_model, FastCluster):
                    index = (batch_num * self.batch_size + within_batch_index) - block_key_start_index

                    pairwise_proba[index] = prediction
                else:
                    i, j = indices[within_batch_index]
                    pairwise_proba[i, j] = prediction

                prev_block_key = block_key

            if count < self.batch_size:
                break

            batch_num += 1

        if not isinstance(self.cluster_model, FastCluster):
            for pairwise_proba in pairwise_probas.values():
                pairwise_proba += pairwise_proba.T
                np.fill_diagonal(pairwise_proba, 0)

        logger.info(f"{len(block_dict)} distance matrices made")
        return pairwise_probas

    def fit(
        self,
        datasets: Union[ANDData, List[ANDData]],
        val_dists_precomputed: Dict[str, Dict[str, np.array]] = None,
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
        assert metric_for_hyperopt in {"b3", "ratio"}
        logger.info("Fitting clusterer")
        if isinstance(datasets, ANDData):
            datasets = [datasets]
        val_block_dict_list = []
        val_cluster_to_signatures_list = []
        val_dists_list = []
        weights = []
        for dataset in datasets:
            # blocks
            train_block_dict, val_block_dict, _ = dataset.split_cluster_signatures()
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

            # distance matrix
            if val_dists_precomputed is None:
                val_dists = self.make_distance_matrices(val_block_dict, dataset)
            else:
                val_dists = val_dists_precomputed[dataset.name]
            val_dists_list.append(val_dists)

            # weights for weighted F1 average: total # of signatures in dataset
            weights.append(np.sum([len(i) for i in val_block_dict.values()]))

        def obj(params):
            self.set_params(params)
            f1s = []
            ratios = []
            for val_block_dict, val_cluster_to_signatures, val_dists in zip(
                val_block_dict_list, val_cluster_to_signatures_list, val_dists_list
            ):
                pred_clusters, _ = self.predict(val_block_dict, dataset=None, dists=val_dists)
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
            rstate=np.random.RandomState(self.random_state),
        )
        # hyperopt has some problems with hp.choice so we need to do this:
        best_params = space_eval(self.search_space, self.hyperopt_trials_store.argmin)
        self.best_params = {k: intify(v) for k, v in best_params.items()}
        self.set_params(self.best_params)

        logger.info("Clusterer fit")
        return self

    def set_params(self, params: Optional[Dict[str, Any]], clone_flag: bool = False):
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
            cluster_model = clone(self.cluster_model)
            cluster_model.set_params(**params)
            return cluster_model
        else:
            self.cluster_model.set_params(**params)

    def predict(
        self,
        block_dict: Dict[str, List[str]],
        dataset: ANDData,
        dists: Optional[Dict[str, np.array]] = None,
        cluster_model_params: Optional[Dict[str, Any]] = None,
        partial_supervision: Dict[Tuple[str, str], Union[int, float]] = {},
        use_s2_clusters: bool = False,
        incremental_dont_use_cluster_seeds: bool = False,
    ) -> Tuple[Dict[str, List[str]], Optional[Dict[str, np.ndarray]]]:
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
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

        Returns
        -------
        Dict: the predicted clusters
        Dict: the predicted distance matrices
        """

        pred_clusters = defaultdict(list)

        if use_s2_clusters:
            for _, signature_list in block_dict.items():
                for _signature in signature_list:
                    s2_cluster_key = dataset.signatures[_signature].author_id
                    pred_clusters[s2_cluster_key].append(_signature)

            return dict(pred_clusters), dists

        if dists is None:
            dists = self.make_distance_matrices(
                block_dict,
                dataset,
                partial_supervision,
                incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            )

        for block_key in block_dict.keys():
            if block_key in dists and len(block_dict[block_key]) > 1:
                cluster_model = self.set_params(cluster_model_params, clone_flag=True)
                with warnings.catch_warnings():
                    # annoying sparse matrix not sorted warning
                    warnings.simplefilter("ignore", category=EfficiencyWarning)
                    cluster_model.fit(dists[block_key])
                labels = cluster_model.labels_
                # in HDBSCAN the labels of -1 are actually "outliers"
                # each of these gets its own label starting at
                # max label + 1 and going up
                max_label = labels.max()
                negative_one_label_locations = np.where(labels == -1)[0]
                for i, loc in enumerate(negative_one_label_locations):
                    labels[loc] = max_label + 1 + i
            else:
                labels = [0]

            for signature, label in zip(block_dict[block_key], labels):
                pred_clusters[block_key + "_" + str(label)].append(signature)

        return dict(pred_clusters), dists

    def predict_incremental(
        self, block_signatures: List[str], dataset: ANDData, prevent_new_incompatibilities: bool = True
    ):
        """
        Predict clustering in incremental mode. This assumes that the majority of the labels are passed
        in using the cluster_seeds parameter of the dataset class, and skips work by simply assigning each
        unassigned signature to the closest cluster if distance is less than eps, and then separately cluster all
        the unassigned signatures that are not within eps of any existing cluster.

        Corrected, claimed profiles should be noted via the altered_cluster_signatures parameter (in ANDData).
        Then predict_incremental performs a pre-clustering step on each altered cluster to determine how
        S2AND would divide it into clusters. Mentions are incrementally added to these new subclusters,
        then reassembled to restore the complete claimed profile when S2AND returns results.

        Currently this would be useful in the following situation. We have a massive block, for which we want
        to cluster a small number of new signatures into (block size * number of new signatures should be less
        than the normal batch size).

        Note: this function was designed to work on a single block at a time.

        Parameters
        ----------
        block_signatures: List[str]
            the signatures in the block to predict from
        dataset: ANDData
            the dataset
        prevent_new_incompatibilities: bool
            if True, prevents the addition to a cluster of new first names that are not prefix match,
            or in the name pairs list, for at least one existing name in the cluster. This can happen
            if a claimed cluster has D Jones and David Jones, s2and would have split that cluster into two,
            and then s2and might add Donald Jones to the D Jones cluster, and once remerged, the resulting
            final cluster would have D Jones, David Jones, and Donald Jones.

        Returns
        -------
        Dict: the predicted clusters
        """
        recluster_map = {}
        cluster_seeds_require = copy.deepcopy(dataset.cluster_seeds_require)
        if dataset.altered_cluster_signatures is not None:
            altered_cluster_nums = set(
                dataset.cluster_seeds_require[altered_signature_id]
                for altered_signature_id in dataset.altered_cluster_signatures
            )
            if len(altered_cluster_nums) > 0:
                cluster_seeds_require_inverse: Dict[int, list] = {}
                for signature_id, cluster_num in dataset.cluster_seeds_require.items():
                    if cluster_num not in cluster_seeds_require_inverse:
                        cluster_seeds_require_inverse[cluster_num] = []
                    cluster_seeds_require_inverse[cluster_num].append(signature_id)
                for altered_cluster_num in altered_cluster_nums:
                    signature_ids_for_cluster_num = cluster_seeds_require_inverse[altered_cluster_num]

                    # Note: incremental_dont_use_cluster_seeds is set to True, because at this stage
                    # of incremental clustering we are splitting up the claimed profiles that we received
                    # from production so that they align with s2and's predictions. When doing this, we
                    # don't want to use the passed in cluster seeds, because they reflect the claimed profile, not
                    # s2and's predictions
                    reclustered_output, _ = self.predict(
                        {"block": signature_ids_for_cluster_num}, dataset, incremental_dont_use_cluster_seeds=True
                    )
                    if len(reclustered_output) > 1:
                        for i, new_cluster_of_signatures in enumerate(reclustered_output.values()):
                            new_cluster_num = str(altered_cluster_num) + f"_{i}"
                            recluster_map[new_cluster_num] = altered_cluster_num
                            for reclustered_signature_id in new_cluster_of_signatures:
                                cluster_seeds_require[reclustered_signature_id] = new_cluster_num  # type: ignore

        all_pairs = []
        for possibly_unassigned_signature in block_signatures:
            if possibly_unassigned_signature in cluster_seeds_require:
                continue
            unassigned_signature = possibly_unassigned_signature
            for signature in cluster_seeds_require.keys():
                label = np.nan
                if self.use_default_constraints_as_supervision:
                    value = dataset.get_constraint(
                        unassigned_signature,
                        signature,
                        dont_merge_cluster_seeds=self.dont_merge_cluster_seeds,
                    )
                    if value is not None:
                        label = value - LARGE_INTEGER
                all_pairs.append((unassigned_signature, signature, label))

        batch_features, _, batch_nameless_features = many_pairs_featurize(
            all_pairs,
            dataset,
            self.featurizer_info,
            self.n_jobs,
            use_cache=self.use_cache,
            chunk_size=DEFAULT_CHUNK_SIZE,
            nameless_featurizer_info=self.nameless_featurizer_info,
        )

        # get predictions where there isn't partial supervision
        # and fill the rest with partial supervision
        # undoing the offset by LARGE_INTEGER from above
        logger.info("Making predict flags")
        batch_labels = np.array([i[2] for i in all_pairs])
        predict_flag = np.isnan(batch_labels)
        not_predict_flag = ~predict_flag
        batch_predictions = np.zeros(len(batch_features))
        # index 0 is p(not the same)
        logger.info("Pairwise classification")
        if np.any(predict_flag):
            if self.nameless_classifier is not None:
                batch_predictions[predict_flag] = (
                    self.classifier.predict_proba(batch_features[predict_flag, :])[:, 0]
                    + self.nameless_classifier.predict_proba(batch_nameless_features[predict_flag, :])[  # type: ignore
                        :, 0
                    ]
                ) / 2
            else:
                batch_predictions[predict_flag] = self.classifier.predict_proba(batch_features[predict_flag, :])[:, 0]
        if np.any(not_predict_flag):
            batch_predictions[not_predict_flag] = batch_labels[not_predict_flag] + LARGE_INTEGER

        logger.info("Computing average distances for unassigned signatures")
        signature_to_cluster_to_average_dist: Dict[str, Dict[int, Tuple[float, int]]] = defaultdict(
            lambda: defaultdict(lambda: (0, 0))
        )
        for signature_pair, dist in zip(all_pairs, batch_predictions):
            unassigned_signature, assigned_signature, _ = signature_pair
            if assigned_signature not in cluster_seeds_require:
                continue
            cluster_id = cluster_seeds_require[assigned_signature]
            previous_average, previous_count = signature_to_cluster_to_average_dist[unassigned_signature][cluster_id]
            signature_to_cluster_to_average_dist[unassigned_signature][cluster_id] = (
                (previous_average * previous_count + dist) / (previous_count + 1),
                previous_count + 1,
            )

        logger.info("Assigning unassigned signatures")
        pred_clusters = defaultdict(list)
        singleton_signatures = []
        for signature_id, cluster_id in dataset.cluster_seeds_require.items():
            pred_clusters[f"{cluster_id}"].append(signature_id)
        for unassigned_signature, cluster_dists in signature_to_cluster_to_average_dist.items():
            best_cluster_id = None
            best_dist = float("inf")
            for cluster_id, (average_dist, _) in cluster_dists.items():
                if average_dist < best_dist and average_dist < self.cluster_model.eps:
                    best_cluster_id = cluster_id
                    best_dist = average_dist
            if best_cluster_id is not None:
                # undo the reclustering step
                new_name_disallowed = False
                if best_cluster_id in recluster_map:
                    best_cluster_id = recluster_map[best_cluster_id]  # type: ignore

                    if prevent_new_incompatibilities:
                        # restrict reclusterings that would add a new name incompatibility to the main cluster
                        main_cluster_signatures = cluster_seeds_require_inverse[best_cluster_id]
                        all_firsts = set(
                            [
                                dataset.signatures[signature_id].author_info_first_normalized_without_apostrophe
                                for signature_id in main_cluster_signatures
                            ]
                        )
                        all_firsts = {first for first in all_firsts if len(first) > 1}

                        # if all the existing first names in the cluster are single characters,
                        # there is nothing else to check
                        if len(all_firsts) > 0:
                            first_unassigned = dataset.signatures[
                                unassigned_signature
                            ].author_info_first_normalized_without_apostrophe
                            match_found = False
                            for first_assigned in all_firsts:
                                prefix = first_assigned.startswith(first_unassigned) or first_unassigned.startswith(
                                    first_assigned
                                )
                                known_alias = (first_assigned, first_unassigned) in dataset.name_tuples

                                if prefix or known_alias:
                                    match_found = True
                                    break
                            # if the candidate name is a prefix or a name alias for any of the existing names,
                            # we will allow it to cluster. If it is not, then it has been clustered with a single
                            # character name, and we don't want to allow it
                            if not match_found:
                                signature = dataset.signatures[unassigned_signature]
                                first = signature.author_info_first
                                last = signature.author_info_last
                                paper_id = signature.paper_id
                                logger.info(
                                    (
                                        "Incremental clustering prevented a name compatibility issue from being "
                                        f"added while clustering {first} {last} on {paper_id}"
                                    )
                                )
                                new_name_disallowed = True

                if new_name_disallowed:
                    singleton_signatures.append(unassigned_signature)
                else:
                    pred_clusters[f"{best_cluster_id}"].append(unassigned_signature)
            else:
                singleton_signatures.append(unassigned_signature)

        reclustered_output, _ = self.predict({"block": singleton_signatures}, dataset)
        new_cluster_id = dataset.max_seed_cluster_id or 0
        for new_cluster in reclustered_output.values():
            pred_clusters[str(new_cluster_id)] = new_cluster
            new_cluster_id += 1

        logger.info("Returning incrementally predicted clusters")
        return dict(pred_clusters)


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
        estimator: Optional[Any] = None,
        search_space: Optional[Dict[str, Any]] = None,
        monotone_constraints: Optional[str] = None,
        n_iter: int = 50,
        n_jobs: int = 16,  # for the model, not the hyperopt
        random_state: int = 42,
    ):
        if estimator is None:
            self.estimator = lgb.LGBMClassifier(
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
        if self.monotone_constraints is not None and isinstance(self.estimator, lgb.LGBMClassifier):
            self.estimator.set_params(monotone_constraints=self.monotone_constraints)
            self.estimator.set_params(monotone_constraints_method="advanced")
            self.search_space["monotone_penalty"] = hp.uniform("monotone_penalty", 0, 5)

        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params: Optional[Dict] = None
        self.hyperopt_trials_store: Optional[Trials] = None
        self.classifier: Optional[Any] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Trials:
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
                rstate=np.random.RandomState(self.random_state),
            )
            best_params = space_eval(self.search_space, self.hyperopt_trials_store.argmin)
            self.best_params = {k: intify(v) for k, v in best_params.items()}
            self.estimator.set_params(**self.best_params)
        else:
            self.best_params = {}
            self.hyperopt_trials_store == {}

        # refitting but only on training data so as not to leak anything
        self.classifier = self.estimator.fit(X_train, y_train)

        return self.hyperopt_trials_store

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.classifier is not None, "You need to call fit first"
        return self.classifier.predict_proba(X)


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
            raise Exception("Voting type must be one of 'soft' or 'hard'")
        return predictions

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
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
            raise AttributeError("predict_proba is not available when" " voting=%r" % self.voting)
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
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators]).T


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
            raise Exception(
                "The 'linkage' parameter has to be one of: "
                + "'single', complete', 'average', 'weighted', 'ward', 'centroid', 'median'."
            )

        self.linkage = linkage
        self.eps = eps
        self.preserve_input = preserve_input
        self.input_as_observation_matrix = input_as_observation_matrix
        self.labels_ = None

    def fit(self, X: np.array) -> np.array:
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
            raise Exception(
                "Input to fit is one-dimensional, but input_as_observation_matrix flag is set to True. "
                "If you intended to pass in an observation matrix, it must be 2-D (N x feature_dimension)."
            )
        elif len(X.shape) == 2 and not self.input_as_observation_matrix:
            raise Exception(
                "Input to fit is two-dimensional, but input_as_observation_matrix flag is set to False. "
                "If you intended to pass in a distance matrix, it must be flattened (1-D)."
            )
        elif len(X.shape) > 2:
            raise Exception("The input to fit can only be one-dimensional or two-dimensional.")
        Z = linkage(X, self.linkage, preserve_input=self.preserve_input)
        self.labels_ = fcluster(Z, t=self.eps, criterion="distance")
        return self

    def fit_transform(self, X: np.array) -> np.array:
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
        self.fit(X)
        return self.labels_

    def transform(self, X: np.array):
        raise Exception("FastCluster has no inductive mode. Use 'fit' or 'fit_transform' instead.")

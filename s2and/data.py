from typing import Optional, Union, Dict, List, Any, Tuple, Set, NamedTuple

import os
import re
import json
import numpy as np
import pandas as pd
import logging
import pickle
import multiprocessing
from tqdm import tqdm

from functools import reduce
from collections import defaultdict, Counter

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from s2and.sampling import sampling, random_sampling
from s2and.consts import (
    NUMPY_NAN,
    NAME_COUNTS_PATH,
    PROJECT_ROOT_PATH,
    LARGE_DISTANCE,
    CLUSTER_SEEDS_LOOKUP,
)
from s2and.file_cache import cached_path
from s2and.text import (
    normalize_text,
    get_text_ngrams,
    compute_block,
    get_text_ngrams_words,
    detect_language,
    AFFILIATIONS_STOP_WORDS,
    VENUE_STOP_WORDS,
    NAME_PREFIXES,
    DROPPED_AFFIXES,
    ORCID_PATTERN,
)

logger = logging.getLogger("s2and")


class NameCounts(NamedTuple):
    first: Optional[int]
    last: Optional[int]
    first_last: Optional[int]
    last_first_initial: Optional[int]


class Signature(NamedTuple):
    author_info_first: Optional[str]
    author_info_first_normalized_without_apostrophe: Optional[str]
    author_info_middle: Optional[str]
    author_info_middle_normalized_without_apostrophe: Optional[str]
    author_info_last_normalized: Optional[str]
    author_info_last: str
    author_info_suffix_normalized: Optional[str]
    author_info_suffix: Optional[str]
    author_info_first_normalized: Optional[str]
    author_info_coauthors: Optional[List[str]]
    author_info_coauthor_blocks: Optional[List[str]]
    author_info_full_name: Optional[str]
    author_info_affiliations: List[str]
    author_info_affiliations_n_grams: Optional[Counter]
    author_info_coauthor_n_grams: Optional[Counter]
    author_info_email: Optional[str]
    author_info_orcid: Optional[str]
    author_info_name_counts: Optional[NameCounts]
    author_info_position: int
    author_info_block: str
    author_info_given_block: Optional[str]
    author_info_estimated_gender: Optional[str]
    author_info_estimated_ethnicity: Optional[str]
    paper_id: int
    sourced_author_source: Optional[str]
    sourced_author_ids: List[str]
    author_id: Optional[int]
    signature_id: str


class Author(NamedTuple):
    author_name: str
    position: int


class Paper(NamedTuple):
    title: str
    has_abstract: Optional[bool]
    in_signatures: Optional[bool]
    is_english: Optional[bool]
    is_reliable: Optional[bool]
    predicted_language: Optional[str]
    title_ngrams_words: Optional[Counter]
    authors: List[Author]
    venue: Optional[str]
    journal_name: Optional[str]
    title_ngrams_chars: Optional[Counter]
    venue_ngrams: Optional[Counter]
    journal_ngrams: Optional[Counter]
    reference_details: Optional[Tuple[Counter, Counter, Counter, Counter]]
    year: Optional[int]
    references: Optional[List[int]]
    paper_id: int


class MiniPaper(NamedTuple):
    title: str
    venue: Optional[str]
    journal_name: Optional[str]
    authors: List[str]


class ANDData:
    """
    The main class for holding our representation of an author disambiguation dataset

    Input:
        signatures: path to the signatures json file (or the json object)
        papers: path to the papers information json file (or the json object)
        name: name of the dataset, used for caching computed features
        mode: 'train' or 'inference'; if 'inference', everything related to splitting will be ignored
        clusters: path to the clusters json file (or the json object)
        specter_embeddings: path to the specter embeddings pickle (or the dictionary object)
        cluster_seeds: path to the cluster seed json file (or the json object)
        altered_cluster_signatures: path to the signature ids \n-separated txt file (or a list or set object)
            Clusters that these signatures appear in will be marked as "altered"
        train_pairs: path to predefined train pairs csv (or the dataframe object)
        val_pairs: path to predefined val pairs csv (or the dataframe object)
        test_pairs: path to predefined test pairs csv (or the dataframe object)
        train_blocks: path to predefined train blocks (or the json object)
        val_blocks: path to predefined val blocks (or the json object)
        test_blocks: path to predefined test blocks (or the json object)
        train_signatures: path to predefined train signatures (or the json object)
        val_signatures: path to predefined val signatures (or the json object)
        test_signatures: path to predefined test signatures (or the json object)
        block_type: can be either "s2" or "original"
        unit_of_data_split: options are ("signatures", "blocks", "time")
        num_clusters_for_block_size: probably leave as default,
            controls train/val/test splits based on block size
        train_ratio: training ratio of instances for clustering
        val_ratio: validation ratio of instances for clustering
        test_ratio: test ratio of instances for clustering
        train_pairs_size: number of training pairs for learning the linkage function
        val_pairs_size: number of validation pairs for fine-tuning the linkage function parameters
        test_pairs_size: number of test pairs for evaluating the linkage function
        pair_sampling_block: sample pairs from only within blocks?,
        pair_sampling_balanced_classes: sample a balanced number of positive and negative pairs?,
        pair_sampling_balanced_homonym_synonym: sample a balanced number of homonymous and synonymous pairs?,
        all_test_pairs_flag: With blocking, for the linkage function evaluation task, should the test
            contain all possible pairs from test blocks, or the given number of pairs (test_pairs_size)
        random_seed: random seed
        load_name_counts: Whether or not to load name counts
        n_jobs: number of cpus to use
        preprocess: whether to preprocess the data (normalization, etc)
        name_tuples: optionally pass in the already created set of name tuples, to avoid recomputation
            can be None or "filtered" or a set of name tuples
        use_orcid_id: whether to use the orcid id for (a) constraints as true if orcids match and
            (b) subblocking so that any sigs with the same orcid are in the same subblock
    """

    def __init__(
        self,
        signatures: Union[str, Dict],
        papers: Union[str, Dict],
        name: str,
        mode: str = "train",
        clusters: Optional[Union[str, Dict]] = None,
        specter_embeddings: Optional[Union[str, Dict]] = None,
        cluster_seeds: Optional[Union[str, Dict]] = None,
        altered_cluster_signatures: Optional[Union[str, List, Set]] = None,
        train_pairs: Optional[Union[str, List]] = None,
        val_pairs: Optional[Union[str, List]] = None,
        test_pairs: Optional[Union[str, List]] = None,
        train_blocks: Optional[Union[str, List]] = None,
        val_blocks: Optional[Union[str, List]] = None,
        test_blocks: Optional[Union[str, List]] = None,
        train_signatures: Optional[Union[str, List]] = None,
        val_signatures: Optional[Union[str, List]] = None,
        test_signatures: Optional[Union[str, List]] = None,
        block_type: str = "s2",
        unit_of_data_split: str = "blocks",
        num_clusters_for_block_size: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        train_pairs_size: int = 30000,
        val_pairs_size: int = 5000,
        test_pairs_size: int = 5000,
        pair_sampling_block: bool = True,
        pair_sampling_balanced_classes: bool = False,
        pair_sampling_balanced_homonym_synonym: bool = False,
        all_test_pairs_flag: bool = False,
        random_seed: int = 1111,
        load_name_counts: Union[bool, Dict] = True,
        n_jobs: int = 1,
        preprocess: bool = True,
        name_tuples: Optional[Union[Set[Tuple[str, str]], str]] = "filtered",
        use_orcid_id: bool = True,
    ):
        if mode == "train":
            if train_blocks is not None and block_type != "original":
                logger.warning("If you are passing in training/val/test blocks, then you may want original blocks.")

            if unit_of_data_split == "blocks" and not pair_sampling_block:
                raise Exception("Block-based cluster splits are not compatible with sampling stratgies 0 and 1.")

            if (clusters is not None and train_pairs is not None) or (
                clusters is None and train_pairs is None and train_blocks is None
            ):
                raise Exception("Set exactly one of clusters and train_pairs")

            if train_blocks is not None and train_pairs is not None:
                raise Exception("Can't pass in both train_blocks and train_pairs")

            if train_blocks is not None and clusters is None:
                raise Exception("Train blocks still needs clusters")

        logger.info("loading papers")
        self.papers = self.maybe_load_json(papers)
        # convert dictionary to namedtuples for memory reduction
        for paper_id, paper in self.papers.items():
            self.papers[paper_id] = Paper(
                title=paper["title"],
                has_abstract=paper["abstract"] not in {"", None},
                in_signatures=None,
                is_english=None,
                is_reliable=None,
                predicted_language=None,
                title_ngrams_words=None,
                authors=[
                    Author(
                        author_name=author["author_name"],
                        position=author["position"],
                    )
                    for author in paper["authors"]
                ],
                venue=paper["venue"],
                journal_name=paper["journal_name"],
                title_ngrams_chars=None,
                venue_ngrams=None,
                journal_ngrams=None,
                reference_details=None,
                year=paper["year"],
                references=paper.get("references", []),
                paper_id=paper["paper_id"],
            )
        logger.info("loaded papers")

        logger.info("loading signatures")
        self.signatures = self.maybe_load_json(signatures)
        # convert dictionary to namedtuples for memory reduction
        for signature_id, signature in self.signatures.items():
            self.signatures[signature_id] = Signature(
                author_info_first=signature["author_info"]["first"],
                author_info_first_normalized_without_apostrophe=None,
                author_info_middle=signature["author_info"]["middle"],
                author_info_middle_normalized_without_apostrophe=None,
                author_info_last_normalized=None,
                author_info_last=signature["author_info"]["last"],
                author_info_suffix_normalized=None,
                author_info_suffix=signature["author_info"]["suffix"],
                author_info_first_normalized=None,
                author_info_coauthors=None,
                author_info_coauthor_blocks=None,
                author_info_full_name=None,
                author_info_affiliations=signature["author_info"]["affiliations"],
                author_info_affiliations_n_grams=None,
                author_info_coauthor_n_grams=None,
                author_info_email=signature["author_info"]["email"],
                author_info_orcid=(
                    signature["author_info"]["source_ids"][0]
                    if use_orcid_id
                    and "source_id_source" in signature["author_info"]
                    and signature["author_info"]["source_id_source"] == "ORCID"
                    else None
                ),
                author_info_name_counts=None,
                author_info_position=signature["author_info"]["position"],
                author_info_block=signature["author_info"]["block"],
                author_info_given_block=signature["author_info"].get("given_block", None),
                author_info_estimated_gender=signature["author_info"].get("estimated_gender", None),
                author_info_estimated_ethnicity=signature["author_info"].get("estimated_ethnicity", None),
                paper_id=signature["paper_id"],
                sourced_author_source=signature.get("sourced_author_source", None),
                sourced_author_ids=signature.get("sourced_author_ids", []),
                author_id=signature.get("author_id", None),
                signature_id=signature["signature_id"],
            )
        logger.info("loaded signatures")
        self.name = name
        self.mode = mode
        logger.info("loading clusters")
        self.clusters: Optional[Dict] = self.maybe_load_json(clusters)
        logger.info("loaded clusters, loading specter")
        self.specter_embeddings = self.maybe_load_specter(specter_embeddings)
        # prevents errors during testing where we have no specter embeddings
        if self.specter_embeddings is None:
            self.specter_embeddings = {}  # type: ignore
        logger.info("loaded specter, loading cluster seeds")
        cluster_seeds_dict = self.maybe_load_json(cluster_seeds)
        self.altered_cluster_signatures = self.maybe_load_list(altered_cluster_signatures)
        self.cluster_seeds_disallow = set()
        self.cluster_seeds_require = {}  # type: ignore
        self.max_seed_cluster_id = None
        if cluster_seeds_dict is not None:
            cluster_num = 0
            for signature_id_a, values in cluster_seeds_dict.items():
                root_added = False
                for signature_id_b, constraint_string in values.items():
                    if constraint_string == "disallow":
                        self.cluster_seeds_disallow.add((signature_id_a, signature_id_b))
                    elif constraint_string == "require":
                        if not root_added:
                            self.cluster_seeds_require[signature_id_a] = cluster_num
                            root_added = True
                        self.cluster_seeds_require[signature_id_b] = cluster_num
                cluster_num += 1
            self.max_seed_cluster_id = cluster_num
        logger.info("loaded cluster seeds")
        # check that all of the altered_clustere_signatures are in the cluster_seeds_require
        if self.altered_cluster_signatures is not None:
            for signature_id in self.altered_cluster_signatures:
                if signature_id not in self.cluster_seeds_require:
                    raise Exception(f"Altered cluster signature {signature_id} not in cluster_seeds_require")
        self.train_pairs = self.maybe_load_dataframe(train_pairs)
        self.val_pairs = self.maybe_load_dataframe(val_pairs)
        self.test_pairs = self.maybe_load_dataframe(test_pairs)
        self.train_blocks = self.maybe_load_json(train_blocks)
        self.val_blocks = self.maybe_load_json(val_blocks)
        self.test_blocks = self.maybe_load_json(test_blocks)
        self.train_signatures = self.maybe_load_json(train_signatures)
        self.val_signatures = self.maybe_load_json(val_signatures)
        self.test_signatures = self.maybe_load_json(test_signatures)
        self.block_type = block_type
        self.unit_of_data_split = unit_of_data_split
        self.num_clusters_for_block_size = num_clusters_for_block_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_pairs_size = train_pairs_size
        self.val_pairs_size = val_pairs_size
        self.test_pairs_size = test_pairs_size
        self.pair_sampling_block = pair_sampling_block
        self.pair_sampling_balanced_classes = pair_sampling_balanced_classes
        self.pair_sampling_balanced_homonym_synonym = pair_sampling_balanced_homonym_synonym
        self.all_test_pairs_flag = all_test_pairs_flag
        self.random_seed = random_seed

        if self.clusters is None:
            self.signature_to_cluster_id = None

        if self.mode == "train":
            if self.clusters is not None:
                self.signature_to_cluster_id = {}
                logger.info("making signature to cluster id")
                for cluster_id, cluster_info in self.clusters.items():
                    for signature in cluster_info["signature_ids"]:
                        self.signature_to_cluster_id[signature] = cluster_id
                logger.info("made signature to cluster id")
        elif self.mode == "inference":
            # sampling within blocks and exhaustive flag is turned on
            self.pair_sampling_block = True
            self.pair_sampling_balanced_classes = False
            self.pair_sampling_balanced_homonym_synonym = False
            self.all_test_pairs_flag = True
            self.block_type = "s2"  # pure inference is for S2 probably?
        else:
            raise Exception(f"Unknown mode: {self.mode}")

        name_counts_loaded = False
        if isinstance(load_name_counts, dict):
            self.first_dict = load_name_counts["first_dict"]
            self.last_dict = load_name_counts["last_dict"]
            self.first_last_dict = load_name_counts["first_last_dict"]
            self.last_first_initial_dict = load_name_counts["last_first_initial_dict"]
            name_counts_loaded = True
        elif load_name_counts:
            logger.info("loading name counts")
            with open(cached_path(NAME_COUNTS_PATH), "rb") as f:
                (
                    first_dict,
                    last_dict,
                    first_last_dict,
                    last_first_initial_dict,
                ) = pickle.load(f)
            self.first_dict = first_dict
            self.last_dict = last_dict
            self.first_last_dict = first_last_dict
            self.last_first_initial_dict = last_first_initial_dict
            name_counts_loaded = True
            logger.info("loaded name counts")

        self.n_jobs = n_jobs
        self.signature_to_block = self.get_signatures_to_block()
        papers_from_signatures = set([str(signature.paper_id) for signature in self.signatures.values()])
        for paper_id, paper in self.papers.items():
            self.papers[paper_id] = paper._replace(in_signatures=str(paper_id) in papers_from_signatures)
        self.preprocess = preprocess

        if name_tuples == "filtered":
            self.name_tuples = set()
            with open(os.path.join(PROJECT_ROOT_PATH, "data", "s2and_name_tuples_filtered.txt"), "r") as f2:  # type: ignore
                for line in f2:
                    line_split = line.strip().split(",")  # type: ignore
                    self.name_tuples.add((line_split[0], line_split[1]))
        elif name_tuples is None:
            self.name_tuples = set()
            with open(os.path.join(PROJECT_ROOT_PATH, "data", "s2and_name_tuples.txt"), "r") as f2:  # type: ignore
                for line in f2:
                    line_split = line.strip().split(",")  # type: ignore
                    self.name_tuples.add((line_split[0], line_split[1]))
        else:
            self.name_tuples = name_tuples  # type: ignore

        logger.info("preprocessing papers")
        self.papers = preprocess_papers_parallel(self.papers, self.n_jobs, self.preprocess)
        logger.info("preprocessed papers")

        logger.info("preprocessing signatures")
        self.preprocess_signatures(name_counts_loaded)
        logger.info("preprocessed signatures")

    @staticmethod
    def get_full_name_for_features(signature: Signature, include_last: bool = True, include_suffix: bool = True) -> str:
        """
        Creates the full name from the name parts.

        Parameters
        ----------
        signature: Signature
            the signature to create the full name for
        include_last: bool
            whether to include the last name
        include_suffix: bool
            whether to include the suffix

        Returns
        -------
        string: the full name
        """
        first = signature.author_info_first_normalized_without_apostrophe or signature.author_info_first
        middle = signature.author_info_middle_normalized_without_apostrophe or signature.author_info_middle
        last = signature.author_info_last_normalized or signature.author_info_last
        suffix = signature.author_info_suffix_normalized or signature.author_info_suffix
        list_of_parts = [first, middle]
        if include_last:
            list_of_parts.append(last)
        if include_suffix:
            list_of_parts.append(suffix)
        name_parts = [part.strip() for part in list_of_parts if part is not None and len(part) != 0]
        return " ".join(name_parts)

    def preprocess_signatures(self, load_name_counts: bool):
        """
        Preprocess the signatures, doing lots of normalization and feature creation

        Parameters
        ----------
        load_name_counts: bool
            whether name counts were loaded (mostly just here so we can not load them when running tests)

        Returns
        -------
        nothing, modifies self.signatures
        """
        for signature_id, signature in tqdm(self.signatures.items(), desc="Preprocessing signatures"):
            # our normalization scheme is to normalize first and middle separately,
            # join them, then take the first token of the combined join
            # TODO: a lot of chinese names are not normalized correctly. they are names like Yue-Hua and Ying-Ying.
            # we need some fix for these
            first_normalized = normalize_text(signature.author_info_first or "")
            first_normalized_without_apostrophe = normalize_text(
                signature.author_info_first or "", special_case_apostrophes=True
            )

            middle_normalized = normalize_text(signature.author_info_middle or "")
            first_middle_normalized_split = (first_normalized + " " + middle_normalized).split(" ")
            if first_middle_normalized_split[0] in NAME_PREFIXES:
                first_middle_normalized_split = first_middle_normalized_split[1:]
            first_middle_normalized_split_without_apostrophe = (
                first_normalized_without_apostrophe + " " + middle_normalized
            ).split(" ")
            if first_middle_normalized_split_without_apostrophe[0] in NAME_PREFIXES:
                first_middle_normalized_split_without_apostrophe = first_middle_normalized_split_without_apostrophe[1:]

            coauthors: Optional[List[str]] = None
            if len(self.papers) != 0:
                paper = self.papers[str(signature.paper_id)]
                coauthors = [
                    author.author_name for author in paper.authors if author.position != signature.author_info_position
                ]

            signature = signature._replace(
                # need this for name counts
                author_info_first_normalized=first_middle_normalized_split[0],
                # need this for featurization
                author_info_first_normalized_without_apostrophe=first_middle_normalized_split_without_apostrophe[0],
                author_info_middle_normalized_without_apostrophe=" ".join(
                    first_middle_normalized_split_without_apostrophe[1:]
                ),
                author_info_last_normalized=normalize_text(signature.author_info_last),
                author_info_suffix_normalized=normalize_text(signature.author_info_suffix or ""),
                author_info_coauthors=set(coauthors) if coauthors is not None else None,
                author_info_coauthor_blocks=(
                    set([compute_block(author) for author in coauthors]) if coauthors is not None else None
                ),
            )

            if self.preprocess:
                affiliations = [normalize_text(affiliation) for affiliation in signature.author_info_affiliations]
                affiliations_n_grams = get_text_ngrams_words(
                    " ".join(affiliations),
                    AFFILIATIONS_STOP_WORDS,
                )

                if load_name_counts:
                    first_last_for_count = (
                        signature.author_info_first_normalized + " " + signature.author_info_last_normalized
                    ).strip()
                    first_initial = (
                        signature.author_info_first_normalized
                        if len(signature.author_info_first_normalized) > 0
                        else ""
                    )
                    last_first_initial_for_count = (signature.author_info_last_normalized + " " + first_initial).strip()
                    counts = NameCounts(
                        first=(
                            self.first_dict.get(signature.author_info_first_normalized, 1)  # type: ignore
                            if len(signature.author_info_first_normalized) > 1
                            else np.nan
                        ),
                        last=self.last_dict.get(signature.author_info_last_normalized, 1),
                        first_last=(
                            self.first_last_dict.get(first_last_for_count, 1)  # type: ignore
                            if len(signature.author_info_first_normalized) > 1
                            else np.nan
                        ),
                        last_first_initial=self.last_first_initial_dict.get(last_first_initial_for_count, 1),
                    )
                else:
                    counts = NameCounts(first=None, last=None, first_last=None, last_first_initial=None)

                signature = signature._replace(
                    author_info_full_name=ANDData.get_full_name_for_features(signature).strip(),
                    author_info_affiliations=affiliations,
                    author_info_affiliations_n_grams=affiliations_n_grams,
                    author_info_coauthor_n_grams=(
                        get_text_ngrams(" ".join(coauthors), stopwords=None, use_bigrams=True)
                        if coauthors is not None
                        else Counter()
                    ),
                    author_info_name_counts=counts,
                )

                # we need a regex to extract the 16 digits, keeping in mind the last digit could be X
                if signature.author_info_orcid is not None:
                    orcid = re.findall(ORCID_PATTERN, signature.author_info_orcid)
                    if len(orcid) > 0:
                        signature = signature._replace(author_info_orcid=orcid[0].upper().replace("-", ""))
                    else:
                        signature = signature._replace(author_info_orcid=None)

            self.signatures[signature_id] = signature

    @staticmethod
    def maybe_load_json(path_or_json: Optional[Union[str, Union[List, Dict]]]) -> Any:
        """
        Either loads a dictionary from a json file or passes through the object

        Parameters
        ----------
        path_or_json: string or Dict
            the file path or the object

        Returns
        -------
        either the loaded json, or the passed in object
        """
        if isinstance(path_or_json, str):
            with open(path_or_json) as _json_file:
                output = json.load(_json_file)
            return output
        else:
            return path_or_json

    @staticmethod
    def maybe_load_list(path_or_list: Optional[Union[str, list, Set]]) -> Optional[Union[list, Set]]:
        """
        Either loads a list from a text file or passes through the object

        Parameters
        ----------
        path_or_list: string or list
            the file path or the object

        Returns
        -------
        either the loaded list, or the passed in object
        """
        if isinstance(path_or_list, str):
            with open(path_or_list, "r") as f:
                return f.read().strip().split("\n")
        else:
            return path_or_list

    @staticmethod
    def maybe_load_dataframe(path_or_dataframe: Optional[Union[str, pd.DataFrame]]) -> Optional[pd.DataFrame]:
        """
        Either loads a dataframe from a csv file or passes through the object

        Parameters
        ----------
        path_or_dataframe: string or dataframe
            the file path or the object

        Returns
        -------
        either the loaded dataframe, or the passed in object
        """
        if type(path_or_dataframe) == str:
            return pd.read_csv(path_or_dataframe, sep=",")
        else:
            return path_or_dataframe

    @staticmethod
    def maybe_load_specter(path_or_pickle: Optional[Union[str, Dict]]) -> Optional[Dict]:
        """
        Either loads a dictionary from a pickle file or passes through the object

        Parameters
        ----------
        path_or_pickle: string or dictionary
            the file path or the object

        Returns
        -------
        either the loaded json, or the passed in object
        """
        if isinstance(path_or_pickle, str):
            with open(path_or_pickle, "rb") as _pickle_file:
                X, keys = pickle.load(_pickle_file)
            D = {}
            for i, key in enumerate(keys):
                D[key] = X[i, :]
            return D
        else:
            return path_or_pickle

    def get_original_blocks(self) -> Dict[str, List[str]]:
        """
        Gets the block dict based on the blocks provided with the dataset

        Returns
        -------
        Dict: mapping from block id to list of signatures in the block
        """
        block = {}
        for signature_id, signature in self.signatures.items():
            block_id = signature.author_info_given_block
            if block_id not in block:
                block[block_id] = [signature_id]
            else:
                block[block_id].append(signature_id)
        return block

    def get_s2_blocks(self) -> Dict[str, List[str]]:
        """
        Gets the block dict based on the blocks provided by Semantic Scholar data

        Returns
        -------
        Dict: mapping from block id to list of signatures in the block
        """
        block: Dict[str, List[str]] = {}
        for signature_id, signature in self.signatures.items():
            block_id = signature.author_info_block
            if block_id not in block:
                block[block_id] = [signature_id]
            else:
                block[block_id].append(signature_id)
        return block

    def get_blocks(self) -> Dict[str, List[str]]:
        """
        Gets the block dict

        Returns
        -------
        Dict: mapping from block id to list of signatures in the block
        """
        if self.block_type == "s2":
            return self.get_s2_blocks()
        elif self.block_type == "original":
            return self.get_original_blocks()
        else:
            raise Exception(f"Unknown block type: {self.block_type}")

    def get_constraint(
        self,
        signature_id_1: str,
        signature_id_2: str,
        low_value: Union[float, int] = 0,
        high_value: Union[float, int] = LARGE_DISTANCE,
        dont_merge_cluster_seeds: bool = True,
        incremental_dont_use_cluster_seeds: bool = False,
    ) -> Optional[float]:
        """Applies cluster_seeds and generates the default
        constraints which are:

        First we apply the passed-in cluster_seeds, then:

        (1) if not a.prefix(b) or b.prefix(a) and (a, b) not in self.name_tuples:
            distance(a, b) = high_value

        (2) if len(a_middle) > 0 and len(b_middle) > 0 and
            intersection(a_middle_chars, b_middle_chars) == 0:
            distance(a, b) = high_value

        There is currently no rule to assign low_value but it would be good
        to potentially add an ORCID rule to use low_value

        Parameters
        ----------
        signature_id_1: string
            one signature id in the pair
        signature_id_2: string
            the other signature id in the pair
        low_value: float
            value to assign to same person override
        high_value: float
            value to assign to different person overrid
        dont_merge_cluster_seeds: bool
            this flag controls whether to use cluster seeds to enforce "dont merge"
            as well as "must merge" constraints
        incremental_dont_use_cluster_seeds: bool
            Are we clustering in incremental mode? If so, don't use the cluster seeds that came with the dataset

        Returns
        -------
        float: the constraint value
        """
        first_1 = self.signatures[signature_id_1].author_info_first_normalized_without_apostrophe
        first_2 = self.signatures[signature_id_2].author_info_first_normalized_without_apostrophe
        middle_1 = self.signatures[signature_id_1].author_info_middle_normalized_without_apostrophe.split()

        paper_1 = self.papers[str(self.signatures[signature_id_1].paper_id)]
        paper_2 = self.papers[str(self.signatures[signature_id_2].paper_id)]

        orcid_1 = self.signatures[signature_id_1].author_info_orcid
        orcid_2 = self.signatures[signature_id_2].author_info_orcid

        # cluster seeds have precedence
        if (signature_id_1, signature_id_2) in self.cluster_seeds_disallow or (
            signature_id_2,
            signature_id_1,
        ) in self.cluster_seeds_disallow:
            return CLUSTER_SEEDS_LOOKUP["disallow"]
        elif (
            self.cluster_seeds_require.get(signature_id_1, -1) == self.cluster_seeds_require.get(signature_id_2, -2)
        ) and (not incremental_dont_use_cluster_seeds):
            return CLUSTER_SEEDS_LOOKUP["require"]
        elif (
            dont_merge_cluster_seeds
            and (signature_id_1 in self.cluster_seeds_require and signature_id_2 in self.cluster_seeds_require)
            and (self.cluster_seeds_require[signature_id_1] != self.cluster_seeds_require[signature_id_2])
        ):
            return CLUSTER_SEEDS_LOOKUP["disallow"]
        # orcid is a very reliable indicator: if 2 orcids are present and equal, then they are the same person
        # but if they are not equal, we can't say much
        elif orcid_1 is not None and orcid_2 is not None and orcid_1 == orcid_2:
            return low_value
        # just-in-case last name constraint: if last names are different, then disallow
        elif (
            self.signatures[signature_id_1].author_info_last_normalized
            != self.signatures[signature_id_2].author_info_last_normalized
        ):
            return high_value
        # just-in-case first initial constraint: if first initials are different, then disallow
        elif len(first_1) > 0 and len(first_2) > 0 and first_1[0] != first_2[0]:
            return high_value
        # and then language constraints
        elif (paper_1.is_reliable and paper_2.is_reliable) and (
            paper_1.predicted_language != paper_2.predicted_language
        ):
            return high_value
        # and then name based constraints
        else:
            signature_2 = self.signatures[signature_id_2]
            # either a known alias or a prefix of the other
            # if neither, then we'll say it's impossible to be the same person
            known_alias = (first_1, first_2) in self.name_tuples
            prefix = first_1.startswith(first_2) or first_2.startswith(first_1)
            if not prefix and not known_alias:
                return high_value
            # dont cluster together if there is no intersection between the sets of middle initials
            # and both sets are not empty
            elif len(middle_1) > 0:
                middle_2 = signature_2.author_info_middle_normalized_without_apostrophe.split()
                if len(middle_2) > 0:
                    overlapping_affixes = set(middle_2).intersection(middle_1).intersection(DROPPED_AFFIXES)
                    middle_1_all = [word for word in middle_1 if len(word) > 0 and word not in overlapping_affixes]
                    middle_2_all = [word for word in middle_2 if len(word) > 0 and word not in overlapping_affixes]
                    middle_1_words = set([word for word in middle_1_all if len(word) > 1])
                    middle_2_words = set([word for word in middle_2_all if len(word) > 1])
                    middle_1_firsts = set([word[0] for word in middle_1_all])
                    middle_2_firsts = set([word[0] for word in middle_2_all])
                    conflicting_initials = (
                        len(middle_1_firsts) > 0
                        and len(middle_2_firsts) > 0
                        and len(middle_1_firsts.intersection(middle_2_firsts)) == 0
                    )
                    conflicting_full_names = (
                        len(middle_1_words) > 0
                        and len(middle_2_words) > 0
                        and len(middle_1_words.intersection(middle_2_words)) == 0
                        and set("".join(middle_1_words)) != set("".join(middle_2_words))
                    )
                    if conflicting_initials or conflicting_full_names:
                        return high_value
        return None

    def get_signatures_to_block(self) -> Dict[str, str]:
        """
        Creates a dictionary mapping signature id to block key

        Returns
        -------
        Dict: the signature to block dictionary
        """
        signatures_to_block: Dict[str, str] = {}
        block_dict = self.get_blocks()
        for block_key, signatures in block_dict.items():
            for signature in signatures:
                signatures_to_block[signature] = block_key
        return signatures_to_block

    def split_blocks_helper(
        self, blocks_dict: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks

        Parameters
        ----------
        blocks_dict: Dict
            the full block dictionary

        Returns
        -------
        train/val/test block dictionaries
        """
        x = []
        y = []
        for block_id, signature in blocks_dict.items():
            x.append(block_id)
            y.append(len(signature))

        clustering_model = KMeans(
            n_clusters=self.num_clusters_for_block_size,
            random_state=self.random_seed,
        ).fit(np.array(y).reshape(-1, 1))
        y_group = clustering_model.labels_

        train_blocks, val_test_blocks, _, val_test_length = train_test_split(
            x,
            y_group,
            test_size=self.val_ratio + self.test_ratio,
            stratify=y_group,
            random_state=self.random_seed,
        )
        val_blocks, test_blocks = train_test_split(
            val_test_blocks,
            test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
            stratify=val_test_length,
            random_state=self.random_seed,
        )

        train_block_dict = {k: blocks_dict[k] for k in train_blocks}
        val_block_dict = {k: blocks_dict[k] for k in val_blocks}
        test_block_dict = {k: blocks_dict[k] for k in test_blocks}

        return train_block_dict, val_block_dict, test_block_dict

    def group_signature_helper(self, signature_list: List[str]) -> Dict[str, List[str]]:
        """
        creates a block dict containing a specific input signature list

        Parameters
        ----------
        signature_list: List
            the list of signatures to include

        Returns
        -------
        Dict: the block dict for the input signatures
        """
        block_to_signatures: Dict[str, List[str]] = {}

        for s in signature_list:
            if self.signature_to_block[s] not in block_to_signatures:
                block_to_signatures[self.signature_to_block[s]] = [s]
            else:
                block_to_signatures[self.signature_to_block[s]].append(s)
        return block_to_signatures

    def split_cluster_signatures(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks based on split type requested.
        Options for splitting are `signatures`, `blocks`, and `time`

        Returns
        -------
        train/val/test block dictionaries
        """
        blocks = self.get_blocks()
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1, "train/val/test ratio should add to 1"

        if self.unit_of_data_split == "signatures":
            signature_keys = list(self.signatures.keys())
            train_signatures, val_test_signatures = train_test_split(
                signature_keys,
                test_size=self.val_ratio + self.test_ratio,
                random_state=self.random_seed,
            )
            val_signatures, test_signatures = train_test_split(
                val_test_signatures,
                test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
                random_state=self.random_seed,
            )
            train_block_dict = self.group_signature_helper(train_signatures)
            val_block_dict = self.group_signature_helper(val_signatures)
            test_block_dict = self.group_signature_helper(test_signatures)
            return train_block_dict, val_block_dict, test_block_dict

        elif self.unit_of_data_split == "blocks":
            (
                train_block_dict,
                val_block_dict,
                test_block_dict,
            ) = self.split_blocks_helper(blocks)
            return train_block_dict, val_block_dict, test_block_dict

        elif self.unit_of_data_split == "time":
            signature_to_year = {}
            for signature_id, signature in self.signatures.items():
                # paper_id should be kept as string, so it can be matched to papers.json
                paper_id = str(signature.paper_id)
                if self.papers[paper_id].year is None:
                    signature_to_year[signature_id] = 0
                else:
                    signature_to_year[signature_id] = self.papers[paper_id].year

            train_size = int(len(signature_to_year) * self.train_ratio)
            val_size = int(len(signature_to_year) * self.val_ratio)
            signatures_sorted_by_year = [i[0] for i in (sorted(signature_to_year.items(), key=lambda x: x[1]))]

            train_signatures = signatures_sorted_by_year[0:train_size]
            val_signatures = signatures_sorted_by_year[train_size : train_size + val_size]
            test_signatures = signatures_sorted_by_year[train_size + val_size : len(signatures_sorted_by_year)]

            train_block_dict = self.group_signature_helper(train_signatures)
            val_block_dict = self.group_signature_helper(val_signatures)
            test_block_dict = self.group_signature_helper(test_signatures)
            return train_block_dict, val_block_dict, test_block_dict

        else:
            raise Exception(f"Unknown unit_of_data_split: {self.unit_of_data_split}")

    def split_cluster_signatures_fixed(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks based on a fixed block
        based split

        Returns
        -------
        train/val/test block dictionaries
        """
        blocks = self.get_blocks()

        train_block_dict: Dict[str, List[str]] = {}
        val_block_dict: Dict[str, List[str]] = {}
        test_block_dict: Dict[str, List[str]] = {}

        logger.info("split_cluster_signatures_fixed")
        if self.val_blocks is None:
            logger.info("Val blocks are None")
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
            logger.info(f"train_prob {train_prob, self.train_ratio, self.val_ratio}")
            np.random.seed(self.random_seed)
            split_prob = np.random.rand(len(self.train_blocks))
            for block_id, signature in blocks.items():
                if block_id in self.train_blocks:
                    lookup = self.train_blocks.index(block_id)
                    if split_prob[lookup] < train_prob:
                        train_block_dict[block_id] = signature
                    else:
                        val_block_dict[block_id] = signature
                elif block_id in self.test_blocks:
                    test_block_dict[block_id] = signature
        else:
            for block_id, signature in blocks.items():
                if block_id in self.train_blocks:
                    train_block_dict[block_id] = signature
                elif block_id in self.val_blocks:
                    val_block_dict[block_id] = signature
                elif block_id in self.test_blocks:
                    test_block_dict[block_id] = signature

        logger.info(f"shuffled train/val/test {len(train_block_dict), len(val_block_dict), len(test_block_dict)}")

        train_set = set(reduce(lambda x, y: x + y, train_block_dict.values()))  # type: ignore
        val_set = set(reduce(lambda x, y: x + y, val_block_dict.values()))  # type: ignore
        test_set = set(reduce(lambda x, y: x + y, test_block_dict.values()))  # type: ignore
        intersection_1 = train_set.intersection(test_set)
        intersection_2 = train_set.intersection(val_set)
        intersection_3 = val_set.intersection(test_set)
        intersection = intersection_1.union(intersection_2).union(intersection_3)

        assert len(intersection) == 0, f"Intersection between train/val/test is {intersection}"

        return train_block_dict, val_block_dict, test_block_dict

    def split_data_signatures_fixed(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks based on a fixed signature
        based split

        Returns
        -------
        train/val/test block dictionaries
        """
        train_block_dict: Dict[str, List[str]] = {}
        val_block_dict: Dict[str, List[str]] = {}
        test_block_dict: Dict[str, List[str]] = {}

        test_signatures = self.test_signatures
        logger.info("fixed signatures split")

        if self.val_signatures is None:
            train_signatures = []
            val_signatures = []
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
            np.random.seed(self.random_seed)
            split_prob = np.random.rand(len(self.train_signatures))
            for signature, p in zip(self.train_signatures, split_prob):
                if p < train_prob:
                    train_signatures.append(signature)
                else:
                    val_signatures.append(signature)
            logger.info(f"size of signatures {len(train_signatures), len(val_signatures)}")
        else:
            train_signatures = self.train_signatures
            val_signatures = self.val_signatures

        train_block_dict = self.group_signature_helper(train_signatures)
        val_block_dict = self.group_signature_helper(val_signatures)
        test_block_dict = self.group_signature_helper(test_signatures)

        return train_block_dict, val_block_dict, test_block_dict

    def split_pairs(
        self,
        train_signatures: Dict[str, List[str]],
        val_signatures: Dict[str, List[str]],
        test_signatures: Dict[str, List[str]],
    ) -> Tuple[
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
    ]:
        """
        creates pairs for the pairwise classification task

        Parameters
        ----------
        train_signatures: Dict
            the train block dict
        val_signatures: Dict
            the val block dict
        test_signatures: Dict
            the test block dict

        Returns
        -------
        train/val/test pairs, where each pair is (signature_id_1, signature_id_2, label)
        """
        assert (
            isinstance(train_signatures, dict)
            and isinstance(val_signatures, dict)
            and isinstance(test_signatures, dict)
        )
        train_pairs = self.pair_sampling(
            self.train_pairs_size,
            [],
            train_signatures,
        )
        val_pairs = (
            self.pair_sampling(
                self.val_pairs_size,
                [],
                val_signatures,
            )
            if len(val_signatures) > 0
            else []
        )

        test_pairs = self.pair_sampling(self.test_pairs_size, [], test_signatures, self.all_test_pairs_flag)

        return train_pairs, val_pairs, test_pairs

    def construct_cluster_to_signatures(
        self,
        block_dict: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        creates a dictionary mapping cluster to signatures

        Parameters
        ----------
        block_dict: Dict
            the block dict to construct cluster to signatures for

        Returns
        -------
        Dict: the dictionary mapping cluster to signatures
        """
        cluster_to_signatures = defaultdict(list)
        for signatures in block_dict.values():
            for signature in signatures:
                true_cluster_id = self.signature_to_cluster_id[signature]
                cluster_to_signatures[true_cluster_id].append(signature)

        return dict(cluster_to_signatures)

    def fixed_pairs(
        self,
    ) -> Tuple[
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
    ]:
        """
        creates pairs for the pairwise classification task from a fixed train/val/test split

        Returns
        -------
        train/val/test pairs, where each pair is (signature_id_1, signature_id_2, label)
        """
        assert (
            self.train_pairs is not None and self.test_pairs is not None
        ), "You need to pass in train and test pairs to use this function"
        self.train_pairs.loc[:, "label"] = self.train_pairs["label"].map(
            {"NO": 0, "YES": 1, "0": 0, 0: 0, "1": 1, 1: 1}
        )
        if self.val_pairs is not None:
            self.val_pairs.loc[:, "label"] = self.val_pairs["label"].map(
                {"NO": 0, "YES": 1, "0": 0, 0: 0, "1": 1, 1: 1}
            )
            train_pairs = list(self.train_pairs.to_records(index=False))
            val_pairs = list(self.val_pairs.to_records(index=False))
        else:
            np.random.seed(self.random_seed)
            # split train into train/val
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
            msk = np.random.rand(len(self.train_pairs)) < train_prob
            train_pairs = list(self.train_pairs[msk].to_records(index=False))
            val_pairs = list(self.train_pairs[~msk].to_records(index=False))
        self.test_pairs.loc[:, "label"] = self.test_pairs["label"].map({"NO": 0, "YES": 1, "0": 0, 0: 0, "1": 1, 1: 1})
        test_pairs = list(self.test_pairs.to_records(index=False))

        return train_pairs, val_pairs, test_pairs

    def all_pairs(self) -> List[Tuple[str, str, Union[int, float]]]:
        """
        creates all pairs within blocks, probably used for inference

        Returns
        -------
        all pairs, where each pair is (signature_id_1, signature_id_2, label)
        """
        all_pairs_output = self.pair_sampling(
            0,  # ignored when all_test_pairs_flag is True
            [],  # no training/test pairs
            self.get_blocks(),
            self.all_test_pairs_flag,
        )
        return all_pairs_output

    def get_full_name(self, signature_id: str) -> str:
        """
        Creates the full name from the name parts.

        Parameters
        ----------
        signature_id: str
            the signature id to create the full name for

        Returns
        -------
        string: the full name
        """
        first = self.signatures[signature_id].author_info_first
        middle = self.signatures[signature_id].author_info_middle
        last = self.signatures[signature_id].author_info_last
        suffix = self.signatures[signature_id].author_info_suffix
        name_parts = [part.strip() for part in [first, middle, last, suffix] if part is not None]
        return " ".join(name_parts)

    def pair_sampling(
        self,
        sample_size: int,
        signature_ids: List[str],
        blocks: Dict[str, List[str]],
        all_pairs: bool = False,
    ) -> List[Tuple[str, str, Union[int, float]]]:
        """
        Enumerates all pairs exhaustively, and samples pairs according to the four different strategies.

        Parameters
        ----------
        sample_size: integer
            The desired sample size
        signature_ids: list
            List of signature ids from which pairs can be sampled from.
            List must be provided if blocking is not used
        blocks: dict
            It has block ids as keys, and list of signature ids under each block as values.
            Must be provided when blocking is used
        all_pairs: bool
            Whether or not to return all pairs

        Returns
        -------
        list: list of signature pairs
        """
        assert (
            (not self.pair_sampling_block and self.pair_sampling_balanced_classes)
            or (self.pair_sampling_block and self.pair_sampling_balanced_classes)
            or (
                self.pair_sampling_block
                and not self.pair_sampling_balanced_homonym_synonym
                and not self.pair_sampling_balanced_classes
            )
        ), f"You chose sample within blocks? {self.pair_sampling_block}, sample balanced pos/neg?\
             {self.pair_sampling_balanced_classes}, sample balanced homonym/synonym?\
              {self.pair_sampling_balanced_homonym_synonym}. This is not a valid combination.\
               Not using blocks and not doing balancing is not supported, and homonym/synonym\
                balancing without pos/neg balancing is not supported"

        same_name_different_cluster: List[Tuple[str, str, Union[int, float]]] = []
        same_name_same_cluster: List[Tuple[str, str, Union[int, float]]] = []
        different_name_same_cluster: List[Tuple[str, str, Union[int, float]]] = []
        different_name_different_cluster: List[Tuple[str, str, Union[int, float]]] = []
        possible: List[Tuple[str, str, Union[int, float]]] = []

        if not self.pair_sampling_block:
            for i, s1 in enumerate(signature_ids):
                for s2 in signature_ids[i + 1 :]:
                    s1_name = self.get_full_name(s1)
                    s2_name = self.get_full_name(s2)
                    s1_cluster = self.signature_to_cluster_id[s1]
                    s2_cluster = self.signature_to_cluster_id[s2]
                    if s1_cluster == s2_cluster:
                        if s1_name == s2_name:
                            same_name_same_cluster.append((s1, s2, 1))
                        else:
                            different_name_same_cluster.append((s1, s2, 1))
                    else:
                        if s1_name == s2_name:
                            same_name_different_cluster.append((s1, s2, 0))
                        else:
                            different_name_different_cluster.append((s1, s2, 0))
        elif not self.pair_sampling_balanced_homonym_synonym and not self.pair_sampling_balanced_classes:
            for _, signatures in blocks.items():
                for i, s1 in enumerate(signatures):
                    for s2 in signatures[i + 1 :]:
                        if self.signature_to_cluster_id is not None:
                            s1_cluster = self.signature_to_cluster_id[s1]
                            s2_cluster = self.signature_to_cluster_id[s2]
                            if s1_cluster == s2_cluster:
                                possible.append((s1, s2, 1))
                            else:
                                possible.append((s1, s2, 0))
                        else:
                            possible.append((s1, s2, NUMPY_NAN))
        else:
            for _, signatures in blocks.items():
                for i, s1 in enumerate(signatures):
                    for s2 in signatures[i + 1 :]:
                        s1_name = self.get_full_name(s1)
                        s2_name = self.get_full_name(s2)
                        s1_cluster = self.signature_to_cluster_id[s1]
                        s2_cluster = self.signature_to_cluster_id[s2]
                        if s1_cluster == s2_cluster:
                            if s1_name == s2_name:
                                same_name_same_cluster.append((s1, s2, 1))
                            else:
                                different_name_same_cluster.append((s1, s2, 1))
                        else:
                            if s1_name == s2_name:
                                same_name_different_cluster.append((s1, s2, 0))
                            else:
                                different_name_different_cluster.append((s1, s2, 0))

        if all_pairs:
            if (
                self.pair_sampling_balanced_homonym_synonym
                or self.pair_sampling_balanced_classes
                or not self.pair_sampling_block
            ):
                all_pairs_output: List[Tuple[str, str, Union[int, float]]] = (
                    same_name_different_cluster
                    + same_name_same_cluster
                    + different_name_same_cluster
                    + different_name_different_cluster
                )
                return all_pairs_output
            else:
                return possible
        else:
            if self.pair_sampling_balanced_classes:
                pairs = sampling(
                    same_name_different_cluster,
                    different_name_same_cluster,
                    same_name_same_cluster,
                    different_name_different_cluster,
                    sample_size,
                    self.pair_sampling_balanced_homonym_synonym,
                    self.random_seed,
                )
            elif (
                self.pair_sampling_block
                and not self.pair_sampling_balanced_classes
                and not self.pair_sampling_balanced_homonym_synonym
            ):
                sample_size = min(len(possible), sample_size)
                pairs = random_sampling(possible, sample_size, self.random_seed)

            return pairs


def preprocess_paper_1(item: Tuple[str, Paper]) -> Tuple[str, Paper]:
    """
    helper function to perform most of the preprocessing of a paper

    Parameters
    ----------
    item: Tuple[str, Paper]
        tuple of paper id and Paper object

    Returns
    -------
    Tuple[str, Paper]: tuple of paper id and preprocessed Paper object
    """
    global global_preprocess  # type: ignore

    key, paper = item

    if paper.in_signatures:
        is_reliable, is_english, predicted_language = detect_language(paper.title)
        paper = paper._replace(is_english=is_english, predicted_language=predicted_language, is_reliable=is_reliable)
    title = normalize_text(paper.title)
    title_ngrams_words = get_text_ngrams_words(title)
    authors = [
        Author(
            position=author.position,
            author_name=normalize_text(author.author_name),
        )
        for author in paper.authors
    ]
    paper = paper._replace(title=title, title_ngrams_words=title_ngrams_words, authors=authors)

    if global_preprocess:  # type: ignore
        venue = normalize_text(paper.venue)
        journal_name = normalize_text(paper.journal_name)
        paper = paper._replace(venue=venue, journal_name=journal_name)
        if paper.in_signatures:
            title_ngrams_chars = get_text_ngrams(paper.title, use_bigrams=True)
            venue_ngrams = get_text_ngrams(paper.venue, stopwords=VENUE_STOP_WORDS, use_bigrams=True)
            journal_ngrams = get_text_ngrams(paper.journal_name, stopwords=VENUE_STOP_WORDS, use_bigrams=True)
            paper = paper._replace(
                title_ngrams_chars=title_ngrams_chars,
                venue_ngrams=venue_ngrams,
                journal_ngrams=journal_ngrams,
            )

    return (key, paper)


def preprocess_paper_2(item: Tuple[str, Paper, List[MiniPaper]]) -> Tuple[str, Paper]:
    """
    helper function to perform preprocessing of the reference details for a paper.
    Note: this happens after the main paper preprocessing has occurred.

    Parameters
    ----------
    item: Tuple[str, Paper, List[MiniPaper]]
        tuple of paper id, Paper object, and list of MiniPaper objects for the references

    Returns
    -------
    Tuple[str, Paper]: tuple of paper id and preprocessed Paper object
    """
    key, paper, reference_papers = item

    titles = " ".join(filter(None, [paper.title for paper in reference_papers]))
    venues = " ".join(filter(None, [paper.venue for paper in reference_papers]))
    journals = " ".join(filter(None, [paper.journal_name for paper in reference_papers]))

    authors: List[str] = list(
        filter(
            None,
            [author.strip() for paper in reference_papers for author in paper.authors],
        )
    )
    blocks = [compute_block(author) for author in authors]
    names = " ".join(authors)
    reference_details = (
        get_text_ngrams(names.strip(), use_bigrams=True, stopwords=None),
        get_text_ngrams(titles, use_bigrams=True),
        get_text_ngrams(
            venues + " " + journals if venues != journals else venues, stopwords=VENUE_STOP_WORDS, use_bigrams=True
        ),
        Counter(blocks),
    )
    paper = paper._replace(reference_details=reference_details)

    return (key, paper)


def _init_pool(flag: bool):
    global global_preprocess
    global_preprocess = flag


def preprocess_papers_parallel(papers_dict: Dict, n_jobs: int, preprocess: bool) -> Dict:
    """
    helper function to preprocess papers

    Parameters
    ----------
    papers_dict: Dict
        the papers dictionary
    n_jobs: int
        how many cpus to use
    preprocess: bool
        whether to do all of the preprocessing, or just a small piece of it

    Returns
    -------
    Dict: the preprocessed papers dictionary
    """
    # we still want it set in the parent for the single-CPU path
    global global_preprocess
    global_preprocess = preprocess

    output: Dict = {}
    if n_jobs > 1:
        # use spawn everywhere; on Unix this makes behaviour identical
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=n_jobs,
            initializer=_init_pool,
            initargs=(preprocess,),  # give the flag to every worker
        ) as p:
            _max = len(papers_dict)
            with tqdm(total=_max, desc="Preprocessing papers 1/2") as pbar:
                for key, value in p.imap(preprocess_paper_1, papers_dict.items(), 1000):
                    output[key] = value
                    pbar.update()
    else:
        for item in tqdm(papers_dict.items(), total=len(papers_dict), desc="Preprocessing papers 1/2"):
            k, v = preprocess_paper_1(item)
            output[k] = v

    # -------- second stage is identical: reuse the same pool setup -------
    if preprocess:
        input_2 = [
            (
                key,
                value,
                [
                    MiniPaper(
                        title=p.title,
                        venue=p.venue,
                        journal_name=p.journal_name,
                        authors=[a.author_name for a in p.authors],
                    )
                    for p in filter(None, [output.get(str(rid)) for rid in (value.references or [])])
                ],
            )
            for key, value in output.items()
        ]
        if n_jobs > 1:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                processes=n_jobs,
                initializer=_init_pool,
                initargs=(preprocess,),
            ) as p:
                _max = len(input_2)
                with tqdm(total=_max, desc="Preprocessing papers 2/2") as pbar:
                    for key, value in p.imap(preprocess_paper_2, input_2, 100):
                        output[key] = value
                        pbar.update()
        else:
            for item in tqdm(input_2, total=len(input_2), desc="Preprocessing papers 2/2"):
                k, v = preprocess_paper_2(item)
                output[k] = v

    return output

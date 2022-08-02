from typing import Optional, Union, Dict, List, Any, Tuple, Set, NamedTuple

import random
import json
import numpy as np
import pandas as pd
import logging
import pickle
import multiprocessing
from tqdm import tqdm

from collections import defaultdict, Counter

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


from s2and.consts import (
    NUMPY_NAN,
    NAME_COUNTS_PATH,
    LARGE_DISTANCE,
    CLUSTER_SEEDS_LOOKUP,
    ORPHAN_CLUSTER_KEY
)
from s2and.file_cache import cached_path
from s2and.text import (
    normalize_text,
    get_text_ngrams,
    get_text_ngrams_words,
    AFFILIATIONS_STOP_WORDS,
    VENUE_STOP_WORDS,
    NAME_PREFIXES,
)

logger = logging.getLogger("s2and")


class NameCounts(NamedTuple):
    first: Optional[int]
    last: Optional[int]
    first_last: Optional[int]
    last_first_initial: Optional[int]


class Author(NamedTuple):
    author_info_first: Optional[str]
    author_info_first_normalized_without_apostrophe: Optional[str]
    author_info_middle: Optional[str]
    author_info_middle_normalized_without_apostrophe: Optional[str]
    author_info_last_normalized: Optional[str]
    author_info_last: str
    author_info_suffix_normalized: Optional[str]
    author_info_suffix: Optional[str]
    author_info_first_normalized: Optional[str]
    author_info_middle_normalized: Optional[str]
    author_info_full_name: Optional[str]
    author_info_first_letters: Optional[set]
    author_info_affiliations: List[str]
    author_info_affiliations_joined: Optional[Counter]
    author_info_email: Optional[str]
    author_info_email_prefix: Optional[str]
    author_info_email_suffix: Optional[str]
    author_info_name_counts: Optional[NameCounts]
    author_info_position: int

class Paper(NamedTuple):
    title: str
    abstract: str
    has_abstract: Optional[bool]
    title_ngrams_words: Optional[Counter]
    abstract_ngrams_words: Optional[Counter]
    authors: List[Author]
    venue: Optional[str]
    journal_name: Optional[str]
    title_ngrams_chars: Optional[Counter]
    venue_ngrams: Optional[Counter]
    journal_ngrams: Optional[Counter]
    author_info_coauthor_n_grams: Optional[Counter]
    author_info_coauthor_email_prefix_n_grams: Optional[Counter]
    author_info_coauthor_email_suffix_n_grams: Optional[Counter]
    author_info_coauthor_affiliations_n_grams: Optional[Counter]
    year: Optional[int]
    paper_id: int
    block: Optional[str]
    # TODO: some key that represents how the papers are currently clustered so we can compare against it


class PDData:
    """
    The main class for holding our representation of an paper disambiguation data

    Input:
        papers: path to the papers information json file (or the json object)
        name: name of the dataset, used for caching computed features
        mode: 'train' or 'inference'; if 'inference', everything related to splitting will be ignored
        clusters: path to the clusters json file (or the json object)
            - a cluster may span multiple blocks, but we will only consider in-block clusters
            - there will be individual papers that definitely do not belong to any of the known clusters
              but which may or may not cluster with each other.
              papers in these clusters will all appear in clusters.json under the key ORPHAN_CLUSTER_KEY
        specter_embeddings: path to the specter embeddings pickle (or the dictionary object)
        cluster_seeds: path to the cluster seed json file (or the json object)
        altered_cluster_papers: path to the paper ids \n-separated txt file (or a list or set object)
            Clusters that these papers appear in will be marked as "altered"
        train_pairs: path to predefined train pairs csv (or the dataframe object)
        val_pairs: path to predefined val pairs csv (or the dataframe object)
        test_pairs: path to predefined test pairs csv (or the dataframe object)
        train_papers: path to predefined train papers (or the json object)
        val_papers: path to predefined val papers (or the json object)
        test_papers: path to predefined test papers (or the json object)
        unit_of_data_split: options are ("papers", "blocks", "time")
        num_clusters_for_block_size: probably leave as default,
            controls train/val/test splits based on block size
        train_ratio: training ratio of instances for clustering
        val_ratio: validation ratio of instances for clustering
        test_ratio: test ratio of instances for clustering
        train_pairs_size: number of training pairs for learning the linkage function
        val_pairs_size: number of validation pairs for fine-tuning the linkage function parameters
        test_pairs_size: number of test pairs for evaluating the linkage function
        all_test_pairs_flag: With blocking, for the linkage function evaluation task, should the test
            contain all possible pairs from test blocks, or the given number of pairs (test_pairs_size)
        random_seed: random seed
        load_name_counts: Whether or not to load name counts
        n_jobs: number of cpus to use
    """

    def __init__(
        self,
        papers: Union[str, Dict],
        name: str,
        mode: str = "train",
        clusters: Optional[Union[str, Dict]] = None,
        specter_embeddings: Optional[Union[str, Dict]] = None,
        cluster_seeds: Optional[Union[str, Dict]] = None,
        altered_cluster_papers: Optional[Union[str, List, Set]] = None,
        train_pairs: Optional[Union[str, List]] = None,
        val_pairs: Optional[Union[str, List]] = None,
        test_pairs: Optional[Union[str, List]] = None,
        train_papers: Optional[Union[str, List]] = None,
        val_papers: Optional[Union[str, List]] = None,
        test_papers: Optional[Union[str, List]] = None,
        unit_of_data_split: str = "blocks",
        num_clusters_for_block_size: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        train_pairs_size: int = 30000,
        val_pairs_size: int = 5000,
        test_pairs_size: int = 5000,
        all_test_pairs_flag: bool = False,
        random_seed: int = 1111,
        load_name_counts: Union[bool, Dict] = True,
        n_jobs: int = 1,
    ):

        logger.info("loading papers")
        self.papers = self.maybe_load_json(papers)
        
        # convert dictionary to namedtuples for memory reduction
        for paper_id, paper in self.papers.items():
            paper_id = str(paper_id)
            self.papers[paper_id] = Paper(
                title=paper.get("title", ''),
                abstract=paper.get("abstract", None),
                has_abstract=paper.get("abstract", None) not in {"", None},
                title_ngrams_words=None,
                abstract_ngrams_words=None,
                authors=[
                    Author(
                        author_info_first=author["first"],
                        author_info_first_normalized_without_apostrophe=None,
                        author_info_middle=' '.join(author["middle"]),
                        author_info_middle_normalized_without_apostrophe=None,
                        author_info_last_normalized=None,
                        author_info_last=author["last"],
                        author_info_suffix_normalized=None,
                        author_info_suffix=author["suffix"],
                        author_info_first_normalized=None,
                        author_info_middle_normalized=None,
                        author_info_full_name=None,
                        author_info_affiliations=author["affiliations"],
                        author_info_first_letters=None,
                        author_info_affiliations_joined=None,
                        author_info_email=author["email"],
                        author_info_email_prefix=None,
                        author_info_email_suffix=None,
                        author_info_name_counts=None,
                        author_info_position=author["position"],
                    )
                    for author in paper["authors"]
                ],
                venue=paper.get("venue", None),
                journal_name=paper.get('journal_name', None),
                title_ngrams_chars=None,
                venue_ngrams=None,
                journal_ngrams=None,
                author_info_coauthor_n_grams=None,
                author_info_coauthor_email_prefix_n_grams=None,
                author_info_coauthor_email_suffix_n_grams=None,
                author_info_coauthor_affiliations_n_grams=None,
                year=paper.get("year", None),
                paper_id=paper_id,
                block=paper.get("block", None),
            )
        logger.info("loaded papers")

        self.name = name
        self.mode = mode
        logger.info("loading clusters")
        self.clusters: Optional[Dict] = self.maybe_load_json(clusters)
        logger.info("loaded clusters, loading specter")
        self.specter_embeddings = self.maybe_load_specter(specter_embeddings)
        logger.info("loaded specter, loading cluster seeds")
        cluster_seeds_dict = self.maybe_load_json(cluster_seeds)
        self.altered_cluster_papers = self.maybe_load_list(altered_cluster_papers)
        self.cluster_seeds_disallow = set()
        self.cluster_seeds_require = {}
        self.max_seed_cluster_id = None
        if cluster_seeds_dict is not None:
            cluster_num = 0
            for paper_id_a, values in cluster_seeds_dict.items():
                root_added = False
                for paper_id_b, constraint_string in values.items():
                    if constraint_string == "disallow":
                        self.cluster_seeds_disallow.add((paper_id_a, paper_id_b))
                    elif constraint_string == "require":
                        if not root_added:
                            self.cluster_seeds_require[paper_id_a] = cluster_num
                            root_added = True
                        self.cluster_seeds_require[paper_id_b] = cluster_num
                cluster_num += 1
            self.max_seed_cluster_id = cluster_num
        logger.info("loaded cluster seeds")
        self.train_pairs = self.maybe_load_dataframe(train_pairs)
        self.val_pairs = self.maybe_load_dataframe(val_pairs)
        self.test_pairs = self.maybe_load_dataframe(test_pairs)
        self.train_papers = self.maybe_load_json(train_papers)
        self.val_papers = self.maybe_load_json(val_papers)
        self.test_papers = self.maybe_load_json(test_papers)
        self.unit_of_data_split = unit_of_data_split
        self.num_clusters_for_block_size = num_clusters_for_block_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1, "train/val/test ratio should add to 1"
        self.train_pairs_size = train_pairs_size
        self.val_pairs_size = val_pairs_size
        self.test_pairs_size = test_pairs_size
        self.all_test_pairs_flag = all_test_pairs_flag
        self.random_seed = random_seed

        if self.clusters is None:
            self.paper_to_cluster_id = None

        if self.mode == "train":
            if self.clusters is not None:
                self.paper_to_cluster_id = {}
                logger.info("making paper to cluster id")
                for cluster_id, cluster_info in self.clusters.items():
                    for paper_id in cluster_info["paper_ids"]:
                        self.paper_to_cluster_id[str(paper_id)] = cluster_id
                logger.info("made paper to cluster id")
        elif self.mode == "inference":
            self.all_test_pairs_flag = True
        else:
            raise Exception(f"Unknown mode: {self.mode}")

        self.load_name_counts = load_name_counts
        if isinstance(load_name_counts, dict):
            self.first_dict = load_name_counts["first_dict"]
            self.last_dict = load_name_counts["last_dict"]
            self.first_last_dict = load_name_counts["first_last_dict"]
            self.last_first_initial_dict = load_name_counts["last_first_initial_dict"]
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
            logger.info("loaded name counts")

        self.n_jobs = n_jobs
        self.paper_to_block = self.get_papers_to_block()

        logger.info("preprocessing papers")
        self.papers = preprocess_papers_parallel(self.papers, self.n_jobs)
        logger.info("preprocessed papers")

        # name counts for each author
        # has to happen after papers + authors are preprocessed
        for paper_id in self.papers.keys():
            paper = self.papers[paper_id]
            authors = [
                self.lookup_name_counts(author, load_name_counts) for author in paper.authors
            ]
            self.papers[paper_id] = paper._replace(authors=authors)

    def lookup_name_counts(self, author, load_name_counts):
        if load_name_counts:
            first_last_for_count = (
                author.author_info_first_normalized + " " + author.author_info_last_normalized
            ).strip()
            first_initial = (
                author.author_info_first_normalized
                if len(author.author_info_first_normalized) > 0
                else ""
            )
            last_first_initial_for_count = (author.author_info_last_normalized + " " + first_initial).strip()
            counts = NameCounts(
                first=self.first_dict.get(author.author_info_first_normalized, 1)
                if len(author.author_info_first_normalized) > 1
                else np.nan,
                last=self.last_dict.get(author.author_info_last_normalized, 1),
                first_last=self.first_last_dict.get(first_last_for_count, 1)
                if len(author.author_info_first_normalized) > 1
                else np.nan,
                last_first_initial=self.last_first_initial_dict.get(last_first_initial_for_count, 1),
            )
        else:
            counts = NameCounts(first=None, last=None, first_last=None, last_first_initial=None)
            
        author = author._replace(
            author_info_name_counts=counts,
        )
        return author

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

    def get_constraint(
        self,
        paper_id_1: str,
        paper_id_2: str,
        low_value: Union[float, int] = 0,
        high_value: Union[float, int] = LARGE_DISTANCE,
        dont_merge_cluster_seeds: bool = True,
        incremental_dont_use_cluster_seeds: bool = False,
    ) -> Optional[float]:
        """Applies cluster_seeds and generates the default
        constraints which are:

        We apply the passed-in cluster_seeds

        Parameters
        ----------
        paper_id_1: string
            one paper id in the pair
        paper_id_2: string
            the other paper id in the pair
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

        # TODO: do we need rules of any kind here?
        paper_1 = self.papers[str(self.papers[paper_id_1].paper_id)]
        paper_2 = self.papers[str(self.papers[paper_id_2].paper_id)]

        # cluster seeds have precedence
        if (paper_id_1, paper_id_2) in self.cluster_seeds_disallow or (
            paper_id_2,
            paper_id_1,
        ) in self.cluster_seeds_disallow:
            return CLUSTER_SEEDS_LOOKUP["disallow"]
        elif (
            self.cluster_seeds_require.get(paper_id_1, -1) == self.cluster_seeds_require.get(paper_id_2, -2)
        ) and (not incremental_dont_use_cluster_seeds):
            return CLUSTER_SEEDS_LOOKUP["require"]
        elif (
            dont_merge_cluster_seeds
            and (paper_id_1 in self.cluster_seeds_require and paper_id_2 in self.cluster_seeds_require)
            and (self.cluster_seeds_require[paper_id_1] != self.cluster_seeds_require[paper_id_2])
        ):
            return CLUSTER_SEEDS_LOOKUP["disallow"]
        else:
            return None

    def get_blocks(self) -> Dict[str, List[str]]:
        """
        Gets the block dict based on the blocks provided

        Returns
        -------
        Dict: mapping from block id to list of papers in the block
        """
        block: Dict[str, List[str]] = {}
        for paper_id, paper in self.papers.items():
            block_id = paper.block
            if block_id not in block:
                block[block_id] = [paper_id]
            else:
                block[block_id].append(paper_id)
        return block
    
    def get_s2_blocks(self) -> Dict[str, List[str]]:
        """
        Gets the block dict based on the blocks provided by Semantic Scholar data
        Returns
        -------
        Dict: mapping from block id to list of signatures in the block
        """
        block: Dict[str, List[str]] = {}
        for paper_id, paper_ID in self.papers.items():
            # TODO: replace with the name that we'll use for original S2 block
            block_id = paper_ID.author_info_block
            if block_id not in block:
                block[block_id] = [paper_id]
            else:
                block[block_id].append(paper_id)
        return block

    def get_papers_to_block(self) -> Dict[str, str]:
        """
        Creates a dictionary mapping paper id to block key

        Each paper can only belong to a single block

        Returns
        -------
        Dict: the papers to block dictionary
        """
        paper_to_block: Dict[str, str] = {}
        block_dict = self.get_blocks()
        for block_key, papers in block_dict.items():
            for paper_id in papers:
                if paper_id in paper_to_block:
                    raise ValueError(
                        f"Paper {paper_id} is in multiple blocks: {paper_to_block[paper_id]} and {block_key}"
                    )
                paper_to_block[paper_id] = block_key
        return paper_to_block

    def split_blocks_helper(
        self, blocks: Dict[str, List[str]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks, while trying to preserve
        the distribution of block sizes between the splits.

        Parameters
        ----------
        blocks: Dict
            the full block dictionary

        Returns
        -------
        train/val/test block dictionaries
        """
        x = []
        y = []
        for block_id, papers in blocks.items():
            x.append(block_id)
            y.append(len(papers))

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

        train_block_dict = {k: blocks[k] for k in train_blocks}
        val_block_dict = {k: blocks[k] for k in val_blocks}
        test_block_dict = {k: blocks[k] for k in test_blocks}

        return train_block_dict, val_block_dict, test_block_dict

    def group_paper_helper(self, paper_list: List[str]) -> Dict[str, List[str]]:
        """
        Creates a block dict containing a specific input paper list

        Parameters
        ----------
        paper_list: List
            the list of papers to include

        Returns
        -------
        Dict: the block dict for the input papers
        """
        block_to_papers: Dict[str, List[str]] = {}

        for s in paper_list:
            if self.paper_to_block[s] not in block_to_papers:
                block_to_papers[self.paper_to_block[s]] = [s]
            else:
                block_to_papers[self.paper_to_block[s]].append(s)
        return block_to_papers

    def split_cluster_papers(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks based on split type requested.
        Options for splitting are `papers`, `blocks`, and `time`

        Returns
        -------
        train/val/test block dictionaries
        """
        blocks = self.get_blocks()
        

        if self.unit_of_data_split == "papers":
            paper_keys = list(self.papers.keys())
            train_papers, val_test_papers = train_test_split(
                paper_keys,
                test_size=self.val_ratio + self.test_ratio,
                random_state=self.random_seed,
            )
            val_papers, test_papers = train_test_split(
                val_test_papers,
                test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
                random_state=self.random_seed,
            )
            train_block_dict = self.group_paper_helper(train_papers)
            val_block_dict = self.group_paper_helper(val_papers)
            test_block_dict = self.group_paper_helper(test_papers)
            return train_block_dict, val_block_dict, test_block_dict

        elif self.unit_of_data_split == "blocks":
            (
                train_block_dict,
                val_block_dict,
                test_block_dict,
            ) = self.split_blocks_helper(blocks)
            return train_block_dict, val_block_dict, test_block_dict

        elif self.unit_of_data_split == "time":
            paper_to_year = {}
            for paper_id, paper in self.papers.items():
                # paper_id should be kept as string, so it can be matched to papers.json
                paper_id = str(paper.paper_id)
                if paper.year is None:
                    paper_to_year[paper_id] = 0
                else:
                    paper_to_year[paper_id] = paper.year

            train_size = int(len(paper_to_year) * self.train_ratio)
            val_size = int(len(paper_to_year) * self.val_ratio)
            papers_sorted_by_year = [i[0] for i in (sorted(paper_to_year.items(), key=lambda x: x[1]))]

            train_papers = papers_sorted_by_year[0:train_size]
            val_papers = papers_sorted_by_year[train_size : train_size + val_size]
            test_papers = papers_sorted_by_year[train_size + val_size : len(papers_sorted_by_year)]

            train_block_dict = self.group_paper_helper(train_papers)
            val_block_dict = self.group_paper_helper(val_papers)
            test_block_dict = self.group_paper_helper(test_papers)
            return train_block_dict, val_block_dict, test_block_dict

        else:
            raise Exception(f"Unknown unit_of_data_split: {self.unit_of_data_split}")

    def split_data_papers_fixed(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Splits the block dict into train/val/test blocks based on a fixed paper
        based split

        Returns
        -------
        train/val/test block dictionaries
        """
        train_block_dict: Dict[str, List[str]] = {}
        val_block_dict: Dict[str, List[str]] = {}
        test_block_dict: Dict[str, List[str]] = {}

        test_papers = self.test_papers
        logger.info("fixed papers split")

        if self.val_papers is None:
            train_papers = []
            val_papers = []
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
            np.random.seed(self.random_seed)
            split_prob = np.random.rand(len(self.train_papers))
            for paper, p in zip(self.train_papers, split_prob):
                if p < train_prob:
                    train_papers.append(paper)
                else:
                    val_papers.append(paper)
            logger.info(f"size of papers {len(train_papers), len(val_papers)}")
        else:
            train_papers = self.train_papers
            val_papers = self.val_papers

        train_block_dict = self.group_paper_helper(train_papers)
        val_block_dict = self.group_paper_helper(val_papers)
        test_block_dict = self.group_paper_helper(test_papers)

        return train_block_dict, val_block_dict, test_block_dict

    def split_pairs(
        self,
        train_papers_dict: Dict[str, List[str]],
        val_papers_dict: Dict[str, List[str]],
        test_papers_dict: Dict[str, List[str]],
    ) -> Tuple[
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
        List[Tuple[str, str, Union[int, float]]],
    ]:
        """
        creates pairs for the pairwise classification task

        Parameters
        ----------
        train_papers_dict: Dict
            the train block dict
        val_papers_dict: Dict
            the val block dict
        test_papers_dict: Dict
            the test block dict

        Returns
        -------
        train/val/test pairs, where each pair is (paper_id_1, paper_id_2, label)
        """
        assert (
            isinstance(train_papers_dict, dict)
            and isinstance(val_papers_dict, dict)
            and isinstance(test_papers_dict, dict)
        )
        train_pairs = self.pair_sampling(
            self.train_pairs_size,
            train_papers_dict,
        )
        val_pairs = (
            self.pair_sampling(
                self.val_pairs_size,
                val_papers_dict,
            )
            if len(val_papers_dict) > 0
            else []
        )

        test_pairs = self.pair_sampling(self.test_pairs_size, test_papers_dict, self.all_test_pairs_flag)

        return train_pairs, val_pairs, test_pairs

    def construct_cluster_to_papers(
        self,
        block_dict: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        creates a dictionary mapping cluster to papers

        Parameters
        ----------
        block_dict: Dict
            the block dict to construct cluster to papers for

        Returns
        -------
        Dict: the dictionary mapping cluster to papers
        """
        cluster_to_papers = defaultdict(list)
        for papers in block_dict.values():
            for paper in papers:
                true_cluster_id = self.paper_to_cluster_id[paper]
                cluster_to_papers[true_cluster_id].append(paper)

        return dict(cluster_to_papers)

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
        train/val/test pairs, where each pair is (paper_id_1, paper_id_2, label)
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
        all pairs, where each pair is (paper_id_1, paper_id_2, label)
        """
        all_pairs_output = self.pair_sampling(
            0,  # ignored when all_test_pairs_flag is True
            self.get_blocks(),
            self.all_test_pairs_flag,
        )
        return all_pairs_output

    def pair_sampling(
        self,
        sample_size: int,
        blocks: Dict[str, List[str]],
        all_pairs: bool = False,
    ) -> List[Tuple[str, str, Union[int, float]]]:
        """
        Enumerates all pairs exhaustively, and samples pairs according to the four different strategies.
        
        Note: we don't know the label when both of papers have the cluster ORPHAN_CLUSTER_KEY. 
        But ("orphan", any other cluster) is allowed and is always a negative by definition

        Parameters
        ----------
        sample_size: integer
            The desired sample size
        paper_ids: list
            List of paper ids from which pairs can be sampled from.
            List must be provided if blocking is not used
        blocks: dict
            It has block ids as keys, and list of paper ids under each block as values.
            Must be provided when blocking is used
        all_pairs: bool
            Whether or not to return all pairs

        Returns
        -------
        list: list of paper pairs
        """

        possible: List[Tuple[str, str, Union[int, float]]] = []

        for _, papers in blocks.items():
            for i, s1 in enumerate(papers):
                for s2 in papers[i + 1 :]:
                    if self.paper_to_cluster_id is not None:
                        s1_cluster = self.paper_to_cluster_id[s1]
                        s2_cluster = self.paper_to_cluster_id[s2]
                        if s1_cluster == s2_cluster:
                            if s1_cluster != ORPHAN_CLUSTER_KEY:
                                possible.append((s1, s2, 1))
                        else:
                            possible.append((s1, s2, 0))
                    else:
                        possible.append((s1, s2, NUMPY_NAN))

        if all_pairs:
            pairs = possible
        else:
            random.seed(self.random_seed)
            pairs = random.sample(possible, min(len(possible), sample_size))
        return pairs


def get_full_name_for_features(author: Author, include_last: bool = True, include_suffix: bool = True) -> str:
    """
    Creates the full name from the name parts.

    Parameters
    ----------
    authir: Author
        the author to create the full name for
    include_last: bool
        whether to include the last name
    include_suffix: bool
        whether to include the suffix

    Returns
    -------
    string: the full name
    """
    first = author.author_info_first_normalized_without_apostrophe or author.author_info_first
    middle = author.author_info_middle_normalized_without_apostrophe or author.author_info_middle
    last = author.author_info_last_normalized or author.author_info_last
    suffix = author.author_info_suffix_normalized or author.author_info_suffix
    list_of_parts = [first, middle]
    if include_last:
        list_of_parts.append(last)
    if include_suffix:
        list_of_parts.append(suffix)
    name_parts = [part.strip() for part in list_of_parts if part is not None and len(part) != 0]
    return " ".join(name_parts)


def preprocess_authors(author):
    """
    Preprocess the authors, doing lots of normalization and feature creation
    """

    # our normalization scheme is to normalize first and middle separately,
    # join them, then take the first token of the combined join
    first_normalized = normalize_text(author.author_info_first or "")
    first_normalized_without_apostrophe = normalize_text(
        author.author_info_first or "", special_case_apostrophes=True
    )

    middle_normalized = normalize_text(author.author_info_middle or "")
    first_middle_normalized_split = (first_normalized + " " + middle_normalized).split(" ")
    if first_middle_normalized_split[0] in NAME_PREFIXES:
        first_middle_normalized_split = first_middle_normalized_split[1:]
    first_middle_normalized_split_without_apostrophe = (
        first_normalized_without_apostrophe + " " + middle_normalized
    ).split(" ")
    if first_middle_normalized_split_without_apostrophe[0] in NAME_PREFIXES:
        first_middle_normalized_split_without_apostrophe = first_middle_normalized_split_without_apostrophe[1:]
    author_info_last_normalized = normalize_text(author.author_info_last or "")

    author = author._replace(
        author_info_first_normalized=first_middle_normalized_split[0],
        author_info_first_normalized_without_apostrophe=first_middle_normalized_split_without_apostrophe[0],
        author_info_middle_normalized=" ".join(first_middle_normalized_split[1:]),
        author_info_middle_normalized_without_apostrophe=" ".join(
            first_middle_normalized_split_without_apostrophe[1:]
        ),
        author_info_last_normalized=author_info_last_normalized,
        author_info_suffix_normalized=normalize_text(author.author_info_suffix or ""),
    )

    author_info_first_letters = set()
    if len(first_normalized_without_apostrophe) > 0:
        author_info_first_letters.add(first_normalized_without_apostrophe[0])
    if len(author_info_last_normalized) > 0:
        author_info_first_letters.add(author_info_last_normalized[0])

    affiliations = [normalize_text(affiliation) for affiliation in author.author_info_affiliations]
    affiliations_joined = " ".join(affiliations)

    if author.author_info_email is not None and len(author.author_info_email) > 0:
        email = author.author_info_email if "@" in author.author_info_email else author.author_info_email + "@MISSING"
        split_email = email.split("@")
        author_info_email_prefix = "".join(split_email[:-1])
        author_info_email_suffix = split_email[-1]
    else:
        author_info_email_prefix = ''
        author_info_email_suffix = ''

    author = author._replace(
        author_info_full_name=get_full_name_for_features(author).strip(),
        author_info_affiliations=affiliations,
        author_info_first_letters=author_info_first_letters,
        author_info_affiliations_joined=affiliations_joined,
        author_info_email_prefix=author_info_email_prefix,
        author_info_email_suffix=author_info_email_suffix,
    )
    
    return author


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

    key, paper = item

    title = normalize_text(paper.title)
    abstract = normalize_text(paper.abstract)
    title_ngrams_words = get_text_ngrams_words(title)
    abstract_ngrams_words = get_text_ngrams_words(abstract)
    authors = [
        preprocess_authors(author) for author in paper.authors
    ]
    paper = paper._replace(title=title, title_ngrams_words=title_ngrams_words, abstract_ngrams_words=abstract_ngrams_words, authors=authors)
    venue = normalize_text(paper.venue)
    journal_name = normalize_text(paper.journal_name)
    paper = paper._replace(venue=venue, journal_name=journal_name)
    title_ngrams_chars = get_text_ngrams(paper.title, use_bigrams=True)
    venue_ngrams = get_text_ngrams(paper.venue, stopwords=VENUE_STOP_WORDS, use_bigrams=True)
    journal_ngrams = get_text_ngrams(paper.journal_name, stopwords=VENUE_STOP_WORDS, use_bigrams=True)
    author_info_coauthor_n_grams = get_text_ngrams(" ".join([i.author_info_full_name for i in authors]), stopwords=None, use_unigrams=True, use_bigrams=True)
    author_info_coauthor_email_prefix_n_grams = get_text_ngrams(" ".join([i.author_info_email_prefix for i in authors]), stopwords=None, use_unigrams=True, use_bigrams=True)
    author_info_coauthor_email_suffix_n_grams = get_text_ngrams(" ".join([i.author_info_email_suffix for i in authors]), stopwords=None, use_bigrams=True)
    affils = [i.author_info_affiliations_joined for i in authors]
    author_info_coauthor_affiliations_n_grams = get_text_ngrams(" ".join(affils), stopwords=AFFILIATIONS_STOP_WORDS, use_bigrams=True)
    paper = paper._replace(
        title_ngrams_chars=title_ngrams_chars,
        venue_ngrams=venue_ngrams,
        journal_ngrams=journal_ngrams,
        author_info_coauthor_n_grams=author_info_coauthor_n_grams,
        author_info_coauthor_email_prefix_n_grams=author_info_coauthor_email_prefix_n_grams,
        author_info_coauthor_email_suffix_n_grams=author_info_coauthor_email_suffix_n_grams,
        author_info_coauthor_affiliations_n_grams=author_info_coauthor_affiliations_n_grams,
    )

    return (key, paper)


def preprocess_papers_parallel(papers_dict: Dict, n_jobs: int) -> Dict:
    """
    helper function to preprocess papers

    Parameters
    ----------
    papers_dict: Dict
        the papers dictionary
    n_jobs: int
        how many cpus to use

    Returns
    -------
    Dict: the preprocessed papers dictionary
    """

    output = {}
    if n_jobs > 1:
        with multiprocessing.Pool(processes=n_jobs) as p:
            _max = len(papers_dict)
            with tqdm(total=_max, desc="Preprocessing papers") as pbar:
                for key, value in p.imap(preprocess_paper_1, papers_dict.items(), 1000):
                    output[key] = value
                    pbar.update()
    else:
        for item in tqdm(papers_dict.items(), total=len(papers_dict), desc="Preprocessing papers"):
            result = preprocess_paper_1(item)
            output[result[0]] = result[1]

    return output


if __name__ == "__main__":
    pddata = PDData(papers='data/test/test_papers.json', clusters='data/test/test_clusters.json', name='test')
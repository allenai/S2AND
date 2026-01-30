from typing import Tuple, List, Union, Dict, Callable, Any, Optional

import os
import json
import tempfile
import orjson
import numpy as np
import functools
import logging
from collections import Counter
import threading
import queue
import time

from tqdm import tqdm

from s2and.data import ANDData
from s2and.mp import UniversalPool
from s2and import feature_port
from s2and.consts import (
    CACHE_ROOT,
    NUMPY_NAN,
    FEATURIZER_VERSION,
    LARGE_INTEGER,
    DEFAULT_CHUNK_SIZE,
)
from s2and.text import (
    equal,
    equal_middle,
    diff,
    name_counts,
    TEXT_FUNCTIONS,
    name_text_features,
    jaccard,
    counter_jaccard,
    cosine_sim,
)

logger = logging.getLogger("s2and")

TupleOfArrays = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]

CACHED_FEATURES: Dict[str, Dict[str, Any]] = {}

# Environment configuration caches (read once per process for consistency)
_ENV_TRUE_VALUES = {"1", "true", "yes"}
_USE_RUST_FEATURIZER_CACHE: Optional[bool] = None
_USE_RUST_BATCH_CACHE: Optional[bool] = None
_RUST_BATCH_THRESHOLD_CACHE: Optional[int] = None


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in _ENV_TRUE_VALUES


def _use_rust_featurizer() -> bool:
    global _USE_RUST_FEATURIZER_CACHE
    if _USE_RUST_FEATURIZER_CACHE is None:
        _USE_RUST_FEATURIZER_CACHE = _env_flag("S2AND_USE_RUST_FEATURIZER", "1")
    return _USE_RUST_FEATURIZER_CACHE


def _use_rust_batch() -> bool:
    global _USE_RUST_BATCH_CACHE
    if _USE_RUST_BATCH_CACHE is None:
        _USE_RUST_BATCH_CACHE = _env_flag("S2AND_RUST_BATCH", "1")
    return _USE_RUST_BATCH_CACHE


def _rust_batch_threshold() -> int:
    global _RUST_BATCH_THRESHOLD_CACHE
    if _RUST_BATCH_THRESHOLD_CACHE is None:
        threshold_str = os.environ.get("S2AND_RUST_BATCH_THRESHOLD", "0")
        try:
            threshold = int(threshold_str)
        except ValueError:
            threshold = 0
        _RUST_BATCH_THRESHOLD_CACHE = max(0, threshold)
    return _RUST_BATCH_THRESHOLD_CACHE


# ── constants for cache writes ───────────────────────────
INCREMENTAL_WRITE_THRESHOLD = 1000  # only write incrementally if we have at least this many new features
BACKGROUND_WRITE_INTERVAL = 30  # seconds between background writes


class BackgroundCacheWriter:
    """Async background writer to reduce cache write bottleneck"""

    def __init__(self):
        self.write_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None

    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self._writer_worker, daemon=True)
            self.worker_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def queue_write(self, featurizer_info, cached_features, dataset_name):
        """Queue a cache write operation"""
        self.write_queue.put((featurizer_info, cached_features.copy(), dataset_name))

    def _writer_worker(self):
        """Background worker that processes write queue"""
        while not self.stop_event.is_set():
            try:
                # Process all queued writes
                writes_processed = 0
                while not self.write_queue.empty() and writes_processed < 10:  # Batch limit
                    try:
                        featurizer_info, cached_features, dataset_name = self.write_queue.get_nowait()
                        if len(cached_features.get("__new_features__", {})) > 0:
                            featurizer_info.write_cache(cached_features, dataset_name, incremental=True)
                        writes_processed += 1
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.warning(f"Background cache write failed: {e}")

                # Sleep before next batch
                time.sleep(BACKGROUND_WRITE_INTERVAL)

            except Exception as e:
                logger.error(f"Background writer error: {e}")
                time.sleep(5)


# Global background writer
_background_writer = BackgroundCacheWriter()


class FeaturizationInfo:
    """
    Class to store information about how to generate and cache features

    Inputs:
        features_to_use: List[str[]]
            list of feature types to use
        featurizer_version: int
            What version of the featurizer we are on. This should be
            incremented when changing how features are computed so that a new cache
            is created
    """

    def __init__(
        self,
        features_to_use: List[str] = [
            "name_similarity",
            "affiliation_similarity",
            "email_similarity",
            "coauthor_similarity",
            "venue_similarity",
            "year_diff",
            "title_similarity",
            "reference_features",
            "misc_features",
            "name_counts",
            "embedding_similarity",
            "journal_similarity",
            "advanced_name_similarity",
        ],
        featurizer_version: int = FEATURIZER_VERSION,
    ):
        self.features_to_use = features_to_use

        self.feature_group_to_index = {
            "name_similarity": [0, 1, 2, 3, 4, 5],
            "affiliation_similarity": [6],
            "email_similarity": [7, 8],
            "coauthor_similarity": [9, 10, 11],
            "venue_similarity": [12],
            "year_diff": [13],
            "title_similarity": [14, 15],
            "reference_features": [16, 17, 18, 19, 20, 21],
            "misc_features": [22, 23, 24, 25, 26],
            "name_counts": [27, 28, 29, 30, 31, 32],
            "embedding_similarity": [33],
            "journal_similarity": [34],
            "advanced_name_similarity": [35, 36, 37, 38],
        }

        self.number_of_features = max(functools.reduce(max, self.feature_group_to_index.values())) + 1  # type: ignore

        lightgbm_monotone_constraints = {
            "name_similarity": ["1", "1", "1", "0", "0", "0"],
            "affiliation_similarity": ["0"],
            "email_similarity": ["1", "1"],
            "coauthor_similarity": ["1", "0", "1"],
            "venue_similarity": ["0"],
            "year_diff": ["-1"],
            "title_similarity": ["1", "1"],
            "reference_features": ["1", "1", "1", "1", "1", "1"],
            "misc_features": ["0", "0", "0", "0", "0"],
            "name_counts": ["0", "-1", "-1", "-1", "0", "-1"],
            "embedding_similarity": ["0"],
            "journal_similarity": ["0"],
            "advanced_name_similarity": ["0", "0", "0", "0"],
        }

        self.lightgbm_monotone_constraints = ",".join(
            [
                ",".join(constraints)
                for feature_category, constraints in lightgbm_monotone_constraints.items()
                if feature_category in features_to_use
            ]
        )
        self.nameless_lightgbm_monotone_constraints = ",".join(
            [
                ",".join(constraints)
                for feature_category, constraints in lightgbm_monotone_constraints.items()
                if feature_category in features_to_use
                and feature_category not in {"advanced_name_similarity", "name_similarity", "name_counts"}
            ]
        )

        # NOTE: Increment this anytime a change is made to the featurization logic
        self.featurizer_version = featurizer_version

    def get_feature_names(self) -> List[str]:
        """
        Gets all of the feature names

        Returns
        -------
        List[string]: List of all the features names
        """
        feature_names = []

        # name features
        if "name_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "first_names_equal",
                    "middle_initials_overlap",
                    "middle_names_equal",
                    "middle_one_missing",
                    "single_char_first",
                    "single_char_middle",
                ]
            )

        # affiliation features
        if "affiliation_similarity" in self.features_to_use:
            feature_names.append("affiliation_overlap")

        # email features
        if "email_similarity" in self.features_to_use:
            feature_names.extend(["email_prefix_equal", "email_suffix_equal"])

        # co author features
        if "coauthor_similarity" in self.features_to_use:
            feature_names.extend(
                [
                    "coauthor_overlap",
                    "coauthor_similarity",
                    "coauthor_match",
                ]
            )

        # venue features
        if "venue_similarity" in self.features_to_use:
            feature_names.append("venue_overlap")

        # year features
        if "year_diff" in self.features_to_use:
            feature_names.append("year_diff")

        # title features
        if "title_similarity" in self.features_to_use:
            feature_names.extend(["title_overlap_words", "title_overlap_chars"])

        # reference features
        if "reference_features" in self.features_to_use:
            feature_names.extend(
                [
                    "references_authors_overlap",
                    "references_titles_overlap",
                    "references_venues_overlap",
                    "references_author_blocks_jaccard",
                    "references_self_citation",
                    "references_overlap",
                ]
            )

        # position features
        if "misc_features" in self.features_to_use:
            feature_names.extend(
                ["position_diff", "abstract_count", "english_count", "same_language", "language_reliability_count"]
            )

        # name count features
        if "name_counts" in self.features_to_use:
            feature_names.extend(
                [
                    "first_name_count_min",
                    "last_first_name_count_min",
                    "last_name_count_min",
                    "last_first_initial_count_min",
                    "first_name_count_max",
                    "last_first_name_count_max",
                ]
            )

        # specter features
        if "embedding_similarity" in self.features_to_use:
            feature_names.append("specter_cosine_sim")

        if "journal_similarity" in self.features_to_use:
            feature_names.append("journal_overlap")

        if "advanced_name_similarity" in self.features_to_use:
            similarity_names = [func[1] for func in TEXT_FUNCTIONS]
            feature_names.extend(similarity_names)

        return feature_names

    @staticmethod
    def feature_cache_key(signature_pair: Tuple) -> str:
        """
        returns the key in the feature cache dictionary for a signature pair

        Parameters
        ----------
        signature_pair: Tuple[string]
            pair of signature ids

        Returns
        -------
        string: the cache key
        """
        return signature_pair[0] + "___" + signature_pair[1]

    def cache_directory(self, dataset_name: str) -> str:
        """
        returns the cache directory for this dataset and featurizer version

        Parameters
        ----------
        dataset_name: string
            the name of the dataset

        Returns
        -------
        string: the cache directory
        """
        return os.path.join(CACHE_ROOT, dataset_name, str(self.featurizer_version))

    def cache_file_path(self, dataset_name: str) -> str:
        """
        returns the file path for the features cache

        Parameters
        ----------
        dataset_name: string
            the name of the dataset

        Returns
        -------
        string: the full file path for the features cache file
        """
        return os.path.join(
            self.cache_directory(dataset_name),
            "all_features.json",
        )

    def write_cache(self, cached_features: Dict, dataset_name: str, incremental: bool = False):
        """
        Writes the cached features to the features cache file

        Parameters
        ----------
        cached_features: Dict
            the features, keyed by signature pair
        dataset_name: str
            the name of the dataset
        incremental: bool
            if True, only write new features from __new_features__ key

        Returns
        -------
        nothing, writes the cache file
        """
        path = self.cache_file_path(dataset_name)

        if incremental and "__new_features__" in cached_features:
            # Load existing cache and merge with new features
            existing_cache: Dict[str, Any] = {"features": {}}
            if os.path.exists(path):
                try:
                    with open(path, "rb") as fh:
                        existing_cache = orjson.loads(fh.read())
                except (ValueError, orjson.JSONDecodeError):
                    logger.warning(f"Could not load existing cache at {path}, creating new cache")
                    existing_cache = {"features": {}}

            # CONCURRENCY RISK: Another process could write between load and write
            # This could cause loss of features computed by other processes

            # Merge new features
            existing_cache["features"].update(cached_features["__new_features__"])
            existing_cache["features_to_use"] = cached_features.get("features_to_use", [])
            features_to_write = existing_cache

            # Clear the new features buffer
            cached_features["__new_features__"] = {}
        else:
            features_to_write = cached_features

        # ---- atomic write using temp file with retry logic ----
        max_retries = 3
        for attempt in range(max_retries):
            try:
                json_bytes = orjson.dumps(features_to_write, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2)
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(path)) as tmp:
                    tmp.write(json_bytes.decode("utf-8"))
                    tmp_path = tmp.name

                # atomic; old cache either stays or is replaced in a single syscall
                os.replace(tmp_path, path)
                break  # Success, exit retry loop
            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Cache write attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Cache write failed after {max_retries} attempts: {e}")
                    raise


NUM_FEATURES = FeaturizationInfo().number_of_features


def _single_pair_featurize(work_input: Tuple[str, str], index: int = -1) -> Tuple[List[Union[int, float]], int]:
    """
    Creates the features array for a single signature pair
    NOTE: This function uses a global variable to support faster multiprocessing. That means that this function
    should only be called from the many_pairs_featurize function below (or if you have carefully set your own global
    variable)

    Parameters
    ----------
    work_input: Tuple[str, str]
        pair of signature ids
    index: int
        the index of the pair in the list of all pairs,
        used to keep track of cached features

    Returns
    -------
    Tuple: tuple of the features array, and the index, which is simply passed through
    """
    global global_dataset

    features = []

    use_rust = _use_rust_featurizer()
    if use_rust and feature_port.s2and_rust is not None:
        features = feature_port.featurize_pair_rust(global_dataset, work_input[0], work_input[1])  # type: ignore
        return features, index

    signature_1 = global_dataset.signatures[work_input[0]]  # type: ignore
    signature_2 = global_dataset.signatures[work_input[1]]  # type: ignore

    paper_id_1 = signature_1.paper_id
    paper_id_2 = signature_2.paper_id

    paper_1 = global_dataset.papers[str(paper_id_1)]  # type: ignore
    paper_2 = global_dataset.papers[str(paper_id_2)]  # type: ignore

    features.extend(
        [
            equal(
                signature_1.author_info_first_normalized_without_apostrophe,
                signature_2.author_info_first_normalized_without_apostrophe,
            ),
            counter_jaccard(
                Counter(
                    [
                        p[0]
                        for p in signature_1.author_info_middle_normalized_without_apostrophe.split(" ")
                        if len(p) > 0
                    ]
                ),
                Counter(
                    [
                        p[0]
                        for p in signature_2.author_info_middle_normalized_without_apostrophe.split(" ")
                        if len(p) > 0
                    ]
                ),
            ),
            equal_middle(
                signature_1.author_info_middle_normalized_without_apostrophe,
                signature_2.author_info_middle_normalized_without_apostrophe,
            ),
            (
                len(signature_1.author_info_middle_normalized_without_apostrophe) == 0
                and len(signature_2.author_info_middle_normalized_without_apostrophe) != 0
            )
            or (
                len(signature_2.author_info_middle_normalized_without_apostrophe) == 0
                and len(signature_1.author_info_middle_normalized_without_apostrophe) != 0
            ),
            len(signature_1.author_info_first_normalized_without_apostrophe) == 1
            or len(signature_2.author_info_first_normalized_without_apostrophe) == 1,
            any(len(middle) == 1 for middle in signature_1.author_info_middle_normalized_without_apostrophe.split(" "))
            or any(
                len(middle) == 1 for middle in signature_2.author_info_middle_normalized_without_apostrophe.split(" ")
            ),
        ]
    )

    features.append(
        counter_jaccard(
            signature_1.author_info_affiliations_n_grams,
            signature_2.author_info_affiliations_n_grams,
        )
    )

    email_prefix_1: Optional[str] = None
    email_prefix_2: Optional[str] = None
    email_suffix_1: Optional[str] = None
    email_suffix_2: Optional[str] = None
    if (
        signature_1.author_info_email is not None
        and len(signature_1.author_info_email) > 0
        and signature_2.author_info_email is not None
        and len(signature_2.author_info_email) > 0
    ):
        email_1 = signature_1.author_info_email
        email_2 = signature_2.author_info_email
        email_1 = email_1 if "@" in email_1 else email_1 + "@MISSING"
        email_2 = email_2 if "@" in email_2 else email_2 + "@MISSING"
        split_email_1 = email_1.split("@")
        split_email_2 = email_2.split("@")
        email_prefix_1 = "".join(split_email_1[:-1]).strip(".").lower()
        email_prefix_2 = "".join(split_email_2[:-1]).strip(".").lower()
        email_suffix_1 = split_email_1[-1].strip(".").lower()
        email_suffix_2 = split_email_2[-1].strip(".").lower()

    features.extend(
        [
            (
                email_prefix_1 == email_prefix_2
                if email_prefix_1 is not None and email_prefix_2 is not None
                else NUMPY_NAN
            ),
            (
                email_suffix_1 == email_suffix_2
                if email_suffix_1 is not None and email_suffix_2 is not None
                else NUMPY_NAN
            ),
        ]
    )

    features.extend(
        [
            jaccard(signature_1.author_info_coauthor_blocks, signature_2.author_info_coauthor_blocks),
            counter_jaccard(
                signature_1.author_info_coauthor_n_grams,
                signature_2.author_info_coauthor_n_grams,
                denominator_max=5000,
            ),
            jaccard(signature_1.author_info_coauthors, signature_2.author_info_coauthors),
        ]
    )

    features.append(counter_jaccard(paper_1.venue_ngrams, paper_2.venue_ngrams))

    features.append(
        np.minimum(
            diff(
                paper_1.year if paper_1.year is not None and paper_1.year > 0 else None,
                paper_2.year if paper_2.year is not None and paper_2.year > 0 else None,
            ),
            50,
        )
    )  # magic number!

    features.extend(
        [
            counter_jaccard(paper_1.title_ngrams_words, paper_2.title_ngrams_words),
            counter_jaccard(paper_1.title_ngrams_chars, paper_2.title_ngrams_chars),
        ]
    )

    # Reference-derived features: optionally disabled
    references_1 = set(paper_1.references or [])
    references_2 = set(paper_2.references or [])
    compute_ref = bool(getattr(global_dataset, "compute_reference_features", False))  # type: ignore
    if compute_ref and paper_1.reference_details is not None and paper_2.reference_details is not None:
        features.extend(
            [
                counter_jaccard(paper_1.reference_details[0], paper_2.reference_details[0], denominator_max=5000),
                counter_jaccard(paper_1.reference_details[1], paper_2.reference_details[1]),
                counter_jaccard(paper_1.reference_details[2], paper_2.reference_details[2]),
                counter_jaccard(paper_1.reference_details[3], paper_2.reference_details[3]),
                int(paper_id_2 in references_1 or paper_id_1 in references_2),
                jaccard(references_1, references_2),
            ]
        )
    else:
        # When reference features are not computed, fill with NaNs to preserve shape
        features.extend([NUMPY_NAN, NUMPY_NAN, NUMPY_NAN, NUMPY_NAN, NUMPY_NAN, NUMPY_NAN])

    english_or_unknown_count = int(paper_1.predicted_language in {"en", "un"}) + int(
        paper_2.predicted_language in {"en", "un"}
    )

    features.extend(
        [
            np.minimum(
                diff(
                    signature_1.author_info_position,
                    signature_2.author_info_position,
                ),
                50,
            ),
            int(paper_1.has_abstract) + int(paper_2.has_abstract),
            english_or_unknown_count,
            paper_1.predicted_language == paper_2.predicted_language,
            int(paper_1.is_reliable) + int(paper_2.is_reliable),
        ]
    )

    features.extend(
        name_counts(
            signature_1.author_info_name_counts,
            signature_2.author_info_name_counts,
        )
    )

    specter_1 = None
    specter_2 = None
    if english_or_unknown_count == 2 and global_dataset.specter_embeddings is not None:  # type: ignore
        if str(paper_id_1) in global_dataset.specter_embeddings:  # type: ignore
            specter_1 = global_dataset.specter_embeddings[str(paper_id_1)]  # type: ignore
            if np.all(specter_1 == 0):
                specter_1 = None
        if str(paper_id_2) in global_dataset.specter_embeddings:  # type: ignore
            specter_2 = global_dataset.specter_embeddings[str(paper_id_2)]  # type: ignore
            if np.all(specter_2 == 0):
                specter_2 = None

    if specter_1 is not None and specter_2 is not None:
        specter_sim = cosine_sim(specter_1, specter_2) + 1
    else:
        specter_sim = NUMPY_NAN

    features.append(specter_sim)  # , abstract_count, english_count])

    features.append(counter_jaccard(paper_1.journal_ngrams, paper_2.journal_ngrams))

    features.extend(
        name_text_features(
            signature_1.author_info_first_normalized_without_apostrophe,
            signature_2.author_info_first_normalized_without_apostrophe,
        )
    )

    # unifying feature type in features array
    features = [float(val) if type(val) in [np.float32, np.float64, float] else int(val) for val in features]

    return features, index


def parallel_helper(piece_of_work: Tuple, worker_func: Callable):
    """
    Helper function to explode tuple arguments

    Parameters
    ----------
    piece_of_work: Tuple
        the input for the worker func, in tuple form
    worker_func: Callable
        the function that will do the work

    Returns
    -------
    returns the result of calling the worker function
    """
    result = worker_func(*piece_of_work)
    return result


def many_pairs_featurize(
    signature_pairs: List[Tuple[str, str, Union[int, float]]],
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int,
    use_cache: bool,
    chunk_size: int,
    nameless_featurizer_info: Optional[FeaturizationInfo] = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Featurizes many pairs

    Parameters
    ----------
    signature_pairs: List[pairs]
        the pairs to featurize
    dataset: ANDData
        the dataset containing the relevant data
    featurizer_info: FeaturizationInfo
        the FeautrizationInfo object containing the listing of features to use
        and featurizer version
    n_jobs: int
        the number of cpus to use
    use_cache: bool
        whether or not to use write to/read from the features cache
    chunk_size: int
        the chunk size for multiprocessing
    nameless_featurizer_info: FeaturizationInfo
        the FeaturizationInfo for creating the features that do not use any name features,
        these will not be computed if this is None
    nan_value: float
        the value to replace nans with
    delete_training_data: bool
        Whether to delete some suspicious training rows

    Returns
    -------
    np.ndarray: the main features for all the pairs
    np.ndarray: the labels for all the pairs
    np.ndarray: the nameless features for all the pairs
    """
    # Strict guard: don't allow reference_features when dataset disabled them
    if "reference_features" in featurizer_info.features_to_use and not getattr(
        dataset, "compute_reference_features", False
    ):
        raise ValueError(
            "'reference_features' requested in features_to_use but dataset.compute_reference_features is False."
        )

    global global_dataset
    global_dataset = dataset  # type: ignore

    cached_features: Dict[str, Any] = {"features": {}}
    cache_changed = False
    new_features_count = 0

    if use_cache:
        logger.info("Loading cache...")
        if not os.path.exists(featurizer_info.cache_directory(dataset.name)):
            os.makedirs(featurizer_info.cache_directory(dataset.name))
        cache_path = featurizer_info.cache_file_path(dataset.name)
        if os.path.exists(cache_path):
            if cache_path in CACHED_FEATURES:
                cached_features = CACHED_FEATURES[cache_path]
            else:
                # fast path: orjson, fallback: stdlib json (handles legacy NaN)
                try:
                    with open(cache_path, "rb") as fh:
                        cached_features = orjson.loads(fh.read())
                except ValueError:
                    with open(cache_path, "r", encoding="utf-8") as fh:
                        cached_features = json.load(fh)
                logger.info(f"Cache loaded with {len(cached_features['features'])} keys")
        else:
            logger.info("Cache initiated.")
            cached_features = {}
            cached_features["features"] = {}
            cached_features["features_to_use"] = featurizer_info.features_to_use

        # Initialize buffer for new features if not already present
        if "__new_features__" not in cached_features:
            cached_features["__new_features__"] = {}

    features = np.ones((len(signature_pairs), NUM_FEATURES)) * (-LARGE_INTEGER)
    labels = np.zeros(len(signature_pairs))
    pieces_of_work = []
    logger.info(f"Creating {len(signature_pairs)} pieces of work")
    for i, pair in tqdm(enumerate(signature_pairs), desc="Creating work", disable=len(signature_pairs) <= 100000):
        labels[i] = pair[2]

        # negative labels are an indication of partial supervision
        if pair[2] < 0:
            continue

        cache_key = pair[0] + "___" + pair[1]
        if use_cache and cache_key in cached_features["features"]:
            cached_vector = cached_features["features"][cache_key]
            features[i, :] = cached_vector
            continue

        cache_key = pair[1] + "___" + pair[0]
        if use_cache and cache_key in cached_features["features"]:
            cached_vector = cached_features["features"][cache_key]
            features[i, :] = cached_vector
            continue

        cache_changed = True
        pieces_of_work.append(((pair[0], pair[1]), i))

    logger.info("Created pieces of work")

    indices_to_use_set = set()
    for feature_name in featurizer_info.features_to_use:
        indices_to_use_set.update(featurizer_info.feature_group_to_index[feature_name])
    indices_to_use: List[int] = sorted(list(indices_to_use_set))

    if nameless_featurizer_info:
        nameless_indices_to_use_set = set()
        for feature_name in nameless_featurizer_info.features_to_use:
            nameless_indices_to_use_set.update(nameless_featurizer_info.feature_group_to_index[feature_name])
        nameless_indices_to_use: List[int] = sorted(list(nameless_indices_to_use_set))

    if cache_changed:
        use_rust = _use_rust_featurizer()
        use_rust_batch = _use_rust_batch()
        rust_batch_threshold = _rust_batch_threshold()
        if rust_batch_threshold > 0 and len(pieces_of_work) < rust_batch_threshold:
            use_rust_batch = False
            logger.info(
                f"Rust batch mode disabled for small batch (size={len(pieces_of_work)} < {rust_batch_threshold})"
            )
        did_rust_batch = False
        if use_rust and use_rust_batch and feature_port.s2and_rust is not None and len(pieces_of_work) > 0:
            logger.info(f"Making {len(pieces_of_work)} feature vectors in Rust batch mode")
            write_cache_flag = use_cache if use_cache else None
            rust_featurizer = feature_port._get_rust_featurizer(dataset, write_cache=write_cache_flag)
            rust_pairs = [pair for pair, _ in pieces_of_work]
            num_threads = max(1, int(n_jobs))
            rust_features = rust_featurizer.featurize_pairs(rust_pairs, num_threads=num_threads)
            for feature_output, (_, index) in zip(rust_features, pieces_of_work):
                if use_cache:
                    cache_key = featurizer_info.feature_cache_key(signature_pairs[index])
                    cached_features["features"][cache_key] = feature_output
                    cached_features["__new_features__"][cache_key] = feature_output
                    new_features_count += 1
                features[index, :] = feature_output
            did_rust_batch = True

        if not did_rust_batch and n_jobs > 1:
            if use_cache:
                logger.info(f"Cache changed, making {len(pieces_of_work)} feature vectors in parallel")
            else:
                logger.info(f"Making {len(pieces_of_work)} feature vectors in parallel")

            with UniversalPool(processes=n_jobs if len(pieces_of_work) > 1000 else 1) as p:
                _max = len(pieces_of_work)
                with tqdm(total=_max, desc="Doing work", disable=_max <= 10000) as pbar:
                    for feature_output, index in p.imap(
                        functools.partial(parallel_helper, worker_func=_single_pair_featurize),
                        pieces_of_work,
                        min(chunk_size, max(1, int((_max / n_jobs) / 2))),
                    ):
                        # Write to in memory cache if we're not skipping
                        if use_cache:
                            cache_key = featurizer_info.feature_cache_key(signature_pairs[index])
                            cached_features["features"][cache_key] = feature_output
                            cached_features["__new_features__"][cache_key] = feature_output
                            new_features_count += 1

                        features[index, :] = feature_output
                        pbar.update()
        elif not did_rust_batch:
            if use_cache:
                logger.info(f"Cache changed, making {len(pieces_of_work)} feature vectors in serial")
            else:
                logger.info(f"Making {len(pieces_of_work)} feature vectors in serial")
            partial_func = functools.partial(parallel_helper, worker_func=_single_pair_featurize)
            for piece in tqdm(pieces_of_work, total=len(pieces_of_work), desc="Doing work"):
                result = partial_func(piece)
                if use_cache:
                    cache_key = featurizer_info.feature_cache_key(signature_pairs[result[1]])
                    cached_features["features"][cache_key] = result[0]
                    cached_features["__new_features__"][cache_key] = result[0]
                    new_features_count += 1

                features[result[1], :] = result[0]
        logger.info("Work completed")

    if use_cache and cache_changed:
        # Only do incremental writes if we have enough new features to justify the overhead
        new_features_in_buffer = len(cached_features.get("__new_features__", {}))

        if new_features_in_buffer >= INCREMENTAL_WRITE_THRESHOLD:
            logger.info(f"Queueing {new_features_in_buffer} new features for background write")
            _background_writer.start()
            _background_writer.queue_write(featurizer_info, cached_features, dataset.name)
        else:
            logger.info(f"Only {new_features_in_buffer} new features - will write at end")

    # Always write any remaining new features at the end
    if use_cache and cache_changed and len(cached_features.get("__new_features__", {})) > 0:
        new_features_in_buffer = len(cached_features["__new_features__"])
        logger.info(f"Writing final {new_features_in_buffer} new features to cache")
        featurizer_info.write_cache(cached_features, dataset.name, incremental=True)
        logger.info(f"Cache written with {len(cached_features['features'])} total keys.")

    if use_cache:
        logger.info("Writing to in memory cache")
        # use the variable from above, to be sure we are using the same path
        cache_path = featurizer_info.cache_file_path(dataset.name)
        CACHED_FEATURES[cache_path] = cached_features
        logger.info("In memory cache written")

    if delete_training_data:
        logger.info("Deleting some training rows")
        negative_label_indices = labels == 0
        high_coauthor_sim_indices = features[:, featurizer_info.get_feature_names().index("coauthor_similarity")] > 0.95
        indices_to_remove = negative_label_indices & high_coauthor_sim_indices
        logger.info(f"Intending to remove {sum(indices_to_remove)} rows")
        original_size = len(labels)
        features = features[~indices_to_remove, :]
        labels = labels[~indices_to_remove]
        logger.info(f"Removed {original_size - features.shape[0]} rows and {original_size - len(labels)} labels")

    logger.info("Making numpy arrays for features and labels")
    # have to do this before subselecting features
    if nameless_featurizer_info is not None:
        nameless_features = features[:, nameless_indices_to_use]
        nameless_features[np.isnan(nameless_features)] = nan_value
    else:
        nameless_features = None  # type: ignore

    features = features[:, indices_to_use]
    features[np.isnan(features)] = nan_value

    logger.info("Numpy arrays made")
    return features, labels, nameless_features


def featurize(
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    n_jobs: int = 1,
    use_cache: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nameless_featurizer_info: Optional[FeaturizationInfo] = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
) -> Union[Tuple[TupleOfArrays, TupleOfArrays, TupleOfArrays], TupleOfArrays]:
    """
    Featurizes the input dataset

    Parameters
    ----------
    dataset: ANDData
        the dataset containing the relevant data
    featurizer_info: FeaturizationInfo
        the FeautrizationInfo object containing the listing of features to use
        and featurizer version
    n_jobs: int
        the number of cpus to use
    use_cache: bool
        whether or not to use write to/read from the features cache
    chunk_size: int
        the chunk size for multiprocessing
    nameless_featurizer_info: FeaturizationInfo
        the FeaturizationInfo for creating the features that do not use any name features,
        these will not be computed if this is None
    nan_value: float
        the value to replace nans with
    delete_training_data: bool
        Whether to delete some suspicious training examples

    Returns
    -------
    train/val/test features and labels if mode is 'train',
    features and labels for all pairs if mode is 'inference'
    """
    # Strict guard: don't allow reference_features when dataset disabled them
    if "reference_features" in featurizer_info.features_to_use and not getattr(
        dataset, "compute_reference_features", False
    ):
        raise ValueError(
            "'reference_features' requested in features_to_use but dataset.compute_reference_features is False."
        )

    if dataset.mode == "inference":
        logger.info("featurizing all pairs")
        all_pairs = dataset.all_pairs()
        all_features = many_pairs_featurize(
            all_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
        )
        logger.info("featurized all pairs")
        return all_features
    else:
        if dataset.train_pairs is None:
            if dataset.train_blocks is not None:
                (
                    train_signatures,
                    val_signatures,
                    test_signatures,
                ) = dataset.split_cluster_signatures_fixed()
            elif dataset.train_signatures is not None:
                (
                    train_signatures,
                    val_signatures,
                    test_signatures,
                ) = dataset.split_data_signatures_fixed()
            else:
                (
                    train_signatures,
                    val_signatures,
                    test_signatures,
                ) = dataset.split_cluster_signatures()  # type: ignore

            train_pairs, val_pairs, test_pairs = dataset.split_pairs(train_signatures, val_signatures, test_signatures)

        else:
            train_pairs, val_pairs, test_pairs = dataset.fixed_pairs()

        logger.info("featurizing train")
        train_features = many_pairs_featurize(
            train_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            delete_training_data,
        )
        logger.info("featurized train, featurizing val")
        val_features = many_pairs_featurize(
            val_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
        )
        logger.info("featurized val, featurizing test")
        test_features = many_pairs_featurize(
            test_pairs,
            dataset,
            featurizer_info,
            n_jobs,
            use_cache,
            chunk_size,
            nameless_featurizer_info,
            nan_value,
            False,
        )
        logger.info("featurized test")
        return train_features, val_features, test_features

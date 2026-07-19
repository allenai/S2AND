import json
import logging
import math
import os
import pickle
import platform
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from s2and.consts import (
    _PACKAGE_DATA_DIR,
    CLUSTER_SEEDS_LOOKUP,
    LARGE_DISTANCE,
    NORMALIZATION_VERSION,
    NUMPY_NAN,
)
from s2and.mp import UniversalPool
from s2and.name_counts_index import (
    NameCountsIndex,
    readonly_name_counts_provenance,
    validated_name_counts_provenance,
)
from s2and.name_tuple_artifact import load_name_tuple_artifact, load_packaged_name_tuple_artifact
from s2and.runtime import (
    RuntimeContext,
    build_runtime_context,
    stage_uses_rust,
)
from s2and.rust_lifecycle import PYTHON_ONLY_POLICY, RUST_ARROW_TRAINING_POLICY
from s2and.sampling import random_sampling, sampling
from s2and.text import (
    AFFILIATIONS_STOP_WORDS,
    DROPPED_AFFIXES,
    VENUE_STOP_WORDS,
    CanonicalNameParts,
    canonical_lasts_equivalent,
    canonical_name_count_keys,
    canonical_name_tuple_pair,
    canonicalize_name_parts,
    canonicalize_name_text,
    compute_block,
    detect_language,
    first_names_name_compatible,
    get_text_ngrams,
    get_text_ngrams_words,
    normalize_orcid_compact,
    normalize_text,
    normalize_title,
)
from s2and.thread_config import resolve_n_jobs

logger = logging.getLogger("s2and")

CHUNK_SIZE = 1000  # for multiprocessing imap chunks
_PAIR_LABEL_MAP: dict[str | int, int] = {"NO": 0, "YES": 1, "0": 0, 0: 0, "1": 1, 1: 1}

SIGNATURE_PREPROCESS_BATCH_SIZE = 2048
# Canonical artifacts use ``<last> <first initial>`` for this count key.
NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR = "initial_char"
PairSamplingMode = Literal[
    "within_block_random",
    "within_block_balanced_classes",
    "within_block_balanced_homonym_synonym",
    "global_balanced_classes",
]
_PAIR_SAMPLING_MODES: frozenset[str] = frozenset(
    {
        "within_block_random",
        "within_block_balanced_classes",
        "within_block_balanced_homonym_synonym",
        "global_balanced_classes",
    }
)


def _normalize_specter_keys(embeddings: Iterable[tuple[Any, Any]]) -> dict[str, Any]:
    """Return SPECTER embeddings keyed by collision-free string paper IDs.

    Args:
        embeddings: Raw paper ID and embedding pairs.

    Returns:
        The embeddings indexed by string paper ID.

    Raises:
        ValueError: If distinct raw keys collapse to the same string key.
    """

    normalized: dict[str, Any] = {}
    raw_key_by_normalized_key: dict[str, Any] = {}
    for raw_key, embedding in embeddings:
        normalized_key = str(raw_key)
        if normalized_key in normalized:
            previous_raw_key = raw_key_by_normalized_key[normalized_key]
            raise ValueError(
                "SPECTER embedding keys collide after string normalization: "
                f"{previous_raw_key!r} and {raw_key!r} both map to {normalized_key!r}"
            )
        normalized[normalized_key] = embedding
        raw_key_by_normalized_key[normalized_key] = raw_key
    return normalized


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate train, validation, and test split ratios.

    Args:
        train_ratio: Fraction assigned to training.
        val_ratio: Fraction assigned to validation.
        test_ratio: Fraction assigned to testing.

    Raises:
        ValueError: If a ratio is outside ``[0, 1]`` or the ratios do not sum
            to one within floating-point tolerance.
    """

    ratios = (train_ratio, val_ratio, test_ratio)
    if any(not 0.0 <= ratio <= 1.0 for ratio in ratios):
        raise ValueError(
            "train/val/test ratios must each be between 0 and 1; "
            f"got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )
    if not math.isclose(sum(ratios), 1.0):
        raise ValueError(
            f"train/val/test ratios must add to 1; got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )


def _map_fixed_pair_labels(pair_frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Copy a fixed-pair frame and map its labels to binary integers.

    Args:
        pair_frame: Input frame containing a ``label`` column.
        split_name: Human-readable split name for validation errors.

    Returns:
        A copy of ``pair_frame`` with mapped integer labels.

    Raises:
        ValueError: If any label is outside the fixed-pair label vocabulary.
    """

    output = pair_frame.copy()
    mapped_labels = output["label"].map(_PAIR_LABEL_MAP)
    unknown_mask = mapped_labels.isna()
    if bool(unknown_mask.any()):
        unknown_labels = pd.unique(output.loc[unknown_mask, "label"]).tolist()
        raise ValueError(f"Unknown fixed-pair labels in {split_name} split: {unknown_labels!r}")
    output.loc[:, "label"] = mapped_labels
    return output


def _validate_pair_sampling_mode(mode: str) -> PairSamplingMode:
    """Return a validated pair sampling mode."""

    if mode not in _PAIR_SAMPLING_MODES:
        raise ValueError(f"Unknown pair_sampling_mode: {mode!r}")
    return cast(PairSamplingMode, mode)


def _pair_sampling_uses_blocks(mode: PairSamplingMode) -> bool:
    """Return whether a pair sampling mode samples within blocks."""

    return mode != "global_balanced_classes"


def _load_name_tuples_from_file(filename: str) -> set[tuple[str, str]]:
    """Load one canonical artifact under the strict adjacent-sidecar contract."""

    if filename == "s2and_name_tuples_canonical.txt":
        return set(load_packaged_name_tuple_artifact().pairs)
    return set(load_name_tuple_artifact(Path(_PACKAGE_DATA_DIR) / filename).pairs)


def _signature_preprocess_backend_decision(runtime_context: RuntimeContext) -> bool:
    use_rust_backend = stage_uses_rust(runtime_context)
    if not use_rust_backend:
        return False
    from s2and import feature_port

    feature_port._require_rust_runtime()  # noqa: SLF001
    return True


def _ordered_coauthors_for_signature(signature: "Signature", papers: dict[str, "Paper"]) -> list[str]:
    if signature.author_info_position is None:
        raise ValueError(
            "Signature is missing author_info_position for coauthor ngram materialization "
            f"(signature_id={signature.signature_id} paper_id={signature.paper_id})"
        )
    paper = papers.get(str(signature.paper_id))
    if paper is None:
        logger.warning(
            "Missing paper for signature ngram materialization; treating coauthors as empty "
            "(signature_id=%s paper_id=%s)",
            signature.signature_id,
            signature.paper_id,
        )
        return []
    # Rust deferred paper preprocessing can leave `paper.authors` as raw names here.
    return [
        normalize_text(author.author_name)
        for author in paper.authors
        if author.position != signature.author_info_position
    ]


def _python_signature_ngrams_batch(
    coauthor_texts: list[str], affiliation_texts: list[str]
) -> tuple[list[Counter], list[Counter]]:
    coauthor_counters = [
        get_text_ngrams(text, stopwords=None, use_bigrams=True, drop_short_tokens=False) if text else Counter()
        for text in coauthor_texts
    ]
    affiliation_counters = [
        get_text_ngrams_words(text, stopwords=AFFILIATIONS_STOP_WORDS) if text else Counter()
        for text in affiliation_texts
    ]
    return coauthor_counters, affiliation_counters


def _assemble_full_name(parts: list[str | None]) -> str:
    return " ".join([part.strip() for part in parts if part is not None and len(part) != 0]).strip()


def _build_signature_ngram_texts(
    *,
    coauthors: list[str],
    affiliations: list[str],
    normalize_coauthors: bool,
    normalize_affiliations: bool,
) -> tuple[str, str]:
    coauthor_values = [normalize_text(coauthor) for coauthor in coauthors] if normalize_coauthors else coauthors
    affiliation_values = (
        [normalize_text(affiliation) for affiliation in affiliations] if normalize_affiliations else affiliations
    )
    coauthor_values = [value for value in coauthor_values if value]
    affiliation_values = [value for value in affiliation_values if value]
    coauthor_text = " ".join(coauthor_values) if len(coauthor_values) > 0 else ""
    affiliation_text = " ".join(affiliation_values)
    return coauthor_text, affiliation_text


class NameCounts(NamedTuple):
    first: float | None
    last: float | None
    first_last: float | None
    last_first_initial: float | None


class Signature(NamedTuple):
    author_info_first: str | None
    author_info_first_normalized_without_apostrophe: str | None
    author_info_middle: str | None
    author_info_middle_normalized_without_apostrophe: str | None
    author_info_last_normalized: str | None
    author_info_last: str
    author_info_suffix_normalized: str | None
    author_info_suffix: str | None
    author_info_coauthors: set[str] | None
    author_info_coauthor_blocks: set[str] | None
    author_info_full_name: str | None
    author_info_affiliations: list[str]
    author_info_affiliations_n_grams: Counter | None
    author_info_coauthor_n_grams: Counter | None
    author_info_email: str | None
    author_info_orcid: str | None
    author_info_name_counts: NameCounts | None
    author_info_position: int
    author_info_block: str
    author_info_given_block: str | None
    author_info_estimated_gender: str | None
    author_info_estimated_ethnicity: str | None
    paper_id: int
    sourced_author_source: str | None
    sourced_author_ids: list[str]
    author_id: int | None
    signature_id: str


class Author(NamedTuple):
    author_name: str
    position: int


class Paper(NamedTuple):
    title: str
    has_abstract: bool | None
    in_signatures: bool | None
    is_english: bool | None
    is_reliable: bool | None
    language_reliability: float | None
    predicted_language: str | None
    title_ngrams_words: Counter | None
    authors: list[Author]
    venue: str | None
    journal_name: str | None
    title_ngrams_chars: Counter | None
    venue_ngrams: Counter | None
    journal_ngrams: Counter | None
    year: int | None
    paper_id: int


class ANDData:
    """
    The main class for holding our representation of an author disambiguation dataset

    Input:
        signatures: path to the signatures json file (or the json object)
        papers: path to the papers information json file (or the json object)
        name: name of the dataset, used for caching computed features
        mode: 'train' or 'inference'; if 'inference', everything related to dataset
            splitting will be ignored
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
        pair_sampling_mode: strategy for sampling training/eval pairs.
        all_test_pairs_flag: With blocking, for the linkage function evaluation task, should the test
            contain all possible pairs from test blocks, or the given number of pairs (test_pairs_size)
        random_seed: random seed
        name_counts_index: Manifest-backed binary name-count index, or ``None``
            to leave Python-side name-count features unmaterialized. A path is
            verified and opened once per immutable manifest generation; an
            already-open ``NameCountsIndex`` handle can be shared explicitly.
        n_jobs: number of cpus to use
        preprocess: whether to preprocess the data (normalization, etc)
        name_tuples: Canonical first-name aliases. ``None`` and ``"filtered"``
            both select the packaged canonical artifact. Pass an explicit empty
            set to disable aliases, or a set of pairs; pair order is ignored.
        use_orcid_id: whether to use the orcid id for (a) constraints as true if orcids match and
            (b) subblocking so that any sigs with the same orcid are in the same subblock
        rust_arrow_featurization: set by s2and.arrow_training when this dataset's Rust
            featurizer is built from Arrow IPC artifacts; defers paper preprocessing and
            signature n-gram/field materialization to the Rust Arrow readers
    """

    def __init__(
        self,
        signatures: str | dict,
        papers: str | dict,
        name: str,
        mode: str = "train",
        clusters: str | dict | None = None,
        specter_embeddings: str | dict | tuple | None = None,
        cluster_seeds: str | dict | None = None,
        altered_cluster_signatures: str | list | set | None = None,
        train_pairs: str | pd.DataFrame | None = None,
        val_pairs: str | pd.DataFrame | None = None,
        test_pairs: str | pd.DataFrame | None = None,
        train_blocks: str | list | None = None,
        val_blocks: str | list | None = None,
        test_blocks: str | list | None = None,
        train_signatures: str | list | None = None,
        val_signatures: str | list | None = None,
        test_signatures: str | list | None = None,
        block_type: str = "s2",
        unit_of_data_split: str = "blocks",
        num_clusters_for_block_size: int = 1,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        train_pairs_size: int = 30000,
        val_pairs_size: int = 5000,
        test_pairs_size: int = 5000,
        pair_sampling_mode: PairSamplingMode = "within_block_random",
        all_test_pairs_flag: bool = False,
        random_seed: int = 1111,
        name_counts_index: str | os.PathLike[str] | NameCountsIndex | None = None,
        n_jobs: int = 1,
        preprocess: bool = True,
        name_tuples: set[tuple[str, str]] | str | None = "filtered",
        use_orcid_id: bool = True,
        compute_block_fn: Callable[[str], str] = compute_block,
        rust_arrow_featurization: bool = False,
    ):
        init_start = time.perf_counter()
        if rust_arrow_featurization and (mode != "train" or not preprocess):
            raise ValueError("Rust Arrow training requires mode='train' and preprocess=True")
        self.runtime_context = build_runtime_context(
            "dataset_build",
            backend="rust" if rust_arrow_featurization else "python",
        )
        self.original_signatures_path = signatures if isinstance(signatures, str) else None
        self.original_papers_path = papers if isinstance(papers, str) else None
        self.signatures_path = self.original_signatures_path
        self.papers_path = self.original_papers_path
        self._s2and_python_pair_ngrams_ready: bool = False
        self._rust_cluster_seeds_require_id: int | None = None
        self._rust_cluster_seeds_require_len: int | None = None
        self._rust_cluster_seeds_disallow_id: int | None = None
        self._rust_cluster_seeds_disallow_len: int | None = None
        self.clusters_path = clusters if isinstance(clusters, str) else None
        self.cluster_seeds_path = cluster_seeds if isinstance(cluster_seeds, str) else None
        self.specter_embeddings_path = specter_embeddings if isinstance(specter_embeddings, str) else None
        # Explicit Arrow prediction artifacts; populated by s2and.arrow_training
        # when the dataset is built from an Arrow bundle.
        self.arrow_paths: Mapping[str, str] | None = None
        self.arrow_artifact_generation: str | None = None
        self.compute_block_fn = compute_block_fn
        self.rust_lifecycle_policy = RUST_ARROW_TRAINING_POLICY if rust_arrow_featurization else PYTHON_ONLY_POLICY
        pair_sampling_mode = _validate_pair_sampling_mode(pair_sampling_mode)

        if mode == "train":
            if train_blocks is not None and block_type != "original":
                logger.warning("If you are passing in training/val/test blocks, then you may want original blocks.")

            if unit_of_data_split == "blocks" and not _pair_sampling_uses_blocks(pair_sampling_mode):
                raise ValueError("Block-based cluster splits are not compatible with sampling strategies 0 and 1.")

            if (clusters is not None and train_pairs is not None) or (
                clusters is None and train_pairs is None and train_blocks is None
            ):
                raise ValueError("Set exactly one of clusters and train_pairs")

            if train_blocks is not None and train_pairs is not None:
                raise ValueError("Can't pass in both train_blocks and train_pairs")

            if train_blocks is not None and clusters is None:
                raise ValueError("Train blocks still needs clusters")

        # Load signatures first so we can restrict papers/specter to relevant subset
        signatures_stage_start = time.perf_counter()
        logger.info("loading signatures")
        raw_signatures = self.maybe_load_json(signatures)
        self.signatures = {}
        # convert dictionary to namedtuples for memory reduction
        for signature_id, signature in raw_signatures.items():
            self.signatures[signature_id] = Signature(
                author_info_first=signature["author_info"]["first"],
                author_info_first_normalized_without_apostrophe=None,
                author_info_middle=signature["author_info"]["middle"],
                author_info_middle_normalized_without_apostrophe=None,
                author_info_last_normalized=None,
                author_info_last=signature["author_info"]["last"],
                author_info_suffix_normalized=None,
                author_info_suffix=signature["author_info"]["suffix"],
                author_info_coauthors=None,
                author_info_coauthor_blocks=None,
                author_info_full_name=None,
                author_info_affiliations=signature["author_info"]["affiliations"],
                author_info_affiliations_n_grams=None,
                author_info_coauthor_n_grams=None,
                author_info_email=signature["author_info"]["email"],
                # use_orcid_id is an offline data-prep knob used by training data
                # construction (incremental_linking_training.data_loading) to build
                # datasets that strip ORCIDs entirely. Production callers leave the
                # default True and let the per-request `Clusterer.suppress_orcid` flag
                # drive ORCID enablement (which threads to Rust via `orcid_enabled` in
                # raw_arrow_features). The two control surfaces are equivalent in
                # effect; do not mix them.
                author_info_orcid=(
                    (signature["author_info"].get("source_ids") or [None])[0]
                    if use_orcid_id and signature["author_info"].get("source_id_source") == "ORCID"
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
        logger.debug(
            "Telemetry stage: stage=anddata_ingest_signatures seconds=%.3f signatures=%d",
            time.perf_counter() - signatures_stage_start,
            len(self.signatures),
        )

        # Determine the set of papers referenced by signatures.
        needed_paper_ids: set[str] = set(str(sig.paper_id) for sig in self.signatures.values())

        papers_stage_start = time.perf_counter()
        logger.info("loading papers (subset referenced by signatures)")
        raw_papers = self.maybe_load_json(papers)
        filtered_papers = {pid: p for pid, p in raw_papers.items() if str(pid) in needed_paper_ids}
        self.papers = {}
        # convert dictionary to namedtuples for memory reduction
        for paper_id, paper in filtered_papers.items():
            self.papers[paper_id] = Paper(
                title=paper["title"],
                has_abstract=paper["abstract"] not in {"", None},
                in_signatures=None,
                is_english=None,
                is_reliable=None,
                language_reliability=None,
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
                year=paper["year"],
                paper_id=paper["paper_id"],
            )
        logger.info(f"loaded papers subset: {len(self.papers)}/{len(raw_papers)} relevant")
        logger.debug(
            "Telemetry stage: stage=anddata_ingest_papers seconds=%.3f retained_papers=%d source_papers=%d",
            time.perf_counter() - papers_stage_start,
            len(self.papers),
            len(raw_papers),
        )

        self.name = name
        self.mode = mode
        logger.info("loading clusters")
        self.clusters: dict | None = self.maybe_load_json(clusters)
        logger.info("loaded clusters, loading specter")
        self.specter_embeddings = self.maybe_load_specter(specter_embeddings)
        # prevents errors during testing where we have no specter embeddings
        if self.specter_embeddings is None:
            self.specter_embeddings = {}
        else:
            # Only keep embeddings for papers we retained
            needed_keys = {str(paper_id) for paper_id in self.papers}
            self.specter_embeddings = {k: v for k, v in self.specter_embeddings.items() if k in needed_keys}
        logger.info("loaded specter, loading cluster seeds")
        cluster_seeds_dict = self.maybe_load_json(cluster_seeds)
        self.altered_cluster_signatures = self.maybe_load_list(altered_cluster_signatures)
        self.cluster_seeds_disallow = set()
        self.cluster_seeds_require = {}
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
                if root_added:
                    cluster_num += 1
            self.max_seed_cluster_id = cluster_num
        logger.info("loaded cluster seeds")
        # Versioned seed state for Rust sync dedupe.
        self._cluster_seeds_version = 1
        self._rust_cluster_seeds_synced_version = 0
        self._rust_cluster_seeds_sync_calls = 0
        self._rust_cluster_seeds_sync_attempted = 0
        self._rust_cluster_seeds_sync_succeeded = 0
        self._rust_cluster_seeds_sync_skipped_unchanged = 0
        self._rust_cluster_seeds_sync_seconds_total = 0.0
        self._rust_cluster_seeds_sync_seconds_max = 0.0
        # check that all altered_cluster_signatures are in cluster_seeds_require
        if self.altered_cluster_signatures is not None:
            for signature_id in self.altered_cluster_signatures:
                if signature_id not in self.cluster_seeds_require:
                    raise ValueError(f"Altered cluster signature {signature_id} not in cluster_seeds_require")
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
        self.pair_sampling_mode = pair_sampling_mode
        self.all_test_pairs_flag = all_test_pairs_flag
        self.random_seed = random_seed
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
            self.pair_sampling_mode = "within_block_random"
            self.all_test_pairs_flag = True
            self.block_type = "s2"  # pure inference is for S2 probably?
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.normalization_version = NORMALIZATION_VERSION
        self._name_counts_provenance: Mapping[str, Any] | None = None
        self.name_counts_index: NameCountsIndex | None = None
        if name_counts_index is not None:
            logger.info("opening name-count index (generation-cached)")
            self.name_counts_index = (
                name_counts_index
                if isinstance(name_counts_index, NameCountsIndex)
                else NameCountsIndex.open(name_counts_index)
            )
            self.name_counts_provenance = self.name_counts_index.source_provenance
            logger.info("opened name-count index")
        self.name_counts_loaded = self.name_counts_index is not None

        self.n_jobs = resolve_n_jobs(n_jobs)
        self.signature_to_block = self.get_signatures_to_block()
        papers_from_signatures = {str(signature.paper_id) for signature in self.signatures.values()}
        for paper_id, paper in self.papers.items():
            self.papers[paper_id] = paper._replace(in_signatures=str(paper_id) in papers_from_signatures)
        self.preprocess = preprocess

        resolved_name_tuples: set[tuple[str, str]]
        if name_tuples == "filtered" or name_tuples is None:
            # canonical_v2 alias artifact, regenerated deterministically by
            # scripts/production/generate_canonical_name_tuples.py.
            resolved_name_tuples = _load_name_tuples_from_file("s2and_name_tuples_canonical.txt")
        elif isinstance(name_tuples, set):
            resolved_name_tuples = {canonical_name_tuple_pair(first_a, first_b) for first_a, first_b in name_tuples}
        else:
            raise ValueError("name_tuples must be None, 'filtered', or a set of canonical (first_a, first_b) tuples")
        self.name_tuples = resolved_name_tuples

        preprocess_papers_stage_start = time.perf_counter()
        if self.rust_lifecycle_policy.skip_python_paper_preprocess:
            # Rust paper preprocessing will fill missing fields in the build path; avoid duplicate Python work.
            logger.info("Rust deferred paper preprocessing active: skipping Python paper preprocessing")
        else:
            logger.info("preprocessing papers")
            self.papers = preprocess_papers_parallel(
                self.papers,
                self.n_jobs,
                self.preprocess,
            )
            logger.info("preprocessed papers")
        logger.debug(
            "Telemetry stage: stage=anddata_preprocess_papers seconds=%.3f papers=%d",
            time.perf_counter() - preprocess_papers_stage_start,
            len(self.papers),
        )

        preprocess_signatures_stage_start = time.perf_counter()
        logger.info("preprocessing signatures")
        self.preprocess_signatures()
        logger.info("preprocessed signatures")
        logger.debug(
            "Telemetry stage: stage=anddata_preprocess_signatures seconds=%.3f signatures=%d",
            time.perf_counter() - preprocess_signatures_stage_start,
            len(self.signatures),
        )
        logger.debug(
            "Telemetry stage: stage=anddata_total_init seconds=%.3f",
            time.perf_counter() - init_start,
        )

    @property
    def name_counts_provenance(self) -> Mapping[str, Any] | None:
        """Return verified name-count provenance through a read-only view."""

        return self._name_counts_provenance

    @name_counts_provenance.setter
    def name_counts_provenance(self, value: Mapping[str, Any] | None) -> None:
        if value is None:
            self._name_counts_provenance = None
            return
        validated = validated_name_counts_provenance(value, context="ANDData.name_counts_provenance")
        self._name_counts_provenance = readonly_name_counts_provenance(validated)

    def _signature_name_count_keys(
        self,
        signature: Signature,
        *,
        first_raw: str,
        middle_raw: str,
        first_without_apostrophe: str | None,
        last_normalized: str | None,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        # canonical_v2 count keys (D6/D8): keys are the canonical fields after missing
        # and informativeness gating. A missing/uninformative component means no lookup
        # (NaN feature), never a sentinel count; a present key that is absent from the
        # corpus dictionaries keeps the default count of 1.
        canonical_first = first_without_apostrophe
        canonical_last = last_normalized
        if canonical_first is None or canonical_last is None:
            parts = canonicalize_name_parts(first_raw, middle_raw, signature.author_info_last)
            if canonical_first is None:
                canonical_first = parts.first
            if canonical_last is None:
                canonical_last = parts.last
        keys = canonical_name_count_keys(CanonicalNameParts(first=canonical_first, middle="", last=canonical_last))
        first_key = keys["first"]
        last_key = keys["last"]
        first_last_key = keys["first_last"]
        last_first_initial_key = keys["last_first_initial"]
        return (
            first_key,
            last_key,
            first_last_key,
            last_first_initial_key,
        )

    def _compute_signature_name_counts(
        self,
        signature: Signature,
        *,
        first_raw: str,
        middle_raw: str,
        first_without_apostrophe: str | None,
        last_normalized: str | None,
    ) -> NameCounts:
        """Resolve one signature through the binary index for targeted refreshes."""

        if self.name_counts_index is None:
            return NameCounts(first=None, last=None, first_last=None, last_first_initial=None)
        keys = self._signature_name_count_keys(
            signature,
            first_raw=first_raw,
            middle_raw=middle_raw,
            first_without_apostrophe=first_without_apostrophe,
            last_normalized=last_normalized,
        )
        columns = self.name_counts_index.lookup_many(
            [keys[0]],
            [keys[1]],
            [keys[2]],
            [keys[3]],
        )
        return NameCounts(*(float(column[0]) for column in columns))

    def preprocess_signatures(self) -> None:
        """
        Preprocess the signatures, doing lots of normalization and feature creation

        Returns
        -------
        nothing, modifies self.signatures
        """
        runtime_context = self.runtime_context
        use_rust_backend = _signature_preprocess_backend_decision(runtime_context)
        use_rust_featurizer = use_rust_backend
        rust_module_available = use_rust_backend
        defer_signature_ngrams_to_rust = self.rust_lifecycle_policy.defer_signature_ngrams_to_rust
        defer_signature_fields_to_rust = self.rust_lifecycle_policy.defer_signature_fields_to_rust
        logger.info(
            "Signature preprocessing backend decision: backend=%s use_rust_featurizer=%s rust_module_available=%s "
            "defer_signature_ngrams_to_rust=%s defer_signature_fields_to_rust=%s "
            "run_id=%s",
            "rust" if use_rust_backend else "python",
            use_rust_featurizer,
            rust_module_available,
            defer_signature_ngrams_to_rust,
            defer_signature_fields_to_rust,
            runtime_context.run_id,
        )

        signature_ids = list(self.signatures.keys())
        with tqdm(total=len(signature_ids), desc="Preprocessing signatures") as progress_bar:
            for batch_start in range(0, len(signature_ids), SIGNATURE_PREPROCESS_BATCH_SIZE):
                batch_signature_ids = signature_ids[batch_start : batch_start + SIGNATURE_PREPROCESS_BATCH_SIZE]
                batch_rows = []
                batch_coauthor_texts: list[str] = []
                batch_affiliation_texts: list[str] = []

                for signature_id in batch_signature_ids:
                    signature = self.signatures[signature_id]

                    first_raw = signature.author_info_first or ""
                    middle_raw = signature.author_info_middle or ""
                    stored_first_without_apostrophe: str | None = (
                        signature.author_info_first_normalized_without_apostrophe
                    )
                    stored_middle_without_apostrophe: str | None = (
                        signature.author_info_middle_normalized_without_apostrophe
                    )
                    stored_last_normalized: str | None = signature.author_info_last_normalized
                    stored_suffix_normalized: str | None = signature.author_info_suffix_normalized

                    coauthors: list[str] | None = None
                    if len(self.papers) != 0 and not defer_signature_fields_to_rust:
                        coauthors = _ordered_coauthors_for_signature(signature, self.papers)

                    coauthor_set = set(coauthors) if coauthors is not None else None
                    coauthor_blocks = (
                        set(self.compute_block_fn(author) for author in coauthors) if coauthors is not None else None
                    )

                    affiliations: list[str] = signature.author_info_affiliations
                    full_name = signature.author_info_full_name
                    normalized_orcid = signature.author_info_orcid
                    count_keys: tuple[str | None, str | None, str | None, str | None] | None = None
                    coauthor_text = ""
                    affiliation_text = ""

                    if self.preprocess:
                        if defer_signature_fields_to_rust:
                            stored_first_without_apostrophe = None
                            stored_middle_without_apostrophe = None
                            stored_last_normalized = None
                            stored_suffix_normalized = None
                            coauthor_set = None
                            coauthor_blocks = None
                            # Rust derives the canonical query-author facet from
                            # the Arrow name fields. Do not retain or synthesize
                            # a raw-text full name on the Python object.
                            full_name = None
                        else:
                            # canonical_v2 normalization: one routine for first/middle/last
                            # (apostrophe-like marks deleted, dash-like characters uniform,
                            # dash-bound given-name compounds stay together, spill on space,
                            # spaced canonical surnames). Suffixes stay on the generic
                            # normalizer; suffix policy is outside canonical_v2.
                            canonical_parts = canonicalize_name_parts(
                                first_raw,
                                middle_raw,
                                signature.author_info_last,
                            )
                            stored_first_without_apostrophe = canonical_parts.first
                            stored_middle_without_apostrophe = canonical_parts.middle
                            stored_last_normalized = canonical_parts.last
                            stored_suffix_normalized = normalize_text(signature.author_info_suffix or "")
                            affiliations = [
                                normalized_affiliation
                                for affiliation in signature.author_info_affiliations
                                if (normalized_affiliation := normalize_text(affiliation))
                            ]
                            if not defer_signature_ngrams_to_rust:
                                coauthor_text, affiliation_text = _build_signature_ngram_texts(
                                    coauthors=coauthors or [],
                                    affiliations=affiliations,
                                    normalize_coauthors=False,
                                    normalize_affiliations=False,
                                )

                        count_keys = None
                        if self.name_counts_index is not None:
                            count_keys = self._signature_name_count_keys(
                                signature,
                                first_raw=first_raw,
                                middle_raw=middle_raw,
                                first_without_apostrophe=stored_first_without_apostrophe,
                                last_normalized=stored_last_normalized,
                            )

                        if not defer_signature_fields_to_rust:
                            full_name = _assemble_full_name(
                                [
                                    stored_first_without_apostrophe,
                                    stored_middle_without_apostrophe,
                                    stored_last_normalized,
                                    stored_suffix_normalized,
                                ]
                            )

                            if signature.author_info_orcid is not None:
                                normalized_orcid = normalize_orcid_compact(signature.author_info_orcid)

                    batch_rows.append(
                        {
                            "signature_id": signature_id,
                            "signature": signature,
                            "first_without_apostrophe": stored_first_without_apostrophe,
                            "middle_without_apostrophe": stored_middle_without_apostrophe,
                            "last_normalized": stored_last_normalized,
                            "suffix_normalized": stored_suffix_normalized,
                            "coauthor_set": coauthor_set,
                            "coauthor_blocks": coauthor_blocks,
                            "affiliations": affiliations,
                            "full_name": full_name,
                            "count_keys": count_keys,
                            "normalized_orcid": normalized_orcid,
                            "coauthor_text": coauthor_text,
                            "affiliation_text": affiliation_text,
                        }
                    )

                    if self.preprocess and not defer_signature_ngrams_to_rust:
                        batch_coauthor_texts.append(coauthor_text)
                        batch_affiliation_texts.append(affiliation_text)

                batch_coauthor_ngrams: list[Counter] = []
                batch_affiliation_ngrams: list[Counter] = []
                if self.preprocess and not defer_signature_ngrams_to_rust:
                    batch_coauthor_ngrams, batch_affiliation_ngrams = _python_signature_ngrams_batch(
                        batch_coauthor_texts,
                        batch_affiliation_texts,
                    )

                batch_name_counts: list[NameCounts] = []
                if self.preprocess:
                    if self.name_counts_index is None:
                        batch_name_counts = [
                            NameCounts(first=None, last=None, first_last=None, last_first_initial=None)
                            for _row in batch_rows
                        ]
                    else:
                        key_rows = [row["count_keys"] for row in batch_rows]
                        if any(keys is None for keys in key_rows):  # pragma: no cover - construction invariant
                            raise RuntimeError("name-count index batch is missing canonical keys")
                        first_keys = [keys[0] for keys in key_rows]
                        last_keys = [keys[1] for keys in key_rows]
                        first_last_keys = [keys[2] for keys in key_rows]
                        last_first_initial_keys = [keys[3] for keys in key_rows]
                        count_columns = self.name_counts_index.lookup_many(
                            first_keys,
                            last_keys,
                            first_last_keys,
                            last_first_initial_keys,
                        )
                        batch_name_counts = [
                            NameCounts(*(float(column[index]) for column in count_columns))
                            for index in range(len(batch_rows))
                        ]

                for idx, row in enumerate(batch_rows):
                    replace_kwargs = {
                        "author_info_first_normalized_without_apostrophe": row["first_without_apostrophe"],
                        "author_info_middle_normalized_without_apostrophe": row["middle_without_apostrophe"],
                        "author_info_last_normalized": row["last_normalized"],
                        "author_info_suffix_normalized": row["suffix_normalized"],
                        "author_info_coauthors": row["coauthor_set"],
                        "author_info_coauthor_blocks": row["coauthor_blocks"],
                    }
                    if self.preprocess:
                        replace_kwargs.update(
                            {
                                "author_info_full_name": row["full_name"],
                                "author_info_affiliations": row["affiliations"],
                                "author_info_affiliations_n_grams": (
                                    None if defer_signature_ngrams_to_rust else batch_affiliation_ngrams[idx]
                                ),
                                "author_info_coauthor_n_grams": (
                                    None if defer_signature_ngrams_to_rust else batch_coauthor_ngrams[idx]
                                ),
                                "author_info_name_counts": batch_name_counts[idx],
                                "author_info_orcid": row["normalized_orcid"],
                            }
                        )
                    self.signatures[row["signature_id"]] = row["signature"]._replace(**replace_kwargs)

                progress_bar.update(len(batch_signature_ids))

    def materialize_signature_ngrams_python(self, batch_size: int = SIGNATURE_PREPROCESS_BATCH_SIZE) -> None:
        """
        Materialize signature n-gram Counters in Python for signatures that are missing them.

        This is primarily intended for fallback/debug paths when Rust-owned signature n-grams
        are deferred during preprocessing.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        signature_ids = list(self.signatures.keys())
        logger.info("Materializing missing signature ngrams in Python for %d signatures", len(signature_ids))
        with tqdm(total=len(signature_ids), desc="Materializing signature ngrams") as progress_bar:
            for batch_start in range(0, len(signature_ids), batch_size):
                batch_signature_ids = signature_ids[batch_start : batch_start + batch_size]
                pending_signature_ids: list[str] = []
                batch_coauthor_texts: list[str] = []
                batch_affiliation_texts: list[str] = []

                for signature_id in batch_signature_ids:
                    signature = self.signatures[signature_id]
                    if (
                        signature.author_info_affiliations_n_grams is not None
                        and signature.author_info_coauthor_n_grams is not None
                    ):
                        continue

                    coauthors = _ordered_coauthors_for_signature(signature, self.papers)
                    normalized_affiliations = list(signature.author_info_affiliations or [])
                    # `get_text_ngrams_words` performs stopword and single-character filtering.
                    # Keep this normalization path idempotent as a safe fallback for deferred Rust paths.
                    coauthor_text, affiliation_text = _build_signature_ngram_texts(
                        coauthors=coauthors,
                        affiliations=normalized_affiliations,
                        normalize_coauthors=True,
                        normalize_affiliations=True,
                    )

                    pending_signature_ids.append(signature_id)
                    batch_coauthor_texts.append(coauthor_text)
                    batch_affiliation_texts.append(affiliation_text)

                if pending_signature_ids:
                    batch_coauthor_ngrams, batch_affiliation_ngrams = _python_signature_ngrams_batch(
                        batch_coauthor_texts,
                        batch_affiliation_texts,
                    )
                    for idx, signature_id in enumerate(pending_signature_ids):
                        signature = self.signatures[signature_id]
                        self.signatures[signature_id] = signature._replace(
                            author_info_affiliations_n_grams=batch_affiliation_ngrams[idx],
                            author_info_coauthor_n_grams=batch_coauthor_ngrams[idx],
                        )

                progress_bar.update(len(batch_signature_ids))

    @staticmethod
    def maybe_load_json(path_or_json: str | list | dict | None) -> Any:
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
            with open(path_or_json, encoding="utf-8") as _json_file:
                output = json.load(_json_file)
            return output
        else:
            return path_or_json

    @staticmethod
    def maybe_load_list(path_or_list: str | list | set | None) -> list | set | None:
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
            with open(path_or_list, encoding="utf-8") as f:
                contents = f.read().strip()
                if not contents:
                    return []
                return contents.splitlines()
        else:
            return path_or_list

    @staticmethod
    def maybe_load_dataframe(path_or_dataframe: str | pd.DataFrame | None) -> pd.DataFrame | None:
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
        if isinstance(path_or_dataframe, str):
            return pd.read_csv(path_or_dataframe, sep=",")
        if path_or_dataframe is None or isinstance(path_or_dataframe, pd.DataFrame):
            return path_or_dataframe
        raise TypeError(f"Expected dataframe path or DataFrame, got {type(path_or_dataframe)}")

    @staticmethod
    def maybe_load_specter(path_or_pickle: str | dict | tuple | None) -> dict | None:
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
        loaded: dict | tuple | Any | None
        if isinstance(path_or_pickle, str):
            with open(path_or_pickle, "rb") as _pickle_file:
                loaded = pickle.load(_pickle_file)
        else:
            loaded = path_or_pickle

        if loaded is None:
            return None

        if isinstance(loaded, dict):
            return _normalize_specter_keys(loaded.items())

        if isinstance(loaded, tuple) and len(loaded) == 2:
            matrix, keys = loaded
            return _normalize_specter_keys((key, matrix[i, :]) for i, key in enumerate(keys))

        raise TypeError(f"Unsupported specter pickle payload type: {type(loaded)}")

    def _build_block_dict(self, key_attr: str) -> dict[str, list[str]]:
        block: dict[str, list[str]] = defaultdict(list)
        for signature_id, signature in self.signatures.items():
            block_key = getattr(signature, key_attr)
            block[block_key].append(signature_id)
        return dict(block)

    def get_original_blocks(self) -> dict[str, list[str]]:
        """
        Gets the block dict based on the blocks provided with the dataset

        Returns
        -------
        Dict: mapping from block id to list of signatures in the block
        """
        return self._build_block_dict("author_info_given_block")

    def get_s2_blocks(self) -> dict[str, list[str]]:
        """
        Gets the block dict based on the blocks provided by Semantic Scholar data

        Returns
        -------
        Dict: mapping from block id to list of signatures in the block
        """
        return self._build_block_dict("author_info_block")

    def get_blocks(self) -> dict[str, list[str]]:
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
            raise ValueError(f"Unknown block type: {self.block_type}")

    def get_constraint(
        self,
        signature_id_1: str,
        signature_id_2: str,
        low_value: float | int = 0,
        high_value: float | int = LARGE_DISTANCE,
        dont_merge_cluster_seeds: bool = True,
        incremental_dont_use_cluster_seeds: bool = False,
        suppress_orcid: bool = False,
    ) -> float | None:
        """Apply pairwise hard constraints for a signature pair.

        Precedence:
        1) Apply passed-in cluster seed constraints first (`disallow`/`require`).
        2) Optionally disallow merging signatures that belong to different
           required-seed groups when `dont_merge_cluster_seeds` is enabled.
        3) If both ORCIDs are present and equal and `suppress_orcid` is false, return `low_value`.
        4) Return `high_value` for deterministic conflicts:
           - normalized last names disagree (hyphen/space-insensitive)
           - first initials disagree
           - first names are neither compatible prefixes nor known aliases
             from `self.name_tuples`
           - middle-name evidence is mutually conflicting (initials or full
             middle tokens)

        If no hard rule applies, return `None`.

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
            If true, ignore cluster-seed require groups, including the derived
            cross-group disallow rule. Explicit `cluster_seeds_disallow` pairs
            still apply as hard negatives.
        suppress_orcid: bool
            If true, do not use same-ORCID equality as a must-link constraint.

        Returns
        -------
        float: the constraint value
        """
        signature_1 = self.signatures[signature_id_1]
        signature_2 = self.signatures[signature_id_2]

        def _materialize_constraint_name_parts(signature: Signature) -> tuple[str, str]:
            first = signature.author_info_first_normalized_without_apostrophe
            middle = signature.author_info_middle_normalized_without_apostrophe
            if first is None or middle is None:
                computed = canonicalize_name_parts(
                    signature.author_info_first,
                    signature.author_info_middle,
                    None,
                )
                if first is None:
                    first = computed.first
                if middle is None:
                    middle = computed.middle
            return first or "", middle or ""

        def _materialize_constraint_last_normalized(signature: Signature) -> str:
            if signature.author_info_last_normalized is not None:
                return signature.author_info_last_normalized
            return canonicalize_name_text(signature.author_info_last)

        first_1, middle_1_text = _materialize_constraint_name_parts(signature_1)
        first_2, middle_2_text = _materialize_constraint_name_parts(signature_2)
        middle_1 = middle_1_text.split()

        orcid_1 = normalize_orcid_compact(signature_1.author_info_orcid)
        orcid_2 = normalize_orcid_compact(signature_2.author_info_orcid)

        # Explicit disallow pairs are hard negatives; the incremental flag only
        # suppresses seed-cluster require groups and derived cross-group disallows.
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
            and (not incremental_dont_use_cluster_seeds)
            and (signature_id_1 in self.cluster_seeds_require and signature_id_2 in self.cluster_seeds_require)
            and (self.cluster_seeds_require[signature_id_1] != self.cluster_seeds_require[signature_id_2])
        ):
            return CLUSTER_SEEDS_LOOKUP["disallow"]
        # orcid is a very reliable indicator: if 2 orcids are present and equal, then they are the same person
        # but if they are not equal, we can't say much
        elif not suppress_orcid and orcid_1 is not None and orcid_2 is not None and orcid_1 == orcid_2:
            return low_value
        # just-in-case last name constraint: if canonical last names differ at
        # compare time, then disallow. Dash/space variants canonicalize to the
        # same spaced form (D5); joined-vs-spaced spellings are additionally
        # equivalent here by compare-time policy (see canonical_lasts_equivalent).
        elif not canonical_lasts_equivalent(
            _materialize_constraint_last_normalized(signature_1),
            _materialize_constraint_last_normalized(signature_2),
        ):
            return high_value
        # just-in-case first initial constraint: if first initials are different, then disallow
        elif len(first_1) > 0 and len(first_2) > 0 and first_1[0] != first_2[0]:
            return high_value
        # and then name based constraints
        else:
            # either a known alias or a prefix of the other
            # if neither, then we'll say it's impossible to be the same person
            if not first_names_name_compatible(first_1, first_2, self.name_tuples):
                return high_value
            # dont cluster together if there is no intersection between the sets of middle initials
            # and both sets are not empty
            elif len(middle_1) > 0:
                middle_2 = middle_2_text.split()
                if len(middle_2) > 0:
                    overlapping_affixes = set(middle_2).intersection(middle_1).intersection(DROPPED_AFFIXES)
                    middle_1_all = [word for word in middle_1 if len(word) > 0 and word not in overlapping_affixes]
                    middle_2_all = [word for word in middle_2 if len(word) > 0 and word not in overlapping_affixes]
                    middle_1_words = {word for word in middle_1_all if len(word) > 1}
                    middle_2_words = {word for word in middle_2_all if len(word) > 1}
                    middle_1_firsts = {word[0] for word in middle_1_all}
                    middle_2_firsts = {word[0] for word in middle_2_all}
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

    def get_signatures_to_block(self) -> dict[str, str]:
        """
        Creates a dictionary mapping signature id to block key

        Returns
        -------
        Dict: the signature to block dictionary
        """
        signatures_to_block: dict[str, str] = {}
        block_dict = self.get_blocks()
        for block_key, signatures in block_dict.items():
            for signature in signatures:
                signatures_to_block[signature] = block_key
        return signatures_to_block

    def split_blocks_helper(
        self, blocks_dict: dict[str, list[str]]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
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
        # The seeded stratified split is order-sensitive. Preserve the incoming
        # block order here; sorting changes pinned production-eval test sets.
        for block_id, signature in blocks_dict.items():
            x.append(block_id)
            y.append(len(signature))

        # Explicitly set n_init to silence upcoming sklearn default-change warning
        clustering_model = KMeans(
            n_clusters=self.num_clusters_for_block_size,
            random_state=self.random_seed,
            n_init=10,
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

    def group_signature_helper(self, signature_list: list[str]) -> dict[str, list[str]]:
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
        block_to_signatures: dict[str, list[str]] = defaultdict(list)
        for signature_id in signature_list:
            block_to_signatures[self.signature_to_block[signature_id]].append(signature_id)
        return dict(block_to_signatures)

    def split_cluster_signatures(
        self,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        """
        Splits the block dict into train/val/test blocks based on split type requested.
        Options for splitting are `signatures`, `blocks`, and `time`

        Returns
        -------
        train/val/test block dictionaries
        """
        _validate_split_ratios(self.train_ratio, self.val_ratio, self.test_ratio)
        blocks = self.get_blocks()

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
            signature_to_year: dict[str, int] = {}
            for signature_id, signature in self.signatures.items():
                # paper_id should be kept as string, so it can be matched to papers.json
                paper_id = str(signature.paper_id)
                if self.papers[paper_id].year is None:
                    signature_to_year[signature_id] = 0
                else:
                    # mypy: year is Optional[int] on Paper; guarded above, so cast to int here
                    signature_to_year[signature_id] = int(self.papers[paper_id].year)

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
            raise ValueError(f"Unknown unit_of_data_split: {self.unit_of_data_split}")

    def split_cluster_signatures_fixed(
        self,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        """
        Splits the block dict into train/val/test blocks based on a fixed block
        based split

        Returns
        -------
        train/val/test block dictionaries
        """
        blocks = self.get_blocks()

        train_block_dict: dict[str, list[str]] = {}
        val_block_dict: dict[str, list[str]] = {}
        test_block_dict: dict[str, list[str]] = {}

        if self.val_blocks is None:
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
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

        train_set = {signature for signatures in train_block_dict.values() for signature in signatures}
        val_set = {signature for signatures in val_block_dict.values() for signature in signatures}
        test_set = {signature for signatures in test_block_dict.values() for signature in signatures}
        intersection_1 = train_set.intersection(test_set)
        intersection_2 = train_set.intersection(val_set)
        intersection_3 = val_set.intersection(test_set)
        intersection = intersection_1.union(intersection_2).union(intersection_3)

        assert len(intersection) == 0, f"Intersection between train/val/test is {intersection}"

        return train_block_dict, val_block_dict, test_block_dict

    def split_data_signatures_fixed(
        self,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
        """
        Splits the block dict into train/val/test blocks based on a fixed signature
        based split

        Returns
        -------
        train/val/test block dictionaries
        """
        train_block_dict: dict[str, list[str]] = {}
        val_block_dict: dict[str, list[str]] = {}
        test_block_dict: dict[str, list[str]] = {}

        test_signatures = self.test_signatures

        if self.val_signatures is None:
            train_signatures = []
            val_signatures = []
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
            np.random.seed(self.random_seed)
            split_prob = np.random.rand(len(self.train_signatures))
            for signature, p in zip(self.train_signatures, split_prob, strict=True):
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
        train_signatures: dict[str, list[str]],
        val_signatures: dict[str, list[str]],
        test_signatures: dict[str, list[str]],
    ) -> tuple[
        list[tuple[str, str, int | float]],
        list[tuple[str, str, int | float]],
        list[tuple[str, str, int | float]],
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
        use_block_sampling = _pair_sampling_uses_blocks(self.pair_sampling_mode)
        train_signature_ids = (
            [] if use_block_sampling else [sig for signatures in train_signatures.values() for sig in signatures]
        )
        val_signature_ids = (
            [] if use_block_sampling else [sig for signatures in val_signatures.values() for sig in signatures]
        )
        test_signature_ids = (
            [] if use_block_sampling else [sig for signatures in test_signatures.values() for sig in signatures]
        )

        train_pairs = self.pair_sampling(
            self.train_pairs_size,
            train_signature_ids,
            train_signatures,
        )
        val_pairs = (
            self.pair_sampling(
                self.val_pairs_size,
                val_signature_ids,
                val_signatures,
            )
            if len(val_signatures) > 0
            else []
        )

        test_pairs = self.pair_sampling(
            self.test_pairs_size,
            test_signature_ids,
            test_signatures,
            self.all_test_pairs_flag,
        )

        return train_pairs, val_pairs, test_pairs

    def construct_cluster_to_signatures(
        self,
        block_dict: dict[str, list[str]],
    ) -> dict[str, list[str]]:
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
        if self.signature_to_cluster_id is None:
            raise ValueError("signature_to_cluster_id is required to construct cluster_to_signatures")
        signature_to_cluster_id = self.signature_to_cluster_id
        cluster_to_signatures = defaultdict(list)
        for signatures in block_dict.values():
            for signature in signatures:
                true_cluster_id = signature_to_cluster_id[signature]
                cluster_to_signatures[true_cluster_id].append(signature)

        return dict(cluster_to_signatures)

    def fixed_pairs(
        self,
    ) -> tuple[
        list[tuple[str, str, int | float]],
        list[tuple[str, str, int | float]],
        list[tuple[str, str, int | float]],
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
        train_pairs_df = _map_fixed_pair_labels(self.train_pairs, "train")
        if self.val_pairs is not None:
            val_pairs_df = _map_fixed_pair_labels(self.val_pairs, "val")
            train_pairs = list(train_pairs_df.to_records(index=False))
            val_pairs = list(val_pairs_df.to_records(index=False))
        else:
            np.random.seed(self.random_seed)
            # split train into train/val
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
            msk = np.random.rand(len(train_pairs_df)) < train_prob
            train_pairs = list(train_pairs_df[msk].to_records(index=False))
            val_pairs = list(train_pairs_df[~msk].to_records(index=False))
        test_pairs_df = _map_fixed_pair_labels(self.test_pairs, "test")
        test_pairs = list(test_pairs_df.to_records(index=False))

        return train_pairs, val_pairs, test_pairs

    def all_pairs(self) -> list[tuple[str, str, int | float]]:
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
        signature_ids: list[str],
        blocks: dict[str, list[str]],
        all_pairs: bool = False,
    ) -> list[tuple[str, str, int | float]]:
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
        pair_sampling_mode = _validate_pair_sampling_mode(str(self.pair_sampling_mode))

        same_name_different_cluster: list[tuple[str, str, int | float]] = []
        same_name_same_cluster: list[tuple[str, str, int | float]] = []
        different_name_same_cluster: list[tuple[str, str, int | float]] = []
        different_name_different_cluster: list[tuple[str, str, int | float]] = []
        possible: list[tuple[str, str, int | float]] = []

        if pair_sampling_mode == "global_balanced_classes":
            if self.signature_to_cluster_id is None:
                raise ValueError("signature_to_cluster_id is required for non-block pair sampling")
            signature_to_cluster_id = self.signature_to_cluster_id
            for i, s1 in enumerate(signature_ids):
                for s2 in signature_ids[i + 1 :]:
                    s1_name = self.get_full_name(s1)
                    s2_name = self.get_full_name(s2)
                    s1_cluster = signature_to_cluster_id[s1]
                    s2_cluster = signature_to_cluster_id[s2]
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
        elif pair_sampling_mode == "within_block_random":
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
            if self.signature_to_cluster_id is None:
                raise ValueError("signature_to_cluster_id is required for balanced pair sampling")
            signature_to_cluster_id = self.signature_to_cluster_id
            for _, signatures in blocks.items():
                for i, s1 in enumerate(signatures):
                    for s2 in signatures[i + 1 :]:
                        s1_name = self.get_full_name(s1)
                        s2_name = self.get_full_name(s2)
                        s1_cluster = signature_to_cluster_id[s1]
                        s2_cluster = signature_to_cluster_id[s2]
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
            if pair_sampling_mode != "within_block_random":
                all_pairs_output: list[tuple[str, str, int | float]] = (
                    same_name_different_cluster
                    + same_name_same_cluster
                    + different_name_same_cluster
                    + different_name_different_cluster
                )
                return all_pairs_output
            else:
                return possible
        else:
            if pair_sampling_mode in {
                "within_block_balanced_classes",
                "within_block_balanced_homonym_synonym",
                "global_balanced_classes",
            }:
                pairs = sampling(
                    same_name_different_cluster,
                    different_name_same_cluster,
                    same_name_same_cluster,
                    different_name_different_cluster,
                    sample_size,
                    pair_sampling_mode == "within_block_balanced_homonym_synonym",
                    self.random_seed,
                )
            elif pair_sampling_mode == "within_block_random":
                sample_size = min(len(possible), sample_size)
                pairs = random_sampling(possible, sample_size, self.random_seed)
            else:
                raise ValueError(
                    "Unsupported pair sampling configuration for non-exhaustive sampling "
                    f"(pair_sampling_mode={pair_sampling_mode})"
                )
            return pairs


def preprocess_paper_1(item: tuple[str, Paper], *, preprocess: bool = True) -> tuple[str, Paper]:
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

    if paper.in_signatures:
        language_detection = detect_language(paper.title)
        paper = paper._replace(
            is_english=language_detection.is_english,
            predicted_language=language_detection.predicted_language,
            is_reliable=language_detection.is_reliable,
            language_reliability=language_detection.language_reliability,
        )
    title = normalize_title(paper.title)
    title_ngrams_words = get_text_ngrams_words(title, drop_short_tokens=False)
    authors = [
        Author(
            position=author.position,
            author_name=normalize_text(author.author_name),
        )
        for author in paper.authors
    ]
    paper = paper._replace(title=title, title_ngrams_words=title_ngrams_words, authors=authors)

    if preprocess:
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


def preprocess_papers_parallel(
    papers_dict: dict,
    n_jobs: int,
    preprocess: bool,
) -> dict:
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
    output: dict = {}
    use_pool_stage_1 = n_jobs > 1 and platform.system() == "Linux"
    if use_pool_stage_1:
        # Linux/WSL2: force process workers for CPU-bound paper preprocessing.
        with UniversalPool(processes=n_jobs, use_threads=False) as p:
            _max = len(papers_dict)
            with tqdm(total=_max, desc="Preprocessing papers") as pbar:
                func = partial(preprocess_paper_1, preprocess=preprocess)
                for key, value in p.imap(func, papers_dict.items(), CHUNK_SIZE):
                    output[key] = value
                    pbar.update()
    else:
        for item in tqdm(papers_dict.items(), total=len(papers_dict), desc="Preprocessing papers"):
            k, v = preprocess_paper_1(item, preprocess=preprocess)
            output[k] = v

    return output

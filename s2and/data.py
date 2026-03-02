import json
import logging
import os
import pickle
import platform
import re
import tempfile
import threading
import time
from collections import Counter, defaultdict
from functools import partial, reduce
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from s2and.consts import (
    CLUSTER_SEEDS_LOOKUP,
    LARGE_DISTANCE,
    NAME_COUNTS_PATH,
    NUMPY_NAN,
    PROJECT_ROOT_PATH,
)
from s2and.file_cache import cached_path
from s2and.mp import UniversalPool
from s2and.runtime import (
    RuntimeContext,
    build_runtime_context,
    detect_rust_runtime_capabilities,
    stage_uses_rust,
)
from s2and.rust_lifecycle import build_rust_lifecycle_policy
from s2and.sampling import random_sampling, sampling
from s2and.text import (
    AFFILIATIONS_STOP_WORDS,
    DROPPED_AFFIXES,
    NAME_PREFIXES,
    ORCID_PATTERN,
    VENUE_STOP_WORDS,
    compute_block,
    detect_language,
    get_text_ngrams,
    get_text_ngrams_words,
    normalize_text,
    same_prefix_tokens,
    split_first_middle_hyphen_aware,
)

logger = logging.getLogger("s2and")

# Lazy-initialized global for Sinonym detector within worker processes
_SINONYM_DETECTOR = None  # type: ignore
_SINONYM_DETECTOR_LOCK = threading.Lock()
CHUNK_SIZE = 1000  # for multiprocessing imap chunks
_PAIR_LABEL_MAP: dict[str | int, int] = {"NO": 0, "YES": 1, "0": 0, 0: 0, "1": 1, 1: 1}

# Cache for large, immutable resources loaded across instances
_NAME_COUNTS_CACHE: tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]] | None = None
_NAME_COUNTS_CACHE_LOCK = threading.Lock()
SIGNATURE_PREPROCESS_BATCH_SIZE = 2048
NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY = "legacy_full_first_token"
NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR = "initial_char"
NameCountsLastFirstInitialSemantics = Literal[
    "legacy_full_first_token",
    "initial_char",
]


# ------------------------ Local helpers (backcompat shims) ------------------------


def _lasts_equivalent_for_constraint(l1: str, l2: str) -> bool:
    """Treat hyphen/space variants as equivalent for last-name constraint checks.

    Examples: "ou yang" == "ouyang"; strictly unequal strings otherwise.

    TODO(s2and): Remove when canonicalization is unified end-to-end and constraints
    operate on canonical forms only.
    """
    if l1 == l2:
        return True
    return l1.replace(" ", "") == l2.replace(" ", "")


def _canonicalize_last_for_counts(raw_last: str | None, normalized_last: str) -> str:
    """Canonicalize last name for legacy count lookups.

    Join internal spaces for hyphen/compound surnames so historical single-token
    count keys still match (e.g., "ou yang" -> "ouyang").

    TODO(s2and): Remove once name count tables are regenerated with hyphen-aware surnames.
    """
    if (raw_last is not None and "-" in raw_last) or (" " in normalized_last):
        return (normalized_last or "").replace(" ", "")
    return normalized_last or ""


def _load_name_counts_cached() -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Load name count dictionaries once per process and cache them.

    Avoids repeatedly unpickling ~600MB file in tests/short runs.
    """
    global _NAME_COUNTS_CACHE
    if _NAME_COUNTS_CACHE is not None:
        return _NAME_COUNTS_CACHE  # type: ignore[return-value]
    with _NAME_COUNTS_CACHE_LOCK:
        # Double-check after acquiring lock (another thread may have loaded).
        if _NAME_COUNTS_CACHE is None:
            with open(cached_path(NAME_COUNTS_PATH), "rb") as f:
                _NAME_COUNTS_CACHE = pickle.load(f)
    return _NAME_COUNTS_CACHE  # type: ignore[return-value]


def _load_name_tuples_from_file(filename: str) -> set[tuple[str, str]]:
    resolved: set[tuple[str, str]] = set()
    with open(os.path.join(PROJECT_ROOT_PATH, "data", filename), encoding="utf-8") as tuples_file:
        for line in tuples_file:
            line_split = line.strip().split(",")
            if len(line_split) >= 2:
                resolved.add((line_split[0], line_split[1]))
    return resolved


def _resolve_name_counts_last_first_initial_semantics(
    value: str | None,
    *,
    default: NameCountsLastFirstInitialSemantics = NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
    strict: bool = True,
) -> NameCountsLastFirstInitialSemantics:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {
        NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY,
        NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
    }:
        return normalized  # type: ignore[return-value]
    if strict:
        raise ValueError(
            "name_counts_last_first_initial_semantics must be one of "
            f"{NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY!r}, {NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR!r}; got {value!r}"
        )
    return default


def _signature_preprocess_backend_decision(runtime_context: RuntimeContext) -> bool:
    use_rust_backend = stage_uses_rust(runtime_context)
    if not use_rust_backend:
        return False

    rust_module_available = False
    try:
        from s2and import feature_port

        rust_module_available = feature_port.rust_featurizer_available()
    except Exception:
        rust_module_available = False

    if not rust_module_available:
        raise RuntimeError(
            "Rust backend requested for ingest_preprocess but s2and_rust extension is unavailable "
            f"(run_id={runtime_context.run_id})"
        )

    return True


def _ordered_coauthors_for_signature(signature: "Signature", papers: dict[str, "Paper"]) -> list[str]:
    paper = papers.get(str(signature.paper_id))
    if paper is None:
        logger.warning(
            "Missing paper for signature ngram materialization; treating coauthors as empty "
            "(signature_id=%s paper_id=%s)",
            signature.signature_id,
            signature.paper_id,
        )
        return []
    return [author.author_name for author in paper.authors if author.position != signature.author_info_position]


def _python_signature_ngrams_batch(
    coauthor_texts: list[str], affiliation_texts: list[str]
) -> tuple[list[Counter], list[Counter]]:
    coauthor_counters = [
        get_text_ngrams(text, stopwords=None, use_bigrams=True) if text else Counter() for text in coauthor_texts
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
    author_info_first_normalized: str | None
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
    predicted_language: str | None
    title_ngrams_words: Counter | None
    authors: list[Author]
    venue: str | None
    journal_name: str | None
    title_ngrams_chars: Counter | None
    venue_ngrams: Counter | None
    journal_ngrams: Counter | None
    reference_details: tuple[Counter, Counter, Counter, Counter] | None
    year: int | None
    references: list[int] | None
    paper_id: int


class MiniPaper(NamedTuple):
    title: str
    venue: str | None
    journal_name: str | None
    authors: list[str]


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
        use_sinonym_overwrite: if True, run a pre-step that batch-detects Chinese names per paper via
            Sinonym and overwrites the corresponding signature name parts with Sinonym's normalized output.
            Also applies Sinonym-normalized names to the per-paper author list so co-author features
            (coauthor sets/blocks and n-grams) are derived from the normalized names as well.
        name_counts_last_first_initial_semantics: semantics for constructing the
            `last_first_initial` lookup key in `name_counts`.
            - "initial_char": `<last> <first[0]>` (current semantics)
            - "legacy_full_first_token": `<last> <first_token>` (legacy compatibility)
        sinonym_overwrite_min_ratio: optional gating threshold; only counts multi-author papers.
            Let a,b be single-author flips/not-flips and x,y be multi-author flips/not-flips.
            If (x + y) > 0: overwrite when x >= min_ratio * y; else (no multi-author evidence):
            overwrite when a > 0 (otherwise do not overwrite).
            it qualifies by default (equivalent to 1 vs 0: flip).
            (building reference_details Counters). Defaults to False. When False, reference_details
            are initialized to empty Counters to maintain featurization compatibility while
            avoiding the expensive reference graph materialization.
    """

    def __init__(
        self,
        signatures: str | dict,
        papers: str | dict,
        name: str,
        mode: str = "train",
        clusters: str | dict | None = None,
        specter_embeddings: str | dict | None = None,
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
        pair_sampling_block: bool = True,
        pair_sampling_balanced_classes: bool = False,
        pair_sampling_balanced_homonym_synonym: bool = False,
        all_test_pairs_flag: bool = False,
        random_seed: int = 1111,
        load_name_counts: bool | dict = True,
        n_jobs: int = 1,
        preprocess: bool = True,
        name_tuples: set[tuple[str, str]] | str | None = "filtered",
        use_orcid_id: bool = True,
        use_sinonym_overwrite: bool = False,
        name_counts_last_first_initial_semantics: NameCountsLastFirstInitialSemantics | None = None,
        sinonym_overwrite_min_ratio: float | None = 3.0,
        compute_reference_features: bool = False,
    ):
        init_start = time.perf_counter()
        self.runtime_context = build_runtime_context("dataset_build")
        self.signatures_path = signatures if isinstance(signatures, str) else None
        self.papers_path = papers if isinstance(papers, str) else None
        self._rust_ingest_tmpdir = None  # TemporaryDirectory; prevent leak
        self._s2and_python_pair_ngrams_ready: bool = False
        self._rust_cluster_seeds_require_id: int | None = None
        self._rust_cluster_seeds_require_len: int | None = None
        self._rust_cluster_seeds_disallow_id: int | None = None
        self._rust_cluster_seeds_disallow_len: int | None = None
        self.clusters_path = clusters if isinstance(clusters, str) else None
        self.cluster_seeds_path = cluster_seeds if isinstance(cluster_seeds, str) else None
        self.specter_embeddings_path = specter_embeddings if isinstance(specter_embeddings, str) else None
        rust_capabilities = detect_rust_runtime_capabilities()
        self.rust_lifecycle_policy = build_rust_lifecycle_policy(
            backend=self.runtime_context.resolved_backend,
            mode=mode,
            has_signatures_path=self.signatures_path is not None,
            has_papers_path=self.papers_path is not None,
            preprocess=preprocess,
            compute_reference_features=compute_reference_features,
            use_rust=self.runtime_context.use_rust,
            from_dataset_paper_preprocess_available=rust_capabilities.from_dataset_paper_preprocess_available,
            use_sinonym_overwrite=use_sinonym_overwrite,
        )
        defer_rust_json_ingest_write_for_sinonym = self.rust_lifecycle_policy.defer_rust_json_ingest_write_for_sinonym

        if mode == "train":
            if train_blocks is not None and block_type != "original":
                logger.warning("If you are passing in training/val/test blocks, then you may want original blocks.")

            if unit_of_data_split == "blocks" and not pair_sampling_block:
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
        raw_signatures_for_json: dict[Any, Any] | None = None
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
        logger.debug(
            "Telemetry stage: stage=anddata_ingest_signatures seconds=%.3f signatures=%d",
            time.perf_counter() - signatures_stage_start,
            len(self.signatures),
        )

        # Determine the set of papers referenced by signatures
        needed_paper_ids: set[str] = set(str(sig.paper_id) for sig in self.signatures.values())

        papers_stage_start = time.perf_counter()
        logger.info("loading papers (subset referenced by signatures)")
        raw_papers = self.maybe_load_json(papers)
        filtered_papers = {pid: p for pid, p in raw_papers.items() if str(pid) in needed_paper_ids}
        rust_json_ingest_from_paths = bool(
            self.rust_lifecycle_policy.rust_build_path == "from_json_paths"
            and self.signatures_path is not None
            and self.papers_path is not None
        )
        if (
            rust_json_ingest_from_paths
            and len(filtered_papers) < len(raw_papers)
            and not defer_rust_json_ingest_write_for_sinonym
        ):
            if raw_signatures_for_json is None:
                raw_signatures_for_json = dict(raw_signatures)
            self._materialize_rust_json_ingest_filtered_payload(
                raw_signatures_for_json=raw_signatures_for_json,
                filtered_papers_for_json=filtered_papers,
                source_papers_count=len(raw_papers),
            )
        self.papers = {}
        # convert dictionary to namedtuples for memory reduction
        for paper_id, paper in filtered_papers.items():
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
        logger.info(f"loaded papers subset: {len(self.papers)}/{len(raw_papers)} relevant")
        logger.debug(
            "Telemetry stage: stage=anddata_ingest_papers seconds=%.3f retained_papers=%d source_papers=%d",
            time.perf_counter() - papers_stage_start,
            len(self.papers),
            len(raw_papers),
        )

        # Optional Sinonym pre-step: normalize Chinese names from papers and overwrite signatures
        # This runs before other preprocessing so downstream steps use updated names
        if use_sinonym_overwrite:
            sinonym_results = sinonym_preprocess_papers_parallel(self.papers, n_jobs)
            allow_overwrite_pos = None
            # Optional gating: only overwrite names that are flipped >= min_ratio * not_flipped
            if sinonym_overwrite_min_ratio is not None:
                try:
                    allow_overwrite_pos = compute_sinonym_overwrite_allowlist(
                        self.signatures, sinonym_results, min_ratio=sinonym_overwrite_min_ratio
                    )
                except (TypeError, ValueError, AttributeError, KeyError) as e:
                    logger.warning(
                        "Sinonym overwrite gating failed (%s), proceeding without gating: %s",
                        type(e).__name__,
                        e,
                    )
                    allow_overwrite_pos = None
            # Only allow block overwrites during inference to keep train/val/test splits reproducible
            overwrite_count = apply_sinonym_overwrites(
                self.signatures,
                sinonym_results,
                overwrite_blocks=(mode == "inference"),
                allow_overwrite_pos=allow_overwrite_pos,
            )
            logger.info(f"Sinonym overwrote {overwrite_count} signature name(s)")
            # Update paper-level author strings so co-author features use Sinonym-normalized names
            paper_overwrite_count = apply_sinonym_overwrites_to_papers(
                self.papers, sinonym_results, allow_overwrite_pos=allow_overwrite_pos
            )
            logger.info(f"Sinonym overwrote {paper_overwrite_count} paper author name(s)")
            if defer_rust_json_ingest_write_for_sinonym:
                if raw_signatures_for_json is None:
                    raw_signatures_for_json = dict(raw_signatures)
                self._sync_in_memory_names_to_rust_json_payload(
                    raw_signatures_for_json=raw_signatures_for_json,
                    filtered_papers_for_json=filtered_papers,
                )
                self._materialize_rust_json_ingest_filtered_payload(
                    raw_signatures_for_json=raw_signatures_for_json,
                    filtered_papers_for_json=filtered_papers,
                    source_papers_count=len(raw_papers),
                )

        self.name = name
        self.mode = mode
        self.name_counts_last_first_initial_semantics = _resolve_name_counts_last_first_initial_semantics(
            name_counts_last_first_initial_semantics,
            default=NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
            strict=True,
        )
        logger.info("loading clusters")
        self.clusters: dict | None = self.maybe_load_json(clusters)
        logger.info("loaded clusters, loading specter")
        self.specter_embeddings = self.maybe_load_specter(specter_embeddings)
        # prevents errors during testing where we have no specter embeddings
        if self.specter_embeddings is None:
            self.specter_embeddings = {}  # type: ignore
        else:
            # Only keep embeddings for papers we retained
            needed_keys = set(self.papers.keys())
            self.specter_embeddings = {k: v for k, v in self.specter_embeddings.items() if str(k) in needed_keys}  # type: ignore
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
        self.pair_sampling_block = pair_sampling_block
        self.pair_sampling_balanced_classes = pair_sampling_balanced_classes
        self.pair_sampling_balanced_homonym_synonym = pair_sampling_balanced_homonym_synonym
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
            self.pair_sampling_block = True
            self.pair_sampling_balanced_classes = False
            self.pair_sampling_balanced_homonym_synonym = False
            self.all_test_pairs_flag = True
            self.block_type = "s2"  # pure inference is for S2 probably?
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        name_counts_loaded = False
        if isinstance(load_name_counts, dict):
            self.first_dict = load_name_counts["first_dict"]
            self.last_dict = load_name_counts["last_dict"]
            self.first_last_dict = load_name_counts["first_last_dict"]
            self.last_first_initial_dict = load_name_counts["last_first_initial_dict"]
            name_counts_loaded = True
        elif load_name_counts:
            logger.info("loading name counts (cached)")
            (
                first_dict,
                last_dict,
                first_last_dict,
                last_first_initial_dict,
            ) = _load_name_counts_cached()
            self.first_dict = first_dict
            self.last_dict = last_dict
            self.first_last_dict = first_last_dict
            self.last_first_initial_dict = last_first_initial_dict
            name_counts_loaded = True
            logger.info("loaded name counts")
        self.name_counts_loaded = bool(name_counts_loaded)

        self.n_jobs = n_jobs
        self.compute_reference_features = compute_reference_features
        self.signature_to_block = self.get_signatures_to_block()
        papers_from_signatures = {str(signature.paper_id) for signature in self.signatures.values()}
        for paper_id, paper in self.papers.items():
            self.papers[paper_id] = paper._replace(in_signatures=str(paper_id) in papers_from_signatures)
        self.preprocess = preprocess

        resolved_name_tuples: set[tuple[str, str]]
        if name_tuples == "filtered":
            resolved_name_tuples = _load_name_tuples_from_file("s2and_name_tuples_filtered.txt")
        elif name_tuples is None:
            resolved_name_tuples = _load_name_tuples_from_file("s2and_name_tuples.txt")
        elif isinstance(name_tuples, set):
            resolved_name_tuples = name_tuples
        else:
            raise ValueError("name_tuples must be None, 'filtered', or a set of (first_a, first_b) tuples")
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
                compute_reference_features=self.compute_reference_features,
            )
            logger.info("preprocessed papers")
        logger.debug(
            "Telemetry stage: stage=anddata_preprocess_papers seconds=%.3f papers=%d",
            time.perf_counter() - preprocess_papers_stage_start,
            len(self.papers),
        )

        preprocess_signatures_stage_start = time.perf_counter()
        logger.info("preprocessing signatures")
        self.preprocess_signatures(name_counts_loaded)
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

    def _sync_in_memory_names_to_rust_json_payload(
        self,
        *,
        raw_signatures_for_json: dict[Any, Any],
        filtered_papers_for_json: dict[Any, Any],
    ) -> None:
        for signature_id, signature in self.signatures.items():
            signature_payload = raw_signatures_for_json.get(signature_id)
            if signature_payload is None:
                signature_payload = raw_signatures_for_json.get(str(signature_id))
            if not isinstance(signature_payload, dict):
                continue
            author_info_payload = signature_payload.get("author_info")
            if not isinstance(author_info_payload, dict):
                continue
            author_info_payload["first"] = signature.author_info_first
            author_info_payload["middle"] = signature.author_info_middle
            author_info_payload["last"] = signature.author_info_last
            author_info_payload["block"] = signature.author_info_block

        for paper_id, paper_payload in filtered_papers_for_json.items():
            paper = self.papers.get(paper_id)
            if paper is None:
                paper = self.papers.get(str(paper_id))
            if paper is None or not isinstance(paper_payload, dict):
                continue
            author_payloads = paper_payload.get("authors")
            if not isinstance(author_payloads, list):
                continue
            position_to_name: dict[Any, str] = {}
            for author in paper.authors:
                position_to_name[author.position] = author.author_name
                position_to_name[str(author.position)] = author.author_name
            for author_payload in author_payloads:
                if not isinstance(author_payload, dict):
                    continue
                author_name = position_to_name.get(author_payload.get("position"))
                if author_name is not None:
                    author_payload["author_name"] = author_name

    def _materialize_rust_json_ingest_filtered_payload(
        self,
        *,
        raw_signatures_for_json: dict[Any, Any],
        filtered_papers_for_json: dict[Any, Any],
        source_papers_count: int,
    ) -> None:
        filtered_paths_start = time.perf_counter()
        if self._rust_ingest_tmpdir is None:
            self._rust_ingest_tmpdir = tempfile.TemporaryDirectory(prefix="s2and_rust_json_ingest_")
        filtered_dir = self._rust_ingest_tmpdir.name
        filtered_signatures_path = os.path.join(filtered_dir, "signatures_filtered.json")
        filtered_papers_path = os.path.join(filtered_dir, "papers_filtered.json")
        with open(filtered_signatures_path, "w", encoding="utf-8") as signatures_file:
            json.dump(raw_signatures_for_json, signatures_file)
        with open(filtered_papers_path, "w", encoding="utf-8") as papers_file:
            json.dump(filtered_papers_for_json, papers_file)
        self.signatures_path = filtered_signatures_path
        self.papers_path = filtered_papers_path
        logger.info(
            "Rust JSON ingest: materialized filtered JSON payloads signatures=%d papers=%d source_papers=%d "
            "seconds=%.3f",
            len(raw_signatures_for_json),
            len(filtered_papers_for_json),
            source_papers_count,
            time.perf_counter() - filtered_paths_start,
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
        # Backward-compatibility for name count keys:
        # - Historically, counts used the legacy single-token `author_info_first_normalized`.
        # - With Sinonym, `author_info_first_normalized_without_apostrophe` can contain multiple tokens
        #   for hyphenated Chinese given names (e.g., "qi xin"). For counts only, we heuristically
        #   join internal spaces to form a single token ("qixin") IF the raw first contained a hyphen.
        # - This preserves old behavior for most names while improving lookups for hyphenated cases.
        # TODO: revisit once we re-extract name_counts using Sinonym-aware canonicalization.
        counts_first_without_apostrophe = first_without_apostrophe
        counts_last_normalized = last_normalized
        if counts_first_without_apostrophe is None or counts_last_normalized is None:
            counts_first_without_apostrophe, _ = split_first_middle_hyphen_aware(first_raw, middle_raw)
            counts_last_normalized = normalize_text(signature.author_info_last)
        # need this for name counts (legacy single-token behavior)
        first_normalized_token_for_counts = (
            counts_first_without_apostrophe.split(" ")[0] if counts_first_without_apostrophe else ""
        )
        first_for_counts = first_normalized_token_for_counts
        if "-" in first_raw:
            joined = (counts_first_without_apostrophe or "").replace(" ", "")
            if joined:
                first_for_counts = joined

        # Backward-compatibility for last name keys:
        # - Historically, last names were single tokens; normalization turns hyphens into spaces
        #   (e.g., "ou-yang" -> "ou yang"). For counts only, treat space/hyphen variants as the
        #   same token by joining internal spaces ("ouyang").
        # TODO(s2and): remove this once name_counts are regenerated with hyphen-aware surnames.
        last_for_counts = _canonicalize_last_for_counts(signature.author_info_last, counts_last_normalized)

        first_last_for_count = (first_for_counts + " " + last_for_counts).strip()
        if self.name_counts_last_first_initial_semantics == NAME_COUNTS_LAST_FIRST_INITIAL_LEGACY:
            last_first_initial_for_count = (last_for_counts + " " + first_for_counts).strip()
        else:
            first_initial = first_for_counts[0] if first_for_counts else ""
            last_first_initial_for_count = (last_for_counts + " " + first_initial).strip()

        return NameCounts(
            first=(self.first_dict.get(first_for_counts, 1) if len(first_for_counts) > 1 else np.nan),  # type: ignore
            last=self.last_dict.get(last_for_counts, 1),
            first_last=(
                self.first_last_dict.get(first_last_for_count, 1) if len(first_for_counts) > 1 else np.nan  # type: ignore
            ),
            last_first_initial=self.last_first_initial_dict.get(last_first_initial_for_count, 1),
        )

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
        runtime_context = self.runtime_context
        use_rust_backend = _signature_preprocess_backend_decision(runtime_context)
        use_rust_featurizer = use_rust_backend
        rust_module_available = use_rust_backend
        defer_signature_ngrams_to_rust = self.rust_lifecycle_policy.defer_signature_ngrams_to_rust
        defer_signature_fields_to_rust = self.rust_lifecycle_policy.defer_signature_fields_to_rust
        logger.info(
            "Signature preprocessing backend decision: backend=%s use_rust_featurizer=%s rust_module_available=%s "
            "defer_signature_ngrams_to_rust=%s defer_signature_fields_to_rust=%s "
            "requested_backend=%s resolved_backend=%s run_id=%s",
            "rust" if use_rust_backend else "python",
            use_rust_featurizer,
            rust_module_available,
            defer_signature_ngrams_to_rust,
            defer_signature_fields_to_rust,
            runtime_context.requested_backend,
            runtime_context.resolved_backend,
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
                    stored_first_normalized_token: str | None = signature.author_info_first_normalized
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
                        set(compute_block(author) for author in coauthors) if coauthors is not None else None
                    )

                    affiliations: list[str] = signature.author_info_affiliations
                    full_name = signature.author_info_full_name
                    counts = signature.author_info_name_counts
                    normalized_orcid = signature.author_info_orcid
                    coauthor_text = ""
                    affiliation_text = ""

                    if self.preprocess:
                        if defer_signature_fields_to_rust:
                            stored_first_normalized_token = None
                            stored_first_without_apostrophe = None
                            stored_middle_without_apostrophe = None
                            stored_last_normalized = None
                            stored_suffix_normalized = None
                            coauthor_set = None
                            coauthor_blocks = None
                            if full_name is None:
                                full_name = _assemble_full_name(
                                    [
                                        signature.author_info_first,
                                        signature.author_info_middle,
                                        signature.author_info_last,
                                        signature.author_info_suffix,
                                    ]
                                )
                        else:
                            # our normalization scheme is to normalize first and middle separately,
                            # join them, then take the first token of the combined join
                            # NOTE: Hyphen-aware handling
                            # - First/middle: handled via split_first_middle_hyphen_aware (keeps hyphenated Chinese
                            #   given names together).
                            # - Surname: for downstream lookups/constraints we also treat hyphen/space variants
                            #   equivalently.
                            # TODO(s2and): Once name_counts/name_tuples are regenerated with Sinonym-aware
                            #              canonicalization, remove the backward-compat shims added below for
                            #              last-name counts/constraints.
                            # Default normalization (keeps legacy behavior for counts/lookups)
                            first_normalized = normalize_text(first_raw)
                            middle_normalized = normalize_text(middle_raw)
                            first_middle_normalized_split = (first_normalized + " " + middle_normalized).split(" ")
                            if first_middle_normalized_split and first_middle_normalized_split[0] in NAME_PREFIXES:
                                first_middle_normalized_split = first_middle_normalized_split[1:]

                            # Hyphen-preserving split for the "without_apostrophe" canonical fields
                            # Centralize in s2and.text for reuse by other scripts
                            first_without_apostrophe, middle_without_apostrophe = split_first_middle_hyphen_aware(
                                first_raw,
                                middle_raw,
                            )
                            # need this for name counts (legacy single-token behavior)
                            # canonical fields used across featurization, prediction, etc.
                            stored_first_normalized_token = (
                                first_middle_normalized_split[0] if first_middle_normalized_split else ""
                            )
                            stored_first_without_apostrophe = first_without_apostrophe
                            stored_middle_without_apostrophe = middle_without_apostrophe
                            stored_last_normalized = normalize_text(signature.author_info_last)
                            stored_suffix_normalized = normalize_text(signature.author_info_suffix or "")
                            affiliations = [
                                normalize_text(affiliation) for affiliation in signature.author_info_affiliations
                            ]
                            if not defer_signature_ngrams_to_rust:
                                coauthor_text, affiliation_text = _build_signature_ngram_texts(
                                    coauthors=coauthors or [],
                                    affiliations=affiliations,
                                    normalize_coauthors=False,
                                    normalize_affiliations=False,
                                )

                        if load_name_counts:
                            counts = self._compute_signature_name_counts(
                                signature,
                                first_raw=first_raw,
                                middle_raw=middle_raw,
                                first_without_apostrophe=stored_first_without_apostrophe,
                                last_normalized=stored_last_normalized,
                            )
                        else:
                            counts = NameCounts(first=None, last=None, first_last=None, last_first_initial=None)

                        if not defer_signature_fields_to_rust:
                            full_name = _assemble_full_name(
                                [
                                    stored_first_without_apostrophe or signature.author_info_first,
                                    stored_middle_without_apostrophe or signature.author_info_middle,
                                    stored_last_normalized or signature.author_info_last,
                                    stored_suffix_normalized or signature.author_info_suffix,
                                ]
                            )

                            # we need a regex to extract the 16 digits, keeping in mind the last digit could be X
                            if signature.author_info_orcid is not None:
                                orcid = re.findall(ORCID_PATTERN, signature.author_info_orcid)
                                if len(orcid) > 0:
                                    normalized_orcid = orcid[0].upper().replace("-", "")
                                else:
                                    normalized_orcid = None

                    batch_rows.append(
                        {
                            "signature_id": signature_id,
                            "signature": signature,
                            "first_normalized_token": stored_first_normalized_token,
                            "first_without_apostrophe": stored_first_without_apostrophe,
                            "middle_without_apostrophe": stored_middle_without_apostrophe,
                            "last_normalized": stored_last_normalized,
                            "suffix_normalized": stored_suffix_normalized,
                            "coauthor_set": coauthor_set,
                            "coauthor_blocks": coauthor_blocks,
                            "affiliations": affiliations,
                            "full_name": full_name,
                            "counts": counts,
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

                for idx, row in enumerate(batch_rows):
                    replace_kwargs = {
                        "author_info_first_normalized": row["first_normalized_token"],
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
                                "author_info_name_counts": row["counts"],
                                "author_info_orcid": row["normalized_orcid"],
                            }
                        )
                    self.signatures[row["signature_id"]] = row["signature"]._replace(**replace_kwargs)

                progress_bar.update(len(batch_signature_ids))

    def _refresh_signature_name_counts(self) -> int:
        if not self.name_counts_loaded:
            return 0
        updated = 0
        for signature_id, signature in self.signatures.items():
            refreshed_counts = self._compute_signature_name_counts(
                signature,
                first_raw=signature.author_info_first or "",
                middle_raw=signature.author_info_middle or "",
                first_without_apostrophe=signature.author_info_first_normalized_without_apostrophe,
                last_normalized=signature.author_info_last_normalized,
            )
            if signature.author_info_name_counts == refreshed_counts:
                continue
            self.signatures[signature_id] = signature._replace(author_info_name_counts=refreshed_counts)
            updated += 1
        return updated

    def set_name_counts_last_first_initial_semantics(self, semantics: str) -> bool:
        resolved = _resolve_name_counts_last_first_initial_semantics(
            semantics,
            default=NAME_COUNTS_LAST_FIRST_INITIAL_INITIAL_CHAR,
            strict=True,
        )
        if resolved == self.name_counts_last_first_initial_semantics:
            return False
        previous = self.name_counts_last_first_initial_semantics
        self.name_counts_last_first_initial_semantics = resolved
        signatures_updated = self._refresh_signature_name_counts()
        try:
            from s2and import feature_port
        except ImportError:
            logger.info(
                "Skipping Rust featurizer eviction while updating name-count semantics "
                "(dataset=%s mode=%s run_id=%s old=%s new=%s): feature_port unavailable",
                self.name,
                self.mode,
                self.runtime_context.run_id,
                previous,
                resolved,
            )
        else:
            try:
                feature_port.evict_rust_featurizer(self)
            except (RuntimeError, AttributeError):
                logger.exception(
                    "Failed to evict Rust featurizer cache during name-count semantics refresh "
                    "(dataset=%s mode=%s run_id=%s old=%s new=%s)",
                    self.name,
                    self.mode,
                    self.runtime_context.run_id,
                    previous,
                    resolved,
                )
                raise
        logger.info(
            "Updated name-count semantics for last_first_initial old=%s new=%s signatures_updated=%d mode=%s",
            previous,
            resolved,
            signatures_updated,
            self.mode,
        )
        return True

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
            with open(path_or_json) as _json_file:
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
            with open(path_or_list) as f:
                return f.read().strip().split("\n")
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
    def maybe_load_specter(path_or_pickle: str | dict | None) -> dict | None:
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

        if loaded is None or isinstance(loaded, dict):
            return loaded

        if isinstance(loaded, tuple) and len(loaded) == 2:
            matrix, keys = loaded
            specter_by_key: dict[Any, Any] = {}
            for i, key in enumerate(keys):
                specter_by_key[key] = matrix[i, :]
            return specter_by_key

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
    ) -> float | None:
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
        signature_1 = self.signatures[signature_id_1]
        signature_2 = self.signatures[signature_id_2]

        def _materialize_constraint_name_parts(signature: Signature) -> tuple[str, str]:
            first = signature.author_info_first_normalized_without_apostrophe
            middle = signature.author_info_middle_normalized_without_apostrophe
            if first is None or middle is None:
                computed_first, computed_middle = split_first_middle_hyphen_aware(
                    signature.author_info_first,
                    signature.author_info_middle,
                )
                if first is None:
                    first = computed_first
                if middle is None:
                    middle = computed_middle
            return first or "", middle or ""

        def _materialize_constraint_last_normalized(signature: Signature) -> str:
            if signature.author_info_last_normalized is not None:
                return signature.author_info_last_normalized
            return normalize_text(signature.author_info_last)

        first_1, middle_1_text = _materialize_constraint_name_parts(signature_1)
        first_2, middle_2_text = _materialize_constraint_name_parts(signature_2)
        middle_1 = middle_1_text.split()

        paper_1 = self.papers[str(signature_1.paper_id)]
        paper_2 = self.papers[str(signature_2.paper_id)]

        orcid_1 = signature_1.author_info_orcid
        orcid_2 = signature_2.author_info_orcid

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
        # just-in-case last name constraint: if last names are different (hyphen/space-insensitive), then disallow
        # TODO(s2and): remove hyphen/space-insensitive shim once canonicalization is unified end-to-end
        elif not _lasts_equivalent_for_constraint(
            _materialize_constraint_last_normalized(signature_1),
            _materialize_constraint_last_normalized(signature_2),
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
            # either a known alias or a prefix of the other
            # if neither, then we'll say it's impossible to be the same person
            # Backward-compatibility: `first_1`/`first_2` can now be multi-token (Sinonym output).
            # Legacy name_tuples were curated over single-token first names. To remain compatible,
            # try multiple forms for alias membership: exact, joined-without-spaces, and first-token only.
            # TODO: revisit once we re-extract name_tuples aligned with Sinonym canonicalization.
            first_1_parts = first_1.split()
            first_2_parts = first_2.split()
            f1_join = "".join(first_1_parts)
            f2_join = "".join(first_2_parts)
            f1_tok = first_1_parts[0] if first_1_parts else first_1
            f2_tok = first_2_parts[0] if first_2_parts else first_2
            known_alias = (
                (first_1, first_2) in self.name_tuples
                or (f1_join, f2_join) in self.name_tuples
                or (f1_tok, f2_tok) in self.name_tuples
            )
            prefix = same_prefix_tokens(first_1, first_2)
            if not prefix and not known_alias:
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
            signature_to_year: dict[str, int] = {}
            for signature_id, signature in self.signatures.items():
                # paper_id should be kept as string, so it can be matched to papers.json
                paper_id = str(signature.paper_id)
                if self.papers[paper_id].year is None:
                    signature_to_year[signature_id] = 0
                else:
                    # mypy: year is Optional[int] on Paper; guarded above, so cast to int here
                    signature_to_year[signature_id] = int(self.papers[paper_id].year)  # type: ignore[arg-type]

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
        train_pairs_df = self.train_pairs.copy()
        train_pairs_df.loc[:, "label"] = train_pairs_df["label"].map(_PAIR_LABEL_MAP)
        if self.val_pairs is not None:
            val_pairs_df = self.val_pairs.copy()
            val_pairs_df.loc[:, "label"] = val_pairs_df["label"].map(_PAIR_LABEL_MAP)
            train_pairs = list(train_pairs_df.to_records(index=False))
            val_pairs = list(val_pairs_df.to_records(index=False))
        else:
            np.random.seed(self.random_seed)
            # split train into train/val
            train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
            msk = np.random.rand(len(train_pairs_df)) < train_prob
            train_pairs = list(train_pairs_df[msk].to_records(index=False))
            val_pairs = list(train_pairs_df[~msk].to_records(index=False))
        test_pairs_df = self.test_pairs.copy()
        test_pairs_df.loc[:, "label"] = test_pairs_df["label"].map(_PAIR_LABEL_MAP)
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
        valid_configuration = (
            (not self.pair_sampling_block and self.pair_sampling_balanced_classes)
            or (self.pair_sampling_block and self.pair_sampling_balanced_classes)
            or (
                self.pair_sampling_block
                and not self.pair_sampling_balanced_homonym_synonym
                and not self.pair_sampling_balanced_classes
            )
        )
        if not valid_configuration:
            raise ValueError(
                f"You chose sample within blocks? {self.pair_sampling_block}, sample balanced pos/neg?\
                 {self.pair_sampling_balanced_classes}, sample balanced homonym/synonym?\
                  {self.pair_sampling_balanced_homonym_synonym}. This is not a valid combination.\
                   Not using blocks and not doing balancing is not supported, and homonym/synonym\
                    balancing without pos/neg balancing is not supported"
            )

        same_name_different_cluster: list[tuple[str, str, int | float]] = []
        same_name_same_cluster: list[tuple[str, str, int | float]] = []
        different_name_same_cluster: list[tuple[str, str, int | float]] = []
        different_name_different_cluster: list[tuple[str, str, int | float]] = []
        possible: list[tuple[str, str, int | float]] = []

        if not self.pair_sampling_block:
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
            if (
                self.pair_sampling_balanced_homonym_synonym
                or self.pair_sampling_balanced_classes
                or not self.pair_sampling_block
            ):
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
            else:
                raise ValueError(
                    "Unsupported pair sampling configuration for non-exhaustive sampling "
                    f"(pair_sampling_block={self.pair_sampling_block}, "
                    f"pair_sampling_balanced_classes={self.pair_sampling_balanced_classes}, "
                    f"pair_sampling_balanced_homonym_synonym={self.pair_sampling_balanced_homonym_synonym})"
                )
            return pairs


# ------------------------ Sinonym integration helpers ------------------------


def _ensure_sinonym_detector():
    """Lazily import and initialize a process-level default detector."""
    global _SINONYM_DETECTOR  # type: ignore
    if _SINONYM_DETECTOR is not None:
        return _SINONYM_DETECTOR
    with _SINONYM_DETECTOR_LOCK:
        if _SINONYM_DETECTOR is None:
            try:
                from sinonym.detector import ChineseNameDetector  # type: ignore
            except Exception as e:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Sinonym is not installed or failed to import. Install 'sinonym' to enable this feature."
                ) from e
            _SINONYM_DETECTOR = ChineseNameDetector()
    return _SINONYM_DETECTOR


def _parse_sinonym_name(name_or_struct: Any) -> tuple[str, str, str]:
    """Extract (first, middle, last) from Sinonym output using ParsedName only.

    Expected input is a structure derived from ParseResult.parsed, either:
      - a ParsedName-like object with attributes: surname_tokens, given_tokens
      - or a dict with keys: 'surname_tokens', 'given_tokens', and optional 'original_compound_surname'

    Returns (first, middle, last), where 'first' is the joined given-name tokens,
    and 'last' uses the original compound surname formatting if provided, otherwise
    joins surname tokens with spaces. 'middle' is empty by design.
    """
    # Handle ParsedName-like object
    if hasattr(name_or_struct, "given_tokens") and hasattr(name_or_struct, "surname_tokens"):
        given_tokens = getattr(name_or_struct, "given_tokens", [])
        surname_tokens = getattr(name_or_struct, "surname_tokens", [])
        original_compound = getattr(name_or_struct, "original_compound_surname", None)
        # Middle can be provided as tokens or as a pre-joined string
        middle_tokens = getattr(name_or_struct, "middle_tokens", None)
        middle_name = getattr(name_or_struct, "middle_name", None)

        first = " ".join([t for t in given_tokens if isinstance(t, str) and t])

        # Prefer explicit middle_name string if present; otherwise join tokens
        middle = ""
        if isinstance(middle_name, str) and middle_name.strip():
            middle = middle_name.strip()
        elif isinstance(middle_tokens, list):
            mt = [t for t in middle_tokens if isinstance(t, str) and t]
            if mt:
                middle = " ".join(mt)

        if isinstance(original_compound, str) and original_compound.strip():
            last = original_compound.strip()
        else:
            last = " ".join([t for t in surname_tokens if isinstance(t, str) and t])
        return first, middle, last

    # Handle dict form
    if isinstance(name_or_struct, dict):
        given_tokens = name_or_struct.get("given_tokens")
        surname_tokens = name_or_struct.get("surname_tokens")
        original_compound = name_or_struct.get("original_compound_surname")
        middle_tokens = name_or_struct.get("middle_tokens")
        middle_name = name_or_struct.get("middle_name")
        if isinstance(given_tokens, list) and isinstance(surname_tokens, list):
            first = " ".join([t for t in given_tokens if isinstance(t, str) and t])

            # Build middle string if available
            middle = ""
            if isinstance(middle_name, str) and middle_name.strip():
                middle = middle_name.strip()
            elif isinstance(middle_tokens, list):
                mt = [t for t in middle_tokens if isinstance(t, str) and t]
                if mt:
                    middle = " ".join(mt)

            if isinstance(original_compound, str) and original_compound.strip():
                last = original_compound.strip()
            else:
                last = " ".join([t for t in surname_tokens if isinstance(t, str) and t])
            return first, middle, last

    # If we got here, we don't have a parsed structure we recognize
    return "", "", ""


def _normalized_first_last_from_signature(sig: Signature) -> tuple[str, str]:
    """Construct normalized (first, last) similar to preprocess_signatures().

    Uses hyphen-aware first/middle split and normalize_text for last.
    """
    first_raw = sig.author_info_first or ""
    middle_raw = sig.author_info_middle or ""
    first_noapos, _ = split_first_middle_hyphen_aware(first_raw, middle_raw)
    last_norm = normalize_text(sig.author_info_last)
    return first_noapos, last_norm


def compute_sinonym_overwrite_allowlist(
    signatures: dict[str, Signature],
    per_paper_results: dict[str, dict[int, Any]],
    min_ratio: float = 3.0,
) -> dict[str, set[int]]:
    """Compute overwrite allowlist.

    Use multi-author ratio (x >= min_ratio * y) when any multi-author evidence exists; otherwise, use
    single-author rule (flip if a > 0).
    """

    def _canon(s: str) -> str:
        # Lower and drop non-letters to align spaces/hyphens variants
        return re.sub(r"[^a-z]", "", (s or "").lower())

    # Detect multi-author papers via unique author positions present among signatures
    paper_pos_sets: dict[str, set[int]] = defaultdict(set)
    for sig in signatures.values():
        paper_pos_sets[str(sig.paper_id)].add(sig.author_info_position)

    # Counts per normalized name: [a, b, x, y]
    # a,b from single-author papers; x,y from multi-author papers
    name_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    # Aggregate counts
    for sig in signatures.values():
        paper_id_str = str(sig.paper_id)
        by_pos = per_paper_results.get(paper_id_str)
        if not by_pos:
            continue
        norm_struct = by_pos.get(sig.author_info_position)
        if not norm_struct:
            continue
        s_first, _, s_last = _parse_sinonym_name(norm_struct)
        if not (s_first and s_last):
            continue
        o_first, o_last = _normalized_first_last_from_signature(sig)
        name_key = (o_first + " " + o_last).strip()

        flipped = _canon(s_first) == _canon(o_last) and _canon(s_last) == _canon(o_first)
        is_multi = len(paper_pos_sets.get(paper_id_str, set())) > 1
        if is_multi:
            if flipped:
                name_counts[name_key][2] += 1  # x
            else:
                name_counts[name_key][3] += 1  # y
        else:
            if flipped:
                name_counts[name_key][0] += 1  # a
            else:
                name_counts[name_key][1] += 1  # b

    # Decide qualified names
    qualified_names: set[str] = set()
    for name, counts in name_counts.items():
        a, b, x, y = counts
        if (x + y) > 0:
            if x >= min_ratio * y:
                qualified_names.add(name)
        else:
            if a > 0:
                qualified_names.add(name)

    # Build allowlist for all occurrences of qualified names
    allow: dict[str, set[int]] = {}
    for sig in signatures.values():
        o_first, o_last = _normalized_first_last_from_signature(sig)
        name_key = (o_first + " " + o_last).strip()
        if name_key in qualified_names:
            allow.setdefault(str(sig.paper_id), set()).add(sig.author_info_position)

    return allow


def sinonym_preprocess_papers_parallel(papers_dict: dict[str, Paper], n_jobs: int) -> dict[str, dict[int, Any]]:
    """Parallel wrapper for running Sinonym preprocessing across papers.

    Returns a mapping: paper_id -> { position -> structured result }, where each
    structured result is:
      - { 'surname_tokens': [...], 'given_tokens': [...], 'original_compound_surname': Optional[str] }
    """
    output: dict[str, dict[int, Any]] = {}
    if n_jobs > 1:
        with UniversalPool(processes=n_jobs) as p:  # type: ignore
            _max = len(papers_dict)
            with tqdm(total=_max, desc="Sinonym: analyzing author batches") as pbar:
                # Build a lightweight iterable to minimize serialization overhead
                light_iter = (
                    (
                        key,
                        [(a.position, a.author_name) for a in paper.authors if a.author_name is not None],
                    )
                    for key, paper in papers_dict.items()
                )
                for key, value in p.imap(_sinonym_preprocess_paper_light, light_iter, CHUNK_SIZE):
                    output[key] = value
                    pbar.update()
    else:
        # Serial path uses the same lightweight items
        light_iter = (
            (
                key,
                [(a.position, a.author_name) for a in paper.authors if a.author_name is not None],
            )
            for key, paper in papers_dict.items()
        )
        for item in tqdm(light_iter, total=len(papers_dict), desc="Sinonym: analyzing author batches"):
            k, v = _sinonym_preprocess_paper_light(item)
            output[k] = v
    return output


def _sinonym_preprocess_paper_light(item: tuple[str, list[tuple[int, str]]]) -> tuple[str, dict[int, Any]]:
    """Lightweight variant: input is (paper_id, [(position, author_name), ...]).
    Returns a mapping: paper_id -> { position -> structured result }, where each structured result is:
    {
        'surname_tokens': [...],
        'given_tokens': [...],
        'original_compound_surname': Optional[str],
        'middle_tokens': Optional[list[str]]  # may be present if available
    }
    """
    key, pos_names = item

    # Collect positions and names, skipping None defensively
    positions: list[int] = []
    names: list[str] = []
    for pos, name in pos_names:
        if name is not None:
            positions.append(pos)
            names.append(name)

    if not names:
        return key, {}

    detector = _ensure_sinonym_detector()
    results = detector.process_name_batch(names)  # type: ignore[attr-defined]

    pos_to_norm: dict[int, Any] = {}

    # If any author in the batch is non-Chinese (unsuccessful), then for the
    # Chinese authors use parsed_original_order instead of parsed.
    any_non_success = any(not getattr(res, "success", False) for res in (results or []))

    # Keep only successful (Chinese) parses; align safely via zip
    for pos, res in zip(positions, (results or []), strict=False):
        success = getattr(res, "success", False)
        if not success:
            continue

        # Choose which parsed structure to use
        if any_non_success:
            parsed = getattr(res, "parsed_original_order", None)
        else:
            parsed = getattr(res, "parsed", None)
        original_compound = getattr(res, "original_compound_surname", None)

        if parsed is not None and hasattr(parsed, "surname_tokens") and hasattr(parsed, "given_tokens"):
            surname_tokens = getattr(parsed, "surname_tokens", [])
            given_tokens = getattr(parsed, "given_tokens", [])
            middle_tokens = None
            if hasattr(parsed, "middle_tokens"):
                middle_tokens = getattr(parsed, "middle_tokens", None)

            entry = {
                "surname_tokens": surname_tokens,
                "given_tokens": given_tokens,
                "original_compound_surname": original_compound,
            }
            if middle_tokens:
                entry["middle_tokens"] = middle_tokens
            pos_to_norm[pos] = entry

    return key, pos_to_norm


def apply_sinonym_overwrites(
    signatures: dict[str, Signature],
    per_paper_results: dict[str, dict[int, Any]],
    *,
    overwrite_blocks: bool = False,
    allow_overwrite_pos: dict[str, set[int]] | None = None,
) -> int:
    """Overwrite signature name parts with Sinonym-normalized names where applicable.

    Args:
        signatures: signature_id -> Signature
        per_paper_results: paper_id(str) -> { position -> parsed_struct }
        overwrite_blocks: if True, also overwrite author_info_block with the new
            block derived from normalized first/last. Use only in inference to
            avoid changing dataset splits.

    Returns:
        Number of signatures updated.
    """
    overwrite_count = 0
    for sig_id, sig in list(signatures.items()):
        paper_id_str = str(sig.paper_id)
        by_pos = per_paper_results.get(paper_id_str)
        if not by_pos:
            continue
        norm_struct = by_pos.get(sig.author_info_position)
        if not norm_struct:
            continue
        first, middle, last = _parse_sinonym_name(norm_struct)
        if first or last:
            # Gate overwrites if allowlist provided
            if allow_overwrite_pos is not None:
                allowed = allow_overwrite_pos.get(paper_id_str, set())
                if sig.author_info_position not in allowed:
                    continue
            # Build the new block only when BOTH first and last are present.
            # Otherwise, do not change the existing block value.
            new_block = None
            try:
                if first and last:
                    # Match standard block computation: first-initial + last, then normalize.
                    # TODO(s2and): When blocks are recomputed everywhere from Sinonym-aware
                    # canonical forms, remove compatibility handling in this overwrite path.
                    new_block = normalize_text(f"{first[:1]} {last}")
            except Exception:
                # Log any unexpected formatting issues; keep prior block on error
                logger.exception(
                    "Error computing new block for signature_id=%s (paper_id=%s, position=%s)",
                    sig_id,
                    sig.paper_id,
                    sig.author_info_position,
                )
                new_block = None

            # Always update first/middle/last; conditionally update block in inference
            new_sig = sig._replace(
                author_info_first=first,
                author_info_middle=middle,
                author_info_last=last,
            )
            if overwrite_blocks and new_block is not None:
                # Note: changing blocks will affect clustering; only do this in inference
                new_sig = new_sig._replace(author_info_block=new_block)
            signatures[sig_id] = new_sig
            overwrite_count += 1
    return overwrite_count


def apply_sinonym_overwrites_to_papers(
    papers: dict[str, Paper],
    per_paper_results: dict[str, dict[int, Any]],
    allow_overwrite_pos: dict[str, set[int]] | None = None,
) -> int:
    """Apply Sinonym-normalized names to Paper.authors for co-author features.

    For each paper and author position recognized by Sinonym, replace the
    Author.author_name with a reconstructed full name built from Sinonym
    (first, middle, last). Per-paper preprocessing will later normalize
    casing/spacing consistently.

    Returns number of author entries updated.
    """
    updates = 0
    for key, paper in papers.items():
        by_pos = per_paper_results.get(str(key))
        if not by_pos:
            continue
        new_authors = []
        for a in paper.authors:
            repl = by_pos.get(a.position) if isinstance(by_pos, dict) else None
            if repl:
                # Gate overwrites by position
                if allow_overwrite_pos is not None:
                    allowed = allow_overwrite_pos.get(str(key), set())
                    if a.position not in allowed:
                        new_authors.append(a)
                        continue
                first, middle, last = _parse_sinonym_name(repl)
                if first or middle or last:
                    parts = [p for p in [first, middle, last] if isinstance(p, str) and p]
                    new_name = " ".join(parts).strip()
                    if new_name and new_name != a.author_name:
                        new_authors.append(Author(author_name=new_name, position=a.position))
                        updates += 1
                        continue
            new_authors.append(a)
        if new_authors and new_authors != list(paper.authors):
            papers[key] = paper._replace(authors=new_authors)
    return updates


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


def preprocess_paper_2(item: tuple[str, Paper, list[MiniPaper]]) -> tuple[str, Paper]:
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

    authors: list[str] = list(
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


def preprocess_papers_parallel(
    papers_dict: dict,
    n_jobs: int,
    preprocess: bool,
    *,
    compute_reference_features: bool = False,
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
        # Use UniversalPool to replicate the original p.imap() streaming behavior
        with UniversalPool(processes=n_jobs) as p:  # type: ignore
            _max = len(papers_dict)
            with tqdm(total=_max, desc="Preprocessing papers 1/2") as pbar:
                func = partial(preprocess_paper_1, preprocess=preprocess)
                for key, value in p.imap(func, papers_dict.items(), CHUNK_SIZE):
                    output[key] = value
                    pbar.update()
    else:
        for item in tqdm(papers_dict.items(), total=len(papers_dict), desc="Preprocessing papers 1/2"):
            k, v = preprocess_paper_1(item, preprocess=preprocess)
            output[k] = v

    # -------- second stage (reference features) -------
    if preprocess and compute_reference_features:
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
                    for p in [output.get(str(rid)) for rid in (value.references or [])]
                    if p is not None
                ],
            )
            for key, value in output.items()
        ]
        for item in tqdm(input_2, total=len(input_2), desc="Preprocessing papers 2/2"):
            k, v = preprocess_paper_2(item)
            output[k] = v
    elif preprocess and not compute_reference_features:
        # Ensure reference_details exists as empty counters to keep downstream code safe
        empty_tuple: tuple[Counter, Counter, Counter, Counter] = (
            Counter(),
            Counter(),
            Counter(),
            Counter(),
        )
        for k, v in output.items():
            if v.reference_details is None:
                output[k] = v._replace(reference_details=empty_tuple)

    return output

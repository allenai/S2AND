import json
import logging
import math
import os
import pickle
import platform
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict, cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import (
    _PACKAGE_DATA_DIR,
    CLUSTER_SEEDS_LOOKUP,
    LARGE_DISTANCE,
    NAME_COUNTS_INDEX_PATH,
    NUMPY_NAN,
)
from s2and.mp import UniversalPool
from s2and.name_counts_index import NameCountsIndex
from s2and.name_tuple_artifact import load_name_tuple_artifact, load_packaged_name_tuple_artifact
from s2and.runtime import (
    RuntimeContext,
    build_runtime_context,
    stage_uses_rust,
)
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
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"train/val/test ratios must add to 1; got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )


def _split_train_val_test(
    items: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    stratify: Any = None,
) -> tuple[list[str], list[str], list[str]]:
    """Split items while allowing any partition to have a zero ratio.

    Args:
        items: Identifiers to split.
        train_ratio: Fraction assigned to training.
        val_ratio: Fraction assigned to validation.
        test_ratio: Fraction assigned to testing.
        random_seed: Random seed passed to scikit-learn.
        stratify: Optional labels aligned with ``items`` for stratified splits.

    Returns:
        Training, validation, and test identifiers.
    """

    _validate_split_ratios(train_ratio, val_ratio, test_ratio)
    ratio_total = math.fsum((train_ratio, val_ratio, test_ratio))
    train_ratio, val_ratio, test_ratio = (
        train_ratio / ratio_total,
        val_ratio / ratio_total,
        test_ratio / ratio_total,
    )

    heldout_ratio = 1.0 - train_ratio
    if train_ratio == 0.0:
        train_items: list[str] = []
        val_test_items = list(items)
        val_test_stratify = stratify
    elif heldout_ratio == 0.0:
        return list(items), [], []
    elif stratify is None:
        train_items, val_test_items = train_test_split(
            items,
            test_size=heldout_ratio,
            random_state=random_seed,
        )
        val_test_stratify = None
    else:
        train_items, val_test_items, _, val_test_stratify = train_test_split(
            items,
            stratify,
            test_size=heldout_ratio,
            stratify=stratify,
            random_state=random_seed,
        )

    if val_ratio == 0.0:
        return train_items, [], val_test_items
    if test_ratio == 0.0:
        return train_items, val_test_items, []

    val_items, test_items = train_test_split(
        val_test_items,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=val_test_stratify,
        random_state=random_seed,
    )
    return train_items, val_items, test_items


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


def _upper_triangle_pair_indices(block_size: int, pair_rank: int) -> tuple[int, int]:
    """Map a lexicographic upper-triangle rank to its signature indices.

    The rank order matches the nested loops used by ``ANDData.pair_sampling``:
    ``(0, 1), (0, 2), ..., (1, 2), ...``.

    Args:
        block_size: Number of signatures in the block.
        pair_rank: Zero-based rank within the block's upper triangle.

    Returns:
        The two signature indices for ``pair_rank``.
    """

    low = 0
    high = block_size - 2
    while low < high:
        candidate = (low + high + 1) // 2
        candidate_start = candidate * (2 * block_size - candidate - 1) // 2
        if candidate_start <= pair_rank:
            low = candidate
        else:
            high = candidate - 1

    first_index = low
    row_start = first_index * (2 * block_size - first_index - 1) // 2
    second_index = first_index + 1 + pair_rank - row_start
    return first_index, second_index


def _sample_within_block_random_pairs(
    blocks: Mapping[str, list[str]],
    signature_to_cluster_id: Mapping[str, Any] | None,
    sample_size: int,
    random_seed: int,
) -> list[tuple[str, str, int | float]]:
    """Sample within-block pairs without materializing every candidate.

    Sampling integer ranks from ``range(total_pairs)`` produces the same
    selections and output order as sampling the legacy, exhaustively built
    candidate list because ``random.sample`` depends only on population length
    and rank. Memory is proportional to the requested sample size plus the
    number of nontrivial blocks.

    Args:
        blocks: Ordered mapping of block IDs to ordered signature IDs.
        signature_to_cluster_id: Optional signature-to-cluster labels.
        sample_size: Maximum number of pairs to return.
        random_seed: Seed passed to the deterministic sampler.

    Returns:
        Sampled signature pairs in legacy ``random.sample`` order.
    """

    started = time.perf_counter()
    pair_blocks: list[tuple[list[str], int]] = []
    cumulative_pair_counts: list[int] = []
    total_pairs = 0
    max_block_size = 0
    for signatures in blocks.values():
        block_size = len(signatures)
        if block_size < 2:
            continue
        max_block_size = max(max_block_size, block_size)

        if signature_to_cluster_id is not None:
            # Preserve the legacy failure contract: missing cluster labels fail
            # even when their pair would not have been selected.
            for signature_id in signatures:
                signature_to_cluster_id[signature_id]

        block_pair_count = block_size * (block_size - 1) // 2
        total_pairs += block_pair_count
        pair_blocks.append((signatures, total_pairs - block_pair_count))
        cumulative_pair_counts.append(total_pairs)

    resolved_sample_size = min(total_pairs, sample_size)
    sampled_pair_ranks = random_sampling(range(total_pairs), resolved_sample_size, random_seed)
    sampled_pairs: list[tuple[str, str, int | float]] = []
    for pair_rank in sampled_pair_ranks:
        block_index = bisect_right(cumulative_pair_counts, pair_rank)
        signatures, block_start = pair_blocks[block_index]
        local_pair_rank = pair_rank - block_start
        first_index, second_index = _upper_triangle_pair_indices(len(signatures), local_pair_rank)
        first_signature = signatures[first_index]
        second_signature = signatures[second_index]
        if signature_to_cluster_id is None:
            label: int | float = NUMPY_NAN
        else:
            label = int(signature_to_cluster_id[first_signature] == signature_to_cluster_id[second_signature])
        sampled_pairs.append((first_signature, second_signature, label))
    logger.info(
        "Telemetry stage: stage=within_block_random_pair_sampling seconds=%.3f "
        "candidate_pairs=%d requested_pairs=%d returned_pairs=%d nontrivial_blocks=%d max_block_size=%d",
        time.perf_counter() - started,
        total_pairs,
        sample_size,
        len(sampled_pairs),
        len(pair_blocks),
        max_block_size,
    )
    return sampled_pairs


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
    author_info_estimated_gender: str | None
    author_info_estimated_ethnicity: str | None
    paper_id: str | int
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
    paper_id: str | int


class _SignaturePreprocessRow(TypedDict):
    """Typed intermediate values for one signature preprocessing batch row."""

    signature_id: str
    signature: Signature
    first_without_apostrophe: str | None
    middle_without_apostrophe: str | None
    last_normalized: str | None
    suffix_normalized: str | None
    coauthor_set: set[str] | None
    coauthor_blocks: set[str] | None
    affiliations: list[str]
    full_name: str | None
    count_keys: tuple[str | None, str | None, str | None, str | None] | None
    normalized_orcid: str | None
    coauthor_text: str
    affiliation_text: str


class ANDData:
    """
    The main class for holding our representation of an author disambiguation dataset

    Blocking uses ``author_info.block`` exclusively. The legacy
    ``block_type`` selector, ``author_info.given_block``,
    ``get_original_blocks()``, and ``get_s2_blocks()`` are removed; callers
    that need a historical partition must retain it outside ``ANDData``.

    Input:
        signatures: path to the signatures json file (or the json object)
        papers: path to the papers information json file (or the json object)
        name: human-readable dataset name used in logs and metrics
        mode: 'train' or 'inference'; if 'inference', everything related to dataset
            splitting will be ignored
        clusters: path to the clusters json file (or the json object)
        specter_embeddings: path to the specter embeddings pickle (or the dictionary object)
        cluster_seeds: path to the cluster seed json file (or the json object)
            Require pairs form transitive connected groups. Explicit disallow
            pairs remain hard negatives, including within a require group.
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
        name_counts_index: Manifest-backed binary name-count index. Defaults to
            the canonical configured ``NAME_COUNTS_INDEX_PATH``; pass ``None``
            explicitly to leave Python-side name-count features unmaterialized.
            A path is verified and opened once per immutable manifest
            generation; an already-open ``NameCountsIndex`` handle can be
            shared explicitly.
        n_jobs: number of cpus to use
        preprocess: whether to preprocess the data (normalization, etc)
        name_tuples: Canonical first-name aliases. ``None`` selects the
            packaged canonical artifact. Pass an explicit empty set or
            frozenset to disable aliases; pair order is ignored.
        use_orcid_id: Whether ingestion retains ORCID IDs. ``False`` strips
            them before preprocessing, disabling both ORCID constraints and
            ORCID-aware subblocking. Arrow-backed training also threads this
            policy into its native featurizer.
    """

    @classmethod
    def _from_arrow_training(
        cls,
        signatures: dict[str, Signature],
        name: str,
        *,
        arrow_dataset: ArrowDataset,
        **kwargs: Any,
    ) -> "ANDData":
        """Construct a Rust-training dataset from one open Arrow dataset.

        Args:
            signatures: Final lightweight Python signature metadata.
            name: Human-readable dataset name used in logs and metrics.
            arrow_dataset: Open Arrow dataset retained by the returned object.
            **kwargs: Remaining train-mode ``ANDData`` construction arguments.

        Returns:
            A train-mode dataset whose Arrow state is available throughout
            initialization.
        """

        return cls(
            signatures=signatures,
            papers={},
            name=name,
            mode="train",
            specter_embeddings=None,
            name_counts_index=None,
            preprocess=True,
            _arrow_dataset=arrow_dataset,
            **kwargs,
        )

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
        name_counts_index: str | os.PathLike[str] | NameCountsIndex | None = NAME_COUNTS_INDEX_PATH,
        n_jobs: int = 1,
        preprocess: bool = True,
        name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None = None,
        use_orcid_id: bool = True,
        compute_block_fn: Callable[[str], str] = compute_block,
        _arrow_dataset: ArrowDataset | None = None,
    ):
        init_start = time.perf_counter()
        if _arrow_dataset is not None:
            if mode != "train" or not preprocess:
                raise ValueError("Arrow training requires mode='train' and preprocess=True")
            if specter_embeddings is not None or name_counts_index is not None:
                raise ValueError("Arrow training reads SPECTER and name counts from its ArrowDataset")
        self.runtime_context = build_runtime_context(
            "dataset_build",
            backend="rust" if _arrow_dataset is not None else "python",
        )
        self.original_signatures_path = signatures if isinstance(signatures, str) else None
        self.original_papers_path = papers if isinstance(papers, str) else None
        self.signatures_path = self.original_signatures_path
        self.papers_path = self.original_papers_path
        self._s2and_python_pair_ngrams_ready: bool = False
        self.clusters_path = clusters if isinstance(clusters, str) else None
        self.cluster_seeds_path = cluster_seeds if isinstance(cluster_seeds, str) else None
        self.specter_embeddings_path = specter_embeddings if isinstance(specter_embeddings, str) else None
        self.arrow_dataset = _arrow_dataset
        self._arrow_paper_ids: set[str] | None = None
        self.compute_block_fn = compute_block_fn
        self.use_orcid_id = bool(use_orcid_id)
        pair_sampling_mode = _validate_pair_sampling_mode(pair_sampling_mode)

        if mode == "train":
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
        if self.arrow_dataset is not None:
            # Arrow ingestion already applied use_orcid_id before construction.
            # Preserve those objects; native training owns preprocessing.
            self.signatures = cast(dict[str, Signature], signatures)
        else:
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
                    # Ingest-time stripping is stronger than scoring-time
                    # `Clusterer.suppress_orcid`: it also affects subblocking.
                    author_info_orcid=(
                        (signature["author_info"].get("source_ids") or [None])[0]
                        if self.use_orcid_id and signature["author_info"].get("source_id_source") == "ORCID"
                        else None
                    ),
                    author_info_name_counts=None,
                    author_info_position=signature["author_info"]["position"],
                    author_info_block=signature["author_info"]["block"],
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

        papers_stage_start = time.perf_counter()
        logger.info("loading papers (subset referenced by signatures)")
        if self.arrow_dataset is not None:
            needed_paper_ids = {str(signature.paper_id) for signature in self.signatures.values()}
            self._arrow_paper_ids = needed_paper_ids
            retained_paper_count = source_paper_count = len(needed_paper_ids)
        else:
            needed_paper_ids = {str(signature.paper_id) for signature in self.signatures.values()}
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
            retained_paper_count = len(self.papers)
            source_paper_count = len(raw_papers)
        logger.info(f"loaded papers subset: {retained_paper_count}/{source_paper_count} relevant")
        logger.debug(
            "Telemetry stage: stage=anddata_ingest_papers seconds=%.3f retained_papers=%d source_papers=%d",
            time.perf_counter() - papers_stage_start,
            retained_paper_count,
            source_paper_count,
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
        self.cluster_seeds_require: dict[str, int | str] = {}
        self.max_seed_cluster_id = None
        if cluster_seeds_dict is not None:
            parents: dict[str, str] = {}
            sizes: dict[str, int] = {}

            def find(signature_id: str) -> str:
                """Find a require component root with path compression."""
                while parents[signature_id] != signature_id:
                    parents[signature_id] = parents[parents[signature_id]]
                    signature_id = parents[signature_id]
                return signature_id

            for signature_id_a, values in cluster_seeds_dict.items():
                for signature_id_b, constraint_string in values.items():
                    if constraint_string == "disallow":
                        self.cluster_seeds_disallow.add((signature_id_a, signature_id_b))
                    elif constraint_string == "require":
                        for signature_id in (signature_id_a, signature_id_b):
                            if signature_id not in parents:
                                parents[signature_id] = signature_id
                                sizes[signature_id] = 1
                        root_a, root_b = find(signature_id_a), find(signature_id_b)
                        if root_a != root_b:
                            if sizes[root_a] < sizes[root_b]:
                                root_a, root_b = root_b, root_a
                            parents[root_b] = root_a
                            sizes[root_a] += sizes[root_b]

            # Assign compact IDs in first-seen order after all bridges are merged.
            component_ids: dict[str, int] = {}
            for signature_id in parents:
                root = find(signature_id)
                cluster_id = component_ids.setdefault(root, len(component_ids))
                self.cluster_seeds_require[signature_id] = cluster_id
            self.max_seed_cluster_id = len(component_ids)
        logger.info("loaded cluster seeds")
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
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.name_counts_index: NameCountsIndex | None = None
        if name_counts_index is not None:
            logger.info("opening name-count index (manifest-cached)")
            self.name_counts_index = (
                name_counts_index
                if isinstance(name_counts_index, NameCountsIndex)
                else NameCountsIndex.open(name_counts_index)
            )
            logger.info("opened name-count index")
        self.name_counts_loaded = self.name_counts_index is not None

        self.n_jobs = resolve_n_jobs(n_jobs)
        self.signature_to_block = self.get_signatures_to_block()
        if self.arrow_dataset is None:
            papers_from_signatures = {str(signature.paper_id) for signature in self.signatures.values()}
            for paper_id, paper in self.papers.items():
                self.papers[paper_id] = paper._replace(in_signatures=str(paper_id) in papers_from_signatures)
        self.preprocess = preprocess

        resolved_name_tuples: set[tuple[str, str]]
        if name_tuples is None:
            resolved_name_tuples = set(load_packaged_name_tuple_artifact().pairs)
        elif isinstance(name_tuples, set | frozenset):
            resolved_name_tuples = {canonical_name_tuple_pair(first_a, first_b) for first_a, first_b in name_tuples}
        else:
            raise TypeError("name_tuples must be None or a set/frozenset of (first_a, first_b) tuples")
        self.name_tuples = frozenset(resolved_name_tuples) if self.arrow_dataset is not None else resolved_name_tuples

        preprocess_papers_stage_start = time.perf_counter()
        if self.arrow_dataset is not None:
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
            retained_paper_count,
        )

        preprocess_signatures_stage_start = time.perf_counter()
        if self.arrow_dataset is not None:
            logger.info("Rust deferred signature preprocessing active: skipping Python signature preprocessing")
        else:
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

    @cached_property
    def papers(self) -> dict[str, Paper]:
        """Materialize Python paper objects when an Arrow-backed caller needs them."""

        from s2and.arrow_training import load_papers_from_arrow

        assert self.arrow_dataset is not None
        assert self._arrow_paper_ids is not None
        with self.arrow_dataset.use() as lease:
            with lease.open_file("papers") as papers, lease.open_file("paper_authors") as paper_authors:
                return load_papers_from_arrow(
                    papers,
                    paper_authors,
                    needed_paper_ids=self._arrow_paper_ids,
                )

    @property
    def name_counts_manifest_sha256(self) -> str | None:
        """Return the name-count identity from the retained resource."""

        if self.name_counts_index is not None:
            return self.name_counts_index.manifest_sha256
        if self.arrow_dataset is not None and self.arrow_dataset.name_counts_index is not None:
            return self.arrow_dataset.name_counts_index.manifest_sha256
        return None

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
        defer_signature_ngrams_to_rust = self.arrow_dataset is not None
        defer_signature_fields_to_rust = self.arrow_dataset is not None
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
                batch_rows: list[_SignaturePreprocessRow] = []
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
                        key_rows: list[tuple[str | None, str | None, str | None, str | None]] = []
                        for row in batch_rows:
                            keys = row["count_keys"]
                            if keys is None:  # pragma: no cover - construction invariant
                                raise RuntimeError("name-count index batch is missing canonical keys")
                            key_rows.append(keys)
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
                    replace_kwargs: dict[str, Any] = {
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

    def get_blocks(self) -> dict[str, list[str]]:
        """Return signatures grouped by their canonical Semantic Scholar block.

        ``author_info.block`` is the sole grouping authority. Legacy
        ``author_info.given_block`` values are intentionally ignored during
        ingestion.

        Returns
        -------
        Dict: mapping from block id to list of signatures in the block
        """
        blocks: dict[str, list[str]] = defaultdict(list)
        for signature_id, signature in self.signatures.items():
            blocks[signature.author_info_block].append(signature_id)
        return dict(blocks)

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
        return self._get_constraint(
            signature_id_1,
            signature_id_2,
            cluster_seeds_require=self.cluster_seeds_require,
            cluster_seeds_disallow=self.cluster_seeds_disallow,
            low_value=low_value,
            high_value=high_value,
            dont_merge_cluster_seeds=dont_merge_cluster_seeds,
            incremental_dont_use_cluster_seeds=incremental_dont_use_cluster_seeds,
            suppress_orcid=suppress_orcid,
        )

    def _get_constraint(
        self,
        signature_id_1: str,
        signature_id_2: str,
        *,
        cluster_seeds_require: Mapping[str, int | str],
        cluster_seeds_disallow: set[tuple[str, str]],
        low_value: float | int = 0,
        high_value: float | int = LARGE_DISTANCE,
        dont_merge_cluster_seeds: bool = True,
        incremental_dont_use_cluster_seeds: bool = False,
        suppress_orcid: bool = False,
    ) -> float | None:
        """Apply hard constraints using explicitly supplied seed state."""
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
        if (signature_id_1, signature_id_2) in cluster_seeds_disallow or (
            signature_id_2,
            signature_id_1,
        ) in cluster_seeds_disallow:
            return CLUSTER_SEEDS_LOOKUP["disallow"]
        elif (cluster_seeds_require.get(signature_id_1, -1) == cluster_seeds_require.get(signature_id_2, -2)) and (
            not incremental_dont_use_cluster_seeds
        ):
            return CLUSTER_SEEDS_LOOKUP["require"]
        elif (
            dont_merge_cluster_seeds
            and (not incremental_dont_use_cluster_seeds)
            and (signature_id_1 in cluster_seeds_require and signature_id_2 in cluster_seeds_require)
            and (cluster_seeds_require[signature_id_1] != cluster_seeds_require[signature_id_2])
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

        train_blocks, val_blocks, test_blocks = _split_train_val_test(
            x,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
            self.random_seed,
            stratify=y_group,
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
            train_signatures, val_signatures, test_signatures = _split_train_val_test(
                signature_keys,
                self.train_ratio,
                self.val_ratio,
                self.test_ratio,
                self.random_seed,
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
                year = self.papers[paper_id].year
                if year is None:
                    signature_to_year[signature_id] = 0
                else:
                    signature_to_year[signature_id] = int(year)

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
            train_blocks = set(self.train_blocks)
            val_blocks = set(self.val_blocks)
            test_blocks = set(self.test_blocks)
            for block_id, signature in blocks.items():
                if block_id in train_blocks:
                    train_block_dict[block_id] = signature
                elif block_id in val_blocks:
                    val_block_dict[block_id] = signature
                elif block_id in test_blocks:
                    test_block_dict[block_id] = signature
            del train_blocks, val_blocks, test_blocks

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

    def _fixed_train_val_pairs(
        self,
        split_probabilities: np.ndarray | None,
    ) -> tuple[list[tuple[str, str, int | float]], list[tuple[str, str, int | float]]]:
        """Map fixed labels and apply an optional train/validation split."""

        assert self.train_pairs is not None
        train_pairs_df = _map_fixed_pair_labels(self.train_pairs, "train")
        if self.val_pairs is not None:
            val_pairs_df = _map_fixed_pair_labels(self.val_pairs, "val")
            return list(train_pairs_df.to_records(index=False)), list(val_pairs_df.to_records(index=False))

        assert split_probabilities is not None
        train_prob = self.train_ratio / (self.train_ratio + self.val_ratio)
        train_mask = split_probabilities < train_prob
        return (
            list(train_pairs_df[train_mask].to_records(index=False)),
            list(train_pairs_df[~train_mask].to_records(index=False)),
        )

    def fixed_train_val_pairs(
        self,
    ) -> tuple[list[tuple[str, str, int | float]], list[tuple[str, str, int | float]]]:
        """Resolve fixed train/validation pairs without accessing test pairs.

        Returns:
            Train and validation pairs with binary integer labels.
        """

        assert self.train_pairs is not None, "You need to pass in train pairs to use this function"
        split_probabilities = (
            np.random.RandomState(self.random_seed).rand(len(self.train_pairs)) if self.val_pairs is None else None
        )
        return self._fixed_train_val_pairs(split_probabilities)

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
        assert self.train_pairs is not None and self.test_pairs is not None, (
            "You need to pass in train and test pairs to use this function"
        )
        split_probabilities = None
        if self.val_pairs is None:
            np.random.seed(self.random_seed)
            split_probabilities = np.random.rand(len(self.train_pairs))
        train_pairs, val_pairs = self._fixed_train_val_pairs(split_probabilities)
        test_pairs_df = _map_fixed_pair_labels(self.test_pairs, "test")
        test_pairs = list(test_pairs_df.to_records(index=False))

        identities = {
            split_name: {tuple(sorted((str(pair[0]), str(pair[1])))) for pair in pairs}
            for split_name, pairs in (("train", train_pairs), ("val", val_pairs), ("test", test_pairs))
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = identities[left] & identities[right]
            if overlap:
                raise ValueError(
                    f"Fixed pair splits {left!r} and {right!r} overlap by unordered signature pair: "
                    f"count={len(overlap)}, sample={sorted(overlap)[:5]}"
                )

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
        Samples pairs according to the configured strategy.

        Random within-block sampling maps sampled integer ranks directly to
        signature pairs. Exhaustive output and balanced strategies still
        enumerate their candidate pairs.

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

        if pair_sampling_mode == "within_block_random" and not all_pairs:
            return _sample_within_block_random_pairs(
                blocks,
                self.signature_to_cluster_id,
                sample_size,
                self.random_seed,
            )

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


def _resolve_signature_splits(
    dataset: ANDData,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """Resolve split identities without adding prediction context.

    Explicit block splits take precedence over explicit signature splits.
    Otherwise use the dataset's configured random or chronological split,
    preserving its ordering and random seed.

    Args:
        dataset: Dataset containing the split configuration.

    Returns:
        Train, validation, and test signatures grouped by block.
    """
    if dataset.train_blocks is not None:
        return dataset.split_cluster_signatures_fixed()
    if dataset.train_signatures is not None:
        return dataset.split_data_signatures_fixed()
    return dataset.split_cluster_signatures()


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

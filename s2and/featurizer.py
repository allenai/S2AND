import functools
import logging
import platform
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from tqdm import tqdm

from s2and import feature_port, memory_budget
from s2and.consts import (
    DEFAULT_CHUNK_SIZE,
    LARGE_INTEGER,
    NUMPY_NAN,
)
from s2and.data import ANDData
from s2and.mp import UniversalPool
from s2and.runtime import RuntimeContext, build_runtime_context, dataset_stage_uses_rust, stage_uses_rust
from s2and.text import (
    TEXT_FUNCTIONS,
    cosine_sim,
    counter_jaccard,
    diff,
    email_prefix_suffix,
    equal,
    equal_middle,
    jaccard,
    name_counts,
    name_text_features,
)

logger = logging.getLogger("s2and")

TupleOfArrays = tuple[np.ndarray, np.ndarray, np.ndarray | None]
_PreprocessedValueT = TypeVar("_PreprocessedValueT")

# This order defines persisted model columns and is shared by training,
# evaluation, and linker aggregation.
DEFAULT_FEATURE_GROUPS: tuple[str, ...] = (
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "misc_features",
    "name_counts",
    "embedding_similarity",
    "journal_similarity",
    "advanced_name_similarity",
)
NAME_DEPENDENT_FEATURE_GROUPS: frozenset[str] = frozenset(
    {"name_similarity", "advanced_name_similarity", "name_counts"}
)
DEFAULT_NAMELESS_FEATURE_GROUPS: tuple[str, ...] = tuple(
    feature_group for feature_group in DEFAULT_FEATURE_GROUPS if feature_group not in NAME_DEPENDENT_FEATURE_GROUPS
)

_FEATURIZATION_WORKER_STATE = threading.local()


def _initialize_featurization_worker(dataset: ANDData) -> None:
    """Bind one dataset to the current featurization worker."""

    _FEATURIZATION_WORKER_STATE.dataset = dataset


def _use_rust_featurizer(
    runtime_context: RuntimeContext | None = None,
    dataset: ANDData | None = None,
) -> bool:
    """Return whether Rust pair featurization applies.

    With a dataset, this is the dataset-shape decision: Rust featurizers are
    built exclusively from Arrow artifacts, so datasets without
    ``arrow_dataset`` use the Python featurizer (or raise when
    the Rust backend was requested explicitly). Without a dataset, this is the
    backend-level decision only.
    """

    if runtime_context is None:
        runtime_context = (
            dataset.runtime_context
            if dataset is not None
            else build_runtime_context(
                "pair_featurization",
                backend="python",
            )
        )
    if dataset is None:
        return stage_uses_rust(runtime_context)
    return dataset_stage_uses_rust(runtime_context, dataset)


def _has_missing_signature_ngrams_for_pairs(
    dataset: ANDData,
    signature_pairs: list[tuple[str, str, int | float]],
) -> tuple[bool, int]:
    signatures = getattr(dataset, "signatures", {})
    if not signatures:
        return False, 0

    inspected = 0
    inspected_signature_ids = set()
    for sig_id_1, sig_id_2, _ in signature_pairs:
        for signature_id in (sig_id_1, sig_id_2):
            if signature_id in inspected_signature_ids:
                continue
            inspected_signature_ids.add(signature_id)
            signature = signatures.get(signature_id)
            if signature is None:
                continue
            inspected += 1
            if signature.author_info_affiliations_n_grams is None or signature.author_info_coauthor_n_grams is None:
                return True, inspected
    return False, inspected


def _ensure_python_pair_signature_ngrams(
    dataset: ANDData,
    signature_pairs: list[tuple[str, str, int | float]],
    runtime_context: RuntimeContext,
) -> None:
    if _use_rust_featurizer(runtime_context, dataset):
        return
    if getattr(dataset, "arrow_dataset", None) is not None and signature_pairs:
        raise RuntimeError(
            "Python featurization cannot run on a dataset built with normalized signature fields deferred to Rust. "
            "Rebuild the dataset with S2AND_BACKEND=python/backend='python', or keep Rust featurization active with "
            "the open Arrow dataset."
        )
    if getattr(dataset, "_s2and_python_pair_ngrams_ready", False):
        return

    materialize_fn = getattr(dataset, "materialize_signature_ngrams_python", None)
    if materialize_fn is None:
        return

    has_missing_ngrams, inspected_signature_count = _has_missing_signature_ngrams_for_pairs(dataset, signature_pairs)
    if not has_missing_ngrams:
        if inspected_signature_count == len(getattr(dataset, "signatures", {})):
            dataset._s2and_python_pair_ngrams_ready = True
        return

    materialize_start = time.perf_counter()
    materialize_fn()
    dataset._s2and_python_pair_ngrams_ready = True
    logger.info(
        "Telemetry stage: stage=python_pair_signature_ngrams_materialize seconds=%.3f "
        "inspected_signatures=%d total_signatures=%d backend=%s run_id=%s",
        time.perf_counter() - materialize_start,
        inspected_signature_count,
        len(getattr(dataset, "signatures", {})),
        runtime_context.backend,
        runtime_context.run_id,
    )


def _log_featurization_backend_decision(
    runtime_context: RuntimeContext,
    pieces_of_work_count: int,
    n_jobs: int,
    use_rust_featurizer: bool,
    rust_module_available: bool,
) -> None:
    if pieces_of_work_count <= 0:
        logger.info("Featurization backend decision: skipped compute (all pairs were pre-labeled)")
        return

    if use_rust_featurizer and rust_module_available:
        backend = "rust_batch"
    else:
        backend = "python_parallel" if n_jobs > 1 else "python_serial"

    logger.info(
        "Featurization backend decision: backend=%s pieces=%d n_jobs=%d "
        "use_rust_featurizer=%s rust_module_available=%s "
        "runtime_backend=%s run_id=%s",
        backend,
        pieces_of_work_count,
        n_jobs,
        use_rust_featurizer,
        rust_module_available,
        runtime_context.backend,
        runtime_context.run_id,
    )

    notes = []
    if not use_rust_featurizer:
        notes.append("pair_featurization stage set to Python by runtime context")
    if use_rust_featurizer and not rust_module_available:
        notes.append("s2and_rust extension unavailable")
    if notes:
        logger.info("Featurization backend notes: %s", "; ".join(notes))


def _contiguous_index_slice(indices: list[int]) -> slice | None:
    if not indices:
        return None
    start = int(indices[0])
    for offset, index in enumerate(indices):
        if int(index) != start + offset:
            return None
    return slice(start, start + len(indices))


@dataclass(frozen=True)
class ScatterContext:
    features: np.ndarray
    nameless_features: np.ndarray | None
    coauthor_similarity_values: np.ndarray | None
    identity_selected_indices: bool
    indices_to_use: list[int]
    nameless_indices_to_use: list[int]
    selected_positions: list[int]
    nameless_positions: list[int]
    coauthor_similarity_index: int | None
    coauthor_position: int | None


@dataclass(frozen=True)
class RustBatchExecutionResult:
    rust_batch_plan: memory_budget.RustBatchChunkPlan
    rust_batch_total_ram_for_stage: int | None
    rust_batch_rss_before_bytes: int
    rust_batch_rss_peak_bytes: int
    rust_batch_rss_source: str
    rust_batch_adaptive_halvings: int


def _scatter_feature_row_from_source(
    *,
    feature_output: np.ndarray,
    output_index: int,
    scatter_context: ScatterContext,
    rust_chunk_is_full: bool,
) -> None:
    features = scatter_context.features
    nameless_features = scatter_context.nameless_features
    coauthor_similarity_values = scatter_context.coauthor_similarity_values
    if rust_chunk_is_full:
        if scatter_context.identity_selected_indices:
            features[output_index, :] = feature_output
        else:
            features[output_index, :] = feature_output[scatter_context.indices_to_use]
        if nameless_features is not None:
            nameless_features[output_index, :] = feature_output[scatter_context.nameless_indices_to_use]
        if coauthor_similarity_values is not None and scatter_context.coauthor_similarity_index is not None:
            coauthor_similarity_values[output_index] = feature_output[scatter_context.coauthor_similarity_index]
        return

    features[output_index, :] = feature_output[scatter_context.selected_positions]
    if nameless_features is not None:
        nameless_features[output_index, :] = feature_output[scatter_context.nameless_positions]
    if (
        coauthor_similarity_values is not None
        and scatter_context.coauthor_similarity_index is not None
        and scatter_context.coauthor_position is not None
    ):
        coauthor_similarity_values[output_index] = feature_output[scatter_context.coauthor_position]


def _scatter_chunk_to_output(
    *,
    rust_features_chunk: np.ndarray,
    chunk_indices: list[int],
    scatter_context: ScatterContext,
    rust_chunk_is_full: bool,
) -> None:
    features = scatter_context.features
    nameless_features = scatter_context.nameless_features
    coauthor_similarity_values = scatter_context.coauthor_similarity_values
    chunk_slice = _contiguous_index_slice(chunk_indices)
    if chunk_slice is not None:
        if rust_chunk_is_full:
            if scatter_context.identity_selected_indices:
                features[chunk_slice, :] = rust_features_chunk
            else:
                np.take(
                    rust_features_chunk,
                    scatter_context.indices_to_use,
                    axis=1,
                    out=features[chunk_slice, :],
                )
            if nameless_features is not None:
                np.take(
                    rust_features_chunk,
                    scatter_context.nameless_indices_to_use,
                    axis=1,
                    out=nameless_features[chunk_slice, :],
                )
            if coauthor_similarity_values is not None and scatter_context.coauthor_similarity_index is not None:
                coauthor_similarity_values[chunk_slice] = rust_features_chunk[
                    :,
                    scatter_context.coauthor_similarity_index,
                ]
            return

        np.take(
            rust_features_chunk,
            scatter_context.selected_positions,
            axis=1,
            out=features[chunk_slice, :],
        )
        if nameless_features is not None:
            np.take(
                rust_features_chunk,
                scatter_context.nameless_positions,
                axis=1,
                out=nameless_features[chunk_slice, :],
            )
        if (
            coauthor_similarity_values is not None
            and scatter_context.coauthor_similarity_index is not None
            and scatter_context.coauthor_position is not None
        ):
            coauthor_similarity_values[chunk_slice] = rust_features_chunk[:, scatter_context.coauthor_position]
        return

    if rust_chunk_is_full:
        if scatter_context.identity_selected_indices:
            features[chunk_indices, :] = rust_features_chunk
        else:
            features[chunk_indices, :] = rust_features_chunk[:, scatter_context.indices_to_use]
        if nameless_features is not None:
            nameless_features[chunk_indices, :] = rust_features_chunk[:, scatter_context.nameless_indices_to_use]
        if coauthor_similarity_values is not None and scatter_context.coauthor_similarity_index is not None:
            coauthor_similarity_values[chunk_indices] = rust_features_chunk[
                :,
                scatter_context.coauthor_similarity_index,
            ]
        return

    features[chunk_indices, :] = rust_features_chunk[:, scatter_context.selected_positions]
    if nameless_features is not None:
        nameless_features[chunk_indices, :] = rust_features_chunk[:, scatter_context.nameless_positions]
    if (
        coauthor_similarity_values is not None
        and scatter_context.coauthor_similarity_index is not None
        and scatter_context.coauthor_position is not None
    ):
        coauthor_similarity_values[chunk_indices] = rust_features_chunk[:, scatter_context.coauthor_position]


def _signature_id_to_index_or_raise(signature_id_to_index: dict[Any, int], signature_id: Any) -> int:
    if signature_id in signature_id_to_index:
        return int(signature_id_to_index[signature_id])
    signature_id_str = str(signature_id)
    if signature_id_str in signature_id_to_index:
        return int(signature_id_to_index[signature_id_str])
    raise ValueError(
        "Rust indexed pair featurization received signature_id not present in Rust featurizer signature_ids: "
        f"{signature_id!r}"
    )


class FeaturizationInfo:
    """
    Class to store information about how to generate features

    Inputs:
        features_to_use: List[str]
            list of feature types to use
    """

    def __init__(
        self,
        features_to_use: list[str] | None = None,
    ):
        if features_to_use is None:
            features_to_use = list(DEFAULT_FEATURE_GROUPS)
        self.features_to_use = list(features_to_use)

        self.feature_group_to_index = {
            "name_similarity": [0, 1, 2, 3, 4, 5],
            "affiliation_similarity": [6],
            "email_similarity": [7, 8],
            "coauthor_similarity": [9, 10, 11],
            "venue_similarity": [12],
            "year_diff": [13],
            "title_similarity": [14, 15],
            "misc_features": [16, 17, 18, 19, 20],
            "name_counts": [21, 22, 23, 24, 25, 26],
            "embedding_similarity": [27],
            "journal_similarity": [28],
            "advanced_name_similarity": [29, 30, 31, 32],
        }
        unknown_feature_groups = sorted(set(self.features_to_use) - set(self.feature_group_to_index))
        if unknown_feature_groups:
            known_feature_groups = ", ".join(sorted(self.feature_group_to_index))
            raise ValueError(
                f"Unknown feature group(s): {unknown_feature_groups}. Known feature groups: {known_feature_groups}"
            )

        max_feature_index = max(
            (feature_index for group in self.feature_group_to_index.values() for feature_index in group),
            default=-1,
        )
        self.number_of_features = max_feature_index + 1

        lightgbm_monotone_constraints = {
            "name_similarity": ["1", "1", "1", "0", "0", "0"],
            "affiliation_similarity": ["0"],
            "email_similarity": ["1", "1"],
            "coauthor_similarity": ["1", "0", "1"],
            "venue_similarity": ["0"],
            "year_diff": ["-1"],
            "title_similarity": ["1", "1"],
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
                if feature_category in features_to_use and feature_category not in NAME_DEPENDENT_FEATURE_GROUPS
            ]
        )

    def selected_feature_indices(self) -> list[int]:
        """Return the canonical sorted feature columns selected by this configuration."""
        return sorted(
            {index for feature_group in self.features_to_use for index in self.feature_group_to_index[feature_group]}
        )

    def get_feature_names(self) -> list[str]:
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

        # position features
        if "misc_features" in self.features_to_use:
            feature_names.extend(
                ["position_diff", "abstract_count", "english_count", "same_language", "language_reliability_min"]
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


NUM_FEATURES = FeaturizationInfo().number_of_features


def _specter_vector_or_none(embedding: Any, *, paper_id: str) -> np.ndarray | None:
    """Normalize one SPECTER embedding and treat an all-zero vector as missing.

    Args:
        embedding: Array-like embedding payload.
        paper_id: Paper ID used to contextualize validation failures.

    Returns:
        A one-dimensional float vector, or ``None`` for an all-zero vector.

    Raises:
        ValueError: If the embedding is non-numeric or not one-dimensional.
    """

    try:
        vector = np.asarray(embedding, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SPECTER embedding for paper {paper_id!r} must be numeric") from exc
    if vector.ndim != 1:
        raise ValueError(f"SPECTER embedding for paper {paper_id!r} must be one-dimensional; got shape={vector.shape}")
    if np.all(vector == 0):
        return None
    return vector


def _require_preprocessed_value(
    value: _PreprocessedValueT | None,
    *,
    field: str,
) -> _PreprocessedValueT:
    """Return a materialized Python featurization value or surface the lifecycle error."""

    if value is None:
        raise RuntimeError(f"Python pair featurization requires preprocessed field {field}")
    return value


def _single_pair_featurize(
    work_input: tuple[str, str],
    index: int = -1,
    *,
    dataset: ANDData | None = None,
) -> tuple[list[int | float], int]:
    """
    Creates the features array for a single signature pair
    Pool workers read the dataset initialized once in worker-local state so
    process pools do not pickle it into every submitted chunk. Direct and
    serial callers pass it explicitly.

    Parameters
    ----------
    work_input: Tuple[str, str]
        pair of signature ids
    index: int
        the index of the pair in the list of all pairs,
        used to scatter results into the output row

    Returns
    -------
    Tuple: tuple of the features array, and the index, which is simply passed through
    """
    if dataset is None:
        dataset = getattr(_FEATURIZATION_WORKER_STATE, "dataset", None)
    if dataset is None:
        raise RuntimeError("featurization worker dataset is not initialized")

    features = []

    signature_1 = dataset.signatures[work_input[0]]
    signature_2 = dataset.signatures[work_input[1]]

    paper_id_1 = signature_1.paper_id
    paper_id_2 = signature_2.paper_id

    paper_1 = dataset.papers[str(paper_id_1)]
    paper_2 = dataset.papers[str(paper_id_2)]

    first_1 = _require_preprocessed_value(
        signature_1.author_info_first_normalized_without_apostrophe,
        field="signature_1.author_info_first_normalized_without_apostrophe",
    )
    first_2 = _require_preprocessed_value(
        signature_2.author_info_first_normalized_without_apostrophe,
        field="signature_2.author_info_first_normalized_without_apostrophe",
    )
    middle_1 = _require_preprocessed_value(
        signature_1.author_info_middle_normalized_without_apostrophe,
        field="signature_1.author_info_middle_normalized_without_apostrophe",
    )
    middle_2 = _require_preprocessed_value(
        signature_2.author_info_middle_normalized_without_apostrophe,
        field="signature_2.author_info_middle_normalized_without_apostrophe",
    )
    affiliation_ngrams_1 = _require_preprocessed_value(
        signature_1.author_info_affiliations_n_grams,
        field="signature_1.author_info_affiliations_n_grams",
    )
    affiliation_ngrams_2 = _require_preprocessed_value(
        signature_2.author_info_affiliations_n_grams,
        field="signature_2.author_info_affiliations_n_grams",
    )
    coauthor_blocks_1 = _require_preprocessed_value(
        signature_1.author_info_coauthor_blocks,
        field="signature_1.author_info_coauthor_blocks",
    )
    coauthor_blocks_2 = _require_preprocessed_value(
        signature_2.author_info_coauthor_blocks,
        field="signature_2.author_info_coauthor_blocks",
    )
    coauthor_ngrams_1 = _require_preprocessed_value(
        signature_1.author_info_coauthor_n_grams,
        field="signature_1.author_info_coauthor_n_grams",
    )
    coauthor_ngrams_2 = _require_preprocessed_value(
        signature_2.author_info_coauthor_n_grams,
        field="signature_2.author_info_coauthor_n_grams",
    )
    coauthors_1 = _require_preprocessed_value(
        signature_1.author_info_coauthors,
        field="signature_1.author_info_coauthors",
    )
    coauthors_2 = _require_preprocessed_value(
        signature_2.author_info_coauthors,
        field="signature_2.author_info_coauthors",
    )
    signature_name_counts_1 = _require_preprocessed_value(
        signature_1.author_info_name_counts,
        field="signature_1.author_info_name_counts",
    )
    signature_name_counts_2 = _require_preprocessed_value(
        signature_2.author_info_name_counts,
        field="signature_2.author_info_name_counts",
    )
    venue_ngrams_1 = _require_preprocessed_value(paper_1.venue_ngrams, field="paper_1.venue_ngrams")
    venue_ngrams_2 = _require_preprocessed_value(paper_2.venue_ngrams, field="paper_2.venue_ngrams")
    title_word_ngrams_1 = _require_preprocessed_value(
        paper_1.title_ngrams_words,
        field="paper_1.title_ngrams_words",
    )
    title_word_ngrams_2 = _require_preprocessed_value(
        paper_2.title_ngrams_words,
        field="paper_2.title_ngrams_words",
    )
    title_char_ngrams_1 = _require_preprocessed_value(
        paper_1.title_ngrams_chars,
        field="paper_1.title_ngrams_chars",
    )
    title_char_ngrams_2 = _require_preprocessed_value(
        paper_2.title_ngrams_chars,
        field="paper_2.title_ngrams_chars",
    )
    has_abstract_1 = _require_preprocessed_value(paper_1.has_abstract, field="paper_1.has_abstract")
    has_abstract_2 = _require_preprocessed_value(paper_2.has_abstract, field="paper_2.has_abstract")
    journal_ngrams_1 = _require_preprocessed_value(paper_1.journal_ngrams, field="paper_1.journal_ngrams")
    journal_ngrams_2 = _require_preprocessed_value(paper_2.journal_ngrams, field="paper_2.journal_ngrams")

    features.extend(
        [
            equal(
                first_1,
                first_2,
            ),
            counter_jaccard(
                Counter([p[0] for p in middle_1.split(" ") if len(p) > 0]),
                Counter([p[0] for p in middle_2.split(" ") if len(p) > 0]),
            ),
            equal_middle(middle_1, middle_2),
            (len(middle_1) == 0 and len(middle_2) != 0) or (len(middle_2) == 0 and len(middle_1) != 0),
            len(first_1) == 1 or len(first_2) == 1,
            any(len(middle) == 1 for middle in middle_1.split(" "))
            or any(len(middle) == 1 for middle in middle_2.split(" ")),
        ]
    )

    features.append(
        counter_jaccard(
            affiliation_ngrams_1,
            affiliation_ngrams_2,
        )
    )

    email_prefix_1: str | None = None
    email_prefix_2: str | None = None
    email_suffix_1: str | None = None
    email_suffix_2: str | None = None
    if (
        signature_1.author_info_email is not None
        and len(signature_1.author_info_email) > 0
        and signature_2.author_info_email is not None
        and len(signature_2.author_info_email) > 0
    ):
        email_prefix_1, email_suffix_1 = email_prefix_suffix(signature_1.author_info_email)
        email_prefix_2, email_suffix_2 = email_prefix_suffix(signature_2.author_info_email)

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
            jaccard(coauthor_blocks_1, coauthor_blocks_2),
            counter_jaccard(
                coauthor_ngrams_1,
                coauthor_ngrams_2,
                denominator_max=5000,
            ),
            jaccard(coauthors_1, coauthors_2),
        ]
    )

    features.append(counter_jaccard(venue_ngrams_1, venue_ngrams_2))

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
            counter_jaccard(title_word_ngrams_1, title_word_ngrams_2),
            counter_jaccard(title_char_ngrams_1, title_char_ngrams_2),
        ]
    )

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
            int(has_abstract_1) + int(has_abstract_2),
            english_or_unknown_count,
            paper_1.predicted_language == paper_2.predicted_language,
            min(float(paper_1.language_reliability or 0.0), float(paper_2.language_reliability or 0.0)),
        ]
    )

    features.extend(
        name_counts(
            signature_name_counts_1,
            signature_name_counts_2,
        )
    )

    specter_1 = None
    specter_2 = None
    if english_or_unknown_count == 2 and dataset.specter_embeddings is not None:
        if str(paper_id_1) in dataset.specter_embeddings:
            specter_1 = _specter_vector_or_none(
                dataset.specter_embeddings[str(paper_id_1)],
                paper_id=str(paper_id_1),
            )
        if str(paper_id_2) in dataset.specter_embeddings:
            specter_2 = _specter_vector_or_none(
                dataset.specter_embeddings[str(paper_id_2)],
                paper_id=str(paper_id_2),
            )

    if specter_1 is not None and specter_2 is not None:
        specter_sim = cosine_sim(specter_1, specter_2) + 1
    else:
        specter_sim = NUMPY_NAN

    features.append(specter_sim)  # , abstract_count, english_count])

    features.append(counter_jaccard(journal_ngrams_1, journal_ngrams_2))

    features.extend(
        name_text_features(
            first_1,
            first_2,
        )
    )

    # unifying feature type in features array
    features = [float(val) if isinstance(val, np.floating | float) else int(val) for val in features]

    return features, index


def parallel_helper(piece_of_work: tuple, worker_func: Callable):
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


def _execute_python_featurization_phase(
    *,
    dataset: ANDData,
    pieces_of_work: list[tuple[tuple[str, str], int]],
    n_jobs: int,
    chunk_size: int,
    scatter_context: ScatterContext,
) -> str:
    if n_jobs > 1:
        backend_used = "python_parallel"
        logger.info("Making %d feature vectors in parallel", len(pieces_of_work))

        pool_size = n_jobs if len(pieces_of_work) > 1000 else 1
        # Explicit platform policy to avoid implicit UniversalPool defaults at call sites.
        use_threads = platform.system() in ("Windows", "Darwin")
        with UniversalPool(
            processes=pool_size,
            use_threads=use_threads,
            initializer=_initialize_featurization_worker,
            initargs=(dataset,),
        ) as p:
            work_count = len(pieces_of_work)
            with tqdm(total=work_count, desc="Doing work", disable=work_count <= 10000) as pbar:
                for feature_output, index in p.imap(
                    functools.partial(parallel_helper, worker_func=_single_pair_featurize),
                    pieces_of_work,
                    min(chunk_size, max(1, int((work_count / n_jobs) / 2))),
                ):
                    _scatter_feature_row_from_source(
                        feature_output=np.asarray(feature_output, dtype=np.float64),
                        output_index=int(index),
                        scatter_context=scatter_context,
                        rust_chunk_is_full=True,
                    )
                    pbar.update()
        return backend_used

    backend_used = "python_serial"
    logger.info("Making %d feature vectors in serial", len(pieces_of_work))
    partial_func = functools.partial(
        parallel_helper,
        worker_func=functools.partial(_single_pair_featurize, dataset=dataset),
    )
    for piece in tqdm(pieces_of_work, total=len(pieces_of_work), desc="Doing work"):
        result = partial_func(piece)
        _scatter_feature_row_from_source(
            feature_output=np.asarray(result[0], dtype=np.float64),
            output_index=int(result[1]),
            scatter_context=scatter_context,
            rust_chunk_is_full=True,
        )
    return backend_used


def _execute_rust_batch_featurization_phase(
    *,
    dataset: ANDData,
    signature_pairs: list[tuple[str, str, int | float]],
    pieces_of_work: list[tuple[tuple[str, str], int]],
    featurizer_info: FeaturizationInfo,
    runtime_context: RuntimeContext,
    n_jobs: int,
    total_ram_bytes: int | None,
    rust_batch_total_ram_for_stage: int | None,
    rust_batch_rss_before_bytes: int,
    rust_batch_rss_peak_bytes: int,
    rust_batch_rss_source: str,
    rust_batch_rss_baseline_locked: bool,
    indices_to_use: list[int],
    nameless_indices_to_use: list[int],
    indices_needed_for_compute: list[int],
    identity_selected_indices: bool,
    coauthor_similarity_index: int | None,
    features: np.ndarray,
    nameless_features: np.ndarray | None,
    coauthor_similarity_values: np.ndarray | None,
) -> RustBatchExecutionResult:
    if len(pieces_of_work) <= 0:
        raise ValueError("Rust batch execution requires non-empty pieces_of_work")

    def _sample_rss_peak() -> None:
        nonlocal rust_batch_rss_peak_bytes
        if rust_batch_total_ram_for_stage is None:
            return
        rss_now, _ = memory_budget.current_rss_bytes_best_effort(rust_batch_total_ram_for_stage)
        if rss_now > rust_batch_rss_peak_bytes:
            rust_batch_rss_peak_bytes = rss_now

    rust_featurizer = feature_port._get_rust_feature_data(dataset)
    rust_selected_indices: list[int] | None = None
    if len(indices_needed_for_compute) > 0:
        rust_selected_indices = indices_needed_for_compute
    signature_id_to_index: dict[Any, int] = {}
    rust_signature_ids = rust_featurizer.signature_ids()
    for idx, sig_id in enumerate(rust_signature_ids):
        signature_id_to_index[sig_id] = int(idx)
        signature_id_to_index[str(sig_id)] = int(idx)
    logger.info("Rust indexed pair API enabled (signature_count=%d)", len(signature_id_to_index))
    rust_feature_count = NUM_FEATURES if rust_selected_indices is None else len(rust_selected_indices)
    rust_prediction_params = memory_budget.resolve_rust_batch_prediction_params()
    rust_batch_plan = memory_budget.compute_rust_batch_chunk_plan(
        num_features=rust_feature_count,
        total_pairs=len(pieces_of_work),
        total_rows=len(signature_pairs),
        selected_feature_count=len(indices_to_use),
        nameless_feature_count=len(nameless_indices_to_use),
        total_ram_bytes=(
            rust_batch_total_ram_for_stage if rust_batch_total_ram_for_stage is not None else total_ram_bytes
        ),
        base_chunk_pairs=int(rust_prediction_params["base_chunk_pairs"]),
        row_overhead_bytes=int(rust_prediction_params["row_overhead_bytes"]),
        persistent_row_overhead_bytes=int(rust_prediction_params["persistent_row_overhead_bytes"]),
        fixed_overhead_bytes=int(rust_prediction_params["fixed_overhead_bytes"]),
    )
    target_chunk_size = int(rust_batch_plan.chunk_pairs)
    total_ram_for_stage = int(rust_batch_plan.total_ram_bytes)
    predicted_stage_peak_delta_bytes = int(rust_batch_plan.predicted_stage_peak_delta_bytes)
    predicted_stage_peak_rss_bytes = int(rust_batch_plan.predicted_stage_peak_rss_bytes)
    if rust_batch_total_ram_for_stage != total_ram_for_stage:
        rust_batch_total_ram_for_stage = total_ram_for_stage
        if not rust_batch_rss_baseline_locked:
            rust_batch_rss_before_bytes, rust_batch_rss_source = memory_budget.current_rss_bytes_best_effort(
                total_ram_for_stage
            )
            rust_batch_rss_peak_bytes = rust_batch_rss_before_bytes
    _sample_rss_peak()
    logger.info(
        "Making %d feature vectors in Rust batch mode (target_chunk_size=%d "
        "base_chunk_pairs=%d bytes_per_pair_row=%d predicted_chunk_bytes=%d "
        "predicted_stage_peak_delta_bytes=%d predicted_stage_peak_rss_bytes=%d stage_budget_bytes=%d "
        "total_ram=%d total_ram_source=%s available=%d)",
        len(pieces_of_work),
        target_chunk_size,
        int(rust_batch_plan.base_chunk_pairs),
        int(rust_batch_plan.bytes_per_pair_row),
        int(rust_batch_plan.predicted_chunk_bytes),
        predicted_stage_peak_delta_bytes,
        predicted_stage_peak_rss_bytes,
        int(rust_batch_plan.stage_budget_bytes),
        int(rust_batch_plan.total_ram_bytes),
        str(rust_batch_plan.total_ram_source),
        int(rust_batch_plan.available_bytes),
    )

    selected_positions: list[int] = indices_to_use
    nameless_positions: list[int] = nameless_indices_to_use
    coauthor_position: int | None = coauthor_similarity_index
    if rust_selected_indices is not None:
        pos_by_feature_idx = {int(feature_idx): int(pos) for pos, feature_idx in enumerate(rust_selected_indices)}
        selected_positions = [pos_by_feature_idx[idx] for idx in indices_to_use]
        nameless_positions = [pos_by_feature_idx[idx] for idx in nameless_indices_to_use]
        if coauthor_similarity_index is not None:
            coauthor_position = pos_by_feature_idx[coauthor_similarity_index]

    rust_scatter_context = ScatterContext(
        features=features,
        nameless_features=nameless_features,
        coauthor_similarity_values=coauthor_similarity_values,
        identity_selected_indices=identity_selected_indices,
        indices_to_use=indices_to_use,
        nameless_indices_to_use=nameless_indices_to_use,
        selected_positions=selected_positions,
        nameless_positions=nameless_positions,
        coauthor_similarity_index=coauthor_similarity_index,
        coauthor_position=coauthor_position,
    )

    num_threads = max(1, int(n_jobs))
    rust_batch_adaptive_halvings = 0
    with tqdm(
        total=len(pieces_of_work),
        desc="Rust batch featurization",
        disable=len(pieces_of_work) <= 10000,
    ) as pbar:
        start_index = 0
        while start_index < len(pieces_of_work):
            chunk_work = pieces_of_work[start_index : start_index + target_chunk_size]
            rust_pairs_chunk = [pair for pair, _ in chunk_work]
            rust_pairs_chunk_indexed = [
                (
                    _signature_id_to_index_or_raise(signature_id_to_index, pair[0]),
                    _signature_id_to_index_or_raise(signature_id_to_index, pair[1]),
                )
                for pair in rust_pairs_chunk
            ]
            rust_features_chunk = np.asarray(
                rust_featurizer.featurize_pairs_matrix_indexed(
                    rust_pairs_chunk_indexed,
                    rust_selected_indices,
                    num_threads,
                    np.nan,
                ),
                dtype=np.float64,
            )

            if rust_features_chunk.shape[0] != len(chunk_work):
                raise RuntimeError(
                    "Rust batch featurizer returned mismatched feature count: "
                    f"expected={len(chunk_work)} got={rust_features_chunk.shape[0]}"
                )
            rust_chunk_columns = int(rust_features_chunk.shape[1])
            selected_column_count = len(rust_selected_indices) if rust_selected_indices is not None else NUM_FEATURES
            if rust_selected_indices is None and rust_chunk_columns != NUM_FEATURES:
                raise RuntimeError(
                    "Rust batch featurizer returned unexpected feature width: "
                    f"expected={NUM_FEATURES} got={rust_chunk_columns}"
                )
            if rust_selected_indices is not None and rust_chunk_columns not in {
                NUM_FEATURES,
                selected_column_count,
            }:
                raise RuntimeError(
                    "Rust batch featurizer returned unexpected feature width: "
                    f"expected={selected_column_count} (selected) or {NUM_FEATURES} (full) "
                    f"got={rust_chunk_columns}"
                )
            rust_chunk_is_full = rust_chunk_columns == NUM_FEATURES
            chunk_indices = [index for _, index in chunk_work]

            _scatter_chunk_to_output(
                rust_features_chunk=rust_features_chunk,
                chunk_indices=chunk_indices,
                scatter_context=rust_scatter_context,
                rust_chunk_is_full=rust_chunk_is_full,
            )
            _sample_rss_peak()
            if (
                rust_batch_total_ram_for_stage is not None
                and rust_batch_adaptive_halvings < 3
                and predicted_stage_peak_delta_bytes > 0
            ):
                observed_delta = max(0, rust_batch_rss_peak_bytes - rust_batch_rss_before_bytes)
                if observed_delta > predicted_stage_peak_delta_bytes * 1.2:
                    target_chunk_size = max(1, target_chunk_size // 2)
                    rust_batch_adaptive_halvings += 1
                    logger.warning(
                        "Rust batch adaptive chunking: observed_delta=%d > predicted_delta=%d * 1.2; "
                        "halving target_chunk_size to %d (halving %d/3) run_id=%s",
                        observed_delta,
                        predicted_stage_peak_delta_bytes,
                        target_chunk_size,
                        rust_batch_adaptive_halvings,
                        runtime_context.run_id,
                    )
            pbar.update(len(chunk_work))
            start_index += len(chunk_work)

    _sample_rss_peak()
    return RustBatchExecutionResult(
        rust_batch_plan=rust_batch_plan,
        rust_batch_total_ram_for_stage=rust_batch_total_ram_for_stage,
        rust_batch_rss_before_bytes=int(rust_batch_rss_before_bytes),
        rust_batch_rss_peak_bytes=int(rust_batch_rss_peak_bytes),
        rust_batch_rss_source=str(rust_batch_rss_source),
        rust_batch_adaptive_halvings=int(rust_batch_adaptive_halvings),
    )


def _is_partial_supervision_label(label: int | float) -> bool:
    """Return whether a pair label encodes a partial-supervision constraint.

    Negative labels are constraints. ``NaN`` labels represent unlabeled
    inference pairs and therefore remain eligible for compute.

    Args:
        label: Pair label supplied to featurization.

    Returns:
        ``True`` only for negative labels.
    """

    return bool(label < 0)


def many_pairs_featurize(
    signature_pairs: list[tuple[str, str, int | float]],
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    *,
    n_jobs: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nameless_featurizer_info: FeaturizationInfo | None = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
    runtime_context: RuntimeContext | None = None,
    total_ram_bytes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Featurize many signature pairs.

    Args:
        signature_pairs: The ``(signature_id_1, signature_id_2, label)`` pairs
            to featurize. Negative labels encode partial-supervision
            constraints and are not computed.
        dataset: The dataset containing the relevant data.
        featurizer_info: Listing of feature groups to use.
        n_jobs: The number of cpus to use.
        chunk_size: The chunk size for multiprocessing.
        nameless_featurizer_info: FeaturizationInfo for the features that do
            not use any name features; those are not computed when ``None``.
        nan_value: The value to replace NaNs with.
        delete_training_data: Whether to delete some suspicious training rows.
        runtime_context: Optional runtime context override.
        total_ram_bytes: Optional explicit RAM input used for stage-wise
            memory budgeting of Rust batch featurization.

    Returns:
        Tuple of (features, labels, nameless features or None).
    """
    featurize_start = time.perf_counter()
    backend_used = "no_compute_needed"
    if runtime_context is None:
        runtime_context = dataset.runtime_context
    signature_pairs = [(str(pair[0]), str(pair[1]), pair[2]) for pair in signature_pairs]
    _ensure_python_pair_signature_ngrams(dataset, signature_pairs, runtime_context)

    did_rust_batch = False
    rust_batch_plan: memory_budget.RustBatchChunkPlan | None = None
    rust_batch_total_ram_for_stage: int | None = None
    rust_batch_rss_before_bytes = 0
    rust_batch_rss_peak_bytes = 0
    rust_batch_rss_source = "unavailable"
    rust_batch_rss_baseline_locked = False
    rust_batch_adaptive_halvings = 0

    rust_module_available = False
    if _use_rust_featurizer(runtime_context, dataset):
        try:
            # Prewarm so the Arrow featurizer build doesn't land inside the RSS measurement window.
            feature_port._get_rust_feature_data(dataset)
            rust_module_available = True
        except Exception as exc:
            raise RuntimeError(f"Rust featurizer init failed (run_id={runtime_context.run_id} error={exc})") from exc
        try:
            rust_batch_total_ram_for_stage, _ = memory_budget.resolve_total_ram_bytes(total_ram_bytes)
            rust_batch_rss_before_bytes, rust_batch_rss_source = memory_budget.current_rss_bytes_best_effort(
                rust_batch_total_ram_for_stage
            )
            rust_batch_rss_peak_bytes = rust_batch_rss_before_bytes
            rust_batch_rss_baseline_locked = True
        except RuntimeError:
            # Preserve behavior for no-compute paths when RAM autodetection is unavailable.
            rust_batch_total_ram_for_stage = None

    def _sample_rust_batch_rss_peak() -> None:
        nonlocal rust_batch_rss_peak_bytes
        if rust_batch_total_ram_for_stage is None:
            return
        rss_now, _ = memory_budget.current_rss_bytes_best_effort(rust_batch_total_ram_for_stage)
        if rss_now > rust_batch_rss_peak_bytes:
            rust_batch_rss_peak_bytes = rss_now

    indices_to_use = featurizer_info.selected_feature_indices()

    nameless_indices_to_use: list[int] = []
    if nameless_featurizer_info is not None:
        nameless_indices_to_use = nameless_featurizer_info.selected_feature_indices()

    identity_selected_indices = indices_to_use == list(range(NUM_FEATURES))
    coauthor_similarity_index: int | None = None
    coauthor_similarity_values: np.ndarray | None = None
    if delete_training_data:
        coauthor_similarity_index = featurizer_info.feature_group_to_index["coauthor_similarity"][1]
        coauthor_similarity_values = np.full(len(signature_pairs), -float(LARGE_INTEGER), dtype=np.float64)

    indices_needed_for_compute: list[int] = sorted(
        set(indices_to_use)
        | set(nameless_indices_to_use)
        | ({coauthor_similarity_index} if coauthor_similarity_index is not None else set())
    )

    features = np.full(
        (len(signature_pairs), len(indices_to_use)),
        -float(LARGE_INTEGER),
        dtype=np.float64,
    )
    labels = np.zeros(len(signature_pairs))
    nameless_features: np.ndarray | None = None
    if nameless_featurizer_info is not None:
        nameless_features = np.full(
            (len(signature_pairs), len(nameless_indices_to_use)),
            -float(LARGE_INTEGER),
            dtype=np.float64,
        )
    default_scatter_context = ScatterContext(
        features=features,
        nameless_features=nameless_features,
        coauthor_similarity_values=coauthor_similarity_values,
        identity_selected_indices=identity_selected_indices,
        indices_to_use=indices_to_use,
        nameless_indices_to_use=nameless_indices_to_use,
        selected_positions=indices_to_use,
        nameless_positions=nameless_indices_to_use,
        coauthor_similarity_index=coauthor_similarity_index,
        coauthor_position=coauthor_similarity_index,
    )
    _sample_rust_batch_rss_peak()
    pieces_of_work = []
    logger.info("Creating %d pieces of work", len(signature_pairs))
    for i, pair in tqdm(enumerate(signature_pairs), desc="Creating work", disable=len(signature_pairs) <= 100000):
        labels[i] = pair[2]

        # negative labels are an indication of partial supervision
        if _is_partial_supervision_label(pair[2]):
            continue

        pieces_of_work.append(((pair[0], pair[1]), i))

    logger.info("Created pieces of work")

    if pieces_of_work:
        use_rust = _use_rust_featurizer(runtime_context, dataset)
        if use_rust and not rust_module_available:
            raise RuntimeError(
                "Rust backend requested for pair_featurization but s2and_rust extension is unavailable "
                f"(run_id={runtime_context.run_id})"
            )

        _log_featurization_backend_decision(
            runtime_context=runtime_context,
            pieces_of_work_count=len(pieces_of_work),
            n_jobs=n_jobs,
            use_rust_featurizer=use_rust,
            rust_module_available=rust_module_available,
        )

        if use_rust and rust_module_available and len(pieces_of_work) > 0:
            try:
                rust_batch_result = _execute_rust_batch_featurization_phase(
                    dataset=dataset,
                    signature_pairs=signature_pairs,
                    pieces_of_work=pieces_of_work,
                    featurizer_info=featurizer_info,
                    runtime_context=runtime_context,
                    n_jobs=n_jobs,
                    total_ram_bytes=total_ram_bytes,
                    rust_batch_total_ram_for_stage=rust_batch_total_ram_for_stage,
                    rust_batch_rss_before_bytes=rust_batch_rss_before_bytes,
                    rust_batch_rss_peak_bytes=rust_batch_rss_peak_bytes,
                    rust_batch_rss_source=rust_batch_rss_source,
                    rust_batch_rss_baseline_locked=rust_batch_rss_baseline_locked,
                    indices_to_use=indices_to_use,
                    nameless_indices_to_use=nameless_indices_to_use,
                    indices_needed_for_compute=indices_needed_for_compute,
                    identity_selected_indices=identity_selected_indices,
                    coauthor_similarity_index=coauthor_similarity_index,
                    features=features,
                    nameless_features=nameless_features,
                    coauthor_similarity_values=coauthor_similarity_values,
                )
                rust_batch_plan = rust_batch_result.rust_batch_plan
                rust_batch_total_ram_for_stage = rust_batch_result.rust_batch_total_ram_for_stage
                rust_batch_rss_before_bytes = rust_batch_result.rust_batch_rss_before_bytes
                rust_batch_rss_peak_bytes = rust_batch_result.rust_batch_rss_peak_bytes
                rust_batch_rss_source = rust_batch_result.rust_batch_rss_source
                rust_batch_adaptive_halvings = rust_batch_result.rust_batch_adaptive_halvings
                did_rust_batch = True
                backend_used = "rust_batch"
            except Exception as exc:
                raise RuntimeError(
                    "Rust batch featurization failed in strict rust backend "
                    f"(pairs={len(pieces_of_work)} run_id={runtime_context.run_id} "
                    f"failure_reason={exc})"
                ) from exc

        if use_rust and not did_rust_batch and len(pieces_of_work) > 0:
            raise RuntimeError(
                "Rust pair_featurization stage was selected but Rust batch execution did not complete "
                f"(run_id={runtime_context.run_id})"
            )

        if not did_rust_batch:
            backend_used = _execute_python_featurization_phase(
                dataset=dataset,
                pieces_of_work=pieces_of_work,
                n_jobs=n_jobs,
                chunk_size=chunk_size,
                scatter_context=default_scatter_context,
            )
        _sample_rust_batch_rss_peak()
        logger.info("Work completed")
    else:
        logger.info("Featurization backend decision: skipped compute (all pairs were pre-labeled)")
    _sample_rust_batch_rss_peak()

    if delete_training_data:
        logger.info("Deleting some training rows")
        negative_label_indices = labels == 0
        if coauthor_similarity_values is None:
            raise RuntimeError("delete_training_data requires coauthor_similarity_values to be computed")
        high_coauthor_sim_indices = coauthor_similarity_values > 0.95
        indices_to_remove = negative_label_indices & high_coauthor_sim_indices
        logger.info("Intending to remove %d rows", int(sum(indices_to_remove)))
        original_size = len(labels)
        features = features[~indices_to_remove, :]
        if nameless_features is not None:
            nameless_features = nameless_features[~indices_to_remove, :]
        labels = labels[~indices_to_remove]
        logger.info(
            "Removed %d rows and %d labels",
            int(original_size - features.shape[0]),
            int(original_size - len(labels)),
        )
    _sample_rust_batch_rss_peak()

    logger.info("Making numpy arrays for features and labels")
    if nameless_features is not None:
        nameless_features[np.isnan(nameless_features)] = nan_value
        _sample_rust_batch_rss_peak()
    features[np.isnan(features)] = nan_value
    _sample_rust_batch_rss_peak()

    if did_rust_batch and rust_batch_plan is not None:
        _sample_rust_batch_rss_peak()
        rss_after_bytes = rust_batch_rss_before_bytes
        if rust_batch_total_ram_for_stage is not None:
            rss_after_bytes, _ = memory_budget.current_rss_bytes_best_effort(rust_batch_total_ram_for_stage)
            _sample_rust_batch_rss_peak()
        rust_batch_prediction = memory_budget.summarize_prediction_accuracy(
            stage_name="pair_featurization_rust_batch",
            predicted_peak_delta_bytes=int(rust_batch_plan.predicted_stage_peak_delta_bytes),
            rss_before_bytes=rust_batch_rss_before_bytes,
            rss_peak_bytes=rust_batch_rss_peak_bytes,
            rss_after_bytes=rss_after_bytes,
        )
        logger.info(
            "Telemetry: pair_featurization_memory stage=%s "
            "predicted_peak_delta_bytes=%d predicted_peak_rss_bytes=%d "
            "total_rows=%d selected_feature_count=%d nameless_feature_count=%d "
            "predicted_features_matrix_bytes=%d predicted_labels_bytes=%d predicted_chunk_bytes=%d "
            "predicted_persistent_row_overhead_bytes=%d predicted_fixed_overhead_bytes=%d "
            "rss_before_bytes=%d rss_peak_bytes=%d rss_after_bytes=%d observed_peak_delta_bytes=%d "
            "prediction_error_ratio=%.3f underpredicted=%s adaptive_halvings=%d rss_source=%s",
            rust_batch_prediction.stage_name,
            int(rust_batch_prediction.predicted_peak_delta_bytes),
            int(rust_batch_prediction.predicted_peak_rss_bytes),
            int(rust_batch_plan.total_rows),
            int(rust_batch_plan.selected_feature_count),
            int(rust_batch_plan.nameless_feature_count),
            int(rust_batch_plan.predicted_features_matrix_bytes),
            int(rust_batch_plan.predicted_labels_bytes),
            int(rust_batch_plan.predicted_chunk_bytes),
            int(rust_batch_plan.predicted_persistent_row_overhead_bytes),
            int(rust_batch_plan.predicted_fixed_overhead_bytes),
            int(rust_batch_prediction.rss_before_bytes),
            int(rust_batch_prediction.rss_peak_bytes),
            int(rust_batch_prediction.rss_after_bytes),
            int(rust_batch_prediction.observed_peak_delta_bytes),
            float(rust_batch_prediction.prediction_error_ratio),
            bool(rust_batch_prediction.underpredicted),
            rust_batch_adaptive_halvings,
            rust_batch_rss_source,
        )
        memory_budget.emit_memory_telemetry(
            {
                "stage": rust_batch_prediction.stage_name,
                "predicted_peak_delta_bytes": rust_batch_prediction.predicted_peak_delta_bytes,
                "predicted_peak_rss_bytes": rust_batch_prediction.predicted_peak_rss_bytes,
                "total_rows": rust_batch_plan.total_rows,
                "selected_feature_count": rust_batch_plan.selected_feature_count,
                "nameless_feature_count": rust_batch_plan.nameless_feature_count,
                "predicted_features_matrix_bytes": rust_batch_plan.predicted_features_matrix_bytes,
                "predicted_labels_bytes": rust_batch_plan.predicted_labels_bytes,
                "predicted_chunk_bytes": rust_batch_plan.predicted_chunk_bytes,
                "predicted_persistent_row_overhead_bytes": rust_batch_plan.predicted_persistent_row_overhead_bytes,
                "predicted_fixed_overhead_bytes": rust_batch_plan.predicted_fixed_overhead_bytes,
                "rss_before_bytes": rust_batch_prediction.rss_before_bytes,
                "rss_peak_bytes": rust_batch_prediction.rss_peak_bytes,
                "rss_after_bytes": rust_batch_prediction.rss_after_bytes,
                "observed_peak_delta_bytes": rust_batch_prediction.observed_peak_delta_bytes,
                "prediction_error_ratio": rust_batch_prediction.prediction_error_ratio,
                "underpredicted": rust_batch_prediction.underpredicted,
                "adaptive_halvings": rust_batch_adaptive_halvings,
                "rss_source": rust_batch_rss_source,
            }
        )

    logger.info(
        "Telemetry stage: stage=pair_featurization seconds=%.3f total_pairs=%d computed_pairs=%d backend=%s",
        time.perf_counter() - featurize_start,
        len(signature_pairs),
        len(pieces_of_work),
        backend_used,
    )
    logger.info("Numpy arrays made")
    return features, labels, nameless_features


def _training_signature_splits(
    dataset: ANDData,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    """Resolve signature splits from the dataset's configured split authority."""

    if dataset.train_blocks is not None:
        return dataset.split_cluster_signatures_fixed()
    if dataset.train_signatures is not None:
        return dataset.split_data_signatures_fixed()
    return dataset.split_cluster_signatures()


def resolve_selection_pairs(
    dataset: ANDData,
) -> tuple[list[tuple[str, str, int | float]], list[tuple[str, str, int | float]]]:
    """Resolve train/validation pairs without accessing test-pair labels.

    Args:
        dataset: A ``mode='train'`` dataset.

    Returns:
        Tuple of (train_pairs, val_pairs).

    Raises:
        ValueError: If the dataset is not in training mode or the resolved
            splits contain the same unordered signature pair.
    """

    if dataset.mode != "train":
        raise ValueError(f"resolve_selection_pairs requires mode='train', got {dataset.mode!r}")
    if dataset.train_pairs is not None:
        train_pairs, val_pairs = dataset.fixed_train_val_pairs()
    else:
        train_signatures, val_signatures, _ = _training_signature_splits(dataset)
        train_pairs, val_pairs, _ = dataset.split_pairs(train_signatures, val_signatures, {})

    train_identities = {tuple(sorted((str(left), str(right)))) for left, right, _ in train_pairs}
    val_identities = {tuple(sorted((str(left), str(right)))) for left, right, _ in val_pairs}
    overlap = train_identities & val_identities
    if overlap:
        raise ValueError(
            "Pairwise train and validation splits overlap by unordered signature pair: "
            f"count={len(overlap)}, sample={sorted(overlap)[:5]}"
        )
    return train_pairs, val_pairs


def resolve_training_pairs(
    dataset: ANDData,
) -> tuple[
    list[tuple[str, str, int | float]],
    list[tuple[str, str, int | float]],
    list[tuple[str, str, int | float]],
]:
    """Resolve the train/val/test signature pairs for a training dataset.

    Args:
        dataset: A ``mode='train'`` dataset.

    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs).

    Raises:
        ValueError: If the dataset is not in training mode.
    """
    if dataset.mode != "train":
        raise ValueError(f"resolve_training_pairs requires mode='train', got {dataset.mode!r}")
    if dataset.train_pairs is not None:
        return dataset.fixed_pairs()
    train_signatures, val_signatures, test_signatures = _training_signature_splits(dataset)
    return dataset.split_pairs(train_signatures, val_signatures, test_signatures)


def featurize(
    dataset: ANDData,
    featurizer_info: FeaturizationInfo,
    *,
    n_jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nameless_featurizer_info: FeaturizationInfo | None = None,
    nan_value: float = np.nan,
    delete_training_data: bool = False,
    total_ram_bytes: int | None = None,
) -> tuple[TupleOfArrays, TupleOfArrays, TupleOfArrays] | TupleOfArrays:
    """Featurize the input dataset.

    Args:
        dataset: The dataset containing the relevant data.
        featurizer_info: Listing of feature groups to use.
        n_jobs: The number of cpus to use.
        chunk_size: The chunk size for multiprocessing.
        nameless_featurizer_info: FeaturizationInfo for the features that do
            not use any name features; those are not computed when ``None``.
        nan_value: The value to replace NaNs with.
        delete_training_data: Whether to delete some suspicious training examples.
        total_ram_bytes: Optional explicit RAM input used for stage-wise memory budgeting.

    Returns:
        Train/val/test features and labels if mode is 'train'; features and
        labels for all pairs if mode is 'inference'.
    """
    if dataset.mode == "inference":
        logger.info("featurizing all pairs")
        all_pairs = dataset.all_pairs()
        all_features = many_pairs_featurize(
            all_pairs,
            dataset,
            featurizer_info,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
            nameless_featurizer_info=nameless_featurizer_info,
            nan_value=nan_value,
            total_ram_bytes=total_ram_bytes,
        )
        logger.info("featurized all pairs")
        return all_features

    train_pairs, val_pairs, test_pairs = resolve_training_pairs(dataset)
    split_results = []
    for split_name, split_pairs, split_delete_training_data in (
        ("train", train_pairs, delete_training_data),
        ("val", val_pairs, False),
        ("test", test_pairs, False),
    ):
        logger.info("featurizing %s", split_name)
        split_results.append(
            many_pairs_featurize(
                split_pairs,
                dataset,
                featurizer_info,
                n_jobs=n_jobs,
                chunk_size=chunk_size,
                nameless_featurizer_info=nameless_featurizer_info,
                nan_value=nan_value,
                delete_training_data=split_delete_training_data,
                total_ram_bytes=total_ram_bytes,
            )
        )
        logger.info("featurized %s", split_name)
    train_features, val_features, test_features = split_results
    return train_features, val_features, test_features

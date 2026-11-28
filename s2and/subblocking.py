from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from s2and.orcid_prefix_counts import (
    DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256,
    load_packaged_orcid_prefix_counts,
)
from s2and.text import (
    canonicalize_name_parts,
    normalize_orcid,
)


class _LazyCanonicalOrcidPrefixCounts(Mapping[str, Mapping[str, int]]):
    """Defer canonical artifact I/O until subblocking actually needs the priors."""

    def load(self) -> Mapping[str, Mapping[str, int]]:
        return load_packaged_orcid_prefix_counts(DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256).counts

    def __getitem__(self, key: str) -> Mapping[str, int]:
        return self.load()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.load())

    def __len__(self) -> int:
        return len(self.load())


def _resolved_orcid_prefix_counts(
    counts: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    return counts.load() if isinstance(counts, _LazyCanonicalOrcidPrefixCounts) else counts


FIRST_K_LETTER_COUNTS = _LazyCanonicalOrcidPrefixCounts()


def canonical_orcid_prefix_counts_data_sha256() -> str:
    """Return the content hash of the verified default ORCID priors."""

    return load_packaged_orcid_prefix_counts(DEFAULT_ORCID_PREFIX_COUNTS_DATA_SHA256).data_sha256


def normalize_orcid_for_subblocking(value: Any) -> str | None:
    """Return the canonical ORCID key used by Rust Arrow subblocking inputs.

    This mirrors Rust's `normalize_orcid_owned(...)`: find one ORCID-shaped
    token, uppercase the check digit, and format the result with hyphens.
    """

    return normalize_orcid(value)


@dataclass(frozen=True)
class GraphSubblockingConfig:
    """Configuration for the graph fallback used by subblocking."""

    neighbor_mode: str = "projection"
    neighbors: int = 16
    min_edge_score: float = 0.30
    specter_weight: float = 1.0
    coauthor_weight: float = 0.35
    affiliation_weight: float = 0.20
    max_exact_knn_group_size: int = 25_000
    projection_count: int = 12
    projection_window: int = 12
    max_candidate_edges: int = 5_000_000
    pack_components: bool = True
    component_pack_strategy: str = "edge-greedy"
    sparse_evidence_edges: bool = True
    sparse_evidence_max_posting_size: int = 8
    sparse_evidence_neighbors: int = 1
    sparse_evidence_min_weight: float = 0.40
    sparse_evidence_include_coauthors: bool = True
    sparse_evidence_include_affiliations: bool = False
    component_pack_top_k: int = 8
    local_move_passes: int = 0
    adaptive_projection: bool = False
    adaptive_projection_max_group_size: int = 5_000
    adaptive_projection_count: int = 24
    adaptive_projection_window: int = 24


def _resolve_graph_subblocking_config(raw_config: object) -> GraphSubblockingConfig:
    """Resolve one strict graph-subblocking configuration value."""

    if raw_config is None:
        return GraphSubblockingConfig()
    if isinstance(raw_config, GraphSubblockingConfig):
        return raw_config
    if isinstance(raw_config, Mapping):
        return GraphSubblockingConfig(**dict(raw_config))
    raise ValueError(
        f"subblocking_graph_config must be a GraphSubblockingConfig, mapping, or None; got {type(raw_config).__name__}"
    )


def signature_name_parts_for_subblocking(signature) -> tuple[str, str]:
    """Canonical (canonical_v2) first/middle for subblocking keys.

    Dash handling is uniform (D4): every dash-like character binds given-name
    compounds identically.
    """

    first = signature.author_info_first_normalized
    middle = signature.author_info_middle_normalized
    if first is not None and middle is not None:
        return first, middle
    # Rust preprocessing can defer normalized name fields; reconstruct with Python-equivalent logic.
    parts = canonicalize_name_parts(
        signature.author_info_first, signature.author_info_middle, signature.author_info_last
    )
    return parts.first, parts.middle


# A compound keys as its joined form only when the block shows that spelling much more often than the
# split spelling, so that distinct names such as "qi an" and "qian" stay apart on weak evidence.
COMPOUND_JOIN_RATIO = 8


def subblocking_name_parts(names: Sequence[tuple[str, str]], maximum_size: int) -> list[tuple[str, str]]:
    """Return the first and middle names that key each record of one subblocking request.

    Only subblock keys use these names. Canonical names, features, counts and constraints do not
    change. A canonical first with two or more tokens is a dash-bound compound, for example
    ``"hui min"`` from ``Hui-Min`` or ``"j d"`` from ``J.-D.``. Each compound keys as one token:

    - When every token has one character, the first token is the key first. The other tokens go
      before the canonical middle: ``("j d", "t")`` keys as ``("j", "d t")``.
    - Else the compound keys as its joined form with the canonical middle when the request has at
      most ``maximum_size`` records with that canonical first (``n_join``) and
      ``(n_join + 1) / (n_split + 1) > COMPOUND_JOIN_RATIO``. ``n_split`` counts records whose
      canonical first is the first token and whose middle starts with the second token.
    - Else the compound splits as initials do: ``("hui min", "")`` keys as ``("hui", "min")``.

    Args:
        names: Canonical (first, middle) of every record in the request, in request order.
        maximum_size: Maximum subblock size. A joined form with more records is already a prefix
            dead end, and a few joined records would only change its middle-name and fallback groups.

    Returns:
        Key (first, middle) for each record, in the order of ``names``.
    """

    first_counts = Counter(first for first, _middle in names)
    split_counts = Counter((first, middle.split()[0]) for first, middle in names if " " not in first and middle)
    key_names = []
    for first, middle in names:
        tokens = first.split()
        if len(tokens) < 2:
            key_names.append((first, middle))
            continue
        joined = "".join(tokens)
        n_join = first_counts[joined]
        if (
            any(len(token) > 1 for token in tokens)
            and n_join <= maximum_size
            and n_join + 1 > COMPOUND_JOIN_RATIO * (split_counts[(tokens[0], tokens[1])] + 1)
        ):
            key_names.append((joined, middle))
        else:
            key_names.append((tokens[0], " ".join([*tokens[1:], middle] if middle else tokens[1:])))
    return key_names


def _require_unique_subblocking_signature_ids(signature_ids: Iterable[Any]) -> list[Any]:
    """Preserve subblocking IDs while rejecting duplicate canonical identities."""

    original_signature_ids = list(signature_ids)
    seen_signature_ids: set[str] = set()
    for signature_id in original_signature_ids:
        normalized_signature_id = str(signature_id)
        if normalized_signature_id in seen_signature_ids:
            raise ValueError(
                f"Subblocking signature_ids must be unique after string coercion: {normalized_signature_id!r}"
            )
        seen_signature_ids.add(normalized_signature_id)
    return original_signature_ids


def _make_subblocks_with_telemetry_arrow_rust(
    arrow_dataset: Any,
    signature_ids,
    maximum_size=15000,
    first_k_letter_counts_sorted=FIRST_K_LETTER_COUNTS,
    graph_subblocking_config: GraphSubblockingConfig | None = None,
    graph_subblocking_random_seed: int = 0,
    use_orcid_subblocking: bool = True,
) -> tuple[dict[str, list[str]], dict[str, Any], dict[str, list[str]]]:
    """Run native Rust graph subblocking with signature rows loaded from Arrow.

    Returns:
        The subblocks, the planner telemetry, and the no-middle bucket of each first-name dead
        end.
    """

    from s2and.runtime import load_s2and_rust_extension

    first_k_letter_counts_sorted = _resolved_orcid_prefix_counts(first_k_letter_counts_sorted)
    rust_prefix_counts = {prefix: dict(nested_counts) for prefix, nested_counts in first_k_letter_counts_sorted.items()}
    rust_make_subblocks = load_s2and_rust_extension().make_subblocks_with_telemetry_arrow_native_graph
    subblocks, telemetry, no_middle_dead_ends = rust_make_subblocks(
        arrow_dataset.native,
        [str(signature_id) for signature_id in _require_unique_subblocking_signature_ids(signature_ids)],
        int(maximum_size),
        rust_prefix_counts,
        graph_subblocking_config,
        int(graph_subblocking_random_seed),
        bool(use_orcid_subblocking),
    )
    return (
        {str(key): list(values) for key, values in dict(subblocks).items()},
        dict(telemetry),
        {str(first): [str(signature_id) for signature_id in bucket] for first, bucket in no_middle_dead_ends},
    )

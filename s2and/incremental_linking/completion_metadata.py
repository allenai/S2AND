"""Lazy metadata adapters shared by Python and Arrow incremental completion."""

import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol

from s2and.name_tuple_artifact import load_packaged_name_tuple_artifact
from s2and.text import canonical_name_tuple_pair, normalize_orcid_compact

logger = logging.getLogger("s2and")


class CompletionSignature(Protocol):
    """Read-only signature fields used by completion and its diagnostics."""

    @property
    def author_info_first(self) -> str | None: ...

    @property
    def author_info_first_normalized_without_apostrophe(self) -> str | None: ...

    @property
    def author_info_last(self) -> str | None: ...

    @property
    def paper_id(self) -> object: ...


def signature_first(signature: CompletionSignature) -> str:
    """Read the prepared first name with the existing raw-name fallback."""
    return signature.author_info_first_normalized_without_apostrophe or signature.author_info_first or ""


class SignatureFirstNames(Mapping[str, str]):
    """Project signatures to first names without copying or scanning them."""

    def __init__(self, signatures: Mapping[str, CompletionSignature]) -> None:
        self._signatures = signatures

    def __getitem__(self, signature_id: str) -> str:
        return signature_first(self._signatures[signature_id])

    def __iter__(self) -> Iterator[str]:
        return iter(self._signatures)

    def __len__(self) -> int:
        return len(self._signatures)


class SignatureOrcids(Mapping[str, str | None]):
    """Project optional ORCID metadata to its normalized comparison value."""

    def __init__(self, signatures: Mapping[str, CompletionSignature]) -> None:
        self._signatures = signatures

    def __getitem__(self, signature_id: str) -> str | None:
        value = getattr(self._signatures[signature_id], "author_info_orcid", None)
        return None if value is None else normalize_orcid_compact(value)

    def __iter__(self) -> Iterator[str]:
        return iter(self._signatures)

    def __len__(self) -> int:
        return len(self._signatures)


def name_tuples_for_incremental_rules(
    name_tuples: set[tuple[str, str]] | frozenset[tuple[str, str]] | None,
) -> set[tuple[str, str]] | frozenset[tuple[str, str]]:
    """Resolve aliases once at the completion boundary, retaining canonical sets."""
    if isinstance(name_tuples, frozenset):
        return name_tuples
    if isinstance(name_tuples, set):
        return {canonical_name_tuple_pair(first_a, first_b) for first_a, first_b in name_tuples}
    if name_tuples is None:
        return load_packaged_name_tuple_artifact().pairs
    raise TypeError("name_tuples must be None or a set/frozenset of (first_a, first_b) tuples")


def log_rejected_links(signature_ids: Sequence[str], signatures: Mapping[str, CompletionSignature]) -> None:
    """Report name-incompatible assignments without exposing metadata to the core."""
    for signature_id in signature_ids:
        signature = signatures[signature_id]
        logger.info(
            "Incremental clustering prevented a name compatibility issue from being "
            f"added while clustering {signature.author_info_first} {signature.author_info_last} "
            f"on {signature.paper_id}"
        )

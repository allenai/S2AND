"""Signature ordering and cluster-seed validation contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def normalize_cluster_seed_disallow_pairs(
    pairs: Iterable[tuple[Any, Any]],
    *,
    valid_signature_ids: Iterable[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return canonical undirected disallow pairs after schema validation."""

    valid_signature_id_set = None if valid_signature_ids is None else {str(value) for value in valid_signature_ids}
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for left, right in pairs:
        left_id = str(left)
        right_id = str(right)
        if not left_id or not right_id:
            raise ValueError("cluster seed disallow pairs cannot contain empty signature ids")
        if left_id == right_id:
            raise ValueError(f"cluster seed disallow pair cannot be a self-pair: {left_id!r}")
        if valid_signature_id_set is not None:
            missing = sorted({left_id, right_id}.difference(valid_signature_id_set))
            if missing:
                raise ValueError(
                    f"cluster seed disallow pair contains signatures missing from signature set: {missing}"
                )
        pair = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append(pair)
    return tuple(normalized)


def filter_cluster_seed_disallows_for_signature_subset(
    pairs: Iterable[tuple[Any, Any]],
    signature_ids: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    """Keep only disallow pairs whose endpoints are both inside a signature set."""

    signature_id_set = {str(signature_id) for signature_id in signature_ids}
    filtered = [
        (str(left), str(right))
        for left, right in pairs
        if str(left) in signature_id_set and str(right) in signature_id_set
    ]
    return normalize_cluster_seed_disallow_pairs(filtered)


@dataclass(frozen=True)
class FeatureBlockSignatureOrder:
    """Deterministic mini-block signature order for numeric linker arrays."""

    signature_ids: tuple[str, ...]
    query_signature_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        signature_ids = tuple(str(value) for value in self.signature_ids)
        query_signature_ids = tuple(str(value) for value in self.query_signature_ids)
        if len(set(signature_ids)) != len(signature_ids):
            raise ValueError("FeatureBlockSignatureOrder.signature_ids must be unique")
        missing_queries = sorted(set(query_signature_ids) - set(signature_ids))
        if missing_queries:
            raise ValueError(f"query_signature_ids are missing from signature_ids: {missing_queries}")
        object.__setattr__(self, "signature_ids", signature_ids)
        object.__setattr__(self, "query_signature_ids", query_signature_ids)

    @property
    def signature_id_to_index(self) -> dict[str, int]:
        """Return this order as the numeric linker index map."""

        return {signature_id: index for index, signature_id in enumerate(self.signature_ids)}

"""Shared bucket derivation for promoted incremental-linking gates."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np

FirstNameBucket = Literal["single_letter_first", "multi_letter_first"]
QueryView = Literal["full", "initial_only"]

_LETTERS_RE = re.compile(r"[A-Za-z]+")
_VALID_FIRST_NAME_BUCKETS: frozenset[str] = frozenset(
    {
        "single_letter_first",
        "multi_letter_first",
    }
)
_VALID_QUERY_VIEWS: frozenset[str] = frozenset({"full", "initial_only"})


def normalize_bucket_letters(value: Any) -> str:
    """Return lowercase ASCII letters used by promoted gate bucket derivation."""

    if value is None:
        return ""
    return "".join(_LETTERS_RE.findall(str(value))).lower()


def first_name_bucket_from_token_view(query_first_token: Any, query_view: Any) -> FirstNameBucket:
    """Classify a query row into the promoted gate's first-name bucket."""

    token = normalize_bucket_letters(query_first_token)
    if not token and str(query_view) == "initial_only":
        return "single_letter_first"
    return "single_letter_first" if len(token) <= 1 else "multi_letter_first"


def first_name_bucket_array(
    query_first_tokens: Sequence[Any] | np.ndarray, query_views: Sequence[Any] | np.ndarray
) -> np.ndarray:
    """Return one first-name bucket per row signal."""

    tokens = np.asarray(query_first_tokens, dtype=object)
    views = np.asarray(query_views, dtype=object)
    if tokens.ndim != 1 or views.ndim != 1 or len(tokens) != len(views):
        raise ValueError(
            f"query_first_tokens and query_views must be 1D arrays with equal length: {tokens.shape} != {views.shape}"
        )
    return np.asarray(
        [first_name_bucket_from_token_view(token, view) for token, view in zip(tokens, views, strict=True)],
        dtype=object,
    )


def validate_first_name_bucket(value: Any) -> FirstNameBucket:
    """Validate a materialized first-name bucket value."""

    bucket = str(value)
    if bucket not in _VALID_FIRST_NAME_BUCKETS:
        raise ValueError(f"Unknown promoted gate first_name_bucket: {bucket!r}")
    return cast(FirstNameBucket, bucket)


def validate_query_view(value: Any) -> QueryView:
    """Validate a retrieval query-view boundary value."""

    view = str(value)
    if view not in _VALID_QUERY_VIEWS:
        raise ValueError(f"Unknown retrieval query_view: {view!r}")
    return cast(QueryView, view)


def normalize_query_views(query_view: str | Sequence[str], query_count: int) -> QueryView | tuple[QueryView, ...]:
    """Validate scalar or per-query retrieval view input."""

    if isinstance(query_view, str):
        return validate_query_view(query_view)
    views = tuple(validate_query_view(value) for value in query_view)
    if len(views) != int(query_count):
        raise ValueError(f"query_view length must match queries: {len(views)} != {int(query_count)}")
    return views

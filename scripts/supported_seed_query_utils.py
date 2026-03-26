"""Shared helpers for ORCID-supported seed-query evaluation.

This module owns the supported-query label construction used by the `h_wang`
task-1 and task-2 evaluators so downstream consumers do not need to import
private helpers from one another.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ORCID_PATTERN = re.compile(r"(\d{4}-?\d{4}-?\d{4}-?[\dXx]{4})")


@dataclass(frozen=True)
class SupportedQuery:
    """Scored-query metadata for ORCID-supported seed evaluation."""

    query_id: str
    normalized_orcid: str
    support_type: str
    supported_cluster_ids: frozenset[str]
    supported_seed_signature_count: int
    orcid_group_size: int
    query_subblock_key: str
    query_subblock_type: str


def normalize_orcid(orcid: str | None) -> str | None:
    """Normalize ORCID to the compact uppercase 16-character form."""

    if not orcid:
        return None
    matches = ORCID_PATTERN.findall(str(orcid))
    if not matches:
        return None
    return matches[0].upper().replace("-", "")


def extract_signature_orcid(signature_payload: dict[str, Any]) -> str | None:
    """Return the normalized ORCID for a raw extracted signature payload."""

    author_info = signature_payload.get("author_info", {})
    if str(author_info.get("source_id_source", "")) != "ORCID":
        return None
    source_ids = author_info.get("source_ids") or []
    if len(source_ids) == 0:
        return None
    return normalize_orcid(str(source_ids[0]))


def load_query_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load the supported-query metadata payload keyed by query ID."""

    with path.open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    rows = payload.get("query_rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Invalid query-set payload at {path}: expected list under 'query_rows'")
    return {str(row["query_id"]): dict(row) for row in rows}


def build_orcid_seed_cluster_counts(
    *,
    raw_signatures: dict[str, Any],
    signature_to_cluster_id: dict[str, str],
) -> dict[str, Counter[str]]:
    """Count supported seed-cluster memberships by normalized ORCID."""

    counts_by_orcid: dict[str, Counter[str]] = defaultdict(Counter)
    for signature_id, cluster_id in signature_to_cluster_id.items():
        signature_payload = raw_signatures.get(str(signature_id))
        if not isinstance(signature_payload, dict):
            continue
        normalized_orcid = extract_signature_orcid(signature_payload)
        if normalized_orcid is None:
            continue
        counts_by_orcid[normalized_orcid][str(cluster_id)] += 1
    return counts_by_orcid


def build_supported_queries(
    *,
    query_metadata: dict[str, dict[str, Any]],
    seed_cluster_counts_by_orcid: dict[str, Counter[str]],
) -> tuple[list[SupportedQuery], dict[str, Any]]:
    """Build the scored supported-query slice from cached seed support."""

    supported_queries: list[SupportedQuery] = []
    support_type_counts: Counter[str] = Counter()

    for query_id, query_meta in sorted(query_metadata.items()):
        normalized_orcid = str(query_meta["normalized_orcid"])
        positive_support = {
            str(cluster_id): int(count)
            for cluster_id, count in seed_cluster_counts_by_orcid.get(normalized_orcid, Counter()).items()
            if int(count) > 0
        }
        if not positive_support:
            continue
        support_type = "unique" if len(positive_support) == 1 else "ambiguous"
        support_type_counts[support_type] += 1
        supported_queries.append(
            SupportedQuery(
                query_id=str(query_id),
                normalized_orcid=normalized_orcid,
                support_type=support_type,
                supported_cluster_ids=frozenset(positive_support),
                supported_seed_signature_count=sum(positive_support.values()),
                orcid_group_size=int(query_meta["orcid_group_size"]),
                query_subblock_key=str(query_meta["query_subblock_key"]),
                query_subblock_type=str(query_meta["query_subblock_type"]),
            )
        )

    summary = {
        "supported_query_count": len(supported_queries),
        "support_type_counts": {
            "unique": int(support_type_counts.get("unique", 0)),
            "ambiguous": int(support_type_counts.get("ambiguous", 0)),
        },
    }
    return supported_queries, summary

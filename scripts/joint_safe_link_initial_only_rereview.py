"""Model-visible initial-only re-review decisions for safe-link datasets."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL_ONLY_REREVIEW_RESULTS_PATH = (
    REPO_ROOT / "scratch" / "s2and_initial_only_rereview_20260429" / "aggregate_reviewed_results.tsv"
)

FEATURE_CONTRACT_FAILURE_ISSUES = frozenset(
    {
        "raw_query_author_needed",
        "member_author_name_needed",
        "same_author_support_needed",
    }
)


@dataclass(frozen=True)
class InitialOnlyRereviewDecision:
    """Collapsed model-visible re-review decision for one initial-only query."""

    query_group_id: str
    dataset: str
    action: str
    reason_bucket: str
    safe_component_key_texts: tuple[str, ...]
    reviewed_row_count: int
    review_decisions: tuple[str, ...]
    evidence_issues: tuple[str, ...]


def _split_component_keys(value: Any) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split("|") if part.strip())


def _single_component_key_text_for_ledger(value: str) -> bool:
    """Return whether a pipe-containing key should stay atomic in ledgers."""

    return "|middle=" in value or "|specter=" in value


def expand_safe_component_key_texts_for_ledger(values: tuple[str, ...]) -> tuple[str, ...]:
    """Expand unambiguous multi-key review fields for ledger rows."""

    keys: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        if not text:
            continue
        parts = (text,) if _single_component_key_text_for_ledger(text) else _split_component_keys(text)
        for part in parts:
            if part not in seen:
                keys.append(part)
                seen.add(part)
    return tuple(keys)


def resolve_reviewed_safe_component_keys(
    safe_component_key_texts: tuple[str, ...],
    *,
    candidate_component_keys: set[str],
) -> tuple[str, ...]:
    """Resolve reviewed safe-key text against the current candidate surface."""

    resolved: list[str] = []
    seen: set[str] = set()
    for value in safe_component_key_texts:
        text = value.strip()
        if not text:
            continue
        if text in candidate_component_keys:
            if text not in seen:
                resolved.append(text)
                seen.add(text)
            continue
        for part in _split_component_keys(text):
            if part in candidate_component_keys and part not in seen:
                resolved.append(part)
                seen.add(part)
    return tuple(resolved)


def _collapse_review_rows(query_group_id: str, rows: list[dict[str, str]]) -> InitialOnlyRereviewDecision:
    datasets = sorted({str(row.get("dataset", "")) for row in rows if str(row.get("dataset", ""))})
    if len(datasets) != 1:
        raise ValueError(f"Initial-only re-review has inconsistent datasets for {query_group_id!r}: {datasets}")
    review_decisions = tuple(sorted({str(row.get("active_contract_decision", "")) for row in rows}))
    evidence_issues = tuple(sorted({str(row.get("evidence_contract_issue", "")) for row in rows}))
    safe_texts = tuple(
        sorted(
            {
                str(row.get("active_contract_safe_positive_component_keys", "")).strip()
                for row in rows
                if str(row.get("active_contract_decision", "")) == "link"
                and str(row.get("active_contract_safe_positive_component_keys", "")).strip()
            }
        )
    )
    decision_set = set(review_decisions)
    issue_set = set(evidence_issues)
    if decision_set == {"link"}:
        if not safe_texts:
            raise ValueError(f"Initial-only link decision has no safe component key: {query_group_id!r}")
        action = "candidate_positive"
        reason_bucket = "unanimous_model_visible_link"
    elif "link" in decision_set:
        action = "drop_query"
        reason_bucket = "conflicting_model_visible_reviews"
    elif "impossible" in decision_set:
        action = "drop_query"
        reason_bucket = "model_visible_impossible"
    elif issue_set & FEATURE_CONTRACT_FAILURE_ISSUES:
        action = "drop_query"
        reason_bucket = "feature_contract_failure"
    elif decision_set == {"abstain"} and issue_set <= {"packet_too_weak"}:
        action = "force_no_positive"
        reason_bucket = "packet_too_weak"
    else:
        action = "drop_query"
        reason_bucket = "unresolved_model_visible_non_link"
    return InitialOnlyRereviewDecision(
        query_group_id=query_group_id,
        dataset=datasets[0],
        action=action,
        reason_bucket=reason_bucket,
        safe_component_key_texts=safe_texts,
        reviewed_row_count=len(rows),
        review_decisions=review_decisions,
        evidence_issues=evidence_issues,
    )


def read_initial_only_rereview_decisions(
    path: Path = DEFAULT_INITIAL_ONLY_REREVIEW_RESULTS_PATH,
) -> dict[str, InitialOnlyRereviewDecision]:
    """Read and conservatively collapse model-visible initial-only reviews."""

    if not path.exists():
        return {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "query_case_id",
            "dataset",
            "active_contract_decision",
            "active_contract_safe_positive_component_keys",
            "evidence_contract_issue",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Initial-only re-review file missing columns: {missing}")
        for row in reader:
            query_group_id = str(row["query_case_id"])
            if not query_group_id.endswith(":initial_only"):
                raise ValueError(f"Unexpected non-initial-only re-review row: {query_group_id!r}")
            grouped[query_group_id].append(dict(row))
    return {
        query_group_id: _collapse_review_rows(query_group_id, rows)
        for query_group_id, rows in sorted(grouped.items())
    }

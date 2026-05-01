"""Canonical label and filter contract helpers for the official safe-link bundle."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.joint_safe_link_initial_only_rereview import (
    InitialOnlyRereviewDecision,
    expand_safe_component_key_texts_for_ledger,
    read_initial_only_rereview_decisions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_ROOT = REPO_ROOT / "data" / "joint_safe_link_official_stack_20260428p"
CONTRACT_RELATIVE_DIR = Path("dataset_contract")
FILTER_POLICY_RELATIVE_PATH = CONTRACT_RELATIVE_DIR / "filter_policy.json"
CUSTOM_LABEL_LEDGER_RELATIVE_PATH = CONTRACT_RELATIVE_DIR / "custom_label_ledger.csv"
NAME_COMPAT_MANUAL_POSITIVE_CORRECTIONS_RELATIVE_PATH = (
    CONTRACT_RELATIVE_DIR / "name_compat_manual_positive_corrections.csv"
)
CUSTOM_LABEL_LEDGER_SUMMARY_RELATIVE_PATH = CONTRACT_RELATIVE_DIR / "custom_label_ledger_summary.json"
CUSTOM_LABEL_LEDGER_COMPARISON_RELATIVE_PATH = CONTRACT_RELATIVE_DIR / "custom_label_ledger_comparison.json"
CUSTOM_LABEL_LEDGER_REPORT_RELATIVE_PATH = CONTRACT_RELATIVE_DIR / "custom_label_ledger_report.md"

LEDGER_COLUMNS = [
    "ledger_source",
    "slice_key",
    "dataset",
    "split",
    "query_group_id",
    "query_view",
    "decision_scope",
    "candidate_component_key",
    "target_label",
    "action",
    "reason_bucket",
    "review_source_path",
    "notes",
]

DEFAULT_FILTER_POLICY = {
    "policy_name": "official_candidate_filter_policy_v1",
    "retrieval_rank_limit": 25,
    "drop_hard_disallow_query_candidate_pairs": True,
    "drop_raw_candidate_components_containing_query_signature": True,
    "derive_query_safe_target_after_filtering": "retrieved_window_max_surviving_candidate_label",
    "target_semantics": "retrieved_window_safe_target",
    "query_level_positive_targets_are_not_authoritative": True,
    "retrieval_misses_are_not_true_negatives": True,
}


def _to_int(value: Any, default: int = 0) -> int:
    """Parse one numeric CSV-ish value as an integer."""

    if value in (None, ""):
        return default
    return int(float(value))


@dataclass(frozen=True)
class CandidateFilterResult:
    """Result of applying one deterministic candidate-row filter."""

    kept_rows: tuple[dict[str, Any], ...]
    dropped_rows: tuple[dict[str, Any], ...]
    reason: str

    @property
    def rows_before(self) -> int:
        """Return input row count."""

        return int(len(self.kept_rows) + len(self.dropped_rows))

    @property
    def rows_after(self) -> int:
        """Return surviving row count."""

        return int(len(self.kept_rows))

    @property
    def positive_rows_before(self) -> int:
        """Return positive input row count."""

        return int(sum(1 for row in (*self.kept_rows, *self.dropped_rows) if _to_int(row.get("label")) == 1))

    @property
    def positive_rows_after(self) -> int:
        """Return positive surviving row count."""

        return int(sum(1 for row in self.kept_rows if _to_int(row.get("label")) == 1))

    @property
    def dropped_candidate_component_keys(self) -> tuple[str, ...]:
        """Return dropped candidate component keys."""

        return tuple(str(row.get("candidate_component_key", "")) for row in self.dropped_rows)

    @property
    def dropped_positive_candidate_component_keys(self) -> tuple[str, ...]:
        """Return dropped candidate component keys that had positive labels."""

        return tuple(
            str(row.get("candidate_component_key", "")) for row in self.dropped_rows if _to_int(row.get("label")) == 1
        )


@dataclass(frozen=True)
class ComponentFilterResult:
    """Result of applying one deterministic component-key filter."""

    kept_component_keys: tuple[str, ...]
    dropped_component_keys: tuple[str, ...]
    reason: str

    @property
    def components_before(self) -> int:
        """Return input component count."""

        return int(len(self.kept_component_keys) + len(self.dropped_component_keys))

    @property
    def components_after(self) -> int:
        """Return surviving component count."""

        return int(len(self.kept_component_keys))


def retrieval_rank_limit_from_policy(policy: dict[str, Any] | None = None) -> int:
    """Return the candidate retrieval-rank limit from one filter policy."""

    active_policy = DEFAULT_FILTER_POLICY if policy is None else policy
    return int(active_policy["retrieval_rank_limit"])


def apply_retrieval_rank_filter(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    retrieval_rank_limit: int | None = None,
) -> CandidateFilterResult:
    """Apply the official retrieval-rank window cap to candidate rows."""

    limit = retrieval_rank_limit_from_policy() if retrieval_rank_limit is None else int(retrieval_rank_limit)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        if _to_int(copied.get("retrieval_rank")) <= limit:
            kept.append(copied)
        else:
            dropped.append(copied)
    return CandidateFilterResult(
        kept_rows=tuple(kept),
        dropped_rows=tuple(dropped),
        reason=f"retrieval_rank_lte_{limit}",
    )


def apply_self_containment_filter(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    contains_query_signature: Callable[[dict[str, Any]], bool],
) -> CandidateFilterResult:
    """Drop rows whose raw candidate component contains the query signature."""

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        if bool(contains_query_signature(copied)):
            dropped.append(copied)
        else:
            kept.append(copied)
    return CandidateFilterResult(
        kept_rows=tuple(kept),
        dropped_rows=tuple(dropped),
        reason="raw_candidate_component_contains_query_signature",
    )


def apply_hard_disallow_component_filter(
    component_keys: list[str] | tuple[str, ...],
    *,
    disallow_pair_count_by_component: dict[str, int],
    preserve_component_keys: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
) -> ComponentFilterResult:
    """Drop components whose query-to-cluster pair set has a hard-disallow pair.

    Known positive components are preserved because the disallow constraint can
    be a name-prefix artifact rather than evidence that the candidate is wrong.
    """

    preserved = frozenset(str(component_key) for component_key in (preserve_component_keys or ()))
    kept: list[str] = []
    dropped: list[str] = []
    for component_key in component_keys:
        key = str(component_key)
        if key in preserved or int(disallow_pair_count_by_component.get(key, 0)) <= 0:
            kept.append(key)
        else:
            dropped.append(key)
    return ComponentFilterResult(
        kept_component_keys=tuple(kept),
        dropped_component_keys=tuple(dropped),
        reason="hard_disallow_query_candidate_pair",
    )


def _read_csv(path: Path, *, sep: str = ",") -> pd.DataFrame:
    """Read a CSV/TSV file with gzip inferred from the suffix."""

    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, sep=sep, compression=compression, low_memory=False, keep_default_na=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _split_component_keys(value: Any) -> list[str]:
    """Split a review field containing zero or more candidate component keys."""

    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _split_review_safe_component_keys(row: dict[str, Any]) -> list[str]:
    """Split safe component keys, preserving singleton keys that contain pipes."""

    text = str(row.get("safe_positive_component_keys", "") or "").strip()
    if not text:
        return []
    candidate_text = str(row.get("candidate_component_keys", "") or "").strip()
    candidate_count_text = str(row.get("candidate_count", "") or "").strip()
    if candidate_text and candidate_count_text:
        try:
            if int(float(candidate_count_text)) == 1 and text == candidate_text:
                return [text]
        except ValueError:
            pass
    return _split_component_keys(text)


def _ledger_row(
    *,
    ledger_source: str,
    slice_key: str,
    dataset: str,
    split: str,
    query_group_id: str,
    query_view: str,
    decision_scope: str,
    candidate_component_key: str = "",
    target_label: int | str = "",
    action: str,
    reason_bucket: str = "",
    review_source_path: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build one normalized custom-label ledger row."""

    return {
        "ledger_source": str(ledger_source),
        "slice_key": str(slice_key),
        "dataset": str(dataset),
        "split": str(split),
        "query_group_id": str(query_group_id),
        "query_view": str(query_view),
        "decision_scope": str(decision_scope),
        "candidate_component_key": str(candidate_component_key),
        "target_label": target_label,
        "action": str(action),
        "reason_bucket": str(reason_bucket),
        "review_source_path": str(review_source_path),
        "notes": str(notes),
    }


def _ledger_rows_from_initial_only_rereview_decision(
    *,
    decision: InitialOnlyRereviewDecision,
    ledger_source: str,
    slice_key: str,
    split: str,
    query_view: str,
    review_source_path: str,
) -> list[dict[str, Any]]:
    """Represent one collapsed model-visible initial-only decision in the ledger."""

    common = {
        "ledger_source": ledger_source,
        "slice_key": slice_key,
        "dataset": decision.dataset,
        "split": split,
        "query_group_id": decision.query_group_id,
        "query_view": query_view,
        "reason_bucket": decision.reason_bucket,
        "review_source_path": review_source_path,
        "notes": (
            f"model_visible_initial_only_rereview rows={decision.reviewed_row_count}; "
            f"decisions={','.join(decision.review_decisions)}; issues={','.join(decision.evidence_issues)}"
        ),
    }
    if decision.action == "candidate_positive":
        return [
            _ledger_row(
                **common,
                decision_scope="candidate",
                candidate_component_key=candidate_key,
                target_label=1,
                action="candidate_positive",
            )
            for candidate_key in expand_safe_component_key_texts_for_ledger(decision.safe_component_key_texts)
        ]
    if decision.action == "force_no_positive":
        return [
            _ledger_row(
                **common,
                decision_scope="query",
                target_label=0,
                action="force_no_positive",
            )
        ]
    if decision.action == "drop_query":
        return [
            _ledger_row(
                **common,
                decision_scope="query",
                target_label="",
                action="drop_query",
            )
        ]
    raise ValueError(f"Unsupported initial-only re-review action: {decision.action!r}")


def _build_hwang_ledger(bundle_root: Path) -> list[dict[str, Any]]:
    """Build ledger rows from the active H-Wang candidate-level manifest."""

    manifest_path = bundle_root / "test" / "hwang_candidate_level_label_overrides.csv"
    manifest = _read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        action = str(row["label_action"])
        candidate_key = str(row.get("reviewed_candidate_component_key", "") or "")
        if action == "add_reviewed_positive":
            decision_scope = "candidate"
            target_label: int | str = 1
            normalized_action = "candidate_positive"
        elif action == "force_no_positive":
            decision_scope = "query"
            target_label = 0
            normalized_action = "force_no_positive"
        else:
            decision_scope = "query"
            target_label = int(row.get("manual_safe_target", 0))
            normalized_action = "query_target_from_surviving_labels"
        rows.append(
            _ledger_row(
                ledger_source="hwang_candidate_level_label_overrides",
                slice_key="hwang_eval",
                dataset="h_wang",
                split="test",
                query_group_id=str(row["query_group_id"]),
                query_view="",
                decision_scope=decision_scope,
                candidate_component_key=candidate_key,
                target_label=target_label,
                action=normalized_action,
                reason_bucket=str(row.get("correction_type", "")),
                review_source_path=str(row.get("review_source_path", "")),
            )
        )
    return rows


def _build_review_ledger_from_frame(
    *,
    reviews: pd.DataFrame,
    ledger_source: str,
    slice_key: str,
    review_path: Path,
    initial_only_decisions: dict[str, InitialOnlyRereviewDecision] | None = None,
) -> list[dict[str, Any]]:
    """Build ledger rows from a manual packet-review frame."""

    active_initial_only_decisions = {} if initial_only_decisions is None else initial_only_decisions
    rows: list[dict[str, Any]] = []
    for row in reviews.to_dict(orient="records"):
        query_group_id = str(row["query_case_id"])
        initial_only_decision = active_initial_only_decisions.get(query_group_id)
        if initial_only_decision is not None:
            rows.extend(
                _ledger_rows_from_initial_only_rereview_decision(
                    decision=initial_only_decision,
                    ledger_source=ledger_source,
                    slice_key=slice_key,
                    split=str(row.get("split", "")),
                    query_view=str(row.get("query_view", "")),
                    review_source_path=str(row.get("review_file", review_path)),
                )
            )
            continue
        candidate_keys = _split_review_safe_component_keys(row)
        correction_type = str(row.get("correction_type", "") or "")
        manual_assessment = str(row.get("manual_assessment", "") or "")
        if manual_assessment == "impossible" or correction_type == "drop_impossible":
            rows.append(
                _ledger_row(
                    ledger_source=ledger_source,
                    slice_key=slice_key,
                    dataset=str(row.get("dataset", "")),
                    split=str(row.get("split", "")),
                    query_group_id=query_group_id,
                    query_view=str(row.get("query_view", "")),
                    decision_scope="query",
                    target_label="",
                    action="drop_query",
                    reason_bucket=str(row.get("reason_bucket", "")),
                    review_source_path=str(row.get("review_file", review_path)),
                    notes=str(row.get("notes", "")),
                )
            )
            continue
        if candidate_keys:
            for candidate_key in candidate_keys:
                rows.append(
                    _ledger_row(
                        ledger_source=ledger_source,
                        slice_key=slice_key,
                        dataset=str(row.get("dataset", "")),
                        split=str(row.get("split", "")),
                        query_group_id=query_group_id,
                        query_view=str(row.get("query_view", "")),
                        decision_scope="candidate",
                        candidate_component_key=candidate_key,
                        target_label=1,
                        action="candidate_positive",
                        reason_bucket=str(row.get("reason_bucket", "")),
                        review_source_path=str(row.get("review_file", review_path)),
                        notes=str(row.get("notes", "")),
                    )
                )
            continue
        rows.append(
            _ledger_row(
                ledger_source=ledger_source,
                slice_key=slice_key,
                dataset=str(row.get("dataset", "")),
                split=str(row.get("split", "")),
                query_group_id=query_group_id,
                query_view=str(row.get("query_view", "")),
                decision_scope="query",
                target_label=0,
                action="force_no_positive",
                reason_bucket=str(row.get("reason_bucket", "")),
                review_source_path=str(row.get("review_file", review_path)),
                notes=str(row.get("notes", "")),
            )
        )
    return rows


def _build_review_ledger(
    *,
    review_path: Path,
    sep: str,
    ledger_source: str,
    slice_key: str,
    initial_only_decisions: dict[str, InitialOnlyRereviewDecision] | None = None,
) -> list[dict[str, Any]]:
    """Build ledger rows from a manual packet-review table."""

    return _build_review_ledger_from_frame(
        reviews=_read_csv(review_path, sep=sep),
        ledger_source=ledger_source,
        slice_key=slice_key,
        review_path=review_path,
        initial_only_decisions=initial_only_decisions,
    )


def _build_s2and_rescue_review_ledger(
    *,
    initial_only_decisions: dict[str, InitialOnlyRereviewDecision] | None = None,
) -> list[dict[str, Any]]:
    """Build ledger rows for the reviewed public-S2AND rescue application."""

    review_path = (
        REPO_ROOT
        / "scratch"
        / "s2and_rescue_manual_review_20260428"
        / "full_queue"
        / "all_reviews_merged.tsv"
    )
    if not review_path.exists():
        return []
    reviews = _read_csv(review_path, sep="\t")
    singleton_review_paths = (
        REPO_ROOT / "scratch" / "singleton_gate_check_review_20260429" / "queue" / "all_reviews_merged.tsv",
        REPO_ROOT / "scratch" / "singleton_gate_check_review_20260429" / "topup_queue" / "all_reviews_merged.tsv",
        REPO_ROOT / "scratch" / "singleton_gate_check_review_20260429" / "topup2_queue" / "all_reviews_merged.tsv",
    )
    singleton_override_query_ids: set[str] = set()
    for singleton_review_path in singleton_review_paths:
        if singleton_review_path.exists():
            singleton_reviews = _read_csv(singleton_review_path, sep="\t")
            singleton_override_query_ids.update(singleton_reviews["query_case_id"].astype(str))
    if singleton_override_query_ids:
        reviews = reviews[~reviews["query_case_id"].astype(str).isin(singleton_override_query_ids)].copy()
    rows: list[dict[str, Any]] = []
    split_text = reviews["split"].astype(str)
    rows.extend(
        _build_review_ledger_from_frame(
            reviews=reviews[split_text == "train"].copy(),
            ledger_source="s2and_rescue_manual_review",
            slice_key="s2and_rescue_reviewed_train",
            review_path=review_path,
            initial_only_decisions=initial_only_decisions,
        )
    )
    rows.extend(
        _build_review_ledger_from_frame(
            reviews=reviews[split_text != "train"].copy(),
            ledger_source="s2and_rescue_manual_review",
            slice_key="s2and_rescue_reviewed_eval",
            review_path=review_path,
            initial_only_decisions=initial_only_decisions,
        )
    )
    for singleton_review_path in singleton_review_paths:
        if not singleton_review_path.exists():
            continue
        rows.extend(
            _build_review_ledger(
                review_path=singleton_review_path,
                sep="\t",
                ledger_source="singleton_gate_check_review_20260429",
                slice_key="s2and_rescue_reviewed_eval",
                initial_only_decisions=None,
            )
        )
    return rows


def _build_singleton_repair_ledger(bundle_root: Path) -> list[dict[str, Any]]:
    """Build ledger rows from the singleton repair and quarantine artifacts."""

    manifest_path = bundle_root / "training" / "singleton_near_distance_repair_manifest.csv"
    quarantine_path = bundle_root / "training" / "singleton_near_distance_quarantined_query_groups.txt"
    manifest = _read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    for row in manifest.to_dict(orient="records"):
        repair_action = str(row["repair_action"])
        action = "candidate_positive" if repair_action == "auto_flip_candidate" else "candidate_negative"
        target_label: int | str = 1 if repair_action == "auto_flip_candidate" else 0
        rows.append(
            _ledger_row(
                ledger_source="singleton_near_distance_repair_manifest",
                slice_key="training_singleton_repair",
                dataset=str(row.get("dataset", "")),
                split="training",
                query_group_id=str(row["query_group_id"]),
                query_view="",
                decision_scope="candidate",
                candidate_component_key=str(row.get("candidate_component_key", "") or ""),
                target_label=target_label,
                action=action,
                reason_bucket=repair_action,
                review_source_path=str(manifest_path.relative_to(REPO_ROOT)),
            )
        )
    if quarantine_path.exists():
        for query_group_id in quarantine_path.read_text(encoding="utf-8").splitlines():
            query_group_id = query_group_id.strip()
            if not query_group_id:
                continue
            rows.append(
                _ledger_row(
                    ledger_source="singleton_near_distance_quarantined_query_groups",
                    slice_key="training_singleton_repair",
                    dataset=query_group_id.split(":", 1)[0],
                    split="training",
                    query_group_id=query_group_id,
                    query_view="",
                    decision_scope="query",
                    target_label="",
                    action="quarantine_query",
                    reason_bucket="quarantined_query_group",
                    review_source_path=str(quarantine_path.relative_to(REPO_ROOT)),
                )
            )
    return rows


def _build_active_row_label_ledger(
    *,
    bundle_root: Path,
    relative_path: Path,
    ledger_source: str,
    slice_key: str,
) -> list[dict[str, Any]]:
    """Represent active reviewed row labels as candidate/query ledger decisions."""

    row_path = bundle_root / relative_path
    usecols = ["dataset", "split", "query_group_id", "query_view", "candidate_component_key", "label"]
    labels = _read_csv(row_path)[usecols].copy()
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce").fillna(0).astype(int)
    rows: list[dict[str, Any]] = []
    for query_group_id, group in labels.groupby(labels["query_group_id"].astype(str), sort=False):
        metadata = group.iloc[0]
        positive_rows = group[group["label"] > 0].copy()
        if positive_rows.empty:
            rows.append(
                _ledger_row(
                    ledger_source=ledger_source,
                    slice_key=slice_key,
                    dataset=str(metadata.get("dataset", "")),
                    split=str(metadata.get("split", "")),
                    query_group_id=str(query_group_id),
                    query_view=str(metadata.get("query_view", "")),
                    decision_scope="query",
                    target_label=0,
                    action="force_no_positive",
                    reason_bucket="active_reviewed_row_labels",
                    review_source_path=str(relative_path).replace("/", "\\"),
                )
            )
            continue
        for row in positive_rows.to_dict(orient="records"):
            rows.append(
                _ledger_row(
                    ledger_source=ledger_source,
                    slice_key=slice_key,
                    dataset=str(row.get("dataset", "")),
                    split=str(row.get("split", "")),
                    query_group_id=str(query_group_id),
                    query_view=str(row.get("query_view", "")),
                    decision_scope="candidate",
                    candidate_component_key=str(row.get("candidate_component_key", "") or ""),
                    target_label=1,
                    action="candidate_positive",
                    reason_bucket="active_reviewed_row_labels",
                    review_source_path=str(relative_path).replace("/", "\\"),
                )
            )
    return rows


def _build_strict_singleton_manual_review_ledger() -> list[dict[str, Any]]:
    """Build ledger rows for the manually reviewed strict singleton training rows."""

    review_path = (
        REPO_ROOT
        / "scratch"
        / "singleton_gate_check_review_20260429"
        / "strict_s2and_singleton_neg_train"
        / "full_manual_review"
        / "review.tsv"
    )
    if not review_path.exists():
        return []
    reviews = _read_csv(review_path, sep="\t")
    reviews = reviews[reviews["manual_label"].astype(str).isin({"positive", "negative"})].copy()
    relative_path = review_path.relative_to(REPO_ROOT)
    rows: list[dict[str, Any]] = []
    for row in reviews.to_dict(orient="records"):
        manual_label = str(row["manual_label"])
        is_positive = manual_label == "positive"
        rows.append(
            _ledger_row(
                ledger_source="s2and_singleton_manual_review_20260429",
                slice_key="s2and_singleton_reviewed_train",
                dataset=str(row.get("dataset", "")),
                split="train",
                query_group_id=str(row["query_group_id"]),
                query_view=str(row.get("query_view", "")),
                decision_scope="candidate" if is_positive else "query",
                candidate_component_key=str(row.get("candidate_component_key", "") or "") if is_positive else "",
                target_label=1 if is_positive else 0,
                action="candidate_positive" if is_positive else "force_no_positive",
                reason_bucket=f"manual_{manual_label}",
                review_source_path=str(relative_path).replace("/", "\\"),
                notes=str(row.get("notes", "") or ""),
            )
        )
    return rows


def _build_name_compat_manual_positive_correction_ledger(bundle_root: Path) -> list[dict[str, Any]]:
    """Read manually verified positives surfaced by name-compatible retrieval."""

    correction_path = bundle_root / NAME_COMPAT_MANUAL_POSITIVE_CORRECTIONS_RELATIVE_PATH
    if not correction_path.exists():
        return []
    corrections = _read_csv(correction_path)
    missing_columns = [column for column in LEDGER_COLUMNS if column not in corrections.columns]
    if missing_columns:
        raise ValueError(
            f"{NAME_COMPAT_MANUAL_POSITIVE_CORRECTIONS_RELATIVE_PATH} is missing columns: {missing_columns}"
        )
    rows: list[dict[str, Any]] = []
    for raw_row in corrections.to_dict(orient="records"):
        row = {column: raw_row.get(column, "") for column in LEDGER_COLUMNS}
        if str(row["action"]) != "candidate_positive":
            raise ValueError(
                "Name-compatible manual corrections must use action=candidate_positive; "
                f"got {row['action']!r} for {row['query_group_id']!r}."
            )
        if str(row["decision_scope"]) != "candidate":
            raise ValueError(
                "Name-compatible manual corrections must use decision_scope=candidate; "
                f"got {row['decision_scope']!r} for {row['query_group_id']!r}."
            )
        if not str(row["candidate_component_key"]):
            raise ValueError(f"Name-compatible manual correction lacks a candidate key: {row['query_group_id']!r}.")
        if _to_int(row["target_label"]) != 1:
            raise ValueError(
                "Name-compatible manual corrections must use target_label=1; "
                f"got {row['target_label']!r} for {row['query_group_id']!r}."
            )
        rows.append(
            _ledger_row(
                ledger_source=str(row["ledger_source"]),
                slice_key=str(row["slice_key"]),
                dataset=str(row["dataset"]),
                split=str(row["split"]),
                query_group_id=str(row["query_group_id"]),
                query_view=str(row["query_view"]),
                decision_scope="candidate",
                candidate_component_key=str(row["candidate_component_key"]),
                target_label=1,
                action="candidate_positive",
                reason_bucket=str(row["reason_bucket"]),
                review_source_path=str(row["review_source_path"]),
                notes=str(row["notes"]),
            )
        )
    return rows


def _apply_name_compat_manual_positive_corrections(
    rows: list[dict[str, Any]], correction_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace stale active decisions with verified name-compatible candidate positives."""

    if not correction_rows:
        return rows
    corrected_query_keys = {
        (str(row["slice_key"]), str(row["query_group_id"]))
        for row in correction_rows
        if str(row.get("action", "")) == "candidate_positive"
    }
    corrected_candidate_keys = {
        (str(row["slice_key"]), str(row["query_group_id"]), str(row["candidate_component_key"]))
        for row in correction_rows
        if str(row.get("action", "")) == "candidate_positive" and str(row.get("candidate_component_key", ""))
    }
    retained_rows = [
        row
        for row in rows
        if not (
            str(row.get("decision_scope", "")) == "query"
            and str(row.get("action", "")) in {"force_no_positive", "query_target_from_surviving_labels"}
            and (str(row.get("slice_key", "")), str(row.get("query_group_id", ""))) in corrected_query_keys
        )
        and not (
            str(row.get("action", "")) == "candidate_positive"
            and (
                str(row.get("slice_key", "")),
                str(row.get("query_group_id", "")),
                str(row.get("candidate_component_key", "")),
            )
            in corrected_candidate_keys
        )
    ]
    return [*retained_rows, *correction_rows]


def build_custom_label_ledger(bundle_root: Path = DEFAULT_BUNDLE_ROOT) -> pd.DataFrame:
    """Build the canonical custom-label ledger from current official artifacts."""

    bundle_root = bundle_root.resolve()
    initial_only_decisions = read_initial_only_rereview_decisions()
    rows: list[dict[str, Any]] = []
    rows.extend(
        _build_active_row_label_ledger(
            bundle_root=bundle_root,
            relative_path=Path("calibration") / "classic_gate_possible_manual_w5_rows.csv.gz",
            ledger_source="new_block_manual_calibration_source_active_labels",
            slice_key="new_block_calibration_source",
        )
    )
    rows.extend(_build_hwang_ledger(bundle_root))
    for slice_key, relative_path in {
        "s_park_eval": Path("test") / "s_park_eval_rows.csv.gz",
        "s_lee_eval": Path("test") / "s_lee_eval_rows.csv.gz",
        "j_smith_eval": Path("test") / "j_smith_eval_rows.csv.gz",
        "a_khan_eval": Path("test") / "a_khan_eval_rows.csv.gz",
        "a_silva_eval": Path("test") / "a_silva_eval_rows.csv.gz",
        "s_gupta_eval": Path("test") / "s_gupta_eval_rows.csv.gz",
    }.items():
        rows.extend(
            _build_active_row_label_ledger(
                bundle_root=bundle_root,
                relative_path=relative_path,
                ledger_source=f"{slice_key}_active_labels",
                slice_key=slice_key,
            )
        )
    rows.extend(
        _build_review_ledger(
            review_path=REPO_ROOT / "scratch" / "s2and_full_relabel_20260424" / "all_reviews_merged.tsv",
            sep="\t",
            ledger_source="s2and_full_relabel_reviews",
            slice_key="s2and_eval",
            initial_only_decisions=initial_only_decisions,
        )
    )
    rows.extend(
        _build_review_ledger(
            review_path=REPO_ROOT
            / "scratch"
            / "training_s2and_source_no_positive_review_20260424"
            / "all_reviews_merged.tsv",
            sep="\t",
            ledger_source="training_s2and_source_review",
            slice_key="training_s2and_source_reviewed_eval",
            initial_only_decisions=initial_only_decisions,
        )
    )
    rows.extend(
        _build_review_ledger(
            review_path=REPO_ROOT / "scratch" / "extra_s2and_no_positive_review_20260424" / "reviews" / "manual.tsv",
            sep="\t",
            ledger_source="extra_s2and_no_positive_review",
            slice_key="s2and_extra_no_positive_eval",
            initial_only_decisions=initial_only_decisions,
        )
    )
    rows.extend(_build_s2and_rescue_review_ledger(initial_only_decisions=initial_only_decisions))
    rows.extend(_build_singleton_repair_ledger(bundle_root))
    rows.extend(_build_strict_singleton_manual_review_ledger())
    rows = _apply_name_compat_manual_positive_corrections(
        rows,
        _build_name_compat_manual_positive_correction_ledger(bundle_root),
    )
    ledger = pd.DataFrame(rows, columns=LEDGER_COLUMNS).fillna("")
    return ledger.sort_values(
        ["slice_key", "query_group_id", "decision_scope", "candidate_component_key", "action"],
        kind="stable",
    ).reset_index(drop=True)


def _row_label_index(path: Path) -> pd.DataFrame:
    """Return the row-level label index for one active candidate row file."""

    usecols = ["query_group_id", "candidate_component_key", "label"]
    rows = _read_csv(path)[usecols].copy()
    rows["query_group_id"] = rows["query_group_id"].astype(str)
    rows["candidate_component_key"] = rows["candidate_component_key"].astype(str)
    rows["label"] = pd.to_numeric(rows["label"], errors="coerce").fillna(0).astype(int)
    return rows


def _summarize_row_labels(rows: pd.DataFrame) -> dict[str, Any]:
    """Summarize active row labels for one slice."""

    query_targets = rows.groupby("query_group_id")["label"].max()
    return {
        "rows": int(len(rows)),
        "queries": int(rows["query_group_id"].nunique()),
        "positive_rows": int(rows["label"].sum()),
        "positive_queries": int(query_targets.sum()),
        "no_positive_queries": int((query_targets == 0).sum()),
    }


def _compare_ledger_slice(
    *,
    ledger: pd.DataFrame,
    slice_key: str,
    active_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Compare one ledger slice against the current active candidate rows."""

    slice_ledger = ledger[ledger["slice_key"].astype(str) == slice_key].copy()
    row_index = active_rows.set_index(["query_group_id", "candidate_component_key"])["label"]
    query_targets = active_rows.groupby("query_group_id")["label"].max()
    active_queries = set(query_targets.index.astype(str))
    candidate_positive_checks = 0
    candidate_positive_mismatches: list[dict[str, str]] = []
    candidate_positive_missing_active_query = 0
    candidate_positive_missing_filtered_query = 0
    candidate_negative_checks = 0
    candidate_negative_mismatches: list[dict[str, str]] = []
    candidate_negative_missing_active_query = 0
    force_no_positive_checks = 0
    force_no_positive_mismatches: list[str] = []
    query_target_checks = 0
    query_target_mismatches: list[dict[str, Any]] = []
    dropped_query_checks = 0
    dropped_query_still_active = 0
    quarantine_query_checks = 0
    quarantine_query_still_active = 0
    for row in slice_ledger.to_dict(orient="records"):
        action = str(row["action"])
        query_group_id = str(row["query_group_id"])
        candidate_key = str(row.get("candidate_component_key", "") or "")
        if action == "candidate_positive":
            candidate_positive_checks += 1
            key = (query_group_id, candidate_key)
            if key in row_index.index:
                if int(row_index.loc[key]) != 1:
                    candidate_positive_mismatches.append(
                        {"query_group_id": query_group_id, "candidate_component_key": candidate_key}
                    )
            elif query_group_id in active_queries:
                candidate_positive_missing_active_query += 1
            else:
                candidate_positive_missing_filtered_query += 1
        elif action == "candidate_negative":
            candidate_negative_checks += 1
            key = (query_group_id, candidate_key)
            if key in row_index.index:
                if int(row_index.loc[key]) != 0:
                    candidate_negative_mismatches.append(
                        {"query_group_id": query_group_id, "candidate_component_key": candidate_key}
                    )
            elif query_group_id in active_queries:
                candidate_negative_missing_active_query += 1
        elif action == "force_no_positive":
            if query_group_id not in active_queries:
                continue
            force_no_positive_checks += 1
            if int(query_targets.loc[query_group_id]) != 0:
                force_no_positive_mismatches.append(query_group_id)
        elif action == "query_target_from_surviving_labels":
            if query_group_id not in active_queries:
                continue
            query_target_checks += 1
            expected_target = int(row.get("target_label") or 0)
            actual_target = int(query_targets.loc[query_group_id])
            if actual_target != expected_target:
                query_target_mismatches.append(
                    {
                        "query_group_id": query_group_id,
                        "expected_target": expected_target,
                        "actual_target": actual_target,
                    }
                )
        elif action == "drop_query":
            dropped_query_checks += 1
            if query_group_id in active_queries:
                dropped_query_still_active += 1
        elif action == "quarantine_query":
            quarantine_query_checks += 1
            if query_group_id in active_queries:
                quarantine_query_still_active += 1

    mismatches = {
        "candidate_positive_wrong_label": candidate_positive_mismatches[:20],
        "candidate_negative_wrong_label": candidate_negative_mismatches[:20],
        "force_no_positive_nonzero_target": force_no_positive_mismatches[:20],
        "query_target_from_surviving_labels_mismatch": query_target_mismatches[:20],
        "drop_query_still_active": int(dropped_query_still_active),
        "quarantine_query_still_active": int(quarantine_query_still_active),
    }
    fatal_mismatch_count = (
        len(candidate_positive_mismatches)
        + len(candidate_negative_mismatches)
        + len(force_no_positive_mismatches)
        + len(query_target_mismatches)
        + int(dropped_query_still_active)
        + int(quarantine_query_still_active)
    )
    return {
        "slice_key": slice_key,
        "ledger_rows": int(len(slice_ledger)),
        "ledger_action_counts": {
            str(key): int(value) for key, value in slice_ledger["action"].value_counts().sort_index().items()
        },
        "active_rows": _summarize_row_labels(active_rows),
        "candidate_positive_checks": int(candidate_positive_checks),
        "candidate_positive_missing_active_query": int(candidate_positive_missing_active_query),
        "candidate_positive_missing_filtered_query": int(candidate_positive_missing_filtered_query),
        "candidate_negative_checks": int(candidate_negative_checks),
        "candidate_negative_missing_active_query": int(candidate_negative_missing_active_query),
        "force_no_positive_checks": int(force_no_positive_checks),
        "query_target_checks": int(query_target_checks),
        "dropped_query_checks": int(dropped_query_checks),
        "quarantine_query_checks": int(quarantine_query_checks),
        "fatal_mismatch_count": int(fatal_mismatch_count),
        "mismatches": mismatches,
    }


def compare_ledger_to_current_bundle(
    ledger: pd.DataFrame,
    *,
    bundle_root: Path = DEFAULT_BUNDLE_ROOT,
) -> dict[str, Any]:
    """Compare custom-label ledger decisions with the current official active rows."""

    bundle_root = bundle_root.resolve()
    slice_paths = {
        "new_block_calibration_source": bundle_root / "calibration" / "classic_gate_possible_manual_w5_rows.csv.gz",
        "hwang_eval": bundle_root / "test" / "hwang_eval_rows.csv.gz",
        "s_park_eval": bundle_root / "test" / "s_park_eval_rows.csv.gz",
        "s_lee_eval": bundle_root / "test" / "s_lee_eval_rows.csv.gz",
        "j_smith_eval": bundle_root / "test" / "j_smith_eval_rows.csv.gz",
        "a_khan_eval": bundle_root / "test" / "a_khan_eval_rows.csv.gz",
        "a_silva_eval": bundle_root / "test" / "a_silva_eval_rows.csv.gz",
        "s_gupta_eval": bundle_root / "test" / "s_gupta_eval_rows.csv.gz",
        "s2and_eval": bundle_root / "test" / "s2and_eval_rows.csv.gz",
        "training_s2and_source_reviewed_eval": bundle_root / "test" / "training_s2and_source_reviewed_eval_rows.csv.gz",
        "s2and_extra_no_positive_eval": bundle_root / "test" / "s2and_extra_no_positive_eval_rows.csv.gz",
        "s2and_rescue_reviewed_eval": bundle_root / "test" / "s2and_rescue_reviewed_eval_rows.csv.gz",
        "s2and_rescue_reviewed_train": bundle_root
        / "training"
        / (
            "classic_train_union21_plus_s_lee_raw_plus_public_loo_q100_seed71_"
            "neg100_plus_reviewed_splitpos_hardneg_rows.csv.gz"
        ),
        "s2and_singleton_reviewed_train": bundle_root
        / "training"
        / (
            "classic_train_union21_plus_s_lee_raw_plus_public_loo_q100_seed71_"
            "neg100_plus_reviewed_splitpos_hardneg_rows.csv.gz"
        ),
        "training_singleton_repair": bundle_root
        / "training"
        / (
            "classic_train_union21_plus_s_lee_raw_plus_public_loo_q100_seed71_"
            "neg100_plus_reviewed_splitpos_hardneg_rows.csv.gz"
        ),
    }
    slices = {
        slice_key: _compare_ledger_slice(
            ledger=ledger,
            slice_key=slice_key,
            active_rows=_row_label_index(path),
        )
        for slice_key, path in slice_paths.items()
    }
    return {
        "bundle_root": str(bundle_root.relative_to(REPO_ROOT)),
        "filter_policy": DEFAULT_FILTER_POLICY,
        "slices": slices,
        "fatal_mismatch_count": int(sum(int(payload["fatal_mismatch_count"]) for payload in slices.values())),
    }


def summarize_ledger(ledger: pd.DataFrame, comparison: dict[str, Any]) -> dict[str, Any]:
    """Summarize the canonical custom-label ledger."""

    return {
        "ledger_rows": int(len(ledger)),
        "slice_counts": {
            str(key): int(value) for key, value in ledger["slice_key"].value_counts().sort_index().items()
        },
        "action_counts": {str(key): int(value) for key, value in ledger["action"].value_counts().sort_index().items()},
        "comparison_fatal_mismatch_count": int(comparison["fatal_mismatch_count"]),
    }


def format_comparison_report(summary: dict[str, Any], comparison: dict[str, Any]) -> str:
    """Format a concise Markdown comparison report."""

    lines = [
        "# Custom Label Ledger Comparison",
        "",
        f"- ledger rows: `{summary['ledger_rows']}`",
        f"- fatal mismatches vs current bundle: `{comparison['fatal_mismatch_count']}`",
        "",
        (
            "| slice | ledger rows | active queries | active positive rows | candidate-positive checks | "
            "missing positive in active query | force-no-positive checks | fatal mismatches |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for slice_key, payload in sorted(comparison["slices"].items()):
        active = payload["active_rows"]
        lines.append(
            "| "
            f"{slice_key} | "
            f"{int(payload['ledger_rows'])} | "
            f"{int(active['queries'])} | "
            f"{int(active['positive_rows'])} | "
            f"{int(payload['candidate_positive_checks'])} | "
            f"{int(payload['candidate_positive_missing_active_query'])} | "
            f"{int(payload['force_no_positive_checks'])} | "
            f"{int(payload['fatal_mismatch_count'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `fatal mismatches` must be zero before a slice is considered migrated into the centralized contract.",
            (
                "- `missing positive in active query` is reported but not fatal in this stage because it can occur "
                "when a reviewed positive candidate was removed by the current candidate filter and the active query "
                "target is now no-positive."
            ),
            (
                "- Query-level safe targets are derived from surviving candidate labels; the ledger does not trust "
                "standalone query-level positive labels."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_contract_artifacts(bundle_root: Path = DEFAULT_BUNDLE_ROOT) -> dict[str, Any]:
    """Build and write the official dataset contract artifacts."""

    bundle_root = bundle_root.resolve()
    contract_dir = bundle_root / CONTRACT_RELATIVE_DIR
    contract_dir.mkdir(parents=True, exist_ok=True)
    ledger = build_custom_label_ledger(bundle_root)
    comparison = compare_ledger_to_current_bundle(ledger, bundle_root=bundle_root)
    summary = summarize_ledger(ledger, comparison)

    ledger.to_csv(bundle_root / CUSTOM_LABEL_LEDGER_RELATIVE_PATH, index=False)
    _write_json(bundle_root / FILTER_POLICY_RELATIVE_PATH, DEFAULT_FILTER_POLICY)
    _write_json(bundle_root / CUSTOM_LABEL_LEDGER_SUMMARY_RELATIVE_PATH, summary)
    _write_json(bundle_root / CUSTOM_LABEL_LEDGER_COMPARISON_RELATIVE_PATH, comparison)
    (bundle_root / CUSTOM_LABEL_LEDGER_REPORT_RELATIVE_PATH).write_text(
        format_comparison_report(summary, comparison),
        encoding="utf-8",
    )
    return {
        "filter_policy_path": str(FILTER_POLICY_RELATIVE_PATH).replace("/", "\\"),
        "custom_label_ledger_path": str(CUSTOM_LABEL_LEDGER_RELATIVE_PATH).replace("/", "\\"),
        "custom_label_ledger_summary_path": str(CUSTOM_LABEL_LEDGER_SUMMARY_RELATIVE_PATH).replace("/", "\\"),
        "custom_label_ledger_comparison_path": str(CUSTOM_LABEL_LEDGER_COMPARISON_RELATIVE_PATH).replace("/", "\\"),
        "custom_label_ledger_report_path": str(CUSTOM_LABEL_LEDGER_REPORT_RELATIVE_PATH).replace("/", "\\"),
        **summary,
    }


def refresh_contract_comparison_from_ledger(bundle_root: Path = DEFAULT_BUNDLE_ROOT) -> dict[str, Any]:
    """Refresh contract summaries from the bundled ledger without historical scratch inputs."""

    bundle_root = bundle_root.resolve()
    ledger_path = bundle_root / CUSTOM_LABEL_LEDGER_RELATIVE_PATH
    if not ledger_path.exists():
        raise FileNotFoundError(f"Missing custom label ledger: {ledger_path}")
    ledger = pd.read_csv(ledger_path, low_memory=False, keep_default_na=False)
    comparison = compare_ledger_to_current_bundle(ledger, bundle_root=bundle_root)
    summary = summarize_ledger(ledger, comparison)
    _write_json(bundle_root / FILTER_POLICY_RELATIVE_PATH, DEFAULT_FILTER_POLICY)
    _write_json(bundle_root / CUSTOM_LABEL_LEDGER_SUMMARY_RELATIVE_PATH, summary)
    _write_json(bundle_root / CUSTOM_LABEL_LEDGER_COMPARISON_RELATIVE_PATH, comparison)
    (bundle_root / CUSTOM_LABEL_LEDGER_REPORT_RELATIVE_PATH).write_text(
        format_comparison_report(summary, comparison),
        encoding="utf-8",
    )
    return {
        "filter_policy_path": str(FILTER_POLICY_RELATIVE_PATH).replace("/", "\\"),
        "custom_label_ledger_path": str(CUSTOM_LABEL_LEDGER_RELATIVE_PATH).replace("/", "\\"),
        "custom_label_ledger_summary_path": str(CUSTOM_LABEL_LEDGER_SUMMARY_RELATIVE_PATH).replace("/", "\\"),
        "custom_label_ledger_comparison_path": str(CUSTOM_LABEL_LEDGER_COMPARISON_RELATIVE_PATH).replace("/", "\\"),
        "custom_label_ledger_report_path": str(CUSTOM_LABEL_LEDGER_REPORT_RELATIVE_PATH).replace("/", "\\"),
        **summary,
    }

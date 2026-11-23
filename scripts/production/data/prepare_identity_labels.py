"""Join verified Gold ORCID labels onto Arrow block signatures with strict audits."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.text import canonicalize_name_parts, normalize_orcid  # noqa: E402
from scripts.production.counts.generate_orcid_name_prefix_counts import EXCLUDED_ORCID_GROUPS  # noqa: E402

_WHITESPACE_OR_UNDERSCORE = re.compile(r"[\s_]+")


def normalize_block_key(value: str | None) -> str:
    """Normalize block keys for comparison across underscore and space conventions."""
    if not value:
        return ""
    return _WHITESPACE_OR_UNDERSCORE.sub(" ", str(value).strip().lower())


def names_are_compatible(
    sig_first: str | None,
    sig_middle: str | None,
    sig_last: str | None,
    gold_first: str | None,
    gold_middle: str | None,
    gold_last: str | None,
) -> bool:
    """Return whether a signature name and gold ORCID author name are canonically compatible."""
    sig = canonicalize_name_parts(sig_first, sig_middle, sig_last)
    gold = canonicalize_name_parts(gold_first, gold_middle, gold_last)
    if sig.last and gold.last and sig.last != gold.last:
        return False
    if sig.first and gold.first:
        if sig.first == gold.first:
            return True
        if len(sig.first) == 1 and gold.first.startswith(sig.first):
            return True
        if len(gold.first) == 1 and sig.first.startswith(gold.first):
            return True
        return False
    return True


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _read_arrow_signatures(arrow_dir: Path) -> pd.DataFrame:
    sig_path = arrow_dir / "signatures.arrow"
    if not sig_path.is_file():
        raise FileNotFoundError(sig_path)
    with pa.memory_map(str(sig_path), "r") as source:
        table = ipc.open_file(source).read_all()
    want = [
        col
        for col in (
            "signature_id",
            "paper_id",
            "author_position",
            "author_block",
            "author_first",
            "author_middle",
            "author_last",
        )
        if col in table.column_names
    ]
    missing = sorted({"signature_id", "paper_id", "author_position", "author_block"} - set(want))
    if missing:
        raise ValueError(f"Arrow signatures table missing required columns: {missing}")
    return table.select(want).to_pandas()


def prepare_identity_labels(
    arrow_dataset_dir: Path,
    gold_parquet_path: Path,
    output_parquet_path: Path,
    *,
    dataset_name: str | None = None,
    audit_json_path: Path | None = None,
    error_signature_ids: Collection[str] = (),
) -> dict[str, Any]:
    """Verify and join Gold ORCID rows onto one Arrow block dataset.

    Enforces:
    - ``scored == True`` when a ``scored`` column is present
    - normalized valid ORCID outside ``EXCLUDED_ORCID_GROUPS``
    - unique signature per ``(paper_id, author_position)`` and per ``(paper_id, author_block)``
    - ``cluster_block_key`` match when present in gold
    - canonical first/last name compatibility when gold author name columns are present
    - exclusion of any reviewed wrong-ORCID ``error_signature_ids``
    """
    arrow_dir = arrow_dataset_dir.resolve()
    gold_path = gold_parquet_path.resolve()
    out_path = output_parquet_path.resolve()
    dataset = dataset_name or arrow_dir.name
    error_set = {str(sid) for sid in error_signature_ids}

    sigs = _read_arrow_signatures(arrow_dir)
    sigs["signature_id"] = sigs["signature_id"].astype(str)
    sigs["paper_id_norm"] = sigs["paper_id"].astype(str).str.strip()
    sigs["pos_norm"] = pd.to_numeric(sigs["author_position"], errors="coerce").astype("Int64")
    sigs["block_norm"] = sigs["author_block"].map(normalize_block_key)

    # Identify unique mentions per (paper_id, author_block) and per (paper_id, author_position).
    paper_block_counts = Counter(zip(sigs["paper_id_norm"], sigs["block_norm"], strict=True))
    paper_pos_counts = Counter(
        (pid, int(pos)) for pid, pos in zip(sigs["paper_id_norm"], sigs["pos_norm"], strict=True) if pd.notna(pos)
    )
    sig_by_paper_pos: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sigs.to_dict(orient="records"):
        pos = row["pos_norm"]
        if pd.isna(pos):
            continue
        key = (str(row["paper_id_norm"]), int(pos))
        if paper_pos_counts[key] == 1:
            sig_by_paper_pos[key] = row

    gold = pd.read_parquet(gold_path)
    paper_col = _pick_column(gold, ("corpus_paper_id", "paper_id"))
    pos_col = _pick_column(gold, ("position", "author_position"))
    orcid_col = _pick_column(gold, ("orcid", "author_orcid"))
    if paper_col is None or pos_col is None or orcid_col is None:
        raise ValueError(
            f"Gold parquet must contain paper_id/corpus_paper_id, position/author_position, and orcid: "
            f"found={list(gold.columns)}"
        )
    scored_col = _pick_column(gold, ("scored",))
    block_col = _pick_column(gold, ("cluster_block_key", "block_key", "author_block"))
    first_col = _pick_column(gold, ("author_first", "first"))
    middle_col = _pick_column(gold, ("author_middle", "middle"))
    last_col = _pick_column(gold, ("author_last", "last"))

    audit: dict[str, Any] = {
        "dataset": dataset,
        "arrow_dataset_dir": str(arrow_dir),
        "gold_parquet_path": str(gold_path),
        "arrow_signature_count": int(len(sigs)),
        "gold_rows_total": int(len(gold)),
        "dropped_unscored": 0,
        "dropped_invalid_or_missing_orcid": 0,
        "dropped_excluded_orcid_group": 0,
        "dropped_unmatched_paper_position": 0,
        "dropped_ambiguous_arrow_position": 0,
        "dropped_ambiguous_paper_block": 0,
        "dropped_block_key_mismatch": 0,
        "dropped_name_mismatch": 0,
        "dropped_reviewed_error_signature": 0,
        "dropped_conflicting_gold_rows": 0,
        "verified_rows": 0,
        "unique_orcids": 0,
        "repeated_orcids": 0,
        "repeated_orcid_signatures": 0,
    }

    # First pass: collect candidate verified assignments per signature_id to detect gold conflicts.
    by_signature: dict[str, list[dict[str, Any]]] = {}
    for record in gold.to_dict(orient="records"):
        if scored_col is not None and not bool(record.get(scored_col)):
            audit["dropped_unscored"] += 1
            continue
        raw_orcid = _optional_str(record.get(orcid_col))
        orcid = normalize_orcid(raw_orcid) if raw_orcid else None
        if not orcid:
            audit["dropped_invalid_or_missing_orcid"] += 1
            continue
        if orcid in EXCLUDED_ORCID_GROUPS:
            audit["dropped_excluded_orcid_group"] += 1
            continue
        pid = _optional_str(record.get(paper_col))
        pos = _optional_int(record.get(pos_col))
        if pid is None or pos is None:
            audit["dropped_unmatched_paper_position"] += 1
            continue
        key = (pid, pos)
        if paper_pos_counts.get(key, 0) > 1:
            audit["dropped_ambiguous_arrow_position"] += 1
            continue
        sig_row = sig_by_paper_pos.get(key)
        if sig_row is None:
            audit["dropped_unmatched_paper_position"] += 1
            continue
        if paper_block_counts.get((pid, str(sig_row["block_norm"])), 0) > 1:
            audit["dropped_ambiguous_paper_block"] += 1
            continue
        if block_col is not None:
            gold_block = normalize_block_key(_optional_str(record.get(block_col)))
            if gold_block and gold_block != str(sig_row["block_norm"]):
                audit["dropped_block_key_mismatch"] += 1
                continue
        if first_col is not None or last_col is not None:
            if not names_are_compatible(
                _optional_str(sig_row.get("author_first")),
                _optional_str(sig_row.get("author_middle")),
                _optional_str(sig_row.get("author_last")),
                _optional_str(record.get(first_col)) if first_col else None,
                _optional_str(record.get(middle_col)) if middle_col else None,
                _optional_str(record.get(last_col)) if last_col else None,
            ):
                audit["dropped_name_mismatch"] += 1
                continue
        sid = str(sig_row["signature_id"])
        if sid in error_set:
            audit["dropped_reviewed_error_signature"] += 1
            continue
        by_signature.setdefault(sid, []).append(
            {
                "signature_id": sid,
                "orcid": orcid,
                "paper_id": pid,
                "position": pos,
                "author_block": str(sig_row["author_block"]),
                "dataset": dataset,
            }
        )

    verified: list[dict[str, Any]] = []
    for sid in sorted(by_signature):
        entries = by_signature[sid]
        distinct_orcids = {entry["orcid"] for entry in entries}
        if len(distinct_orcids) > 1:
            audit["dropped_conflicting_gold_rows"] += len(entries)
            continue
        verified.append(entries[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_frame = pd.DataFrame(
        verified,
        columns=pd.Index(["signature_id", "orcid", "paper_id", "position", "author_block", "dataset"]),
    )
    out_frame.to_parquet(out_path, index=False)

    orcid_counts = Counter(row["orcid"] for row in verified)
    repeated = {oid: cnt for oid, cnt in orcid_counts.items() if cnt >= 2}
    audit["verified_rows"] = int(len(verified))
    audit["unique_orcids"] = int(len(orcid_counts))
    audit["repeated_orcids"] = int(len(repeated))
    audit["repeated_orcid_signatures"] = int(sum(repeated.values()))
    audit["output_parquet_path"] = str(out_path)

    if audit_json_path is not None:
        audit_json_path.parent.mkdir(parents=True, exist_ok=True)
        audit_json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-dataset-dir", type=Path, required=True)
    parser.add_argument("--gold-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument(
        "--error-signatures-json",
        type=Path,
        default=None,
        help="Optional JSON list of [dataset, signature_id] pairs to exclude.",
    )
    args = parser.parse_args(argv)
    dataset = args.dataset_name or args.arrow_dataset_dir.resolve().name
    errors: set[str] = set()
    if args.error_signatures_json is not None:
        raw = json.loads(args.error_signatures_json.read_text(encoding="utf-8"))
        for item in raw:
            if isinstance(item, list) and len(item) == 2 and item[0] == dataset:
                errors.add(str(item[1]))
            elif isinstance(item, str):
                errors.add(item)
    report = prepare_identity_labels(
        args.arrow_dataset_dir,
        args.gold_parquet,
        args.output_parquet,
        dataset_name=dataset,
        audit_json_path=args.audit_json,
        error_signature_ids=errors,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Join verified Gold ORCID labels onto Arrow block signatures with streaming and strict audits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.text import canonical_lasts_equivalent, canonicalize_name_parts, normalize_orcid  # noqa: E402
from scripts.production.counts.generate_orcid_name_prefix_counts import EXCLUDED_ORCID_GROUPS  # noqa: E402
from scripts.production.data.block_roles import BlockRoles, load_block_roles, normalize_block_key  # noqa: E402


def names_are_compatible(
    sig_first: str | None,
    sig_middle: str | None,
    sig_last: str | None,
    gold_first: str | None,
    gold_middle: str | None,
    gold_last: str | None,
) -> bool:
    """Return whether a signature name and gold ORCID author name are canonically compatible.

    Uses ``canonical_lasts_equivalent`` for surnames (supporting spaced/joined particles,
    listed spelling variants, and Slavic endings), supports initial and multi-token lead
    matching on first names, and rejects conflicting explicit middle initials.
    """
    sig = canonicalize_name_parts(sig_first, sig_middle, sig_last)
    gold = canonicalize_name_parts(gold_first, gold_middle, gold_last)
    if sig.last and gold.last and not canonical_lasts_equivalent(sig.last, gold.last):
        return False
    if sig.first and gold.first:
        sig_lead = sig.first.split()[0]
        gold_lead = gold.first.split()[0]
        first_ok = (
            sig.first == gold.first
            or sig_lead == gold_lead
            or (len(sig.first) == 1 and gold.first.startswith(sig.first))
            or (len(gold.first) == 1 and sig.first.startswith(gold.first))
            or (len(sig_lead) == 1 and gold_lead.startswith(sig_lead))
            or (len(gold_lead) == 1 and sig_lead.startswith(gold_lead))
        )
        if not first_ok:
            return False
    if sig.middle and gold.middle:
        sig_m_init = sig.middle.split()[0][0]
        gold_m_init = gold.middle.split()[0][0]
        if sig_m_init != gold_m_init:
            return False
    return True


def _pick_name(names: Collection[str], candidates: tuple[str, ...]) -> str | None:
    name_set = set(names)
    for candidate in candidates:
        if candidate in name_set:
            return candidate
    return None


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_paper_id(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return str(int(value))
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _strict_nonneg_int(value: Any) -> int | None:
    """Parse an exact non-negative integer; reject booleans, NaNs, and fractional floats like 1.9."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        ival = int(value)
        return ival if ival >= 0 else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


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
    expected_split: str | None = None,
    block_roles: BlockRoles | None = None,
) -> dict[str, Any]:
    """Stream, verify, and join Gold ORCID rows onto one Arrow block dataset.

    Streams ``gold_parquet_path`` with column projection and predicate pushdown so multi-tens-of-millions
    of rows (e.g. the 74M-row ``orcid-blocks-v2/labels.parquet``) run in seconds with bounded RAM.

    Enforces:
    - ``scored is True`` when a ``scored`` column is present (rejects ``None``/``NaN``/``False``)
    - ``split == expected_split`` when ``expected_split`` is specified
    - normalized valid ORCID outside ``EXCLUDED_ORCID_GROUPS``
    - exact non-negative integer ``position``
    - unique signature per ``(paper_id, author_position)`` and per ``(paper_id, author_block)``
    - ``cluster_block_key`` match when present in gold
    - canonical name compatibility when gold author name columns are present
    - exclusion of any reviewed wrong-ORCID ``error_signature_ids``
    - ORCID home roles when frozen block assignments are supplied
    """
    arrow_dir = arrow_dataset_dir.resolve()
    gold_path = gold_parquet_path.resolve()
    out_path = output_parquet_path.resolve()
    dataset = dataset_name or arrow_dir.name
    error_set = {str(sid) for sid in error_signature_ids}

    sigs = _read_arrow_signatures(arrow_dir)
    sigs["signature_id"] = sigs["signature_id"].astype(str)
    sigs["paper_id_norm"] = sigs["paper_id"].map(_normalize_paper_id)
    sigs["pos_norm"] = sigs["author_position"].map(_strict_nonneg_int)
    sigs["block_norm"] = sigs["author_block"].map(normalize_block_key)

    paper_block_counts = Counter(zip(sigs["paper_id_norm"], sigs["block_norm"], strict=True))
    paper_pos_counts = Counter(
        (pid, int(pos))
        for pid, pos in zip(sigs["paper_id_norm"], sigs["pos_norm"], strict=True)
        if pid is not None and pos is not None and not pd.isna(pos)
    )
    sig_by_paper_pos: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sigs.to_dict(orient="records"):
        pid = row["paper_id_norm"]
        pos = row["pos_norm"]
        if pid is None or pos is None or pd.isna(pos):
            continue
        key = (str(pid), int(pos))
        if paper_pos_counts[key] == 1:
            sig_by_paper_pos[key] = row

    arrow_paper_ids = {str(pid) for pid in sigs["paper_id_norm"].dropna().unique()}
    arrow_blocks_norm = {str(b) for b in sigs["block_norm"].unique() if b}
    if block_roles is not None:
        for block in arrow_blocks_norm:
            block_roles.role_for(block)

    schema = pq.read_schema(gold_path)
    col_names = list(schema.names)
    paper_col = _pick_name(col_names, ("corpus_paper_id", "paper_id"))
    pos_col = _pick_name(col_names, ("position", "author_position"))
    orcid_col = _pick_name(col_names, ("orcid", "author_orcid"))
    if paper_col is None or pos_col is None or orcid_col is None:
        raise ValueError(
            f"Gold parquet must contain paper_id/corpus_paper_id, position/author_position, and orcid: "
            f"found={col_names}"
        )
    scored_col = _pick_name(col_names, ("scored",))
    split_col = _pick_name(col_names, ("split",))
    block_col = _pick_name(col_names, ("cluster_block_key", "block_key", "author_block"))
    first_col = _pick_name(col_names, ("author_first", "first"))
    middle_col = _pick_name(col_names, ("author_middle", "middle"))
    last_col = _pick_name(col_names, ("author_last", "last"))

    if expected_split is not None and split_col is None:
        raise ValueError(f"--expected-split={expected_split!r} requested but gold parquet has no 'split' column")

    needed_cols = [
        col
        for col in (paper_col, pos_col, orcid_col, scored_col, split_col, block_col, first_col, middle_col, last_col)
        if col is not None
    ]
    name_columns_present = first_col is not None or last_col is not None

    # Prefer a tiny block-key predicate pushdown (e.g. ["a khan", "a_khan"]) when block_col exists:
    # Parquet row-group / dictionary pruning skips ~98% of a 74M-row file in <1s, whereas pushing a
    # 100k-element paper_id isin() set across 74M rows stalls PyArrow evaluation.
    filter_expr = None
    if block_col is not None and arrow_blocks_norm:
        block_variants: set[str] = set()
        for blk in arrow_blocks_norm:
            block_variants.add(blk)
            block_variants.add(blk.replace(" ", "_"))
        filter_expr = ds.field(block_col).isin(sorted(block_variants))
    elif arrow_paper_ids and len(arrow_paper_ids) <= 4096:
        paper_field = schema.field(paper_col)
        if pa.types.is_integer(paper_field.type):
            int_ids = [int(pid) for pid in arrow_paper_ids if pid.isdigit()]
            if int_ids:
                filter_expr = ds.field(paper_col).isin(int_ids)
        elif pa.types.is_string(paper_field.type) or pa.types.is_large_string(paper_field.type):
            filter_expr = ds.field(paper_col).isin(sorted(arrow_paper_ids))

    dataset_reader = ds.dataset(gold_path, format="parquet")
    scanner = dataset_reader.scanner(columns=needed_cols, filter=filter_expr, batch_size=65_536)

    audit: dict[str, Any] = {
        "dataset": dataset,
        "arrow_dataset_dir": str(arrow_dir),
        "gold_parquet_path": str(gold_path),
        "arrow_signature_count": int(len(sigs)),
        "arrow_blocks": sorted(arrow_blocks_norm),
        "expected_split": expected_split,
        "name_columns_present": name_columns_present,
        "verification_mode": "position_block_and_name" if name_columns_present else "position_and_block_only",
        "gold_rows_scanned": 0,
        "dropped_wrong_split": 0,
        "dropped_unscored": 0,
        "dropped_non_home_role": 0,
        "block_role_assignments_sha256": block_roles.sha256 if block_roles is not None else None,
        "dropped_invalid_or_missing_orcid": 0,
        "dropped_excluded_orcid_group": 0,
        "dropped_invalid_position": 0,
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
        "dropped_ambiguous_paper_block_fraction": 0.0,
    }

    by_signature: dict[str, list[dict[str, Any]]] = {}
    for batch in scanner.to_batches():
        cols = {name: batch.column(name).to_pylist() for name in needed_cols}
        n_rows = batch.num_rows
        audit["gold_rows_scanned"] += int(n_rows)
        for i in range(n_rows):
            row_split = _optional_str(cols[split_col][i]) if split_col is not None else None
            if expected_split is not None and row_split != expected_split:
                audit["dropped_wrong_split"] += 1
                continue
            if scored_col is not None and cols[scored_col][i] is not True:
                audit["dropped_unscored"] += 1
                continue
            raw_orcid = _optional_str(cols[orcid_col][i])
            orcid = normalize_orcid(raw_orcid) if raw_orcid else None
            if not orcid:
                audit["dropped_invalid_or_missing_orcid"] += 1
                continue
            if orcid in EXCLUDED_ORCID_GROUPS:
                audit["dropped_excluded_orcid_group"] += 1
                continue
            pid = _normalize_paper_id(cols[paper_col][i])
            pos = _strict_nonneg_int(cols[pos_col][i])
            if pos is None:
                audit["dropped_invalid_position"] += 1
                continue
            if pid is None:
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
                gold_block = normalize_block_key(_optional_str(cols[block_col][i]))
                if gold_block and gold_block != str(sig_row["block_norm"]):
                    audit["dropped_block_key_mismatch"] += 1
                    continue
            if name_columns_present:
                if not names_are_compatible(
                    _optional_str(sig_row.get("author_first")),
                    _optional_str(sig_row.get("author_middle")),
                    _optional_str(sig_row.get("author_last")),
                    _optional_str(cols[first_col][i]) if first_col else None,
                    _optional_str(cols[middle_col][i]) if middle_col else None,
                    _optional_str(cols[last_col][i]) if last_col else None,
                ):
                    audit["dropped_name_mismatch"] += 1
                    continue
            sid = str(sig_row["signature_id"])
            if sid in error_set:
                audit["dropped_reviewed_error_signature"] += 1
                continue
            entry: dict[str, Any] = {
                "signature_id": sid,
                "orcid": orcid,
                "paper_id": pid,
                "position": pos,
                "author_block": str(sig_row["author_block"]),
                "dataset": dataset,
                "name_verified": name_columns_present,
            }
            if split_col is not None:
                entry["split"] = row_split
            by_signature.setdefault(sid, []).append(entry)

    verified: list[dict[str, Any]] = []
    for sid in sorted(by_signature):
        entries = by_signature[sid]
        distinct_orcids = {entry["orcid"] for entry in entries}
        if len(distinct_orcids) > 1:
            audit["dropped_conflicting_gold_rows"] += len(entries)
            continue
        if block_roles is not None and not block_roles.scores(entries[0]["author_block"], entries[0]["orcid"]):
            audit["dropped_non_home_role"] += len(entries)
            continue
        verified.append(entries[0])

    out_cols = ["signature_id", "orcid", "paper_id", "position", "author_block", "dataset", "name_verified"]
    if split_col is not None:
        out_cols.append("split")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_frame = pd.DataFrame(verified, columns=pd.Index(out_cols))
    out_frame.to_parquet(out_path, index=False)

    orcid_counts = Counter(row["orcid"] for row in verified)
    repeated = {oid: cnt for oid, cnt in orcid_counts.items() if cnt >= 2}
    matched_or_ambig = len(verified) + audit["dropped_ambiguous_paper_block"]
    audit["verified_rows"] = int(len(verified))
    audit["unique_orcids"] = int(len(orcid_counts))
    audit["repeated_orcids"] = int(len(repeated))
    audit["repeated_orcid_signatures"] = int(sum(repeated.values()))
    audit["dropped_ambiguous_paper_block_fraction"] = (
        audit["dropped_ambiguous_paper_block"] / matched_or_ambig if matched_or_ambig else 0.0
    )
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
    parser.add_argument("--block-role-assignments", type=Path, help="Apply the frozen ORCID home-role mask.")
    parser.add_argument(
        "--expected-split",
        default=None,
        help="Require gold rows to match this split (e.g. 'development' or 'holdout').",
    )
    parser.add_argument(
        "--error-signatures-json",
        type=Path,
        default=None,
        help="Optional JSON list of [dataset, signature_id] pairs to exclude.",
    )
    args = parser.parse_args(argv)
    dataset = args.dataset_name or args.arrow_dataset_dir.resolve().name
    dataset_norm = normalize_block_key(dataset)
    errors: set[str] = set()
    if args.error_signatures_json is not None:
        raw = json.loads(args.error_signatures_json.read_text(encoding="utf-8"))
        for item in raw:
            if isinstance(item, list) and len(item) == 2 and normalize_block_key(str(item[0])) == dataset_norm:
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
        expected_split=args.expected_split,
        block_roles=load_block_roles(args.block_role_assignments) if args.block_role_assignments is not None else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

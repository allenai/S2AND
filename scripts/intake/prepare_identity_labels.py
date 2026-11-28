"""Verify Gold ORCID joins against Arrow block signatures."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.ipc as ipc

from s2and.text import canonical_lasts_equivalent, canonicalize_name_parts
from scripts.common.block_roles import normalize_block_key


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
            "author_orcid",
        )
        if col in table.column_names
    ]
    missing = sorted({"signature_id", "paper_id", "author_position", "author_block"} - set(want))
    if missing:
        raise ValueError(f"Arrow signatures table missing required columns: {missing}")
    return table.select(want).to_pandas()


class SignatureJoin:
    """Verify a corpus author position against one Arrow block."""

    def __init__(self, sigs: pd.DataFrame) -> None:
        sigs["signature_id"] = sigs["signature_id"].astype(str)
        sigs["paper_id_norm"] = sigs["paper_id"].map(_normalize_paper_id)
        sigs["pos_norm"] = sigs["author_position"].map(_strict_nonneg_int)
        sigs["block_norm"] = sigs["author_block"].map(normalize_block_key)
        self.rows: list[dict[str, Any]] = sigs.to_dict(orient="records")
        self.paper_blocks = Counter((row["paper_id_norm"], row["block_norm"]) for row in self.rows)
        self.positions: dict[tuple[str, int], dict[str, Any]] = {}
        self.ambiguous: set[tuple[str, int]] = set()
        for row in self.rows:
            pid, pos = row["paper_id_norm"], row["pos_norm"]
            if pid is None or pd.isna(pos):
                continue
            key = (pid, int(pos))
            if key in self.positions:
                self.ambiguous.add(key)
            else:
                self.positions[key] = row

    def match(
        self, paper_id: str, position: int, block: str, names: tuple[str | None, str | None, str | None]
    ) -> tuple[dict[str, Any] | None, str]:
        """Return a verified signature row or its audit reason."""
        key = (paper_id, position)
        if key in self.ambiguous:
            return None, "dropped_ambiguous_arrow_position"
        row = self.positions.get(key)
        if row is None:
            return None, "dropped_unmatched_paper_position"
        if self.paper_blocks[paper_id, row["block_norm"]] > 1:
            return None, "dropped_ambiguous_paper_block"
        if block and block != row["block_norm"]:
            return None, "dropped_block_key_mismatch"
        if not names_are_compatible(
            _optional_str(row.get("author_first")),
            _optional_str(row.get("author_middle")),
            _optional_str(row.get("author_last")),
            *names,
        ):
            return None, "dropped_name_mismatch"
        return row, ""


def canonical_gold_block_filter(block_col: str, blocks: Collection[str]) -> ds.Expression:
    """Select canonical ASCII keys and Unicode spellings for exact Python normalization."""
    raw_key = ds.field(block_col)
    ascii_key = pc.ascii_lower(pc.utf8_trim(raw_key, characters=" \t\n\v\f\r\x1c\x1d\x1e\x1f"))
    ascii_key = pc.replace_substring_regex(ascii_key, pattern=r"[\s_\x0B\x1C-\x1F]+", replacement=" ")
    return ascii_key.isin(sorted(blocks)) | pc.match_substring_regex(raw_key, pattern=r"[^\x00-\x7F]")

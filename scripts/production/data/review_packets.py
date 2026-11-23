"""Leak-proof blind review packet builder and validator for Sol 6.2 manual labeling.

Enforces strict field allowlists (raw paper metadata only — never predicted cluster
summaries, ORCIDs, scores, ranks, model names, or ground-truth flags), seeded order
randomization, embedded ORCID controls, shard size caps, and pre-launch leak checks.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.production.data.hash_release_emails import _find_plaintext_emails, scrub_text_emails  # noqa: E402

ALLOWED_PAPER_ITEM_FIELDS = frozenset(
    {
        "signature_id",
        "paper_id",
        "title",
        "abstract",
        "venue",
        "year",
        "authors",
        "author_name",
        "affiliations",
        "coauthors",
    }
)
ALLOWED_PAIR_RECORD_FIELDS = frozenset({"item_id", "block_key", "paper_a", "paper_b"})
ALLOWED_LINKER_CANDIDATE_FIELDS = frozenset({"candidate_slot", "exemplars"})
ALLOWED_LINKER_RECORD_FIELDS = frozenset({"item_id", "block_key", "query", "candidates"})

_ORCID_VALUE_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[0-9X]\b")
_FORBIDDEN_KEY_SUBSTRINGS = (
    "orcid",
    "score",
    "prob",
    "rank",
    "cluster",
    "model",
    "label",
    "gold",
    "control",
    "summary",
    "margin",
    "gate",
    "action",
    "eps",
    "truth",
    "pred",
    "source_model",
    "winner",
)


def sanitize_paper_item(raw: Mapping[str, Any], *, max_abstract_chars: int = 1200) -> dict[str, Any]:
    """Extract only allowlisted raw paper metadata fields and scrub plaintext emails."""
    clean: dict[str, Any] = {}
    for key in sorted(ALLOWED_PAPER_ITEM_FIELDS):
        if key not in raw:
            continue
        value = raw[key]
        if value is None:
            clean[key] = None
        elif isinstance(value, str):
            text = scrub_text_emails(value) or ""
            if key == "abstract" and len(text) > max_abstract_chars:
                text = text[:max_abstract_chars].rstrip() + "..."
            clean[key] = text
        elif isinstance(value, list):
            clean[key] = [scrub_text_emails(v) if isinstance(v, str) else v for v in value]
        elif isinstance(value, int | float | bool):
            clean[key] = value
        else:
            clean[key] = scrub_text_emails(str(value))
    return clean


def _scan_blind_value(value: Any, *, path: str, violations: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_str = str(key)
            key_lower = key_str.lower()
            if any(token in key_lower for token in _FORBIDDEN_KEY_SUBSTRINGS):
                violations.append(f"{path}.{key_str}: forbidden key")
            _scan_blind_value(nested, path=f"{path}.{key_str}", violations=violations)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _scan_blind_value(item, path=f"{path}[{idx}]", violations=violations)
    elif isinstance(value, str):
        if _ORCID_VALUE_RE.search(value):
            violations.append(f"{path}: ORCID pattern found in blind packet text")
        if "@" in value and _find_plaintext_emails(value):
            violations.append(f"{path}: plaintext email found in blind packet text")


def validate_blind_packet_records(records: Sequence[Mapping[str, Any]], *, packet_type: str) -> list[str]:
    """Validate that blind packet records obey the strict allowlist and leak zero side-channel info."""
    if packet_type not in {"pair", "linker"}:
        raise ValueError(f"Unsupported packet_type: {packet_type!r}")
    violations: list[str] = []
    allowed_top = ALLOWED_PAIR_RECORD_FIELDS if packet_type == "pair" else ALLOWED_LINKER_RECORD_FIELDS
    for idx, record in enumerate(records):
        prefix = f"records[{idx}]"
        extra_top = sorted(set(record) - allowed_top)
        if extra_top:
            violations.append(f"{prefix}: disallowed top-level keys {extra_top}")
        if packet_type == "pair":
            for side in ("paper_a", "paper_b"):
                item = record.get(side)
                if not isinstance(item, Mapping):
                    violations.append(f"{prefix}.{side}: expected mapping")
                    continue
                extra = sorted(set(item) - ALLOWED_PAPER_ITEM_FIELDS)
                if extra:
                    violations.append(f"{prefix}.{side}: disallowed item keys {extra}")
        else:
            query = record.get("query")
            if not isinstance(query, Mapping):
                violations.append(f"{prefix}.query: expected mapping")
            else:
                extra_q = sorted(set(query) - ALLOWED_PAPER_ITEM_FIELDS)
                if extra_q:
                    violations.append(f"{prefix}.query: disallowed item keys {extra_q}")
            candidates = record.get("candidates")
            if not isinstance(candidates, list):
                violations.append(f"{prefix}.candidates: expected list")
            else:
                for c_idx, cand in enumerate(candidates):
                    if not isinstance(cand, Mapping):
                        violations.append(f"{prefix}.candidates[{c_idx}]: expected mapping")
                        continue
                    extra_c = sorted(set(cand) - ALLOWED_LINKER_CANDIDATE_FIELDS)
                    if extra_c:
                        violations.append(f"{prefix}.candidates[{c_idx}]: disallowed candidate keys {extra_c}")
                    exemplars = cand.get("exemplars", [])
                    if not isinstance(exemplars, list):
                        violations.append(f"{prefix}.candidates[{c_idx}].exemplars: expected list")
                        continue
                    for e_idx, exemplar in enumerate(exemplars):
                        if not isinstance(exemplar, Mapping):
                            violations.append(f"{prefix}.candidates[{c_idx}].exemplars[{e_idx}]: expected mapping")
                            continue
                        extra_e = sorted(set(exemplar) - ALLOWED_PAPER_ITEM_FIELDS)
                        if extra_e:
                            violations.append(
                                f"{prefix}.candidates[{c_idx}].exemplars[{e_idx}]: disallowed item keys {extra_e}"
                            )
        _scan_blind_value(record, path=prefix, violations=violations)
    return violations


def _shard_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_items_per_shard: int,
    max_bytes_per_shard: int,
) -> list[list[Mapping[str, Any]]]:
    if max_items_per_shard <= 0 or max_bytes_per_shard <= 0:
        raise ValueError("Shard limits must be positive")
    shards: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    current_bytes = 0
    for record in records:
        encoded_len = len((json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        if encoded_len > max_bytes_per_shard:
            raise ValueError(f"Single record exceeds max_bytes_per_shard ({encoded_len} > {max_bytes_per_shard})")
        if current and (len(current) >= max_items_per_shard or current_bytes + encoded_len > max_bytes_per_shard):
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += encoded_len
    if current:
        shards.append(current)
    return shards


def build_blind_pair_packets(
    *,
    items: Sequence[Mapping[str, Any]],
    control_items: Sequence[Mapping[str, Any]] = (),
    output_dir: Path,
    seed: int = 20261123,
    control_fraction: float = 0.12,
    max_items_per_shard: int = 40,
    max_bytes_per_shard: int = 180_000,
    max_abstract_chars: int = 1200,
) -> dict[str, Any]:
    """Build leak-checked, order-randomized blind pair packets plus a separate private answer key.

    Each input item must contain ``source_id``, ``block_key``, ``paper_a``, and ``paper_b``
    (plus optional ``metadata`` stored ONLY in ``answer_key.json``). Control items should
    additionally include ``gold_label`` in ``metadata`` (never emitted to blind packets).
    """
    if not 0.0 <= control_fraction <= 0.5:
        raise ValueError(f"control_fraction must be in [0.0, 0.5], got {control_fraction}")
    rng = random.Random(seed)
    combined: list[tuple[bool, Mapping[str, Any]]] = [(False, item) for item in items]
    if control_items and control_fraction > 0.0:
        desired = max(1, round(len(items) * control_fraction / (1.0 - control_fraction)))
        target_controls = min(len(control_items), desired)
        chosen_controls = rng.sample(list(control_items), target_controls)
        combined.extend((True, item) for item in chosen_controls)
    rng.shuffle(combined)

    blind_records: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for index, (is_control, raw) in enumerate(combined, start=1):
        item_id = f"pair_{index:05d}"
        swap_sides = rng.random() < 0.5
        raw_a = raw["paper_b"] if swap_sides else raw["paper_a"]
        raw_b = raw["paper_a"] if swap_sides else raw["paper_b"]
        blind_records.append(
            {
                "item_id": item_id,
                "block_key": str(raw["block_key"]),
                "paper_a": sanitize_paper_item(raw_a, max_abstract_chars=max_abstract_chars),
                "paper_b": sanitize_paper_item(raw_b, max_abstract_chars=max_abstract_chars),
            }
        )
        answer_key.append(
            {
                "item_id": item_id,
                "source_id": str(raw["source_id"]),
                "block_key": str(raw["block_key"]),
                "is_control": is_control,
                "sides_swapped": swap_sides,
                "metadata": dict(raw.get("metadata", {})),
            }
        )

    violations = validate_blind_packet_records(blind_records, packet_type="pair")
    if violations:
        raise ValueError(f"Blind pair packet failed leak check ({len(violations)} violations): {violations[:5]}")

    shards = _shard_records(
        blind_records,
        max_items_per_shard=max_items_per_shard,
        max_bytes_per_shard=max_bytes_per_shard,
    )
    output_dir = output_dir.resolve()
    packets_dir = output_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    for shard_index, shard in enumerate(shards, start=1):
        shard_path = packets_dir / f"pair_shard_{shard_index:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8", newline="\n") as handle:
            for rec in shard:
                handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        shard_paths.append(shard_path.relative_to(output_dir).as_posix())

    key_payload = {
        "packet_type": "pair",
        "seed": int(seed),
        "total_items": len(blind_records),
        "control_items": sum(1 for row in answer_key if row["is_control"]),
        "shards": shard_paths,
        "items": answer_key,
    }
    (output_dir / "answer_key.json").write_text(
        json.dumps(key_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "packet_type": "pair",
        "output_dir": str(output_dir),
        "shard_count": len(shard_paths),
        "total_items": len(blind_records),
        "control_items": key_payload["control_items"],
        "shards": shard_paths,
    }


def build_blind_linker_packets(
    *,
    queries: Sequence[Mapping[str, Any]],
    control_queries: Sequence[Mapping[str, Any]] = (),
    output_dir: Path,
    seed: int = 20261123,
    control_fraction: float = 0.12,
    max_items_per_shard: int = 25,
    max_bytes_per_shard: int = 180_000,
    max_abstract_chars: int = 900,
) -> dict[str, Any]:
    """Build leak-checked blind linker candidate packets with shuffled candidate slots.

    Each input query must contain ``source_id``, ``block_key``, ``query`` (paper metadata),
    and ``candidates`` (list of ``{"candidate_key": str, "exemplars": [paper_dict, ...]}``).
    Candidate order is randomized per query and mapped to neutral labels ``C1..Ck`` in
    the blind packet, with the true mapping saved only in ``answer_key.json``.
    """
    if not 0.0 <= control_fraction <= 0.5:
        raise ValueError(f"control_fraction must be in [0.0, 0.5], got {control_fraction}")
    rng = random.Random(seed)
    combined: list[tuple[bool, Mapping[str, Any]]] = [(False, q) for q in queries]
    if control_queries and control_fraction > 0.0:
        target_controls = min(
            len(control_queries), max(1, round(len(queries) * control_fraction / (1.0 - control_fraction)))
        )
        chosen_controls = rng.sample(list(control_queries), target_controls)
        combined.extend((True, q) for q in chosen_controls)
    rng.shuffle(combined)

    blind_records: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for index, (is_control, raw) in enumerate(combined, start=1):
        item_id = f"linker_{index:05d}"
        candidates_in = list(raw["candidates"])
        rng.shuffle(candidates_in)
        blind_candidates: list[dict[str, Any]] = []
        slot_to_key: dict[str, str] = {}
        for slot_idx, cand in enumerate(candidates_in, start=1):
            slot = f"C{slot_idx}"
            slot_to_key[slot] = str(cand["candidate_key"])
            blind_candidates.append(
                {
                    "candidate_slot": slot,
                    "exemplars": [
                        sanitize_paper_item(ex, max_abstract_chars=max_abstract_chars)
                        for ex in cand.get("exemplars", ())
                    ],
                }
            )
        blind_records.append(
            {
                "item_id": item_id,
                "block_key": str(raw["block_key"]),
                "query": sanitize_paper_item(raw["query"], max_abstract_chars=max_abstract_chars),
                "candidates": blind_candidates,
            }
        )
        answer_key.append(
            {
                "item_id": item_id,
                "source_id": str(raw["source_id"]),
                "block_key": str(raw["block_key"]),
                "is_control": is_control,
                "slot_to_candidate_key": slot_to_key,
                "metadata": dict(raw.get("metadata", {})),
            }
        )

    violations = validate_blind_packet_records(blind_records, packet_type="linker")
    if violations:
        raise ValueError(f"Blind linker packet failed leak check ({len(violations)} violations): {violations[:5]}")

    shards = _shard_records(
        blind_records,
        max_items_per_shard=max_items_per_shard,
        max_bytes_per_shard=max_bytes_per_shard,
    )
    output_dir = output_dir.resolve()
    packets_dir = output_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    for shard_index, shard in enumerate(shards, start=1):
        shard_path = packets_dir / f"linker_shard_{shard_index:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8", newline="\n") as handle:
            for rec in shard:
                handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        shard_paths.append(shard_path.relative_to(output_dir).as_posix())

    key_payload = {
        "packet_type": "linker",
        "seed": int(seed),
        "total_items": len(blind_records),
        "control_items": sum(1 for row in answer_key if row["is_control"]),
        "shards": shard_paths,
        "items": answer_key,
    }
    (output_dir / "answer_key.json").write_text(
        json.dumps(key_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "packet_type": "linker",
        "output_dir": str(output_dir),
        "shard_count": len(shard_paths),
        "total_items": len(blind_records),
        "control_items": key_payload["control_items"],
        "shards": shard_paths,
    }

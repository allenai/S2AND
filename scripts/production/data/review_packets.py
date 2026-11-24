"""Leak-proof blind review packet builder and validator for Sol 6.2 manual labeling.

Enforces strict field allowlists (raw paper metadata only — never ``signature_id``,
``paper_id``, ``author_id``, predicted cluster summaries, ORCIDs, scores, ranks, model
names, or ground-truth flags), flattens author lists to plain strings, scrubs incidental
ORCIDs and emails, requires a fresh packet directory with the private answer key stored
outside the packet tree, distributes embedded ORCID controls evenly per shard, and separates
``control_seed`` from ``order_seed`` for reproducible two-pass judging.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.consts import PLACEHOLDER_YEARS, normalize_paper_year  # noqa: E402
from s2and.text import ORCID_DASH_CLASS, ORCID_PATTERN, normalize_orcid  # noqa: E402
from scripts.production.counts.generate_orcid_name_prefix_counts import EXCLUDED_ORCID_GROUPS  # noqa: E402
from scripts.production.data.block_roles import BlockRoles, load_block_roles, normalize_block_key  # noqa: E402
from scripts.production.data.hash_release_emails import _find_plaintext_emails, scrub_text_emails  # noqa: E402

ORCID_REDACTED_TOKEN = "[ORCID_REDACTED]"
# Note: signature_id and paper_id are intentionally EXCLUDED from blind payloads (kept only in answer_key)
# so synthetic prefixes like visibleq:/visiblep:/qian___ can never leak to judges.
ALLOWED_PAPER_ITEM_FIELDS = frozenset(
    {
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

# Matches hyphenated, Unicode-dash, space-separated, and compact 16-char ORCIDs.
_BROAD_ORCID_RE = re.compile(
    rf"(?i)(?<![0-9x])"
    rf"[0-9]{{4}}[{ORCID_DASH_CLASS}\s]?"
    rf"[0-9]{{4}}[{ORCID_DASH_CLASS}\s]?"
    rf"[0-9]{{4}}[{ORCID_DASH_CLASS}\s]?"
    rf"[0-9]{{3}}[0-9x](?![0-9x])"
)
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
    "author_id",
    "signature_id",
    "paper_id",
    "decision",
    "identity",
)


def _contains_valid_orcid(text: str) -> bool:
    for pattern in (ORCID_PATTERN, _BROAD_ORCID_RE):
        for match in pattern.finditer(text):
            compact = re.sub(r"\s+", "", match.group(0))
            if normalize_orcid(compact) is not None:
                return True
    return False


def scrub_text_orcids(value: str) -> str:
    """Replace any valid ISO-7064 ORCID (hyphenated, Unicode-dash, spaced, or compact) with a neutral token."""

    def _replace(match: re.Match[str]) -> str:
        compact = re.sub(r"\s+", "", match.group(0))
        return ORCID_REDACTED_TOKEN if normalize_orcid(compact) is not None else match.group(0)

    return _BROAD_ORCID_RE.sub(_replace, value)


def _clean_text(value: str) -> str:
    scrubbed = scrub_text_emails(value) or ""
    return scrub_text_orcids(scrubbed)


def _flatten_author_entry(entry: Any) -> str | None:
    """Extract only a plain author name string from an author list item (stripping IDs/ORCIDs)."""
    if isinstance(entry, str):
        text = _clean_text(entry).strip()
        return text or None
    if isinstance(entry, Mapping):
        for key in ("author_name", "name"):
            val = entry.get(key)
            if isinstance(val, str) and val.strip():
                return _clean_text(val).strip()
        return None
    return None


def sanitize_paper_item(raw: Mapping[str, Any], *, max_abstract_chars: int = 1200) -> dict[str, Any]:
    """Extract only allowlisted raw paper metadata fields as strict scalars or lists of strings."""
    clean: dict[str, Any] = {}
    for key in sorted(ALLOWED_PAPER_ITEM_FIELDS):
        if key not in raw:
            continue
        value = raw[key]
        if value is None:
            clean[key] = None
        elif key in {"authors", "coauthors"}:
            if isinstance(value, list):
                names = [name for item in value if (name := _flatten_author_entry(item)) is not None]
                clean[key] = names
            elif isinstance(value, str):
                clean[key] = [_clean_text(value)]
            else:
                clean[key] = []
        elif key == "affiliations":
            if isinstance(value, list):
                clean[key] = [_clean_text(item) for item in value if isinstance(item, str) and item.strip()]
            elif isinstance(value, str):
                clean[key] = [_clean_text(value)] if value.strip() else []
            else:
                clean[key] = []
        elif key == "year":
            if isinstance(value, int) and not isinstance(value, bool):
                clean[key] = normalize_paper_year(value)
            elif isinstance(value, str) and value.strip().isdigit():
                clean[key] = normalize_paper_year(int(value.strip()))
            else:
                clean[key] = None
        elif isinstance(value, str):
            text = _clean_text(value)
            if key == "abstract" and len(text) > max_abstract_chars:
                text = text[:max_abstract_chars].rstrip() + "..."
            clean[key] = text
        else:
            clean[key] = _clean_text(str(value))
    return clean


def _scan_blind_value(value: Any, *, path: str, violations: list[str], allow_nested_dicts: bool = True) -> None:
    if isinstance(value, Mapping):
        if not allow_nested_dicts:
            violations.append(f"{path}: unexpected nested mapping inside blind leaf list")
            return
        for key, nested in value.items():
            key_str = str(key)
            key_lower = key_str.lower()
            if any(token in key_lower for token in _FORBIDDEN_KEY_SUBSTRINGS):
                violations.append(f"{path}.{key_str}: forbidden key")
            _scan_blind_value(nested, path=f"{path}.{key_str}", violations=violations, allow_nested_dicts=True)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _scan_blind_value(item, path=f"{path}[{idx}]", violations=violations, allow_nested_dicts=allow_nested_dicts)
    elif isinstance(value, str):
        if _contains_valid_orcid(value):
            violations.append(f"{path}: ORCID pattern found in blind packet text")
        if ("@" in value or "\uff20" in value) and _find_plaintext_emails(value):
            violations.append(f"{path}: plaintext email found in blind packet text")


def _validate_paper_item(item: Any, *, prefix: str, violations: list[str]) -> None:
    if not isinstance(item, Mapping):
        violations.append(f"{prefix}: expected mapping")
        return
    extra = sorted(set(item) - ALLOWED_PAPER_ITEM_FIELDS)
    if extra:
        violations.append(f"{prefix}: disallowed item keys {extra}")
    for key, val in item.items():
        if key in {"authors", "coauthors", "affiliations"}:
            if val is not None:
                if not isinstance(val, list) or any(not isinstance(elem, str) for elem in val):
                    violations.append(f"{prefix}.{key}: must be a list of plain strings")
        elif key == "year":
            if val is not None and (isinstance(val, bool) or not isinstance(val, int)):
                violations.append(f"{prefix}.year: must be an integer or None")
            elif val in PLACEHOLDER_YEARS:
                violations.append(f"{prefix}.year: replace a corpus placeholder with None")
        elif val is not None and not isinstance(val, str):
            violations.append(f"{prefix}.{key}: must be a plain string or None")
    _scan_blind_value(item, path=prefix, violations=violations, allow_nested_dicts=True)


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
            _validate_paper_item(record.get("paper_a"), prefix=f"{prefix}.paper_a", violations=violations)
            _validate_paper_item(record.get("paper_b"), prefix=f"{prefix}.paper_b", violations=violations)
        else:
            _validate_paper_item(record.get("query"), prefix=f"{prefix}.query", violations=violations)
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
                        _validate_paper_item(
                            exemplar,
                            prefix=f"{prefix}.candidates[{c_idx}].exemplars[{e_idx}]",
                            violations=violations,
                        )
        _scan_blind_value(record, path=prefix, violations=violations)
    return violations


def _resolve_answer_key_path(output_dir: Path, answer_key_path: Path | None) -> Path:
    resolved_out = output_dir.resolve()
    key_path = (
        (resolved_out.parent / f"{resolved_out.name}_answer_key.json")
        if answer_key_path is None
        else answer_key_path.resolve()
    )
    if key_path.is_relative_to(resolved_out):
        raise ValueError(
            f"Private answer_key_path ({key_path}) must be outside the blind packet directory ({resolved_out})"
        )
    if key_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing answer_key_path: {key_path}")
    return key_path


def _select_controls(
    items_count: int,
    control_items: Sequence[Mapping[str, Any]],
    *,
    control_fraction: float,
    control_seed: int,
) -> list[Mapping[str, Any]]:
    if not control_items or control_fraction <= 0.0:
        return []
    rng = random.Random(control_seed)
    desired = max(1, round(items_count * control_fraction / (1.0 - control_fraction)))
    target = min(len(control_items), desired)
    return rng.sample(list(control_items), target)


def _validate_packet_roles(
    items: Sequence[Mapping[str, Any]], block_roles: BlockRoles | None, allowed_roles: Collection[str]
) -> None:
    if (block_roles is None) == bool(allowed_roles):
        raise ValueError("Supply a role manifest and at least one allowed packet role together")
    if block_roles is not None:
        for item in items:
            block = str(item["block_key"])
            if block_roles.role_for(block) not in allowed_roles:
                raise ValueError(f"Block {block!r} is outside the allowed packet roles")


def _scored_controls(
    items: Sequence[Mapping[str, Any]],
    block_roles: BlockRoles | None,
    *,
    packet_type: str,
    allowed_roles: Collection[str],
) -> list[Mapping[str, Any]]:
    if block_roles is None:
        return list(items)
    result = []
    for item in items:
        block = str(item["block_key"])
        if block_roles.role_for(block) not in allowed_roles:
            continue
        if packet_type == "pair":
            papers = [item["paper_a"], item["paper_b"]]
        else:
            papers = [item["query"], *(paper for cand in item["candidates"] for paper in cand["exemplars"])]
        orcids = [normalize_orcid(str(paper.get("orcid") or paper.get("author_orcid") or "")) for paper in papers]
        if all(
            orcid
            and orcid not in EXCLUDED_ORCID_GROUPS
            and paper.get("scored", True) is True
            and block_roles.scores(block, orcid)
            for paper, orcid in zip(papers, orcids, strict=True)
        ):
            result.append(item)
    return result


def _stratified_interleave(
    regular: Sequence[tuple[bool, Mapping[str, Any]]],
    controls: Sequence[tuple[bool, Mapping[str, Any]]],
    *,
    max_items_per_shard: int,
    order_seed: int,
) -> list[tuple[bool, Mapping[str, Any]]]:
    """Distribute controls evenly across every shard, then shuffle within each shard."""
    rng = random.Random(order_seed)
    reg = list(regular)
    ctrl = list(controls)
    rng.shuffle(reg)
    rng.shuffle(ctrl)
    total = len(reg) + len(ctrl)
    if total == 0:
        return []
    n_shards = max(1, (total + max_items_per_shard - 1) // max_items_per_shard)
    buckets: list[list[tuple[bool, Mapping[str, Any]]]] = [[] for _ in range(n_shards)]
    for idx, entry in enumerate(ctrl):
        buckets[idx % n_shards].append(entry)
    for idx, entry in enumerate(reg):
        buckets[idx % n_shards].append(entry)
    ordered: list[tuple[bool, Mapping[str, Any]]] = []
    for bucket in buckets:
        rng.shuffle(bucket)
        ordered.extend(bucket)
    return ordered


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
    answer_key_path: Path | None = None,
    seed: int = 20261123,
    control_seed: int | None = None,
    control_fraction: float = 0.12,
    max_items_per_shard: int = 40,
    max_bytes_per_shard: int = 180_000,
    max_abstract_chars: int = 1200,
    block_roles: BlockRoles | None = None,
    allowed_roles: Collection[str] = (),
) -> dict[str, Any]:
    """Build leak-checked, order-randomized blind pair packets plus a separate private answer key.

    Requires a fresh ``output_dir`` and places ``answer_key.json`` OUTSIDE ``output_dir``
    so a judge pointed at ``output_dir`` can never read the key. Separating ``control_seed``
    from ``seed`` (presentation order) enables reproducible two-pass judging on the exact
    same item+control population.
    """
    if not 0.0 <= control_fraction <= 0.5:
        raise ValueError(f"control_fraction must be in [0.0, 0.5], got {control_fraction}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Blind packet builder requires a fresh output directory: {output_dir}")
    key_path = _resolve_answer_key_path(output_dir, answer_key_path)

    effective_control_seed = seed if control_seed is None else control_seed
    _validate_packet_roles(items, block_roles, allowed_roles)
    eligible_controls = _scored_controls(control_items, block_roles, packet_type="pair", allowed_roles=allowed_roles)
    chosen_controls = _select_controls(
        len(items), eligible_controls, control_fraction=control_fraction, control_seed=effective_control_seed
    )
    combined = _stratified_interleave(
        [(False, item) for item in items],
        [(True, item) for item in chosen_controls],
        max_items_per_shard=max_items_per_shard,
        order_seed=seed,
    )

    rng = random.Random(seed)
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
                "paper_a_signature_id": raw_a.get("signature_id"),
                "paper_b_signature_id": raw_b.get("signature_id"),
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
    packets_dir = output_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=False)
    shard_paths: list[str] = []
    key_items = iter(answer_key)
    for shard_index, shard in enumerate(shards, start=1):
        shard_path = packets_dir / f"pair_shard_{shard_index:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8", newline="\n") as handle:
            for rec in shard:
                handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        shard_paths.append(shard_path.relative_to(output_dir).as_posix())
        for _ in shard:
            next(key_items)["shard"] = shard_paths[-1]

    key_payload = {
        "packet_type": "pair",
        "order_seed": int(seed),
        "control_seed": int(effective_control_seed),
        "total_items": len(blind_records),
        "control_items": sum(1 for row in answer_key if row["is_control"]),
        "shards": shard_paths,
        "items": answer_key,
    }
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {
        "packet_type": "pair",
        "output_dir": str(output_dir),
        "answer_key_path": str(key_path),
        "shard_count": len(shard_paths),
        "total_items": len(blind_records),
        "control_items": key_payload["control_items"],
        "excluded_controls": len(control_items) - len(eligible_controls),
        "shards": shard_paths,
    }


def build_blind_linker_packets(
    *,
    queries: Sequence[Mapping[str, Any]],
    control_queries: Sequence[Mapping[str, Any]] = (),
    output_dir: Path,
    answer_key_path: Path | None = None,
    seed: int = 20261123,
    control_seed: int | None = None,
    control_fraction: float = 0.12,
    max_items_per_shard: int = 25,
    max_bytes_per_shard: int = 180_000,
    max_abstract_chars: int = 900,
    block_roles: BlockRoles | None = None,
    allowed_roles: Collection[str] = (),
) -> dict[str, Any]:
    """Build leak-checked blind linker candidate packets with shuffled neutral candidate slots."""
    if not 0.0 <= control_fraction <= 0.5:
        raise ValueError(f"control_fraction must be in [0.0, 0.5], got {control_fraction}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Blind packet builder requires a fresh output directory: {output_dir}")
    key_path = _resolve_answer_key_path(output_dir, answer_key_path)

    effective_control_seed = seed if control_seed is None else control_seed
    _validate_packet_roles(queries, block_roles, allowed_roles)
    eligible_controls = _scored_controls(
        control_queries, block_roles, packet_type="linker", allowed_roles=allowed_roles
    )
    chosen_controls = _select_controls(
        len(queries), eligible_controls, control_fraction=control_fraction, control_seed=effective_control_seed
    )
    combined = _stratified_interleave(
        [(False, q) for q in queries],
        [(True, q) for q in chosen_controls],
        max_items_per_shard=max_items_per_shard,
        order_seed=seed,
    )

    rng = random.Random(seed)
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
                "query_signature_id": raw["query"].get("signature_id"),
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
    packets_dir = output_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=False)
    shard_paths: list[str] = []
    key_items = iter(answer_key)
    for shard_index, shard in enumerate(shards, start=1):
        shard_path = packets_dir / f"linker_shard_{shard_index:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8", newline="\n") as handle:
            for rec in shard:
                handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        shard_paths.append(shard_path.relative_to(output_dir).as_posix())
        for _ in shard:
            next(key_items)["shard"] = shard_paths[-1]

    key_payload = {
        "packet_type": "linker",
        "order_seed": int(seed),
        "control_seed": int(effective_control_seed),
        "total_items": len(blind_records),
        "control_items": sum(1 for row in answer_key if row["is_control"]),
        "shards": shard_paths,
        "items": answer_key,
    }
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {
        "packet_type": "linker",
        "output_dir": str(output_dir),
        "answer_key_path": str(key_path),
        "shard_count": len(shard_paths),
        "total_items": len(blind_records),
        "control_items": key_payload["control_items"],
        "excluded_controls": len(control_queries) - len(eligible_controls),
        "shards": shard_paths,
    }


_DECIDED = frozenset({"same", "different"})
_VERDICTS = _DECIDED | {"insufficient_evidence", "cannot_tell", "abstain"}


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _judge_metrics(rows: Sequence[tuple[str | None, str | None]]) -> dict[str, Any]:
    decided = sum(verdict in _DECIDED for verdict, _ in rows)
    controls = [(verdict, truth) for verdict, truth in rows if truth is not None]
    same = sum(truth == "same" for _, truth in controls)
    different = len(controls) - same
    false_same = sum(verdict == "same" and truth == "different" for verdict, truth in controls)
    false_different = sum(verdict == "different" and truth == "same" for verdict, truth in controls)
    false_same_rate = _fraction(false_same, different)
    false_different_rate = _fraction(false_different, same)
    return {
        "items": len(rows),
        "missing": sum(verdict is None for verdict, _ in rows),
        "decided": decided,
        "decided_share": _fraction(decided, len(rows)),
        "controls": len(controls),
        "control_accuracy": _fraction(sum(verdict == truth for verdict, truth in controls), len(controls)),
        "same_controls": same,
        "different_controls": different,
        "false_same": false_same,
        "false_different": false_different,
        "false_same_rate": false_same_rate,
        "false_different_rate": false_different_rate,
        "false_same_minus_false_different": (
            false_same_rate - false_different_rate
            if false_same_rate is not None and false_different_rate is not None
            else None
        ),
    }


def _read_judge_verdicts(
    verdicts_path: Path, answer_key: Path
) -> tuple[dict[str, Any], dict[tuple[str, str, str], str | None]]:
    key = json.loads(answer_key.read_text(encoding="utf-8"))
    by_item = {row["item_id"]: row for row in key["items"]}
    if len(by_item) != len(key["items"]):
        raise ValueError(f"Duplicate item IDs in {answer_key}")
    verdicts: dict[str, Any] = {}
    with verdicts_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record["item_id"]
            if item_id not in by_item or item_id in verdicts:
                raise ValueError(f"Unknown or duplicate item {item_id!r} in {verdicts_path}:{line_number}")
            verdicts[item_id] = record

    by_shard: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    by_stratum: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    aligned: dict[tuple[str, str, str], str | None] = {}
    for item_id, item in by_item.items():
        record = verdicts.get(item_id, {})
        metadata = item.get("metadata", {})
        shard = item.get("shard")
        if shard is None and len(key["shards"]) == 1:
            shard = key["shards"][0]
        if shard not in key["shards"]:
            raise ValueError(f"Missing or invalid shard for {item_id!r} in {answer_key}")
        stratum = str(metadata.get("stratum", "unstratified"))
        if key["packet_type"] == "pair":
            slots = {"": ""}
            item_verdicts = {"": record["verdict"]} if record else {}
            gold_label = metadata.get("gold_label")
            gold_keys = [""] if gold_label in (1, "same") else []
            if item["is_control"] and gold_label not in (0, 1, "same", "different"):
                raise ValueError(f"Missing or invalid gold_label for control {item_id!r}")
        elif key["packet_type"] == "linker":
            slots = item["slot_to_candidate_key"]
            item_verdicts = record["candidate_verdicts"] if record else {}
            gold_keys = metadata.get("gold_candidate_keys")
            if item["is_control"] and (not isinstance(gold_keys, list) or set(gold_keys) - set(slots.values())):
                raise ValueError(f"Missing or invalid gold_candidate_keys for control {item_id!r}")
        else:
            raise ValueError(f"Unknown packet type in {answer_key}")
        if set(item_verdicts) - set(slots) or any(value not in _VERDICTS for value in item_verdicts.values()):
            raise ValueError(f"Invalid verdict or slot for {item_id!r} in {verdicts_path}")
        for slot, candidate in slots.items():
            identity = (normalize_block_key(item["block_key"]), item["source_id"], candidate)
            if identity in aligned:
                raise ValueError(f"Duplicate source and candidate {identity!r} in {answer_key}")
            verdict = item_verdicts.get(slot)
            truth = ("same" if candidate in gold_keys else "different") if item["is_control"] else None
            aligned[identity] = verdict
            by_shard[shard].append((verdict, truth))
            by_stratum[stratum].append((verdict, truth))
    return {
        "by_shard": {shard: _judge_metrics(rows) for shard, rows in sorted(by_shard.items())},
        "by_stratum": {stratum: _judge_metrics(rows) for stratum, rows in sorted(by_stratum.items())},
    }, aligned


def score_judges(judges: Sequence[tuple[str, Path, Path]]) -> dict[str, Any]:
    """Score JSONL verdicts and pairwise agreement across judge families.

    Args:
        judges: Family, verdict file, and private answer key for each judge.

    Returns:
        Per-judge control metrics and agreement on the same source items.
    """
    families = [family.strip().casefold() for family, _, _ in judges]
    if not all(families) or len(set(families)) < 2:
        raise ValueError("The Phase 0 scorer requires at least two distinct judge families")
    if len({path.resolve() for _, path, _ in judges}) != len(judges):
        raise ValueError("Each judge must supply a distinct verdict file")
    reports = []
    verdicts = []
    for family, (_, verdict_path, key_path) in zip(families, judges, strict=True):
        report, aligned = _read_judge_verdicts(verdict_path, key_path)
        reports.append({"family": family, "verdicts": str(verdict_path), **report})
        verdicts.append(aligned)
    agreement = []
    for i, j in combinations(range(len(judges)), 2):
        if families[i] == families[j]:
            continue
        left, right = verdicts[i], verdicts[j]
        overlap = left.keys() & right.keys()
        paired = [(left[key], right[key]) for key in overlap if left[key] in _DECIDED and right[key] in _DECIDED]
        left_counts = Counter(a for a, _ in paired)
        right_counts = Counter(b for _, b in paired)
        n = len(paired)
        matches = sum(a == b for a, b in paired)
        expected = sum(left_counts[decision] * right_counts[decision] for decision in _DECIDED)
        agreement.append(
            {
                "judges": [i, j],
                "families": [families[i], families[j]],
                "overlap": len(overlap),
                "both_decided": n,
                "both_decided_share": _fraction(n, len(overlap)),
                "agreement": _fraction(matches, n),
                "cohen_kappa": _fraction(matches * n - expected, n * n - expected),
            }
        )
    return {"judges": reports, "cross_family_agreement": agreement}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_p = subparsers.add_parser("validate", help="Validate a blind JSONL shard or directory of shards.")
    validate_p.add_argument("--path", type=Path, required=True)
    validate_p.add_argument("--packet-type", choices=["pair", "linker"], required=True)

    build_p = subparsers.add_parser("build", help="Build blind review packets from a JSON input file.")
    build_p.add_argument("--packet-type", choices=["pair", "linker"], required=True)
    build_p.add_argument("--items-json", type=Path, required=True)
    build_p.add_argument("--controls-json", type=Path, default=None)
    build_p.add_argument("--block-role-assignments", type=Path)
    build_p.add_argument("--allowed-role", action="append", default=[])
    build_p.add_argument("--output-dir", type=Path, required=True)
    build_p.add_argument("--answer-key-path", type=Path, default=None)
    build_p.add_argument("--seed", type=int, default=20261123)
    build_p.add_argument("--control-seed", type=int, default=None)
    build_p.add_argument("--control-fraction", type=float, default=0.12)
    build_p.add_argument("--max-items-per-shard", type=int, default=40)

    score_p = subparsers.add_parser("score", help="Score controls and agreement across judge families.")
    score_p.add_argument(
        "--judge", action="append", nargs=3, metavar=("FAMILY", "VERDICTS", "ANSWER_KEY"), required=True
    )
    score_p.add_argument("--output-json", type=Path)

    args = parser.parse_args(argv)
    if args.command == "score":
        report = score_judges([(family, Path(verdicts), Path(key)) for family, verdicts, key in args.judge])
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output_json is not None:
            args.output_json.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    if args.command == "validate":
        paths = sorted(args.path.rglob("*.jsonl")) if args.path.is_dir() else [args.path]
        all_violations: list[str] = []
        for path in paths:
            recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for violation in validate_blind_packet_records(recs, packet_type=args.packet_type):
                all_violations.append(f"{path.name}: {violation}")
        print(json.dumps({"files_checked": len(paths), "violations": all_violations}, indent=2))
        return 1 if all_violations else 0

    items = json.loads(args.items_json.read_text(encoding="utf-8"))
    controls = json.loads(args.controls_json.read_text(encoding="utf-8")) if args.controls_json else ()
    block_roles = load_block_roles(args.block_role_assignments) if args.block_role_assignments else None
    if args.packet_type == "pair":
        report = build_blind_pair_packets(
            items=items,
            control_items=controls,
            output_dir=args.output_dir,
            answer_key_path=args.answer_key_path,
            seed=args.seed,
            control_seed=args.control_seed,
            control_fraction=args.control_fraction,
            max_items_per_shard=args.max_items_per_shard,
            block_roles=block_roles,
            allowed_roles=args.allowed_role,
        )
    else:
        report = build_blind_linker_packets(
            queries=items,
            control_queries=controls,
            output_dir=args.output_dir,
            answer_key_path=args.answer_key_path,
            seed=args.seed,
            control_seed=args.control_seed,
            control_fraction=args.control_fraction,
            max_items_per_shard=args.max_items_per_shard,
            block_roles=block_roles,
            allowed_roles=args.allowed_role,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and.text import ORCID_DASH_CLASS, ORCID_PATTERN, normalize_orcid  # noqa: E402
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
                clean[key] = value
            elif isinstance(value, str) and value.strip().isdigit():
                clean[key] = int(value.strip())
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
    chosen_controls = _select_controls(
        len(items), control_items, control_fraction=control_fraction, control_seed=effective_control_seed
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
    for shard_index, shard in enumerate(shards, start=1):
        shard_path = packets_dir / f"pair_shard_{shard_index:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8", newline="\n") as handle:
            for rec in shard:
                handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        shard_paths.append(shard_path.relative_to(output_dir).as_posix())

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
) -> dict[str, Any]:
    """Build leak-checked blind linker candidate packets with shuffled neutral candidate slots."""
    if not 0.0 <= control_fraction <= 0.5:
        raise ValueError(f"control_fraction must be in [0.0, 0.5], got {control_fraction}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Blind packet builder requires a fresh output directory: {output_dir}")
    key_path = _resolve_answer_key_path(output_dir, answer_key_path)

    effective_control_seed = seed if control_seed is None else control_seed
    chosen_controls = _select_controls(
        len(queries), control_queries, control_fraction=control_fraction, control_seed=effective_control_seed
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
    for shard_index, shard in enumerate(shards, start=1):
        shard_path = packets_dir / f"linker_shard_{shard_index:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8", newline="\n") as handle:
            for rec in shard:
                handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        shard_paths.append(shard_path.relative_to(output_dir).as_posix())

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
        "shards": shard_paths,
    }


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
    build_p.add_argument("--output-dir", type=Path, required=True)
    build_p.add_argument("--answer-key-path", type=Path, default=None)
    build_p.add_argument("--seed", type=int, default=20261123)
    build_p.add_argument("--control-seed", type=int, default=None)
    build_p.add_argument("--control-fraction", type=float, default=0.12)
    build_p.add_argument("--max-items-per-shard", type=int, default=40)

    args = parser.parse_args(argv)
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
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

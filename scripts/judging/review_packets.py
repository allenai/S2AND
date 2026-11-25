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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import is_lowercase_sha256  # noqa: E402
from s2and.consts import PLACEHOLDER_YEARS, normalize_paper_year  # noqa: E402
from s2and.incremental_linking.contracts import canonical_json_digest  # noqa: E402
from s2and.text import ORCID_DASH_CLASS, ORCID_PATTERN, normalize_orcid  # noqa: E402
from scripts.common.block_roles import (  # noqa: E402
    BLOCK_BINDINGS_FIELD,
    GATE_BINDINGS_FIELD,
    GATE_CANDIDATE_ORCIDS,
    BlockRoles,
    load_block_roles,
    normalize_block_key,
)
from scripts.common.email_redaction import find_plaintext_emails, scrub_text_emails  # noqa: E402
from scripts.common.identity_ledger import (  # noqa: E402
    EVIDENCE_SHA256_FIELD,
    BlockEligibility,
    IdentityLedger,
    evidence_identities,
    gate_veto_identities,
    ledger_for_roles,
)
from scripts.common.linker_seeds import GATE_SEED_DIGESTS  # noqa: E402
from scripts.common.orcid_references import EXCLUDED_ORCID_GROUPS  # noqa: E402
from scripts.judging.gate_exemplars import frozen_packet_papers, resolve_gate_exemplars  # noqa: E402

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
        if ("@" in value or "\uff20" in value) and find_plaintext_emails(value):
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
    packet_type: str,
    control_fraction: float,
    control_seed: int,
    balance_by_score_band: bool,
    minimum: int = 2,
) -> list[Mapping[str, Any]]:
    if not items_count or control_fraction <= 0.0:
        return []
    rng = random.Random(control_seed)
    target = max(minimum, round(items_count * control_fraction / (1.0 - control_fraction)))
    if balance_by_score_band:
        target += target % 2
    by_class: dict[str, tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]] = {}
    for item in control_items:
        metadata = item.get("metadata", {})
        band = metadata.get("score_band_v14") if balance_by_score_band else ""
        if not isinstance(band, str) or (balance_by_score_band and not band.strip()):
            raise ValueError("Score-band balancing requires metadata.score_band_v14 on every eligible control")
        positive = (
            metadata.get("gold_label") in (1, "same")
            if packet_type == "pair"
            else bool(metadata.get("gold_candidate_keys"))
        )
        by_class.setdefault(band, ([], []))[int(positive)].append(item)
    for negative, positive in by_class.values():
        rng.shuffle(negative)
        rng.shuffle(positive)

    pairs = [(negative, positive) for _, (negative, positive) in sorted(by_class.items()) if negative and positive]
    if not pairs:
        scope = " in a shared score band" if balance_by_score_band else ""
        raise ValueError(f"Expected both verified control classes{scope}")
    selected: list[Mapping[str, Any]] = []
    while pairs and len(selected) + 2 <= target:
        for negative, positive in pairs:
            if len(selected) + 2 > target:
                break
            selected.extend((negative.pop(), positive.pop()))
        pairs = [(negative, positive) for negative, positive in pairs if negative and positive]
    if not balance_by_score_band and target - len(selected) == 1:
        extra = max(by_class[""], key=len)
        if extra:
            selected.append(extra.pop())
    if len(selected) != target:
        raise ValueError(f"Insufficient balanced controls: selected {len(selected)}, needed {target}")
    return selected


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


def _require_ledger_packet_roles(roles: BlockRoles | None, ledger: IdentityLedger | None) -> None:
    if roles is not None and roles.evidence_sha256 is not None and (ledger is None or ledger.roles != roles):
        raise ValueError("Packet roles do not match the identity ledger")


def _validate_natural_items(
    items: Sequence[Mapping[str, Any]], ledger: IdentityLedger | None, packet_type: str
) -> dict[str, set[str]]:
    named: dict[str, set[str]] = defaultdict(set)
    if ledger is None:
        return named
    views: dict[str, BlockEligibility] = {}
    validated: set[tuple[str, str]] = set()
    for item in items:
        block = normalize_block_key(item["block_key"])
        if block not in views:
            views[block] = ledger.for_block(block)
        if packet_type == "pair":
            papers = [item["paper_a"], item["paper_b"]]
        else:
            papers = [item["query"]]
            if views[block].role == "gate_thresh":
                named.setdefault(block, set())
            for candidate in item["candidates"]:
                exemplars = candidate["exemplars"]
                if views[block].role != "gate_thresh":
                    papers.extend(exemplars)
                    continue
                component = (block, candidate["candidate_key"])
                if component in validated:
                    continue
                for sid in candidate["member_signature_ids"]:
                    if views[block].eligibility(sid).identity_veto:
                        raise ValueError(f"Natural judging item {item['source_id']!r} contains a vetoed identity")
                    named[block].update(gate_veto_identities(views[block].rows[sid]))
                validated.add(component)
        if any(not views[block].eligibility(str(paper["signature_id"])).fit_ok for paper in papers):
            raise ValueError(f"Natural judging item {item['source_id']!r} contains an ineligible signature")
    return named


def _scored_controls(
    items: Sequence[Mapping[str, Any]],
    block_roles: BlockRoles | None,
    *,
    packet_type: str,
    allowed_roles: Collection[str],
    ledger: IdentityLedger | None = None,
) -> list[Mapping[str, Any]]:
    result = []
    views: dict[str, BlockEligibility] = {}
    for item in items:
        block = str(item["block_key"])
        if block_roles is not None and block_roles.role_for(block) not in allowed_roles:
            continue
        candidates = item["candidates"] if packet_type == "linker" else ()
        if packet_type == "pair":
            papers = [item["paper_a"], item["paper_b"]]
        else:
            if not candidates or any(not cand["exemplars"] for cand in candidates):
                continue
            papers = [item["query"], *(paper for cand in candidates for paper in cand["exemplars"])]
        if ledger is None:
            orcids = [normalize_orcid(str(paper.get("orcid") or paper.get("author_orcid") or "")) for paper in papers]
            if not all(
                orcid
                and orcid not in EXCLUDED_ORCID_GROUPS
                and paper.get("scored", True) is True
                and (block_roles is None or block_roles.scores(block, orcid))
                for paper, orcid in zip(papers, orcids, strict=True)
            ):
                continue
        else:
            key = normalize_block_key(block)
            if key not in views:
                views[key] = ledger.for_block(key)
            view = views[key]
            if packet_type == "linker" and view.role == "gate_thresh":
                query_orcid = view.eligibility(str(item["query"]["signature_id"])).score_orcid
                component_orcids = [view.component_label(candidate["member_signature_ids"]) for candidate in candidates]
                if not query_orcid or any(
                    not orcid
                    or not any(
                        orcid in evidence_identities(view.rows[paper["signature_id"]])
                        for paper in candidate["exemplars"]
                    )
                    for candidate, orcid in zip(candidates, component_orcids, strict=True)
                ):
                    continue
                orcids = [
                    query_orcid,
                    *(
                        orcid
                        for candidate, orcid in zip(candidates, component_orcids, strict=True)
                        for _ in candidate["exemplars"]
                    ),
                ]
            else:
                orcids = [view.eligibility(str(paper["signature_id"])).score_orcid for paper in papers]
            if not all(orcids):
                continue
            if view.role != "gate_thresh" and any(
                not isinstance(candidate.get("member_signature_ids"), list)
                or not {str(paper["signature_id"]) for paper in candidate["exemplars"]}.issubset(
                    candidate["member_signature_ids"]
                )
                or view.component_label(candidate["member_signature_ids"]) is None
                for candidate in candidates
            ):
                continue
        metadata = dict(item.get("metadata", {}))
        if packet_type == "pair":
            gold = int(orcids[0] == orcids[1])
            supplied = metadata.get("gold_label", gold)
            if supplied not in (gold, "same" if gold else "different"):
                raise ValueError(f"Control {item['source_id']!r} gold_label contradicts its verified ORCIDs")
            metadata["gold_label"] = gold
        else:
            remaining_orcids = iter(orcids[1:])
            gold_keys = []
            for candidate in candidates:
                identities = {next(remaining_orcids) for _ in candidate["exemplars"]}
                if len(identities) != 1:
                    break
                if orcids[0] in identities:
                    gold_keys.append(str(candidate["candidate_key"]))
            else:
                supplied_keys = metadata.get("gold_candidate_keys", gold_keys)
                if not isinstance(supplied_keys, list) or set(supplied_keys) != set(gold_keys):
                    raise ValueError(f"Control {item['source_id']!r} gold_candidate_keys contradict verified ORCIDs")
                metadata["gold_candidate_keys"] = gold_keys
                result.append({**item, "metadata": metadata})
            continue
        result.append({**item, "metadata": metadata})
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


def _load_packet_displays(
    items: Sequence[tuple[bool, Mapping[str, Any]]],
    *,
    packet_type: str,
    snapshot: Path | None,
    ledger: IdentityLedger | None,
    max_abstract_chars: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    ids: dict[str, set[str]] = defaultdict(set)
    for _, item in items:
        if packet_type == "pair":
            refs = [item["paper_a"], item["paper_b"]]
        else:
            refs = [item["query"], *(ex for candidate in item["candidates"] for ex in candidate.get("exemplars", ()))]
        ids[normalize_block_key(item["block_key"])].update(str(ref["signature_id"]) for ref in refs)
    if not ids:
        return {}
    papers = frozen_packet_papers(ids, snapshot=snapshot, evidence=ledger.evidence if ledger is not None else None)
    return {key: sanitize_paper_item(paper, max_abstract_chars=max_abstract_chars) for key, paper in papers.items()}


def _validate_linker_membership(items: Sequence[Mapping[str, Any]]) -> None:
    for item in items:
        query = str(item["query"]["signature_id"])
        for candidate in item["candidates"]:
            members = candidate.get("member_signature_ids")
            if not isinstance(members, (list, tuple)) or not members:
                raise ValueError(f"Linker candidate {candidate['candidate_key']!r} requires component members")
            member_ids = set(members)
            exemplars = {str(paper["signature_id"]) for paper in candidate.get("exemplars", ())}
            if query in member_ids or query in exemplars:
                raise ValueError(f"Linker query {query!r} is a member of candidate {candidate['candidate_key']!r}")
            if not exemplars <= member_ids:
                raise ValueError(f"Linker candidate {candidate['candidate_key']!r} has exemplars outside its component")


def build_blind_pair_packets(
    *,
    items: Sequence[Mapping[str, Any]],
    control_items: Sequence[Mapping[str, Any]] = (),
    output_dir: Path,
    answer_key_path: Path | None = None,
    seed: int = 20261123,
    control_seed: int | None = None,
    control_fraction: float = 0.12,
    balance_controls_by_score_band: bool = False,
    max_items_per_shard: int = 40,
    max_bytes_per_shard: int = 180_000,
    max_abstract_chars: int = 1200,
    block_roles: BlockRoles | None = None,
    allowed_roles: Collection[str] = (),
    ledger: IdentityLedger | None = None,
    snapshot: Path | None = None,
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
    _require_ledger_packet_roles(block_roles, ledger)
    _validate_natural_items(items, ledger, "pair")
    eligible_controls = _scored_controls(
        control_items, block_roles, packet_type="pair", allowed_roles=allowed_roles, ledger=ledger
    )
    chosen_controls = _select_controls(
        len(items),
        eligible_controls,
        packet_type="pair",
        control_fraction=control_fraction,
        control_seed=effective_control_seed,
        balance_by_score_band=balance_controls_by_score_band,
        minimum=_MIN_CONTROL_REQUESTS if ledger is not None else 2,
    )
    combined = _stratified_interleave(
        [(False, item) for item in items],
        [(True, item) for item in chosen_controls],
        max_items_per_shard=max_items_per_shard,
        order_seed=seed,
    )
    displays = _load_packet_displays(
        combined, packet_type="pair", snapshot=snapshot, ledger=ledger, max_abstract_chars=max_abstract_chars
    )

    rng = random.Random(seed)
    blind_records: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for index, (is_control, raw) in enumerate(combined, start=1):
        item_id = f"pair_{index:05d}"
        block = normalize_block_key(raw["block_key"])
        swap_sides = rng.random() < 0.5
        raw_a = raw["paper_b"] if swap_sides else raw["paper_a"]
        raw_b = raw["paper_a"] if swap_sides else raw["paper_b"]
        blind_records.append(
            {
                "item_id": item_id,
                "block_key": block,
                "paper_a": displays[block, str(raw_a["signature_id"])],
                "paper_b": displays[block, str(raw_b["signature_id"])],
            }
        )
        answer_key.append(
            {
                "item_id": item_id,
                "source_id": str(raw["source_id"]),
                "display_sha256": canonical_json_digest(blind_records[-1]),
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

    bindings = (
        {BLOCK_BINDINGS_FIELD: ledger.roles.bindings_for({item["block_key"] for item in answer_key})}
        if ledger is not None
        else {}
    )
    key_payload = {
        **bindings,
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
        **bindings,
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
    balance_controls_by_score_band: bool = False,
    max_items_per_shard: int = 25,
    max_bytes_per_shard: int = 180_000,
    max_abstract_chars: int = 900,
    block_roles: BlockRoles | None = None,
    allowed_roles: Collection[str] = (),
    ledger: IdentityLedger | None = None,
    linker_task_manifest: Path | None = None,
    snapshot: Path | None = None,
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
    _require_ledger_packet_roles(block_roles, ledger)
    seed_digests: dict[str, str] = {}
    roles = ledger.roles if ledger is not None else block_roles
    allowed_controls = [
        item for item in control_queries if roles is None or roles.role_for(item["block_key"]) in allowed_roles
    ]
    if ledger is not None:
        queries, seed_digests = resolve_gate_exemplars(queries, ledger.roles, linker_task_manifest)
        allowed_controls, _ = resolve_gate_exemplars(
            allowed_controls, ledger.roles, linker_task_manifest, control_ledger=ledger
        )
    _validate_linker_membership(queries)
    _validate_linker_membership(allowed_controls)
    gate_candidates = _validate_natural_items(queries, ledger, "linker")
    eligible_controls = _scored_controls(
        allowed_controls, block_roles, packet_type="linker", allowed_roles=allowed_roles, ledger=ledger
    )
    chosen_controls = _select_controls(
        len(queries),
        eligible_controls,
        packet_type="linker",
        control_fraction=control_fraction,
        control_seed=effective_control_seed,
        balance_by_score_band=balance_controls_by_score_band,
        minimum=_MIN_CONTROL_REQUESTS if ledger is not None else 2,
    )
    combined = _stratified_interleave(
        [(False, q) for q in queries],
        [(True, q) for q in chosen_controls],
        max_items_per_shard=max_items_per_shard,
        order_seed=seed,
    )
    displays = _load_packet_displays(
        combined, packet_type="linker", snapshot=snapshot, ledger=ledger, max_abstract_chars=max_abstract_chars
    )

    rng = random.Random(seed)
    blind_records: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for index, (is_control, raw) in enumerate(combined, start=1):
        item_id = f"linker_{index:05d}"
        block = normalize_block_key(raw["block_key"])
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
                    "exemplars": [displays[block, str(ex["signature_id"])] for ex in cand.get("exemplars", ())],
                }
            )
        blind_records.append(
            {
                "item_id": item_id,
                "block_key": block,
                "query": displays[block, str(raw["query"]["signature_id"])],
                "candidates": blind_candidates,
            }
        )
        answer_key.append(
            {
                "item_id": item_id,
                "source_id": str(raw["source_id"]),
                "display_sha256": canonical_json_digest(blind_records[-1]),
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

    bindings = (
        {BLOCK_BINDINGS_FIELD: ledger.roles.bindings_for({item["block_key"] for item in answer_key})}
        if ledger is not None
        else {}
    )
    gate_bindings = (
        {
            GATE_BINDINGS_FIELD: ledger.roles.gate_judgment_bindings(gate_candidates),
            GATE_CANDIDATE_ORCIDS: {block: sorted(identities) for block, identities in gate_candidates.items()},
            GATE_SEED_DIGESTS: {block: seed_digests[block] for block in gate_candidates},
        }
        if ledger is not None and gate_candidates
        else {}
    )
    key_payload = {
        **bindings,
        **gate_bindings,
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
        **bindings,
        "packet_type": "linker",
        "output_dir": str(output_dir),
        "answer_key_path": str(key_path),
        "shard_count": len(shard_paths),
        "total_items": len(blind_records),
        "control_items": key_payload["control_items"],
        "excluded_controls": len(control_queries) - len(eligible_controls),
        "shards": shard_paths,
    }


_CONTROL_PASS_MINIMUM = 0.90
_MIN_CONTROL_REQUESTS = 10
RequestKey = tuple[str, str, str]
JudgmentKey = tuple[str, str, str, str]
_DECIDED = frozenset({"same", "different"})
_VERDICTS = _DECIDED | {"insufficient_evidence", "cannot_tell", "abstain"}


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _judge_metrics(
    rows: Sequence[tuple[str | None, str | None]],
    control_requests: Sequence[tuple[bool, bool]],
    zero_candidate_requests: int = 0,
) -> dict[str, Any]:
    decided = sum(verdict in _DECIDED for verdict, _ in rows)
    controls = [(verdict, truth) for verdict, truth in rows if truth is not None]
    same = sum(truth == "same" for _, truth in controls)
    different = len(controls) - same
    false_same = sum(verdict == "same" and truth == "different" for verdict, truth in controls)
    false_different = sum(verdict == "different" and truth == "same" for verdict, truth in controls)
    false_same_rate = _fraction(false_same, different)
    false_different_rate = _fraction(false_different, same)
    same_accuracy = _fraction(sum(verdict == truth == "same" for verdict, truth in controls), same)
    different_accuracy = _fraction(sum(verdict == truth == "different" for verdict, truth in controls), different)
    lower_class_accuracy = (
        min(same_accuracy, different_accuracy) if same_accuracy is not None and different_accuracy is not None else None
    )
    request_accuracy = _fraction(sum(success for _, success in control_requests), len(control_requests))
    positive_requests = [success for positive, success in control_requests if positive]
    negative_requests = [success for positive, success in control_requests if not positive]
    positive_request_accuracy = _fraction(sum(positive_requests), len(positive_requests))
    negative_request_accuracy = _fraction(sum(negative_requests), len(negative_requests))
    return {
        "items": len(rows),
        "missing": sum(verdict is None for verdict, _ in rows),
        "decided": decided,
        "decided_share": _fraction(decided, len(rows)),
        "controls": len(controls),
        "control_accuracy": _fraction(sum(verdict == truth for verdict, truth in controls), len(controls)),
        "control_requests": len(control_requests),
        "positive_control_requests": len(positive_requests),
        "negative_control_requests": len(negative_requests),
        "positive_control_request_accuracy": positive_request_accuracy,
        "negative_control_request_accuracy": negative_request_accuracy,
        "zero_candidate_requests": zero_candidate_requests,
        "control_request_accuracy": request_accuracy,
        "same_controls": same,
        "same_control_accuracy": same_accuracy,
        "different_control_accuracy": different_accuracy,
        "lower_control_class_accuracy": lower_class_accuracy,
        "passes_control_bar": (
            len(control_requests) >= _MIN_CONTROL_REQUESTS
            and lower_class_accuracy is not None
            and request_accuracy is not None
            and positive_request_accuracy is not None
            and negative_request_accuracy is not None
            and min(lower_class_accuracy, request_accuracy, positive_request_accuracy, negative_request_accuracy)
            >= _CONTROL_PASS_MINIMUM
        ),
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
    verdicts_path: Path, answer_key: Path, key: Mapping[str, Any], *, stale_control_blocks: Collection[str]
) -> tuple[dict[str, Any], dict[JudgmentKey, tuple[str, str | None, bool]], dict[RequestKey, tuple[str, bool]]]:
    by_item = {row["item_id"]: row for row in key["items"]}
    if len(by_item) != len(key["items"]):
        raise ValueError(f"Duplicate item IDs in {answer_key}")
    if any(not is_lowercase_sha256(item.get("display_sha256")) for item in by_item.values()):
        raise ValueError(f"Missing or invalid display hash in {answer_key}")
    verdicts: dict[str, Any] = {}
    with verdicts_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            item_id = record["item_id"]
            if item_id not in by_item or item_id in verdicts:
                raise ValueError(f"Unknown or duplicate item {item_id!r} in {verdicts_path}:{line_number}")
            if record.get("display_sha256") != by_item[item_id]["display_sha256"]:
                raise ValueError(f"Verdict display hash differs from the key for {item_id!r} in {verdicts_path}")
            verdicts[item_id] = record

    by_shard: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    by_stratum: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    shard_controls: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    stratum_controls: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    zero_by_shard: Counter[str] = Counter()
    zero_by_stratum: Counter[str] = Counter()
    aligned: dict[JudgmentKey, tuple[str, str | None, bool]] = {}
    requests: dict[RequestKey, tuple[str, bool]] = {}
    seen_requests: set[RequestKey] = set()
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
        request_key: RequestKey = (key["packet_type"], normalize_block_key(item["block_key"]), item["source_id"])
        if request_key in seen_requests:
            raise ValueError(f"Duplicate source request {request_key!r} in {answer_key}")
        seen_requests.add(request_key)
        if item["is_control"] and request_key[1] in stale_control_blocks:
            continue
        by_shard[shard]
        by_stratum[stratum]
        if not slots:
            zero_by_shard[shard] += 1
            zero_by_stratum[stratum] += 1
        if not item["is_control"]:
            requests[request_key] = (stratum, not slots)
        control_success = bool(slots)
        for slot, candidate in slots.items():
            identity = (*request_key, candidate)
            if identity in aligned:
                raise ValueError(f"Duplicate source and candidate {identity!r} in {answer_key}")
            verdict = item_verdicts.get(slot)
            truth = ("same" if candidate in gold_keys else "different") if item["is_control"] else None
            aligned[identity] = (stratum, verdict, item["is_control"])
            control_success &= verdict == truth
            by_shard[shard].append((verdict, truth))
            by_stratum[stratum].append((verdict, truth))
        if item["is_control"]:
            outcome = (bool(gold_keys), control_success)
            shard_controls[shard].append(outcome)
            stratum_controls[stratum].append(outcome)
    return (
        {
            "overall": _judge_metrics(
                [row for rows in by_shard.values() for row in rows],
                [outcome for outcomes in shard_controls.values() for outcome in outcomes],
                zero_by_shard.total(),
            ),
            "by_shard": {
                shard: _judge_metrics(rows, shard_controls[shard], zero_by_shard[shard])
                for shard, rows in sorted(by_shard.items())
            },
            "by_stratum": {
                stratum: _judge_metrics(rows, stratum_controls[stratum], zero_by_stratum[stratum])
                for stratum, rows in sorted(by_stratum.items())
            },
        },
        aligned,
        requests,
    )


def _paired_judge_metrics(rows: Sequence[tuple[str | None, str | None]]) -> dict[str, Any]:
    present = [(a, b) for a, b in rows if a is not None and b is not None]
    decided = [(a, b) for a, b in present if a in _DECIDED and b in _DECIDED]
    left_counts = Counter(a for a, _ in decided)
    right_counts = Counter(b for _, b in decided)
    n = len(decided)
    matches = sum(a == b for a, b in decided)
    expected = sum(left_counts[value] * right_counts[value] for value in _DECIDED)
    both_abstain = sum(a not in _DECIDED and b not in _DECIDED for a, b in present)
    return {
        "overlap": len(rows),
        "paired_verdicts": len(present),
        "paired_verdict_share": _fraction(len(present), len(rows)),
        "both_decided": n,
        "both_decided_share": _fraction(n, len(rows)),
        "unresolved_share": _fraction(len(rows) - matches, len(rows)),
        "agreement": _fraction(matches, n),
        "cohen_kappa": _fraction(matches * n - expected, n * n - expected),
        "decided_vs_abstain": len(present) - n - both_abstain,
        "agreement_including_abstentions": _fraction(matches + both_abstain, len(present)),
    }


def score_judges(judges: Sequence[tuple[str, Path, Path]], *, ledger: IdentityLedger | None = None) -> dict[str, Any]:
    """Score JSONL verdicts and pairwise agreement across judge families.

    Args:
        judges: Family, verdict file, and private answer key for each judge.
            List the two primary families first, followed by any audit families.

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
    requests = []
    scored_blocks: set[str] = set()
    gate_seed_digests: dict[str, str] = {}
    gate_candidates: dict[str, set[str]] = defaultdict(set)
    gate_votes: dict[tuple[str, str, str], dict[str, set[str | None]]] = defaultdict(lambda: defaultdict(set))
    for family, (_, verdict_path, key_path) in zip(families, judges, strict=True):
        key = json.loads(key_path.read_text(encoding="utf-8"))
        gate_blocks: set[str] = set()
        stale_control_blocks: set[str] = set()
        if ledger is not None:
            blocks = {normalize_block_key(item["block_key"]) for item in key["items"]}
            natural_blocks = {normalize_block_key(item["block_key"]) for item in key["items"] if not item["is_control"]}
            if key["packet_type"] == "linker":
                gate_blocks = {block for block in natural_blocks if ledger.roles.role_for(block) == "gate_thresh"}
            if gate_blocks or GATE_BINDINGS_FIELD in key:
                candidates = key.get(GATE_CANDIDATE_ORCIDS)
                if (
                    not gate_blocks
                    or not isinstance(candidates, dict)
                    or candidates.keys() != gate_blocks
                    or key.get(GATE_BINDINGS_FIELD) != ledger.roles.gate_judgment_bindings(candidates)
                ):
                    raise ValueError(f"{family} gate judging key block provenance does not match the identity ledger")
                for block, identities in candidates.items():
                    gate_candidates[block].update(identities)
                # A changed control-only block cannot supply accuracy metrics for the current ledger.
                saved = key.get(BLOCK_BINDINGS_FIELD)
                current = ledger.roles.bindings_for(blocks & ledger.roles.block_to_role.keys())
                if (
                    not isinstance(saved, dict)
                    or saved.keys() != blocks
                    or any(saved[block] != current[block] for block in natural_blocks)
                ):
                    raise ValueError(f"{family} judging key block provenance does not match the identity ledger")
                stale_control_blocks = {
                    block for block in blocks if block not in current or saved[block] != current[block]
                }
            else:
                ledger.roles.require_bindings(key, blocks, context=f"{family} judging key")
            scored_blocks.update(blocks - stale_control_blocks)
        elif BLOCK_BINDINGS_FIELD in key or EVIDENCE_SHA256_FIELD in key:
            raise ValueError("Ledger-backed judging keys require frozen roles and identity evidence")
        report, aligned, natural_requests = _read_judge_verdicts(
            verdict_path, key_path, key, stale_control_blocks=stale_control_blocks
        )
        reports.append({"family": family, "verdicts": str(verdict_path), **report})
        verdicts.append(aligned)
        requests.append(natural_requests)
        if gate_blocks:
            for item in key["items"]:
                block = normalize_block_key(item["block_key"])
                if item["is_control"] or block not in gate_blocks:
                    continue
                digest = key.get(GATE_SEED_DIGESTS, {}).get(block)
                if not is_lowercase_sha256(digest) or gate_seed_digests.setdefault(block, digest) != digest:
                    raise ValueError("Gate threshold judging keys must use the same frozen seed assignments")
                signature = item["query_signature_id"]
                if not isinstance(signature, str) or not signature:
                    raise ValueError("Gate threshold judging keys must identify each query signature")
                for candidate in item["slot_to_candidate_key"].values():
                    _, verdict, _ = aligned["linker", block, item["source_id"], candidate]
                    gate_votes[block, signature, candidate][family].add(verdict)
    agreement = []
    for i, j in combinations(range(len(judges)), 2):
        if families[i] == families[j]:
            continue
        left, right = verdicts[i], verdicts[j]
        left_natural = {key for key, (_, _, control) in left.items() if not control}
        right_natural = {key for key, (_, _, control) in right.items() if not control}
        shared_natural = left_natural & right_natural
        left_requests, right_requests = requests[i], requests[j]
        shared_requests = left_requests.keys() & right_requests.keys()
        by_task: dict[str, dict[str, list[tuple[str | None, str | None]]]] = defaultdict(lambda: defaultdict(list))
        zero_candidate_requests = 0
        planned_requests = left_requests.keys() | right_requests.keys()
        for request in sorted(planned_requests):
            default = left_requests.get(request) or right_requests[request]
            stratum, left_zero = left_requests.get(request, default)
            right_stratum, right_zero = right_requests.get(request, default)
            if stratum != right_stratum:
                raise ValueError(f"Mismatched strata for jointly judged item {request!r}")
            if left_zero or right_zero:
                zero_candidate_requests += 1
                by_task[request[0]][stratum].append((None, None))
        planned_natural = left_natural | right_natural
        for key in sorted(planned_natural):
            planned_stratum = (left[key] if key in left_natural else right[key])[0]
            missing: tuple[str, str | None, bool] = (planned_stratum, None, False)
            stratum, a, _ = left[key] if key in left_natural else missing
            right_stratum, b, _ = right[key] if key in right_natural else missing
            if stratum != right_stratum:
                raise ValueError(f"Mismatched strata for jointly judged item {key!r}")
            by_task[key[0]][stratum].append((a, b))
        agreement.append(
            {
                "judges": [i, j],
                "families": [families[i], families[j]],
                "natural_key_coverage": _fraction(len(shared_natural), len(left_natural | right_natural)),
                "missing_natural_keys": {
                    families[i]: sorted(right_natural - left_natural),
                    families[j]: sorted(left_natural - right_natural),
                },
                "request_coverage": _fraction(len(shared_requests), len(left_requests.keys() | right_requests.keys())),
                "missing_request_keys": {
                    families[i]: sorted(right_requests.keys() - left_requests.keys()),
                    families[j]: sorted(left_requests.keys() - right_requests.keys()),
                },
                "planned_zero_candidate_requests": zero_candidate_requests,
                **_paired_judge_metrics(
                    [row for strata in by_task.values() for rows in strata.values() for row in rows]
                ),
                "by_task": {
                    task: {stratum: _paired_judge_metrics(rows) for stratum, rows in sorted(strata.items())}
                    for task, strata in sorted(by_task.items())
                },
            }
        )
    report = {"judges": reports, "cross_family_agreement": agreement}
    if ledger is not None:
        report[BLOCK_BINDINGS_FIELD] = ledger.roles.bindings_for(scored_blocks)
    if ledger is not None and gate_votes:
        primary = tuple(dict.fromkeys(families))[:2]
        primary_candidates = {key for key, votes in gate_votes.items() if any(family in votes for family in primary)}
        audit_coverage = {}
        for family in dict.fromkeys(families):
            if family in primary:
                continue
            planned = [gate_votes[key][family] for key in primary_candidates if family in gate_votes[key]]
            audit_coverage[family] = {
                "planned_candidates": len(planned),
                "planned_share": _fraction(len(planned), len(primary_candidates)),
                "candidates_with_verdict": sum(bool(votes - {None}) for votes in planned),
            }
        labels = []
        for (block, query, candidate), votes in sorted(gate_votes.items()):
            consensus = set().union(*votes.values())
            if not set(primary) <= votes.keys() or len(consensus) != 1 or not consensus <= _DECIDED:
                continue
            labels.append(
                {
                    "dataset": block,
                    "query_signature_id": query,
                    "candidate_component_key": candidate,
                    "verdict": next(iter(consensus)),
                }
            )
        report[GATE_BINDINGS_FIELD] = ledger.roles.gate_judgment_bindings(gate_candidates)
        report[GATE_CANDIDATE_ORCIDS] = {block: sorted(identities) for block, identities in gate_candidates.items()}
        report[GATE_SEED_DIGESTS] = gate_seed_digests
        report["gate_thresh_labels"] = labels
        report["gate_thresh_planned_candidates"] = len(primary_candidates)
        report["gate_thresh_primary_families"] = list(primary)
        report["gate_thresh_audit_coverage"] = audit_coverage
    return report


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
    build_p.add_argument("--identity-evidence", type=Path)
    build_p.add_argument("--snapshot", type=Path, help="Frozen Arrow collection when building without a ledger.")
    build_p.add_argument(
        "--linker-task-manifest", type=Path, help="Frozen linker_tasks.json for ledger-backed gate packets."
    )
    build_p.add_argument("--block-role-assignments", type=Path, required=True)
    build_p.add_argument("--allowed-role", action="append", required=True)
    build_p.add_argument("--output-dir", type=Path, required=True)
    build_p.add_argument("--answer-key-path", type=Path, default=None)
    build_p.add_argument("--seed", type=int, default=20261123)
    build_p.add_argument("--control-seed", type=int, default=None)
    build_p.add_argument("--control-fraction", type=float, default=0.12)
    build_p.add_argument(
        "--balance-controls-by-score-band",
        action="store_true",
        help="Draw same/different or linkable/unlinkable controls within each metadata.score_band_v14 value.",
    )
    build_p.add_argument("--max-items-per-shard", type=int, default=40)

    score_p = subparsers.add_parser("score", help="Score controls and agreement across judge families.")
    score_p.add_argument(
        "--judge", action="append", nargs=3, metavar=("FAMILY", "VERDICTS", "ANSWER_KEY"), required=True
    )
    score_p.add_argument("--identity-evidence", type=Path)
    score_p.add_argument("--block-role-assignments", type=Path)
    score_p.add_argument("--output-json", type=Path)

    args = parser.parse_args(argv)
    if args.command == "score":
        roles = load_block_roles(args.block_role_assignments) if args.block_role_assignments else None
        report = score_judges(
            [(family, Path(verdicts), Path(key)) for family, verdicts, key in args.judge],
            ledger=ledger_for_roles(args.identity_evidence, roles),
        )
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
    block_roles = load_block_roles(args.block_role_assignments)
    ledger = ledger_for_roles(args.identity_evidence, block_roles)
    if args.packet_type == "pair":
        report = build_blind_pair_packets(
            items=items,
            control_items=controls,
            output_dir=args.output_dir,
            answer_key_path=args.answer_key_path,
            seed=args.seed,
            control_seed=args.control_seed,
            control_fraction=args.control_fraction,
            balance_controls_by_score_band=args.balance_controls_by_score_band,
            max_items_per_shard=args.max_items_per_shard,
            block_roles=block_roles,
            allowed_roles=args.allowed_role,
            ledger=ledger,
            snapshot=args.snapshot,
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
            balance_controls_by_score_band=args.balance_controls_by_score_band,
            max_items_per_shard=args.max_items_per_shard,
            block_roles=block_roles,
            allowed_roles=args.allowed_role,
            ledger=ledger,
            linker_task_manifest=args.linker_task_manifest,
            snapshot=args.snapshot,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

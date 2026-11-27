"""Build blind author-pair packages and maintain a versioned bank of verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

from s2and._atomic_io import write_fresh_json
from s2and.incremental_linking.contracts import canonical_json_digest
from s2and.text import normalize_orcid
from scripts.common.block_roles import load_block_roles, normalize_block_key
from scripts.common.identity_ledger import IdentityLedger, load_identity_ledger
from scripts.judging.gate_exemplars import frozen_packet_papers
from scripts.judging.review_packets import render_author_pair_package, resolve_answer_key_path, sanitize_paper_item

Verdict = Literal["same", "different", "cannot_tell"]
Gold = Literal["same", "different"]
ControlKind = Literal["same", "different", "different_same_name"]
VERDICTS: tuple[Verdict, ...] = ("same", "different", "cannot_tell")
CLASSES: tuple[Gold, ...] = ("same", "different")
CONTROL_KINDS: tuple[ControlKind, ...] = (*CLASSES, "different_same_name")
CONFIDENCES = ("low", "medium", "high")
FORMAT_VERSION = 1
VERSION_NAME = re.compile(r"[0-9]{8}")
LOGGER = logging.getLogger(__name__)

_STRING_LIST = pa.list_(pa.string())
_OCCURRENCE = [
    ("package", pa.string()),
    ("item", pa.int64()),
    ("stratum", pa.string()),
    ("target_component", pa.string()),
    ("target_signature_ids", _STRING_LIST),
    ("model", pa.string()),
    ("arm", pa.string()),
    ("score", pa.float64()),
    ("cutoff", pa.float64()),
    ("inclusion_probability", pa.float64()),
    ("control", pa.bool_()),
    ("gold_verdict", pa.string()),
    ("disputed_control", pa.bool_()),
]
_PAIR = [
    ("pair_id", pa.string()),
    ("block", pa.string()),
    ("query_signature_id", pa.string()),
    ("shown_group_signature_ids", _STRING_LIST),
]
JUDGMENTS_SCHEMA = pa.schema(
    [
        *_PAIR,
        *_OCCURRENCE,
        ("judge", pa.string()),
        ("verdict", pa.string()),
        ("confidence", pa.string()),
        ("key_sha256", pa.string()),
        ("verdict_sha256", pa.string()),
    ]
)
PAIRS_SCHEMA = pa.schema(
    [
        *_PAIR,
        ("occurrences", pa.list_(pa.struct(_OCCURRENCE))),
        (
            "consensus",
            pa.list_(
                pa.struct(
                    [
                        ("package", pa.string()),
                        ("item", pa.int64()),
                        ("judge_a", pa.string()),
                        ("judge_b", pa.string()),
                        ("verdict", pa.string()),
                    ]
                )
            ),
        ),
    ]
)


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _optional_text(value: object, field: str) -> str | None:
    return None if value is None else _required_text(value, field)


def _signature_ids(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of signature IDs")
    ids = tuple(_required_text(sid, field) for sid in value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field} contains duplicate signature IDs")
    return ids


def _probability(value: object, field: str, *, positive: bool = False) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        or (positive and value == 0)
    ):
        raise ValueError(f"Invalid {field}: {value!r}")
    return float(value)


@dataclass(frozen=True)
class Pair:
    """A query and the exact set of records shown to a judge."""

    block: str
    query_signature_id: str
    shown_group_signature_ids: tuple[str, ...]

    @property
    def pair_id(self) -> str:
        """Hash the query and order-independent shown signatures."""
        return canonical_json_digest([self.query_signature_id, sorted(self.shown_group_signature_ids)])

    @classmethod
    def read(cls, row: Mapping[str, Any]) -> Pair:
        """Validate pair evidence at the capture or answer key boundary."""
        block = normalize_block_key(_required_text(row["block"], "block"))
        query = _required_text(row["query_signature_id"], "query_signature_id")
        shown = _signature_ids(row["shown_group_signature_ids"], "shown_group_signature_ids")
        if not block or not shown or query in shown:
            raise ValueError("A pair needs a block and a nonempty Group B disjoint from the query")
        return cls(block, query, shown)


@dataclass(frozen=True)
class Item:
    """One occurrence of a pair in a package."""

    pair: Pair
    stratum: str
    target_signature_ids: tuple[str, ...] | None = None
    target_component: str | None = None
    model: str | None = None
    arm: str | None = None
    score: float | None = None
    cutoff: float | None = None
    inclusion_probability: float | None = None
    control: bool = False
    gold_verdict: Gold | None = None
    disputed_control: bool = False

    @classmethod
    def read(cls, row: Mapping[str, Any]) -> Item:
        """Validate private item metadata and its candidate membership."""
        pair = Pair.read(row)
        raw_members = row.get("target_signature_ids")
        members = None if raw_members is None else _signature_ids(raw_members, "target_signature_ids")
        if members is not None and (
            pair.query_signature_id in members or not set(pair.shown_group_signature_ids).issubset(members)
        ):
            raise ValueError(f"Invalid candidate membership for {pair.query_signature_id!r}")
        gold = row.get("gold_verdict")
        control = row.get("control", False)
        disputed = row.get("disputed_control", False)
        if (
            (gold is not None and gold not in CLASSES)
            or not isinstance(control, bool)
            or not isinstance(disputed, bool)
            or (control and gold is None)
            or (disputed and not control)
        ):
            raise ValueError("Invalid gold verdict or control flags")
        return cls(
            pair,
            _required_text(row["stratum"], "stratum"),
            members,
            _optional_text(row.get("target_component"), "target_component"),
            _optional_text(row.get("model"), "model"),
            _optional_text(row.get("arm"), "arm"),
            _probability(row.get("score"), "score"),
            _probability(row.get("cutoff"), "cutoff"),
            _probability(row.get("inclusion_probability"), "inclusion_probability", positive=True),
            control,
            gold,
            disputed,
        )

    def record(self) -> dict[str, Any]:
        """Flatten the pair and its occurrence for the private key."""
        row = asdict(self)
        row.pop("pair")
        return {**asdict(self.pair), "pair_id": self.pair.pair_id, **row}


def covers(
    shown: Iterable[str],
    candidate: Iterable[str],
    labeled_or_shown: Iterable[str],
) -> bool:
    """Check containment and a strict majority of labeled or displayed IDs."""
    old = set(shown)
    members = set(candidate)
    evidence = (set(labeled_or_shown) & members) | old
    return old <= members and 2 * len(old) > len(evidence)


def consensus(first: str, second: str) -> str:
    """Return agreement on identity or an undecided outcome."""
    return first if first == second and first != "cannot_tell" else "undecided"


def latest_version(bank: Path) -> Path | None:
    """Find the last fully published bank version."""
    versions = bank / "versions"
    return (
        max(
            (path for path in versions.iterdir() if path.is_dir() and VERSION_NAME.fullmatch(path.name)),
            default=None,
        )
        if versions.is_dir()
        else None
    )


def _bank_rows(version: Path | None, table: str) -> list[dict[str, Any]]:
    if version is None:
        return []
    if _read_json(version / "manifest.json")["format_version"] != FORMAT_VERSION:
        raise ValueError(f"Unsupported bank format in {version}")
    return pq.read_table(version / f"{table}.parquet").to_pylist()


def _pairs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, Any]] = {}
    by_item: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = row["pair_id"]
        pair = {field: row[field] for field, _ in _PAIR}
        pair["shown_group_signature_ids"] = sorted(pair["shown_group_signature_ids"])
        entry = by_pair.setdefault(pair_id, {**pair, "occurrences": [], "consensus": []})
        if any(entry[field] != value for field, value in pair.items()):
            raise ValueError(f"Conflicting evidence for pair {pair_id}")
        by_item[row["package"], row["item"]][row["judge"]] = row
    for (package, item), votes in by_item.items():
        row = next(iter(votes.values()))
        entry = by_pair[row["pair_id"]]
        entry["occurrences"].append({field: row[field] for field, _ in _OCCURRENCE})
        for a, b in combinations(sorted(votes), 2):
            entry["consensus"].append(
                {
                    "package": package,
                    "item": item,
                    "judge_a": a,
                    "judge_b": b,
                    "verdict": consensus(votes[a]["verdict"], votes[b]["verdict"]),
                }
            )
    return list(by_pair.values())


def read_verdicts(payload: bytes, expected: set[int], source: Path) -> dict[int, tuple[Verdict, str]]:
    """Require one valid verdict and confidence for every numbered item."""
    votes: dict[int, tuple[Verdict, str]] = {}
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{source}:{line_number}: expected an object")
        item, verdict, confidence = row.get("item"), row.get("verdict"), row.get("confidence")
        if type(item) is not int or item not in expected or item in votes:
            raise ValueError(f"{source}:{line_number}: unknown or repeated item {item!r}")
        if verdict not in VERDICTS or confidence not in CONFIDENCES:
            raise ValueError(f"{source}:{line_number}: invalid verdict or confidence")
        votes[item] = (verdict, confidence)
    if missing := expected - votes.keys():
        raise ValueError(f"{source}: missing {len(missing)} items, starting with {sorted(missing)[:5]}")
    return votes


def merge_verdicts(bank: Path, key: Path, judges: Sequence[tuple[str, Path]]) -> Path:
    """Validate and atomically append all supplied judge files in one bank version."""
    key_bytes = key.read_bytes()
    key_sha = hashlib.sha256(key_bytes).hexdigest()
    data = json.loads(key_bytes)
    if data["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported answer key format")
    package = _required_text(data["package"], "package")
    items = data["items"]
    if not items or set(items) != {str(i) for i in range(1, len(items) + 1)}:
        raise ValueError("Key item numbers must be contiguous from 1")
    records = {int(number): Item.read(row).record() for number, row in items.items()}
    names = [_required_text(name, "judge") for name, _ in judges]
    if not names or len(set(names)) != len(names):
        raise ValueError("Supply one or more distinct judge names")

    previous = latest_version(bank)
    rows = _bank_rows(previous, "judgments")
    prior = {(row["package"], row["judge"]): (row["key_sha256"], row["verdict_sha256"]) for row in rows}
    if any(p == package and sha != key_sha for (p, _), (sha, _) in prior.items()):
        raise ValueError(f"Package {package!r} is already bound to a different key")
    frozen = {}
    for name, source in judges:
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        votes = read_verdicts(payload, set(records), source)
        if (package, name) in prior:
            if prior[package, name] != (key_sha, digest):
                raise ValueError(f"Verdicts already frozen for {package!r}, judge {name!r}")
            continue
        frozen[digest] = payload
        rows.extend(
            {
                **records[item],
                "package": package,
                "item": item,
                "judge": name,
                "verdict": verdict,
                "confidence": confidence,
                "key_sha256": key_sha,
                "verdict_sha256": digest,
            }
            for item, (verdict, confidence) in votes.items()
        )
    if not frozen:
        assert previous is not None
        return previous

    rows.sort(key=lambda row: (row["package"], row["item"], row["judge"]))
    pairs = _pairs(rows)
    versions = bank / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    revision = 1 if previous is None else int(previous.name) + 1
    destination = versions / f"{revision:08d}"
    with TemporaryDirectory(prefix=".merge-", dir=versions) as temporary:
        staging = Path(temporary)
        (staging / "frozen").mkdir()
        for digest, payload in frozen.items():
            (staging / "frozen" / f"{digest}.jsonl").write_bytes(payload)
        pq.write_table(pa.Table.from_pylist(rows, schema=JUDGMENTS_SCHEMA), staging / "judgments.parquet")
        pq.write_table(pa.Table.from_pylist(pairs, schema=PAIRS_SCHEMA), staging / "pairs.parquet")
        (staging / "manifest.json").write_text(
            _json(
                {
                    "format_version": FORMAT_VERSION,
                    "previous": previous.name if previous else None,
                    "judgments": len(rows),
                    "pairs": len(pairs),
                    "frozen_verdict_sha256": sorted(frozen),
                }
            ),
            encoding="utf-8",
        )
        # An existing destination is a complete nonempty version. A concurrent writer must retry.
        staging.rename(destination)
    return destination


@dataclass(frozen=True)
class Decision:
    """Captured decision with the labels and display needed for bank lookup."""

    item: Item
    labeled_or_shown: frozenset[str]

    @classmethod
    def read(cls, row: dict[str, Any], rng: Random, max_shown: int) -> Decision:
        """Validate a decision and choose its display when the capture omits it."""
        members = _signature_ids(row["target_signature_ids"], "target_signature_ids")
        labels = _signature_ids(row["labeled_signature_ids"], "labeled_signature_ids")
        if not members or not set(labels).issubset(members):
            raise ValueError("Expected a nonempty candidate and labeled signatures drawn from it")
        shown = row.get("shown_group_signature_ids")
        if shown is None:
            shown = rng.sample(sorted(members), min(max_shown, len(members)))
        item = Item.read({**row, "shown_group_signature_ids": shown})
        if item.control or item.gold_verdict is not None or item.inclusion_probability is not None:
            raise ValueError("Control gold and inclusion probabilities are assigned by package assembly")
        return cls(item, frozenset(labels) | frozenset(item.pair.shown_group_signature_ids))


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _package_rows(captured: Path, quotas: Mapping[str, int], ledger: IdentityLedger | None) -> list[dict[str, Any]]:
    table = pq.read_table(captured)
    if "stratum" in table.column_names:
        return table.to_pylist()
    if "candidate_signature_ids" not in table.column_names or quotas.keys() - {"full", "initial_only"}:
        raise ValueError("Production capture quotas must use the full and initial_only query views")

    labels: dict[str, set[str]] = {}
    rows = []
    for row in table.to_pylist():
        if row["exclusion_reason"] is not None or row["query_view"] not in quotas:
            continue
        if ledger is not None and row["block"] not in labels:
            labels[row["block"]] = set(ledger.for_block(row["block"]).scored())
        members = row["candidate_signature_ids"]
        rows.append(
            {
                "block": row["block"],
                "query_signature_id": row["query_signature_id"],
                "target_signature_ids": members,
                "labeled_signature_ids": sorted(labels.get(row["block"], set()).intersection(members)),
                "stratum": row["query_view"],
                "target_component": row["candidate_component"],
                "score": row["chosen_probability"],
            }
        )
    return rows


def _given_names(orcid: str, cache: Path) -> str:
    if normalize_orcid(orcid) != orcid:
        raise ValueError(f"Invalid ledger ORCID {orcid!r}")
    path = cache / f"{orcid}.json"
    data: dict[str, Any]
    if path.is_file():
        data = _read_json(path)
    else:
        request = Request(f"https://pub.orcid.org/v3.0/{orcid}/person", headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:
                data = {"status": response.status, "person": json.load(response)}
        except HTTPError as exc:
            if exc.code not in (404, 410):
                LOGGER.error("ORCID lookup for %s returned HTTP %s", orcid, exc.code)
                raise
            data = {"status": exc.code, "person": None}
        except (URLError, TimeoutError):
            LOGGER.exception("ORCID lookup failed for %s", orcid)
            raise
        cache.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cache) as temporary:
            staged = Path(temporary) / path.name
            staged.write_text(_json(data), encoding="utf-8")
            staged.replace(path)
    if data["status"] != 200:
        return ""
    person = data["person"]
    if not isinstance(person, Mapping):
        raise ValueError(f"Invalid ORCID person record for {orcid}")
    person_path = person.get("path")
    if person_path is not None and person_path != f"/{orcid}/person":
        raise ValueError(f"ORCID person record does not match the requested ID {orcid}")
    names = (person.get("name") or {}).get("given-names") or {}
    return " ".join(unicodedata.normalize("NFKC", names.get("value") or "").casefold().split())


def _plant_controls(
    pool: list[Item],
    quotas: dict[ControlKind, int],
    ledger: IdentityLedger,
    cache: Path,
    rng: Random,
) -> list[Item]:
    rng.shuffle(pool)
    views = {}
    names: dict[str, str] = {}
    same: list[Item] = []
    different: list[Item] = []
    same_name: list[Item] = []
    either: list[Item] = []
    order: dict[Pair, int] = {}
    for index, item in enumerate(pool):
        different_needed = max(0, quotas["different"] - len(different))
        same_name_needed = max(0, quotas["different_same_name"] - len(same_name))
        if len(same) == quotas["same"] and different_needed + same_name_needed <= len(either):
            break
        pair = item.pair
        if pair.block not in views:
            view = ledger.for_block(pair.block)
            views[pair.block] = view, view.scored()
        view, labels = views[pair.block]
        query_identity = labels.get(pair.query_signature_id)
        if not query_identity or labels.keys().isdisjoint(pair.shown_group_signature_ids):
            continue
        assert item.target_signature_ids is not None
        target_identity = view.component_label(item.target_signature_ids)
        if target_identity is None:
            continue
        order[pair] = index
        if query_identity == target_identity:
            if len(same) < quotas["same"]:
                same.append(item)
            continue
        if different_needed + same_name_needed <= len(either):
            continue
        for orcid in (query_identity, target_identity):
            if orcid not in names:
                names[orcid] = _given_names(orcid, cache)
        first, second = names[query_identity], names[target_identity]
        if not first or not second:
            continue
        matches_initial = first[0] == second[0] and quotas["different_same_name"] > 0
        if matches_initial and first != second and quotas["different"]:
            either.append(item)
        elif matches_initial and same_name_needed:
            same_name.append(item)
        elif first != second and different_needed:
            different.append(item)

    same_name_needed = quotas["different_same_name"] - len(same_name)
    same_name.extend(either[:same_name_needed])
    different.extend(either[same_name_needed : same_name_needed + quotas["different"] - len(different)])
    candidates: dict[ControlKind, list[Item]] = {
        "same": same,
        "different": different,
        "different_same_name": same_name,
    }
    if shortfall := {
        kind: quotas[kind] - len(items) for kind, items in candidates.items() if len(items) < quotas[kind]
    }:
        raise ValueError(f"Insufficient ledger-pure control candidates, shortfall {shortfall}")
    selected = [
        replace(item, control=True, gold_verdict="same" if kind == "same" else "different", stratum=f"control_{kind}")
        for kind, items in candidates.items()
        for item in items
    ]
    return sorted(selected, key=lambda item: order[item.pair])


def _sample_decisions(
    decisions: Sequence[Decision],
    quotas: dict[str, int],
    version: Path | None,
    rng: Random,
    *,
    sample_before_reuse: bool,
) -> tuple[list[Item], list[Item], list[dict[str, Any]], dict[str, dict[str, int]]]:
    indexed: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for row in _bank_rows(version, "pairs"):
        pair = Pair.read(row)
        indexed[pair.block, pair.query_signature_id].append(pair)

    pools: dict[str, list[Item]] = defaultdict(list)
    reused = []
    matched: dict[str, list[str]] = {}
    skipped: Counter[str] = Counter()
    seen = set()
    for decision in decisions:
        item = decision.item
        pair = item.pair
        if pair.pair_id in seen:
            raise ValueError(f"Captured pair occurs more than once: {pair.pair_id}")
        seen.add(pair.pair_id)
        assert item.target_signature_ids is not None
        hits = [
            old.pair_id
            for old in indexed[pair.block, pair.query_signature_id]
            if covers(old.shown_group_signature_ids, item.target_signature_ids, decision.labeled_or_shown)
        ]
        if hits and not sample_before_reuse:
            reused.append({**item.record(), "bank_pair_ids": hits})
            skipped[item.stratum] += 1
        else:
            pools[item.stratum].append(item)
            if hits:
                matched[pair.pair_id] = hits

    selected = []
    remaining = []
    counts = {}
    for stratum, quota in quotas.items():
        pool = pools[stratum]
        take = min(quota, len(pool))
        sample = rng.sample(range(len(pool)), take)
        indices = set(sample)
        reused_in_sample = 0
        for i in sample:
            item = replace(pool[i], inclusion_probability=take / len(pool))
            if hits := matched.get(item.pair.pair_id):
                reused.append({**item.record(), "bank_pair_ids": hits})
                reused_in_sample += 1
            else:
                selected.append(item)
        remaining.extend(item for i, item in enumerate(pool) if i not in indices and item.pair.pair_id not in matched)
        counts[stratum] = {
            "eligible": len(pool),
            "reused": skipped[stratum] + reused_in_sample,
            "sampled": take - reused_in_sample,
        }
    return selected, remaining, reused, counts


def build_package(
    *,
    captured: Path,
    strata: Path,
    bank: Path,
    output_dir: Path,
    key_path: Path,
    snapshot: Path | None = None,
    ledger: IdentityLedger | None = None,
    orcid_cache: Path | None = None,
    sample_before_reuse: bool = False,
    max_authors: int | None = None,
) -> dict[str, int]:
    """Filter captured decisions, sample and plant controls, then publish a blind package."""
    key_path = resolve_answer_key_path(output_dir, key_path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    spec = _read_json(strata)
    package = _required_text(spec["package"], "package")
    quotas = {name: _nonnegative_int(n, f"quota for {name}") for name, n in spec["quotas"].items()}
    controls = {name: _nonnegative_int(n, f"control quota for {name}") for name, n in spec["controls"].items()}
    if controls.keys() - set(CONTROL_KINDS) or set(quotas) & {f"control_{kind}" for kind in CONTROL_KINDS}:
        raise ValueError("Use control quotas separately from decision strata")
    control_quotas = {kind: controls.get(kind, 0) for kind in CONTROL_KINDS}
    if any(control_quotas.values()) and ledger is None:
        raise ValueError("Planting controls requires a frozen identity ledger")
    if (control_quotas["different"] or control_quotas["different_same_name"]) and (
        orcid_cache is None or orcid_cache.resolve().is_relative_to(output_dir.resolve())
    ):
        raise ValueError("Different-person controls require a private ORCID cache outside the package directory")
    seed = _nonnegative_int(spec["seed"], "seed")
    max_shown = _nonnegative_int(spec.get("max_shown", 8), "max_shown")
    if not max_shown:
        raise ValueError("max_shown must be at least 1")
    if max_authors is not None and max_authors < 1:
        raise ValueError("max_authors must be at least 1")

    version = latest_version(bank)
    rows = _package_rows(captured, quotas, ledger)
    if unknown := {row["stratum"] for row in rows} - quotas.keys():
        raise ValueError(f"No quotas for captured strata {sorted(unknown)}")
    display_rng = Random(seed)
    decisions = [Decision.read(row, display_rng, max_shown) for row in rows]
    rng = Random(seed)
    selected, remaining, reused, counts = _sample_decisions(
        decisions, quotas, version, rng, sample_before_reuse=sample_before_reuse
    )
    if ledger is not None and any(control_quotas.values()):
        selected.extend(_plant_controls(remaining, control_quotas, ledger, orcid_cache or key_path.parent, rng))
    rng.shuffle(selected)
    ids_by_block: dict[str, set[str]] = defaultdict(set)
    for item in selected:
        pair = item.pair
        ids_by_block[pair.block].update((pair.query_signature_id, *pair.shown_group_signature_ids))
    papers = (
        frozen_packet_papers(ids_by_block, snapshot=snapshot, evidence=ledger.evidence if ledger else None)
        if ids_by_block
        else {}
    )
    private_ids = set()
    records = []

    def display(paper: Mapping[str, Any]) -> dict[str, Any]:
        clean = sanitize_paper_item(paper)
        authors = clean.get("authors") or []
        if max_authors is not None and len(authors) > max_authors:
            clean["authors"] = [*authors[:max_authors], "…"]
        return clean

    for item in selected:
        pair = item.pair
        private_ids.update((pair.query_signature_id, *(item.target_signature_ids or ())))
        if item.target_component:
            private_ids.add(item.target_component)
        query = papers[pair.block, pair.query_signature_id]
        shown = [papers[pair.block, sid] for sid in pair.shown_group_signature_ids]
        private_ids.update(str(paper["paper_id"]) for paper in [query, *shown])
        records.append(
            {
                "query": display(query),
                "candidates": [{"exemplars": [display(p) for p in shown]}],
            }
        )
    body = render_author_pair_package(records, private_ids=private_ids)
    key = {
        "format_version": FORMAT_VERSION,
        "package": package,
        "package_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "bank_version": version.name if version else None,
        "ledger": ledger.bindings if ledger else None,
        "strata": counts,
        "reused": reused,
        "items": {str(i): item.record() for i, item in enumerate(selected, 1)},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".package-", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        (staging / "package.md").write_bytes(body.encode())
        write_fresh_json(key_path, key)
        try:
            staging.rename(output_dir)
        except OSError:
            key_path.unlink()
            raise
    return {
        "sampled": sum(count["sampled"] for count in counts.values()),
        "reused": len(reused),
        "items": len(selected),
    }


def judge_check(
    bank: Path,
    *,
    package: str | None = None,
    judges: Sequence[str] = (),
    exclude_disputed: bool = False,
) -> list[dict[str, Any]]:
    """Count judge and pairwise-consensus accuracy on planted controls."""
    grouped: dict[tuple[str, int, str], dict[str, str]] = defaultdict(dict)
    for row in _bank_rows(latest_version(bank), "judgments"):
        if row["control"] and (
            (not exclude_disputed or not row["disputed_control"])
            and (package is None or row["package"] == package)
            and (not judges or row["judge"] in judges)
        ):
            grouped[row["package"], row["item"], row["gold_verdict"]][row["judge"]] = row["verdict"]

    totals: Counter[tuple[str, tuple[str, ...], str]] = Counter()
    correct: Counter[tuple[str, tuple[str, ...], str]] = Counter()
    for (name, _, gold), votes in grouped.items():
        for judge, vote in votes.items():
            key = name, (judge,), gold
            totals[key] += 1
            correct[key] += vote == gold
        for a, b in combinations(sorted(votes), 2):
            key = name, (a, b), gold
            totals[key] += 1
            correct[key] += consensus(votes[a], votes[b]) == gold
    return [
        {
            "package": name,
            "judges": names,
            "gold": gold,
            "correct": correct[name, names, gold],
            "total": total,
            "accuracy": correct[name, names, gold] / total,
        }
        for (name, names, gold), total in sorted(totals.items())
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run blind package assembly, verified import, or control scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-package")
    build.add_argument("--captured", type=Path, required=True)
    build.add_argument("--strata", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--key", type=Path, required=True)
    build.add_argument("--snapshot", type=Path)
    build.add_argument("--block-role-assignments", type=Path)
    build.add_argument("--identity-evidence", type=Path)
    build.add_argument("--orcid-cache", type=Path)
    build.add_argument("--sample-before-reuse", action="store_true")
    build.add_argument("--max-authors", type=int)
    merge = commands.add_parser("merge-verdicts")
    merge.add_argument("--key", type=Path, required=True)
    merge.add_argument("--judge", nargs=2, metavar=("NAME", "JSONL"), action="append", required=True)
    check = commands.add_parser("judge-check")
    check.add_argument("--package", required=True)
    check.add_argument("--judge", action="append", default=[])
    check.add_argument("--exclude-disputed", action="store_true")
    for kind in ("judge", "consensus"):
        for gold in CLASSES:
            check.add_argument(f"--{kind}-{gold}", type=float)
    for command in (build, merge, check):
        command.add_argument("--bank", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "build-package":
        roles, evidence = args.block_role_assignments, args.identity_evidence
        if bool(roles) != bool(evidence):
            parser.error("Supply --block-role-assignments and --identity-evidence together")
        ledger = load_identity_ledger(evidence, load_block_roles(roles)) if evidence else None
        print(
            _json(
                build_package(
                    captured=args.captured,
                    strata=args.strata,
                    bank=args.bank,
                    output_dir=args.output_dir,
                    key_path=args.key,
                    snapshot=args.snapshot,
                    ledger=ledger,
                    orcid_cache=args.orcid_cache,
                    sample_before_reuse=args.sample_before_reuse,
                    max_authors=args.max_authors,
                )
            ),
            end="",
        )
        return 0
    if args.command == "merge-verdicts":
        print(merge_verdicts(args.bank, args.key, [(name, Path(path)) for name, path in args.judge]))
        return 0

    thresholds = {
        (n, gold): _probability(getattr(args, f"{kind}_{gold}"), f"{kind}-{gold}")
        for n, kind in ((1, "judge"), (2, "consensus"))
        for gold in CLASSES
    }
    if all(value is None for value in thresholds.values()):
        parser.error("Supply at least one judge or consensus threshold")
    rows = judge_check(args.bank, package=args.package, judges=args.judge, exclude_disputed=args.exclude_disputed)
    available = {name for row in rows for name in row["judges"]}
    by_group = {(row["judges"], row["gold"]): row for row in rows}
    passed = bool(available) and set(args.judge) <= available
    for (n, gold), threshold in thresholds.items():
        if threshold is None:
            continue
        groups = list(combinations(sorted(available), n))
        passed &= bool(groups)
        for group in groups:
            row = by_group.get((group, gold))
            passed &= row is not None and row["accuracy"] >= threshold
    print(_json({"passed": passed, "controls": rows}), end="")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())

"""Hash public replay emails without changing their pairwise feature evidence.

Also supports intake-once redaction of structured email fields and free-text
email mentions (`affiliations`, `title`, `abstract`) across JSON and Arrow
trees, plus plaintext-email scanning of staged release bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

from s2and._sha256 import sha256_file
from s2and.incremental_linking.feature_block_arrow import write_arrow_batch_lookup_index
from s2and.text import email_prefix_suffix

# Legacy 1.4 giant dataset names kept for backwards compatibility; release hashing
# discovers datasets dynamically from replay_manifest.json and dataset directories.
GIANT_DATASETS = ("a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park")
EMAIL_REDACTED_TOKEN = "[EMAIL_REDACTED]"
_HASHED_EMAIL = re.compile(r"[0-9a-f]{64}@[0-9a-f]{64}|(?:hmac-sha256|sha256):[0-9a-f]{64}")
# Matches standard emails, Unicode-dash variants (\u2010-\u2015), and malformed "@." forms.
_EMAIL_LOCAL_CHARS = r"A-Za-z0-9._%+\-\u2010-\u2015"
_EMAIL_DOMAIN_CHARS = r"A-Za-z0-9.\-\u2010-\u2015"
_PLAINTEXT_EMAIL_RE = re.compile(
    rf"(?<![{_EMAIL_LOCAL_CHARS}])[{_EMAIL_LOCAL_CHARS}]+@\.?[{_EMAIL_DOMAIN_CHARS}]+\.[A-Za-z]{{2,}}(?![A-Za-z])"
)
_TEXT_SCAN_SUFFIXES = frozenset({".json", ".jsonl", ".tsv", ".csv", ".txt"})
_TEXT_SCAN_CHUNK_CHARS = 2 * 1024 * 1024
_TEXT_SCAN_OVERLAP_CHARS = 512


def hash_feature_email(value: str | None) -> str | None:
    """Preserve normalized email equality and malformed-value missingness.

    Args:
        value: A raw email, existing component hash, or missing value.

    Returns:
        Separate normalized component hashes, or a whole-value hash for malformed input.
    """
    if not value or _HASHED_EMAIL.fullmatch(value):
        return value
    local, domain = email_prefix_suffix(value)
    if local is None or domain is None:
        return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    return "@".join(hashlib.sha256(part.encode("utf-8")).hexdigest() for part in (local, domain))


def scrub_text_emails(value: str | None, *, replacement: str = EMAIL_REDACTED_TOKEN) -> str | None:
    """Replace plaintext email mentions in free text without touching existing hashes.

    Handles standard ASCII addresses as well as Unicode-hyphen (U+2010..U+2015) and
    malformed ``user@.domain.tld`` variants observed in paper metadata and affiliations.
    """
    if not value:
        return value

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return token if _HASHED_EMAIL.fullmatch(token) else replacement

    return _PLAINTEXT_EMAIL_RE.sub(_replace, value)


def _find_plaintext_emails(text: str) -> list[str]:
    return [
        match.group(0) for match in _PLAINTEXT_EMAIL_RE.finditer(text) if not _HASHED_EMAIL.fullmatch(match.group(0))
    ]


def _write_json(path: Path, payload: Any) -> None:
    # Replace the directory entry because release staging may use hardlinks.
    temporary = path.with_suffix(path.suffix + ".email_hashing.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _scrub_arrow_string_or_list(array: pa.Array, field: pa.Field) -> tuple[pa.Array, int]:
    """Scrub plaintext emails in a string or list<string> Arrow column."""
    values = array.to_pylist()
    if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
        after = [scrub_text_emails(v) if isinstance(v, str) else v for v in values]
        changed = sum(a != b for a, b in zip(values, after, strict=True))
        return pa.array(after, type=field.type), changed
    if (pa.types.is_list(field.type) or pa.types.is_large_list(field.type)) and (
        pa.types.is_string(field.type.value_type) or pa.types.is_large_string(field.type.value_type)
    ):
        after_list: list[list[str | None] | None] = []
        changed = 0
        for row in values:
            if row is None:
                after_list.append(None)
                continue
            cleaned = [scrub_text_emails(item) if isinstance(item, str) else item for item in row]
            changed += sum(a != b for a, b in zip(row, cleaned, strict=True))
            after_list.append(cleaned)
        return pa.array(after_list, type=field.type), changed
    return array, 0


def _hash_arrow(path: Path, *, scrub_free_text: bool = False) -> int:
    temporary = path.with_suffix(".email_hashing.tmp")
    changed = 0
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        email_column = reader.schema.get_field_index("author_email")
        affiliation_columns = (
            [
                idx
                for name in ("author_affiliations", "affiliations")
                if (idx := reader.schema.get_field_index(name)) >= 0
            ]
            if scrub_free_text
            else []
        )
        if email_column < 0 and not affiliation_columns:
            return 0
        with pa.OSFile(str(temporary), "wb") as sink, pa.ipc.new_file(sink, reader.schema) as writer:
            for index in range(reader.num_record_batches):
                batch = reader.get_batch(index)
                if email_column >= 0:
                    before = batch.column(email_column).to_pylist()
                    after = [hash_feature_email(value) for value in before]
                    changed += sum(a != b for a, b in zip(before, after, strict=True))
                    batch = batch.set_column(
                        email_column,
                        reader.schema.field(email_column),
                        pa.array(after, type=reader.schema.field(email_column).type),
                    )
                for col_idx in affiliation_columns:
                    field = reader.schema.field(col_idx)
                    cleaned_col, col_changed = _scrub_arrow_string_or_list(batch.column(col_idx), field)
                    changed += col_changed
                    batch = batch.set_column(col_idx, field, cleaned_col)
                writer.write_batch(batch)
    if changed:
        temporary.replace(path)
    else:
        temporary.unlink()
    return changed


def _scrub_papers_arrow(path: Path) -> int:
    temporary = path.with_suffix(".email_hashing.tmp")
    changed = 0
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        text_columns = [idx for name in ("title", "abstract") if (idx := reader.schema.get_field_index(name)) >= 0]
        if not text_columns:
            return 0
        with pa.OSFile(str(temporary), "wb") as sink, pa.ipc.new_file(sink, reader.schema) as writer:
            for index in range(reader.num_record_batches):
                batch = reader.get_batch(index)
                for col_idx in text_columns:
                    field = reader.schema.field(col_idx)
                    cleaned_col, col_changed = _scrub_arrow_string_or_list(batch.column(col_idx), field)
                    changed += col_changed
                    batch = batch.set_column(col_idx, field, cleaned_col)
                writer.write_batch(batch)
    if changed:
        temporary.replace(path)
    else:
        temporary.unlink()
    return changed


def _hash_source_json(path: Path, *, scrub_free_text: bool = False) -> int:
    changed = 0
    temporary = path.with_suffix(".email_hashing.tmp")
    with path.open("rb") as source, temporary.open("x", encoding="utf-8", newline="\n") as output:
        output.write("{")
        first = True
        for key, record in ijson.kvitems(source, "", use_float=True):
            info = record["author_info"]
            for field in ("email", "emails"):
                if field not in info:
                    continue
                before = info[field]
                after = (
                    [hash_feature_email(value) for value in before]
                    if isinstance(before, list)
                    else hash_feature_email(before)
                )
                changed += before != after
                info[field] = after
            if scrub_free_text:
                for field in ("affiliations", "affiliation"):
                    if field not in info:
                        continue
                    before = info[field]
                    if isinstance(before, list):
                        after = [scrub_text_emails(v) if isinstance(v, str) else v for v in before]
                    elif isinstance(before, str):
                        after = scrub_text_emails(before)
                    else:
                        continue
                    changed += before != after
                    info[field] = after
            if not first:
                output.write(",")
            first = False
            output.write(json.dumps(key) + ":" + json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        output.write("}\n")
    if changed:
        temporary.replace(path)
    else:
        temporary.unlink()
    return changed


def _scrub_papers_json(path: Path) -> int:
    changed = 0
    temporary = path.with_suffix(".email_hashing.tmp")
    with path.open("rb") as source, temporary.open("x", encoding="utf-8", newline="\n") as output:
        output.write("{")
        first = True
        for key, record in ijson.kvitems(source, "", use_float=True):
            if isinstance(record, dict):
                for field in ("title", "abstract"):
                    before = record.get(field)
                    if isinstance(before, str):
                        after = scrub_text_emails(before)
                        if after != before:
                            changed += 1
                            record[field] = after
            if not first:
                output.write(",")
            first = False
            output.write(json.dumps(key) + ":" + json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        output.write("}\n")
    if changed:
        temporary.replace(path)
    else:
        temporary.unlink()
    return changed


def _refresh_arrow_dataset_manifest(directory: Path, *, signatures_changed: bool, papers_changed: bool) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest.get("paths", {})
    files = manifest.setdefault("files", {})
    updated = False
    if signatures_changed and "signatures" in paths:
        sig_path = directory / paths["signatures"]
        if "signatures_batch_index" in paths:
            index_path = directory / paths["signatures_batch_index"]
            write_arrow_batch_lookup_index(sig_path, index_path, key_column="signature_id", table_name="signatures")
            files["signatures_batch_index"] = {
                "byte_count": index_path.stat().st_size,
                "sha256": sha256_file(index_path),
            }
        files["signatures"] = {"byte_count": sig_path.stat().st_size, "sha256": sha256_file(sig_path)}
        updated = True
    if papers_changed and "papers" in paths:
        papers_path = directory / paths["papers"]
        files["papers"] = {"byte_count": papers_path.stat().st_size, "sha256": sha256_file(papers_path)}
        updated = True
    if updated:
        _write_json(manifest_path, manifest)
    return updated


def _discover_release_datasets(output_root: Path) -> tuple[str, ...]:
    """Discover dataset names dynamically from manifests and staged directories."""
    names: set[str] = set()
    dataset_root = output_root / "dataset"
    replay_path = dataset_root / "replay_manifest.json"
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        manifests = replay.get("dataset_manifests")
        if isinstance(manifests, dict):
            names.update(str(name) for name in manifests)
    datasets_dir = dataset_root / "datasets"
    if datasets_dir.is_dir():
        for child in datasets_dir.iterdir():
            if child.is_dir() and (child / "signatures.arrow").is_file():
                names.add(child.name)
    raw_dir = output_root / "source" / "corrected_replay" / "raw"
    if raw_dir.is_dir():
        for child in raw_dir.iterdir():
            if child.is_dir() and (child / "signatures.json").is_file():
                names.add(child.name)
    return tuple(sorted(names))


def redact_source_tree(source_root: Path, *, scrub_free_text: bool = True) -> dict[str, int]:
    """Redact emails once at intake across JSON and Arrow files under ``source_root``.

    Hashes structured ``email`` / ``emails`` / ``author_email`` fields while preserving
    normalized pairwise equality, and optionally scrubs plaintext email mentions from
    ``affiliations``, ``title``, and ``abstract``. Refreshes any sibling Arrow manifests.
    """
    source_root = source_root.resolve()
    counts = {
        "signature_json_fields": 0,
        "paper_json_fields": 0,
        "signature_arrow_values": 0,
        "paper_arrow_values": 0,
    }
    arrow_dirs: dict[Path, list[bool]] = {}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name == "signatures.json" or name.endswith("_signatures.json"):
            counts["signature_json_fields"] += _hash_source_json(path, scrub_free_text=scrub_free_text)
        elif scrub_free_text and (name == "papers.json" or name.endswith("_papers.json")):
            counts["paper_json_fields"] += _scrub_papers_json(path)
        elif name == "signatures.arrow":
            changed = _hash_arrow(path, scrub_free_text=scrub_free_text)
            counts["signature_arrow_values"] += changed
            if changed:
                flags = arrow_dirs.setdefault(path.parent, [False, False])
                flags[0] = True
        elif scrub_free_text and name == "papers.arrow":
            changed = _scrub_papers_arrow(path)
            counts["paper_arrow_values"] += changed
            if changed:
                flags = arrow_dirs.setdefault(path.parent, [False, False])
                flags[1] = True
    for directory, (sig_changed, papers_changed) in sorted(arrow_dirs.items()):
        _refresh_arrow_dataset_manifest(directory, signatures_changed=sig_changed, papers_changed=papers_changed)
    return counts


def _scan_arrow_batch(batch: pa.RecordBatch, rel_path: str, findings: list[dict[str, str]], max_findings: int) -> None:
    for col_idx, field in enumerate(batch.schema):
        if len(findings) >= max_findings:
            return
        ftype = field.type
        is_str = pa.types.is_string(ftype) or pa.types.is_large_string(ftype)
        is_list_str = (pa.types.is_list(ftype) or pa.types.is_large_list(ftype)) and (
            pa.types.is_string(ftype.value_type) or pa.types.is_large_string(ftype.value_type)
        )
        if not (is_str or is_list_str):
            continue
        for value in batch.column(col_idx).to_pylist():
            if value is None:
                continue
            items = [value] if is_str else value
            for item in items:
                if not isinstance(item, str) or "@" not in item:
                    continue
                for match in _find_plaintext_emails(item):
                    findings.append(
                        {
                            "path": rel_path,
                            "location": f"column:{field.name}",
                            "match_sha256": hashlib.sha256(match.encode("utf-8")).hexdigest()[:12],
                        }
                    )
                    if len(findings) >= max_findings:
                        return


def scan_tree_for_plaintext_emails(root: Path, *, max_findings: int = 20) -> list[dict[str, str]]:
    """Scan JSON, text, Arrow, and Parquet files under ``root`` for plaintext emails.

    Returns compact findings without leaking the raw email addresses.
    """
    root = root.resolve()
    findings: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if len(findings) >= max_findings:
            break
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel_path = path.relative_to(root).as_posix()
        if suffix in _TEXT_SCAN_SUFFIXES:
            with path.open("r", encoding="utf-8", errors="ignore") as stream:
                tail = ""
                chunk_index = 0
                while len(findings) < max_findings:
                    chunk = stream.read(_TEXT_SCAN_CHUNK_CHARS)
                    if not chunk:
                        break
                    window = tail + chunk
                    if "@" in window:
                        for match in _find_plaintext_emails(window):
                            findings.append(
                                {
                                    "path": rel_path,
                                    "location": f"chunk:{chunk_index}",
                                    "match_sha256": hashlib.sha256(match.encode("utf-8")).hexdigest()[:12],
                                }
                            )
                            if len(findings) >= max_findings:
                                break
                    tail = window[-_TEXT_SCAN_OVERLAP_CHARS:]
                    chunk_index += 1
        elif suffix == ".arrow":
            with pa.memory_map(str(path), "r") as source:
                reader = pa.ipc.open_file(source)
                for batch_index in range(reader.num_record_batches):
                    _scan_arrow_batch(reader.get_batch(batch_index), rel_path, findings, max_findings)
                    if len(findings) >= max_findings:
                        break
        elif suffix == ".parquet":
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches():
                _scan_arrow_batch(batch, rel_path, findings, max_findings)
                if len(findings) >= max_findings:
                    break
    return findings


def hash_release_emails(output_root: Path, *, scrub_free_text: bool = False) -> dict[str, dict[str, int]]:
    """Hash replay email columns and source fields, then refresh checksum references.

    Discovers datasets dynamically from ``replay_manifest.json`` and staged directories
    rather than relying on a fixed giant list.

    Args:
        output_root: Staged root containing the dataset and source bundles.
        scrub_free_text: When True, also scrub plaintext emails from free-text
            ``affiliations``, ``title``, and ``abstract`` fields.

    Returns:
        Changed Arrow values and source fields for each discovered dataset.
    """
    report: dict[str, dict[str, int]] = {}
    dataset = output_root / "dataset"
    replay_path = dataset / "replay_manifest.json"
    replay: dict[str, Any] | None = None
    replay_dirty = False
    for name in _discover_release_datasets(output_root):
        directory = dataset / "datasets" / name
        if not directory.exists():
            continue
        sig_changed = _hash_arrow(directory / "signatures.arrow", scrub_free_text=scrub_free_text)
        papers_arrow = directory / "papers.arrow"
        papers_changed = _scrub_papers_arrow(papers_arrow) if scrub_free_text and papers_arrow.is_file() else 0
        changed = sig_changed + papers_changed

        source_dir = output_root / "source" / "corrected_replay" / "raw" / name
        source = source_dir / "signatures.json"
        source_changed = _hash_source_json(source, scrub_free_text=scrub_free_text) if source.exists() else 0
        source_papers = source_dir / "papers.json"
        if scrub_free_text and source_papers.is_file():
            source_changed += _scrub_papers_json(source_papers)

        report[name] = {"arrow_values": changed, "source_fields": source_changed}
        if not changed:
            continue
        _refresh_arrow_dataset_manifest(
            directory, signatures_changed=bool(sig_changed), papers_changed=bool(papers_changed)
        )
        if replay_path.is_file():
            if replay is None:
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
            if isinstance(replay.get("dataset_manifests"), dict) and name in replay["dataset_manifests"]:
                replay["dataset_manifests"][name]["sha256"] = sha256_file(directory / "manifest.json")
                replay_dirty = True
    if replay is not None and replay_dirty:
        _write_json(replay_path, replay)
        public_path = dataset / "manifest.json"
        if public_path.is_file():
            public = json.loads(public_path.read_text(encoding="utf-8"))
            if isinstance(public.get("replay_bundles"), dict) and "incremental" in public["replay_bundles"]:
                public["replay_bundles"]["incremental"]["sha256"] = sha256_file(replay_path)
                _write_json(public_path, public)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    redact_parser = subparsers.add_parser("redact-source-tree", help="Redact emails once at intake in a source tree.")
    redact_parser.add_argument("--source-root", type=Path, required=True)
    redact_parser.add_argument("--keep-free-text", action="store_true", help="Only hash structured email fields.")

    scan_parser = subparsers.add_parser("scan", help="Scan a directory tree for plaintext emails.")
    scan_parser.add_argument("--root", type=Path, required=True)
    scan_parser.add_argument("--max-findings", type=int, default=20)

    args = parser.parse_args(argv)
    if args.command == "redact-source-tree":
        print(json.dumps(redact_source_tree(args.source_root, scrub_free_text=not args.keep_free_text), indent=2))
        return 0
    findings = scan_tree_for_plaintext_emails(args.root, max_findings=args.max_findings)
    print(json.dumps({"findings": findings, "count": len(findings)}, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

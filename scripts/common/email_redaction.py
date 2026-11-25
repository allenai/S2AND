"""Hash and scrub emails in source trees, and scan trees for plaintext emails."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Collection
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

# Use token boundaries instead of Unicode letter classes so contextual IDNA characters survive matching.
_TOKEN_DELIMITERS = r'\s@\uff20<>()\[\]{}"“”‘’,;:/\\，；：、。！？〈〉《》「」『』【】（）'
_LOCAL_CHAR = rf"[^{_TOKEN_DELIMITERS}]"
_DOMAIN_CHAR = rf"[^{_TOKEN_DELIMITERS}.!?]"
_PLAINTEXT_EMAIL_RE = re.compile(
    rf'(?:\{{[^{{}}@\uff20\r\n]+\}}|"[^"@\uff20\r\n]+"|(?<!{_LOCAL_CHAR}){_LOCAL_CHAR}++)'
    rf"[@\uff20]\.?{_DOMAIN_CHAR}++(?:\.{_DOMAIN_CHAR}++)++"
)
# Avoid flagging chemical nanoparticle formulas like "Fe3O4@SiO2.The" in abstracts.
_CHEMICAL_FORMULA_PREFIX_RE = re.compile(r"^[A-Z][a-z]?\d+(?:[A-Z][a-z]?\d*)*[@\uFF20](?:[A-Z][a-z]?\d*)+\.")
_JSON_ESCAPED_AT_RE = re.compile(r"\\u0040|\\uff20", re.IGNORECASE)
_TEXT_SCAN_SUFFIXES = frozenset({".json", ".jsonl", ".tsv", ".csv", ".txt", ".md", ".yaml", ".yml", ".html"})
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


def _is_real_plaintext_email(token: str) -> bool:
    if _HASHED_EMAIL.fullmatch(token):
        return False
    if _CHEMICAL_FORMULA_PREFIX_RE.match(token):
        return False
    return True


def scrub_text_emails(value: str | None, *, replacement: str = EMAIL_REDACTED_TOKEN) -> str | None:
    """Replace plaintext email mentions in free text without touching existing hashes.

    Handles dotted domains in any script, fullwidth ``＠``, brace-grouped and quoted
    locals, and malformed ``user@.domain.tld`` variants. Redacts through the next delimiter.
    """
    if not value or ("@" not in value and "\uff20" not in value):
        return value

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return replacement if _is_real_plaintext_email(token) else token

    return _PLAINTEXT_EMAIL_RE.sub(_replace, value)


def _find_plaintext_emails_with_offsets(text: str) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    if "@" not in text and "\uff20" not in text:
        return found
    for match in _PLAINTEXT_EMAIL_RE.finditer(text):
        token = match.group(0)
        if _is_real_plaintext_email(token):
            found.append((match.start(), match.end(), token))
    return found


def find_plaintext_emails(text: str) -> list[str]:
    return [token for _, _, token in _find_plaintext_emails_with_offsets(text)]


def _atomic_temp_path(target: Path, suffix: str = ".email_hashing.tmp") -> Path:
    handle = tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=suffix, delete=False)
    temp_path = Path(handle.name)
    handle.close()
    return temp_path


def _write_json(path: Path, payload: Any) -> None:
    # Replace the directory entry atomically because release staging may use hardlinks.
    temporary = _atomic_temp_path(path, suffix=".json.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


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
    temporary = _atomic_temp_path(path)
    changed = 0
    try:
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
                temporary.unlink(missing_ok=True)
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
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return changed


def _scrub_papers_arrow(path: Path) -> int:
    return _scrub_arrow_text_columns(path, ("title", "abstract"))


def _scrub_paper_authors_arrow(path: Path) -> int:
    # Production author lists occasionally carry an email as the author name.
    return _scrub_arrow_text_columns(path, ("author_name",))


def _scrub_arrow_text_columns(path: Path, column_names: tuple[str, ...]) -> int:
    temporary = _atomic_temp_path(path)
    changed = 0
    try:
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_file(source)
            text_columns = [idx for name in column_names if (idx := reader.schema.get_field_index(name)) >= 0]
            if not text_columns:
                temporary.unlink(missing_ok=True)
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
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return changed


def _hash_source_json(path: Path, *, scrub_free_text: bool = False) -> int:
    changed = 0
    temporary = _atomic_temp_path(path)
    try:
        with path.open("rb") as source, temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write("{")
            first = True
            for key, record in ijson.kvitems(source, "", use_float=True):
                if isinstance(record, dict) and isinstance(record.get("author_info"), dict):
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
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return changed


def _scrub_papers_json(path: Path) -> int:
    changed = 0
    temporary = _atomic_temp_path(path)
    try:
        with path.open("rb") as source, temporary.open("w", encoding="utf-8", newline="\n") as output:
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
                    for author in record.get("authors") or ():
                        name = author.get("author_name") if isinstance(author, dict) else None
                        if isinstance(name, str) and (cleaned := scrub_text_emails(name)) != name:
                            changed += 1
                            author["author_name"] = cleaned
                if not first:
                    output.write(",")
                first = False
                output.write(json.dumps(key) + ":" + json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output.write("}\n")
        if changed:
            temporary.replace(path)
        else:
            temporary.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return changed


_TABLE_INDEX_KEYS = {"signatures": "signature_id", "papers": "paper_id", "paper_authors": "paper_id"}


def _refresh_arrow_dataset_manifest(directory: Path, changed_tables: Collection[str]) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest.get("paths", {})
    files = manifest.setdefault("files", {})
    updated = False
    for table in changed_tables:
        if table not in paths:
            continue
        table_path = directory / paths[table]
        index_name = f"{table}_batch_index"
        if index_name in paths:
            # A rewritten table invalidates its lookup index, so rebuild it with the table.
            index_path = directory / paths[index_name]
            write_arrow_batch_lookup_index(
                table_path, index_path, key_column=_TABLE_INDEX_KEYS[table], table_name=table, overwrite=True
            )
            files[index_name] = {"byte_count": index_path.stat().st_size, "sha256": sha256_file(index_path)}
        files[table] = {"byte_count": table_path.stat().st_size, "sha256": sha256_file(table_path)}
        updated = True
    if updated:
        _write_json(manifest_path, manifest)
    return updated


def _refresh_source_inventory(source_root: Path, modified_paths: list[Path]) -> None:
    """Update SHA-256 entries in ``source/corrected_sources/manifest.json`` after source edits."""
    inventory_path = source_root / "corrected_sources" / "manifest.json"
    if not modified_paths or not inventory_path.is_file():
        return
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict):
        return
    dirty = False
    for path in modified_paths:
        rel = path.relative_to(source_root).as_posix()
        if rel in files:
            files[rel] = sha256_file(path)
            dirty = True
    if dirty:
        _write_json(inventory_path, payload)


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
    ``affiliations``, ``title``, ``abstract`` and author names. Refreshes any sibling Arrow
    manifests and the batch lookup index of each rewritten table.
    """
    source_root = source_root.resolve()
    counts = {
        "signature_json_fields": 0,
        "paper_json_fields": 0,
        "signature_arrow_values": 0,
        "paper_arrow_values": 0,
        "paper_author_arrow_values": 0,
    }
    arrow_dirs: dict[Path, set[str]] = {}
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
                arrow_dirs.setdefault(path.parent, set()).add("signatures")
        elif scrub_free_text and name == "papers.arrow":
            changed = _scrub_papers_arrow(path)
            counts["paper_arrow_values"] += changed
            if changed:
                arrow_dirs.setdefault(path.parent, set()).add("papers")
        elif scrub_free_text and name == "paper_authors.arrow":
            changed = _scrub_paper_authors_arrow(path)
            counts["paper_author_arrow_values"] += changed
            if changed:
                arrow_dirs.setdefault(path.parent, set()).add("paper_authors")
    for directory, tables in sorted(arrow_dirs.items()):
        _refresh_arrow_dataset_manifest(directory, sorted(tables))
    return counts


def _type_contains_string(ftype: pa.DataType) -> bool:
    if pa.types.is_string(ftype) or pa.types.is_large_string(ftype) or pa.types.is_string_view(ftype):
        return True
    if pa.types.is_dictionary(ftype):
        return _type_contains_string(ftype.value_type)
    if pa.types.is_list(ftype) or pa.types.is_large_list(ftype) or pa.types.is_fixed_size_list(ftype):
        return _type_contains_string(ftype.value_type)
    if pa.types.is_struct(ftype):
        return any(_type_contains_string(field.type) for field in ftype)
    if pa.types.is_map(ftype):
        return _type_contains_string(ftype.key_type) or _type_contains_string(ftype.item_type)
    return False


def _scan_nested_python_value(
    value: Any,
    *,
    rel_path: str,
    location: str,
    findings: list[dict[str, str]],
    max_findings: int,
) -> None:
    if value is None or len(findings) >= max_findings:
        return
    if isinstance(value, str):
        if "@" not in value and "\uff20" not in value:
            return
        for match in find_plaintext_emails(value):
            findings.append(
                {
                    "path": rel_path,
                    "location": location,
                    "match_sha256": hashlib.sha256(match.encode("utf-8")).hexdigest()[:12],
                }
            )
            if len(findings) >= max_findings:
                return
    elif isinstance(value, list | tuple):
        for item in value:
            _scan_nested_python_value(
                item, rel_path=rel_path, location=location, findings=findings, max_findings=max_findings
            )
            if len(findings) >= max_findings:
                return
    elif isinstance(value, dict):
        for nested in value.values():
            _scan_nested_python_value(
                nested, rel_path=rel_path, location=location, findings=findings, max_findings=max_findings
            )
            if len(findings) >= max_findings:
                return


def _scan_arrow_batch(batch: pa.RecordBatch, rel_path: str, findings: list[dict[str, str]], max_findings: int) -> None:
    for col_idx, field in enumerate(batch.schema):
        if len(findings) >= max_findings:
            return
        if not _type_contains_string(field.type):
            continue
        col = batch.column(col_idx)
        # For dictionary-encoded string columns, scanning the unique dictionary is both faster and complete.
        if pa.types.is_dictionary(field.type):
            values = col.dictionary.to_pylist()
        else:
            values = col.to_pylist()
        location = f"column:{field.name}"
        for value in values:
            _scan_nested_python_value(
                value, rel_path=rel_path, location=location, findings=findings, max_findings=max_findings
            )
            if len(findings) >= max_findings:
                return


def scan_tree_for_plaintext_emails(root: Path, *, max_findings: int = 20) -> list[dict[str, str]]:
    """Scan JSON, text, Arrow, and Parquet files under ``root`` for plaintext emails.

    Returns compact findings without leaking raw email addresses. Deduplicates chunk-overlap
    matches, handles JSON-escaped ``\\u0040``, and recursively inspects nested Arrow types
    (including ``dictionary``, ``string_view``, ``list<struct>``, and ``map``).
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
                window_start_offset = 0
                seen_spans: set[tuple[int, int]] = set()
                chunk_index = 0
                while len(findings) < max_findings:
                    chunk = stream.read(_TEXT_SCAN_CHUNK_CHARS)
                    if not chunk:
                        break
                    window = tail + chunk
                    if suffix in {".json", ".jsonl"} and (
                        "\\u0040" in window or "\\u0040" in window.lower() or "\\uff20" in window.lower()
                    ):
                        scan_window = _JSON_ESCAPED_AT_RE.sub("@", window)
                    else:
                        scan_window = window
                    if "@" in scan_window or "\uff20" in scan_window:
                        for start, end, match in _find_plaintext_emails_with_offsets(scan_window):
                            # Skip partial matches abutting the right boundary before next chunk is read.
                            if len(chunk) == _TEXT_SCAN_CHUNK_CHARS and end == len(scan_window):
                                continue
                            abs_span = (window_start_offset + start, window_start_offset + end)
                            if abs_span in seen_spans:
                                continue
                            seen_spans.add(abs_span)
                            findings.append(
                                {
                                    "path": rel_path,
                                    "location": f"chunk:{chunk_index}",
                                    "match_sha256": hashlib.sha256(match.encode("utf-8")).hexdigest()[:12],
                                }
                            )
                            if len(findings) >= max_findings:
                                break
                    next_tail_len = min(_TEXT_SCAN_OVERLAP_CHARS, len(window))
                    window_start_offset += len(window) - next_tail_len
                    tail = window[-next_tail_len:] if next_tail_len else ""
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
            ``affiliations``, ``title``, ``abstract`` and author-name fields.

    Returns:
        Changed Arrow values and source fields for each discovered dataset.
    """
    report: dict[str, dict[str, int]] = {}
    dataset = output_root / "dataset"
    source_root = output_root / "source"
    replay_path = dataset / "replay_manifest.json"
    replay: dict[str, Any] | None = None
    replay_dirty = False
    modified_sources: list[Path] = []
    for name in _discover_release_datasets(output_root):
        directory = dataset / "datasets" / name
        if not directory.exists():
            continue
        sig_changed = _hash_arrow(directory / "signatures.arrow", scrub_free_text=scrub_free_text)
        papers_arrow = directory / "papers.arrow"
        papers_changed = _scrub_papers_arrow(papers_arrow) if scrub_free_text and papers_arrow.is_file() else 0
        authors_arrow = directory / "paper_authors.arrow"
        authors_changed = (
            _scrub_paper_authors_arrow(authors_arrow) if scrub_free_text and authors_arrow.is_file() else 0
        )
        changed = sig_changed + papers_changed + authors_changed

        source_dir = source_root / "corrected_replay" / "raw" / name
        source = source_dir / "signatures.json"
        source_sig_changed = _hash_source_json(source, scrub_free_text=scrub_free_text) if source.exists() else 0
        if source_sig_changed:
            modified_sources.append(source)
        source_papers = source_dir / "papers.json"
        source_papers_changed = _scrub_papers_json(source_papers) if scrub_free_text and source_papers.is_file() else 0
        if source_papers_changed:
            modified_sources.append(source_papers)
        source_changed = source_sig_changed + source_papers_changed

        report[name] = {"arrow_values": changed, "source_fields": source_changed}
        if not changed:
            continue
        table_counts = {"signatures": sig_changed, "papers": papers_changed, "paper_authors": authors_changed}
        changed_tables = [table for table, count in table_counts.items() if count]
        _refresh_arrow_dataset_manifest(directory, changed_tables)
        if replay_path.is_file():
            if replay is None:
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
            if isinstance(replay.get("dataset_manifests"), dict) and name in replay["dataset_manifests"]:
                replay["dataset_manifests"][name]["sha256"] = sha256_file(directory / "manifest.json")
                replay_dirty = True
    if modified_sources:
        _refresh_source_inventory(source_root, modified_sources)
    if replay is not None and replay_dirty:
        _write_json(replay_path, replay)
        public_path = dataset / "manifest.json"
        if public_path.is_file():
            public = json.loads(public_path.read_text(encoding="utf-8"))
            if isinstance(public.get("replay_bundles"), dict) and "incremental" in public["replay_bundles"]:
                public["replay_bundles"]["incremental"]["sha256"] = sha256_file(replay_path)
                _write_json(public_path, public)
    return report

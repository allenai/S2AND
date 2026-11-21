"""Hash public replay emails without changing their pairwise feature evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import ijson
import pyarrow as pa

from s2and._sha256 import sha256_file
from s2and.incremental_linking.feature_block_arrow import write_arrow_batch_lookup_index
from s2and.text import email_prefix_suffix

GIANT_DATASETS = ("a_khan", "a_silva", "h_wang", "j_smith", "s_gupta", "s_lee", "s_park")
_HASHED_EMAIL = re.compile(r"[0-9a-f]{64}@[0-9a-f]{64}|(?:hmac-sha256|sha256):[0-9a-f]{64}")


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


def _write_json(path: Path, payload: Any) -> None:
    # Replace the directory entry because release staging may use hardlinks.
    temporary = path.with_suffix(path.suffix + ".email_hashing.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _hash_arrow(path: Path) -> int:
    temporary = path.with_suffix(".email_hashing.tmp")
    changed = 0
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        column = reader.schema.get_field_index("author_email")
        with pa.OSFile(str(temporary), "wb") as sink, pa.ipc.new_file(sink, reader.schema) as writer:
            for index in range(reader.num_record_batches):
                batch = reader.get_batch(index)
                before = batch.column(column).to_pylist()
                after = [hash_feature_email(value) for value in before]
                changed += sum(a != b for a, b in zip(before, after, strict=True))
                writer.write_batch(batch.set_column(column, reader.schema.field(column), pa.array(after, pa.string())))
    if changed:
        temporary.replace(path)
    else:
        temporary.unlink()
    return changed


def _hash_source_json(path: Path) -> int:
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


def hash_release_emails(output_root: Path) -> dict[str, dict[str, int]]:
    """Hash giant replay email columns and source fields, then refresh checksum references.

    Args:
        output_root: Staged root containing the dataset and source bundles.

    Returns:
        Changed Arrow values and source fields for each giant dataset.
    """
    report = {}
    dataset = output_root / "dataset"
    replay_path = dataset / "replay_manifest.json"
    replay: dict[str, Any] | None = None
    for name in GIANT_DATASETS:
        directory = dataset / "datasets" / name
        if not directory.exists():
            continue
        changed = _hash_arrow(directory / "signatures.arrow")
        source = output_root / "source" / "corrected_replay" / "raw" / name / "signatures.json"
        source_changed = _hash_source_json(source) if source.exists() else 0
        report[name] = {"arrow_values": changed, "source_fields": source_changed}
        if not changed:
            continue
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = directory / manifest["paths"]["signatures_batch_index"]
        write_arrow_batch_lookup_index(
            directory / "signatures.arrow", index_path, key_column="signature_id", table_name="signatures"
        )
        for role in ("signatures", "signatures_batch_index"):
            path = directory / manifest["paths"][role]
            manifest["files"][role] = {"byte_count": path.stat().st_size, "sha256": sha256_file(path)}
        _write_json(manifest_path, manifest)
        if replay is None:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
        replay["dataset_manifests"][name]["sha256"] = sha256_file(manifest_path)
    if replay is not None:
        _write_json(replay_path, replay)
        public_path = dataset / "manifest.json"
        public = json.loads(public_path.read_text(encoding="utf-8"))
        public["replay_bundles"]["incremental"]["sha256"] = sha256_file(replay_path)
        _write_json(public_path, public)
    return report

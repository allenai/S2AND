"""Generate canonical name-count artifacts with explicit warehouse guardrails."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from s2and._atomic_io import exclusive_file_lock, fsync_directory
from s2and.consts import NORMALIZATION_VERSION
from s2and.text import canonical_name_count_keys, canonicalize_name_parts

QUERY = """
select concat(concat(nvl(first_name, ''), '|||'), nvl(last_name, '')) as concat,
       count(*) as count
from content.authors
group by concat(concat(nvl(first_name, ''), '|||'), nvl(last_name, ''))
""".strip()

NameCountMappings = tuple[
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, int],
]


def _query_text(limit: int | None) -> str:
    """Return the exact deterministic warehouse query for this run."""

    ordered = f"{QUERY}\norder by concat"
    return ordered if limit is None else f"{ordered}\nlimit {int(limit)}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-full",
        action="store_true",
        help="authorize access to the internal warehouse",
    )
    source.add_argument(
        "--fixture-input",
        type=Path,
        help="local JSON rows with first_name, last_name, and count",
    )
    parser.add_argument("--source-snapshot-id", required=True)
    parser.add_argument("--limit", type=int, help="Maximum source rows")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace the published manifest pointer")
    return parser


def _validated_limit(value: int | None) -> int | None:
    if value is not None and value < 1:
        raise ValueError("--limit must be positive")
    return value


def _query_rows(limit: int | None) -> Iterator[tuple[str, int]]:
    try:
        from pys2 import _evaluate_redshift_query  # type: ignore
    except ImportError as exc:
        raise RuntimeError("warehouse generation requires the internal pys2 package") from exc
    frame = _evaluate_redshift_query(_query_text(limit))
    for raw, count in zip(frame["concat"], frame["count"], strict=True):
        yield str(raw), int(count)


def _fixture_rows(path: Path, limit: int | None) -> Iterator[tuple[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("fixture input must contain a JSON list")
    for row_index, row in enumerate(payload):
        if limit is not None and row_index >= limit:
            return
        if not isinstance(row, dict):
            raise ValueError(f"fixture row {row_index} must be an object")
        first = row.get("first_name", "")
        last = row.get("last_name", "")
        count = row.get("count")
        if not isinstance(first, str) or not isinstance(last, str):
            raise ValueError(f"fixture row {row_index} names must be strings")
        if not isinstance(count, int) or count < 1:
            raise ValueError(f"fixture row {row_index} count must be a positive integer")
        yield f"{first}|||{last}", count


def build_name_count_dicts(
    rows: Iterable[tuple[str, int]],
) -> tuple[NameCountMappings, dict[str, int | str]]:
    """Canonicalize source rows and return the four filtered lookup mappings."""

    counters: list[Counter[str]] = [Counter(), Counter(), Counter(), Counter()]
    key_names = ("first", "last", "first_last", "last_first_initial")
    source_rows = 0
    rejected_rows = 0
    selected_rows_digest = hashlib.sha256()
    for raw_concat, count in rows:
        source_rows += 1
        raw_bytes = raw_concat.encode("utf-8")
        selected_rows_digest.update(len(raw_bytes).to_bytes(8, "little", signed=False))
        selected_rows_digest.update(raw_bytes)
        selected_rows_digest.update(int(count).to_bytes(8, "little", signed=True))
        raw_first, separator, raw_last = raw_concat.partition("|||")
        if not separator:
            rejected_rows += 1
            continue
        keys = canonical_name_count_keys(canonicalize_name_parts(raw_first, None, raw_last))
        accepted = False
        for counter, key_name in zip(counters, key_names, strict=True):
            value = keys[key_name]
            if value is not None:
                counter[value] += count
                accepted = True
        if not accepted:
            rejected_rows += 1
    # Keep only the four Counters themselves resident.  Constructing four full
    # filtered dictionaries temporarily doubled this already-large artifact.
    for counter in counters:
        rejected_keys = [key for key, value in counter.items() if value <= 1]
        for key in rejected_keys:
            del counter[key]
    mappings = (counters[0], counters[1], counters[2], counters[3])
    return mappings, {
        "source_row_count": source_rows,
        "selected_row_count": source_rows,
        "selected_rows_sha256": selected_rows_digest.hexdigest(),
        "rejected_row_count": rejected_rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as source:
        os.fsync(source.fileno())


@contextmanager
def _publish_lock(root: Path) -> Iterator[None]:
    with exclusive_file_lock(root / ".publish.lock"):
        yield


def publish_name_counts(
    mappings: NameCountMappings,
    *,
    output_dir: Path,
    source_snapshot_id: str,
    source_kind: str,
    query_digest: str,
    row_metrics: Mapping[str, int | str],
    overwrite: bool,
) -> dict[str, Any]:
    """Publish one immutable generation and replace its manifest last."""

    if len(query_digest) != 64:
        raise ValueError("query_digest must be a SHA-256 hex digest")
    if row_metrics.get("source_row_count") != row_metrics.get("selected_row_count"):
        raise ValueError("row_metrics source_row_count/selected_row_count mismatch")
    selected_rows_sha256 = row_metrics.get("selected_rows_sha256")
    if not isinstance(selected_rows_sha256, str) or len(selected_rows_sha256) != 64:
        raise ValueError("row_metrics requires selected_rows_sha256")
    safe_snapshot = re.sub(r"[^A-Za-z0-9._-]+", "-", source_snapshot_id).strip("-")
    if not safe_snapshot:
        raise ValueError("--source-snapshot-id must contain a filename-safe character")
    root = output_dir / "name_counts"
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation_id = f"{safe_snapshot}-{uuid.uuid4().hex}"
    staging = Path(tempfile.mkdtemp(prefix=f".{generation_id}.", dir=str(generations)))
    final_generation = generations / generation_id
    manifest_path = root / "manifest.json"
    manifest_tmp = root / f".manifest.{generation_id}.json"
    generation_renamed = False
    manifest_committed = False
    try:
        pickle_path = staging / "name_counts.pickle"
        with pickle_path.open("wb") as output:
            pickle.dump(mappings, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.flush()
            os.fsync(output.fileno())
        cardinalities = dict(
            zip(
                ("first", "last", "first_last", "last_first_initial"),
                (len(mapping) for mapping in mappings),
                strict=True,
            )
        )
        metadata = {
            "schema_version": "name_counts_provenance_v1",
            "normalization_version": NORMALIZATION_VERSION,
            "generation_id": generation_id,
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "source_kind": source_kind,
            "source_snapshot_id": source_snapshot_id,
            "source_query_sha256": query_digest,
            "pickle_sha256": _sha256(pickle_path),
            "pickle_byte_count": pickle_path.stat().st_size,
            "cardinalities": cardinalities,
            **row_metrics,
        }
        metadata_path = staging / "name_counts.meta.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _fsync_file(metadata_path)
        fsync_directory(staging)
        provenance_sha256 = _sha256(metadata_path)
        provenance_byte_count = metadata_path.stat().st_size
        manifest = {
            "schema_version": "name_counts_manifest_v1",
            "normalization_version": NORMALIZATION_VERSION,
            "generation_id": generation_id,
            "source_snapshot_id": source_snapshot_id,
            "files": {
                "pickle": f"generations/{generation_id}/name_counts.pickle",
                "provenance": f"generations/{generation_id}/name_counts.meta.json",
            },
            "pickle_sha256": metadata["pickle_sha256"],
            "provenance_sha256": provenance_sha256,
            "provenance_byte_count": provenance_byte_count,
        }
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _fsync_file(manifest_tmp)
        with _publish_lock(root):
            if manifest_path.exists() and not overwrite:
                raise FileExistsError(f"published manifest already exists: {manifest_path}; pass --overwrite")
            staging.rename(final_generation)
            generation_renamed = True
            fsync_directory(generations)
            manifest_tmp.replace(manifest_path)
            manifest_committed = True
            fsync_directory(root)
        return {**metadata, "manifest_path": str(manifest_path.resolve())}
    finally:
        manifest_tmp.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
        if generation_renamed and not manifest_committed and final_generation.exists():
            shutil.rmtree(final_generation)
            fsync_directory(generations)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    limit = _validated_limit(args.limit)
    source_kind = "redshift:content.authors" if args.run_full else f"fixture:{args.fixture_input.resolve()}"
    query_text = _query_text(limit)
    plan = {
        "source_kind": source_kind,
        "source_snapshot_id": args.source_snapshot_id,
        "limit": limit,
        "output_dir": str(args.output_dir.resolve()),
        "query_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
        "query": query_text,
        "cost_context": "internal warehouse scan; inspect the snapshot and limit before authorizing --run-full",
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps({"plan": plan}, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    rows = _query_rows(limit) if args.run_full else _fixture_rows(args.fixture_input, limit)
    mappings, row_metrics = build_name_count_dicts(rows)
    metadata = publish_name_counts(
        mappings,
        output_dir=args.output_dir,
        source_snapshot_id=args.source_snapshot_id,
        source_kind=source_kind,
        query_digest=plan["query_sha256"],
        row_metrics=row_metrics,
        overwrite=bool(args.overwrite),
    )
    from s2and.incremental_linking.feature_block_arrow import write_name_counts_index

    index_path, index_metrics = write_name_counts_index(
        args.output_dir,
        mappings,
        metadata,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {"result": metadata, "name_counts_index": index_path, "name_counts_index_metrics": index_metrics},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

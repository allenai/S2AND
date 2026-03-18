"""Extract a monolithic big-block export into ANDData-friendly files.

This converts a JSON file with top-level `signatures`, `papers`, and
`paper_embeddings` sections into:

- `signatures.json`
- `papers.json`
- `specter.pickle`
- `cluster_seeds.json`
- `altered_cluster_signatures.txt`
- `meta.json`

The extractor is streaming and does not materialize the full monolith in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SIGNATURES_START = '  "signatures" : [ {'
PAPERS_START = '  "papers" : [ {'
PAPER_EMBEDDINGS_START = '  "paper_embeddings" : {'
OBJECT_SEPARATOR = "  }, {"
OBJECT_SECTION_END = re.compile(r"^  }\s*],?$")
EMBEDDING_SECTION_END = re.compile(r"^  },?$")
EMBEDDING_ENTRY = re.compile(r'^    "([^"]+)" : (\[.*\]),?$')

SIGNATURE_LOG_INTERVAL = 25_000
PAPER_LOG_INTERVAL = 25_000
EMBEDDING_LOG_INTERVAL = 25_000


@dataclass(slots=True)
class MonolithCensus:
    """Summary of the selected subset inside the monolithic export."""

    input_path: Path
    input_bytes: int
    signature_count: int
    paper_count: int
    embedding_count: int
    embedding_dim: int | None
    block_counts: Counter[str]
    needed_paper_ids: set[str]
    limit_signatures: int | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit-signatures",
        type=int,
        default=None,
        help="Optional debug cap. When set, only the first N signatures are selected.",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Required for an unbounded full extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite known output files if they already exist.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    if args.limit_signatures is not None and args.limit_signatures <= 0:
        raise ValueError("--limit-signatures must be > 0 when set")
    if args.limit_signatures is None and not args.full_run:
        raise ValueError("Refusing unbounded extraction without explicit confirmation. Use --full-run.")


def _section_for_line(line: str) -> str | None:
    """Return the section name for a top-level section start line."""
    if line == SIGNATURES_START:
        return "signatures"
    if line == PAPERS_START:
        return "papers"
    if line == PAPER_EMBEDDINGS_START:
        return "paper_embeddings"
    return None


def _parse_json_object(lines: list[str]) -> dict[str, Any]:
    """Parse one top-level signature or paper object."""
    return json.loads("\n".join(lines))


def _parse_embedding_line(line: str) -> tuple[str, np.ndarray]:
    """Parse one `paper_embeddings` entry."""
    match = EMBEDDING_ENTRY.match(line)
    if match is None:
        raise ValueError(f"Malformed embedding line: {line[:200]}")
    key = match.group(1)
    vector = np.fromstring(match.group(2).strip()[1:-1], sep=",", dtype=np.float32)
    return key, vector


def iter_monolith_records(path: Path) -> Iterator[tuple[str, Any]]:
    """Yield signature objects, paper objects, and embedding rows from the monolith."""
    active_object_section: str | None = None
    current_object_lines: list[str] | None = None
    active_section: str | None = None

    with path.open("r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")

            if active_object_section is None:
                maybe_section = _section_for_line(line)
                if maybe_section in {"signatures", "papers"}:
                    active_object_section = maybe_section
                    active_section = maybe_section
                    current_object_lines = ["{"]
                    continue
                if maybe_section == "paper_embeddings":
                    active_section = maybe_section
                    continue

            if active_object_section is not None:
                if line == OBJECT_SEPARATOR:
                    assert current_object_lines is not None
                    current_object_lines.append("}")
                    yield active_object_section, _parse_json_object(current_object_lines)
                    current_object_lines = ["{"]
                    continue
                if OBJECT_SECTION_END.match(line):
                    assert current_object_lines is not None
                    current_object_lines.append("}")
                    yield active_object_section, _parse_json_object(current_object_lines)
                    current_object_lines = None
                    active_object_section = None
                    active_section = None
                    continue
                assert current_object_lines is not None
                current_object_lines.append(line)
                continue

            if active_section == "paper_embeddings":
                if EMBEDDING_SECTION_END.match(line):
                    active_section = None
                    continue
                if not line.startswith("    "):
                    continue
                yield "paper_embedding", _parse_embedding_line(line)


def census_monolith(path: Path, *, limit_signatures: int | None = None) -> MonolithCensus:
    """Scan the monolith once and collect counts for the selected subset."""
    signature_count = 0
    paper_count = 0
    embedding_count = 0
    embedding_dim: int | None = None
    block_counts: Counter[str] = Counter()
    needed_paper_ids: set[str] = set()

    for record_type, payload in iter_monolith_records(path):
        if record_type == "signatures":
            if limit_signatures is not None and signature_count >= limit_signatures:
                continue
            signature_count += 1
            signature = payload
            paper_id = str(signature["paper_id"])
            needed_paper_ids.add(paper_id)
            block_counts[str(signature["author_info"]["block"])] += 1
        elif record_type == "papers":
            paper = payload
            paper_id = str(paper["paper_id"])
            if paper_id in needed_paper_ids:
                paper_count += 1
        elif record_type == "paper_embedding":
            paper_id, vector = payload
            if paper_id in needed_paper_ids:
                if embedding_dim is None:
                    embedding_dim = int(vector.shape[0])
                elif int(vector.shape[0]) != embedding_dim:
                    raise ValueError(
                        f"Inconsistent embedding dimension for paper_id={paper_id}: "
                        f"{vector.shape[0]} != {embedding_dim}"
                    )
                embedding_count += 1

    return MonolithCensus(
        input_path=path,
        input_bytes=path.stat().st_size,
        signature_count=signature_count,
        paper_count=paper_count,
        embedding_count=embedding_count,
        embedding_dim=embedding_dim,
        block_counts=block_counts,
        needed_paper_ids=needed_paper_ids,
        limit_signatures=limit_signatures,
    )


def _target_paths(output_dir: Path) -> dict[str, Path]:
    """Return the canonical output file paths."""
    return {
        "signatures": output_dir / "signatures.json",
        "papers": output_dir / "papers.json",
        "specter": output_dir / "specter.pickle",
        "cluster_seeds": output_dir / "cluster_seeds.json",
        "altered_cluster_signatures": output_dir / "altered_cluster_signatures.txt",
        "meta": output_dir / "meta.json",
    }


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> dict[str, Path]:
    """Create the output directory and validate target-file overwrite behavior."""
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = _target_paths(output_dir)
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        existing_names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output directory already contains target files ({existing_names}). " "Use --overwrite to replace them."
        )
    return targets


def _write_mapping_entry(outfile, *, key: str, payload: dict[str, Any], first_entry: bool) -> None:
    """Write one JSON mapping entry to an open output file."""
    if not first_entry:
        outfile.write(",\n")
    outfile.write(json.dumps(key))
    outfile.write(":")
    # ANDData opens JSON paths without an explicit UTF-8 encoding on Windows, so
    # emit ASCII-escaped JSON for compatibility with the existing loader contract.
    json.dump(payload, outfile, ensure_ascii=True, separators=(",", ":"))


def extract_monolith_dataset(
    input_path: Path,
    output_dir: Path,
    *,
    limit_signatures: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract the selected subset into ANDData-friendly files."""
    targets = _prepare_output_dir(output_dir, overwrite=overwrite)
    census = census_monolith(input_path, limit_signatures=limit_signatures)

    temp_paths = {name: path.with_suffix(path.suffix + ".tmp") for name, path in targets.items() if name != "meta"}
    for temp_path in temp_paths.values():
        if temp_path.exists():
            temp_path.unlink()

    start = time.perf_counter()
    signatures_written = 0
    papers_written = 0
    embeddings_written = 0

    embedding_keys: list[str] = []
    matrix_tmp_path = output_dir / "specter.matrix.tmp"
    if matrix_tmp_path.exists():
        matrix_tmp_path.unlink()

    embedding_matrix = None
    if census.embedding_count > 0:
        assert census.embedding_dim is not None
        embedding_matrix = np.memmap(
            matrix_tmp_path,
            dtype=np.float32,
            mode="w+",
            shape=(census.embedding_count, census.embedding_dim),
        )

    with (
        temp_paths["signatures"].open("w", encoding="utf-8") as signatures_out,
        temp_paths["papers"].open("w", encoding="utf-8") as papers_out,
    ):
        signatures_out.write("{\n")
        papers_out.write("{\n")
        first_signature = True
        first_paper = True

        for record_type, payload in iter_monolith_records(input_path):
            if record_type == "signatures":
                if limit_signatures is not None and signatures_written >= limit_signatures:
                    continue
                signature = payload
                _write_mapping_entry(
                    signatures_out,
                    key=str(signature["signature_id"]),
                    payload=signature,
                    first_entry=first_signature,
                )
                first_signature = False
                signatures_written += 1
                if signatures_written % SIGNATURE_LOG_INTERVAL == 0:
                    elapsed = time.perf_counter() - start
                    print(f"[extract] signatures={signatures_written:,} elapsed={elapsed:,.1f}s")
            elif record_type == "papers":
                paper = payload
                paper_id = str(paper["paper_id"])
                if paper_id not in census.needed_paper_ids:
                    continue
                _write_mapping_entry(
                    papers_out,
                    key=paper_id,
                    payload=paper,
                    first_entry=first_paper,
                )
                first_paper = False
                papers_written += 1
                if papers_written % PAPER_LOG_INTERVAL == 0:
                    elapsed = time.perf_counter() - start
                    print(f"[extract] papers={papers_written:,} elapsed={elapsed:,.1f}s")
            elif record_type == "paper_embedding":
                paper_id, vector = payload
                if paper_id not in census.needed_paper_ids:
                    continue
                if embedding_matrix is None:
                    raise AssertionError("Encountered embedding rows despite zero expected embeddings")
                if census.embedding_dim is None:
                    raise AssertionError("Missing embedding dimension during write pass")
                if int(vector.shape[0]) != census.embedding_dim:
                    raise ValueError(
                        f"Inconsistent embedding dimension for paper_id={paper_id}: "
                        f"{vector.shape[0]} != {census.embedding_dim}"
                    )
                embedding_matrix[embeddings_written, :] = vector
                embedding_keys.append(paper_id)
                embeddings_written += 1
                if embeddings_written % EMBEDDING_LOG_INTERVAL == 0:
                    elapsed = time.perf_counter() - start
                    print(f"[extract] embeddings={embeddings_written:,} elapsed={elapsed:,.1f}s")

        signatures_out.write("\n}\n")
        papers_out.write("\n}\n")

    if embedding_matrix is not None:
        embedding_matrix.flush()
        with temp_paths["specter"].open("wb") as specter_out:
            pickle.dump((np.asarray(embedding_matrix), embedding_keys), specter_out, protocol=pickle.HIGHEST_PROTOCOL)
        del embedding_matrix
        if matrix_tmp_path.exists():
            matrix_tmp_path.unlink()
    else:
        with temp_paths["specter"].open("wb") as specter_out:
            pickle.dump((np.zeros((0, 0), dtype=np.float32), []), specter_out, protocol=pickle.HIGHEST_PROTOCOL)

    with temp_paths["cluster_seeds"].open("w", encoding="utf-8") as cluster_seeds_out:
        json.dump({}, cluster_seeds_out)
        cluster_seeds_out.write("\n")
    with temp_paths["altered_cluster_signatures"].open("w", encoding="utf-8") as altered_out:
        altered_out.write("")

    if signatures_written != census.signature_count:
        raise AssertionError(f"Wrote {signatures_written} signatures, expected {census.signature_count}")
    if papers_written != census.paper_count:
        raise AssertionError(f"Wrote {papers_written} papers, expected {census.paper_count}")
    if embeddings_written != census.embedding_count:
        raise AssertionError(f"Wrote {embeddings_written} embeddings, expected {census.embedding_count}")

    meta = {
        "input_path": str(input_path),
        "input_bytes": int(census.input_bytes),
        "limit_signatures": census.limit_signatures,
        "signature_count": int(census.signature_count),
        "paper_count": int(census.paper_count),
        "embedding_count": int(census.embedding_count),
        "embedding_dim": int(census.embedding_dim) if census.embedding_dim is not None else None,
        "unique_blocks": int(len(census.block_counts)),
        "top_blocks": [[block, int(count)] for block, count in census.block_counts.most_common(20)],
        "created_at_epoch_seconds": time.time(),
        "output_dir": str(output_dir),
        "outputs": {name: str(path) for name, path in targets.items()},
        "elapsed_seconds": time.perf_counter() - start,
    }

    meta_tmp_path = targets["meta"].with_suffix(targets["meta"].suffix + ".tmp")
    with meta_tmp_path.open("w", encoding="utf-8") as meta_out:
        json.dump(meta, meta_out, indent=2)
        meta_out.write("\n")

    for name, temp_path in temp_paths.items():
        os.replace(temp_path, targets[name])
    os.replace(meta_tmp_path, targets["meta"])

    return meta


def main() -> None:
    """Run the extractor CLI."""
    args = parse_args()
    validate_args(args)
    meta = extract_monolith_dataset(
        args.input_path,
        args.output_dir,
        limit_signatures=args.limit_signatures,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

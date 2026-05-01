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
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SIGNATURE_LOG_INTERVAL = 25_000
PAPER_LOG_INTERVAL = 25_000
EMBEDDING_LOG_INTERVAL = 25_000
STREAM_CHUNK_SIZE = 1 << 20


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


class JsonStream:
    """Incrementally read JSON tokens from a text file without loading it all."""

    def __init__(self, path: Path, *, chunk_size: int | None = None) -> None:
        self.path = path
        self.chunk_size = STREAM_CHUNK_SIZE if chunk_size is None else chunk_size
        self._buffer = ""
        self._position = 0
        self._eof = False
        self._infile = path.open("r", encoding="utf-8")

    def close(self) -> None:
        """Close the underlying file handle."""
        self._infile.close()

    def __enter__(self) -> JsonStream:
        """Enter the stream context."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the stream context."""
        self.close()

    def _fill(self) -> None:
        """Append more text to the buffer if the file has remaining content."""
        if self._eof:
            return
        chunk = self._infile.read(self.chunk_size)
        if chunk == "":
            self._eof = True
            return
        self._buffer += chunk

    def _compact(self) -> None:
        """Drop already-consumed bytes so the buffer stays bounded."""
        if self._position == 0:
            return
        if self._position >= self.chunk_size or self._position == len(self._buffer):
            self._buffer = self._buffer[self._position :]
            self._position = 0

    def _ensure_available(self) -> None:
        """Ensure there is at least one character available or the stream is at EOF."""
        while self._position >= len(self._buffer) and not self._eof:
            self._fill()

    def _skip_whitespace(self) -> None:
        """Advance past JSON whitespace."""
        while True:
            self._ensure_available()
            while self._position < len(self._buffer) and self._buffer[self._position].isspace():
                self._position += 1
            if self._position < len(self._buffer) or self._eof:
                break
        self._compact()

    def peek(self) -> str:
        """Return the next non-whitespace character without consuming it."""
        self._skip_whitespace()
        self._ensure_available()
        if self._position >= len(self._buffer):
            raise EOFError(f"Unexpected end of file while reading {self.path}")
        return self._buffer[self._position]

    def read_char(self, expected: str | None = None) -> str:
        """Consume and return the next non-whitespace character."""
        char = self.peek()
        if expected is not None and char != expected:
            raise ValueError(f"Expected {expected!r}, found {char!r} in {self.path}")
        self._position += 1
        self._compact()
        return char

    def read_json_value_text(self) -> str:
        """Read the next JSON value and return its source text."""
        self._skip_whitespace()
        self._ensure_available()
        if self._position >= len(self._buffer):
            raise EOFError(f"Unexpected end of file while reading {self.path}")

        start = self._position
        token = self._buffer[self._position]
        if token in "[{":
            return self._read_balanced_value(start)
        if token == '"':
            return self._read_string_value(start)
        return self._read_scalar_value(start)

    def _read_balanced_value(self, start: int) -> str:
        """Read an object or array, preserving nested content."""
        closing_by_open = {"{": "}", "[": "]"}
        stack: list[str] = [closing_by_open[self._buffer[start]]]
        in_string = False
        escaped = False
        self._position = start + 1

        while stack:
            self._ensure_available()
            if self._position >= len(self._buffer):
                raise EOFError(f"Unexpected end of file while reading structured JSON in {self.path}")

            char = self._buffer[self._position]
            self._position += 1

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in closing_by_open:
                stack.append(closing_by_open[char])
            elif char in {"}", "]"}:
                expected = stack.pop()
                if char != expected:
                    raise ValueError(f"Malformed JSON nesting in {self.path}: expected {expected!r}, found {char!r}")

        value_text = self._buffer[start : self._position]
        self._compact()
        return value_text

    def _read_string_value(self, start: int) -> str:
        """Read a JSON string token."""
        escaped = False
        self._position = start + 1

        while True:
            self._ensure_available()
            if self._position >= len(self._buffer):
                raise EOFError(f"Unexpected end of file while reading string JSON in {self.path}")

            char = self._buffer[self._position]
            self._position += 1
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                value_text = self._buffer[start : self._position]
                self._compact()
                return value_text

    def _read_scalar_value(self, start: int) -> str:
        """Read a JSON scalar such as a number, boolean, or null."""
        while True:
            self._ensure_available()
            if self._position >= len(self._buffer):
                break
            char = self._buffer[self._position]
            if char in {",", "]", "}"} or char.isspace():
                break
            self._position += 1

        value_text = self._buffer[start : self._position]
        self._compact()
        return value_text


def _read_json_string(stream: JsonStream) -> str:
    """Read one JSON string token and decode it."""
    value = json.loads(stream.read_json_value_text())
    if not isinstance(value, str):
        raise ValueError(f"Expected a JSON string in {stream.path}, found {type(value).__name__}")
    return value


def _consume_delimited_sequence_end(stream: JsonStream, *, separator: str, end: str) -> bool:
    """Consume a separator or closing delimiter and report whether more items remain."""
    token = stream.peek()
    if token == separator:
        stream.read_char(separator)
        return True
    if token == end:
        stream.read_char(end)
        return False
    raise ValueError(f"Expected {separator!r} or {end!r}, found {token!r} in {stream.path}")


def _iter_object_section(stream: JsonStream, record_type: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield objects from a top-level JSON array."""
    stream.read_char("[")
    if stream.peek() == "]":
        stream.read_char("]")
        return

    while True:
        payload = json.loads(stream.read_json_value_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object entries in {record_type}, found {type(payload).__name__}")
        yield record_type, payload
        if not _consume_delimited_sequence_end(stream, separator=",", end="]"):
            return


def _iter_embedding_section(stream: JsonStream) -> Iterator[tuple[str, tuple[str, np.ndarray]]]:
    """Yield `(paper_id, vector)` rows from the top-level embedding mapping."""
    stream.read_char("{")
    if stream.peek() == "}":
        stream.read_char("}")
        return

    while True:
        key = _read_json_string(stream)
        stream.read_char(":")
        vector_text = stream.read_json_value_text().strip()
        if not (vector_text.startswith("[") and vector_text.endswith("]")):
            raise ValueError(f"Expected embedding array for paper_id={key!r} in {stream.path}")
        vector = np.fromstring(vector_text[1:-1], sep=",", dtype=np.float32)
        yield "paper_embedding", (key, vector)
        if not _consume_delimited_sequence_end(stream, separator=",", end="}"):
            return


def iter_monolith_records(path: Path, *, chunk_size: int | None = None) -> Iterator[tuple[str, Any]]:
    """Yield signature objects, paper objects, and embedding rows from the monolith."""
    with JsonStream(path, chunk_size=chunk_size) as stream:
        stream.read_char("{")
        if stream.peek() == "}":
            stream.read_char("}")
            return

        while True:
            section_name = _read_json_string(stream)
            stream.read_char(":")

            if section_name in {"signatures", "papers"}:
                yield from _iter_object_section(stream, section_name)
            elif section_name == "paper_embeddings":
                yield from _iter_embedding_section(stream)
            else:
                stream.read_json_value_text()

            if not _consume_delimited_sequence_end(stream, separator=",", end="}"):
                return


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

"""Prepare current metadata and SPECTER2 inputs for the fixed linker replay IDs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import shutil
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPLAY_DATASETS = (
    "a_khan",
    "a_silva",
    "arnetminer",
    "h_wang",
    "inspire",
    "j_smith",
    "kisti",
    "pubmed",
    "qian",
    "s_gupta",
    "s_lee",
    "s_park",
    "zbmath",
)
BENCHMARK_DATASETS = frozenset({"arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath"})
MAX_SOURCE_BYTES = 30 * 1024**3


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _vectors(path: Path) -> tuple[np.ndarray, list[str], dict[str, int]]:
    with path.open("rb") as stream:
        vectors, original_keys = pickle.load(stream)
    vectors = np.asarray(vectors)
    keys = list(map(str, original_keys))
    if vectors.ndim != 2 or vectors.shape != (len(keys), 768):
        raise ValueError(f"Expected one 768-dimensional SPECTER2 vector per key: {path}")
    index = {key: position for position, key in enumerate(keys)}
    if len(index) != len(keys):
        raise ValueError(f"Duplicate SPECTER2 paper IDs: {path}")
    return vectors, keys, index


def _prepare_dataset(
    dataset: str, raw_root: Path, embeddings_root: Path, benchmark_root: Path, output_root: Path
) -> dict[str, Any]:
    raw_source = raw_root / dataset
    old_embedding_path = embeddings_root / dataset / "specter2.pkl"
    raw_output = output_root / "raw" / dataset
    embedding_output = output_root / "embeddings" / dataset
    raw_output.mkdir(parents=True)
    embedding_output.mkdir(parents=True)
    source_files = [raw_source / "signatures.json", raw_source / "papers.json", old_embedding_path]
    if dataset not in BENCHMARK_DATASETS:
        destinations = [raw_output / "signatures.json", raw_output / "papers.json", embedding_output / "specter2.pkl"]
        for source, destination in zip(source_files, destinations, strict=True):
            shutil.copyfile(source, destination)
            if _sha256(source) != _sha256(destination):
                raise ValueError(f"The unchanged replay copy differs from its source: {dataset}/{source.name}")
        return {"dataset": dataset, "policy": "unchanged_giant_source", "byte_exact_copies": True}

    benchmark = benchmark_root / dataset
    signature_path = benchmark / f"{dataset}_signatures.json"
    paper_path = benchmark / f"{dataset}_papers.json"
    fresh_embedding_path = benchmark / f"{dataset}_specter2.pkl"
    source_files.extend([signature_path, paper_path, fresh_embedding_path])
    old_signatures = _load_json(source_files[0])
    current_signatures = _load_json(signature_path)
    missing_ids = set(old_signatures) - set(current_signatures)
    if missing_ids:
        raise ValueError(f"Current {dataset} metadata omits {len(missing_ids)} fixed replay signature IDs")
    signatures = {sid: copy.deepcopy(current_signatures[sid]) for sid in old_signatures}
    projected_email_count = 0
    changed_fields: Counter[str] = Counter()
    for sid, row in signatures.items():
        if str(row["signature_id"]) != sid:
            raise ValueError(f"The current signature ID differs from its object key: {dataset}/{sid}")
        info = row["author_info"]
        if "email" not in info:
            emails = info.get("emails") or []
            info["email"] = emails[0] if len(emails) == 1 else None
            projected_email_count += 1
        for field in ("first", "middle", "last", "suffix", "block", "position", "affiliations", "email"):
            if info.get(field) != old_signatures[sid]["author_info"].get(field):
                changed_fields[field] += 1
        if row["paper_id"] != old_signatures[sid]["paper_id"]:
            changed_fields["paper_id"] += 1

    old_papers = _load_json(source_files[1])
    current_papers = _load_json(paper_path)
    needed_papers = {str(row["paper_id"]) for row in signatures.values()}
    if needed_papers - set(current_papers):
        raise ValueError(f"Current {dataset} papers omit fixed replay signature references")
    papers = {pid: current_papers.get(pid, row) for pid, row in old_papers.items()}
    for pid in sorted(needed_papers - set(papers)):
        papers[pid] = current_papers[pid]
    changed_text = {
        pid
        for pid in old_papers
        if pid in current_papers
        and any(old_papers[pid].get(field) != current_papers[pid].get(field) for field in ("title", "abstract"))
    }
    old_vectors, old_keys, old_index = _vectors(old_embedding_path)
    fresh_vectors, _fresh_keys, fresh_index = _vectors(fresh_embedding_path)
    if set(old_keys) - set(old_papers):
        raise ValueError(f"The old {dataset} vectors have no raw paper record for encoder-input verification")
    keys = list(old_keys)
    keys.extend(sorted(needed_papers - set(keys)))
    output_keys = []
    output_vectors = []
    fresh_count = retained_count = different_count = 0
    for key in keys:
        if key in fresh_index:
            vector = fresh_vectors[fresh_index[key]]
            fresh_count += 1
            if key in old_index and not np.array_equal(vector, old_vectors[old_index[key]]):
                different_count += 1
        elif key in old_index:
            if key in changed_text:
                raise ValueError(f"Changed {dataset} encoder input lacks a current SPECTER2 vector: {key}")
            vector = old_vectors[old_index[key]]
            retained_count += 1
        else:
            continue
        if not np.isfinite(vector).all() or not np.any(vector != 0):
            raise ValueError(f"Invalid non-missing SPECTER2 vector: {dataset}/{key}")
        output_keys.append(key)
        output_vectors.append(vector)
    matrix = np.stack(output_vectors) if output_vectors else np.empty((0, 768), dtype=fresh_vectors.dtype)
    _write_json(raw_output / "signatures.json", signatures)
    _write_json(raw_output / "papers.json", papers)
    with (embedding_output / "specter2.pkl").open("xb") as stream:
        pickle.dump((matrix, np.asarray(output_keys, dtype=str)), stream, protocol=4)
    # Check the written file, not the in-memory copy, against the fixed IDs and current raw names.
    saved = _load_json(raw_output / "signatures.json")
    ids_and_order_equal = list(saved) == list(old_signatures)
    raw_name_parts_exact = all(
        row["author_info"][field] == current_signatures[sid]["author_info"][field]
        for sid, row in saved.items()
        for field in ("first", "middle", "last")
    )
    if not ids_and_order_equal or not raw_name_parts_exact:
        raise ValueError(f"The saved {dataset} replay signatures differ from the fixed IDs or current raw names")
    return {
        "dataset": dataset,
        "policy": "current_raw_benchmark_metadata",
        "signature_count": len(signatures),
        "signature_ids_and_order_equal": ids_and_order_equal,
        "changed_signature_records": sum(signatures[sid] != row for sid, row in old_signatures.items()),
        "changed_signature_fields": dict(changed_fields),
        "scalar_email_projection_count": projected_email_count,
        "paper_count": len(papers),
        "old_paper_count": len(old_papers),
        "changed_paper_records": sum(papers[pid] != row for pid, row in old_papers.items()),
        "changed_title_or_abstract_records": len(changed_text),
        "fresh_vectors": fresh_count,
        "retained_unchanged_input_vectors": retained_count,
        "numerically_changed_old_vectors": different_count,
        "added_vectors": len(output_keys) - len(old_keys),
        "old_missing_embedding_count": len(set(old_papers) - set(old_index)),
        "missing_embedding_count": len(set(papers) - set(output_keys)),
        "raw_name_parts_exact": raw_name_parts_exact,
        "source_files": [{"path": str(path), "sha256": _sha256(path)} for path in source_files],
    }


def prepare_linker_replay(
    *,
    raw_root: Path,
    embeddings_root: Path,
    benchmark_root: Path,
    output_root: Path,
    datasets: Sequence[str],
    n_jobs: int = 1,
    max_source_bytes: int = MAX_SOURCE_BYTES,
) -> dict[str, Any]:
    """Prepare a bounded replay source without a new model prediction.

    Args:
        raw_root: Frozen replay signature and paper JSON directories.
        embeddings_root: Frozen replay SPECTER2 directories.
        benchmark_root: Current benchmark JSON and SPECTER2 directories.
        output_root: New output directory for raw and embedding inputs.
        datasets: At most the 13 known replay dataset names.
        n_jobs: At most 20 concurrent dataset tasks. Use one for low RAM use.
        max_source_bytes: Maximum total bytes in selected input files.

    Returns:
        Dataset counts, vector changes, source identities, and output paths.
    """
    names = tuple(datasets)
    if not names or len(set(names)) != len(names) or set(names) - set(REPLAY_DATASETS):
        raise ValueError("Select distinct names from the 13 fixed replay datasets")
    if not 1 <= n_jobs <= 20 or max_source_bytes < 1:
        raise ValueError("Use 1-20 jobs and a positive source-byte bound")
    raw_root, embeddings_root, benchmark_root, output_root = (
        path.resolve() for path in (raw_root, embeddings_root, benchmark_root, output_root)
    )
    if output_root.exists():
        raise FileExistsError(output_root)
    if any(output_root.is_relative_to(root) for root in (raw_root, embeddings_root, benchmark_root)):
        raise ValueError("The output directory must not be inside a source directory")
    files = set()
    for dataset in names:
        files.update(
            (
                raw_root / dataset / "signatures.json",
                raw_root / dataset / "papers.json",
                embeddings_root / dataset / "specter2.pkl",
            )
        )
        if dataset in BENCHMARK_DATASETS:
            files.update(
                benchmark_root / dataset / f"{dataset}_{suffix}"
                for suffix in ("signatures.json", "papers.json", "specter2.pkl")
            )
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > max_source_bytes:
        raise ValueError(f"Selected source bytes {total_bytes} exceed max_source_bytes={max_source_bytes}")
    output_root.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=min(n_jobs, len(names))) as executor:
        futures = [
            executor.submit(_prepare_dataset, dataset, raw_root, embeddings_root, benchmark_root, output_root)
            for dataset in names
        ]
        records = []
        for future in futures:
            record = future.result()
            records.append(record)
            with (output_root / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    result = {
        "kind": "s2and_linker_replay_preparation",
        "status": "complete",
        "datasets": records,
        "source_bytes": total_bytes,
        "n_jobs": n_jobs,
        "max_source_bytes": max_source_bytes,
        "raw_root": str(output_root / "raw"),
        "embeddings_root": str(output_root / "embeddings"),
    }
    _write_json(output_root / "preparation.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare replay inputs through the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root", "embeddings-root", "benchmark-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--datasets", nargs="+")
    selection.add_argument("--run-full", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-source-bytes", type=int, default=MAX_SOURCE_BYTES)
    args = parser.parse_args(argv)
    prepare_linker_replay(
        raw_root=args.raw_root,
        embeddings_root=args.embeddings_root,
        benchmark_root=args.benchmark_root,
        output_root=args.output_root,
        datasets=REPLAY_DATASETS if args.run_full else args.datasets,
        n_jobs=args.n_jobs,
        max_source_bytes=args.max_source_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

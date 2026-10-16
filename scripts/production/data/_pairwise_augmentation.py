"""Create bounded augmentation from current labels and fixed release populations."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import pickle
import pickletools
import random
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from s2and.data import _map_fixed_pair_labels  # noqa: E402
from s2and.text import normalize_text  # noqa: E402
from scripts.production.data.prepare_pairwise_inputs import (  # noqa: E402
    DATASETS,
    dataset_for_split,
    members,
    read,
    sha,
    write,
)


def selected_papers(path: Path, needed: set[str]) -> dict:
    """Read only the paper variants selected by pair candidates."""
    import ijson

    result = {}
    with path.open("rb") as stream:
        for key, value in ijson.kvitems(stream, "", use_float=True):
            if key in needed:
                result[key] = value
    if result.keys() != needed:
        raise ValueError(f"Missing papers: {path}, {sorted(needed - result.keys())[:5]}")
    return result


# The public opcode list gives the same map as the unstubbed pickletools.code2op.
_PICKLE_OPCODES = {opcode.code: opcode for opcode in pickletools.opcodes}


def source_vectors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Map current large exports and load small float32 tuple exports."""
    spans, ops = [], []
    with path.open("rb") as stream:
        while True:
            code = stream.read(1)
            if not code:
                raise ValueError("Missing pickle STOP")
            opcode = _PICKLE_OPCODES[code.decode("latin1")]
            if opcode.name in {"BINBYTES", "BINBYTES8", "BYTEARRAY8"}:
                width = 4 if opcode.name == "BINBYTES" else 8
                length = struct.unpack("<I" if width == 4 else "<Q", stream.read(width))[0]
                spans.append((stream.tell(), length))
                stream.seek(length, 1)
                argument = None
            else:
                argument = opcode.arg.reader(stream) if opcode.arg else None
            ops.append((opcode.name, argument))
            if opcode.name == "STOP":
                if stream.read(1):
                    raise ValueError("Trailing pickle bytes")
                break
    strings = [value for name, value in ops if name in {"SHORT_BINUNICODE", "BINUNICODE"}]
    if "f4" in strings:
        with path.open("rb") as stream:
            values, keys = pickle.load(stream)
        if values.shape != (len(keys), 768) or len(set(map(str, keys))) != len(keys):
            raise ValueError(f"Invalid vector tuple: {path}")
        return values, keys
    if len(spans) != 2 or "f8" not in strings:
        raise ValueError(f"Unsupported source vector format: {path}")
    key_type = next(value for value in strings if value.startswith("U") and value[1:].isdigit())
    rows = spans[1][1] // (int(key_type[1:]) * 4)
    if spans[0][1] != rows * 768 * 8:
        raise ValueError("Source vector shape mismatch")
    values = np.memmap(path, dtype="<f8", mode="r", offset=spans[0][0], shape=(rows, 768))
    keys = np.memmap(path, dtype="<" + key_type, mode="r", offset=spans[1][0], shape=(rows,))
    return values, keys


def candidates(
    name: str,
    source: Path,
    *,
    prepared_root: Path,
    unknown_filename: str,
    seed: int,
    pair_seed: int,
    train_candidates: int,
    train_pool_cap: int,
    evaluation_candidates: int,
    train_positive_cap: int,
    evaluation_positive_cap: int,
    negative_multiplier: float,
) -> tuple[dict, dict]:
    """Select bounded known pairs from the actual source populations."""
    signatures = read(source / f"{name}_signatures.json")
    if name == "medline":
        frame = _map_fixed_pair_labels(pd.read_csv(source / "train_pairs.csv"), "medline train")
        mask = np.random.RandomState(seed).rand(len(frame)) < (0.8 / 0.9)
        selections = {
            "train": list(frame[mask].itertuples(index=False, name=None)),
            "val": list(frame[~mask].itertuples(index=False, name=None)),
            "test": list(
                _map_fixed_pair_labels(pd.read_csv(source / "test_pairs.csv"), "medline test").itertuples(
                    index=False, name=None
                )
            ),
        }
    else:
        clusters = read(source / f"{name}_clusters.json")
        unknown = set(json.loads((source / unknown_filename).read_bytes()))
        populations = read(prepared_root / "populations/identities" / f"{name}.json")
        dataset = dataset_for_split(name, signatures, clusters, seed=seed)
        dataset.random_seed = pair_seed
        selections = {}
        for split, population in (("train", "train"), ("val", "validation"), ("test", "external_test")):
            known = {
                block: [sid for sid in ids if sid not in unknown] for block, ids in populations[population].items()
            }
            known = {block: ids for block, ids in known.items() if ids}
            selections[split] = dataset.pair_sampling(
                train_candidates if split == "train" else evaluation_candidates,
                [],
                known,
            )
            if split == "train":
                selections[split] = selections[split][:train_pool_cap]
    result = {}
    for split, pairs in selections.items():
        positive_cap = train_positive_cap if split == "train" else evaluation_positive_cap
        counts = Counter()
        kept = []
        seen = set()
        for left, right, label in pairs:
            left, right, label = str(left), str(right), int(label)
            pair = tuple(sorted((left, right)))
            if pair in seen:
                raise ValueError(f"Duplicate pair: {name} {split} {pair}")
            seen.add(pair)
            cap = positive_cap if label else int(positive_cap * negative_multiplier)
            if counts[label] < cap:
                kept.append((left, right, label))
                counts[label] += 1
        result[split] = kept
    return signatures, result


def make_variant(
    name: str,
    sid: str,
    original_signature: dict,
    original_paper: dict,
    *,
    old_paper: dict,
    cached_paper: dict | None,
    rng: random.Random,
    omission_rate: float,
) -> tuple[dict, dict, Counter[str]]:
    """Own one paper variant and preserve accepted raw name fields."""
    original_pid = str(original_signature["paper_id"])
    counts: Counter[str] = Counter()
    augmented_sid = f"{name}___{sid}"
    augmented_pid = f"{name}___{sid}___{original_pid}"
    signature, paper = copy.deepcopy(original_signature), copy.deepcopy(original_paper)
    signature["signature_id"], signature["paper_id"] = augmented_sid, augmented_pid
    paper["paper_id"] = augmented_pid
    drop_abstract = rng.random() < omission_rate
    drop_references = rng.random() < omission_rate
    drop_affiliations = rng.random() < omission_rate
    drop_venue = rng.random() < omission_rate
    drop_first = rng.random() < omission_rate
    drop_coauthors = rng.random() < omission_rate
    # Preserve the accepted omission sequence across cached translation decisions.
    rng.random()
    if drop_abstract:
        paper["abstract"] = ""
    if drop_references:
        paper["references"] = []
    if drop_affiliations:
        signature["author_info"]["affiliations"] = []
    if drop_venue:
        paper["venue"], paper["journal_name"] = None, None
    if drop_first and signature["author_info"]["first"]:
        signature["author_info"]["first"] = signature["author_info"]["first"][0]
    if drop_coauthors:
        paper["authors"] = [row for row in paper["authors"] if row["position"] == signature["author_info"]["position"]]
    cached = cached_paper
    old_title = normalize_text(old_paper["title"])
    source_match = old_paper["title"] == original_paper["title"]
    if cached and cached["title"] != old_title:
        if source_match:
            paper["title"] = cached["title"]
            counts["cached_translation_applied"] += 1
        else:
            counts["cached_translation_source_change"] += 1
    elif cached is None:
        counts["paper_without_translation_cache"] += 1
    for feature, chosen in (
        ("abstract", drop_abstract),
        ("references", drop_references),
        ("affiliations", drop_affiliations),
        ("venue", drop_venue),
        ("first", drop_first),
        ("coauthors", drop_coauthors),
    ):
        counts[f"{feature}_omission_selected"] += chosen
    return signature, paper, counts


def vector_action(
    paper: dict,
    original_paper: dict,
    *,
    has_source_vector: bool,
) -> Literal["absent", "reuse", "refresh"]:
    """Use exact text equality and retain absence for blank variants."""
    if not any((paper.get(field) or "").strip() for field in ("title", "abstract")):
        return "absent"
    same_text = all((paper.get(field) or "") == (original_paper.get(field) or "") for field in ("title", "abstract"))
    return "reuse" if same_text and has_source_vector else "refresh"


def build_augmentation(
    *,
    prepared_root: Path,
    output: Path,
    legacy_root: Path | None = None,
    datasets: tuple[str, ...] = DATASETS,
    unknown_filename: str = "restored_unknown_signature_ids_20261009.json",
    seed: int = 1111,
    pair_seed: int = 12455678,
    train_candidates: int = 100000,
    train_pool_cap: int = 25000,
    evaluation_candidates: int = 10000,
    train_positive_cap: int = 10000,
    evaluation_positive_cap: int = 1000,
    negative_multiplier: float = 1.5,
    omission_rate: float = 0.5,
) -> dict[str, Any]:
    """Create fresh variants from accepted names and current split labels."""
    destination = output
    destination.mkdir(parents=True, exist_ok=False)
    old_augmented = read(legacy_root / "augmented/augmented_papers.json") if legacy_root is not None else {}
    rng = random.Random(seed)
    all_signatures, all_papers, pairs, vectors, refresh, refresh_positions = (
        {},
        {},
        {key: [] for key in ("train", "val", "test")},
        [],
        [],
        {},
    )
    paper_ids = []
    report: dict[str, Any] = {"seed": seed, "pair_seed": pair_seed, "omission_rate": omission_rate, "datasets": {}}
    for name in datasets:
        source = prepared_root / "benchmark_source" / name
        signatures, selected = candidates(
            name,
            source,
            prepared_root=prepared_root,
            unknown_filename=unknown_filename,
            seed=seed,
            pair_seed=pair_seed,
            train_candidates=train_candidates,
            train_pool_cap=train_pool_cap,
            evaluation_candidates=evaluation_candidates,
            train_positive_cap=train_positive_cap,
            evaluation_positive_cap=evaluation_positive_cap,
            negative_multiplier=negative_multiplier,
        )
        wanted = sorted({sid for rows in selected.values() for left, right, _ in rows for sid in (left, right)})
        needed = {str(signatures[sid]["paper_id"]) for sid in wanted}
        papers = selected_papers(source / f"{name}_papers.json", needed)
        old_papers = (
            selected_papers(legacy_root / name / f"{name}_papers.json", needed) if legacy_root is not None else papers
        )
        matrix, keys = source_vectors(source / f"{name}_specter2.pkl")
        positions = {str(key): index for index, key in enumerate(keys)}
        if len(positions) != len(keys):
            raise ValueError(f"Duplicate source vector IDs: {name}")
        counts = Counter()
        for sid in wanted:
            original_signature = signatures[sid]
            original_pid = str(original_signature["paper_id"])
            original_paper = papers[original_pid]
            signature, paper, variant_counts = make_variant(
                name,
                sid,
                original_signature,
                original_paper,
                old_paper=old_papers[original_pid],
                cached_paper=old_augmented.get(original_pid),
                rng=rng,
                omission_rate=omission_rate,
            )
            counts.update(variant_counts)
            augmented_sid, augmented_pid = signature["signature_id"], signature["paper_id"]
            all_signatures[augmented_sid], all_papers[augmented_pid] = signature, paper
            action = vector_action(paper, original_paper, has_source_vector=original_pid in positions)
            if action == "absent":
                counts["absent_blank_vectors"] += 1
                continue
            paper_ids.append(augmented_pid)
            if action == "reuse":
                vectors.append(np.asarray(matrix[positions[original_pid]], dtype=np.float32))
                counts["fresh_vector_reuse"] += 1
            else:
                if original_pid not in positions:
                    counts["missing_source_vector_refresh"] += 1
                text_key = hashlib.sha256(
                    orjson.dumps([paper.get("title") or "", paper.get("abstract") or ""])
                ).hexdigest()
                if text_key not in refresh_positions:
                    refresh_positions[text_key] = len(refresh)
                    refresh.append(
                        {
                            "dataset": "augmented",
                            "paper_id": text_key,
                            "title": paper.get("title"),
                            "abstract": paper.get("abstract"),
                        }
                    )
                vectors.append(np.full(768, np.nan, dtype=np.float32))
                counts["vectors_to_refresh"] += 1
                signature["augmentation_vector_text_key"] = text_key
        for split, rows in selected.items():
            pairs[split].extend((f"{name}___{left}", f"{name}___{right}", label) for left, right, label in rows)
        report["datasets"][name] = {
            "signatures": len(wanted),
            "paper_variants": len(wanted),
            "corruptions": dict(counts),
            "pairs": {
                split: {
                    "count": len(rows),
                    "positive": sum(label for _, _, label in rows),
                    "negative": sum(1 - label for _, _, label in rows),
                }
                for split, rows in selected.items()
            },
        }
        print(name, json.dumps(report["datasets"][name]), flush=True)
        del signatures, papers, old_papers, matrix, keys, positions
    write(destination / "augmented_signatures.json", all_signatures)
    write(destination / "augmented_papers.json", all_papers)
    for split, rows in pairs.items():
        with (destination / f"{split if split != 'val' else 'val'}_pairs.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("signature_id_1", "signature_id_2", "label"))
            writer.writerows(rows)
    write(
        destination / "test_pairs.json",
        [{"signature_id_1": left, "signature_id_2": right, "label": label} for left, right, label in pairs["test"]],
    )
    schema = pa.schema([(name, pa.string()) for name in ("dataset", "paper_id", "title", "abstract")])
    refresh_root = destination / "embedding_refresh"
    refresh_root.mkdir()
    with (
        pa.OSFile(str(refresh_root / "changed_papers.arrow"), "wb") as stream,
        pa.ipc.new_file(stream, schema) as writer,
    ):
        writer.write_table(pa.Table.from_pylist(refresh, schema=schema), max_chunksize=4096)
    np.save(destination / "vectors_partial.npy", np.asarray(vectors, dtype=np.float32).reshape(-1, 768))
    write(destination / "vector_ids.json", paper_ids)
    report.update(
        signature_count=len(all_signatures),
        unique_embedding_texts_to_refresh=len(refresh),
        paper_variant_policy="One paper variant per signature prevents conflicting coauthor omissions.",
        translation_policy="Preserve cached translations on unchanged source titles. Use fresh omissions.",
    )
    write(destination / "augmentation_report.json", report)

    return report


def finalize_augmentation(
    *,
    root: Path,
    prepared_root: Path,
    datasets: tuple[str, ...] = DATASETS,
    unknown_filename: str = "restored_unknown_signature_ids_20261009.json",
) -> dict[str, Any]:
    """Attach exact vectors and verify source labels and split boundaries."""
    signatures = read(root / "augmented_signatures.json")
    papers = read(root / "augmented_papers.json")
    ids = read(root / "vector_ids.json")
    partial = np.load(root / "vectors_partial.npy", mmap_mode="r")
    output = np.asarray(partial).copy()
    refresh_root = root / "embedding_refresh"
    with (refresh_root / "changed_papers.arrow").open("rb") as stream:
        input_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with pa.memory_map(str(refresh_root / "changed_papers.arrow"), "r") as stream:
        text_rows = pa.ipc.open_file(stream).read_all().to_pylist()
    if text_rows:
        refresh_vectors = np.load(refresh_root / "vectors.npy", mmap_mode="r")
        progress = read(refresh_root / "progress.json")
        if progress.get("status") != "complete" or progress["completed_rows"] != progress["rows"]:
            raise ValueError("Embedding refresh is incomplete")
        if input_sha != progress["input_sha256"] or progress["rows"] != len(text_rows):
            raise ValueError("Embedding refresh input drift")
        if refresh_vectors.shape != (len(text_rows), 768):
            raise ValueError("Embedding refresh shape mismatch")
    else:
        refresh_vectors = np.empty((0, 768), dtype=np.float32)
    text_positions = {row["paper_id"]: index for index, row in enumerate(text_rows)}
    if len(text_positions) != len(text_rows):
        raise ValueError("Duplicate embedding text keys")
    positions = {pid: index for index, pid in enumerate(ids)}
    owned = {str(signature["paper_id"]) for signature in signatures.values()}
    if len(positions) != len(ids) or len(owned) != len(signatures) or owned != papers.keys():
        raise ValueError("Signature/paper/vector coverage mismatch")
    if not positions.keys() <= owned:
        raise ValueError("Unknown vector paper IDs")
    for sid, signature in signatures.items():
        if signature["signature_id"] != sid:
            raise ValueError("Signature key mismatch")
        pid = str(signature["paper_id"])
        key = signature.get("augmentation_vector_text_key")
        if pid not in positions:
            if key or any((papers[pid].get(field) or "").strip() for field in ("title", "abstract")):
                raise ValueError(f"Missing informative vector: {pid}")
            continue
        if key:
            output[positions[pid]] = refresh_vectors[text_positions[key]]
            row = text_rows[text_positions[key]]
            if any((papers[pid][field] or "") != (row[field] or "") for field in ("title", "abstract")):
                raise ValueError(f"Embedding text mismatch: {pid}")
    if output.shape != (len(ids), 768) or not np.isfinite(output).all():
        raise ValueError("Incomplete or invalid output vectors")
    if not np.all(np.linalg.norm(output, axis=1) > 0):
        raise ValueError("Zero output vectors")
    validation: dict[str, Any] = {
        "signatures": len(signatures),
        "papers": len(papers),
        "vector_rows": len(ids),
        "absent_blank_vectors": len(owned) - len(ids),
        "refreshed_texts": len(text_rows),
        "datasets": {},
        "pair_splits": {},
    }
    all_pairs: dict[str, set[tuple[str, ...]]] = {}
    frames = {}
    for split in ("train", "val", "test"):
        with (root / f"{split}_pairs.csv").open(encoding="utf-8", newline="") as stream:
            frames[split] = list(csv.DictReader(stream))
        identities = {tuple(sorted((row["signature_id_1"], row["signature_id_2"]))) for row in frames[split]}
        if len(identities) != len(frames[split]) or not identities:
            raise ValueError(f"Duplicate or empty {split} pairs")
        all_pairs[split] = identities
        for row in frames[split]:
            for field in ("signature_id_1", "signature_id_2"):
                sid = row[field]
                if sid not in signatures or sid.partition("___")[0] not in datasets:
                    raise ValueError(f"Unknown augmentation pair member: {sid}")
        validation["pair_splits"][split] = {
            "pairs": len(identities),
            "positive": sum(int(row["label"]) for row in frames[split]),
        }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if all_pairs[left] & all_pairs[right]:
            raise ValueError(f"Pair leakage: {left}, {right}")
        left_members = {sid for pair in all_pairs[left] for sid in pair}
        right_members = {sid for pair in all_pairs[right] for sid in pair}
        if left_members & right_members:
            raise ValueError(f"Signature leakage: {left}, {right}")
    for name in datasets:
        source = prepared_root / "benchmark_source" / name
        if name == "medline":
            accepted = {}
            allowed_pairs = {}
            frame = _map_fixed_pair_labels(pd.read_csv(source / "train_pairs.csv"), "medline train")
            seed = read(root / "augmentation_report.json")["seed"]
            mask = np.random.RandomState(seed).rand(len(frame)) < (0.8 / 0.9)
            for split, rows in (("train", frame[mask]), ("val", frame[~mask])):
                allowed_pairs[split] = {
                    tuple(sorted((str(left), str(right)))) for left, right, _ in rows.itertuples(index=False, name=None)
                }
            for split in ("train", "test"):
                frame = _map_fixed_pair_labels(pd.read_csv(source / f"{split}_pairs.csv"), f"medline {split}")
                for left, right, label in frame.itertuples(index=False, name=None):
                    accepted[tuple(sorted((str(left), str(right))))] = int(label)
                if split == "test":
                    allowed_pairs[split] = {
                        tuple(sorted((str(left), str(right))))
                        for left, right, _ in frame.itertuples(index=False, name=None)
                    }
            labels = None
            unknown = set()
            populations = None
        else:
            labels = members(read(source / f"{name}_clusters.json"))
            unknown = set(read(source / unknown_filename))
            populations = read(prepared_root / "populations/identities" / f"{name}.json")
        allowed_by_split = (
            {
                split: {sid for block in populations[key].values() for sid in block}
                for split, key in (("train", "train"), ("val", "validation"), ("test", "external_test"))
            }
            if populations is not None
            else {}
        )
        counts = {"pairs": 0, "unknown_pair_members": 0, "label_disagreements": 0}
        for split, rows in frames.items():
            for row in rows:
                prefix, _, left = row["signature_id_1"].partition("___")
                if prefix != name:
                    continue
                right_prefix, _, right = row["signature_id_2"].partition("___")
                if right_prefix != name:
                    raise ValueError("Cross-source augmentation pair")
                counts["pairs"] += 1
                if left in unknown or right in unknown:
                    raise ValueError(f"Unknown augmentation member: {name}")
                expected = (
                    int(labels[left] == labels[right]) if labels is not None else accepted[tuple(sorted((left, right)))]
                )
                if int(row["label"]) != expected:
                    raise ValueError(f"Stale augmentation label: {name} {left}, {right}")
                if populations is None and tuple(sorted((left, right))) not in allowed_pairs[split]:
                    raise ValueError(f"Fixed pair population leakage: {name} {split}")
                if populations is not None:
                    allowed = allowed_by_split[split]
                    if left not in allowed or right not in allowed:
                        raise ValueError(f"Population leakage: {name} {split}")
        validation["datasets"][name] = counts
    output_path = root / "augmented_specter2.pkl"
    if output_path.exists():
        raise FileExistsError(output_path)
    with output_path.open("wb") as stream:
        pickle.dump((output, np.asarray(ids)), stream, protocol=4)
    with output_path.open("rb") as stream:
        reloaded, reloaded_ids = pickle.load(stream)
    if not np.array_equal(output, reloaded) or list(reloaded_ids) != ids:
        raise ValueError("Saved embedding roundtrip drift")
    validation["outputs"] = {
        filename: {"path": str(root / filename), "sha256": sha(root / filename)}
        for filename in (
            "augmented_signatures.json",
            "augmented_papers.json",
            "augmented_specter2.pkl",
            "train_pairs.csv",
            "val_pairs.csv",
            "test_pairs.csv",
            "test_pairs.json",
        )
    }
    write(root / "validation.json", validation)
    print(json.dumps(validation), flush=True)

    return validation

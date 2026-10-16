"""Freeze and score gold identity cases through the public incremental Arrow API."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from s2and.arrow_inputs import ArrowDataset  # noqa: E402
from s2and.production_model import load_production_model  # noqa: E402
from scripts.production.model.unknown_signatures import load_unknown_signatures  # noqa: E402

Context = Literal["positive", "author_absent"]
Outcome = Literal["correct_link", "correct_abstain", "false_abstain", "false_link", "wrong_candidate"]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Require a JSON object: {path}")
    return value


def _rank(seed: int, *values: str) -> str:
    return hashlib.sha256(json.dumps([seed, *values], ensure_ascii=False).encode("utf-8")).hexdigest()


def case_seeds(block: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, str]:
    """Exclude the query paper and, for absent cases, the complete query author."""
    query_paper = block["signature_papers"][case["query_signature_id"]]
    return {
        signature_id: author_id
        for author_id, members in block["profiles"].items()
        if case["context"] != "author_absent" or author_id != case["query_author_id"]
        for signature_id in members
        if block["signature_papers"][signature_id] != query_paper
    }


def prepare_cases(
    *, snapshot: Path, eligible_mask: Path, max_blocks: int, max_cases: int, seed: int = 13
) -> dict[str, Any]:
    """Freeze one query per known author without any model or model scores."""
    if max_blocks <= 0 or max_cases <= 0:
        raise ValueError("Require positive block and case limits")
    snapshot = snapshot.resolve()
    mask = _read_json(eligible_mask)
    datasets = mask["datasets"]
    block_count = sum(len(entry["block_ids"]) for entry in datasets.values())
    if block_count > max_blocks:
        raise ValueError(f"Eligible pool exceeds block limit: {block_count} > {max_blocks}")
    blocks: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    sources: dict[str, dict[str, str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for dataset, entry in sorted(datasets.items()):
        if not dataset or Path(dataset).name != dataset or any(character in dataset for character in "\\/:"):
            raise ValueError(f"Require a dataset path component: {dataset!r}")
        selected = set(entry["block_ids"])
        if len(selected) != len(entry["block_ids"]):
            raise ValueError(f"Duplicate eligible block IDs: {dataset}")
        count: Counter[str] = Counter(eligible_blocks=len(selected))
        counts[dataset] = dict(count)
        if not selected:
            continue
        source = snapshot / "corrected_sources" / dataset
        signatures_path = source / f"{dataset}_signatures.json"
        clusters_path = source / f"{dataset}_clusters.json"
        signatures = _read_json(signatures_path)
        clusters = _read_json(clusters_path)
        unknown = load_unknown_signatures({"clusters": clusters_path})
        unknown_ids = unknown.ids | frozenset(map(str, entry["unknown_signature_ids"]))
        allowed_queries = set(map(str, entry["known_query_signature_ids"]))
        author_by_signature: dict[str, str] = {}
        for author_id, cluster in clusters.items():
            for value in cluster["signature_ids"]:
                signature_id = str(value)
                if signature_id in author_by_signature:
                    raise ValueError(f"Duplicate gold signature: {dataset}/{signature_id}")
                author_by_signature[signature_id] = author_id
        source_paths = [signatures_path, clusters_path, snapshot / "benchmarks" / dataset / "manifest.json"]
        if unknown.path is not None:
            source_paths.append(unknown.path)
        sources[dataset] = {path.relative_to(snapshot).as_posix(): sha256_file(path) for path in source_paths}
        members_by_block: dict[str, list[str]] = defaultdict(list)
        for signature_id, signature in signatures.items():
            block_id = signature["author_info"]["block"]
            if block_id in selected:
                members_by_block[block_id].append(signature_id)
        if members_by_block.keys() != selected:
            raise ValueError(f"Eligible blocks lack source signatures: {dataset}/{selected - members_by_block.keys()}")
        author_blocks: dict[str, set[str]] = defaultdict(set)
        for block_id, members in members_by_block.items():
            for signature_id in members:
                if signature_id not in unknown_ids and signature_id in author_by_signature:
                    author_blocks[author_by_signature[signature_id]].add(block_id)
        for author_id, known_blocks in author_blocks.items():
            if len(known_blocks) > 1:
                raise ValueError(
                    f"Known gold author spans selected blocks: {dataset}/{author_id}/{sorted(known_blocks)}"
                )
        count["source_signatures"] = sum(map(len, members_by_block.values()))
        for block_id in sorted(selected, key=lambda value: _rank(seed, dataset, value)):
            profiles: dict[str, list[str]] = defaultdict(list)
            papers: dict[str, str] = {}
            for signature_id in sorted(members_by_block[block_id]):
                if signature_id in unknown_ids:
                    count["excluded_unknown_signatures"] += 1
                    continue
                if signature_id not in author_by_signature:
                    raise ValueError(f"Known signature lacks gold author: {dataset}/{signature_id}")
                papers[signature_id] = str(signatures[signature_id]["paper_id"])
                profiles[author_by_signature[signature_id]].append(signature_id)
            count["known_signatures"] += len(papers)
            count["known_authors"] += len(profiles)
            block = {
                "dataset": dataset,
                "block_id": block_id,
                "profiles": dict(sorted(profiles.items())),
                "signature_papers": papers,
            }
            block_index = len(blocks)
            blocks.append(block)
            for author_id, author_members in sorted(profiles.items()):
                candidates = [signature_id for signature_id in author_members if signature_id in allowed_queries]
                if not candidates:
                    count["authors_without_eligible_query"] += 1
                    continue
                query = min(
                    candidates,
                    key=lambda signature_id: (
                        not any(papers[other] != papers[signature_id] for other in author_members),
                        _rank(seed, dataset, block_id, author_id, signature_id),
                    ),
                )
                count["selected_queries"] += 1
                for context in ("positive", "author_absent"):
                    case = {
                        "case_id": _rank(seed, dataset, block_id, author_id, query, context),
                        "block_index": block_index,
                        "query_signature_id": query,
                        "query_author_id": author_id,
                        "context": context,
                    }
                    seeds = case_seeds(block, case)
                    if context == "positive" and author_id not in seeds.values():
                        count["positive_without_distinct_paper"] += 1
                        continue
                    if not seeds:
                        count[f"empty_profiles_{context}"] += 1
                        continue
                    cases.append(case)
                    count[f"cases_{context}"] += 1
                    if len(cases) > max_cases:
                        raise ValueError(f"Prepared cases exceed case limit: {len(cases)} > {max_cases}")
        counts[dataset] = dict(count)
    return {
        "kind": "incremental_gold_identity_cases",
        "seed": seed,
        "bounds": {"max_blocks": max_blocks, "max_cases": max_cases},
        "labels": "Corrected gold identity facts; no manual safe-action judgments.",
        "source_sha256": sources,
        "counts": counts,
        "blocks": blocks,
        "cases": cases,
    }


def classify_query(
    clusters: Mapping[str, Sequence[str]], query: str, seeds: Mapping[str, str], author: str, context: Context
) -> tuple[Outcome, list[str]]:
    """Classify identity from the query cluster and its actual seed members."""
    request = {query, *seeds}
    members = [str(member) for cluster in clusters.values() for member in cluster]
    if len(members) != len(request) or set(members) != request:
        raise ValueError("Prediction must partition the exact request once")
    signature_partition = {
        str(member): partition_index for partition_index, cluster in enumerate(clusters.values()) for member in cluster
    }
    author_partitions: dict[str, set[int]] = defaultdict(set)
    partition_authors: dict[int, set[str]] = defaultdict(set)
    for signature_id, author_id in seeds.items():
        partition_index = signature_partition[signature_id]
        author_partitions[author_id].add(partition_index)
        partition_authors[partition_index].add(author_id)
    for author_id, partitions in author_partitions.items():
        if len(partitions) > 1:
            raise ValueError(f"Seed profile splits across result clusters: {author_id}")
    for partition_index, authors in partition_authors.items():
        if len(authors) > 1:
            raise ValueError(f"Seed profiles merge in result cluster {partition_index}: {sorted(authors)}")
    query_members = next(cluster for cluster in clusters.values() if query in cluster)
    selected_authors = sorted({seeds[member] for member in query_members if member in seeds})
    if context == "author_absent":
        return ("false_link" if selected_authors else "correct_abstain"), selected_authors
    if not selected_authors:
        return "false_abstain", selected_authors
    return ("correct_link" if selected_authors == [author] else "wrong_candidate"), selected_authors


def evaluate_cases(
    *, snapshot: Path, cases_path: Path, model_dir: Path, output_dir: Path, n_jobs: int, total_ram_bytes: int
) -> dict[str, Any]:
    """Score frozen cases with a complete serialized production model."""
    if n_jobs <= 0 or total_ram_bytes <= 0:
        raise ValueError("Require positive core and RAM limits")
    payload = _read_json(cases_path)
    if payload["kind"] != "incremental_gold_identity_cases":
        raise ValueError("Require frozen gold identity cases")
    for paths in payload["source_sha256"].values():
        for relative_path, expected in paths.items():
            path = (snapshot / relative_path).resolve()
            if not path.is_relative_to(snapshot.resolve()) or sha256_file(path) != expected:
                raise ValueError(f"Frozen source differs: {relative_path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    clusterer = load_production_model(model_dir, n_jobs=n_jobs)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    sizes: dict[str, list[dict[str, int]]] = defaultdict(list)
    start = time.perf_counter()
    with ExitStack() as stack, (output_dir / "rows.jsonl").open("x", encoding="utf-8") as rows:
        handles: dict[str, ArrowDataset] = {}
        for dataset in payload["source_sha256"]:
            handles[dataset] = stack.enter_context(ArrowDataset.open(snapshot / "benchmarks" / dataset))
        for case in payload["cases"]:
            block = payload["blocks"][case["block_index"]]
            dataset = block["dataset"]
            seeds = case_seeds(block, case)
            if not seeds:
                raise ValueError(f"Frozen case has empty profiles: {case['case_id']}")
            query = case["query_signature_id"]
            try:
                result = clusterer.predict_incremental_from_arrow(
                    [query, *seeds], handles[dataset], cluster_seeds_require=seeds, total_ram_bytes=total_ram_bytes
                )
                outcome, predicted_authors = classify_query(
                    result["clusters"], query, seeds, case["query_author_id"], case["context"]
                )
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                raise RuntimeError(f"Gold case fails: {dataset}/{case['case_id']}/{query}") from exc
            profile_sizes = Counter(seeds.values())
            row = {
                **case,
                "dataset": dataset,
                "block_id": block["block_id"],
                "outcome": outcome,
                "predicted_author_ids": predicted_authors,
                "profile_sizes": dict(profile_sizes),
            }
            rows.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows.flush()
            counts[dataset]["cases"] += 1
            counts[dataset][case["context"]] += 1
            counts[dataset][outcome] += 1
            sizes[dataset].append(dict(profile_sizes))
    reports: dict[str, dict[str, Any]] = {}
    for dataset, count in counts.items():
        errors = sum(count[label] for label in ("false_abstain", "false_link", "wrong_candidate"))
        profile_sizes = [size for case_sizes in sizes[dataset] for size in case_sizes.values()]
        reports[dataset] = {
            **dict(count),
            "identity_error_rate": errors / count["cases"],
            "profile_signature_count_min": min(profile_sizes),
            "profile_signature_count_max": max(profile_sizes),
            "profile_signature_count_mean": sum(profile_sizes) / len(profile_sizes),
        }
    summary = {
        "label_type": payload["labels"],
        "preparation_counts": payload["counts"],
        "datasets": reports,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze cases before any model scores")
    prepare.add_argument("--snapshot", type=Path, required=True)
    prepare.add_argument("--eligible-mask", type=Path, required=True)
    prepare.add_argument("--max-blocks", type=int, required=True)
    prepare.add_argument("--max-cases", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=13)
    prepare.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Score a complete production model")
    evaluate.add_argument("--snapshot", type=Path, required=True)
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--model-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--n-jobs", type=int, required=True)
    evaluate.add_argument("--total-ram-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_cases(
            snapshot=args.snapshot,
            eligible_mask=args.eligible_mask,
            max_blocks=args.max_blocks,
            max_cases=args.max_cases,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, ensure_ascii=False, sort_keys=True)
            output.write("\n")
        print(json.dumps({"cases": len(payload["cases"]), "counts": payload["counts"]}, sort_keys=True))
    else:
        print(
            json.dumps(
                evaluate_cases(
                    snapshot=args.snapshot,
                    cases_path=args.cases,
                    model_dir=args.model_dir,
                    output_dir=args.output_dir,
                    n_jobs=args.n_jobs,
                    total_ram_bytes=args.total_ram_bytes,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()

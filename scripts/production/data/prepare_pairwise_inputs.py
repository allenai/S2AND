"""Prepare current benchmark sources and exact release populations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import orjson

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from s2and.data import ANDData  # noqa: E402
from scripts.convert_to_arrow import join_canonical_benchmark_names  # noqa: E402

DATASETS = ("aminer", "arnetminer", "inspire", "kisti", "medline", "pubmed", "qian", "zbmath")


def read(path: Path) -> Any:
    """Read one preparation input or report."""
    return orjson.loads(path.read_bytes())


def write(path: Path, value: object) -> None:
    """Save one preparation output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(value))


def sha(path: Path) -> str:
    """Return a content identity for input checks."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def members(clusters: dict) -> dict[str, str]:
    """Reject duplicate gold memberships."""
    result = {sid: cid for cid, row in clusters.items() for sid in row["signature_ids"]}
    if len(result) != sum(len(row["signature_ids"]) for row in clusters.values()):
        raise ValueError("Duplicate cluster membership")
    return result


def dataset_for_split(
    name: str,
    signatures: dict,
    clusters: dict,
    *,
    seed: int = 1111,
    n_jobs: int = 1,
) -> ANDData:
    """Use the real splitter without text features or predictions."""
    paper_ids = {str(signature["paper_id"]) for signature in signatures.values()}
    papers = {
        pid: {
            "paper_id": pid,
            "title": "",
            "abstract": "",
            "authors": [],
            "venue": None,
            "journal_name": None,
            "year": None,
        }
        for pid in paper_ids
    }
    # Block selection reads names, block IDs, and block sizes. Paper text has no effect.
    return ANDData(
        signatures=signatures,
        papers=papers,
        clusters=clusters,
        name=name,
        mode="train",
        preprocess=False,
        random_seed=seed,
        n_jobs=n_jobs,
        name_counts_index=None,
        name_tuples=frozenset(),
    )


def subset_clusters(clusters: dict, ids: set[str]) -> dict:
    """Preserve gold row order in a selected population."""
    return {
        cid: {**row, "signature_ids": selected}
        for cid, row in clusters.items()
        if (selected := [sid for sid in row["signature_ids"] if sid in ids])
    }


def link(source: Path, target: Path) -> None:
    """Share immutable paper and vector inputs across populations."""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def prepare_benchmark(
    name: str,
    *,
    source_root: Path,
    output_root: Path,
    fixed_pair_root: Path,
    correction_receipt: Path | None = None,
    correction_root: Path | None = None,
    unknown_filename: str = "restored_unknown_signature_ids_20261009.json",
    seed: int = 1111,
    pair_seed: int = 12455678,
    evaluation_pairs: int = 10000,
    n_jobs: int = 1,
) -> dict:
    """Prepare accepted raw names, current gold, and nested release splits."""
    root = output_root / "benchmark_source"
    destination = root / name
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    source = source_root / name
    signature_path = source / f"{name}_signatures.json"
    signatures = read(signature_path)
    report: dict = {"source_signatures_sha256": sha(signature_path), "signature_count": len(signatures)}
    rows = []
    scalar_emails = 0
    for sid, signature in signatures.items():
        info = signature["author_info"]
        rows.append(
            {"signature_id": sid, "first": info.get("first"), "middle": info.get("middle"), "last": info.get("last")}
        )
        emails = info.get("emails", [])
        info["email"] = emails[0] if len(emails) == 1 else None
        scalar_emails += info["email"] is not None
    joined, names_report = join_canonical_benchmark_names(signatures, rows)
    signatures = {sid: joined[sid] for sid in signatures}
    write(destination / f"{name}_signatures.json", signatures)
    write(output_root / "canonical_names" / f"{name}.json", rows)
    report["canonical_names"] = names_report
    report["name_row_provenance"] = "Accepted upstream source fields; the runtime owns canonicalization."
    report["scalar_email_count"] = scalar_emails
    report["source_signature_order_preserved"] = True
    if name == "medline":
        for split in ("train", "test"):
            shutil.copyfile(fixed_pair_root / "medline" / f"{split}_pairs.csv", destination / f"{split}_pairs.csv")
        report["fixed_pair_label_source"] = "s2and/data-backup/medline"
    else:
        original_path = source / f"{name}_clusters.json"
        clusters = read(original_path)
        old_members = members(clusters)
        sidecar = source / unknown_filename
        unknown = set(json.loads(sidecar.read_bytes()))
        if not unknown <= signatures.keys():
            raise ValueError(f"Unknown IDs are absent from source signatures: {name}")
        report["source_clusters_sha256"] = sha(original_path)
        receipt = read(correction_receipt)["datasets"].get(name) if correction_receipt is not None else None
        if receipt is not None:
            if correction_root is None:
                raise ValueError("A correction receipt requires a correction root")
            staged_path = correction_root / name / original_path.name
            if report["source_clusters_sha256"] != receipt["before_sha256"]:
                raise ValueError(f"Correction baseline drift: {name}")
            if sha(staged_path) != receipt["staged_sha256"]:
                raise ValueError(f"Correction output drift: {name}")
            staged = read(staged_path)
            assigned = members(staged)
            missing = set(old_members) - assigned.keys()
            if missing != set(receipt["unlabeled_ids"]):
                raise ValueError(f"Unexpected correction exclusions: {name}")
            for sid in sorted(missing):
                cid = old_members[sid]
                if cid not in staged:
                    staged[cid] = {**copy.deepcopy(clusters[cid]), "signature_ids": []}
                staged[cid]["signature_ids"].append(sid)
            unknown.update(missing)
            clusters = staged
            report["review_overlay"] = {
                "rows": receipt["rows"],
                "moved": receipt["moved"],
                "new_unknown": len(missing),
                "sha256": sha(staged_path),
            }
            current = members(clusters)
            if set(current) != set(old_members):
                raise ValueError(f"Correction membership coverage drift: {name}")
            if any(current[sid] != cid for sid, cid in assigned.items()):
                raise ValueError(f"Correction membership changed: {name}")
        if set(members(clusters)) != set(signatures):
            raise ValueError(f"Signature/cluster identity mismatch: {name}")
        write(destination / original_path.name, clusters)
        write(destination / sidecar.name, sorted(unknown))
        report["unknown_count"] = len(unknown)
        dataset = dataset_for_split(name, signatures, clusters, seed=seed, n_jobs=n_jobs)
        outer_train, outer_val, heldout = dataset.split_cluster_signatures()
        development_ids = {sid for block in (*outer_train.values(), *outer_val.values()) for sid in block}
        heldout_ids = {sid for block in heldout.values() for sid in block}
        model_signatures = {sid: row for sid, row in signatures.items() if sid in development_ids}
        model_clusters = subset_clusters(clusters, development_ids)
        model_dataset = dataset_for_split(name, model_signatures, model_clusters, seed=seed, n_jobs=n_jobs)
        model_train, model_val, model_unused = model_dataset.split_cluster_signatures()
        populations = {
            "train": model_train,
            "validation": model_val,
            "unused_internal_test": model_unused,
            "external_test": heldout,
        }
        if development_ids & heldout_ids or development_ids | heldout_ids != set(signatures):
            raise ValueError(f"Population overlap or loss: {name}")
        selected_root = output_root / "populations"
        model_directory = selected_root / "model_source" / name
        eval_directory = selected_root / "evaluation_source" / name
        for directory, ids in ((model_directory, development_ids), (eval_directory, heldout_ids)):
            directory.mkdir(parents=True, exist_ok=True)
            write(directory / f"{name}_signatures.json", {sid: row for sid, row in signatures.items() if sid in ids})
            write(directory / f"{name}_clusters.json", subset_clusters(clusters, ids))
            write(directory / sidecar.name, sorted(unknown & ids))
        write(selected_root / "identities" / f"{name}.json", populations)
        write(eval_directory / "test_blocks.json", heldout)
        dataset.random_seed = pair_seed
        dataset.test_pairs_size = evaluation_pairs
        known_heldout = {block: [sid for sid in ids if sid not in unknown] for block, ids in heldout.items()}
        known_heldout = {block: ids for block, ids in known_heldout.items() if ids}
        test_pairs = dataset.pair_sampling(evaluation_pairs, [], known_heldout)
        write(
            eval_directory / "test_pairs.json",
            [
                {"signature_id_1": left, "signature_id_2": right, "label": int(label)}
                for left, right, label in test_pairs
            ],
        )
        report["populations"] = {
            split: {"blocks": len(blocks), "signatures": sum(map(len, blocks.values()))}
            for split, blocks in populations.items()
        }
        report["external_test_pairs"] = len(test_pairs)
        report["external_pair_labels"] = {str(label): sum(row[2] == label for row in test_pairs) for label in (0, 1)}
    for filename in (f"{name}_papers.json", f"{name}_specter2.pkl"):
        shutil.copyfile(source / filename, destination / filename)
    added_signatures = source / f"{name}_added_signatures.json"
    if added_signatures.is_file():
        shutil.copyfile(added_signatures, destination / added_signatures.name)
    if name != "medline":
        for mode in ("model_source", "evaluation_source"):
            for filename in (f"{name}_papers.json", f"{name}_specter2.pkl"):
                link(destination / filename, output_root / "populations" / mode / name / filename)
    write(destination / "preparation_report.json", report)
    print(name, json.dumps(report.get("populations", {})), flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded pairwise data preparation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    benchmark = stages.add_parser("benchmarks", help="Prepare raw names, corrected gold, and release splits")
    for option in ("source-root", "fixed-pair-root", "output"):
        benchmark.add_argument(f"--{option}", type=Path, required=True)
    benchmark.add_argument("--correction-receipt", type=Path)
    benchmark.add_argument("--correction-root", type=Path)
    benchmark.add_argument("--evaluation-pairs", type=int, default=10000)
    augmentation = stages.add_parser("augmentation", help="Create fresh omissions and vector refresh inputs")
    for option in ("prepared-root", "output"):
        augmentation.add_argument(f"--{option}", type=Path, required=True)
    augmentation.add_argument("--legacy-root", type=Path, help="Reuse explicitly supplied cached translations")
    for option, default in (
        ("train-candidates", 100000),
        ("train-pool-cap", 25000),
        ("evaluation-candidates", 10000),
        ("train-positive-cap", 10000),
        ("evaluation-positive-cap", 1000),
    ):
        augmentation.add_argument(f"--{option}", type=int, default=default)
    augmentation.add_argument("--negative-multiplier", type=float, default=1.5)
    augmentation.add_argument("--omission-rate", type=float, default=0.5)
    finalize = stages.add_parser("finalize", help="Attach vectors and verify labels and populations")
    finalize.add_argument("--root", type=Path, required=True)
    finalize.add_argument("--prepared-root", type=Path, required=True)
    finalize.add_argument("--encode", action="store_true", help="Use a local SPECTER2 model for changed text")
    finalize.add_argument("--models-root", type=Path)
    finalize.add_argument("--batch-size", type=int, default=16)
    finalize.add_argument("--device", default="cuda")
    for stage in (benchmark, augmentation, finalize):
        stage.add_argument("--datasets", nargs="+", default=DATASETS)
        stage.add_argument("--unknown-filename", default="restored_unknown_signature_ids_20261009.json")
        stage.add_argument("--n-jobs", type=int, default=1)
    for stage in (benchmark, augmentation):
        stage.add_argument("--seed", type=int, default=1111)
        stage.add_argument("--pair-seed", type=int, default=12455678)
    args = parser.parse_args(argv)
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
    import pyarrow as pa
    from threadpoolctl import threadpool_limits

    from scripts.production.data._pairwise_augmentation import build_augmentation, finalize_augmentation

    pa.set_cpu_count(args.n_jobs)
    pa.set_io_thread_count(args.n_jobs)
    options = vars(args).copy()
    options.pop("stage")
    options["datasets"] = tuple(options["datasets"])
    with threadpool_limits(limits=args.n_jobs):
        if args.stage == "benchmarks":
            options["output_root"] = options.pop("output")
            datasets = options.pop("datasets")
            report = {"seed": args.seed, "pair_seed": args.pair_seed, "datasets": {}}
            for name in datasets:
                report["datasets"][name] = prepare_benchmark(name, **options)
                write(args.output / "population_summary.json", report)
        elif args.stage == "augmentation":
            options.pop("n_jobs")
            if not 0 <= args.omission_rate <= 1 or args.negative_multiplier <= 0:
                parser.error("Require 0 <= --omission-rate <= 1 and positive --negative-multiplier")
            for option in (
                "train_candidates",
                "train_pool_cap",
                "evaluation_candidates",
                "train_positive_cap",
                "evaluation_positive_cap",
            ):
                if options[option] < 1:
                    parser.error(f"--{option.replace('_', '-')} must be positive")
            report = build_augmentation(**options)
        else:
            encode = options.pop("encode")
            models_root = options.pop("models_root")
            batch_size = options.pop("batch_size")
            device = options.pop("device")
            options.pop("n_jobs")
            if encode:
                if models_root is None:
                    parser.error("--encode requires --models-root")
                from scripts.production.data._pairwise_embeddings import encode_refresh

                encode_refresh(
                    root=args.root / "embedding_refresh",
                    models_root=models_root,
                    batch_size=batch_size,
                    device=device,
                    n_jobs=args.n_jobs,
                )
            report = finalize_augmentation(**options)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare fresh linker support from reviewed benchmark and ORCID labels."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from s2and._sha256 import sha256_file  # noqa: E402
from s2and.incremental_linking_training.classic import (  # noqa: E402
    _apply_classic_train_holdout_filter,
    _load_classic_stratified_eval_rows,
    _validate_stratified_split_assignments,
    load_bundle,
    read_classic_holdout_identities,
)
from s2and.incremental_linking_training.source_bundle_preflight import (  # noqa: E402
    validate_source_bundle_support_files,
)

BENCHMARK_DATASETS = ("arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath")
Relation = tuple[str, str, str]
Query = tuple[str, str]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _child(root: Path, relative: str) -> Path:
    path = Path(relative.replace("\\", "/"))
    target = (root / path).resolve()
    if path.is_absolute() or PureWindowsPath(relative).drive or not target.is_relative_to(root.resolve()):
        raise ValueError(f"Support path must remain inside its root: {relative}")
    return target


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _review(handoff: Path, units_path: Path) -> tuple[set[Query], set[Relation], list[Path]]:
    inputs = [handoff / "SHA256SUMS", units_path]
    verified: set[str] = set()
    for line in (handoff / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = _child(handoff, relative.lstrip("*"))
        if sha256_file(path) != expected:
            raise ValueError(f"Review digest mismatch: {path}")
        inputs.append(path)
        verified.add(path.name)
    required = {"final_linker_review.jsonl", "orcid_error_query_signatures.json"}
    if not required <= verified:
        raise ValueError(f"The review checksum file lacks required decisions: {sorted(required - verified)}")
    errors: set[Query] = set()
    for pair in _read_json(handoff / "orcid_error_query_signatures.json"):
        if len(pair) != 2:
            raise ValueError("ORCID exclusions must contain dataset and signature ID")
        errors.add((str(pair[0]), str(pair[1])))
    restore = {
        (str(unit["dataset"]), str(query), str(component))
        for unit in _read_json(units_path)
        if unit["kind"] == "big_block_orcid_group"
        for query, component in unit["pairs"]
        if (str(unit["dataset"]), str(query)) not in errors
    }
    return errors, restore, inputs


def _gold(
    root: Path, datasets: Sequence[str], unknown_filename: str,
) -> tuple[dict[str, dict[str, str]], list[Path]]:
    result = {}
    inputs = []
    for dataset in datasets:
        clusters_path = _child(root, f"{dataset}/{dataset}_clusters.json")
        unknown_path = _child(root, f"{dataset}/{unknown_filename}")
        unknown = set(map(str, _read_json(unknown_path)))
        membership: dict[str, str] = {}
        for cluster, row in _read_json(clusters_path).items():
            for signature in row["signature_ids"]:
                signature = str(signature)
                if signature in membership:
                    raise ValueError(f"Duplicate gold signature: {dataset}/{signature}")
                membership[signature] = str(cluster)
        if not unknown <= membership.keys():
            raise ValueError(f"Unknown signatures lack fallback clusters: {dataset}")
        result[dataset] = {signature: cluster for signature, cluster in membership.items() if signature not in unknown}
        inputs.extend((clusters_path, unknown_path))
    return result, inputs


def _assignment_signatures(assignments: pd.DataFrame) -> pd.Series:
    signatures = []
    for dataset, group in assignments[["dataset", "query_group_id"]].itertuples(index=False, name=None):
        parts = str(group).split(":")
        if len(parts) != 3 or parts[0] != str(dataset) or not parts[1] or parts[2] not in {"full", "initial_only"}:
            raise ValueError(f"Expected dataset:signature:view query identity: {group}")
        signatures.append(f"{dataset}:{parts[1]}")
    return pd.Series(signatures, index=assignments.index)


def _write_rows(path: Path, rows: pd.DataFrame, schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(rows, schema=schema, preserve_index=False), path)


def _read_rows(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def _counts(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(rows), "query_views": int(rows["query_group_id"].nunique()),
        "positive_rows": int(rows["label"].sum()),
        "dataset_rows": {str(key): int(value) for key, value in rows["dataset"].value_counts().items()},
    }


def prepare_linker_support(
    *, source_root: Path, benchmark_labels_root: Path, review_handoff: Path, review_units: Path,
    unknown_filename: str, output_root: Path, report_path: Path,
    benchmark_datasets: Sequence[str] = BENCHMARK_DATASETS,
) -> dict[str, Any]:
    """Write corrected labels and active splits without new identity judgments.

    Keep original component and split files. Exclude complete query views when
    their targets lack known evidence. Preserve each surviving split allocation.
    """
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if output_root.exists() or report_path.exists():
        raise FileExistsError("Support output and report paths must be fresh")
    if report_path.resolve().is_relative_to(output_root):
        raise ValueError("Support report must remain outside the support output")
    for root in (source_root, benchmark_labels_root.resolve(), review_handoff.resolve()):
        if output_root.is_relative_to(root) or root.is_relative_to(output_root):
            raise ValueError("Support output must not overlap its inputs")
        if report_path.resolve().is_relative_to(root):
            raise ValueError("Support report must not overwrite its inputs")
    bundle = _read_json(source_root / "bundle.json")
    for asset in ("featureless_rows", "candidate_members"):
        mapping = bundle["assets"][asset]["files" if asset == "featureless_rows" else "datasets"]
        for key, relative in mapping.items():
            mapping[key] = str(relative).replace("\\", "/")
    internal_eval = bundle["models"]["classic"]["classic_gate_internal_eval_base_groups_path"]
    bundle["models"]["classic"]["classic_gate_internal_eval_base_groups_path"] = internal_eval.replace("\\", "/")
    errors, restore, inputs = _review(review_handoff, review_units)
    gold, gold_inputs = _gold(benchmark_labels_root, benchmark_datasets, unknown_filename)
    inputs.extend([source_root / "bundle.json", *gold_inputs])
    files = bundle["assets"]["featureless_rows"]["files"]
    assignments_path = _child(source_root, bundle["assets"]["splits"]["assignments_path"])
    summary_path = _child(source_root, bundle["assets"]["splits"]["summary_path"])
    assignments = pd.read_csv(assignments_path)
    _validate_stratified_split_assignments(assignments)
    identities = _assignment_signatures(assignments)
    split_counts = assignments.assign(signature_identity=identities).groupby("signature_identity")["split"].nunique()
    conflicting = set(split_counts.loc[split_counts.gt(1)].index.astype(str))
    members: dict[str, dict[str, set[str]]] = {}
    for dataset in benchmark_datasets:
        path = _child(source_root, bundle["assets"]["candidate_members"]["datasets"][dataset])
        rows = _read_rows(path, columns=["candidate_component_key", "signature_id"])
        grouped: dict[str, set[str]] = defaultdict(set)
        for component, signature in rows.itertuples(index=False, name=None):
            grouped[str(component)].add(str(signature))
        members[dataset] = dict(grouped)

    decisions: dict[Relation, tuple[int, str | None]] = {}
    excluded: dict[Query, set[str]] = defaultdict(set)
    observed_errors: set[Query] = set()
    for relative in files.values():
        path = _child(source_root, relative)
        inputs.append(path)
        rows = _read_rows(path, columns=[
            "dataset", "query_signature_id", "query_group_id", "candidate_component_key", "supervision_type",
        ])
        for dataset, query, group, component, supervision in rows.itertuples(index=False, name=None):
            relation = (str(dataset), str(query), str(component))
            group_key = (str(dataset), str(group))
            if (str(dataset), str(query)) in errors and "orcid" in str(supervision):
                excluded[group_key].add("accepted_orcid_error")
                observed_errors.add((str(dataset), str(query)))
            if supervision != "labeled" or dataset not in gold:
                continue
            if relation not in decisions:
                known = gold[str(dataset)]
                author = known.get(str(query))
                candidate = members[str(dataset)][str(component)] - {str(query)}
                positive = author is not None and any(known.get(signature) == author for signature in candidate)
                reason = (
                    "unknown_query" if author is None else
                    "unknown_negative_candidate"
                    if not positive and any(signature not in known for signature in candidate)
                    else None
                )
                decisions[relation] = (int(positive), reason)
            if (reason := decisions[relation][1]) is not None:
                excluded[group_key].add(reason)
    if observed_errors != errors:
        raise ValueError(f"ORCID exclusions lack source coverage: {sorted(errors - observed_errors)}")

    output_root.mkdir(parents=True)
    for directory in ("components", "splits"):
        shutil.copytree(source_root / directory, output_root / directory)
        inputs.extend(path for path in sorted((source_root / directory).rglob("*")) if path.is_file())
    assignment_base = {
        (str(dataset), str(group)): str(base)
        for dataset, group, base in assignments[["dataset", "query_group_id", "base_group_id"]]
        .itertuples(index=False, name=None)
    }
    canonical_base: dict[Query, str] = {}
    for relative in files.values():
        rows = _read_rows(
            _child(source_root, relative), columns=["dataset", "query_signature_id", "query_group_id"],
        )
        for dataset, query, group in rows.itertuples(index=False, name=None):
            if (str(dataset), str(group)) in excluded or f"{dataset}:{query}" in conflicting:
                continue
            if (base := assignment_base.get((str(dataset), str(group)))) is not None:
                canonical_base.setdefault((str(dataset), str(query)), base)
    before_counts = {}
    for key, relative in files.items():
        table = pq.read_table(_child(source_root, relative), use_threads=False)
        rows = table.to_pandas(use_threads=False)
        before_counts[key] = _counts(rows)
        keep, labels, bases = [], [], []
        for dataset, query, group, component, supervision, label, base in rows[[
            "dataset", "query_signature_id", "query_group_id", "candidate_component_key",
            "supervision_type", "label", "base_group_id",
        ]].itertuples(index=False, name=None):
            relation = (str(dataset), str(query), str(component))
            group_key = (str(dataset), str(group))
            keep.append(group_key not in excluded and f"{dataset}:{query}" not in conflicting)
            target = decisions[relation][0] if relation in decisions and supervision == "labeled" else int(label)
            labels.append(1 if relation in restore and "orcid" in str(supervision) else target)
            bases.append(assignment_base.get(group_key, canonical_base.get((str(dataset), str(query)), str(base))))
        rows["label"] = pd.Series(labels, index=rows.index, dtype=rows["label"].dtype)
        rows["base_group_id"] = bases
        retained = rows.loc[keep]
        if retained.empty:
            raise ValueError(f"Support table has no usable query views: {key}")
        _write_rows(_child(output_root, relative), retained, table.schema)
    available: set[Query] = set()
    for key, relative in files.items():
        if key != "train_path":
            rows = _read_rows(_child(output_root, relative), columns=["dataset", "query_group_id"])
            available.update((str(dataset), str(group)) for dataset, group in rows.itertuples(index=False, name=None))
    keep_assignments = [
        (str(dataset), str(group)) in available
        for dataset, group in assignments[["dataset", "query_group_id"]].itertuples(index=False, name=None)
    ]
    active = assignments.loc[keep_assignments].copy()
    active_relative = f"splits/active_{assignments_path.name}"
    active_summary_relative = f"splits/active_{summary_path.name}"
    active.to_csv(output_root / active_relative, index=False)
    summary = _read_json(summary_path)
    summary["assignment_rows"] = len(active)
    summary["split_counts"] = {str(key): int(value) for key, value in active["split"].value_counts().items()}
    summary["removed_release_support_assignments"] = {"removed_assignment_rows": len(assignments) - len(active)}
    (output_root / active_summary_relative).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    bundle["assets"]["splits"].update(assignments_path=active_relative, summary_path=active_summary_relative)
    classic = bundle["models"]["classic"]
    classic["extra_eval_paths"] = {}
    for key, relative in files.items():
        if key.startswith("extra_eval_paths."):
            classic["extra_eval_paths"][key.split(".", 1)[1]] = relative
        else:
            classic[key] = relative
    classic["stratified_eval_test_split"].update(assignments_path=active_relative, summary_path=active_summary_relative)
    bundle["notes"] += (
        " Derived support uses accepted gold and ORCID decisions."
        " Active assignments preserve each surviving split."
    )
    (output_root / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(
        review_handoff / "orcid_error_query_signatures.json", output_root / "orcid_error_query_signatures.json",
    )
    result = load_bundle(output_root)
    validate_source_bundle_support_files(result)
    holdout = read_classic_holdout_identities(result)
    train_path = _child(output_root, files["train_path"])
    train_table = pq.read_table(train_path, use_threads=False)
    train, train_filter = _apply_classic_train_holdout_filter(
        train_table.to_pandas(use_threads=False), holdout_query_group_ids=set(holdout.query_group_ids),
        holdout_base_group_ids=set(holdout.base_group_ids), holdout_sources=list(holdout.sources),
    )
    if train.empty:
        raise ValueError("Support train table has no rows after the holdout filter")
    held_signatures: set[Query] = set()
    for key, relative in files.items():
        if key != "train_path":
            rows = _read_rows(_child(output_root, relative), columns=["dataset", "query_signature_id"])
            held_signatures.update(
                (str(dataset), str(query)) for dataset, query in rows.itertuples(index=False, name=None)
            )
    if any((str(dataset), str(query)) in held_signatures for dataset, query
           in train[["dataset", "query_signature_id"]].itertuples(index=False, name=None)):
        raise ValueError("The production train filter retains a held signature identity")
    _write_rows(train_path, train, train_table.schema)
    bound = {}
    for split in ("calibration_fit", "calibration_check", "test"):
        rows, selected = _load_classic_stratified_eval_rows(
            result, classic, classic["stratified_eval_test_split"], splits=[split], include_calibration_source=True,
        )
        bound[split] = {"rows": len(rows), "query_views": len(selected), "positive_rows": int(rows["label"].sum())}
    active_identities = _assignment_signatures(active)
    if active.assign(signature_identity=active_identities).groupby("signature_identity")["split"].nunique().gt(1).any():
        raise ValueError("Active lifecycle splits share a signature identity")
    after = {}
    for key, relative in files.items():
        path = _child(output_root, relative)
        rows = _read_rows(path)
        after[key] = {"before": before_counts[key], "after": _counts(rows), "output": _identity(path)}
    for path in inputs:
        if path.is_relative_to(source_root) and path.relative_to(source_root).parts[0] in {"components", "splits"}:
            if sha256_file(path) != sha256_file(output_root / path.relative_to(source_root)):
                raise ValueError(f"Changed frozen support: {path}")
    report = {
        "output_root": str(output_root), "tables": after, "original_assignment_rows": len(assignments),
        "active_assignment_rows": len(active), "removed_assignment_rows": len(assignments) - len(active),
        "orcid_excluded_query_signatures": len(errors), "legacy_cross_split_signature_identities": sorted(conflicting),
        "excluded_query_views_by_reason": dict(Counter(reason for reasons in excluded.values() for reason in reasons)),
        "train_holdout_filter": train_filter, "active_split_loader_counts": bound,
        "active_assignments": _identity(output_root / active_relative),
        "inputs": [_identity(path) for path in dict.fromkeys(inputs)],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "source-root", "benchmark-labels-root", "review-handoff", "review-units", "output-root", "report-path",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--unknown-filename", required=True)
    parser.add_argument("--benchmark-datasets", nargs="+", default=BENCHMARK_DATASETS)
    print(json.dumps(prepare_linker_support(**vars(parser.parse_args(argv))), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

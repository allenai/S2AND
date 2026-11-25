"""Build the 1.4 training inventory from private, corpus-keyed ORCID labels."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking_training.source_bundle import load_bundle
from scripts.common.arrow_roots import generic_dataset_manifests, read_json_object
from scripts.common.block_roles import normalize_block_key
from scripts.common.initial_query_sources import load_initial_query_sources
from scripts.intake.check_v14_training_overlap import canonical_id, checked_orcid, paper_position
from scripts.intake.prepare_identity_labels import canonical_gold_block_filter
from scripts.linker.train_linker_and_finalize import _release_table_plan

V14_LINKER_GIANTS = ("a khan", "h wang", "j smith", "s feldman", "s gupta", "s lee", "s park")
_SCHEMA = pa.schema(
    [("signature_id", pa.string()), ("orcid", pa.string()), ("paper_id", pa.string()), ("position", pa.int64())]
)


def _source_batches(path: Path, *, gold_giants: bool) -> Iterator[pa.RecordBatch]:
    source = ds.dataset(path, format="parquet")
    names = source.schema.names
    paper_column = "corpus_paper_id" if "corpus_paper_id" in names else "paper_id" if "paper_id" in names else None
    position_column = "position" if "position" in names else "author_position" if "author_position" in names else None
    signature_column = None if gold_giants or "signature_id" not in names else "signature_id"
    orcid_column = "orcid" if "orcid" in names else "author_orcid"
    if orcid_column not in names or (paper_column is None) != (position_column is None):
        raise ValueError(f"Expected ORCID evidence and complete corpus paper positions in {path}")
    if signature_column is None and paper_column is None:
        raise ValueError(f"Expected signature_id or corpus paper positions in {path}")
    predicate = canonical_gold_block_filter("cluster_block_key", V14_LINKER_GIANTS) if gold_giants else None
    columns = [field for field in (signature_column, orcid_column, paper_column, position_column) if field is not None]
    if gold_giants:
        columns.append("cluster_block_key")
    for batch in source.scanner(columns=columns, filter=predicate).to_batches():
        rows = []
        for row in batch.to_pylist():
            if gold_giants and normalize_block_key(row["cluster_block_key"]) not in V14_LINKER_GIANTS:
                continue
            orcid = checked_orcid(row[orcid_column], path)
            if gold_giants and orcid is None:
                continue
            pid, pos = (None, None)
            if paper_column is not None and position_column is not None:
                pid, pos = paper_position(row[paper_column], row[position_column], path)
            sid = canonical_id(row[signature_column], "signature_id", path) if signature_column else f"{pid}-{pos}"
            rows.append({"signature_id": sid, "orcid": orcid, "paper_id": pid, "position": pos})
        if rows:
            yield pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)


def _training_linker_mentions(root: Path) -> Iterator[tuple[pa.RecordBatch, int, int]]:
    """Resolve training and gate calibration queries and full candidate components to corpus positions."""
    bundle = read_json_object(root / "bundle.json")
    assets = bundle["assets"]
    tables = assets["featureless_rows"]["files"]
    training_keys = ["train_path"]
    if "classic_gate_source_path" in tables:
        training_keys.append("classic_gate_source_path")
    query_ids: dict[str, set[str]] = {}
    if "stratified_eval_test_split" in bundle.get("models", {}).get("classic", {}):
        planned_keys, query_ids, _ = _release_table_plan(load_bundle(root))
        training_keys = list(planned_keys)
    member_paths = assets["candidate_members"]["datasets"]
    datasets = generic_dataset_manifests(root)
    corpus_datasets = {name for name in datasets if normalize_block_key(name) in V14_LINKER_GIANTS}
    origins = load_initial_query_sources(root, dataset_names=sorted(corpus_datasets)).origins
    queries: dict[str, set[str]] = defaultdict(set)
    components: dict[str, set[str]] = defaultdict(set)
    fields = ["dataset", "query_signature_id", "candidate_component_key"]
    for key in training_keys:
        source = ds.dataset(root / tables[key], format="parquet")
        predicate = ds.field("dataset").isin(sorted(corpus_datasets))
        if key in query_ids:
            predicate &= ds.field("query_group_id").isin(sorted(query_ids[key]))
        for batch in source.to_batches(columns=fields, filter=predicate):
            for row in batch.to_pylist():
                name = row["dataset"]
                sid = row["query_signature_id"]
                queries[name].add(origins.get(name, {}).get(sid, sid))
                components[name].add(row["candidate_component_key"])
    for name in sorted(queries):
        directory = datasets[name].parent
        members_path = root / member_paths[name]
        memberships = ds.dataset(members_path, format="parquet")
        member_groups: dict[str, set[str]] = defaultdict(set)
        for batch in memberships.to_batches(
            columns=["candidate_component_key", "signature_id"],
            filter=ds.field("candidate_component_key").isin(sorted(components[name])),
        ):
            for row in batch.to_pylist():
                member_groups[row["candidate_component_key"]].add(row["signature_id"])
        if missing := components[name] - member_groups.keys():
            raise ValueError(f"Missing trained component memberships in {name!r}: {sorted(missing)[:5]}")
        raw_members = set().union(*member_groups.values())
        requested = queries[name] | raw_members
        evidence = {}
        signature_blocks = {}
        with (
            ArrowDataset.open(directory) as arrow,
            arrow.use() as lease,
            lease.open_file("signatures") as stream,
            pa.PythonFile(stream, mode="r") as handle,
        ):
            reader = pa.ipc.open_file(handle)
            columns = ["signature_id", "paper_id", "author_position", "author_block"]
            if "author_orcid" in reader.schema.names:
                columns.append("author_orcid")
            for index in range(reader.num_record_batches):
                for row in reader.get_batch(index).select(columns).to_pylist():
                    sid = row["signature_id"]
                    if sid not in requested:
                        continue
                    paper, position = paper_position(row["paper_id"], row["author_position"], directory)
                    if not paper.isdecimal() or sid in evidence:
                        raise ValueError(f"Expected one corpus position for trained signature {name}/{sid}")
                    signature_blocks[sid] = row["author_block"]
                    evidence[sid] = {
                        "signature_id": canonical_id(sid, "signature_id", directory),
                        "orcid": checked_orcid(row.get("author_orcid"), directory),
                        "paper_id": paper,
                        "position": position,
                    }
        if missing := requested - evidence.keys():
            raise ValueError(
                f"Source Arrow signatures do not cover trained mentions in {name!r}: {sorted(missing)[:5]}"
            )
        members = set()
        for component, signature_ids in member_groups.items():
            block, separator, _ = component.partition("::")
            local = {sid for sid in signature_ids if not separator or signature_blocks[sid] == block}
            if not local:
                raise ValueError(f"No block-local members for trained component {name}/{component}")
            members.update(local)
        rows = [evidence[sid] for sid in sorted(queries[name] | members)]
        yield pa.RecordBatch.from_pylist(rows, schema=_SCHEMA), len(queries[name]), len(members)


def build_training_inventory(
    gold: Path, private_labels: Iterable[Path], output: Path, *, linker_sources: Sequence[Path] = ()
) -> dict[str, int]:
    """Write giant gold, actual linker training mentions, and private sources."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        with pq.ParquetWriter(staging, _SCHEMA) as writer:
            for source in linker_sources:
                total = queries = members = 0
                for batch, query_count, member_count in _training_linker_mentions(source):
                    writer.write_batch(batch)
                    total += batch.num_rows
                    queries += query_count
                    members += member_count
                if not total:
                    raise ValueError(f"No corpus-keyed linker training mentions in {source}")
                counts[str(source)] = total
                counts[f"{source} queries"] = queries
                counts[f"{source} candidate members"] = members
            for path, gold_giants in [(gold, True), *((path, False) for path in private_labels)]:
                count = 0
                for batch in _source_batches(path, gold_giants=gold_giants):
                    writer.write_batch(batch)
                    count += batch.num_rows
                if count == 0:
                    raise ValueError(f"No training rows in {path}")
                counts[str(path)] = count
        staging.rename(output)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """Build a Parquet inventory using the known 1.4 linker training giants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-parquet", type=Path, required=True)
    parser.add_argument("--linker-source-root", type=Path, action="append", required=True)
    parser.add_argument("--private-labels", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    counts = build_training_inventory(
        args.gold_parquet, args.private_labels, args.output, linker_sources=args.linker_source_root
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

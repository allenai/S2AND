"""Build the 1.4 training inventory from private, corpus-keyed ORCID labels."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from scripts.intake.check_v14_training_overlap import canonical_id, checked_orcid, paper_position

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
    predicate = ds.field(orcid_column).is_valid()
    if gold_giants:
        predicate = predicate & ds.field("cluster_block_key").isin(
            [variant for block in V14_LINKER_GIANTS for variant in (block, block.replace(" ", "_"))]
        )
    columns = [field for field in (signature_column, orcid_column, paper_column, position_column) if field is not None]
    for batch in source.scanner(columns=columns, filter=predicate).to_batches():
        rows = []
        for row in batch.to_pylist():
            orcid = checked_orcid(row[orcid_column], path)
            pid, pos = (None, None)
            if paper_column is not None and position_column is not None:
                pid, pos = paper_position(row[paper_column], row[position_column], path)
            sid = canonical_id(row[signature_column], "signature_id", path) if signature_column else f"{pid}-{pos}"
            rows.append({"signature_id": sid, "orcid": orcid, "paper_id": pid, "position": pos})
        if rows:
            yield pa.RecordBatch.from_pylist(rows, schema=_SCHEMA)


def build_training_inventory(gold: Path, private_labels: Iterable[Path], output: Path) -> dict[str, int]:
    """Write the giant labels and any explicitly identified private training labels."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        with pq.ParquetWriter(staging, _SCHEMA) as writer:
            for path, gold_giants in [(gold, True), *((path, False) for path in private_labels)]:
                count = 0
                for batch in _source_batches(path, gold_giants=gold_giants):
                    writer.write_batch(batch)
                    count += batch.num_rows
                if count == 0:
                    raise ValueError(f"No ORCID-bearing training rows in {path}")
                counts[str(path)] = count
        staging.rename(output)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """Build a Parquet inventory using the known 1.4 linker training giants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-parquet", type=Path, required=True)
    parser.add_argument("--private-labels", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_training_inventory(args.gold_parquet, args.private_labels, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

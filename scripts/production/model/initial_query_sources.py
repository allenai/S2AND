"""Verify derived query origins against the source signature tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from s2and._sha256 import is_lowercase_sha256
from s2and.arrow_inputs import ArrowDataset

MAP_RELATIVE_PATH = Path("incremental_patent_tasks/initial_query_source_map.jsonl")
MAP_SHA256_FIELD = "initial_query_sources_sha256"


def derived_query_id(prefix: str, signature_id: str, paper_id: str) -> str:
    """Bind a derived record ID to its source signature and paper."""
    payload = json.dumps(["initial-source-v1", signature_id, paper_id], ensure_ascii=False, separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InitialQuerySource:
    """One declared original signature and optional paper identities."""

    signature_id: str
    source_paper_id: str | None
    initial_paper_id: str | None


@dataclass(frozen=True)
class InitialQuerySources:
    """Parsed origin records and the digest of their exact source bytes."""

    datasets: Mapping[str, Mapping[str, InitialQuerySource]]
    sha256: str | None

    @property
    def origins(self) -> dict[str, dict[str, str]]:
        """Return original signature IDs keyed by dataset and copy ID."""
        return {name: {clone: row.signature_id for clone, row in rows.items()} for name, rows in self.datasets.items()}

    def _validate_reader(self, dataset: str, reader: pa.ipc.RecordBatchFileReader) -> None:
        entries = self.datasets.get(dataset, {})
        needed = set(entries) | {row.signature_id for row in entries.values()}
        papers: dict[str, str] = {}
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index).select(["signature_id", "paper_id"])
            for signature, paper in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
                if isinstance(signature, str) and signature.startswith("visibleq:") and signature not in entries:
                    raise ValueError(f"Derived query has no source-map origin: {dataset}/{signature}")
                if signature not in needed:
                    continue
                if signature in papers or not isinstance(paper, str) or not paper:
                    raise ValueError(f"Source-map signature requires one nonempty paper ID: {dataset}/{signature}")
                papers[signature] = paper
        if missing := needed - papers.keys():
            raise ValueError(f"Source-map records are absent from dataset {dataset!r}: {sorted(missing)[:5]}")
        for clone, row in entries.items():
            source_paper, clone_paper = papers[row.signature_id], papers[clone]
            if row.source_paper_id is not None and row.source_paper_id != source_paper:
                raise ValueError(f"Source-map source paper differs from the signature: {dataset}/{clone}")
            if row.initial_paper_id is not None and row.initial_paper_id != clone_paper:
                raise ValueError(f"Source-map initial paper differs from the signature: {dataset}/{clone}")
            if (
                derived_query_id("visibleq:", row.signature_id, source_paper) != clone
                or derived_query_id("visiblep:", row.signature_id, source_paper) != clone_paper
            ):
                raise ValueError(f"Derived query IDs differ from the source-map origin: {dataset}/{clone}")

    def validate_arrow_dataset(self, dataset: str, arrow_dataset: ArrowDataset) -> None:
        """Require complete, consistent origins before a dataset supplies features."""
        with arrow_dataset.use() as lease, lease.open_file("signatures") as stream, pa.PythonFile(stream) as source:
            self._validate_reader(dataset, pa.ipc.open_file(source))

    def validate_snapshot(self, root: Path, dataset_names: Sequence[str] | None = None) -> None:
        """Check source tables without a model load or a feature allocation."""
        datasets_root = root / "datasets"
        names = (
            set(dataset_names)
            if dataset_names is not None
            else set(self.datasets) | {path.parent.name for path in datasets_root.glob("*/manifest.json")}
        )
        for name in sorted(names):
            dataset_dir = datasets_root / name
            manifest_path = dataset_dir / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"Source-map dataset has no signature manifest: {name}")
            manifest = json.loads(manifest_path.read_bytes())
            signatures = (dataset_dir / manifest["paths"]["signatures"]).resolve()
            with pa.memory_map(str(signatures), "r") as source:
                self._validate_reader(name, pa.ipc.open_file(source))


def load_initial_query_sources(
    root: Path,
    *,
    arrow_datasets: Mapping[str, ArrowDataset] | None = None,
    dataset_names: Sequence[str] | None = None,
    expected_sha256: str | None = None,
) -> InitialQuerySources:
    """Load and validate origin records and any declared digest.

    Args:
        root: Source snapshot root.
        arrow_datasets: Open source datasets, when the caller already has them.
        dataset_names: Snapshot datasets to check when open handles are absent.
        expected_sha256: Optional digest from the caller's frozen run inputs.

    Returns:
        Validated origin records and their byte digest.

    Raises:
        ValueError: An origin, derived record, or expected digest is invalid.
    """
    bundle_path = root / "bundle.json"
    bundle = json.loads(bundle_path.read_bytes()) if bundle_path.is_file() else {}
    if not isinstance(bundle, dict):
        raise ValueError("Source bundle manifest must be an object")
    declared = bundle.get(MAP_SHA256_FIELD)
    for digest in (declared, expected_sha256):
        if digest is not None and not is_lowercase_sha256(digest):
            raise ValueError("Initial-query source-map digest must be a lowercase SHA-256")
    if declared is not None and expected_sha256 is not None and declared != expected_sha256:
        raise ValueError("Initial-query source-map expected digests conflict")
    path = root / MAP_RELATIVE_PATH
    contents = path.read_bytes() if path.exists() else None
    actual = None if contents is None else hashlib.sha256(contents).hexdigest()
    if (declared is not None and declared != actual) or (expected_sha256 is not None and expected_sha256 != actual):
        raise ValueError("Initial-query source-map SHA-256 differs from the expected digest")
    datasets: dict[str, dict[str, InitialQuerySource]] = {}
    source_ids: dict[str, set[str]] = {}
    for number, line in enumerate([] if contents is None else contents.decode("utf-8").splitlines(), 1):
        row: Any = json.loads(line)
        required = ("dataset", "source_query_signature_id", "initial_query_signature_id")
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise ValueError(f"Invalid initial-query source-map row: {path}:{number}")
        dataset, source, clone = (row[key] for key in required)
        if (
            dataset in {".", ".."}
            or Path(dataset).name != dataset
            or any(character in dataset for character in ("/", "\\", ":"))
        ):
            raise ValueError(f"Invalid source-map dataset: {dataset!r}")
        entries = datasets.setdefault(dataset, {})
        sources = source_ids.setdefault(dataset, set())
        if clone in entries or source in sources:
            raise ValueError(f"Duplicate initial-query source-map identity: {dataset}/{clone}")
        if source == clone or source.startswith("visibleq:"):
            raise ValueError(f"Source-map origin must be an original signature: {dataset}/{clone}")
        for key in ("source_paper_id", "initial_paper_id"):
            if key in row and (not isinstance(row[key], str) or not row[key]):
                raise ValueError(f"Invalid source-map paper identity: {path}:{number}")
        entries[clone] = InitialQuerySource(source, row.get("source_paper_id"), row.get("initial_paper_id"))
        sources.add(source)
    for dataset, entries in datasets.items():
        if any(entry.signature_id in entries for entry in entries.values()):
            raise ValueError(f"Source-map origins include another derived query: {dataset}")
    result = InitialQuerySources(datasets, actual)
    if arrow_datasets is None:
        result.validate_snapshot(root, dataset_names)
    else:
        for dataset, arrow_dataset in arrow_datasets.items():
            result.validate_arrow_dataset(dataset, arrow_dataset)
    return result

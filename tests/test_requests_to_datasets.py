import tarfile
from pathlib import Path

import pyarrow as pa
import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset, read_arrow_collection_root
from s2and.arrow_tables import write_name_counts_index
from s2and.incremental_linking.feature_block import write_arrow_ipc_table
from tests.helpers import write_minimal_arrow_prediction_bundle

pytestmark = pytest.mark.tooling


def test_intake_publishes_and_resumes_snapshot(tmp_path: Path) -> None:
    from scripts.intake.requests_to_datasets import convert_requests

    source = tmp_path / "source"
    write_minimal_arrow_prediction_bundle(source, include_specter=True)
    signature_path = source / "signatures.arrow"
    table = pa.ipc.open_file(str(signature_path)).read_all()
    table = table.append_column("author_email", pa.array(["ada@example.org"] * table.num_rows))
    table = table.set_column(
        table.schema.get_field_index("author_affiliations"),
        "author_affiliations",
        pa.array([["Write to ada@example.org"]] * table.num_rows),
    )
    write_arrow_ipc_table(table, signature_path)
    paper_path = source / "papers.arrow"
    papers = pa.ipc.open_file(str(paper_path)).read_all().append_column("abstract", pa.array(["Private text"]))
    papers = pa.concat_tables(
        [
            papers,
            pa.table(
                {
                    "paper_id": ["p1"],
                    "title": ["Restricted"],
                    "venue": [""],
                    "journal_name": [""],
                    "abstract": ["Hidden"],
                }
            ),
        ]
    )
    write_arrow_ipc_table(papers, paper_path)
    bundle = tmp_path / "a_lovelace.arrow.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in source.glob("*.arrow"):
            archive.add(path, arcname=path.name)
        archive.add(source / "manifest.json", arcname="../unrelated")
    index, _ = write_name_counts_index(tmp_path, ({}, {}, {}, {}))
    root = tmp_path / "snapshot"
    permissions = {"allowed_papers": frozenset({"p0"}), "abstract_allowed_papers": frozenset()}
    result = convert_requests([bundle], root, Path(index), **permissions)
    assert result["a_lovelace"]["source_bundle_sha256"] == sha256_file(bundle)
    datasets, _, _ = read_arrow_collection_root(root / "manifest.json")
    dataset = datasets["a_lovelace"].parent
    assert dataset == root / "datasets" / "a_lovelace"
    with ArrowDataset.open(dataset, require_specter=True, require_name_counts_index=True):
        pass
    assert not any(root.rglob("unrelated"))
    signatures = pa.ipc.open_file(str(dataset / "signatures.arrow")).read_all()
    assert signatures["author_email"][0].as_py() != "ada@example.org"
    assert signatures["author_affiliations"][0].as_py() == ["Write to [EMAIL_REDACTED]"]
    papers = pa.ipc.open_file(str(dataset / "papers.arrow")).read_all()
    assert papers["paper_id"].to_pylist() == ["p0"]
    assert papers["abstract"].to_pylist() == ["Has Abstract"]
    assert convert_requests([bundle], root, Path(index), **permissions)["a_lovelace"]["reused"]
    (dataset / "papers.arrow").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="papers"):
        convert_requests([bundle], root, Path(index), **permissions)


def test_intake_rejects_bundle_named_for_another_block(tmp_path: Path) -> None:
    from scripts.intake.requests_to_datasets import convert_requests

    source = tmp_path / "source"
    write_minimal_arrow_prediction_bundle(source, include_specter=True)
    bundle = tmp_path / "g hopper.arrow.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in source.glob("*.arrow"):
            archive.add(path, arcname=path.name)
    index, _ = write_name_counts_index(tmp_path, ({}, {}, {}, {}))
    with pytest.raises(ValueError, match="author_block"):
        convert_requests(
            [bundle],
            tmp_path / "snapshot",
            Path(index),
            allowed_papers=frozenset({"p0"}),
            abstract_allowed_papers=frozenset(),
        )

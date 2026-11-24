import tarfile
from pathlib import Path

import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset, read_arrow_collection_root
from s2and.arrow_tables import write_name_counts_index
from tests.helpers import write_minimal_arrow_prediction_bundle

pytestmark = pytest.mark.tooling


def test_intake_publishes_and_resumes_snapshot(tmp_path: Path) -> None:
    from scripts.intake.requests_to_datasets import convert_requests

    source = tmp_path / "source"
    write_minimal_arrow_prediction_bundle(source, include_specter=True)
    bundle = tmp_path / "a_lovelace.arrow.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in source.glob("*.arrow"):
            archive.add(path, arcname=path.name)
        archive.add(source / "manifest.json", arcname="../unrelated")
    index, _ = write_name_counts_index(tmp_path, ({}, {}, {}, {}))
    root = tmp_path / "snapshot"
    result = convert_requests([bundle], root, Path(index))
    assert result["a_lovelace"]["source_bundle_sha256"] == sha256_file(bundle)
    datasets, _, _ = read_arrow_collection_root(root / "manifest.json")
    dataset = datasets["a_lovelace"].parent
    assert dataset == root / "datasets" / "a_lovelace"
    with ArrowDataset.open(dataset, require_specter=True, require_name_counts_index=True):
        pass
    assert not any(root.rglob("unrelated"))
    assert convert_requests([bundle], root, Path(index))["a_lovelace"]["reused"]

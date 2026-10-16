from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from s2and.arrow_inputs import ArrowDataset
from scripts.production.data.package_pairwise_inputs import package_pairwise_inputs
from tests.fixture_templates import copy_template
from tests.pairwise_release_helpers import (
    convert_release_arrow,
    write_pairwise_artifacts,
    write_pairwise_dataset,
)


def _write_prepared(tmp_path: Path) -> Path:
    root = tmp_path / "prepared"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _, index, _ = write_pairwise_artifacts(artifacts)
    shutil.copytree(index, root / "name_counts_global" / "name_counts_index")
    model = root / "populations" / "model_source"
    evaluation = root / "populations" / "evaluation_source"
    write_pairwise_dataset(model)
    shutil.copytree(model / "qian", evaluation / "qian")
    sidecar = "restored_unknown_signature_ids_20261009.json"
    for source in (model / "qian", evaluation / "qian"):
        (source / sidecar).write_text("[]", encoding="utf-8")
    for filename in ("test_pairs.json", "test_blocks.json"):
        (evaluation / "qian" / filename).write_text("{}", encoding="utf-8")
    write_pairwise_dataset(root, "augmented")
    for folder, source in (("release", model), ("evaluation", evaluation)):
        arrow = root / "arrow" / folder
        convert_release_arrow(source, "qian", arrow, index)
        convert_release_arrow(root, "augmented", arrow, index)
    for filename, first, second in (("train_pairs.csv", "0-0", "0-1"), ("val_pairs.csv", "0-2", "0-3")):
        (root / "augmented" / filename).write_text(
            f"signature_id_1,signature_id_2,label\nsignature-{first},signature-{second},1\n",
            encoding="utf-8",
        )
    (root / "augmented" / "test_pairs.json").write_text(
        json.dumps([{"signature_id_1": "signature-1-0", "signature_id_2": "signature-1-1", "label": 1}]),
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="module")
def _prepared_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("prepared_template")
    _write_prepared(root)
    return root


@pytest.fixture
def prepared(_prepared_template: Path, tmp_path: Path) -> Path:
    # Tests change the prepared inputs, so each test gets its own copy.
    copy_template(_prepared_template, tmp_path)
    return tmp_path / "prepared"


def test_package_preserves_sources_and_opens_after_copy(prepared: Path, tmp_path: Path) -> None:
    output = tmp_path / "package"
    result = package_pairwise_inputs(prepared, output, datasets=("qian",))
    portable = tmp_path / "portable"
    shutil.copytree(output, portable)
    for mode, collection in result["arrow"].items():
        for name in ("qian", "augmented"):
            with ArrowDataset.open(
                portable / collection / name,
                require_specter=True,
                require_name_counts_index=True,
            ) as dataset:
                assert dataset.name_counts_index is not None
            manifest = json.loads((portable / collection / name / "manifest.json").read_text())
            counter_path = (portable / collection / name / manifest["paths"]["name_counts_index"]).resolve()
            assert counter_path == portable / "name_counts_index"
        for files in result["sources"][mode].values():
            for spec in files.values():
                assert not Path(spec["path"]).is_absolute()
                assert (portable / spec["path"]).is_file()
    assert result["sources"]["model"]["qian"]["papers"] == result["sources"]["evaluation"]["qian"]["papers"]
    for role in ("signatures", "clusters"):
        copied = output / result["sources"]["model"]["qian"][role]["path"]
        original = prepared / "populations" / "model_source" / "qian" / f"qian_{role}.json"
        assert copied.read_bytes() == original.read_bytes()
    with pytest.raises(FileExistsError):
        package_pairwise_inputs(prepared, output, datasets=("qian",))


def test_package_rejects_incompatible_shared_source(prepared: Path, tmp_path: Path) -> None:
    source = prepared / "populations" / "evaluation_source" / "qian" / "qian_papers.json"
    data = json.loads(source.read_text())
    next(iter(data.values()))["title"] = "Changed source"
    source.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Model and evaluation papers differ"):
        package_pairwise_inputs(prepared, tmp_path / "package", datasets=("qian",))

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from s2and.arrow_inputs import ArrowDataset
from scripts.production.data.package_pairwise_inputs import package_pairwise_inputs
from scripts.production.data.prepare_pairwise_inputs import prepare_benchmark
from scripts.production.model.unknown_signatures import load_unknown_signatures
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


@pytest.mark.parametrize("sidecar_value", [["duplicate", "duplicate"], {"unknown": []}])
def test_package_rejects_invalid_sidecars_before_output(prepared: Path, tmp_path: Path, sidecar_value: object) -> None:
    source = prepared / "populations" / "model_source" / "qian"
    (source / "restored_unknown_signature_ids_20261009.json").write_text(json.dumps(sidecar_value), encoding="utf-8")
    output = tmp_path / "package"
    with pytest.raises(ValueError, match="sidecar"):
        package_pairwise_inputs(prepared, output, datasets=("qian",))
    assert not output.exists()


def test_prepared_custom_sidecar_reaches_packaging_and_training_discovery(prepared: Path, tmp_path: Path) -> None:
    source = prepared / "populations" / "model_source"
    original = source / "qian" / "restored_unknown_signature_ids_20261009.json"
    filename = "restored_unknown_signature_ids_reviewed.json"
    original.rename(original.with_name(filename))
    unknown_ids = {f"signature-{block}-0" for block in range(10)}
    original.with_name(filename).write_text(json.dumps(sorted(unknown_ids)), encoding="utf-8")
    current = tmp_path / "current"
    prepare_benchmark(
        "qian",
        source_root=source,
        fixed_pair_root=source,
        output_root=current,
        unknown_filename=filename,
        evaluation_pairs=10,
    )
    shutil.copytree(prepared / "name_counts_global", current / "name_counts_global")
    shutil.copytree(prepared / "augmented", current / "augmented")
    (current / "augmented" / "test_pairs.csv").write_text(
        "signature_id_1,signature_id_2,label\nsignature-1-0,signature-1-1,1\n", encoding="utf-8"
    )
    index = current / "name_counts_global" / "name_counts_index"
    for folder, population in (("release", "model_source"), ("evaluation", "evaluation_source")):
        arrow = current / "arrow" / folder
        convert_release_arrow(current / "populations" / population, "qian", arrow, index)
        convert_release_arrow(current, "augmented", arrow, index)
    output = tmp_path / "package"
    result = package_pairwise_inputs(current, output, datasets=("qian",))
    for mode in ("model", "evaluation"):
        files = {role: output / spec["path"] for role, spec in result["sources"][mode]["qian"].items()}
        actual = load_unknown_signatures(files)
        signatures = json.loads(files["signatures"].read_bytes())
        assert actual.ids == unknown_ids & signatures.keys()
        assert actual.ids
        assert actual.path == files["clusters"].with_name(filename)
        assert actual.sha256 == result["sources"][mode]["qian"]["unknown_signatures"]["sha256"]

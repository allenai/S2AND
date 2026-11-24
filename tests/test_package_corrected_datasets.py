from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from s2and._sha256 import sha256_file
from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block_arrow import write_name_counts_index
from s2and.production_training_contract import load_model_plan
from scripts.common.arrow_roots import write_root_manifest
from scripts.common.unknown_signatures import load_unknown_signatures
from scripts.linker.linker_source_bundle import validate_portable_source_bundle
from scripts.pairwise.package_pairwise_inputs import package_pairwise_inputs
from scripts.release.package_corrected_datasets import (
    DEFAULT_EPS_GRID,
    main,
    package_corrected_datasets,
    validate_corrected_datasets,
)
from tests.fixture_templates import copy_template
from tests.pairwise_release_helpers import write_pairwise_dataset
from tests.test_linker_source_bundle import (
    _link_directory,
    _write_arrow_root,
    _write_linker_inputs,
    _write_singleton_orcid_views,
)
from tests.test_package_pairwise_inputs import _write_prepared

pytestmark = pytest.mark.tooling


@pytest.fixture(scope="module")
def _package_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("package_template")
    prepared_root = _write_prepared(tmp_path)
    pairwise = tmp_path / "pairwise-inputs"
    package_pairwise_inputs(prepared_root, pairwise, datasets=("qian",))
    corrected = tmp_path / "corrected"
    shutil.copytree(prepared_root / "populations" / "model_source" / "qian", corrected / "qian")
    (corrected / "qian" / "qian_added_signatures.json").write_text("{}", encoding="utf-8")
    (corrected / "qian" / "preparation_report.before_raw_names.json").write_text("{}", encoding="utf-8")
    write_pairwise_dataset(corrected, "patents")
    public = tmp_path / "public"
    _write_arrow_root(public, dataset="qian", replay=False, name_counts_source=pairwise / "name_counts_index")
    write_root_manifest(public, dataset_manifests={"qian": "qian/manifest.json"}, release_version="1.3")
    linker = tmp_path / "linker"
    _write_arrow_root(linker, dataset="qian", replay=True, name_counts_source=pairwise / "name_counts_index")
    _write_linker_inputs(linker, leak=False, dataset="qian")
    return tmp_path


@pytest.fixture
def package_inputs(_package_template: Path, tmp_path: Path) -> dict[str, Path]:
    # Tests change the inputs, so each test gets its own copy of the module template.
    copy_template(_package_template, tmp_path)
    pairwise = tmp_path / "pairwise-inputs"
    return {
        "public_data_root": tmp_path / "public",
        "corrected_source_root": tmp_path / "corrected",
        "pairwise_inputs_root": pairwise,
        "linker_source_bundle": tmp_path / "linker",
        "name_counts_index": pairwise / "name_counts_index",
        "output_root": tmp_path / "complete",
    }


def _package(
    inputs: dict[str, Path],
    *,
    eps_grid: Sequence[float] = DEFAULT_EPS_GRID,
    corrected_replay_root: Path | None = None,
    corrected_replay_embeddings_root: Path | None = None,
    precision_f1_tolerance: float = 0.0,
) -> dict[str, Any]:
    return package_corrected_datasets(
        public_data_root=inputs["public_data_root"],
        corrected_source_root=inputs["corrected_source_root"],
        pairwise_inputs_root=inputs["pairwise_inputs_root"],
        linker_source_bundle=inputs["linker_source_bundle"],
        name_counts_index=inputs["name_counts_index"],
        output_root=inputs["output_root"],
        eps_grid=eps_grid,
        corrected_replay_root=corrected_replay_root,
        corrected_replay_embeddings_root=corrected_replay_embeddings_root,
        precision_f1_tolerance=precision_f1_tolerance,
    )


def test_model_plan_records_the_explicit_precision_tolerance(package_inputs: dict[str, Path]) -> None:
    tolerance = 0.001
    _package(package_inputs, precision_f1_tolerance=tolerance)

    plan_path = package_inputs["output_root"] / "pairwise" / "model_plan.json"
    eps = json.loads(plan_path.read_text(encoding="utf-8"))["eps"]
    assert eps["precision_f1_tolerance"] == tolerance
    assert load_model_plan(plan_path).eps.precision_f1_tolerance == tolerance


def test_packaging_cli_requires_a_precision_tolerance(
    package_inputs: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    args = [arg for name, path in package_inputs.items() for arg in (f"--{name.replace('_', '-')}", str(path))]

    with pytest.raises(SystemExit) as exc:
        main(args)

    assert exc.value.code == 2
    assert "--precision-f1-tolerance" in capsys.readouterr().err
    assert not package_inputs["output_root"].exists()


@pytest.mark.parametrize("problem", ["missing-test-group", "unreviewed-singletons", "missing-replay-dataset"])
def test_packaging_rejects_invalid_linker_labels_before_copy(package_inputs: dict[str, Path], problem: str) -> None:
    linker = package_inputs["linker_source_bundle"]
    if problem == "missing-test-group":
        labels = linker / "labels" / "hwang_eval.parquet"
        pd.read_parquet(labels).assign(query_group_id="hwang:other").to_parquet(labels, index=False)
        message = "did not all match source rows.*hwang:q1"
    elif problem == "unreviewed-singletons":
        _write_singleton_orcid_views(linker, view_count=10)
        message = "0/10 marked negative_singleton_orcid views"
    else:
        labels = linker / "labels" / "train.parquet"
        pd.read_parquet(labels).assign(dataset="unshipped").to_parquet(labels, index=False)
        message = "absent from the replay collection.*unshipped"

    with pytest.raises(ValueError, match=message):
        _package(package_inputs)

    assert not package_inputs["output_root"].exists()


def test_complete_snapshot_preserves_current_sources_and_uses_one_index(
    package_inputs: dict[str, Path], tmp_path: Path
) -> None:
    original_headers = {
        name: sha256_file(package_inputs[name] / filename)
        for name, filename in (
            ("public_data_root", "manifest.json"),
            ("pairwise_inputs_root", "inputs.json"),
            ("linker_source_bundle", "bundle.json"),
        )
    }
    original_labels = (package_inputs["linker_source_bundle"] / "labels" / "train.parquet").read_bytes()
    report = _package(package_inputs)
    root = package_inputs["output_root"]
    assert report["corrected_datasets"] == ["patents", "qian"]
    assert report["model_datasets"] == ["qian", "augmented"]
    assert report["incremental_source_rows"] == 4
    index_directories = sorted(path.relative_to(root).as_posix() for path in root.rglob("name_counts_index"))
    assert index_directories == ["name_counts_index"]
    source = package_inputs["corrected_source_root"]
    for name in ("qian", "patents"):
        for file in (source / name).iterdir():
            if file.name.startswith("preparation_report"):
                assert not (root / "corrected_sources" / name / file.name).exists()
            else:
                assert (root / "corrected_sources" / name / file.name).read_bytes() == file.read_bytes()
    papers = json.loads((root / "corrected_sources" / "qian" / "qian_papers.json").read_text())
    assert next(iter(papers.values()))["abstract"] == "Deterministic abstract for topic 0."
    inputs = json.loads((root / "pairwise" / "inputs.json").read_text())
    assert inputs["name_counts_index"] == "../name_counts_index"
    assert inputs["corrected_source_root"] == "../corrected_sources"
    assert inputs["linker_source_bundle"] == ".."
    for mode in ("model", "evaluation"):
        assert inputs["sources"][mode]["qian"]["papers"]["path"] == "../corrected_sources/qian/qian_papers.json"
        assert not (root / "pairwise" / "sources" / "common" / "qian" / "qian_papers.json").exists()
    plan = load_model_plan(root / "pairwise" / "model_plan.json")
    assert list(plan.datasets) == ["qian", "augmented"]
    assert plan.eps.grid == tuple(step / 100 for step in range(101))
    assert json.loads((root / "pairwise" / "model_plan.json").read_text())["eps"]["precision_f1_tolerance"] == 0.0
    assert plan.eps.precision_f1_tolerance == 0.0
    assert plan.datasets["qian"].files["papers"] == root / "corrected_sources" / "qian" / "qian_papers.json"
    for name, filename in (
        ("public_data_root", "manifest.json"),
        ("pairwise_inputs_root", "inputs.json"),
        ("linker_source_bundle", "bundle.json"),
    ):
        assert sha256_file(package_inputs[name] / filename) == original_headers[name]
    relocated = tmp_path / "other" / "directory"
    shutil.copytree(root, relocated)
    for name in (
        "public_data_root",
        "corrected_source_root",
        "pairwise_inputs_root",
        "linker_source_bundle",
        "output_root",
    ):
        shutil.rmtree(package_inputs[name])
    assert validate_corrected_datasets(relocated)["model_datasets"] == ["qian", "augmented"]
    assert (relocated / "labels" / "train.parquet").read_bytes() == original_labels
    assert pd.read_parquet(relocated / "labels" / "train.parquet")["label"].tolist() == [1]
    with ArrowDataset.open(relocated / "pairwise" / "arrow" / "release" / "qian") as data:
        assert data.name_counts_index is not None
        counts = data.name_counts_index.lookup_many(["alex"], [None], [None], [None])
        assert counts[0][0] == 20


def test_snapshot_validation_rejects_changed_outputs(package_inputs: dict[str, Path], tmp_path: Path) -> None:
    # Package once and mutate copies: each packaging run costs about a second.
    _package(package_inputs)
    packaged = package_inputs["output_root"]

    def changed_copy(name: str) -> Path:
        root = tmp_path / "changed" / name
        shutil.copytree(packaged, root)
        return root

    root = changed_copy("duplicate_index")
    shutil.copytree(root / "name_counts_index", root / "pairwise" / "name_counts_index")
    with pytest.raises(ValueError, match="Require one shared"):
        validate_corrected_datasets(root)

    # Benchmark and replay datasets share the name qian; validation must check the benchmark copy.
    root = changed_copy("benchmark_payload")
    paper_table = root / "benchmarks" / "qian" / "papers.arrow"
    paper_table.write_bytes(paper_table.read_bytes().replace(b"Notes", b"WRONG"))
    with pytest.raises(ValueError, match="mismatch"):
        validate_portable_source_bundle(root)

    root = changed_copy("master_bytes")
    (root / "corrected_sources" / "qian" / "qian_clusters.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrected source SHA-256 mismatch"):
        validate_corrected_datasets(root)


def test_complete_snapshot_rejects_changed_pairwise_source(package_inputs: dict[str, Path]) -> None:
    papers = package_inputs["pairwise_inputs_root"] / "sources" / "common" / "qian" / "qian_papers.json"
    papers.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Source SHA-256 mismatch"):
        _package(package_inputs)


def test_complete_snapshot_rejects_invalid_eps_before_copy(package_inputs: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="EPS grid"):
        _package(package_inputs, eps_grid=[0.5, 0.5])
    assert not package_inputs["output_root"].exists()


def test_complete_snapshot_rejects_a_different_count_index(package_inputs: dict[str, Path], tmp_path: Path) -> None:
    index_path, _metrics = write_name_counts_index(
        tmp_path / "other-counts", ({"alex": 21}, {"adams": 4}, {"alex adams": 2}, {"adams a": 4})
    )
    package_inputs["name_counts_index"] = Path(index_path)
    with pytest.raises(ValueError, match="authoritative index"):
        _package(package_inputs)
    assert not package_inputs["output_root"].exists()


def test_complete_snapshot_does_not_recertify_a_changed_replay_payload(package_inputs: dict[str, Path]) -> None:
    paper_table = package_inputs["linker_source_bundle"] / "datasets" / "qian" / "papers.arrow"
    paper_table.write_bytes(paper_table.read_bytes().replace(b"Notes", b"WRONG"))
    with pytest.raises(ValueError, match="inventory mismatch"):
        _package(package_inputs)


def test_complete_snapshot_rejects_benchmark_names_that_escape_output(package_inputs: dict[str, Path]) -> None:
    manifest_path = package_inputs["public_data_root"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_manifests"]["../escape"] = manifest["dataset_manifests"].pop("qian")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid benchmark dataset name"):
        _package(package_inputs)
    assert not package_inputs["output_root"].exists()


def test_complete_snapshot_rejects_a_linked_input_before_copy(package_inputs: dict[str, Path], tmp_path: Path) -> None:
    source = package_inputs["linker_source_bundle"]
    target = tmp_path / "external-labels"
    shutil.move(source / "labels", target)
    _link_directory(source / "labels", target)
    output = package_inputs["output_root"]
    with pytest.raises(ValueError, match="dataset snapshot input contains a filesystem link"):
        _package(package_inputs)
    assert not output.exists()
    assert (target / "train.parquet").is_file()


def test_complete_snapshot_preserves_current_replay_text_and_vectors(
    package_inputs: dict[str, Path], tmp_path: Path
) -> None:
    replay_raw = tmp_path / "replay-raw"
    files = write_pairwise_dataset(replay_raw, "giant_demo")
    replay_embeddings = tmp_path / "replay-embeddings"
    replay_embeddings.mkdir()
    vector_path = replay_embeddings / files["specter_embeddings"].name
    shutil.move(files["specter_embeddings"], vector_path)
    expected_vector = vector_path.read_bytes()
    _package(package_inputs, corrected_replay_root=replay_raw, corrected_replay_embeddings_root=replay_embeddings)
    root = package_inputs["output_root"]
    text_path = root / "corrected_replay" / "raw" / "giant_demo" / "giant_demo_papers.json"
    assert text_path.read_bytes() == files["papers"].read_bytes()
    assert next(iter(json.loads(text_path.read_text()).values()))["abstract"] == "Deterministic abstract for topic 0."
    assert (root / "corrected_replay" / "embeddings" / vector_path.name).read_bytes() == expected_vector
    inventory = json.loads((root / "corrected_sources" / "manifest.json").read_text())["files"]
    assert inventory[text_path.relative_to(root).as_posix()] == sha256_file(text_path)
    relocated = tmp_path / "different" / "snapshot"
    shutil.copytree(root, relocated)
    shutil.rmtree(root)
    shutil.rmtree(replay_raw)
    shutil.rmtree(replay_embeddings)
    assert validate_corrected_datasets(relocated)["corrected_datasets"] == ["patents", "qian"]


def test_complete_snapshot_preserves_the_population_uncertainty_sidecar(package_inputs: dict[str, Path]) -> None:
    pairwise = package_inputs["pairwise_inputs_root"]
    inputs_path = pairwise / "inputs.json"
    inputs = json.loads(inputs_path.read_text())
    spec = inputs["sources"]["model"]["qian"]["unknown_signatures"]
    unknown = pairwise / spec["path"]
    unknown.write_text('["signature-0-0"]', encoding="utf-8")
    spec["sha256"] = sha256_file(unknown)
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
    corrected = package_inputs["corrected_source_root"] / "qian"
    shutil.copyfile(unknown, corrected / unknown.name)
    master_clusters = corrected / "qian_clusters.json"
    clusters = json.loads(master_clusters.read_text())
    next(iter(clusters.values()))["model_version"] = 0
    master_clusters.write_text(json.dumps(clusters), encoding="utf-8")
    _package(package_inputs)
    root = package_inputs["output_root"]
    plan = load_model_plan(root / "pairwise" / "model_plan.json")
    actual = load_unknown_signatures(plan.datasets["qian"].files)
    assert actual.ids == frozenset({"signature-0-0"})
    assert actual.path is not None
    assert actual.path == plan.datasets["qian"].files["clusters"].parent / unknown.name
    assert actual.path.is_relative_to(root / "pairwise")


def test_complete_snapshot_requires_current_labels_for_each_master(package_inputs: dict[str, Path]) -> None:
    (package_inputs["corrected_source_root"] / "patents" / "patents_clusters.json").unlink()
    with pytest.raises(ValueError, match="Require cluster labels or train/test fixed pairs"):
        _package(package_inputs)


def test_complete_snapshot_copies_public_license_when_linker_has_none(package_inputs: dict[str, Path]) -> None:
    (package_inputs["linker_source_bundle"] / "LICENSE.txt").unlink()
    license_source = package_inputs["public_data_root"] / "LICENSE.txt"
    license_source.write_text("Current public data license\n", encoding="utf-8")
    _package(package_inputs)
    root = package_inputs["output_root"]
    assert (root / "LICENSE.txt").read_bytes() == license_source.read_bytes()
    assert validate_corrected_datasets(root)["public_dataset_count"] == 1


def test_complete_snapshot_requires_public_license_before_copy(package_inputs: dict[str, Path]) -> None:
    (package_inputs["public_data_root"] / "LICENSE.txt").unlink()
    with pytest.raises(FileNotFoundError, match="LICENSE.txt"):
        _package(package_inputs)
    assert not package_inputs["output_root"].exists()


def test_copy_file_hardlinks_and_is_idempotent_on_matching_sha256(tmp_path: Path) -> None:
    import os

    from scripts.release.package_corrected_datasets import _copy_file

    source = tmp_path / "source.txt"
    source.write_text("immutable payload\n", encoding="utf-8")
    destination = tmp_path / "nested" / "dest.txt"

    digest = _copy_file(source, destination)
    assert os.path.samefile(source, destination)
    # Matching SHA-256 on retry succeeds without error.
    assert _copy_file(source, destination, expected_sha256=digest) == digest

    other = tmp_path / "other.txt"
    other.write_text("different payload\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing destination SHA-256 mismatch"):
        _copy_file(other, destination)

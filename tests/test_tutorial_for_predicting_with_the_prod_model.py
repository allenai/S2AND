"""Exercise the prediction tutorial's block grouping, routes, and output."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from s2and.arrow_inputs import ArrowDataset
from scripts import tutorial_for_predicting_with_the_prod_model as tutorial
from tests.helpers import write_minimal_arrow_prediction_bundle

pytestmark = pytest.mark.tooling


def _load(monkeypatch: pytest.MonkeyPatch, clusterer: object) -> list[tuple[Path, int]]:
    from s2and import production_model

    loads: list[tuple[Path, int]] = []

    def load(path: Path, *, n_jobs: int) -> object:
        loads.append((path, n_jobs))
        return clusterer

    monkeypatch.setattr(production_model, "load_production_model", load)
    return loads


def test_arrow_route_predicts_every_block_and_writes_new_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa

    from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_arrow_ipc_table
    from tests.helpers import write_test_arrow_artifact_manifest

    root = tmp_path / "arrow"
    paths = write_minimal_arrow_prediction_bundle(root, include_specter=True)
    with pa.OSFile(paths["signatures"], "rb") as source:
        signatures = pa.ipc.open_file(source).read_all()
    block_column = signatures.schema.get_field_index("author_block")
    updated = signatures.set_column(
        block_column, "author_block", [pa.array(["b smith"] + ["a lovelace"] * (len(signatures) - 1))]
    )
    write_arrow_ipc_table(updated, Path(paths["signatures"]))
    write_arrow_batch_lookup_index(
        paths["signatures"], paths["signatures_batch_index"], key_column="signature_id", table_name="signatures"
    )
    write_test_arrow_artifact_manifest(root, paths)

    expected_blocks = {"a lovelace": [*map(str, range(1, 10)), "q1", "q2", "seed1"], "b smith": ["0"]}

    def predict_from_arrow(block_dict, arrow_dataset, **options):
        assert block_dict == expected_blocks
        assert options == {"batching_threshold": 17, "total_ram_bytes": 123}
        return {"c1": list(reversed(block_dict["a lovelace"])), "c2": block_dict["b smith"]}, None

    loads = _load(monkeypatch, SimpleNamespace(predict_from_arrow=predict_from_arrow))
    real_open = ArrowDataset.open
    opened = []

    def open_dataset(path: Path, **options: bool) -> ArrowDataset:
        assert path == root
        assert options == {"require_specter": True, "require_name_counts_index": True}
        dataset = real_open(path, require_specter=True)
        opened.append(dataset)
        return dataset

    monkeypatch.setattr(tutorial.ArrowDataset, "open", open_dataset)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    output = tmp_path / "clusters.json"
    arguments = [
        "--model-path",
        str(tmp_path / "model"),
        "--arrow-dataset",
        str(root),
        "--n-jobs",
        "2",
        "--batching-threshold",
        "17",
        "--total-ram-bytes",
        "123",
        "--output-json",
        str(output),
    ]
    tutorial.main(arguments)

    assert loads == [(tmp_path / "model", 2)]
    assert opened[0].closed
    assert json.loads(output.read_text(encoding="utf-8")) == {"c1": expected_blocks["a lovelace"], "c2": ["0"]}
    with pytest.raises(SystemExit):
        tutorial.main(arguments)


def test_json_route_builds_inference_data_and_predicts_its_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from s2and import data

    built: list[dict] = []

    class Dataset:
        def __init__(self, **options: object) -> None:
            built.append(options)

        def get_blocks(self) -> dict[str, list[str]]:
            return {"a lovelace": ["s1"]}

    def predict(block_dict, dataset):
        assert block_dict == {"a lovelace": ["s1"]} and isinstance(dataset, Dataset)
        return {"c1": ["s1"]}, None

    _load(monkeypatch, SimpleNamespace(predict=predict))
    monkeypatch.setattr(data, "ANDData", Dataset)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    records = tmp_path / "qian"

    tutorial.main(
        [
            "--model-path",
            "model",
            "--json-dir",
            str(records),
            "--specter-pickle",
            "specter2.pkl",
            "--name-counts-index",
            "index",
            "--n-jobs",
            "1",
        ]
    )

    assert built == [
        {
            "signatures": str(records / "signatures.json"),
            "papers": str(records / "papers.json"),
            "name": "qian",
            "mode": "inference",
            "specter_embeddings": "specter2.pkl",
            "name_counts_index": "index",
            "n_jobs": 1,
        }
    ]


@pytest.mark.parametrize("missing", ["--specter-pickle", "--name-counts-index"])
def test_json_route_requires_embeddings_and_index(missing: str) -> None:
    options = {"--specter-pickle": "specter2.pkl", "--name-counts-index": "index"}
    del options[missing]
    arguments = ["--model-path", "model", "--json-dir", "records"]
    for flag, value in options.items():
        arguments += [flag, value]

    with pytest.raises(SystemExit):
        tutorial.main(arguments)


def test_json_route_rejects_batching_threshold_before_loading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loads = _load(monkeypatch, SimpleNamespace())
    arguments = ["--model-path", "model", "--json-dir", "records", "--specter-pickle", "specter2.pkl"]
    arguments += ["--name-counts-index", "index", "--batching-threshold", "17"]

    with pytest.raises(SystemExit):
        tutorial.main(arguments)

    assert loads == []
    assert "--batching-threshold requires --arrow-dataset" in capsys.readouterr().err


def test_incremental_route_links_one_block_against_seeded_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa

    clusters_path = tmp_path / "aminer_clusters.json"
    clusters_path.write_text(json.dumps({"c1": {"signature_ids": ["s1", "s2", "s3"]}}), encoding="utf-8")
    opened = object()

    class OpenDataset:
        def __enter__(self) -> object:
            return opened

        def __exit__(self, *_exc: object) -> None:
            return None

    tables = {
        "signatures": pa.table(
            {
                "signature_id": ["s1", "s2", "s3", "s4", "s5"],
                "paper_id": ["p1", "p2", "p3", "p4", "p5"],
                "author_block": ["j wang", "j wang", "j wang", "j wang", "other"],
            }
        ),
        "papers": pa.table({"paper_id": ["p1", "p2", "p3", "p4", "p5"], "year": [2010, 2020, 1800, 2001, 2010]}),
    }

    def read_columns(dataset, name, columns):
        assert dataset is opened
        return tables[name].select(list(columns))

    def predict_incremental_from_arrow(block_signatures, arrow_dataset, **options):
        assert block_signatures == ["s1", "s2", "s3", "s4"] and arrow_dataset is opened
        assert options == {"cluster_seeds_require": {"s1": "c1"}, "batching_threshold": None, "total_ram_bytes": None}
        return {"clusters": {"c1": ["s1", "s2"], "new": ["s3"], "unlabeled": ["s4"]}}

    _load(monkeypatch, SimpleNamespace(predict_incremental_from_arrow=predict_incremental_from_arrow))
    monkeypatch.setattr(tutorial.ArrowDataset, "open", lambda path, **options: OpenDataset())
    monkeypatch.setattr(tutorial, "_read_columns", read_columns)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    tutorial.main(
        [
            "--model-path",
            "model",
            "--arrow-dataset",
            str(tmp_path / "aminer"),
            "--incremental-block",
            "j wang",
            "--seed-clusters",
            str(clusters_path),
            "--seed-before-year",
            "2012",
            "--n-jobs",
            "1",
        ]
    )

    tables["papers"] = tables["papers"].drop(["year"])
    with pytest.raises(ValueError, match="papers.year"):
        tutorial.seed_profiles_before_year(opened, ["s1"], labeled_clusters={}, before_year=2012)


@pytest.mark.parametrize("dropped", ["--arrow-dataset", "--seed-clusters"])
def test_incremental_route_requires_arrow_data_and_seed_clusters(dropped: str) -> None:
    options = {"--arrow-dataset": "aminer", "--seed-clusters": "clusters.json"}
    del options[dropped]
    arguments = ["--model-path", "model", "--incremental-block", "j wang"]
    for flag, value in options.items():
        arguments += [flag, value]
    if dropped == "--arrow-dataset":
        arguments += ["--json-dir", "records", "--specter-pickle", "s.pkl", "--name-counts-index", "index"]

    with pytest.raises(SystemExit):
        tutorial.main(arguments)

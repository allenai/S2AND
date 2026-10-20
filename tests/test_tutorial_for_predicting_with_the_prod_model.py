"""Exercise the prediction tutorial's block grouping, routes, and output."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from s2and.arrow_inputs import ArrowDataset
from scripts import tutorial_for_predicting_with_the_prod_model as tutorial
from tests.helpers import write_minimal_arrow_prediction_bundle


def test_arrow_blocks_group_every_signature_by_author_block(tmp_path: Path) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path / "arrow", include_specter=True)
    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        blocks = tutorial.arrow_blocks(arrow_dataset)

    assert blocks == {"a lovelace": [*map(str, range(10)), "q1", "q2", "seed1"]}


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
    lifecycle: list[str] = []
    blocks = {"a lovelace": ["s1", "s2"], "b smith": ["s3"]}

    class OpenDataset:
        def __enter__(self) -> "OpenDataset":
            lifecycle.append("open")
            return self

        def __exit__(self, *_exc: object) -> None:
            lifecycle.append("close")

    opened = OpenDataset()

    def open_dataset(path: Path, **options: bool) -> OpenDataset:
        assert path == tmp_path / "qian"
        assert options == {"require_specter": True, "require_name_counts_index": True}
        return opened

    def predict_from_arrow(block_dict, arrow_dataset, **options):
        assert block_dict == blocks and arrow_dataset is opened
        assert options == {"batching_threshold": 17, "total_ram_bytes": 123}
        lifecycle.append("predict")
        return {"c1": ["s2", "s1"], "c2": ["s3"]}, None

    loads = _load(monkeypatch, SimpleNamespace(predict_from_arrow=predict_from_arrow))
    monkeypatch.setattr(tutorial.ArrowDataset, "open", open_dataset)
    monkeypatch.setattr(tutorial, "arrow_blocks", lambda dataset: blocks)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    output = tmp_path / "clusters.json"
    arguments = [
        "--model-path",
        str(tmp_path / "model"),
        "--arrow-dataset",
        str(tmp_path / "qian"),
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
    assert lifecycle == ["open", "predict", "close"]
    assert json.loads(output.read_text(encoding="utf-8")) == {"c1": ["s1", "s2"], "c2": ["s3"]}
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

    def predict(block_dict, dataset, *, batching_threshold):
        assert block_dict == {"a lovelace": ["s1"]} and isinstance(dataset, Dataset)
        assert batching_threshold is None
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

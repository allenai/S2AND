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


def test_seed_profiles_use_labeled_signatures_on_earlier_papers(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyarrow as pa

    tables = {
        "signatures": pa.table({"signature_id": ["s1", "s2", "s3", "s4"], "paper_id": ["p1", "p2", "p3", "p4"]}),
        "papers": pa.table({"paper_id": ["p1", "p2", "p3", "p4"], "year": pa.array([2010, 2020, None, 2001])}),
    }
    monkeypatch.setattr(tutorial, "_read_columns", lambda _dataset, name, columns: tables[name].select(list(columns)))
    labeled = {"c1": {"signature_ids": ["s1", "s2", "s3"]}}

    seeds = tutorial.seed_profiles_before_year(
        None, ["s1", "s2", "s3", "s4"], labeled_clusters=labeled, before_year=2015
    )

    # s2 is too new, s3 has no year, and s4 has no label, so all three arrive as new signatures.
    assert seeds == {"s1": "c1"}


def test_seed_profiles_require_paper_years(tmp_path: Path) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path / "arrow", include_specter=True)
    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        with pytest.raises(ValueError, match="papers.year"):
            tutorial.seed_profiles_before_year(arrow_dataset, ["q1"], labeled_clusters={}, before_year=2015)


def test_incremental_route_links_one_block_against_seeded_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clusters_path = tmp_path / "aminer_clusters.json"
    clusters_path.write_text(json.dumps({"c1": {"signature_ids": ["s1"]}}), encoding="utf-8")
    opened = object()

    class OpenDataset:
        def __enter__(self) -> object:
            return opened

        def __exit__(self, *_exc: object) -> None:
            return None

    def seeds(arrow_dataset, signature_ids, *, labeled_clusters, before_year):
        assert arrow_dataset is opened and signature_ids == ["s1", "s2", "s3"]
        assert labeled_clusters == {"c1": {"signature_ids": ["s1"]}} and before_year == 2012
        return {"s1": "c1"}

    def predict_incremental_from_arrow(block_signatures, arrow_dataset, **options):
        assert block_signatures == ["s1", "s2", "s3"] and arrow_dataset is opened
        assert options == {"cluster_seeds_require": {"s1": "c1"}, "batching_threshold": None, "total_ram_bytes": None}
        return {"clusters": {"c1": ["s1", "s2"], "new": ["s3"]}}

    _load(monkeypatch, SimpleNamespace(predict_incremental_from_arrow=predict_incremental_from_arrow))
    monkeypatch.setattr(tutorial.ArrowDataset, "open", lambda path, **options: OpenDataset())
    monkeypatch.setattr(tutorial, "arrow_blocks", lambda dataset: {"j wang": ["s1", "s2", "s3"], "other": ["s4"]})
    monkeypatch.setattr(tutorial, "seed_profiles_before_year", seeds)
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

    output = capsys.readouterr().out
    assert "Seeded 1 profiles with 1 signatures." in output
    assert "Linked 2 new signatures: 1 joined a profile, 1 did not." in output


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

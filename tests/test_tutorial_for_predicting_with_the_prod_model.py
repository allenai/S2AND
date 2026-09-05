"""Exercise the tutorial's routing, prediction options, and evaluation results."""

import sys
from types import SimpleNamespace

import pytest

from scripts import tutorial_for_predicting_with_the_prod_model as tutorial


def test_tutorial_json_eval_uses_selected_split_and_scores_predictions() -> None:
    split = "val"
    block_dict = {"a": ["s1", "s2"]}
    predict_calls = []

    class Dataset:
        def get_blocks(self):
            return block_dict

        def split_blocks_helper(self, blocks):
            assert blocks is block_dict
            return tuple(blocks if name == split else {} for name in ("train", "val", "test"))

        def construct_cluster_to_signatures(self, blocks):
            assert blocks is block_dict
            return {"truth": ["s1", "s2"]}

    class Clusterer:
        def predict(self, blocks, dataset, *, use_s2_clusters, batching_threshold):
            predict_calls.append((blocks, dataset, use_s2_clusters, batching_threshold))
            return {"predicted1": ["s1"], "predicted2": ["s2"]}, None

    dataset = Dataset()
    clusterer = Clusterer()
    metrics, per_signature = tutorial._cluster_eval_with_predict_options(
        dataset,
        clusterer,
        split=split,
        use_s2_clusters=False,
        batching_threshold=5_000,
    )

    assert predict_calls == [(block_dict, dataset, False, 5_000)]
    assert metrics["B3 (P, R, F1)"] == pytest.approx((1.0, 0.5, 0.667))
    assert set(per_signature) == {"s1", "s2"}


@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_tutorial_arrow_cli_forwards_options_and_closes_dataset(monkeypatch, evaluation_fails: bool) -> None:
    from s2and import production_model
    from scripts import eval_prod_models

    lifecycle = []
    clusterer = SimpleNamespace(n_jobs=None)

    class ArrowDataset:
        def __enter__(self):
            lifecycle.append("open")
            return self

        def __exit__(self, *_exc):
            lifecycle.append("close")

    arrow_dataset = ArrowDataset()

    def evaluate(dataset, model, **options):
        assert dataset is arrow_dataset
        assert model is clusterer
        assert model.n_jobs == 1
        assert options == {
            "random_seed": 42,
            "n_jobs": 1,
            "split": "val",
            "total_ram_bytes": 123_456,
            "batching_threshold": 17,
        }
        lifecycle.append("evaluate")
        if evaluation_fails:
            raise ValueError("invalid Arrow input")
        return {"B3 (P, R, F1)": (1.0, 1.0, 1.0)}, {}

    monkeypatch.setattr(production_model, "load_production_model", lambda _path: clusterer)
    monkeypatch.setattr(eval_prod_models, "resolve_arrow_dataset", lambda *_args: arrow_dataset)
    monkeypatch.setattr(eval_prod_models, "cluster_eval_arrow", evaluate)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tutorial",
            "--model-path",
            "unused",
            "--dataset",
            "dummy",
            "--input-format",
            "arrow",
            "--n-jobs",
            "1",
            "--split",
            "val",
            "--arrow-total-ram-bytes",
            "123456",
            "--batching-threshold",
            "17",
        ],
    )

    if evaluation_fails:
        with pytest.raises(ValueError, match="invalid Arrow input"):
            tutorial.main()
    else:
        tutorial.main()

    assert lifecycle == ["open", "evaluate", "close"]


@pytest.mark.parametrize(
    ("requested", "failure", "expected"),
    [
        ("auto", FileNotFoundError, "json"),
        ("json", AssertionError, "json"),
        ("arrow", FileNotFoundError, FileNotFoundError),
        ("auto", ValueError, ValueError),
    ],
)
def test_tutorial_input_fallback_only_handles_missing_optional_arrow(requested, failure, expected) -> None:
    def resolve(*_args):
        raise failure("cannot open dataset")

    options = {
        "requested_input_format": requested,
        "dataset_name": "dummy",
        "arrow_data_root": "unused",
        "specter_suffix": "_specter2.pkl",
        "resolve_arrow_dataset": resolve,
    }
    if isinstance(expected, str):
        assert tutorial._select_input_route(**options) == (expected, None)
    else:
        with pytest.raises(expected, match="cannot open dataset"):
            tutorial._select_input_route(**options)

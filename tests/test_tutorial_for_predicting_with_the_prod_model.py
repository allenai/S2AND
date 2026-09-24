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


@pytest.mark.parametrize(("split", "evaluation_fails"), [("val", False), ("test", False), ("test", True)])
def test_tutorial_arrow_cli_forwards_options_and_closes_dataset(
    monkeypatch, split: str, evaluation_fails: bool
) -> None:
    from s2and import production_model
    from scripts import eval_prod_models

    lifecycle = []
    clusterer = SimpleNamespace(n_jobs=None)
    recorded_split = {"block_membership_sha256": "0" * 64, "test_block_ids": ["a"]}

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
            "random_seed": 7,
            "n_jobs": 1,
            "split": split,
            "total_ram_bytes": 123_456,
            "batching_threshold": 17,
            "test_split": recorded_split if split == "test" else None,
        }
        lifecycle.append("evaluate")
        if evaluation_fails:
            raise ValueError("invalid Arrow input")
        return {"B3 (P, R, F1)": (1.0, 1.0, 1.0)}, {}

    def load(_path, *, n_jobs):
        clusterer.n_jobs = n_jobs
        return clusterer

    monkeypatch.setattr(production_model, "load_production_model", load)
    monkeypatch.setattr(eval_prod_models, "bundle_data_random_seed", lambda _path: 7)
    monkeypatch.setattr(eval_prod_models, "bundle_cluster_test_splits", lambda _path: {"dummy": recorded_split})
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
            split,
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


def test_tutorial_json_eval_scores_recorded_test_blocks() -> None:
    from s2and.production_training_contract import block_membership_sha256

    blocks = {"a": ["s1", "s2"], "b": ["s3"]}
    record = {"block_membership_sha256": block_membership_sha256(blocks), "test_block_ids": ["a"]}
    predicted_blocks = []

    class Dataset:
        def get_blocks(self):
            return blocks

        def split_blocks_helper(self, _blocks):
            raise AssertionError("recorded test blocks must not be re-derived from a seed")

        def construct_cluster_to_signatures(self, block_dict):
            assert block_dict == {"a": ["s1", "s2"]}
            return {"truth": ["s1", "s2"]}

    class Clusterer:
        def predict(self, block_dict, dataset, *, use_s2_clusters, batching_threshold):
            predicted_blocks.append(block_dict)
            return {"predicted": ["s1", "s2"]}, None

    metrics, _per_signature = tutorial._cluster_eval_with_predict_options(
        Dataset(),
        Clusterer(),
        split="test",
        use_s2_clusters=False,
        batching_threshold=None,
        test_split=record,
    )

    assert predicted_blocks == [{"a": ["s1", "s2"]}]
    assert metrics["B3 (P, R, F1)"] == pytest.approx((1.0, 1.0, 1.0))


@pytest.mark.parametrize(
    ("present", "suffix", "expected"),
    [
        (["qian_specter2.pkl", "specter2.pkl"], "_specter2.pkl", "qian_specter2.pkl"),
        (["specter2.pkl", "specter.pickle"], "_specter2.pkl", "specter2.pkl"),
        (["specter.pickle"], "_specter2.pkl", None),
        (["specter.pickle"], "_specter.pickle", "specter.pickle"),
    ],
    ids=["mini-naming", "fixture-naming", "no-specter1-fallback", "explicit-specter1"],
)
def test_tutorial_json_route_selects_requested_specter_generation(tmp_path, present, suffix, expected) -> None:
    dataset_dir = tmp_path / "qian"
    dataset_dir.mkdir()
    for name in present:
        (dataset_dir / name).write_bytes(b"")

    if expected is None:
        with pytest.raises(FileNotFoundError):
            tutorial._json_specter_path(str(tmp_path), "qian", suffix)
    else:
        assert tutorial._json_specter_path(str(tmp_path), "qian", suffix) == str(dataset_dir / expected)

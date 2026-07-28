import argparse
import os

import pytest

from scripts import tutorial_for_predicting_with_the_prod_model as tutorial


def test_tutorial_defaults_to_real_mini_dataset_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    class ParsedDefault(Exception):
        pass

    def assert_default(parser: argparse.ArgumentParser) -> None:
        assert parser.get_default("data_root") == os.path.join("s2and", "data-backup", "s2and_mini")
        raise ParsedDefault

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", assert_default)

    with pytest.raises(ParsedDefault):
        tutorial.main()


def test_tutorial_json_eval_uses_current_predict_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    import s2and.eval as eval_module

    block_dict = {"a": ["s1"]}
    predict_calls = []

    class Dataset:
        def get_blocks(self):
            return block_dict

        def split_blocks_helper(self, blocks):
            assert blocks is block_dict
            return {}, {}, blocks

        def construct_cluster_to_signatures(self, blocks):
            assert blocks is block_dict
            return {"truth": ["s1"]}

    class Clusterer:
        def predict(self, blocks, dataset, *, use_s2_clusters, batching_threshold):
            predict_calls.append((blocks, dataset, use_s2_clusters, batching_threshold))
            return {"predicted": ["s1"]}, None

    monkeypatch.setattr(
        eval_module,
        "b3_precision_recall_fscore",
        lambda *_args: (1.0, 1.0, 1.0, {}, [], []),
    )
    monkeypatch.setattr(eval_module, "pairwise_precision_recall_fscore", lambda *_args: (1.0, 1.0, 1.0))

    dataset = Dataset()
    clusterer = Clusterer()
    tutorial._cluster_eval_with_predict_options(
        dataset,
        clusterer,
        split="test",
        use_s2_clusters=False,
        batching_threshold=5_000,
    )

    assert predict_calls == [(block_dict, dataset, False, 5_000)]


def test_tutorial_arrow_route_accepts_native_subblocking_threshold() -> None:
    arrow_dataset = object()

    input_format, resolved_dataset = tutorial._select_input_route(
        requested_input_format="auto",
        dataset_name="dummy",
        arrow_data_root="unused",
        specter_suffix="_specter2.pkl",
        resolve_arrow_dataset=lambda *_args: arrow_dataset,
    )

    assert input_format == "arrow"
    assert resolved_dataset is arrow_dataset

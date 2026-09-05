from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import s2and.model as model_module
from s2and.arrow_inputs import ArrowDataset
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer
from s2and.prediction_state import PredictionState
from tests.helpers import write_minimal_arrow_prediction_bundle


@pytest.mark.parametrize("case", ["clean", "dataset", "partial", "direct_soft", "direct_hard", "initial_only"])
def test_public_arrow_subblocking_preserves_effective_seed_disallows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    """Real native subblocking carries seed overrides into initial-only passes."""
    import pyarrow as pa

    from s2and.consts import LARGE_DISTANCE
    from s2and.incremental_linking.artifact import (
        load_incremental_linking_artifact,
        save_incremental_linking_artifact,
    )
    from s2and.incremental_linking.feature_block import write_arrow_batch_lookup_index, write_arrow_ipc_table
    from tests.helpers import write_test_arrow_artifact_manifest
    from tests.promoted_linking_helpers import build_tiny_promoted_booster, synthetic_pairwise_bundle_binding
    from tests.test_incremental_linking_artifact import _logistic_gate_config

    paths = write_minimal_arrow_prediction_bundle(tmp_path / "arrow", include_specter=True)
    with pa.OSFile(paths["signatures"], "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    firsts = table["author_first"].to_pylist()
    firsts[2] = "A"
    if case == "initial_only":
        firsts[1] = "A"
    table = table.set_column(table.schema.get_field_index("author_first"), "author_first", pa.array(firsts))
    write_arrow_ipc_table(table, Path(paths["signatures"]))
    write_arrow_batch_lookup_index(
        Path(paths["signatures"]),
        Path(paths["signatures_batch_index"]),
        key_column="signature_id",
        table_name="signatures",
    )
    write_test_arrow_artifact_manifest(tmp_path / "arrow", paths)
    monkeypatch.setattr("s2and.subblocking._resolved_orcid_prefix_counts", lambda counts: {})

    clusterer = Clusterer(FeaturizationInfo(["year_diff"]), classifier=None, n_jobs=1)
    booster, _ = build_tiny_promoted_booster()
    artifact_dir = tmp_path / "linker"
    save_incremental_linking_artifact(
        booster,
        artifact_dir,
        gate_config=_logistic_gate_config(),
        target_spec={},
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
    )
    clusterer.incremental_linker_artifact = load_incremental_linking_artifact(artifact_dir)
    seeds = {"0": "claimed", "1": "claimed", "2": "initial"}
    pair = ("0", "1")
    if case == "initial_only":
        seeds = {"0": "full", "1": "claimed", "2": "claimed"}
        pair = ("1", "2")
    disallows = {pair} if case in {"dataset", "initial_only"} else set()
    partial: dict[tuple[str, str], int | float] = {}
    if case == "partial":
        partial[pair] = LARGE_DISTANCE
    elif case in {"direct_soft", "direct_hard"}:
        partial[pair] = 0.2 if case == "direct_soft" else LARGE_DISTANCE
        partial[(pair[1], pair[0])] = LARGE_DISTANCE if case == "direct_soft" else 0.2
    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        outputs = [
            clusterer.predict_from_arrow(
                {"block": ["0", "1", "2"]},
                arrow_dataset,
                batching_threshold=threshold,
                cluster_seeds_require=seeds,
                cluster_seeds_disallow=disallows,
                partial_supervision=partial,
                name_tuples=frozenset(),
            )[0]
            for threshold in (None, 2)
        ]
    partitions = [{frozenset(members) for members in output.values()} for output in outputs]
    assert partitions[0] == partitions[1]
    assert any(set(pair) <= members for members in partitions[1]) == (case in {"clean", "direct_soft"})
    assert sorted(signature_id for members in outputs[1].values() for signature_id in members) == ["0", "1", "2"]


def _clusterer() -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
    )


def test_large_arrow_block_uses_native_subblocking_and_reuses_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cover the complete bounded Arrow large-block orchestration path."""
    prediction_state = PredictionState()

    write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    arrow_dataset = ArrowDataset.open(tmp_path)

    events: list[str] = []
    build_calls: list[dict[str, Any]] = []
    native_calls: list[dict[str, Any]] = []
    incremental_calls: list[dict[str, Any]] = []

    class FakeRustFeaturizer:
        def cluster_seeds_require(self) -> list[tuple[str, str]]:
            return [("0", "claimed"), ("1", "claimed")]

    def fake_native_subblocking(dataset: Any, signature_ids: list[str], **kwargs: Any):
        events.append("native_subblocking")
        native_calls.append({"dataset": dataset, "signature_ids": list(signature_ids), **kwargs})
        subblocks = {
            "initial": ["6", "7", "8"],
            "multi_a": ["0", "2", "5"],
            "multi_b": ["1", "3", "4"],
        }
        assert max(map(len, subblocks.values())) <= kwargs["maximum_size"]
        return subblocks, {"final_subblock_count": 3, "input_signature_count": 9}

    def fake_load_representatives(_dataset: Any, signature_ids: list[str]):
        events.append("representatives")
        assert signature_ids == ["6", "0", "5", "3", "9"]
        return {
            signature_id: SimpleNamespace(
                author_info_first="j" if signature_id == "6" else "john",
                author_info_first_normalized_without_apostrophe=("j" if signature_id == "6" else "john"),
            )
            for signature_id in signature_ids
        }

    def fake_build_featurizer(dataset: ArrowDataset, **kwargs: Any) -> FakeRustFeaturizer:
        events.append("featurizer")
        assert dataset is arrow_dataset
        build_calls.append(dict(kwargs))
        return FakeRustFeaturizer()

    def fake_predict_multiple(self: Clusterer, block_dict: dict[str, list[str]], **kwargs: Any):
        del self
        events.append("bulk_predict")
        assert list(block_dict) == [
            "large|subblock=multi_a|repair_part=0000",
            "large|subblock=multi_a|repair_part=0001",
            "large|subblock=multi_b",
            "small",
        ]
        assert block_dict["large|subblock=multi_a|repair_part=0000"] == ["0", "1", "2"]
        assert kwargs["dataset"].cluster_seeds_require == {"0": "claimed", "1": "claimed"}
        return {
            "claimed": ["0", "1"],
            "bulk_a": ["2"],
            "bulk_b": ["3", "4", "5", "9"],
        }

    def fake_predict_incremental(
        self: Clusterer,
        block_signatures: list[str],
        dataset: ArrowDataset,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del self
        events.append("incremental_attach")
        assert dataset is arrow_dataset
        incremental_calls.append({"block_signatures": list(block_signatures), **kwargs})
        assert kwargs["cluster_seeds_require"] == {
            "0": "claimed",
            "1": "claimed",
            "2": "bulk_a",
            "3": "bulk_b",
            "4": "bulk_b",
            "5": "bulk_b",
            "9": "bulk_b",
        }
        return {
            "clusters": {
                "claimed": ["0", "1"],
                "bulk_a": ["2", "6", "7", "8"],
                "bulk_b": ["3", "4", "5", "9"],
            }
        }

    monkeypatch.setattr(model_module, "_make_subblocks_with_telemetry_arrow_rust", fake_native_subblocking)
    monkeypatch.setattr(model_module, "_load_arrow_incremental_signature_info", fake_load_representatives)
    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_dataset", fake_build_featurizer)
    monkeypatch.setattr(Clusterer, "_predict_subblocked_multiple_letter_groups", fake_predict_multiple)
    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow", fake_predict_incremental)
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Python subblocking must not run")),
    )
    clusterer = _clusterer()
    predicted, dists = clusterer._predict_from_arrow_request(
        {"small": ["9"], "large": [str(index) for index in range(9)]},
        arrow_dataset,
        batching_threshold=3,
        name_tuples=set(),
        cluster_seeds_require={"0": "claimed", "1": "claimed"},
        prediction_state=prediction_state,
    )

    assert dists is None
    assert events == [
        "native_subblocking",
        "representatives",
        "featurizer",
        "bulk_predict",
        "incremental_attach",
    ]
    assert native_calls[0]["signature_ids"] == [str(index) for index in range(9)]
    assert native_calls[0]["maximum_size"] == 3
    assert len(build_calls) == 1
    assert build_calls[0]["signature_ids"] == ["0", "1", "2", "5", "3", "4", "9"]
    assert build_calls[0]["cluster_seeds_path"] is not None
    assert len(incremental_calls) == 1
    assert incremental_calls[0]["block_signatures"] == ["6", "7", "8"]
    assert {signature_id for members in predicted.values() for signature_id in members} == {
        str(index) for index in range(10)
    }

    subblocking_telemetry = prediction_state.telemetry["rust_arrow_subblocking"]
    assert subblocking_telemetry["enabled"] == 1
    assert subblocking_telemetry["maximum_size"] == 3
    assert subblocking_telemetry["input_block_count"] == 2
    assert subblocking_telemetry["oversized_block_count"] == 1
    assert subblocking_telemetry["blocks"]["large"]["final_subblock_count"] == 3
    assert subblocking_telemetry["blocks"]["large"]["seed_components_repacked"] == 1
    assert subblocking_telemetry["blocks"]["large"]["repaired_final_subblock_count"] == 4
    predict_telemetry = prediction_state.telemetry["arrow_predict"]
    assert predict_telemetry["signature_count"] == 10
    assert predict_telemetry["featurizer_signature_count"] == 7
    assert predict_telemetry["block_count"] == 2


def test_arrow_subblocking_avoids_generated_key_collisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_native_subblocking(_dataset: Any, signature_ids: list[str], **_kwargs: Any):
        assert signature_ids == ["s1", "s2"]
        return {"x": ["s1"], "y": ["s2"]}, {}

    monkeypatch.setattr(model_module, "_make_subblocks_with_telemetry_arrow_rust", fake_native_subblocking)
    write_minimal_arrow_prediction_bundle(tmp_path)
    arrow_dataset = ArrowDataset.open(tmp_path)

    result = _clusterer()._build_arrow_subblocked_block_dict(
        {
            "a": ["s1", "s2"],
            "a|subblock=x": ["other"],
        },
        arrow_dataset,
        batching_threshold=1,
        cluster_seeds_require={},
    )

    assert result == {
        "a|subblock=x|collision=0001": ["s1"],
        "a|subblock=y": ["s2"],
        "a|subblock=x": ["other"],
    }


def test_arrow_subblocking_presplits_altered_profiles_before_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    arrow_dataset = ArrowDataset.open(tmp_path)
    rewritten_seeds = {"s1": "claimed_0", "s2": "claimed_0", "s3": "claimed_1"}
    events: list[str] = []

    def fake_seed_setup(self: Clusterer, dataset: Any, *_args: Any, **_kwargs: Any):
        del self
        events.append("presplit")
        assert dataset.cluster_seeds_require == {"s1": "claimed", "s2": "claimed", "s3": "claimed"}
        assert dataset.altered_cluster_signatures == ["s1"]
        return rewritten_seeds, {"claimed_0": "claimed", "claimed_1": "claimed"}, {}, {}

    def fake_load_signature_info(_dataset: Any, signature_ids: list[str]):
        return {
            signature_id: SimpleNamespace(
                author_info_first="john",
                author_info_first_normalized_without_apostrophe="john",
                author_info_orcid=None,
            )
            for signature_id in signature_ids
        }

    def fake_predict_from_arrow(
        self: Clusterer,
        block_dict: dict[str, list[str]],
        dataset: ArrowDataset,
        _lease: Any,
        sidecars: Mapping[str, str],
        **kwargs: Any,
    ):
        del self
        events.append("prediction")
        assert dataset is arrow_dataset
        assert block_dict == {"block": ["s1", "s2", "s3"]}
        assert "cluster_seeds" in sidecars
        assert kwargs["prediction_cluster_seeds_require"] == rewritten_seeds
        assert kwargs["needs_subblocking"] is True
        return {"cluster": ["s1", "s2", "s3"]}, None

    monkeypatch.setattr(Clusterer, "_build_incremental_seed_setup", fake_seed_setup)
    monkeypatch.setattr(model_module, "_load_arrow_incremental_signature_info", fake_load_signature_info)
    monkeypatch.setattr(Clusterer, "_predict_from_arrow", fake_predict_from_arrow)

    result, _ = _clusterer().predict_from_arrow(
        {"block": ["s1", "s2", "s3"]},
        arrow_dataset,
        batching_threshold=2,
        name_tuples=set(),
        cluster_seeds_require={"s1": "claimed", "s2": "claimed", "s3": "claimed"},
        altered_cluster_signatures=["s1"],
    )

    assert events == ["presplit", "prediction"]
    assert result == {"cluster": ["s1", "s2", "s3"]}


def test_all_initial_arrow_subblocks_stay_on_bulk_path() -> None:
    block_dict = {
        "block|subblock=a": ["i0", "i1"],
        "block|subblock=b": ["i2"],
    }
    dataset = SimpleNamespace(
        signatures={
            signature_id: SimpleNamespace(
                author_info_first="J",
                author_info_first_normalized_without_apostrophe="j",
            )
            for signature_id in ("i0", "i1", "i2")
        }
    )

    bulk_blocks, incremental_blocks, alert_flag = _clusterer()._partition_subblocked_first_name_groups(
        block_dict,
        dataset,
    )

    assert bulk_blocks == block_dict
    assert incremental_blocks == {}
    assert alert_flag is True


def test_seeded_initial_only_arrow_subblock_uses_sequential_attachment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep seeded initial-only groups on the cross-subblock attachment path."""

    write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    arrow_dataset = ArrowDataset.open(tmp_path)
    bulk_calls: list[dict[str, list[str]]] = []
    incremental_calls: list[dict[str, Any]] = []

    def fake_native_subblocking(
        _dataset: Any,
        signature_ids: list[str],
        **_kwargs: Any,
    ) -> tuple[dict[str, list[str]], dict[str, int]]:
        assert signature_ids == ["0", "1", "2", "3", "4"]
        return {
            "initial": ["2", "3", "4"],
            "multiple": ["0", "1"],
        }, {"final_subblock_count": 2}

    def fake_load_representatives(
        _dataset: Any,
        signature_ids: list[str],
    ) -> dict[str, SimpleNamespace]:
        assert signature_ids == ["2", "0"]
        return {
            signature_id: SimpleNamespace(
                author_info_first="j" if signature_id == "2" else "john",
                author_info_first_normalized_without_apostrophe=("j" if signature_id == "2" else "john"),
            )
            for signature_id in signature_ids
        }

    def fake_build_featurizer(dataset: ArrowDataset, **kwargs: Any) -> object:
        assert dataset is arrow_dataset
        assert kwargs["signature_ids"] == ["0", "1"]
        return object()

    def fake_predict_multiple(
        self: Clusterer,
        block_dict: dict[str, list[str]],
        **_kwargs: Any,
    ) -> dict[str, list[str]]:
        del self
        bulk_calls.append(dict(block_dict))
        assert block_dict == {"large|subblock=multiple": ["0", "1"]}
        return {"established": ["0", "1"]}

    def fake_predict_incremental(
        self: Clusterer,
        block_signatures: list[str],
        dataset: ArrowDataset,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del self
        assert dataset is arrow_dataset
        incremental_calls.append({"block_signatures": list(block_signatures), **kwargs})
        assert block_signatures == ["2", "3", "4"]
        assert kwargs["cluster_seeds_require"] == {
            "0": "established",
            "1": "established",
            "2": "initial_component",
            "3": "initial_component",
        }
        return {"clusters": {"established": ["0", "1", "2", "3", "4"]}}

    monkeypatch.setattr(model_module, "_make_subblocks_with_telemetry_arrow_rust", fake_native_subblocking)
    monkeypatch.setattr(model_module, "_load_arrow_incremental_signature_info", fake_load_representatives)
    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_dataset", fake_build_featurizer)
    monkeypatch.setattr(Clusterer, "_predict_subblocked_multiple_letter_groups", fake_predict_multiple)
    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow", fake_predict_incremental)

    predicted, dists = _clusterer().predict_from_arrow(
        {"large": ["0", "1", "2", "3", "4"]},
        arrow_dataset,
        batching_threshold=3,
        cluster_seeds_require={"2": "initial_component", "3": "initial_component"},
        name_tuples=set(),
    )

    assert dists is None
    assert bulk_calls == [{"large|subblock=multiple": ["0", "1"]}]
    assert len(incremental_calls) == 1
    assert predicted == {"established": ["0", "1", "2", "3", "4"]}


def test_large_arrow_block_rejects_native_subblock_over_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    arrow_dataset = ArrowDataset.open(tmp_path)
    monkeypatch.setattr(
        model_module,
        "_make_subblocks_with_telemetry_arrow_rust",
        lambda *_args, **_kwargs: ({"too_large": ["0", "1", "2", "3"]}, {}),
    )

    with pytest.raises(RuntimeError, match="Rust Arrow subblocking exceeded batching_threshold"):
        _clusterer().predict_from_arrow(
            {"large": [str(index) for index in range(5)]},
            arrow_dataset,
            batching_threshold=3,
            name_tuples=set(),
        )


def test_large_arrow_block_rejects_oversized_explicit_seed_component_before_native_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    arrow_dataset = ArrowDataset.open(tmp_path)
    native_called = False
    featurizer_called = False

    def fail_native(*_args: Any, **_kwargs: Any):
        nonlocal native_called
        native_called = True
        raise AssertionError("native subblocking must not start")

    def fail_featurizer(*_args: Any, **_kwargs: Any):
        nonlocal featurizer_called
        featurizer_called = True
        raise AssertionError("featurizer construction must not start")

    monkeypatch.setattr(model_module, "_make_subblocks_with_telemetry_arrow_rust", fail_native)
    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_dataset", fail_featurizer)

    with pytest.raises(
        ValueError,
        match="cluster_seeds_require component exceeds batching_threshold before Arrow subblocking",
    ):
        _clusterer().predict_from_arrow(
            {"large": [str(index) for index in range(5)]},
            arrow_dataset,
            batching_threshold=3,
            cluster_seeds_require={str(index): "claimed" for index in range(4)},
            name_tuples=set(),
        )

    assert native_called is False
    assert featurizer_called is False


def test_large_arrow_block_rejects_invalid_native_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    arrow_dataset = ArrowDataset.open(tmp_path)
    monkeypatch.setattr(
        model_module,
        "build_rust_featurizer_from_arrow_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("featurizer construction must not start")),
    )
    cases = (
        ("missing", {"left": ["0", "1"], "right": ["2", "3"]}, "missing=['4']"),
        ("duplicate", {"left": ["0", "1", "2"], "right": ["2", "3", "4"]}, "duplicates=['2']"),
        ("unexpected", {"left": ["0", "1", "2"], "right": ["3", "foreign"]}, "unexpected=['foreign']"),
    )
    for case_id, native_subblocks, expected_detail in cases:
        monkeypatch.setattr(
            model_module,
            "_make_subblocks_with_telemetry_arrow_rust",
            lambda *_args, _subblocks=native_subblocks, **_kwargs: (_subblocks, {}),
        )
        with pytest.raises(
            RuntimeError,
            match="Rust Arrow subblocking must return every input signature exactly once",
        ) as exc_info:
            _clusterer().predict_from_arrow(
                {"large": [str(index) for index in range(5)]},
                arrow_dataset,
                batching_threshold=3,
                name_tuples=set(),
            )
        assert expected_detail in str(exc_info.value), case_id

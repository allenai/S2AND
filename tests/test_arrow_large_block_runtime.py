from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import s2and.model as model_module
import s2and.subblocking as subblocking_module
from s2and.arrow_inputs import ValidatedArrowInputs, validate_arrow_prediction_artifacts
from s2and.consts import NORMALIZATION_VERSION, PROJECT_ROOT_PATH
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.feature_block_arrow import write_cluster_seeds_arrow
from s2and.model import Clusterer
from scripts._rust_suite.promoted_incremental_arrow_profile_cmd import (
    _block_dict,
    _read_signature_rows,
    _select_workload,
)
from tests.helpers import write_minimal_arrow_prediction_bundle, write_test_arrow_artifact_manifest

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"


def _is_lfs_pointer(path: Path) -> bool:
    return path.is_file() and path.read_bytes()[: len(_LFS_POINTER_PREFIX)] == _LFS_POINTER_PREFIX


def _skip_if_missing_or_lfs_pointer(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise pytest.skip.Exception(f"missing local Arrow/prod artifact(s): {missing}")
    pointers = [str(path) for path in paths if _is_lfs_pointer(path)]
    if pointers:
        raise pytest.skip.Exception(f"Git LFS artifact(s) not materialized: {pointers}")


def _cluster_partition(clusters: Mapping[str, list[str]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(signature_ids) for signature_ids in clusters.values() if signature_ids)


def _resolve_manifest_path(dataset_root: Path, value: Any) -> str:
    raw_path = Path(str(value))
    candidates = [raw_path] if raw_path.is_absolute() else [dataset_root / raw_path, Path(PROJECT_ROOT_PATH) / raw_path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0])


def _arrow_prediction_paths(dataset_root: Path) -> ValidatedArrowInputs:
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_paths = manifest.get("paths")
    if not isinstance(manifest_paths, Mapping):
        raise ValueError(f"Arrow manifest is missing object paths: {manifest_path}")
    paths: dict[str, str] = {}
    for key in (
        "signatures",
        "papers",
        "paper_authors",
        "name_counts_index",
        "signatures_batch_index",
        "papers_batch_index",
        "paper_authors_batch_index",
    ):
        value = manifest_paths.get(key)
        if value is not None:
            paths[key] = _resolve_manifest_path(dataset_root, value)

    specter_value = manifest_paths.get("specter", manifest_paths.get("specter2"))
    specter_index_value = manifest_paths.get("specter_batch_index", manifest_paths.get("specter2_batch_index"))
    if specter_value is not None:
        paths["specter"] = _resolve_manifest_path(dataset_root, specter_value)
    if specter_index_value is not None:
        paths["specter_batch_index"] = _resolve_manifest_path(dataset_root, specter_index_value)
    return validate_arrow_prediction_artifacts(
        paths,
        require_specter=True,
        require_name_counts_index=True,
        expected_normalization_version=NORMALIZATION_VERSION,
        context="canonical PubMed large-block Arrow runtime integration test",
        producer_hint="publish the canonical PubMed Arrow bundle and v1.3 production model",
    )


def _clusterer() -> Clusterer:
    return Clusterer(
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        classifier=None,
        cluster_model=None,
        n_jobs=1,
    )


@pytest.mark.requires_lfs
def test_canonical_pubmed_large_block_arrow_subblocking_and_incremental_no_anddata_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise native large-block and incremental paths against real release artifacts."""

    pytest.importorskip("s2and_rust")
    from s2and.production_model import load_production_model

    dataset_root = Path("s2and/data/s2and_and_big_blocks_linker_dataset_20260525/datasets/pubmed")
    model_root = Path("s2and/data/production_model_v1.3")
    _skip_if_missing_or_lfs_pointer(
        [
            dataset_root / "manifest.json",
            model_root / "manifest.json",
            model_root / "clusterer.json",
            model_root / "pairwise/main.lgb",
            model_root / "pairwise/nameless.lgb",
            model_root / "incremental_linker/booster.lgb",
            model_root / "incremental_linker/metadata.json",
        ]
    )

    arrow_paths = _arrow_prediction_paths(dataset_root)
    rows = _read_signature_rows(Path(arrow_paths["signatures"]))
    blocks = _block_dict(rows)
    target_block = "r agarwal"
    block_signature_ids = blocks[target_block]
    seed_signature_to_cluster = {
        signature_id: f"seed_component_{index}" for index, signature_id in enumerate(block_signature_ids[:20])
    }
    workload = _select_workload(
        blocks=blocks,
        signature_to_cluster_id=seed_signature_to_cluster,
        target_block=target_block,
        query_limit=2,
        max_seed_clusters=2,
    )

    monkeypatch.setattr(
        subblocking_module,
        "cluster_with_specter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Python SPECTER subblocking ran")),
    )

    clusterer = load_production_model(str(model_root))
    clusterer.n_jobs = 1
    pred_clusters, dists = clusterer.predict_from_arrow_paths(
        {workload.target_block: block_signature_ids},
        arrow_paths,
        batching_threshold=64,
        cluster_seeds_require=workload.seed_signature_to_cluster,
        total_ram_bytes=1_000_000_000_000,
    )

    assert dists is None
    predicted_signature_ids = [signature_id for members in pred_clusters.values() for signature_id in members]
    assert len(predicted_signature_ids) == len(set(predicted_signature_ids))
    assert set(predicted_signature_ids) == set(block_signature_ids)
    subblocking_telemetry = cast(dict[str, Any], clusterer._last_rust_arrow_subblocking_telemetry)
    assert subblocking_telemetry["enabled"] == 1
    assert subblocking_telemetry["oversized_block_count"] == 1
    block_telemetry = cast(dict[str, Any], subblocking_telemetry["blocks"][workload.target_block])
    assert block_telemetry["input_signature_count"] == len(block_signature_ids)
    assert block_telemetry["graph_fallback_native"] is True

    explicit_result = clusterer.predict_incremental_from_arrow_paths(
        workload.block_signatures,
        arrow_paths,
        prevent_new_incompatibilities=False,
        batching_threshold=1,
        cluster_seeds_require=workload.seed_signature_to_cluster,
        total_ram_bytes=1_000_000_000_000,
    )
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, workload.seed_signature_to_cluster)
    sidecar_result = clusterer.predict_incremental_from_arrow_paths(
        workload.block_signatures,
        {**arrow_paths, "cluster_seeds": str(cluster_seeds_path)},
        prevent_new_incompatibilities=False,
        batching_threshold=1,
        total_ram_bytes=1_000_000_000_000,
    )

    assert _cluster_partition(cast(dict[str, list[str]], explicit_result["clusters"])) == _cluster_partition(
        cast(dict[str, list[str]], sidecar_result["clusters"])
    )
    for result in (explicit_result, sidecar_result):
        telemetry = cast(dict[str, Any], result["incremental_linker_telemetry"])
        assert result["incremental_linker_query_view"] == "raw_arrow"
        assert telemetry["arrow_promoted_incremental"] == 1
        assert result["clusters"]


def test_large_arrow_block_uses_native_subblocking_and_reuses_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cover the complete bounded Arrow large-block orchestration path."""

    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    cluster_seeds_path = tmp_path / "cluster_seeds.arrow"
    write_cluster_seeds_arrow(cluster_seeds_path, {"0": "claimed", "1": "claimed"})
    arrow_paths["cluster_seeds"] = str(cluster_seeds_path)
    write_test_arrow_artifact_manifest(tmp_path, arrow_paths)

    events: list[str] = []
    build_calls: list[dict[str, Any]] = []
    native_calls: list[dict[str, Any]] = []
    incremental_calls: list[dict[str, Any]] = []

    class FakeRustFeaturizer:
        def cluster_seeds_require(self) -> list[tuple[str, str]]:
            return [("0", "claimed"), ("1", "claimed")]

    def fake_native_subblocking(paths: dict[str, str], signature_ids: list[str], **kwargs: Any):
        events.append("native_subblocking")
        native_calls.append({"paths": dict(paths), "signature_ids": list(signature_ids), **kwargs})
        subblocks = {
            "initial": ["6", "7", "8"],
            "multi_a": ["0", "2", "5"],
            "multi_b": ["1", "3", "4"],
        }
        assert max(map(len, subblocks.values())) <= kwargs["maximum_size"]
        return subblocks, {"final_subblock_count": 3, "input_signature_count": 9}

    def fake_load_representatives(_paths: dict[str, str], signature_ids: list[str]):
        events.append("representatives")
        assert signature_ids == ["6", "0", "5", "3", "9"]
        return {
            signature_id: SimpleNamespace(
                author_info_first="j" if signature_id == "6" else "john",
                author_info_first_normalized_without_apostrophe=("j" if signature_id == "6" else "john"),
            )
            for signature_id in signature_ids
        }

    def fake_build_featurizer(paths: dict[str, str], **kwargs: Any) -> FakeRustFeaturizer:
        events.append("featurizer")
        build_calls.append({"paths": dict(paths), **kwargs})
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
        paths: dict[str, str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del self
        events.append("incremental_attach")
        incremental_calls.append({"block_signatures": list(block_signatures), "paths": dict(paths), **kwargs})
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
    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_featurizer)
    monkeypatch.setattr(Clusterer, "_predict_subblocked_multiple_letter_groups", fake_predict_multiple)
    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow_paths", fake_predict_incremental)
    monkeypatch.setattr(
        model_module,
        "make_subblocks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Python subblocking must not run")),
    )
    clusterer = _clusterer()
    predicted, dists = clusterer.predict_from_arrow_paths(
        {"small": ["9"], "large": [str(index) for index in range(9)]},
        arrow_paths,
        batching_threshold=3,
        name_tuples=set(),
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
    assert build_calls[0]["paths"]["cluster_seeds"] == str(cluster_seeds_path)
    assert len(incremental_calls) == 1
    assert incremental_calls[0]["block_signatures"] == ["6", "7", "8"]
    assert incremental_calls[0]["paths"]["cluster_seeds"] == str(cluster_seeds_path)
    assert {signature_id for members in predicted.values() for signature_id in members} == {
        str(index) for index in range(10)
    }

    subblocking_telemetry = clusterer._last_rust_arrow_subblocking_telemetry
    assert subblocking_telemetry["enabled"] == 1
    assert subblocking_telemetry["maximum_size"] == 3
    assert subblocking_telemetry["input_block_count"] == 2
    assert subblocking_telemetry["oversized_block_count"] == 1
    assert subblocking_telemetry["blocks"]["large"]["final_subblock_count"] == 3
    assert subblocking_telemetry["blocks"]["large"]["seed_components_repacked"] == 1
    assert subblocking_telemetry["blocks"]["large"]["repaired_final_subblock_count"] == 4
    predict_telemetry = clusterer._last_arrow_predict_telemetry
    assert predict_telemetry["signature_count"] == 10
    assert predict_telemetry["featurizer_signature_count"] == 7
    assert predict_telemetry["block_count"] == 2


def test_arrow_subblocking_avoids_generated_key_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_native_subblocking(_paths: Mapping[str, Any], signature_ids: list[str], **_kwargs: Any):
        assert signature_ids == ["s1", "s2"]
        return {"x": ["s1"], "y": ["s2"]}, {}

    monkeypatch.setattr(model_module, "_make_subblocks_with_telemetry_arrow_rust", fake_native_subblocking)

    result = _clusterer()._build_arrow_subblocked_block_dict(
        {
            "a": ["s1", "s2"],
            "a|subblock=x": ["other"],
        },
        {},
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
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    rewritten_seeds = {"s1": "claimed_0", "s2": "claimed_0", "s3": "claimed_1"}
    events: list[str] = []

    def fake_seed_setup(self: Clusterer, dataset: Any, *_args: Any, **_kwargs: Any):
        del self
        events.append("presplit")
        assert dataset.cluster_seeds_require == {"s1": "claimed", "s2": "claimed", "s3": "claimed"}
        assert dataset.altered_cluster_signatures == ["s1"]
        return rewritten_seeds, {"claimed_0": "claimed", "claimed_1": "claimed"}, {}, {}

    def fake_load_signature_info(_paths: Mapping[str, Any], signature_ids: list[str]):
        return {
            signature_id: SimpleNamespace(
                author_info_first="john",
                author_info_first_normalized_without_apostrophe="john",
                author_info_orcid=None,
            )
            for signature_id in signature_ids
        }

    def fake_predict_validated(
        self: Clusterer,
        block_dict: dict[str, list[str]],
        request_paths: ValidatedArrowInputs,
        **kwargs: Any,
    ):
        del self
        events.append("prediction")
        assert block_dict == {"block": ["s1", "s2", "s3"]}
        assert model_module._read_cluster_seeds_arrow(Path(request_paths["cluster_seeds"])) == rewritten_seeds
        assert kwargs["prediction_cluster_seeds_require"] == rewritten_seeds
        assert kwargs["needs_subblocking"] is True
        return {"cluster": ["s1", "s2", "s3"]}, None

    monkeypatch.setattr(Clusterer, "_build_incremental_seed_setup", fake_seed_setup)
    monkeypatch.setattr(model_module, "_load_arrow_incremental_signature_info", fake_load_signature_info)
    monkeypatch.setattr(Clusterer, "_predict_from_validated_arrow_paths", fake_predict_validated)

    result, _ = _clusterer().predict_from_arrow_paths(
        {"block": ["s1", "s2", "s3"]},
        arrow_paths,
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

    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    bulk_calls: list[dict[str, list[str]]] = []
    incremental_calls: list[dict[str, Any]] = []

    def fake_native_subblocking(
        _paths: Mapping[str, Any],
        signature_ids: list[str],
        **_kwargs: Any,
    ) -> tuple[dict[str, list[str]], dict[str, int]]:
        assert signature_ids == ["0", "1", "2", "3", "4"]
        return {
            "initial": ["2", "3", "4"],
            "multiple": ["0", "1"],
        }, {"final_subblock_count": 2}

    def fake_load_representatives(
        _paths: Mapping[str, Any],
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

    def fake_build_featurizer(_paths: Mapping[str, Any], **kwargs: Any) -> object:
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
        _paths: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del self
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
    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fake_build_featurizer)
    monkeypatch.setattr(Clusterer, "_predict_subblocked_multiple_letter_groups", fake_predict_multiple)
    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow_paths", fake_predict_incremental)

    predicted, dists = _clusterer().predict_from_arrow_paths(
        {"large": ["0", "1", "2", "3", "4"]},
        arrow_paths,
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
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    monkeypatch.setattr(
        model_module,
        "_make_subblocks_with_telemetry_arrow_rust",
        lambda *_args, **_kwargs: ({"too_large": ["0", "1", "2", "3"]}, {}),
    )

    with pytest.raises(RuntimeError, match="Rust Arrow subblocking exceeded batching_threshold"):
        _clusterer().predict_from_arrow_paths(
            {"large": [str(index) for index in range(5)]},
            arrow_paths,
            batching_threshold=3,
            name_tuples=set(),
        )


def test_large_arrow_block_rejects_oversized_explicit_seed_component_before_native_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
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
    monkeypatch.setattr(model_module, "build_rust_featurizer_from_arrow_paths", fail_featurizer)

    with pytest.raises(
        ValueError,
        match="cluster_seeds_require component exceeds batching_threshold before Arrow subblocking",
    ):
        _clusterer().predict_from_arrow_paths(
            {"large": [str(index) for index in range(5)]},
            arrow_paths,
            batching_threshold=3,
            cluster_seeds_require={str(index): "claimed" for index in range(4)},
            name_tuples=set(),
        )

    assert native_called is False
    assert featurizer_called is False


@pytest.mark.parametrize(
    ("native_subblocks", "expected_detail"),
    [
        ({"left": ["0", "1"], "right": ["2", "3"]}, "missing=\\['4'\\]"),
        ({"left": ["0", "1", "2"], "right": ["2", "3", "4"]}, "duplicates=\\['2'\\]"),
        ({"left": ["0", "1", "2"], "right": ["3", "foreign"]}, "unexpected=\\['foreign'\\]"),
    ],
)
def test_large_arrow_block_rejects_invalid_native_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    native_subblocks: dict[str, list[str]],
    expected_detail: str,
) -> None:
    arrow_paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    monkeypatch.setattr(
        model_module,
        "_make_subblocks_with_telemetry_arrow_rust",
        lambda *_args, **_kwargs: (native_subblocks, {}),
    )
    monkeypatch.setattr(
        model_module,
        "build_rust_featurizer_from_arrow_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("featurizer construction must not start")),
    )

    with pytest.raises(
        RuntimeError,
        match=rf"Rust Arrow subblocking must return every input signature exactly once.*{expected_detail}",
    ):
        _clusterer().predict_from_arrow_paths(
            {"large": [str(index) for index in range(5)]},
            arrow_paths,
            batching_threshold=3,
            name_tuples=set(),
        )

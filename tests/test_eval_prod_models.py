from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import scripts.eval_prod_models as eval_prod_models
from s2and.arrow_inputs import (
    ARROW_COLLECTION_KIND,
    ArrowDataset,
    build_arrow_artifact_manifest,
    write_arrow_artifact_manifest,
)
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from s2and.incremental_linking.feature_block import write_name_counts_index
from s2and.production_training_contract import block_membership_sha256, frozen_test_blocks
from tests.helpers import tiny_name_counts_tuple, write_minimal_arrow_prediction_bundle


def _open_minimal_arrow_dataset(
    root: Path,
    *,
    clusters: Mapping[str, Sequence[str]] | None = None,
) -> ArrowDataset:
    write_minimal_arrow_prediction_bundle(root)
    if clusters is not None:
        (root / f"{root.name}_clusters.json").write_text(
            json.dumps(
                {
                    str(cluster_id): {"signature_ids": [str(signature_id) for signature_id in signature_ids]}
                    for cluster_id, signature_ids in clusters.items()
                }
            ),
            encoding="utf-8",
        )
    return ArrowDataset.open(root)


def _write_eval_arrow_root(root: Path, *, include_name_counts_index: bool = True) -> Path:
    dataset_root = root / "aliased-location"
    paths = write_minimal_arrow_prediction_bundle(dataset_root, include_specter=True)
    specter2 = dataset_root / "specter2.arrow"
    Path(paths["specter"]).rename(specter2)
    paths["specter"] = str(specter2)
    if include_name_counts_index:
        paths["name_counts_index"], _metrics = write_name_counts_index(root, tiny_name_counts_tuple())
    manifest_path = write_arrow_artifact_manifest(build_arrow_artifact_manifest(paths, dataset_root), dataset_root)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": ARROW_COLLECTION_KIND,
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "dataset_manifests": {
                    "pubmed": {
                        "path": "aliased-location/manifest.json",
                        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset_root


def test_arrow_availability_reads_only_manifests_and_opens_once_for_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = _write_eval_arrow_root(tmp_path)
    opened: list[Path] = []
    real_open = ArrowDataset.open

    def track_open(root, **kwargs):
        opened.append(Path(root))
        return real_open(root, **kwargs)

    monkeypatch.setattr(ArrowDataset, "open", track_open)
    assert eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter2.pkl"]) is None
    assert opened == []
    with eval_prod_models.resolve_arrow_dataset(str(tmp_path), "pubmed", "_specter2.pkl") as dataset:
        assert dataset.root == dataset_root.resolve()
    assert opened == [dataset_root.resolve()]


def test_arrow_availability_requires_name_counts_declaration(tmp_path: Path) -> None:
    _write_eval_arrow_root(tmp_path, include_name_counts_index=False)
    assert eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter2.pkl"]) is not None
    error = eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter2.pkl"])
    assert isinstance(error, FileNotFoundError)
    assert "name_counts_index" in str(error)
    with pytest.raises(FileNotFoundError, match="name_counts_index"):
        eval_prod_models.resolve_arrow_dataset(str(tmp_path), "pubmed", "_specter2.pkl")


def test_arrow_availability_uses_declared_specter_name_on_case_insensitive_filesystem(tmp_path: Path) -> None:
    dataset_root = _write_eval_arrow_root(tmp_path)
    declared_path = dataset_root / "specter2.arrow"
    renamed_path = dataset_root / "SPECTER2.ARROW"
    intermediate_path = declared_path.rename(dataset_root / "rename-intermediate.arrow")
    intermediate_path.rename(renamed_path)
    if not declared_path.is_file():
        pytest.skip("The filesystem distinguishes the declared and physical filename case")

    with ArrowDataset.open(dataset_root, require_specter=True, require_name_counts_index=True):
        pass
    assert eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter2.pkl"]) is None
    with eval_prod_models.resolve_arrow_dataset(str(tmp_path), "pubmed", "_specter2.pkl") as dataset:
        assert dataset.root == dataset_root.resolve()


def test_arrow_availability_accepts_utf8_bom_dataset_manifest(tmp_path: Path) -> None:
    dataset_root = _write_eval_arrow_root(tmp_path)
    manifest_path = dataset_root / "manifest.json"
    manifest_path.write_bytes(b"\xef\xbb\xbf" + manifest_path.read_bytes())
    collection_path = tmp_path / "manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    collection["dataset_manifests"]["pubmed"]["sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    with ArrowDataset.open(dataset_root, require_specter=True, require_name_counts_index=True):
        pass
    assert eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter2.pkl"]) is None
    with eval_prod_models.resolve_arrow_dataset(str(tmp_path), "pubmed", "_specter2.pkl") as dataset:
        assert dataset.root == dataset_root.resolve()


@pytest.mark.parametrize(
    "artifact,corruption",
    [("papers", "missing"), ("name_counts_index", "checksum"), ("name_counts_index", "missing")],
)
def test_available_arrow_payload_corruption_fails_the_real_open(tmp_path: Path, corruption: str, artifact: str) -> None:
    dataset_root = _write_eval_arrow_root(tmp_path)
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    payload = dataset_root / manifest["paths"][artifact]
    if artifact == "name_counts_index":
        payload /= "manifest.json"
    if corruption == "checksum":
        content = payload.read_bytes()
        payload.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    else:
        payload.unlink()
    assert eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter2.pkl"]) is None
    with pytest.raises(
        ValueError if corruption == "checksum" else FileNotFoundError,
        match="checksum mismatch" if corruption == "checksum" else None,
    ):
        eval_prod_models.resolve_arrow_dataset(str(tmp_path), "pubmed", "_specter2.pkl")


def test_arrow_availability_rejects_missing_base_artifact(tmp_path: Path) -> None:
    dataset_root = _write_eval_arrow_root(tmp_path)
    root_manifest = tmp_path / "manifest.json"
    child_manifest = dataset_root / "manifest.json"
    child = json.loads(child_manifest.read_text(encoding="utf-8"))
    del child["paths"]["papers"]
    del child["files"]["papers"]
    child_manifest.write_text(json.dumps(child), encoding="utf-8")
    collection = json.loads(root_manifest.read_text(encoding="utf-8"))
    collection["dataset_manifests"]["pubmed"]["sha256"] = hashlib.sha256(child_manifest.read_bytes()).hexdigest()
    root_manifest.write_text(json.dumps(collection), encoding="utf-8")
    with pytest.raises(ValueError):
        eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter2.pkl"])


def test_arrow_availability_distinguishes_absence_and_embedding_generation(tmp_path: Path) -> None:
    assert eval_prod_models.first_missing_arrow_dataset_error(None, ["pubmed"], ["_specter2.pkl"]) is not None
    assert eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter2.pkl"]) is not None
    _write_eval_arrow_root(tmp_path)
    assert eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["absent"], ["_specter2.pkl"]) is not None
    assert (
        eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter.pickle"]) is not None
    )
    assert eval_prod_models.first_missing_arrow_dataset_error(str(tmp_path), ["pubmed"], ["_specter2.pkl"]) is None


def test_train_mode_validation_rejects_conflicting_request_or_wrong_dataset() -> None:
    assert eval_prod_models._resolve_requested_train_modes(None, compare_train_modes=True) == [
        "anddata-python",
        "arrow-rust",
    ]
    with pytest.raises(ValueError, match="either --compare-train-modes or --train-modes"):
        eval_prod_models._resolve_requested_train_modes(["arrow-rust"], compare_train_modes=True)
    with pytest.raises(ValueError, match="qian-only"):
        eval_prod_models._validate_train_mode_scope(["arrow-rust"], ["pubmed"])


def test_training_mode_metric_assertion_accepts_identical_and_rejects_different_metrics() -> None:
    results = {
        ("_specter2.pkl", "anddata-python"): [{"B3 (P, R, F1)": (0.1, 0.2, 0.3)}],
        ("_specter2.pkl", "arrow-rust"): [{"B3 (P, R, F1)": (0.1, 0.2, 0.3)}],
    }

    eval_prod_models._assert_training_mode_metrics_identical(
        results,
        specter_suffixes_to_check=["_specter2.pkl"],
        train_modes=["anddata-python", "arrow-rust"],
        datasets=["qian"],
    )
    results[("_specter2.pkl", "arrow-rust")][0]["B3 (P, R, F1)"] = (0.1, 0.2, 0.4)

    with pytest.raises(AssertionError, match="Training mode metrics differ"):
        eval_prod_models._assert_training_mode_metrics_identical(
            results,
            specter_suffixes_to_check=["_specter2.pkl"],
            train_modes=["anddata-python", "arrow-rust"],
            datasets=["qian"],
        )


def test_build_pairwise_clusterer_can_disable_hyperopt(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    import s2and.model as model_module

    captured_pairwise_search_spaces: list[dict[str, Any] | None] = []
    captured_cluster_kwargs: dict[str, Any] = {}

    class FakePairwiseModeler:
        def __init__(self, **kwargs: Any) -> None:
            captured_pairwise_search_spaces.append(cast(dict[str, Any] | None, kwargs["search_space"]))
            self.classifier = None

        def fit(self, *_args: Any) -> None:
            self.classifier = object()

    class FakeClusterer:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            captured_cluster_kwargs.update(kwargs)

    monkeypatch.setattr(model_module, "PairwiseModeler", FakePairwiseModeler)
    monkeypatch.setattr(model_module, "Clusterer", FakeClusterer)
    info = SimpleNamespace(lightgbm_monotone_constraints=None)
    train = (np.zeros((2, 3)), np.array([0, 1]), np.zeros((2, 2)))
    val = (np.zeros((2, 3)), np.array([0, 1]), np.zeros((2, 2)))

    eval_prod_models.build_pairwise_clusterer_from_features(
        train,
        val,
        featurization_info=info,
        nameless_featurization_info=info,
        n_jobs=1,
        random_seed=42,
        pairwise_n_iter=25,
        cluster_n_iter=25,
        fixed_lightgbm_params=True,
        fixed_cluster_eps=0.5,
    )

    assert captured_pairwise_search_spaces == [{}, {}]
    assert captured_cluster_kwargs["search_space"] == {}
    assert captured_cluster_kwargs["cluster_model"].eps == 0.5


def test_read_arrow_s2_blocks_reads_from_open_dataset(tmp_path: Path) -> None:
    with _open_minimal_arrow_dataset(tmp_path) as arrow_dataset:
        assert arrow_dataset.root == tmp_path.resolve()
        assert eval_prod_models.read_arrow_s2_blocks(arrow_dataset) == {
            "a lovelace": [*[str(index) for index in range(10)], "q1", "q2", "seed1"]
        }


def test_pair_splits_from_arrow_dataset_samples_within_block_random_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eval_prod_models, "read_arrow_s2_blocks", lambda _path: {"block": ["s1", "s2", "s3"]})
    monkeypatch.setattr(
        eval_prod_models,
        "split_blocks_like_anddata",
        lambda blocks, *, random_seed: (dict(blocks), {}, {}),
    )

    with _open_minimal_arrow_dataset(tmp_path, clusters={"c1": ["s1", "s2"], "c2": ["s3"]}) as arrow_dataset:
        splits = eval_prod_models.pair_splits_from_arrow_dataset(
            arrow_dataset,
            random_seed=42,
            train_pairs_size=10,
            val_pairs_size=10,
            test_pairs_size=10,
        )

    assert set(splits.train_pairs) == {
        ("s1", "s2", 1),
        ("s1", "s3", 0),
        ("s2", "s3", 0),
    }
    assert splits.val_pairs == []
    assert splits.test_pairs == []


@pytest.mark.parametrize("n_jobs", [3, -1])
def test_arrow_feature_splits_reuse_open_dataset_and_project_one_union_call(tmp_path, monkeypatch, n_jobs) -> None:
    from s2and import feature_port, thread_config

    monkeypatch.setattr(thread_config.os, "cpu_count", lambda: 3)

    calls: list[dict[str, Any]] = []

    class FakeRustFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["s1", "s2", "s3"]

        def featurize_pairs_matrix_indexed(
            self,
            indexed_pairs: list[tuple[int, int]],
            selected_indices: list[int],
            n_jobs: int,
            nan_value: float,
        ) -> list[list[float]]:
            calls.append(
                {
                    "indexed_pairs": indexed_pairs,
                    "selected_indices": selected_indices,
                    "n_jobs": n_jobs,
                    "nan_value": nan_value,
                }
            )
            return [
                [float("nan") if index == 4 else float(row * 100 + index) for index in selected_indices]
                for row in range(len(indexed_pairs))
            ]

    main_info = SimpleNamespace(selected_feature_indices=lambda: [1, 3, 4])
    nameless_info = SimpleNamespace(selected_feature_indices=lambda: [2, 4])

    def build(dataset, **kwargs):
        assert dataset is arrow_dataset
        assert kwargs == {"name_tuples": None, "num_threads": n_jobs}
        return FakeRustFeaturizer()

    monkeypatch.setattr(feature_port, "build_rust_featurizer_from_arrow_dataset", build)
    pairs = [("s3", "s1", 0.25), ("s1", "s2", 1)]
    splits = eval_prod_models.PairwiseTrainingSplits(pairs, [], [], {}, {}, {}, {})
    with _open_minimal_arrow_dataset(tmp_path) as arrow_dataset:
        train, val, test, _ = eval_prod_models.arrow_training_feature_splits(
            arrow_dataset,
            splits,
            featurizer_info=main_info,
            nameless_featurizer_info=nameless_info,
            n_jobs=n_jobs,
            nan_value=-7.5,
        )
    features, labels, nameless = train
    assert val[0].shape == test[0].shape == (0, 3)

    # One union call; NaN is requested from Rust and nan_value is applied afterwards,
    # as many_pairs_featurize does.
    assert len(calls) == 1
    assert math.isnan(calls[0].pop("nan_value"))
    assert calls == [{"indexed_pairs": [(2, 0), (0, 1)], "selected_indices": [1, 2, 3, 4], "n_jobs": 3}]
    assert features.tolist() == [[1.0, 3.0, -7.5], [101.0, 103.0, -7.5]]
    assert labels.tolist() == [0.25, 1.0]
    assert nameless is not None
    assert nameless.tolist() == [[2.0, -7.5], [102.0, -7.5]]


def test_split_blocks_like_anddata_matches_anddata_and_rejects_tiny_inputs() -> None:
    for block_count in (1, 2, 4):
        blocks = {f"b{index}": [f"s{index}"] for index in range(block_count)}
        with pytest.raises(ValueError):
            eval_prod_models.split_blocks_like_anddata(blocks, random_seed=1)

    from s2and.data import ANDData

    blocks = {
        f"block {index}": [f"s{index}_{position}" for position in range((index * 7) % 13 + 1)] for index in range(40)
    }
    for seed in (42, 1111):
        fake_anddata = SimpleNamespace(
            num_clusters_for_block_size=1,
            random_seed=seed,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        expected = ANDData.split_blocks_helper(cast(Any, fake_anddata), blocks)
        actual = eval_prod_models.split_blocks_like_anddata(blocks, random_seed=seed)

        assert actual == expected, f"seed-{seed}"


def _write_bundle_training_config(bundle_dir: Path, payload: Mapping[str, Any]) -> None:
    reproducibility_dir = bundle_dir / "reproducibility"
    reproducibility_dir.mkdir(parents=True, exist_ok=True)
    (reproducibility_dir / "pairwise_training_config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_recorded_holdout_survives_arrow_order_and_rejects_population_drift() -> None:
    blocks = {f"block {index}": [str(2 * index), str(2 * index + 1)] for index in range(40)}
    train, _, test = eval_prod_models.split_blocks_like_anddata(blocks, random_seed=1111)
    reordered = {key: list(reversed(blocks[key])) for key in sorted(blocks, key=lambda key: blocks[key][0])}
    _, _, wrong_test = eval_prod_models.split_blocks_like_anddata(reordered, random_seed=1111)
    assert set(wrong_test) & set(train), "Fixture must reproduce the seed-only holdout leak"
    record = {"block_membership_sha256": block_membership_sha256(blocks), "test_block_ids": list(test)}
    actual = frozen_test_blocks(reordered, record)
    assert set(actual) == set(test)
    assert not set(actual) & set(train)
    assert all(actual[key] is reordered[key] for key in actual)
    reordered["block 0"] = ["replaced", "1"]
    with pytest.raises(ValueError, match="membership differs"):
        frozen_test_blocks(reordered, record)


@pytest.mark.parametrize("ids", [[], ["b", "b"], [1], "b"])
def test_recorded_holdout_rejects_invalid_ids(ids: Any) -> None:
    blocks = {"b": ["s"]}
    with pytest.raises(ValueError, match="unique test_block_ids"):
        frozen_test_blocks(blocks, {"block_membership_sha256": block_membership_sha256(blocks), "test_block_ids": ids})


def test_bundle_holdout_requires_recorded_identities(tmp_path: Path) -> None:
    _write_bundle_training_config(tmp_path, {"data_random_seed": 1111})
    with pytest.raises(ValueError, match="no recorded cluster test splits"):
        eval_prod_models.bundle_cluster_test_splits(tmp_path)


def test_bundle_holdout_rejects_null_record_instead_of_falling_back_to_seed(tmp_path: Path) -> None:
    _write_bundle_training_config(tmp_path, {"data_random_seed": 1111})
    (tmp_path / "reproducibility" / "pairwise_training_summary.json").write_text(
        json.dumps({"cluster_test_splits": {"pubmed": None}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity records"):
        eval_prod_models.bundle_cluster_test_splits(tmp_path)
    (tmp_path / "reproducibility" / "pairwise_training_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="no recorded cluster test splits"):
        eval_prod_models.bundle_cluster_test_splits(tmp_path)


def test_arrow_eval_uses_recorded_holdout_without_resplitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = {"training": ["s0"], "heldout": ["s1"]}
    record = {"block_membership_sha256": block_membership_sha256(blocks), "test_block_ids": ["heldout"]}
    monkeypatch.setattr(eval_prod_models, "read_arrow_s2_blocks", lambda _dataset: blocks)
    monkeypatch.setattr(
        eval_prod_models,
        "split_blocks_like_anddata",
        lambda *_a, **_k: pytest.fail("Must not recreate split from seed"),
    )

    class Clusterer:
        def predict_from_arrow(self, actual_blocks, _dataset, **_kwargs):
            assert actual_blocks == {"heldout": ["s1"]}
            return {"pred": ["s1"]}, None

    with _open_minimal_arrow_dataset(tmp_path, clusters={"truth": ["s1"]}) as dataset:
        metrics, _ = eval_prod_models.cluster_eval_arrow(
            dataset, Clusterer(), random_seed=42, n_jobs=1, test_split=record
        )
    assert metrics["B3 (P, R, F1)"] == (1.0, 1.0, 1.0)


def test_bundle_data_random_seed_requires_a_recorded_integer(tmp_path: Path) -> None:
    _write_bundle_training_config(tmp_path, {"data_random_seed": 1111})
    assert eval_prod_models.bundle_data_random_seed(tmp_path) == 1111

    missing_root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="records no training split seed"):
        eval_prod_models.bundle_data_random_seed(missing_root)

    for case_id, bad_seed in (
        ("none", None),
        ("string", "1111"),
        ("boolean", True),
        ("float", 1111.0),
    ):
        invalid_root = tmp_path / case_id
        _write_bundle_training_config(invalid_root, {"data_random_seed": bad_seed})
        with pytest.raises(ValueError, match="integer data_random_seed"):
            eval_prod_models.bundle_data_random_seed(invalid_root)


def test_resolve_arrow_dataset_root_requires_explicit_collection_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    (dataset_root / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Arrow root manifest does not exist"):
        eval_prod_models.resolve_arrow_dataset_root(str(tmp_path / "arrow"), "dummy")


def test_resolve_arrow_dataset_root_does_not_bypass_its_manifest(tmp_path: Path) -> None:
    root = tmp_path / "arrow"
    declared = root / "declared"
    undeclared = root / "undeclared"
    declared.mkdir(parents=True)
    undeclared.mkdir()
    declared_manifest = declared / "manifest.json"
    declared_manifest.write_text("{}\n", encoding="utf-8")
    (undeclared / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": ARROW_COLLECTION_KIND,
                "format_version": PUBLIC_DATA_FORMAT_VERSION,
                "dataset_manifests": {
                    "declared": {
                        "path": "declared/manifest.json",
                        "sha256": hashlib.sha256(declared_manifest.read_bytes()).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert eval_prod_models.resolve_arrow_dataset_root(str(root), "declared") == str(declared.resolve())
    with pytest.raises(FileNotFoundError, match="does not declare dataset 'undeclared'"):
        eval_prod_models.resolve_arrow_dataset_root(str(root), "undeclared")


def test_cluster_eval_arrow_passes_open_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeClusterer:
        def predict_from_arrow(self, block_dict, arrow_dataset, **kwargs):
            captured["block_dict"] = dict(block_dict)
            captured["arrow_dataset"] = arrow_dataset
            captured["kwargs"] = dict(kwargs)
            return {"pred": ["s1"]}, None

    monkeypatch.setattr(eval_prod_models, "read_arrow_s2_blocks", lambda _path: {"block": ["s1"]})
    monkeypatch.setattr(
        eval_prod_models,
        "split_blocks_like_anddata",
        lambda blocks, *, random_seed: ({}, {}, dict(blocks)),
    )
    monkeypatch.setattr(eval_prod_models, "read_signature_to_cluster_id", lambda _path: {"s1": "truth"})

    with _open_minimal_arrow_dataset(tmp_path, clusters={"truth": ["s1"]}) as arrow_dataset:
        eval_prod_models.cluster_eval_arrow(
            arrow_dataset,
            FakeClusterer(),
            random_seed=42,
            n_jobs=1,
            batching_threshold=7,
        )

        assert captured["arrow_dataset"] is arrow_dataset
    assert captured["block_dict"] == {"block": ["s1"]}
    assert "load_name_counts" not in captured["kwargs"]
    assert captured["kwargs"]["batching_threshold"] == 7


def test_eval_main_use_arrow_calls_arrow_eval_without_anddata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import s2and.data as data_module
    import s2and.production_model as production_model

    captured: dict[str, Any] = {}

    class RaisingANDData:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("ANDData should not be constructed for --use-arrow eval")

    dataset_root = tmp_path / "arrow" / "pubmed"
    arrow_dataset = _open_minimal_arrow_dataset(dataset_root)

    def fake_resolve_arrow_dataset(arrow_root: str, dataset_name: str, specter_suffix: str) -> ArrowDataset:
        captured["resolve"] = (arrow_root, dataset_name, specter_suffix)
        return arrow_dataset

    def fake_cluster_eval_arrow(actual_dataset: ArrowDataset, clusterer: Any, **kwargs: Any):
        captured["arrow_dataset"] = actual_dataset
        captured["clusterer"] = clusterer
        captured["kwargs"] = dict(kwargs)
        return {"B3 (P, R, F1)": (1.0, 1.0, 1.0)}, {}

    monkeypatch.setattr(data_module, "ANDData", RaisingANDData)
    monkeypatch.setattr(eval_prod_models, "first_missing_arrow_dataset_error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(eval_prod_models, "resolve_arrow_dataset", fake_resolve_arrow_dataset)
    monkeypatch.setattr(eval_prod_models, "cluster_eval_arrow", fake_cluster_eval_arrow)
    monkeypatch.setattr(
        production_model,
        "load_production_model",
        lambda model_path: SimpleNamespace(model_path=model_path),
    )
    model_path = tmp_path / "explicit-model"
    model_path.mkdir()
    _write_bundle_training_config(model_path, {"data_random_seed": 1111})
    test_split = {"block_membership_sha256": "0" * 64, "test_block_ids": ["heldout"]}
    (model_path / "reproducibility" / "pairwise_training_summary.json").write_text(
        json.dumps({"cluster_test_splits": {"pubmed": test_split}}), encoding="utf-8"
    )
    monkeypatch.setattr(eval_prod_models.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_prod_models.py",
            "--dataset",
            "mini",
            "--datasets",
            "pubmed",
            "--specter2-model-path",
            str(model_path),
            "--use-arrow",
            "--arrow-data-root",
            "arrow-root",
            "--n_jobs",
            "1",
        ],
    )

    eval_prod_models.main()

    assert captured["resolve"] == (str(Path("arrow-root").resolve()), "pubmed", "_specter2.pkl")
    assert captured["arrow_dataset"] is arrow_dataset
    assert arrow_dataset.closed
    assert captured["kwargs"]["n_jobs"] == 1
    assert captured["kwargs"]["random_seed"] == 1111
    assert captured["kwargs"]["test_split"] == test_split
    assert captured["clusterer"].model_path == model_path


@pytest.mark.parametrize("force_arrow", [False, True])
@pytest.mark.parametrize("corruption", ["payload", "manifest"])
def test_eval_main_surfaces_arrow_corruption_with_json_fallback_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_arrow: bool, corruption: str
) -> None:
    import s2and.production_model as production_model

    arrow_root = tmp_path / "arrow"
    dataset_root = _write_eval_arrow_root(arrow_root)
    if corruption == "payload":
        papers = dataset_root / "papers.arrow"
        papers.write_bytes(papers.read_bytes().replace(b"Notes", b"WRONG"))
        message = "checksum mismatch"
        # Availability reads manifests only; the real open must reject the
        # damaged payload even when JSON fallback is configured.
        assert (
            eval_prod_models.first_missing_arrow_dataset_error(str(arrow_root), ["pubmed"], ["_specter2.pkl"]) is None
        )
    else:
        (arrow_root / "manifest.json").write_text("{", encoding="utf-8")
        message = "invalid JSON"

    model_path = tmp_path / "model"
    _write_bundle_training_config(model_path, {"data_random_seed": 1111})
    (model_path / "reproducibility" / "pairwise_training_summary.json").write_text(
        json.dumps({"cluster_test_splits": {"pubmed": {"test_block_ids": ["heldout"]}}}), encoding="utf-8"
    )
    monkeypatch.setattr(production_model, "load_production_model", lambda _path: SimpleNamespace())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_prod_models.py",
            "--dataset",
            "mini",
            "--datasets",
            "pubmed",
            "--specter2-model-path",
            str(model_path),
            "--arrow-data-root",
            str(arrow_root),
            "--json-data-root",
            str(tmp_path),
            *(["--use-arrow"] if force_arrow else []),
        ],
    )
    # Automatic selection used to conceal the integrity error and ask for JSON
    # assets; forced Arrow used to relabel it as missing data.
    with pytest.raises(ValueError, match=message):
        eval_prod_models.main()


@pytest.mark.parametrize("automatic", [False, True])
def test_eval_main_json_uses_recorded_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], automatic: bool
) -> None:
    import s2and.name_tuple_artifact as name_tuple_artifact
    import s2and.production_model as production_model
    from s2and.data import ANDData

    blocks = {"training": ["s0"], "heldout": ["s1"]}
    record = {"block_membership_sha256": block_membership_sha256(blocks), "test_block_ids": ["heldout"]}
    model_path = tmp_path / "model"
    _write_bundle_training_config(model_path, {"data_random_seed": 1111})
    (model_path / "reproducibility" / "pairwise_training_summary.json").write_text(
        json.dumps({"cluster_test_splits": {"pubmed": record}}), encoding="utf-8"
    )

    class Dataset:
        train_blocks = None
        val_blocks = None
        test_blocks = None
        split_cluster_signatures_fixed = ANDData.split_cluster_signatures_fixed

        def get_blocks(self):
            return blocks

        def split_cluster_signatures(self):
            pytest.fail("Must not recreate the test split from its seed")

        def construct_cluster_to_signatures(self, actual):
            assert actual == {"heldout": ["s1"]}
            return {"truth": ["s1"]}

    predicted = []

    class Clusterer:
        def predict(self, actual, _dataset):
            assert actual == {"heldout": ["s1"]}
            predicted.append(actual)
            return {"pred": ["s1"]}, None

    monkeypatch.setattr(eval_prod_models, "build_eval_anddata", lambda **_kwargs: Dataset())
    monkeypatch.setattr(name_tuple_artifact, "load_name_tuple_artifact", lambda _path: SimpleNamespace(pairs=set()))
    monkeypatch.setattr(production_model, "load_production_model", lambda _path: Clusterer())
    arrow_root = tmp_path / "arrow"
    if automatic:
        _write_eval_arrow_root(arrow_root, include_name_counts_index=False)
    else:
        # An explicitly selected JSON evaluation never consults unrelated Arrow data.
        arrow_root.mkdir()
        (arrow_root / "manifest.json").write_text("{", encoding="utf-8")

    def fail_arrow_probe(*_args, **_kwargs):
        pytest.fail("--no-arrow must bypass Arrow availability")

    if not automatic:
        monkeypatch.setattr(eval_prod_models, "first_missing_arrow_dataset_error", fail_arrow_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_prod_models.py",
            "--dataset",
            "mini",
            "--datasets",
            "pubmed",
            *([] if automatic else ["--no-arrow"]),
            "--specter2-model-path",
            str(model_path),
            "--json-data-root",
            str(tmp_path),
            "--arrow-data-root",
            str(arrow_root),
            "--name-counts-index-root",
            str(tmp_path),
            "--name-tuples-path",
            str(tmp_path / "tuples"),
        ],
    )
    eval_prod_models.main()
    assert len(predicted) == 1
    output = capsys.readouterr().out
    if automatic:
        assert "Arrow unavailable; using JSON:" in output
        assert "dataset='pubmed'" in output
        assert "name_counts_index" in output
    else:
        assert "Arrow unavailable; using JSON:" not in output


def test_eval_main_forced_arrow_rejects_undeclared_name_counts_with_json_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrow_root = tmp_path / "arrow"
    _write_eval_arrow_root(arrow_root, include_name_counts_index=False)
    missing_arrow_error = eval_prod_models.first_missing_arrow_dataset_error(
        str(arrow_root), ["pubmed"], ["_specter2.pkl"]
    )
    assert isinstance(missing_arrow_error, FileNotFoundError)
    model_path = tmp_path / "model"
    _write_bundle_training_config(model_path, {"data_random_seed": 1111})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_prod_models.py",
            "--dataset",
            "mini",
            "--datasets",
            "pubmed",
            "--use-arrow",
            "--specter2-model-path",
            str(model_path),
            "--arrow-data-root",
            str(arrow_root),
            "--json-data-root",
            str(tmp_path),
        ],
    )
    # The automatic-selection case above falls back to JSON for this missing
    # declaration. Explicit Arrow must retain the original missing-artifact error.
    with pytest.raises(FileNotFoundError, match="name_counts_index") as error:
        eval_prod_models.main()
    assert str(error.value) == str(missing_arrow_error)
    assert "dataset='pubmed'" in str(error.value)


def test_eval_main_unavailable_arrow_without_json_preserves_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrow_root = tmp_path / "arrow"
    _write_eval_arrow_root(arrow_root, include_name_counts_index=False)
    model_path = tmp_path / "model"
    _write_bundle_training_config(model_path, {"data_random_seed": 1111})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_prod_models.py",
            "--dataset",
            "mini",
            "--datasets",
            "pubmed",
            "--specter2-model-path",
            str(model_path),
            "--arrow-data-root",
            str(arrow_root),
        ],
    )
    with pytest.raises(ValueError, match="name_counts_index") as error:
        eval_prod_models.main()
    assert "dataset='pubmed'" in str(error.value)
    assert "--json-data-root" in str(error.value)
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_eval_main_rejects_invalid_mode_combinations(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = (
        (["--dataset", "mini", "--specter-suffixes", "_specter2.pkl"], "requires an explicit model path"),
        (
            ["--dataset", "mini", "--specter-suffixes", "_specter.pickle"],
            "SPECTER1 production evaluation was removed",
        ),
        (["--train", "--specter2-model-path", "unused-model"], "cannot be combined with --train"),
        (["--dataset", "mini", "--use-arrow", "--train"], "cannot be combined with --train"),
        (
            ["--dataset", "mini", "--specter2-model-path", "some-model", "--seed", "42"],
            "--seed applies only to --train",
        ),
    )
    for argv, message in cases:
        monkeypatch.setattr(sys, "argv", ["eval_prod_models.py", *argv])
        with pytest.raises(ValueError, match=message):
            eval_prod_models.main()


def test_eval_main_requires_explicit_data_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = (
        ("json-training-root", ["--train", "--dataset", "mini"], "requires an explicit --json-data-root"),
        (
            "arrow-training-root",
            [
                "--train",
                "--dataset",
                "mini",
                "--datasets",
                "qian",
                "--train-modes",
                eval_prod_models.TRAIN_MODE_ARROW_RUST,
            ],
            "requires an explicit --arrow-data-root",
        ),
        (
            "evaluation-data-root",
            ["--dataset", "mini", "--specter2-model-path", "model"],
            "requires --arrow-data-root or --json-data-root",
        ),
        (
            "name-assets",
            ["--train", "--dataset", "mini", "--json-data-root", "json"],
            "require explicit --name-counts-index-root and --name-tuples-path",
        ),
    )
    monkeypatch.setattr(eval_prod_models, "bundle_data_random_seed", lambda _path: 42)
    for case_id, argv, message in cases:
        monkeypatch.setattr(sys, "argv", ["eval_prod_models.py", *argv])

        try:
            eval_prod_models.main()
        except ValueError as error:
            assert message in str(error), f"{case_id}: {error}"
        else:
            raise AssertionError(f"{case_id}: incomplete data roots were accepted")


def test_construct_cluster_to_signatures_reports_missing_assignments() -> None:
    with pytest.raises(ValueError, match="missing cluster assignments"):
        eval_prod_models.construct_cluster_to_signatures({"s1": "c1"}, {"block": ["s1", "s2"]})


def test_arrow_calibration_deferred_search_matches_explicit_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from hyperopt import hp

    from s2and.featurizer import FeaturizationInfo
    from s2and.model import Clusterer

    blocks = {"block": ["s1", "s2", "s3"]}
    distances = {"block": np.array([0.2, 0.8, 0.8], dtype=np.float64)}
    splits = eval_prod_models.PairwiseTrainingSplits(
        [], [], [], {}, blocks, {}, {"s1": "together", "s2": "together", "s3": "apart"}
    )
    rust_featurizer = SimpleNamespace(cluster_seeds_require=lambda: [])
    distance_calls = []

    def cached_distances(self, block_dict, actual_featurizer):
        assert block_dict == blocks
        assert actual_featurizer is rust_featurizer
        distance_calls.append(self)
        return distances

    monkeypatch.setattr(Clusterer, "make_distance_matrices_from_rust_featurizer", cached_distances)
    deferred = Clusterer(FeaturizationInfo(), classifier=None, n_jobs=1, n_iter=8)
    explicit = Clusterer(
        FeaturizationInfo(), classifier=None, n_jobs=1, n_iter=8, search_space={"eps": hp.uniform("eps", 0, 1)}
    )
    assert deferred.search_space is None

    for clusterer in (deferred, explicit):
        result = eval_prod_models.fit_clusterer_from_arrow_validation(
            clusterer, splits, rust_featurizer, random_seed=42
        )
        assert result is clusterer

    assert distance_calls == [deferred, explicit]
    assert deferred.best_params == explicit.best_params
    deferred_trials = deferred.hyperopt_trials_store
    explicit_trials = explicit.hyperopt_trials_store
    assert deferred_trials is not None and explicit_trials is not None
    assert deferred_trials.losses() == explicit_trials.losses()
    assert [trial["misc"]["vals"] for trial in deferred_trials.trials] == [
        trial["misc"]["vals"] for trial in explicit_trials.trials
    ]
    assert min(deferred_trials.losses()) == -1.0
    for clusterer in (deferred, explicit):
        predictions, _ = clusterer.predict_from_rust_featurizer(blocks, rust_featurizer, dists=distances)
        assert {frozenset(signatures) for signatures in predictions.values()} == {
            frozenset({"s1", "s2"}),
            frozenset({"s3"}),
        }
    np.testing.assert_array_equal(distances["block"], [0.2, 0.8, 0.8])

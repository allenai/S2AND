from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import scripts.eval_prod_models as eval_prod_models
from s2and.arrow_inputs import ARROW_COLLECTION_KIND, ArrowDataset
from s2and.consts import PUBLIC_DATA_FORMAT_VERSION
from tests.helpers import write_minimal_arrow_prediction_bundle


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


def test_feature_tuple_from_rust_featurizer_uses_one_union_call_and_projects_feature_groups() -> None:
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
            return [[float(row * 100 + index) for index in selected_indices] for row in range(len(indexed_pairs))]

    main_info = SimpleNamespace(
        features_to_use=["main_b", "main_a"],
        feature_group_to_index={"main_a": [3], "main_b": [4, 1]},
    )
    nameless_info = SimpleNamespace(features_to_use=["nameless"], feature_group_to_index={"nameless": [4, 2]})

    features, labels, nameless = eval_prod_models._feature_tuple_from_rust_featurizer(
        FakeRustFeaturizer(),
        [("s3", "s1", 0.25), ("s1", "s2", 1)],
        featurizer_info=main_info,
        nameless_featurizer_info=nameless_info,
        n_jobs=3,
        nan_value=-7.5,
    )

    assert calls == [
        {
            "indexed_pairs": [(2, 0), (0, 1)],
            "selected_indices": [1, 2, 3, 4],
            "n_jobs": 3,
            "nan_value": -7.5,
        }
    ]
    assert features.tolist() == [[1.0, 3.0, 4.0], [101.0, 103.0, 104.0]]
    assert labels.tolist() == [0.25, 1.0]
    assert nameless is not None
    assert nameless.tolist() == [[2.0, 4.0], [102.0, 104.0]]


def test_arrow_training_feature_splits_uses_open_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from s2and import feature_port

    captured: dict[str, Any] = {}

    class FakeRustFeaturizer:
        def signature_ids(self) -> list[str]:
            return []

    def fake_build(arrow_dataset, **kwargs):
        captured["arrow_dataset"] = arrow_dataset
        captured["kwargs"] = kwargs
        return FakeRustFeaturizer()

    monkeypatch.setattr(feature_port, "build_rust_featurizer_from_arrow_dataset", fake_build)
    splits = eval_prod_models.PairwiseTrainingSplits([], [], [], {}, {}, {}, {})

    with _open_minimal_arrow_dataset(tmp_path) as arrow_dataset:
        eval_prod_models.arrow_training_feature_splits(
            arrow_dataset,
            splits,
            featurizer_info=SimpleNamespace(features_to_use=[], feature_group_to_index={}),
            nameless_featurizer_info=None,
            n_jobs=1,
            nan_value=float("nan"),
        )

        assert captured["arrow_dataset"] is arrow_dataset
    assert captured["kwargs"] == {"name_tuples": None, "num_threads": 1}


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
    assert captured["clusterer"].model_path == model_path


def test_eval_main_rejects_invalid_mode_combinations(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = (
        (["--dataset", "mini", "--specter-suffixes", "_specter2.pkl"], "requires an explicit model path"),
        (
            ["--dataset", "mini", "--specter-suffixes", "_specter.pickle"],
            "SPECTER1 production evaluation was removed",
        ),
        (["--train", "--specter2-model-path", "unused-model"], "cannot be combined with --train"),
        (["--dataset", "mini", "--use-arrow", "--train"], "cannot be combined with --train"),
        (["--dataset", "inventors_s2and", "--use-arrow"], "supports --dataset mini and --dataset full only"),
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

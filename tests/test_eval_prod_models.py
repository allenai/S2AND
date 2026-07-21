from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import scripts.eval_prod_models as eval_prod_models
from s2and.arrow_inputs import ValidatedArrowInputs
from s2and.consts import NORMALIZATION_VERSION
from s2and.incremental_linking.feature_block import write_name_counts_index
from tests.helpers import import_s2and_rust, tiny_name_counts_provenance, tiny_name_counts_tuple

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"

requires_canonical_source_bundle = pytest.mark.skip(
    reason=(
        "the explicitly named v1.21 source bundle predates canonical_v2 and fails fast at load; "
        "unskip after a canonical source bundle is available "
        "(docs/normalization_migration_blocked.md, cutover readiness checklist item 4)"
    )
)


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.read_bytes()[: len(_LFS_POINTER_PREFIX)] == _LFS_POINTER_PREFIX


def _skip_if_missing_or_lfs_pointer(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        if os.environ.get("CI"):
            raise pytest.fail.Exception(f"missing LFS-backed artifact(s) in CI: {missing}")
        raise pytest.skip.Exception(f"missing LFS-backed artifact(s): {missing}")
    pointers = [str(path) for path in paths if _is_lfs_pointer(path)]
    if pointers:
        if os.environ.get("CI"):
            raise pytest.fail.Exception(f"Git LFS artifact(s) not materialized in CI: {pointers}")
        raise pytest.skip.Exception(f"Git LFS artifact(s) not materialized: {pointers}")


def _cluster_partition(clusters: Mapping[str, Sequence[str]]) -> frozenset[frozenset[str]]:
    return frozenset(frozenset(str(signature_id) for signature_id in signatures) for signatures in clusters.values())


def _touch_eval_batch_indexes(dataset_root: Path, *, specter_stem: str = "specter") -> None:
    for index_name in (
        "signatures.signatures_batch_index.bin",
        "papers.papers_batch_index.bin",
        "paper_authors.paper_authors_batch_index.bin",
        f"{specter_stem}.specter_batch_index.bin",
    ):
        (dataset_root / index_name).touch()


def _bypass_arrow_artifact_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import s2and.arrow_inputs as arrow_inputs

    monkeypatch.setattr(
        arrow_inputs,
        "validate_arrow_prediction_artifacts",
        lambda paths, **_kwargs: dict(paths),
    )


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


def test_read_arrow_s2_blocks_reads_columns_without_row_dicts(tmp_path: Path) -> None:
    import pyarrow as pa

    table = pa.table(
        {
            "signature_id": pa.array(["s1", "s2", "s3"], type=pa.string()),
            "author_block": pa.array(["a smith", "a smith", "b jones"], type=pa.string()),
        }
    )
    path = tmp_path / "signatures.arrow"
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    assert eval_prod_models.read_arrow_s2_blocks(str(path)) == {
        "a smith": ["s1", "s2"],
        "b jones": ["s3"],
    }


def test_pair_splits_from_arrow_paths_samples_within_block_random_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clusters_path = tmp_path / "qian_clusters.json"
    clusters_path.write_text(
        json.dumps(
            {
                "c1": {"signature_ids": ["s1", "s2"]},
                "c2": {"signature_ids": ["s3"]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(eval_prod_models, "read_arrow_s2_blocks", lambda _path: {"block": ["s1", "s2", "s3"]})
    monkeypatch.setattr(
        eval_prod_models,
        "split_blocks_like_anddata",
        lambda blocks, *, random_seed: (dict(blocks), {}, {}),
    )

    splits = eval_prod_models.pair_splits_from_arrow_paths(
        {"signatures": "signatures.arrow", "clusters": str(clusters_path)},
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


def test_feature_tuple_from_rust_featurizer_uses_selected_feature_groups() -> None:
    class FakeRustFeaturizer:
        def signature_ids(self) -> list[str]:
            return ["s1", "s2"]

        def featurize_pairs_matrix_indexed(
            self,
            indexed_pairs: list[tuple[int, int]],
            selected_indices: list[int],
            _n_jobs: int,
            _nan_value: float,
        ) -> list[list[float]]:
            assert indexed_pairs == [(0, 1)]
            return [[float(index) for index in selected_indices]]

    main_info = SimpleNamespace(features_to_use=["main"], feature_group_to_index={"main": [2, 0]})
    nameless_info = SimpleNamespace(features_to_use=["nameless"], feature_group_to_index={"nameless": [1]})

    features, labels, nameless = eval_prod_models._feature_tuple_from_rust_featurizer(
        FakeRustFeaturizer(),
        [("s1", "s2", 1)],
        featurizer_info=main_info,
        nameless_featurizer_info=nameless_info,
        n_jobs=1,
        nan_value=float("nan"),
    )

    assert features.tolist() == [[0.0, 2.0]]
    assert labels.tolist() == [1.0]
    assert nameless is not None
    assert nameless.tolist() == [[1.0]]


@pytest.mark.parametrize("name_counts_owner", ["main", "nameless"])
def test_arrow_training_feature_splits_loads_selected_name_counts(monkeypatch, name_counts_owner: str) -> None:
    from s2and import feature_port

    captured: dict[str, Any] = {}

    class FakeRustFeaturizer:
        def signature_ids(self) -> list[str]:
            return []

    def fake_build(paths, **kwargs):
        captured["paths"] = paths
        captured["kwargs"] = kwargs
        return FakeRustFeaturizer()

    monkeypatch.setattr(feature_port, "build_rust_featurizer_from_arrow_paths", fake_build)
    validated_paths = ValidatedArrowInputs._from_verified(
        paths={"signatures": "signatures.arrow", "name_counts_index": "name_counts_index"},
        generation_id="generation",
        normalization_version=NORMALIZATION_VERSION,
    )
    splits = eval_prod_models.PairwiseTrainingSplits([], [], [], {}, {}, {}, {})

    eval_prod_models.arrow_training_feature_splits(
        validated_paths,
        splits,
        featurizer_info=SimpleNamespace(
            features_to_use=["name_counts"] if name_counts_owner == "main" else [],
            feature_group_to_index={"name_counts": [0]},
        ),
        nameless_featurizer_info=(
            SimpleNamespace(features_to_use=["name_counts"], feature_group_to_index={"name_counts": [0]})
            if name_counts_owner == "nameless"
            else None
        ),
        n_jobs=1,
        nan_value=float("nan"),
    )

    assert captured["kwargs"]["load_name_counts"] is True


@pytest.mark.parametrize("block_count", [1, 2, 4])
def test_split_blocks_like_anddata_rejects_tiny_smoke_datasets_like_anddata(block_count: int) -> None:
    blocks = {f"b{index}": [f"s{index}"] for index in range(block_count)}

    with pytest.raises(ValueError):
        eval_prod_models.split_blocks_like_anddata(blocks, random_seed=1)


def _read_minimal_incremental_signatures(signatures_path: Path) -> dict[str, Any]:
    import pyarrow as pa

    with pa.memory_map(str(signatures_path), "r") as source:
        table = pa.ipc.open_file(source).read_all()
    signatures: dict[str, Any] = {}
    for row in table.to_pylist():
        signature_id = str(row["signature_id"])
        signatures[signature_id] = SimpleNamespace(
            signature_id=signature_id,
            paper_id=str(row["paper_id"]),
            author_info_first=row["author_first"],
            author_info_first_normalized_without_apostrophe=row["author_first"],
            author_info_last=row["author_last"],
            author_info_orcid=row.get("author_orcid"),
        )
    return signatures


class _ArrowIncrementalFixtureDataset:
    def __init__(
        self,
        arrow_paths: Mapping[str, str],
        signatures: dict[str, Any],
        cluster_seeds_path: Path,
    ) -> None:
        self.arrow_paths = {str(key): str(value) for key, value in arrow_paths.items() if key != "clusters"}
        self.arrow_paths["cluster_seeds"] = str(cluster_seeds_path)
        self.signatures = signatures
        self.cluster_seeds_require: dict[str, str] = {}
        self.cluster_seeds_disallow: set[tuple[str, str]] = set()
        self.altered_cluster_signatures: list[str] = []
        self.name_tuples = None
        self.max_seed_cluster_id = 0
        self.name = "pubmed_specter2_arrow_incremental_fixture"


def test_resolve_arrow_dataset_paths_includes_name_counts_index_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_arrow_artifact_validation(monkeypatch)
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    name_counts_index, _metrics = write_name_counts_index(
        tmp_path, tiny_name_counts_tuple(), tiny_name_counts_provenance()
    )
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()
    _touch_eval_batch_indexes(dataset_root)
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"name_counts_index": str(name_counts_index)}}),
        encoding="utf-8",
    )

    resolved = eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")

    assert resolved["name_counts_index"] == str(name_counts_index)


def test_resolve_arrow_dataset_paths_supports_nested_datasets_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_arrow_artifact_validation(monkeypatch)
    dataset_root = tmp_path / "arrow" / "datasets" / "dummy"
    dataset_root.mkdir(parents=True)
    name_counts_index, _metrics = write_name_counts_index(
        tmp_path / "arrow", tiny_name_counts_tuple(), tiny_name_counts_provenance()
    )
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()
    _touch_eval_batch_indexes(dataset_root)
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"name_counts_index": "../../name_counts_index"}}),
        encoding="utf-8",
    )

    resolved = eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")

    assert resolved["signatures"] == str(dataset_root / "signatures.arrow")
    assert resolved["name_counts_index"] == str(name_counts_index)


def test_resolve_arrow_dataset_paths_supports_release_parent_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_arrow_artifact_validation(monkeypatch)
    dataset_root = tmp_path / "arrow" / "release_parent" / "datasets" / "dummy"
    dataset_root.mkdir(parents=True)
    name_counts_index, _metrics = write_name_counts_index(
        tmp_path / "arrow" / "release_parent", tiny_name_counts_tuple(), tiny_name_counts_provenance()
    )
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter2.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()
    _touch_eval_batch_indexes(dataset_root, specter_stem="specter2")
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"name_counts_index": "../../name_counts_index"}}),
        encoding="utf-8",
    )
    (tmp_path / "arrow" / "release_parent" / "manifest.json").write_text(
        json.dumps({"dataset_manifests": [{"manifest_path": "datasets/dummy/manifest.json"}]}),
        encoding="utf-8",
    )

    resolved = eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter2.pkl")

    assert resolved["signatures"] == str(dataset_root / "signatures.arrow")
    assert resolved["name_counts_index"] == str(Path(name_counts_index).resolve())


def test_resolve_arrow_dataset_root_rejects_ambiguous_release_parent(tmp_path: Path) -> None:
    for release_name in ("release_20260525", "release_20260526"):
        release_root = tmp_path / "arrow" / release_name
        dataset_root = release_root / "datasets" / "dummy"
        dataset_root.mkdir(parents=True)
        (release_root / "manifest.json").write_text(
            json.dumps({"dataset_manifests": [{"manifest_path": "datasets/dummy/manifest.json"}]}),
            encoding="utf-8",
        )
        (dataset_root / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Ambiguous Arrow release parent"):
        eval_prod_models.resolve_arrow_dataset_root(str(tmp_path / "arrow"), "dummy")


def test_resolve_arrow_dataset_paths_requires_eval_name_counts_index(tmp_path: Path) -> None:
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()

    with pytest.raises(FileNotFoundError, match="Missing Arrow name_counts_index"):
        eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")


def test_resolve_arrow_dataset_paths_rejects_bad_manifest_name_counts_index(tmp_path: Path) -> None:
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    (tmp_path / "arrow" / "name_counts_index").mkdir()
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"name_counts_index": "missing/name_counts_index"}}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="specifies name_counts_index path that does not exist"):
        eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")


def test_resolve_arrow_dataset_paths_supports_repo_relative_manifest_name_counts_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bypass_arrow_artifact_validation(monkeypatch)
    dataset_root = tmp_path / "arrow" / "dummy"
    dataset_root.mkdir(parents=True)
    index_root = tmp_path / "repo" / "s2and" / "data" / "name_counts_index"
    write_name_counts_index(index_root.parent, tiny_name_counts_tuple(), tiny_name_counts_provenance())
    for filename in (
        "signatures.arrow",
        "papers.arrow",
        "paper_authors.arrow",
        "specter.arrow",
        "dummy_clusters.json",
    ):
        (dataset_root / filename).touch()
    _touch_eval_batch_indexes(dataset_root)
    (dataset_root / "manifest.json").write_text(
        json.dumps({"paths": {"name_counts_index": "s2and/data/name_counts_index"}}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path / "repo")

    resolved = eval_prod_models.resolve_arrow_dataset_paths(str(tmp_path / "arrow"), "dummy", "_specter.pickle")

    assert resolved["name_counts_index"] == str(index_root.resolve())


def test_cluster_eval_arrow_passes_name_counts_index_and_batch_indexes(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClusterer:
        def predict_from_arrow_paths(self, block_dict, arrow_paths, **kwargs):
            captured["block_dict"] = dict(block_dict)
            captured["arrow_paths"] = dict(arrow_paths)
            captured["kwargs"] = dict(kwargs)
            return {"pred": ["s1"]}, None

    monkeypatch.setattr(eval_prod_models, "read_arrow_s2_blocks", lambda _path: {"block": ["s1"]})
    monkeypatch.setattr(
        eval_prod_models,
        "split_blocks_like_anddata",
        lambda blocks, *, random_seed: ({}, {}, dict(blocks)),
    )
    monkeypatch.setattr(eval_prod_models, "read_signature_to_cluster_id", lambda _path: {"s1": "truth"})

    arrow_paths = ValidatedArrowInputs._from_verified(
        paths={
            "signatures": "signatures.arrow",
            "papers": "papers.arrow",
            "paper_authors": "paper_authors.arrow",
            "specter": "specter.arrow",
            "clusters": "clusters.json",
            "name_counts_index": "name_counts_index",
            "signatures_batch_index": "signatures.signatures_batch_index.bin",
        },
        generation_id="test-generation",
        normalization_version=NORMALIZATION_VERSION,
    )
    eval_prod_models.cluster_eval_arrow(
        arrow_paths,
        SimpleNamespace(predict_from_arrow_paths=FakeClusterer().predict_from_arrow_paths),
        random_seed=42,
        n_jobs=1,
    )

    assert captured["block_dict"] == {"block": ["s1"]}
    assert "load_name_counts" not in captured["kwargs"]
    assert captured["arrow_paths"]["name_counts_index"] == "name_counts_index"
    assert captured["arrow_paths"]["signatures_batch_index"] == "signatures.signatures_batch_index.bin"
    assert "clusters" not in captured["arrow_paths"]


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

    def fake_cluster_eval_arrow(arrow_paths: dict[str, str], clusterer: Any, **kwargs: Any):
        captured["arrow_paths"] = dict(arrow_paths)
        captured["clusterer"] = clusterer
        captured["kwargs"] = dict(kwargs)
        return {"B3 (P, R, F1)": (1.0, 1.0, 1.0)}, {}

    monkeypatch.setattr(data_module, "ANDData", RaisingANDData)
    monkeypatch.setattr(eval_prod_models, "first_missing_arrow_dataset_error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        eval_prod_models,
        "resolve_arrow_dataset_paths",
        lambda arrow_root, dataset_name, specter_suffix: {
            "dataset": dataset_name,
            "specter_suffix": specter_suffix,
            "root": arrow_root,
        },
    )
    monkeypatch.setattr(eval_prod_models, "cluster_eval_arrow", fake_cluster_eval_arrow)
    monkeypatch.setattr(
        production_model,
        "load_production_model",
        lambda model_path: SimpleNamespace(model_path=model_path),
    )
    model_path = tmp_path / "explicit-model"
    model_path.mkdir()
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
            "--specter-suffixes",
            "_specter2.pkl",
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

    assert captured["arrow_paths"] == {
        "dataset": "pubmed",
        "specter_suffix": "_specter2.pkl",
        "root": "arrow-root",
    }
    assert captured["kwargs"]["n_jobs"] == 1
    assert captured["kwargs"]["random_seed"] == 42
    assert captured["clusterer"].model_path == model_path


def test_eval_main_rejects_invalid_mode_combinations(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = (
        (["--dataset", "mini", "--specter-suffixes", "_specter2.pkl"], "requires an explicit model path"),
        (["--train", "--specter2-model-path", "unused-model"], "cannot be combined with --train"),
        (["--dataset", "mini", "--use-arrow", "--train"], "cannot be combined with --train"),
        (["--dataset", "inventors_s2and", "--use-arrow"], "supports --dataset mini and --dataset full only"),
    )
    for argv, message in cases:
        monkeypatch.setattr(sys, "argv", ["eval_prod_models.py", *argv])
        with pytest.raises(ValueError, match=message):
            eval_prod_models.main()


def test_construct_cluster_to_signatures_reports_missing_assignments() -> None:
    with pytest.raises(ValueError, match="missing cluster assignments"):
        eval_prod_models.construct_cluster_to_signatures({"s1": "c1"}, {"block": ["s1", "s2"]})


@requires_canonical_source_bundle
@pytest.mark.requires_lfs
def test_pubmed_specter2_arrow_fixture_matches_production_eval() -> None:
    rust_available, rust_payload = import_s2and_rust()
    if not rust_available:
        raise pytest.skip.Exception(f"s2and_rust unavailable: {rust_payload!r}")

    from s2and.production_model import load_production_model

    fixture_root = Path("tests/fixtures/arrow/pubmed_specter2")
    fixture_dataset = fixture_root / "pubmed"
    production_model = Path("s2and/data/production_model_v1.21")
    _skip_if_missing_or_lfs_pointer(
        [
            fixture_dataset / "signatures.arrow",
            fixture_dataset / "papers.arrow",
            fixture_dataset / "paper_authors.arrow",
            fixture_dataset / "specter2.arrow",
            fixture_dataset / "signatures.signatures_batch_index.bin",
            fixture_dataset / "papers.papers_batch_index.bin",
            fixture_dataset / "paper_authors.paper_authors_batch_index.bin",
            fixture_dataset / "specter2.specter_batch_index.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/first.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/last.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/first_last.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/last_first_initial.bin",
            production_model / "manifest.json",
            production_model / "clusterer.json",
            production_model / "pairwise/main.lgb",
            production_model / "pairwise/nameless.lgb",
        ]
    )

    arrow_paths = eval_prod_models.resolve_arrow_dataset_paths(str(fixture_root), "pubmed", "_specter2.pkl")
    assert Path(arrow_paths["specter"]).name == "specter2.arrow"
    assert Path(arrow_paths["name_counts_index"]).resolve() == (fixture_dataset / "name_counts_index").resolve()

    clusterer = load_production_model(str(production_model))
    clusterer.n_jobs = 4
    cluster_metrics, _ = eval_prod_models.cluster_eval_arrow(
        arrow_paths,
        clusterer,
        random_seed=42,
        n_jobs=4,
    )

    assert cluster_metrics["B3 (P, R, F1)"] == pytest.approx((1.0, 0.9, 0.948), abs=5e-4)


@requires_canonical_source_bundle
@pytest.mark.requires_lfs
def test_pubmed_specter2_arrow_fixture_incremental_smoke_matches_expected_b3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from s2and.eval import b3_precision_recall_fscore
    from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow
    from s2and.production_model import load_production_model

    monkeypatch.setenv("S2AND_BACKEND", "rust")
    fixture_root = Path("tests/fixtures/arrow/pubmed_specter2")
    fixture_dataset = fixture_root / "pubmed"
    production_model = Path("s2and/data/production_model_v1.21")
    _skip_if_missing_or_lfs_pointer(
        [
            fixture_dataset / "signatures.arrow",
            fixture_dataset / "papers.arrow",
            fixture_dataset / "paper_authors.arrow",
            fixture_dataset / "specter2.arrow",
            fixture_dataset / "signatures.signatures_batch_index.bin",
            fixture_dataset / "papers.papers_batch_index.bin",
            fixture_dataset / "paper_authors.paper_authors_batch_index.bin",
            fixture_dataset / "specter2.specter_batch_index.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/first.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/last.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/first_last.bin",
            fixture_dataset / "name_counts_index/generations/pubmed-specter2/last_first_initial.bin",
            production_model / "manifest.json",
            production_model / "clusterer.json",
            production_model / "pairwise/main.lgb",
            production_model / "pairwise/nameless.lgb",
            production_model / "incremental_linker/booster.lgb",
            production_model / "incremental_linker/metadata.json",
        ]
    )

    arrow_paths = eval_prod_models.resolve_arrow_dataset_paths(str(fixture_root), "pubmed", "_specter2.pkl")
    signatures = _read_minimal_incremental_signatures(fixture_dataset / "signatures.arrow")
    _train_block_dict, _val_block_dict, test_block_dict = eval_prod_models.split_blocks_like_anddata(
        eval_prod_models.read_arrow_s2_blocks(arrow_paths["signatures"]),
        random_seed=42,
    )
    signature_to_cluster_id = eval_prod_models.read_signature_to_cluster_id(arrow_paths["clusters"])
    cluster_to_signatures = eval_prod_models.construct_cluster_to_signatures(signature_to_cluster_id, test_block_dict)

    clusterer = load_production_model(str(production_model))
    clusterer.n_jobs = 4
    predicted_clusters: dict[str, list[str]] = {}
    total_query_count = 0
    total_candidate_row_count = 0

    for block_index, (block_key, block_signatures) in enumerate(sorted(test_block_dict.items())):
        seed_signature_to_cluster: dict[str, str] = {}
        seen_cluster_ids: set[str] = set()
        for signature_id in block_signatures:
            cluster_id = signature_to_cluster_id[signature_id]
            if cluster_id in seen_cluster_ids:
                continue
            seed_signature_to_cluster[signature_id] = cluster_id
            seen_cluster_ids.add(cluster_id)

        cluster_seeds_path = tmp_path / f"cluster_seeds_{block_index}.arrow"
        write_cluster_seeds_arrow(cluster_seeds_path, seed_signature_to_cluster)
        dataset = _ArrowIncrementalFixtureDataset(arrow_paths, signatures, cluster_seeds_path)
        result = cast(
            dict[str, Any],
            clusterer.predict_incremental(
                list(block_signatures),
                cast(Any, dataset),
                prevent_new_incompatibilities=False,
                total_ram_bytes=1_000_000_000_000,
            ),
        )
        if block_index == 0:
            direct_result = cast(
                dict[str, Any],
                clusterer.predict_incremental_from_arrow_paths(
                    list(block_signatures),
                    {**arrow_paths, "cluster_seeds": str(cluster_seeds_path)},
                    prevent_new_incompatibilities=False,
                    batching_threshold=None,
                    total_ram_bytes=1_000_000_000_000,
                ),
            )
            assert _cluster_partition(
                cast(Mapping[str, Sequence[str]], direct_result["clusters"])
            ) == _cluster_partition(cast(Mapping[str, Sequence[str]], result["clusters"]))
            direct_telemetry = cast(Mapping[str, Any], direct_result["incremental_linker_telemetry"])
            assert direct_result["incremental_linker_query_view"] == "raw_arrow"
            assert direct_telemetry["arrow_promoted_incremental"] == 1
            assert direct_telemetry["seed_setup_cluster_seeds_source"] == "arrow"
        telemetry = cast(Mapping[str, Any], result["incremental_linker_telemetry"])
        query_count = len(block_signatures) - len(seed_signature_to_cluster)
        assert result["incremental_linker_query_view"] == "raw_arrow"
        assert telemetry["arrow_promoted_incremental"] == 1
        assert telemetry["seed_setup_cluster_seeds_source"] == "arrow"
        assert telemetry["seed_arrow_reused_source"] == 1
        assert telemetry["query_count"] == query_count
        total_query_count += int(telemetry["query_count"])
        total_candidate_row_count += int(telemetry["candidate_row_count"])

        block_signature_set = set(block_signatures)
        for cluster_id, members in cast(Mapping[str, Sequence[str]], result["clusters"]).items():
            kept_members = [str(member) for member in members if str(member) in block_signature_set]
            if kept_members:
                predicted_clusters[f"{block_index}:{block_key}:{cluster_id}"] = kept_members

    cluster_metrics = b3_precision_recall_fscore(cluster_to_signatures, predicted_clusters)
    assert total_query_count == 127
    assert total_candidate_row_count > 0
    # Was (1.0, 0.816, 0.899) before FEATURIZER_VERSION 5. The is_reliable
    # "both detectors must agree" change perturbs the language features fed to
    # the (not-yet-retrained) production model, shifting B3 recall/F1 by ~0.001
    # on this fixture. This pins the transient old-model-with-new-features value
    # and will move again once the v1.3 model is retrained on the new features.
    assert cluster_metrics[:3] == pytest.approx((1.0, 0.815, 0.898), abs=5e-4)

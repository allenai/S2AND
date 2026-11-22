from __future__ import annotations

import hashlib
import json
import pickle
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import s2and.subblocking as subblocking
from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from scripts.eps_sweep import common, sweep_eps_on_linking_gold
from tests.incremental_test_helpers import _dataset, prepare_seed_arrow_bundle
from tests.model_helpers import ConstantDistanceClassifier


def _cli(*extra: str) -> list[str]:
    return [
        "--dataset",
        "dummy",
        "--arrow-root",
        "arrow",
        "--model-path",
        "model",
        *extra,
    ]


def test_load_gold_drops_unlabeled_singleton_orcid_rows(tmp_path) -> None:
    gold_path = tmp_path / "gold.parquet"
    pd.DataFrame(
        [
            {
                "dataset": "unit",
                "table_name": "train.parquet",
                "split": "train",
                "source_key": "train",
                "supervision_type": "unlabeled_singleton_orcid",
                "query_signature_id": "q1",
                "member_signature_id": "m1",
                "query_view": "full",
                "label": 0,
                "weight_pair": 1.0,
                "weight_query_balanced": 1.0,
                "weight_query_label_balanced": 1.0,
                "weight_query_class_balanced": 1.0,
            },
            {
                "dataset": "unit",
                "table_name": "train.parquet",
                "split": "train",
                "source_key": "train",
                "supervision_type": "positive_repeat_orcid",
                "query_signature_id": "q2",
                "member_signature_id": "m2",
                "query_view": "full",
                "label": 1,
                "weight_pair": 1.0,
                "weight_query_balanced": 1.0,
                "weight_query_label_balanced": 1.0,
                "weight_query_class_balanced": 1.0,
            },
        ]
    ).to_parquet(gold_path, index=False)

    loaded = sweep_eps_on_linking_gold._load_gold(gold_path)

    assert loaded["query_signature_id"].tolist() == ["q2"]
    assert loaded["supervision_type"].tolist() == ["positive_repeat_orcid"]


def test_ensure_distance_caches_skips_singleton_without_compute_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        sweep_eps_on_linking_gold,
        "_build_arrow_featurizer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("singleton block should not build featurizer")),
    )
    args = SimpleNamespace(
        arrow_root=tmp_path / "arrow",
        batching_threshold=10,
        compute_missing_dists=False,
        dataset="dummy",
        model_path=tmp_path / "model.pkl",
        overwrite_dists=False,
        pair_chunk_size=3,
        suppress_orcid_constraints=False,
        use_orcid_subblocking=False,
    )
    clusterer = SimpleNamespace(batch_size=99)

    rows = sweep_eps_on_linking_gold._ensure_distance_caches(
        cast(Any, args),
        clusterer,
        {"singleton": ["s1"]},
        tmp_path / "cache",
        cast(Any, SimpleNamespace(generation_id="test-generation")),
    )

    assert rows[0]["block_key"] == "singleton"
    assert rows[0]["pair_count"] == 0
    assert rows[0]["computed"] is False
    assert clusterer.batch_size == 99


def test_distance_cache_metadata_rejects_changed_inputs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model")
    args = SimpleNamespace(
        arrow_root=tmp_path / "arrow",
        batching_threshold=10,
        dataset="dummy",
        model_path=model_path,
        pair_chunk_size=3,
        suppress_orcid_constraints=False,
        use_orcid_subblocking=False,
    )
    metadata = sweep_eps_on_linking_gold._cache_metadata(
        cast(Any, args),
        "block",
        ["s1", "s2"],
        "first-generation",
    )

    cache_path = tmp_path / "cache.pkl"

    def cache(record: dict) -> None:
        with cache_path.open("wb") as outfile:
            pickle.dump({"metadata": record, "dist": [0.25]}, outfile)

    def expected(generation: str = "first-generation") -> dict:
        return sweep_eps_on_linking_gold._cache_metadata(cast(Any, args), "block", ["s1", "s2"], generation)

    cache(metadata)
    with pytest.raises(ValueError, match="arrow_generation_id"):
        sweep_eps_on_linking_gold._load_cached_distance(cache_path, expected("second-generation"))

    cache({key: value for key, value in metadata.items() if key != "s2and_version"})
    with pytest.raises(ValueError, match="s2and_version"):
        sweep_eps_on_linking_gold._load_cached_distance(cache_path, expected())

    cache(metadata)
    with monkeypatch.context() as patch:
        patch.setattr(Clusterer, "orcid_name_waiver", None)
        with pytest.raises(ValueError, match="orcid_name_waiver"):
            sweep_eps_on_linking_gold._load_cached_distance(cache_path, expected())

    model_path.write_bytes(b"second model with different contents")
    with pytest.raises(ValueError, match="model_"):
        sweep_eps_on_linking_gold._load_cached_distance(cache_path, expected())


def test_distance_cache_recomputes_for_the_new_name_policy(tmp_path) -> None:
    arrow_root = tmp_path / "arrow"
    model_path = tmp_path / "model"
    model_path.write_bytes(b"constant pairwise distance")
    args = sweep_eps_on_linking_gold.parse_args(
        [
            "--dataset",
            "dummy",
            "--arrow-root",
            str(arrow_root),
            "--model-path",
            str(model_path),
            "--compute-missing-dists",
            "--n-jobs",
            "1",
        ]
    )
    scorer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.1),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
    )
    dataset = _dataset([("孝", "", "鈴木"), ("孝徳", "", "鈴木")], {})
    prepare_seed_arrow_bundle(dataset, arrow_root)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    block = {"block": ["0", "1"]}
    cache_path = sweep_eps_on_linking_gold._distance_cache_path(cache_dir, args.dataset, 0, "block")
    with ArrowDataset.open(arrow_root) as arrow:
        metadata = sweep_eps_on_linking_gold._cache_metadata(args, "block", block["block"], arrow.generation_id)
        for stale in ({}, {"distance_policy_version": metadata["distance_policy_version"] - 1}):
            old_metadata = {key: value for key, value in metadata.items() if key != "distance_policy_version"}
            with cache_path.open("wb") as outfile:
                pickle.dump({"metadata": old_metadata | stale, "dist": [0.1]}, outfile)
            with pytest.raises(ValueError, match="key=distance_policy_version"):
                sweep_eps_on_linking_gold._ensure_distance_caches(args, scorer, block, cache_dir, arrow)

        args.overwrite_dists = True
        rows = sweep_eps_on_linking_gold._ensure_distance_caches(args, scorer, block, cache_dir, arrow)
        assert rows[0]["computed"]
        assert list(sweep_eps_on_linking_gold._load_cached_distance(cache_path, metadata)) == [LARGE_DISTANCE]

        args.overwrite_dists = False
        reused = sweep_eps_on_linking_gold._ensure_distance_caches(args, scorer, block, cache_dir, arrow)
        assert not reused[0]["computed"]


def test_eps_sweep_resolves_bound_model_and_arrow_inputs(tmp_path) -> None:
    model_path = tmp_path / "production_model_v9.9"
    (model_path / "pairwise").mkdir(parents=True)
    (model_path / "manifest.json").write_text("{}", encoding="utf-8")
    (model_path / "pairwise" / "main.lgb").write_bytes(b"model")
    args = SimpleNamespace(model_path=model_path)

    fingerprint = sweep_eps_on_linking_gold._model_fingerprint(cast(Any, args))  # noqa: SLF001

    assert fingerprint["model_path"] == str(model_path.resolve())
    assert fingerprint["model_size"] == 7
    assert isinstance(fingerprint["model_sha256"], str)
    assert len(fingerprint["model_sha256"]) == 64

    arrow_root = tmp_path / "arrow"
    dataset_manifest = arrow_root / "dummy" / "manifest.json"
    dataset_manifest.parent.mkdir(parents=True)
    dataset_manifest.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(dataset_manifest.read_bytes()).hexdigest()
    (arrow_root / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "s2and_arrow_collection",
                "format_version": 1,
                "dataset_manifests": {
                    "dummy": {
                        "path": "dummy/manifest.json",
                        "sha256": digest,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert common.arrow_dataset_dir(arrow_root, "dummy") == dataset_manifest.parent.resolve()
    with pytest.raises(ValueError, match="does not declare"):
        common.arrow_dataset_dir(arrow_root, "other")

    undeclared_root = tmp_path / "undeclared"
    (undeclared_root / "dummy").mkdir(parents=True)
    (undeclared_root / "dummy" / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        common.arrow_dataset_dir(undeclared_root, "dummy")


def test_cli_enforces_production_route_and_compute_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate(*extra: str):
        args = sweep_eps_on_linking_gold.parse_args(_cli(*extra))
        sweep_eps_on_linking_gold._validate_args(args)
        return args

    with pytest.raises(ValueError, match="--max-subblocks"):
        validate("--compute-missing-dists")
    with pytest.raises(ValueError, match="--route bulk"):
        validate("--max-subblocks", "1")
    with pytest.raises(ValueError, match="--use-orcid-subblocking exactly"):
        validate("--use-orcid-subblocking")

    validate("--compute-missing-dists", "--max-subblocks", "1", "--route", "bulk")
    assert validate("--compute-missing-dists", "--allow-full-run").suppress_orcid_constraints is True
    args = validate("--use-orcid-subblocking", "--use-orcid-constraints", "--n-jobs", "2")
    assert args.suppress_orcid_constraints is False
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    sweep_eps_on_linking_gold._configure_runtime_environment(args)
    assert sweep_eps_on_linking_gold.os.environ["OMP_NUM_THREADS"] == "2"
    assert sweep_eps_on_linking_gold.os.environ["RAYON_NUM_THREADS"] == "2"


@pytest.mark.parametrize(
    ("raw_config", "neighbors", "orcid"),
    [(None, 16, False), ({"neighbors": 7}, 7, True), (subblocking.GraphSubblockingConfig(neighbors=5), 5, False)],
)
def test_eps_sweep_uses_production_native_planner(monkeypatch, raw_config, neighbors, orcid):
    calls = []
    dataset = object()
    counts = {"ch": {"cy": 16}}

    def native(arrow_dataset, signature_ids, **kwargs):
        assert arrow_dataset is dataset
        calls.append((signature_ids, kwargs))
        return {"a": ["1", "2"], "b": ["3"]}, {"graph_fallback_native": True}, {}

    def reject_python(*args, **kwargs):
        raise AssertionError("EPS selection must not use the Python graph fallback")

    monkeypatch.setattr(subblocking, "_make_subblocks_with_telemetry_arrow_rust", native)
    monkeypatch.setattr(subblocking, "make_subblocks_with_telemetry", reject_python)
    blocks, telemetry = sweep_eps_on_linking_gold._build_arrow_subblocked_block_dict(
        args=SimpleNamespace(batching_threshold=2, use_orcid_subblocking=orcid),
        clusterer=SimpleNamespace(
            subblocking_graph_config=raw_config, random_state=7, _subblocking_prefix_counts=lambda: counts
        ),
        block_dict={"large": ["1", "2", "3"], "small": ["4"], "irrelevant": ["5", "6", "7"]},
        target_signature_ids={"1", "4"},
        arrow_dataset=dataset,
    )
    assert blocks == {"large|subblock=a": ["1", "2"], "large|subblock=b": ["3"], "small": ["4"]}
    assert len(calls) == 1
    assert calls[0][0] == ["1", "2", "3"]
    assert calls[0][1]["graph_subblocking_config"].neighbors == neighbors
    assert calls[0][1]["graph_subblocking_random_seed"] == 7
    assert calls[0][1]["first_k_letter_counts_sorted"] is counts
    assert calls[0][1]["use_orcid_subblocking"] is orcid
    assert telemetry[0]["subblocking_telemetry"]["graph_fallback_native"] is True


@pytest.mark.parametrize(
    "result",
    [{"a": ["1", "2"], "b": ["2", "3"]}, {"a": ["1", "2", "3"]}],
)
def test_eps_sweep_rejects_invalid_native_partition(monkeypatch, result):
    monkeypatch.setattr(subblocking, "_make_subblocks_with_telemetry_arrow_rust", lambda *a, **k: (result, {}, {}))
    with pytest.raises(ValueError, match="every input signature exactly once|oversized subblock"):
        sweep_eps_on_linking_gold._build_arrow_subblocked_block_dict(
            args=SimpleNamespace(batching_threshold=2, use_orcid_subblocking=False),
            clusterer=SimpleNamespace(_subblocking_prefix_counts=dict),
            block_dict={"block": ["1", "2", "3"]},
            target_signature_ids={"1"},
            arrow_dataset=object(),
        )


def test_eps_sweep_rejects_invalid_graph_config_type():
    with pytest.raises(ValueError, match="GraphSubblockingConfig, mapping, or None"):
        sweep_eps_on_linking_gold._build_arrow_subblocked_block_dict(
            args=SimpleNamespace(batching_threshold=2, use_orcid_subblocking=False),
            clusterer=SimpleNamespace(subblocking_graph_config="invalid"),
            block_dict={},
            target_signature_ids=set(),
            arrow_dataset=object(),
        )


def test_cache_order_appends_non_target_subblocks_after_target_subblocks() -> None:
    selected = {"b|subblock=z": ["3"], "b|subblock=a": ["1"]}
    every = {**selected, "b|subblock=m": ["2"], "b|subblock=0": ["4"]}

    # Bulk runs and production runs give target subblocks the same cache indexes.
    assert list(sweep_eps_on_linking_gold._cache_order(selected, None)) == ["b|subblock=a", "b|subblock=z"]  # noqa: SLF001
    assert list(sweep_eps_on_linking_gold._cache_order(selected, every)) == [  # noqa: SLF001
        "b|subblock=a",
        "b|subblock=z",
        "b|subblock=0",
        "b|subblock=m",
    ]


def test_production_assignments_take_cached_clusters_for_the_bulk_step() -> None:
    cluster_blocks = {
        "b|subblock=full": sweep_eps_on_linking_gold.CachedBlockClusters(
            "b|subblock=full", ["1", "2", "3"], {0.5: [1, 1, 2], 0.95: [1, 1, 1]}
        ),
        "b|subblock=solo": sweep_eps_on_linking_gold.CachedBlockClusters(
            "b|subblock=solo", ["4"], {0.5: [0], 0.95: [0]}
        ),
    }
    calls: list[dict[str, Any]] = []

    class Model:
        cluster_model = SimpleNamespace(set_params=lambda **params: calls.append(params))

        def _predict_from_arrow_request(self, block_dict, arrow_dataset, **options):
            calls.append({"blocks": block_dict, "threshold": options["batching_threshold"]})
            bulk = options["bulk_clusters"]({"b|subblock=full": ["1", "2", "3"], "b|subblock=solo": ["4"]})
            # The later production steps attach "5" to the bulk cluster that holds "1".
            return {**bulk, "b|subblock=full_1": [*bulk["b|subblock=full_1"], "5"]}, None

    args = SimpleNamespace(batching_threshold=3, total_ram_bytes=1)
    assignments = sweep_eps_on_linking_gold._production_assignments_for_eps(  # noqa: SLF001
        eps=0.5,
        clusterer=Model(),
        block_dict={"b": ["1", "2", "3", "4", "5"]},
        cluster_blocks=cluster_blocks,
        arrow_dataset=cast(Any, None),
        args=args,
    )

    assert calls[0] == {"eps": 0.5}
    assert calls[1]["threshold"] == 3
    assert assignments == {
        "1": "b|subblock=full_1",
        "2": "b|subblock=full_1",
        "5": "b|subblock=full_1",
        "3": "b|subblock=full_2",
        "4": "b|subblock=solo_0",
    }

    class ReorderedPlan(Model):
        def _predict_from_arrow_request(self, block_dict, arrow_dataset, **options):
            return options["bulk_clusters"]({"b|subblock=full": ["2", "1", "3"]}), None

    with pytest.raises(ValueError, match="No cached clusters match"):
        sweep_eps_on_linking_gold._production_assignments_for_eps(  # noqa: SLF001
            eps=0.5,
            clusterer=ReorderedPlan(),
            block_dict={"b": ["1", "2", "3"]},
            cluster_blocks=cluster_blocks,
            arrow_dataset=cast(Any, None),
            args=args,
        )

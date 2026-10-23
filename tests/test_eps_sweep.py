from __future__ import annotations

import hashlib
import json
import pickle
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import s2and.subblocking as subblocking
from scripts.eps_sweep import common, sweep_eps_on_linking_gold


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


def test_eps_sweep_runtime_environment_sets_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    sweep_eps_on_linking_gold._configure_runtime_environment(cast(Any, SimpleNamespace(n_jobs=2)))

    assert sweep_eps_on_linking_gold.os.environ["OMP_NUM_THREADS"] == "2"
    assert sweep_eps_on_linking_gold.os.environ["RAYON_NUM_THREADS"] == "2"


def test_eps_sweep_cli_has_one_real_orcid_constraint_switch() -> None:
    default_args = sweep_eps_on_linking_gold.parse_args(_cli())
    enabled_args = sweep_eps_on_linking_gold.parse_args(_cli("--use-orcid-constraints"))

    assert default_args.suppress_orcid_constraints is True
    assert enabled_args.suppress_orcid_constraints is False


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


@pytest.mark.parametrize(
    ("replacement_model", "expected_generation", "message"),
    (
        (b"second model with different contents", "first-generation", "model_"),
        (None, "second-generation", "arrow_generation_id"),
    ),
    ids=("model-content", "arrow-generation"),
)
def test_distance_cache_metadata_rejects_changed_inputs(
    tmp_path,
    replacement_model: bytes | None,
    expected_generation: str,
    message: str,
) -> None:
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
    with cache_path.open("wb") as outfile:
        pickle.dump({"metadata": metadata, "dist": [0.25]}, outfile)

    if replacement_model is not None:
        model_path.write_bytes(replacement_model)
    expected_metadata = sweep_eps_on_linking_gold._cache_metadata(
        cast(Any, args),
        "block",
        ["s1", "s2"],
        expected_generation,
    )

    with pytest.raises(ValueError, match=message):
        sweep_eps_on_linking_gold._load_cached_distance(cache_path, expected_metadata)


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


def test_validate_args_requires_limit_or_full_run_for_compute_missing() -> None:
    args = sweep_eps_on_linking_gold.parse_args(_cli("--compute-missing-dists"))

    with pytest.raises(ValueError, match="--max-subblocks"):
        sweep_eps_on_linking_gold._validate_args(args)  # noqa: SLF001

    limited_args = sweep_eps_on_linking_gold.parse_args(
        _cli("--compute-missing-dists", "--max-subblocks", "1", "--route", "bulk")
    )
    sweep_eps_on_linking_gold._validate_args(limited_args)  # noqa: SLF001

    full_run_args = sweep_eps_on_linking_gold.parse_args(_cli("--compute-missing-dists", "--allow-full-run"))
    sweep_eps_on_linking_gold._validate_args(full_run_args)  # noqa: SLF001


@pytest.mark.parametrize(
    "raw_config,neighbors", [(None, 16), ({"neighbors": 7}, 7), (subblocking.GraphSubblockingConfig(neighbors=5), 5)]
)
@pytest.mark.parametrize("orcid", [False, True])
def test_eps_sweep_uses_production_native_planner(monkeypatch, raw_config, neighbors, orcid):
    calls = []
    dataset = object()

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
        clusterer=SimpleNamespace(subblocking_graph_config=raw_config, random_state=7),
        block_dict={"large": ["1", "2", "3"], "small": ["4"], "irrelevant": ["5", "6", "7"]},
        target_signature_ids={"1", "4"},
        arrow_dataset=dataset,
    )
    assert blocks == {"large|subblock=a": ["1", "2"], "large|subblock=b": ["3"], "small": ["4"]}
    assert len(calls) == 1
    assert calls[0][0] == ["1", "2", "3"]
    assert calls[0][1]["graph_subblocking_config"].neighbors == neighbors
    assert calls[0][1]["graph_subblocking_random_seed"] == 7
    assert calls[0][1]["use_orcid_subblocking"] is orcid
    assert telemetry[0]["subblocking_telemetry"]["graph_fallback_native"] is True


@pytest.mark.parametrize(
    "result",
    [{"a": ["1", "2"]}, {"a": ["1", "2"], "b": ["2", "3"]}, {"a": ["1", "2"], "b": ["other"]}, {"a": ["1", "2", "3"]}],
)
def test_eps_sweep_rejects_invalid_native_partition(monkeypatch, result):
    monkeypatch.setattr(subblocking, "_make_subblocks_with_telemetry_arrow_rust", lambda *a, **k: (result, {}, {}))
    with pytest.raises(ValueError, match="every input signature exactly once|oversized subblock"):
        sweep_eps_on_linking_gold._build_arrow_subblocked_block_dict(
            args=SimpleNamespace(batching_threshold=2, use_orcid_subblocking=False),
            clusterer=SimpleNamespace(),
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


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--max-subblocks", "1"), "--route bulk"),
        (("--use-orcid-subblocking",), "--use-orcid-subblocking exactly"),
        (("--use-orcid-constraints",), "--use-orcid-subblocking exactly"),
    ],
)
def test_production_route_rejects_partial_blocks_and_mismatched_orcid_repair(extra, message) -> None:
    with pytest.raises(ValueError, match=message):
        sweep_eps_on_linking_gold._validate_args(sweep_eps_on_linking_gold.parse_args(_cli(*extra)))  # noqa: SLF001
    sweep_eps_on_linking_gold._validate_args(  # noqa: SLF001
        sweep_eps_on_linking_gold.parse_args(_cli("--use-orcid-subblocking", "--use-orcid-constraints"))
    )
    sweep_eps_on_linking_gold._validate_args(  # noqa: SLF001
        sweep_eps_on_linking_gold.parse_args(_cli("--route", "bulk", "--max-subblocks", "1"))
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


def test_production_assignments_cut_cached_trees_for_the_bulk_step() -> None:
    from scipy.cluster.hierarchy import linkage

    tree = linkage([0.1, 0.9, 0.9], method="average")
    linkage_blocks = {
        "b|subblock=full": sweep_eps_on_linking_gold.CachedLinkageBlock("b|subblock=full", ["1", "2", "3"], tree),
        "b|subblock=solo": sweep_eps_on_linking_gold.CachedLinkageBlock("b|subblock=solo", ["4"], None),
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
        linkage_blocks=linkage_blocks,
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

    with pytest.raises(ValueError, match="No cached linkage matches"):
        sweep_eps_on_linking_gold._production_assignments_for_eps(  # noqa: SLF001
            eps=0.5,
            clusterer=ReorderedPlan(),
            block_dict={"b": ["1", "2", "3"]},
            linkage_blocks=linkage_blocks,
            arrow_dataset=cast(Any, None),
            args=args,
        )

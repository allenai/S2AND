from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import s2and.subblocking as subblocking
from scripts.eps_sweep import sweep_eps_on_linking_gold


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


def test_eps_sweep_runtime_environment_sets_backend_and_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    sweep_eps_on_linking_gold._configure_runtime_environment(cast(Any, SimpleNamespace(n_jobs=2)))

    assert sweep_eps_on_linking_gold.os.environ["S2AND_BACKEND"] == "rust"
    assert sweep_eps_on_linking_gold.os.environ["OMP_NUM_THREADS"] == "2"
    assert sweep_eps_on_linking_gold.os.environ["RAYON_NUM_THREADS"] == "2"


def test_eps_sweep_cli_has_one_real_orcid_constraint_switch() -> None:
    default_args = sweep_eps_on_linking_gold.parse_args(["--dataset", "dummy", "--model-path", "model"])
    enabled_args = sweep_eps_on_linking_gold.parse_args(
        ["--dataset", "dummy", "--model-path", "model", "--use-orcid-constraints"]
    )

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


def test_distance_cache_metadata_rejects_overwritten_model_path(tmp_path) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"first model")
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
        "arrow-digest",
    )
    cache_path = tmp_path / "cache.pkl"
    with cache_path.open("wb") as outfile:
        pickle.dump({"metadata": metadata, "dist": [0.25]}, outfile)

    model_path.write_bytes(b"second model with different contents")
    expected_metadata = sweep_eps_on_linking_gold._cache_metadata(
        cast(Any, args),
        "block",
        ["s1", "s2"],
        "arrow-digest",
    )

    with pytest.raises(ValueError, match="model_"):
        sweep_eps_on_linking_gold._load_cached_distance(cache_path, expected_metadata)


def test_distance_cache_metadata_rejects_different_arrow_generation(tmp_path) -> None:
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

    expected_metadata = sweep_eps_on_linking_gold._cache_metadata(
        cast(Any, args),
        "block",
        ["s1", "s2"],
        "second-generation",
    )

    with pytest.raises(ValueError, match="arrow_generation_id"):
        sweep_eps_on_linking_gold._load_cached_distance(cache_path, expected_metadata)


def test_model_fingerprint_accepts_directory_model_path(tmp_path) -> None:
    model_path = tmp_path / "production_model_v1.21"
    (model_path / "pairwise").mkdir(parents=True)
    (model_path / "manifest.json").write_text("{}", encoding="utf-8")
    (model_path / "pairwise" / "main.lgb").write_bytes(b"model")
    args = SimpleNamespace(model_path=model_path)

    fingerprint = sweep_eps_on_linking_gold._model_fingerprint(cast(Any, args))  # noqa: SLF001

    assert fingerprint["model_path"] == str(model_path.resolve())
    assert fingerprint["model_size"] == 7
    assert isinstance(fingerprint["model_sha256"], str)
    assert len(fingerprint["model_sha256"]) == 64


def test_validate_args_requires_limit_or_full_run_for_compute_missing() -> None:
    args = sweep_eps_on_linking_gold.parse_args(
        ["--dataset", "dummy", "--model-path", "model", "--compute-missing-dists"]
    )

    with pytest.raises(ValueError, match="--max-subblocks"):
        sweep_eps_on_linking_gold._validate_args(args)  # noqa: SLF001

    limited_args = sweep_eps_on_linking_gold.parse_args(
        ["--dataset", "dummy", "--model-path", "model", "--compute-missing-dists", "--max-subblocks", "1"]
    )
    sweep_eps_on_linking_gold._validate_args(limited_args)  # noqa: SLF001

    full_run_args = sweep_eps_on_linking_gold.parse_args(
        ["--dataset", "dummy", "--model-path", "model", "--compute-missing-dists", "--allow-full-run"]
    )
    sweep_eps_on_linking_gold._validate_args(full_run_args)  # noqa: SLF001


@pytest.mark.parametrize(
    ("raw_config", "expected_neighbors", "expect_same_instance"),
    [
        (None, 16, False),
        ({"neighbors": 7}, 7, False),
        (subblocking.GraphSubblockingConfig(neighbors=5), 5, True),
    ],
)
def test_eps_sweep_uses_strict_shared_graph_config_resolver(
    monkeypatch,
    raw_config: object,
    expected_neighbors: int,
    expect_same_instance: bool,
) -> None:
    captured: dict[str, Any] = {}

    def fake_factory(
        arrow_dataset: object,
        *,
        config: subblocking.GraphSubblockingConfig,
        random_seed: int,
    ) -> object:
        del arrow_dataset, random_seed
        captured["config"] = config
        return object()

    monkeypatch.setattr(subblocking, "make_arrow_graph_subblocking_cluster_fn", fake_factory)
    clusterer = SimpleNamespace(subblocking_graph_config=raw_config, random_state=3)

    sweep_eps_on_linking_gold._make_arrow_specter_cluster_fn(
        clusterer,
        object(),
    )

    config = captured["config"]
    assert config.neighbors == expected_neighbors
    if expect_same_instance:
        assert config is raw_config


def test_eps_sweep_rejects_invalid_graph_config_type() -> None:
    clusterer = SimpleNamespace(subblocking_graph_config="invalid")

    with pytest.raises(ValueError, match="GraphSubblockingConfig, mapping, or None"):
        sweep_eps_on_linking_gold._make_arrow_specter_cluster_fn(
            clusterer,
            object(),
        )

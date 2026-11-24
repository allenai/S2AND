from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pytest

import s2and.subblocking as subblocking
from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from scripts.eps_sweep import common, sweep_eps_on_linking_gold
from tests.incremental_test_helpers import _dataset, prepare_seed_arrow_bundle
from tests.model_helpers import ConstantDistanceClassifier


def _cli(*extra: str, include_batching_threshold: bool = True) -> list[str]:
    return [
        "--dataset",
        "dummy",
        "--arrow-root",
        "arrow",
        "--model-path",
        "model",
        *(("--batching-threshold", "15000") if include_batching_threshold else ()),
        *extra,
    ]


def test_eps_sweep_requires_and_records_the_subblock_limit(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        sweep_eps_on_linking_gold.parse_args(_cli(include_batching_threshold=False))
    assert exc.value.code == 2
    assert "--batching-threshold" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        sweep_eps_on_linking_gold.parse_args(_cli("--score-mode", "safe-action-proxy"))
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err

    args = sweep_eps_on_linking_gold.parse_args(_cli())
    gold = pd.DataFrame({"query_signature_id": ["q"], "member_signature_id": ["m"], "label": [1]})
    _, plan, _ = sweep_eps_on_linking_gold._plan_subblocks(  # noqa: SLF001
        args, object(), {"block": ["q", "m"]}, gold, cast(ArrowDataset, None)
    )
    assert plan["batching_threshold"] == 15000


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
            "--batching-threshold",
            "15000",
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

    validate("--compute-missing-dists", "--max-subblocks", "1", "--route", "bulk")
    assert validate("--compute-missing-dists", "--allow-full-run").suppress_orcid_constraints is True
    args = validate("--n-jobs", "2")
    assert args.score_mode == "orcid-b3"
    assert not args.use_orcid_subblocking
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
    dataset = cast(ArrowDataset, object())
    counts = {"ch": {"cy": 16}}

    def native(arrow_dataset, signature_ids, **kwargs):
        assert arrow_dataset is dataset
        calls.append((signature_ids, kwargs))
        return {"a": ["1", "2"], "b": ["3"]}, {"graph_fallback_native": True}, {}

    monkeypatch.setattr(subblocking, "_make_subblocks_with_telemetry_arrow_rust", native)
    blocks, telemetry = sweep_eps_on_linking_gold._build_arrow_subblocked_block_dict(
        args=argparse.Namespace(batching_threshold=2, use_orcid_subblocking=orcid),
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
            args=argparse.Namespace(batching_threshold=2, use_orcid_subblocking=False),
            clusterer=SimpleNamespace(_subblocking_prefix_counts=dict),
            block_dict={"block": ["1", "2", "3"]},
            target_signature_ids={"1"},
            arrow_dataset=cast(ArrowDataset, object()),
        )


def test_eps_sweep_rejects_invalid_graph_config_type():
    with pytest.raises(ValueError, match="GraphSubblockingConfig, mapping, or None"):
        sweep_eps_on_linking_gold._build_arrow_subblocked_block_dict(
            args=argparse.Namespace(batching_threshold=2, use_orcid_subblocking=False),
            clusterer=SimpleNamespace(subblocking_graph_config="invalid"),
            block_dict={},
            target_signature_ids=set(),
            arrow_dataset=cast(ArrowDataset, object()),
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

    args = argparse.Namespace(batching_threshold=3, total_ram_bytes=1)
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


def test_distance_cache_key_reuses_pairwise_across_linker_and_eps_variants(tmp_path) -> None:
    def write_bundle(root: Path, *, pairwise_bytes: bytes, linker_bytes: bytes, eps: float) -> Path:
        (root / "pairwise").mkdir(parents=True)
        (root / "incremental_linker").mkdir(parents=True)
        (root / "reproducibility").mkdir(parents=True)
        (root / "pairwise" / "main.lgb").write_bytes(pairwise_bytes)
        (root / "incremental_linker" / "linker.lgb").write_bytes(linker_bytes)
        (root / "reproducibility" / "incremental_linker_training_target.json").write_text(
            json.dumps({"linker_len": len(linker_bytes)}), encoding="utf-8"
        )
        (root / "clusterer.json").write_text(
            json.dumps(
                {
                    "feature_contract": {"version": 1},
                    "best_params": {"eps": eps},
                    "cluster_model": {"params": {"eps": eps}},
                    "incremental_linker": {"gate_version": len(linker_bytes)},
                    "production_model_release_version": f"1.5-eps-{eps}",
                }
            ),
            encoding="utf-8",
        )
        (root / "manifest.json").write_text(json.dumps({"linker_len": len(linker_bytes)}), encoding="utf-8")
        return root

    bundle_a = write_bundle(tmp_path / "model_a", pairwise_bytes=b"pw-v1", linker_bytes=b"lnk-v1", eps=0.38)
    bundle_b = write_bundle(tmp_path / "model_b", pairwise_bytes=b"pw-v1", linker_bytes=b"lnk-v2-different", eps=0.46)
    bundle_c = write_bundle(tmp_path / "model_c", pairwise_bytes=b"pw-v2", linker_bytes=b"lnk-v1", eps=0.38)

    def meta(model_path):
        args = SimpleNamespace(
            arrow_root=tmp_path / "arrow",
            batching_threshold=10,
            dataset="dummy",
            model_path=model_path,
            pair_chunk_size=3,
            suppress_orcid_constraints=True,
            use_orcid_subblocking=False,
        )
        return sweep_eps_on_linking_gold._cache_metadata(cast(Any, args), "block", ["s1", "s2"], "gen1")

    meta_a = meta(bundle_a)
    meta_b = meta(bundle_b)
    meta_c = meta(bundle_c)
    # Pairwise fingerprint matches across linker/reproducibility/EPS variants, while full model_sha256 differs.
    assert meta_a["pairwise_model_sha256"] == meta_b["pairwise_model_sha256"]
    assert meta_a["model_sha256"] != meta_b["model_sha256"]
    assert meta_a["pairwise_model_sha256"] != meta_c["pairwise_model_sha256"]

    cache_path = tmp_path / "cache.pkl"
    with cache_path.open("wb") as outfile:
        pickle.dump({"metadata": meta_a, "dist": [0.25]}, outfile)
    # Linker/EPS/target variant at a different path reuses the cached pairwise distance matrix.
    assert sweep_eps_on_linking_gold._load_cached_distance(cache_path, meta_b) == [0.25]
    # Changing pairwise booster invalidates the cache.
    with pytest.raises(ValueError, match="pairwise_model_sha256"):
        sweep_eps_on_linking_gold._load_cached_distance(cache_path, meta_c)


def test_orcid_b3_external_labels_and_allow_all_identities(tmp_path) -> None:
    from scripts.eps_sweep import orcid_b3
    from tests.helpers import write_arrow_tables_bundle, write_test_arrow_artifact_manifest
    from tests.raw_arrow_helpers import base_arrow_paths

    raw = base_arrow_paths(tmp_path / "raw")
    with pa.OSFile(raw["signatures"], "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    rows = table.to_pylist()
    valid = "0000-0002-1694-233X"
    # Regression test (Astra #3): rows[0], rows[1], rows[2] share valid Arrow ORCID; external labels
    # only cover rows[0] and rows[1], omitting final-test rows[2].
    rows[0]["author_orcid"] = valid
    rows[1]["author_orcid"] = valid
    rows[2]["author_orcid"] = valid
    paths = write_arrow_tables_bundle(tmp_path / "raw", {"signatures": pa.Table.from_pylist(rows, schema=table.schema)})
    write_test_arrow_artifact_manifest(tmp_path / "raw", {**raw, **paths})

    labels_path = tmp_path / "gold_orcid.parquet"
    pd.DataFrame(
        {
            "signature_id": [rows[0]["signature_id"], rows[1]["signature_id"]],
            "orcid": [f"https://orcid.org/{valid}", valid],
            "dataset": ["unit", "unit"],
        }
    ).to_parquet(labels_path, index=False)
    external = orcid_b3.load_external_reference_labels(labels_path, dataset="unit")

    with ArrowDataset.open(Path(paths["signatures"]).parent) as dataset:
        # Even though rows[2] is omitted from external_labels, its Arrow ORCID excludes the whole group!
        with pytest.raises(ValueError, match="No repeated ORCID references"):
            orcid_b3.read_reference(
                dataset,
                tmp_path,
                "unit",
                test_identities=frozenset({rows[2]["signature_id"]}),
                error_records=frozenset(),
                external_labels=external,
                allow_all_identities=False,
            )

        ref = orcid_b3.read_reference(
            dataset,
            tmp_path,
            "unit",
            test_identities=frozenset(),
            error_records=frozenset(),
            external_labels=external,
            allow_all_identities=True,
        )
        assert ref.repeated == {rows[0]["signature_id"]: valid, rows[1]["signature_id"]: valid}
        assert ref.stats["reference_policy"] == orcid_b3.REFERENCE_POLICY_ALL_IDENTITIES
        assert ref.stats["reference_source"] == "external_labels"
        assert ref.stats["external_labels_matched"] == 2
        with pytest.raises(ValueError, match="allow_all_identities"):
            orcid_b3.read_reference(
                dataset,
                tmp_path,
                "unit",
                test_identities=frozenset({rows[0]["signature_id"]}),
                error_records=frozenset(),
                external_labels=external,
                allow_all_identities=True,
            )


def test_distance_cache_rejects_a_different_pairwise_contract(tmp_path: Path) -> None:
    from s2and.production_model import canonical_artifact_hashes
    from tests.promoted_linking_helpers import write_synthetic_pairwise_bundle

    bundle = tmp_path / "model"
    write_synthetic_pairwise_bundle(bundle, artifact_hashes=canonical_artifact_hashes(), release_version="1.4")
    args = SimpleNamespace(
        arrow_root=tmp_path / "arrow",
        batching_threshold=10,
        dataset="test",
        model_path=bundle,
        pair_chunk_size=3,
        suppress_orcid_constraints=True,
        use_orcid_subblocking=False,
    )
    previous = sweep_eps_on_linking_gold._cache_metadata(cast(Any, args), "block", ["1", "2"], "generation")
    cache = tmp_path / "cache.pkl"
    with cache.open("wb") as outfile:
        pickle.dump({"metadata": previous, "dist": [0.25]}, outfile)

    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["feature_contract_version"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    current = sweep_eps_on_linking_gold._cache_metadata(cast(Any, args), "block", ["1", "2"], "generation")
    with pytest.raises(ValueError, match="pairwise_model_sha256"):
        sweep_eps_on_linking_gold._load_cached_distance(cache, current)


def test_eps_sweep_rejects_sealed_dataset_before_reference_collection(tmp_path, monkeypatch) -> None:
    from contextlib import nullcontext

    from scripts.eps_sweep import orcid_b3

    roles = tmp_path / "roles.json"
    roles.write_text(
        json.dumps(
            {
                "roles": {"final_test_a": ["dummy"]},
                "block_orcids": {"dummy": []},
                "verification": {"cross_role_orcid_overlap": 0},
            }
        )
    )
    args = sweep_eps_on_linking_gold.parse_args(
        _cli(
            "--score-mode",
            "orcid-b3",
            "--allow-all-reference-identities",
            "--block-role-assignments",
            str(roles),
            "--output-dir",
            str(tmp_path / "out"),
            "--arrow-root",
            str(tmp_path),
            "--n-jobs",
            "1",
        )
    )
    sources = SimpleNamespace(validate_arrow_dataset=lambda *_: None)
    monkeypatch.setattr(sweep_eps_on_linking_gold, "load_initial_query_sources", lambda *a, **kw: sources)
    monkeypatch.setattr(sweep_eps_on_linking_gold, "arrow_dataset_dir", lambda *_: tmp_path)
    monkeypatch.setattr(ArrowDataset, "open", lambda *a, **kw: nullcontext(object()))
    monkeypatch.setattr("s2and.production_model.load_production_model", lambda _: SimpleNamespace())

    def reject_read(*args, **kwargs):
        pytest.fail("sealed identities reached reference collection")

    monkeypatch.setattr(orcid_b3, "read_reference", reject_read)
    with pytest.raises(ValueError, match="not in the frozen eps_cal pool"):
        sweep_eps_on_linking_gold.run(args)

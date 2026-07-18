from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest

from s2and.consts import FEATURIZER_VERSION, LARGE_INTEGER
from s2and.featurizer import DEFAULT_FEATURE_GROUPS, DEFAULT_NAMELESS_FEATURE_GROUPS, FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from scripts._pair_ablation import b3_cache as b3_cache_module
from scripts._pair_ablation.b3_cache import (
    B3RawFeatureStore,
    b3_cache_builder_identity,
    build_or_load_b3_raw_feature_store,
    score_b3_raw_feature_store,
)
from scripts._pair_ablation.evaluation import (
    B3EvaluationPlan,
    B3PlanBlock,
    b3_for_threshold,
    build_block_linkages,
    load_gold_block_data,
    tune_b3_threshold,
)
from scripts._pair_ablation.legacy_rust import build_legacy_rust_featurizer, resolve_legacy_arrow_manifest
from scripts._pair_ablation.modeling import load_pairwise_models
from scripts._pair_ablation.run_identity import rust_extension_binary_sha256

_TEST_RUST_EXTENSION_SHA256 = hashlib.sha256(b"test-rust-extension").hexdigest()
_TEST_CACHE_BUILDER_IDENTITY = hashlib.sha256(b"test-cache-builder").hexdigest()


def _plan() -> B3EvaluationPlan:
    return B3EvaluationPlan(
        dataset="d",
        role="heldout_test",
        evaluation_seed=1111,
        pair_budget=None,
        blocks=(
            B3PlanBlock("multi", ("s0", "s1", "s2")),
            B3PlanBlock("singleton", ("single",)),
        ),
        gold_assignments=(("s0", "c0"), ("s1", "c0"), ("s2", "c1"), ("single", "c2")),
    )


def _feature_info(indices: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        features_to_use=["selected"],
        feature_group_to_index={"selected": indices},
        featurizer_version="test-v1",
    )


def _clusterer() -> SimpleNamespace:
    return SimpleNamespace(
        featurizer_info=_feature_info([0, 1]),
        nameless_featurizer_info=_feature_info([2]),
        classifier=object(),
        nameless_classifier=object(),
        n_jobs=2,
        use_cache=False,
        use_default_constraints_as_supervision=True,
        dont_merge_cluster_seeds=True,
        suppress_orcid=False,
    )


class _FakeRustFeaturizer:
    def __init__(self, *, fail_if_called: bool = False) -> None:
        self.fail_if_called = fail_if_called
        self.constraint_calls = 0
        self.feature_calls = 0

    def signature_ids(self) -> list[str]:
        if self.fail_if_called:
            raise AssertionError("valid cache hits must not call Rust")
        return ["s0", "s1", "s2", "single"]

    def get_constraints_block_upper_triangle_indexed(
        self,
        block_signature_indices: list[int],
        start_offset: int,
        max_pairs: int,
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[list[int], list[int], list[float | None]]:
        if self.fail_if_called:
            raise AssertionError("valid cache hits must not call Rust")
        self.constraint_calls += 1
        pairs = [
            (left, right)
            for left in range(len(block_signature_indices))
            for right in range(left + 1, len(block_signature_indices))
        ][start_offset : start_offset + max_pairs]
        constraint_by_pair_offset: list[float | None] = [0.0, None, 10_000.0]
        values = constraint_by_pair_offset[start_offset : start_offset + max_pairs]
        return [left for left, _ in pairs], [right for _, right in pairs], values

    def featurize_block_upper_triangle_matrix_indexed(
        self,
        _block_signature_indices: list[int],
        start_offset: int,
        max_pairs: int,
        selected_indices: list[int],
        *_args: Any,
        **_kwargs: Any,
    ) -> np.ndarray:
        if self.fail_if_called:
            raise AssertionError("valid cache hits must not call Rust")
        self.feature_calls += 1
        output = np.empty((max_pairs, len(selected_indices)), dtype=np.float64)
        for row in range(max_pairs):
            pair_offset = start_offset + row
            output[row] = [10.0 * pair_offset + feature_index for feature_index in selected_indices]
        return output


def _build_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        b3_cache_module,
        "require_rust_featurizer_name_counts_binding_for_clusterer",
        lambda *_args, **_kwargs: None,
    )
    rust = _FakeRustFeaturizer()
    clusterer = _clusterer()
    store = build_or_load_b3_raw_feature_store(
        tmp_path,
        plan=_plan(),
        rust_featurizer=rust,
        feature_artifact_identity={"sha256": "feature-input"},
        rust_featurizer_identity={"preprocess": False, "name_tuples": "filtered"},
        clusterer=clusterer,
        rust_version="test-rust",
        rust_extension_sha256=_TEST_RUST_EXTENSION_SHA256,
        cache_builder_identity=_TEST_CACHE_BUILDER_IDENTITY,
        pair_chunk_size=2,
    )
    return store, rust, clusterer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_b3_raw_feature_cache_roundtrip_is_exact_and_cache_hit_skips_rust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, rust, clusterer = _build_store(tmp_path, monkeypatch)

    np.testing.assert_array_equal(store.main, np.asarray([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]))
    np.testing.assert_array_equal(store.nameless, np.asarray([[2.0], [12.0], [22.0]]))
    np.testing.assert_allclose(
        store.staged_labels,
        np.asarray([-LARGE_INTEGER, np.nan, 10_000.0 - LARGE_INTEGER]),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert [(block.block_key, block.row_start, block.row_stop) for block in store.layout] == [
        ("multi", 0, 3),
        ("singleton", 3, 3),
    ]
    assert rust.constraint_calls == 2
    assert rust.feature_calls == 4

    loaded = build_or_load_b3_raw_feature_store(
        tmp_path,
        plan=_plan(),
        rust_featurizer=_FakeRustFeaturizer(fail_if_called=True),
        feature_artifact_identity={"sha256": "feature-input"},
        rust_featurizer_identity={"preprocess": False, "name_tuples": "filtered"},
        clusterer=clusterer,
        rust_version="test-rust",
        rust_extension_sha256=_TEST_RUST_EXTENSION_SHA256,
        cache_builder_identity=_TEST_CACHE_BUILDER_IDENTITY,
        pair_chunk_size=1,
    )
    assert loaded.cache_digest == store.cache_digest
    np.testing.assert_array_equal(loaded.main, store.main)


def test_b3_cache_builder_identity_binds_full_implementation_and_runtime() -> None:
    implementation = {
        "s2and/model.py": hashlib.sha256(b"model").hexdigest(),
        "scripts/_pair_ablation/b3_cache.py": hashlib.sha256(b"cache").hexdigest(),
    }
    runtime = {"python": "3.11.14", "numpy": "1.26.4", "pyarrow": "21.0.0"}
    reference = b3_cache_builder_identity(
        implementation_sha256=implementation,
        runtime_versions=runtime,
    )
    reordered = b3_cache_builder_identity(
        implementation_sha256=dict(reversed(list(implementation.items()))),
        runtime_versions=dict(reversed(list(runtime.items()))),
    )
    changed_implementation = b3_cache_builder_identity(
        implementation_sha256={**implementation, "s2and/model.py": hashlib.sha256(b"changed").hexdigest()},
        runtime_versions=runtime,
    )
    changed_runtime = b3_cache_builder_identity(
        implementation_sha256=implementation,
        runtime_versions={**runtime, "pyarrow": "22.0.0"},
    )

    assert reference == reordered
    assert reference != changed_implementation
    assert reference != changed_runtime


def test_b3_cache_digest_binds_rust_binary_and_cache_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _rust, clusterer = _build_store(tmp_path, monkeypatch)
    changed_rust = build_or_load_b3_raw_feature_store(
        tmp_path,
        plan=_plan(),
        rust_featurizer=_FakeRustFeaturizer(),
        feature_artifact_identity={"sha256": "feature-input"},
        rust_featurizer_identity={"preprocess": False, "name_tuples": "filtered"},
        clusterer=clusterer,
        rust_version="test-rust",
        rust_extension_sha256=hashlib.sha256(b"changed-rust-extension").hexdigest(),
        cache_builder_identity=_TEST_CACHE_BUILDER_IDENTITY,
        pair_chunk_size=2,
    )
    changed_builder = build_or_load_b3_raw_feature_store(
        tmp_path,
        plan=_plan(),
        rust_featurizer=_FakeRustFeaturizer(),
        feature_artifact_identity={"sha256": "feature-input"},
        rust_featurizer_identity={"preprocess": False, "name_tuples": "filtered"},
        clusterer=clusterer,
        rust_version="test-rust",
        rust_extension_sha256=_TEST_RUST_EXTENSION_SHA256,
        cache_builder_identity=hashlib.sha256(b"changed-cache-builder").hexdigest(),
        pair_chunk_size=2,
    )

    assert len({first.cache_digest, changed_rust.cache_digest, changed_builder.cache_digest}) == 3
    assert first.cache_identity["rust_extension_sha256"] == _TEST_RUST_EXTENSION_SHA256
    assert first.cache_identity["cache_builder_identity"] == _TEST_CACHE_BUILDER_IDENTITY


def test_validated_store_memo_skips_repeat_validation_but_not_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        b3_cache_module,
        "require_rust_featurizer_name_counts_binding_for_clusterer",
        lambda *_args, **_kwargs: None,
    )
    load_calls = 0
    original_load = b3_cache_module._load_b3_raw_feature_store

    def counted_load(*args: Any, **kwargs: Any) -> B3RawFeatureStore:
        nonlocal load_calls
        load_calls += 1
        return original_load(*args, **kwargs)

    score_calls = 0

    def counted_score(*_args: Any, **_kwargs: Any) -> tuple[np.ndarray, float]:
        nonlocal score_calls
        score_calls += 1
        return np.zeros(3, dtype=np.float64), 0.0

    monkeypatch.setattr(b3_cache_module, "_load_b3_raw_feature_store", counted_load)
    monkeypatch.setattr(b3_cache_module.model_module, "_predict_and_combine", counted_score)
    clusterer = _clusterer()
    validated_stores: dict[str, B3RawFeatureStore] = {}
    first = build_or_load_b3_raw_feature_store(
        tmp_path,
        plan=_plan(),
        rust_featurizer=_FakeRustFeaturizer(),
        feature_artifact_identity={"sha256": "feature-input"},
        rust_featurizer_identity={"preprocess": False, "name_tuples": "filtered"},
        clusterer=clusterer,
        rust_version="test-rust",
        rust_extension_sha256=_TEST_RUST_EXTENSION_SHA256,
        cache_builder_identity=_TEST_CACHE_BUILDER_IDENTITY,
        pair_chunk_size=2,
        validated_stores=validated_stores,
    )
    second = build_or_load_b3_raw_feature_store(
        tmp_path,
        plan=_plan(),
        rust_featurizer=_FakeRustFeaturizer(fail_if_called=True),
        feature_artifact_identity={"sha256": "feature-input"},
        rust_featurizer_identity={"preprocess": False, "name_tuples": "filtered"},
        clusterer=clusterer,
        rust_version="test-rust",
        rust_extension_sha256=_TEST_RUST_EXTENSION_SHA256,
        cache_builder_identity=_TEST_CACHE_BUILDER_IDENTITY,
        validated_stores=validated_stores,
    )
    score_b3_raw_feature_store(first, clusterer=clusterer, total_ram_bytes=1024**3)
    score_b3_raw_feature_store(second, clusterer=clusterer, total_ram_bytes=1024**3)

    assert second is first
    assert load_calls == 1
    assert score_calls == 2
    assert list(validated_stores) == [first.cache_digest]


def test_validated_store_memo_rejects_wrong_store_under_cache_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, _rust, clusterer = _build_store(tmp_path, monkeypatch)
    wrong_store = replace(first, cache_dir=tmp_path / "wrong-cache-directory")

    with pytest.raises(RuntimeError, match="Memoized B3 raw-feature store identity mismatch"):
        build_or_load_b3_raw_feature_store(
            tmp_path,
            plan=_plan(),
            rust_featurizer=_FakeRustFeaturizer(fail_if_called=True),
            feature_artifact_identity={"sha256": "feature-input"},
            rust_featurizer_identity={"preprocess": False, "name_tuples": "filtered"},
            clusterer=clusterer,
            rust_version="test-rust",
            rust_extension_sha256=_TEST_RUST_EXTENSION_SHA256,
            cache_builder_identity=_TEST_CACHE_BUILDER_IDENTITY,
            validated_stores={first.cache_digest: wrong_store},
        )


@pytest.mark.parametrize("corruption", ["array_hash", "shape", "dtype", "layout"])
def test_b3_raw_feature_cache_strictly_rejects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    store, _rust, clusterer = _build_store(tmp_path, monkeypatch)
    manifest_path = store.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "array_hash":
        with (store.cache_dir / "main.npy").open("ab") as handle:
            handle.write(b"corrupt")
    elif corruption == "shape":
        manifest["arrays"]["main.npy"]["shape"] = [999, 2]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif corruption == "dtype":
        manifest["arrays"]["main.npy"]["dtype"] = "float32"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        layout_path = store.cache_dir / "layout.json"
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        layout["blocks"][0]["signatures"].reverse()
        layout_path.write_text(json.dumps(layout), encoding="utf-8")
        manifest["layout_sha256"] = _sha256(layout_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="B3 raw-feature cache"):
        build_or_load_b3_raw_feature_store(
            tmp_path,
            plan=_plan(),
            rust_featurizer=_FakeRustFeaturizer(fail_if_called=True),
            feature_artifact_identity={"sha256": "feature-input"},
            rust_featurizer_identity={"preprocess": False, "name_tuples": "filtered"},
            clusterer=clusterer,
            rust_version="test-rust",
            rust_extension_sha256=_TEST_RUST_EXTENSION_SHA256,
            cache_builder_identity=_TEST_CACHE_BUILDER_IDENTITY,
        )


def test_b3_raw_feature_cache_rejects_infinite_values_even_with_matching_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _rust, clusterer = _build_store(tmp_path, monkeypatch)
    main_path = store.cache_dir / "main.npy"
    main = np.load(main_path, mmap_mode="r+")
    main[0, 0] = np.inf
    main.flush()
    del main
    manifest_path = store.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["arrays"]["main.npy"]["sha256"] = _sha256(main_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="infinite values"):
        build_or_load_b3_raw_feature_store(
            tmp_path,
            plan=_plan(),
            rust_featurizer=_FakeRustFeaturizer(fail_if_called=True),
            feature_artifact_identity={"sha256": "feature-input"},
            rust_featurizer_identity={"preprocess": False, "name_tuples": "filtered"},
            clusterer=clusterer,
            rust_version="test-rust",
            rust_extension_sha256=_TEST_RUST_EXTENSION_SHA256,
            cache_builder_identity=_TEST_CACHE_BUILDER_IDENTITY,
        )


def test_b3_cached_scoring_calls_canonical_predict_and_combine_and_restores_block_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _rust, clusterer = _build_store(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    def fake_predict_and_combine(
        classifier: Any,
        nameless_classifier: Any,
        main: np.ndarray,
        staged_labels: np.ndarray,
        nameless: np.ndarray,
        batch_label: str,
        **kwargs: Any,
    ) -> tuple[np.ndarray, float]:
        captured.update(
            classifier=classifier,
            nameless_classifier=nameless_classifier,
            main=main,
            staged_labels=staged_labels,
            nameless=nameless,
            batch_label=batch_label,
            kwargs=kwargs,
        )
        return np.asarray([0.1, 0.2, 10_000.0]), 0.0

    monkeypatch.setattr(b3_cache_module.model_module, "_predict_and_combine", fake_predict_and_combine)
    distances = score_b3_raw_feature_store(store, clusterer=clusterer, total_ram_bytes=1024**3)

    assert captured["classifier"] is clusterer.classifier
    assert captured["nameless_classifier"] is clusterer.nameless_classifier
    assert captured["main"] is store.main
    assert captured["staged_labels"] is store.staged_labels
    assert captured["nameless"] is store.nameless
    assert captured["batch_label"] == store.cache_digest
    assert captured["kwargs"] == {"num_threads": 2, "total_ram_bytes": 1024**3}
    np.testing.assert_array_equal(distances["multi"], np.asarray([0.1, 0.2, 10_000.0]))
    assert distances["singleton"].shape == (0,)


def _manual_plan(
    dataset: str,
    role: Literal["calibration", "heldout_test"],
    block_key: str,
    signatures: list[str],
    gold: dict[str, str],
) -> B3EvaluationPlan:
    return B3EvaluationPlan(
        dataset=dataset,
        role=role,
        evaluation_seed=1111,
        pair_budget=20 if role == "calibration" else None,
        blocks=(B3PlanBlock(block_key, tuple(signatures)),),
        gold_assignments=tuple((signature_id, gold[signature_id]) for signature_id in signatures),
    )


@pytest.mark.requires_lfs
@pytest.mark.skipif(
    os.environ.get("S2AND_RUN_REAL_RUST_B3_CACHE_PARITY") != "1",
    reason="set S2AND_RUN_REAL_RUST_B3_CACHE_PARITY=1 for bounded real-Rust cache parity",
)
def test_real_rust_b3_cache_matches_direct_distance_linkage_threshold_and_b3(tmp_path: Path) -> None:
    import s2and_rust

    rust_version = s2and_rust.__version__
    if not isinstance(rust_version, str):
        raise AssertionError("s2and_rust must expose a string version")

    data_root = Path(os.environ.get("S2AND_REAL_RUST_PUBMED_ROOT", "s2and/data/pubmed"))
    pairwise_model_root = Path(
        os.environ.get("S2AND_REAL_RUST_PAIRWISE_MODEL_ROOT", "s2and/data/production_model_v1.21/pairwise")
    )
    artifacts = resolve_legacy_arrow_manifest(data_root / "manifest.json")
    gold = load_gold_block_data("pubmed", data_root / "signatures.arrow", data_root / "pubmed_clusters.json")
    candidates = sorted(
        (block_key, signatures) for block_key, signatures in gold.blocks.items() if 2 <= len(signatures) <= 6
    )
    if len(candidates) < 2:
        raise AssertionError("PubMed fixture needs two small non-singleton blocks")
    calibration = _manual_plan("pubmed", "calibration", *candidates[0], gold.cluster_by_signature)
    heldout = _manual_plan("pubmed", "heldout_test", *candidates[1], gold.cluster_by_signature)

    selected_signature_ids = [
        signature_id for plan in (calibration, heldout) for block in plan.blocks for signature_id in block.signatures
    ]
    rust_featurizer = build_legacy_rust_featurizer(
        artifacts,
        n_jobs=2,
        signature_ids=selected_signature_ids,
    )
    models = load_pairwise_models(pairwise_model_root, n_jobs=2)
    clusterer = Clusterer(
        FeaturizationInfo(features_to_use=list(DEFAULT_FEATURE_GROUPS), featurizer_version=FEATURIZER_VERSION),
        models.main,
        cluster_model=FastCluster(linkage="average"),
        n_jobs=2,
        use_cache=False,
        nameless_classifier=models.nameless,
        nameless_featurizer_info=FeaturizationInfo(
            features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS),
            featurizer_version=FEATURIZER_VERSION,
        ),
    )

    direct_distances = {}
    cached_distances = {}
    for plan in (calibration, heldout):
        direct_distances[plan.role] = clusterer.make_distance_matrices_from_rust_featurizer(
            plan.blocks_dict(),
            rust_featurizer,
            total_ram_bytes=2 * 1024**3,
        )
        store = build_or_load_b3_raw_feature_store(
            tmp_path,
            plan=plan,
            rust_featurizer=rust_featurizer,
            feature_artifact_identity={"dataset": "pubmed", "manifest_sha256": artifacts.manifest_sha256},
            rust_featurizer_identity={"adapter": "practice_legacy", "preprocess": False},
            clusterer=clusterer,
            rust_version=rust_version,
            rust_extension_sha256=rust_extension_binary_sha256(),
            cache_builder_identity=_TEST_CACHE_BUILDER_IDENTITY,
            pair_chunk_size=2,
        )
        cached_distances[plan.role] = score_b3_raw_feature_store(
            store,
            clusterer=clusterer,
            total_ram_bytes=2 * 1024**3,
        )
        for block_key in plan.blocks_dict():
            np.testing.assert_allclose(
                cached_distances[plan.role][block_key],
                direct_distances[plan.role][block_key],
                rtol=0,
                atol=1e-12,
            )

    direct_calibration_linkage = build_block_linkages(calibration.blocks_dict(), direct_distances["calibration"])
    cached_calibration_linkage = build_block_linkages(calibration.blocks_dict(), cached_distances["calibration"])
    direct_heldout_linkage = build_block_linkages(heldout.blocks_dict(), direct_distances["heldout_test"])
    cached_heldout_linkage = build_block_linkages(heldout.blocks_dict(), cached_distances["heldout_test"])
    for block_key in calibration.blocks_dict():
        direct_tree = direct_calibration_linkage[block_key].tree
        cached_tree = cached_calibration_linkage[block_key].tree
        assert direct_tree is not None and cached_tree is not None
        np.testing.assert_allclose(
            cached_tree,
            direct_tree,
            rtol=0,
            atol=1e-12,
        )
    for block_key in heldout.blocks_dict():
        direct_tree = direct_heldout_linkage[block_key].tree
        cached_tree = cached_heldout_linkage[block_key].tree
        assert direct_tree is not None and cached_tree is not None
        np.testing.assert_allclose(
            cached_tree,
            direct_tree,
            rtol=0,
            atol=1e-12,
        )

    thresholds = [0.3, 0.5, 0.7]
    direct_threshold, _ = tune_b3_threshold(
        {"pubmed": direct_calibration_linkage},
        {"pubmed": calibration.blocks_dict()},
        {"pubmed": calibration.gold_dict()},
        thresholds,
    )
    cached_threshold, _ = tune_b3_threshold(
        {"pubmed": cached_calibration_linkage},
        {"pubmed": calibration.blocks_dict()},
        {"pubmed": calibration.gold_dict()},
        thresholds,
    )
    assert cached_threshold == pytest.approx(direct_threshold, rel=0, abs=1e-12)
    direct_b3 = b3_for_threshold(
        {"pubmed": direct_heldout_linkage},
        {"pubmed": heldout.blocks_dict()},
        {"pubmed": heldout.gold_dict()},
        direct_threshold,
    )
    cached_b3 = b3_for_threshold(
        {"pubmed": cached_heldout_linkage},
        {"pubmed": heldout.blocks_dict()},
        {"pubmed": heldout.gold_dict()},
        cached_threshold,
    )
    np.testing.assert_allclose(cached_b3, direct_b3, rtol=0, atol=1e-12)

from __future__ import annotations

import importlib.util
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import pytest

from s2and.consts import PROJECT_ROOT_PATH
from s2and.data import ANDData, NameCounts
from s2and.featurizer import FeaturizationInfo


def _import_s2and_rust():
    try:
        import s2and_rust

        rust_featurizer = getattr(s2and_rust, "RustFeaturizer", None)
        if rust_featurizer is None or not hasattr(rust_featurizer, "from_json_paths"):
            return False, None
        return True, s2and_rust
    except Exception:
        return False, None


HAS_FROM_JSON_PATHS, s2and_rust = _import_s2and_rust()
if not HAS_FROM_JSON_PATHS:
    pytest.skip("s2and_rust RustFeaturizer.from_json_paths is unavailable", allow_module_level=True)

_FEATURIZATION_INFO = FeaturizationInfo()

EXPECTED_STAGE_SECONDS_KEYS = {
    "json_parse_seconds",
    "paper_preprocess_seconds",
    "reference_counter_seconds",
    "signature_preprocess_seconds",
    "cluster_seed_seconds",
}
EXPECTED_CALLBACK_COUNT_KEYS = {
    "normalize_text_calls",
    "split_first_middle_hyphen_aware_calls",
    "compute_block_calls",
    "detect_language_calls",
    "get_text_ngrams_calls",
    "get_text_ngrams_words_calls",
}


def _load_stress_module():
    script_path = Path(PROJECT_ROOT_PATH) / "scripts" / "rust_suite.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing rust suite script: {script_path}")
    spec = importlib.util.spec_from_file_location("rust_suite", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _force_python_backend(monkeypatch):
    """Ensure all tests run with the Python backend and skip fastText."""
    monkeypatch.setenv("S2AND_BACKEND", "python")
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")


def _equalish(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-3) -> bool:
    if math.isnan(float(a)) and math.isnan(float(b)):
        return True
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def _load_dataset_from_dir(data_dir: str, name: str, *, compute_reference_features: bool) -> ANDData:
    cluster_seeds_path = os.path.join(data_dir, "cluster_seeds.json")
    cluster_seeds = cluster_seeds_path if os.path.exists(cluster_seeds_path) else None
    return ANDData(
        signatures=os.path.join(data_dir, "signatures.json"),
        papers=os.path.join(data_dir, "papers.json"),
        name=name,
        mode="train",
        specter_embeddings=None,
        clusters=os.path.join(data_dir, "clusters.json"),
        cluster_seeds=cluster_seeds,
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=1000,
        val_pairs_size=1000,
        test_pairs_size=1000,
        n_jobs=1,
        load_name_counts=False,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=False,
        compute_reference_features=compute_reference_features,
    )


def _sample_pairs(signature_ids: list[str], count: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    while len(pairs) < count and len(signature_ids) >= 2:
        s1 = rng.choice(signature_ids)
        s2 = rng.choice(signature_ids)
        if s1 == s2:
            continue
        pairs.append((s1, s2))
    return pairs


def _build_rust_from_json_paths(data_dir: str, *, compute_reference_features: bool):
    signatures_path = os.path.join(data_dir, "signatures.json")
    papers_path = os.path.join(data_dir, "papers.json")
    clusters_path = os.path.join(data_dir, "clusters.json")
    cluster_seeds_path = os.path.join(data_dir, "cluster_seeds.json")
    cluster_seeds_path_arg = cluster_seeds_path if os.path.exists(cluster_seeds_path) else None
    return s2and_rust.RustFeaturizer.from_json_paths(
        signatures_path,
        papers_path,
        clusters_path,
        cluster_seeds_path_arg,
        None,  # specter_embeddings_path
        None,  # name_tuples_path
        None,  # name_counts_path
        True,  # preprocess
        compute_reference_features,
        0.0,
        10000.0,
        1,
    )


def test_from_json_paths_feature_parity_vs_from_dataset_dummy():
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    dataset = _load_dataset_from_dir(data_dir, "dummy_from_json_parity", compute_reference_features=False)
    rust_from_dataset = s2and_rust.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_from_json = _build_rust_from_json_paths(data_dir, compute_reference_features=False)

    pairs = _sample_pairs(list(dataset.signatures.keys()), count=12, seed=1337)
    for s1, s2 in pairs:
        ref_features = rust_from_dataset.featurize_pair(s1, s2)
        got_features = rust_from_json.featurize_pair(s1, s2)
        assert len(ref_features) == len(got_features)
        for idx, (ref, got) in enumerate(zip(ref_features, got_features, strict=False)):
            assert _equalish(ref, got), f"Mismatch idx={idx} pair=({s1},{s2}) ref={ref} got={got}"


def test_from_json_paths_constraint_parity_vs_from_dataset_dummy():
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    dataset = _load_dataset_from_dir(data_dir, "dummy_from_json_constraints", compute_reference_features=False)
    rust_from_dataset = s2and_rust.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_from_json = _build_rust_from_json_paths(data_dir, compute_reference_features=False)

    pairs = _sample_pairs(list(dataset.signatures.keys()), count=10, seed=17)
    pairs.extend(list(dataset.cluster_seeds_disallow)[:5])
    by_cluster: dict[object, list[str]] = defaultdict(list)
    for sig_id, cluster_id in dataset.cluster_seeds_require.items():
        by_cluster[cluster_id].append(sig_id)
    for sig_ids in by_cluster.values():
        if len(sig_ids) >= 2:
            pairs.append((sig_ids[0], sig_ids[1]))
    deduped_pairs = []
    seen = set()
    for pair in pairs:
        if pair[0] == pair[1]:
            continue
        key = tuple(sorted(pair))
        if key in seen:
            continue
        seen.add(key)
        deduped_pairs.append(pair)

    for s1, s2 in deduped_pairs[:20]:
        ref_constraint = rust_from_dataset.get_constraint(s1, s2)
        got_constraint = rust_from_json.get_constraint(s1, s2)
        assert ref_constraint == got_constraint


def test_from_json_paths_reference_feature_parity_vs_from_dataset_qian():
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "qian")
    if not os.path.exists(os.path.join(data_dir, "signatures.json")):
        pytest.skip("qian fixture unavailable")

    dataset = _load_dataset_from_dir(data_dir, "qian_from_json_parity", compute_reference_features=True)
    rust_from_dataset = s2and_rust.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_from_json = _build_rust_from_json_paths(data_dir, compute_reference_features=True)

    pairs = _sample_pairs(list(dataset.signatures.keys()), count=10, seed=99)
    reference_feature_indices = set(_FEATURIZATION_INFO.feature_group_to_index["reference_features"])
    for s1, s2 in pairs:
        ref_features = rust_from_dataset.featurize_pair(s1, s2)
        got_features = rust_from_json.featurize_pair(s1, s2)
        assert len(ref_features) == len(got_features)
        for idx, (ref, got) in enumerate(zip(ref_features, got_features, strict=False)):
            if idx not in reference_feature_indices:
                continue
            assert _equalish(ref, got), f"Reference mismatch idx={idx} pair=({s1},{s2}) ref={ref} got={got}"


def test_from_json_paths_emits_telemetry_payload_dummy():
    reset_fn = getattr(s2and_rust, "reset_last_json_ingest_telemetry", None)
    get_fn = getattr(s2and_rust, "get_last_json_ingest_telemetry", None)
    if not callable(reset_fn) or not callable(get_fn):
        pytest.skip("json ingest telemetry helpers unavailable")

    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")

    reset_fn()
    _build_rust_from_json_paths(data_dir, compute_reference_features=False)
    telemetry = get_fn()

    assert isinstance(telemetry, dict)
    stage_seconds = telemetry.get("stage_seconds")
    callback_counts = telemetry.get("callback_counts")
    assert isinstance(stage_seconds, dict)
    assert isinstance(callback_counts, dict)
    assert EXPECTED_STAGE_SECONDS_KEYS.issubset(stage_seconds.keys())
    assert EXPECTED_CALLBACK_COUNT_KEYS.issubset(callback_counts.keys())
    for key in EXPECTED_STAGE_SECONDS_KEYS:
        assert float(stage_seconds[key]) >= 0.0
    for key in EXPECTED_CALLBACK_COUNT_KEYS:
        assert int(callback_counts[key]) >= 0
    assert int(callback_counts["normalize_text_calls"]) == 0
    assert int(callback_counts["split_first_middle_hyphen_aware_calls"]) == 0
    assert int(callback_counts["compute_block_calls"]) == 0
    assert int(callback_counts["detect_language_calls"]) == 0
    assert int(callback_counts["get_text_ngrams_calls"]) == 0
    assert int(callback_counts["get_text_ngrams_words_calls"]) == 0


def test_from_json_paths_signature_name_counts_overlay_parity_dummy():
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    dataset = _load_dataset_from_dir(data_dir, "dummy_from_json_name_counts_overlay", compute_reference_features=False)

    # Inject deterministic per-signature count tuples without relying on external artifacts.
    for idx, sig_id in enumerate(sorted(dataset.signatures.keys()), start=1):
        signature = dataset.signatures[sig_id]
        dataset.signatures[sig_id] = signature._replace(
            author_info_name_counts=NameCounts(
                first=float(10 + idx),
                last=float(20 + idx),
                first_last=float(30 + idx),
                last_first_initial=float(40 + idx),
            )
        )

    rust_from_dataset = s2and_rust.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_from_json = _build_rust_from_json_paths(data_dir, compute_reference_features=False)
    if not hasattr(rust_from_json, "update_signature_name_counts"):
        pytest.skip("RustFeaturizer.update_signature_name_counts is unavailable")

    updated = rust_from_json.update_signature_name_counts(dataset.signatures)
    assert updated == len(dataset.signatures)

    pairs = _sample_pairs(list(dataset.signatures.keys()), count=12, seed=123)
    name_count_indices = _FEATURIZATION_INFO.feature_group_to_index["name_counts"]
    for s1, s2 in pairs:
        ref_features = rust_from_dataset.featurize_pair(s1, s2)
        got_features = rust_from_json.featurize_pair(s1, s2)
        for idx in name_count_indices:
            ref_val = ref_features[idx]
            got_val = got_features[idx]
            assert _equalish(
                ref_val, got_val
            ), f"Name-count mismatch idx={idx} pair=({s1},{s2}) ref={ref_val} got={got_val}"


@pytest.mark.parametrize("build_path", ["from_json_paths", "from_dataset"])
def test_repeated_rust_featurizer_rebuild_dummy_smoke(build_path, tmp_path):
    stress_module = _load_stress_module()
    output_path = tmp_path / f"stress_{build_path}_dummy.json"
    result = stress_module.run_rebuild_stress(
        dataset="dummy",
        build_path=build_path,
        repeats=3,
        num_threads=1,
        write_json=str(output_path),
    )

    assert result["dataset"] == "dummy"
    assert result["build_path"] == build_path
    assert result["success_count"] == 3
    assert result["failure_count"] == 0
    assert output_path.exists()
    assert len(result["iterations"]) == 3
    assert all(iteration["status"] == "ok" for iteration in result["iterations"])
    assert "rss_peak_gb_by_iteration" in result
    assert len(result["rss_peak_gb_by_iteration"]) == 3
    assert "rss_growth_fraction" in result
    assert all("rss_peak_gb" in iteration for iteration in result["iterations"])


@pytest.mark.skipif(
    os.environ.get("S2AND_RUN_HEAVY_RUST_STRESS", "0") != "1",
    reason="Set S2AND_RUN_HEAVY_RUST_STRESS=1 to run AMiner rebuild stress.",
)
def test_repeated_from_json_paths_aminer_opt_in(tmp_path):
    aminer_signatures = Path(PROJECT_ROOT_PATH) / "data" / "aminer" / "aminer_signatures.json"
    if not aminer_signatures.exists():
        pytest.skip(f"AMiner signatures fixture unavailable: {aminer_signatures}")

    stress_module = _load_stress_module()
    output_path = tmp_path / "stress_rust_from_json_paths_aminer.json"
    result = stress_module.run_rebuild_stress(
        dataset="aminer",
        build_path="from_json_paths",
        repeats=6,
        num_threads=1,
        write_json=str(output_path),
    )

    assert result["dataset"] == "aminer"
    assert result["build_path"] == "from_json_paths"
    assert result["success_count"] == 6
    assert result["failure_count"] == 0
    assert output_path.exists()

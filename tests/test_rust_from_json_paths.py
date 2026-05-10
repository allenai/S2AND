from __future__ import annotations

import importlib.util
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pytest

from s2and.consts import PROJECT_ROOT_PATH
from s2and.data import ANDData, NameCounts
from s2and.featurizer import FeaturizationInfo
from tests.helpers import equalish, import_s2and_rust

HAS_FROM_JSON_PATHS, s2and_rust = import_s2and_rust(required_method="from_json_paths")
if not HAS_FROM_JSON_PATHS:
    raise pytest.skip.Exception("s2and_rust RustFeaturizer.from_json_paths is unavailable", allow_module_level=True)
assert s2and_rust is not None and not isinstance(s2and_rust, Exception)
_S2AND_RUST = cast(Any, s2and_rust)

_FEATURIZATION_INFO = FeaturizationInfo()


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


def _build_rust_from_json_paths(
    data_dir: str,
    *,
    compute_reference_features: bool,
    specter_embeddings: dict[str, list[float]] | None = None,
):
    signatures_path = os.path.join(data_dir, "signatures.json")
    papers_path = os.path.join(data_dir, "papers.json")
    cluster_seeds_path = os.path.join(data_dir, "cluster_seeds.json")
    cluster_seeds_path_arg = cluster_seeds_path if os.path.exists(cluster_seeds_path) else None
    return _S2AND_RUST.RustFeaturizer.from_json_paths(
        signatures_path,
        papers_path,
        cluster_seeds_path_arg,
        specter_embeddings,
        None,  # name_tuples_path
        None,  # name_counts_path
        True,  # preprocess
        compute_reference_features,
        0.0,
        10000.0,
        1,
    )


def test_from_json_paths_language_matches_python_cld2_detail_for_beta_title(tmp_path: Path):
    data_dir = tmp_path / "beta_language"
    data_dir.mkdir()
    signatures = {
        "q": {
            "signature_id": "q",
            "paper_id": 1,
            "given_name": "Seoyeon Lee",
            "sourced_author_ids": [],
            "sourced_author_source": None,
            "author_info": {
                "first": "Seoyeon",
                "middle": None,
                "last": "Lee",
                "suffix": None,
                "position": 0,
                "email": None,
                "affiliations": [],
                "block": "s lee",
                "given_block": "s lee",
                "estimated_ethnicity": None,
                "estimated_gender": None,
            },
        },
        "m": {
            "signature_id": "m",
            "paper_id": 2,
            "given_name": "Seo-Young Lee",
            "sourced_author_ids": [],
            "sourced_author_source": None,
            "author_info": {
                "first": "Seo-Young",
                "middle": None,
                "last": "Lee",
                "suffix": None,
                "position": 0,
                "email": None,
                "affiliations": [],
                "block": "s lee",
                "given_block": "s lee",
                "estimated_ethnicity": None,
                "estimated_gender": None,
            },
        },
    }
    papers = {
        "1": {
            "paper_id": 1,
            "title": "Molecular programs of fibrotic change in aging human lung",
            "abstract": "",
            "journal_name": None,
            "venue": None,
            "year": 2024,
            "sources": [],
            "fields_of_study": [],
            "authors": [{"position": 0, "author_name": "Seoyeon Lee"}],
            "references": [],
        },
        "2": {
            "paper_id": 2,
            "title": (
                "Fibroblast TGF-\u03b2 signaling defines spatial tumor ecosystems linked to "
                "immune checkpoint blockade resistance"
            ),
            "abstract": "",
            "journal_name": None,
            "venue": None,
            "year": 2024,
            "sources": [],
            "fields_of_study": [],
            "authors": [{"position": 0, "author_name": "Seo-Young Lee"}],
            "references": [],
        },
    }
    clusters = {"1": {"cluster_id": "1", "signature_ids": ["q", "m"], "model_version": -1}}
    (data_dir / "signatures.json").write_text(json.dumps(signatures), encoding="utf-8")
    (data_dir / "papers.json").write_text(json.dumps(papers), encoding="utf-8")
    (data_dir / "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")

    dataset = _load_dataset_from_dir(str(data_dir), "beta_language_from_dataset", compute_reference_features=False)
    rust_from_dataset = _S2AND_RUST.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_from_json = _build_rust_from_json_paths(
        str(data_dir),
        compute_reference_features=False,
        specter_embeddings={"1": [1.0, 0.0], "2": [0.0, 1.0]},
    )
    ref_features = rust_from_dataset.featurize_pair("q", "m")
    got_features = rust_from_json.featurize_pair("q", "m")
    feature_names = _FEATURIZATION_INFO.get_feature_names()

    for feature_name in ("english_count", "same_language", "language_reliability_count"):
        feature_index = feature_names.index(feature_name)
        assert got_features[feature_index] == ref_features[feature_index]
    assert ref_features[feature_names.index("english_count")] == 1
    assert ref_features[feature_names.index("same_language")] == 0
    assert ref_features[feature_names.index("language_reliability_count")] == 2
    assert math.isnan(got_features[feature_names.index("specter_cosine_sim")])


def test_from_json_paths_reports_default_and_missing_input_counts(tmp_path: Path):
    data_dir = tmp_path / "default_counts"
    data_dir.mkdir()
    signatures = {
        "q": {
            "signature_id": "q",
            "paper_id": 1,
            "author_info": {
                "first": "Alice",
                "middle": None,
                "last": "Smith",
                "suffix": None,
                "email": None,
                "affiliations": [],
            },
        },
        "m": {
            "signature_id": "m",
            "paper_id": 2,
            "author_info": {
                "first": "Bob",
                "middle": None,
                "last": "Jones",
                "suffix": None,
                "position": 0,
                "email": None,
                "affiliations": [],
            },
        },
    }
    papers = {
        "1": {
            "paper_id": 1,
            "title": "Shared topic one",
            "abstract": "",
            "journal_name": None,
            "venue": None,
            "year": 2024,
            "authors": [{"author_name": "Alice Smith"}],
            "references": [],
        },
        "2": {
            "paper_id": 2,
            "title": "Shared topic two",
            "abstract": "",
            "journal_name": None,
            "venue": None,
            "year": 2024,
            "authors": [{"position": 0, "author_name": "Bob Jones"}],
            "references": [],
        },
    }
    name_counts = {
        "normalization_version": "legacy_compat",
        "first_dict": {"alice": 5.0, "bob": 6.0},
        "last_dict": {"jones": 7.0},
        "first_last_dict": {"bob jones": 8.0},
        "last_first_initial_dict": {"smith a": 9.0, "jones b": 10.0},
    }
    (data_dir / "signatures.json").write_text(json.dumps(signatures), encoding="utf-8")
    (data_dir / "papers.json").write_text(json.dumps(papers), encoding="utf-8")
    name_counts_path = data_dir / "name_counts.json"
    name_counts_path.write_text(json.dumps(name_counts), encoding="utf-8")

    _S2AND_RUST.RustFeaturizer.from_json_paths(
        str(data_dir / "signatures.json"),
        str(data_dir / "papers.json"),
        None,
        {"1": [1.0, 0.0]},
        None,
        str(name_counts_path),
        True,
        False,
        0.0,
        10000.0,
        1,
        "legacy_compat",
        False,
    )

    telemetry = _S2AND_RUST.get_last_json_ingest_telemetry()
    assert telemetry is not None
    counts = telemetry["counts"]
    assert counts["missing_specter_paper_count"] == 1
    assert counts["defaulted_signature_author_position_count"] == 1
    assert counts["defaulted_paper_author_position_count"] == 1
    assert counts["defaulted_name_count_signature_count"] == 1
    assert counts["defaulted_name_count_first_count"] == 0
    assert counts["defaulted_name_count_first_last_count"] == 1
    assert counts["defaulted_name_count_last_count"] == 1
    assert counts["defaulted_name_count_last_first_initial_count"] == 0


def test_from_json_paths_feature_parity_vs_from_dataset_dummy():
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    dataset = _load_dataset_from_dir(data_dir, "dummy_from_json_parity", compute_reference_features=False)
    rust_from_dataset = _S2AND_RUST.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_from_json = _build_rust_from_json_paths(data_dir, compute_reference_features=False)

    pairs = _sample_pairs(list(dataset.signatures.keys()), count=12, seed=1337)
    for s1, s2 in pairs:
        ref_features = rust_from_dataset.featurize_pair(s1, s2)
        got_features = rust_from_json.featurize_pair(s1, s2)
        assert len(ref_features) == len(got_features)
        for idx, (ref, got) in enumerate(zip(ref_features, got_features, strict=True)):
            assert equalish(ref, got), f"Mismatch idx={idx} pair=({s1},{s2}) ref={ref} got={got}"


def test_from_json_paths_constraint_parity_vs_from_dataset_dummy():
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    dataset = _load_dataset_from_dir(data_dir, "dummy_from_json_constraints", compute_reference_features=False)
    rust_from_dataset = _S2AND_RUST.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
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
        raise pytest.skip.Exception("qian fixture unavailable")

    dataset = _load_dataset_from_dir(data_dir, "qian_from_json_parity", compute_reference_features=True)
    rust_from_dataset = _S2AND_RUST.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_from_json = _build_rust_from_json_paths(data_dir, compute_reference_features=True)

    pairs = _sample_pairs(list(dataset.signatures.keys()), count=10, seed=99)
    reference_feature_indices = set(_FEATURIZATION_INFO.feature_group_to_index["reference_features"])
    for s1, s2 in pairs:
        ref_features = rust_from_dataset.featurize_pair(s1, s2)
        got_features = rust_from_json.featurize_pair(s1, s2)
        assert len(ref_features) == len(got_features)
        for idx, (ref, got) in enumerate(zip(ref_features, got_features, strict=True)):
            if idx not in reference_feature_indices:
                continue
            assert equalish(ref, got), f"Reference mismatch idx={idx} pair=({s1},{s2}) ref={ref} got={got}"


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

    rust_from_dataset = _S2AND_RUST.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)
    rust_from_json = _build_rust_from_json_paths(data_dir, compute_reference_features=False)
    if not hasattr(rust_from_json, "update_signature_name_counts"):
        raise pytest.skip.Exception("RustFeaturizer.update_signature_name_counts is unavailable")

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
            assert equalish(
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


@pytest.mark.heavy
@pytest.mark.skip(reason="Run explicitly with: uv run pytest -m heavy")
def test_repeated_from_json_paths_aminer_opt_in(tmp_path):
    aminer_signatures = Path(PROJECT_ROOT_PATH) / "data" / "aminer" / "aminer_signatures.json"
    if not aminer_signatures.exists():
        raise pytest.skip.Exception(f"AMiner signatures fixture unavailable: {aminer_signatures}")

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

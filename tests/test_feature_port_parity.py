import os
import random
import math
from collections import defaultdict
from contextlib import contextmanager

import pytest
import numpy as np

import s2and.featurizer as featurizer_mod
from s2and.featurizer import _single_pair_featurize
from s2and.data import ANDData
from s2and.consts import PROJECT_ROOT_PATH
from s2and.feature_port import (
    featurize_pair_rust,
    get_constraint_rust,
    _get_rust_featurizer,
    update_rust_cluster_seeds,
)


def _import_s2and_rust():
    try:
        import s2and_rust  # noqa: F401

        return True, None
    except Exception as e:
        # If local source shadowed the installed extension, retry from site-packages.
        try:
            import importlib
            import importlib.util
            from importlib.machinery import PathFinder

            import sys

            sys.modules.pop("s2and_rust", None)
            sys.modules.pop("s2and_rust.s2and_rust", None)
            site_paths = [p for p in sys.path if "site-packages" in p]
            spec = PathFinder.find_spec("s2and_rust", site_paths)
            if spec is None or spec.loader is None:
                raise e
            module = importlib.util.module_from_spec(spec)
            sys.modules["s2and_rust"] = module
            spec.loader.exec_module(module)
            return True, None
        except Exception as e2:
            return False, e2


HAS_RUST, _RUST_IMPORT_ERROR = _import_s2and_rust()
print("s2and_rust import OK" if HAS_RUST else f"s2and_rust import FAILED: {_RUST_IMPORT_ERROR}")


def _equalish(a, b, rel_tol=1e-6, abs_tol=1e-3):
    a_val = float(a)
    b_val = float(b)
    if math.isnan(a_val) and math.isnan(b_val):
        return True
    return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)


def _paper_for_sig(dataset, sig_id):
    sig = dataset.signatures[sig_id]
    return dataset.papers[str(sig.paper_id)]


def _find_signatures(dataset, predicate, limit=2):
    out = []
    for sig_id in dataset.signatures.keys():
        paper = _paper_for_sig(dataset, sig_id)
        if predicate(sig_id, paper):
            out.append(sig_id)
            if len(out) >= limit:
                break
    return out


def _specter_vec(dataset, paper_id):
    if dataset.specter_embeddings is None:
        return None
    vec = dataset.specter_embeddings.get(str(paper_id))
    if vec is None:
        vec = dataset.specter_embeddings.get(paper_id)
    if vec is None:
        return None
    if np.all(vec == 0):
        return None
    return vec


def _ref_details_empty(paper):
    if paper.reference_details is None:
        return False
    return all(len(counter) == 0 for counter in paper.reference_details)


def _ref_details_nonempty(paper, idx=None):
    if paper.reference_details is None:
        return False
    if idx is None:
        return any(len(counter) > 0 for counter in paper.reference_details)
    return len(paper.reference_details[idx]) > 0


@contextmanager
def _temporary_cluster_seeds(dataset, require_map, disallow_set):
    orig_require = dataset.cluster_seeds_require
    orig_disallow = dataset.cluster_seeds_disallow
    dataset.cluster_seeds_require = require_map
    dataset.cluster_seeds_disallow = disallow_set
    if HAS_RUST:
        update_rust_cluster_seeds(dataset)
    try:
        yield
    finally:
        dataset.cluster_seeds_require = orig_require
        dataset.cluster_seeds_disallow = orig_disallow
        if HAS_RUST:
            update_rust_cluster_seeds(dataset)


def _load_dataset_from_dir(data_dir, name, *, compute_reference_features=False):
    cluster_seeds_path = os.path.join(data_dir, "cluster_seeds.json")
    cluster_seeds = cluster_seeds_path if os.path.exists(cluster_seeds_path) else None
    ds = ANDData(
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
        train_pairs_size=100000,
        val_pairs_size=10000,
        test_pairs_size=10000,
        n_jobs=1,
        load_name_counts=True,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=False,
        compute_reference_features=compute_reference_features,
    )
    return ds


def _attach_fake_specter_embeddings(ds, max_papers=2, dim=8):
    rng = np.random.RandomState(123)
    if ds.specter_embeddings is None:
        ds.specter_embeddings = {}
    added = 0
    for sig_id in ds.signatures.keys():
        paper = _paper_for_sig(ds, sig_id)
        if paper.predicted_language in {"en", "un"}:
            paper_id = str(paper.paper_id)
            if paper_id not in ds.specter_embeddings:
                ds.specter_embeddings[paper_id] = rng.normal(size=(dim,)).astype(np.float32)
                added += 1
                if added >= max_papers:
                    break
    return ds


@pytest.fixture(scope="session")
def dataset():
    # Speed/safety: skip fastText (optional) to avoid large model loads in tests
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    # Force python reference featurizer for parity tests
    os.environ["S2AND_USE_RUST_FEATURIZER"] = "0"
    # Avoid disk-cached Rust featurizer snapshots in parity tests
    os.environ.setdefault("S2AND_RUST_FEATURIZER_DISK_CACHE", "0")

    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    ds = _load_dataset_from_dir(data_dir, "dummy")
    ds = _attach_fake_specter_embeddings(ds)
    # set global for _single_pair_featurize
    featurizer_mod.global_dataset = ds  # type: ignore
    return ds


@pytest.fixture(scope="session")
def dataset_with_refs():
    # Speed/safety: skip fastText (optional) to avoid large model loads in tests
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    # Force python reference featurizer for parity tests
    os.environ.setdefault("S2AND_USE_RUST_FEATURIZER", "0")
    # Avoid disk-cached Rust featurizer snapshots in parity tests
    os.environ.setdefault("S2AND_RUST_FEATURIZER_DISK_CACHE", "0")

    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "qian")
    ds = _load_dataset_from_dir(data_dir, "qian", compute_reference_features=True)
    return ds


@pytest.fixture(scope="session")
def sample_pairs(dataset):
    rng = random.Random(123)
    sig_ids = list(dataset.signatures.keys())
    pairs = []
    while len(pairs) < 10:
        s1 = rng.choice(sig_ids)
        s2 = rng.choice(sig_ids)
        if s1 == s2:
            continue
        pairs.append((s1, s2))
    return pairs


@pytest.fixture(scope="session")
def constraint_pairs(dataset, sample_pairs):
    pairs = list(sample_pairs)
    seen = set(pairs)

    # Add a few disallow pairs if present
    for a, b in list(dataset.cluster_seeds_disallow)[:5]:
        if (a, b) not in seen and (b, a) not in seen and a != b:
            pairs.append((a, b))
            seen.add((a, b))

    # Add a few require pairs (same cluster id) and cross-cluster pairs
    by_cluster = defaultdict(list)
    for sig_id, cluster_id in dataset.cluster_seeds_require.items():
        by_cluster[cluster_id].append(sig_id)

    for sigs in by_cluster.values():
        if len(sigs) >= 2:
            pair = (sigs[0], sigs[1])
            if pair not in seen and pair[::-1] not in seen:
                pairs.append(pair)
                seen.add(pair)

    cluster_groups = [sigs for sigs in by_cluster.values() if len(sigs) > 0]
    if len(cluster_groups) >= 2:
        pair = (cluster_groups[0][0], cluster_groups[1][0])
        if pair not in seen and pair[::-1] not in seen:
            pairs.append(pair)

    return pairs


def test_rust_extension_available():
    assert HAS_RUST, f"s2and_rust not available: {_RUST_IMPORT_ERROR}"

def test_featurize_pair_rust_parity(dataset, sample_pairs):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    for s1, s2 in sample_pairs:
        ref_features, _ = _single_pair_featurize((s1, s2))
        rust_features = featurize_pair_rust(dataset, s1, s2)
        assert len(ref_features) == len(rust_features)
        for idx, (ref_val, got_val) in enumerate(zip(ref_features, rust_features)):
            assert _equalish(ref_val, got_val), (
                f"Featurize pair mismatch at index {idx} for pair {s1},{s2}: "
                f"ref={ref_val}, got={got_val}"
            )


def test_get_constraint_rust_parity(dataset, constraint_pairs):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    for s1, s2 in constraint_pairs:
        ref_val = dataset.get_constraint(s1, s2)
        got_val = get_constraint_rust(dataset, s1, s2)
        if ref_val is None or got_val is None:
            assert ref_val is None and got_val is None
        else:
            assert ref_val == got_val, f"Constraint mismatch for pair {s1},{s2}: ref={ref_val}, got={got_val}"


def test_featurize_pairs_rust_batch_parity(dataset, sample_pairs):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    featurizer_mod.global_dataset = dataset  # type: ignore
    rust_featurizer = _get_rust_featurizer(dataset)
    rust_features = rust_featurizer.featurize_pairs(sample_pairs)

    assert len(rust_features) == len(sample_pairs)
    for (s1, s2), rust_vec in zip(sample_pairs, rust_features):
        ref_vec, _ = _single_pair_featurize((s1, s2))
        assert len(ref_vec) == len(rust_vec)
        for idx, (ref_val, got_val) in enumerate(zip(ref_vec, rust_vec)):
            assert _equalish(ref_val, got_val), (
                f"Batch featurize mismatch at index {idx} for pair {s1},{s2}: "
                f"ref={ref_val}, got={got_val}"
            )


def test_featurize_pair_rust_parity_reference_details_empty(dataset_with_refs):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    featurizer_mod.global_dataset = dataset_with_refs  # type: ignore
    empty_sigs = _find_signatures(dataset_with_refs, lambda _s, p: _ref_details_empty(p), limit=1)
    if not empty_sigs:
        pytest.skip("No papers with empty reference_details counters found")
    s1 = empty_sigs[0]
    # pick a distinct second signature
    s2 = next(sig_id for sig_id in dataset_with_refs.signatures.keys() if sig_id != s1)

    ref_features, _ = _single_pair_featurize((s1, s2))
    rust_features = featurize_pair_rust(dataset_with_refs, s1, s2)
    # Ensure reference branch executed in Python (self-citation is computed even with empty refs)
    assert not math.isnan(ref_features[20])
    for idx, (ref_val, got_val) in enumerate(zip(ref_features, rust_features)):
        assert _equalish(ref_val, got_val), (
            f"Reference-empty mismatch at index {idx} for pair {s1},{s2}: ref={ref_val}, got={got_val}"
        )


def test_featurize_pair_rust_parity_reference_details_nonempty(dataset_with_refs):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    featurizer_mod.global_dataset = dataset_with_refs  # type: ignore
    sigs = _find_signatures(dataset_with_refs, lambda _s, p: _ref_details_nonempty(p, idx=0), limit=2)
    if len(sigs) < 2:
        pytest.skip("Not enough papers with non-empty reference author counters found")
    s1, s2 = sigs[0], sigs[1]

    ref_features, _ = _single_pair_featurize((s1, s2))
    rust_features = featurize_pair_rust(dataset_with_refs, s1, s2)
    assert not math.isnan(ref_features[16]), "Reference author overlap should be computed for non-empty counters"
    for idx, (ref_val, got_val) in enumerate(zip(ref_features, rust_features)):
        assert _equalish(ref_val, got_val), (
            f"Reference-nonempty mismatch at index {idx} for pair {s1},{s2}: ref={ref_val}, got={got_val}"
        )


def test_featurize_pair_rust_parity_missing_email(dataset):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    featurizer_mod.global_dataset = dataset  # type: ignore
    missing_email_sigs = _find_signatures(
        dataset,
        lambda s, _p: (dataset.signatures[s].author_info_email is None)
        or (len(dataset.signatures[s].author_info_email or "") == 0),
        limit=1,
    )
    if not missing_email_sigs:
        pytest.skip("No signatures with missing email found")
    s1 = missing_email_sigs[0]
    s2 = next(sig_id for sig_id in dataset.signatures.keys() if sig_id != s1)

    ref_features, _ = _single_pair_featurize((s1, s2))
    rust_features = featurize_pair_rust(dataset, s1, s2)
    assert math.isnan(ref_features[7]) and math.isnan(ref_features[8])
    for idx, (ref_val, got_val) in enumerate(zip(ref_features, rust_features)):
        assert _equalish(ref_val, got_val), (
            f"Missing-email mismatch at index {idx} for pair {s1},{s2}: ref={ref_val}, got={got_val}"
        )


def test_featurize_pair_rust_parity_specter_present(dataset):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    featurizer_mod.global_dataset = dataset  # type: ignore
    sigs = _find_signatures(
        dataset,
        lambda s, p: p.predicted_language in {"en", "un"} and _specter_vec(dataset, p.paper_id) is not None,
        limit=2,
    )
    if len(sigs) < 2:
        pytest.skip("Not enough papers with valid specter embeddings found")
    s1, s2 = sigs[0], sigs[1]

    ref_features, _ = _single_pair_featurize((s1, s2))
    rust_features = featurize_pair_rust(dataset, s1, s2)
    assert not math.isnan(ref_features[33])
    for idx, (ref_val, got_val) in enumerate(zip(ref_features, rust_features)):
        assert _equalish(ref_val, got_val), (
            f"Specter-present mismatch at index {idx} for pair {s1},{s2}: ref={ref_val}, got={got_val}"
        )


def test_featurize_pair_rust_parity_specter_absent(dataset):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    featurizer_mod.global_dataset = dataset  # type: ignore
    sigs = _find_signatures(
        dataset,
        lambda s, p: p.predicted_language not in {"en", "un"} or _specter_vec(dataset, p.paper_id) is None,
        limit=1,
    )
    if not sigs:
        pytest.skip("No papers with missing specter embeddings or non-EN/UN language found")
    s1 = sigs[0]
    s2 = next(sig_id for sig_id in dataset.signatures.keys() if sig_id != s1)

    ref_features, _ = _single_pair_featurize((s1, s2))
    rust_features = featurize_pair_rust(dataset, s1, s2)
    assert math.isnan(ref_features[33])
    for idx, (ref_val, got_val) in enumerate(zip(ref_features, rust_features)):
        assert _equalish(ref_val, got_val), (
            f"Specter-absent mismatch at index {idx} for pair {s1},{s2}: ref={ref_val}, got={got_val}"
        )


def test_get_constraint_rust_parity_incremental_flag(dataset):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    by_cluster = defaultdict(list)
    for sig_id, cluster_id in dataset.cluster_seeds_require.items():
        by_cluster[cluster_id].append(sig_id)
    sigs = None
    for sig_list in by_cluster.values():
        if len(sig_list) >= 2:
            sigs = (sig_list[0], sig_list[1])
            break
    if sigs is None:
        sig_ids = list(dataset.signatures.keys())
        if len(sig_ids) < 2:
            pytest.skip("Not enough signatures to synthesize cluster_seeds_require pairs")
        s1, s2 = sig_ids[0], sig_ids[1]
        require_map = {s1: "test_cluster", s2: "test_cluster"}
        disallow_set = set()
        with _temporary_cluster_seeds(dataset, require_map, disallow_set):
            ref_val = dataset.get_constraint(s1, s2, incremental_dont_use_cluster_seeds=True)
            got_val = get_constraint_rust(dataset, s1, s2, incremental_dont_use_cluster_seeds=True)
    else:
        s1, s2 = sigs
        ref_val = dataset.get_constraint(s1, s2, incremental_dont_use_cluster_seeds=True)
        got_val = get_constraint_rust(dataset, s1, s2, incremental_dont_use_cluster_seeds=True)
    if ref_val is None or got_val is None:
        assert ref_val is None and got_val is None
    else:
        assert ref_val == got_val, f"Incremental constraint mismatch for pair {s1},{s2}: ref={ref_val}, got={got_val}"


def test_get_constraint_rust_parity_dont_merge_cluster_seeds_false(dataset):
    if not HAS_RUST:
        pytest.fail(f"s2and_rust extension not built: {_RUST_IMPORT_ERROR}")

    sigs = None
    cluster_ids = defaultdict(list)
    for sig_id, cluster_id in dataset.cluster_seeds_require.items():
        cluster_ids[cluster_id].append(sig_id)
    clusters = list(cluster_ids.values())
    if len(clusters) >= 2 and clusters[0] and clusters[1]:
        sigs = (clusters[0][0], clusters[1][0])
    if sigs is None:
        sig_ids = list(dataset.signatures.keys())
        if len(sig_ids) < 2:
            pytest.skip("Not enough signatures to synthesize distinct cluster_seeds_require entries")
        s1, s2 = sig_ids[0], sig_ids[1]
        require_map = {s1: "cluster_a", s2: "cluster_b"}
        disallow_set = set()
        with _temporary_cluster_seeds(dataset, require_map, disallow_set):
            ref_val = dataset.get_constraint(s1, s2, dont_merge_cluster_seeds=False)
            got_val = get_constraint_rust(dataset, s1, s2, dont_merge_cluster_seeds=False)
    else:
        s1, s2 = sigs
        ref_val = dataset.get_constraint(s1, s2, dont_merge_cluster_seeds=False)
        got_val = get_constraint_rust(dataset, s1, s2, dont_merge_cluster_seeds=False)
    if ref_val is None or got_val is None:
        assert ref_val is None and got_val is None
    else:
        assert ref_val == got_val, f"Dont-merge constraint mismatch for pair {s1},{s2}: ref={ref_val}, got={got_val}"

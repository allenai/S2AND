import os
import random
import math
from collections import defaultdict

import pytest

import s2and.featurizer as featurizer_mod
from s2and.featurizer import _single_pair_featurize
from s2and.data import ANDData
from s2and.consts import PROJECT_ROOT_PATH
from s2and.feature_port import featurize_pair_rust, get_constraint_rust


try:
    import s2and_rust  # noqa: F401

    HAS_RUST = True
    _RUST_IMPORT_ERROR = None
    print("s2and_rust import OK")
except Exception as e:  # pragma: no cover - environment specific
    HAS_RUST = False
    _RUST_IMPORT_ERROR = e
    print(f"s2and_rust import FAILED: {e}")


def _equalish(a, b, rel_tol=1e-6, abs_tol=1e-3):
    a_val = float(a)
    b_val = float(b)
    if math.isnan(a_val) and math.isnan(b_val):
        return True
    return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)


@pytest.fixture(scope="session")
def dataset():
    # Speed/safety: skip fastText (optional) to avoid large model loads in tests
    os.environ.setdefault("S2AND_SKIP_FASTTEXT", "1")
    # Force python reference featurizer for parity tests
    os.environ.setdefault("S2AND_USE_RUST_FEATURIZER", "0")
    # Avoid disk-cached Rust featurizer snapshots in parity tests
    os.environ.setdefault("S2AND_RUST_FEATURIZER_DISK_CACHE", "0")

    data_original = os.path.join(PROJECT_ROOT_PATH, "data", "s2and_mini")
    dataset_name = "zbmath"

    ds = ANDData(
        signatures=os.path.join(data_original, dataset_name, dataset_name + "_signatures.json"),
        papers=os.path.join(data_original, dataset_name, dataset_name + "_papers.json"),
        name=dataset_name,
        mode="train",
        specter_embeddings=os.path.join(data_original, dataset_name, dataset_name + "_specter.pickle"),
        clusters=os.path.join(data_original, dataset_name, dataset_name + "_clusters.json"),
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
    )
    # set global for _single_pair_featurize
    featurizer_mod.global_dataset = ds  # type: ignore
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

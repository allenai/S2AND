import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import pytest

import s2and.featurizer as featurizer_mod
from s2and.consts import LARGE_DISTANCE, LARGE_INTEGER, PROJECT_ROOT_PATH
from s2and.data import ANDData
from s2and.feature_port import (
    _get_rust_featurizer,
    build_linker_pair_distance_accumulators_rust,
    get_constraint_labels_index_arrays_rust,
    get_constraints_matrix_indexed_rust,
)
from s2and.featurizer import _single_pair_featurize
from s2and.text import canonicalize_name_parts, detect_language
from scripts.convert_to_arrow import join_canonical_benchmark_names
from tests.helpers import build_arrow_training_dataset, equalish, import_s2and_rust, tiny_name_counts_index

HAS_RUST, _rust_import_payload = import_s2and_rust()
_RUST_IMPORT_ERROR = None if HAS_RUST else _rust_import_payload
if not HAS_RUST:
    raise pytest.skip.Exception(
        f"s2and_rust extension not built/installed: {_RUST_IMPORT_ERROR}",
        allow_module_level=True,
    )


def _paper_for_sig(dataset, sig_id):
    sig = dataset.signatures[sig_id]
    return dataset.papers[str(sig.paper_id)]


def _featurize_pair_indexed_rust(dataset, sig_id_1: str, sig_id_2: str) -> np.ndarray:
    rust_featurizer = _get_rust_featurizer(dataset)
    signature_id_to_index = {str(sig_id): index for index, sig_id in enumerate(rust_featurizer.signature_ids())}
    return np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed(
            [(signature_id_to_index[str(sig_id_1)], signature_id_to_index[str(sig_id_2)])],
            None,
            getattr(dataset, "n_jobs", 1),
            np.nan,
        ),
        dtype=np.float64,
    )[0]


def _constraint_indexed_rust(dataset, sig_id_1: str, sig_id_2: str, **kwargs):
    rust_featurizer = kwargs.pop("featurizer", None)
    if rust_featurizer is None:
        rust_featurizer = _get_rust_featurizer(dataset)
    signature_id_to_index = {str(sig_id): index for index, sig_id in enumerate(rust_featurizer.signature_ids())}
    return get_constraints_matrix_indexed_rust(
        [(signature_id_to_index[str(sig_id_1)], signature_id_to_index[str(sig_id_2)])],
        featurizer=rust_featurizer,
        **kwargs,
    )[0]


def _load_dataset_from_dir(data_dir, name, *, signatures=None):
    cluster_seeds_path = os.path.join(data_dir, "cluster_seeds.json")
    cluster_seeds = cluster_seeds_path if os.path.exists(cluster_seeds_path) else None
    ds = ANDData(
        signatures=signatures if signatures is not None else os.path.join(data_dir, "signatures.json"),
        papers=os.path.join(data_dir, "papers.json"),
        name=name,
        mode="train",
        specter_embeddings=None,
        clusters=os.path.join(data_dir, "clusters.json"),
        cluster_seeds=cluster_seeds,
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=100000,
        val_pairs_size=10000,
        test_pairs_size=10000,
        n_jobs=1,
        name_counts_index=tiny_name_counts_index(),
        preprocess=True,
        random_seed=42,
        name_tuples=None,
        use_orcid_id=True,
    )
    return ds


def _build_two_signature_dataset(signatures, papers, name, *, name_tuples=None):
    return ANDData(
        signatures=signatures,
        papers=papers,
        name=name,
        mode="train",
        specter_embeddings=None,
        clusters={"c1": {"cluster_id": "c1", "signature_ids": ["s1", "s2"], "model_version": -1}},
        cluster_seeds=None,
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=10,
        val_pairs_size=10,
        test_pairs_size=10,
        n_jobs=1,
        name_counts_index=None,
        preprocess=True,
        random_seed=42,
        name_tuples=name_tuples,
        use_orcid_id=True,
    )


def _attach_fake_specter_embeddings(ds, max_papers=2, dim=8):
    rng = np.random.RandomState(123)
    if ds.specter_embeddings is None:
        ds.specter_embeddings = {}
    added = 0
    # Independent of preprocessing backend: paper fields like predicted_language
    # may be deferred to Rust, so attach purely by signature iteration order.
    for sig_id in ds.signatures.keys():
        paper = _paper_for_sig(ds, sig_id)
        paper_id = str(paper.paper_id)
        if paper_id not in ds.specter_embeddings:
            ds.specter_embeddings[paper_id] = rng.normal(size=(dim,)).astype(np.float32)
            added += 1
            if added >= max_papers:
                break
    return ds


def _reset_featurizer_env_caches():
    featurizer_mod.__dict__["_RUST_BATCH_CHUNK_SIZE_CACHE"] = None
    featurizer_mod.__dict__["_RUST_BATCH_MAX_CHUNK_MB_CACHE"] = None


def _build_labeled_pairs(sig_ids, count=20, seed=123):
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < count:
        s1 = rng.choice(sig_ids)
        s2 = rng.choice(sig_ids)
        if s1 == s2:
            continue
        pairs.append((s1, s2, 0))
    return pairs


@pytest.fixture(scope="session")
def source_dataset():
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("S2AND_BACKEND", "python")
        # Avoid reusing stale process-level env caches between parity fixtures.
        _reset_featurizer_env_caches()

        data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
        with open(os.path.join(data_dir, "signatures.json"), encoding="utf-8") as infile:
            raw_signatures = json.load(infile)
        canonical_rows = [
            {
                "signature_id": signature_id,
                **canonicalize_name_parts(
                    signature["author_info"].get("first"),
                    signature["author_info"].get("middle"),
                    signature["author_info"].get("last"),
                )._asdict(),
            }
            for signature_id, signature in raw_signatures.items()
        ]
        signatures, report = join_canonical_benchmark_names(raw_signatures, canonical_rows)
        assert report["rows"] == len(raw_signatures)
        assert report["changed_signatures"] > 0
        return _attach_fake_specter_embeddings(
            _load_dataset_from_dir(data_dir, "dummy_parity_session", signatures=signatures)
        )


@pytest.fixture(scope="session")
def arrow_dataset(source_dataset, tmp_path_factory):
    # Rust featurizers build exclusively from Arrow artifacts. Keep this
    # reconstructed object distinct from the Python reference dataset.
    return build_arrow_training_dataset(source_dataset, tmp_path_factory.mktemp("parity_session_bundle"))


@pytest.fixture(scope="session")
def sample_pairs(source_dataset):
    rng = random.Random(123)
    sig_ids = list(source_dataset.signatures.keys())
    pairs = []
    while len(pairs) < 10:
        s1 = rng.choice(sig_ids)
        s2 = rng.choice(sig_ids)
        if s1 == s2:
            continue
        pairs.append((s1, s2))
    return pairs


@pytest.fixture(scope="session")
def constraint_pairs(source_dataset, sample_pairs):
    pairs = list(sample_pairs)
    seen = set(pairs)

    # Add a few disallow pairs if present
    for a, b in list(source_dataset.cluster_seeds_disallow)[:5]:
        if (a, b) not in seen and (b, a) not in seen and a != b:
            pairs.append((a, b))
            seen.add((a, b))

    # Add a few require pairs (same cluster id) and cross-cluster pairs
    by_cluster = defaultdict(list)
    for sig_id, cluster_id in source_dataset.cluster_seeds_require.items():
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


def test_rust_indexed_pair_featurization_preserves_empty_selected_width(arrow_dataset):
    rust_featurizer = _get_rust_featurizer(arrow_dataset)

    matrix = np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed([], [0, 1], 1, np.nan),
        dtype=np.float64,
    )

    assert matrix.shape == (0, 2)
    with pytest.raises(ValueError, match="selected_indices contains out-of-range index"):
        rust_featurizer.featurize_pairs_matrix_indexed([], [10_000], 1, np.nan)


def test_rust_indexed_block_featurization_preserves_empty_selected_width(arrow_dataset):
    rust_featurizer = _get_rust_featurizer(arrow_dataset)
    signature_count = len(rust_featurizer.signature_ids())
    assert signature_count >= 2

    singleton = np.asarray(
        rust_featurizer.featurize_block_upper_triangle_matrix_indexed([0], 0, None, [0, 1], 1, np.nan),
        dtype=np.float64,
    )
    empty_block = np.asarray(
        rust_featurizer.featurize_block_upper_triangle_matrix_indexed([], 0, None, [0, 1], 1, np.nan),
        dtype=np.float64,
    )
    capped_empty = np.asarray(
        rust_featurizer.featurize_block_upper_triangle_matrix_indexed([0, 1], 0, 0, [0, 1], 1, np.nan),
        dtype=np.float64,
    )

    assert singleton.shape == (0, 2)
    assert empty_block.shape == (0, 2)
    assert capped_empty.shape == (0, 2)


def test_rust_indexed_block_featurization_validates_empty_inputs(arrow_dataset):
    rust_featurizer = _get_rust_featurizer(arrow_dataset)
    signature_count = len(rust_featurizer.signature_ids())

    with pytest.raises(ValueError, match="selected_indices contains out-of-range index"):
        rust_featurizer.featurize_block_upper_triangle_matrix_indexed([0], 0, None, [10_000], 1, np.nan)
    with pytest.raises(ValueError, match="selected_indices contains out-of-range index"):
        rust_featurizer.featurize_block_upper_triangle_matrix_indexed([0, 1], 0, 0, [10_000], 1, np.nan)
    with pytest.raises(IndexError, match="block signature index out of range"):
        rust_featurizer.featurize_block_upper_triangle_matrix_indexed(
            [signature_count],
            0,
            None,
            [0, 1],
            1,
            np.nan,
        )


def test_arrow_training_helper_does_not_reuse_source_records(source_dataset, arrow_dataset):
    signature_id = next(iter(source_dataset.signatures))
    paper_id = str(source_dataset.signatures[signature_id].paper_id)

    assert arrow_dataset.signatures[signature_id] is not source_dataset.signatures[signature_id]
    assert arrow_dataset.papers[paper_id] is not source_dataset.papers[paper_id]


def test_rust_featurizer_supports_string_paper_ids(tmp_path):
    # Regression guard: datasets may use non-numeric paper IDs (e.g., "app:123").
    signatures = {
        "s1": {
            "signature_id": "s1",
            "paper_id": "app:1",
            "author_info": {
                "position": 0,
                "block": "alice_smith",
                "first": "Alice",
                "middle": "",
                "last": "Smith",
                "suffix": None,
                "email": None,
                "affiliations": [],
            },
        },
        "s2": {
            "signature_id": "s2",
            "paper_id": "app:2",
            "author_info": {
                "position": 0,
                "block": "alice_smith",
                "first": "Alice",
                "middle": "",
                "last": "Smith",
                "suffix": None,
                "email": None,
                "affiliations": [],
            },
        },
    }
    papers = {
        "app:1": {
            "paper_id": "app:1",
            "title": "A",
            "abstract": "",
            "authors": [
                {"author_name": "Alice Smith", "position": 0},
                {"author_name": "Bob Jones", "position": 1},
            ],
            "venue": "",
            "journal_name": "",
            "year": 2020,
            "references": [],
        },
        "app:2": {
            "paper_id": "app:2",
            "title": "B",
            "abstract": "",
            "authors": [
                {"author_name": "Alice Smith", "position": 0},
                {"author_name": "Carol Lee", "position": 1},
            ],
            "venue": "",
            "journal_name": "",
            "year": 2021,
            "references": [],
        },
    }
    ds = _build_two_signature_dataset(signatures, papers, "rust_string_id_regression")

    ds = build_arrow_training_dataset(ds, tmp_path, name_counts="empty")
    features = _featurize_pair_indexed_rust(ds, "s1", "s2")
    assert len(features) > 0

    constraint = _constraint_indexed_rust(ds, "s1", "s2")
    assert constraint is None or isinstance(constraint, int | float)


def test_single_initial_name_text_features_match_rust(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("S2AND_BACKEND", "python")
    _reset_featurizer_env_caches()
    signatures = {
        "s1": {
            "signature_id": "s1",
            "paper_id": "app:1",
            "author_info": {
                "position": 0,
                "block": "a_smith",
                "first": "A",
                "middle": "",
                "last": "Smith",
                "suffix": None,
                "email": None,
                "affiliations": [],
            },
        },
        "s2": {
            "signature_id": "s2",
            "paper_id": "app:2",
            "author_info": {
                "position": 0,
                "block": "alice_smith",
                "first": "Alice",
                "middle": "",
                "last": "Smith",
                "suffix": None,
                "email": None,
                "affiliations": [],
            },
        },
    }
    papers = {
        "app:1": {
            "paper_id": "app:1",
            "title": "A",
            "abstract": "",
            "authors": [{"author_name": "A Smith", "position": 0}],
            "venue": "",
            "journal_name": "",
            "year": 2020,
            "references": [],
        },
        "app:2": {
            "paper_id": "app:2",
            "title": "B",
            "abstract": "",
            "authors": [{"author_name": "Alice Smith", "position": 0}],
            "venue": "",
            "journal_name": "",
            "year": 2021,
            "references": [],
        },
    }
    ds = _build_two_signature_dataset(signatures, papers, "single_initial_name_text_parity")
    arrow_dataset = build_arrow_training_dataset(ds, tmp_path, name_counts="empty")
    ref_features, _ = _single_pair_featurize(("s1", "s2"), dataset=ds)
    rust_features = _featurize_pair_indexed_rust(arrow_dataset, "s1", "s2")
    feature_names = featurizer_mod.FeaturizationInfo().get_feature_names()

    assert ds.get_constraint("s1", "s2") is None
    for feature_name in ("levenshtein", "prefix", "lcs", "jaro"):
        idx = feature_names.index(feature_name)
        assert not math.isnan(ref_features[idx])
        assert equalish(ref_features[idx], rust_features[idx])


def test_indexed_pair_matrix_rust_parity(source_dataset, arrow_dataset, sample_pairs):
    for s1, s2 in sample_pairs:
        ref_features, _ = _single_pair_featurize((s1, s2), dataset=source_dataset)
        rust_features = _featurize_pair_indexed_rust(arrow_dataset, s1, s2)
        assert len(ref_features) == len(rust_features)
        for idx, (ref_val, got_val) in enumerate(zip(ref_features, rust_features, strict=True)):
            assert equalish(ref_val, got_val), (
                f"Featurize pair mismatch at index {idx} for pair {s1},{s2}: ref={ref_val}, got={got_val}"
            )


def test_language_reliability_min_is_pair_order_invariant_in_python_and_rust(tmp_path):
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    dataset = _load_dataset_from_dir(data_dir, "dummy_language_reliability_pair_order")
    signature_id_1 = "0"
    signature_id_2 = "2"
    paper_id_1 = str(dataset.signatures[signature_id_1].paper_id)
    paper_id_2 = str(dataset.signatures[signature_id_2].paper_id)
    dataset.papers[paper_id_1] = dataset.papers[paper_id_1]._replace(
        predicted_language="en",
        is_reliable=True,
        language_reliability=0.25,
    )
    dataset.papers[paper_id_2] = dataset.papers[paper_id_2]._replace(
        predicted_language="fr",
        is_reliable=True,
        language_reliability=0.75,
    )
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path)
    feature_names = featurizer_mod.FeaturizationInfo().get_feature_names()
    reliability_index = feature_names.index("language_reliability_min")

    python_forward, _ = _single_pair_featurize((signature_id_1, signature_id_2), dataset=dataset)
    python_reverse, _ = _single_pair_featurize((signature_id_2, signature_id_1), dataset=dataset)
    rust_forward = _featurize_pair_indexed_rust(arrow_dataset, signature_id_1, signature_id_2)
    rust_reverse = _featurize_pair_indexed_rust(arrow_dataset, signature_id_2, signature_id_1)

    observed = [
        python_forward[reliability_index],
        python_reverse[reliability_index],
        rust_forward[reliability_index],
        rust_reverse[reliability_index],
    ]
    assert all(equalish(float(value), 0.25) for value in observed), observed


@pytest.mark.parametrize(
    "control",
    ["\u001c", "\u0080"],
    ids=["c0-file-separator", "c1-padding-character"],
)
def test_raw_arrow_language_detection_matches_python_for_rejected_controls(tmp_path, control):
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    dataset = _load_dataset_from_dir(data_dir, "dummy_rejected_language_control_parity")
    signature_id_1 = "0"
    signature_id_2 = "2"
    paper_id_1 = str(dataset.signatures[signature_id_1].paper_id)
    paper_id_2 = str(dataset.signatures[signature_id_2].paper_id)
    plain_title = "This is a detailed English research title about neural systems and scientific evaluation."
    control_title = (
        f"This is a detailed English research title {control} about neural systems and scientific evaluation."
    )
    plain_detection = detect_language(plain_title)
    control_detection = detect_language(control_title)

    assert plain_detection.predicted_language == "en"
    assert plain_detection.language_reliability > 0.0
    assert control_detection.predicted_language == "un"
    assert control_detection.language_reliability == 0.0

    dataset.papers[paper_id_1] = dataset.papers[paper_id_1]._replace(
        title=plain_title,
        predicted_language=plain_detection.predicted_language,
        is_reliable=plain_detection.is_reliable,
        language_reliability=plain_detection.language_reliability,
    )
    dataset.papers[paper_id_2] = dataset.papers[paper_id_2]._replace(
        title=control_title,
        predicted_language=control_detection.predicted_language,
        is_reliable=control_detection.is_reliable,
        language_reliability=control_detection.language_reliability,
    )
    python_features, _ = _single_pair_featurize((signature_id_1, signature_id_2), dataset=dataset)

    for paper_id in (paper_id_1, paper_id_2):
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(
            predicted_language=None,
            is_reliable=None,
            language_reliability=None,
        )
    arrow_dataset = build_arrow_training_dataset(dataset, tmp_path)
    rust_features = _featurize_pair_indexed_rust(arrow_dataset, signature_id_1, signature_id_2)
    feature_names = featurizer_mod.FeaturizationInfo().get_feature_names()

    for feature_name in ("same_language", "language_reliability_min"):
        feature_index = feature_names.index(feature_name)
        assert equalish(python_features[feature_index], rust_features[feature_index])
        assert equalish(python_features[feature_index], 0.0)


def test_many_pairs_end_to_end_parity_python_vs_rust(monkeypatch, tmp_path):
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")

    monkeypatch.setenv("S2AND_BACKEND", "python")
    _reset_featurizer_env_caches()
    ds_python = _load_dataset_from_dir(data_dir, "dummy_python_end_to_end")
    ds_python = _attach_fake_specter_embeddings(ds_python)
    sig_ids = list(ds_python.signatures.keys())
    pairs = _build_labeled_pairs(sig_ids, count=25, seed=7)
    featurizer_info = featurizer_mod.FeaturizationInfo()
    features_python, labels_python, _ = featurizer_mod.many_pairs_featurize(
        pairs,
        ds_python,
        featurizer_info,
        n_jobs=2,
        chunk_size=4,
        nan_value=np.nan,
    )

    monkeypatch.setenv("S2AND_BACKEND", "rust")
    _reset_featurizer_env_caches()
    ds_rust = _load_dataset_from_dir(data_dir, "dummy_rust_end_to_end")
    ds_rust = _attach_fake_specter_embeddings(ds_rust)
    ds_rust = build_arrow_training_dataset(ds_rust, tmp_path)
    features_rust, labels_rust, _ = featurizer_mod.many_pairs_featurize(
        pairs,
        ds_rust,
        featurizer_info,
        n_jobs=2,
        chunk_size=4,
        nan_value=np.nan,
    )

    assert np.array_equal(labels_python, labels_rust)
    assert features_python.shape == features_rust.shape
    close_mask = np.isclose(features_python, features_rust, rtol=0.0, atol=1e-6, equal_nan=True)
    assert np.all(close_mask), f"Feature matrix mismatch count: {int((~close_mask).sum())}"
    _reset_featurizer_env_caches()


def test_indexed_constraint_rust_ignores_reliable_language_mismatch(tmp_path):
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    source_dataset = _load_dataset_from_dir(data_dir, "dummy_language_constraint_removed")

    s1 = "0"
    s2 = "2"
    paper_id_1 = str(source_dataset.signatures[s1].paper_id)
    paper_id_2 = str(source_dataset.signatures[s2].paper_id)

    source_dataset.papers[paper_id_1] = source_dataset.papers[paper_id_1]._replace(
        predicted_language="en",
        is_reliable=True,
    )
    source_dataset.papers[paper_id_2] = source_dataset.papers[paper_id_2]._replace(
        predicted_language="fr",
        is_reliable=True,
    )
    arrow_dataset = build_arrow_training_dataset(source_dataset, tmp_path)

    ref_val = source_dataset.get_constraint(s1, s2)
    got_val = _constraint_indexed_rust(arrow_dataset, s1, s2)

    assert ref_val is None
    assert got_val is None

    rust_featurizer = _get_rust_featurizer(arrow_dataset)
    signature_ids = list(rust_featurizer.signature_ids())
    signature_index = {sig_id: idx for idx, sig_id in enumerate(signature_ids)}

    got_indexed = get_constraints_matrix_indexed_rust(
        [(signature_index[s1], signature_index[s2])],
        featurizer=rust_featurizer,
    )

    assert got_indexed == [None]


def test_indexed_constraint_rust_uses_dataset_name_tuple_aliases(tmp_path):
    signatures = {
        "s1": {
            "signature_id": "s1",
            "paper_id": "p1",
            "author_info": {
                "first": "Yu",
                "middle": None,
                "last": "Chen",
                "suffix": None,
                "affiliations": [],
                "email": None,
                "position": 0,
                "block": "y chen",
            },
        },
        "s2": {
            "signature_id": "s2",
            "paper_id": "p2",
            "author_info": {
                "first": "Yi",
                "middle": None,
                "last": "Chen",
                "suffix": None,
                "affiliations": [],
                "email": None,
                "position": 0,
                "block": "y chen",
            },
        },
    }
    papers = {
        "p1": {
            "paper_id": "p1",
            "title": "A",
            "abstract": "",
            "authors": [{"author_name": "Yu Chen", "position": 0}],
            "venue": "",
            "journal_name": "",
            "year": 1964,
            "references": [],
        },
        "p2": {
            "paper_id": "p2",
            "title": "B",
            "abstract": "",
            "authors": [{"author_name": "Yi Chen", "position": 0}],
            "venue": "",
            "journal_name": "",
            "year": 1970,
            "references": [],
        },
    }
    ds = _build_two_signature_dataset(
        signatures,
        papers,
        "name_tuple_alias_constraint_parity",
        name_tuples={("yu", "yi")},
    )

    assert ds.get_constraint("s1", "s2") is None
    ds = build_arrow_training_dataset(ds, tmp_path, name_counts="empty")
    rust_featurizer = _get_rust_featurizer(ds)
    assert _constraint_indexed_rust(ds, "s1", "s2", featurizer=rust_featurizer) is None
    signature_ids = list(rust_featurizer.signature_ids())
    signature_index = {sig_id: idx for idx, sig_id in enumerate(signature_ids)}
    indexed_values = get_constraints_matrix_indexed_rust(
        [(signature_index["s1"], signature_index["s2"])],
        featurizer=rust_featurizer,
    )
    assert indexed_values == [None]


def test_get_constraints_matrix_indexed_rust_parity(source_dataset, arrow_dataset, constraint_pairs):
    rust_featurizer = _get_rust_featurizer(arrow_dataset)
    signature_ids = list(rust_featurizer.signature_ids())
    signature_index = {sig_id: idx for idx, sig_id in enumerate(signature_ids)}
    indexed_pairs = [(signature_index[s1], signature_index[s2]) for s1, s2 in constraint_pairs]

    expected = [source_dataset.get_constraint(s1, s2) for s1, s2 in constraint_pairs]
    indexed_values = get_constraints_matrix_indexed_rust(indexed_pairs, featurizer=rust_featurizer)
    assert len(indexed_values) == len(expected)
    for pair, ref_val, indexed_val in zip(
        constraint_pairs,
        expected,
        indexed_values,
        strict=True,
    ):
        assert ref_val == indexed_val, (
            f"Batch indexed constraint mismatch for pair {pair}: ref={ref_val}, indexed={indexed_val}"
        )


def test_linker_constraint_labels_index_arrays_match_indexed_constraints_large(arrow_dataset, constraint_pairs):
    rust_featurizer = _get_rust_featurizer(arrow_dataset)
    signature_ids = list(rust_featurizer.signature_ids())
    signature_index = {sig_id: idx for idx, sig_id in enumerate(signature_ids)}
    base_pairs = list(constraint_pairs)
    for left in signature_ids[:8]:
        for right in signature_ids[:8]:
            if left != right:
                base_pairs.append((left, right))
    pairs = [base_pairs[offset % len(base_pairs)] for offset in range(4096)]
    indexed_pairs = [(signature_index[s1], signature_index[s2]) for s1, s2 in pairs]
    left_indices = np.asarray([left for left, _right in indexed_pairs], dtype=np.uint32)
    right_indices = np.asarray([right for _left, right in indexed_pairs], dtype=np.uint32)

    expected_values = get_constraints_matrix_indexed_rust(indexed_pairs, featurizer=rust_featurizer)
    expected_labels = np.asarray(
        [np.nan if value is None else float(value - LARGE_INTEGER) for value in expected_values],
        dtype=np.float64,
    )
    got_labels = get_constraint_labels_index_arrays_rust(
        left_indices,
        right_indices,
        featurizer=rust_featurizer,
        num_threads=2,
    )

    np.testing.assert_allclose(got_labels, expected_labels, equal_nan=True)


def test_linker_pair_distance_accumulators_match_python_large(arrow_dataset):
    rust_featurizer = _get_rust_featurizer(arrow_dataset)
    rng = np.random.default_rng(20260509)
    row_count = 503
    pair_count = 12000
    row_indices = rng.integers(0, row_count, size=pair_count, dtype=np.uint32)
    model_distances = rng.random(pair_count, dtype=np.float64)
    labels = np.full(pair_count, np.nan, dtype=np.float64)
    labels[::17] = -float(LARGE_INTEGER)
    labels[::29] = float(LARGE_DISTANCE - LARGE_INTEGER)

    expected_counts = np.zeros(row_count, dtype=np.uint32)
    expected_sums = np.zeros(row_count, dtype=np.float64)
    expected_mins = np.full(row_count, np.inf, dtype=np.float64)
    expected_top = np.full((row_count, 5), np.inf, dtype=np.float64)
    expected_hard_disallow = 0
    for row_raw, model_distance, label in zip(row_indices, model_distances, labels, strict=True):
        row = int(row_raw)
        value = float(model_distance if np.isnan(label) else label + LARGE_INTEGER)
        expected_counts[row] += 1
        expected_sums[row] += value
        expected_mins[row] = min(expected_mins[row], value)
        if value >= LARGE_DISTANCE:
            expected_hard_disallow += 1
        if value < expected_top[row, -1]:
            expected_top[row, -1] = value
            expected_top[row].sort()

    counts, sums, mins, top, hard_disallow = build_linker_pair_distance_accumulators_rust(
        row_indices,
        row_count,
        model_distances,
        pair_labels=labels,
        featurizer=rust_featurizer,
        num_threads=2,
    )

    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_allclose(sums, expected_sums)
    np.testing.assert_allclose(mins, expected_mins)
    np.testing.assert_allclose(top, expected_top)
    assert hard_disallow == expected_hard_disallow


@pytest.mark.parametrize(
    ("constraint_kwargs"),
    [
        {"dont_merge_cluster_seeds": False},
        {"incremental_dont_use_cluster_seeds": True},
    ],
)
def test_get_constraints_matrix_indexed_rust_flag_parity(
    source_dataset,
    arrow_dataset,
    constraint_pairs,
    constraint_kwargs,
):
    rust_featurizer = _get_rust_featurizer(arrow_dataset)
    expected = [source_dataset.get_constraint(s1, s2, **constraint_kwargs) for s1, s2 in constraint_pairs]

    signature_ids = list(rust_featurizer.signature_ids())
    signature_index = {sig_id: idx for idx, sig_id in enumerate(signature_ids)}
    indexed_pairs = [(signature_index[s1], signature_index[s2]) for s1, s2 in constraint_pairs]
    got_indexed = get_constraints_matrix_indexed_rust(
        indexed_pairs,
        featurizer=rust_featurizer,
        **constraint_kwargs,
    )

    for pair, ref_val, indexed_val in zip(constraint_pairs, expected, got_indexed, strict=True):
        assert ref_val == indexed_val, (
            f"Flag parity mismatch (indexed) for pair {pair}: ref={ref_val}, got={indexed_val}"
        )

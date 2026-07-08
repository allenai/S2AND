import os
import random
from contextlib import contextmanager
from itertools import combinations

import numpy as np
import pytest

from s2and import feature_port
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo, _single_pair_featurize
from s2and.subblocking import make_subblocks
from tests.helpers import attach_arrow_featurizer_bundle, build_dummy_dataset, equalish, tiny_name_counts

if not feature_port.rust_featurizer_available():
    raise pytest.skip.Exception("s2and_rust featurizer API is unavailable", allow_module_level=True)


@contextmanager
def _temporary_env(name: str, value: str | None):
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


def _signature_scalar_fields(signature) -> dict[str, object]:
    return {
        "author_info_first_normalized": signature.author_info_first_normalized,
        "author_info_first_normalized_without_apostrophe": signature.author_info_first_normalized_without_apostrophe,
        "author_info_middle_normalized_without_apostrophe": signature.author_info_middle_normalized_without_apostrophe,
        "author_info_last_normalized": signature.author_info_last_normalized,
        "author_info_coauthors": signature.author_info_coauthors,
        "author_info_coauthor_blocks": signature.author_info_coauthor_blocks,
        "author_info_affiliations": signature.author_info_affiliations,
        "author_info_name_counts": signature.author_info_name_counts,
        "author_info_orcid": signature.author_info_orcid,
    }


def _sample_pairs(signature_ids: list[str], limit: int = 8) -> list[tuple[str, str]]:
    pairs = []
    for s1, s2 in combinations(signature_ids, 2):
        pairs.append((s1, s2))
        if len(pairs) >= limit:
            break
    return pairs


def _short_coauthor_dataset(name: str, backend: str) -> ANDData:
    signatures = {
        "s1": {
            "author_info": {
                "first": "Alice",
                "middle": "",
                "last": "Smith",
                "suffix": None,
                "position": 0,
                "email": None,
                "affiliations": [],
                "block": "a smith",
            },
            "signature_id": "s1",
            "paper_id": 1,
        },
        "s2": {
            "author_info": {
                "first": "Alice",
                "middle": "",
                "last": "Smith",
                "suffix": None,
                "position": 0,
                "email": None,
                "affiliations": [],
                "block": "a smith",
            },
            "signature_id": "s2",
            "paper_id": 2,
        },
    }
    papers = {
        "1": {
            "paper_id": 1,
            "title": "Short coauthor parity study",
            "abstract": "",
            "journal_name": "",
            "venue": "",
            "year": 2024,
            "authors": [
                {"position": 0, "author_name": "Alice Smith"},
                {"position": 1, "author_name": "Li"},
            ],
            "references": [],
        },
        "2": {
            "paper_id": 2,
            "title": "Short coauthor parity followup",
            "abstract": "",
            "journal_name": "",
            "venue": "",
            "year": 2025,
            "authors": [
                {"position": 0, "author_name": "Alice Smith"},
                {"position": 1, "author_name": "Li"},
            ],
            "references": [],
        },
    }
    with _temporary_env("S2AND_BACKEND", backend):
        return ANDData(
            signatures,
            papers,
            clusters={},
            name=name,
            mode="inference",
            load_name_counts=tiny_name_counts(),
            preprocess=True,
            n_jobs=1,
        )


def test_signature_preprocess_json_dataset_rust_backend_uses_python_signature_fields():
    with _temporary_env("S2AND_BACKEND", "python"):
        dataset_python = build_dummy_dataset("dummy_signature_preprocess_python")
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_rust")

    assert set(dataset_python.signatures.keys()) == set(dataset_rust.signatures.keys())
    for signature_id in dataset_python.signatures:
        signature_python = dataset_python.signatures[signature_id]
        signature_rust = dataset_rust.signatures[signature_id]
        assert _signature_scalar_fields(signature_rust) == _signature_scalar_fields(signature_python)


def test_signature_preprocess_pair_features_and_constraints_parity_with_arrow_fields(tmp_path):
    with _temporary_env("S2AND_BACKEND", "python"):
        dataset_python = build_dummy_dataset("dummy_signature_preprocess_materialize_python", load_name_counts=True)
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_materialize_rust", load_name_counts=True)
    attach_arrow_featurizer_bundle(dataset_rust, tmp_path)

    signature_ids = list(dataset_python.signatures.keys())
    pairs = _sample_pairs(signature_ids, limit=8)
    assert len(pairs) > 0
    rust_featurizer = feature_port._get_rust_featurizer(dataset_rust)  # noqa: SLF001
    rust_signature_id_to_index = {
        str(signature_id): index for index, signature_id in enumerate(rust_featurizer.signature_ids())
    }

    for s1, s2 in pairs:
        python_features, _ = _single_pair_featurize((s1, s2), dataset=dataset_python)
        rust_features = np.asarray(
            rust_featurizer.featurize_pairs_matrix_indexed(
                [(rust_signature_id_to_index[str(s1)], rust_signature_id_to_index[str(s2)])],
                None,
                getattr(dataset_rust, "n_jobs", 1),
                np.nan,
            ),
            dtype=np.float64,
        )[0]
        assert len(python_features) == len(rust_features)
        for idx, (python_value, rust_value) in enumerate(zip(python_features, rust_features, strict=True)):
            assert equalish(python_value, rust_value), (
                f"Feature mismatch for pair ({s1}, {s2}) at idx={idx}: " f"python={python_value} rust={rust_value}"
            )

        python_constraint = dataset_python.get_constraint(s1, s2)
        rust_constraint = feature_port.get_constraints_matrix_indexed_rust(
            dataset_rust,
            [(rust_signature_id_to_index[str(s1)], rust_signature_id_to_index[str(s2)])],
            featurizer=rust_featurizer,
        )[0]
        if python_constraint is None or rust_constraint is None:
            assert python_constraint is None and rust_constraint is None
        else:
            assert python_constraint == rust_constraint


def test_signature_preprocess_lazy_materialization_ngrams_match_python():
    with _temporary_env("S2AND_BACKEND", "python"):
        dataset_python = build_dummy_dataset("dummy_signature_preprocess_materialize_python_ngrams")
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_materialize_rust_ngrams")
    dataset_rust.materialize_signature_ngrams_python()

    for signature_id in dataset_python.signatures:
        signature_python = dataset_python.signatures[signature_id]
        signature_rust = dataset_rust.signatures[signature_id]
        assert signature_python.author_info_affiliations_n_grams == signature_rust.author_info_affiliations_n_grams
        assert signature_python.author_info_coauthor_n_grams == signature_rust.author_info_coauthor_n_grams


def test_short_coauthor_tokens_match_python_and_rust_featurizers(tmp_path):
    dataset_python = _short_coauthor_dataset("short_coauthor_python", "python")
    python_features, _ = _single_pair_featurize(("s1", "s2"), dataset=dataset_python)
    coauthor_similarity_idx = FeaturizationInfo().feature_group_to_index["coauthor_similarity"][1]

    python_coauthor_ngrams = dataset_python.signatures["s1"].author_info_coauthor_n_grams
    assert python_coauthor_ngrams is not None
    assert "li" in python_coauthor_ngrams
    assert python_features[coauthor_similarity_idx] == 1.0

    dataset_rust = _short_coauthor_dataset("short_coauthor_rust", "rust")
    attach_arrow_featurizer_bundle(dataset_rust, tmp_path)
    rust_featurizer = feature_port._get_rust_featurizer(dataset_rust)  # noqa: SLF001
    rust_signature_id_to_index = {
        str(signature_id): index for index, signature_id in enumerate(rust_featurizer.signature_ids())
    }
    rust_features = np.asarray(
        rust_featurizer.featurize_pairs_matrix_indexed(
            [(rust_signature_id_to_index["s1"], rust_signature_id_to_index["s2"])],
            None,
            getattr(dataset_rust, "n_jobs", 1),
            np.nan,
        ),
        dtype=np.float64,
    )[0]

    assert equalish(python_features[coauthor_similarity_idx], rust_features[coauthor_similarity_idx])
    assert rust_features[coauthor_similarity_idx] == 1.0


def test_subblocking_handles_missing_signature_affiliation_ngrams():
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_subblocking_rust")
    signature_ids = list(dataset_rust.signatures.keys())
    output = make_subblocks(signature_ids, dataset_rust, maximum_size=2)
    assert sum(len(subblock) for subblock in output.values()) == len(signature_ids)


def test_subblocking_membership_parity_python_vs_rust():
    with _temporary_env("S2AND_BACKEND", "python"):
        dataset_python = build_dummy_dataset("dummy_signature_preprocess_subblocking_python")
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_rust = build_dummy_dataset("dummy_signature_preprocess_subblocking_rust_parity")

    signature_ids = list(dataset_python.signatures.keys())
    random.seed(12345)
    output_python = make_subblocks(signature_ids, dataset_python, maximum_size=2)
    random.seed(12345)
    output_rust = make_subblocks(signature_ids, dataset_rust, maximum_size=2)

    clusters_python = {tuple(sorted(subblock)) for subblock in output_python.values()}
    clusters_rust = {tuple(sorted(subblock)) for subblock in output_rust.values()}
    assert clusters_python == clusters_rust


def test_rust_inference_without_arrow_featurizer_runs_python_paper_preprocess():
    with _temporary_env("S2AND_BACKEND", "rust"):
        dataset_inference = build_dummy_dataset("dummy_signature_preprocess_minimal_papers", mode="inference")

    paper_id = next(iter(dataset_inference.papers.keys()))
    inference_paper = dataset_inference.papers[paper_id]

    assert inference_paper.title_ngrams_chars is not None
    assert inference_paper.title_ngrams_words is not None

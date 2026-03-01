from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from s2and.consts import PROJECT_ROOT_PATH
from s2and.data import ANDData
from tests.conftest import import_s2and_rust


def test_python_maybe_load_specter_accepts_dict_payload():
    payload = {"p1": np.array([0.1, 0.2], dtype=np.float32)}
    loaded = ANDData.maybe_load_specter(payload)
    assert isinstance(loaded, dict)
    assert np.allclose(loaded["p1"], payload["p1"])


def test_python_maybe_load_specter_accepts_tuple_payload(tmp_path):
    matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    keys = ["p1", "p2"]
    payload_path = tmp_path / "specter_tuple.pickle"
    with payload_path.open("wb") as out_file:
        pickle.dump((matrix, keys), out_file, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = ANDData.maybe_load_specter(str(payload_path))
    assert isinstance(loaded, dict)
    assert set(loaded.keys()) == {"p1", "p2"}
    assert np.allclose(loaded["p1"], matrix[0])
    assert np.allclose(loaded["p2"], matrix[1])


def test_rust_from_json_paths_accepts_tuple_specter_pickle(tmp_path):
    has_rust, s2and_rust = import_s2and_rust(required_method="from_json_paths")
    if not has_rust:
        pytest.skip("s2and_rust RustFeaturizer.from_json_paths is unavailable")

    signatures_path = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy", "signatures.json")
    papers_path = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy", "papers.json")
    clusters_path = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy", "clusters.json")
    cluster_seeds_path = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy", "cluster_seeds.json")

    keys = ["53235312", "27077319", "19901392", "21094749", "38029096", "1", "2"]
    matrix = np.random.RandomState(7).normal(size=(len(keys), 8)).astype(np.float32)
    specter_path = tmp_path / "specter_tuple.pickle"
    with specter_path.open("wb") as out_file:
        pickle.dump((matrix, keys), out_file, protocol=pickle.HIGHEST_PROTOCOL)

    rust_featurizer = s2and_rust.RustFeaturizer.from_json_paths(
        signatures_path,
        papers_path,
        clusters_path,
        cluster_seeds_path,
        str(specter_path),
        None,
        None,
        True,
        False,
        0.0,
        10000.0,
        1,
    )
    features = rust_featurizer.featurize_pair("0", "1")
    assert isinstance(features, list)
    assert len(features) > 0


def test_rust_from_json_paths_accepts_dict_specter():
    has_rust, s2and_rust = import_s2and_rust(required_method="from_json_paths")
    if not has_rust:
        pytest.skip("s2and_rust RustFeaturizer.from_json_paths is unavailable")

    signatures_path = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy", "signatures.json")
    papers_path = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy", "papers.json")
    clusters_path = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy", "clusters.json")
    cluster_seeds_path = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy", "cluster_seeds.json")

    keys = ["53235312", "27077319", "19901392", "21094749", "38029096", "1", "2"]
    matrix = np.random.RandomState(7).normal(size=(len(keys), 8)).astype(np.float32)
    specter_dict = {k: matrix[i] for i, k in enumerate(keys)}

    rust_featurizer = s2and_rust.RustFeaturizer.from_json_paths(
        signatures_path,
        papers_path,
        clusters_path,
        cluster_seeds_path,
        specter_dict,
        None,
        None,
        True,
        False,
        0.0,
        10000.0,
        1,
    )
    features = rust_featurizer.featurize_pair("0", "1")
    assert isinstance(features, list)
    assert len(features) > 0

from s2and.ingest_contract import build_rust_json_ingest_contract


class DummyDataset:
    signatures_path = "signatures.json"
    papers_path = "papers.json"
    clusters_path = "clusters.json"
    cluster_seeds_path = "cluster_seeds.json"
    specter_embeddings_path = "specter.pkl"
    preprocess = False
    compute_reference_features = True


def test_build_rust_json_ingest_contract_collects_canonical_fields():
    contract = build_rust_json_ingest_contract(
        DummyDataset(),
        name_counts_path="name_counts.json",
        cluster_seed_require_value=0.0,
        cluster_seed_disallow_value=10000.0,
        num_threads=4,
        name_tuples_path="name_tuples.txt",
    )
    assert contract.signatures_path == "signatures.json"
    assert contract.papers_path == "papers.json"
    assert contract.clusters_path == "clusters.json"
    assert contract.cluster_seeds_path == "cluster_seeds.json"
    assert contract.specter_embeddings == "specter.pkl"
    assert contract.name_tuples_path == "name_tuples.txt"
    assert contract.name_counts_path == "name_counts.json"
    assert contract.preprocess is False
    assert contract.compute_reference_features is True
    assert contract.num_threads == 4
    assert contract.expected_normalization_version is None
    assert contract.allow_normalization_version_mismatch is False
    assert contract.as_from_json_paths_args() == (
        "signatures.json",
        "papers.json",
        "cluster_seeds.json",
        "specter.pkl",
        "name_tuples.txt",
        "name_counts.json",
        False,
        True,
        0.0,
        10000.0,
        4,
        None,
        False,
    )


def test_build_rust_json_ingest_contract_prefers_loaded_dict_over_path():
    class DatasetWithLoadedSpecter:
        signatures_path = "signatures.json"
        papers_path = "papers.json"
        clusters_path = None
        cluster_seeds_path = None
        specter_embeddings_path = "specter.pkl"
        specter_embeddings = {"p1": [0.1, 0.2]}
        preprocess = True
        compute_reference_features = False

    contract = build_rust_json_ingest_contract(
        DatasetWithLoadedSpecter(),
        name_counts_path=None,
        cluster_seed_require_value=0.0,
        cluster_seed_disallow_value=10000.0,
        num_threads=1,
    )
    assert contract.specter_embeddings == {"p1": [0.1, 0.2]}


def test_build_rust_json_ingest_contract_requires_signature_and_paper_paths():
    class MissingPathsDataset:
        signatures_path = None
        papers_path = None

    try:
        build_rust_json_ingest_contract(
            MissingPathsDataset(),
            name_counts_path=None,
            cluster_seed_require_value=0.0,
            cluster_seed_disallow_value=10000.0,
            num_threads=1,
        )
    except RuntimeError as exc:
        assert "signatures_path/papers_path" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing signatures_path/papers_path")

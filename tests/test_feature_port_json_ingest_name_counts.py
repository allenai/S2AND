from __future__ import annotations

import os

import pytest

from s2and import feature_port
from s2and.consts import PROJECT_ROOT_PATH
from s2and.data import ANDData, NameCounts
from s2and.featurizer import FeaturizationInfo

if not feature_port.rust_featurizer_available():
    pytest.skip("s2and_rust extension is unavailable", allow_module_level=True)

_FEATURIZATION_INFO = FeaturizationInfo()


def _build_dummy_dataset(name: str, *, mode: str = "train", load_name_counts=False) -> ANDData:
    data_dir = os.path.join(PROJECT_ROOT_PATH, "tests", "dummy")
    return ANDData(
        signatures=os.path.join(data_dir, "signatures.json"),
        papers=os.path.join(data_dir, "papers.json"),
        name=name,
        mode=mode,
        specter_embeddings=None,
        clusters=os.path.join(data_dir, "clusters.json"),
        cluster_seeds=None,
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=100,
        val_pairs_size=50,
        test_pairs_size=50,
        n_jobs=1,
        load_name_counts=load_name_counts,
        preprocess=True,
        random_seed=42,
        name_tuples="filtered",
        use_orcid_id=True,
        use_sinonym_overwrite=False,
        compute_reference_features=False,
    )


def test_feature_port_json_ingest_overlays_dataset_name_counts_without_artifact(monkeypatch):
    dataset = _build_dummy_dataset("dummy_json_ingest_signature_counts_overlay")
    for idx, sig_id in enumerate(sorted(dataset.signatures.keys()), start=1):
        signature = dataset.signatures[sig_id]
        dataset.signatures[sig_id] = signature._replace(
            author_info_name_counts=NameCounts(
                first=float(100 + idx),
                last=float(200 + idx),
                first_last=float(300 + idx),
                last_first_initial=float(400 + idx),
            )
        )

    feature_port.clear_rust_featurizer_cache()
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.delenv("S2AND_RUST_NAME_COUNTS_JSON", raising=False)
    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")

    rust_from_feature_port = feature_port._get_rust_featurizer(dataset)
    rust_from_dataset = feature_port.s2and_rust.RustFeaturizer.from_dataset(dataset, 0.0, 10000.0, 1)

    signature_ids = sorted(dataset.signatures.keys())
    pair = (signature_ids[0], signature_ids[1])
    via_feature_port = rust_from_feature_port.featurize_pair(pair[0], pair[1])
    via_dataset = rust_from_dataset.featurize_pair(pair[0], pair[1])

    name_count_indices = _FEATURIZATION_INFO.feature_group_to_index["name_counts"]
    for idx in name_count_indices:
        assert via_feature_port[idx] == pytest.approx(via_dataset[idx], rel=1e-6, abs=1e-6)

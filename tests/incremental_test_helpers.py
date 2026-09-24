"""Shared tiny seed datasets and public prediction inputs; test oracles stay local."""

import numpy as np
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.artifact import load_incremental_linking_artifact, save_incremental_linking_artifact
from s2and.incremental_linking.feature_block_arrow import write_raw_arrow_batch_lookup_indexes
from s2and.model import Clusterer, FastCluster
from scripts.arrow_conversion_helpers import write_raw_planner_arrow_from_anddata
from tests.helpers import write_test_arrow_artifact_manifest
from tests.model_helpers import ConstantDistanceClassifier
from tests.promoted_linking_helpers import (
    build_tiny_promoted_booster,
    synthetic_pairwise_bundle_binding,
    tiny_logistic_gate_config,
)


def _dataset(names: list[tuple[str, str, str]], seeds: dict[str, str]) -> ANDData:
    signatures, papers = {}, {}
    for index, (first, middle, last) in enumerate(names):
        sid, pid = str(index), str(100 + index)
        signatures[sid] = {
            "signature_id": sid,
            "paper_id": int(pid),
            "author_info": {
                "first": first,
                "middle": middle,
                "last": last,
                "suffix": "",
                "position": 0,
                "email": None,
                "affiliations": [],
                "block": f"a {last.lower()}",
            },
        }
        papers[pid] = {
            "paper_id": int(pid),
            "title": "Notes on mathematics",
            "abstract": "",
            "year": 2000,
            "venue": "",
            "journal_name": "",
            "authors": [{"position": 0, "author_name": f"{first} {last}"}],
        }
    dataset = ANDData(
        signatures,
        papers,
        name="seed_boundaries",
        mode="inference",
        name_counts_index=None,
        name_tuples=set(),
        specter_embeddings={pid: np.array([1.0, 0.0], dtype=np.float32) for pid in papers},
    )
    dataset.cluster_seeds_require = seeds.copy()
    dataset.max_seed_cluster_id = len(set(seeds.values()))
    return dataset


def _predict(
    tmp_path, dataset, blocks, route, *, threshold=3, distance=0.1, classifier=None, linkage="average", **kwargs
):
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(distance) if classifier is None else classifier,
        cluster_model=FastCluster(linkage=linkage, eps=0.5),
        n_jobs=1,
        use_default_constraints_as_supervision=kwargs.pop("use_default_constraints_as_supervision", True),
    )
    if route == "python":
        return clusterer.predict(blocks, dataset, batching_threshold=threshold, **kwargs)[0]
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker")
    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        return clusterer.predict_from_arrow(
            blocks,
            arrow_dataset,
            batching_threshold=threshold,
            name_tuples=frozenset(),
            cluster_seeds_require=dataset.cluster_seeds_require,
            cluster_seeds_disallow=dataset.cluster_seeds_disallow,
            **kwargs,
        )[0]


def prepare_seed_arrow_bundle(dataset, arrow_dir):
    """Write the matching raw Arrow tables, indexes and test manifest."""
    paths = write_raw_planner_arrow_from_anddata(dataset, arrow_dir, include_specter=True)
    paths, _ = write_raw_arrow_batch_lookup_indexes(paths, arrow_dir)
    write_test_arrow_artifact_manifest(arrow_dir, paths)


def attach_tiny_incremental_linker(clusterer, artifact_dir):
    """Attach the same deterministic promoted linker used by seed route tests."""
    booster, _ = build_tiny_promoted_booster()
    save_incremental_linking_artifact(
        booster,
        artifact_dir,
        gate_config=tiny_logistic_gate_config(),
        target_spec={},
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
    )
    clusterer.incremental_linker_artifact = load_incremental_linking_artifact(artifact_dir)


@pytest.fixture(autouse=True)
def empty_orcid_priors(monkeypatch):
    """Keep optional generated ORCID priors out of name/seed-only fixtures."""
    monkeypatch.setattr("s2and.subblocking._resolved_orcid_prefix_counts", lambda counts: {})

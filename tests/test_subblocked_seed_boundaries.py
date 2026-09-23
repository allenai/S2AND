"""Public prediction regressions for block-local, atomic claimed profiles."""

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


def _predict(tmp_path, dataset, blocks, route, *, threshold=3, distance=0.1, **kwargs):
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(distance),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
        use_default_constraints_as_supervision=kwargs.pop("use_default_constraints_as_supervision", True),
    )
    if route == "python":
        return clusterer.predict(blocks, dataset, batching_threshold=threshold, **kwargs)[0]
    paths = write_raw_planner_arrow_from_anddata(dataset, tmp_path / "arrow", include_specter=True)
    paths, _ = write_raw_arrow_batch_lookup_indexes(paths, tmp_path / "arrow")
    write_test_arrow_artifact_manifest(tmp_path / "arrow", paths)
    booster, _ = build_tiny_promoted_booster()
    save_incremental_linking_artifact(
        booster,
        tmp_path / "linker",
        gate_config=tiny_logistic_gate_config(),
        target_spec={},
        pairwise_bundle_binding=synthetic_pairwise_bundle_binding(),
    )
    clusterer.incremental_linker_artifact = load_incremental_linking_artifact(tmp_path / "linker")
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


@pytest.fixture(autouse=True)
def _empty_orcid_priors(monkeypatch):
    # Stage 1's optional generated prior is unrelated to these name/seed cases.
    monkeypatch.setattr("s2and.subblocking._resolved_orcid_prefix_counts", lambda counts: {})


def _assert_partition(clusters, blocks):
    ids = [sid for members in clusters.values() for sid in members]
    assert sorted(ids) == sorted(sid for members in blocks.values() for sid in members)
    assert all(any(set(members) <= set(block) for block in blocks.values()) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize("same_last", [False, True])
def test_claimed_profile_and_initial_attachment_stay_in_original_block(tmp_path, route, same_last):
    names = [("Ada", "", "Smith")] * 3 + [("Ada", "", "Smith" if same_last else "Jones")] * 3
    names.append(("A", "", "Smith" if same_last else "Jones"))
    dataset = _dataset(names, {"0": "claimed", "3": "claimed"})
    blocks = {"left": ["0", "1", "2"], "right": ["3", "4", "5", "6"]}
    clusters = _predict(tmp_path, dataset, blocks, route)
    _assert_partition(clusters, blocks)
    assert any({"0", "1", "2"} <= set(members) for members in clusters.values())
    assert any({"3", "4", "5"} <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_claimed_component_is_atomic_across_first_name_subblocks(tmp_path, route):
    dataset = _dataset([("Abdul", "", "Smith"), ("Ahmed", "", "Smith")] * 2, {"0": "claimed", "1": "claimed"})
    clusters = _predict(tmp_path, dataset, {"one": list(dataset.signatures)}, route, threshold=2, distance=0.9)
    assert {frozenset(members) for members in clusters.values()} == {
        frozenset({"0", "1"}),
        frozenset({"2"}),
        frozenset({"3"}),
    }


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_claimed_component_does_not_absorb_incompatible_names(tmp_path, route):
    dataset = _dataset(
        [("Ada", "", "Smith")] * 3 + [("Amy", "", "Smith")] * 3 + [("A", "", "Smith")], {"0": "claimed", "3": "claimed"}
    )
    blocks = {"one": list(dataset.signatures)}
    clusters = _predict(tmp_path, dataset, blocks, route)
    _assert_partition(clusters, blocks)
    assert any({"0", "3"} <= set(members) for members in clusters.values())
    assert not any({"1", "4"} <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_initial_only_block_uses_bulk_even_beside_full_name_block(tmp_path, route):
    dataset = _dataset([("Ada", "", "Smith")] + [("A", "", "Smith")] * 4, {})
    blocks = {"full": ["0"], "initial": ["1", "2", "3", "4"]}
    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2)
    _assert_partition(clusters, blocks)


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_profile_size_limit_applies_to_each_original_block(tmp_path, route):
    dataset = _dataset(
        [("Ada", "", "Smith"), ("Ada", "", "Smith"), ("A", "", "Smith")] * 2,
        dict.fromkeys(["0", "1", "3", "4"], "claimed"),
    )
    blocks = {"left": ["0", "1", "2"], "right": ["3", "4", "5"]}
    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2)
    _assert_partition(clusters, blocks)
    assert any({"0", "1"} <= set(members) for members in clusters.values())
    assert any({"3", "4"} <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_residual_cluster_ids_from_independent_blocks_preserve_all_outputs(tmp_path, route):
    dataset = _dataset([("Ada", "", "Smith"), ("Ada", "", "Smith"), ("A", "", "Smith")] * 2, {})
    dataset.cluster_seeds_disallow = {("0", "2"), ("1", "2"), ("3", "5"), ("4", "5")}
    blocks = {"left": ["0", "1", "2"], "right": ["3", "4", "5"]}
    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2)
    _assert_partition(clusters, blocks)
    assert {frozenset(members) for members in clusters.values()} == {
        frozenset({"0", "1"}),
        frozenset({"2"}),
        frozenset({"3", "4"}),
        frozenset({"5"}),
    }


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize("constraint_source", ["explicit_disallow", "partial_supervision", "ignored_seeds"])
@pytest.mark.parametrize("disallow", [("1", "3"), ("1", "2")])
def test_query_disallows_survive_subblocked_attachment(tmp_path, route, constraint_source, disallow):
    dataset = _dataset(
        [
            ("Ada", "", "Smith"),
            ("A", "", "Smith"),
            ("A", "", "Smith"),
            ("A", "Alpha", "Smith"),
            ("A", "Alpha", "Smith"),
        ],
        {},
    )
    # A missing middle name is compatible with Alpha, so name rules cannot
    # accidentally mask a dropped explicit negative across initial-only groups.
    assert dataset.get_constraint(*disallow) is None
    kwargs = {}
    if constraint_source == "partial_supervision":
        kwargs["partial_supervision"] = {disallow: 10000.0}
    else:
        dataset.cluster_seeds_disallow = {disallow}
        if constraint_source == "ignored_seeds":
            dataset.cluster_seeds_require = dict.fromkeys(disallow, "claimed")
            dataset.max_seed_cluster_id = 1
            kwargs["incremental_dont_use_cluster_seeds"] = True
    blocks = {"one": list(dataset.signatures)}
    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2, **kwargs)
    _assert_partition(clusters, blocks)
    assert not any(set(disallow) <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_query_must_link_survives_subblocked_attachment(tmp_path, route):
    """A must-link whose partner is attached in a later initial-only group."""
    dataset = _dataset(
        [
            ("Ada", "", "Smith"),
            ("A", "", "Smith"),
            ("A", "", "Smith"),
            ("A", "Alpha", "Smith"),
            ("A", "Alpha", "Smith"),
        ],
        {},
    )
    blocks = {"one": list(dataset.signatures)}
    clusters = _predict(
        tmp_path, dataset, blocks, route, threshold=2, distance=0.9, partial_supervision={("1", "3"): 0.0}
    )
    _assert_partition(clusters, blocks)
    assert any({"1", "3"} <= set(members) for members in clusters.values())


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize(
    "flags", [{}, {"incremental_dont_use_cluster_seeds": True}, {"use_default_constraints_as_supervision": False}]
)
def test_oversized_required_component_only_rejected_when_active(tmp_path, route, flags):
    dataset = _dataset([("Ada", "", "Smith")] * 3, dict.fromkeys(["0", "1", "2"], "claimed"))
    if not flags:
        with pytest.raises(ValueError, match="component exceeds batching_threshold"):
            _predict(tmp_path, dataset, {"one": list(dataset.signatures)}, route, threshold=2, **flags)
    else:
        clusters = _predict(
            tmp_path, dataset, {"one": list(dataset.signatures)}, route, threshold=2, distance=0.9, **flags
        )
        assert all(len(members) == 1 for members in clusters.values())


def test_legacy_author_id_prediction_does_not_pack_unused_seeds(tmp_path):
    dataset = _dataset([("Ada", "", "Smith")] * 3, dict.fromkeys(["0", "1", "2"], "claimed"))
    for sid, signature in dataset.signatures.items():
        dataset.signatures[sid] = signature._replace(author_id=f"legacy-{sid}")
    result = _predict(tmp_path, dataset, {"one": list(dataset.signatures)}, "python", threshold=2, use_s2_clusters=True)
    assert result == {f"legacy-{sid}": [sid] for sid in dataset.signatures}

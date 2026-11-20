"""Exercise guarded ORCID joins through bulk clustering and the EPS grid."""

from pathlib import Path

import numpy as np
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.block_corrections import correction_fingerprint
from s2and.data import ANDData
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.incremental_test_helpers import _dataset, prepare_seed_arrow_bundle
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors
from tests.model_helpers import ConstantDistanceClassifier

ORCID = "0000-0001-2345-6789"
OTHER = "0000-0002-1234-5678"
LEFT = frozenset(("0", "1", "2"))
RIGHT = frozenset(("3", "4", "5"))


def _model(distance: float) -> Clusterer:
    model = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(distance),
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
    )
    model.name_conflict_repair_max_distance = None
    model.year_gap_split = False
    return model


def _set_orcids(dataset: ANDData, orcids: dict[str, str]) -> None:
    for sid, orcid in orcids.items():
        dataset.signatures[sid] = dataset.signatures[sid]._replace(author_info_orcid=orcid)


def _partition(clusters: dict[str, list[str]]) -> set[frozenset[str]]:
    return {frozenset(members) for members in clusters.values()}


def _within_block(orcids: dict[str, str]) -> tuple[ANDData, dict[tuple[str, str], float]]:
    dataset = _dataset([("Ada", "", "Smith")] * 6, {})
    _set_orcids(dataset, orcids)
    # Keep each fragment intact and put the average of the cross pairs just above EPS.
    supervision = {(a, b): 0.01 for side in (LEFT, RIGHT) for a in side for b in side if a < b}
    supervision.update({("0", "3"): 0.10, ("1", "4"): 0.25, ("0", "4"): 0.75, ("1", "3"): 0.75})
    return dataset, supervision


@pytest.mark.parametrize("route", ["python", "arrow"])
@pytest.mark.parametrize(
    "guard",
    [
        "enabled",
        "off",
        "suppressed",
        "disallow",
        "exclusion",
        "seeds",
        "seeds_allowed",
        "majority",
        "tie",
        "noise",
        "far",
        "anchor_only",
    ],
)
def test_guarded_join_in_one_block(tmp_path: Path, route: str, guard: str) -> None:
    observed = {"0": ORCID, "1": ORCID, "3": ORCID, "4": ORCID}
    if guard == "majority":
        observed.update({"4": OTHER, "5": OTHER})
    if guard == "tie":
        observed["4"] = OTHER
    if guard == "noise":
        del observed["1"]
    dataset, supervision = _within_block(observed)
    supervision[("2", "5")] = 0.25 if guard == "anchor_only" else 0.19
    model = _model(0.75 if guard != "far" else 0.90)
    model.orcid_join = guard != "off"
    model.suppress_orcid = guard == "suppressed"
    model.dont_merge_cluster_seeds = guard != "seeds_allowed"
    if guard in ("seeds", "seeds_allowed"):
        dataset.cluster_seeds_require = {"2": "L", "5": "R"}
    if guard == "disallow":
        dataset.cluster_seeds_disallow = {("2", "4")}
        # The explicit disallow still applies to the join when a caller distance overrides it.
        supervision[("2", "4")] = 0.75
    exclusions = [("2", "4")] if guard == "exclusion" else []
    blocks = {"block": list(dataset.signatures)}
    ignore_seeds = guard in ("seeds", "seeds_allowed")
    if route == "python":
        clusters, _ = model.predict(
            blocks,
            dataset,
            partial_supervision=supervision,
            profile_exclusions=exclusions,
            incremental_dont_use_cluster_seeds=ignore_seeds,
        )
    else:
        prepare_seed_arrow_bundle(dataset, tmp_path)
        with ArrowDataset.open(tmp_path) as arrow:
            clusters, _ = model.predict_from_arrow(
                blocks,
                arrow,
                name_tuples=frozenset(),
                cluster_seeds_disallow=dataset.cluster_seeds_disallow,
                cluster_seeds_require=dataset.cluster_seeds_require,
                partial_supervision=supervision,
                profile_exclusions=exclusions,
                incremental_dont_use_cluster_seeds=ignore_seeds,
            )
    expected = {LEFT | RIGHT} if guard in ("enabled", "seeds_allowed") else {LEFT, RIGHT}
    assert _partition(clusters) == expected


def test_within_block_join_uses_the_name_rule_to_explain_hard_pairs() -> None:
    # Only the unlabeled endpoints introduce a hard conflict. An arbitrary hard
    # distance on a different pair must still veto the same proposed join.
    from s2and.orcid_join import join_orcids_in_block
    from s2and.pair_codes import HARD_CODE, condensed_index

    ids = list("012345")
    labels = [0, 0, 0, 1, 1, 1]
    codes = np.full(15, 30, dtype=np.uint8)
    codes[condensed_index(6, 2, 5)] = HARD_CODE
    kwargs = {
        "cut_height": 0.5,
        "forbidden": {},
        "seed_groups": None,
        "name_conflicts": lambda pairs: [pair == ("2", "5") for pair in pairs],
        "block_key": "block",
    }
    orcids = dict.fromkeys(("0", "1", "3", "4"), ORCID)
    assert set(join_orcids_in_block(ids, labels, codes, orcids, **kwargs)) == {0}
    codes[condensed_index(6, 2, 5)] = 30
    codes[condensed_index(6, 0, 3)] = HARD_CODE
    assert join_orcids_in_block(ids, labels, codes, orcids, **kwargs) is labels


@pytest.mark.parametrize("guard", ["enabled", "off", "suppressed", "disallow", "exclusion", "seeds", "name", "noise"])
def test_native_cross_subblock_join_stays_in_original_block(tmp_path: Path, guard: str) -> None:
    names = (
        [("Ada", "Jane", "Smith")] * 3 + [("Alma", "Jane", "Smith")] * 3
        if guard == "name"
        else [("Ada", "Jane", "Smith")] * 6
    )
    dataset = _dataset(names * 2, {})
    observed = dict.fromkeys(dataset.signatures, ORCID)
    if guard == "noise":
        observed = {s: ORCID for s in ("0", "1", "2", "3", "6", "7", "8", "9")}
    _set_orcids(dataset, observed)
    prepare_seed_arrow_bundle(dataset, tmp_path)
    model = _model(0.15)
    model.orcid_join = guard != "off"
    model.suppress_orcid = guard == "suppressed"
    blocks = {"one": list("012345"), "two": [str(i) for i in range(6, 12)]}
    with ArrowDataset.open(tmp_path) as arrow:
        clusters, _ = model.predict_from_arrow(
            blocks,
            arrow,
            batching_threshold=3,
            name_tuples=frozenset(),
            cluster_seeds_disallow=[("0", "5"), ("6", "11")] if guard == "disallow" else None,
            profile_exclusions=[("0", "5"), ("6", "11")] if guard == "exclusion" else None,
            cluster_seeds_require={"0": "A", "5": "B", "6": "C", "11": "D"} if guard == "seeds" else None,
        )
    partition = _partition(clusters)
    assert set().union(*partition) == set(dataset.signatures)
    assert sum(map(len, partition)) == len(dataset.signatures)
    if guard == "enabled":
        assert partition == {frozenset(members) for members in blocks.values()}
    else:
        assert len(partition) >= 4
        assert all(any(cluster <= set(group) for group in blocks.values()) for cluster in partition)


def test_eps_grids_follow_toggle_and_suppression(tmp_path: Path) -> None:
    dataset, supervision = _within_block({s: ORCID for s in ("0", "1", "3", "4")})
    supervision.update({("1", "4"): 0.25, ("2", "5"): 0.19})
    model = _model(0.75)
    ids = list(dataset.signatures)
    blocks = {"block": ids}
    dists = {"block": np.asarray([supervision.get((a, b), 0.75) for i, a in enumerate(ids) for b in ids[i + 1 :]])}
    prepare_seed_arrow_bundle(dataset, tmp_path)
    with ArrowDataset.open(tmp_path) as arrow:
        rust = build_rust_featurizer_from_arrow_dataset(
            arrow, signature_ids=ids, name_tuples=frozenset(), preprocess=True, num_threads=1
        )
        initial = correction_fingerprint(model)
        for on, suppressed, count in [(True, False, 1), (False, False, 2), (True, True, 2)]:
            model.orcid_join = on
            model.suppress_orcid = suppressed
            ids_py, grid_py = model.predict_eps_grid(blocks, dataset, dists, [0.5])
            ids_rs, grid_rs = model.predict_eps_grid_from_rust_featurizer(blocks, rust, dists, [0.5])
            assert ids_py == ids_rs == ids
            np.testing.assert_array_equal(grid_py[0.5], grid_rs[0.5])
            assert len(set(grid_py[0.5])) == count
            assert (correction_fingerprint(model) == initial) == (count == 1)


@pytest.mark.parametrize(("size", "joined"), [(40, True), (41, False)])
def test_cross_subblock_join_requires_orcid_coverage(size: int, joined: bool) -> None:
    from s2and.orcid_join import join_orcids_across_subblocks

    clusters = {"left": [f"l{i}" for i in range(size)], "right": ["r0", "r1", "r2"]}
    orcids = dict.fromkeys(("l0", "l1", "r0", "r1"), ORCID)
    actual = join_orcids_across_subblocks(
        clusters,
        orcids,
        dict.fromkeys(orcids, "original"),
        {sid: sid[0] for sid in orcids},
        lambda signature_ids: [("ada", "")] * len(signature_ids),
        forbidden={},
        seed_groups=None,
    )

    assert (len(actual) == 1) == joined


@pytest.mark.parametrize("explicit", [False, True])
def test_join_honors_a_featurizer_disallow_hidden_by_a_distance_override(tmp_path: Path, explicit: bool) -> None:
    # Joins one cluster pair through ORCID support; caller supervision then hides the disallow
    # ("2", "4") from the matrix, so only the featurizer still knows it.
    dataset, supervision = _within_block(dict.fromkeys("0134", ORCID))
    supervision.update({("2", "5"): 0.19, ("2", "4"): 0.75})
    disallow = {("2", "4")}
    dataset.cluster_seeds_disallow = disallow
    ids = list(dataset.signatures)
    model = _model(0.75)
    model.orcid_join = True
    prepare_seed_arrow_bundle(dataset, tmp_path)
    with ArrowDataset.open(tmp_path) as arrow:
        featurizer = build_rust_featurizer_from_arrow_dataset(
            arrow, signature_ids=ids, name_tuples=frozenset(), preprocess=True, num_threads=1
        ).with_cluster_seeds({}, disallow)
        assert featurizer.cluster_seeds_disallow() == [("2", "4")]
        clusters, _ = model.predict_from_rust_featurizer(
            {"block": ids},
            featurizer,
            partial_supervision=supervision,
            cluster_seeds_disallow=disallow if explicit else None,
        )
    assert _partition(clusters) == {LEFT, RIGHT}


def test_adjacency_union_reads_both_maps_without_copying() -> None:
    from s2and.block_corrections import _AdjacencyUnion

    union = _AdjacencyUnion({"a": {"b"}, "c": {"d"}}, {"a": {"e"}, "f": {"g"}})
    assert union.get("a") == {"b", "e"}
    assert union["c"] == {"d"} and union["f"] == {"g"}
    assert union.get("z", ()) == ()
    assert sorted(union) == ["a", "c", "f"] and len(union) == 3
    assert bool(_AdjacencyUnion({}, {"a": {"b"}})) and not _AdjacencyUnion({}, {})

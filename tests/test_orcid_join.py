"""Exercise the guarded ORCID join across subblocks."""

from pathlib import Path

import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.incremental_test_helpers import _dataset, prepare_seed_arrow_bundle
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors
from tests.model_helpers import ConstantDistanceClassifier

ORCID = "0000-0001-2345-6789"


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


@pytest.mark.parametrize(
    ("left_name", "right_name", "joined"),
    [
        (("ana", ""), ("ana", "c"), True),
        (("ana", "c"), ("ana", "m"), False),
        (("jean pierre", ""), ("jean luc", ""), False),
    ],
)
def test_cross_subblock_join_refuses_clusters_the_name_rule_keeps_apart(
    left_name: tuple[str, str], right_name: tuple[str, str], joined: bool
) -> None:
    from s2and.orcid_join import join_orcids_across_subblocks

    clusters = {"left": ["l0", "l1", "l2"], "right": ["r0", "r1", "r2"]}
    orcids = dict.fromkeys(("l0", "l1", "r0", "r1"), ORCID)
    names = {
        sid: (left_name if sid.startswith("l") else right_name) for members in clusters.values() for sid in members
    }
    actual = join_orcids_across_subblocks(
        clusters,
        orcids,
        dict.fromkeys(orcids, "original"),
        {sid: sid[0] for sid in orcids},
        lambda signature_ids: [names[sid] for sid in signature_ids],
        forbidden={},
        seed_groups=None,
    )

    assert (len(actual) == 1) == joined


@pytest.mark.parametrize(("bonus", "scans"), [(0.0, 0), (8.0, 1)])
def test_arrow_prediction_reads_small_blocks_only_for_the_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bonus: float, scans: int
) -> None:
    # The year-gap split needs a close-pair bonus; without one, nothing reads the small-block set.
    import s2and.model as model_module

    dataset = _dataset([("Ada", "", "Smith")] * 4, {})
    prepare_seed_arrow_bundle(dataset, tmp_path)
    model = _model(0.1)
    model.year_gap_split = True
    model.cluster_model.close_pair_bonus = bonus
    model.cluster_model.close_pair_distance = 0.05 if bonus else 0.0
    calls: list[object] = []
    original = model_module.arrow_small_block_signatures
    monkeypatch.setattr(model_module, "arrow_small_block_signatures", lambda *a: calls.append(a) or original(*a))
    with ArrowDataset.open(tmp_path) as arrow:
        model.predict_from_arrow({"b": list(dataset.signatures)}, arrow, name_tuples=frozenset())
    assert len(calls) == scans


def test_same_orcid_mentions_of_one_block_end_in_one_cluster() -> None:
    # Why no join runs inside a block: equal ORCIDs get distance 0 before the name rules, so
    # linkage merges their mentions first, however far apart the model puts them.
    dataset = _dataset([("Ada", "", "Smith")] * 6, {})
    _set_orcids(dataset, dict.fromkeys("0134", ORCID))
    model = _model(0.99)
    clusters, _ = model.predict({"block": list(dataset.signatures)}, dataset)
    holding = [members for members in clusters.values() if {"0", "1", "3", "4"} & set(members)]
    assert len(holding) == 1 and {"0", "1", "3", "4"} <= set(holding[0])

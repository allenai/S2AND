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


def _join(clusters, orcids, names, subblocks=None, single_letter=frozenset()):
    from s2and.orcid_join import join_orcids_across_subblocks

    return join_orcids_across_subblocks(
        clusters,
        orcids,
        dict.fromkeys(orcids, "original"),
        subblocks or {sid: sid[0] for sid in orcids},
        lambda signature_ids: [names[sid] for sid in signature_ids],
        forbidden={},
        seed_groups=None,
        single_letter_subblocks=single_letter,
    )


@pytest.mark.parametrize(("anchors", "joined"), [(2, False), (3, True)])
def test_cross_subblock_join_takes_an_initial_only_cluster_with_enough_orcid_coverage(
    anchors: int, joined: bool
) -> None:
    # "A. M." holds no full first name; 3 of 14 mentions carry the ORCID (21%), 2 of 14 do not (14%).
    clusters = {"full": [f"f{i}" for i in range(20)], "initials": [f"i{i}" for i in range(14)]}
    orcids = {**dict.fromkeys(("f0", "f1"), ORCID), **dict.fromkeys([f"i{i}" for i in range(anchors)], ORCID)}
    names = {**dict.fromkeys(clusters["full"], ("ahsan", "mehmood")), **dict.fromkeys(clusters["initials"], ("a", "m"))}
    assert (len(_join(clusters, orcids, names)) == 1) == joined


def test_cross_subblock_join_measures_first_name_dominance_among_full_names_only() -> None:
    # Half the big cluster is initials only; its full names are all "ahsan", so it still qualifies.
    clusters = {"big": [f"b{i}" for i in range(20)], "small": ["s0", "s1", "s2"]}
    orcids = dict.fromkeys(("b0", "b1", "s0", "s1"), ORCID)
    names = {
        **{f"b{i}": (("ahsan", "") if i < 10 else ("a", "")) for i in range(20)},
        **dict.fromkeys(clusters["small"], ("ahsan", "")),
    }
    assert len(_join(clusters, orcids, names)) == 1


def test_cross_subblock_join_leaves_initial_only_clusters_when_two_full_names_share_the_orcid() -> None:
    clusters = {"ana": ["a0", "a1", "a2"], "anna": ["n0", "n1", "n2"], "initials": ["i0", "i1", "i2"]}
    orcids = dict.fromkeys(("a0", "a1", "n0", "n1", "i0", "i1"), ORCID)
    names = {
        **dict.fromkeys(clusters["ana"], ("ana", "")),
        **dict.fromkeys(clusters["anna"], ("anna", "")),
        **dict.fromkeys(clusters["initials"], ("a", "")),
    }
    joined = _join(clusters, orcids, names, subblocks={sid: sid[0] for sid in orcids})
    assert {frozenset(members) for members in joined.values()} >= {frozenset(clusters["initials"])}


def test_cross_subblock_join_ignores_a_stray_conflicting_form_but_not_a_supported_one() -> None:
    clusters = {"ana": [f"a{i}" for i in range(10)], "anac": ["c0", "c1", "c2"]}
    orcids = dict.fromkeys(("a0", "a1", "c0", "c1"), ORCID)
    names = {**dict.fromkeys(clusters["ana"], ("ana", "")), **dict.fromkeys(clusters["anac"], ("ana", "c"))}
    names["a9"] = ("ana", "maria")  # one stray mention of ten
    assert len(_join(clusters, orcids, names)) == 1
    names.update(dict.fromkeys(("a7", "a8"), ("ana", "maria")))  # now 30% of the cluster
    assert len(_join(clusters, orcids, names)) == 2


def test_cross_subblock_join_does_not_count_the_single_letter_subblock_as_shared() -> None:
    clusters = {"full": ["f0", "f1", "f2"], "initials": ["i0", "i1", "i2"]}
    orcids = dict.fromkeys(("f0", "f1", "i0", "i1"), ORCID)
    names = {**dict.fromkeys(clusters["full"], ("ana", "")), **dict.fromkeys(clusters["initials"], ("a", ""))}
    # Both clusters hold anchors from the single-letter subblock "a".
    subblocks = {"f0": "ana", "f1": "a", "i0": "a", "i1": "a"}
    assert len(_join(clusters, orcids, names, subblocks)) == 2
    assert len(_join(clusters, orcids, names, subblocks, single_letter=frozenset({"a"}))) == 1

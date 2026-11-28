"""Public prediction regressions for block-local, atomic claimed profiles."""

import numpy as np
import pytest

from tests.incremental_test_helpers import _dataset, _predict
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors


def _assert_partition(clusters, blocks):
    ids = [sid for members in clusters.values() for sid in members]
    assert sorted(ids) == sorted(sid for members in blocks.values() for sid in members)
    assert all(any(set(members) <= set(block) for block in blocks.values()) for members in clusters.values())


def test_subblocking_normalizes_integer_ids_and_keeps_compatible_neighbors(tmp_path):
    names = [("A", "", "Smith")] * 2 + [("Ada", "", "Smith")] * 2 + [("Bob", "", "Smith")] * 2
    dataset = _dataset(names * 2 + [("A", "", "Smith")] * 2, {"0": "claimed", "2": "claimed"})
    blocks = {"claimed": list(range(6)), "unclaimed": list(range(6, 12)), "initials": [12, 13]}

    clusters = _predict(tmp_path, dataset, blocks, "arrow", threshold=3)

    _assert_partition(clusters, {block: [str(sid) for sid in ids] for block, ids in blocks.items()})
    assert {frozenset(members) for members in clusters.values()} == {
        frozenset({"0", "1", "2", "3"}),
        frozenset({"4", "5"}),
        frozenset({"6", "7", "8", "9"}),
        frozenset({"10", "11"}),
        frozenset({"12", "13"}),
    }


@pytest.mark.parametrize("route", ["arrow"])
def test_mixed_initial_claim_at_capacity_preserves_membership_and_name_constraints(tmp_path, route):
    names = [("A", "", "Smith")] * 2 + [("Ada", "", "Smith")] * 2 + [("Bob", "", "Smith")] * 2
    dataset = _dataset(names, {"0": "claimed", "2": "claimed"})
    blocks = {"one": list(dataset.signatures)}

    clusters = _predict(tmp_path, dataset, blocks, route, threshold=2)

    _assert_partition(clusters, blocks)
    assert any({"0", "2"} <= set(members) for members in clusters.values())
    assert all(not (set(members) & {"0", "1", "2", "3"} and set(members) & {"4", "5"}) for members in clusters.values())


@pytest.mark.parametrize("route", ["arrow"])
def test_claimed_component_is_atomic_across_first_name_subblocks(tmp_path, route):
    dataset = _dataset([("Abdul", "", "Smith"), ("Ahmed", "", "Smith")] * 2, {"0": "claimed", "1": "claimed"})
    clusters = _predict(tmp_path, dataset, {"one": list(dataset.signatures)}, route, threshold=2, distance=0.9)
    assert {frozenset(members) for members in clusters.values()} == {
        frozenset({"0", "1"}),
        frozenset({"2"}),
        frozenset({"3"}),
    }


@pytest.mark.parametrize("route", ["arrow"])
def test_claimed_component_does_not_absorb_incompatible_names(tmp_path, route):
    dataset = _dataset(
        [("Ada", "", "Smith")] * 3 + [("Amy", "", "Smith")] * 3 + [("A", "", "Smith")], {"0": "claimed", "3": "claimed"}
    )
    blocks = {"one": list(dataset.signatures)}
    clusters = _predict(tmp_path, dataset, blocks, route)
    _assert_partition(clusters, blocks)
    assert any({"0", "3"} <= set(members) for members in clusters.values())
    assert not any({"1", "4"} <= set(members) for members in clusters.values())


def test_subblocked_profile_capacity_and_residual_ids_are_block_local(tmp_path):
    dataset = _dataset(
        [("Ada", "", "Smith"), ("Ada", "", "Smith"), ("A", "", "Smith")] * 2,
        dict.fromkeys(["0", "1", "3", "4"], "claimed"),
    )
    dataset.cluster_seeds_disallow = {("0", "2"), ("1", "2"), ("3", "5"), ("4", "5")}
    blocks = {"left": ["0", "1", "2"], "right": ["3", "4", "5"]}
    clusters = _predict(tmp_path, dataset, blocks, "arrow", threshold=2)

    _assert_partition(clusters, blocks)
    assert {frozenset(members) for members in clusters.values()} == {
        frozenset({"0", "1"}),
        frozenset({"2"}),
        frozenset({"3", "4"}),
        frozenset({"5"}),
    }


@pytest.mark.parametrize("route", ["arrow"])
@pytest.mark.parametrize(
    ("constraint_source", "disallow"),
    [("explicit_disallow", ("1", "2")), ("partial_supervision", ("1", "3")), ("ignored_seeds", ("1", "3"))],
)
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


@pytest.mark.parametrize("route", ["arrow"])
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


@pytest.mark.parametrize("route", ["arrow"])
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
            tmp_path,
            dataset,
            {"one": list(dataset.signatures)},
            route,
            threshold=2,
            distance=0.9,
            gate_link=False,
            **flags,
        )
        assert all(len(members) == 1 for members in clusters.values())


def test_legacy_author_id_prediction_does_not_pack_unused_seeds(tmp_path):
    dataset = _dataset([("Ada", "", "Smith")] * 3, dict.fromkeys(["0", "1", "2"], "claimed"))
    for sid, signature in dataset.signatures.items():
        dataset.signatures[sid] = signature._replace(author_id=f"legacy-{sid}")
    result = _predict(
        tmp_path, dataset, {"one": list(dataset.signatures)}, "python", threshold=None, use_s2_clusters=True
    )
    assert result == {f"legacy-{sid}": [sid] for sid in dataset.signatures}


class _YearDistanceClassifier:
    """Make two tight groups whose average distance dilutes one negative."""

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        distances = np.where(features[:, 0] == 0, 0.0, 0.1)
        return np.column_stack([distances, 1.0 - distances])


@pytest.mark.parametrize("route", ["arrow"])
@pytest.mark.parametrize(("source", "exclude_query"), [("dataset", False), ("caller", True)])
def test_generated_bulk_seed_negatives_do_not_reject_initial_attachment(tmp_path, route, source, exclude_query):
    """Real bulk linkage can contain a negative without being a caller seed claim."""
    dataset = _dataset([("Ada", "", "Smith")] * 3 + [("A", "", "Smith")], {})
    negatives = {("0", "2")}
    if exclude_query:
        negatives.add(("0", "3"))
    dataset.cluster_seeds_disallow = negatives.copy() if source == "dataset" else set()
    partial_supervision = dict.fromkeys(negatives, 10000.0) if source == "caller" else {}
    original_disallows = dataset.cluster_seeds_disallow.copy()
    original_supervision = partial_supervision.copy()
    blocks = {"one": list(dataset.signatures)}

    clusters = _predict(
        tmp_path,
        dataset,
        blocks,
        route,
        threshold=3,
        linkage="single",
        partial_supervision=partial_supervision,
    )

    _assert_partition(clusters, blocks)
    # Single linkage joins the three full names around their internal negative;
    # that generated group must not become an inconsistent caller seed claim.
    expected = {frozenset({"0", "1", "2"}), frozenset({"3"})}
    if not exclude_query:
        expected = {frozenset({"0", "1", "2", "3"})}
    assert {frozenset(members) for members in clusters.values()} == expected
    assert dataset.cluster_seeds_require == {}
    assert dataset.cluster_seeds_disallow == original_disallows
    assert partial_supervision == original_supervision


@pytest.mark.parametrize("route", ["arrow"])
def test_average_linkage_keeps_bulk_negative_apart_in_large_merge(tmp_path, route):
    """A negative stays hard when a large average-linkage merge would dilute it."""
    dataset = _dataset([("Ada", "", "Smith")] * 400 + [("A", "", "Smith")] + [("Bea", "", "Smith")] * 11, {})
    for index in range(200, 400):
        paper_id = str(100 + index)
        dataset.papers[paper_id] = dataset.papers[paper_id]._replace(year=2001)
    dataset.cluster_seeds_disallow = {("0", "200"), ("0", "400")}
    partial_supervision = {}
    original_disallows = dataset.cluster_seeds_disallow.copy()
    original_supervision = partial_supervision.copy()
    # Let an independent block with no hard negatives keep its ordinary merge.
    ordinary = [str(index) for index in range(401, 412)]
    blocks = {"one": list(dataset.signatures)[:401], "ordinary": ordinary}

    clusters = _predict(
        tmp_path,
        dataset,
        blocks,
        route,
        threshold=400,
        classifier=_YearDistanceClassifier(),
        partial_supervision=partial_supervision,
    )

    _assert_partition(clusters, blocks)
    # A finite sentinel averages to about .35 over 200 x 200 pairs, below
    # eps=.5. The query keeps its own negative and joins the other group.
    expected = {
        frozenset(str(index) for index in range(200)),
        frozenset(str(index) for index in range(200, 401)),
        frozenset(ordinary),
    }
    assert {frozenset(members) for members in clusters.values()} == expected
    assert dataset.cluster_seeds_require == {}
    assert dataset.cluster_seeds_disallow == original_disallows
    assert partial_supervision == original_supervision


@pytest.mark.parametrize("route", ["arrow"])
@pytest.mark.parametrize("claimed", [False, True])
def test_dash_initial_compound_subblock_is_classified_by_its_key_first(tmp_path, monkeypatch, route, claimed):
    """A "J.-D." record keys as "j", so its subblock attaches like initials and never homes a profile."""
    from s2and.model import Clusterer

    dataset = _dataset([("J.-D.", "", "Smith")] + [("Jean", "", "Smith")] * 3, {"0": "c", "1": "c"} if claimed else {})
    blocks = {"one": list(dataset.signatures)}
    observed = {}
    partition = Clusterer._partition_subblocked_first_name_groups

    def record_partition(self, block_dict_subblocked, *args, **kwargs):
        result = partition(self, block_dict_subblocked, *args, **kwargs)
        observed.update(subblocks=dict(block_dict_subblocked), bulk=result[0], attach=result[1])
        return result

    monkeypatch.setattr(Clusterer, "_partition_subblocked_first_name_groups", record_partition)
    clusters = _predict(tmp_path, dataset, blocks, route, threshold=3)

    _assert_partition(clusters, blocks)
    if claimed:
        # The full-name subblock is the profile's home, so the "j" subblock empties.
        assert observed["subblocks"] == {
            "one|subblock=je|repair_part=0000": ["1", "0", "2"],
            "one|subblock=je|repair_part=0001": ["3"],
        }
        assert observed["attach"] == {}
    else:
        assert observed["subblocks"] == {"one|subblock=j": ["0"], "one|subblock=je": ["1", "2", "3"]}
        assert observed["bulk"] == {"one|subblock=je": ["1", "2", "3"]}
        assert observed["attach"] == {"one|subblock=j": ["0"]}


def test_dash_initial_compound_attaches_under_cluster_name_checks(tmp_path):
    dataset = _dataset(
        [("J.-D.", "", "Smith")]
        + [("Jean-Daniel", "", "Smith")] * 3
        + [("J.-D.", "", "Smith")]
        + [("Jean-Marc", "", "Smith")] * 3,
        {},
    )
    blocks = {"compatible": ["0", "1", "2", "3"], "incompatible": ["4", "5", "6", "7"]}

    clusters = _predict(tmp_path, dataset, blocks, "arrow", threshold=3)

    _assert_partition(clusters, blocks)
    assert {frozenset(members) for members in clusters.values()} == {
        frozenset({"0", "1", "2", "3"}),
        frozenset({"4"}),
        frozenset({"5", "6", "7"}),
    }

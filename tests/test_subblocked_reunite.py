"""Reunite-only: no-middle dead-end records that the size cap cut across subblocks."""

import json

import pytest

from s2and.model import _ARROW_ATTACH_NAME_CHECK_EXEMPT, _ARROW_BULK_DISTANCE_OVERRIDES, Clusterer
from tests.incremental_test_helpers import _dataset, _predict
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors

ROUTES = ["python", "arrow"]
ATTACH_METHODS = {
    "python": "_predict_subblocked_single_letter_incremental_groups",
    "arrow": "_predict_subblocked_single_letter_groups_from_arrow",
}
WEI = ("Wei", "", "Wang")
WEI_MIN = ("Wei", "Min", "Wang")
WEN = ("Wen", "", "Wang")
HAO = ("Hao", "", "Wang")
HAO_XU = ("Hao", "Xu", "Wang")
# With a cap of 3, the nine "Wei" records without a middle name are a no-middle dead-end bucket. The
# graph fallback cuts it into the pieces 0-2, 3-5 and 6-8. "Wei Min" (9-12) is not marked; its
# pieces are 10-12 and 9.
NAMES = [WEI] * 9 + [WEI_MIN] * 4 + [WEN]
NO_MIDDLE = {str(index) for index in range(9)}


def _partition(clusters):
    return {frozenset(members) for members in clusters.values()}


def _assert_partition(clusters, signature_ids):
    members = [signature_id for cluster in clusters.values() for signature_id in cluster]
    assert sorted(members) == sorted(signature_ids)


def _without_reunite(monkeypatch):
    """Make the step a no-op, which is the behavior before the step existed."""
    monkeypatch.setattr(Clusterer, "_reunite_no_middle_dead_ends", lambda self, pred_clusters, **kwargs: pred_clusters)


def _record_attach_calls(monkeypatch, route):
    """Record each sequential attachment pass: its query groups, visible clusters and result."""
    calls = []
    method_name = ATTACH_METHODS[route]
    original = getattr(Clusterer, method_name)

    def record(self, groups, *, pred_clusters, **kwargs):
        call = {
            "groups": {key: list(members) for key, members in groups.items()},
            "visible": {key: list(members) for key, members in pred_clusters.items()},
        }
        call["result"] = original(self, groups, pred_clusters=pred_clusters, **kwargs)
        # A pass without groups returns its input unchanged.
        if groups:
            calls.append(call)
        return call["result"]

    monkeypatch.setattr(Clusterer, method_name, record)
    return calls


def _record_incremental_seeds(monkeypatch, route):
    """Record the seed map and attach context that each incremental call receives."""
    seen = []
    if route == "arrow":
        original = Clusterer.predict_incremental_from_arrow

        def record_arrow(self, block_signatures, arrow_dataset, **kwargs):
            seen.append(
                {
                    "queries": [str(signature_id) for signature_id in block_signatures],
                    "seeds": {str(key) for key in kwargs["cluster_seeds_require"]},
                    "name_check_exempt": _ARROW_ATTACH_NAME_CHECK_EXEMPT.get(),
                    "bulk_distance_overrides": _ARROW_BULK_DISTANCE_OVERRIDES.get(),
                }
            )
            return original(self, block_signatures, arrow_dataset, **kwargs)

        monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow", record_arrow)
    else:
        original_python = Clusterer._predict_incremental_python

        def record_python(self, block_signatures, dataset, **kwargs):
            seen.append(
                {
                    "queries": [str(signature_id) for signature_id in block_signatures],
                    "seeds": set(kwargs["prediction_state"].cluster_seeds_require),
                }
            )
            return original_python(self, block_signatures, dataset, **kwargs)

        monkeypatch.setattr(Clusterer, "_predict_incremental_python", record_python)
    return seen


@pytest.mark.parametrize("route", ROUTES)
def test_reunite_joins_no_middle_pieces_and_leaves_other_clusters(tmp_path, monkeypatch, route):
    dataset = _dataset(NAMES, {})
    blocks = {"w wang": list(dataset.signatures)}
    calls = _record_attach_calls(monkeypatch, route)

    clusters = _predict(tmp_path / "reunite", dataset, blocks, route)

    _assert_partition(clusters, blocks["w wang"])
    assert _partition(clusters) == {
        frozenset(NO_MIDDLE),
        frozenset({"10", "11", "12"}),
        frozenset({"9"}),
        frozenset({"13"}),
    }
    # The block has no initial-only group, so the reunite pass is the only attachment pass.
    [call] = calls
    # The pieces tie in size, so the first in key order keeps its bulk cluster as the seed. Each
    # other piece is one query group, in key order.
    assert _partition(call["visible"]) == {frozenset({"0", "1", "2"})}
    assert [set(members) for members in call["groups"].values()] == [{"3", "4", "5"}, {"6", "7", "8"}]
    assert sorted(call["groups"]) == list(call["groups"])

    _without_reunite(monkeypatch)
    before = _predict(tmp_path / "before", dataset, blocks, route)
    assert _partition(before) == {
        frozenset({"0", "1", "2"}),
        frozenset({"3", "4", "5"}),
        frozenset({"6", "7", "8"}),
        frozenset({"10", "11", "12"}),
        frozenset({"9"}),
        frozenset({"13"}),
    }


@pytest.mark.parametrize("route", ROUTES)
def test_reunite_never_shows_clusters_with_unmarked_records(tmp_path, monkeypatch, route):
    """Required members are not marked, so their bulk cluster stays hidden and unchanged."""
    dataset = _dataset(NAMES, {"3": "claimed", "4": "claimed"})
    blocks = {"w wang": list(dataset.signatures)}
    calls = _record_attach_calls(monkeypatch, route)
    seen = _record_incremental_seeds(monkeypatch, route)

    clusters = _predict(tmp_path / "reunite", dataset, blocks, route)

    _assert_partition(clusters, blocks["w wang"])
    marked = NO_MIDDLE - {"3", "4"}
    [call] = calls
    visible = {signature_id for members in call["visible"].values() for signature_id in members}
    queries = {signature_id for members in call["groups"].values() for signature_id in members}
    reunited = {signature_id for members in call["result"].values() for signature_id in members}
    assert visible | queries <= marked
    assert reunited <= marked
    assert all(call_seen["seeds"] <= marked and set(call_seen["queries"]) <= marked for call_seen in seen)
    # Record 5 is marked, but bulk put it with the required pair, so its cluster is hidden.
    assert _partition(clusters) == {
        frozenset({"0", "1", "2", "6", "7", "8"}),
        frozenset({"3", "4", "5"}),
        frozenset({"10", "11", "12"}),
        frozenset({"9"}),
        frozenset({"13"}),
    }

    _without_reunite(monkeypatch)
    before = _predict(tmp_path / "before", dataset, blocks, route)
    required_clusters = [members for members in clusters.values() if "3" in members]
    assert required_clusters == [members for members in before.values() if "3" in members]


@pytest.mark.parametrize("route", ROUTES)
def test_required_group_across_pieces_is_never_marked_or_split(tmp_path, monkeypatch, route):
    """Packing moves the required pair into the first piece; its repair parts are the pieces."""
    dataset = _dataset(NAMES, {"0": "claimed", "6": "claimed"})
    blocks = {"w wang": list(dataset.signatures)}
    calls = _record_attach_calls(monkeypatch, route)

    clusters = _predict(tmp_path, dataset, blocks, route)

    _assert_partition(clusters, blocks["w wang"])
    [call] = calls
    # The largest piece (3-5) keeps its bulk cluster as the seed, although the part with 2 sorts first.
    assert _partition(call["visible"]) == {frozenset({"3", "4", "5"})}
    assert [set(members) for members in call["groups"].values()] == [{"2"}, {"7", "8"}]
    assert _partition(clusters) == {
        frozenset({"0", "1", "6"}),
        frozenset({"2", "3", "4", "5", "7", "8"}),
        frozenset({"10", "11", "12"}),
        frozenset({"9"}),
        frozenset({"13"}),
    }


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("hao_count", [3, 6], ids=["second-dead-end-in-one-piece", "second-dead-end-cut"])
def test_reunite_works_on_each_dead_end_alone(tmp_path, monkeypatch, route, hao_count):
    """A second dead end ("hao") neither sees nor joins the "wei" clusters, and an uncut bucket stays."""
    dataset = _dataset([WEI] * 9 + [WEI_MIN] * 4 + [HAO] * hao_count + [HAO_XU], {})
    blocks = {"w wang": list(dataset.signatures)}
    hao = {str(index) for index in range(13, 13 + hao_count)}
    calls = _record_attach_calls(monkeypatch, route)

    clusters = _predict(tmp_path / "reunite", dataset, blocks, route)

    _assert_partition(clusters, blocks["w wang"])
    assert _partition(clusters) == {
        frozenset(NO_MIDDLE),
        frozenset(hao),
        frozenset({"10", "11", "12"}),
        frozenset({"9"}),
        frozenset({str(13 + hao_count)}),
    }
    wei_call = {"visible": {frozenset({"0", "1", "2"})}, "groups": [{"3", "4", "5"}, {"6", "7", "8"}]}
    hao_call = {"visible": {frozenset({"13", "14", "15"})}, "groups": [{"16", "17", "18"}]}
    # One pass per cut bucket, in bucket order. Each pass sees only the clusters of its bucket.
    expected_calls = [wei_call] if hao_count == 3 else [hao_call, wei_call]
    assert [
        {"visible": _partition(call["visible"]), "groups": [set(members) for members in call["groups"].values()]}
        for call in calls
    ] == expected_calls
    if hao_count == 3:
        _without_reunite(monkeypatch)
        before = _predict(tmp_path / "before", dataset, blocks, route)
        assert [members for members in clusters.values() if "13" in members] == [
            members for members in before.values() if "13" in members
        ]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("constraint_source", ["dataset_disallow", "partial_supervision"])
def test_reunite_keeps_disallows(tmp_path, route, constraint_source):
    dataset = _dataset(NAMES, {})
    kwargs = {}
    if constraint_source == "dataset_disallow":
        dataset.cluster_seeds_disallow = {("0", "6")}
    else:
        kwargs["partial_supervision"] = {("0", "6"): 10000.0}
    blocks = {"w wang": list(dataset.signatures)}

    clusters = _predict(tmp_path, dataset, blocks, route, **kwargs)

    _assert_partition(clusters, blocks["w wang"])
    assert not any({"0", "6"} <= set(members) for members in clusters.values())
    # The step reuses each route's attachment. The promoted Arrow linker decides each query alone,
    # while classic attachment first clusters a piece's queries and decides for the whole group.
    piece_6 = (
        [{"0", "1", "2", "3", "4", "5", "7", "8"}, {"6"}]
        if route == "arrow"
        else [NO_MIDDLE - {"6", "7", "8"}, {"6", "7", "8"}]
    )
    assert _partition(clusters) == {frozenset(members) for members in piece_6} | {
        frozenset({"10", "11", "12"}),
        frozenset({"9"}),
        frozenset({"13"}),
    }


def test_arrow_reunite_runs_with_the_attach_name_checks(tmp_path, monkeypatch):
    """The reunite pass uses the same name checks and bulk distance semantics as initial-only attachment."""
    dataset = _dataset(NAMES + [("W", "", "Wang")], {"9": "claimed", "10": "claimed"})
    blocks = {"w wang": list(dataset.signatures)}
    seen = _record_incremental_seeds(monkeypatch, "arrow")

    _predict(tmp_path, dataset, blocks, "arrow")

    reunite_calls = [call for call in seen if set(call["queries"]) <= NO_MIDDLE]
    initial_calls = [call for call in seen if call["queries"] == ["14"]]
    assert len(reunite_calls) == 2 and len(initial_calls) == 1
    # A non-None set turns the checks on. No required member is visible in the reunite pass.
    assert all(call["name_check_exempt"] == frozenset() for call in reunite_calls)
    assert initial_calls[0]["name_check_exempt"] == frozenset({"9", "10"})
    assert all(call["bulk_distance_overrides"] is True for call in seen)


@pytest.mark.parametrize("route", ROUTES)
def test_initial_only_attachment_runs_after_reunite_with_every_cluster(tmp_path, monkeypatch, route):
    dataset = _dataset(NAMES + [("W", "", "Wang")], {})
    blocks = {"w wang": list(dataset.signatures)}
    calls = _record_attach_calls(monkeypatch, route)

    clusters = _predict(tmp_path, dataset, blocks, route)

    _assert_partition(clusters, blocks["w wang"])
    reunite_call, initial_call = calls
    assert set(reunite_call["groups"]) != set(initial_call["groups"])
    assert [members for members in initial_call["groups"].values()] == [["14"]]
    assert _partition(initial_call["visible"]) == {
        frozenset(NO_MIDDLE),
        frozenset({"10", "11", "12"}),
        frozenset({"9"}),
        frozenset({"13"}),
    }


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize(
    "names",
    [
        pytest.param([WEI] * 3 + [WEI_MIN] * 2 + [WEN], id="no-middle-bucket-in-one-piece"),
        pytest.param([WEI, WEN, ("Wu", "", "Wang"), ("Wang", "", "Wang")], id="no-dead-end"),
        pytest.param([WEI] * 2 + [WEN], id="block-at-cap"),
        pytest.param([WEI] * 3 + [WEI_MIN] + [HAO] * 3 + [HAO_XU], id="two-dead-ends-each-in-one-piece"),
    ],
)
def test_reunite_is_a_no_op_without_marked_clusters_in_two_pieces(tmp_path, monkeypatch, route, names):
    dataset = _dataset(names, {})
    blocks = {"w wang": list(dataset.signatures)}
    calls = _record_attach_calls(monkeypatch, route)

    clusters = _predict(tmp_path / "reunite", dataset, blocks, route)
    assert calls == []

    _without_reunite(monkeypatch)
    before = _predict(tmp_path / "before", dataset, blocks, route)
    assert json.dumps(clusters) == json.dumps(before)


@pytest.mark.parametrize(
    ("seeds", "reunited"),
    [
        pytest.param({}, NO_MIDDLE, id="no-seeds"),
        pytest.param({"12": "claimed", "13": "claimed"}, NO_MIDDLE, id="required-group-elsewhere"),
        pytest.param({"0": "claimed", "6": "claimed"}, NO_MIDDLE - {"0", "1", "6"}, id="required-group-across-pieces"),
    ],
)
def test_reunite_gives_identical_clusters_on_both_routes_and_repeats(tmp_path, seeds, reunited):
    dataset = _dataset(NAMES, seeds)
    blocks = {"w wang": list(dataset.signatures)}

    python = _predict(tmp_path / "python", dataset, blocks, "python")
    arrow = _predict(tmp_path / "arrow", dataset, blocks, "arrow")

    assert _partition(python) == _partition(arrow)
    assert frozenset(reunited) in _partition(arrow)
    assert json.dumps(_predict(tmp_path / "python_again", dataset, blocks, "python")) == json.dumps(python)
    assert json.dumps(_predict(tmp_path / "arrow_again", dataset, blocks, "arrow")) == json.dumps(arrow)

"""Bulk distance overrides and Arrow ID normalization at the public boundary."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.consts import LARGE_DISTANCE
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.incremental_test_helpers import _dataset, _predict, attach_tiny_incremental_linker, prepare_seed_arrow_bundle
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors
from tests.model_helpers import ConstantDistanceClassifier

CASES = {
    "between_residual_queries": (
        [("Ada", "", "Smith"), ("A", "", "Smith"), ("A", "", "Smith")],
        {("1", "2"): 0.0},
        2,
        1,
    ),
    "outside_retrieval_window": (
        [("Ada", "", "Smith")] * 40 + [("A", "", "Smith")] * 2,
        {("40", "0"): 0.0},
        40,
        40,
    ),
    "conflicting_seed_components": (
        [("Ada", "", "Smith"), ("Ada", "", "Smith"), ("A", "", "Smith")],
        {("2", "0"): 0.0, ("2", "1"): 0.0},
        2,
        2,
    ),
}


def _clusterer(distance):
    return Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(distance),
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("route", ["arrow"])
def test_bulk_zero_overrides_do_not_require_incremental_links(tmp_path, case, route):
    names, supervision, threshold, _ = CASES[case]
    dataset = _dataset(names, {})
    blocks = {"one": list(dataset.signatures)}
    caller_supervision = dict(supervision)

    clusters = _predict(
        tmp_path,
        dataset,
        blocks,
        route,
        threshold=threshold,
        distance=0.9,
        partial_supervision=caller_supervision,
    )

    assert sorted(sid for members in clusters.values() for sid in members) == sorted(dataset.signatures)
    if case == "between_residual_queries":
        # The exact zero must survive into residual pairwise clustering.
        assert any({"1", "2"} <= set(members) for members in clusters.values())
    assert caller_supervision == supervision
    assert dataset.cluster_seeds_require == {}


@pytest.mark.parametrize("case", CASES)
def test_standalone_arrow_incremental_keeps_strict_zero_requirements(tmp_path, case):
    names, supervision, threshold, seed_count = CASES[case]
    if case == "outside_retrieval_window":
        # Equal-score candidates use lexical component order; component 39 is
        # outside the top 25 with these direct, singleton seed assignments.
        supervision = {("40", "39"): 0.0}
    dataset = _dataset(names, {})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    clusterer = _clusterer(0.9)
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker")
    seeds = {str(index): str(index) for index in range(seed_count)}
    queries = [str(index) for index in range(seed_count, len(names))]

    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        with pytest.raises(ValueError, match=f"partial_supervision_require_{case}"):
            clusterer.predict_incremental_from_arrow(
                queries,
                arrow_dataset,
                batching_threshold=threshold,
                partial_supervision=dict(supervision),
                cluster_seeds_require=seeds,
                name_tuples=frozenset(),
            )


# Entry stringifies str and int keys alike, so int keys alone catch a missed stringify.
@pytest.mark.parametrize("distance", [0.0, LARGE_DISTANCE])
@pytest.mark.parametrize("threshold", [None, 3])
def test_arrow_bulk_normalizes_ids_and_supervision_on_every_route(tmp_path, distance, threshold):
    names = [("Ada", "", "Smith")] * 4 + [("Bob", "", "Smith")] * 2
    dataset = _dataset(names, {})
    blocks = {"small": [0, 1], "big": [2, 3, 4, 5]}
    supervision = {(2, 3): distance}

    clusters = _predict(
        tmp_path,
        dataset,
        blocks,
        "arrow",
        threshold=threshold,
        distance=0.9 if distance == 0.0 else 0.1,
        partial_supervision=supervision,
    )

    assert sorted(sid for members in clusters.values() for sid in members) == list(dataset.signatures)
    assert any({"2", "3"} <= set(members) for members in clusters.values()) == (distance == 0.0)
    assert blocks == {"small": [0, 1], "big": [2, 3, 4, 5]}
    assert supervision == {(2, 3): distance}


def test_arrow_bulk_rejects_supervision_conflicts_after_id_normalization():
    clusterer = _clusterer(0.9)
    with pytest.raises(ValueError, match="conflicting distances"):
        clusterer.predict_from_arrow(
            {"one": [0, 1]},
            None,
            partial_supervision={(0, 1): 0.0, ("1", "0"): 0.9},
        )


@pytest.mark.parametrize("route", ["arrow"])
def test_all_initial_blocks_use_bulk_and_keep_zero_overrides(tmp_path, monkeypatch, route):
    def reject_incremental(*_args, **_kwargs):
        pytest.fail("All-initial bulk blocks must not enter incremental attachment")

    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow", reject_incremental)
    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow", reject_incremental)
    dataset = _dataset([("A", "", "Smith")] * 4, {})
    clusters = _predict(
        tmp_path,
        dataset,
        {"one": list(dataset.signatures)},
        route,
        threshold=2,
        distance=0.9,
        partial_supervision={("0", "1"): 0.0},
    )
    assert sorted(sid for members in clusters.values() for sid in members) == list(dataset.signatures)


@pytest.mark.parametrize("fail_bulk", [False, True])
def test_bulk_context_survives_public_wrappers_and_is_isolated(tmp_path, fail_bulk):
    dataset = _dataset([("Ada", "", "Smith"), ("A", "", "Smith"), ("A", "", "Smith")], {})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    calls = []

    class WrappedClusterer(Clusterer):
        def predict_incremental_from_arrow(self, *args, **kwargs):
            calls.append(kwargs)
            assert "partial_supervision_is_distance_override" not in kwargs
            if len(calls) == 1:
                # A concurrent standalone request must retain strict semantics
                # while this thread is inside bulk attachment.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    standalone = executor.submit(
                        Clusterer.predict_incremental_from_arrow,
                        self,
                        ["1", "2"],
                        args[1],
                        cluster_seeds_require={"0": "seed"},
                        partial_supervision={("1", "2"): 0.0},
                        name_tuples=frozenset(),
                    )
                    with pytest.raises(ValueError, match="partial_supervision_require_between_residual_queries"):
                        standalone.result()
                if fail_bulk:
                    raise RuntimeError("custom attachment failure")
            return super().predict_incremental_from_arrow(*args, **kwargs)

    clusterer = WrappedClusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.9),
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
    )
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker")
    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:

        def predict_bulk():
            return clusterer.predict_from_arrow(
                {"one": ["0", "1", "2"]},
                arrow_dataset,
                batching_threshold=2,
                partial_supervision={("1", "2"): 0.0},
                name_tuples=frozenset(),
            )

        if fail_bulk:
            with pytest.raises(RuntimeError, match="custom attachment failure"):
                predict_bulk()
        else:
            clusters, _ = predict_bulk()
            assert any({"1", "2"} <= set(members) for members in clusters.values())
        assert len(calls) == 1
        # Bulk context must reset on both success and an override exception.
        with pytest.raises(ValueError, match="partial_supervision_require_between_residual_queries"):
            clusterer.predict_incremental_from_arrow(
                ["1", "2"],
                arrow_dataset,
                cluster_seeds_require={"0": "seed"},
                partial_supervision={("1", "2"): 0.0},
                name_tuples=frozenset(),
            )
        assert len(calls) == 2


def test_attach_name_check_covers_only_initial_only_attachment(tmp_path, monkeypatch):
    """Subblocked attachment exempts caller-required seeds; the public API stays unchanged."""
    import s2and.model as model_module

    dataset = _dataset([("Ada", "", "Smith"), ("A", "", "Smith"), ("A", "", "Smith")], {})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    observed = []
    original = model_module.predict_incremental_promoted_linker_from_arrow

    def capture(clusterer, block_signatures, request_dataset, **kwargs):
        observed.append(request_dataset.name_check_exempt_signature_ids)
        return original(clusterer, block_signatures, request_dataset, **kwargs)

    monkeypatch.setattr(model_module, "predict_incremental_promoted_linker_from_arrow", capture)
    clusterer = _clusterer(0.9)
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker")
    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        clusterer.predict_from_arrow(
            {"one": ["0", "1", "2"]},
            arrow_dataset,
            batching_threshold=2,
            cluster_seeds_require={"0": "profile"},
            name_tuples=frozenset(),
        )
        assert observed == [frozenset({"0"})]
        clusterer.predict_incremental_from_arrow(
            ["1", "2"],
            arrow_dataset,
            cluster_seeds_require={"0": "seed"},
            name_tuples=frozenset(),
        )
    assert observed[-1] is None
    assert model_module._ARROW_ATTACH_NAME_CHECK_EXEMPT.get() is None


@pytest.mark.parametrize("prevent_new_incompatibilities", [True, False])
def test_incremental_keeps_a_required_link_to_an_altered_profile(tmp_path, prevent_new_incompatibilities):
    # Seeds Alan and Brian form one altered profile; the caller requires Caleb to join Alan.
    # The final name check must not undo the required link or hide it from the residual count.
    names = [("Alan", "", "Lee"), ("Brian", "", "Lee"), ("Caleb", "", "Lee")]
    dataset = _dataset(names, {"0": "anchor", "1": "anchor"})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]), ConstantDistanceClassifier(0.1), cluster_model=FastCluster(eps=0.5), n_jobs=1
    )
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker", link=False)

    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        result = clusterer.predict_incremental_from_arrow(
            ["2"],
            arrow_dataset,
            partial_supervision={("0", "2"): 0.0},
            cluster_seeds_require={"0": "anchor", "1": "anchor"},
            altered_cluster_signatures=["0"],
            prevent_new_incompatibilities=prevent_new_incompatibilities,
            name_tuples=frozenset(),
        )

    assert sorted(result["clusters"]["anchor"]) == ["0", "1", "2"]
    assert result["phase_b_residual_count"] == 0


def test_incremental_residuals_keep_equivalent_cjk_given_names_together(tmp_path):
    # U+FA45 is a compatibility form of 海; full-block clustering merges these two mentions.
    names = [("Alice", "", "Lee"), ("\ufa45\u62d3", "", "Lee"), ("\u6d77\u62d3", "", "Lee")]
    dataset = _dataset(names, {"0": "anchor"})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    clusterer = _clusterer(0.1)
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker", link=False)

    with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
        result = clusterer.predict_incremental_from_arrow(
            ["1", "2"], arrow_dataset, cluster_seeds_require={"0": "anchor"}, name_tuples=frozenset()
        )

    assert sorted(sorted(members) for members in result["clusters"].values()) == [["0"], ["1", "2"]]


@pytest.mark.parametrize("distance", [0.0, 0.2, LARGE_DISTANCE - 1])
@pytest.mark.parametrize("route", ["python", "arrow"])
def test_bulk_rejects_a_caller_distance_that_joins_two_seed_groups(tmp_path, route, distance):
    # Seed restoration would merge both profiles through the one pair, past the hard pair (1, 2).
    dataset = _dataset([("Ada", "", "Smith")] * 3, {"0": "profileA", "1": "profileA", "2": "profileB"})
    blocks = {"probe": ["0", "2", "1"]}
    supervision = {("2", "0"): distance, ("0", "1"): 0.2}
    clusterer = _clusterer(0.1)
    with pytest.raises(ValueError, match=r"partial_supervision_joins_seed_groups: pair=\('0', '2'\)"):
        if route == "python":
            clusterer.predict(blocks, dataset, partial_supervision=supervision)
        else:
            prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
            with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
                clusterer.predict_from_arrow(
                    blocks,
                    arrow_dataset,
                    partial_supervision=supervision,
                    cluster_seeds_require=dataset.cluster_seeds_require,
                    name_tuples=frozenset(),
                )


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_bulk_accepts_a_hard_distance_between_seed_groups(tmp_path, route):
    dataset = _dataset([("Ada", "", "Smith")] * 3, {"0": "profileA", "1": "profileA", "2": "profileB"})
    blocks = {"probe": ["0", "2", "1"]}
    supervision = {("0", "2"): LARGE_DISTANCE, ("0", "1"): 0.2}
    clusterer = _clusterer(0.1)
    if route == "python":
        clusters, _ = clusterer.predict(blocks, dataset, partial_supervision=supervision)
    else:
        prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
        with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
            clusters, _ = clusterer.predict_from_arrow(
                blocks,
                arrow_dataset,
                partial_supervision=supervision,
                cluster_seeds_require=dataset.cluster_seeds_require,
                name_tuples=frozenset(),
            )
    assert sorted(sorted(members) for members in clusters.values()) == [["0", "1"], ["2"]]


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_bulk_ignores_a_caller_distance_outside_the_blocks(tmp_path, route):
    # Pair (2, 3) joins two seed groups, but signature 3 is in no block of this request.
    dataset = _dataset([("Ada", "", "Smith")] * 4, {"0": "profileA", "1": "profileA", "2": "profileB", "3": "profileA"})
    blocks = {"probe": ["0", "2", "1"]}
    supervision = {("2", "3"): 0.0, ("0", "1"): 0.2}
    clusterer = _clusterer(0.1)
    if route == "python":
        clusters, _ = clusterer.predict(blocks, dataset, partial_supervision=supervision)
    else:
        prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
        with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
            clusters, _ = clusterer.predict_from_arrow(
                blocks,
                arrow_dataset,
                partial_supervision=supervision,
                cluster_seeds_require=dataset.cluster_seeds_require,
                name_tuples=frozenset(),
            )
    assert sorted(sorted(members) for members in clusters.values()) == [["0", "1"], ["2"]]


def test_predict_helper_rejects_a_caller_distance_that_joins_two_seed_groups():
    dataset = _dataset([("Ada", "", "Smith")] * 3, {"0": "profileA", "1": "profileA", "2": "profileB"})
    with pytest.raises(ValueError, match="partial_supervision_joins_seed_groups"):
        _clusterer(0.1).predict_helper({"probe": ["0", "1", "2"]}, dataset, partial_supervision={("0", "2"): 0.0})


def test_python_seed_check_follows_the_default_constraint_switch():
    # Without default constraints the Python route does not separate seed groups.
    dataset = _dataset([("Ada", "", "Smith")] * 2, {"0": "profileA", "1": "profileB"})
    clusterer = _clusterer(0.1)
    clusterer.use_default_constraints_as_supervision = False
    clusters, _ = clusterer.predict({"probe": ["0", "1"]}, dataset, partial_supervision={("0", "1"): 0.1})
    assert sorted(len(members) for members in clusters.values()) == [2]


def test_seed_check_skips_pairs_outside_the_requested_blocks():
    # Residual clustering clusters queries only; seed pairs elsewhere are not its concern.
    dataset = _dataset([("Ada", "", "Smith")] * 4, {"0": "profileA", "1": "profileB"})
    clusters, _ = _clusterer(0.1).predict_helper(
        {"probe": ["2", "3"]}, dataset, partial_supervision={("0", "1"): 0.0, ("2", "3"): 0.1}
    )
    assert sorted(len(members) for members in clusters.values()) == [2]

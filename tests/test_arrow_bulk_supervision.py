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


@pytest.mark.parametrize("case", ["between_residual_queries", "outside_retrieval_window"])
def test_bulk_zero_overrides_do_not_require_incremental_links(tmp_path, case):
    names, supervision, threshold, _ = CASES[case]
    dataset = _dataset(names, {})
    blocks = {"one": list(dataset.signatures)}
    caller_supervision = dict(supervision)

    clusters = _predict(
        tmp_path,
        dataset,
        blocks,
        "arrow",
        threshold=threshold,
        distance=0.9,
        partial_supervision=caller_supervision,
    )

    assert sorted(sid for members in clusters.values() for sid in members) == sorted(dataset.signatures)
    if case == "between_residual_queries":
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
@pytest.mark.parametrize(("distance", "threshold"), [(LARGE_DISTANCE, None), (0.0, 3)])
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


def test_all_initial_blocks_use_bulk_and_keep_zero_overrides(tmp_path, monkeypatch):
    def reject_incremental(*_args, **_kwargs):
        pytest.fail("All-initial bulk blocks must not enter incremental attachment")

    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow", reject_incremental)
    dataset = _dataset([("A", "", "Smith")] * 4, {})
    clusters = _predict(
        tmp_path,
        dataset,
        {"one": list(dataset.signatures)},
        "arrow",
        threshold=2,
        distance=0.9,
        partial_supervision={("0", "1"): 0.0},
    )
    assert sorted(sid for members in clusters.values() for sid in members) == list(dataset.signatures)


def test_bulk_context_survives_public_wrappers_and_is_isolated(tmp_path):
    fail_bulk = True
    dataset = _dataset([("Ada", "", "Smith"), ("A", "", "Smith"), ("A", "", "Smith")], {})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    calls = []

    class WrappedClusterer(Clusterer):
        def predict_incremental_from_arrow(self, *args, **kwargs):
            calls.append(kwargs)
            assert "partial_supervision_is_distance_override" not in kwargs
            if fail_bulk and len(calls) == 1:
                # Keep standalone validation strict during a concurrent bulk attachment.
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

        def check_standalone():
            with pytest.raises(ValueError, match="partial_supervision_require_between_residual_queries"):
                clusterer.predict_incremental_from_arrow(
                    ["1", "2"],
                    arrow_dataset,
                    cluster_seeds_require={"0": "seed"},
                    partial_supervision={("1", "2"): 0.0},
                    name_tuples=frozenset(),
                )

        with pytest.raises(RuntimeError, match="custom attachment failure"):
            predict_bulk()
        check_standalone()
        fail_bulk = False
        clusters, _ = predict_bulk()
        assert any({"1", "2"} <= set(members) for members in clusters.values())
        check_standalone()
        assert len(calls) == 4


def test_attach_name_check_covers_only_initial_only_attachment(tmp_path, monkeypatch):
    """Subblocked attachment exempts caller-required seeds; the public API stays unchanged."""
    import s2and.model as model_module
    from s2and.model import incremental

    dataset = _dataset([("Ada", "", "Smith"), ("A", "", "Smith"), ("A", "", "Smith")], {})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    observed = []
    original = incremental.predict_incremental_promoted_linker_from_arrow

    def capture(clusterer, block_signatures, request_dataset, **kwargs):
        observed.append(request_dataset.name_check_exempt_signature_ids)
        return original(clusterer, block_signatures, request_dataset, **kwargs)

    monkeypatch.setattr(incremental, "predict_incremental_promoted_linker_from_arrow", capture)
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


@pytest.mark.parametrize("route", ["python", "arrow"])
def test_bulk_caller_distances_preserve_seed_groups(tmp_path, route):
    dataset = _dataset([("Ada", "", "Smith")] * 4, {"0": "profileA", "1": "profileA", "2": "profileB", "3": "profileA"})
    clusterer = _clusterer(0.1)
    blocks = {"probe": ["0", "2", "1"], "other": ["3"]}

    def check(predict):
        # Restoring seeds would merge profiles through a caller-supplied soft distance.
        with pytest.raises(ValueError, match="partial_supervision_joins_seed_groups"):
            predict({("2", "0"): 0.2})
        clusters, _ = predict({("0", "2"): LARGE_DISTANCE, ("2", "3"): 0.0, ("0", "1"): 0.2})
        assert sorted(sorted(members) for members in clusters.values()) == [["0", "1"], ["2"], ["3"]]

    if route == "python":
        check(lambda supervision: clusterer.predict(blocks, dataset, partial_supervision=supervision))
    else:
        prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
        with ArrowDataset.open(tmp_path / "arrow") as arrow_dataset:
            check(
                lambda supervision: clusterer.predict_from_arrow(
                    blocks,
                    arrow_dataset,
                    partial_supervision=supervision,
                    cluster_seeds_require=dataset.cluster_seeds_require,
                    name_tuples=frozenset(),
                )
            )


def test_predict_helper_limits_seed_validation_to_the_requested_block():
    dataset = _dataset([("Ada", "", "Smith")] * 4, {"0": "profileA", "1": "profileB"})
    clusterer = _clusterer(0.1)
    supervision = {("0", "1"): 0.0, ("2", "3"): 0.1}

    with pytest.raises(ValueError, match="partial_supervision_joins_seed_groups"):
        clusterer.predict_helper({"probe": ["0", "1"]}, dataset, partial_supervision=supervision)

    clusters, _ = clusterer.predict_helper({"probe": ["2", "3"]}, dataset, partial_supervision=supervision)
    assert list(clusters.values()) == [["2", "3"]]

    clusterer.use_default_constraints_as_supervision = False
    clusters, _ = clusterer.predict({"probe": ["0", "1"]}, dataset, partial_supervision=supervision)
    assert list(clusters.values()) == [["0", "1"]]

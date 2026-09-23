"""Regression coverage for distinct original and split seed maps at both routes."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import s2and.incremental_linking.production as production_module
from s2and.featurizer import FeaturizationInfo
from s2and.model import Clusterer, FastCluster
from tests.helpers import incremental_seed_setup
from tests.model_helpers import ConstantDistanceClassifier
from tests.test_cluster_incremental import (
    _minimal_arrow_dataset,
    _mock_promoted_limits,
    _patch_fake_raw_arrow_planner,
    _unexpected_residual_clustering,
)
from tests.test_subblocked_seed_boundaries import _dataset


@pytest.mark.parametrize("route", ["python", "promoted"])
@pytest.mark.parametrize(
    ("linked_seed_name", "query_name", "accepted"),
    [("D", "Donald", True), ("Donald", "David", False)],
    ids=["initial_split_accepts_new_full_name", "full_name_split_rejects_original_profile_match"],
)
def test_incremental_routes_use_split_members_for_names_and_original_members_for_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    route: str,
    linked_seed_name: str,
    query_name: str,
    accepted: bool,
) -> None:
    """Changing either inverse map must alter the real assignment result.

    Retrieval/scoring is fixed at the proposed split link. Assignment and
    completion run unchanged, including rejection to a residual singleton.
    """

    def signature(first: str) -> SimpleNamespace:
        return SimpleNamespace(
            author_info_first=first,
            author_info_first_normalized_without_apostrophe=first.lower(),
            author_info_last="Jones",
            author_info_orcid=None,
            paper_id=f"p-{first}",
        )

    signatures = {
        "seed_original": signature("David"),
        "seed_linked": signature(linked_seed_name),
        "query": signature(query_name),
    }
    original_inverse: dict[int | str, list[str]] = {"claimed": ["seed_original", "seed_linked"]}
    split_inverse: dict[int | str, list[str]] = {"split_other": ["seed_original"], "split_linked": ["seed_linked"]}
    split_seed_map = {"seed_original": "split_other", "seed_linked": "split_linked"}
    recluster_map: dict[int | str, int | str] = {"split_other": "claimed", "split_linked": "claimed"}
    original_seed_map = {"seed_original": "claimed", "seed_linked": "claimed"}
    dataset = production_module._DirectArrowIncrementalDataset(
        signatures=signatures,
        name_tuples=set(),
        cluster_seeds_require=original_seed_map,
        cluster_seeds_disallow=set(),
        altered_cluster_signatures=["seed_original"],
        max_seed_cluster_id=0,
    )
    clusterer = SimpleNamespace(
        use_default_constraints_as_supervision=True,
        suppress_orcid=True,
        n_jobs=1,
        featurizer_info=FeaturizationInfo(features_to_use=["year_diff", "misc_features"]),
        feature_contract={},
        _build_incremental_seed_setup=lambda *_args, **_kwargs: incremental_seed_setup(
            split_seed_map,
            recluster_map,
            original_inverse,
            split_inverse,
        ),
    )
    runtime_context = cast(Any, SimpleNamespace(run_id="seed-inverse-wiring"))
    if route == "python":
        clusters = Clusterer._finish_incremental_with_seed_links(
            cast(Any, clusterer),
            ["query"],
            cast(Any, dataset),
            {"query": "split_linked"},
            recluster_map,
            original_inverse,
            prevent_new_incompatibilities=True,
            partial_supervision={},
            runtime_context=runtime_context,
            split_cluster_seeds_require_inverse=split_inverse,
        )
    else:
        _patch_fake_raw_arrow_planner(monkeypatch)
        monkeypatch.setattr(
            production_module,
            "compute_promoted_incremental_limits",
            lambda **_kwargs: _mock_promoted_limits(),
        )
        monkeypatch.setattr(
            production_module.runtime_module,
            "_predict_incremental_link_or_abstain_from_preplanned_raw_arrow",
            lambda *_args, **_kwargs: SimpleNamespace(
                linked_signature_clusters={"query": "split_linked"},
                telemetry={"query_count": 1, "candidate_row_count": 1, "pair_count": 1},
            ),
        )
        payload = production_module.predict_incremental_promoted_linker_from_arrow(
            clusterer,
            list(signatures),
            dataset,
            cluster_residuals=_unexpected_residual_clustering,
            arrow_dataset=_minimal_arrow_dataset(tmp_path),
            artifact=cast(
                Any,
                SimpleNamespace(artifact_dir=tmp_path, retrieval_top_k=25, feature_columns=("test_link_probability",)),
            ),
            prevent_new_incompatibilities=True,
            partial_supervision={},
            runtime_context=runtime_context,
            total_ram_bytes=None,
            batching_threshold=None,
        )
        clusters = payload["clusters"]

    expected = (
        {"claimed": ["seed_original", "seed_linked", "query"]}
        if accepted
        else {
            "claimed": ["seed_original", "seed_linked"],
            "0": ["query"],
        }
    )
    assert clusters == expected
    assert original_inverse == {"claimed": ["seed_original", "seed_linked"]}
    assert split_inverse == {"split_other": ["seed_original"], "split_linked": ["seed_linked"]}


@pytest.mark.parametrize(
    ("linked_seed_name", "query_name", "accepted"),
    [("D", "Donald", True), ("Donald", "David", False)],
    ids=["initial_split_accepts_new_full_name", "full_name_split_rejects_original_profile_match"],
)
def test_classic_incremental_hands_seed_setup_maps_to_completion(
    monkeypatch: pytest.MonkeyPatch, linked_seed_name: str, query_name: str, accepted: bool
) -> None:
    """The public Python route must pass setup's split and original maps through.

    Only seed setup is fixed; phases B-D, seed-link assignment and completion run.
    """
    dataset = _dataset([("David", "", "Jones"), (linked_seed_name, "", "Jones"), (query_name, "", "Jones")], {})
    dataset.cluster_seeds_require = {"0": "claimed", "1": "claimed"}
    dataset.max_seed_cluster_id = 1
    original_members: dict[int | str, list[str]] = {"claimed": ["0", "1"]}
    split_members: dict[int | str, list[str]] = {"split_other": ["0"], "split_linked": ["1"]}
    setup = incremental_seed_setup(
        {"0": "split_other", "1": "split_linked"},
        {"split_other": "claimed", "split_linked": "claimed"},
        original_members,
        split_members,
    )
    monkeypatch.setattr(Clusterer, "_build_incremental_seed_setup", lambda *_args, **_kwargs: setup)
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.9),
        cluster_model=FastCluster(eps=0.5),
        n_jobs=1,
    )

    result = clusterer.predict_incremental(
        ["2"],
        dataset,
        prevent_new_incompatibilities=True,
        partial_supervision={("2", "1"): 0.0},
    )

    partition = {frozenset(members) for members in result["clusters"].values()}
    assert partition == ({frozenset({"0", "1", "2"})} if accepted else {frozenset({"0", "1"}), frozenset({"2"})})
    assert result["clusters"]["claimed"][:2] == ["0", "1"]
    assert original_members == {"claimed": ["0", "1"]}
    assert split_members == {"split_other": ["0"], "split_linked": ["1"]}

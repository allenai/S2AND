"""Check exact offline attachment reuse against independent requests."""

import json
from dataclasses import replace

import numpy as np
import pytest

from s2and.arrow_inputs import ArrowDataset
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking import runtime
from s2and.incremental_linking.attach_reuse import AttachReuse, retained_bytes
from s2and.model import Clusterer, FastCluster, distances
from s2and.model.incremental import _AttachmentBlockState
from s2and.prediction_state import PredictionState
from s2and.production_model import NativeLightGBMBinaryClassifier
from tests.incremental_test_helpers import _dataset, attach_tiny_incremental_linker, prepare_seed_arrow_bundle
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors
from tests.promoted_linking_helpers import tiny_binary_booster


def _native_classifier(tmp_path, info, seed):
    path = tmp_path / f"pairwise-{seed}.txt"
    tiny_binary_booster(len(info.selected_feature_indices()), seed=seed).booster_.save_model(path)
    return NativeLightGBMBinaryClassifier(path)


@pytest.fixture
def multi_pass_block(tmp_path):
    names = [("Wei", "", "Wang")] * 9 + [("Wei", "Min", "Wang")] * 4 + [("W", "", "Wang")] * 4
    dataset = _dataset(names, {})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    blocks = {"w wang": list(dataset.signatures)}
    info = FeaturizationInfo(["year_diff"])
    clusterer = Clusterer(
        info,
        _native_classifier(tmp_path, info, 101),
        cluster_model=FastCluster(linkage="average"),
        n_jobs=1,
    )
    clusterer.suppress_orcid = True
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker")
    with ArrowDataset.open(tmp_path / "arrow") as arrow:
        yield arrow, blocks, clusterer


def test_setup_and_residual_reuse_across_eps(multi_pass_block, tmp_path, monkeypatch):
    arrow, blocks, clusterer = multi_pass_block
    attach_tiny_incremental_linker(clusterer, tmp_path / "abstain", link=False)
    calls = []
    original = Clusterer.predict_incremental_from_arrow

    def record(self, signature_ids, *args, **kwargs):
        calls.append(tuple(signature_ids))
        return original(self, signature_ids, *args, **kwargs)

    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow", record)
    with AttachReuse() as reuse:

        def predict(eps, reuse=None, threshold=3):
            calls.clear()
            state = PredictionState()
            clusters, _ = clusterer._predict_from_arrow_request(
                blocks,
                arrow,
                cluster_model_params={"eps": eps},
                batching_threshold=threshold,
                name_tuples=frozenset(),
                prediction_state=state,
                attach_reuse=reuse,
            )
            assert state.telemetry["subblocked_reunite"]["query_group_count"] >= 1
            assert len(calls) >= 2
            return json.dumps(clusters).encode()

        for eps in (0.36, 0.46):
            assert predict(eps, reuse) == predict(eps)
        assert reuse.subblocks.hits == 1
        assert len(reuse.blocks.entries) == 1
        assert reuse.blocks.hits >= 1
        assert reuse.residuals.hits >= 1
        assert predict(0.46, reuse, threshold=4) == predict(0.46, threshold=4)
        assert reuse.subblocks.hits == 1
        assert len(reuse.subblocks.entries) == 2


def test_residual_model_cache_reapplies_constraints_and_checks_model(tmp_path):
    names = [("Ada", "", "Smith"), ("Bob", "", "Smith"), ("Ada", "", "Smith"), ("A", "", "Smith")]
    dataset = _dataset(names, {})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    info = FeaturizationInfo(["year_diff"])

    clusterer = Clusterer(
        info, _native_classifier(tmp_path, info, 101), cluster_model=FastCluster(linkage="average"), n_jobs=1
    )
    clusterer.suppress_orcid = True
    with ArrowDataset.open(tmp_path / "arrow") as arrow, AttachReuse() as reuse:
        group = list(dataset.signatures)
        state = _AttachmentBlockState(arrow, group, frozenset(), 1)
        state.attach_reuse = reuse
        state.residual_group = tuple(group)

        def verify(ids, supervision, *, cached=True):
            args = (clusterer, {"block": ids}, state.featurizer())
            expected = distances._make_distance_matrices_from_verified_rust_featurizer(
                *args, partial_supervision=supervision
            )["block"]
            observed = distances._make_distance_matrices_from_verified_rust_featurizer(
                *args, partial_supervision=supervision, residual_reuse=state if cached else None
            )["block"]
            np.testing.assert_array_equal(observed, expected)
            assert observed.tobytes() == expected.tobytes()

        try:
            verify(group, {})
            verify(group[1:], {("1", "2"): 0.123})
            verify(["0", "2", "3"], {("3", "0"): 0.456})
            assert len(reuse.residuals.entries) == 1
            assert reuse.residuals.hits == 2
            verify(list(reversed(group)), {})
            assert reuse.residuals.hits == 3
            clusterer.classifier = _native_classifier(tmp_path, info, 102)
            verify(group[1:], {})
            assert len(reuse.residuals.entries) == 2
            assert reuse.residuals.hits == 3
            clusterer.suppress_orcid = False
            verify(group[1:], {})
            assert reuse.residuals.hits == 3
        finally:
            state.release()


def test_snapshot_reuses_only_the_first_pass_and_checks_retrieval_key(multi_pass_block, tmp_path, monkeypatch):
    arrow, blocks, clusterer = multi_pass_block
    link = clusterer.incremental_linker_artifact
    attach_tiny_incremental_linker(clusterer, tmp_path / "abstain", link=False)
    abstain = clusterer.incremental_linker_artifact
    original = runtime.compute_candidate_batch_pairwise_model_and_aggregate_stats
    score_calls = []

    def record(*args, **kwargs):
        score_calls.append(args[1].pair_count)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "compute_candidate_batch_pairwise_model_and_aggregate_stats", record)

    def predict(artifact, reuse=None):
        score_calls.clear()
        clusterer.incremental_linker_artifact = artifact
        state = PredictionState()
        clusters, _ = clusterer._predict_from_arrow_request(
            blocks,
            arrow,
            cluster_model_params={"eps": 0.36},
            batching_threshold=3,
            name_tuples=frozenset(),
            prediction_state=state,
            attach_reuse=reuse,
        )
        assert state.telemetry["subblocked_reunite"]["query_group_count"] >= 1
        return json.dumps(clusters).encode(), len(score_calls)

    with AttachReuse(snapshot_eps=0.36) as reuse:
        baseline_link, link_calls = predict(link)
        baseline_abstain, abstain_calls = predict(abstain)
        assert baseline_link != baseline_abstain
        assert min(link_calls, abstain_calls) >= 2
        assert predict(link, reuse) == (baseline_link, link_calls)
        assert predict(abstain, reuse) == (baseline_abstain, abstain_calls - 1)
        assert reuse.snapshot_plans.hits == reuse.snapshots.hits == 1
        assert len(reuse.snapshots.entries) == 1
        changed_top_k = replace(abstain, retrieval_top_k=1)
        assert predict(changed_top_k, reuse) == predict(changed_top_k)
        assert reuse.snapshot_plans.hits == reuse.snapshots.hits == 1
        assert len(reuse.snapshots.entries) == 2


def test_snapshot_budget_counts_external_array_buffers():
    data = np.frombuffer(bytearray(8192), dtype=np.uint32)
    assert not data.flags.owndata
    assert retained_bytes(data[1:2]) >= data.nbytes
    assert retained_bytes((data[:32], data[32:])) < 2 * data.nbytes

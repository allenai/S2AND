"""Check exact offline attachment reuse against independent requests."""

import json

import numpy as np
from lightgbm import LGBMClassifier

from s2and.arrow_inputs import ArrowDataset
from s2and.featurizer import FeaturizationInfo
from s2and.incremental_linking.attach_reuse import AttachReuse
from s2and.model import Clusterer, FastCluster, distances
from s2and.model.incremental import _AttachmentBlockState
from s2and.prediction_state import PredictionState
from s2and.production_model import NativeLightGBMBinaryClassifier
from tests.incremental_test_helpers import _dataset, attach_tiny_incremental_linker, prepare_seed_arrow_bundle
from tests.incremental_test_helpers import empty_orcid_priors as empty_orcid_priors
from tests.model_helpers import ConstantDistanceClassifier


def test_setup_reuse_across_eps(tmp_path, monkeypatch):
    names = [("Wei", "", "Wang")] * 9 + [("Wei", "Min", "Wang")] * 4 + [("W", "", "Wang")] * 4
    dataset = _dataset(names, {})
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    blocks = {"w wang": list(dataset.signatures)}
    clusterer = Clusterer(
        FeaturizationInfo(["year_diff"]),
        ConstantDistanceClassifier(0.39),
        cluster_model=FastCluster(linkage="average"),
        n_jobs=1,
    )
    clusterer.suppress_orcid = True
    attach_tiny_incremental_linker(clusterer, tmp_path / "linker")
    calls = []
    original = Clusterer.predict_incremental_from_arrow

    def record(self, signature_ids, *args, **kwargs):
        calls.append(tuple(signature_ids))
        return original(self, signature_ids, *args, **kwargs)

    monkeypatch.setattr(Clusterer, "predict_incremental_from_arrow", record)
    with ArrowDataset.open(tmp_path / "arrow") as arrow, AttachReuse() as reuse:

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
        assert predict(0.46, reuse, threshold=4) == predict(0.46, threshold=4)
        assert reuse.subblocks.hits == 1
        assert len(reuse.subblocks.entries) == 2


def test_residual_model_cache_reapplies_constraints_and_checks_model(tmp_path):
    ada = ("Ada", "", "Smith")
    names = [ada, ("Bob", "", "Smith"), ada, ("A", "", "Smith"), ada, ("Ada", "May", "Smith")]
    dataset = _dataset(names, {}, years=(2000, 2001, 2000, 2012, 2001, 2004))
    prepare_seed_arrow_bundle(dataset, tmp_path / "arrow")
    info = FeaturizationInfo(["name_similarity", "year_diff"])

    def classifier(*, invert=False):
        path = tmp_path / f"pairwise-{invert}.txt"
        features = info.get_feature_names()
        year_gaps = np.repeat(np.arange(13), 4)
        matrix = np.zeros((len(year_gaps), len(features)))
        matrix[:, features.index("year_diff")] = year_gaps
        labels = np.tile(np.arange(4), 13) < 4 - np.minimum(year_gaps, 3)
        booster = LGBMClassifier(n_jobs=1, n_estimators=4, min_child_samples=1, verbosity=-1)
        booster.fit(matrix, labels ^ invert).booster_.save_model(path)
        return NativeLightGBMBinaryClassifier(path)

    clusterer = Clusterer(info, classifier(), cluster_model=FastCluster(linkage="average"), n_jobs=1)
    clusterer.suppress_orcid = True
    with ArrowDataset.open(tmp_path / "arrow") as arrow, AttachReuse() as reuse:
        group = list(dataset.signatures)
        state = _AttachmentBlockState(arrow, group, frozenset(), 1)
        state.attach_reuse = reuse
        state.residual_group = tuple(group)
        native = state.featurizer()
        scored_counts = []

        class RecordingFeaturizer:
            def __getattr__(self, name):
                return getattr(native, name)

            def fill_block_distances_indexed(self, *args, **kwargs):
                counts = native.fill_block_distances_indexed(*args, **kwargs)
                scored_counts.append(counts[1])
                return counts

        def verify(ids, supervision):
            expected = distances._make_distance_matrices_from_verified_rust_featurizer(
                clusterer, {"block": ids}, native, partial_supervision=supervision
            )["block"]
            observed = distances._make_distance_matrices_from_verified_rust_featurizer(
                clusterer, {"block": ids}, RecordingFeaturizer(), partial_supervision=supervision, residual_reuse=state
            )["block"]
            np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))

        try:
            first = ["0", "2", "4", "5"]
            verify(first, {})
            [cached] = reuse.residuals.entries.values()
            filled = cached.values[~np.isnan(cached.values)]
            assert len(filled) == 6
            assert len(np.unique(filled)) >= 2
            verify(["1", "2", "4", "5"], {("1", "2"): 0.123})
            verify(first, {("5", "0"): 0.456})
            verify(group, {})
            assert scored_counts == [6, 3, 0, 6]
            assert not np.isnan(cached.values).any()
            assert reuse.residuals.hits == 3
            verify(list(reversed(group)), {})
            assert reuse.residuals.hits == 4
            clusterer.classifier = classifier(invert=True)
            verify(group[1:], {})
            assert len(reuse.residuals.entries) == 2
            assert scored_counts[-1] == 10
            assert reuse.residuals.hits == 4
            clusterer.suppress_orcid = False
            verify(group[1:], {})
            assert reuse.residuals.hits == 4
        finally:
            state.release()

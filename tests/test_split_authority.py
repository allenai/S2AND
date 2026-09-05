"""Keep training, calibration and evaluation on the configured signature splits."""

import copy
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hyperopt import Trials

from s2and.data import ANDData, _resolve_signature_splits
from s2and.eval import cluster_eval, incremental_cluster_eval
from s2and.featurizer import FeaturizationInfo, resolve_training_pairs
from s2and.model import Clusterer


@pytest.fixture
def dataset() -> ANDData:
    """Create 20 two-signature blocks with distinct chronological years."""
    signature_template = json.loads(Path("tests/dummy/signatures.json").read_text())["0"]
    paper_template = json.loads(Path("tests/dummy/papers.json").read_text())["53235312"]
    signatures, papers, clusters = {}, {}, {}
    for block in range(20):
        members = []
        for member in range(2):
            sid = f"s{block}_{member}"
            signatures[sid] = copy.deepcopy(signature_template)
            signatures[sid].update(signature_id=sid, paper_id=sid)
            signatures[sid]["author_info"]["block"] = f"b{block}"
            papers[sid] = copy.deepcopy(paper_template)
            papers[sid].update(paper_id=sid, year=1980 + block)
            members.append(sid)
        clusters[f"c{block}"] = {"signature_ids": members}
    return ANDData(
        signatures,
        papers,
        "split_authority",
        clusters=clusters,
        name_counts_index=None,
        name_tuples=set(),
        preprocess=False,
    )


class RecordingClusterer:
    """Capture evaluation identities while returning a valid oracle partition."""

    def predict(self, blocks: dict[str, list[str]], dataset: ANDData, **kwargs: Any) -> tuple[dict, None]:
        """Record the request without altering input membership lists."""
        self.blocks = copy.deepcopy(blocks)
        self.supervision = kwargs.get("partial_supervision", {})
        return dataset.construct_cluster_to_signatures(blocks), None


def _members(blocks: dict[str, list[str]]) -> set[str]:
    """Return signature coverage without discarding block order in comparisons."""
    return {signature for signatures in blocks.values() for signature in signatures}


@pytest.mark.parametrize("kind", ["fixed_blocks", "fixed_signatures", "blocks", "signatures", "time"])
@pytest.mark.parametrize("split,index", [("train", 0), ("val", 1), ("test", 2)])
def test_cluster_eval_matches_configured_split(dataset: ANDData, kind: str, split: str, index: int) -> None:
    """Evaluation honors explicit populations and preserves default split order."""
    if kind == "fixed_blocks":
        dataset.train_blocks = [f"b{i}" for i in range(16)]
        dataset.val_blocks = ["b16", "b17"]
        dataset.test_blocks = ["b18", "b19"]
        expected = dataset.split_cluster_signatures_fixed()
    elif kind == "fixed_signatures":
        ids = list(dataset.signatures)
        dataset.train_signatures, dataset.val_signatures, dataset.test_signatures = ids[:20], ids[20:30], ids[30:]
        expected = dataset.split_data_signatures_fixed()
    else:
        dataset.unit_of_data_split = kind
        expected = dataset.split_cluster_signatures()

    before = copy.deepcopy(expected)
    recorder = RecordingClusterer()
    metrics, per_signature = cluster_eval(dataset, cast(Clusterer, recorder), split=split)

    assert recorder.blocks == expected[index]
    assert list(recorder.blocks) == list(expected[index])
    assert set(per_signature) == _members(expected[index])
    assert metrics["B3 (P, R, F1)"] == (1.0, 1.0, 1.0)
    assert _resolve_signature_splits(dataset) == before
    train_pairs, val_pairs, test_pairs = resolve_training_pairs(dataset)
    for pairs, blocks in zip((train_pairs, val_pairs, test_pairs), expected, strict=True):
        assert {sid for left, right, _ in pairs for sid in (left, right)} <= _members(blocks)
    if split == "test":
        assert not _members(recorder.blocks) & _members(expected[0])


@pytest.mark.parametrize("split", ["val", "test"])
def test_incremental_eval_uses_only_selected_context_and_scored_ids(dataset: ANDData, split: str) -> None:
    """Unselected source records cannot join the prediction or metric population."""
    ids = ["s0_0", "s0_1", "s1_0", "s1_1"]
    for sid in ids:
        dataset.signatures[sid] = dataset.signatures[sid]._replace(author_info_block="selected")
    dataset.signature_to_block = dataset.get_signatures_to_block()
    dataset.train_signatures = [ids[0]]
    dataset.val_signatures = [ids[1]]
    dataset.test_signatures = [ids[2]]
    recorder = RecordingClusterer()

    metrics, per_signature = incremental_cluster_eval(dataset, cast(Clusterer, recorder), split=split)

    expected_context = [ids[1], ids[0]] if split == "val" else ids[:3]
    assert recorder.blocks == {"selected": expected_context}
    assert set(per_signature) == {ids[1] if split == "val" else ids[2]}
    assert metrics["B3 (P, R, F1)"] == (1.0, 1.0, 1.0)
    assert ids[3] not in _members(recorder.blocks)
    assert dataset.train_signatures == [ids[0]]
    assert dataset.val_signatures == [ids[1]]
    assert dataset.test_signatures == [ids[2]]


@pytest.mark.parametrize("precomputed", [False, True])
def test_cluster_fit_calibrates_on_fixed_validation_only(dataset: ANDData, monkeypatch, precomputed: bool) -> None:
    """EPS calibration consumes declared validation matrices, including cached ones."""
    dataset.train_blocks = [f"b{i}" for i in range(16)]
    dataset.val_blocks = ["b16", "b17"]
    dataset.test_blocks = ["b18", "b19"]
    expected = {key: dataset.get_blocks()[key] for key in dataset.val_blocks}
    distances = {key: np.array([0.0], dtype=np.float64) for key in expected}
    originals = {key: value.copy() for key, value in distances.items()}
    clusterer = Clusterer(FeaturizationInfo(["year_diff"]), classifier=None, n_jobs=1, n_iter=2)

    def make_distances(blocks, supplied_dataset):
        assert supplied_dataset is dataset
        assert blocks == expected
        assert not precomputed
        return distances

    monkeypatch.setattr(clusterer, "make_distance_matrices", make_distances)
    assert clusterer.fit(dataset, val_dists_precomputed={dataset.name: distances} if precomputed else None) is clusterer
    assert isinstance(clusterer.hyperopt_trials_store, Trials)
    assert clusterer.hyperopt_trials_store.losses() == [-1.0, -1.0]
    for key, value in distances.items():
        np.testing.assert_array_equal(value, originals[key])


@pytest.mark.parametrize("kind", ["blocks", "signatures"])
def test_inferred_validation_matches_training(dataset: ANDData, kind: str) -> None:
    """Missing explicit validation lists use the same seeded split in evaluation."""
    if kind == "blocks":
        dataset.train_blocks = [f"b{i}" for i in range(18)]
        dataset.test_blocks = ["b18", "b19"]
    else:
        dataset.train_signatures = list(dataset.signatures)[:36]
        dataset.test_signatures = list(dataset.signatures)[36:]
    expected = _resolve_signature_splits(dataset)
    recorder = RecordingClusterer()
    cluster_eval(dataset, cast(Clusterer, recorder), split="val")
    assert recorder.blocks == expected[1]
    assert not _members(recorder.blocks) & _members(expected[0])
    assert not _members(recorder.blocks) & _members(expected[2])


@pytest.mark.parametrize("kind", ["blocks", "signatures", "time"])
def test_empty_validation_split_stays_empty(dataset: ANDData, kind: str) -> None:
    """Zero-ratio evaluation uses the existing empty-metric contract."""
    dataset.unit_of_data_split = kind
    dataset.train_ratio, dataset.val_ratio, dataset.test_ratio = 0.8, 0.0, 0.2
    recorder = RecordingClusterer()
    metrics, per_signature = cluster_eval(dataset, cast(Clusterer, recorder), split="val")
    assert recorder.blocks == {}
    assert per_signature == {}
    assert metrics["B3 (P, R, F1)"] == (0.0, 0.0, 0.0)

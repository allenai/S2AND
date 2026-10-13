"""Release featurization on provenance-bound Arrow keeps Python-selected pairs, features and clusters."""

from __future__ import annotations

import argparse
import json
import pickle
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from lightgbm import LGBMClassifier

from s2and import model as model_module
from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_provenance import ArrowProvenanceError, verify_conversion_provenance
from s2and.data import ANDData
from s2and.feature_port import (
    build_rust_featurizer_from_arrow_dataset,
    build_verified_rust_featurizer,
    featurize_pairs_with_rust,
)
from s2and.featurizer import (
    DEFAULT_FEATURE_GROUPS,
    DEFAULT_NAMELESS_FEATURE_GROUPS,
    FeaturizationInfo,
    many_pairs_featurize,
    resolve_selection_pairs,
)
from s2and.model import Clusterer, FastCluster
from s2and.name_counts_index import NameCountsIndex
from s2and.name_tuple_artifact import load_name_tuple_artifact
from s2and.production_training_contract import ProductionArtifactAuthority
from scripts.production.model import release_pairwise, train_pairwise
from tests.pairwise_release_helpers import (
    convert_release_arrow,
    source_sha256,
    write_pairwise_artifacts,
    write_pairwise_dataset,
)

MAIN_INFO = FeaturizationInfo(features_to_use=list(DEFAULT_FEATURE_GROUPS))
NAMELESS_INFO = FeaturizationInfo(features_to_use=list(DEFAULT_NAMELESS_FEATURE_GROUPS))
EPS_GRID = [round(float(eps), 3) for eps in np.linspace(0.05, 0.95, 19)]


@pytest.fixture
def release_inputs(tmp_path: Path) -> dict[str, Any]:
    data_dir = tmp_path / "data"
    files = write_pairwise_dataset(data_dir)
    tuple_path, name_counts_path, _orcid = write_pairwise_artifacts(tmp_path)
    arrow_root = tmp_path / "arrow"
    convert_release_arrow(data_dir, "qian", arrow_root, name_counts_path)
    return {
        "data_dir": data_dir,
        "files": files,
        "arrow_root": arrow_root,
        "name_counts_path": name_counts_path,
        "name_tuples": load_name_tuple_artifact(tuple_path).pairs,
    }


def _json_dataset(inputs: dict[str, Any], *, selection_only: bool = False, **selection: Any) -> ANDData:
    """Load the fixture as the Python featurizer needs it, or as release selection loads it."""

    files = inputs["files"]
    if not selection:
        selection = {"clusters": str(files["clusters"]), "train_pairs_size": 24, "val_pairs_size": 6}
    features: dict[str, Any] = {
        "specter_embeddings": str(files["specter_embeddings"]),
        "name_counts_index": NameCountsIndex.open(inputs["name_counts_path"]),
        "preprocess": True,
    }
    if selection_only:
        features = {"name_counts_index": None, "preprocess": False}
    return ANDData(
        signatures=str(files["signatures"]),
        papers=str(files["papers"]),
        name="qian",
        mode="train",
        random_seed=1111,
        n_jobs=1,
        name_tuples=inputs["name_tuples"],
        **selection,
        **features,
    )


def _pair_rows(pairs: list[Any]) -> list[tuple[str, str, float]]:
    return [(str(pair[0]), str(pair[1]), float(pair[2])) for pair in pairs]


def _selections(inputs: dict[str, Any], **selection: Any) -> tuple[ANDData, ANDData, list[Any], list[Any]]:
    """Return (reference, selection-only) datasets and their identical train and validation pairs."""

    reference = _json_dataset(inputs, **selection)
    selection_only = _json_dataset(inputs, selection_only=True, **selection)
    train_pairs, val_pairs = resolve_selection_pairs(selection_only)
    expected_train, expected_val = resolve_selection_pairs(reference)
    assert _pair_rows(train_pairs) == _pair_rows(expected_train)
    assert _pair_rows(val_pairs) == _pair_rows(expected_val)
    return reference, selection_only, train_pairs, val_pairs


def _featurizer(
    inputs: dict[str, Any],
    dataset: ANDData,
    arrow_dataset: ArrowDataset,
    signature_ids: set[str],
) -> Any:
    return build_verified_rust_featurizer(
        dataset,
        arrow_dataset,
        signature_ids=signature_ids,
        name_counts_manifest_sha256=NameCountsIndex.open(inputs["name_counts_path"]).manifest_sha256,
        num_threads=2,
    )


def _pair_members(*pair_lists: list[Any]) -> set[str]:
    return {str(pair[index]) for pairs in pair_lists for pair in pairs for index in (0, 1)}


def _block_members(blocks: dict[str, list[str]]) -> set[str]:
    return {signature for signatures in blocks.values() for signature in signatures}


def _open(inputs: dict[str, Any]) -> ArrowDataset:
    return ArrowDataset.open(inputs["arrow_root"] / "qian", require_specter=True, require_name_counts_index=True)


def _python_features(pairs: list[Any], dataset: ANDData, nan_value: float = np.nan) -> tuple[Any, ...]:
    return many_pairs_featurize(
        pairs,
        dataset,
        MAIN_INFO,
        n_jobs=1,
        nameless_featurizer_info=NAMELESS_INFO,
        nan_value=nan_value,
    )


def _rust_features(pairs: list[Any], featurizer: Any, nan_value: float = np.nan) -> tuple[Any, ...]:
    return featurize_pairs_with_rust(
        featurizer,
        pairs,
        MAIN_INFO,
        nameless_featurizer_info=NAMELESS_INFO,
        n_jobs=2,
        nan_value=nan_value,
        total_ram_bytes=1_000_000_000,
    )


def _assert_features_match(python: tuple[Any, ...], rust: tuple[Any, ...]) -> None:
    for expected, observed in ((python[0], rust[0]), (python[2], rust[2])):
        assert observed.shape == expected.shape
        assert np.array_equal(np.isnan(observed), np.isnan(expected))
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(rust[1], python[1])


def _canonical_partition(clusters: dict[str, list[str]]) -> list[list[str]]:
    return sorted(sorted(str(member) for member in members) for members in clusters.values())


def test_fixed_pair_datasets_featurize_like_python(release_inputs: dict[str, Any], tmp_path: Path) -> None:
    """Fixed train/val pair files (as for augmented) resolve to numpy records, not tuples."""

    signature_ids = sorted(json.loads(release_inputs["files"]["signatures"].read_text(encoding="utf-8")))
    split_files = {}
    for split, offset in (("train", 0), ("val", 20)):
        rows = ["signature_id_1,signature_id_2,label"] + [
            f"{signature_ids[offset + index]},{signature_ids[offset + index + 1]},{index % 2}"
            for index in range(0, 18, 2)
        ]
        path = tmp_path / f"{split}_pairs.csv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        split_files[split] = str(path)
    reference, dataset, train_pairs, val_pairs = _selections(
        release_inputs, clusters=None, train_pairs=split_files["train"], val_pairs=split_files["val"]
    )
    assert isinstance(train_pairs[0], np.record)

    with _open(release_inputs) as arrow_dataset:
        train, val = train_pairwise._featurize_selection(
            dataset,
            arrow_dataset,
            MAIN_INFO,
            name_counts_manifest_sha256=reference.name_counts_manifest_sha256,
            n_jobs=1,
            nameless_featurizer_info=NAMELESS_INFO,
            total_ram_bytes=None,
        )
    _assert_features_match(_python_features(train_pairs, reference), train)
    _assert_features_match(_python_features(val_pairs, reference), val)


def test_rust_features_match_python_on_the_selected_training_pairs(release_inputs: dict[str, Any]) -> None:
    reference, dataset, train_pairs, val_pairs = _selections(release_inputs)
    assert len(train_pairs) == 24 and len(val_pairs) == 6

    with _open(release_inputs) as arrow_dataset:
        featurizer = _featurizer(release_inputs, dataset, arrow_dataset, _pair_members(train_pairs, val_pairs))
        assert len(featurizer.signature_ids()) < len(dataset.signatures)
        for pairs in (train_pairs, val_pairs):
            _assert_features_match(_python_features(pairs, reference), _rust_features(pairs, featurizer))
        # A non-NaN nan_value is applied after featurization, as many_pairs_featurize does.
        replaced = _rust_features(train_pairs, featurizer, nan_value=-1.0)
        _assert_features_match(_python_features(train_pairs, reference, nan_value=-1.0), replaced)
        assert not np.isnan(replaced[0]).any()


@pytest.mark.parametrize("unknown_scope", ["partial", "all"])
def test_unknowns_leave_supervised_pairs_but_retain_dataset_and_features(
    release_inputs: dict[str, Any],
    unknown_scope: str,
) -> None:
    reference, dataset, train_pairs, val_pairs = _selections(release_inputs)
    unknown = frozenset({str(train_pairs[0][0]), str(val_pairs[0][1])})
    if unknown_scope == "all":
        unknown = frozenset(dataset.signatures)
    expected = [
        [pair for pair in pairs if str(pair[0]) not in unknown and str(pair[1]) not in unknown]
        for pairs in (train_pairs, val_pairs)
    ]
    original_blocks = dataset.get_blocks()
    assert reference.name_counts_manifest_sha256 is not None
    with _open(release_inputs) as arrow_dataset:
        observed = train_pairwise._featurize_selection(
            dataset,
            arrow_dataset,
            MAIN_INFO,
            name_counts_manifest_sha256=reference.name_counts_manifest_sha256,
            n_jobs=1,
            nameless_featurizer_info=NAMELESS_INFO,
            total_ram_bytes=None,
            unknown_signature_ids=unknown,
        )
    for pairs, arrays in zip(expected, observed, strict=True):
        _assert_features_match(_python_features(pairs, reference), arrays)
    assert dataset.get_blocks() == original_blocks
    assert unknown <= set(dataset.signatures)


def test_unknown_signature_retains_its_effect_on_cluster_predictions(release_inputs: dict[str, Any]) -> None:
    dataset = _json_dataset(release_inputs, selection_only=True)
    first, unknown, second = next(iter(dataset.get_blocks().values()))[:3]
    classifier = LGBMClassifier(n_estimators=1, n_jobs=1, verbosity=-1).fit(
        np.zeros((2, len(MAIN_INFO.selected_feature_indices()))),
        [0, 1],
    )
    clusterer = Clusterer(
        MAIN_INFO,
        classifier,
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
        use_default_constraints_as_supervision=False,
    )
    predictions, _ = clusterer.predict(
        {"block": [first, unknown, second]},
        dataset,
        dists={"block": np.array([0.1, 0.8, 0.1])},
    )
    assert _canonical_partition(predictions) == [sorted([first, unknown, second])]
    score = release_pairwise._b3_report(
        {"a": [first, unknown], "b": [second]},
        predictions,
        frozenset({unknown}),
    )
    assert score == {"precision": 0.5, "recall": 1.0, "f1": 0.667, "signature_count": 2}
    without_unknown, _ = clusterer.predict(
        {"block": [first, second]},
        dataset,
        dists={"block": np.array([0.8])},
    )
    assert len(without_unknown) == 2


def test_rust_distances_give_python_partitions_at_every_eps(release_inputs: dict[str, Any]) -> None:
    """Release clustering (selection-only dataset, Rust distances) partitions like Python fused predict."""

    reference, dataset, train_pairs, _ = _selections(release_inputs)
    features, labels, nameless_features = _python_features(train_pairs, reference)
    classifiers = [
        LGBMClassifier(n_estimators=25, num_leaves=5, min_child_samples=2, random_state=0, n_jobs=1, verbose=-1).fit(
            matrix, labels
        )
        for matrix in (features, nameless_features)
    ]
    clusterer = Clusterer(
        MAIN_INFO,
        classifiers[0],
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
        nameless_classifier=classifiers[1],
        nameless_featurizer_info=NAMELESS_INFO,
        use_default_constraints_as_supervision=True,
        dont_merge_cluster_seeds=True,
    )
    clusterer.feature_contract = {
        "name_counts_manifest_sha256": reference.name_counts_manifest_sha256,
        "name_counts_last_first_initial_semantics": "initial_char",
    }
    blocks = clusterer.filter_blocks(dataset.get_blocks())
    assert blocks == clusterer.filter_blocks(reference.get_blocks())
    assert dataset.split_cluster_signatures() == reference.split_cluster_signatures()

    python_distances = clusterer.make_distance_matrices(blocks, reference)
    with _open(release_inputs) as arrow_dataset:
        seeded = _featurizer(release_inputs, dataset, arrow_dataset, _block_members(blocks)).with_cluster_seeds(
            dataset.cluster_seeds_require, dataset.cluster_seeds_disallow
        )
        rust_distances = clusterer.make_distance_matrices_from_rust_featurizer(blocks, seeded)
    for block_key in blocks:
        np.testing.assert_allclose(rust_distances[block_key], python_distances[block_key], rtol=0.0, atol=1e-12)

    for eps in EPS_GRID:
        clusterer.cluster_model.eps = eps
        fused, _ = clusterer.predict(blocks, reference)
        from_python, _ = clusterer.predict(blocks, reference, dists=python_distances)
        from_rust, _ = clusterer.predict(blocks, dataset, dists=rust_distances)
        assert _canonical_partition(from_rust) == _canonical_partition(from_python) == _canonical_partition(fused), eps


def test_subset_featurizer_matches_a_full_build_bit_for_bit(release_inputs: dict[str, Any]) -> None:
    dataset = _json_dataset(release_inputs)
    blocks = dict(sorted(dataset.get_blocks().items()))
    first_key, second_key = list(blocks)[:2]
    block = {first_key: blocks[first_key]}
    inside, outside = blocks[first_key], blocks[second_key]
    # Seeds that also name signatures the subset featurizer never builds.
    require = {inside[0]: 0, inside[1]: 0, outside[0]: 0, outside[1]: 1}
    disallow = {(inside[2], inside[3]), (inside[0], outside[2])}
    pairs = [(inside[0], inside[1], 1), (inside[0], inside[2], 0), (inside[2], inside[3], 0)]
    clusterer = Clusterer(
        MAIN_INFO,
        LGBMClassifier(n_estimators=5, num_leaves=3, min_child_samples=1, random_state=0, verbose=-1).fit(
            np.nan_to_num(_python_features(pairs, dataset)[0]), [1, 0, 0]
        ),
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
        use_default_constraints_as_supervision=True,
        dont_merge_cluster_seeds=True,
    )
    clusterer.feature_contract = {
        "name_counts_manifest_sha256": dataset.name_counts_manifest_sha256,
        "name_counts_last_first_initial_semantics": "initial_char",
    }

    with _open(release_inputs) as arrow_dataset:
        full = build_rust_featurizer_from_arrow_dataset(arrow_dataset, name_tuples=dataset.name_tuples, num_threads=1)
        subset = _featurizer(release_inputs, dataset, arrow_dataset, set(inside))
        assert len(subset.signature_ids()) == len(inside) < len(full.signature_ids())
        for expected, observed in zip(_rust_features(pairs, full), _rust_features(pairs, subset), strict=True):
            np.testing.assert_array_equal(np.asarray(observed).view(np.uint64), np.asarray(expected).view(np.uint64))
        distances = [
            clusterer.make_distance_matrices_from_rust_featurizer(
                block, featurizer.with_cluster_seeds(require, disallow)
            )[first_key]
            for featurizer in (full, subset)
        ]
    np.testing.assert_array_equal(distances[1].view(np.uint64), distances[0].view(np.uint64))


@pytest.mark.parametrize("constraints_enabled", [True, False])
@pytest.mark.parametrize("dont_merge_seeds", [True, False])
def test_cluster_release_reuses_index_and_preserves_seed_policy_and_block_order(
    release_inputs: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constraints_enabled: bool,
    dont_merge_seeds: bool,
) -> None:
    """The actual release route matches its old loop and frees each consumed block matrix."""

    reference, dataset, train_pairs, _ = _selections(release_inputs)
    features, labels, nameless = _python_features(train_pairs, reference)
    classifiers = [
        LGBMClassifier(n_estimators=25, num_leaves=3, min_child_samples=1, n_jobs=1, verbose=-1, random_state=0).fit(
            matrix, labels
        )
        for matrix in (features, nameless)
    ]
    clusterer = Clusterer(
        MAIN_INFO,
        classifiers[0],
        cluster_model=FastCluster(linkage="average", eps=0.5),
        n_jobs=1,
        nameless_classifier=classifiers[1],
        nameless_featurizer_info=NAMELESS_INFO,
        use_default_constraints_as_supervision=constraints_enabled,
        dont_merge_cluster_seeds=dont_merge_seeds,
    )
    assert reference.name_counts_manifest_sha256 is not None
    clusterer.feature_contract = {
        "name_counts_manifest_sha256": reference.name_counts_manifest_sha256,
        "name_counts_last_first_initial_semantics": "initial_char",
    }
    all_blocks = dataset.get_blocks()
    first, second, singleton, outside = list(all_blocks)[:4]
    blocks = {second: all_blocks[second], first: all_blocks[first], singleton: all_blocks[singleton][:1]}
    left, right = blocks[first], blocks[second]
    dataset.cluster_seeds_require = {
        left[0]: 0,
        left[1]: 0,
        left[2]: 1,
        right[0]: 0,
        right[1]: 1,
        right[2]: 0,
        all_blocks[outside][0]: 0,
    }
    dataset.cluster_seeds_disallow = {(left[0], left[1]), (right[0], right[2])}
    expected: dict[str, list[str]] = {}
    expected_distances: dict[str, np.ndarray] = {}
    with _open(release_inputs) as arrow_dataset:
        seeded = _featurizer(release_inputs, dataset, arrow_dataset, _block_members(blocks)).with_cluster_seeds(
            dataset.cluster_seeds_require, dataset.cluster_seeds_disallow
        )
        for key, members in blocks.items():
            block = {key: members}
            distances = clusterer.make_distance_matrices_from_rust_featurizer(block, seeded)
            expected_distances[key] = distances[key].copy()
            predicted, _ = clusterer.predict(block, dataset, dists=distances)
            expected.update(predicted)
    # The conflicting require/disallow pair exercises enabled and disabled seed behavior.
    assert any(left[0] in members and left[1] in members for members in expected.values()) is not constraints_enabled

    blocks_path = tmp_path / "blocks.json"
    blocks_path.write_text(json.dumps(blocks), encoding="utf-8")
    specs = {
        role: (path, release_pairwise._sha256_file(path))
        for role, path in {**release_inputs["files"], "blocks": blocks_path}.items()
    }
    monkeypatch.setattr(release_pairwise, "load_run_binding", lambda _path: {"run_binding_sha256": "a" * 64})
    monkeypatch.setattr(release_pairwise, "require_run_binding_matches", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release_pairwise, "_evaluation_model", lambda _args: (_authority(release_inputs), clusterer))
    monkeypatch.setattr(
        release_pairwise, "_load_evaluation_plan", lambda _path: SimpleNamespace(cluster={"qian": specs})
    )
    monkeypatch.setattr(release_pairwise, "_load_anddata", lambda *_args, **_kwargs: dataset)
    observed: list[dict[str, list[str]]] = []
    original_report = release_pairwise._b3_report

    def report(
        truth: Any,
        predicted: dict[str, list[str]],
        unknown_signature_ids: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        observed.append(predicted)
        return original_report(truth, predicted, unknown_signature_ids)

    monkeypatch.setattr(release_pairwise, "_b3_report", report)
    index_builds = 0
    original_index = model_module._build_signature_index_by_id

    def index(featurizer: Any) -> dict[str, int]:
        nonlocal index_builds
        index_builds += 1
        return original_index(featurizer)

    monkeypatch.setattr(model_module, "_build_signature_index_by_id", index)
    previous_matrices: list[weakref.ReferenceType[np.ndarray]] = []
    matrix_blocks: list[str] = []
    original_matrices = Clusterer._make_distance_matrices_from_verified_rust_featurizer

    def matrices(self: Clusterer, block: dict[str, list[str]], *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        assert all(matrix() is None for matrix in previous_matrices)
        result = original_matrices(self, block, *args, **kwargs)
        for key, matrix in result.items():
            np.testing.assert_array_equal(matrix.view(np.uint64), expected_distances[key].view(np.uint64))
        previous_matrices[:] = [weakref.ref(matrix) for matrix in result.values()]
        matrix_blocks.extend(block)
        return result

    monkeypatch.setattr(Clusterer, "_make_distance_matrices_from_verified_rust_featurizer", matrices)
    release_pairwise.evaluate_clusters(
        argparse.Namespace(
            output_report=tmp_path / "report.json",
            run_binding=tmp_path / "binding.json",
            model=tmp_path / "model",
            evaluation_plan=tmp_path / "plan.json",
            arrow_root=release_inputs["arrow_root"],
            n_jobs=1,
        )
    )

    assert observed == [expected]
    assert list(observed[0].items()) == list(expected.items())
    assert index_builds == 1
    assert matrix_blocks == [second, first]
    assert all(matrix() is None for matrix in previous_matrices)


def _changed_feature_source(files: dict[str, Path], role: str) -> None:
    """Change one real feature input while keeping every signature and paper ID."""
    if role == "specter_embeddings":
        with files[role].open("rb") as source:
            embeddings = pickle.load(source)
        embeddings["paper-0-0"][0] = 0.5
        with files[role].open("wb") as output:
            pickle.dump(embeddings, output)
    else:
        payload = json.loads(files[role].read_text(encoding="utf-8"))
        if role == "signatures":
            payload["signature-0-0"]["author_info"]["first"] = "Alexandra"
        else:
            payload["paper-0-0"]["title"] = "Changed title"
        files[role].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _authority(inputs: dict[str, Any]) -> ProductionArtifactAuthority:
    return ProductionArtifactAuthority(
        name_counts_index=NameCountsIndex.open(inputs["name_counts_path"]),
        name_tuples=SimpleNamespace(pairs=inputs["name_tuples"]),
        orcid_prefix_counts=SimpleNamespace(),
    )


@pytest.mark.parametrize("role", ["signatures", "papers", "specter_embeddings"])
def test_release_rejects_stale_arrow_through_provenance_not_the_plan_file_check(
    release_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    files = release_inputs["files"]
    _changed_feature_source(files, role)
    # The plan is updated to the changed JSON, so the existing plan-file digest check passes;
    # only the stale Arrow directory and its stale provenance record remain.
    updated_plan_digests = source_sha256(files)
    dataset = _json_dataset(release_inputs)
    real_open = ArrowDataset.open
    monkeypatch.setattr(
        ArrowDataset,
        "open",
        staticmethod(lambda *_args, **_kwargs: pytest.fail("stale Arrow data must be rejected before opening")),
    )

    with pytest.raises(ArrowProvenanceError, match=f"qian was converted from a different {role} source"):
        with release_pairwise._release_rust_featurizer(
            release_inputs["arrow_root"],
            "qian",
            source_sha256=updated_plan_digests,
            dataset=dataset,
            signature_ids=set(dataset.signatures),
            artifacts=_authority(release_inputs),
            n_jobs=1,
        ):
            pass

    # Control: regenerating Arrow data and provenance from the changed JSON is accepted.
    monkeypatch.setattr(ArrowDataset, "open", real_open)
    convert_release_arrow(
        release_inputs["data_dir"], "qian", release_inputs["arrow_root"], release_inputs["name_counts_path"]
    )
    with release_pairwise._release_rust_featurizer(
        release_inputs["arrow_root"],
        "qian",
        source_sha256=updated_plan_digests,
        dataset=dataset,
        signature_ids=set(dataset.signatures),
        artifacts=_authority(release_inputs),
        n_jobs=1,
    ) as featurizer:
        assert {str(signature_id) for signature_id in featurizer.signature_ids()} == set(dataset.signatures)


def test_provenance_rejects_tables_that_differ_from_the_recorded_conversion(release_inputs: dict[str, Any]) -> None:
    dataset_dir = release_inputs["arrow_root"] / "qian"
    expected = source_sha256(release_inputs["files"])
    verify_conversion_provenance(dataset_dir, expected_source_sha256=expected)

    record_path = dataset_dir / "conversion_provenance.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["arrow_files"]["papers"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ArrowProvenanceError, match="tables differ from the conversion"):
        verify_conversion_provenance(dataset_dir, expected_source_sha256=expected)

    record_path.unlink()
    with pytest.raises(ArrowProvenanceError, match="has no conversion provenance record"):
        verify_conversion_provenance(dataset_dir, expected_source_sha256=expected)


def test_verified_featurizer_rejects_mismatched_counts_and_signatures(release_inputs: dict[str, Any]) -> None:
    dataset = _json_dataset(release_inputs, selection_only=True)
    with _open(release_inputs) as arrow_dataset:
        members = set(list(dataset.signatures)[:4])
        with pytest.raises(ValueError, match="'qian' uses the wrong name-count index"):
            build_verified_rust_featurizer(
                dataset, arrow_dataset, signature_ids=members, name_counts_manifest_sha256="0" * 64, num_threads=1
            )
        with pytest.raises(ValueError, match="Requested signatures are not in 'qian'"):
            _featurizer(release_inputs, dataset, arrow_dataset, members | {"unknown-signature"})
        extra = next(iter(dataset.signatures.values()))
        dataset.signatures["not-in-arrow"] = extra._replace(signature_id="not-in-arrow")
        # The full-set check still runs when only a subset is built.
        with pytest.raises(ValueError, match="Arrow signatures for 'qian' differ from its JSON signatures"):
            _featurizer(release_inputs, dataset, arrow_dataset, members)


def test_rust_pair_featurization_handles_empty_input_and_rejects_constraint_labels(
    release_inputs: dict[str, Any],
) -> None:
    dataset = _json_dataset(release_inputs)
    with _open(release_inputs) as arrow_dataset:
        featurizer = _featurizer(release_inputs, dataset, arrow_dataset, set(list(dataset.signatures)[:2]))
        features, labels, nameless = _rust_features([], featurizer)
        assert features.shape == (0, len(MAIN_INFO.selected_feature_indices()))
        assert labels.shape == (0,)
        assert nameless is not None and nameless.shape == (0, len(NAMELESS_INFO.selected_feature_indices()))
        with pytest.raises(ValueError, match="partial-supervision"):
            _rust_features([("signature-0-0", "signature-0-1", -1)], featurizer)

"""Real native coverage for caller-owned Arrow datasets without manifests."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, NoReturn

import numpy as np
import pyarrow as pa
import pytest

import s2and.arrow_inputs as arrow_inputs
from s2and.arrow_inputs import ArrowDataset, MissingArrowArtifactError
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.incremental_linking.feature_block import write_name_counts_index
from s2and.runtime import load_s2and_rust_extension
from tests.helpers import (
    tiny_name_counts_tuple,
    write_minimal_arrow_prediction_bundle,
)


def _features(dataset: ArrowDataset) -> np.ndarray:
    featurizer = build_rust_featurizer_from_arrow_dataset(dataset, name_tuples=frozenset(), num_threads=1)
    return np.asarray(featurizer.featurize_pairs_matrix_indexed([(0, 1)], None, 1, np.nan))


@pytest.mark.parametrize("include_specter", [False, True])
def test_trusted_features_match_verified_dataset_without_manifest(tmp_path: Path, include_specter: bool) -> None:
    paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=include_specter)
    with ArrowDataset.open(tmp_path, require_specter=include_specter) as verified:
        expected = _features(verified)
    (tmp_path / "manifest.json").unlink()

    with ArrowDataset.open_trusted(paths, require_specter=include_specter) as dataset:
        assert dataset.root == tmp_path.resolve()
        assert dataset.has("specter") == include_specter
        np.testing.assert_array_equal(_features(dataset), expected)
        with dataset.use() as lease, lease.open_file("signatures") as source:
            assert pa.ipc.open_file(source).read_all().num_rows == 13
            with pytest.raises(RuntimeError, match="active use"):
                dataset.close()
    assert dataset.closed
    with pytest.raises(RuntimeError, match="closed"):
        _features(dataset)


@pytest.mark.parametrize(
    ("key", "argument"),
    [
        ("cluster_seeds", "cluster_seeds_require"),
        ("cluster_seed_disallows", "cluster_seeds_disallow"),
        ("altered_cluster_signatures", "altered_cluster_signatures"),
    ],
)
def test_trusted_open_rejects_request_local_tables(tmp_path: Path, key: str, argument: str) -> None:
    # A caller ported from 0.51 passes seed tables as paths; ignoring them would drop exclusions silently.
    paths = write_minimal_arrow_prediction_bundle(tmp_path)
    paths[key] = str(tmp_path / f"{key}.arrow")
    with pytest.raises(ValueError, match=f"for '{key}' pass {argument} "):
        ArrowDataset.open_trusted(paths)


def test_trusted_open_skips_hashing_and_isolates_generations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_minimal_arrow_prediction_bundle(tmp_path)
    (tmp_path / "manifest.json").unlink()

    def unexpected_hash(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("trusted Arrow open must not hash runtime table/index contents")

    monkeypatch.setattr(arrow_inputs, "_hash_retained_file", unexpected_hash)
    monkeypatch.setattr(arrow_inputs, "_sha256_file", unexpected_hash)
    native = load_s2and_rust_extension()
    monkeypatch.setattr(native, "_arrow_retained_file_sha256", unexpected_hash)
    with ArrowDataset.open_trusted(paths) as first, ArrowDataset.open_trusted(paths) as second:
        assert first.generation_id != second.generation_id
        np.testing.assert_array_equal(_features(first), _features(second))


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("magic", "magic"),
        ("old_format", "magic"),
        ("source_size", "indexed size"),
        ("key_column", "key column"),
        ("length", "length"),
        ("batch_bounds", "references record batch"),
    ],
)
def test_trusted_open_still_rejects_malformed_indexes(tmp_path: Path, corruption: str, message: str) -> None:
    paths = write_minimal_arrow_prediction_bundle(tmp_path)
    index_path = Path(paths["signatures_batch_index"])
    data = bytearray(index_path.read_bytes())
    if corruption == "magic":
        data[:8] = b"BADMAGIC"
    elif corruption == "old_format":
        data[:8] = b"S2ABI002"
    elif corruption == "source_size":
        struct.pack_into("<Q", data, 16, Path(paths["signatures"]).stat().st_size + 1)
    elif corruption == "key_column":
        struct.pack_into("<Q", data, 24, 0)
    elif corruption == "length":
        data.pop()
    else:
        struct.pack_into("<I", data, 40, 1_000_000)
    index_path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        ArrowDataset.open_trusted(paths)


def test_trusted_open_requires_complete_tables_indexes_and_requested_profiles(tmp_path: Path) -> None:
    paths = write_minimal_arrow_prediction_bundle(tmp_path, include_specter=True)
    with pytest.raises(MissingArrowArtifactError, match="papers_batch_index"):
        ArrowDataset.open_trusted({key: value for key, value in paths.items() if key != "papers_batch_index"})
    with pytest.raises(ValueError, match="specter.*specter_batch_index"):
        ArrowDataset.open_trusted({key: value for key, value in paths.items() if key != "specter_batch_index"})
    without_specter = {key: value for key, value in paths.items() if not key.startswith("specter")}
    with pytest.raises(MissingArrowArtifactError, match="specter"):
        ArrowDataset.open_trusted(without_specter, require_specter=True)
    with pytest.raises(MissingArrowArtifactError, match="name_counts_index"):
        ArrowDataset.open_trusted(paths, require_name_counts_index=True)


def test_trusted_bad_schema_closes_every_opened_handle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = write_minimal_arrow_prediction_bundle(tmp_path)
    table = pa.table({"signature_id": pa.array(["bad"], type=pa.string())})
    with pa.OSFile(paths["signatures"], "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    original_open = arrow_inputs._open_retained_path  # noqa: SLF001
    opened: list[BinaryIO] = []

    def track_open(path: Path) -> BinaryIO:
        source = original_open(path)
        opened.append(source)
        return source

    monkeypatch.setattr(arrow_inputs, "_open_retained_path", track_open)
    with pytest.raises(ValueError, match="Invalid Arrow signatures schema"):
        ArrowDataset.open_trusted(paths)
    assert len(opened) == len(paths)
    assert all(source.closed for source in opened)


def test_trusted_open_preserves_independent_name_counts_validation(tmp_path: Path) -> None:
    paths = write_minimal_arrow_prediction_bundle(tmp_path)
    name_counts_path, _ = write_name_counts_index(tmp_path / "counts", tiny_name_counts_tuple())
    paths["name_counts_index"] = name_counts_path
    (tmp_path / "manifest.json").unlink()
    with ArrowDataset.open_trusted(paths, require_name_counts_index=True) as dataset:
        assert dataset.has("name_counts_index")
        assert dataset.name_counts_index is not None
        assert dataset.native_name_counts_index is not None
    # Use a fresh path: previously opened immutable name-count handles are cached.
    invalid_path, _ = write_name_counts_index(tmp_path / "invalid-counts", tiny_name_counts_tuple())
    (Path(invalid_path) / "manifest.json").write_text("invalid-json", encoding="utf-8")
    paths["name_counts_index"] = invalid_path
    with pytest.raises(ValueError):
        ArrowDataset.open_trusted(paths, require_name_counts_index=True)


def test_featurizer_view_matches_a_fresh_build_with_seed_sidecars(tmp_path: Path) -> None:
    # Subblocked attachment reuses one block featurizer through views, so a view must
    # featurize and constrain exactly like a fresh build over the same signatures.
    from s2and.incremental_linking.feature_block_arrow import temporary_cluster_seed_sidecars

    paths = write_minimal_arrow_prediction_bundle(tmp_path)
    all_ids = [str(index) for index in range(10)] + ["q1", "q2", "seed1"]
    subset = ["q2", "3", "seed1", "7", "0", "q1", "5"]
    pairs = [(left, right) for left in range(len(subset)) for right in range(left + 1, len(subset))]
    seeds = {"seed1": "profile", "3": "profile", "7": "other", "9": "outside"}
    disallows = [("q1", "3"), ("0", "5"), ("8", "9")]
    with (
        ArrowDataset.open_trusted(paths) as dataset,
        temporary_cluster_seed_sidecars(seeds, prefix="view_", cluster_seeds_disallow=disallows) as sidecars,
    ):
        fresh = build_rust_featurizer_from_arrow_dataset(
            dataset,
            signature_ids=subset,
            name_tuples=frozenset(),
            num_threads=1,
            cluster_seeds_path=sidecars["cluster_seeds"],
            cluster_seed_disallows_path=sidecars["cluster_seed_disallows"],
        )
        block = build_rust_featurizer_from_arrow_dataset(
            dataset, signature_ids=all_ids, name_tuples=frozenset(), num_threads=1
        )
        view = block.view_with_cluster_seed_sidecars(
            subset, sidecars["cluster_seeds"], sidecars["cluster_seed_disallows"]
        )

        def features(featurizer):
            return np.asarray(featurizer.featurize_pairs_matrix_indexed(pairs, None, 1, np.nan))

        def constraints(featurizer):
            return featurizer.get_constraints_matrix_indexed(pairs, 0.0, 10000.0, False, False, 1, False)

        np.testing.assert_array_equal(features(view), features(fresh))
        assert constraints(view) == constraints(fresh)
        assert view.cluster_seeds_require() == fresh.cluster_seeds_require()
        with pytest.raises(KeyError, match="missing requested signature ids"):
            block.view_with_cluster_seed_sidecars(["missing"], None, None)


def _same_plan(left: object, right: object) -> bool:
    # Telemetry holds timings and read counts, which differ between a build and a reload.
    if isinstance(left, dict) and "telemetry" in left and isinstance(right, dict):
        left = {key: value for key, value in left.items() if key != "telemetry"}
        right = {key: value for key, value in right.items() if key != "telemetry"}
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_plan(left[key], right[key]) for key in left)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_plan(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def test_planner_reload_matches_a_fresh_build_for_a_seed_superset(tmp_path: Path) -> None:
    # Subblocked attachment reloads one planner across passes, so a reload must plan
    # exactly like a fresh build over the same sidecars, and refuse anything else.
    from s2and.feature_port import _require_rust_runtime
    from s2and.incremental_linking.feature_block import (
        write_cluster_seed_disallows_arrow,
        write_cluster_seeds_arrow,
    )

    paths = write_minimal_arrow_prediction_bundle(tmp_path / "data", include_specter=True)
    first_seeds = {"0": "a", "1": "a", "2": "b"}
    later_seeds = {**first_seeds, "3": "b", "4": "c", "seed1": "c", "5": "a"}
    sidecars = {}
    for name, seeds in (("first", first_seeds), ("later", later_seeds), ("subset", {"0": "a"})):
        sidecars[name] = tmp_path / f"{name}_seeds.arrow"
        write_cluster_seeds_arrow(sidecars[name], seeds)
    disallows = tmp_path / "disallows.arrow"
    write_cluster_seed_disallows_arrow(disallows, [("q1", "4")])
    planner_class = _require_rust_runtime().RawBlockQueryCandidatePlanner
    options = {"top_k": 3, "orcid_enabled": True, "num_threads": 1, "max_exemplars": 4}
    queries = ["q1", "q2", "6"]
    with ArrowDataset.open_trusted(paths, require_specter=True) as dataset:
        fresh = planner_class.from_auto_queries(
            dataset.native, str(sidecars["later"]), cluster_seed_disallows_path=str(disallows), **options
        )
        expected = fresh.plan(queries)
        reused = planner_class.from_auto_queries(
            dataset.native, str(sidecars["first"]), retain_seed_rows=True, **options
        )
        reused.plan(queries)
        assert reused.reload_seed_sidecars(str(sidecars["later"]), str(disallows)) is True
        assert _same_plan(reused.plan(queries), expected)

        # Dropping a current seed cannot reuse rows; the planner stays as it was.
        assert reused.reload_seed_sidecars(str(sidecars["subset"])) is False
        assert _same_plan(reused.plan(queries), expected)

        # Without retained rows a planner never reloads.
        assert fresh.reload_seed_sidecars(str(sidecars["later"]), str(disallows)) is False

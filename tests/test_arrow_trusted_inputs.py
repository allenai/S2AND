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
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset, build_rust_featurizer_from_planner
from s2and.incremental_linking.feature_block import write_name_counts_index
from s2and.runtime import load_s2and_rust_extension
from tests.helpers import (
    tiny_name_counts_tuple,
    write_arrow_tables_bundle,
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


def test_planner_reload_that_fails_keeps_the_rows_for_a_retry(tmp_path: Path) -> None:
    # A seed without a signature row fails the reload. The planner must keep its
    # rows, so the next reload and plan work as before.
    from s2and.feature_port import _require_rust_runtime
    from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow

    paths = write_minimal_arrow_prediction_bundle(tmp_path / "data", include_specter=True)
    initial = tmp_path / "initial_seeds.arrow"
    bad = tmp_path / "bad_seeds.arrow"
    write_cluster_seeds_arrow(initial, {"0": "a", "1": "b"})
    write_cluster_seeds_arrow(bad, {"0": "a", "1": "b", "missing": "c"})
    planner_class = _require_rust_runtime().RawBlockQueryCandidatePlanner
    with ArrowDataset.open_trusted(paths, require_specter=True) as dataset:
        planner = planner_class.from_auto_queries(
            dataset.native, str(initial), top_k=2, retain_seed_rows=True, num_threads=1
        )
        expected = planner.plan(["q1"])
        with pytest.raises(KeyError, match="missing"):
            planner.reload_seed_sidecars(str(bad))
        assert planner.build_telemetry()["signature_count"] == 2
        assert planner.reload_seed_sidecars(str(initial)) is True
        assert _same_plan(planner.plan(["q1"]), expected)


def test_released_planner_and_featurizer_refuse_further_work(tmp_path: Path) -> None:
    # A request releases its native handles on exit; they must not serve work afterwards,
    # and a released view must not touch the featurizer it shares data with.
    from s2and.feature_port import _require_rust_runtime, build_rust_featurizer_from_arrow_dataset
    from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow

    paths = write_minimal_arrow_prediction_bundle(tmp_path / "data", include_specter=True)
    seeds = tmp_path / "seeds.arrow"
    write_cluster_seeds_arrow(seeds, {"0": "a", "1": "b"})
    planner_class = _require_rust_runtime().RawBlockQueryCandidatePlanner
    with ArrowDataset.open_trusted(paths, require_specter=True) as dataset:
        planner = planner_class.from_auto_queries(dataset.native, str(seeds), top_k=2, num_threads=1)
        planner.release()
        with pytest.raises(RuntimeError, match="released"):
            planner.plan(["q1"])
        featurizer = build_rust_featurizer_from_arrow_dataset(
            dataset, signature_ids=["0", "1", "q1"], name_tuples=None, preprocess=True, num_threads=1
        )
        view = featurizer.view_with_cluster_seed_sidecars(["0"], None, None)
        view.release()
        assert view.signature_ids() == []
        assert featurizer.signature_ids() == ["0", "1", "q1"]
        surviving_view = featurizer.view_with_cluster_seed_sidecars(["0", "1"], None, None)
        expected_features = np.asarray(surviving_view.featurize_pairs_matrix_indexed([(0, 1)], None, 1, np.nan))
        featurizer.release()
        assert featurizer.signature_ids() == []
        np.testing.assert_equal(
            surviving_view.featurize_pairs_matrix_indexed([(0, 1)], None, 1, np.nan), expected_features
        )
        surviving_view.release()


def test_planner_reload_reuses_unchanged_state_and_plans_like_a_fresh_build(tmp_path: Path) -> None:
    # A reload keeps the features of current seeds and moves over the summary of each
    # component whose ordered members are unchanged, under the component's new label.
    from s2and.feature_port import _require_rust_runtime
    from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow

    paths = write_minimal_arrow_prediction_bundle(tmp_path / "data", include_specter=True)
    first_seeds = {"0": "a", "1": "a", "2": "b"}
    # Component a keeps members 0, 1 as "x"; b gains 3; c is new.
    later_seeds = {"0": "x", "1": "x", "2": "b", "3": "b", "4": "c", "seed1": "c"}
    sidecars = {}
    for name, seeds in (("first", first_seeds), ("later", later_seeds)):
        sidecars[name] = tmp_path / f"{name}_seeds.arrow"
        write_cluster_seeds_arrow(sidecars[name], seeds)
    planner_class = _require_rust_runtime().RawBlockQueryCandidatePlanner
    options = {"top_k": 3, "orcid_enabled": True, "num_threads": 2, "max_exemplars": 4}
    queries = ["q1", "q2", "5", "6"]
    with ArrowDataset.open_trusted(paths, require_specter=True) as dataset:
        expected = planner_class.from_auto_queries(dataset.native, str(sidecars["later"]), **options).plan(queries)
        reused = planner_class.from_auto_queries(
            dataset.native, str(sidecars["first"]), retain_seed_rows=True, **options
        )
        reused.plan(queries)
        assert reused.reload_seed_sidecars(str(sidecars["later"])) is True
        telemetry = reused.build_telemetry()
        assert telemetry["reused_seed_feature_count"] == len(first_seeds)
        assert telemetry["reused_summary_count"] == 1
        assert _same_plan(reused.plan(queries), expected)


def test_planner_prefetched_block_rows_plan_and_reload_like_reads(tmp_path: Path) -> None:
    # Attachment passes prefetch the block's rows: plans and reloads must equal
    # those that read the Arrow tables, and a reload must read nothing new.
    from s2and.feature_port import _require_rust_runtime
    from s2and.incremental_linking.feature_block import write_cluster_seeds_arrow

    paths = write_minimal_arrow_prediction_bundle(tmp_path / "data", include_specter=True)
    with pa.ipc.open_file(paths["signatures"]) as reader:
        block = [str(signature_id) for signature_id in reader.read_all().column("signature_id").to_pylist()]
    first_seeds = {"0": "a", "1": "a", "2": "b"}
    later_seeds = {**first_seeds, "q1": "a", "5": "b", "4": "c"}
    sidecars = {}
    for name, seeds in (("first", first_seeds), ("later", later_seeds)):
        sidecars[name] = tmp_path / f"{name}_seeds.arrow"
        write_cluster_seeds_arrow(sidecars[name], seeds)
    planner_class = _require_rust_runtime().RawBlockQueryCandidatePlanner
    options = {"top_k": 3, "orcid_enabled": True, "num_threads": 2, "max_exemplars": 4}
    first_queries = ["q1", "q2", "5", "6"]
    later_queries = ["q2", "6"]
    with ArrowDataset.open_trusted(paths, require_specter=True) as dataset:
        expected_first = planner_class.from_auto_queries(dataset.native, str(sidecars["first"]), **options).plan(
            first_queries
        )
        expected_later = planner_class.from_auto_queries(dataset.native, str(sidecars["later"]), **options).plan(
            later_queries
        )
        prefetched = planner_class.from_auto_queries(
            dataset.native,
            str(sidecars["first"]),
            retain_seed_rows=True,
            prefetch_signature_ids=block,
            **options,
        )
        assert _same_plan(prefetched.plan(first_queries), expected_first)
        assert prefetched.reload_seed_sidecars(str(sidecars["later"])) is True
        telemetry = prefetched.build_telemetry()
        assert telemetry["signature_batches_read"] == 0
        assert telemetry["paper_author_batches_read"] == 0
        assert _same_plan(prefetched.plan(later_queries), expected_later)


def _write_varied_arrow_bundle(bundle_dir: Path) -> dict[str, str]:
    # Signatures on several papers with distinct names, coauthors, years, and
    # vectors. Paper p4 has no author rows and p5 has no SPECTER row.
    signature_rows = [
        ("0", "p0", "Ada", 0, None),
        ("1", "p1", "Ada", 1, "0000-0001"),
        ("2", "p1", "Augusta", 0, None),
        ("3", "p2", "A", 2, None),
        ("4", "p3", "Ada", 0, None),
        ("5", "p4", "Ada", 0, None),
        ("6", "p5", "Alan", 1, "0000-0001"),
        ("7", "p3", "Annabella", 1, None),
        ("q1", "p2", "Ada", 0, None),
        ("q2", "p5", "Ada", 0, None),
        ("q3", "p0", "Ann", 1, None),
    ]
    author_rows = [
        ("p0", 0, "Ada Lovelace"),
        ("p0", 1, "Ann Lovelace"),
        ("p0", 2, "Charles Babbage"),
        ("p1", 0, "Augusta Lovelace"),
        ("p1", 1, "Ada Lovelace"),
        ("p2", 0, "Ada Lovelace"),
        ("p2", 1, "Mary Somerville"),
        ("p2", 2, "A Lovelace"),
        ("p3", 0, "Ada Lovelace"),
        ("p3", 1, "Annabella Lovelace"),
        ("p5", 0, "Ada Lovelace"),
        ("p5", 1, "Alan Lovelace"),
    ]
    paper_ids = ["p0", "p1", "p2", "p3", "p4", "p5"]
    tables = {
        "signatures": pa.table(
            {
                "signature_id": [row[0] for row in signature_rows],
                "paper_id": [row[1] for row in signature_rows],
                "author_first": [row[2] for row in signature_rows],
                "author_middle": [""] * len(signature_rows),
                "author_last": ["Lovelace"] * len(signature_rows),
                "author_suffix": [""] * len(signature_rows),
                "author_affiliations": pa.array(
                    [["Analytical Society"] if index % 2 else [] for index in range(len(signature_rows))],
                    type=pa.list_(pa.string()),
                ),
                "author_position": pa.array([row[3] for row in signature_rows], type=pa.int64()),
                "author_block": ["a lovelace"] * len(signature_rows),
                "author_orcid": pa.array([row[4] for row in signature_rows], type=pa.string()),
            }
        ),
        "papers": pa.table(
            {
                "paper_id": paper_ids,
                "title": [
                    "Notes on the analytical engine",
                    "Sketch of the engine",
                    "On the connexion of the physical sciences",
                    "Bernoulli numbers by machine",
                    "Letters",
                    "A poetical science",
                ],
                "venue": ["Taylor", "", "Murray", "Taylor", "", "Murray"],
                "journal_name": ["Scientific Memoirs", "", "", "Scientific Memoirs", "", ""],
                "year": pa.array([1843, 1842, 1834, None, 1840, 1843], type=pa.int64()),
                "abstract": ["Engine notes", "", "Physics", "", "", "Poetry"],
            }
        ),
        "paper_authors": pa.table(
            {
                "paper_id": [row[0] for row in author_rows],
                "position": pa.array([row[1] for row in author_rows], type=pa.int64()),
                "author_name": [row[2] for row in author_rows],
            }
        ),
        "specter": pa.table(
            {
                "paper_id": paper_ids[:5],
                "embedding": pa.array(
                    [[1.0, 0.0, 0.5], [0.9, 0.1, 0.4], [0.0, 1.0, 0.2], [0.7, 0.7, 0.0], [0.2, 0.3, 0.9]],
                    type=pa.list_(pa.float32(), 3),
                ),
            }
        ),
    }
    return write_arrow_tables_bundle(bundle_dir, tables)


def test_planner_featurizer_matches_an_arrow_build(tmp_path: Path) -> None:
    # Request windows build featurizers from the planner's seed, query, and
    # prefetched rows; each must featurize and constrain like an Arrow build.
    from s2and.feature_port import _require_rust_runtime
    from s2and.incremental_linking.feature_block import (
        write_cluster_seed_disallows_arrow,
        write_cluster_seeds_arrow,
    )

    paths = _write_varied_arrow_bundle(tmp_path / "data")
    first_seeds = {"0": "a", "1": "a", "2": "b", "3": "b", "4": "c", "5": "c"}
    later_seeds = {**first_seeds, "q1": "b", "6": "d"}
    sidecars = {}
    for name, seeds in (("first", first_seeds), ("later", later_seeds)):
        sidecars[name] = tmp_path / f"{name}_seeds.arrow"
        write_cluster_seeds_arrow(sidecars[name], seeds)
    disallows = tmp_path / "disallows.arrow"
    write_cluster_seed_disallows_arrow(disallows, [("q1", "4"), ("0", "q3")])
    name_tuples = frozenset({("ada", "augusta"), ("ann", "annabella")})
    planner_class = _require_rust_runtime().RawBlockQueryCandidatePlanner
    options = {"top_k": 3, "orcid_enabled": True, "num_threads": 2, "max_exemplars": 4}

    def assert_matches_arrow(planner: object, signature_ids: list[str], seeds_path: Path) -> None:
        sidecar_paths = {"cluster_seeds_path": seeds_path, "cluster_seed_disallows_path": disallows}
        from_planner = build_rust_featurizer_from_planner(
            planner, signature_ids=signature_ids, name_tuples=name_tuples, num_threads=2, **sidecar_paths
        )
        from_arrow = build_rust_featurizer_from_arrow_dataset(
            dataset, signature_ids=signature_ids, name_tuples=name_tuples, num_threads=2, **sidecar_paths
        )
        assert from_planner is not None
        count = len(dict.fromkeys(signature_ids))
        pairs = [(left, right) for left in range(count) for right in range(left + 1, count)]
        np.testing.assert_array_equal(
            np.asarray(from_planner.featurize_pairs_matrix_indexed(pairs, None, 1, np.nan)),
            np.asarray(from_arrow.featurize_pairs_matrix_indexed(pairs, None, 1, np.nan)),
        )
        assert from_planner.get_constraints_matrix_indexed(
            pairs, 0.0, 10000.0, False, False, 1, False
        ) == from_arrow.get_constraints_matrix_indexed(pairs, 0.0, 10000.0, False, False, 1, False)
        assert from_planner.cluster_seeds_require() == from_arrow.cluster_seeds_require()

    with ArrowDataset.open_trusted(paths, require_specter=True) as dataset:
        # Seed rows plus the latest plan's query rows, in any order, with repeats.
        planner = planner_class.from_auto_queries(
            dataset.native,
            str(sidecars["first"]),
            cluster_seed_disallows_path=str(disallows),
            retain_seed_rows=True,
            **options,
        )
        planner.plan(["q1", "q2", "q3", "6"])
        assert_matches_arrow(planner, ["q2", "3", "q1", "0", "6", "5", "q3", "1", "2", "4", "q1"], sidecars["first"])
        # A signature outside the seeds and the latest queries is not held.
        assert build_rust_featurizer_from_planner(planner, signature_ids=["0", "7"], name_tuples=name_tuples) is None
        planner.plan(["q2"])
        assert build_rust_featurizer_from_planner(planner, signature_ids=["q1"], name_tuples=name_tuples) is None
        # Without retained rows the planner holds no paper rows.
        unretained = planner_class.from_auto_queries(dataset.native, str(sidecars["first"]), **options)
        assert build_rust_featurizer_from_planner(unretained, signature_ids=["0"], name_tuples=name_tuples) is None

        # Prefetched block rows, before and after a reload adds seeds.
        with pa.ipc.open_file(paths["signatures"]) as reader:
            block = [str(value) for value in reader.read_all().column("signature_id").to_pylist()]
        prefetched = planner_class.from_auto_queries(
            dataset.native,
            str(sidecars["first"]),
            cluster_seed_disallows_path=str(disallows),
            retain_seed_rows=True,
            prefetch_signature_ids=block,
            **options,
        )
        prefetched.plan(["q1", "6"])
        assert_matches_arrow(prefetched, ["q1", "6", "7", "0", "4"], sidecars["first"])
        assert prefetched.reload_seed_sidecars(str(sidecars["later"]), str(disallows)) is True
        prefetched.plan(["q2", "q3"])
        assert_matches_arrow(prefetched, ["q3", "q2", "q1", "6", "7", "5", "0"], sidecars["later"])

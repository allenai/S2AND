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
    # Sidecars are request inputs and need not exist when opening the dataset.
    paths["cluster_seeds"] = str(tmp_path / "nonexistent-sidecar.arrow")

    with ArrowDataset.open_trusted(paths, require_specter=include_specter) as dataset:
        assert dataset.root == tmp_path.resolve()
        assert dataset.has("specter") == include_specter
        assert not dataset.has("cluster_seeds")
        np.testing.assert_array_equal(_features(dataset), expected)
        with dataset.use() as lease, lease.open_file("signatures") as source:
            assert pa.ipc.open_file(source).read_all().num_rows == 13
            with pytest.raises(RuntimeError, match="active use"):
                dataset.close()
    assert dataset.closed
    with pytest.raises(RuntimeError, match="closed"):
        _features(dataset)


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

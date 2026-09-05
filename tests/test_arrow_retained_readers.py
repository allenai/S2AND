"""Exercise retained Python streams against the real native Arrow readers."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa

from s2and.arrow_inputs import ArrowDataset
from s2and.feature_port import build_rust_featurizer_from_arrow_dataset
from s2and.incremental_linking.feature_block import write_arrow_ipc_table
from tests.helpers import write_minimal_arrow_prediction_bundle


def _native_features(dataset: ArrowDataset) -> np.ndarray:
    """Read the retained native tables and score one fixed signature pair."""
    featurizer = build_rust_featurizer_from_arrow_dataset(dataset, num_threads=1)
    return np.asarray(featurizer.featurize_pairs_matrix_indexed([(0, 1)], None, 1, np.nan))


def test_native_and_python_readers_share_dataset_across_threads(tmp_path: Path) -> None:
    """Interleave a native request with another request's Python file lease."""
    write_minimal_arrow_prediction_bundle(tmp_path)
    expected = (tmp_path / "signatures.arrow").read_bytes()
    barrier = threading.Barrier(2)

    with ArrowDataset.open(tmp_path) as dataset:
        expected_features = _native_features(dataset)

        def read_python() -> bytes:
            with dataset.use() as lease, lease.open_file("signatures") as source:
                prefix = source.read(8)
                position = source.tell()
                barrier.wait(timeout=30)
                barrier.wait(timeout=30)
                assert source.tell() == position
                return prefix + source.read()

        def read_native() -> np.ndarray:
            with dataset.use():
                barrier.wait(timeout=30)
                try:
                    return _native_features(dataset)
                finally:
                    barrier.wait(timeout=30)

        with ThreadPoolExecutor(max_workers=2) as pool:
            python_result = pool.submit(read_python)
            native_result = pool.submit(read_native)
            assert python_result.result(timeout=30) == expected
            np.testing.assert_array_equal(native_result.result(timeout=30), expected_features)


def test_interleaved_readers_retain_original_file_after_path_replacement(tmp_path: Path) -> None:
    """Independent cursors must still refer to the originally validated file."""
    write_minimal_arrow_prediction_bundle(tmp_path)
    signature_path = tmp_path / "signatures.arrow"
    expected = signature_path.read_bytes()
    with pa.BufferReader(expected) as source:
        original = pa.ipc.open_file(source).read_all()
        replacement_table = original.set_column(
            original.schema.get_field_index("author_first"),
            "author_first",
            pa.array(["Other"] + ["Ada"] * (original.num_rows - 1), type=pa.string()),
        )
        replacement_path = tmp_path / "replacement.arrow"
        write_arrow_ipc_table(replacement_table, replacement_path)
    del original, replacement_table

    with ArrowDataset.open(tmp_path) as dataset:
        expected_features = _native_features(dataset)
        if os.name == "nt":
            signature_path.unlink()
            replacement_path.rename(signature_path)
        else:
            replacement_path.replace(signature_path)
        assert signature_path.read_bytes() != expected

        with dataset.use() as lease, lease.open_file("signatures") as source:
            prefix = source.read(8)
            observed_features = _native_features(dataset)
            assert prefix + source.read() == expected
        np.testing.assert_array_equal(observed_features, expected_features)

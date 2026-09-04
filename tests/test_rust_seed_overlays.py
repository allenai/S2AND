"""Native request seed overlays preserve reusable feature data and prior requests."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from s2and.feature_port import _get_rust_featurizer
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset, import_s2and_rust

HAS_RUST, _RUST_IMPORT_PAYLOAD = import_s2and_rust()
if not HAS_RUST:
    pytest.skip(f"Rust extension unavailable: {_RUST_IMPORT_PAYLOAD}", allow_module_level=True)


def test_native_seed_overlays_are_independent_after_interleaving_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two seed policies on one native backing survive interleaved calls and errors."""
    monkeypatch.setenv("S2AND_BACKEND", "python")
    source = build_dummy_dataset("seed-overlay", name_counts_index=True)
    dataset = build_arrow_training_dataset(source, tmp_path)
    base = _get_rust_featurizer(dataset)
    ids = base.signature_ids()
    pairs = [(ids.index("3"), ids.index("4"))]
    together = base.with_cluster_seeds({"3": "same", "4": "same"}, set())
    apart = base.with_cluster_seeds({"3": "left", "4": "right"}, set())
    base_constraints = base.get_constraints_matrix_indexed(pairs)
    assert together.get_constraints_matrix_indexed(pairs) == [0.0]
    assert apart.get_constraints_matrix_indexed(pairs) == [10000.0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda handle: handle.get_constraints_matrix_indexed(pairs), [apart, together] * 4))
    assert results == [[10000.0], [0.0]] * 4
    with pytest.raises((TypeError, ValueError)):
        base.with_cluster_seeds({"3": "replacement"}, {(1,)})
    assert base.get_constraints_matrix_indexed(pairs) == base_constraints
    assert together.get_constraints_matrix_indexed(pairs) == [0.0]
    assert apart.get_constraints_matrix_indexed(pairs) == [10000.0]
    np.testing.assert_equal(together.featurize_pairs_matrix_indexed(pairs), apart.featurize_pairs_matrix_indexed(pairs))

from __future__ import annotations

import numpy as np
import pytest

import s2and.featurizer as featurizer_module
from s2and.featurizer import FeaturizationInfo, ScatterContext


@pytest.mark.parametrize(
    ("platform_name", "expected_use_threads"),
    [
        ("Windows", True),
        ("Linux", False),
    ],
)
def test_execute_python_featurization_phase_explicit_pool_mode(monkeypatch, platform_name, expected_use_threads):
    monkeypatch.setattr(featurizer_module.platform, "system", lambda: platform_name)
    scatter_context = ScatterContext(
        features=np.zeros((0, 0)),
        nameless_features=None,
        coauthor_similarity_values=None,
        identity_selected_indices=True,
        indices_to_use=[],
        nameless_indices_to_use=[],
        selected_positions=[],
        nameless_positions=[],
        coauthor_similarity_index=None,
        coauthor_position=None,
    )

    class FakeUniversalPool:
        init_calls = 0
        last_use_threads = None
        last_processes = None

        def __init__(self, processes: int | None = None, use_threads: bool | None = None):
            type(self).init_calls += 1
            type(self).last_use_threads = use_threads
            type(self).last_processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap(self, _func, _iterable, chunksize=1, max_prefetch=4):
            _ = chunksize, max_prefetch
            return iter(())

    monkeypatch.setattr(featurizer_module, "UniversalPool", FakeUniversalPool)

    backend, new_features_count = featurizer_module._execute_python_featurization_phase(
        pieces_of_work=[],
        n_jobs=4,
        chunk_size=64,
        use_cache=False,
        signature_pairs=[],
        featurizer_info=FeaturizationInfo(features_to_use=[]),
        scatter_context=scatter_context,
        cached_features={},
    )

    assert backend == "python_parallel"
    assert new_features_count == 0
    assert FakeUniversalPool.init_calls == 1
    assert FakeUniversalPool.last_processes == 1
    assert FakeUniversalPool.last_use_threads is expected_use_threads

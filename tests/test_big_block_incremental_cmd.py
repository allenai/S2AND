from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts._rust_suite.big_block_incremental_cmd as big_block_incremental_cmd


def _base_args(**overrides):
    args = {
        "mode": "single",
        "backend": "rust",
        "subset_dir": "",
        "target_block": "",
        "total_signatures": 3,
        "seed_signatures": 2,
        "seed_cluster_count": 1,
        "batching_threshold": 12,
        "n_jobs": 20,
        "random_seed": 7,
        "use_orcid_id": 1,
        "specter_path": "",
        "cluster_seeds_path": "",
        "altered_cluster_signatures_path": "",
        "total_ram_bytes": 34,
        "model_path": "model.pickle",
        "emit_signature_map": 0,
        "write_json": "",
        "single_write_json": "",
        "fail_on_cluster_mismatch": 1,
        "require_rust_release": 0,
        "full_run": True,
    }
    args.update(overrides)
    return Namespace(**args)


def test_run_single_forwards_explicit_artifact_paths(monkeypatch, tmp_path: Path):
    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    specter_path = tmp_path / "specter.pickle"
    cluster_seeds_path = tmp_path / "cluster_seeds.json"
    altered_path = tmp_path / "altered_cluster_signatures.txt"
    for path, contents in [
        (specter_path, "specter"),
        (cluster_seeds_path, "{}"),
        (altered_path, "s1\n"),
    ]:
        path.write_text(contents, encoding="utf-8")
    model_path = tmp_path / "model.pickle"
    model_path.write_text("model", encoding="utf-8")

    signatures = {
        "s1": {"paper_id": "p1", "author_info": {"block": "h wang"}},
        "s2": {"paper_id": "p2", "author_info": {"block": "h wang"}},
        "s3": {"paper_id": "p3", "author_info": {"block": "h wang"}},
    }
    papers = {
        "p1": {"authors": [{"author_name": "A"}]},
        "p2": {"authors": [{"author_name": "B"}]},
        "p3": {"authors": [{"author_name": "C"}]},
    }

    monkeypatch.setattr(
        big_block_incremental_cmd,
        "_load_subset_payload",
        lambda _subset_dir: (signatures, papers, "h wang"),
    )
    monkeypatch.setattr(
        big_block_incremental_cmd,
        "_build_cluster_seeds",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("synthetic seed builder should not run")),
    )
    monkeypatch.setattr(big_block_incremental_cmd, "collect_rust_extension_identity", lambda **_kwargs: {"rust": 1})
    monkeypatch.setattr(
        big_block_incremental_cmd,
        "ProcessTreeRSSMonitor",
        type(
            "FakeMonitor",
            (),
            {
                "__init__": lambda self, interval_seconds=0.05: setattr(self, "peak_gb", 1.25),
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: False,
            },
        ),
    )

    captured_anddata_kwargs = {}

    class FakeANDData:
        def __init__(self, **kwargs):
            captured_anddata_kwargs.update(kwargs)
            self.cluster_seeds_require = {"s1": 0}
            self.cluster_seeds_disallow = set()
            self.altered_cluster_signatures = ["s1"]
            self.max_seed_cluster_id = 1
            self._rust_cluster_seeds_sync_calls = 0
            self._rust_cluster_seeds_sync_attempted = 0
            self._rust_cluster_seeds_sync_succeeded = 0
            self._rust_cluster_seeds_sync_skipped_unchanged = 0
            self._rust_cluster_seeds_sync_seconds_total = 0.0
            self._rust_cluster_seeds_sync_seconds_max = 0.0

    monkeypatch.setattr("s2and.data.ANDData", FakeANDData)
    monkeypatch.setattr(
        "s2and.serialization.load_pickle_with_verified_label_encoder_compat",
        lambda _path: {
            "clusterer": SimpleNamespace(
                classifier=SimpleNamespace(),
                nameless_classifier=SimpleNamespace(),
                use_cache=False,
                n_jobs=0,
                predict_incremental=lambda block, dataset, **kwargs: {
                    "clusters": {"0": list(block)},
                    "phase_b_mode": "exact",
                    "phase_b_budget_bytes": 1,
                    "phase_b_required_bytes": 1,
                },
            )
        },
    )
    monkeypatch.setattr("s2and.model._ensure_lightgbm_fitted", lambda *_args, **_kwargs: None)

    result = big_block_incremental_cmd._run_single(
        _base_args(
            subset_dir=str(subset_dir),
            use_orcid_id=0,
            specter_path=str(specter_path),
            cluster_seeds_path=str(cluster_seeds_path),
            altered_cluster_signatures_path=str(altered_path),
            model_path=str(model_path),
        )
    )

    assert captured_anddata_kwargs["use_orcid_id"] is False
    assert captured_anddata_kwargs["specter_embeddings"] == str(specter_path)
    assert captured_anddata_kwargs["cluster_seeds"] == str(cluster_seeds_path)
    assert captured_anddata_kwargs["altered_cluster_signatures"] == str(altered_path)
    assert result["use_orcid_id"] is False
    assert result["cluster_seeds_source"] == str(cluster_seeds_path)
    assert result["specter_embeddings_source"] == str(specter_path)
    assert result["altered_cluster_signatures_source"] == str(altered_path)
    assert result["seed_signatures"] == 1
    assert result["unassigned_signatures"] == 2
    assert result["estimated_incremental_pairs"] == 2


def test_run_single_uses_synthetic_cluster_seeds_when_no_path_is_supplied(monkeypatch, tmp_path: Path):
    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    model_path = tmp_path / "model.pickle"
    model_path.write_text("model", encoding="utf-8")

    signatures = {
        "s1": {"paper_id": "p1", "author_info": {"block": "h wang"}},
        "s2": {"paper_id": "p2", "author_info": {"block": "h wang"}},
        "s3": {"paper_id": "p3", "author_info": {"block": "h wang"}},
    }
    papers = {
        "p1": {"authors": [{"author_name": "A"}]},
        "p2": {"authors": [{"author_name": "B"}]},
        "p3": {"authors": [{"author_name": "C"}]},
    }

    synthetic_cluster_seeds = {"s1": {"s2": "require"}}
    monkeypatch.setattr(
        big_block_incremental_cmd,
        "_load_subset_payload",
        lambda _subset_dir: (signatures, papers, "h wang"),
    )
    cluster_seed_build_calls = {}

    def fake_build_cluster_seeds(seed_signature_ids, seed_cluster_count):
        cluster_seed_build_calls["seed_signature_ids"] = list(seed_signature_ids)
        cluster_seed_build_calls["seed_cluster_count"] = seed_cluster_count
        return synthetic_cluster_seeds

    monkeypatch.setattr(big_block_incremental_cmd, "_build_cluster_seeds", fake_build_cluster_seeds)
    monkeypatch.setattr(big_block_incremental_cmd, "collect_rust_extension_identity", lambda **_kwargs: {"rust": 1})
    monkeypatch.setattr(
        big_block_incremental_cmd,
        "ProcessTreeRSSMonitor",
        type(
            "FakeMonitor",
            (),
            {
                "__init__": lambda self, interval_seconds=0.05: setattr(self, "peak_gb", 1.25),
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: False,
            },
        ),
    )

    captured_anddata_kwargs = {}

    class FakeANDData:
        def __init__(self, **kwargs):
            captured_anddata_kwargs.update(kwargs)
            self.cluster_seeds_require = {"s1": 0}
            self.cluster_seeds_disallow = set()
            self.altered_cluster_signatures = None
            self.max_seed_cluster_id = 1
            self._rust_cluster_seeds_sync_calls = 0
            self._rust_cluster_seeds_sync_attempted = 0
            self._rust_cluster_seeds_sync_succeeded = 0
            self._rust_cluster_seeds_sync_skipped_unchanged = 0
            self._rust_cluster_seeds_sync_seconds_total = 0.0
            self._rust_cluster_seeds_sync_seconds_max = 0.0

    monkeypatch.setattr("s2and.data.ANDData", FakeANDData)
    monkeypatch.setattr(
        "s2and.serialization.load_pickle_with_verified_label_encoder_compat",
        lambda _path: {
            "clusterer": SimpleNamespace(
                classifier=SimpleNamespace(),
                nameless_classifier=SimpleNamespace(),
                use_cache=False,
                n_jobs=0,
                predict_incremental=lambda block, dataset, **kwargs: {
                    "clusters": {"0": list(block)},
                    "phase_b_mode": "exact",
                    "phase_b_budget_bytes": 1,
                    "phase_b_required_bytes": 1,
                },
            )
        },
    )
    monkeypatch.setattr("s2and.model._ensure_lightgbm_fitted", lambda *_args, **_kwargs: None)

    result = big_block_incremental_cmd._run_single(
        _base_args(
            subset_dir=str(subset_dir),
            model_path=str(model_path),
        )
    )

    assert len(cluster_seed_build_calls["seed_signature_ids"]) == 2
    assert set(cluster_seed_build_calls["seed_signature_ids"]).issubset(signatures)
    assert cluster_seed_build_calls["seed_cluster_count"] == 1
    assert captured_anddata_kwargs["specter_embeddings"] is None
    assert captured_anddata_kwargs["cluster_seeds"] == synthetic_cluster_seeds
    assert captured_anddata_kwargs["altered_cluster_signatures"] is None
    assert captured_anddata_kwargs["use_orcid_id"] is True
    assert result["cluster_seeds_source"] == "synthetic"
    assert result["specter_embeddings_source"] == "unset"
    assert result["altered_cluster_signatures_source"] == "unset"
    assert result["seed_signatures_requested"] == 2
    assert result["seed_signatures"] == 1
    assert result["unassigned_signatures"] == 2
    assert result["estimated_incremental_pairs"] == 2


def test_run_single_reports_actual_external_seed_counts(monkeypatch, tmp_path: Path):
    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    cluster_seeds_path = tmp_path / "cluster_seeds.json"
    cluster_seeds_path.write_text("{}", encoding="utf-8")
    model_path = tmp_path / "model.pickle"
    model_path.write_text("model", encoding="utf-8")

    signatures = {
        "s1": {"paper_id": "p1", "author_info": {"block": "h wang"}},
        "s2": {"paper_id": "p2", "author_info": {"block": "h wang"}},
        "s3": {"paper_id": "p3", "author_info": {"block": "h wang"}},
        "s4": {"paper_id": "p4", "author_info": {"block": "h wang"}},
    }
    papers = {
        "p1": {"authors": [{"author_name": "A"}]},
        "p2": {"authors": [{"author_name": "B"}]},
        "p3": {"authors": [{"author_name": "C"}]},
        "p4": {"authors": [{"author_name": "D"}]},
    }

    monkeypatch.setattr(
        big_block_incremental_cmd,
        "_load_subset_payload",
        lambda _subset_dir: (signatures, papers, "h wang"),
    )
    monkeypatch.setattr(big_block_incremental_cmd, "collect_rust_extension_identity", lambda **_kwargs: {"rust": 1})
    monkeypatch.setattr(
        big_block_incremental_cmd,
        "ProcessTreeRSSMonitor",
        type(
            "FakeMonitor",
            (),
            {
                "__init__": lambda self, interval_seconds=0.05: setattr(self, "peak_gb", 1.25),
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: False,
            },
        ),
    )

    class FakeANDData:
        def __init__(self, **kwargs):
            self.cluster_seeds_require = {"s1": 0, "s2": 0, "s3": 1, "s4": 1}
            self.cluster_seeds_disallow = set()
            self.altered_cluster_signatures = None
            self.max_seed_cluster_id = 2
            self._rust_cluster_seeds_sync_calls = 0
            self._rust_cluster_seeds_sync_attempted = 0
            self._rust_cluster_seeds_sync_succeeded = 0
            self._rust_cluster_seeds_sync_skipped_unchanged = 0
            self._rust_cluster_seeds_sync_seconds_total = 0.0
            self._rust_cluster_seeds_sync_seconds_max = 0.0

    monkeypatch.setattr("s2and.data.ANDData", FakeANDData)
    monkeypatch.setattr(
        "s2and.serialization.load_pickle_with_verified_label_encoder_compat",
        lambda _path: {
            "clusterer": SimpleNamespace(
                classifier=SimpleNamespace(),
                nameless_classifier=SimpleNamespace(),
                use_cache=False,
                n_jobs=0,
                predict_incremental=lambda block, dataset, **kwargs: {
                    "clusters": {"0": ["s1", "s2"], "1": ["s3", "s4"]},
                    "phase_b_mode": "exact",
                    "phase_b_budget_bytes": 1,
                    "phase_b_required_bytes": 1,
                },
            )
        },
    )
    monkeypatch.setattr("s2and.model._ensure_lightgbm_fitted", lambda *_args, **_kwargs: None)

    result = big_block_incremental_cmd._run_single(
        _base_args(
            subset_dir=str(subset_dir),
            total_signatures=4,
            seed_signatures=1,
            seed_cluster_count=1,
            cluster_seeds_path=str(cluster_seeds_path),
            model_path=str(model_path),
        )
    )

    assert result["seed_signatures_requested"] == 1
    assert result["seed_signatures"] == 4
    assert result["unassigned_signatures"] == 0
    assert result["seed_clusters_effective"] == 2
    assert result["estimated_incremental_pairs"] == 0


def test_run_single_rejects_external_seed_signatures_outside_selected_subset(monkeypatch, tmp_path: Path):
    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    cluster_seeds_path = tmp_path / "cluster_seeds.json"
    cluster_seeds_path.write_text("{}", encoding="utf-8")
    model_path = tmp_path / "model.pickle"
    model_path.write_text("model", encoding="utf-8")

    signatures = {
        "s1": {"paper_id": "p1", "author_info": {"block": "h wang"}},
        "s2": {"paper_id": "p2", "author_info": {"block": "h wang"}},
        "s3": {"paper_id": "p3", "author_info": {"block": "h wang"}},
    }
    papers = {
        "p1": {"authors": [{"author_name": "A"}]},
        "p2": {"authors": [{"author_name": "B"}]},
        "p3": {"authors": [{"author_name": "C"}]},
    }

    monkeypatch.setattr(
        big_block_incremental_cmd,
        "_load_subset_payload",
        lambda _subset_dir: (signatures, papers, "h wang"),
    )
    monkeypatch.setattr(big_block_incremental_cmd, "collect_rust_extension_identity", lambda **_kwargs: {"rust": 1})
    monkeypatch.setattr(
        big_block_incremental_cmd,
        "ProcessTreeRSSMonitor",
        type(
            "FakeMonitor",
            (),
            {
                "__init__": lambda self, interval_seconds=0.05: setattr(self, "peak_gb", 1.25),
                "__enter__": lambda self: self,
                "__exit__": lambda self, exc_type, exc, tb: False,
            },
        ),
    )

    class FakeANDData:
        def __init__(self, **kwargs):
            self.cluster_seeds_require = {"outside": 0, "s1": 0}
            self.cluster_seeds_disallow = set()
            self.altered_cluster_signatures = None
            self.max_seed_cluster_id = 1
            self._rust_cluster_seeds_sync_calls = 0
            self._rust_cluster_seeds_sync_attempted = 0
            self._rust_cluster_seeds_sync_succeeded = 0
            self._rust_cluster_seeds_sync_skipped_unchanged = 0
            self._rust_cluster_seeds_sync_seconds_total = 0.0
            self._rust_cluster_seeds_sync_seconds_max = 0.0

    monkeypatch.setattr("s2and.data.ANDData", FakeANDData)
    monkeypatch.setattr(
        "s2and.serialization.load_pickle_with_verified_label_encoder_compat",
        lambda _path: {
            "clusterer": SimpleNamespace(
                classifier=SimpleNamespace(),
                nameless_classifier=SimpleNamespace(),
                use_cache=False,
                n_jobs=0,
                predict_incremental=lambda block, dataset, **kwargs: {
                    "clusters": {"0": list(block)},
                    "phase_b_mode": "exact",
                    "phase_b_budget_bytes": 1,
                    "phase_b_required_bytes": 1,
                },
            )
        },
    )
    monkeypatch.setattr("s2and.model._ensure_lightgbm_fitted", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="outside the selected subset"):
        big_block_incremental_cmd._run_single(
            _base_args(
                subset_dir=str(subset_dir),
                cluster_seeds_path=str(cluster_seeds_path),
                model_path=str(model_path),
            )
        )


def test_validate_args_ignores_seed_counts_when_external_seed_path_is_supplied():
    big_block_incremental_cmd._validate_args(
        _base_args(
            cluster_seeds_path="precomputed.json",
            seed_signatures=0,
            seed_cluster_count=0,
        )
    )

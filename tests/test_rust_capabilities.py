from __future__ import annotations

import numpy as np
import pytest

import s2and.rust_capabilities as rust_capabilities
from s2and import runtime as _runtime

# A rust extension version that satisfies the current minimum. Derived from the
# guard itself so these fixtures track the lockstep version bump instead of going
# stale on every release (see MIN_SUPPORTED_RUST_EXTENSION_VERSION).
_SUPPORTED_VERSION = _runtime.min_supported_rust_extension_version_string()


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def _make_core_rust_featurizer(*, supports_from_dataset_paper_preprocess: bool = False):
    class RustFeaturizer:
        SUPPORTS_FROM_DATASET_PAPER_PREPROCESS = supports_from_dataset_paper_preprocess

        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        @staticmethod
        def from_json_paths(*args, **kwargs):
            return None

        def json_ingest_telemetry(self):
            return {}

        def signature_ids(self):
            return []

        def get_constraint(self, *args, **kwargs):
            return None

        def get_constraints_matrix(self, *args, **kwargs):
            return []

        def get_constraints_matrix_indexed(self, *args, **kwargs):
            return []

        def featurize_pairs_matrix_indexed(self, *args, **kwargs):
            return None

        def update_signature_name_counts(self, *args, **kwargs):
            return 0

    return RustFeaturizer


def test_load_s2and_rust_extension_prefers_versioned_candidate_on_tie(monkeypatch):
    RustFeaturizer = _make_core_rust_featurizer()

    class ShimModule:
        __version__ = None

    ShimModule.RustFeaturizer = RustFeaturizer

    class NativeModule:
        __version__ = _SUPPORTED_VERSION

    NativeModule.RustFeaturizer = RustFeaturizer

    def _fake_import_module(name: str):
        if name == "s2and_rust":
            return ShimModule
        if name == "s2and_rust._s2and_rust":
            return NativeModule
        raise _missing_module(name)

    monkeypatch.setattr(rust_capabilities.importlib, "import_module", _fake_import_module)
    loaded = rust_capabilities.load_s2and_rust_extension()
    assert loaded is NativeModule


def test_load_s2and_rust_extension_returns_none_for_missing_top_level(monkeypatch):
    def _fake_import_module(name: str):
        raise _missing_module(name)

    monkeypatch.setattr(rust_capabilities.importlib, "import_module", _fake_import_module)

    assert rust_capabilities.load_s2and_rust_extension() is None


def test_load_s2and_rust_extension_returns_none_for_missing_workspace_native_module(monkeypatch):
    def _fake_import_module(name: str):
        if name == "s2and_rust":
            raise _missing_module("s2and_rust.s2and_rust._s2and_rust")
        raise _missing_module(name)

    monkeypatch.setattr(rust_capabilities.importlib, "import_module", _fake_import_module)

    assert rust_capabilities.load_s2and_rust_extension() is None


def test_load_s2and_rust_extension_reraises_import_crash(monkeypatch):
    def _fake_import_module(_name: str):
        raise RuntimeError("bad dll")

    monkeypatch.setattr(rust_capabilities.importlib, "import_module", _fake_import_module)

    with pytest.raises(RuntimeError, match="bad dll"):
        rust_capabilities.load_s2and_rust_extension()


def test_load_s2and_rust_extension_reraises_nested_missing_dependency(monkeypatch):
    RustFeaturizer = _make_core_rust_featurizer()

    class ShimModule:
        __version__ = _SUPPORTED_VERSION

    ShimModule.RustFeaturizer = RustFeaturizer

    def _fake_import_module(name: str):
        if name == "s2and_rust":
            return ShimModule
        if name == "s2and_rust._s2and_rust":
            raise _missing_module("missing_dependency")
        raise _missing_module(name)

    monkeypatch.setattr(rust_capabilities.importlib, "import_module", _fake_import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        rust_capabilities.load_s2and_rust_extension()
    assert exc_info.value.name == "missing_dependency"


def test_detect_rust_runtime_capabilities_requires_core_markers():
    class MissingMarkerRustFeaturizer:
        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        @staticmethod
        def from_json_paths(*args, **kwargs):
            return None

    class Module:
        __version__ = "0.49.0"

    Module.RustFeaturizer = MissingMarkerRustFeaturizer

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert capabilities.extension_importable is True
    assert capabilities.core_runtime_available is False
    assert capabilities.reason.startswith("rust_core_missing_markers:")


def test_detect_rust_runtime_capabilities_rejects_old_version():
    RustFeaturizer = _make_core_rust_featurizer()

    class Module:
        __version__ = "0.49.9"

    Module.RustFeaturizer = RustFeaturizer

    blocked = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert blocked.core_runtime_available is False
    assert blocked.reason.startswith("rust_version_below_minimum:")


def test_detect_rust_runtime_capabilities_rejects_unparseable_version():
    RustFeaturizer = _make_core_rust_featurizer()

    class Module:
        __version__ = "dev-local"

    Module.RustFeaturizer = RustFeaturizer

    blocked = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert blocked.core_runtime_available is False
    assert blocked.reason.startswith("rust_version_unparseable:")


def test_detect_rust_runtime_capabilities_reads_from_dataset_paper_preprocess_marker():
    RustFeaturizer = _make_core_rust_featurizer(supports_from_dataset_paper_preprocess=True)

    class Module:
        __version__ = _SUPPORTED_VERSION

    Module.RustFeaturizer = RustFeaturizer

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert capabilities.core_runtime_available is True
    assert capabilities.from_dataset_paper_preprocess_available is True


def test_detect_rust_runtime_capabilities_requires_json_ingest_telemetry():
    class RustFeaturizerWithoutTelemetry:
        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        @staticmethod
        def from_json_paths(*args, **kwargs):
            return None

        def signature_ids(self):
            return []

        def get_constraint(self, *args, **kwargs):
            return None

        def get_constraints_matrix(self, *args, **kwargs):
            return []

        def get_constraints_matrix_indexed(self, *args, **kwargs):
            return []

        def featurize_pairs_matrix_indexed(self, *args, **kwargs):
            return None

        def update_signature_name_counts(self, *args, **kwargs):
            return 0

    class Module:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = RustFeaturizerWithoutTelemetry

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)

    assert capabilities.core_runtime_available is False
    assert "json_ingest_telemetry" in capabilities.reason


def test_detect_rust_runtime_capabilities_reports_incremental_linker_names():
    class NamedRustFeaturizer(_make_core_rust_featurizer()):
        def linker_pair_index_arrays_aggregate_stats(self, *args, **kwargs):
            return None

        def linker_pair_index_arrays_constraint_labels(self, *args, **kwargs):
            return None

        def linker_pair_distance_accumulators(self, *args, **kwargs):
            return None

    class PairPlanMethod:
        __text_signature__ = (
            "($self, queries, query_signature_indices, component_member_indices_by_key, top_k, "
            "num_threads=None, query_signature_ids=None, retrieval_subblock_index=None, "
            "query_candidate_component_keys_by_signature_id=None, full_first_global_backfill_count=0)"
        )

        def __call__(self, *args, **kwargs):
            return None

    class NamedRustHybridCentroidRetriever:
        def top_k_hybrid_centroid(self, *args, **kwargs):
            return None

        top_k_hybrid_centroid_pair_plan = PairPlanMethod()

    class Module:
        __version__ = _SUPPORTED_VERSION
        INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS = ("row_orcid_match",)
        RustFeaturizer = NamedRustFeaturizer
        RustHybridCentroidRetriever = NamedRustHybridCentroidRetriever

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)

    assert "hybrid_centroid_retriever_v1" in capabilities.named_capabilities
    assert "indexed_pair_array_featurization_v1" in capabilities.named_capabilities
    assert "incremental_linking_pair_plan_v1" in capabilities.named_capabilities
    assert "incremental_linking_constraint_arrays_v1" in capabilities.named_capabilities


def test_detect_rust_runtime_capabilities_rejects_stale_incremental_pair_plan_abi():
    class NamedRustFeaturizer(_make_core_rust_featurizer()):
        def linker_pair_index_arrays_aggregate_stats(self, *args, **kwargs):
            return None

    class StalePairPlanMethod:
        __text_signature__ = (
            "($self, queries, query_signature_indices, component_member_indices_by_key, top_k, "
            "num_threads=None, query_signature_ids=None, retrieval_subblock_index=None, "
            "full_first_global_backfill_count=0)"
        )

        def __call__(self, *args, **kwargs):
            return None

    class NamedRustHybridCentroidRetriever:
        def top_k_hybrid_centroid(self, *args, **kwargs):
            return None

        top_k_hybrid_centroid_pair_plan = StalePairPlanMethod()

    class Module:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = NamedRustFeaturizer
        RustHybridCentroidRetriever = NamedRustHybridCentroidRetriever

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)

    assert "hybrid_centroid_retriever_v1" in capabilities.named_capabilities
    assert "indexed_pair_array_featurization_v1" in capabilities.named_capabilities
    assert "incremental_linking_pair_plan_v1" not in capabilities.named_capabilities


def test_detect_rust_runtime_capabilities_requires_pair_plan_orcid_signal_marker():
    class NamedRustFeaturizer(_make_core_rust_featurizer()):
        def linker_pair_index_arrays_aggregate_stats(self, *args, **kwargs):
            return None

    class PairPlanMethod:
        __text_signature__ = (
            "($self, queries, query_signature_indices, component_member_indices_by_key, top_k, "
            "num_threads=None, query_signature_ids=None, retrieval_subblock_index=None, "
            "query_candidate_component_keys_by_signature_id=None, full_first_global_backfill_count=0)"
        )

        def __call__(self, *args, **kwargs):
            return None

    class NamedRustHybridCentroidRetriever:
        def top_k_hybrid_centroid(self, *args, **kwargs):
            return None

        top_k_hybrid_centroid_pair_plan = PairPlanMethod()

    class Module:
        __version__ = _SUPPORTED_VERSION
        INCREMENTAL_LINKING_PAIR_PLAN_ROW_SIGNALS = ()
        RustFeaturizer = NamedRustFeaturizer
        RustHybridCentroidRetriever = NamedRustHybridCentroidRetriever

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)

    assert "hybrid_centroid_retriever_v1" in capabilities.named_capabilities
    assert "indexed_pair_array_featurization_v1" in capabilities.named_capabilities
    assert "incremental_linking_pair_plan_v1" not in capabilities.named_capabilities


def test_rust_get_build_info_contract():
    s2and_rust = pytest.importorskip("s2and_rust")

    get_build_info = getattr(s2and_rust, "get_build_info", None)
    if not callable(get_build_info):
        pytest.skip("s2and_rust.get_build_info unavailable")

    info = get_build_info()
    assert isinstance(info, dict)
    for key in ("crate_version", "profile", "debug_assertions", "opt_level", "target"):
        assert key in info
    row_signals = tuple(info.get("incremental_linking_pair_plan_row_signals", ()))
    if not row_signals:
        plan = s2and_rust.RustHybridCentroidRetriever([], include_exemplars=True).top_k_hybrid_centroid_pair_plan(
            [],
            np.asarray([], dtype=np.uint32),
            {},
            1,
            1,
        )
        row_signals = tuple(plan)
    assert "row_orcid_match" in row_signals

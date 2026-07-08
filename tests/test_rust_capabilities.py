from __future__ import annotations

from typing import Any, cast

import pytest

import s2and.runtime as rust_capabilities
from tests.helpers import import_s2and_rust

# A rust extension version that satisfies the current minimum. Derived from the
# guard itself so these fixtures track the lockstep version bump instead of going
# stale on every release (see MIN_SUPPORTED_RUST_EXTENSION_VERSION).
_SUPPORTED_VERSION = rust_capabilities.min_supported_rust_extension_version_string()


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def _make_core_rust_featurizer():
    class RustFeaturizer:
        @staticmethod
        def from_arrow_paths(*args, **kwargs):
            return None

        def signature_ids(self):
            return []

        def get_constraints_matrix_indexed(self, *args, **kwargs):
            return []

        def featurize_pairs_matrix_indexed(self, *args, **kwargs):
            return None

        def update_signature_name_counts(self, *args, **kwargs):
            return 0

    return RustFeaturizer


def _pair_plan_build_info(
    *,
    supported_kwargs: tuple[str, ...] = ("query_candidate_component_keys_by_signature_id",),
    row_signals: tuple[str, ...] = ("row_orcid_match",),
    raw_planner_methods: tuple[str, ...] = (
        "from_query_signatures",
        "plan_query_signatures",
        "build_telemetry",
    ),
) -> dict[str, tuple[str, ...]]:
    return {
        "incremental_linking_pair_plan_supported_kwargs": supported_kwargs,
        "incremental_linking_pair_plan_row_signals": row_signals,
        "raw_arrow_query_signature_planner_methods": raw_planner_methods,
    }


def test_load_s2and_rust_extension_prefers_versioned_candidate_on_tie(monkeypatch):
    RustFeaturizer = _make_core_rust_featurizer()

    class ShimModule:
        __version__ = None

    cast(Any, ShimModule).RustFeaturizer = RustFeaturizer

    class NativeModule:
        __version__ = _SUPPORTED_VERSION

    cast(Any, NativeModule).RustFeaturizer = RustFeaturizer

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

    cast(Any, ShimModule).RustFeaturizer = RustFeaturizer

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
        def signature_ids(*args, **kwargs):
            return []

    class Module:
        __version__ = "0.49.0"

    cast(Any, Module).RustFeaturizer = MissingMarkerRustFeaturizer

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert capabilities.extension_importable is True
    assert capabilities.core_runtime_available is False
    assert capabilities.reason.startswith("rust_core_missing_markers:")


def test_detect_rust_runtime_capabilities_rejects_old_version():
    RustFeaturizer = _make_core_rust_featurizer()

    class Module:
        __version__ = "0.49.9"

    cast(Any, Module).RustFeaturizer = RustFeaturizer

    blocked = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert blocked.core_runtime_available is False
    assert blocked.reason.startswith("rust_version_below_minimum:")


def test_detect_rust_runtime_capabilities_rejects_unparseable_version():
    RustFeaturizer = _make_core_rust_featurizer()

    class Module:
        __version__ = "dev-local"

    cast(Any, Module).RustFeaturizer = RustFeaturizer

    blocked = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert blocked.core_runtime_available is False
    assert blocked.reason.startswith("rust_version_unparseable:")


def test_detect_rust_runtime_capabilities_does_not_require_json_ingest_markers():
    class RustFeaturizerWithoutJsonCompat:
        @staticmethod
        def from_arrow_paths(*args, **kwargs):
            return None

        def signature_ids(self):
            return []

        def get_constraints_matrix_indexed(self, *args, **kwargs):
            return []

        def featurize_pairs_matrix_indexed(self, *args, **kwargs):
            return None

        def update_signature_name_counts(self, *args, **kwargs):
            return 0

    class Module:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = RustFeaturizerWithoutJsonCompat

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)

    assert capabilities.core_runtime_available is True
    assert capabilities.reason == "rust_core_available"


def test_detect_rust_runtime_capabilities_reports_incremental_linker_names():
    class NamedRustFeaturizer(_make_core_rust_featurizer()):
        def linker_pair_index_arrays_and_aggregate_stats(self, *args, **kwargs):
            return None

        def linker_pair_index_arrays_constraint_labels(self, *args, **kwargs):
            return None

        def linker_pair_distance_accumulators(self, *args, **kwargs):
            return None

    class NamedRustHybridCentroidRetriever:
        def top_k_hybrid_centroid_pair_plan(self, *args, **kwargs):
            return None

    class NamedRawBlockQueryCandidatePlanner:
        @staticmethod
        def from_query_signatures(*args, **kwargs):
            return None

    class Module:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = NamedRustFeaturizer
        RustHybridCentroidRetriever = NamedRustHybridCentroidRetriever
        RawBlockQueryCandidatePlanner = NamedRawBlockQueryCandidatePlanner

        @staticmethod
        def get_build_info():
            return _pair_plan_build_info()

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)

    assert "hybrid_centroid_retriever_v1" in capabilities.named_capabilities
    assert "indexed_pair_array_featurization_v1" in capabilities.named_capabilities
    assert "incremental_linking_pair_plan_v1" in capabilities.named_capabilities
    assert "incremental_linking_constraint_arrays_v1" in capabilities.named_capabilities
    assert "raw_arrow_query_signature_planner_v1" in capabilities.named_capabilities


def test_detect_rust_runtime_capabilities_rejects_stale_raw_query_signature_planner_abi():
    class NamedRawPlanner:
        @staticmethod
        def from_query_signatures(*args, **kwargs):
            return None

    class Module:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = _make_core_rust_featurizer()
        RawBlockQueryCandidatePlanner = NamedRawPlanner

        @staticmethod
        def get_build_info():
            return _pair_plan_build_info(raw_planner_methods=("from_query_signatures",))

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)

    assert capabilities.core_runtime_available is True
    assert "raw_arrow_query_signature_planner_v1" not in capabilities.named_capabilities


def test_detect_rust_runtime_capabilities_rejects_stale_incremental_pair_plan_abi():
    class NamedRustFeaturizer(_make_core_rust_featurizer()):
        def linker_pair_index_arrays_and_aggregate_stats(self, *args, **kwargs):
            return None

    class NamedRustHybridCentroidRetriever:
        def top_k_hybrid_centroid_pair_plan(self, *args, **kwargs):
            return None

    class Module:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = NamedRustFeaturizer
        RustHybridCentroidRetriever = NamedRustHybridCentroidRetriever

        @staticmethod
        def get_build_info():
            return _pair_plan_build_info(supported_kwargs=())

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)

    assert "hybrid_centroid_retriever_v1" in capabilities.named_capabilities
    assert "indexed_pair_array_featurization_v1" in capabilities.named_capabilities
    assert "incremental_linking_pair_plan_v1" not in capabilities.named_capabilities


def test_detect_rust_runtime_capabilities_requires_pair_plan_orcid_signal_marker():
    class NamedRustFeaturizer(_make_core_rust_featurizer()):
        def linker_pair_index_arrays_and_aggregate_stats(self, *args, **kwargs):
            return None

    class NamedRustHybridCentroidRetriever:
        def top_k_hybrid_centroid_pair_plan(self, *args, **kwargs):
            return None

    class Module:
        __version__ = _SUPPORTED_VERSION
        RustFeaturizer = NamedRustFeaturizer
        RustHybridCentroidRetriever = NamedRustHybridCentroidRetriever

        @staticmethod
        def get_build_info():
            return _pair_plan_build_info(row_signals=())

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)

    assert "hybrid_centroid_retriever_v1" in capabilities.named_capabilities
    assert "indexed_pair_array_featurization_v1" in capabilities.named_capabilities
    assert "incremental_linking_pair_plan_v1" not in capabilities.named_capabilities


def test_rust_get_build_info_contract():
    rust_available, rust_payload = import_s2and_rust(
        required_module_attrs=("get_build_info",),
        prefer_site_packages=True,
    )
    if not rust_available:
        raise pytest.skip.Exception(f"s2and_rust.get_build_info unavailable: {rust_payload!r}")
    s2and_rust = rust_payload

    get_build_info = getattr(s2and_rust, "get_build_info", None)
    if not callable(get_build_info):
        raise pytest.skip.Exception("s2and_rust.get_build_info unavailable")

    info = get_build_info()
    assert isinstance(info, dict)
    for key in ("crate_version", "profile", "debug_assertions", "opt_level", "target"):
        assert key in info
    row_signals = tuple(info.get("incremental_linking_pair_plan_row_signals", ()))
    assert "row_orcid_match" in row_signals
    supported_kwargs = tuple(info.get("incremental_linking_pair_plan_supported_kwargs", ()))
    assert "query_candidate_component_keys_by_signature_id" in supported_kwargs
    if "raw_arrow_query_signature_planner_methods" not in info:
        raise pytest.skip.Exception(
            "installed local s2and_rust extension was built before raw-planner build-info markers"
        )
    raw_planner_methods = tuple(info.get("raw_arrow_query_signature_planner_methods", ()))
    assert "from_query_signatures" in raw_planner_methods
    assert "plan_query_signatures" in raw_planner_methods
    assert "build_telemetry" in raw_planner_methods

from __future__ import annotations

import pytest

import s2and.rust_capabilities as rust_capabilities


def _make_core_rust_featurizer(*, supports_from_dataset_paper_preprocess: bool = False):
    class RustFeaturizer:
        SUPPORTS_FROM_DATASET_PAPER_PREPROCESS = supports_from_dataset_paper_preprocess

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

    return RustFeaturizer


def test_load_s2and_rust_extension_prefers_versioned_candidate_on_tie(monkeypatch):
    RustFeaturizer = _make_core_rust_featurizer()

    class ShimModule:
        __version__ = None

    ShimModule.RustFeaturizer = RustFeaturizer

    class NativeModule:
        __version__ = "0.40.0"

    NativeModule.RustFeaturizer = RustFeaturizer

    def _fake_import_module(name: str):
        if name == "s2and_rust":
            return ShimModule
        if name == "s2and_rust._s2and_rust":
            return NativeModule
        raise ImportError(name)

    monkeypatch.setattr(rust_capabilities.importlib, "import_module", _fake_import_module)
    loaded = rust_capabilities.load_s2and_rust_extension()
    assert loaded is NativeModule


def test_detect_rust_runtime_capabilities_requires_core_markers():
    class MissingMarkerRustFeaturizer:
        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        @staticmethod
        def from_json_paths(*args, **kwargs):
            return None

    class Module:
        __version__ = "0.40.0"

    Module.RustFeaturizer = MissingMarkerRustFeaturizer

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert capabilities.extension_importable is True
    assert capabilities.core_runtime_available is False
    assert capabilities.reason.startswith("rust_core_missing_markers:")


def test_detect_rust_runtime_capabilities_rejects_old_version():
    RustFeaturizer = _make_core_rust_featurizer()

    class Module:
        __version__ = "0.39.9"

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
        __version__ = "0.40.0"

    Module.RustFeaturizer = RustFeaturizer

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert capabilities.core_runtime_available is True
    assert capabilities.from_dataset_paper_preprocess_available is True


def test_rust_get_build_info_contract():
    s2and_rust = pytest.importorskip("s2and_rust")

    get_build_info = getattr(s2and_rust, "get_build_info", None)
    if not callable(get_build_info):
        return

    info = get_build_info()
    assert isinstance(info, dict)
    for key in ("crate_version", "profile", "debug_assertions", "opt_level", "target"):
        assert key in info

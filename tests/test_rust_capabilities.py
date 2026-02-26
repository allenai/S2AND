from __future__ import annotations

import s2and.rust_capabilities as rust_capabilities


def test_load_s2and_rust_extension_prefers_versioned_candidate_on_tie(monkeypatch):
    class RustFeaturizer:
        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        @staticmethod
        def from_json_paths(*args, **kwargs):
            return None

        def signature_ids(self):
            return []

        def featurize_pairs_matrix_indexed(self, *args, **kwargs):
            return None

        def update_signature_name_counts(self, *args, **kwargs):
            return 0

    class ShimModule:
        __version__ = None

    ShimModule.RustFeaturizer = RustFeaturizer

    class NativeModule:
        __version__ = "0.31.0"

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
        __version__ = "0.31.0"

    Module.RustFeaturizer = MissingMarkerRustFeaturizer

    capabilities = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert capabilities.extension_importable is True
    assert capabilities.core_runtime_available is False
    assert capabilities.reason.startswith("rust_core_missing_markers:")


def test_detect_rust_runtime_capabilities_rejects_old_version():
    class RustFeaturizer:
        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        @staticmethod
        def from_json_paths(*args, **kwargs):
            return None

        def featurize_pairs_matrix_indexed(self, *args, **kwargs):
            return None

        def update_signature_name_counts(self, *args, **kwargs):
            return 0

    class Module:
        __version__ = "0.30.9"

    Module.RustFeaturizer = RustFeaturizer

    blocked = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert blocked.core_runtime_available is False
    assert blocked.reason.startswith("rust_version_below_minimum:")


def test_detect_rust_runtime_capabilities_rejects_unparseable_version():
    class RustFeaturizer:
        @staticmethod
        def from_dataset(*args, **kwargs):
            return None

        @staticmethod
        def from_json_paths(*args, **kwargs):
            return None

        def featurize_pairs_matrix_indexed(self, *args, **kwargs):
            return None

        def update_signature_name_counts(self, *args, **kwargs):
            return 0

    class Module:
        __version__ = "dev-local"

    Module.RustFeaturizer = RustFeaturizer

    blocked = rust_capabilities.detect_rust_runtime_capabilities(extension_module=Module)
    assert blocked.core_runtime_available is False
    assert blocked.reason.startswith("rust_version_unparseable:")

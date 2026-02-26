from __future__ import annotations

import pytest

from s2and import runtime


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "S2AND_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    runtime.reset_runtime_warning_state_for_tests()


def test_resolve_backend_unset_auto_falls_back_to_python_when_rust_core_missing(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "_auto_backend_capability_probe",
        lambda: (False, "rust_extension_unavailable"),
    )
    resolution = runtime.resolve_backend(emit_startup_warning=False)
    assert resolution.resolved_backend == "python"
    assert resolution.source == "default"
    assert resolution.requested_backend is None
    assert resolution.capability_reason == "rust_extension_unavailable"


def test_resolve_backend_unset_auto_uses_rust_when_core_capability_available(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "_auto_backend_capability_probe",
        lambda: (True, "rust_core_available"),
    )
    resolution = runtime.resolve_backend(emit_startup_warning=False)
    assert resolution.resolved_backend == "rust"
    assert resolution.source == "default"
    assert resolution.requested_backend is None
    assert resolution.capability_reason == "rust_core_available"


def test_resolve_backend_prefers_explicit_s2and_backend(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "python")
    resolution = runtime.resolve_backend(emit_startup_warning=False)
    assert resolution.resolved_backend == "python"
    assert resolution.source == "S2AND_BACKEND"


def test_resolve_backend_auto_env_uses_capability_probe(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "auto")
    monkeypatch.setattr(
        runtime,
        "_auto_backend_capability_probe",
        lambda: (True, "rust_core_available"),
    )
    resolution = runtime.resolve_backend(emit_startup_warning=False)
    assert resolution.requested_backend == "auto"
    assert resolution.resolved_backend == "rust"
    assert resolution.source == "S2AND_BACKEND"
    assert resolution.capability_reason == "rust_core_available"


def test_resolve_backend_invalid_value_raises(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "invalid")
    with pytest.raises(ValueError, match="Invalid S2AND_BACKEND"):
        runtime.resolve_backend(emit_startup_warning=False)


def test_runtime_context_stage_enablement(monkeypatch: pytest.MonkeyPatch):
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("S2AND_BACKEND", "python")
    python_context = runtime.build_runtime_context("unit_test", emit_startup_warning=False)
    assert python_context.stage_backend("pair_featurization") == "python"
    assert python_context.stage_backend("constraints") == "python"

    monkeypatch.setenv("S2AND_BACKEND", "rust")
    rust_context = runtime.build_runtime_context("unit_test", emit_startup_warning=False)
    assert rust_context.stage_backend("pair_featurization") == "rust"
    assert rust_context.stage_backend("constraints") == "rust"



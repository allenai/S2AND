from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

import s2and
import s2and.runtime as runtime


def test_runtime_backend_routing_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime,
        "load_s2and_rust_extension",
        lambda: pytest.fail("default Python routing must not import Rust"),
    )
    context = runtime.build_runtime_context("unit_test")
    assert context.backend == "python"
    assert runtime.stage_uses_rust(context) is False

    context = runtime.build_runtime_context("unit_test", backend="python")
    assert context.backend == "python"

    sentinel = object()
    monkeypatch.setattr(runtime, "load_s2and_rust_extension", lambda: sentinel)
    context = runtime.build_runtime_context("unit_test", backend="rust")
    assert context.backend == "rust"
    assert runtime.stage_uses_rust(context) is True

    with pytest.raises(ValueError, match="expected 'python' or 'rust'"):
        runtime.build_runtime_context("unit_test", backend="auto")  # type: ignore[arg-type]


def test_importing_model_with_python_backend_does_not_load_rust_extension() -> None:
    script = """
import s2and.runtime as runtime

def fail():
    raise AssertionError("Python model import must not load the Rust extension")

runtime.load_s2and_rust_extension = fail
import s2and.model
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_load_rust_extension_validates_package_and_version() -> None:
    exact = SimpleNamespace(__version__=s2and.__version__)
    assert runtime.load_s2and_rust_extension(import_module=lambda _name: exact) is exact

    mismatched = SimpleNamespace(__version__="1.0.1")
    with pytest.raises(RuntimeError, match="does not match the Python package"):
        runtime.load_s2and_rust_extension(import_module=lambda _name: mismatched)

    def missing(_name: str) -> object:
        raise ModuleNotFoundError("missing", name="s2and._s2and_rust")

    with pytest.raises(RuntimeError, match="not importable"):
        runtime.load_s2and_rust_extension(import_module=missing)

    def broken(_name: str) -> object:
        raise ModuleNotFoundError("missing dependency", name="native_dependency")

    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        runtime.load_s2and_rust_extension(import_module=broken)


def test_load_rust_extension_matches_cargo_prerelease_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    # The extension reports its Cargo version; the wheel and s2and use the PEP 440 spelling.
    monkeypatch.setattr(runtime, "__version__", "1.0.0b1")
    beta = SimpleNamespace(__version__="1.0.0-beta.1")
    assert runtime.load_s2and_rust_extension(import_module=lambda _name: beta) is beta

    other_beta = SimpleNamespace(__version__="1.0.0-beta.2")
    with pytest.raises(RuntimeError, match="does not match the Python package"):
        runtime.load_s2and_rust_extension(import_module=lambda _name: other_beta)


def test_dataset_stage_routing_requires_arrow_only_for_rust() -> None:
    rust_context = runtime.RuntimeContext(
        operation="unit_test",
        backend="rust",
        run_id="test-run",
    )

    assert runtime.dataset_stage_uses_rust(rust_context, SimpleNamespace(arrow_dataset=object())) is True
    with pytest.raises(RuntimeError, match="dataset has no ArrowDataset"):
        runtime.dataset_stage_uses_rust(rust_context, SimpleNamespace(arrow_dataset=None))

    python_context = runtime.RuntimeContext(
        operation="unit_test",
        backend="python",
        run_id="test-run",
    )

    assert runtime.dataset_stage_uses_rust(python_context, SimpleNamespace(arrow_dataset=None)) is False

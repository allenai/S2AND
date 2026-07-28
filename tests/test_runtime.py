from __future__ import annotations

from types import SimpleNamespace

import pytest

import s2and.runtime as runtime


@pytest.fixture(autouse=True)
def _clear_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("S2AND_BACKEND", raising=False)


def test_default_backend_is_python_without_rust_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime,
        "load_s2and_rust_extension",
        lambda: pytest.fail("default Python routing must not import Rust"),
    )

    context = runtime.build_runtime_context("unit_test")

    assert context.backend == "python"
    assert runtime.stage_uses_rust(context) is False


def test_backend_argument_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S2AND_BACKEND", "rust")
    monkeypatch.setattr(
        runtime,
        "load_s2and_rust_extension",
        lambda: pytest.fail("explicit Python routing must not import Rust"),
    )

    context = runtime.build_runtime_context("unit_test", backend="python")

    assert context.backend == "python"


def test_explicit_rust_checks_pinned_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(runtime, "load_s2and_rust_extension", lambda: sentinel)

    context = runtime.build_runtime_context("unit_test", backend="rust")

    assert context.backend == "rust"
    assert runtime.stage_uses_rust(context) is True


@pytest.mark.parametrize("value", ["auto", "", "gpu"])
def test_backend_rejects_every_value_except_python_or_rust(value: str) -> None:
    with pytest.raises(ValueError, match="expected 'python' or 'rust'"):
        runtime.build_runtime_context("unit_test", backend=value)  # type: ignore[arg-type]


def test_load_rust_extension_requires_exact_version() -> None:
    exact = SimpleNamespace(__version__=runtime.REQUIRED_RUST_EXTENSION_VERSION)
    assert runtime.load_s2and_rust_extension(import_module=lambda _name: exact) is exact

    mismatched = SimpleNamespace(__version__="1.0.1")
    with pytest.raises(RuntimeError, match="does not match the pinned dependency"):
        runtime.load_s2and_rust_extension(import_module=lambda _name: mismatched)


def test_load_rust_extension_reports_missing_package() -> None:
    def missing(_name: str) -> object:
        raise ModuleNotFoundError("missing", name="s2and_rust")

    with pytest.raises(RuntimeError, match="not importable"):
        runtime.load_s2and_rust_extension(import_module=missing)


def test_load_rust_extension_does_not_hide_broken_transitive_import() -> None:
    def broken(_name: str) -> object:
        raise ModuleNotFoundError("missing dependency", name="native_dependency")

    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        runtime.load_s2and_rust_extension(import_module=broken)


def test_dataset_rust_stage_requires_arrow_dataset() -> None:
    context = runtime.RuntimeContext(
        operation="unit_test",
        backend="rust",
        run_id="test-run",
    )

    assert runtime.dataset_stage_uses_rust(context, SimpleNamespace(arrow_dataset=object())) is True
    with pytest.raises(RuntimeError, match="dataset has no ArrowDataset"):
        runtime.dataset_stage_uses_rust(context, SimpleNamespace(arrow_dataset=None))


def test_dataset_python_stage_never_requires_arrow_dataset() -> None:
    context = runtime.RuntimeContext(
        operation="unit_test",
        backend="python",
        run_id="test-run",
    )

    assert runtime.dataset_stage_uses_rust(context, SimpleNamespace(arrow_dataset=None)) is False

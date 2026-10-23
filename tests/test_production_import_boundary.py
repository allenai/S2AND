"""Verify inference and evaluation imports do not initialize plotting or training."""

import subprocess
import sys
import textwrap

import pytest


def _run_in_fresh_python(script: str) -> None:
    """Execute import assertions without pollution from pytest collection."""
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_does_not_load_plotting_or_training() -> None:
    """Check import boundaries in a fresh interpreter, outside pytest's imports.

    test_production_model.py::test_complete_bundle_native_inference_does_not_load_training_libraries
    checks the s2and.production_model boundary.
    """
    script = textwrap.dedent(
        """
        import importlib
        import sys
        importlib.import_module("s2and.eval")
        forbidden = ["seaborn", "matplotlib.pyplot", "hyperopt"]
        loaded = [name for name in forbidden if name in sys.modules]
        assert not loaded, f"Unexpected import dependencies: {loaded}"
        """
    )
    _run_in_fresh_python(script)


@pytest.mark.parametrize("module", ["s2and.production_model", "s2and.eval"])
def test_import_preserves_caller_plotting_configuration(module: str) -> None:
    _run_in_fresh_python(
        f"""
        import importlib
        import matplotlib

        matplotlib.rcParams["axes.labelsize"] = 13.0
        matplotlib.rcParams["axes.facecolor"] = "ivory"
        before = dict.copy(matplotlib.rcParams)
        importlib.import_module({module!r})
        after = dict.copy(matplotlib.rcParams)
        changed = {{key for key in before if before[key] != after[key]}}
        assert not changed, f"Import changed Matplotlib configuration: {{changed}}"
        """
    )


def test_incremental_linking_modules_import_without_training_or_model_dependencies() -> None:
    """Resolve real transitive imports, including relative and dynamic imports."""
    _run_in_fresh_python(
        """
        import importlib
        import importlib.abc
        import pkgutil
        import sys

        class RuntimeBoundary(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                forbidden = ("scripts", "s2and.model", "s2and.incremental_linking_training")
                if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
                    raise AssertionError(f"Runtime imported training dependency: {fullname}")

        sys.meta_path.insert(0, RuntimeBoundary())
        package = importlib.import_module("s2and.incremental_linking")
        modules = list(pkgutil.walk_packages(package.__path__, package.__name__ + "."))
        assert modules, "No incremental linking modules were discovered"
        for module in modules:
            importlib.import_module(module.name)
        """
    )


def test_arrow_string_array_matches_pyarrow_without_importing_pandas() -> None:
    """pa.array imports pandas, so request-path string arrays are built from buffers."""
    _run_in_fresh_python(
        """
        import sys
        from s2and.arrow_inputs import arrow_string_array

        values = ["a", "caf\u00e9", "", "xyz"]
        built = arrow_string_array(values)
        built.validate(full=True)
        empty = arrow_string_array([])
        assert "pandas" not in sys.modules, "arrow_string_array imported pandas"

        import pyarrow as pa

        assert built.equals(pa.array(values, type=pa.string()))
        assert empty.equals(pa.array([], type=pa.string()))
        """
    )

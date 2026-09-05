"""Verify inference and evaluation imports do not initialize plotting or training."""

import subprocess
import sys
import textwrap

import pytest


def _run_in_fresh_python(script: str) -> None:
    """Execute import assertions without pollution from pytest collection."""
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", ["s2and.production_model", "s2and.eval"])
def test_import_does_not_load_plotting_or_training(module: str) -> None:
    """Check import boundaries in a fresh interpreter, outside pytest's imports."""
    script = textwrap.dedent(
        f"""
        import importlib
        import sys
        importlib.import_module({module!r})
        forbidden = ["seaborn", "matplotlib.pyplot", "hyperopt"]
        if {module!r} == "s2and.production_model":
            forbidden.append("s2and.eval")
        loaded = [name for name in forbidden if name in sys.modules]
        assert not loaded, f"Unexpected import dependencies: {{loaded}}"
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

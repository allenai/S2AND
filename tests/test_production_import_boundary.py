"""Verify inference and evaluation imports do not initialize plotting or training."""

import subprocess
import sys
import textwrap


def _run_in_fresh_python(script: str) -> None:
    """Execute import assertions without pollution from pytest collection."""
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_request_and_eval_imports_skip_pandas_plotting_and_training() -> None:
    """Check import boundaries in a fresh interpreter, outside pytest's imports.

    Share one interpreter between checks: each fresh interpreter costs seconds.
    test_production_model.py::test_complete_bundle_native_inference_does_not_load_training_libraries
    checks the s2and.production_model boundary.
    """
    _run_in_fresh_python(
        """
        import importlib
        import sys

        # pa.array imports pandas, so request-path string arrays are built from buffers.
        from s2and.arrow_inputs import arrow_string_array

        values = ["a", "café", "", "xyz"]
        built = arrow_string_array(values)
        built.validate(full=True)
        empty = arrow_string_array([])
        assert "pandas" not in sys.modules, "arrow_string_array imported pandas"

        # Import matplotlib only after this check: a loaded matplotlib lets dependencies import pyplot.
        importlib.import_module("s2and.eval")
        forbidden = ["seaborn", "matplotlib.pyplot", "hyperopt"]
        loaded = [name for name in forbidden if name in sys.modules]
        assert not loaded, f"Unexpected import dependencies: {loaded}"

        import pyarrow as pa

        assert built.equals(pa.array(values, type=pa.string()))
        assert empty.equals(pa.array([], type=pa.string()))
        """
    )


def test_imports_preserve_caller_plotting_configuration() -> None:
    _run_in_fresh_python(
        """
        import importlib
        import matplotlib

        matplotlib.rcParams["axes.labelsize"] = 13.0
        matplotlib.rcParams["axes.facecolor"] = "ivory"
        before = dict.copy(matplotlib.rcParams)
        for module in ("s2and.production_model", "s2and.eval"):
            importlib.import_module(module)
            after = dict.copy(matplotlib.rcParams)
            changed = {key for key in before if before[key] != after[key]}
            assert not changed, f"{module} import changed Matplotlib configuration: {changed}"
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


def test_reproduction_scripts_import_without_optional_extras() -> None:
    """docs/reproduce.md runs these scripts after a plain `uv sync`, without extras."""
    _run_in_fresh_python(
        """
        import importlib
        import importlib.abc
        import sys

        class NoExtras(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"orjson", "boto3", "duckdb", "ijson"}:
                    raise ModuleNotFoundError(f"optional extra imported: {fullname}", name=fullname)

        sys.meta_path.insert(0, NoExtras())
        for module in (
            "scripts.production.model.train_pairwise",
            "scripts.production.model.release_pairwise",
            "scripts.production.model.train_linker_and_finalize",
            "scripts.verification.evaluate_incremental_gold",
            "scripts.eps_sweep.apply_reviewed_eps",
        ):
            importlib.import_module(module)
        """
    )

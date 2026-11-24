"""Verify inference, evaluation, and reproduction imports stay isolated in a clean interpreter."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_clean_interpreter_import_boundaries() -> None:
    """Verify runtime, Arrow, evaluation, plotting, and optional-extra boundaries in one clean Python process.

    Stages are ordered so earlier imports never mask later sys.modules checks:
    1. incremental_linking modules must not import scripts, s2and.model, or training packages;
    2. arrow_string_array must not import pandas;
    3. s2and.eval must not eagerly load seaborn, matplotlib.pyplot, or hyperopt;
    4. importing s2and.production_model and reloading s2and.eval must preserve caller matplotlib.rcParams;
    5. reproduction CLI scripts must import without optional extras (orjson, boto3, duckdb, ijson).
    """
    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import pkgutil
        import sys

        # 1. Runtime incremental_linking boundary
        class RuntimeBoundary(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                forbidden = ("scripts", "s2and.model", "s2and.incremental_linking_training")
                if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
                    raise AssertionError(f"Runtime imported training dependency: {fullname}")

        boundary = RuntimeBoundary()
        sys.meta_path.insert(0, boundary)
        package = importlib.import_module("s2and.incremental_linking")
        modules = list(pkgutil.walk_packages(package.__path__, package.__name__ + "."))
        assert modules, "No incremental linking modules were discovered"
        for module in modules:
            importlib.import_module(module.name)
        sys.meta_path.remove(boundary)

        # 2. Request-path string array skips pandas
        from s2and.arrow_inputs import arrow_string_array

        values = ["a", "café", "", "xyz"]
        built = arrow_string_array(values)
        built.validate(full=True)
        empty = arrow_string_array([])
        assert "pandas" not in sys.modules, "arrow_string_array imported pandas"

        # 3. Evaluation import skips plotting & hyperopt before matplotlib is loaded
        assert "matplotlib" not in sys.modules
        eval_mod = importlib.import_module("s2and.eval")
        forbidden_eval = ["seaborn", "matplotlib.pyplot", "hyperopt"]
        loaded = [name for name in forbidden_eval if name in sys.modules]
        assert not loaded, f"Unexpected import dependencies: {loaded}"

        import pyarrow as pa

        assert built.equals(pa.array(values, type=pa.string()))
        assert empty.equals(pa.array([], type=pa.string()))

        # 4. Importing production_model and eval preserves caller matplotlib.rcParams
        import matplotlib

        matplotlib.rcParams["axes.labelsize"] = 13.0
        matplotlib.rcParams["axes.facecolor"] = "ivory"
        before = dict.copy(matplotlib.rcParams)
        importlib.import_module("s2and.production_model")
        importlib.reload(eval_mod)
        after = dict.copy(matplotlib.rcParams)
        changed = {key for key in before if before[key] != after[key]}
        assert not changed, f"Import changed Matplotlib configuration: {changed}"

        # 5. Reproduction scripts import without optional extras
        class NoExtras(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"orjson", "boto3", "duckdb", "ijson"}:
                    raise ModuleNotFoundError(f"optional extra imported: {fullname}", name=fullname)

        sys.meta_path.insert(0, NoExtras())
        for module_name in (
            "scripts.pairwise.train_pairwise",
            "scripts.pairwise.release_pairwise",
            "scripts.linker.train_linker_and_finalize",
            "scripts.release.verification.evaluate_incremental_gold",
            "scripts.eps.apply_reviewed_eps",
        ):
            importlib.import_module(module_name)
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

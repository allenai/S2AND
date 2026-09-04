"""Verify inference and evaluation imports do not initialize plotting or training."""

import subprocess
import sys
import textwrap

import pytest


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
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fixed_search_space_construction_does_not_load_hyperopt() -> None:
    """An explicitly configured inference model does not need the optimizer."""
    script = textwrap.dedent(
        """
        import sys
        from lightgbm import LGBMClassifier
        from s2and.featurizer import FeaturizationInfo
        from s2and.model import Clusterer

        clusterer = Clusterer(FeaturizationInfo(), LGBMClassifier(), search_space={})
        assert clusterer.search_space == {}
        assert "hyperopt" not in sys.modules
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", ["s2and.production_model", "s2and.eval"])
def test_import_preserves_matplotlib_configuration(module: str) -> None:
    """Preserve caller-owned plotting configuration, including custom settings."""
    script = textwrap.dedent(
        f"""
        import importlib
        import matplotlib

        matplotlib.rcParams["axes.labelsize"] = 13.0
        matplotlib.rcParams["axes.facecolor"] = "ivory"
        before = dict.copy(matplotlib.rcParams)
        importlib.import_module({module!r})
        after = dict.copy(matplotlib.rcParams)
        changed = {{key: (before[key], after[key]) for key in before if before[key] != after[key]}}
        assert not changed, f"Import changed Matplotlib configuration: {{changed}}"
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

from __future__ import annotations

import json
from pathlib import Path

import pytest

from s2and import plotting_utils


def test_plot_facets_uses_explicit_names_and_writes_inside_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int, Path]] = []

    def fake_plot_box(
        _s2and_performance,
        _s2_performance,
        figs_path,
        title,
        total_bins=5,
    ) -> None:
        calls.append((title, total_bins, Path(figs_path)))

    monkeypatch.setattr(plotting_utils, "plot_box", fake_plot_box)
    s2and_facets = {
        "homonymity": {"0.5": [0.8]},
        "year": {"2020": [0.9]},
    }
    s2_facets = {
        "year": {"2020": [0.7]},
        "homonymity": {"0.5": [0.6]},
    }

    plotting_utils.plot_facets(s2and_facets, s2_facets, tmp_path / "facets")

    output_dir = tmp_path / "facets"
    assert calls == [("year", 4, output_dir), ("homonymity", 10, output_dir)]
    assert json.loads((output_dir / "year_dict_pred.json").read_text(encoding="utf-8")) == {"2020": [0.9]}
    assert json.loads((output_dir / "homonymity_dict_s2.json").read_text(encoding="utf-8")) == {"0.5": [0.6]}


def test_plot_facets_requires_matching_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must contain the same names"):
        plotting_utils.plot_facets(
            {"year": {"2020": [0.9]}},
            {"homonymity": {"0.5": [0.6]}},
            tmp_path,
        )


def test_plot_facets_rejects_unknown_names(tmp_path: Path) -> None:
    facets = {"invented facet": {"1": [0.5]}}

    with pytest.raises(ValueError, match="Unknown facet names"):
        plotting_utils.plot_facets(facets, facets, tmp_path)

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FACET_BIN_COUNTS = {
    "gender": 0,
    "ethnicity": 0,
    "number of authors": 8,
    "year": 4,
    "block size": 8,
    "cluster size": 8,
    "homonymity": 10,
    "synonymity": 10,
}


def plot_box(
    s2and_performance: Mapping[Any, Sequence[Any]],
    s2_performance: Mapping[Any, Sequence[Any]],
    figs_path: str | Path,
    title: str,
    total_bins: int = 5,
) -> None:
    b3 = []
    keylist = []
    model = []

    if title == "ethnicity":
        sns.set(rc={"figure.figsize": (15, 7)})
    else:
        sns.set(rc={"figure.figsize": (12, 6)})

    for facet, f1 in s2and_performance.items():
        if title == "gender":
            if facet == "-":
                continue

        if title == "year":
            if int(facet) == 0:
                continue

        for _f1 in f1:
            if title != "gender" and title != "ethnicity":
                keylist.append(float(facet))
            else:
                keylist.append(facet)
            b3.append(_f1)
            model.append("S2AND")

        for _f1 in s2_performance[facet]:
            if title != "gender" and title != "ethnicity":
                keylist.append(float(facet))
            else:
                keylist.append(facet)
            b3.append(_f1)
            model.append("S2")

    if title == "year":
        bins = pd.IntervalIndex.from_tuples(
            [
                (int(min(keylist)), 1960),
                (1960, 1980),
                (1980, 1990),
                (1990, 2000),
                (2000, 2005),
                (2005, 2010),
                (2010, 2015),
                (2015, 2020),
            ]
        )
    elif title == "number of authors":
        bins = pd.IntervalIndex.from_tuples(
            [
                (0, 1),
                (1, 2),
                (2, 5),
                (5, 10),
                (10, 15),
                (15, 25),
                (25, 50),
                (50, 100),
                (100, 1000),
            ]
        )
    elif title == "block size":
        bins = pd.IntervalIndex.from_tuples(
            [(0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 200), (200, 400), (400, 800), (800, 3000)]
        )
    elif title == "cluster size":
        bins = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 100), (100, 800)])
    elif total_bins > 0:
        bins = np.linspace(
            min(keylist), max(keylist), total_bins + 1
        )  # need a + 1 because of how bins interacts with boxplot

    df = pd.DataFrame({"X": keylist, "Y": b3, "Model": model})
    if total_bins > 0:
        data_cut = pd.cut(df.X, bins)
        df["group"] = data_cut
    else:
        df["group"] = keylist

    ax = sns.boxplot(
        x="group",
        y="Y",
        hue="Model",
        data=df,
        showmeans=True,
        meanprops={"marker": "s", "markerfacecolor": "white", "markeredgecolor": "blue"},
    )

    if title != "gender" and title != "ethnicity":
        plt.xlabel(title, fontsize=15)
    else:
        plt.xlabel("", fontsize=15)

    plt.ylabel("B3 F1", fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], loc="lower left")
    plt.savefig(Path(figs_path) / f"{title}_facet.png", bbox_inches="tight")
    plt.clf()
    plt.close()


def plot_facets(
    s2and_facets: Mapping[str, Mapping[Any, Sequence[Any]]],
    s2_facets: Mapping[str, Mapping[Any, Sequence[Any]]],
    figs_path: str | Path,
) -> None:
    """Write facet score payloads and comparison plots for explicitly named facets."""

    s2and_names = set(s2and_facets)
    s2_names = set(s2_facets)
    if s2and_names != s2_names:
        raise ValueError(
            "S2AND and S2 facet mappings must contain the same names: "
            f"only_s2and={sorted(s2and_names - s2_names)}, only_s2={sorted(s2_names - s2and_names)}"
        )

    unknown_names = s2and_names - FACET_BIN_COUNTS.keys()
    if unknown_names:
        raise ValueError(f"Unknown facet names: {sorted(unknown_names)}")

    figs_dir = Path(figs_path)
    figs_dir.mkdir(parents=True, exist_ok=True)
    for plot_name, bin_size in FACET_BIN_COUNTS.items():
        if plot_name not in s2and_facets:
            continue
        s2and_facet = s2and_facets[plot_name]
        s2_facet = s2_facets[plot_name]
        with (figs_dir / f"{plot_name}_dict_pred.json").open("w", encoding="utf-8") as outfile:
            json.dump(dict(s2and_facet), outfile, indent=4)
        with (figs_dir / f"{plot_name}_dict_s2.json").open("w", encoding="utf-8") as outfile:
            json.dump(dict(s2_facet), outfile, indent=4)

        plot_box(s2and_facet, s2_facet, figs_dir, plot_name, total_bins=bin_size)

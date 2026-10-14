"""Plot giant-block safe-action curves for manual EPS review."""

from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eps_sweep.common import read_sweep_curve  # noqa: E402

HISTORICAL_ORDER = ("j_smith", "s_gupta", "a_khan", "a_silva", "s_park", "s_lee", "h_wang")
WEIGHTS = ("weight_query_class_balanced", "weight_query_label_balanced")


def _format_eps_label(eps: float) -> str:
    """Show the cut value beside each point."""
    return f"{eps:.4g}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit sweep and plot paths."""
    parser = argparse.ArgumentParser(description="Plot positive recall vs negative separation for EPS sweeps.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="EPS sweep tradeoff across giant datasets\nUpper-right is better")
    parser.add_argument("--reference-eps", type=float, default=None)
    parser.add_argument("--weight", choices=WEIGHTS, default="weight_query_label_balanced")
    return parser.parse_args(argv)


def _load_q_class(root: Path, dataset: str, weight: str = "weight_query_class_balanced") -> pd.DataFrame:
    """Reject a partial review plot when a requested source lacks results."""
    plan, q_class = read_sweep_curve(root, dataset, "safe-action-proxy", weight=weight)
    if q_class.empty:
        raise ValueError(f"No {weight} EPS results for dataset={dataset}")
    columns = ["eps", "positive_recall", "negative_separation", "accuracy"]
    if q_class[columns].isna().any().any() or q_class["eps"].duplicated().any():
        raise ValueError(f"Incomplete or duplicate EPS results for dataset={dataset}")
    if not q_class["accuracy"].map(lambda value: math.isfinite(value) and 0 <= value <= 1).all():
        raise ValueError(f"Invalid EPS accuracy for dataset={dataset}")
    q_class["dataset"] = dataset
    q_class.attrs["model_manifest_sha256"] = plan["model_manifest_sha256"]
    q_class.attrs["limited_run"] = plan["limited_run"]
    return q_class.sort_values("eps")


def _best_accuracy_point(curve: pd.DataFrame) -> pd.Series:
    """Use the lowest EPS when tested accuracy values have a tie."""
    return curve.sort_values(["accuracy", "eps"], ascending=[False, True]).iloc[0]


def plot(args: argparse.Namespace) -> Path:
    """Write the original recall/separation view without an automatic EPS choice."""
    requested = list(dict.fromkeys(args.datasets))
    datasets = [dataset for dataset in HISTORICAL_ORDER if dataset in requested]
    datasets.extend(dataset for dataset in requested if dataset not in HISTORICAL_ORDER)
    frames = [_load_q_class(args.root, dataset, args.weight) for dataset in datasets]
    if len({frame.attrs["model_manifest_sha256"] for frame in frames}) != 1:
        raise ValueError("All review curves must use the same model")
    limited = any(frame.attrs["limited_run"] for frame in frames)
    combined = pd.concat(frames, ignore_index=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    label_dataset = "j_smith" if "j_smith" in datasets else datasets[0]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=170)
    for index, dataset in enumerate(datasets):
        q = combined[combined["dataset"].eq(dataset)]
        color_index = HISTORICAL_ORDER.index(dataset) if dataset in HISTORICAL_ORDER else index
        color = color_cycle[color_index % len(color_cycle)]
        ax.plot(
            q["negative_separation"],
            q["positive_recall"],
            marker="o",
            linewidth=1.8,
            markersize=5.2,
            label=dataset,
            color=color,
            alpha=0.9,
        )
        if dataset == label_dataset:
            for point_index, row in q.reset_index(drop=True).iterrows():
                eps = float(row["eps"])
                reference = args.reference_eps is not None and math.isclose(eps, args.reference_eps, abs_tol=1e-10)
                if not reference and not math.isclose(eps * 20, round(eps * 20), abs_tol=1e-10):
                    continue
                offset_y = 4 if point_index % 2 == 0 else -9
                ax.annotate(
                    _format_eps_label(eps),
                    (row["negative_separation"], row["positive_recall"]),
                    xytext=(6, offset_y),
                    textcoords="offset points",
                    fontsize=8,
                    color="#333333",
                )
        best = _best_accuracy_point(q)
        ax.scatter(
            [best["negative_separation"]],
            [best["positive_recall"]],
            marker="*",
            s=220,
            color=color,
            edgecolor="black",
            linewidth=0.9,
            zorder=6,
        )
        default = (
            q[q["eps"].round(10).eq(round(args.reference_eps, 10))] if args.reference_eps is not None else q.iloc[:0]
        )
        if not default.empty:
            row = default.iloc[0]
            ax.scatter(
                [row["negative_separation"]],
                [row["positive_recall"]],
                s=105,
                facecolors="none",
                edgecolor=color,
                linewidth=1.6,
                zorder=5,
            )

    ax.set_title(args.title + (" (bounded smoke only)" if limited else ""))
    ax.set_xlabel("Negative separation")
    ax.set_ylabel("Positive recall")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(float(combined["negative_separation"].min()) - 0.025, 1.015)
    ax.set_ylim(max(0, float(combined["positive_recall"].min()) - 0.035), 1)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", linestyle="none", color="gray", markeredgecolor="black", markersize=13))
    labels.append("Best tested accuracy")
    if args.reference_eps is not None:
        handles.append(
            Line2D([], [], marker="o", linestyle="none", color="black", markerfacecolor="none", markersize=9)
        )
        labels.append(f"Current EPS {_format_eps_label(args.reference_eps)}")
    ax.legend(handles, labels, fontsize=9, ncols=2, loc="lower right")
    fig.text(
        0.075,
        0.012,
        f"Rows: eps_overall.tsv; scope=overall/all; weight={args.weight}. "
        f"EPS labels at 0.05 steps: {label_dataset}. Stars use diagnostic accuracy. Within-subblock pairs only.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    return args.output.resolve()


def main(argv: Sequence[str] | None = None) -> None:
    """Print the path to the saved plot."""
    print(plot(parse_args(argv)))


if __name__ == "__main__":
    main()

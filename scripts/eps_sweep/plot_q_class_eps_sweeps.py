"""Plot giant-block safe-action curves for manual EPS review."""

from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eps_sweep.common import read_sweep_curve  # noqa: E402


def _format_eps_label(eps: float) -> str:
    """Show the cut value beside each point."""
    return f"{eps:.4g}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit sweep and plot paths."""
    parser = argparse.ArgumentParser(description="Plot q-class positive recall vs negative separation for EPS sweeps.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="EPS sweep: q-class positive recall vs negative separation")
    parser.add_argument("--reference-eps", type=float, default=None)
    return parser.parse_args(argv)


def _load_q_class(root: Path, dataset: str) -> pd.DataFrame:
    """Reject a partial review plot when a requested source lacks results."""
    plan, q_class = read_sweep_curve(root, dataset, "safe-action-proxy")
    if q_class.empty:
        raise ValueError(f"No query-class-balanced EPS results for dataset={dataset}")
    columns = ["eps", "positive_recall", "negative_separation"]
    if q_class[columns].isna().any().any() or q_class["eps"].duplicated().any():
        raise ValueError(f"Incomplete or duplicate EPS results for dataset={dataset}")
    q_class["dataset"] = dataset
    q_class.attrs["model_manifest_sha256"] = plan["model_manifest_sha256"]
    q_class.attrs["limited_run"] = plan["limited_run"]
    return q_class.sort_values("eps")


def plot(args: argparse.Namespace) -> Path:
    """Write the original recall/separation view without an automatic EPS choice."""
    datasets = list(dict.fromkeys(args.datasets))
    frames = [_load_q_class(args.root, dataset) for dataset in datasets]
    if len({frame.attrs["model_manifest_sha256"] for frame in frames}) != 1:
        raise ValueError("All review curves must use the same model")
    limited = any(frame.attrs["limited_run"] for frame in frames)
    combined = pd.concat(frames, ignore_index=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=170)
    for index, dataset in enumerate(datasets):
        q = combined[combined["dataset"].eq(dataset)]
        color = color_cycle[index % len(color_cycle)]
        marker = markers[index % len(markers)]
        ax.plot(
            q["negative_separation"],
            q["positive_recall"],
            marker=marker,
            linewidth=1.8,
            markersize=5.2,
            label=dataset,
            color=color,
            alpha=0.9,
        )
        for point_index, row in q.reset_index(drop=True).iterrows():
            offset_y = 4 if (point_index + index) % 2 == 0 else -7
            ax.annotate(
                _format_eps_label(float(row["eps"])),
                (row["negative_separation"], row["positive_recall"]),
                xytext=(4, offset_y),
                textcoords="offset points",
                fontsize=5.4,
                color=color,
                alpha=0.85,
            )
        default = (
            q[q["eps"].round(10).eq(round(args.reference_eps, 10))] if args.reference_eps is not None else q.iloc[:0]
        )
        if not default.empty:
            row = default.iloc[0]
            ax.scatter(
                [row["negative_separation"]],
                [row["positive_recall"]],
                s=85,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

    ax.set_title(args.title + (" (bounded smoke only)" if limited else ""))
    ax.set_xlabel("Negative separation")
    ax.set_ylabel("Positive recall")
    ax.grid(True, alpha=0.25)
    ax.legend(title="block", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    return args.output.resolve()


def main(argv: Sequence[str] | None = None) -> None:
    """Print the path to the saved plot."""
    print(plot(parse_args(argv)))


if __name__ == "__main__":
    main()

"""Plot ORCID-masked giant-block B3 curves for manual EPS review."""

from __future__ import annotations

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


def plot(root: Path, datasets: Sequence[str], output: Path, reference_eps: float | None = None) -> Path:
    """Show every requested dataset and reject missing or incomplete curves."""
    frames: list[pd.DataFrame] = []
    model_hashes: set[str] = set()
    limited = False
    for dataset in dict.fromkeys(datasets):
        plan, frame = read_sweep_curve(root, dataset, "orcid-b3")
        model_hashes.add(plan["model_manifest_sha256"])
        limited = limited or plan["limited_run"]
        if frame.empty or frame[["eps", "precision", "recall", "f1"]].isna().any().any():
            raise ValueError(f"Incomplete B3 curve for dataset={dataset}")
        if frame["eps"].duplicated().any():
            raise ValueError(f"Duplicate EPS values for dataset={dataset}")
        frames.append(frame.assign(dataset=dataset).sort_values("eps"))
    if not frames:
        raise ValueError("At least one dataset is required")
    if len(model_hashes) != 1:
        raise ValueError("All review curves must use the same model")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    for axis, metric in zip(axes, ["precision", "recall", "f1"], strict=True):
        for frame in frames:
            axis.plot(frame["eps"], frame[metric], marker="o", markersize=3, label=str(frame["dataset"].iloc[0]))
        if reference_eps is not None:
            axis.axvline(reference_eps, color="black", linestyle="--", alpha=0.5)
        axis.set_xlabel("EPS")
        axis.set_ylabel(f"B3 {metric}")
        axis.set_ylim(bottom=max(0, axis.get_ylim()[0]), top=min(1, axis.get_ylim()[1]))
        axis.grid(True, alpha=0.25)
    axes[-1].legend(title="Dataset", fontsize=8)
    fig.suptitle(
        "Giant-block EPS review: ORCID references hidden from prediction" + (" (bounded smoke only)" if limited else "")
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output.resolve()


def main(argv: Sequence[str] | None = None) -> None:
    """Print the saved plot path without an automatic EPS choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-eps", type=float, default=None)
    args = parser.parse_args(argv)
    print(plot(args.root, args.datasets, args.output, args.reference_eps))


if __name__ == "__main__":
    main()

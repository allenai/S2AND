from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from s2and._sha256 import sha256_file
from scripts.eps_sweep import plot_q_class_eps_sweeps as tradeoff


def _write_curve(root: Path) -> Path:
    dataset = root / "j_smith"
    dataset.mkdir(parents=True)
    plan_path = dataset / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "score_mode": "safe-action-proxy",
                "eps_values": [0.55, 0.65],
                "model_manifest_sha256": "a" * 64,
                "limited_run": False,
            }
        ),
        encoding="utf-8",
    )
    scores = dataset / "eps_overall.tsv"
    pd.DataFrame(
        {
            "eps": [0.55, 0.65, 0.55, 0.65, 0.55],
            "weight": [
                "weight_query_class_balanced",
                "weight_query_class_balanced",
                "weight_query_label_balanced",
                "weight_query_label_balanced",
                "weight_query_label_balanced",
            ],
            "scope_type": ["overall", "overall", "overall", "overall", "table"],
            "scope_value": ["all", "all", "all", "all", "other"],
            "positive_recall": [0.8, 1, 0.7, 0.9, 1],
            "negative_separation": [1, 0.8, 0.95, 0.85, 1],
            "accuracy": [0.9, 0.88, 0.8, 0.87, 1],
        },
    ).to_csv(scores, sep="\t", index=False)
    (dataset / "summary.json").write_text(
        json.dumps({"plan_sha256": sha256_file(plan_path), "score_sha256": sha256_file(scores)}), encoding="utf-8"
    )
    return dataset


def test_star_uses_requested_weights_and_overall_scope(tmp_path: Path) -> None:
    _write_curve(tmp_path)
    original = tradeoff._load_q_class(tmp_path, "j_smith", "weight_query_label_balanced")
    previous = tradeoff._load_q_class(tmp_path, "j_smith")
    assert tradeoff._best_accuracy_point(original)["eps"] == 0.65
    assert tradeoff._best_accuracy_point(previous)["eps"] == 0.55
    assert original["positive_recall"].tolist() == [0.7, 0.9]
    tied = original.assign(accuracy=0.8).iloc[::-1]
    assert tradeoff._best_accuracy_point(tied)["eps"] == 0.55


def test_plot_rejects_stale_scores(tmp_path: Path) -> None:
    dataset = _write_curve(tmp_path)
    with (dataset / "eps_overall.tsv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="scores change after completion"):
        tradeoff._load_q_class(tmp_path, "j_smith", "weight_query_label_balanced")


def test_original_style_plot_runs_with_verified_curves(tmp_path: Path) -> None:
    _write_curve(tmp_path)
    output = tmp_path / "tradeoff.png"
    args = tradeoff.parse_args(
        [
            "--root",
            str(tmp_path),
            "--datasets",
            "j_smith",
            "--output",
            str(output),
            "--weight",
            "weight_query_label_balanced",
            "--reference-eps",
            "0.55",
        ]
    )
    assert tradeoff.plot(args) == output.resolve()
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

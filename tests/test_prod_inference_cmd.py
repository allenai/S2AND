from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts._rust_suite import featurizer_reuse_cmd, prod_inference_cmd


def test_single_run_arrow_requires_rust_backend() -> None:
    with pytest.raises(ValueError, match="requires --backend rust"):
        prod_inference_cmd._single_run(  # noqa: SLF001
            backend="python",
            dataset_name="qian",
            n_jobs=2,
            profile_output_path="profile.txt",
            model_path="model",
            input_format="arrow",
        )


def test_single_run_json_rejects_rust_before_loading_data() -> None:
    with pytest.raises(ValueError, match="rust requires --input-format arrow"):
        prod_inference_cmd._single_run(  # noqa: SLF001
            backend="rust",
            dataset_name="qian",
            n_jobs=2,
            profile_output_path="profile.txt",
            model_path="missing-model",
            input_format="json",
        )


def test_featurizer_reuse_rejects_json_before_loading_rust_or_data() -> None:
    with pytest.raises(ValueError, match="requires --input-format arrow"):
        featurizer_reuse_cmd.run_reuse_profile(
            dataset_name="qian",
            n_jobs=1,
            repeats=1,
            model_path="missing-model",
            input_format="json",
        )


def test_main_single_arrow_emits_marked_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_json = tmp_path / "single.json"

    def fake_single_run(**kwargs: Any) -> dict[str, Any]:
        return {"input_format": kwargs["input_format"], "dataset": kwargs["dataset_name"]}

    monkeypatch.setattr(prod_inference_cmd, "_single_run", fake_single_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prod_inference_cmd.py",
            "--mode",
            "single",
            "--backend",
            "rust",
            "--input-format",
            "arrow",
            "--dataset-name",
            "qian",
            "--model-path",
            str(tmp_path / "model"),
            "--profile-output-path",
            str(tmp_path / "profile.txt"),
            "--single-write-json",
            str(output_json),
        ],
    )

    prod_inference_cmd.main()

    stdout = capsys.readouterr().out
    assert prod_inference_cmd.RESULT_JSON_START in stdout
    assert prod_inference_cmd.RESULT_JSON_END in stdout
    assert json.loads(output_json.read_text(encoding="utf-8")) == {"dataset": "qian", "input_format": "arrow"}

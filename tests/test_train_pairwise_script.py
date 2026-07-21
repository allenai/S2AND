from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scripts.production.model import train_pairwise


def _import_train_pairwise(env: dict[str, str], repo_root: Path) -> dict[str, str]:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, os;"
                "import scripts.production.model.train_pairwise as train_pairwise;"
                "print(json.dumps({"
                "'backend': os.environ.get('S2AND_BACKEND'),"
                "}))"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, f"stdout:\n{completed.stdout}\n" f"stderr:\n{completed.stderr}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_train_pairwise_does_not_claim_a_rust_backend() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("S2AND_BACKEND", None)

    payload = _import_train_pairwise(env, repo_root)

    assert payload["backend"] is None


def test_train_pairwise_respects_existing_backend_override() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["S2AND_BACKEND"] = "python"

    payload = _import_train_pairwise(env, repo_root)

    assert payload["backend"] == "python"


def test_train_pairwise_rejects_existing_output_before_loading_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "production_model_v9.9"
    output_dir.mkdir()

    def fail_if_artifact_work_starts() -> None:
        raise AssertionError("artifact work started before output validation")

    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_name_tuple_artifact",
        fail_if_artifact_work_starts,
    )

    with pytest.raises(SystemExit, match="must name a new directory"):
        train_pairwise.train_pairwise_bundle(
            cast(
                Namespace,
                SimpleNamespace(
                    run_full=True,
                    data_dir=tmp_path,
                    output_dir=output_dir,
                    production_version="9.9",
                ),
            )
        )


def test_feature_cache_dir_defaults_to_none() -> None:
    args = train_pairwise.build_parser().parse_args(["--production-version", "9.9"])
    assert args.feature_cache_dir is None

    args = train_pairwise.build_parser().parse_args(["--production-version", "9.9", "--feature-cache-dir", "some/dir"])
    assert args.feature_cache_dir == Path("some/dir")

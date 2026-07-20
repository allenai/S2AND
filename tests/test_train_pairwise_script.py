from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
                "from s2and.consts import CACHE_ROOT;"
                "print(json.dumps({"
                "'cache_root': str(CACHE_ROOT),"
                "'default_feature_cache_root': str(train_pairwise.DEFAULT_FEATURE_CACHE_ROOT),"
                "'env': os.environ.get('S2AND_CACHE'),"
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


def test_train_pairwise_sets_default_cache_before_importing_s2and() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("S2AND_CACHE", None)
    env.pop("S2AND_BACKEND", None)

    payload = _import_train_pairwise(env, repo_root)

    expected_cache_root = repo_root / "data" / ".feature_cache"
    assert Path(payload["env"]).resolve() == expected_cache_root.resolve()
    assert Path(payload["cache_root"]).resolve() == expected_cache_root.resolve()
    assert Path(payload["default_feature_cache_root"]).resolve() == expected_cache_root.resolve()
    assert payload["backend"] == "rust"


def test_train_pairwise_respects_existing_cache_and_backend_overrides(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    override_cache_root = tmp_path / "custom_cache"
    env = os.environ.copy()
    env["S2AND_CACHE"] = str(override_cache_root)
    env["S2AND_BACKEND"] = "python"

    payload = _import_train_pairwise(env, repo_root)

    assert Path(payload["env"]).resolve() == override_cache_root.resolve()
    assert Path(payload["cache_root"]).resolve() == override_cache_root.resolve()
    assert payload["backend"] == "python"


def test_train_pairwise_rejects_existing_output_before_loading_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "production_model_v9.9"
    output_dir.mkdir()
    monkeypatch.setattr(
        train_pairwise,
        "load_packaged_name_tuple_artifact",
        lambda: pytest.fail("artifact work started before output validation"),
    )

    with pytest.raises(SystemExit, match="must name a new directory"):
        train_pairwise.train_pairwise_bundle(
            SimpleNamespace(
                run_full=True,
                data_dir=tmp_path,
                output_dir=output_dir,
                production_version="9.9",
            )
        )

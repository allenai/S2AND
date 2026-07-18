from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


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
                "'production_pairwise_recipe': train_pairwise.PRODUCTION_PAIRWISE_RECIPE.as_dict(),"
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
    recipe = payload["production_pairwise_recipe"]
    assert recipe["name"] == "big7_1250_v1"
    assert recipe["uniform_source_domains"] == [
        "aminer",
        "arnetminer",
        "inspire",
        "kisti",
        "pubmed",
        "qian",
        "zbmath",
    ]
    assert recipe["uniform_pairs_per_domain"] == 100_000
    assert recipe["pair_sampling_seed"] == 1111
    assert recipe["additive_linker"]["source_set"] == "big7"
    assert recipe["additive_linker"]["source_domains"] == [
        "a_khan",
        "a_silva",
        "h_wang",
        "j_smith",
        "s_gupta",
        "s_lee",
        "s_park",
    ]
    assert recipe["additive_linker"]["linker_pairs_per_domain"] == 1_250
    assert recipe["additive_linker"]["linker_pairs_per_label"] == 625
    assert recipe["nominal_training_rows"] == 708_750


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

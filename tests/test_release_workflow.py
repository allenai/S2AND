from __future__ import annotations

import io
import json
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.verification.verify_production_model_distributions import (
    LEGACY_PRODUCTION_MODEL_PATHS,
    verify_production_model_distributions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-rust.yml"
MAIN_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "main.yaml"


def _workflow_job_condition(workflow: str, job_name: str) -> str:
    """Return one job-level condition from a workflow source string."""

    job_match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [a-z0-9][a-z0-9-]*:\n|\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert job_match is not None, f"missing workflow job {job_name}"
    condition_match = re.search(r"^    if: (?P<condition>.+)$", job_match.group("body"), flags=re.MULTILINE)
    assert condition_match is not None, f"missing condition for workflow job {job_name}"
    return condition_match.group("condition")


def test_rust_wheel_matrix_matches_supported_python_versions() -> None:
    rust_project = tomllib.loads((REPO_ROOT / "s2and_rust" / "pyproject.toml").read_text(encoding="utf-8"))
    assert rust_project["project"]["requires-python"] == ">=3.11,<3.14"

    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    matrices = re.findall(r"^\s+py: \[([^]]+)\]$", workflow, flags=re.MULTILINE)
    assert matrices == ['"3.11", "3.12", "3.13"'] * 2
    assert "-i python3.11 -i python3.12 -i python3.13" in workflow
    assert "python3.10" not in workflow


def test_python_publish_depends_on_release_validation_and_exact_rust_probe() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "needs: [detect-versions, s2and-dist, release-smoke, release-tests, probe-rust-release]" in workflow
    assert "needs: [detect-versions, publish-rust]" in workflow
    assert "uv run pytest -q" in workflow
    assert "load_production_model()" in workflow
    assert "scripts/verification/smoke_installed_incremental_arrow.py" in workflow
    assert "github.event_name == 'pull_request'" in workflow
    assert "needs.detect-versions.outputs.force_build == 'true'" in workflow
    assert workflow.count("scripts/verification/smoke_installed_rust_api.py") >= 2
    pr_force_build = "github.event_name == 'pull_request' && needs.detect-versions.outputs.force_build == 'true'"
    for job_name in ("s2and-dist", "wheels-windows", "wheels-macos", "wheels-linux", "sdist"):
        condition = _workflow_job_condition(workflow, job_name)
        assert condition.startswith(f"({pr_force_build}) || (github.event_name != 'pull_request' &&")
    release_smoke_condition = _workflow_job_condition(workflow, "release-smoke")
    assert release_smoke_condition.startswith(f"({pr_force_build}) || (github.event_name == 'push' &&")
    for job_name in ("publish-s2and", "publish-rust"):
        publish_condition = _workflow_job_condition(workflow, job_name)
        assert "pull_request" not in publish_condition
        assert "github.ref == 'refs/heads/main'" in publish_condition
    assert 'get("name", "").lower() == "force-build"' in workflow
    assert (
        "ALLOW_LEGACY_DEFAULT_REJECTION: ${{ github.event_name == 'pull_request' "
        "&& needs.detect-versions.outputs.force_build == 'true' }}"
    ) in workflow
    assert "expected_legacy_rejection" in workflow
    assert "str(exc) == expected_legacy_rejection" in workflow
    incremental_smoke = (REPO_ROOT / "scripts/verification/smoke_installed_incremental_arrow.py").read_text(
        encoding="utf-8"
    )
    assert "predict_incremental_from_arrow_paths(" in incremental_smoke


def test_rust_enabled_ci_cannot_convert_import_failures_to_skips() -> None:
    main_workflow = MAIN_WORKFLOW_PATH.read_text(encoding="utf-8")
    helper_source = (REPO_ROOT / "tests" / "helpers.py").read_text(encoding="utf-8")

    assert "S2AND_TEST_REQUIRE_RUST" in main_workflow
    assert "S2AND_TEST_REQUIRE_RUST" in helper_source
    assert "Rust-enabled tests require a working s2and_rust runtime" in helper_source


def test_release_workflow_uses_uv_for_python_commands_and_declared_default() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    main_workflow = MAIN_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert re.search(r"^\s+python\s", workflow, flags=re.MULTILINE) is None
    assert "production_model_v1.21" not in workflow
    assert "production_model_v1.21" not in main_workflow
    assert "default_production_model.json" in workflow
    assert "default_production_model.json" in main_workflow
    assert "scripts/verification/verify_production_model_distributions.py" in workflow


def _write_distribution_fixture(root: Path, *, extra_wheel: bool, extra_sdist: bool) -> tuple[Path, Path]:
    data_dir = root / "s2and" / "data"
    bundle_dir = data_dir / "production_model_v9.9"
    bundle_dir.mkdir(parents=True)
    declaration = {
        "bundle_dir": "production_model_v9.9",
        "bundle_version": "9.9",
        "schema_version": "s2and_default_production_model_v1",
    }
    (data_dir / "default_production_model.json").write_text(json.dumps(declaration), encoding="utf-8")
    (bundle_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    for path in LEGACY_PRODUCTION_MODEL_PATHS:
        source_path = root / path
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"hydrated legacy model")

    files = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    dist_dir = root / "dist"
    dist_dir.mkdir()
    wheel_path = dist_dir / "s2and-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for path, content in files.items():
            wheel.writestr(path, content)
        if extra_wheel:
            wheel.writestr("s2and/data/production_model_v8.8/manifest.json", b"{}\n")

    sdist_path = dist_dir / "s2and-0.0.0.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as sdist:
        for path, content in files.items():
            member = tarfile.TarInfo(f"s2and-0.0.0/{path}")
            member.size = len(content)
            sdist.addfile(member, io.BytesIO(content))
        if extra_sdist:
            content = b"{}\n"
            member = tarfile.TarInfo("s2and-0.0.0/s2and/data/production_model_v8.8/manifest.json")
            member.size = len(content)
            sdist.addfile(member, io.BytesIO(content))
    return dist_dir, root


def test_distribution_verifier_accepts_only_declared_and_legacy_models(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path, extra_wheel=False, extra_sdist=False)

    verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


@pytest.mark.parametrize(
    ("extra_wheel", "extra_sdist"),
    [(True, False), (False, True)],
)
def test_distribution_verifier_rejects_undeclared_model_assets(
    tmp_path: Path,
    extra_wheel: bool,
    extra_sdist: bool,
) -> None:
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=extra_wheel,
        extra_sdist=extra_sdist,
    )

    with pytest.raises(ValueError, match="undeclared production model assets"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


def test_distribution_verifier_rejects_unversioned_orcid_counts(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path, extra_wheel=False, extra_sdist=False)
    wheel_path = next(dist_dir.glob("s2and-*.whl"))
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("s2and/data/first_k_letter_counts_from_orcid.json", b"{}\n")

    with pytest.raises(ValueError, match="forbidden legacy runtime artifacts"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)

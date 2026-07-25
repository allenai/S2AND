from __future__ import annotations

import io
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

from scripts.verification.verify_production_model_distributions import verify_production_model_distributions

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-rust.yml"
MAIN_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "main.yaml"


def _release_workflow_jobs() -> dict[str, dict]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    return workflow["jobs"]


def _all_run_text(jobs: dict[str, dict]) -> str:
    return "\n".join(step.get("run", "") for job in jobs.values() for step in job.get("steps", []))


def _all_action_args(jobs: dict[str, dict]) -> str:
    return "\n".join(
        str(step["with"]["args"])
        for job in jobs.values()
        for step in job.get("steps", [])
        if isinstance(step.get("with"), dict) and "args" in step["with"]
    )


def test_ci_checkout_is_the_single_lfs_hydration_authority() -> None:
    workflow = yaml.safe_load(MAIN_WORKFLOW_PATH.read_text(encoding="utf-8"))
    job = workflow["jobs"]["typecheck-and-test"]
    checkout = next(step for step in job["steps"] if str(step.get("uses", "")).startswith("actions/checkout@"))
    run_text = "\n".join(str(step.get("run", "")) for step in job["steps"])

    assert checkout["with"]["lfs"] is True
    assert "git lfs" not in run_text
    assert "git-lfs.github.com/spec" not in run_text


def test_rust_wheel_matrix_matches_supported_python_versions() -> None:
    rust_project = tomllib.loads((REPO_ROOT / "s2and_rust" / "pyproject.toml").read_text(encoding="utf-8"))
    assert rust_project["project"]["requires-python"] == ">=3.11,<3.14"

    jobs = _release_workflow_jobs()
    python_matrices = [
        job["strategy"]["matrix"]["py"] for job in jobs.values() if "py" in job.get("strategy", {}).get("matrix", {})
    ]
    assert python_matrices == [["3.11", "3.12", "3.13"]] * 2
    interpreter_args = _all_action_args(jobs)
    assert "-i python3.11 -i python3.12 -i python3.13" in interpreter_args
    assert "python3.10" not in interpreter_args
    assert "python3.10" not in _all_run_text(jobs)


def test_release_maturin_action_satisfies_rust_build_system_requirement() -> None:
    rust_project = tomllib.loads((REPO_ROOT / "s2and_rust" / "pyproject.toml").read_text(encoding="utf-8"))
    maturin_requirement = Requirement(
        next(
            requirement for requirement in rust_project["build-system"]["requires"] if requirement.startswith("maturin")
        )
    )

    maturin_action_versions = [
        step["with"]["maturin-version"]
        for job in _release_workflow_jobs().values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("PyO3/maturin-action@")
    ]
    assert maturin_action_versions
    assert len(set(maturin_action_versions)) == 1
    assert all(version.startswith("v") for version in maturin_action_versions)
    assert all(
        Version(version.removeprefix("v")) in maturin_requirement.specifier for version in maturin_action_versions
    )


def test_rust_sdist_contains_vendored_cld2_and_is_clean_installed() -> None:
    rust_project = tomllib.loads((REPO_ROOT / "s2and_rust" / "pyproject.toml").read_text(encoding="utf-8"))
    assert rust_project["tool"]["maturin"]["include"] == [{"path": "vendor/cld2/**/*", "format": "sdist"}]

    sdist_job = _release_workflow_jobs()["sdist"]
    steps = sdist_job["steps"]
    step_names = [step.get("name") for step in steps]
    build_index = step_names.index("Build sdist")
    smoke_index = step_names.index("Install and exercise sdist")
    upload_index = next(
        index for index, step in enumerate(steps) if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    )
    assert build_index < smoke_index < upload_index

    setup_python = next(step for step in steps if str(step.get("uses", "")).startswith("actions/setup-python@"))
    assert setup_python["with"]["python-version"] == "3.11"
    smoke_script = steps[smoke_index]["run"]
    assert "-name '*.tar.gz'" in smoke_script
    assert "uv venv --python 3.11" in smoke_script
    assert "uv pip install --python" in smoke_script
    assert "scripts/verification/smoke_installed_rust_api.py" in smoke_script


def test_release_workflow_consumes_only_final_policy_outputs() -> None:
    jobs = _release_workflow_jobs()

    assert _all_run_text(jobs).count("scripts/sync_version.py --release-policy") == 1
    expected_conditions = {
        "s2and-dist": "needs.detect-versions.outputs.build_s2and == 'true'",
        "wheels-windows": "needs.detect-versions.outputs.build_rust == 'true'",
        "wheels-macos": "needs.detect-versions.outputs.build_rust == 'true'",
        "wheels-linux": "needs.detect-versions.outputs.build_rust == 'true'",
        "sdist": "needs.detect-versions.outputs.build_rust == 'true'",
        "release-smoke": "needs.detect-versions.outputs.run_release_smoke == 'true'",
        "release-tests": "needs.detect-versions.outputs.run_release_smoke == 'true'",
        "publish-s2and": "needs.detect-versions.outputs.publish_s2and == 'true'",
        "publish-rust": "needs.detect-versions.outputs.publish_rust == 'true'",
        "probe-rust-release": (
            "always() && needs.detect-versions.outputs.publish_s2and == 'true' && "
            "(needs.publish-rust.result == 'success' || needs.publish-rust.result == 'skipped')"
        ),
    }
    assert {name: job["if"] for name, job in jobs.items() if "if" in job} == expected_conditions

    expected_needs = {
        "s2and-dist": {"detect-versions"},
        "wheels-windows": {"detect-versions"},
        "wheels-macos": {"detect-versions"},
        "wheels-linux": {"detect-versions"},
        "sdist": {"detect-versions"},
        "release-smoke": {"detect-versions", "s2and-dist", "wheels-linux"},
        "release-tests": {"detect-versions"},
        "publish-s2and": {
            "detect-versions",
            "s2and-dist",
            "release-smoke",
            "release-tests",
            "probe-rust-release",
        },
        "publish-rust": {
            "detect-versions",
            "wheels-windows",
            "wheels-macos",
            "wheels-linux",
            "sdist",
            "release-smoke",
            "release-tests",
        },
        "probe-rust-release": {"detect-versions", "publish-rust"},
    }
    assert {name: set(job["needs"]) for name, job in jobs.items() if "needs" in job} == expected_needs


def test_publish_jobs_require_manual_release_intent_and_full_release_gates() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    jobs = workflow["jobs"]

    assert inputs["publish_s2and"]["type"] == "boolean"
    assert inputs["publish_rust"]["type"] == "boolean"
    assert "Release-ready" in inputs["publish_s2and"]["description"]
    assert "Release-ready" in inputs["publish_rust"]["description"]

    for job_name in ("publish-s2and", "publish-rust"):
        needs = set(jobs[job_name]["needs"])
        assert {"release-smoke", "release-tests"} <= needs


def test_python_package_data_is_explicit() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["setuptools"]["include-package-data"] is False
    package_data = config["tool"]["setuptools"]["package-data"]["s2and"]
    assert "arrow_schema_contract.json" in package_data
    assert all((REPO_ROOT / "s2and" / path).is_file() for path in package_data)
    assert all("first_k_letter_counts_from_orcid" not in pattern for pattern in package_data)
    assert all("production_model" not in pattern for pattern in package_data)
    assert set(package_data).isdisjoint(
        {
            "data/s2and_name_tuples.txt",
            "data/s2and_name_tuples_filtered.txt",
            "data/s2and_unnormalized_filtered_name_tuples.txt",
        }
    )


TUPLE_DATA_PATH = "s2and/data/s2and_name_tuples_canonical.txt"
TUPLE_META_PATH = "s2and/data/s2and_name_tuples_canonical.txt.meta.json"
ORCID_DATA_PATH = "s2and/data/first_k_letter_counts_from_orcid.json"
ORCID_MANIFEST_PATH = "s2and/data/first_k_letter_counts_from_orcid.manifest.json"


def _write_distribution_fixture(
    root: Path,
    *,
    extra_wheel: bool,
    extra_sdist: bool,
    declare_model: bool = True,
    omit_wheel_path: str | None = None,
    omit_sdist_path: str | None = None,
    include_orcid: bool = False,
    declare_orcid: bool = True,
    declare_tuples: bool = True,
    corrupt_wheel_path: str | None = None,
    sdist_package_parent: str | None = None,
) -> tuple[Path, Path]:
    data_dir = root / "s2and" / "data"
    data_dir.mkdir(parents=True)
    (root / "s2and" / "arrow_schema_contract.json").write_text("{}\n", encoding="utf-8")
    (data_dir / "path_config.json").write_text("{}\n", encoding="utf-8")
    (root / TUPLE_DATA_PATH).write_text("alice\talicia\n", encoding="utf-8")
    (root / TUPLE_META_PATH).write_text("{}\n", encoding="utf-8")
    declared = ["arrow_schema_contract.json", "data/path_config.json"]
    if declare_tuples:
        declared += [
            "data/s2and_name_tuples_canonical.txt",
            "data/s2and_name_tuples_canonical.txt.meta.json",
        ]
    if include_orcid:
        (root / ORCID_DATA_PATH).write_text("{}\n", encoding="utf-8")
        (root / ORCID_MANIFEST_PATH).write_text("{}\n", encoding="utf-8")
        if declare_orcid:
            declared += [
                "data/first_k_letter_counts_from_orcid.json",
                "data/first_k_letter_counts_from_orcid.manifest.json",
            ]
    (root / "pyproject.toml").write_text(
        f"[tool.setuptools.package-data]\ns2and = {json.dumps(declared)}\n",
        encoding="utf-8",
    )
    if declare_model:
        bundle_dir = data_dir / "production_model_v9.9"
        bundle_dir.mkdir()
        declaration = {
            "bundle_dir": "production_model_v9.9",
            "bundle_version": "9.9",
            "schema_version": "s2and_default_production_model_v1",
        }
        (data_dir / "default_production_model.json").write_text(json.dumps(declaration), encoding="utf-8")
        (bundle_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    files = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    dist_dir = root / "dist"
    dist_dir.mkdir()
    wheel_path = dist_dir / "s2and-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for path, content in files.items():
            if path == omit_wheel_path:
                continue
            wheel.writestr(path, b"tampered\n" if path == corrupt_wheel_path else content)
        if extra_wheel:
            wheel.writestr("s2and/data/production_model_v8.8/manifest.json", b"{}\n")

    sdist_path = dist_dir / "s2and-0.0.0.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as sdist:
        for path, content in files.items():
            if path == omit_sdist_path:
                continue
            member_path = path if sdist_package_parent is None else f"{sdist_package_parent}/{path}"
            member = tarfile.TarInfo(f"s2and-0.0.0/{member_path}")
            member.size = len(content)
            sdist.addfile(member, io.BytesIO(content))
        if extra_sdist:
            content = b"{}\n"
            member = tarfile.TarInfo("s2and-0.0.0/s2and/data/production_model_v8.8/manifest.json")
            member.size = len(content)
            sdist.addfile(member, io.BytesIO(content))
    return dist_dir, root


def test_distribution_verifier_accepts_only_declared_model(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path, extra_wheel=False, extra_sdist=False)

    verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


def test_distribution_verifier_rejects_nested_sdist_package_decoys(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=False,
        extra_sdist=False,
        sdist_package_parent="docs",
    )

    with pytest.raises(ValueError, match="missing required distribution files"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


def test_distribution_verifier_accepts_no_model_during_cutover(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=False,
        extra_sdist=False,
        declare_model=False,
    )

    verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


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
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


@pytest.mark.parametrize(
    ("omit_wheel_path", "omit_sdist_path"),
    (
        ("s2and/arrow_schema_contract.json", None),
        (None, "s2and/data/path_config.json"),
    ),
)
def test_distribution_verifier_requires_declared_package_data(
    tmp_path: Path,
    omit_wheel_path: str | None,
    omit_sdist_path: str | None,
) -> None:
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=False,
        extra_sdist=False,
        omit_wheel_path=omit_wheel_path,
        omit_sdist_path=omit_sdist_path,
    )

    with pytest.raises(ValueError, match="missing required distribution files"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


@pytest.mark.parametrize(
    "missing_source_path",
    (
        "s2and/arrow_schema_contract.json",
        "s2and/data/path_config.json",
    ),
)
def test_distribution_verifier_rejects_missing_declared_package_data_source(
    tmp_path: Path,
    missing_source_path: str,
) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path, extra_wheel=False, extra_sdist=False)
    (source_root / missing_source_path).unlink()

    with pytest.raises(FileNotFoundError, match="package-data patterns match no source files"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


def test_distribution_verifier_rejects_archive_content_drift(tmp_path: Path) -> None:
    """Inventory alone is fail-open: a declared path can ship stale or truncated bytes."""

    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=False,
        extra_sdist=False,
        corrupt_wheel_path=TUPLE_DATA_PATH,
    )

    with pytest.raises(ValueError, match="content differs from the source tree"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


def test_distribution_verifier_requires_phase_runtime_artifacts(tmp_path: Path) -> None:
    """Removing a declaration must fail, which pyproject-derived inventory alone cannot see."""

    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=False,
        extra_sdist=False,
        declare_tuples=False,
    )

    with pytest.raises(ValueError, match="requires these paths to be declared"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


def test_code_only_phase_forbids_orcid_artifacts(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=False,
        extra_sdist=False,
        include_orcid=True,
    )

    with pytest.raises(ValueError, match="forbids declaring these paths"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


def test_code_only_phase_forbids_undeclared_orcid_source_files(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=False,
        extra_sdist=False,
        include_orcid=True,
        declare_orcid=False,
    )

    with pytest.raises(ValueError, match="forbids these source files"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="code_only")


def test_release_candidate_phase_requires_orcid_artifacts(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path, extra_wheel=False, extra_sdist=False)

    with pytest.raises(ValueError, match="requires these paths to be declared"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="release_candidate")


def test_release_candidate_phase_accepts_promoted_orcid_artifacts(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=False,
        extra_sdist=False,
        include_orcid=True,
    )

    verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root, phase="release_candidate")


def test_current_repository_state_is_a_valid_code_only_phase() -> None:
    """The checked-in tree must satisfy the phase contract it claims to be in."""

    from scripts.verification.verify_production_model_distributions import (
        _load_expected_paths,
    )

    expected = _load_expected_paths(REPO_ROOT, phase="code_only")

    assert TUPLE_DATA_PATH in expected
    assert TUPLE_META_PATH in expected
    assert ORCID_DATA_PATH not in expected

    with pytest.raises(ValueError, match="requires these paths to be declared"):
        _load_expected_paths(REPO_ROOT, phase="release_candidate")

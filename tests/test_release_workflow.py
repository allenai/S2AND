from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import tomllib
import zipfile
from collections import Counter
from pathlib import Path

import pytest
import yaml

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


def test_release_maturin_action_matches_rust_build_system_floor() -> None:
    rust_project = tomllib.loads((REPO_ROOT / "s2and_rust" / "pyproject.toml").read_text(encoding="utf-8"))
    maturin_requirement = next(
        requirement for requirement in rust_project["build-system"]["requires"] if requirement.startswith("maturin")
    )
    floor_match = re.search(r">=([^,]+)", maturin_requirement)
    assert floor_match is not None
    expected_action_version = f"v{floor_match.group(1)}"

    maturin_action_versions = [
        step["with"]["maturin-version"]
        for job in _release_workflow_jobs().values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("PyO3/maturin-action@")
    ]
    assert maturin_action_versions
    assert set(maturin_action_versions) == {expected_action_version}


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
    conditions = Counter(
        condition
        for job in jobs.values()
        for condition in (job.get("if"), *(step.get("if") for step in job.get("steps", [])))
        if condition
    )
    assert conditions["needs.detect-versions.outputs.build_s2and == 'true'"] == 1
    assert conditions["needs.detect-versions.outputs.build_rust == 'true'"] == 4
    assert conditions["needs.detect-versions.outputs.run_release_smoke == 'true'"] == 2
    assert conditions["needs.detect-versions.outputs.publish_s2and == 'true'"] == 1
    assert conditions["needs.detect-versions.outputs.publish_rust == 'true'"] == 1
    needs_lists = [job.get("needs") for job in jobs.values() if isinstance(job.get("needs"), list)]
    assert ["detect-versions", "s2and-dist", "release-smoke", "release-tests", "probe-rust-release"] in needs_lists
    assert [
        "detect-versions",
        "wheels-windows",
        "wheels-macos",
        "wheels-linux",
        "sdist",
        "release-smoke",
        "release-tests",
    ] in needs_lists
    assert ["detect-versions", "publish-rust"] in needs_lists


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
    canonical_orcid_artifacts = {
        "data/first_k_letter_counts_from_orcid.json",
        "data/first_k_letter_counts_from_orcid.manifest.json",
    }
    assert "arrow_schema_contract.json" in package_data
    assert canonical_orcid_artifacts <= set(package_data)
    assert all("production_model" not in pattern for pattern in package_data)
    assert set(package_data).isdisjoint(
        {
            "data/s2and_name_tuples.txt",
            "data/s2and_name_tuples_filtered.txt",
            "data/s2and_unnormalized_filtered_name_tuples.txt",
        }
    )
    excluded_package_data = config["tool"]["setuptools"].get("exclude-package-data", {}).get("s2and", [])
    assert canonical_orcid_artifacts.isdisjoint(excluded_package_data)


def _write_distribution_fixture(
    root: Path,
    *,
    extra_wheel: bool,
    extra_sdist: bool,
    declare_model: bool = True,
    omit_wheel_path: str | None = None,
    omit_sdist_path: str | None = None,
) -> tuple[Path, Path]:
    data_dir = root / "s2and" / "data"
    data_dir.mkdir(parents=True)
    orcid_data_payload = b"{}\n"
    (data_dir / "first_k_letter_counts_from_orcid.json").write_bytes(orcid_data_payload)
    (data_dir / "first_k_letter_counts_from_orcid.manifest.json").write_text(
        json.dumps(
            {
                "data_sha256": hashlib.sha256(orcid_data_payload).hexdigest(),
                "normalization_version": "canonical_v2",
                "pair_key_semantics": "unordered_lexicographic",
                "schema_version": "orcid_prefix_counts_v2",
            },
            sort_keys=True,
        )
        + "\n",
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
            if path != omit_wheel_path:
                wheel.writestr(path, content)
        if extra_wheel:
            wheel.writestr("s2and/data/production_model_v8.8/manifest.json", b"{}\n")

    sdist_path = dist_dir / "s2and-0.0.0.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as sdist:
        for path, content in files.items():
            if path == omit_sdist_path:
                continue
            member = tarfile.TarInfo(f"s2and-0.0.0/{path}")
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

    verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


def test_distribution_verifier_accepts_no_model_during_cutover(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        extra_wheel=False,
        extra_sdist=False,
        declare_model=False,
    )

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


@pytest.mark.parametrize(
    ("omit_wheel_path", "omit_sdist_path"),
    (
        ("s2and/data/first_k_letter_counts_from_orcid.json", None),
        ("s2and/data/first_k_letter_counts_from_orcid.manifest.json", None),
        (None, "s2and/data/first_k_letter_counts_from_orcid.json"),
        (None, "s2and/data/first_k_letter_counts_from_orcid.manifest.json"),
    ),
)
def test_distribution_verifier_requires_canonical_orcid_artifact_pair(
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
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


@pytest.mark.parametrize(
    "missing_source_path",
    (
        "s2and/data/first_k_letter_counts_from_orcid.json",
        "s2and/data/first_k_letter_counts_from_orcid.manifest.json",
    ),
)
def test_distribution_verifier_rejects_missing_canonical_orcid_source(
    tmp_path: Path,
    missing_source_path: str,
) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path, extra_wheel=False, extra_sdist=False)
    (source_root / missing_source_path).unlink()

    with pytest.raises(FileNotFoundError, match="missing required canonical runtime artifacts"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)

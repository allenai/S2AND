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


def test_native_cache_keys_hash_rust_inputs_not_build_outputs() -> None:
    workflow = yaml.safe_load(MAIN_WORKFLOW_PATH.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["typecheck-and-test"]["steps"]
    cache_keys = {
        step["name"]: step["with"]["key"]
        for step in steps
        if step.get("name") in {"Cache cargo build", "Cache uv + venv"}
    }

    assert set(cache_keys) == {"Cache cargo build", "Cache uv + venv"}
    rust_globs = "'s2and_rust/**', '!s2and_rust/target/**', '!s2and_rust/dist/**'"
    assert f"hashFiles({rust_globs})" in cache_keys["Cache cargo build"]
    assert f"hashFiles('pyproject.toml', 'uv.lock', {rust_globs})" in cache_keys["Cache uv + venv"]
    uv_cache = next(step for step in steps if step.get("name") == "Cache uv + venv")
    assert "restore-keys" not in uv_cache["with"]


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


def test_release_workflow_has_one_manual_digest_pinned_path() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert set(workflow[True]) == {"workflow_dispatch"}
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    jobs = workflow["jobs"]

    assert set(inputs) == {"evidence_manifest_url", "evidence_manifest_sha256"}
    assert all(value["required"] is True and value["type"] == "string" for value in inputs.values())
    verification = "\n".join(str(step.get("run", "")) for step in jobs["verify-release-input"]["steps"])
    assert 'GITHUB_REF" = "refs/heads/main' in verification
    assert "https://*" in verification
    assert "sha256sum --check --strict" in verification
    assert "python -m json.tool" in verification
    assert "s2and/release_evidence.py" in verification
    verification_steps = jobs["verify-release-input"]["steps"]
    stage_index = next(
        index
        for index, step in enumerate(verification_steps)
        if step.get("name") == "Download and verify release evidence manifest"
    )
    gate = verification_steps[stage_index + 1]
    assert gate["name"] == "Enforce release evaluation gate"
    assert gate["env"]["EVIDENCE_MANIFEST_SHA256"] == "${{ inputs.evidence_manifest_sha256 }}"
    assert "uv run --frozen python" in gate["run"]
    assert "release_pairwise.py evaluate-release" in gate["run"]
    assert "--evidence-manifest release-evidence/evidence_manifest.json" in gate["run"]
    assert '--expected-evidence-manifest-sha256 "$EVIDENCE_MANIFEST_SHA256"' in gate["run"]
    assert "--output-report release-evidence/evaluation_report.json" in gate["run"]
    assert "verify_production_model_distributions.py" in _all_run_text(jobs)
    assert all("if" not in job for job in jobs.values())

    for job_name in ("s2and-dist", "wheels-windows", "wheels-macos", "wheels-linux", "sdist", "release-tests"):
        assert "verify-release-input" in jobs[job_name]["needs"]


def test_release_publication_is_one_ordered_gated_chain() -> None:
    jobs = _release_workflow_jobs()
    run_text = _all_run_text(jobs)

    assert set(jobs["release-bundle"]["needs"]) == {
        "s2and-dist",
        "wheels-windows",
        "wheels-macos",
        "wheels-linux",
        "sdist",
    }
    assert jobs["release-smoke"]["needs"] == ["release-bundle"]
    assert "sha256sum > SHA256SUMS" in run_text
    assert run_text.count("sha256sum --check --strict SHA256SUMS") == 4
    assert {"release-bundle", "release-smoke", "release-tests"} <= set(jobs["publish-rust"]["needs"])
    assert jobs["probe-rust-release"]["needs"] == ["publish-rust"]
    assert {"release-bundle", "release-smoke", "release-tests", "probe-rust-release"} <= set(
        jobs["publish-s2and"]["needs"]
    )
    assert jobs["probe-s2and-release"]["needs"] == ["publish-s2and"]
    assert jobs["publish-rust"]["environment"]["name"] == "pypi"
    assert jobs["publish-s2and"]["environment"]["name"] == "pypi"


def test_python_package_data_is_explicit() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["setuptools"]["include-package-data"] is False
    package_data = config["tool"]["setuptools"]["package-data"]["s2and"]
    assert "arrow_schema_contract.json" in package_data
    assert all((REPO_ROOT / "s2and" / path).is_file() for path in package_data)
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
DEFAULT_MODEL_PATH = "s2and/data/default_production_model.json"
MODEL_MEMBER_PATH = "s2and/data/production_model_v8.8/manifest.json"
GENERIC_MEMBER_PATH = "s2and/arrow_schema_contract.json"
REQUIRED_RUNTIME_PATHS = (TUPLE_DATA_PATH, TUPLE_META_PATH, ORCID_DATA_PATH, ORCID_MANIFEST_PATH)
RELEASE_MEMBERS = {
    TUPLE_DATA_PATH: b"alice\talicia\n",
    TUPLE_META_PATH: b"{}\n",
    ORCID_DATA_PATH: b"{}\n",
    ORCID_MANIFEST_PATH: b"{}\n",
    GENERIC_MEMBER_PATH: b"{}\n",
}


def _write_package_data_config(root: Path, declared_paths: set[str]) -> None:
    declared = sorted(path.removeprefix("s2and/") for path in declared_paths)
    (root / "pyproject.toml").write_text(
        f"[tool.setuptools.package-data]\ns2and = {json.dumps(declared)}\n",
        encoding="utf-8",
    )


def _with_overrides(
    overrides: dict[str, bytes | None] | None,
) -> dict[str, bytes]:
    members = dict(RELEASE_MEMBERS)
    for path, content in (overrides or {}).items():
        if content is None:
            members.pop(path, None)
        else:
            members[path] = content
    return members


def _write_distribution_fixture(
    root: Path,
    *,
    wheel_overrides: dict[str, bytes | None] | None = None,
    sdist_overrides: dict[str, bytes | None] | None = None,
) -> tuple[Path, Path]:
    for path, content in RELEASE_MEMBERS.items():
        source = root / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    _write_package_data_config(root, set(RELEASE_MEMBERS))

    dist_dir = root / "dist"
    dist_dir.mkdir()
    wheel_path = dist_dir / "s2and-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for path, content in _with_overrides(wheel_overrides).items():
            wheel.writestr(path, content)

    sdist_path = dist_dir / "s2and-0.0.0.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as sdist:
        for path, content in _with_overrides(sdist_overrides).items():
            member = tarfile.TarInfo(f"s2and-0.0.0/{path}")
            member.size = len(content)
            sdist.addfile(member, io.BytesIO(content))
    return dist_dir, root


@pytest.mark.parametrize(
    ("archive_kind", "replacement", "error"),
    (
        ("wheel", None, "missing required distribution files"),
        ("sdist", None, "missing required distribution files"),
        ("wheel", b"tampered\n", "content differs from the source tree"),
        ("sdist", b"tampered\n", "content differs from the source tree"),
    ),
)
def test_distribution_verifier_rejects_missing_or_drifted_member(
    tmp_path: Path,
    archive_kind: str,
    replacement: bytes | None,
    error: str,
) -> None:
    overrides = {GENERIC_MEMBER_PATH: replacement}
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        wheel_overrides=overrides if archive_kind == "wheel" else None,
        sdist_overrides=overrides if archive_kind == "sdist" else None,
    )

    with pytest.raises(ValueError, match=error):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


def test_distribution_verifier_accepts_valid_archives(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path)

    verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


@pytest.mark.parametrize("missing_path", REQUIRED_RUNTIME_PATHS)
def test_distribution_verifier_requires_runtime_artifact(tmp_path: Path, missing_path: str) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path)
    _write_package_data_config(source_root, set(RELEASE_MEMBERS) - {missing_path})

    with pytest.raises(ValueError, match="Release distributions require these paths"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


@pytest.mark.parametrize("archive_kind", ("wheel", "sdist"))
@pytest.mark.parametrize("forbidden_path", (DEFAULT_MODEL_PATH, MODEL_MEMBER_PATH))
def test_distribution_verifier_rejects_model_path(
    tmp_path: Path,
    archive_kind: str,
    forbidden_path: str,
) -> None:
    overrides = {forbidden_path: b"{}\n"}
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        wheel_overrides=overrides if archive_kind == "wheel" else None,
        sdist_overrides=overrides if archive_kind == "sdist" else None,
    )

    with pytest.raises(ValueError, match="forbidden production model paths"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)

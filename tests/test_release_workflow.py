from __future__ import annotations

import io
import json
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts.verification.verify_production_model_distributions import verify_production_model_distributions

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "release-rust.yml"


def test_rust_wheel_matrix_matches_supported_python_versions() -> None:
    rust_project = tomllib.loads((REPO_ROOT / "s2and_rust" / "pyproject.toml").read_text(encoding="utf-8"))
    assert rust_project["project"]["requires-python"] == ">=3.11,<3.14"

    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    matrices = re.findall(r"^\s+py: \[([^]]+)\]$", workflow, flags=re.MULTILINE)
    assert matrices == ['"3.11", "3.12", "3.13"'] * 2
    assert "-i python3.11 -i python3.12 -i python3.13" in workflow
    assert "python3.10" not in workflow


def test_release_workflow_consumes_only_final_policy_outputs() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert workflow.count("scripts/sync_version.py --release-policy") == 1
    assert workflow.count("if: needs.detect-versions.outputs.build_s2and == 'true'") == 1
    assert workflow.count("if: needs.detect-versions.outputs.build_rust == 'true'") == 4
    assert workflow.count("if: needs.detect-versions.outputs.run_release_smoke == 'true'") == 1
    assert workflow.count("if: needs.detect-versions.outputs.publish_s2and == 'true'") == 2
    assert workflow.count("if: needs.detect-versions.outputs.publish_rust == 'true'") == 1
    assert "needs: [detect-versions, s2and-dist, release-smoke, release-tests, probe-rust-release]" in workflow
    assert "needs: [detect-versions, publish-rust]" in workflow


def test_python_package_data_is_explicit() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["setuptools"]["include-package-data"] is False
    package_data = config["tool"]["setuptools"]["package-data"]["s2and"]
    assert "arrow_schema_contract.json" in package_data
    assert all("production_model" not in pattern for pattern in package_data)


def _write_distribution_fixture(
    root: Path,
    *,
    extra_wheel: bool,
    extra_sdist: bool,
    declare_model: bool = True,
) -> tuple[Path, Path]:
    data_dir = root / "s2and" / "data"
    data_dir.mkdir(parents=True)
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


def test_distribution_verifier_rejects_unversioned_orcid_counts(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path, extra_wheel=False, extra_sdist=False)
    wheel_path = next(dist_dir.glob("s2and-*.whl"))
    with zipfile.ZipFile(wheel_path, "a") as wheel:
        wheel.writestr("s2and/data/first_k_letter_counts_from_orcid.json", b"{}\n")

    with pytest.raises(ValueError, match="forbidden legacy runtime artifacts"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)

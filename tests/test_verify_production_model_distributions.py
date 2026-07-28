from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.verification.verify_production_model_distributions import verify_production_model_distributions

TUPLE_DATA_PATH = "s2and/data/s2and_name_tuples_canonical.txt"
ORCID_DATA_PATH = "s2and/data/first_k_letter_counts_from_orcid.json"
ORCID_MANIFEST_PATH = "s2and/data/first_k_letter_counts_from_orcid.manifest.json"
DEFAULT_MODEL_PATH = "s2and/data/default_production_model.json"
MODEL_MEMBER_PATH = "s2and/data/production_model_v8.8/manifest.json"
GENERIC_MEMBER_PATH = "s2and/arrow_schema_contract.json"
REQUIRED_RUNTIME_PATHS = (TUPLE_DATA_PATH, ORCID_DATA_PATH, ORCID_MANIFEST_PATH)
RELEASE_MEMBERS = {
    TUPLE_DATA_PATH: b"alice\talicia\n",
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

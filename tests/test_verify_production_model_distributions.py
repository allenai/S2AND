from __future__ import annotations

import io
import json
import tarfile
import zipfile
from pathlib import Path

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


def test_distribution_verifier_rejects_missing_or_drifted_member(tmp_path: Path) -> None:
    cases = (
        ("wheel", None, "missing required distribution files"),
        ("sdist", None, "missing required distribution files"),
        ("wheel", b"tampered\n", "content differs from the source tree"),
    )
    for archive_kind, replacement, message in cases:
        case_id = f"{archive_kind}-{'missing' if replacement is None else 'drifted'}"
        overrides = {GENERIC_MEMBER_PATH: replacement}
        dist_dir, source_root = _write_distribution_fixture(
            tmp_path / case_id,
            wheel_overrides=overrides if archive_kind == "wheel" else None,
            sdist_overrides=overrides if archive_kind == "sdist" else None,
        )

        try:
            verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)
        except ValueError as error:
            assert message in str(error), f"{case_id}: {error}"
        else:
            raise AssertionError(f"{case_id}: invalid distribution member was accepted")


def test_distribution_verifier_accepts_valid_archives(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path)

    verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


def test_distribution_verifier_requires_runtime_artifact(tmp_path: Path) -> None:
    for index, missing_path in enumerate(sorted(REQUIRED_RUNTIME_PATHS)):
        case_id = f"runtime-artifact-{index}"
        dist_dir, source_root = _write_distribution_fixture(tmp_path / case_id)
        _write_package_data_config(source_root, set(RELEASE_MEMBERS) - {missing_path})

        try:
            verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)
        except ValueError as error:
            assert "Release distributions require these paths" in str(error), f"{missing_path}: {error}"
        else:
            raise AssertionError(f"{missing_path}: missing runtime artifact was accepted")


def test_distribution_verifier_rejects_model_path(tmp_path: Path) -> None:
    cases = (
        ("wheel", DEFAULT_MODEL_PATH),
        ("wheel", MODEL_MEMBER_PATH),
        ("sdist", DEFAULT_MODEL_PATH),
    )
    for archive_kind, forbidden_path in cases:
        case_id = f"{archive_kind}-{Path(forbidden_path).parent.name}-{Path(forbidden_path).name}"
        overrides = {forbidden_path: b"{}\n"}
        dist_dir, source_root = _write_distribution_fixture(
            tmp_path / case_id,
            wheel_overrides=overrides if archive_kind == "wheel" else None,
            sdist_overrides=overrides if archive_kind == "sdist" else None,
        )

        try:
            verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)
        except ValueError as error:
            assert "forbidden production model paths" in str(error), f"{case_id}: {error}"
        else:
            raise AssertionError(f"{case_id}: forbidden model path was accepted")

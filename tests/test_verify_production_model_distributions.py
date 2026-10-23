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
MODULE_PATH = "s2and/__init__.py"
NATIVE_PATH = "s2and/_s2and_rust.cp311-win_amd64.pyd"
METADATA_PATH = "s2and-0.0.0.dist-info/METADATA"
REQUIRED_RUNTIME_PATHS = (TUPLE_DATA_PATH, ORCID_DATA_PATH, ORCID_MANIFEST_PATH)
RELEASE_MEMBERS = {
    TUPLE_DATA_PATH: b"alice\talicia\n",
    ORCID_DATA_PATH: b"{}\n",
    ORCID_MANIFEST_PATH: b"{}\n",
    GENERIC_MEMBER_PATH: b"{}\n",
}
SOURCE_MEMBERS = {**RELEASE_MEMBERS, MODULE_PATH: b""}
WHEEL_MEMBERS = {**SOURCE_MEMBERS, NATIVE_PATH: b"native", METADATA_PATH: b"Name: s2and\n"}


def _write_package_data_config(root: Path, declared_paths: set[str]) -> None:
    declared = ", ".join("{ path = " + json.dumps(path) + " }" for path in sorted(declared_paths))
    (root / "pyproject.toml").write_text(
        f"[tool.maturin]\ninclude = [{declared}]\n",
        encoding="utf-8",
    )


def _with_overrides(
    base: dict[str, bytes],
    overrides: dict[str, bytes | None] | None,
) -> dict[str, bytes]:
    members = dict(base)
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
    for path, content in SOURCE_MEMBERS.items():
        source = root / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(content)
    _write_package_data_config(root, set(RELEASE_MEMBERS))

    dist_dir = root / "dist"
    dist_dir.mkdir()
    wheel_path = dist_dir / "s2and-0.0.0-cp311-cp311-win_amd64.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for path, content in _with_overrides(WHEEL_MEMBERS, wheel_overrides).items():
            wheel.writestr(path, content)

    sdist_path = dist_dir / "s2and-0.0.0.tar.gz"
    with tarfile.open(sdist_path, "w:gz") as sdist:
        for path, content in _with_overrides(SOURCE_MEMBERS, sdist_overrides).items():
            member = tarfile.TarInfo(f"s2and-0.0.0/{path}")
            member.size = len(content)
            sdist.addfile(member, io.BytesIO(content))
    return dist_dir, root


@pytest.mark.parametrize(
    ("archive_kind", "replacement", "message"),
    (
        ("wheel", None, "missing required distribution files"),
        ("sdist", None, "missing required distribution files"),
        ("wheel", b"tampered\n", "content differs from the source tree"),
    ),
)
def test_distribution_verifier_rejects_missing_or_drifted_member(tmp_path, archive_kind, replacement, message) -> None:
    overrides = {GENERIC_MEMBER_PATH: replacement}
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        wheel_overrides=overrides if archive_kind == "wheel" else None,
        sdist_overrides=overrides if archive_kind == "sdist" else None,
    )

    with pytest.raises(ValueError, match=message):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


def test_distribution_verifier_accepts_valid_archives(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path)

    verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


@pytest.mark.parametrize(
    ("platform", "path"),
    [
        ("musllinux_1_2_x86_64", "s2and.libs/extra.py"),
        ("musllinux_1_2_x86_64", "s2and.libs/libunexpected-123abc.so.1"),
        ("musllinux_1_2_x86_64", "s2and.libs/libgcc_s.so.1"),
        ("win_amd64", "s2and.libs/libgcc_s-f685abf1.so.1"),
    ],
)
def test_distribution_verifier_rejects_unexpected_bundled_files(tmp_path: Path, platform: str, path: str) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path, wheel_overrides={path: b"unexpected"})
    wheel = next(dist_dir.glob("*.whl"))
    wheel.rename(dist_dir / f"s2and-0.0.0-cp311-cp311-{platform}.whl")

    with pytest.raises(ValueError, match="unexpected package files"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


@pytest.mark.parametrize("missing_path", sorted(REQUIRED_RUNTIME_PATHS))
def test_distribution_verifier_requires_runtime_artifact(tmp_path: Path, missing_path: str) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path)
    _write_package_data_config(source_root, set(RELEASE_MEMBERS) - {missing_path})

    with pytest.raises(ValueError, match="Release distributions require these paths"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


@pytest.mark.parametrize(
    ("archive_kind", "forbidden_path"),
    (
        ("wheel", DEFAULT_MODEL_PATH),
        ("wheel", MODEL_MEMBER_PATH),
        ("sdist", DEFAULT_MODEL_PATH),
    ),
)
def test_distribution_verifier_rejects_model_path(tmp_path, archive_kind, forbidden_path) -> None:
    overrides: dict[str, bytes | None] = {forbidden_path: b"{}\n"}
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        wheel_overrides=overrides if archive_kind == "wheel" else None,
        sdist_overrides=overrides if archive_kind == "sdist" else None,
    )

    with pytest.raises(ValueError, match="forbidden production model paths"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


@pytest.mark.parametrize("keep", ["wheel", "sdist"])
def test_verify_separate_platform_build_artifacts(tmp_path, keep):
    dist_dir, root = _write_distribution_fixture(tmp_path)
    suffix = "*.tar.gz" if keep == "wheel" else "*.whl"
    for path in dist_dir.glob(suffix):
        path.unlink()
    verify_production_model_distributions(dist_dir=dist_dir, source_root=root)


def test_verify_every_platform_wheel(tmp_path):
    dist_dir, root = _write_distribution_fixture(tmp_path)
    with zipfile.ZipFile(dist_dir / "s2and-0.0.0-cp312-cp312-win_amd64.whl", "w") as wheel:
        for path, payload in WHEEL_MEMBERS.items():
            if path != GENERIC_MEMBER_PATH:
                wheel.writestr(path, payload)
    with pytest.raises(ValueError, match="missing required distribution files"):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=root)


@pytest.mark.parametrize(
    ("archive_kind", "overrides", "message"),
    (
        # Maturin packs any file Git does not ignore, so stray files must fail the inventory.
        ("wheel", {"s2and/data/extra.json": b"{}\n"}, "unexpected package files"),
        ("sdist", {"s2and/data/extra.json": b"{}\n"}, "unexpected package files"),
        ("wheel", {"s2and/__pycache__/model.cpython-311.pyc": b""}, "unexpected package files"),
        ("wheel", {"scripts/extra.py": b""}, "unexpected package files"),
        ("wheel", {MODULE_PATH: None}, "missing source modules"),
        ("sdist", {MODULE_PATH: None}, "missing source modules"),
        ("wheel", {NATIVE_PATH: None}, "must contain 1 native extension"),
        ("wheel", {"s2and/_s2and_rust.cp312-win_amd64.pyd": b"native"}, "must contain 1 native extension"),
        ("sdist", {NATIVE_PATH: b"native"}, "unexpected package files"),
    ),
)
def test_distribution_verifier_requires_exact_package_inventory(tmp_path, archive_kind, overrides, message) -> None:
    dist_dir, source_root = _write_distribution_fixture(
        tmp_path,
        wheel_overrides=overrides if archive_kind == "wheel" else None,
        sdist_overrides=overrides if archive_kind == "sdist" else None,
    )

    with pytest.raises(ValueError, match=message):
        verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)


def test_distribution_verifier_accepts_real_maturin_wheel_layout(tmp_path: Path) -> None:
    dist_dir, source_root = _write_distribution_fixture(tmp_path)
    licensed = {
        "s2and-0.0.0.dist-info/licenses/LICENSE": b"license\n",
        "s2and-0.0.0.dist-info/sboms/s2and_rust.cyclonedx.json": b"{}\n",
    }
    with zipfile.ZipFile(dist_dir / "s2and-0.0.0-cp313-cp313-musllinux_1_2_x86_64.whl", "w") as wheel:
        for path, payload in {**WHEEL_MEMBERS, **licensed}.items():
            if path != NATIVE_PATH:
                wheel.writestr(path, payload)
        wheel.writestr("s2and/_s2and_rust.cpython-313-x86_64-linux-musl.so", b"native")
        wheel.writestr("s2and.libs/libgcc_s-f685abf1.so.1", b"runtime")
        wheel.writestr("s2and.libs/libstdc++-1b2c0963.so.6", b"runtime")

    verify_production_model_distributions(dist_dir=dist_dir, source_root=source_root)

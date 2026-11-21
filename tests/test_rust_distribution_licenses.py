"""License checks reject omitted and altered notices in built archives."""

import io
import tarfile
import zipfile

import pytest

from scripts.verification.verify_rust_distribution_licenses import LICENSE_PATHS, verify_archive


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("damage", [None, "missing", "changed"])
def test_verify_rust_distribution_notices(tmp_path, kind, damage):
    source = tmp_path / "source"
    notices = {path: f"license text: {path}".encode() for path in LICENSE_PATHS}
    for path, payload in notices.items():
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    if damage == "missing":
        del notices[LICENSE_PATHS[-1]]
    elif damage == "changed":
        notices[LICENSE_PATHS[-1]] = b"truncated notice"
    archive = tmp_path / ("s2and_rust.whl" if kind == "wheel" else "s2and_rust.tar.gz")
    if kind == "wheel":
        with zipfile.ZipFile(archive, "w") as wheel:
            wheel.writestr("s2and_rust-1.0.dist-info/METADATA", "Name: s2and-rust\n")
            for path, payload in notices.items():
                wheel.writestr(f"s2and_rust-1.0.dist-info/licenses/{path}", payload)
    else:
        with tarfile.open(archive, "w:gz") as sdist:
            for path, payload in notices.items():
                member = tarfile.TarInfo(f"s2and_rust-1.0/{path}")
                member.size = len(payload)
                sdist.addfile(member, io.BytesIO(payload))
    if damage:
        with pytest.raises(ValueError, match="license notice"):
            verify_archive(archive, source)
    else:
        verify_archive(archive, source)

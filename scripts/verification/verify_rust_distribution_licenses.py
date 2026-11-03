"""Verify that Rust wheels and source distributions retain their license notices."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

LICENSE_PATHS = (
    "LICENSE",
    "LICENSE-DATA",
    "s2and_rust/THIRD_PARTY_NOTICES.md",
    "s2and_rust/vendor/cld2/LICENSE-APACHE.txt",
    "s2and_rust/vendor/cld2/UNLICENSE.txt",
)


def verify_archive(archive: Path, source_root: Path) -> None:
    """Require every notice byte-for-byte in one built distribution."""
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as wheel:
            roots = [name.removesuffix("METADATA") for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(roots) != 1:
                raise ValueError(f"{archive}: expected one wheel metadata directory")
            expected = {f"{roots[0]}licenses/{path}": (source_root / path).read_bytes() for path in LICENSE_PATHS}
            for name, payload in expected.items():
                if name not in wheel.namelist() or wheel.read(name) != payload:
                    raise ValueError(f"{archive}: missing or changed license notice {name}")
    else:
        with tarfile.open(archive, "r:gz") as sdist:
            for path in LICENSE_PATHS:
                matches = [member for member in sdist.getmembers() if member.name.split("/", 1)[-1] == path]
                if len(matches) != 1 or not matches[0].isfile():
                    raise ValueError(f"{archive}: missing or duplicate license notice {path}")
                source = sdist.extractfile(matches[0])
                if source is None or source.read() != (source_root / path).read_bytes():
                    raise ValueError(f"{archive}: changed license notice {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    args = parser.parse_args()
    archives = sorted([*args.dist_dir.glob("*.whl"), *args.dist_dir.glob("*.tar.gz")])
    if not archives:
        raise ValueError(f"No distributions found in {args.dist_dir}")
    for archive in archives:
        verify_archive(archive, args.source_root)
        print(f"Verified license notices: {archive}")


if __name__ == "__main__":
    main()

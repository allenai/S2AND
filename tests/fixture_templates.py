"""Per-test copies of fixture trees that a module builds once."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def copy_template(template: Path, destination: Path) -> None:
    """Copy a template tree into a test directory and rebase recorded absolute paths.

    Arrow conversion records absolute source paths in JSON. The rebase makes each copy
    equal to a tree that the test builds itself, with no reference into the shared template.

    Args:
        template: Module-scope directory that no test writes.
        destination: Test directory that receives the copy at the same relative layout.
    """

    shutil.copytree(template, destination, dirs_exist_ok=True)
    old, new = (json.dumps(str(path))[1:-1].encode() for path in (template, destination))
    for path in destination.rglob("*.json"):
        data = path.read_bytes()
        if old in data:
            path.write_bytes(data.replace(old, new))

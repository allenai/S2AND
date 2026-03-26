from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _rust_suite.calibrate_cmd import run_calibration  # type: ignore  # noqa: E402


def main(argv: list[str]) -> int:
    return run_calibration(argv, profile_key="rust_batch")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

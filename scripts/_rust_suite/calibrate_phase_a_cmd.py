from __future__ import annotations

import sys

from _rust_suite.calibrate_cmd import run_calibration


def main(argv: list[str]) -> int:
    return run_calibration(argv, profile_key="phase_a")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

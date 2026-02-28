import argparse
import datetime as dt
import os
import pickle
from pathlib import Path

import orjson

from s2and.consts import NAME_COUNTS_PATH, PROJECT_ROOT_PATH
from s2and.file_cache import cached_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Python name-count pickle into Rust-ingest JSON artifact format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=NAME_COUNTS_PATH,
        help="Path to input name_counts pickle (defaults to s2and.consts.NAME_COUNTS_PATH).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(PROJECT_ROOT_PATH) / "scratch" / "name_counts_rust.json"),
        help=(
            "Path to output JSON artifact for Rust from_json_paths ingest "
            "(use --output data/... explicitly if desired)."
        ),
    )
    parser.add_argument(
        "--normalization-version",
        type=str,
        default=os.environ.get("S2AND_NORMALIZATION_VERSION", "legacy_compat"),
        help="Normalization policy version tag stored in artifact metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = cached_path(args.input)
    with open(input_path, "rb") as in_file:
        first_dict, last_dict, first_last_dict, last_first_initial_dict = pickle.load(in_file)

    payload = {
        "normalization_version": args.normalization_version,
        "generation_provenance": {
            "script": "scripts/export_name_counts_for_rust.py",
            "generated_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
            "source_path": os.fspath(input_path),
        },
        "first_dict": first_dict,
        "last_dict": last_dict,
        "first_last_dict": first_last_dict,
        "last_first_initial_dict": last_first_initial_dict,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out_file:
        out_file.write(orjson.dumps(payload))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(
        orjson.dumps(
            {
                "input_path": os.fspath(input_path),
                "output_path": str(output_path),
                "first_dict": len(first_dict),
                "last_dict": len(last_dict),
                "first_last_dict": len(first_last_dict),
                "last_first_initial_dict": len(last_first_initial_dict),
                "normalization_version": args.normalization_version,
                "output_size_mb": round(size_mb, 2),
            },
            option=orjson.OPT_INDENT_2,
        ).decode("utf-8")
    )


if __name__ == "__main__":
    main()

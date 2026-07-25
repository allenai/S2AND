"""Outcome tests for canonical ORCID prefix-count generation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from s2and.consts import NORMALIZATION_VERSION, PROJECT_ROOT_PATH
from s2and.orcid_prefix_counts import (
    ORCID_PREFIX_ARTIFACT_SCHEMA_VERSION,
    ORCID_PREFIX_DATA_FILENAME,
    ORCID_PREFIX_MANIFEST_FILENAME,
    ORCID_PREFIX_PAIR_KEY_SEMANTICS,
    load_canonical_orcid_prefix_counts,
)
from scripts.production.counts import generate_orcid_name_prefix_counts as generator

ORCID_1 = "0000-0000-0000-0001"
ORCID_2 = "0000-0000-0000-0002"
NAME_TUPLES_PATH = Path(PROJECT_ROOT_PATH) / "s2and" / "data" / "s2and_name_tuples_canonical.txt"
NAME_TUPLES_SHA256 = hashlib.sha256(NAME_TUPLES_PATH.read_bytes()).hexdigest()
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_module_entrypoint_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.production.counts.generate_orcid_name_prefix_counts", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--expected-name-tuples-sha256" in completed.stdout


def _tuple_args() -> list[str]:
    return [
        "--name-tuples-path",
        str(NAME_TUPLES_PATH),
        "--expected-name-tuples-sha256",
        NAME_TUPLES_SHA256,
    ]


def _write_rows(path: Path, rows: object) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _fixture_args(tmp_path: Path, rows: object, *, output_name: str = "publication") -> list[str]:
    return [
        "--input-json",
        str(_write_rows(tmp_path / f"{output_name}-rows.json", rows)),
        "--output-dir",
        str(tmp_path / output_name),
        "--source-snapshot-id",
        "fixture-snapshot",
        *_tuple_args(),
    ]


def test_module_entrypoint_publishes_fixture(tmp_path: Path) -> None:
    args = _fixture_args(
        tmp_path,
        [
            {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
            {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
        ],
    )
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.production.counts.generate_orcid_name_prefix_counts", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "publication" / ORCID_PREFIX_MANIFEST_FILENAME).is_file()


def _write_guardrails(path: Path, **changes: int) -> Path:
    values = {
        "min_source_rows": 1,
        "max_source_rows": 100,
        "max_names_per_orcid": 100,
        "max_pair_keys": 1_000_000,
        "min_orcid_pair_keys": 1,
        **changes,
    }
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def _publish_fixture(tmp_path: Path) -> Path:
    rows = [
        {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
    ]
    assert generator.main(_fixture_args(tmp_path, rows)) == 0
    return tmp_path / "publication"


def test_fixture_publishes_one_manifest_authority_and_loads_immutably(tmp_path: Path) -> None:
    output_dir = _publish_fixture(tmp_path)

    assert {path.name for path in output_dir.iterdir()} == {
        ORCID_PREFIX_DATA_FILENAME,
        ORCID_PREFIX_MANIFEST_FILENAME,
    }
    manifest_payload = (output_dir / ORCID_PREFIX_MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_payload)
    loaded = load_canonical_orcid_prefix_counts(output_dir)
    assert manifest["schema_version"] == ORCID_PREFIX_ARTIFACT_SCHEMA_VERSION
    assert manifest["normalization_version"] == NORMALIZATION_VERSION
    assert manifest["pair_key_semantics"] == ORCID_PREFIX_PAIR_KEY_SEMANTICS
    assert manifest["source_kind"].startswith("fixture:")
    assert manifest["name_tuples_sha256"] == NAME_TUPLES_SHA256
    assert loaded.data_sha256 == hashlib.sha256((output_dir / ORCID_PREFIX_DATA_FILENAME).read_bytes()).hexdigest()
    assert loaded.manifest_sha256 == hashlib.sha256(manifest_payload).hexdigest()
    assert loaded.name_tuples_sha256 == NAME_TUPLES_SHA256
    with pytest.raises(TypeError):
        loaded.counts["al"] = {}  # type: ignore[index]


def test_existing_or_empty_publication_is_rejected_without_partial_output(tmp_path: Path) -> None:
    output_dir = _publish_fixture(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        generator.main(
            _fixture_args(
                tmp_path,
                [{"orcid": ORCID_1, "first_name": "Alice", "middle": None}],
                output_name=output_dir.name,
            )
        )

    empty_output = tmp_path / "empty"
    with pytest.raises(RuntimeError, match="zero source rows"):
        generator.main(_fixture_args(tmp_path, [], output_name=empty_output.name))
    assert not empty_output.exists()


def test_full_dry_run_uses_one_guardrail_authority_and_max_plus_one_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [
        "--run-full",
        "--output-dir",
        str(tmp_path / "output"),
        "--source-snapshot-id",
        "warehouse-snapshot",
        *_tuple_args(),
        "--dry-run",
    ]
    with pytest.raises(ValueError, match="--guardrails-json"):
        generator.main(args)

    guardrails = _write_guardrails(tmp_path / "guardrails.json", max_source_rows=17)
    assert generator.main([*args, "--guardrails-json", str(guardrails)]) == 0
    assert "limit 18" in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def test_full_run_rejects_invalid_expansion_guard_before_warehouse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guardrails = _write_guardrails(tmp_path / "guardrails.json", max_names_per_orcid=1)
    monkeypatch.setattr(
        generator,
        "_load_warehouse_rows",
        lambda *_args: pytest.fail("warehouse accessed before guardrail validation"),
    )

    with pytest.raises(ValueError, match="max_names_per_orcid must be at least 2"):
        generator.main(
            [
                "--run-full",
                "--output-dir",
                str(tmp_path / "output"),
                "--source-snapshot-id",
                "warehouse-snapshot",
                "--guardrails-json",
                str(guardrails),
                *_tuple_args(),
            ]
        )


def test_name_tuple_digest_is_explicitly_bound_before_publication(tmp_path: Path) -> None:
    args = _fixture_args(
        tmp_path,
        [{"orcid": ORCID_1, "first_name": "Alice", "middle": None}],
    )
    args[args.index(NAME_TUPLES_SHA256)] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        generator.main(args)
    assert not (tmp_path / "publication").exists()


def test_builder_deduplicates_rows_and_hashes_selected_content() -> None:
    rows = [
        {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
    ]
    counts, metrics, digest = generator.build_prefix_counts_from_sorted_rows(
        rows,
        [],
        min_orcid_count=1,
    )
    duplicate_counts, duplicate_metrics, duplicate_digest = generator.build_prefix_counts_from_sorted_rows(
        [*rows, rows[-1]],
        [],
        min_orcid_count=1,
    )

    assert counts == duplicate_counts
    assert digest == duplicate_digest
    assert metrics["source_rows"] == 2
    assert duplicate_metrics["source_rows"] == 3
    assert metrics["orcid_pair_keys_after_threshold"] > 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_names_per_orcid": 1}, "at least 2"),
        ({"max_names_per_orcid": 2}, "more than max_names_per_orcid=2"),
        ({"max_source_rows": 1}, "max_source_rows=1"),
        ({"max_pair_keys": 1}, "max_pair_keys=1"),
    ],
)
def test_builder_enforces_only_live_expansion_bounds(kwargs: dict[str, int], match: str) -> None:
    rows = [
        {"orcid": ORCID_1, "first_name": "Alice", "middle": None},
        {"orcid": ORCID_1, "first_name": "Amy", "middle": None},
        {"orcid": ORCID_1, "first_name": "Ava", "middle": None},
    ]
    with pytest.raises(ValueError, match=match):
        generator.build_prefix_counts_from_sorted_rows(rows, [], min_orcid_count=1, **kwargs)


@pytest.mark.parametrize("tamper", ["data", "cardinality", "source"])
def test_canonical_loader_rejects_tampering(tmp_path: Path, tamper: str) -> None:
    output_dir = _publish_fixture(tmp_path)
    data_path = output_dir / ORCID_PREFIX_DATA_FILENAME
    manifest_path = output_dir / ORCID_PREFIX_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    if tamper == "data":
        data_path.write_bytes(data_path.read_bytes() + b" ")
    elif tamper == "cardinality":
        manifest["metrics"]["output_pair_keys"] += 1
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest["source_kind"] = ""
        manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        load_canonical_orcid_prefix_counts(output_dir)


def test_source_orcid_key_and_query_order_match() -> None:
    raw = "prefix 0000‐0000‐0000‐0001 suffix"
    match = re.search(generator._CANONICAL_SOURCE_ORCID_SQL_PATTERN, raw, flags=re.IGNORECASE)
    assert match is not None
    compact = re.sub(generator._ORCID_DASH_SQL_PATTERN, "", match.group()).upper()
    warehouse_key = f"{compact[:4]}-{compact[4:8]}-{compact[8:12]}-{compact[12:]}"
    assert generator._canonical_source_orcid(match.group()) == warehouse_key
    assert "order by orcid nulls last" in generator._warehouse_query(10)
    assert generator._warehouse_query(10).endswith("limit 11\n")


def test_prefix_pair_lookup_is_order_independent() -> None:
    expected = generator.prefix_pairs_for_names("alice", "amy")
    assert expected == generator.prefix_pairs_for_names("amy", "alice")
    assert all(left < right for left, right in expected)
    assert not generator.prefix_pairs_for_names("alice", "bob")

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from s2and.consts import _PACKAGE_DATA_DIR
from s2and.name_tuple_artifact import load_name_tuple_artifact
from scripts.production import generate_canonical_name_tuples


@pytest.mark.parametrize("argv", [[], ["--source", "source.txt"], ["--output", "output.txt"]])
def test_canonical_name_tuple_generator_requires_explicit_paths(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_canonical_name_tuples.py", *argv])

    with pytest.raises(SystemExit) as exc_info:
        generate_canonical_name_tuples.main()

    assert exc_info.value.code == 2


def test_checked_in_canonical_name_tuples_load_directly() -> None:
    data_path = Path(_PACKAGE_DATA_DIR) / "s2and_name_tuples_canonical.txt"
    artifact = load_name_tuple_artifact(data_path)

    assert len(artifact.pairs) == 5027
    assert artifact.data_sha256 == hashlib.sha256(data_path.read_bytes()).hexdigest()


def test_checked_in_manual_adjudication_matches_promoted_aliases() -> None:
    evidence_path = (
        Path(__file__).resolve().parents[1] / "docs" / "release_evidence" / "name_tuple_legacy_adjudication_v1.jsonl.gz"
    )
    records = [json.loads(line) for line in gzip.decompress(evidence_path.read_bytes()).decode("utf-8").splitlines()]
    accepted = {(record["left"], record["right"]) for record in records if record["label"] == "accept"}
    excluded = {(record["left"], record["right"]) for record in records if record["label"] != "accept"}
    artifact = load_name_tuple_artifact(Path(_PACKAGE_DATA_DIR) / "s2and_name_tuples_canonical.txt")

    assert len(records) == 2266
    assert len(accepted) == 1343
    assert len(excluded) == 923
    assert accepted <= artifact.pairs
    assert artifact.pairs.isdisjoint(excluded)


def test_custom_artifact_loads_directly(tmp_path: Path) -> None:
    artifact_path = tmp_path / "aliases.txt"
    artifact_path.write_bytes(b"alice,ally\n")

    artifact = load_name_tuple_artifact(artifact_path)

    assert artifact.pairs == frozenset({("alice", "ally")})
    assert artifact.data_sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()


def test_loader_reads_data_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_path = tmp_path / "aliases.txt"
    artifact_path.write_bytes(b"alice,ally\n")
    path_type = type(artifact_path)
    read_bytes = path_type.read_bytes
    reads: list[Path] = []

    def recording_read_bytes(path: Path) -> bytes:
        reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(path_type, "read_bytes", recording_read_bytes)

    assert load_name_tuple_artifact(artifact_path).pairs == frozenset({("alice", "ally")})
    assert reads == [artifact_path]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"ally,alice\n", "name_a must be lexicographically less than name_b"),
        (b"alice,ally\nalice,ally\n", "rows must be unique and sorted"),
        (b"alice,alice\n", "identity"),
        (b"ann,anna\n", "prefix-compatible"),
        (b"Alice,ally\n", "noncanonical"),
        (b"alice\n", "two nonempty fields"),
        (b"\xff\n", "not valid UTF-8"),
    ],
)
def test_loader_rejects_semantically_invalid_rows(tmp_path: Path, payload: bytes, message: str) -> None:
    artifact_path = tmp_path / "aliases.txt"
    artifact_path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        load_name_tuple_artifact(artifact_path)


def test_generator_atomically_publishes_one_loadable_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "source.txt"
    output_path = tmp_path / "aliases.txt"
    source_path.write_text("Alice,Ally\nRobert,Bob\n", encoding="utf-8")
    replaced_destinations: list[Path] = []
    real_replace = os.replace

    def recording_replace(source, destination) -> None:
        real_replace(source, destination)
        replaced_destinations.append(Path(destination))

    monkeypatch.setattr(generate_canonical_name_tuples.os, "replace", recording_replace)
    summary = generate_canonical_name_tuples.regenerate(str(source_path), str(output_path))

    assert replaced_destinations == [output_path]
    assert summary["data"]["pair_count"] == 2
    assert load_name_tuple_artifact(output_path).pairs == frozenset({("alice", "ally"), ("bob", "robert")})


def test_generator_reports_filtered_and_duplicate_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "source.txt"
    output_path = tmp_path / "aliases.txt"
    source_path.write_text(
        "Alice,Ally\nally,alice\nALICE,ALLY\nAlice,Alice\nAnn,Anna\n',Alice\n",
        encoding="utf-8",
    )

    summary = generate_canonical_name_tuples.regenerate(str(source_path), str(output_path))

    assert summary["generation_counts"] == {
        "input_pair_count": 6,
        "dropped_identity": 1,
        "dropped_prefix_compatible": 1,
        "dropped_empty": 1,
        "dropped_duplicate_canonical": 2,
    }
    assert summary["data"]["pair_count"] == 1


def test_concurrent_generators_leave_one_loadable_artifact(tmp_path: Path) -> None:
    sources = [tmp_path / "source-a.txt", tmp_path / "source-b.txt"]
    sources[0].write_text("Alice,Ally\n", encoding="utf-8")
    sources[1].write_text("Robert,Bob\n", encoding="utf-8")
    output_path = tmp_path / "aliases.txt"
    barrier = threading.Barrier(2)

    def regenerate(source: Path) -> dict:
        barrier.wait(timeout=5)
        return generate_canonical_name_tuples.regenerate(str(source), str(output_path))

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(regenerate, sources))

    artifact = load_name_tuple_artifact(output_path)
    assert artifact.pairs in (
        frozenset({("alice", "ally")}),
        frozenset({("bob", "robert")}),
    )

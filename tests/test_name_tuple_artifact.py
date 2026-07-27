from __future__ import annotations

import gzip
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from s2and.consts import _PACKAGE_DATA_DIR
from s2and.name_tuple_artifact import (
    NAME_TUPLE_ARTIFACT_SEMANTICS,
    build_name_tuple_artifact_metadata,
    load_name_tuple_artifact,
)
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


def _write_artifact(
    path: Path,
    data: bytes = b"alice,ally\n",
    *,
    pair_count: int = 1,
) -> dict:
    metadata = build_name_tuple_artifact_metadata(
        source_filename="source.txt",
        source_bytes=b"Alice,Ally\n",
        data_filename=path.name,
        data_bytes=data,
        pair_count=pair_count,
        generated_at="2026-07-10T00:00:00+00:00",
        input_pair_count=1,
        dropped_identity=0,
        dropped_prefix_compatible=0,
        dropped_empty=0,
        dropped_duplicate_canonical=0,
    )
    path.write_bytes(data)
    Path(str(path) + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _set_pair_count(metadata: dict, pair_count: int) -> None:
    metadata["data"]["pair_count"] = pair_count
    metadata["generation_counts"]["input_pair_count"] = pair_count


def test_checked_in_canonical_name_tuple_hash_matches_unchanged_data() -> None:
    artifact = load_name_tuple_artifact(Path(_PACKAGE_DATA_DIR) / "s2and_name_tuples_canonical.txt")
    metadata = json.loads(
        (Path(_PACKAGE_DATA_DIR) / "s2and_name_tuples_canonical.txt.meta.json").read_text(encoding="utf-8")
    )

    assert len(artifact.pairs) == 5027
    assert artifact.data_sha256 == "b21638351149389c57eca547b0f79c80084e56ad273f31e778cb1db1866945a8"
    assert metadata["generation_counts"]["dropped_duplicate_canonical"] == 3768


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


def test_custom_artifact_requires_sidecar_and_rejects_data_tamper(tmp_path: Path) -> None:
    artifact_path = tmp_path / "aliases.txt"
    artifact_path.write_text("alice,ally\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="aliases.txt.meta.json"):
        load_name_tuple_artifact(artifact_path)

    _write_artifact(artifact_path)
    artifact = load_name_tuple_artifact(artifact_path)
    assert artifact.pairs == frozenset({("alice", "ally")})

    artifact_path.write_bytes(b"alica,ally\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_name_tuple_artifact(artifact_path)


def test_loader_reads_data_and_metadata_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact_path = tmp_path / "aliases.txt"
    metadata_path = Path(str(artifact_path) + ".meta.json")
    _write_artifact(artifact_path)
    path_type = type(artifact_path)
    read_bytes = path_type.read_bytes
    reads: list[Path] = []

    def recording_read_bytes(path: Path) -> bytes:
        reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(path_type, "read_bytes", recording_read_bytes)

    assert load_name_tuple_artifact(artifact_path).pairs == frozenset({("alice", "ally")})
    assert reads == [metadata_path, artifact_path]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.update(schema_version="unknown"), "schema_version"),
        (lambda metadata: _set_pair_count(metadata, 4), "pair_count mismatch"),
        (lambda metadata: metadata["semantics"].update(directionality="one_way"), "unsupported semantics"),
        (
            lambda metadata: metadata["generation_counts"].update(input_pair_count=1 << 64),
            "unsigned 64-bit integer",
        ),
        (
            lambda metadata: metadata["generation_counts"].pop("dropped_duplicate_canonical"),
            "dropped_duplicate_canonical",
        ),
        (
            lambda metadata: metadata["generation_counts"].update(input_pair_count=2),
            "do not account for every input pair",
        ),
    ],
)
def test_metadata_contract_rejects_schema_cardinality_and_semantic_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    artifact_path = tmp_path / "aliases.txt"
    metadata = _write_artifact(artifact_path)
    mutation(metadata)
    Path(str(artifact_path) + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_name_tuple_artifact(artifact_path)


def test_loader_rejects_noncanonical_field_order_even_with_matching_metadata(tmp_path: Path) -> None:
    artifact_path = tmp_path / "aliases.txt"
    _write_artifact(
        artifact_path,
        b"ally,alice\n",
        pair_count=1,
    )

    with pytest.raises(ValueError, match="name_a must be lexicographically less than name_b"):
        load_name_tuple_artifact(artifact_path)


def test_generator_publishes_metadata_last_and_output_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "source.txt"
    output_path = tmp_path / "aliases.txt"
    source_path.write_text("Alice,Ally\nRobert,Bob\n", encoding="utf-8")
    replaced_destinations: list[Path] = []
    real_replace = os.replace

    def recording_replace(source, destination) -> None:
        real_replace(source, destination)
        replaced_destinations.append(Path(destination))

    monkeypatch.setattr(generate_canonical_name_tuples.os, "replace", recording_replace)
    metadata = generate_canonical_name_tuples.regenerate(str(source_path), str(output_path))

    assert replaced_destinations == [output_path, Path(str(output_path) + ".meta.json")]
    assert metadata["semantics"] == NAME_TUPLE_ARTIFACT_SEMANTICS
    assert load_name_tuple_artifact(output_path).pairs == frozenset({("alice", "ally"), ("bob", "robert")})


def test_generator_audits_each_filtered_and_duplicate_input_row(tmp_path: Path) -> None:
    source_path = tmp_path / "source.txt"
    output_path = tmp_path / "aliases.txt"
    source_path.write_text(
        "Alice,Ally\nally,alice\nALICE,ALLY\nAlice,Alice\nAnn,Anna\n',Alice\n",
        encoding="utf-8",
    )

    metadata = generate_canonical_name_tuples.regenerate(str(source_path), str(output_path))

    assert metadata["generation_counts"] == {
        "input_pair_count": 6,
        "dropped_identity": 1,
        "dropped_prefix_compatible": 1,
        "dropped_empty": 1,
        "dropped_duplicate_canonical": 2,
    }
    assert metadata["data"]["pair_count"] == 1


def test_concurrent_generators_leave_one_complete_loadable_artifact(tmp_path: Path) -> None:
    sources = [tmp_path / "source-a.txt", tmp_path / "source-b.txt"]
    sources[0].write_text("Alice,Ally\n", encoding="utf-8")
    sources[1].write_text("Robert,Bob\n", encoding="utf-8")
    output_path = tmp_path / "aliases.txt"
    barrier = threading.Barrier(2)

    def regenerate(source: Path) -> dict:
        barrier.wait(timeout=5)
        return generate_canonical_name_tuples.regenerate(str(source), str(output_path))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(regenerate, source) for source in sources]
        metadata = [future.result() for future in futures]

    artifact = load_name_tuple_artifact(output_path)
    assert artifact.pairs in (
        frozenset({("alice", "ally")}),
        frozenset({("bob", "robert")}),
    )
    assert artifact.data_sha256 in {item["data"]["sha256"] for item in metadata}

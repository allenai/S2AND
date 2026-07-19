from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pytest

from s2and.consts import _PACKAGE_DATA_DIR
from s2and.name_tuple_artifact import (
    NAME_TUPLE_ARTIFACT_SEMANTICS,
    build_name_tuple_artifact_metadata,
    load_name_tuple_artifact,
    load_packaged_name_tuple_artifact,
)
from scripts.production import generate_canonical_name_tuples
from tests.helpers import import_s2and_rust

HAS_RUST_IDENTITY, RUST_MODULE = import_s2and_rust()


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
    )
    path.write_bytes(data)
    Path(str(path) + ".meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def test_checked_in_canonical_name_tuple_identity_matches_unchanged_data() -> None:
    artifact = load_name_tuple_artifact(Path(_PACKAGE_DATA_DIR) / "s2and_name_tuples_canonical.txt")

    assert len(artifact.pairs) == 3684
    assert artifact.identity.pair_count == 3684
    assert artifact.identity.data_sha256 == "a6eafc93ee5af6c883c6d9dfa8abc2c26c88427ef2428fa4b99a681b0eaefb5b"
    assert artifact.identity.source_sha256 == "68f0c70fcf138d08656eaa39e485ba0d513ce6da6ce1164a3cc8bb2c680430f2"


def test_packaged_artifact_cache_is_immutable_and_avoids_rehashing() -> None:
    load_packaged_name_tuple_artifact.cache_clear()

    first = load_packaged_name_tuple_artifact()
    second = load_packaged_name_tuple_artifact()

    assert first is second
    assert isinstance(first.pairs, frozenset)
    assert load_packaged_name_tuple_artifact.cache_info().hits == 1


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.update(schema_version="unknown"), "schema_version"),
        (lambda metadata: metadata["data"].update(pair_count=4), "pair_count mismatch"),
        (lambda metadata: metadata["semantics"].update(directionality="one_way"), "unsupported semantics"),
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
    assert artifact.identity.source_sha256 in {item["source"]["sha256"] for item in metadata}


@pytest.mark.skipif(not HAS_RUST_IDENTITY, reason="current Rust extension lacks tuple identity reader")
def test_python_and_rust_validate_the_same_artifact_identity(tmp_path: Path) -> None:
    artifact_path = tmp_path / "aliases.txt"
    _write_artifact(artifact_path)

    python_identity = asdict(load_name_tuple_artifact(artifact_path).identity)
    rust_identity = dict(RUST_MODULE.read_name_tuple_artifact_identity(str(artifact_path)))

    assert rust_identity == python_identity

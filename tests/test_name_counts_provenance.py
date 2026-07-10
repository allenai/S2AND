"""Canonical name-count provenance enforcement tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import s2and.data as data_module
from s2and.data import ANDData
from s2and.incremental_linking.feature_block import write_name_counts_index
from scripts.production.counts.generate_name_counts import publish_name_counts
from tests.helpers import tiny_name_counts, tiny_name_counts_provenance


def _publish_fixture(tmp_path: Path) -> Path:
    publish_name_counts(
        ({"ada": 3}, {"lovelace": 5}, {"ada lovelace": 2}, {"lovelace a": 7}),
        output_dir=tmp_path,
        source_snapshot_id="fixture-snapshot",
        source_kind="fixture:test",
        query_digest="1" * 64,
        row_metrics=cast(
            Any,
            {
                "source_row_count": 1,
                "selected_row_count": 1,
                "selected_rows_sha256": "2" * 64,
                "rejected_row_count": 0,
            },
        ),
        overwrite=False,
    )
    return tmp_path / "name_counts" / "manifest.json"


def test_verified_name_counts_round_trip_into_index_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_manifest = _publish_fixture(tmp_path / "source")
    counts, provenance = data_module._read_name_counts_artifact(source_manifest)
    monkeypatch.setattr(data_module, "_load_name_counts_artifact", lambda: (counts, provenance))

    index_path, _metrics = write_name_counts_index(tmp_path / "index")
    index_manifest = json.loads((Path(index_path) / "manifest.json").read_text(encoding="utf-8"))
    assert index_manifest["normalization_version"] == provenance["normalization_version"]
    assert index_manifest["source_provenance"] == provenance


def test_tampered_pickle_is_rejected_before_unpickling(tmp_path: Path) -> None:
    source_manifest = _publish_fixture(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    pickle_path = source_manifest.parent / manifest["files"]["pickle"]
    payload = bytearray(pickle_path.read_bytes())
    payload[-1] ^= 1
    pickle_path.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        data_module._read_name_counts_artifact(source_manifest)


def test_tampered_provenance_is_rejected_before_parsing(tmp_path: Path) -> None:
    source_manifest = _publish_fixture(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    provenance_path = source_manifest.parent / manifest["files"]["provenance"]
    payload = bytearray(provenance_path.read_bytes())
    payload[-2] ^= 1
    provenance_path.write_bytes(payload)

    with pytest.raises(ValueError, match="provenance SHA-256 mismatch"):
        data_module._read_name_counts_artifact(source_manifest)


def test_bare_name_count_mapping_requires_explicit_provenance() -> None:
    with pytest.raises(ValueError, match="name_counts_provenance_v1"):
        ANDData(
            signatures={},
            papers={},
            name="bare-counts",
            mode="inference",
            load_name_counts={
                "first_dict": {},
                "last_dict": {},
                "first_last_dict": {},
                "last_first_initial_dict": {},
            },
            preprocess=False,
        )


def test_package_name_count_cache_uses_shared_read_only_views_without_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_counts = (
        {"ada": 3},
        {"lovelace": 5},
        {"ada lovelace": 2},
        {"lovelace a": 7},
    )
    raw_provenance = {
        **tiny_name_counts_provenance(),
        "cardinalities": {"first": 1},
    }
    monkeypatch.setattr(data_module, "_NAME_COUNTS_CACHE", None)
    monkeypatch.setattr(data_module, "_NAME_COUNTS_PROVENANCE_CACHE", None)
    monkeypatch.setattr(
        data_module,
        "_read_name_counts_artifact",
        lambda manifest_path: (raw_counts, raw_provenance),
    )

    first_counts, first_provenance = data_module._load_name_counts_artifact()
    second_counts, second_provenance = data_module._load_name_counts_artifact()

    assert first_counts is second_counts
    assert first_provenance is second_provenance
    for index, (raw_mapping, exposed_mapping) in enumerate(zip(raw_counts, first_counts, strict=True)):
        with pytest.raises(TypeError):
            cast(Any, exposed_mapping)["mutation"] = 1
        no_copy_key = f"no-copy-{index}"
        raw_mapping[no_copy_key] = index
        assert exposed_mapping[no_copy_key] == index

    with pytest.raises(TypeError):
        cast(Any, first_provenance)["generation_id"] = "mutated"
    with pytest.raises(TypeError):
        first_provenance["cardinalities"]["first"] = 2
    raw_provenance["generation_id"] = "mutated-after-cache"
    raw_provenance["cardinalities"]["first"] = 2
    assert first_provenance["generation_id"] == "test-tiny-name-counts"
    assert first_provenance["cardinalities"]["first"] == 1


def test_anddata_provenance_is_read_only_and_stable_after_source_mutation() -> None:
    load_name_counts = tiny_name_counts()
    source_provenance = load_name_counts["provenance"]
    source_provenance["cardinalities"] = {"first": len(load_name_counts["first_dict"])}
    dataset = ANDData(
        signatures={},
        papers={},
        name="read-only-name-count-provenance",
        mode="inference",
        load_name_counts=load_name_counts,
        name_tuples=set(),
        preprocess=False,
    )
    assert dataset.name_counts_provenance is not None

    with pytest.raises(TypeError):
        cast(Any, dataset.name_counts_provenance)["generation_id"] = "mutated"
    with pytest.raises(TypeError):
        dataset.name_counts_provenance["cardinalities"]["first"] = 0

    source_provenance["generation_id"] = "mutated-source"
    source_provenance["cardinalities"]["first"] = 0
    assert dataset.name_counts_provenance["generation_id"] == "test-tiny-name-counts"
    assert dataset.name_counts_provenance["cardinalities"]["first"] == 3


@pytest.mark.parametrize(
    "field_name",
    ("pickle_sha256", "source_query_sha256", "selected_rows_sha256"),
)
@pytest.mark.parametrize("invalid_digest", ("A" * 64, "g" * 64))
def test_name_count_provenance_rejects_non_lowercase_hex_sha256(
    field_name: str,
    invalid_digest: str,
) -> None:
    provenance = tiny_name_counts_provenance()
    provenance[field_name] = invalid_digest

    with pytest.raises(ValueError, match="requires a lowercase hexadecimal SHA-256"):
        data_module._validated_name_counts_provenance(provenance, context="test provenance")


@pytest.mark.parametrize("field_name", ("pickle_sha256", "provenance_sha256"))
@pytest.mark.parametrize("invalid_digest", ("A" * 64, "g" * 64))
def test_name_count_manifest_rejects_non_lowercase_hex_sha256(
    tmp_path: Path,
    field_name: str,
    invalid_digest: str,
) -> None:
    source_manifest = _publish_fixture(tmp_path)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest[field_name] = invalid_digest
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a lowercase hexadecimal SHA-256"):
        data_module._read_name_counts_artifact(source_manifest)

"""Public email hashing preserves feature evidence and staged artifact integrity."""

from __future__ import annotations

import hashlib
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from s2and import feature_port
from s2and._sha256 import sha256_file
from s2and.featurizer import NUM_FEATURES
from s2and.incremental_linking import linker_pairwise
from s2and.incremental_linking.feature_block_arrow import write_arrow_batch_lookup_index
from s2and.text import email_prefix_suffix
from scripts.production.data.hash_release_emails import hash_feature_email, hash_release_emails
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset


@pytest.mark.parametrize(
    "value",
    [None, "", "J.Smith@MIT.EDU.", ".j.smith@mit.edu", "other@mit.edu", "a@b@c", "a @b", "a@", "@b"],
)
def test_hashing_preserves_normalized_email_components(value: str | None) -> None:
    result = hash_feature_email(value)
    before = email_prefix_suffix(value) if value else (None, None)
    after = email_prefix_suffix(result) if result else (None, None)
    if before[0] is None:
        assert after == (None, None)
    else:
        assert before[1] is not None
        assert after == (hashlib.sha256(before[0].encode()).hexdigest(), hashlib.sha256(before[1].encode()).hexdigest())
    assert hash_feature_email(result) == result


def test_existing_component_and_malformed_hashes_remain_unchanged() -> None:
    for value in ("a" * 64 + "@" + "b" * 64, "hmac-sha256:" + "c" * 64, "sha256:" + "d" * 64):
        assert hash_feature_email(value) == value


def test_release_hashing_refreshes_indexes_and_manifests_without_mutating_inputs(tmp_path: Path) -> None:
    original = tmp_path / "original.arrow"
    table = pa.table({"signature_id": ["s1", "s2"], "author_email": ["J.Smith@MIT.edu.", "a@b@c"]})
    with pa.OSFile(str(original), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    dataset = tmp_path / "dataset"
    giant = dataset / "datasets/a_khan"
    giant.mkdir(parents=True)
    arrow = giant / "signatures.arrow"
    os.link(original, arrow)
    index = giant / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(arrow, index, key_column="signature_id")
    original_index = tmp_path / "original_index.bin"
    os.link(index, original_index)
    original_arrow_sha, original_index_sha = sha256_file(original), sha256_file(original_index)
    manifest = {"paths": {"signatures": arrow.name, "signatures_batch_index": index.name}, "files": {}}
    (giant / "manifest.json").write_text(json.dumps(manifest))
    (dataset / "replay_manifest.json").write_text(json.dumps({"dataset_manifests": {"a_khan": {"sha256": "old"}}}))
    (dataset / "manifest.json").write_text(json.dumps({"replay_bundles": {"incremental": {"sha256": "old"}}}))
    source = tmp_path / "source/corrected_replay/raw/a_khan/signatures.json"
    source.parent.mkdir(parents=True)
    records = {
        "s1": {
            "author_info": {
                "email": "J.Smith@MIT.edu.",
                "emails": ["x@y"],
                "first": "J",
                "affiliations": ["Lab contact: office@example.org"],
            }
        }
    }
    source.write_text(json.dumps(records))

    assert hash_release_emails(tmp_path) == {"a_khan": {"arrow_values": 2, "source_fields": 2}}
    assert sha256_file(original) == original_arrow_sha
    assert sha256_file(original_index) == original_index_sha
    with pa.memory_map(str(arrow), "r") as stream:
        restored = pa.ipc.open_file(stream).read_all()
    assert restored["signature_id"] == table["signature_id"]
    assert restored["author_email"].to_pylist() == [hash_feature_email(v) for v in table["author_email"].to_pylist()]
    manifest = json.loads((giant / "manifest.json").read_text())
    for role, path in (("signatures", arrow), ("signatures_batch_index", index)):
        assert manifest["files"][role] == {"sha256": sha256_file(path), "byte_count": path.stat().st_size}
    assert json.loads((dataset / "replay_manifest.json").read_text())["dataset_manifests"]["a_khan"][
        "sha256"
    ] == sha256_file(giant / "manifest.json")
    assert json.loads((dataset / "manifest.json").read_text())["replay_bundles"]["incremental"][
        "sha256"
    ] == sha256_file(dataset / "replay_manifest.json")
    _, metrics = write_arrow_batch_lookup_index(arrow, index, key_column="signature_id", overwrite=False)
    assert metrics["reused"]
    assert json.loads(source.read_text())["s1"]["author_info"]["first"] == "J"
    assert json.loads(source.read_text())["s1"]["author_info"]["affiliations"] == ["Lab contact: office@example.org"]
    settled_hashes = {path: sha256_file(path) for path in tmp_path.rglob("*") if path.is_file()}
    assert hash_release_emails(tmp_path) == {"a_khan": {"arrow_values": 0, "source_fields": 0}}
    assert all(sha256_file(path) == digest for path, digest in settled_hashes.items())


def test_hashing_preserves_native_pairwise_and_incremental_aggregates(tmp_path: Path) -> None:
    dataset = build_dummy_dataset("hashed_emails", name_counts_index=True, n_jobs=1)
    emails = ["J.Smith@MIT.edu.", ".j.smith@mit.edu", "other@mit.edu", "j.smith@other.edu", "a@b@c", None, "a @b"]
    for sid, email in zip(dataset.signatures, emails, strict=False):
        dataset.signatures[sid] = dataset.signatures[sid]._replace(author_info_email=email)
    original = build_arrow_training_dataset(dataset, tmp_path / "original")
    for sid, signature in list(dataset.signatures.items()):
        dataset.signatures[sid] = signature._replace(author_info_email=hash_feature_email(signature.author_info_email))
    hashed = build_arrow_training_dataset(dataset, tmp_path / "hashed")
    matrices = []
    aggregates = []
    for data in (original, hashed):
        native = feature_port._get_rust_featurizer(data)
        size = len(native.signature_ids())
        pairs = list(product(range(size), repeat=2))
        matrices.append(np.asarray(native.featurize_pairs_matrix_indexed(pairs, list(range(NUM_FEATURES)), 1, np.nan)))
        batch = linker_pairwise.LinkerCandidateBatch(
            row_count=size,
            left_signature_indices=np.asarray([a for a, _ in pairs], dtype=np.uint32),
            right_signature_indices=np.asarray([b for _, b in pairs], dtype=np.uint32),
            pair_row_indices=np.repeat(np.arange(size, dtype=np.uint32), size),
        )
        stats = linker_pairwise.compute_candidate_batch_pairwise_aggregate_stats_rust(
            data, batch, n_jobs=1, nan_value=0.0, aggregate_nan_value=0.0, featurizer=native
        )
        aggregates.append(stats.feature_matrix())
    np.testing.assert_allclose(*matrices, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(*aggregates, rtol=0, atol=0, equal_nan=True)

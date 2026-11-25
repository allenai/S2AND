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
from scripts.common.email_redaction import (
    EMAIL_REDACTED_TOKEN,
    find_plaintext_emails,
    hash_feature_email,
    hash_release_emails,
    redact_source_tree,
    scan_tree_for_plaintext_emails,
    scrub_text_emails,
)
from tests.helpers import build_arrow_training_dataset, build_dummy_dataset

pytestmark = pytest.mark.tooling


@pytest.mark.parametrize(
    "value",
    [None, "J.Smith@MIT.EDU.", "a@b@c", "a @b"],
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


def test_scrub_text_emails_handles_unicode_dashes_and_malformed_at_dot() -> None:
    hashed = hash_feature_email("alice@example.org")
    assert hashed is not None
    text = (
        f"Contact first\u2010last@uni\u2013dept.edu, odd@.mail.example.org, "
        f'josé@uni.es, {{john,mary}}@uni.edu, user\uff20universität.de, "j.doe"@mit.edu; '
        f"chemical Fe3O4@SiO2.The and already hashed {hashed} stay intact."
    )
    scrubbed = scrub_text_emails(text)
    assert scrubbed is not None
    assert "first\u2010last@" not in scrubbed
    assert "odd@.mail.example.org" not in scrubbed
    assert "josé@uni.es" not in scrubbed
    assert "{john,mary}@uni.edu" not in scrubbed
    assert "user\uff20universität.de" not in scrubbed
    assert '"j.doe"@mit.edu' not in scrubbed
    assert scrubbed.count(EMAIL_REDACTED_TOKEN) == 6
    assert "Fe3O4@SiO2.The" in scrubbed
    assert hashed in scrubbed


@pytest.mark.parametrize(
    ("prefix", "address", "affiliation_id"),
    [
        ("Xxxxxxxxxxx ", "x.xxxxxxxxx@xxx.xx", "60032882"),
        ("Xxxxxxxxxxx ", "x.xxxxxxxxx@xxx.xx", "60032882"),
        ("xx, Xxxxxxx ", "xxxxxxxxx.xxxxx@xxx-xxxxxxx.xx", "60028229"),
        ("xx, Xxxxxxx ", "x.xxxxx@xxx-xxx.xx", "60010750"),
        ("xx, Xxxxxxx ", "x.xxxxx@xxx-xxx.xx", "60010750"),
        ("xx, Xxxxxxx ", "x.xxxxx@xxx-xxx.xx", "60010750"),
        ("Xxxxxxxxxxx ", "xxxxxx.xxxxxxxxx@xxx.xx", "60012614"),
    ],
    ids=["287947975-0", "289458341-1", "286214046-3", "279001215-0", "279001275-1", "279001287-1", "286770593-6"],
)
def test_affiliation_email_followed_by_id(prefix: str, address: str, affiliation_id: str) -> None:
    # Mask the source addresses while preserving the seven missed cells' structure.
    cell = f"{prefix}{address}{affiliation_id}"
    assert find_plaintext_emails(cell) == [address]
    assert scrub_text_emails(cell) == f"{prefix}{EMAIL_REDACTED_TOKEN}{affiliation_id}"


def test_affiliation_email_followed_by_a_long_run_of_letters() -> None:
    address_and_text = "a.b@example.eduDepartmentofInformationandComputerScience"
    assert find_plaintext_emails(address_and_text) == [address_and_text]
    assert scrub_text_emails(address_and_text) == EMAIL_REDACTED_TOKEN


def test_dynamic_dataset_discovery_redact_source_tree_and_plaintext_scan(tmp_path: Path) -> None:
    # Non-hardcoded dataset name (not in legacy 7-giant list) is discovered and redacted,
    # including rebuilding papers_batch_index and refreshing source inventory!
    dataset_dir = tmp_path / "dataset" / "datasets" / "b_chen"
    dataset_dir.mkdir(parents=True)
    sig_table = pa.table(
        {
            "signature_id": ["s1"],
            "author_email": ["b.chen@example.org"],
            "author_affiliations": [["Dept of AI, contact b\u2010chen@example.org"]],
        }
    )
    with (
        pa.OSFile(str(dataset_dir / "signatures.arrow"), "wb") as sink,
        pa.ipc.new_file(sink, sig_table.schema) as writer,
    ):
        writer.write_table(sig_table)
    sig_idx = dataset_dir / "signatures.signatures_batch_index.bin"
    write_arrow_batch_lookup_index(dataset_dir / "signatures.arrow", sig_idx, key_column="signature_id")

    papers_table = pa.table({"paper_id": ["p1"], "title": ["Study"], "abstract": ["Write to foo@.bar.edu"]})
    with (
        pa.OSFile(str(dataset_dir / "papers.arrow"), "wb") as sink,
        pa.ipc.new_file(sink, papers_table.schema) as writer,
    ):
        writer.write_table(papers_table)
    papers_idx = dataset_dir / "papers.papers_batch_index.bin"
    write_arrow_batch_lookup_index(dataset_dir / "papers.arrow", papers_idx, key_column="paper_id", table_name="papers")

    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "paths": {
                    "signatures": "signatures.arrow",
                    "signatures_batch_index": sig_idx.name,
                    "papers": "papers.arrow",
                    "papers_batch_index": papers_idx.name,
                },
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "dataset" / "replay_manifest.json").write_text(
        json.dumps({"dataset_manifests": {"b_chen": {"sha256": "old"}}}), encoding="utf-8"
    )
    (tmp_path / "dataset" / "manifest.json").write_text(
        json.dumps({"replay_bundles": {"incremental": {"sha256": "old"}}}), encoding="utf-8"
    )
    raw_dir = tmp_path / "source" / "corrected_replay" / "raw" / "b_chen"
    raw_dir.mkdir(parents=True)
    raw_sigs = raw_dir / "signatures.json"
    raw_papers = raw_dir / "papers.json"
    raw_sigs.write_text(
        json.dumps(
            {"s1": {"author_info": {"email": "b.chen@example.org", "affiliations": ["mail: b.chen@example.org"]}}}
        ),
        encoding="utf-8",
    )
    raw_papers.write_text(
        json.dumps({"p1": {"title": "Email x@y.org", "abstract": "Plain abstract"}}), encoding="utf-8"
    )
    inv_path = tmp_path / "source" / "corrected_sources" / "manifest.json"
    inv_path.parent.mkdir(parents=True)
    inv_path.write_text(
        json.dumps(
            {
                "files": {
                    "corrected_replay/raw/b_chen/signatures.json": sha256_file(raw_sigs),
                    "corrected_replay/raw/b_chen/papers.json": sha256_file(raw_papers),
                }
            }
        ),
        encoding="utf-8",
    )

    findings_before = scan_tree_for_plaintext_emails(tmp_path)
    assert len(findings_before) >= 4
    assert "b.chen@example.org" not in json.dumps(findings_before)

    report = hash_release_emails(tmp_path, scrub_free_text=True)
    assert report == {"b_chen": {"arrow_values": 3, "source_fields": 3}}
    assert scan_tree_for_plaintext_emails(tmp_path) == []

    # Verify papers_batch_index was rebuilt and is NOT stale!
    _, papers_metrics = write_arrow_batch_lookup_index(
        dataset_dir / "papers.arrow", papers_idx, key_column="paper_id", table_name="papers", overwrite=False
    )
    assert papers_metrics["reused"]
    refreshed_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    assert refreshed_manifest["files"]["papers_batch_index"]["sha256"] == sha256_file(papers_idx)
    inv_after = json.loads(inv_path.read_text(encoding="utf-8"))["files"]
    assert inv_after["corrected_replay/raw/b_chen/papers.json"] == sha256_file(raw_papers)

    # Scanner catches JSON-escaped \u0040, dictionary<*, string>, and list<struct> Arrow columns.
    probe = tmp_path / "scan_probe"
    probe.mkdir()
    (probe / "escaped.json").write_text('{"note": "alice\\u0040example.org"}\n', encoding="utf-8")
    nested_table = pa.table(
        {
            "dict_col": pa.array(["bob@example.org"]).dictionary_encode(),
            "struct_list": pa.array([[{"author": "carol@example.org"}]]),
        }
    )
    with (
        pa.OSFile(str(probe / "nested.arrow"), "wb") as sink,
        pa.ipc.new_file(sink, nested_table.schema) as writer,
    ):
        writer.write_table(nested_table)
    probe_findings = scan_tree_for_plaintext_emails(probe)
    assert len(probe_findings) == 3

    # Standalone intake-once redaction also handles *_signatures.json / *_papers.json trees.
    intake = tmp_path / "intake"
    intake.mkdir()
    (intake / "block_signatures.json").write_text(
        json.dumps({"s1": {"author_info": {"email": "a@b.edu", "affiliations": ["lab a@b.edu"]}}}),
        encoding="utf-8",
    )
    (intake / "block_papers.json").write_text(
        json.dumps({"p1": {"title": "T", "abstract": "Contact c\u2013d@e.org"}}), encoding="utf-8"
    )
    counts = redact_source_tree(intake)
    assert counts["signature_json_fields"] == 2
    assert counts["paper_json_fields"] == 1
    assert scan_tree_for_plaintext_emails(intake) == []

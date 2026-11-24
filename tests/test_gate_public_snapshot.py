from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts.production.data.gate_public_snapshot import (
    REDACTED_ABSTRACT_MARKER,
    gate_public_dataset,
    load_paper_id_set,
)
from scripts.production.data.hash_release_emails import (
    EMAIL_REDACTED_TOKEN,
    hash_feature_email,
    scan_tree_for_plaintext_emails,
)


def test_load_paper_id_set_rejects_false_and_non_bool_permission_maps(tmp_path: Path) -> None:
    bool_map = tmp_path / "perm.json"
    bool_map.write_text(json.dumps({"allowed": True, "restricted": False}), encoding="utf-8")
    assert load_paper_id_set(bool_map) == frozenset({"allowed"})

    bad_map = tmp_path / "bad.json"
    bad_map.write_text(json.dumps({"allowed": "yes"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must map paper IDs to booleans"):
        load_paper_id_set(bad_map)


def test_gate_public_dataset_preserves_has_abstract_redacts_history_and_filters_sidecars(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    src.mkdir()
    papers_path = src / "demo_papers.json"
    sigs_path = src / "demo_signatures.json"
    clusters_path = src / "demo_clusters.json"
    specter_path = src / "demo_specter2.pkl"
    unknown_path = src / "restored_unknown_signature_ids_test.json"
    pairs_path = src / "train_pairs.csv"

    papers_path.write_text(
        json.dumps(
            {
                "p_public_abs": {
                    "paper_id": "p_public_abs",
                    "title": "Public study",
                    "abstract": "Contact author at alice@example.org for code.",
                },
                "p_public_no_abs": {
                    "paper_id": "p_public_no_abs",
                    "title": "Closed abstract study (bob@example.org)",
                    "abstract": "Proprietary abstract that must be replaced by marker.",
                    "correction_history": [{"field": "abstract", "old_value": "Earlier proprietary abstract text."}],
                },
                "p_public_empty_abs": {
                    "paper_id": "p_public_empty_abs",
                    "title": "No abstract paper",
                    "abstract": None,
                },
                "p_private": {
                    "paper_id": "p_private",
                    "title": "Non-S2AG paper",
                    "abstract": "Must be dropped entirely.",
                },
            }
        ),
        encoding="utf-8",
    )
    sigs_path.write_text(
        json.dumps(
            {
                "s1": {
                    "signature_id": "s1",
                    "paper_id": "p_public_abs",
                    "author_info": {
                        "email": "alice@example.org",
                        "affiliations": ["Lab, mail alice@example.org"],
                    },
                },
                "s2": {
                    "signature_id": "s2",
                    "paper_id": "p_public_no_abs",
                    "author_info": {"email": None, "affiliations": ["Univ"]},
                },
                "s3": {
                    "signature_id": "s3",
                    "paper_id": "p_private",
                    "author_info": {"email": "private@example.org", "affiliations": []},
                },
            }
        ),
        encoding="utf-8",
    )
    clusters_path.write_text(
        json.dumps(
            {
                "c1": {"cluster_id": "c1", "signature_ids": ["s1", "s3"]},
                "c2": {"cluster_id": "c2", "signature_ids": ["s3"]},
            }
        ),
        encoding="utf-8",
    )
    with specter_path.open("wb") as handle:
        pickle.dump(
            {
                "p_public_abs": np.zeros(4, dtype=np.float32),
                "p_public_no_abs": np.ones(4, dtype=np.float32),
                "p_private": np.full(4, 2.0, dtype=np.float32),
            },
            handle,
        )
    unknown_path.write_text(json.dumps(["s1", "s3"]), encoding="utf-8")
    pairs_path.write_text("signature_id_1,signature_id_2,label\ns1,s2,1\ns1,s3,0\n", encoding="utf-8")

    allowed_file = tmp_path / "allowed.json"
    allowed_file.write_text(json.dumps(["p_public_abs", "p_public_no_abs", "p_public_empty_abs"]), encoding="utf-8")
    abs_allowed_file = tmp_path / "abs_allowed.txt"
    abs_allowed_file.write_text("p_public_abs\n", encoding="utf-8")

    out_dir = tmp_path / "gated"
    report = gate_public_dataset(
        dataset_name="demo",
        signatures_json=sigs_path,
        papers_json=papers_path,
        clusters_json=clusters_path,
        specter_pickle=specter_path,
        unknown_signatures_json=unknown_path,
        pair_files=[pairs_path],
        output_dir=out_dir,
        allowed_paper_ids=load_paper_id_set(allowed_file),
        abstract_allowed_paper_ids=load_paper_id_set(abs_allowed_file),
    )

    assert report["papers_before"] == 4
    assert report["papers_after"] == 3
    assert report["papers_dropped_non_s2ag"] == 1
    assert report["signatures_after"] == 2
    assert report["abstracts_redacted"] == 1
    assert report["abstract_history_fields_redacted"] == 1
    assert report["specter_after"] == 2
    assert report["unknown_signatures_after"] == 1
    assert report["pair_files"]["train_pairs.csv"] == {"before": 2, "after": 1}

    gated_papers = json.loads((out_dir / "demo_papers.json").read_text(encoding="utf-8"))
    assert gated_papers["p_public_no_abs"]["abstract"] == REDACTED_ABSTRACT_MARKER
    assert gated_papers["p_public_no_abs"]["correction_history"][0]["old_value"] == REDACTED_ABSTRACT_MARKER
    assert gated_papers["p_public_empty_abs"]["abstract"] is None
    assert EMAIL_REDACTED_TOKEN in gated_papers["p_public_abs"]["abstract"]
    assert EMAIL_REDACTED_TOKEN in gated_papers["p_public_no_abs"]["title"]

    gated_sigs = json.loads((out_dir / "demo_signatures.json").read_text(encoding="utf-8"))
    assert gated_sigs["s1"]["author_info"]["email"] == hash_feature_email("alice@example.org")
    assert gated_sigs["s1"]["author_info"]["affiliations"] == [f"Lab, mail {EMAIL_REDACTED_TOKEN}"]

    with (out_dir / specter_path.name).open("rb") as handle:
        assert set(pickle.load(handle)) == {"p_public_abs", "p_public_no_abs"}
    assert json.loads((out_dir / unknown_path.name).read_text(encoding="utf-8")) == ["s1"]
    assert (out_dir / "preparation_report.public_gate.json").is_file()
    assert not (out_dir / "population_corrections.json").exists()
    assert scan_tree_for_plaintext_emails(out_dir) == []

    # Reusing an existing output directory is rejected.
    with pytest.raises(FileExistsError):
        gate_public_dataset(
            dataset_name="demo",
            signatures_json=sigs_path,
            papers_json=papers_path,
            output_dir=out_dir,
            allowed_paper_ids=load_paper_id_set(allowed_file),
            abstract_allowed_paper_ids=load_paper_id_set(abs_allowed_file),
        )

from __future__ import annotations

import json
from pathlib import Path

from scripts.production.data.gate_public_snapshot import gate_public_dataset, load_paper_id_set
from scripts.production.data.hash_release_emails import (
    EMAIL_REDACTED_TOKEN,
    hash_feature_email,
    scan_tree_for_plaintext_emails,
)


def test_gate_public_dataset_filters_s2ag_redacts_abstracts_and_scrubs_emails(tmp_path: Path) -> None:
    src = tmp_path / "raw"
    src.mkdir()
    papers_path = src / "demo_papers.json"
    sigs_path = src / "demo_signatures.json"
    clusters_path = src / "demo_clusters.json"

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
                    "abstract": "Proprietary abstract that must be removed.",
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

    allowed_file = tmp_path / "allowed.json"
    allowed_file.write_text(json.dumps(["p_public_abs", "p_public_no_abs"]), encoding="utf-8")
    abs_allowed_file = tmp_path / "abs_allowed.txt"
    abs_allowed_file.write_text("p_public_abs\n", encoding="utf-8")

    out_dir = tmp_path / "gated"
    report = gate_public_dataset(
        dataset_name="demo",
        signatures_json=sigs_path,
        papers_json=papers_path,
        clusters_json=clusters_path,
        output_dir=out_dir,
        allowed_paper_ids=load_paper_id_set(allowed_file),
        abstract_allowed_paper_ids=load_paper_id_set(abs_allowed_file),
    )

    assert report["papers_before"] == 3
    assert report["papers_after"] == 2
    assert report["papers_dropped_non_s2ag"] == 1
    assert report["signatures_after"] == 2
    assert report["clusters_before"] == 2
    assert report["clusters_after"] == 1
    assert report["abstracts_redacted"] == 1

    gated_papers = json.loads((out_dir / "demo_papers.json").read_text(encoding="utf-8"))
    assert set(gated_papers) == {"p_public_abs", "p_public_no_abs"}
    assert EMAIL_REDACTED_TOKEN in gated_papers["p_public_abs"]["abstract"]
    assert gated_papers["p_public_no_abs"]["abstract"] is None
    assert EMAIL_REDACTED_TOKEN in gated_papers["p_public_no_abs"]["title"]

    gated_sigs = json.loads((out_dir / "demo_signatures.json").read_text(encoding="utf-8"))
    assert set(gated_sigs) == {"s1", "s2"}
    assert gated_sigs["s1"]["author_info"]["email"] == hash_feature_email("alice@example.org")
    assert gated_sigs["s1"]["author_info"]["affiliations"] == [f"Lab, mail {EMAIL_REDACTED_TOKEN}"]

    gated_clusters = json.loads((out_dir / "demo_clusters.json").read_text(encoding="utf-8"))
    assert gated_clusters == {"c1": {"cluster_id": "c1", "signature_ids": ["s1"]}}
    assert (out_dir / "population_corrections.json").is_file()
    assert scan_tree_for_plaintext_emails(out_dir) == []

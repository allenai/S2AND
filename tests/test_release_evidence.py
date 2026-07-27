from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from s2and import release_evidence


def test_stage_release_evidence_downloads_and_verifies_members(
    tmp_path: Path,
    monkeypatch,
) -> None:
    blobs = {
        f"https://example.test/{role}.json": json.dumps({"role": role}).encode()
        for role in release_evidence.RELEASE_EVIDENCE_ROLES
    }
    payload = {
        "schema_version": release_evidence.RELEASE_EVIDENCE_MANIFEST_SCHEMA,
        "members": {
            role: {
                "path": f"reports/{role}.json",
                "sha256": hashlib.sha256(blobs[url]).hexdigest(),
                "size_bytes": len(blobs[url]),
                "url": url,
            }
            for role in release_evidence.RELEASE_EVIDENCE_ROLES
            for url in [f"https://example.test/{role}.json"]
        },
    }
    manifest = tmp_path / "transport.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        release_evidence,
        "urlopen",
        lambda url, timeout: io.BytesIO(blobs[url]),
    )

    staged = release_evidence.stage_release_evidence(
        manifest,
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        tmp_path / "staged",
    )

    assert staged.name == "evidence_manifest.json"
    release_evidence.validate_release_evidence_manifest(payload, staged, require_urls=True)

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_common_model_json_bytes_match_manifest() -> None:
    """Verify checksum bytes from the actual Git checkout without an LFS fetch."""
    common_root = Path(__file__).resolve().parents[1] / "artifacts" / "common"
    manifests = sorted(common_root.glob("*/production_model_*/manifest.json"))
    assert manifests
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_bytes())
        for relative_path, expected_sha256 in manifest["sha256"].items():
            if Path(relative_path).suffix == ".json":
                artifact_path = manifest_path.parent / relative_path
                assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == expected_sha256, artifact_path

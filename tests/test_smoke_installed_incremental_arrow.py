from __future__ import annotations

from pathlib import Path

import pytest

from s2and import production_model as production_model_module
from scripts.verification import smoke_installed_incremental_arrow as smoke_module


def test_promoted_incremental_arrow_smoke_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_hashes = {
        "name_tuples_data_sha256": "1" * 64,
        "orcid_prefix_counts_data_sha256": "2" * 64,
    }
    monkeypatch.setattr(smoke_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    monkeypatch.setattr(production_model_module, "canonical_artifact_hashes", lambda: dict(artifact_hashes))
    summary = smoke_module.run_smoke(tmp_path)

    assert summary == {
        "arrow_promoted_incremental": 1,
        "cluster_count": 2,
        "query_view": "raw_arrow",
        "signature_count": 3,
    }

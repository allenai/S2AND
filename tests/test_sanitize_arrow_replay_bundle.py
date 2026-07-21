from __future__ import annotations

import json
from pathlib import Path

from scripts.production.model.sanitize_arrow_replay_bundle import sanitize_arrow_replay_bundle


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_sanitize_arrow_replay_bundle_removes_legacy_runtime_assets(tmp_path: Path) -> None:
    bundle_root = tmp_path / "arrow_replay_bundle"
    _write_json(
        bundle_root / "bundle.json",
        {
            "bundle_name": "s2and_and_big_blocks_linker_dataset_20260513",
            "assets": {
                "candidate_members": {"datasets": {"pubmed": "components/pubmed_members.parquet"}},
                "corrected_feature_rows": {"files": {"train_path": "features_corrected/train.parquet"}},
                "embeddings": {"datasets": {"pubmed": "embeddings/pubmed/specter2.pkl"}},
                "featureless_rows": {"files": {"train_path": "labels/train.parquet"}},
                "raw_metadata": {
                    "datasets": {
                        "pubmed": {
                            "papers_path": "raw/pubmed/papers.json",
                            "signatures_path": "raw/pubmed/signatures.json",
                        }
                    }
                },
                "splits": {"assignments_path": "splits/combined_query_split_assignments.csv"},
            },
            "models": {
                "classic": {
                    "best_params": {"n_estimators": 10},
                    "classic_gate_source_path": "features_corrected/calibration_source.parquet",
                    "extra_eval_paths": {"j_smith": "features_corrected/j_smith_eval.parquet"},
                    "train_path": "features_corrected/train.parquet",
                }
            },
            "expected_metrics": {},
            "notes": "legacy note",
        },
    )
    _write_json(
        bundle_root / "source_bundle_manifest.json",
        {
            "raw": [{"dataset": "pubmed", "papers": 2, "signatures": 3, "source_paper_bytes": 10}],
            "embeddings": [
                {
                    "dataset": "pubmed",
                    "embedding_count": 1,
                    "embedding_dim": 768,
                    "missing_embedding_count": 1,
                    "paper_count": 2,
                    "path": "embeddings/pubmed/specter2.pkl",
                    "source_path": "D:/data/pubmed/specter.pickle",
                    "source_kind": "specter2_pickle",
                }
            ],
            "components": [{"dataset": "pubmed", "rows": 3}],
            "table_summaries": [{"table_key": "train_path", "rows": 4}],
            "validation": {"embedding_missing_counts_preserved_as_missing_features": {"pubmed": 1}},
        },
    )

    report = sanitize_arrow_replay_bundle(
        bundle_root,
        write=True,
        legacy_source_bundle_name="s2and_and_big_blocks_linker_dataset_20260513",
        legacy_source_url=("s3://ai2-s2-research-public/s2and-release/s2and_and_big_blocks_linker_dataset_20260513"),
    )

    bundle_payload = json.loads((bundle_root / "bundle.json").read_text(encoding="utf-8"))
    assert bundle_payload["bundle_name"] == "arrow_replay_bundle"
    assert sorted(bundle_payload["assets"]) == ["candidate_members", "featureless_rows", "splits"]
    assert bundle_payload["models"]["classic"] == {"best_params": {"n_estimators": 10}}
    assert bundle_payload["runtime_contract"]["omitted_legacy_assets"] == [
        "raw/*.json",
        "embeddings/*.pkl",
        "features_corrected/*.parquet",
    ]
    assert report["removed_bundle_asset_keys"] == ["corrected_feature_rows", "embeddings", "raw_metadata"]

    provenance = json.loads((bundle_root / "source_bundle_manifest.json").read_text(encoding="utf-8"))
    assert "raw" not in provenance
    assert "embeddings" not in provenance
    assert provenance["legacy_source_counts"]["raw"]["pubmed"] == {"papers": 2, "signatures": 3}
    assert provenance["legacy_source_counts"]["embeddings"]["pubmed"]["missing_embedding_count"] == 1

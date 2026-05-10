# Rust-Promoted `predict_incremental` Status

Status date: 2026-05-10

This is the live implementation checklist for
[predict_incremental_fast_design.md](predict_incremental_fast_design.md).

## Done

- Rust backend selection routes `Clusterer.predict_incremental(...)` through
  the promoted linker when the Rust extension, artifact, and capabilities are
  available.
- The checked-in released linker artifact is `s2and/data/production_incremental_linker_v1.2/`
  with `booster.lgb`, `metadata.json`, and `training_target.json`.
- Training replay defaults to
  `s2and/data/joint_safe_link_minimal_raw_specter_20260507a` plus the tracked target
  spec.
- Runtime code lives under `s2and/incremental_linking/` without importing
  `scripts.*`.
- Promoted feature assembly is tested against the tracked 70-feature target.
- Query batching uses `total_ram_bytes` and `batching_threshold`.
- The residual tail stays exact and receives the resolved RAM budget.
- Local precomputed feature-table mode was removed.

## Still Needed

- Real-block release evidence for quality, speed, pair-count reduction,
  residual count, and RSS.
- Strict residual memory behavior documented from an actual large-block run.
- Optional portable precomputed-feature bundle mode, if repeated replay needs
  compute-once/reuse.

## Verification

Focused checks:

```powershell
uv run pytest -q tests/test_cluster_incremental.py::test_predict_incremental_private_linker_mode_uses_seed_link_seam tests/test_cluster_incremental.py::test_predict_incremental_promoted_linker_batches_queries tests/test_cluster_incremental.py::test_finish_incremental_with_seed_links_reclusters_only_abstains
uv run pytest -q tests/test_incremental_linking_m1_gates.py tests/test_linker_feature_assembly.py tests/test_incremental_linking_default_artifact.py
uv run ruff check scripts/run_joint_safe_link_promoted_train_calibrate_eval.py scripts/joint_safe_link_official_stack.py tests/test_incremental_linking_m1_gates.py tests/test_linker_feature_assembly.py
```

Full PR verification should still run the broader incremental-linking and
cluster-incremental suites.

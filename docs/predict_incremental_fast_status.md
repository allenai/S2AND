# Rust-Promoted `predict_incremental` Status

Status date: 2026-05-13

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
- Promoted feature assembly is tested against the tracked 53-feature target.
- Query batching uses `total_ram_bytes` and `batching_threshold`.
- The residual tail stays exact and receives the resolved RAM budget.
- Local precomputed feature-table mode was removed.
- Portable precomputed promoted-feature bundle mode is available only through
  explicit `--feature-mode precomputed-promoted --precomputed-feature-bundle-root ...`.
  It validates relative bundle paths, row counts, target/schema digests, required
  tables, and exact feature-column equality before training.
- Promoted-53 precomputed replay reused all feature tables and reproduced the
  production metrics: 53 features, 1,636,263 training rows, 300 stratified test
  errors, and `weighted_average_error=0.003968401417923204`.
- Real-block release evidence now covers quality, speed, pair-count reduction,
  residual count, exact-tail memory bytes, and RSS. The current 4k `j kim`
  operational run reduced the broad seed/query scope from 3,000,000 pairs to
  150,000 promoted scored pairs, left 354 exact residual queries, used a
  499,848-byte residual matrix, and peaked at 0.621 GiB process-tree RSS.

## Still Needed

- No known code or documentation blockers for the promoted-53 Rust
  `predict_incremental` release path. Rerun the verification suite after any
  further feature-surface or artifact change.

## Verification

Focused checks:

```powershell
uv run pytest -q tests/test_cluster_incremental.py::test_predict_incremental_private_linker_mode_uses_seed_link_seam tests/test_cluster_incremental.py::test_predict_incremental_promoted_linker_batches_queries tests/test_cluster_incremental.py::test_finish_incremental_with_seed_links_reclusters_only_abstains
uv run pytest -q tests/test_incremental_linking_m1_gates.py tests/test_linker_feature_assembly.py tests/test_incremental_linking_default_artifact.py
uv run pytest -q tests/test_big_block_incremental_cmd.py
uv run ruff check scripts/run_joint_safe_link_promoted_train_calibrate_eval.py scripts/_rust_suite/big_block_incremental_cmd.py tests/test_incremental_linking_m1_gates.py tests/test_linker_feature_assembly.py tests/test_big_block_incremental_cmd.py
```

Full PR verification should still run the broader incremental-linking and
cluster-incremental suites.

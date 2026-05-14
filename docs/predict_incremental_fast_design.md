# Rust-Promoted `predict_incremental`

Status date: 2026-05-13

## Why This Exists

Legacy `Clusterer.predict_incremental(...)` compares each unassigned query
signature against assigned seed signatures in the block, then links or abstains
using pairwise-distance rules. On giant blocks, that creates too much pair work:
large runs can spend most of their time scoring pairs that are not plausible
seed-cluster candidates.

The promoted Rust path changes the problem shape. It retrieves a bounded set of
candidate seed clusters first, scores only those candidate query/member pairs,
assembles the promoted 53-feature linker matrix, and applies a calibrated
link-or-abstain gate. Output parity with the legacy path is not a goal; the
goal is better measured quality and runtime under an explicit memory budget.

## How It Works

```text
seed clusters + unassigned queries
  -> Rust seed-cluster retrieval
  -> Rust candidate pair plan
  -> indexed pair featurization + pairwise model distances + pw_* aggregates
  -> promoted 53-feature row assembly
  -> calibrated score/margin gate
  -> exact residual clustering tail for abstained/no-candidate queries
```

Compared with the legacy Python path:

- **Candidate scope:** legacy scores broad query-vs-seed-signature pairs; promoted
  Rust scores retrieved seed-cluster candidates.
- **Decision model:** legacy uses pairwise distances and `eps`-style cluster
  decisions; promoted Rust uses a trained LightGBM linker plus calibrated gates.
- **Feature surface:** legacy relies on pairwise features; promoted Rust uses 30
  compact row features plus 23 retained `pw_*` aggregate features.
- **Memory behavior:** promoted Rust batches query signatures by
  `total_ram_bytes` and `batching_threshold`; the residual tail stays exact and
  fails before allocation if the exact matrix cannot fit.

## Evidence So Far

The strongest benchmark is the manually reviewed `a_khan` block used during
promotion. It is not a universal production guarantee, but it explains why this
path exists.

| Path | Candidate work | Precision | Recall | Wall time |
|---|---:|---:|---:|---:|
| Legacy `predict_incremental`, all candidates, `n_jobs=12` | 91,671,102 pairs | 94.73% | 93.59% | 2215.5s |
| Promoted linker seed-link path, top 25, fused pairs, `n_jobs=20` | 784,174 pairs | 99.19% | 97.68% | 46.4s |
| Promoted end-to-end path with exact residual tail, `n_jobs=20` | 784,174 seed pairs + 1,770 residual pairs | 99.19% | 97.68% | 72.6s |

The important change is candidate reduction: the promoted path reduced seed-link
pair work from about 91.7M pairs to about 0.8M pairs on this benchmark while
improving reviewed-label precision and recall. The end-to-end number includes
the exact residual clustering tail.

A current promoted-53 operational replay on the existing real-block
`scratch/inventors_topblock_15k` subset (`j kim`, 4,000 selected signatures,
3,000 seed signatures, 1,000 unassigned signatures, 500 synthetic seed clusters,
`n_jobs=20`, `total_ram_bytes=32 GiB`) produced:

| Metric | Value |
|---|---:|
| Setup-inclusive runtime | 9.288s |
| `predict_incremental` runtime | 8.202s |
| Broad seed/query pair scope | 3,000,000 pairs |
| Promoted candidate rows | 25,000 rows |
| Promoted scored query/member pairs | 150,000 pairs |
| Promoted links / abstains | 646 / 354 |
| Exact residual-tail queries | 354 |
| Exact residual-tail pair matrix | 62,481 pairs, 499,848 bytes |
| Process-tree peak RSS | 0.621 GiB |
| Promoted-batch observed RSS delta | 66,285,568 bytes |
| Promoted-batch predicted RSS delta | 277,177,216 bytes |

The command wrote
`scratch/predict_incremental_release_4000_20260513/single.json`. The run used
the production v1.2 pairwise model and required a release Rust extension.

## Current Release State

The promoted-53 Rust path is the current release target for Rust-backed
`Clusterer.predict_incremental(...)`. Backend selection routes through the
promoted linker when the Rust extension, artifact, and required capabilities are
available.

The checked-in release artifact is
`s2and/data/production_incremental_linker_v1.2/`, with `booster.lgb`,
`metadata.json`, and `training_target.json`. Runtime code lives under
`s2and/incremental_linking/` and must not import `scripts.*`.

Feature assembly is tested against the tracked 53-feature target. Query
batching uses `total_ram_bytes` and `batching_threshold`; the residual tail
stays exact and receives the resolved RAM budget.

## Release Inputs

The release surface is:

- `s2and/data/production_model_v1.2.pickle`
- `s2and/data/production_incremental_linker_v1.2/`
- `s2and/data/production_incremental_linker_v1.2/training_target.json`
- `s2and/data/s2and_and_big_blocks_linker_dataset_20260513/`

`training_target.json` is the portable target spec for replay: feature order,
LightGBM params, target metrics, status, and variant. Replay must not depend on
machine-local analysis artifacts.

When the pairwise model changes, update the promoted linker as a coordinated
release unit; see
[production_inference.md](production_inference.md#updating-the-linker-after-a-pairwise-model-change).

## Reusing Computed Features

Official replay defaults to recomputing promoted features from the
self-contained source bundle. Repeated replay can explicitly reuse a portable
precomputed promoted-feature bundle:

1. Materialize features once into an explicit `--output-dir`.
2. Promote the materialized bundle only after it has relative paths, row counts,
   a target-spec digest, a feature-schema digest, and verification metrics.
3. Reuse it with `--feature-mode precomputed-promoted` and an explicit
   `--precomputed-feature-bundle-root`.

That mode validates `bundle.json`, table paths, row counts, required
train/calibrate/eval tables, target-spec digest, feature-schema digest, and
exact feature-column equality with `training_target.json` before training. It
has no machine-local default.

The full promoted-53 replay through precomputed mode wrote
`scratch/precomputed_promoted53_replay_20260513/run_summary.json` and reproduced
the production training/evaluation metrics with all ten feature tables reused:
53 features, 1,636,263 training rows, 300 stratified test errors, and
`weighted_average_error=0.003968401417923204`.

## Release Gates

For each release candidate, report:

- reviewed-label quality;
- setup-inclusive and hot-path wall time;
- candidate rows, scored pairs, and residual pairs;
- residual count and exact-tail memory behavior;
- observed RSS versus `total_ram_bytes`.

Legacy-output parity is not a release gate.

## Verification

Focused release checks:

```powershell
uv run pytest -q tests/test_cluster_incremental.py::test_predict_incremental_rust_promoted_linker_uses_seed_link_seam tests/test_cluster_incremental.py::test_predict_incremental_promoted_linker_batches_queries tests/test_cluster_incremental.py::test_finish_incremental_with_seed_links_reclusters_only_abstains
uv run pytest -q tests/test_incremental_linking_m1_gates.py tests/test_linker_feature_assembly.py tests/test_incremental_linking_default_artifact.py
uv run pytest -q tests/test_big_block_incremental_cmd.py
uv run ruff check scripts/run_joint_safe_link_promoted_train_calibrate_eval.py scripts/_rust_suite/big_block_incremental_cmd.py tests/test_incremental_linking_m1_gates.py tests/test_linker_feature_assembly.py tests/test_big_block_incremental_cmd.py
```

Full PR verification should also run the broader incremental-linking and
cluster-incremental suites.

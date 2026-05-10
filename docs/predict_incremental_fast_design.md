# Rust-Promoted `predict_incremental`

Status date: 2026-05-10

## Why This Exists

Legacy `Clusterer.predict_incremental(...)` compares each unassigned query
signature against assigned seed signatures in the block, then links or abstains
using pairwise-distance rules. On giant blocks, that creates too much pair work:
large runs can spend most of their time scoring pairs that are not plausible
seed-cluster candidates.

The promoted Rust path changes the problem shape. It retrieves a bounded set of
candidate seed clusters first, scores only those candidate query/member pairs,
assembles the released 70-feature linker matrix, and applies a calibrated
link-or-abstain gate. Output parity with the legacy path is not a goal; the
goal is better measured quality and runtime under an explicit memory budget.

## How It Works

```text
seed clusters + unassigned queries
  -> Rust seed-cluster retrieval
  -> Rust candidate pair plan
  -> indexed pair featurization + pairwise model distances + pw_* aggregates
  -> promoted 70-feature row assembly
  -> calibrated score/margin gate
  -> exact residual clustering tail for abstained/no-candidate queries
```

Compared with the legacy Python path:

- **Candidate scope:** legacy scores broad query-vs-seed-signature pairs; promoted
  Rust scores retrieved seed-cluster candidates.
- **Decision model:** legacy uses pairwise distances and `eps`-style cluster
  decisions; promoted Rust uses a trained LightGBM linker plus calibrated gates.
- **Feature surface:** legacy relies on pairwise features; promoted Rust uses 18
  compact row features plus 52 retained `pw_*` aggregate features.
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

## Release Inputs

The release surface is:

- `data/production_model_v1.2.pickle`
- `data/production_incremental_linker_v1.2/`
- `data/production_incremental_linker_v1.2/training_target.json`
- `data/joint_safe_link_minimal_raw_specter_20260507a/`

`training_target.json` is the portable target spec for replay: feature order,
LightGBM params, target metrics, status, and variant. Replay must not depend on
machine-local analysis artifacts.

## Reusing Computed Features

Official replay currently recomputes promoted features from the self-contained
source bundle. If repeated replay needs compute-once/reuse, add a portable
precomputed-feature bundle flow:

1. Materialize features once into an explicit `--output-dir`.
2. Promote the materialized bundle only after it has relative paths, row counts,
   a target-spec digest, a feature-schema digest, and verification metrics.
3. Store release candidates under a tracked or downloadable data path such as
   `data/promoted_feature_bundles/<version>/`.
4. Add `--feature-mode precomputed-promoted` only with an explicit
   `--precomputed-feature-bundle-root`; do not ship a machine-local default.

That mode should validate `bundle.json`, table paths, row counts, required
train/calibrate/eval tables, and exact feature-column equality with
`training_target.json` before training.

## Release Gates

Before making broader claims, report:

- reviewed-label quality;
- setup-inclusive and hot-path wall time;
- candidate rows, scored pairs, and residual pairs;
- residual count and exact-tail memory behavior;
- observed RSS versus `total_ram_bytes`.

Legacy-output parity is not a release gate.

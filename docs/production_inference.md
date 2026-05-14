# Production Inference

This document collects the operational details for using the released S2AND production models.

## Which pairwise model to use

| Model file | Status | Embeddings | Uses reference features? |
| --- | --- | --- | --- |
| `production_model_v1.2.pickle` | Current | SPECTER2 PRX | No |
| `production_model_v1.1.pickle` | Previous | SPECTER1 | No |
| `production_model_v1.0.pickle` | Deprecated | SPECTER1 | Yes |

Recommended default:

- Use `production_model_v1.2.pickle` unless you have a specific compatibility reason to stay on `v1.1`.

Embedding source:

- For `v1.2`, use `embedding.specter_v2` from the Semantic Scholar API.
- For `v1.1`, use `embedding.specter_v1`.

## Promoted incremental linker artifact

The promoted incremental linker artifact is versioned with the pairwise model it
depends on:

| Artifact | Status | Depends on | Format |
| --- | --- | --- | --- |
| `production_incremental_linker_v1.2/` | Current | `production_model_v1.2.pickle` | `booster.lgb` + `metadata.json` |

This is intentionally not a pickle. The directory artifact stores the LightGBM
booster separately from metadata that validates the feature schema,
production contract, retrieval contract, gate thresholds, required Rust
capabilities, and a prediction fixture at load time. It also ships
`training_target.json`, the portable target spec used by replay scripts. The
tracked target is currently the promoted 53-feature schema used by the bundled
booster and metadata.

This artifact is the promoted incremental linker for Rust-backed
`Clusterer.predict_incremental(...)`. It is not intended to reproduce the
legacy incremental output. When Rust mode is selected and the extension plus
artifact pass validation, the target behavior is to use this promoted
retrieval/linker/gate path because it has shown better runtime and quality than
the long-standing legacy implementation.

### Updating the linker after a pairwise model change

Treat the pairwise model and promoted linker as one release unit. If
`production_model_vX.Y.pickle` changes, build a new
`production_incremental_linker_vX.Y/` artifact from features recomputed with
that exact pairwise model. Do not copy the old `booster.lgb`, reuse the old
`metadata.json`, or only edit metadata to point at the new pairwise file. The
promoted linker trains on pairwise-model distances plus `pw_*` aggregate
features, and the artifact audit metadata records the pairwise model path,
version, and SHA.

Before replay, confirm the new pairwise model is compatible with the replay
source bundle. The default minimal-raw bundle does not store reference papers,
so the replay script rejects pairwise models that require
`reference_features`. If the pairwise model changes embedding source or input
contract, rebuild the source bundle and pass it with `--source-bundle-root`.
The current train/calibrate/eval source bundle is published on S3 with the
other release data:

```powershell
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release/s2and_and_big_blocks_linker_dataset_20260513 s2and\data\s2and_and_big_blocks_linker_dataset_20260513
```

#### What the replay script does

`scripts/run_joint_safe_link_promoted_train_calibrate_eval.py` is the official
replay driver for the promoted incremental linker. It does not train the
pairwise pickle. It takes the pairwise pickle as an input, recomputes or loads
the promoted linker feature tables, trains/calibrates/evaluates the downstream
LightGBM linker, and can write the runtime linker artifact.

Its main inputs are:

- `--pairwise-model-path`: the pairwise model whose distances feed the linker.
- `--source-bundle-root`: the raw+SPECTER2+labels train/calibrate/eval bundle.
- `--target-json`: the replay target with feature order, LightGBM params,
  expected metrics, status, and variant.
- `--output-dir`: the scratch run directory for materialized features,
  summaries, and replay outputs.
- `--save-artifact-to`: optional output directory for `booster.lgb` and
  `metadata.json`.

In the default `--feature-mode minimal-raw-rust`, the script rebuilds promoted
features from the source bundle. For each selected table and dataset, it loads
the raw papers, signatures, SPECTER2 embeddings, and labels; applies structural
cleaning; builds block-local query/candidate context; uses the frozen Rust
retrieval policy to choose candidate seed clusters; builds the candidate/member
pair plan; computes pairwise model distances and `pw_*` aggregate features; adds
the non-pairwise row features; then writes target-ordered feature tables and
bundle metadata under `--output-dir`. These feature values are tied to the exact
pairwise model passed with `--pairwise-model-path`.

The other feature modes are narrower:

- `rust-recompute-pw` rematerializes promoted tables from an existing source
  bundle while recomputing the Rust pairwise aggregate columns.
- `precomputed-promoted` loads an already materialized portable feature bundle.
  It validates relative table paths, row counts, required tables, target-spec
  digest, feature-schema digest, and exact target feature-column order before
  training.

After features are available, the script runs the classic train/calibrate/eval
stack. It trains the LightGBM linker with the target params, fits or applies the
configured score/margin gate, evaluates the configured S2AND/Hwang/extra/manual
holdout tables, writes `classic/summary.json`, and writes `run_summary.json`
with observed metrics and deltas from `training_target.json`. Unless
`--allow-metric-drift` is passed, a full replay fails when observed metrics do
not match the target metrics.

When `--save-artifact-to` is set, the script also fits the final production
linker on train rows plus weighted calibration/eval rows, then writes
`booster.lgb` and `metadata.json`. The metadata includes the feature schema,
gate config, required Rust capabilities, prediction fixture, booster digest,
pairwise model path/version/SHA, source bundle, feature mode, observed metrics,
and production training summary. Keep `training_target.json` in the release
artifact directory alongside those files; `--save-artifact-to` writes the model
artifact files, not a new target spec.

Safety behavior is intentional: an unbounded full run requires `--run-full`;
`--datasets`, `--tables`, and `--limit-rows` are smoke/materialization controls
and require `--materialize-only`; and precomputed feature reuse is accepted only
through explicit `--feature-mode precomputed-promoted`.

The required update flow is:

1. Put the candidate pairwise pickle under `s2and/data/`, choose the matching
   linker version, and create or update
   `s2and/data/production_incremental_linker_vX.Y/training_target.json`. Start
   from the previous target only when the 53-feature schema and LightGBM params
   are intentionally unchanged.
2. Run a bounded materialization smoke test before any full replay:

```powershell
uv run python scripts\run_joint_safe_link_promoted_train_calibrate_eval.py `
  --pairwise-model-path s2and\data\production_model_vX.Y.pickle `
  --target-json s2and\data\production_incremental_linker_vX.Y\training_target.json `
  --output-dir scratch\joint_safe_link_promoted_vX.Y_smoke `
  --datasets qian `
  --limit-rows 200 `
  --materialize-only
```

3. Run the full train/calibrate/eval replay and write the new directory
   artifact. This is a large job; report the command, expected runtime, output
   directory, and monitoring plan before starting it.

```powershell
uv run python scripts\run_joint_safe_link_promoted_train_calibrate_eval.py `
  --pairwise-model-path s2and\data\production_model_vX.Y.pickle `
  --target-json s2and\data\production_incremental_linker_vX.Y\training_target.json `
  --save-artifact-to s2and\data\production_incremental_linker_vX.Y `
  --linker-artifact-version vX.Y `
  --output-dir scratch\joint_safe_link_promoted_vX.Y_full `
  --run-full
```

Use `--allow-metric-drift` only for exploratory candidate runs when the target
metrics are intentionally stale. Do not use it as the final release gate.

4. Review `run_summary.json`, `classic/summary.json`, and
   `prod_artifact_summary.json`. Report reviewed-label quality,
   setup-inclusive and hot-path wall time, candidate rows, scored pairs,
   residual pairs, residual count, exact-tail memory behavior, and observed RSS
   versus `total_ram_bytes`.
5. Promote the release as a coordinated update: the pairwise pickle, the linker
   artifact directory (`booster.lgb`, `metadata.json`, and
   `training_target.json`), default paths in `s2and/model.py` and
   `scripts/run_joint_safe_link_promoted_train_calibrate_eval.py`, package data
   entries in `pyproject.toml`, and tests/docs that hard-code the release
   version.
6. Verify the promoted artifact and release wiring:

```powershell
uv run pytest -q tests/test_incremental_linking_m1_gates.py tests/test_linker_feature_assembly.py tests/test_incremental_linking_default_artifact.py
uv run ruff check scripts/run_joint_safe_link_promoted_train_calibrate_eval.py tests/test_incremental_linking_m1_gates.py tests/test_linker_feature_assembly.py tests/test_incremental_linking_default_artifact.py
```

For repeated replay, `--feature-mode precomputed-promoted` is allowed only when
the precomputed bundle was materialized for the same target and pairwise model;
the full default replay recomputes promoted features from the source bundle.

Training/evaluation replay normally recomputes promoted features from the
self-contained minimal-raw source bundle. For compute-once/reuse workflows, the
replay script also supports an explicit portable precomputed bundle mode:

```powershell
uv run python scripts\run_joint_safe_link_promoted_train_calibrate_eval.py `
  --feature-mode precomputed-promoted `
  --precomputed-feature-bundle-root path\to\minimal_raw_feature_bundle `
  --run-full
```

The script validates relative table paths, row counts, target/schema digests,
required tables, and exact 53-feature column equality before training. There is
no shipped machine-local default for precomputed feature tables.

## Reference-feature behavior

Models `v1.1` and `v1.2` were trained with `compute_reference_features=False`. That means they do not use features derived from cited references.

The disabled reference-derived features are:

- `references_authors_overlap`
- `references_titles_overlap`
- `references_venues_overlap`
- `references_author_blocks_jaccard`
- `references_self_citation`
- `references_overlap`

Practical consequence:

- For `v1.1` and `v1.2`, `papers.references` can be omitted or set to `null`.
- Signature fields are still required as usual.

If you use `v1.0`, you must provide the paper-reference lists needed for those features.

## Minimal input contract

Minimal paper entry for `v1.1` and `v1.2`:

```json
{
  "paper_id": 12345,
  "title": "My Paper Title",
  "abstract": "Optional but recommended for the has_abstract feature.",
  "year": 2023,
  "venue": "Conference Name",
  "journal_name": "Journal Name",
  "authors": [
    {"position": 0, "author_name": "Jane Smith"},
    {"position": 1, "author_name": "John Doe"}
  ],
  "references": null
}
```

Minimal signature entry:

```json
{
  "signature_id": "0",
  "paper_id": 12345,
  "author_info": {
    "position": 0,
    "block": "j smith",
    "first": "Jane",
    "middle": null,
    "last": "Smith",
    "suffix": null,
    "email": null,
    "affiliations": ["University of Example"]
  }
}
```

## Name-count semantics compatibility

S2AND supports two runtime semantics for the name-count feature key used by `last_first_initial_count_min`:

- `legacy_full_first_token`: key is `<last> <first_token>`
- `initial_char`: key is `<last> <first[0]>`

Compatibility rules:

- `production_model_v1.1.pickle` and `production_model_v1.2.pickle` use `initial_char` with
  `s2and/data/name_counts.pickle`; that pickle stores keys like `smith j`, not `smith john`.
- In `ANDData(..., mode="inference")`, prediction automatically applies the semantics expected by the loaded model via the stored feature contract.
- Do not mix model artifacts and feature semantics without retraining.

## Minimal prediction flow

```python
from s2and.data import ANDData
from s2and.serialization import load_pickle_with_verified_label_encoder_compat

clusterer = load_pickle_with_verified_label_encoder_compat(
    "s2and/data/production_model_v1.2.pickle"
)["clusterer"]

dataset = ANDData(
    signatures="path/to/signatures.json",
    papers="path/to/papers.json",
    specter_embeddings="path/to/specter_embeddings.pkl",
    mode="inference",
    block_type="s2",
    n_jobs=8,
    name="my_dataset",
)

pred_clusters, pred_distance_matrices = clusterer.predict(dataset.get_blocks(), dataset)
```

`pred_distance_matrices` may be `None` when using memory-optimized fused clustering paths.

## Caching

Public cache control:

- `Clusterer.use_cache`
- `featurize(..., use_cache=...)`
- `many_pairs_featurize(..., use_cache=...)`
- `warm_rust_featurizer(..., use_cache=...)`

Semantics:

- `use_cache=True` enables the persistent pair-feature SQLite cache and the Rust featurizer disk cache.
- `use_cache=False` skips those persistent cache reads and writes.
- Same-process Rust featurizer reuse still stays enabled even when `use_cache=False`.

Recommended defaults:

- Repeated inference on the same dataset or pair set: `use_cache=True`
- One-shot jobs and experiments: `use_cache=False`

Full cache details: [caching.md](caching.md)

## Rust backend

`S2AND_BACKEND` controls runtime backend selection:

- `auto`: use Rust when available and capable, otherwise Python
- `rust`: strict Rust mode
- `python`: Python-only mode

Install contract:

- `uv pip install s2and`: Python-only runtime
- `uv pip install "s2and[rust]"`: Rust-enabled runtime when wheels are available

Full runtime contract: [rust/runtime.md](rust/runtime.md)

## Large blocks and incremental inference

For standard full-block prediction, subblocking keeps peak memory bounded. For
the promoted Rust incremental target, query batching should provide the memory
bound for the promoted retrieval/linker/gate path. The legacy incremental
implementation remains a fallback or compatibility mode, not the output target.

Standard large-block prediction:

```python
pred_clusters, _ = clusterer.predict(
    dataset.get_blocks(),
    dataset,
    batching_threshold=5000,
    desired_memory_use=5000 * 5000,
)
```

Incremental prediction with explicit RAM budget:

```python
result = clusterer.predict_incremental(
    block_signatures,
    dataset,
    batching_threshold=5000,
    total_ram_bytes=32 * 1024**3,
    max_chunk_pairs=50_000_000,
)

clusters = result["clusters"]
phase_b_mode = result["phase_b_mode"]
```

`phase_b_mode` tells you which path ran:

- `exact`: Phase B ran globally and is intended to match monolithic semantics.
- `subblock_local`: runtime fell back to approximate per-subblock behavior to stay within budget.

### Rust promoted incremental target

The target behavior is that `Clusterer.predict_incremental(...)` uses the
promoted Rust linker by default when `S2AND_BACKEND` selects Rust and the
extension has the required promoted-incremental capabilities. Legacy output
parity is not a release goal; the promoted path intentionally uses different
retrieval, linker, and margin-gate decisions.

`S2AND_BACKEND=rust` and `S2AND_BACKEND=auto` now route `predict_incremental`
through the promoted linker when backend resolution selects Rust. There is no
separate public force flag or artifact override: backend selection is the
routing contract. Promoted query batching is available: `batching_threshold`
caps the number of unassigned query signatures per promoted linker batch, while
`total_ram_bytes` derives the default batch size when the caller does not pass a
cap. The first meaningful promoted batch recalibrates rows/pairs per query for
remaining batches, and telemetry records predicted/observed RSS deltas.

The release evidence in [predict_incremental_fast_design.md](predict_incremental_fast_design.md)
includes a current promoted-53 4k real-block run: 3,000,000 broad seed/query
pairs reduced to 150,000 promoted scored pairs, 354 exact residual queries,
499,848 residual-tail bytes, 8.202s `predict_incremental` time, 9.288s
setup-inclusive runtime, and 0.621 GiB process-tree peak RSS.

Supporting docs:

- Subblocking behavior and tradeoffs: [subclustering.md](subclustering.md)
- Threading guidance: [threading.md](threading.md)
- Environment variables: [environment.md](environment.md)

## Warm-starting the Rust featurizer

For long-lived services, you can pre-warm once at startup:

```python
from s2and.feature_port import warm_rust_featurizer

warm_rust_featurizer(dataset, use_cache=True)
```

Use `use_cache=False` if you only want same-process warmup and do not want persistent disk-cache writes.

# Production Inference

This document collects the operational details for using the released S2AND production models.

## Which production model to use

| Model artifact | Status | Embeddings | Usable with current S2AND? | Format |
| --- | --- | --- | --- | --- |
| `production_model_v1.21/` | Current | SPECTER2 PRX | Yes | Native LightGBM + JSON bundle |
| `production_model_v1.2.pickle` | Legacy pairwise pickle | SPECTER2 PRX | Yes | Pickle |
| `production_model_v1.1.pickle` | Legacy | SPECTER1 | Yes | Pickle |
| `production_model_v1.0.pickle` | Deprecated | SPECTER1 | No (required removed reference features) | Pickle |

Recommended default:

- Use `production_model_v1.21/` unless you have a specific compatibility reason to load an older pickle.
- The v1.21 bundle contains pairwise artifacts whose source model version is
  v1.2 and the v1.2 promoted Rust incremental linker. Linker audit metadata may
  record the enclosing bundle path/version used during finalization.
- The bundle is checked into `s2and/data/` and included in package data, so prediction does not require a separate model download.

Embedding source:

- For `v1.21` and `v1.2`, use `embedding.specter_v2` from the Semantic Scholar API.
- For `v1.1`, use `embedding.specter_v1`.

## Production model bundle

The current production model is a single directory:

```text
s2and/data/production_model_v1.21/
  manifest.json
  clusterer.json
  pairwise/
    main.lgb
    nameless.lgb
    metadata.json
    main_prediction_fixture.json
    nameless_prediction_fixture.json
  incremental_linker/
    booster.lgb
    metadata.json
  reproducibility/
    incremental_linker_training_target.json
```

Load this directory once with `load_production_model(...)`. The returned object
is still a normal mutable `Clusterer`, so callers can set `clusterer.n_jobs`,
`clusterer.use_cache`, or `clusterer.cluster_model.eps` just as they did with
the old pickle-loaded clusterer.
For v1.2-derived production artifacts, including `production_model_v1.2.pickle`
and `production_model_v1.21/`, the loader applies the published runtime
FastCluster threshold `eps=0.65` before returning the clusterer.
Native bundles resolve that override from manifest/config metadata plus the
directory name. Legacy pickle artifacts do not carry bundle metadata, so their
override is path-name scoped; a v1.2-derived pickle saved under a noncanonical
filename is loaded with its stored `eps`.

The `pairwise/*.lgb` files are native LightGBM models, so Python can load them
without pickle and Rust/other runtimes can consume the same format directly.
The `incremental_linker/` directory contains the promoted linker used by
Rust-backed
`Clusterer.predict_incremental(...)`. It is not intended to reproduce the
legacy incremental output. When Rust mode is selected and the extension plus
artifact pass validation, the target behavior is to use this promoted
retrieval/linker/gate path because it has shown better runtime and quality than
the long-standing legacy implementation.

The file under `reproducibility/` is not consumed by prediction logic. It is
included in manifest checksum validation at bundle load time and records the
53-feature replay target and LightGBM training params for rebuilding or auditing
the promoted incremental linker.

New production releases are built as a two-stage native bundle. First,
`scripts/production/model/train_pairwise.py` writes the pairwise-only
`production_model_vX.Y/` stage. Then
`scripts/production/model/train_linker_and_finalize.py` trains the promoted
linker into the same directory and writes the final checksummed `manifest.json`.
Production release scripts should not write pickle artifacts.

New bundles must store their intended runtime `eps` in `clusterer.json` at
export time. The loader's runtime eps override is scoped to v1.2-derived
artifacts only (`_RUNTIME_CLUSTER_EPS_OVERRIDE_VERSIONS` in
`s2and/production_model.py`); do not extend it to new versions.

### Updating the linker after a pairwise model change

Treat the pairwise model and promoted linker as one release unit. If
the pairwise model changes, export a new native bundle such as
`production_model_vX.Y/` and rebuild the `incremental_linker/` artifact from
features recomputed with that exact pairwise model. Do not copy the old
`booster.lgb`, reuse the old `metadata.json`, or only edit metadata to point at
the new pairwise file. The promoted linker trains on pairwise-model distances
plus `pw_*` aggregate features, and the artifact audit metadata records the
pairwise model path, version, and digest.

Before replay, confirm the new pairwise model is compatible with the replay
source bundle. The replay script rejects legacy pairwise models whose
`features_to_use` still names the removed `reference_features` group. If the
pairwise model changes embedding source or input contract, rebuild the source
bundle and pass it with `--source-bundle-root`.
The current train/calibrate/eval source bundle is published as an Arrow-only
replay bundle with the other Arrow release data:

```powershell
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release-arrow/s2and_and_big_blocks_linker_dataset_20260525 s2and\data\s2and_and_big_blocks_linker_dataset_20260525
```

This bundle intentionally omits legacy `raw/`, `embeddings/`, and
`features_corrected/` directories. Replay uses Arrow tables under
`datasets/<dataset>/` plus `components/`, `labels/`, and `splits/`; promoted
feature rows are materialized into the run output for the selected pairwise
model.

#### What the replay script does

`scripts/production/model/train_linker_and_finalize.py` is the official release
entrypoint for the promoted incremental linker. It delegates to
`scripts/production/model/linker_train_calibrate_eval.py`. It does not
train the pairwise model. It takes a pairwise production bundle stage as input,
recomputes or loads the promoted linker feature tables,
trains/calibrates/evaluates the downstream LightGBM linker, writes the runtime
linker artifact, and finalizes the complete production bundle.

Its main inputs are:

- `--pairwise-model-path`: the pairwise model whose distances feed the linker.
- `--source-bundle-root`: the Arrow+labels train/calibrate/eval bundle.
- `--target-json`: the replay target with feature order, LightGBM params,
  expected metrics, status, and variant.
- `--output-dir`: the scratch run directory for materialized features,
  summaries, and replay outputs.
- `--save-artifact-to`: optional output directory for `booster.lgb` and
  `metadata.json`; this is a low-level linker-only output.
- `--save-production-bundle-to`: preferred release output. It writes the linker
  under `incremental_linker/`, copies the target JSON into `reproducibility/`,
  refreshes linker audit metadata, and writes the final bundle manifest.

In the default `--feature-mode arrow-rust`, the script rebuilds promoted
features from the Arrow source bundle. For each selected table and dataset, it
loads the Arrow papers, signatures, SPECTER2 rows, and labels; applies
structural cleaning; builds block-local query/candidate context; uses the frozen
Rust retrieval policy to choose candidate seed clusters; builds the
candidate/member pair plan; computes pairwise model distances and `pw_*`
aggregate features; adds the non-pairwise row features; then writes
target-ordered feature tables and bundle metadata under `--output-dir`. These
feature values are tied to the exact pairwise model passed with
`--pairwise-model-path`.

The other feature mode is narrower:

- `precomputed-promoted` loads an already materialized portable feature bundle.
  It validates relative table paths, row counts, required tables, target-spec
  digest, feature-schema digest, and exact target feature-column order before
  training.

After features are available, the script runs the classic train/calibrate/eval
stack. It trains the LightGBM linker with the target params, fits or applies the
configured NumPy logistic gate, evaluates the configured S2AND/Hwang/extra/manual
holdout tables, writes `classic/summary.json`, and writes `run_summary.json`
with observed metrics and deltas from the replay target JSON. Unless
`--allow-metric-drift` is passed, a full replay fails when observed metrics do
not match the target metrics.

When `--save-artifact-to` or `--save-production-bundle-to` is set, the script
also fits the final production linker on train rows plus weighted
`calibration_fit`/`calibration_check` rows, then fits the final NumPy logistic
gate on the configured `test` split and writes `booster.lgb` and
`metadata.json`. The metadata includes the feature schema, gate config, required
Rust capabilities, prediction fixture, booster digest, pairwise model
path/version/digest, source bundle, feature mode, observed metrics, and
production training summary. Keep the replay target under the bundle's
`reproducibility/` directory. `--save-production-bundle-to` is the
normal release path because it assembles the complete runtime directory and
validates it with `load_production_model(...)`.

Safety behavior is intentional: an unbounded full run requires `--run-full`;
`--datasets`, `--tables`, and `--limit-rows` are smoke/materialization controls
and require `--materialize-only`; and precomputed feature reuse is accepted only
through explicit `--feature-mode precomputed-promoted`.

The required update flow is:

1. Train the pairwise stage. This writes native LightGBM files, `clusterer.json`,
   pairwise prediction fixtures, pairwise reproducibility files, and a
   pairwise-only manifest.

```powershell
uv run python scripts\production\model\train_pairwise.py `
  --production-version X.Y `
  --output-dir s2and\data\production_model_vX.Y `
  --run-full
```

2. Create or update
   `s2and/data/production_model_vX.Y/reproducibility/incremental_linker_training_target.json`.
   Start from the previous target only when the 53-feature schema and LightGBM
   params are intentionally unchanged.
3. Run a bounded materialization smoke test before any full replay:

```powershell
uv run python scripts\production\model\train_linker_and_finalize.py `
  --pairwise-model-path s2and\data\production_model_vX.Y `
  --target-json s2and\data\production_model_vX.Y\reproducibility\incremental_linker_training_target.json `
  --output-dir scratch\joint_safe_link_promoted_vX.Y_smoke `
  --datasets qian `
  --limit-rows 200 `
  --materialize-only
```

4. Run the full train/calibrate/eval replay and finalize the full production
   bundle. This is a large job; report the command, expected runtime, output
   directory, and monitoring plan before starting it.

```powershell
uv run python scripts\production\model\train_linker_and_finalize.py `
  --production-bundle-version X.Y `
  --pairwise-model-path s2and\data\production_model_vX.Y `
  --target-json s2and\data\production_model_vX.Y\reproducibility\incremental_linker_training_target.json `
  --save-production-bundle-to s2and\data\production_model_vX.Y `
  --linker-artifact-version vX.Y `
  --output-dir scratch\joint_safe_link_promoted_vX.Y_full `
  --run-full
```

Use `--allow-metric-drift` only for exploratory candidate runs when the target
metrics are intentionally stale. Do not use it as the final release gate.

5. Review `run_summary.json`, `classic/summary.json`, and
   `prod_artifact_summary.json`. Report reviewed-label quality,
   setup-inclusive and hot-path wall time, candidate rows, scored pairs,
   residual pairs, residual count, exact-tail memory behavior, and observed RSS
   versus `total_ram_bytes`.
6. Promote the release as a coordinated update: the finalized
   `production_model_vX.Y/` directory, default paths in `s2and/model.py` and
   `scripts/production/model/linker_train_calibrate_eval.py`, and tests/docs
   that hard-code the release version. `pyproject.toml` uses version-agnostic
   `production_model_v*/...` package-data patterns, so adding a new release
   directory should not require package-data edits.
7. Verify the promoted artifact and release wiring:

```powershell
uv run pytest -q tests/test_promoted_linker_training_cli.py tests/test_linker_feature_assembly.py tests/test_incremental_linking_default_artifact.py tests/test_production_model.py tests/test_production_model_cli_flow.py
uv run ruff check scripts/production/model/linker_train_calibrate_eval.py tests/test_promoted_linker_training_cli.py tests/test_linker_feature_assembly.py tests/test_incremental_linking_default_artifact.py tests/test_production_model.py tests/test_production_model_cli_flow.py
```

For repeated replay, `--feature-mode precomputed-promoted` is allowed only when
the precomputed bundle was materialized for the same target and pairwise model;
the full default replay recomputes promoted features from the source bundle.

Training/evaluation replay normally recomputes promoted features from the
Arrow source bundle. For compute-once/reuse workflows, the replay script also
supports an explicit portable precomputed bundle mode:

```powershell
uv run python scripts\production\model\linker_train_calibrate_eval.py `
  --feature-mode precomputed-promoted `
  --precomputed-feature-bundle-root path\to\minimal_raw_feature_bundle `
  --run-full
```

The script validates relative table paths, row counts, target/schema digests,
required tables, and exact 53-feature column equality before training. There is
no shipped machine-local default for precomputed feature tables.

## Reference-feature behavior

Reference features have been removed from S2AND entirely. The featurizer no
longer computes features derived from cited references, `ANDData` ignores
`papers.references`, and the feature vector no longer reserves columns for
them. Model `v1.0`, which required reference features, is no longer usable
with current S2AND.

## Minimal input contract

Minimal paper entry for `v1.21`, `v1.2`, and `v1.1`:

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
  ]
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

- `production_model_v1.21/`, `production_model_v1.2.pickle`, and `production_model_v1.1.pickle` use `initial_char` with
  `s2and/data/name_counts.pickle`; that pickle stores keys like `smith j`, not `smith john`.
- In `ANDData(..., mode="inference")`, prediction automatically applies the semantics expected by the loaded model via the stored feature contract.
- Do not mix model artifacts and feature semantics without retraining.

## Minimal Arrow Prediction Flow

```python
from scripts.eval_prod_models import (
    read_arrow_s2_blocks,
    resolve_arrow_dataset_paths,
    split_blocks_like_anddata,
)
from s2and.production_model import load_production_model

clusterer = load_production_model("s2and/data/production_model_v1.21")

arrow_paths = resolve_arrow_dataset_paths(
    "s2and/data",
    "qian",
    "_specter2.pkl",
)
_, _, test_blocks = split_blocks_like_anddata(read_arrow_s2_blocks(arrow_paths["signatures"]), random_seed=42)

pred_clusters, pred_distance_matrices = clusterer.predict_from_arrow_paths(
    test_blocks,
    {key: value for key, value in arrow_paths.items() if key != "clusters"},
    total_ram_bytes=32 * 1024**3,
    load_name_counts=True,
    name_tuples="filtered",
)
```

`pred_distance_matrices` may be `None` when using memory-optimized fused clustering paths.
Use `scripts/tutorial_for_predicting_with_the_prod_model.py --input-format arrow`
for a runnable CLI example. JSON/`ANDData` prediction remains a compatibility
path for fixtures, training references, and parity checks; it is not the
production Rust inference route.

## Caching

Public cache controls for non-Arrow featurization paths:

- `Clusterer.use_cache`
- `featurize(..., use_cache=...)`
- `many_pairs_featurize(..., use_cache=...)`

Semantics:

- `use_cache=True` enables the persistent pair-feature SQLite cache.
- `use_cache=False` skips those persistent cache reads and writes.
- Same-process Rust featurizer reuse still stays enabled even when
  `use_cache=False`.
- Direct Arrow/Rust production prediction bypasses the pair-feature SQLite
  cache and reads the request/runtime Arrow artifacts directly.

Recommended defaults:

- Repeated inference on the same dataset or pair set: `use_cache=True`
- One-shot jobs and experiments: `use_cache=False`
- Direct Arrow production inference: keep Arrow artifacts local and do not rely
  on the pair-feature SQLite cache.

Full cache details: [caching.md](caching.md)

## Rust backend

`S2AND_BACKEND` controls runtime backend selection:

- `auto`: use Rust when available and capable, otherwise Python
- `rust`: strict Rust mode
- `python`: Python-only mode

Python callers can also pass `backend="python"`, `backend="rust"`, or `backend="auto"` to
`Clusterer.predict(...)` for a single-call override. Subblocking follows the resolved backend:
direct `make_subblocks(...)` calls remain Python, while indexed Arrow subblocking in
`Clusterer.predict(...)` can run in Rust when the resolved backend is Rust.

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

Production Rust inference should call `Clusterer.predict_from_arrow_paths(...)`
or provide complete Arrow paths to `Clusterer.predict(...)`: at least
`signatures`, `papers`, `paper_authors`, and their raw-planner batch-index
sidecars. Models that use SPECTER features also require `specter` plus
`specter_batch_index`, and models that use name-count features require
`name_counts_index`. The direct Arrow route validates required keys and declared
files before Rust featurizer construction and raises
`MissingArrowArtifactError` with structured `missing_keys` and `missing_files`
fields. When the generic `Clusterer.predict(...)` route resolves to Rust,
including subblocked large-block prediction, it follows the same strict artifact
rule and raises instead of falling back to `ANDData`; select the Python backend
explicitly for compatibility/reference execution.

Request-time filtered Arrow reads validate sidecar magic, key column, file
length, source file size, and whether the source changed during the read. They
intentionally do not fingerprint the full source file per request. Run strict
fingerprint validation when publishing or deploying Arrow artifacts, and rebuild
batch indexes after any source Arrow rewrite, including same-size rewrites.

Incremental prediction with explicit RAM budget:

```python
result = clusterer.predict_incremental(
    block_signatures,
    dataset,
    batching_threshold=5000,
    total_ram_bytes=32 * 1024**3,
)

clusters = result["clusters"]
```

Incremental prediction directly from Arrow paths, without constructing an
`ANDData`-shaped dataset object:

```python
result = clusterer.predict_incremental_from_arrow_paths(
    block_signatures,
    {
        "signatures": ".../signatures.arrow",
        "papers": ".../papers.arrow",
        "paper_authors": ".../paper_authors.arrow",
        "signatures_batch_index": ".../signatures.signatures_batch_index.bin",
        "papers_batch_index": ".../papers.papers_batch_index.bin",
        "paper_authors_batch_index": ".../paper_authors.paper_authors_batch_index.bin",
        "specter": ".../specter2.arrow",
        "specter_batch_index": ".../specter2.specter_batch_index.bin",
        "name_counts_index": ".../name_counts_index",
        "cluster_seeds": ".../cluster_seeds.arrow",
    },
    batching_threshold=5000,
    total_ram_bytes=32 * 1024**3,
    name_tuples="filtered",
)

clusters = result["clusters"]
```

The direct Arrow incremental API preserves caller-supplied seed component ids
from `cluster_seeds` or `cluster_seeds_require`. The legacy `ANDData`
cluster-seed JSON loader may instead densify seed component ids to `0..N-1`.
Compare partitions rather than exact cluster-id keys when checking parity
between these entrypoints.

### Rust promoted incremental target

The target behavior is that `Clusterer.predict_incremental(...)` uses the
promoted Rust linker when `S2AND_BACKEND` selects Rust, the extension has the
required promoted-incremental capabilities, and seed inputs are available from
the dataset or Arrow artifacts. Without base Arrow artifacts or seed inputs,
Rust mode raises `MissingArrowArtifactError` before seed sync or helper
fallback. Legacy output parity is not a release goal; the promoted path
intentionally uses different retrieval, linker, and logistic-gate decisions.

There is no separate public force flag or artifact override: backend selection
plus seed availability is the routing contract. Promoted query batching is
available: `batching_threshold` caps the number of unassigned query signatures
per promoted linker batch, while `total_ram_bytes` derives the default batch
size when the caller does not pass a cap. The first meaningful promoted batch
recalibrates rows/pairs per query for remaining batches, and telemetry records
predicted/observed RSS deltas.

The current Rust inference boundary, direct Arrow path, and remaining
Python-heavy paths are summarized in
[rust/inference_architecture.md](rust/inference_architecture.md).

Promoted Rust `predict_incremental(...)` requires FeatureBlock Arrow artifacts
and seed inputs, then uses the raw Arrow/Rust retrieval and scoring bridge for
Phase A before finishing residual abstains through the normal incremental
completion path. Seeds can come from `cluster_seeds.arrow` or from
`dataset.cluster_seeds_require`; when the latter is used, the runtime writes a
request-local temporary seed table for Rust retrieval. A promoted Rust
incremental request without base Arrow paths or without either seed source fails
before seed sync or helper fallback. `cluster_seed_disallows.arrow` is optional
and means "no pairwise seed disallow constraints" when omitted, but an explicit
path must exist.

### Incremental Seed Telemetry Contract

Seed setup telemetry is part of the promoted incremental compatibility surface.
Refactors may move the implementation, but these keys must keep their names and
meanings until callers have an explicit migration path.

| Key | Meaning | Emitter |
|---|---|---|
| `seed_setup_seconds` | Wall time spent building request seed state. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_seed_signature_count` | Number of signatures with seed assignments after request seed setup. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_component_count` | Number of unique seed components after request seed setup. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_altered_signature_count` | Count of altered-cluster signatures considered during seed setup. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_altered_presplit_block_count` | Altered-cluster blocks sent through pre-split reclustering. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_altered_presplit_signature_count` | Altered-cluster signatures sent through pre-split reclustering. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_altered_presplit_predict_seconds` | Wall time spent in altered-cluster pre-split prediction. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_altered_presplit_cache_hit_count` | Pre-split reclustering cache hits. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_altered_presplit_cache_miss_count` | Pre-split reclustering cache misses. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_altered_presplit_orcid_skip_count` | Altered-cluster reclustering skips caused by compatible normalized ORCID groups. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_recluster_map_entry_count` | Number of temporary recluster component ids mapped back to source components. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_altered_cluster_count` | Number of source altered clusters involved in seed setup. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_cluster_seeds_source` | Seed assignment authority used for the request, currently `python` or `arrow`. | `Clusterer._build_incremental_seed_setup` |
| `seed_setup_cluster_seeds_from_arrow` | Integer flag for `seed_setup_cluster_seeds_source == "arrow"`. | `Clusterer._build_incremental_seed_setup` |

The bulk subblocked altered-profile pre-split path emits a second prefix. These
keys are stored on `_last_subblocked_altered_presplit_telemetry` and may also be
reported alongside promoted incremental telemetry.

| Key | Meaning | Emitter |
|---|---|---|
| `bulk_altered_presplit_applied` | Integer flag indicating whether the bulk pre-split path ran. | `Clusterer.predict` |
| `bulk_altered_presplit_seconds` | Wall time spent in the bulk pre-split setup. | `Clusterer.predict` |
| `bulk_altered_presplit_seed_signature_count` | Number of seed signatures after bulk pre-split seed setup. | `Clusterer.predict` |
| `bulk_altered_presplit_recluster_map_entry_count` | Number of temporary recluster ids mapped back to source components. | `Clusterer.predict` |
| `bulk_altered_presplit_block_count` | Altered pre-split block count copied from `seed_setup_altered_presplit_block_count`. | `Clusterer.predict` |
| `bulk_altered_presplit_signature_count` | Altered pre-split signature count copied from `seed_setup_altered_presplit_signature_count`. | `Clusterer.predict` |
| `bulk_altered_presplit_cache_hit_count` | Cache-hit count copied from `seed_setup_altered_presplit_cache_hit_count`. | `Clusterer.predict` |
| `bulk_altered_presplit_cache_miss_count` | Cache-miss count copied from `seed_setup_altered_presplit_cache_miss_count`. | `Clusterer.predict` |
| `bulk_altered_presplit_orcid_skip_count` | ORCID-skip count copied from `seed_setup_altered_presplit_orcid_skip_count`. | `Clusterer.predict` |

Supporting docs:

- Subblocking behavior and tradeoffs: [subblocking.md](subblocking.md)
- Threading guidance: [threading.md](threading.md)
- Environment variables: [environment.md](environment.md)

## Warm-starting compatibility featurizers

`warm_rust_featurizer(dataset)` preloads the same-process Rust featurizer for
`ANDData`/JSON compatibility paths. It is not the production Arrow inference
entrypoint. Production services should keep model bundles and Arrow runtime
artifacts local, then call `Clusterer.predict_from_arrow_paths(...)` or
Arrow-routed `Clusterer.predict(...)` with complete artifact paths.

For long-lived compatibility services that still own an `ANDData` object, you
can pre-warm once at startup:

```python
from s2and.feature_port import warm_rust_featurizer

warm_rust_featurizer(dataset)
```

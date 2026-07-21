# Production Inference

S2AND is in the canonical-v2 cutover. This branch deliberately distributes no
production model and declares no default. `load_production_model()` without a
path therefore raises; callers must pass an explicit compatible bundle.

The removed `production_model_v1.0.pickle` through
`production_model_v1.2.pickle` files are not package assets. The checked-in
`production_model_v1.21/` directory is only an explicit historical source and
parity artifact. It is not packaged, is rejected by canonical-v2, and must not
be used for inference with this branch. Use the previous compatible S2AND
release when reproducing v1.21 behavior.

The remaining release work is tracked in [work_plan.md](work_plan.md).

## Complete model bundles

The public loader accepts one format: an explicit, complete, canonical native
bundle.

```python
from s2and.production_model import load_production_model

clusterer = load_production_model("/path/to/production_model_v1.3")
```

A complete bundle contains the pairwise boosters, clusterer configuration,
promoted incremental linker, reproducibility target, and a manifest that binds
their schemas, versions, checksums, and normalization provenance. The important
runtime layout is:

```text
production_model_vX.Y/
  manifest.json
  clusterer.json
  pairwise/
    main.lgb
    main_prediction_fixture.json
    nameless.lgb
    nameless_prediction_fixture.json
  incremental_linker/
    booster.lgb
    metadata.json
  reproducibility/
    incremental_linker_training_target.json
```

The v4 complete manifest requires the fixed runtime entries shown above. The
pairwise training config and summary may also appear, but only together as a
reproducibility pair. The `files` mapping must match one of those two shapes
exactly. The loader hashes every declared file once, derives the fixed
`incremental_linker/` directory from the schema, and ignores unrelated files
that are not part of the runtime contract.

`load_production_model(path)` rejects legacy pickles, incomplete directories,
and `pairwise_only` manifests. A pairwise-only bundle is an internal training
stage and has a separate internal loader; it is never a public inference model.
The loader validates the complete bundle, loads the linker once, and retains
that immutable native artifact on the clusterer. Deep copies share it; pickle
round trips revalidate and reload it from the recorded bundle path.

`clusterer.json` is the sole authority for the clustering `eps`. There are no
version- or path-based threshold overrides.

### Staged publication

Pairwise training writes an immutable pairwise-only source. Finalization does
not add a linker to that directory and does not transition its manifest in
place. It assembles a complete sibling staging directory, copies the verified
pairwise files and newly trained linker into it, validates the complete staged
bundle, and renames that directory once into a previously nonexistent final
path. It does not use publication locks, directory fsyncs, or identical-tree
rewrite handling.

A failed finalization therefore leaves the pairwise source unchanged and does
not expose a partial final bundle. Publishing to an existing final path is an
error; a changed release gets a new path.

A release flow is:

```powershell
uv run python scripts\production\model\train_pairwise.py `
  --production-version X.Y `
  --output-dir scratch\pairwise_stage\production_model_vX.Y `
  --run-full

uv run python scripts\production\model\train_linker_and_finalize.py `
  --production-bundle-version X.Y `
  --pairwise-model-path scratch\pairwise_stage\production_model_vX.Y `
  --source-bundle-root path\to\canonical_arrow_training_bundle `
  --target-json scratch\production_linker_vX.Y\incremental_linker_training_target.json `
  --save-production-bundle-to s2and\data\production_model_vX.Y `
  --linker-artifact-version vX.Y `
  --output-dir scratch\joint_safe_link_promoted_vX.Y_full `
  --run-full
```

Run the existing small materialization smoke before the full command. The full
retrain is a large job and requires explicit owner approval, captured logs, and
quality/runtime/RSS evidence. `--allow-metric-drift` is diagnostic-only and
cannot publish a linker or production bundle.

## Explicit execution routes

The input type determines the backend. There is no automatic backend,
capability matrix, or Python/Rust fallback:

| API | Input authority | Backend | Result |
| --- | --- | --- | --- |
| `Clusterer.predict` | `ANDData` | Python | `(clusters, distance_matrices)` |
| `Clusterer.predict_incremental` | `ANDData` | Python | structured mapping |
| `Clusterer.predict_from_arrow_paths` | immutable Arrow paths | Rust | `(clusters, distance_matrices)` |
| `Clusterer.predict_incremental_from_arrow_paths` | immutable Arrow paths | Rust | structured mapping |

Passing a Rust runtime context to an `ANDData` API, or a Python context to an
Arrow API, is an error. Rust entry points import the exact `s2and-rust` version
pinned by the project metadata and fail on a missing or different version. That
dependency is part of the normal S2AND install; there is no `s2and[rust]`
compatibility extra.

Classic `ANDData` prediction remains useful for Python training, fixtures, and
reference checks:

```python
pred_clusters, pred_distance_matrices = clusterer.predict(
    dataset.get_blocks(),
    dataset,
)

result = clusterer.predict_incremental(block_signatures, dataset)
clusters = result["clusters"]
```

Incremental APIs always return their structured result. There is no
`return_clusters_only` mode.

## Immutable Arrow contract

Production Arrow input is a manifest-backed immutable generation. Validation
checks the fixed prediction or training profile, normalization provenance, full
generation inventory, checksums, table schemas, and batch-index fingerprints
at the construction/deployment boundary. A successfully validated generation
is then trusted and reused by exact identity; requests do not install
filesystem watchers or repeatedly probe for same-path mutation.

Never edit an Arrow generation in place. Publish changed content under a new
generation/path, validate it once, and construct a new dataset or featurizer.
Request-local seed sidecars are separate from the immutable base generation.

Prediction requires `signatures`, `papers`, `paper_authors`, and their
raw-planner batch indexes. A model that uses embeddings also requires `specter`
and its batch index; a model that uses global name counts requires
`name_counts_index`. The validator reports missing keys/files explicitly before
building a Rust featurizer.

`author_orcid` is optional in `signatures.arrow`. When the column is absent,
every signature simply has no ORCID evidence; all other required signature
columns retain their normal validation.

### Rust training constructor

Rust training has one constructor:
`s2and.arrow_training.build_training_anddata_from_arrow`. It validates the
fixed training profile and returns a fully initialized train-mode `ANDData`.
The dataset owns one read-only `dataset.arrow_paths` mapping and one verified
generation identity; there is no second Arrow-path field or caller mutation
step.

```python
from s2and.arrow_training import build_training_anddata_from_arrow
from s2and.consts import NORMALIZATION_VERSION

dataset = build_training_anddata_from_arrow(
    arrow_paths,
    "training_dataset",
    expected_normalization_version=NORMALIZATION_VERSION,
    clusters="/path/to/clusters.json",
)
```

Python SPECTER arrays and name-count mappings are not materialized for this
route. Rust featurization reads them from `dataset.arrow_paths`.

### Rust inference

Full-block inference accepts blocks plus the validated Arrow-path mapping:

```python
pred_clusters, pred_distance_matrices = clusterer.predict_from_arrow_paths(
    blocks,
    arrow_paths,
    total_ram_bytes=32 * 1024**3,
)
```

Promoted incremental inference accepts either `cluster_seeds` in the Arrow
paths or an explicit seed mapping:

```python
result = clusterer.predict_incremental_from_arrow_paths(
    block_signatures,
    arrow_paths,
    cluster_seeds_require=signature_to_seed_cluster,
    batching_threshold=5000,
    total_ram_bytes=32 * 1024**3,
)

clusters = result["clusters"]
```

The optional `cluster_seed_disallows` sidecar means “no seed-disallow evidence”
when absent. An explicit sidecar path must exist. Arrow seed component IDs are
preserved, so parity checks should compare partitions when another ingestion
path assigns different cluster IDs.

## Incremental decision semantics

Promoted query decisions are request-global and deterministic across input
order, query batches, and `batching_threshold`. A global conflict is resolved
by rebuilding and rescoring the lower-priority query from its complete
single-query candidate plan with the winning component excluded. The runtime
does not replay a compact retained subset as a substitute for that complete
plan.

RAM limits are refreshed before each batch and after planner and featurizer
allocations. If a changed limit shrinks the batch, its unscored remainder is
queued and the safe prefix is replanned before scoring. Planning uses the loaded
model’s actual final, pairwise, and aggregate feature widths.

## Name-count and cache boundaries

Canonical-v2 uses one binary `NameCountsIndex` in Python and Rust. Classic
`ANDData` accepts either a verified index path, a shared open index handle, or
`None` when count features are intentionally absent. Arrow routes carry the
same index under `arrow_paths["name_counts_index"]`. Runtime code does not open
the historical source pickle. Python deduplicates each 2,048-signature batch
before unique keys cross the native boundary, scatters the four result columns
back onto signatures, and discards all temporary key maps with the batch. The
v1 `pickle_sha256` field is source-lineage metadata, not a runtime dependency.

Python is the sole name-tuple artifact loader. It validates the data and
adjacent metadata once for the packaged immutable artifact, retains frozen
pairs plus `data_sha256`, and passes those pairs explicitly to Rust-backed
flows.

Canonical ORCID prefix counts likewise use a direct JSON data file and adjacent
`.meta.json` sidecar. The lazy runtime loader reads each once, validates the
small metadata schema and data SHA-256, and retains the result in-process. There
is no ORCID generation pointer, publication lock, retry loop, or legacy
fallback. During cutover, the checked-in legacy JSON and absent canonical
sidecar remain excluded from distributions until the approved canonical
generation is produced.

`last_first_initial_count_min` uses
`<canonical last> <canonical first[0]>` when both fields exist and a null key
otherwise. Models, datasets, count indexes, ORCID counts, name tuples, and the
linker must agree on their canonical-v2 provenance; legacy artifacts are
rejected rather than adapted.

Production inference has no persistent cache. Direct Arrow/Rust prediction
reuses already validated immutable native state in-process only. See
[caching.md](caching.md) for details.

## Verification

The focused code gate is:

```powershell
uv run pytest -q tests/test_production_model.py tests/test_production_model_cli_flow.py tests/test_arrow_production_boundary.py tests/test_arrow_training_ingestion.py tests/test_cluster_incremental.py
uv run pytest -q tests/test_name_tuple_artifact.py tests/test_generate_orcid_name_prefix_counts.py
uv run ruff check s2and scripts/production/model tests/test_production_model.py tests/test_production_model_cli_flow.py tests/test_arrow_production_boundary.py tests/test_arrow_training_ingestion.py tests/test_cluster_incremental.py
git diff --check
```

These tests validate code and synthetic bundles only. A release additionally
needs a clean wheel/sdist check, exact-version Rust wheel install, a real
canonical bundle, and the approved quality/runtime/RSS gates.

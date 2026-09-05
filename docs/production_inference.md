# Production Inference

Status date: 2026-07-27.

S2AND is in the `1.0.0` cutover. This branch deliberately distributes no
production model and declares no default. `load_production_model()` without a
path therefore raises; callers must pass an explicit compatible bundle.

The removed production-model pickles and previous v1.21 bundle are not package
assets. Use the previous compatible S2AND release when reproducing v1.21
behavior; current instructions never use its deleted repository path.

The component behavior in this document is implemented, but a releasable v1.3
bundle does not yet exist. Use [release.md](release.md) for
execution order and acceptance requirements.

## Import boundaries

Importing `s2and.production_model` does not load evaluation, plotting, or
Hyperopt modules. Pure clustering metrics live in `s2and.metrics`; existing
imports from `s2and.eval` remain available. Evaluation loads plotting libraries
when rendering and scopes its plot styling to that operation.

Constructing a `Clusterer` or loading a production bundle does not create an
optimizer search space or load Hyperopt. With the default `search_space=None`,
the attribute remains `None` until `Clusterer.fit()` or Arrow validation
calibration creates the default uniform EPS space. Explicit search spaces
remain unchanged. Genie loads only when its subblocking algorithm is used.

## Prediction state ownership

Prediction orchestration keeps effective seed assignments, disallow pairs,
altered-profile overrides, and internal diagnostics in request-owned state.
Bulk subblocking builds synthetic seeds from its completed subblocks without
temporarily replacing the seed collections on `ANDData`. Nested predictions
and failed subblock calls therefore leave the dataset's original seed
collections intact.

Prepared Rust feature data is reusable; the effective seed constraints belong
to the prediction using it. Python orchestration passes those constraints at
the Rust boundary instead of installing request overrides into cached feature
data. The public prediction entry points retain their existing contracts.

This ownership boundary is not a general thread-safety guarantee for every
classifier or cache supplied by an application.

Seed-link application is a pure decision phase shared by classic Python and
promoted Arrow incremental completion. It accepts ordered seed memberships,
link decisions, altered-profile mappings, and first-name metadata; it returns
owned cluster lists, residual signatures, and rejected links. It preserves
split-specific name checks and restores original profile IDs. The orchestrator
resolves name aliases, logs rejections, and clusters the residual signatures.
Name metadata is accessed lazily without copying the prepared dataset.

## Complete model bundles

The public loader accepts one format: an explicit, complete, canonical native
bundle.

```python
from s2and.production_model import load_production_model

clusterer = load_production_model("/path/to/production_model_v1.3")
```

A complete bundle contains the pairwise boosters, clusterer configuration,
promoted incremental linker, reproducibility target, and a manifest that binds
its release, exact generating runtime, EPS lifecycle, and checksums. The
important runtime layout is:

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

The root manifest has exactly `kind: "s2and_model"`, `release_version`,
`generated_by_runtime`, `eps_calibration`, and `sha256`.
`generated_by_runtime` must exactly equal the installed `s2and` and
`s2and-rust` version. The checksum inventory determines whether the bundle is
pairwise-only or complete; `bundle_kind` is not serialized. Pairwise-only
bundles may be `pending` or `calibrated`, while complete bundles must be
`calibrated`. Runtime paths are fixed by the contract rather than serialized
again. The pairwise training config and summary may also appear, but only
together as a reproducibility pair. The loader requires exact checksum
coverage for the derived paths, hashes every declared file once, and ignores
unrelated files that are not part of the runtime contract.

`load_production_model(path)` rejects legacy pickles, incomplete directories,
and `pairwise_only` manifests. A pairwise-only bundle is an internal training
stage and has a separate internal loader; it is never a public inference model.
The loader validates the complete bundle, loads the linker once, and retains
that immutable native artifact on the clusterer. Deep copies share it; pickle
round trips revalidate and reload it from the recorded bundle path.

`clusterer.json` is inference-only. It contains the pairwise feature contracts,
runtime controls, and only the clustering algorithm fields consumed by
prediction (`eps` and `linkage`). Stage 3 records placeholder EPS `0.5`; the
manifest marks that value as pending until validation-only calibration writes
a calibrated sibling bundle. It has no independent schema, normalization, or
featurizer version counter.

### Staged publication

Pairwise training writes an immutable pairwise-only, EPS-pending source.
Calibration changes only `clusterer.json` and `manifest.json` in a fresh
pairwise-only sibling. Linker training and finalization accept only the
calibrated sibling. Finalization does
not add a linker to that directory and does not transition its manifest in
place. It assembles a complete sibling staging directory, copies the verified
pairwise files and newly trained linker into it, validates the complete staged
bundle, and renames that directory once into a previously nonexistent final
path. It does not use publication locks, directory fsyncs, or identical-tree
rewrite handling.

A failed finalization therefore leaves the pairwise source unchanged and does
not expose a partial final bundle. Publishing to an existing final path is an
error; a changed release gets a new path.

Production training first writes an immutable pairwise-only stage. After
validation-only EPS selection, the linker release command rematerializes
pairwise-derived features, performs one fresh linker fit, writes a complete
bundle in a new directory, reloads it, and evaluates through that exact
serialized artifact. The embedded
`reproducibility/incremental_linker_training_target.json` remains part of the
complete manifest.

`train_linker_and_finalize.py` is one direct, expensive-job entrypoint; its
feature materialization and staging are temporary. The
[production command reference](../scripts/production/README.md) documents its
arguments, and the [v1.3 runbook](release.md) owns release order and
gates.

## Explicit execution routes

The input type determines the backend. There is no automatic backend,
capability matrix, or Python/Rust fallback:

| API | Input authority | Backend | Result |
| --- | --- | --- | --- |
| `Clusterer.predict` | `ANDData` | Python | `(clusters, distance_matrices)` |
| `Clusterer.predict_incremental` | `ANDData` | Python | structured mapping |
| `Clusterer.predict_from_arrow` | open `ArrowDataset` | Rust | `(clusters, distance_matrices)` |
| `Clusterer.predict_incremental_from_arrow` | open `ArrowDataset` | Rust | structured mapping |

Passing a Rust runtime context to an `ANDData` API, or a Python context to an
Arrow API, is an error. Rust entry points import the exact `s2and-rust` version
pinned by the project metadata and fail on a missing or different version. That
dependency is part of the normal S2AND install; there is no `s2and[rust]`
compatibility extra.

Classic `ANDData` prediction remains useful for Python training, fixtures, and
reference checks over the canonical S2 partition. `author_info.block` is its
sole grouping authority. The legacy `block_type` selector,
`author_info.given_block`, `get_original_blocks()`, and `get_s2_blocks()` are
not part of the current API; preserve any historical/original partition
outside `ANDData` if it is still needed.

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

Production Arrow input is one manifest-backed immutable root. Open it once with
`ArrowDataset.open(root)`: opening requires `kind: "s2and_arrow_dataset"` and
public `format_version: 1`, validates the flat content inventory, checksums,
table schemas, and batch indexes, then retains the exact files for the handle's
lifetime.

Never edit an open root in place. Publish changed content at a new path and open
a new handle. Request-local seed constraints are explicit prediction arguments,
not mutations to the base dataset.

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
`s2and.arrow_training.build_training_anddata_from_arrow`. It accepts one open
handle, validates the training profile, and returns a fully initialized
train-mode `ANDData` that retains that handle.

```python
from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_training import build_training_anddata_from_arrow

arrow_dataset = ArrowDataset.open(
    "/path/to/arrow_dataset",
    require_name_counts_index=True,
)
dataset = build_training_anddata_from_arrow(
    arrow_dataset,
    "training_dataset",
    clusters="/path/to/clusters.json",
)
```

Python SPECTER arrays and name-count mappings are not materialized for this
route. Rust reads them through the retained handle. Keep `arrow_dataset` open
until the returned training dataset is no longer used.

### Rust inference

Full-block inference accepts blocks plus one open handle:

```python
from s2and.arrow_inputs import ArrowDataset

with ArrowDataset.open("/path/to/arrow_dataset") as arrow_dataset:
    pred_clusters, pred_distance_matrices = clusterer.predict_from_arrow(
        blocks,
        arrow_dataset,
        batching_threshold=15_000,
        cluster_seeds_require=current_seed_assignments,
        altered_cluster_signatures=corrected_claimed_profiles,
        total_ram_bytes=32 * 1024**3,
    )
```

`batching_threshold` is optional. When set, blocks larger than the threshold
use strict Arrow-native Rust subblocking; smaller blocks retain the full-block
route. The native graph fallback requires indexed SPECTER evidence even when
the pairwise model itself does not select an embedding feature. Initial-only
groups attach through the production bundle's promoted incremental linker.
When altered claimed profiles are supplied on a subblocked request, their
components are naturally pre-split before native subblocking; the same explicit
request-local seed view is shared by subblocking and featurization.

Promoted incremental inference requires an explicit seed mapping:

```python
with ArrowDataset.open("/path/to/arrow_dataset") as arrow_dataset:
    result = clusterer.predict_incremental_from_arrow(
        block_signatures,
        arrow_dataset,
        cluster_seeds_require=signature_to_seed_cluster,
        batching_threshold=5000,
        total_ram_bytes=32 * 1024**3,
    )

clusters = result["clusters"]
```

Optional disallow pairs are passed through `cluster_seeds_disallow`. Seed
component IDs are preserved, so parity checks should compare partitions when
another ingestion path assigns different cluster IDs.

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

## Name-count and reuse boundaries

The current runtime uses one binary `NameCountsIndex` in Python and Rust. Classic
`ANDData` defaults to the canonical configured index and also accepts a
verified alternate path, a shared open index handle, or explicit `None` when
count features are intentionally absent. Arrow routes retain the manifest-bound
index inside `ArrowDataset`. Runtime code does not open the
historical source pickle, and the generator no longer publishes one. Models
bind directly to the native index's single `manifest_sha256`. The manifest
contains exactly `kind: "s2and_name_counts"`, public `format_version: 1`, and
byte count plus SHA-256 for the four fixed binary roles. Paths are derived as
`<role>.bin`; producer mode and output-count metrics are reported by the
producer instead of entering the runtime identity. Python deduplicates each
2,048-signature batch before
unique keys cross the native boundary, scatters the four result columns back
onto signatures, and discards all temporary key maps with the batch.

Python is the sole name-tuple artifact loader. It validates the canonical text
directly, retains frozen pairs plus the computed `data_sha256`, and passes
those pairs explicitly to Rust-backed flows.

Canonical ORCID prefix counts use a direct JSON data file and one adjacent
`.manifest.json` containing only the `name_tuples_sha256` dependency. The lazy
runtime loader validates the prefix-pair keys and positive integer counts,
computes the data hash, and retains the result in-process.
Production contracts bind that hash and require the manifest's tuple hash
to match the canonical name-tuple artifact. There is no ORCID generation
pointer, producer-provenance protocol, retry loop, or legacy fallback. The
package declares both ORCID runtime paths, but the current pre-release tree
contains neither file. Stage 1 of the v1.3 runbook must generate, review, and
copy the approved pair before distribution verification can pass.

`last_first_initial_count_min` uses
`<canonical last> <canonical first[0]>` when both fields exist and a null key
otherwise. Production feature contracts bind the exact name-count manifest,
the ORCID and name-tuple data hashes, and the linker. The model root's exact
generating-runtime match owns behavioral compatibility; legacy artifacts are
rejected rather than adapted.

Production inference has no persistent feature-snapshot or artifact cache.
Direct Arrow/Rust prediction reuses already validated immutable native state
in-process; the bounded altered-profile presplit memo is likewise a private
same-process optimization, not a persisted format.

## Verification

The focused code gate is:

```powershell
uv run pytest -q tests/test_production_model.py tests/test_arrow_production_boundary.py tests/test_arrow_training_ingestion.py tests/test_cluster_incremental.py
uv run pytest -q tests/test_name_tuple_artifact.py tests/test_generate_orcid_name_prefix_counts.py
uv run pytest -q tests/test_train_pairwise_script.py tests/test_promoted_linker_training_cli.py tests/test_eval_prod_models.py
uv run ruff check s2and scripts/production/model tests/test_production_model.py tests/test_arrow_production_boundary.py tests/test_arrow_training_ingestion.py tests/test_cluster_incremental.py
git diff --check
```

These tests validate code and synthetic bundles only. A release additionally
needs a clean wheel/sdist check, exact-version Rust wheel install, a real
canonical bundle, and the approved quality/runtime/RSS gates in
[release.md](release.md).

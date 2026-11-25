# Production Inference

Inference users need the common complete model and binary name-count index.
The `inference` S3 bundle holds both; [the data guide](data.md#download) syncs it into `s2and/data/`.
Small canonical name tuples and ORCID prefix counts ship in the Python wheel.

Pass an explicit complete model path to `load_production_model`.
The loader does not select a default model.

## Load a production model

The public loader accepts one format: an explicit, complete, canonical native
bundle.

```python
from s2and.production_model import load_production_model

clusterer = load_production_model("/path/to/production_model_v1.4", n_jobs=8)
```

Without `n_jobs`, a loaded bundle keeps its stored `n_jobs=1` and scores
single-threaded; `clusterer.n_jobs` can also be changed after loading. See
[Complete model bundles](#complete-model-bundles) for the bundle layout and
what the loader validates.

## Explicit execution routes

The input type determines the backend, and there is no fallback between
backends:

| API | Input authority | Backend | Result |
| --- | --- | --- | --- |
| `ANDData(...)` | classic data inputs | Python | initialized `ANDData` |
| `build_training_anddata_from_arrow(...)` | open `ArrowDataset` | Rust | train-mode `ANDData` retaining the handle |
| `Clusterer.predict` | `ANDData` | Python | `(clusters, distance_matrices)`; matrices may be `None` |
| `Clusterer.predict_from_arrow` | open `ArrowDataset` | Rust | `(clusters, distance_matrices)`; matrices may be `None` |
| `Clusterer.predict_incremental_from_arrow` | open `ArrowDataset` | Rust | structured mapping |

Passing a Rust runtime context to an `ANDData` API, or a Python context to an
Arrow API, is an error. Rust entry points load the bundled `s2and._s2and_rust`
extension and reject a missing binary or a version mismatch.
Use `predict_incremental_from_arrow` for incremental prediction and
`predict_from_arrow` with `batching_threshold` for subblocking.

Lower-level callers of `s2and.runtime.build_runtime_context(...)` get Python
when they omit `backend`. No environment variable changes these routes.

Classic `ANDData` prediction remains useful for Python training, fixtures, and
reference checks over the canonical S2 partition. `author_info.block` is its
sole grouping authority; keep any other partition, such as original benchmark
blocks, outside `ANDData`.

```python
# dataset is an ANDData.
pred_clusters, pred_distance_matrices = clusterer.predict(
    dataset.get_blocks(),
    dataset,
)
```

Incremental APIs always return their structured result.

## Immutable Arrow contract

Published Arrow input is one manifest-backed immutable root. Open it once with
`ArrowDataset.open(root)`: opening requires `kind: "s2and_arrow_dataset"` and
public `format_version: 1`, validates the flat content inventory, checksums,
table schemas, and batch indexes, then retains the exact files for the handle's
lifetime.

Opening computes the manifest SHA-256 once per retained file. Reusing the open
handle does not hash files again. Production model cache identities likewise
reuse the model hashes verified when loading the bundle.

Batch indexes contain source sizes, key-column hashes, and key-to-batch records.
They do not contain source fingerprints. Writers always build without a separate
source checksum scan. Build tables and indexes together; `overwrite=False`
checks existing index structure and source size, but the caller must guarantee
that the pair belongs together. Rebuild indexes whenever the source changes.

For trusted caller-owned inputs, skip the dataset manifest entirely:

```python
from s2and.arrow_inputs import ArrowDataset
from s2and.incremental_linking.feature_block import write_raw_arrow_batch_lookup_indexes

paths, _ = write_raw_arrow_batch_lookup_indexes(table_paths, request_directory)
with ArrowDataset.open_trusted(paths) as dataset:
    clusters, distances = clusterer.predict_from_arrow(block_signatures, dataset)
```

`open_trusted` accepts the same table/index path keys and returns the same owning
handle as `open`. It performs no dataset checksum scans and still checks required tables,
paired optional inputs, Arrow schemas, index structure, and source sizes. The
caller is responsible for supplying matching, immutable tables and indexes.
It raises `ValueError` for the request-local keys `cluster_seeds`,
`cluster_seed_disallows`, `altered_cluster_signatures`, and `profile_exclusions`.
Read those tables with `read_cluster_seeds_arrow`, `read_cluster_seed_disallows_arrow`,
`read_altered_cluster_signatures_arrow`, and `read_profile_exclusions_arrow`,
respectively, and pass the results as prediction arguments.
Both modes open a name-count index with the same checks: manifest, file sizes, file headers,
and section bounds.
Each trusted open gets a fresh generation ID, so prediction caches can be reused
within that handle but cannot reuse entries from another trusted open. Its
`root` property is the directory containing the signatures file.

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

For `build_training_anddata_from_arrow` and its input requirements, see
[Build a Rust-backed training dataset](training.md#build-a-rust-backed-training-dataset).

### Rust inference

Full-block inference accepts blocks plus one open handle:

```python
from s2and.arrow_inputs import ArrowDataset

# blocks: {block_key: [signature_id, ...]}
# current_seed_assignments: {signature_id: existing_profile_id}
with ArrowDataset.open("/path/to/arrow_dataset") as arrow_dataset:
    pred_clusters, pred_distance_matrices = clusterer.predict_from_arrow(
        blocks,
        arrow_dataset,
        batching_threshold=15_000,
        cluster_seeds_require=current_seed_assignments,
        total_ram_bytes=32 * 1024**3,
    )
```

`batching_threshold` is optional. When set, blocks larger than the threshold
use strict Arrow-native Rust subblocking; smaller blocks retain the full-block
route. The native graph fallback requires indexed SPECTER evidence even when
the pairwise model itself does not select an embedding feature. Initial-only
groups attach through the production bundle's promoted incremental linker.
Attachment and required-profile restoration stay within each original input
block. Active required components are kept together within that block; a
component larger than the threshold raises before subblocking. Blocks containing
only initials use bulk prediction. See the algorithm and standalone Python
reference in the [subblocking guide](subblocking.md).

Bulk prediction uses ordinary required seed groups on the Python route and on
the Arrow route with or without subblocking. Altered-profile markers do not change bulk seed
membership or attachment policy. The Arrow `altered_cluster_signatures` argument
is accepted but ignored by bulk prediction; the Python route likewise
ignores `ANDData.altered_cluster_signatures` in bulk mode. Required components
larger than the threshold within one original block raise even when marked altered.
Between bulk prediction and initial-only attachment, the
[reunite-only step](subblocking.md#reunite-only-step) reattaches no-middle
records that the size cap cut across subblocks (see
[memory and time](#reunite-only-memory-and-time)).

Promoted incremental inference requires an explicit seed mapping:

```python
# block_signatures: signature IDs in the request. Those absent from
# signature_to_seed_cluster ({signature_id: existing_profile_id}) are the queries.
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

The seed mapping holds the signatures of existing author profiles, for example
the clusters of an earlier run. For a runnable example on released data, the
[prediction tutorial](../scripts/tutorial_for_predicting_with_the_prod_model.py)
seeds profiles from one block's labeled signatures on papers before a chosen year
and links the block's other signatures; see the [README quick start](../README.md#quick-start).

### Profile exclusions

A disallow pair is a hard negative in average linkage. When the excluded paper
sits inside the body of work of the person who excluded it, that one pair blocks
the placed papers from merging with the rest of the person's work, and the
profile keeps only its placed papers.

For "this paper must not be on this claimed profile", pass profile exclusions
instead of disallow pairs on full-block prediction:

```python
clusters, _ = clusterer.predict_from_arrow(
    block_dict,
    arrow_dataset,
    cluster_seeds_require=read_cluster_seeds_arrow(seeds_path),
    profile_exclusions=read_profile_exclusions_arrow(exclusions_path),
    name_tuples=None,
)
```

Each row is `(anchor_signature_id, excluded_signature_id)`. The anchor is any one
signature the user placed on the profile. S2AND clusters the block, evicts each
excluded signature found in its anchor's cluster, and clusters again with that
signature barred from the profile. The excluded signature never ends in its
anchor's cluster, and within its subblock it can still join any other cluster.

- With `batching_threshold`, the anchor and the excluded signature can land in
  different subblocks, and attachment or reunite can then join them. A final pass
  moves such a signature to a cluster of its own.
- An excluded signature that is required together with other signatures leaves
  with that group.
- Rows that name a signature outside the request are ignored.
- A block that holds an anchor keeps a second copy of its distance matrix while it
  clusters. The RAM guard counts one, so leave headroom for one more matrix.

- The Arrow table is `profile_exclusions.arrow` with string columns
  `anchor_signature_id` and `excluded_signature_id`. It is optional.
- `Clusterer.predict` takes the same `profile_exclusions` argument.
- `convert_to_arrow.py service-json` reads a top-level `profile_exclusions` key:
  a list of `[anchor, excluded]` pairs, or an object from anchor to excluded IDs.
- A row that excludes a signature required together with its anchor raises `ValueError`.
- `cluster_seeds_disallow` keeps its meaning. Incremental prediction does not take
  profile exclusions; send its exclusions as disallow pairs, as before.
- Eviction honors exclusions. It does not remove another person's papers that
  the user did not exclude.

### Reunite-only memory and time

Each pass of the [reunite-only step](subblocking.md#reunite-only-step) attaches up
to `batching_threshold` marked records from one subblock to marked-only clusters
in the same dead-end bucket. Set `total_ram_bytes` with headroom above the
expected peak resident set size (RSS). Buckets need marked-only clusters in at
least two subblocks to run the step.

### Arrow validation in release and evaluation tools

Linker source assembly validates each original Arrow manifest and its table and
index hashes before rebinding its name-count index path. The release evaluator
uses the Arrow benchmark and held-out labels. See the
[reproduction guide](reproduce.md) for the evaluation command.

## Clustering corrections and close-pair bonus

Set these runtime-only `Clusterer` switches after loading a model. Each starts
at the package default.

| `Clusterer` switch | Default | Off value | Scope | Bundle export |
| --- | --- | --- | --- | --- |
| `orcid_name_waiver` | `(5, 0.1)` | `None` | Rust block and subblock constraint building | Requires the default |
| `name_conflict_repair_max_distance` | `0.2` | `None` | In-block repair after linkage | Requires the default |
| `orcid_join` | `True` | `False` | Final clusters across Arrow subblocks | Requires the default |
| `year_gap_split` | `True` | `False` | In-block split after repair | Requires the default |

Name-conflict repair and year-gap splitting share a one-byte-per-pair copy of
the distances. The memory guard accounts for that copy when a correction can
use it. Both steps skip ordinary prediction with precomputed `dists=`, including
the epsilon (EPS) search in `Clusterer.fit`. The dedicated
`predict_eps_grid` and `predict_eps_grid_from_rust_featurizer` methods apply both
corrections when eligible. Pass them matrices built from the same dataset or
featurizer, free of caller supervision. See the correction order in
[`s2and/corrections/block_corrections.py`](../s2and/corrections/block_corrections.py).

### ORCID name waiver

When Rust builds a block or subblock distance matrix, an ORCID's own mentions
can override a name conflict with a mention that lacks an ORCID. The outside
name must be compatible with at least two same-ORCID mentions whose names
conflict with the paired mention. The two setting values give the minimum
number of full-first-name mentions and the maximum share of those mentions that
may conflict with the outside name. The minimum must be at least 1 and the
share must be in `[0, 1]`.

Eligible pairs use their model distance, with seed constraints taking
precedence. Pairwise constraint checks in incremental linking use the ordinary
name rules. `suppress_orcid` also disables the waiver. See
[the Rust featurizer](../s2and_rust/src/rust_featurizer.rs) for the full rule.

### Name-conflict repair

Repair can join two clusters in a block of at least 11 mentions when one or two
odd mentions explain the name vetoes. Both sides must retain enough mentions
and full first-name evidence, and the merge must respect ORCIDs, seed
constraints, and profile exclusions. It uses an upper bound on the linkage
height and retains the odd mentions in the merged cluster. When the plain
average bound exceeds the repair limit, the bonus requires at least 10% of
pairs to be certainly close, a plain average bound of at most 0.40, and at
least 10 remaining mentions on each side.

Repair requires `FastCluster` and a one-dimensional condensed matrix with a
floating-point or integer dtype. It converts strided views and other supported
dtypes to contiguous `float64` as needed. Repair needs an exact
`LARGE_DISTANCE` pair and skips matrices with NaN or any other distance above
one. It can run during incremental altered-profile splitting and residual
clustering. The removed `name_conflict_repair_*` settings raise `ValueError`.
See [`s2and/corrections/name_conflict_repair.py`](../s2and/corrections/name_conflict_repair.py) for the
repair rounds and the joined-initials rule.

### ORCID joins

Average-linkage Arrow subblocking can join final clusters supported by the same
dominant ORCID in separate subblocks of one original block. The join requires
sufficient ORCID coverage and compatible name evidence and respects seed
disallows, profile exclusions, and `dont_merge_cluster_seeds`. Joined clusters
can exceed the subblock size limit. `suppress_orcid` also disables the join.
See [`s2and/corrections/orcid_join.py`](../s2and/corrections/orcid_join.py) for the evidence thresholds.

### Year-gap split

After linkage and any in-block repair, the split separates dated mentions
across gaps longer than 50 years. A certainly close pair, a shared ORCID or required seed group,
or a shared first word of a full first name in a small surname block can bridge
the gap. Small-block eligibility uses the original surname block and the
name-count index where available. The years 1601 and 1800 count as undated.

The split requires a cluster model with a close-pair bonus and runs
independently of name-conflict repair. During altered-profile pre-splitting,
`incremental_dont_use_cluster_seeds=True` permits temporary cuts within a seed
group. Incremental linking restores the original profile in its result. See
[`s2and/corrections/year_gap_split.py`](../s2and/corrections/year_gap_split.py) for undated mentions and bridges.

### Close-pair bonus

The bonus gives strong pairwise evidence more weight in average linkage.
The merge height is the average model distance minus `close_pair_bonus` times
the share of pairs below `close_pair_distance`, floored at zero. Impossible
pairs still keep clusters apart.

Calibration selects EPS together with the bonus. The bundle records both bonus
settings under `cluster_model` in `clusterer.json`. Bundles without these fields
use plain average linkage.

The bonus applies to clustering with the configured model, including full
blocks, subblocks, and the clustering steps of incremental prediction.
[Name-conflict repair](#name-conflict-repair) uses an upper bound on the lowered
height when the model sets a bonus, subject to the repair's bonus bounds. The
incremental linker scores attachments separately.

## Caller supervision and dataset constraints

`partial_supervision` maps unordered signature pairs to finite distance
overrides. Either pair orientation is accepted. Equal entries for `(A, B)`
and `(B, A)` are deduplicated; differing values raise `ValueError` identifying
the pair and both distances. NaN and infinity also raise. Validation runs
before candidate selection, subblocking, or scoring, without mutating the
caller's mapping. Finite soft distances keep their normal meaning and range; this does not turn zero distances into globally enforced transitive
must-links. A soft distance must stay at least `1e-6` from 0 and from `10000`; values closer
to either end raise `ValueError`, because scoring would read them as a forced link or a hard
veto. Bulk prediction also rejects a distance below `10000` between members of two different
required seed groups, because seed restoration would then merge both whole groups.

Arrow bulk prediction accepts integer signature IDs by normalizing block members
and supervision keys to strings. Its output members are strings whether or not
subblocking runs. Conflicting integer/string aliases in caller supervision raise
the same error as conflicting reversed pairs.

In subblocked bulk prediction, exact zero overrides retain this distance meaning
during initial-only attachment: they affect retrieved pair scores and residual
clustering without requiring a candidate outside the retrieval window or merging
separate bulk groups. Standalone Arrow incremental prediction retains its stricter
zero-distance require semantics and rejects requirements between residual queries,
outside the retrieval window, or targeting multiple seed components.

Caller distances override dataset constraints for the same pair, consistently
during scoring and seed restoration, and remain active when `use_default_constraints_as_supervision=False`. That setting
disables dataset-derived distance constraints, including dataset disallows,
on both Python and Arrow. Existing profile assignments still identify seed
targets for incremental prediction. Arrow resolves effective disallows before
retrieval so an overridden dataset negative cannot remove a candidate before
the caller's distance is applied. Other retrieval limits still apply.

Incremental prediction preserves each original seed group. If an effective
explicit disallow connects two members of that group, `predict_incremental_from_arrow`
raises `ValueError` with `incremental_seed_disallow_conflict`, the group ID, and
the conflicting signature pair before splitting or scoring. This includes
seed members supplied separately from the query list and requests with no new
papers. Caller distances equal to `LARGE_DISTANCE` are hard disallows; other
finite distances and inferred name constraints do not invalidate existing
membership. Disabled dataset negatives and dataset negatives replaced by
non-hard caller distances do not conflict. Correct contradictory membership or
exclusions in the input; incremental prediction does not choose which existing
paper to remove.
Exclusions involving new papers prevent attachments to the original
profile, including attachments through temporary split components.
Subblocked bulk prediction creates temporary seed groups from its own bulk
output. Average linkage does not place a pair at exactly `LARGE_DISTANCE`
together; other linkage methods give no such guarantee. Follow-on attachment
preserves the computed memberships and omits only their internal hard negatives
from its local attachment request. Exclusions involving fresh signatures or different
groups remain active, and the original input supervision is unchanged. This
exception does not apply to caller-supplied incremental seed groups.

The lower-level `predict_from_rust_featurizer` bridge retains its explicit
seed-override behavior: seed arguments supplied directly to that method remain
active even when default constraints are disabled.

Caller supervision cannot be combined with precomputed matrices, and Arrow
rejects supplied seed constraints with those matrices even when default
constraints are off.

Explicit hard cannot-link supervision also constrains post-clustering seed
merges. Indirect merges cannot reconnect a forbidden pair.

## Incremental decision semantics

Subblocked initial-only attachment adds two hard name checks. Standalone
incremental prediction does not use them. A query does not link to a cluster
when it violates the hard name constraint with a member of that cluster; the
decision becomes an abstain. Require and ORCID evidence keep their links.
Members of caller-required seed groups are not checked, so a curated profile
with conflicting names does not block each query. Then the links of the pass go
in order of forced evidence, score and signature ID. A link that conflicts with
a link accepted earlier into the same cluster is rejected. Abstained and
rejected queries go to residual clustering. The passes of the
[reunite-only step](subblocking.md#reunite-only-step) use the same checks.

Promoted query decisions are request-global and deterministic across input
order, query batches, and `batching_threshold`. A global conflict is resolved
by rebuilding and rescoring the lower-priority query from its complete
single-query candidate plan with the winning component excluded.

RAM limits are refreshed before planning each query window, again after
planning, and before each scoring batch. If a refresh before planning shrinks
the window, only its safe prefix is planned and the remaining queries move to
later windows. If a refresh inside a planned window shrinks the scoring batch,
the smaller batches are scored from contiguous slices of that window's existing
plan rather than replanned. Planning uses the loaded model’s actual final,
pairwise, and aggregate feature widths.

Temporary altered-profile splitting belongs to incremental linking: score new
papers against the natural components, then restore the original claimed
profile. Explicit exclusions still apply to the restored identity, so a query
cannot bypass an exclusion by linking through another component.

Promoted incremental prediction checks disallows against restored profile
identities. A query forbidden from one member of an altered profile cannot
attach through a sibling split. These exclusions apply before retrieval top-k.
Mutually disallowed queries likewise cannot attach to different splits that
restore to the same profile; the lower-priority query is rescored through the
existing exclusion mechanism. The additional bookkeeping uses explicit
constraint edges and split mappings rather than enumerating paper pairs.

## Complete model bundles

A complete bundle contains both pairwise boosters, prediction fixtures, calibrated EPS, the incremental linker, and its target.
The manifest binds the model format, feature contract, ordered features, release identity, lifecycle, and checksums.
The runtime layout is:

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
    pairwise_training_config.json
    pairwise_training_summary.json
```

The root manifest contains `kind: "s2and_model"`, `format_version: 1`, and `feature_contract_version: 1`.
It also contains `feature_names`, `release_version`, `eps_calibration`, and `sha256`.
`feature_names` declares the exact main and nameless column order.
The loader checks supported contract versions and feature names.
The bundle does not record the producer package version, so the loader does not compare it.
The installed Python and Rust packages still require matched versions for their internal interfaces.

The exact checksum inventory determines whether the bundle is pairwise-only or complete.
`bundle_kind` is not serialized.
Pairwise-only stages can have pending or calibrated EPS; complete models require calibrated EPS.
The pairwise training config and summary must both exist or both be absent.
The loader hashes each declared file and checks both mandatory prediction fixtures.
It rejects unversioned model stages instead of a silent legacy fallback.

Pairwise production training records `name_tuples_data_sha256`,
`name_counts_manifest_sha256`, and `orcid_prefix_counts_data_sha256` in
`feature_contract`. Bundle export does not synthesize missing behavior hashes.
Export and load check the name-tuple hash against the packaged artifact by
default. The ORCID prefix-count hash must match an
[installed count set](#name-count-and-reuse-boundaries). The exact name-count
manifest and complete ordered feature contract also bind the model and linker.

The linker metadata records `kind: "s2and_incremental_linker"`, format and feature-contract versions, ordered feature names, retrieval top-k, gate configuration, and binding digests.
Its checksums bind the booster, pairwise bundle, and complete target JSON.
The complete model retains the target at `reproducibility/incremental_linker_training_target.json`.
Finalization and load reject a mismatch.

`load_production_model(path)` rejects legacy pickles, incomplete directories,
and `pairwise_only` manifests. A pairwise-only bundle is an internal training
stage and has a separate internal loader; it is never a public inference model.
The loader validates the complete bundle, loads the linker once, and retains
that immutable native artifact on the clusterer. Deep copies share it; pickle
round trips revalidate and reload it from the recorded bundle path.

`clusterer.json` is inference-only. It contains the pairwise feature contracts,
runtime controls, and only the clustering algorithm fields consumed by
prediction (`eps` and `linkage`, plus `close_pair_bonus` and `close_pair_distance`
when the bundle uses the [close-pair bonus](#close-pair-bonus)). Stage 3 records placeholder EPS `0.5`; the
manifest marks that value as pending until validation-only calibration writes
a calibrated sibling bundle.

### Staged publication

Pairwise training writes an immutable pairwise-only stage with EPS pending.
Validation-only calibration writes a sibling stage, changing only
`clusterer.json` and `manifest.json`. `train_linker_and_finalize.py` uses that
calibrated stage to rematerialize pairwise-derived features and fit the linker
once.

Finalization assembles and validates a complete sibling staging directory
containing the verified pairwise files, the new linker, and
`reproducibility/incremental_linker_training_target.json`. It publishes the
directory with one rename to a new final path. A failed finalization leaves
the pairwise source intact and keeps partial files out of the final path.
The training command reloads the published bundle and evaluates the serialized
artifact.

See the [production command reference](../scripts/release/README.md) for
arguments and the [release process](release.md) for stage order and gates.

## Name-count and reuse boundaries

The current runtime uses one binary `NameCountsIndex` in Python and Rust. Classic
`ANDData` defaults to the canonical configured index and also accepts an
alternate path, a shared open index handle, or explicit `None` when
count features are intentionally absent.
`NameCountsIndex.open` checks the manifest, file sizes, file headers, and section
bounds; it does not hash the index files. `NameCountsIndex.open(path, verify=True)`
and `verify()` also check every file digest and every record. The release-root
validator, `convert_to_arrow.py validate`, dataset packaging, linker-bundle assembly,
the release bundle writer, and the count producer run that check before publication.
Run it again when you deploy an index to a new location, for example with
`NameCountsIndex.open("s2and/data/name_counts_index", verify=True)`. Arrow routes retain the manifest-bound
index inside `ArrowDataset`. Models bind directly to the native index's single
`manifest_sha256`. The manifest
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

The wheel ships two ORCID prefix-count sets. Each has
`first_k_letter_counts_from_orcid.json` and an adjacent
`first_k_letter_counts_from_orcid.manifest.json` with its `name_tuples_sha256`
dependency.

| Count set | Directory |
| --- | --- |
| Used by `1.3.x` models | `s2and/data/` |
| Used by `1.4` models | `s2and/data/orcid_prefix_counts_v1_4/` |

Arrow subblocking selects the set by `orcid_prefix_counts_data_sha256` in the
model's feature contract. The packaged count loader rejects unknown hashes
and data that fails the hash check. New training contracts use the `1.4` set and
verify its tuple hash against the canonical name-tuple artifact. A clusterer
without a contract hash uses the `1.4` set for subblocking. See
[the packaged count loader](../s2and/orcid_prefix_counts.py) and
[`Clusterer._subblocking_prefix_counts`](../s2and/model/bulk.py).

`last_first_initial_count_min` uses
`<canonical last> <canonical first[0]>` when both fields exist and a null key
otherwise. Production feature contracts bind the exact name-count manifest,
the ORCID and name-tuple data hashes, and the linker.
Feature-contract versions and ordered names define compatible feature behavior.
Unversioned artifacts require an explicit rebuild or export.

Direct Arrow/Rust prediction reuses validated immutable native state within the
process, and the bounded altered-profile presplit memo is a private in-process
optimization. Nothing is cached on disk between processes.

## Verification

The focused code gate is:

```powershell
uv run pytest -q tests/test_production_model.py tests/test_arrow_production_boundary.py tests/test_arrow_training_ingestion.py tests/test_cluster_incremental.py
uv run pytest -q tests/test_name_tuple_artifact.py tests/test_generate_orcid_name_prefix_counts.py
uv run pytest -q tests/test_train_pairwise_script.py tests/test_promoted_linker_training_cli.py
uv run ruff check s2and scripts/common scripts/linker scripts/pairwise scripts/release tests/test_production_model.py tests/test_arrow_production_boundary.py tests/test_arrow_training_ingestion.py tests/test_cluster_incremental.py
git diff --check
```

These tests validate code and synthetic bundles only. A release additionally
needs a clean wheel/sdist check, exact-version Rust wheel install, a real
canonical bundle, and the approved quality/runtime/RSS gates in
[release.md](release.md).

## Offline attachment replay

The production route of the EPS sweep holds an attachment reuse context for each open, ORCID-free
Arrow dataset. It retains subblocking, block featurizers and query name metadata across EPS
values. Offline replays can pass an AttachReuse context to the same private Arrow request entry
point. Use the context manager and close it before the Arrow dataset. It logs retained signatures
and hits, and computes setup normally when its signature budget is exhausted.

The residual cache retains condensed model distances for initial-only attachment groups when
ORCID constraints are suppressed and native fused scoring is available. Each pass applies its
own constraints and overrides, then runs linkage and block corrections at its EPS. The context
defaults to a 1 GiB residual cache and falls back to ordinary scoring when the matrix exceeds
the cache or runtime memory budget.

For fixed-EPS ranker and gate replays, pass that EPS as the context's snapshot_eps and reuse the
context across variants. The first attachment pass in each block can reuse its candidate plans,
pairwise aggregates and feature matrices. The ranker and gate run for each variant, and later
passes plan from that variant's links. The 512 MiB snapshot budget is split evenly between plans
and scoring inputs. Global query-disallow rescoring uses fresh plans and features.

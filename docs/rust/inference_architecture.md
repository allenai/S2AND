# Rust Inference Architecture

Status date: 2026-05-25

This is the current map for Rust-backed inference. It replaces the older
raw-candidate-plan design log and the promoted-incremental design note.

The target is not a Rust clone of all `ANDData`. `ANDData` remains the Python
reference object for training, legacy loading, pair sampling, compatibility
tests, and explicit Python/legacy inference. The production Rust target is a
narrow set of typed inference inputs read directly by Rust.

## Before / After

| Area | Before | After |
|---|---|---|
| Inference boundary | Most production paths started from full `ANDData` and crossed into Rust after Python had built namedtuples, blocks, and feature objects. | Direct Arrow paths feed signatures, papers, paper authors, SPECTER, and seed rows into Rust without full-block `ANDData`. |
| Full-block prediction | `Clusterer.predict(...)` built or reused `ANDData`, then used Rust mainly for pairwise feature and clustering kernels. | `Clusterer.predict_from_arrow_paths(...)` and Arrow-routed `predict(...)` build the filtered Rust featurizer from Arrow IPC and reuse Rust blockwise feature, constraint, and clustering logic. Rust mode requires complete Arrow artifacts and fails fast when they are missing. |
| Raw incremental requests | The raw path first materialized Python signal objects or mini compatibility objects before scoring. | `RawBlockQueryCandidatePlanner` performs retrieval and row-signal construction in Rust from Arrow IPC, then the raw Arrow scoring route runs without full-block `ANDData`. |
| Candidate scope | Giant blocks could lead to broad query-vs-seed-signature work before the linker saw compact candidates. | Rust retrieval builds a bounded query-to-component candidate plan before pair scoring. |
| Pairwise feature build | Python object materialization and `ANDData` construction dominated some profiles before Rust pairwise work began. | `RustFeaturizer.from_arrow_paths(...)` (Rust method exposed via PyO3; Python wrapper is `build_rust_featurizer_from_arrow_paths` in `s2and/feature_port.py`) constructs only the selected scoring rows from Arrow and global sidecars. Production filtered reads require batch indexes. |
| Row signals | Several promoted link/abstain row signals were assembled in Python after Rust retrieval. | Rust emits the promoted native row signals needed by the raw Arrow planner/scoring route, including name-count rarity and paper-author overlap signals. |
| Name counts | Docs and tests previously preferred embedding four per-signature count columns in `signatures.arrow`; Rust could skip global artifacts if all selected rows had embedded counts. | Embedded Arrow name-count columns are not a supported direction. Runtime bundles should provide `s2and/data/name_counts_index/`; `name_counts.arrow` is only for generation, inspection, and parity debugging. |
| Name alias data | Some paths could pass per-dataset Arrow `name_pairs` / `name_tuples` overrides. | Runtime aliases now come from the explicit `name_tuples` argument; production path bundles must not carry alias override paths. |
| SPECTER | Pickle remained common in Python paths; Rust paths handled some payloads through Python objects. | Direct Arrow uses fixed-size-list `float32` embedding tables. Safetensors is still only a future benchmark if SPECTER read time becomes material. |
| Cluster seeds | Seed semantics were mostly Python maps on the incremental path. | Seeded/incremental Arrow requires a seed source: either `cluster_seeds.arrow` or a normalized seed mapping that production materializes into request-local Arrow. `cluster_seed_disallows.arrow` is optional unless disallow constraints are declared. Unseeded full predict can omit both. |
| Reference features | Legacy feature slots and training paths still supported citation-derived reference features. | Removed entirely: the featurizer no longer defines the `reference_features` group and the feature vector carries no reference columns. |
| Data ingestion | JSON/pickle plus `ANDData` preprocessing was the default ingestion shape. | Arrow IPC is the preferred table-shaped ingestion format when the hot path stays Rust/columnar. JSON remains compatibility/test input. |
| Verification | Performance and parity evidence lived across several design logs. | Current gates should point to this architecture doc, `arrow_dataset_spec.md`, `artifact_formats.md`, `runtime.md`, and `baselines.md`. |

## Name-Count Decision

Use the sorted exact-verified `s2and/data/name_counts_index/` sidecar as the
Rust hot-path artifact. It is a better fit than SQLite for the current workload
because Rust does exact point lookups against four static dictionaries; the
binary index is memory-map friendly, has exact string verification after hash
lookup, and avoids shipping a query engine or managing SQLite runtime state.

Reconsider SQLite only if the requirement changes to ad hoc querying, partial
updates, cross-process transactional writes, or richer offline inspection.

## One Rust Production Route Per Job

Each production job should have exactly one Rust entrypoint. Other Rust APIs may
exist for parity, training/eval compatibility, or targeted diagnostics, but they
are not alternate production paths.

| Job | Production route | Not production |
|---|---|---|
| Full-block prediction | `Clusterer.predict_from_arrow_paths(...)` or Arrow-routed `Clusterer.predict(...)` -> `feature_port.build_rust_featurizer_from_arrow_paths(...)` -> `RustFeaturizer.from_arrow_paths(...)`. | JSON loaders and raw Python object scoring. |
| Raw incremental candidate planning | `RawBlockQueryCandidatePlanner.from_query_signatures(paths_with_query_signatures_and_batch_indexes, ...)` -> `.plan_query_signatures()` or subset `.plan(...)` calls. | Unindexed filtered Arrow scans, Python mini object materialization, direct retriever wiring from callers. |
| Pairwise feature and prediction inputs | `LinkerCandidateBatch` index arrays -> indexed Rust pairwise APIs. | String-pair feature APIs or ad hoc per-pair calls. |
| Constraints | `get_constraints_matrix_indexed`, `get_constraints_block_upper_triangle_indexed`, or linker label-array APIs. | Single-pair Rust constraints. |
| Arrow graph subblocking | `make_subblocks_with_telemetry_arrow_native_graph(...)` with `signatures_batch_index`. | Full scans or Python callback-based Rust subblocking. |
| Training and materialization | Python owns cleaning, sampling, LightGBM training, calibration, and metrics. `raw_arrow_labeled_candidate_plan(...)` is allowed only as a training/materialization helper. | Treating training helpers as online production inference APIs. |

CI should guard the removed production escape hatches: the unindexed
filtered-read bypass and the single-pair Rust constraint API must not reappear.

## Compatibility And Python-Heavy Paths

Production Rust inference uses `Clusterer.predict_from_arrow_paths(...)`,
Arrow-routed `Clusterer.predict(...)`, promoted
`Clusterer.predict_incremental(...)` with complete base Arrow artifacts, or
`Clusterer.predict_incremental_from_arrow_paths(...)` when callers have Arrow
artifacts and request seed state but do not have an `ANDData`-shaped dataset
object. The paths below remain useful, but they are compatibility, training,
parity, or test surfaces rather than production inference APIs.

Removed bridge surfaces: `RustFeaturizer.from_feature_block(...)` and raw
payload scoring wrappers are no longer Rust inference APIs. They built or
traversed Python `FeatureBlock` objects before scoring; production raw requests
now use typed Arrow query-signature request sidecars for raw planner entry.

| Path | Current Python dependency | Production status |
|---|---|---|
| `Clusterer.predict(...)` without Arrow paths | Explicit Python/legacy routes can still build normal `ANDData` and use Python block orchestration. Auto/default routes also fall back to Python for non-Arrow datasets. | Explicit Rust production raises `MissingArrowArtifactError`. Provide complete Arrow artifacts or select `backend="python"` for compatibility/reference execution. |
| `Clusterer.predict_incremental(...)` without base Arrow paths or seed source | Explicit Python/legacy routes can still use Python incremental helpers and `ANDData` seed state. | Rust production now raises `MissingArrowArtifactError`. Provide `signatures`, `papers`, `paper_authors`, required embedding/name-count artifacts, and a seed source via `cluster_seeds` or `dataset.cluster_seeds_require`. |
| JSON/`ANDData` featurization | Uses Python `ANDData` and Python feature code. | Compatibility/reference execution only; Arrow IPC is the Rust table-shaped target. |
| Training and release replay | Python owns data cleaning, feature table materialization, LightGBM training, calibration, and metrics. | Not a no-`ANDData` inference target; keep Python unless runtime profiling shows a training bottleneck worth porting. |

Incremental seed and altered-profile behavior is part of the production
runtime contract, but the implementation details live in the operational docs:
[../production_inference.md](../production_inference.md) owns caller-visible
telemetry and routing semantics, while [arrow_dataset_spec.md](arrow_dataset_spec.md)
owns the Arrow table contracts. Active cleanup work is tracked in
[../work_plan.md](../work_plan.md).

## Current Verification Focus

- Tiny Arrow fixture tests for schema and row-signal behavior.
- Exact parity gates for direct Arrow full predict: feature matrix, constraints,
  distances, and clusters.
- Raw Arrow incremental checks for candidate rows, pair rows, row signals,
  probabilities, and final link/abstain decisions.
- Stage telemetry that separates Arrow read, name-count index load, retrieval,
  featurizer construction, pair scoring, and raw row-signal construction. Final
  logistic-gate decisions are covered by result telemetry but are not currently
  timed as a separate stage.

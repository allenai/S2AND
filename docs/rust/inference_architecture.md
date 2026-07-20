# Rust Inference Architecture

Status date: 2026-07-10

This is the current map for Rust-backed inference. It replaces the older
raw-candidate-plan design log and the promoted-incremental design note.

The target is not a Rust clone of all `ANDData`. Classic `ANDData` construction,
pair sampling, and prediction remain Python. The Rust target is a narrow set of
typed Arrow inputs read directly by Rust.

## Before / After

| Area | Before | After |
|---|---|---|
| Inference boundary | Most production paths started from full `ANDData` and crossed into Rust after Python had built namedtuples, blocks, and feature objects. | Direct Arrow paths feed signatures, papers, paper authors, SPECTER, and seed rows into Rust without full-block `ANDData`. |
| Full-block prediction | `Clusterer.predict(...)` built or reused `ANDData`, then used Rust mainly for pairwise feature and clustering kernels. | `Clusterer.predict_from_arrow_paths(...)` explicitly builds the filtered Rust featurizer from Arrow IPC and reuses Rust blockwise feature, constraint, and clustering logic. Missing required Arrow artifacts fail immediately. |
| Raw incremental requests | The raw path first materialized Python signal objects or mini compatibility objects before scoring. | `RawBlockQueryCandidatePlanner` performs retrieval and row-signal construction in Rust from Arrow IPC, then the raw Arrow scoring route runs without full-block `ANDData`. |
| Candidate scope | Giant blocks could lead to broad query-vs-seed-signature work before the linker saw compact candidates. | Rust retrieval builds a bounded query-to-component candidate plan before pair scoring. |
| Pairwise feature build | Python object materialization and `ANDData` construction dominated some profiles before Rust pairwise work began. | `RustFeaturizer.from_arrow_paths(...)` (Rust method exposed via PyO3; Python wrapper is `build_rust_featurizer_from_arrow_paths` in `s2and/feature_port.py`) constructs only the selected scoring rows from Arrow and global sidecars. Production filtered reads require batch indexes. |
| Row signals | Several promoted link/abstain row signals were assembled in Python after Rust retrieval. | Rust emits the promoted native row signals needed by the raw Arrow planner/scoring route, including name-count rarity and paper-author overlap signals. |
| Name counts | Docs and tests previously preferred embedding four per-signature count columns in `signatures.arrow`; Rust could skip global artifacts if all selected rows had embedded counts. | Runtime bundles provide the canonical `s2and/data/name_counts_index/` binary sidecar. Embedded Arrow name-count columns and standalone Arrow name-count artifacts are unsupported. |
| Name alias data | Some paths could pass per-dataset Arrow `name_pairs` / `name_tuples` overrides. | Runtime aliases now come from the explicit `name_tuples` argument; production path bundles must not carry alias override paths. |
| SPECTER | Pickle remained common in Python paths; Rust paths handled some payloads through Python objects. | Direct Arrow uses fixed-size-list `float32` embedding tables. Safetensors is still only a future benchmark if SPECTER read time becomes material. |
| Cluster seeds | Seed semantics were mostly Python maps on the incremental path. | Seeded/incremental Arrow requires a seed source: either `cluster_seeds.arrow` or a normalized seed mapping that production materializes into request-local Arrow. `cluster_seed_disallows.arrow` is optional unless disallow constraints are declared. Unseeded full predict can omit both. |
| Reference features | Legacy feature slots and training paths still supported citation-derived reference features. | Removed entirely: the featurizer no longer defines the `reference_features` group and the feature vector carries no reference columns. |
| Data ingestion | JSON/pickle plus `ANDData` preprocessing was the default ingestion shape. | Explicit Rust routes take Arrow IPC. JSON and ordinary `ANDData` use the Python route. |
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
| Full-block prediction | `Clusterer.predict_from_arrow_paths(...)` -> `feature_port.build_rust_featurizer_from_arrow_paths(...)` -> `RustFeaturizer.from_arrow_paths(...)`. | `Clusterer.predict(...)`, JSON loaders, and raw Python object scoring. |
| Raw incremental candidate planning | Explicit requests use `RawBlockQueryCandidatePlanner.from_query_signatures(paths_with_query_signatures_and_batch_indexes, ...)`; automatically selected promoted query windows use `from_auto_queries(paths_with_batch_indexes, ...)`. Both reuse `.plan(...)` calls. | Unindexed filtered Arrow scans, Python mini object materialization, temporary empty request sidecars, direct retriever wiring from callers. |
| Pairwise feature and prediction inputs | `LinkerCandidateBatch` index arrays -> indexed Rust pairwise APIs. | String-pair feature APIs or ad hoc per-pair calls. |
| Constraints | `get_constraints_matrix_indexed`, `get_constraints_block_upper_triangle_indexed`, or linker label-array APIs. | Single-pair Rust constraints. |
| Arrow graph subblocking | `make_subblocks_with_telemetry_arrow_native_graph(...)` with `signatures_batch_index`. | Full scans or Python callback-based Rust subblocking. |
| Training and materialization | Python owns cleaning, sampling, LightGBM training, calibration, and metrics. The fixed `build_training_anddata_from_arrow(...)` constructor supplies Rust featurization from one immutable `dataset.arrow_paths` mapping. `raw_arrow_labeled_candidate_plan(...)` is a training/materialization helper. | Treating training helpers as online production inference APIs. |

CI should guard the removed production escape hatches: the unindexed
filtered-read bypass and the single-pair Rust constraint API must not reappear.

## Python and Rust API Boundaries

Production Rust inference uses `Clusterer.predict_from_arrow_paths(...)` or
`Clusterer.predict_incremental_from_arrow_paths(...)`. Classic
`Clusterer.predict(...)` and `Clusterer.predict_incremental(...)` always use
Python with `ANDData`; they do not inspect Arrow attributes or switch backend.

Removed bridge surfaces: `RustFeaturizer.from_feature_block(...)` and raw
payload scoring wrappers are no longer Rust inference APIs. They built or
traversed Python `FeatureBlock` objects before scoring; production raw requests
now use typed Arrow query-signature request sidecars for raw planner entry.

| Path | Current Python dependency | Production status |
|---|---|---|
| `Clusterer.predict(...)` | Uses `ANDData`, Python feature code, and Python block orchestration. | Explicit Python full-block API. Use `predict_from_arrow_paths(...)` for Rust. |
| `Clusterer.predict_incremental(...)` | Uses Python incremental helpers and `ANDData` seed state. | Explicit Python incremental API. Use `predict_incremental_from_arrow_paths(...)` for Rust. |
| JSON/`ANDData` featurization | Uses Python `ANDData` and Python feature code. | Python input route; it does not silently become Rust. |
| Arrow training | Python owns cleaning, sampling, LightGBM training, calibration, and metrics; the fixed Arrow constructor supplies the Rust featurizer. | Explicit Rust-training route with validated immutable Arrow paths. |

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

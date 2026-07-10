# Rust Public Surface Inventory

Status date: 2026-07-10

This inventory records the current Python-visible `s2and_rust` surface before
module splitting or API deletion. It is intentionally about ownership and
cleanup risk, not a user-facing API promise.

Python callers target exactly `s2and-rust==0.60.0`. Maintained calls are
direct: the Python runtime checks the extension version once when Rust is
requested and does not probe individual methods or constants.

## Module Exports

| Export | Owner / caller | Status |
|---|---|---|
| `RustFeaturizer` | `s2and/feature_port.py`, `s2and/rust_calls.py`, Arrow training/prediction, parity tests | Core class. Maintained Python Rust routes enter through `from_arrow_paths`. |
| `RustHybridCentroidRetriever` | `s2and/incremental_linking/retrieval.py`, raw Arrow planners, training query-support code | Core retrieval class. Production runtime should prefer `top_k_hybrid_centroid_pair_plan(...)`. |
| `RawBlockQueryCandidatePlanner` | `s2and/incremental_linking/production.py`, `s2and/incremental_linking/runtime.py` | Canonical production raw Arrow planner. |
| `raw_arrow_labeled_candidate_plan(...)` | `scripts/production/model/linker_train_calibrate_eval.py` | Training/materialization surface, not request-time inference. |
| `promoted_linker_non_pairwise_features(...)` | `s2and/incremental_linking/row_features.py` | Production promoted-linker row feature builder. |
| `make_subblocks_with_telemetry_arrow_native_graph(...)` | `s2and/subblocking.py` | Arrow-native graph subblocking helper used by large-block prediction. |
| `get_build_info(...)` | `scripts/_rust_suite/common.py`, Rust-suite tests | Build diagnostics for benchmark and verification reports; not a runtime gate. |
| `RustLightGBMBooster` | `s2and/production_model.py` (`NativeLightGBMBinaryClassifier`), `s2and/incremental_linking/artifact.py` (`IncrementalLinkingArtifact`), parity tests | Pure-Rust `.lgb` text-model scorer for binary numerical-split boosters; the production scoring path for pairwise and linker models. Raw scores are bit-exact vs Python `lightgbm` (`tests/test_rust_lightgbm_booster_parity.py`); rejects categorical/linear/multiclass models at load. Python `lgb.Booster` remains only as the lazily-loaded `booster_` surface for bundle writing and SHAP. |

## Module Constants

| Export | Owner / caller | Status |
|---|---|---|
| `RETRIEVAL_FEATURE_ORDER` | `s2and/incremental_linking_training/retrieval_policy.py`, retrieval parity tests | Retrieval feature ordering contract mirrored into Python training/reference policy code. |
| `DEFAULT_HYBRID_CENTROID_POLICY_NAME`, `DEFAULT_HYBRID_CENTROID_WEIGHTS`, `DEFAULT_INITIAL_ONLY_HYBRID_CENTROID_WEIGHTS`, `DEFAULT_HYBRID_EXEMPLAR_4_WEIGHTS` | `s2and/incremental_linking_training/retrieval_policy.py`, promoted-linker training and retrieval tests | Frozen retrieval policy constants; update Python training/reference defaults and tests together with Rust. |
| `RETRIEVAL_MIDDLE_INITIAL_CONFLICT_SCORE`, `RETRIEVAL_YEAR_SCORE_DECAY_YEARS`, `RETRIEVAL_YEAR_SCORE_RANGE_GAP`, `RETRIEVAL_YEAR_SCORE_RANGE_PENALTY`, `RETRIEVAL_HARD_FILTER_MAX_YEAR_GAP` | `s2and/incremental_linking_training/query_support.py`, retrieval parity tests | Training/query-support scoring constants. |

## `RustFeaturizer`

| Method | Owner / caller | Status |
|---|---|---|
| `from_arrow_paths(...)` | `feature_port.build_rust_featurizer_from_arrow_paths(...)`; full predict, subblocked predict, raw Arrow scoring | Production Arrow constructor. The Python production wrapper requires batch indexes for filtered reads. |
| `update_cluster_seeds(...)` and `update_signature_name_counts(...)` | cache/seed update helpers in `feature_port.py` and tests | Compatibility/training lifecycle helpers. |
| `signature_ids(...)` | pairwise matrix wrappers, promoted incremental runtime, parity scripts | Shared index-order contract; keep. |
| `signature_rule_metadata(...)`, `signature_name_counts_present(...)`, `cluster_seeds_require(...)` | `predict_from_rust_featurizer(...)`, parity tests, and state restoration checks | Required metadata for direct Rust-featurizer prediction and parity. |
| `get_constraints_matrix_indexed(...)` | `model.py`, `rust_calls.py`, parity tests | Maintained indexed constraint API. |
| `get_constraints_block_upper_triangle_indexed(...)` | `model.py`, Arrow parity script | Maintained blockwise constraint API. |
| `linker_pair_index_arrays_constraint_labels(...)` | promoted linker training/materialization and runtime tests | Maintained promoted incremental constraint-label API. |
| `linker_pair_distance_accumulators(...)` | promoted incremental runtime and tests | Maintained promoted incremental aggregate API. |
| `featurize_pairs_matrix_indexed(...)` | `s2and/featurizer.py` | Canonical pairwise matrix API for Python Rust batching. |
| `linker_pair_index_arrays_and_aggregate_stats(...)` | `s2and/incremental_linking/linker_pairwise.py` | Canonical promoted linker pair-feature plus aggregate API. |
| `linker_pair_index_arrays_and_aggregate_stats(..., emit_matrix=False)` | `s2and/incremental_linking/linker_pairwise.py` | Canonical aggregate-only mode; preserves the no-matrix fast path without a second PyO3 method. |
| `featurize_block_upper_triangle_matrix_indexed(...)` | blockwise full predict | Maintained blockwise feature API. |

## Retrieval Classes

| Method | Owner / caller | Status |
|---|---|---|
| `RustHybridCentroidRetriever.__new__(...)` | raw Arrow planners, training query support, tests | Maintained constructor. |
| `top_k_hybrid_centroid_pair_plan(...)` | `s2and/incremental_linking/retrieval.py`, raw Arrow planners | Canonical runtime retrieval output. |
| `top_k_experimental_weighted_hybrid_centroid_subset(...)` | `s2and/incremental_linking_training/query_support.py`, tests | Training/query-support scoring surface. |
| `RawBlockQueryCandidatePlanner.from_query_signatures(...)`, `from_auto_queries(...)`, `plan_query_signatures(...)`, `build_telemetry(...)`, `plan(...)` | `s2and/incremental_linking/production.py`, `s2and/incremental_linking/runtime.py`; tests | Canonical reusable production raw Arrow planner. Explicit query-view requests enter through typed `query_signatures.arrow`; automatically selected promoted query windows use the separate constructor without a temporary empty sidecar. |

## Python Wrapper Ownership

| Wrapper | Owner / caller | Status |
|---|---|---|
| `feature_port.build_rust_featurizer_from_arrow_paths(...)` | strict full predict, subblocked predict, raw Arrow scoring, fixed Rust training (`s2and/arrow_training.py`) | Direct wrapper for the pinned native Arrow constructor. |
| `feature_port.build_rust_featurizer(...)`, `_get_rust_featurizer(...)`, `warm_rust_featurizer(...)` | Rust-training datasets with immutable `arrow_paths` | Dataset-scoped dispatcher; classic `ANDData`/JSON datasets use Python featurization. |
| `rust_calls.get_constraints_matrix_indexed_rust(...)` and `get_constraints_block_upper_triangle_indexed_rust(...)` | full predict and parity | Maintained constraint wrappers. |
| `rust_calls.build_linker_pair_features_and_aggregate_stats_arrays_rust(...)` | promoted incremental pairwise scoring | Maintained canonical array wrapper. |
| `rust_calls.build_linker_pair_aggregate_stats_arrays_rust(...)` | promoted incremental aggregate-only path | Thin Python wrapper over `linker_pair_index_arrays_and_aggregate_stats(..., emit_matrix=False)`. |

## Build Information

| Key | Owner / caller | Status |
|---|---|---|
| `crate_version`, `profile`, `debug_assertions`, `opt_level`, `target` | `scripts/_rust_suite/common.py`, Rust-suite tests | Diagnostic report fields. Runtime compatibility is the exact package-version check, not these fields. |

## Cleanup Notes

- Keep `RawBlockQueryCandidatePlanner` as the canonical raw Arrow planning
  API; it owns reusable seed state and strict indexed-read defaults. Callers
  should use `from_query_signatures(...)` plus `plan_query_signatures()` or
  subset `plan(...)`.
- Do not delete `RustNameCompatibleSubblockSelector` internals; the pair-plan
  route still uses them for retrieval subblock filtering.
- Status 2026-05-25: `RustHybridCentroidRetriever.summary_count(...)` was
  removed after a repo-local no-caller scan.
- Status 2026-05-25:
  `linker_pair_features_and_aggregate_stats_indexed(...)` and its Python
  wrapper were removed after the repo-local callers moved to the canonical
  index-array API.
- Status 2026-05-25: aggregate-only remains a runtime mode, but the separate
  `linker_pair_index_arrays_aggregate_stats(...)` PyO3 method was folded into
  `linker_pair_index_arrays_and_aggregate_stats(..., emit_matrix=False)`.
- Status 2026-05-25: the string-pair `get_constraints_matrix(...)` PyO3 method
  and `rust_calls.get_constraints_matrix_rust(...)` wrapper were removed after
  parity tests moved to indexed constraint matrices.
- Status 2026-05-25: direct retriever debug APIs
  `top_k_hybrid_centroid(...)` and `chooser_feature_rows_subset(...)` were
  removed after callers and tests moved to the canonical pair-plan route.
- Status 2026-05-26: Python wrappers
  `feature_port.featurize_pair_rust(...)` and
  `feature_port.build_pair_feature_matrix_rust(...)` were removed. Rust pair
  feature batching from Python now requires
  `featurize_pairs_matrix_indexed(...)`.
- Status 2026-05-26: Rust PyO3 debug methods
  `RustFeaturizer.featurize_pair(...)`, `featurize_pairs(...)`, and
  `featurize_pairs_matrix(...)` were removed after repo-local tests and scripts
  moved to `featurize_pairs_matrix_indexed(...)`.
- Status 2026-05-26: `RustFeaturizer.from_json_paths(...)` and the Python
  JSON-ingest lifecycle were removed. Scripts now use Arrow
  `from_arrow_paths(...)` for Rust featurization.
- Status 2026-07-08: `RustFeaturizer.from_dataset(...)` was removed. Python
  `ANDData` remains responsible for JSON/pickle payloads and uses Python
  featurization unless a validated Arrow bundle is attached.
- Status 2026-05-26: `RustFeaturizer.save(...)` and
  `RustFeaturizer.load(...)` were removed. Counter-data measurement now uses
  build-time RSS deltas rather than Rust featurizer serialization.
- Status 2026-05-25: the one-shot
  `raw_block_query_candidate_plan_arrow(...)` PyO3 wrapper was removed after
  runtime callers moved to `RawBlockQueryCandidatePlanner`.
- Status 2026-05-25: `RustFeaturizer.from_feature_block(...)`,
  `feature_port.build_rust_featurizer_from_feature_block(...)`, and raw
  payload scoring wrappers were removed after a repo-local no-caller scan.
  Lower-level Python `FeatureBlock` builders remain only for fixture,
  compatibility-conversion, and parity-helper tests; production Rust scoring
  uses Arrow request tables.
- Status 2026-05-27: `signature_ngrams_batch(...)`,
  `normalize_text_compat(...)`, and the debug language-detector audit export
  were removed from the Python-visible Rust module. Their implementation
  helpers remain internal where production constructors need them.
- Status 2026-05-27: Arrow name-alias override paths are no longer a production
  input. Runtime aliases come from the explicit `name_tuples` argument.
- Status 2026-05-25: Arrow string columns are strict at the Rust boundary.
  ID, text/language, and alias columns must be Arrow string types; integer
  coercion is not accepted.
- Status 2026-05-25: Arrow graph subblocking uses raw-planner batch lookup
  indexes for filtered evidence reads and no longer exposes the unused Python
  full-table graph loader.
- Status 2026-05-27: the single-pair Rust constraint API was removed from the
  maintained surface. Constraint parity is owned by indexed matrix APIs.
- Status 2026-05-27: the pinned extension directly provides raw
  query-signature planning; `query_signatures.arrow` is request-local planner
  input, not a generic scoring artifact sidecar.
- Status 2026-05-28: callable PyO3 exports, module constants, and
  `get_build_info()` diagnostics were rechecked against the local
  `s2and_rust` module.

# Release Notes

## 0.60.0

- Fixes query-vs-query `cluster_seed_disallows` enforcement in promoted
  incremental linking. Previously a disallow pair whose endpoints were both
  residual queries was silently unenforced at link time, so two mutually
  disallowed queries could link into the same predicted cluster. Enforcement
  now happens at the moment a pair becomes resolvable: same-batch conflicts
  resolve in priority order (require-forced links first, then descending link
  score) over already-scored candidate rows, and a query whose disallowed
  partner linked in an earlier batch has that component excluded as if never
  retrieved. Contradictory hard constraints (a require into a component that a
  mutually-disallowed partner already claimed) raise
  `cluster_seed_disallow_conflicts_with_require_constraint`. New telemetry:
  `cluster_seed_disallow_excluded_row_count`,
  `cluster_seed_disallow_excluded_query_count`, and the
  `cluster_seed_disallow_same_batch_{conflict,reassigned_link,demoted_abstain}_count`
  counters.
- Rust subblocking now fails loudly on duplicate signature ids: the final
  complete-partition check compares multisets (matching the Python assert),
  and the ORCID merge pass raises a typed error instead of silently binding a
  duplicated signature to the last subblock visited.
- Rust language detection's text gate now counts alphabetic characters with
  Python `str.isalpha` semantics (general category L* only) instead of
  `char::is_alphabetic` (which also counts `Other_Alphabetic` combining marks
  and Nl characters). This removes a Python/Rust divergence on titles whose
  alphabetic content includes Indic combining vowel signs and similar marks
  near the zero-alpha and >0.9-uppercase-ratio gate boundaries.
- Breaking: `s2and-rust` is now a required runtime dependency of `s2and`.
  `uv pip install s2and` installs the Rust package; `s2and[rust]` remains only
  as a compatibility alias.
- Breaking: removes the Sinonym-dependent `ANDData` rewrite API. `ANDData(...)`
  no longer accepts `use_sinonym_overwrite` or `sinonym_overwrite_min_ratio`;
  callers must provide upstream-normalized names before constructing `ANDData`.
- Breaking: language detection is now CLD2-only; the fastText runtime
  dependency and `lid.176.bin` release artifact are removed. The pairwise
  language feature `language_reliability_count` is replaced by
  `language_reliability_min`, the minimum CLD2 reliable-confidence score for
  the two papers. Cached `predicted_language`/`is_reliable`/language
  reliability Arrow columns produced under the old policy must be regenerated
  or cleared; when the cached language columns are missing or NULL, Rust
  recomputes language locally from the raw title. `FEATURIZER_VERSION` is
  bumped to 9.
- Breaking: Rust featurization and ingest enter through Arrow IPC artifacts
  only; the JSON `from_dataset` ingestion surface is removed. Classic
  JSON/`ANDData` datasets remain supported through the Python featurizer.
- Feature caches are invalidated: `FEATURIZER_VERSION` is bumped 3 -> 9 across
  the feature-correctness pass (self-cite shared-paper guard, email
  missing-suffix handling, multi-token middle-initial comparison, empty-surname
  name-count keys, and related parity fixes), the language-detection policy
  change, and the Arrow migration.
- Adds a pure-Rust LightGBM evaluator for the production `.lgb` boosters
  (single-model binary-objective GBDTs with numerical splits; unsupported
  model types are rejected at load time). Production scoring no longer round
  trips through the Python `lightgbm` package; parity is pinned bit-for-bit by
  `tests/test_rust_lightgbm_booster_parity.py`.
- Arrow-backed training ingestion now streams IPC record batches for
  signatures/papers/authors and skips Python SPECTER embedding materialization
  by default when Rust featurization is attached. Pass
  `load_python_specter=True` only for Python reference featurization or direct
  Python embedding access.
- Filtered Arrow reads now reject duplicate signature/paper ids even when
  every copy of the duplicated id falls outside the requested id filter, so a
  filtered scan is never more permissive than a full scan.

## 0.51.1

- Ships the package as `0.51.1` and pins optional Rust installs to `s2and-rust==0.51.1`.
- Makes Arrow IPC the production Rust runtime boundary. Direct prediction now uses `Clusterer.predict_from_arrow_paths(...)` or Arrow-routed `Clusterer.predict(...)` over `signatures`, `papers`, `paper_authors`, selected `specter`, raw-planner batch indexes, and shared `name_counts_index`. JSON/`ANDData` remains available for compatibility, training, fixtures, and parity checks.
- Adds canonical Arrow runtime contracts and tooling: `s2and.arrow_inputs`, `s2and/arrow_schema_contract.json`, `scripts/convert_to_arrow.py`, `scripts/arrow_conversion_helpers.py`, local Arrow release validation, and bounded parity/quality verification scripts. The documented public data release is now `s2and-release-arrow`; the legacy JSON/pickle release remains for paper-era inputs.
- Tightens production validation. Missing or malformed Arrow artifacts now raise structured `MissingArrowArtifactError` failures, Rust production routes fail fast instead of silently falling back to `ANDData`, unsupported name-alias path keys are rejected, and direct Arrow prediction refuses models that require reference features.
- Reworks promoted incremental linking around Arrow/Rust. The promoted path reads base Arrow artifacts, query signatures, cluster seeds, cluster seed disallows, and altered-profile sidecars; Rust performs raw candidate planning and promoted row-signal construction; `batching_threshold` controls promoted Rust query batch size.
- Switches promoted linker replay/training to `--feature-mode arrow-rust` by default against the canonical `s2and_and_big_blocks_linker_dataset_20260525` Arrow+labels bundle. `precomputed-promoted` remains an explicit reuse mode.
- Uses the manifest-backed binary `name_counts_index/` sidecar as the Rust hot-path name-count artifact. Embedded Arrow name-count columns are no longer the runtime direction, and `name_counts.arrow` is generation, inspection, and parity-debugging only.
- Updates model, eval, tutorial, and profiling flows for Arrow-first operation. `eval_prod_models.py` can auto-use Arrow when complete artifacts exist, the production tutorial defaults to Arrow input, the documented model path is the native `production_model_v1.21/` bundle, and `rust_suite.py promoted-incremental-arrow-profile` replaces the legacy big-block incremental profiler.
- Adds graph subblocking as the default fallback for oversized name groups and introduces strict Arrow-native Rust graph subblocking with batch-indexed reads, expanded telemetry, ORCID repair behavior, and scoped dash-name compatibility.
- Narrows the Rust production surface to one route per job. Legacy/debug bridge APIs are removed or demoted, including raw FeatureBlock scoring, string-pair constraint matrix APIs, retriever debug APIs, aggregate-only linker helpers, and `s2and/rust_capabilities.py`.
- Clarifies cache behavior for the new runtime boundary. Direct Arrow/Rust production prediction bypasses the persistent pair-feature SQLite cache, Rust featurizer reuse is invalidated by cluster-seed changes, and artifact cache entries no longer probe raw-ETag filenames.

## Rust extension 0.50.0

- `s2and-rust>=0.50.0` is required for Rust-backed incremental linking.
- Native extension load failures now surface as import errors instead of silently falling back to Python. Missing extension modules still use the Python fallback path.
- Incremental linking uses the NumPy logistic link-or-abstain gate artifact format; legacy score/margin gate thresholds are not supported.
- Production linker finalization trains the final booster on train plus weighted calibration splits, then calibrates the final logistic gate on the held-out test split.
- Incremental name compatibility now accepts joined and first-token aliases in addition to exact first-name tuples.
- Artifact cache entries are keyed by validator type. Raw-ETag cache filenames are no longer probed.

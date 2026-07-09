# Work Plan

Status date: 2026-07-09

Landed on `canonical-v2-migration` since the 2026-07-05 status:

- Rust Arrow migration completed: Rust featurization/ingest enters through
  Arrow IPC only (the JSON `from_dataset` ingestion surface is removed);
  classic JSON/`ANDData` datasets stay on the Python featurizer. See
  [rust/runtime.md](rust/runtime.md). `FEATURIZER_VERSION` is now 9.
- Native Rust LightGBM evaluator
  ([../s2and_rust/src/lightgbm_booster.rs](../s2and_rust/src/lightgbm_booster.rs))
  scores production `.lgb` boosters; parity is pinned by
  `tests/test_rust_lightgbm_booster_parity.py`.

This is the active Rust/Arrow platform backlog. Stable architecture and artifact
contracts live in:

- [rust/inference_architecture.md](rust/inference_architecture.md)
- [rust/public_surface_inventory.md](rust/public_surface_inventory.md)
- [rust/artifact_formats.md](rust/artifact_formats.md)
- [rust/arrow_dataset_spec.md](rust/arrow_dataset_spec.md)
- [rust/runtime.md](rust/runtime.md)
- [rust/baselines.md](rust/baselines.md)

## Current Decisions

| Topic | Decision |
|---|---|
| `ANDData` | Keep as Python reference, training/eval, parity, fixture, and compatibility surface. Do not port all of `ANDData` to Rust. |
| Production inference | Production Rust inference should enter through raw Arrow IPC artifacts. JSON and Python objects are Python compatibility surfaces. |
| Arrow preprocessing | Production Arrow rows are runtime inputs, not preprocessed `ANDData` caches. Rust owns local normalization, ngram construction, unidecode, name handling, and language detection from raw Arrow inputs. |
| Name counts | Use manifest-backed `name_counts_index/` for hot-path lookups. Do not satisfy strict production bundles from ambient package/global fallbacks. |
| Batch indexes | Filtered production Arrow reads require raw-planner batch lookup indexes. Full scans are explicit test/compatibility opt-ins only. |
| SPECTER | Missing embedding rows are valid. Present rows are real vectors, including all-zero rows. Select `specter` or `specter2` through the manifest/path mapping. |
| Seeds | Incremental production requires a seed source, but not necessarily a physical `cluster_seeds.arrow`; request/dataset seed mappings may be materialized into request-local Arrow. |
| Optional sidecars | Missing `cluster_seed_disallows` means no seed-disallow constraints. Missing `altered_cluster_signatures` means no altered claimed profiles. If a sidecar key is declared, its file must exist and validate. |

## Canonical Arrow Input Surface

`s2and.arrow_inputs` is the strict production validation authority. Call sites
may resolve manifests or request-local overlays, but they should not reimplement
required-artifact, path-kind, missing-file, or batch-index policy.

The canonical surface owns:

- Path normalization and structured `MissingArrowArtifactError` diagnostics.
- Required and optional artifact policy for prediction, subblocking,
  incremental prediction, feature generation, script profiling, and eval.
- Runtime schema validation policy for string/int/bool/list fields, null
  handling, duplicates, and id semantics. Today the checks still live in the
  table readers, subblocking, and Rust implementation; centralize only when it
  removes duplicated policy.
- Batch lookup index requirements and explicit full-scan opt-ins.
- Signature subset/filtering semantics and request-local seed overlays.
- SPECTER path selection, dimensions, all-zero vectors, and missing-vector
  semantics.
- Manifest-backed `name_counts_index/`, name tuple policy, and alias policy.
- Text normalization/unidecode, local language detection from raw titles, name
  splitting, paper-author ordering, null position, and duplicate-position
  semantics.
- Seed sidecars and request-local seed materialization.
- Subblocking strictness, telemetry keys, and producer hints.

## Open Work

### 1. Performance Targets

Current evidence:

- `scripts/rust_suite.py promoted-incremental-arrow-profile` ran 5 isolated
  runs on the canonical local `pubmed` `r agarwal` block with 25 synthetic seed
  clusters and 25 query signatures because the canonical replay bundle has no
  `clusters` artifact.
- Release-build baseline (debug_assertions=false): p50 predict wall 2.18s,
  read_name_counts p50 0.775s (35.5% of wall), peak RSS 3.84 GB.
- After replacing `fs::read` with `memmap2`-backed reads for the four
  `name_counts_index/*.bin` files: p50 predict wall 2.01s, read_name_counts
  p50 0.622s (-19.7%), peak RSS 3.02 GB (-21.4%). Wall-time gain is -7.9%,
  below the 10% threshold for continued optimization, so no further work is
  scheduled on this workload.
- The 2026-05-27 reading of p50 ~11.15s was the debug-assertions cost; a
  release rebuild alone explains ~5x of that.
- Evidence: [rust/profiling/2026-05-28-promoted-incremental-arrow.md](rust/profiling/2026-05-28-promoted-incremental-arrow.md)
  (release-grade refresh and mmap delta);
  [rust/profiling/2026-05-27-promoted-incremental-arrow.md](rust/profiling/2026-05-27-promoted-incremental-arrow.md)
  (prior debug-assertions snapshot).

Next profiling target:

- Arrow read/summary construction and reusable component summaries on the
  canonical local promoted-incremental workload:
  `s2and/data/s2and_and_big_blocks_linker_dataset_20260525`.
- Use `scripts/rust_suite.py promoted-incremental-arrow-profile`, not the
  deleted JSON/`ANDData` big-block command.

Required metrics:

- p50 wall time over at least five isolated runs.
- Peak RSS.
- Summary-construction allocation volume from a stack-level allocation profiler
  where available.

Act only when:

- Arrow read or summary construction is at least a 10% contributor to p50 wall
  time or allocation volume, or the change removes a real `ANDData` dependency.
- Stop optimizing once measured improvement falls below 10% for the selected
  workload.

### 2. Feature-Space Parity And Correctness Bugs

These bugs change the values produced by featurization for currently-valid
inputs. Defer until they can be fixed together so feature-parity baselines and
trained models can be re-established in a single pass. Source: 2026-05-27
bug-validation pass.

Status update (2026-05-28): Rust-side and pure-logic fixes have landed; the
remaining open items are bugs that exist in Python (or in both Python and Rust)
and would change Python feature values when fixed. See "Fixed in 2026-05-28
correctness pass" below for what changed.

Status update (2026-07-04): a reachability audit re-verified every open item
below against the current tree. All remain present, with two corrections: line
references are refreshed throughout, and the same-signature
`paper_author_list_*` item moved from Tier A to Tier B because the current
linker training path already builds residual summaries that exclude the query
([scripts/production/model/linker_train_calibrate_eval.py:2591-2609](../scripts/production/model/linker_train_calibrate_eval.py#L2591-L2609))
and filters the query out of pair batches
([scripts/production/model/linker_train_calibrate_eval.py:1082](../scripts/production/model/linker_train_calibrate_eval.py#L1082)),
so no live path feeds a query-inclusive summary. When picking up the remaining
items for the re-baseline cycle, verify each on a live call path (not just
code presence) before scoping the fix.

Fixed in 2026-07-04 correctness pass (branch `canonical-v2-migration`): the
prescribed, decision-free items below are now fixed, Python and Rust in
lockstep where both sides were affected, with `FEATURIZER_VERSION` bumped 3->4
to invalidate the pair-feature cache. Covered by `tests/test_text.py`,
`tests/test_correctness_pass.py`, and the existing parity battery
(`test_feature_port_parity.py`, `test_arrow_training_ingestion.py`) which
verifies the two implementations still agree.

- Self-cite shared-paper (parity): guarded on `paper_id_1 != paper_id_2` in
  both `s2and/featurizer.py` and `s2and_rust/src/rust_featurizer.rs`.
- "MISSING" email collision (parity): absent `@` now yields a `None` suffix
  (feature = NaN, not a sentinel match); `s2and.text.email_prefix_suffix` and
  Rust `email_parts`.
- `equal_middle` multi-token (parity): single initial compared against the set
  of all token initials in `s2and/text.py` and Rust `middle_names_equal`.
- Whitespace-only `equal` (latent, parity): strip-then-test-empty in both
  `s2and.text.equal` and Rust `first_names_equal`.
- ORCID Unicode digits (Python): `ORCID_PATTERN` uses `[0-9]` so only ASCII
  digits match, aligning with Rust `is_ascii_digit()`.
- Reader NULL handling (Rust): the raw-Arrow readers coerce NULL -> "" for the
  nullable string columns (author first/middle/last/suffix, title/venue/
  journal_name), matching Python's `normalize_text(None) -> ""`. (An interim
  `required_value` change that errored on NULL was reverted: the producer's
  `_optional_str` serializes empty as Arrow NULL, so NULL is the normal empty
  representation -- e.g. author_suffix is ~100% NULL, and empty surnames are
  valid per D6 -- not a corruption signal. A Rust round-trip test now pins
  NULL -> "".)
- SPECTER all-zero ingest (Rust): `extract_specter_vec` keeps all-zero rows as
  present, matching the Arrow ingest path; the featurizer still treats them as
  missing at feature time (no feature-value change).
- `compute_ref` reference-list features (parity): the two reference-list
  features are computed whenever reference features are enabled (only the four
  ngram-Counter features still require `reference_details`), in both
  `s2and/featurizer.py` and `s2and_rust/src/rust_featurizer.rs`. The Rust side
  needed the same restructure for parity even though the doc had scoped it
  Python-only.
- Empty-surname name counts (parity, = D6): `_compute_signature_name_counts`
  (Python) and `build_name_counts_data_from_artifact` (Rust) both return NaN for
  every last-dependent key when the surname is empty, rather than the sentinel
  default 1. The Rust artifact-build path was a latent divergence the parity
  suite missed (no empty-surname case exercised it) and is now covered by a
  dedicated Rust regression test.
- `get_text_ngrams` short-token filter (Python): decoupled from stopword
  removal so reference-author ngrams (stopwords=None) also drop 1-2 char
  tokens. Rust consumes the Python-built `reference_details` counters, so no
  Rust change was needed.
- Same-signature `paper_author_list_*` guard (Tier B): ATTEMPTED, then REVERTED.
  Moving the `same_signature` continue above the `best_author_*` updates broke
  the pre-existing `test_local10_evidence_ignores_query_signature_member`, which
  deliberately locks paper-author-list self-matching (only the local10 window
  excludes the query). It is a speculative guard for a caller that does not
  exist (production and training already drop the query upstream), so per
  CLAUDE.md it was dropped; the guard stays scoped to local10.

Decided after this pass (implementation on this branch):

- **`detect_language` reliability** — DECIDED: language detection is CLD2-only.
  `predicted_language` is CLD2's top known language, `is_reliable` mirrors
  CLD2's reliable flag for known languages, and `language_reliability` is the
  CLD2 top-language percent divided by 100 when reliable, otherwise 0. Python
  (`s2and/text.py`) and Rust (`s2and_rust/src/language_detection.rs`) changed
  in lockstep. Production Arrow datasets are not expected to carry language
  columns in general: `predicted_language`/`is_reliable`/`language_reliability`
  are optional cached compatibility overrides. When the cached language columns
  are missing or NULL, Rust recomputes language locally from the raw title.
  Bumps `FEATURIZER_VERSION` for the renamed pairwise
  `language_reliability_min` feature.
- **fastText availability** — DECIDED: fastText is removed from runtime
  language detection. The Python dependency, `lid.176.bin` release artifact,
  and `S2AND_SKIP_FASTTEXT` test switch are gone.
- **Sinonym API removal** — DECIDED: `ANDData(..., use_sinonym_overwrite=...)`
  and `sinonym_overwrite_min_ratio` are removed without a compatibility alias.
  Canonical-v2 data preparation is expected to supply upstream-normalized names
  before `ANDData` construction; `ANDData` no longer owns a Sinonym-dependent
  rewrite step. This is an intentional breaking API change for callers that set
  those keyword arguments and must be called out in release notes.

Required when picking these up:

- Fix Python and Rust sides together where a bug exists in both.
- Re-record `compare_existing_arrow_anddata_feature_parity.py` baselines after
  each fix; expect intentional drift on the fixed columns.
- Re-train production pairwise models if cumulative feature drift exceeds the
  current `1e-5` tolerance on any non-changed column.

Remaining open bugs and decision items after the 2026-07-04 correctness pass:

- **Subblocking ORCID gating asymmetry between layers.** RESOLVED 2026-07-04
  (verified consistent, downgraded from open). A code trace shows both
  implementations already gate ORCID on the per-request `orcid_enabled` flag,
  not on field presence: Rust
  [raw_arrow_features.rs:143](../s2and_rust/src/raw_arrow_features.rs#L143)
  gates the `orcid_hash` on `orcid_enabled`, and Python
  `query_adapter.mask_query_features`
  ([:343](../s2and/incremental_linking/query_adapter.py#L343)) sets `orcid=None`
  when `orcid_enabled` is false, so a populated `author_info_orcid` is
  suppressed. The subblocking repair pass uses the same `use_orcid_subblocking`
  flag in both Python ([subblocking.py:1932](../s2and/subblocking.py#L1932)) and
  Rust ([subblocking.rs:2580](../s2and_rust/src/subblocking.rs#L2580)).
  `use_orcid_id` (ingest field-strip) and `suppress_orcid`/`orcid_enabled`
  (per-request) are orthogonal by design; all field×flag combinations are
  consistent, so there is no contradictory combination to guard. The
  flag-is-authoritative invariant is pinned by
  `tests/test_query_adapter.py::test_orcid_enabled_false_suppresses_populated_orcid_field`.
  (A hard single-flag master switch that also disables the subblocking pass on
  `suppress_orcid` remains an available product change, but it alters clustering
  and needs a re-baseline, so it is not done here.)

- **Arrow ingest hardcodes `NameCountsLastFirstInitialSemantics::InitialChar`.**
  RESOLVED 2026-07-04 (option b, write-time assert). `feature_block_from_anddata`
  ([scripts/arrow_conversion_helpers.py](../scripts/arrow_conversion_helpers.py),
  the single ANDData->Arrow funnel) now raises if the source ANDData declares a
  non-`initial_char` `name_counts_last_first_initial_semantics`, so a bundle can
  no longer be built from a `legacy_full_first_token` ANDData without a
  diagnostic. The Rust readers still pin `InitialChar`
  ([raw_arrow_features.rs:49-51](../s2and_rust/src/raw_arrow_features.rs#L49-L51),
  [rust_featurizer.rs:1637-1638](../s2and_rust/src/rust_featurizer.rs#L1637-L1638)),
  now guaranteed correct by construction at conversion time. Covered by
  `tests/test_feature_block.py::test_feature_block_from_anddata_rejects_legacy_name_count_semantics`.

- **Unicode `is_alphabetic` / `is_uppercase` claim in the language text gate.**
  RESOLVED 2026-07-09 (reproduced and fixed). The divergence is real: Python
  `str.isalpha()` accepts general category L* only, while Rust
  `char::is_alphabetic()` is the derived `Alphabetic` property (L* plus Nl plus
  `Other_Alphabetic`, e.g. Indic combining vowel signs). A text whose only
  `Alphabetic` characters are combining marks hits Python's zero-isalpha "un"
  early exit but proceeded to the detectors in Rust, and the same count skew
  could flip the >0.9 uppercase-ratio lowercasing branch (`"A B C ि"`:
  Python ratio 1.0, old Rust 0.75). Fixed by `is_python_alpha` /
  language text-gate counting in
  [s2and_rust/src/language_detection.rs](../s2and_rust/src/language_detection.rs)
  (general-category check via `unicode-properties`); `is_uppercase` needed no
  change (both sides use the derived `Uppercase` property, filtered to L*
  chars on both sides). Pinned by the Rust language-detection tests and the
  Python reference test in `tests/test_text.py`. This was part of the language
  feature-policy bump that now lands at `FEATURIZER_VERSION=9`.

### Second-pass bugs (2026-05-28)

Found during a follow-up sweep that explicitly excluded the original ten items.
The 2026-07-04 correctness pass fixed or rejected the resolved entries; only
the still-open items remain below. Tier A bugs change observable production
behavior today; Tier B bugs are latent, currently masked, or decision cleanup.

Status update (2026-07-08): three Tier B entries were closed.

Status update (2026-07-09): the section is fully closed. The Tier A
query-vs-query disallow bug is fixed (enforced at resolution time), the
duplicate-signature Tier B entry is closed with a loud Rust-side check, the
two-pair veto entry is confirmed as intentional policy and pinned, and the
Unicode `is_alphabetic` gate item in section 2 is reproduced and fixed. No
open bugs remain in this section; the entries below are kept as resolution
records.

- Stale-index `indexed_source_mtime_ns` header field: removed from the entry
  list as stale — the field no longer exists in
  [s2and_rust/src/arrow_batch_lookup.rs](../s2and_rust/src/arrow_batch_lookup.rs)
  (the header carries only size + full-content fingerprint); it had already
  been removed on `main` before this branch.
- Duplicate-id detection depending on the `keep_ids` filter: fixed. The
  signatures reader already checked pre-filter; the papers and specter readers
  now do the same, so a duplicate id whose copies are excluded by the
  keep-filter still fails (a filtered scan is never more permissive than a
  full scan). Pinned by the `filtered_duplicate_detection_tests` Rust unit
  tests in [s2and_rust/src/raw_arrow/readers.rs](../s2and_rust/src/raw_arrow/readers.rs).
- FNV64 batch-lookup collisions: resolved via the prescribed document-loudly
  option. A comment on `batch_indices_for_keys` in
  [s2and_rust/src/arrow_batch_lookup.rs](../s2and_rust/src/arrow_batch_lookup.rs)
  states the hash-only match, why the downstream exact-id re-filter makes it
  safe, the `rows_scanned` inflation side effect, and that any exact-mapping
  consumer must first add key verification material with a format-version
  bump.

#### Tier A — observable in production

- **Query-vs-query `cluster_seed_disallows` pairs silently dropped from raw planner exclusion.**
  RESOLVED 2026-07-09 (enforced at resolution time, not end-of-run
  reconciliation). A query-vs-query disallow pair is unenforceable at plan
  time — neither endpoint has a component — so enforcement happens the moment
  one endpoint resolves, at cost O(#query-vs-query disallow pairs):
  - Same batch: after batch scoring, contended queries finalize in priority
    order (require-forced links first, then descending link score); a
    lower-priority partner that landed on the same component is re-decided
    over its already-scored rows with that component removed (link to
    runner-up if it passes the gate, else abstain into residual Phase B). No
    re-scoring or re-featurization.
    ([runtime.py `_resolve_same_batch_disallow_conflicts`](../s2and/incremental_linking/runtime.py)).
  - Cross batch/window: before each batch, partners already present in
    `linked_signature_clusters` contribute their component to the query's
    hard-excluded set; excluded rows are treated exactly as if the planner had
    excluded them at retrieval (so require/ORCID forcing cannot override — a
    conflicting require raises
    `cluster_seed_disallow_conflicts_with_require_constraint`).
    ([production.py `_query_disallow_partner_ids`](../s2and/incremental_linking/production.py),
    [runtime.py `_cluster_seed_disallow_excluded_rows`](../s2and/incremental_linking/runtime.py)).
  - Both-abstain was already enforced: residual Phase B merges cluster-seed
    disallows into partial supervision
    ([model.py:5667](../s2and/model.py#L5667)).
  Both `cluster_seed_disallows` and partial-supervision disallow pairs feed the
  partner map. Telemetry: `cluster_seed_disallow_excluded_{row,query}_count`
  and `cluster_seed_disallow_same_batch_{conflict,reassigned_link,demoted_abstain}_count`
  (the validation counter `partial_supervision_disallow_between_residual_queries`
  remains as window telemetry). Pinned by the same-batch/cross-batch/require-
  conflict tests in `tests/test_incremental_linking_runtime.py`.

#### Tier B — latent, masked, or training-only

- **`apply_orcid_subblocking` Rust binds duplicate signature IDs to the last subblock visited; Python to flat dict insertion order.**
  RESOLVED 2026-07-09 (Rust now fails loudly, closing the residual exposure).
  `apply_orcid_subblocking` raises a typed error on any duplicated
  signature_id instead of silently binding last-wins (the check is a free
  test of the map insert it already performed), and the Rust final
  complete-partition check now compares multisets, matching the Python
  assert, so a duplicate fails loudly on Rust-only production runs even with
  ORCID subblocking disabled. Pinned by the
  `orcid_subblocking_rejects_duplicate_signature_id_instead_of_binding_last_wins`
  Rust unit test.

- **Disallow-veto coverage gap at `pair_count == 2 && disallow_count == 1`.**
  RESOLVED 2026-07-09 (confirmed intentional policy, no code change). The
  three rules are the discrete form of one coherent policy: veto on unanimous
  disallow evidence at any sample size, or >=80% with `pair_count >= 3`. At
  2 pairs / 1 disallow (50%) one historical member matched fine — vetoing
  there would make n=2 stricter than n>=3, which is non-monotonic. The
  layering also supports it: request-level hard disallows are enforced
  absolutely by candidate exclusion upstream (see the resolved Tier A entry);
  this veto layer weighs derived, noisier constraint evidence and is
  deliberately overridable by require/ORCID
  ([runtime.py:263-268](../s2and/incremental_linking/runtime.py#L263-L268)).
  Policy pinned by
  `test_constraint_disallow_veto_policy_pins_two_pair_half_disallow_fall_through`.

## Watchlist

### Compact Incremental Partial Supervision

[../s2and/incremental_linking/runtime.py](../s2and/incremental_linking/runtime.py)
raises `NotImplementedError` when compact-linker retrieved-candidate scoring
receives `partial_supervision`; that failure mode is asserted in
[../tests/test_incremental_linking_runtime.py](../tests/test_incremental_linking_runtime.py).
This is separate from `FastCluster.transform(...)`, which is intentionally
unsupported inductive-mode API and covered by
[../tests/test_model_pairwise_exceptions.py](../tests/test_model_pairwise_exceptions.py).

Do nothing unless a production compact-linker request path actually needs
partial supervision. If needed, first add a typed request fixture proving the
desired merge semantics, then wire the compact runtime behavior with explicit
tests for require/disallow conflicts.

Verification gate:

```powershell
uv run pytest -q tests/test_incremental_linking_runtime.py::test_private_retrieved_candidate_slice_rejects_partial_supervision
uv run pytest -q tests/test_model_pairwise_exceptions.py
```

## Blocked

### Normalization Canonicalization Migration

Blocked until canonical artifacts and retraining can move together. Full plan:
[normalization_migration_blocked.md](normalization_migration_blocked.md). The
ASCII/non-ASCII dash behavior, tuple probing fallbacks, ORCID prefix
fallbacks, and block compaction workarounds are measured legacy-compatibility
repairs, not the canonical target. Code TODO comments in
[../s2and/data.py](../s2and/data.py) and the production count scripts point at
this migration; do not schedule them as separate cleanup work before canonical
artifacts exist.

Update 2026-07-09: the four Open Decisions gating the migration freeze are
ruled (see the plan doc for the full rulings): single-mode `canonical_v2`
cutover (no runtime compatibility window, which moots the decommission-window
decision), keep the current alignment thresholds, and re-export canonical
names re-joined by signature id for the benchmark datasets used in production
training (decision only — the re-join/re-export tooling is deliberately not
written yet). The remaining blocker is canonical artifact regeneration plus
the v1.3 retrain moving as one release unit.

Update 2026-07-09 (later same day): migration step 2 landed —
`s2and.text.canonicalize_name_parts` and `s2and.text.canonical_name_count_keys`
implement the canonical_v2 pipeline as pure functions with no live consumers,
and the canonical-contract tests in
[../tests/test_canonical_name_examples.py](../tests/test_canonical_name_examples.py)
are active (previously skipped). The shim-removal TODOs in
[../s2and/data.py](../s2and/data.py) and the count-script rewrites remain
gated on canonical artifacts + the v1.3 retrain, as before.

Verification gate (compatibility behavior stays stable):

```powershell
uv run pytest -q tests/test_surname_hyphen_aware.py tests/test_subblocking_telemetry.py tests/test_text.py tests/test_arrow_training_ingestion.py tests/test_cluster_incremental.py
```

## Documentation Cleanup

- If licensing policy is corrected, update [../README.md](../README.md),
  [../pyproject.toml](../pyproject.toml), root [../LICENSE](../LICENSE), and
  dataset docs together. The current MIT / CC-BY-4.0 / ODC-BY mismatch is
  already preserved in README as a known issue.

## Standing Guardrails

These are not TODOs, but they should shape future work:

- Keep production artifact validation routed through `s2and.arrow_inputs`.
- Keep production Rust inference on `Clusterer.predict_from_arrow_paths(...)`
  or complete Arrow paths to `Clusterer.predict(...)`.
- Keep full scans and compatibility fallbacks explicit test-only or
  parity-only options.
- Prefer `Clusterer.predict_from_arrow_paths(...)` or Arrow-routed
  `predict(...)` for production inference; keep JSON/`ANDData` compatibility
  on Python paths.
- Keep production-scale `name_counts_index/` in S3, not Git/LFS;
  `name_counts.arrow` stays available for generation/inspection/parity, not
  request-time reads.
- Do not duplicate strict Arrow validation in scripts or model helpers;
  always go through `s2and.arrow_inputs`.

## Non-Goals

- Do not remove normalization shims before regenerated canonical artifacts are  validated.
- Do not add another strict/compatibility discovery layer beside
  `s2and.arrow_inputs`.
- Do not run S3/network release smokes as default pytest.

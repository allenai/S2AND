# Work Plan

Status date: 2026-05-28

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
| Production inference | Production Rust inference should enter through raw Arrow IPC artifacts. JSON, Python objects, and `RustFeaturizer.from_dataset(...)` are compatibility surfaces. |
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
(`test_feature_port_parity.py`, `test_rust_from_dataset_contract.py`) which
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
- Reader NULL coercion (Rust): required string columns route through
  `required_value` (errors on NULL; empty string still allowed).
- SPECTER all-zero ingest (Rust): `extract_specter_vec` keeps all-zero rows as
  present, matching the Arrow ingest path; the featurizer still treats them as
  missing at feature time (no feature-value change).
- `compute_ref` reference-list features (parity): the two reference-list
  features are computed whenever reference features are enabled (only the four
  ngram-Counter features still require `reference_details`), in both
  `s2and/featurizer.py` and `s2and_rust/src/rust_featurizer.rs`. The Rust side
  needed the same restructure for parity even though the doc had scoped it
  Python-only.
- Empty-surname name counts (Python, = D6): `_compute_signature_name_counts`
  returns `np.nan` for every last-dependent key when the surname is empty,
  rather than the sentinel default 1.
- `get_text_ngrams` short-token filter (Python): decoupled from stopword
  removal so reference-author ngrams (stopwords=None) also drop 1-2 char
  tokens. Rust consumes the Python-built `reference_details` counters, so no
  Rust change was needed.
- Same-signature `paper_author_list_*` guard (Tier B hardening): the
  `same_signature` continue now precedes the `best_author_*` updates in
  `s2and/incremental_linking/query_adapter.py`.
- Sinonym stale fields: `apply_sinonym_overwrites` invalidates the derived
  normalized-name and name-count fields when overwriting raw name parts.

Still open after this pass:

- **`detect_language` single-detector `is_reliable`** — genuine product policy
  call (should a single responding detector count as reliable?). Left unchanged
  pending a decision; not a mechanical fix.

Required when picking these up:

- Fix Python and Rust sides together where a bug exists in both.
- Re-record `compare_existing_arrow_anddata_feature_parity.py` baselines after
  each fix; expect intentional drift on the fixed columns.
- Re-train production pairwise models if cumulative feature drift exceeds the
  current `1e-5` tolerance on any non-changed column.

Open bugs (Python feature changes deferred for the next re-baseline cycle):

- **Sinonym overwrite leaves stale normalized fields when run outside `__init__`.**
  [s2and/data.py:2391-2395](../s2and/data.py#L2391-L2395) replaces only raw
  `author_info_first/middle/last`. Inside `__init__` the subsequent
  `preprocess_signatures()` call at
  [s2and/data.py:815](../s2and/data.py#L815) rebuilds normalized fields, so the
  canonical path is safe. Any call site that mutates signatures post-init
  produces stale `_normalized_*` / `name_counts`. Invalidate or re-derive in
  `apply_sinonym_overwrites` so the invariant is explicit.

- **Self-cite signal returns 1.0 when both signatures share a self-citing paper (parity bug).**
  Same bug exists in Python at
  [s2and/featurizer.py:1183](../s2and/featurizer.py#L1183) and Rust at
  [s2and_rust/src/rust_featurizer.rs:188-193](../s2and_rust/src/rust_featurizer.rs#L188-L193).
  When `s1.paper_id == s2.paper_id` and the paper appears in its own reference
  list, both produce 1.0. Fix both sides simultaneously to preserve parity.

- **"MISSING" email collision (parity bug).**
  Python at
  [s2and/featurizer.py:1117-1124](../s2and/featurizer.py#L1117-L1124) and Rust
  at [s2and_rust/src/features.rs:460-463](../s2and_rust/src/features.rs#L460-L463)
  both map emails without `@` to suffix `"missing"`, so two malformed emails
  produce a false suffix match. Return an `Option`/`None` suffix instead and
  treat absent `@` as missing. Fix both sides.

- **`equal_middle` falls through to 0 for multi-token middles (parity bug).**
  Python at [s2and/text.py:736-743](../s2and/text.py#L736-L743) and Rust at
  [s2and_rust/src/features.rs:381-399](../s2and_rust/src/features.rs#L381-L399)
  only compare the first character when one side is a single initial; later
  tokens of a joined multi-token middle that match the other side's initial
  return 0. Split both sides on whitespace and compare initial sets when either
  side is a single character.

- **Subblocking ORCID gating asymmetry between layers.** (Latent — outputs
  match on the default code path.) Rust at
  [s2and_rust/src/raw_arrow_features.rs:120](../s2and_rust/src/raw_arrow_features.rs#L120)
  gates the ORCID hash on a per-call `orcid_enabled` flag; Python relies on
  `author_info_orcid` being `None` when `use_orcid_id=False` at
  [s2and/data.py:554](../s2and/data.py#L554). Mismatched flag combinations
  between ingest and subblocking would diverge. Align the gating surface so the
  same flag controls both layers in both implementations.

- **Arrow ingest hardcodes `NameCountsLastFirstInitialSemantics::InitialChar`.**
  (Latent — correct by construction because Arrow datasets are contractually
  canonical, but the assumption is not enforced.) Rust
  [s2and_rust/src/raw_arrow_features.rs:49-51](../s2and_rust/src/raw_arrow_features.rs#L49-L51)
  and
  [s2and_rust/src/rust_featurizer.rs:1637-1638](../s2and_rust/src/rust_featurizer.rs#L1637-L1638)
  pin the semantic to `InitialChar` with an inline comment. If a future Arrow
  bundle were generated from a `legacy_full_first_token` ANDData, Rust would
  silently use the wrong `last_first_initial` keys with no diagnostic. Cheap
  defenses: (a) add `name_counts_last_first_initial_semantics` to the Arrow
  dataset manifest (defaulting to `initial_char`) and pass it through to
  `build_name_counts_data_from_artifact`, or (b) when the manifest is present,
  assert it declares `initial_char` and fail fast otherwise.

- **Unicode `is_alphabetic` / `is_uppercase` claim under review in the fastText
  0.9 gate.** Original report claimed Python's `isalpha` counts category Lm
  while Rust's `is_alphabetic` does not. Re-check on 2026-05-28: Python
  `str.isalpha()` returns True for Lu/Ll/Lt/Lm/Lo, and Rust
  `char::is_alphabetic()` is the Unicode `Alphabetic` property which also
  includes Lm. The actual divergence is narrower (Rust's `Alphabetic` includes
  `Other_Alphabetic` characters that Python's `isalpha` does not). If a real
  fastText branch flip is reproducible, re-document the precise category set
  and fix; otherwise close as a documentation correction.
  Sites:
  [s2and/text.py:360-365](../s2and/text.py#L360-L365);
  [s2and_rust/src/language_detection.rs:80-86](../s2and_rust/src/language_detection.rs#L80-L86).

- **`detect_language` reports `is_reliable=True` when only one detector responded.**
  [s2and/text.py:385-399](../s2and/text.py#L385-L399) treats `un_ft` or `un_2`
  as a successful agreement, so single-detector signals are weighted the same
  as two-detector agreement downstream. Decide whether single-detector should
  be reliable; if not, fix and propagate the new `is_reliable` semantics.

- **Query-vs-query `cluster_seed_disallows` pairs are unenforced.**
  Telemetry counter only on the Python side at
  [s2and/incremental_linking/runtime.py:1163-1164](../s2and/incremental_linking/runtime.py#L1163-L1164),
  silently dropped on the Rust side at
  [s2and_rust/src/raw_candidate_planner.rs:174-201](../s2and_rust/src/raw_candidate_planner.rs#L174-L201).
  Rust and Python currently agree on the no-op behavior, but enforcement
  requires post-link reconciliation across both implementations. Pick one
  policy (raise, telemetry-only, or enforce) coordinated with model owners.

### Second-pass bugs (2026-05-28)

Found during a follow-up sweep that explicitly excluded the ten items above.
Split into severity tiers. Tier A bugs change observable production behavior
today; Tier B bugs are latent (currently masked, harmless, or only fire in
labeled training paths) but worth fixing during the same re-baseline cycle.

The remaining items below are Python-side feature changes that would require a
re-baseline, or latent issues that are tracked but not actively fixed.

#### Tier A — observable in production

- **ORCID regex accepts Unicode digits in Python but Rust requires ASCII.**
  Python `ORCID_PATTERN` at
  [s2and/text.py:109-115](../s2and/text.py#L109-L115) uses bare `\d` (no
  `re.ASCII`), so Unicode digit classes (Arabic-Indic `٠-٩`, etc.) match.
  Rust `normalize_orcid_owned` at
  [s2and_rust/src/orcid.rs:19,34](../s2and_rust/src/orcid.rs#L19) requires
  `is_ascii_digit()`. Concrete divergence: a string like
  `"https://orcid.org/٠٠٠٠-٠٠٠٢-١٨٢٥-009X"` returns a normalized ORCID under
  Python and `None` under Rust. Fix the Python side to require ASCII digits
  (add `re.ASCII` or `[0-9]`).

- **Reader silently coerces NULL to `""` for schema-required string columns.**
  [s2and_rust/src/raw_arrow/readers.rs:160-163](../s2and_rust/src/raw_arrow/readers.rs#L160-L163)
  and
  [s2and_rust/src/raw_arrow/readers.rs:249-255](../s2and_rust/src/raw_arrow/readers.rs#L249-L255)
  call `optional_owned(row).unwrap_or_default()` on `signatures.author_first`,
  `author_middle`, `author_last`, `author_suffix` and on `papers.title`,
  `venue`, `journal_name`, all of which the schema contract at
  [s2and/arrow_schema_contract.json:30-43](../s2and/arrow_schema_contract.json)
  declares `required=true`. Compare `paper_authors.author_name` at
  [s2and_rust/src/raw_arrow/readers.rs:302-304](../s2and_rust/src/raw_arrow/readers.rs#L302-L304),
  which correctly errors via `required_value`. The current behavior turns
  upstream NULLs into empty-string name-count lookups (see [Bug 5 below](#empty-last-coalesces-to-1)),
  empty-title term sets, etc. Fix: route the affected fields through
  `required_value` and let `s2and.arrow_inputs` raise structured
  `MissingArrowArtifactError`.

- **SPECTER all-zero rows mean "missing" via Python dict ingest and "present" via Arrow ingest.**
  Python-dict path
  [s2and_rust/src/ingest_dataset.rs:706-743](../s2and_rust/src/ingest_dataset.rs#L706-L743)
  drops all-zero embeddings as missing; Arrow path
  [s2and_rust/src/raw_arrow/readers.rs:490-517](../s2and_rust/src/raw_arrow/readers.rs#L490-L517)
  via
  [s2and_rust/src/raw_arrow/arrow_io.rs:245-287](../s2and_rust/src/raw_arrow/arrow_io.rs#L245-L287)
  keeps them as real centroids. The work plan already declares "Present rows
  are real vectors, including all-zero rows" (Current Decisions table), so the
  Arrow path matches the decision; the Python-dict path silently filters. Pick
  one and align — the cleanest fix is to remove the zero-vector filter in the
  Python-dict path so the two ingest modes share semantics.

- **`compute_ref` drops self-citation and reference-Jaccard signal whenever `reference_details is None`.**
  [s2and/featurizer.py:1175-1189](../s2and/featurizer.py#L1175-L1189)
  gates the entire reference feature block on `compute_ref` AND both papers
  having non-None `reference_details`. But `paper_id_2 in references_1` and
  `jaccard(references_1, references_2)` only need `paper.references` (the raw
  int list, always populated by the loader). On any non-canonical path with
  `preprocess=False` and `compute_reference_features=True`, two computable
  features become NaN. Fix: compute the two reference-list features
  unconditionally when both `references` lists are present, and gate only the
  four ngram-Counter features on `reference_details`.

- <a id="empty-last-coalesces-to-1"></a>**`_compute_signature_name_counts` returns `last=1` for empty surnames (sentinel collision with genuinely once-seen surnames).**
  [s2and/data.py:902-907](../s2and/data.py#L902-L907):
  `last=self.last_dict.get(last_for_counts, 1)` returns the default `1` when
  `last_for_counts == ""`, indistinguishable from a real corpus count of `1`.
  Same for `last_first_initial` on line 906. `first` and `first_last` are
  symmetric and correctly return `np.nan` (lines 903, 905). Together with the
  reader bug above, a NULL `author_last` ends up as a rare-surname signal.
  This is the same shape of bug as the documented "MISSING" email collision —
  fix by returning `np.nan` (or an explicit None) when `last_for_counts == ""`.

- **Query-vs-query `cluster_seed_disallows` pairs silently dropped from raw planner exclusion.**
  [s2and_rust/src/raw_candidate_planner.rs:174-201](../s2and_rust/src/raw_candidate_planner.rs#L174-L201)
  only records exclusions when exactly one endpoint is a query; pairs where
  both endpoints are residual queries write nothing.
  [s2and/incremental_linking/runtime.py:1156-1160](../s2and/incremental_linking/runtime.py#L1156-L1160)
  merely increments a counter for the same case. Two mutually-disallowed
  residual queries can still be linked to the same predicted cluster within
  one batch. Fix: enforce the constraint at batch reconciliation, or reject
  query-vs-query disallows at validation time with a typed error.

- **`get_text_ngrams` couples the short-token filter to `stopwords is not None`.**
  [s2and/text.py:569-600](../s2and/text.py#L569-L600): the
  `len(word) > 2` filter is only applied inside the `stopwords is not None`
  branch. Reference-author ngrams built in
  [s2and/data.py:2528](../s2and/data.py#L2528) pass `stopwords=None`,
  accidentally disabling the short-token filter as well, so reference-author
  ngrams include 1-2 char tokens (`"li"`, `"wu"`) that title/venue ngrams
  drop. Decouple the two filters — either apply `len > 2` unconditionally or
  add an explicit second argument.

#### Tier B — latent, masked, or training-only

- **Same-signature guard in `raw_paper_evidence_features` runs after the `best_author_*` updates.**
  (Moved from Tier A on 2026-07-04: the earlier "pollutes training data" claim
  was inaccurate as written — the residual-summary training path predates that
  entry, commit `f7e98c7`.)
  [s2and/incremental_linking/query_adapter.py:578-611](../s2and/incremental_linking/query_adapter.py#L578-L611):
  the `if same_signature: continue` guard at line 605 skips the local10
  features only after `best_author_jaccard`, `best_author_containment`,
  `best_author_overlap`, and `best_author_count_log_absdiff` are updated with a
  perfect self-match. No live path feeds a query-inclusive summary today:
  production incremental linking excludes seeds via
  `unassigned_signature_ids`, and linker training builds residual summaries
  that exclude the query
  ([scripts/production/model/linker_train_calibrate_eval.py:2591-2609](../scripts/production/model/linker_train_calibrate_eval.py#L2591-L2609))
  and drops the query from pair batches
  ([scripts/production/model/linker_train_calibrate_eval.py:1082](../scripts/production/model/linker_train_calibrate_eval.py#L1082)).
  Fix as cheap hardening for future callers: move the `if same_signature:
  continue` guard above the paper-author-list updates.

- **`equal()` returns 1 (equal) for two whitespace-only first/middle/last inputs.**
  [s2and/text.py:698-707](../s2and/text.py#L698-L707): the empty check uses
  raw `len()` but the comparison uses `.lower().strip()`. `equal(" ", "  ")`
  returns `1` (perfect match) instead of `default_val (NaN)`. Normalized
  pipelines should not produce whitespace-only normalized names, but the
  asymmetry between the raw-length guard and the post-strip comparison is a
  latent foot-gun.

- **Stale-index header reserves `indexed_source_mtime_ns` but freshness check ignores it.**
  [s2and_rust/src/arrow_batch_lookup.rs:78-109](../s2and_rust/src/arrow_batch_lookup.rs#L78-L109):
  the 8-byte field is read into `_indexed_source_mtime_ns` (underscore →
  discarded) and only `(size, fingerprint)` are compared. The fingerprint
  hashes full file contents so freshness is correct, but the field is dead
  documentation in the on-disk format. Either remove from the header (and
  bump the format version) or actually enforce.

- **`FNV64` batch-lookup keys can collide; downstream filtering masks but telemetry double-counts.**
  [s2and_rust/src/arrow_batch_lookup.rs:161-186](../s2and_rust/src/arrow_batch_lookup.rs#L161-L186):
  `lower_bound` matches on 64-bit hash equality, not on raw key, so two
  distinct keys hashing to the same value return both batches. The reader
  re-filters by `keep_ids.contains(...)`, so output rows are correct, but
  `rows_scanned` telemetry is inflated and any future consumer trusting
  `batch_indices_for_keys` as exact would be wrong. Either store enough key
  material to verify, or document the FNV-collision assumption loudly.

- **Duplicate-id detection in Arrow readers depends on the `keep_ids` filter.**
  [s2and_rust/src/raw_arrow/readers.rs:128-146,222-240,473-516](../s2and_rust/src/raw_arrow/readers.rs#L128-L516):
  `seen_paper_ids`/`seen_signature_ids` is populated after the filter runs,
  so an Arrow file with two rows sharing the same id is rejected only when
  both copies survive the filter. Pre-filter detection would catch upstream
  corruption symmetrically.

- **`apply_orcid_subblocking` Rust binds duplicate signature IDs to the last subblock visited; Python to flat dict insertion order.**
  [s2and_rust/src/subblocking.rs:2270-2280](../s2and_rust/src/subblocking.rs#L2270-L2280)
  vs
  [s2and/subblocking.py:1937-1941](../s2and/subblocking.py#L1937-L1941).
  Currently a non-issue because upstream is supposed to partition signatures
  across subblocks (the function does not defend against a duplicate). The
  Python `assert` at
  [s2and/subblocking.py:2031](../s2and/subblocking.py#L2031) only checks set
  equality, not multiplicity, so any future invariant break would silently
  produce different ORCID merge graphs on the two implementations.

- **Disallow-veto coverage gap at `pair_count == 2 && disallow_count == 1`.**
  [s2and/incremental_linking/runtime.py:254-258](../s2and/incremental_linking/runtime.py#L254-L258):
  the three veto rules cover `pair_count <= 1` (`single_pair_disallow`),
  `disallow_count >= pair_count` (`all_pairs_disallow`), and
  `pair_count >= 3 && fraction >= 0.8` (`mostly_disallow`). The 2-pair
  half-disallow case falls through the cracks while the 1-pair 100% disallow
  case is honored. Confirm whether this is intentional product policy; if
  not, add an explicit two-pair rule.

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

Verification gate (compatibility behavior stays stable):

```powershell
uv run pytest -q tests/test_surname_hyphen_aware.py tests/test_subblocking_telemetry.py tests/test_text.py tests/test_rust_from_dataset_contract.py tests/test_cluster_incremental.py
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
  `predict(...)` for production inference; keep
  `feature_block_from_arrow_paths(...)` and `RustFeaturizer.from_dataset(...)`
  as fixture/parity/training surfaces only.
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

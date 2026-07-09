# Normalization Unification Migration Plan

Execution status (last reconfirmed 2026-07-09; originally entered blocked state 2026-03-02)
- Blocked: normalization work is on hold until the required data/artifacts are ready.
- Unblocking in progress (2026-07-04): canonical upstream names are arriving; this plan is
  expected to execute together with the production v1.3 retrain, with regenerated artifacts
  and retrained models moving as one release unit.
- Open Decisions ruled (2026-07-09): single-mode cutover (OD4), decommission window moot (OD1),
  thresholds unchanged (OD2), benchmark training names re-exported by signature-id re-join (OD3,
  decision only — tooling not yet written). See the rulings inline below. The migration freeze is
  no longer gated on decisions, only on canonical artifacts + the v1.3 retrain.
- Step-2 canonical routines landed (2026-07-09): `s2and.text.canonicalize_name_parts` and
  `s2and.text.canonical_name_count_keys` implement the canonical_v2 pipeline as pure functions,
  and the fixture's canonical contract in `tests/test_canonical_name_examples.py` is active
  (no longer skipped). They are deliberately not consumed by live code paths yet; wiring them
  in is the cutover and moves with regenerated artifacts + the v1.3 retrain.
- Keep this plan separate from the active execution plan in `docs/work_plan.md`.

Status
- Draft updated from issue notes through August 31, 2025.
- Rust compatibility-alignment notes added on February 20, 2026; helper porting is no longer tracked here as a separate blocking work item.
- Reviewed on February 24, 2026 alongside big-block execution planning; no normalization-policy changes in that workstream.
- Rechecked on May 26, 2026 during Rust Arrow graph-subblocking work. The current production-quality subblocking
  behavior depends on a localized legacy-compatibility repair for dash-like given names; that repair is documented
  below as current compatibility behavior, not as the canonical target state.
- Path-only audit on May 28, 2026: refreshed Rust code references for
  `build_name_counts_data_from_artifact`, `canonical_last_for_counts`, and
  `normalize_subblocking_signature_rows` after earlier module extractions out of `lib.rs`,
  and noted that Rust now supports both `last_first_initial` semantics via
  `NameCountsLastFirstInitialSemantics`. No policy or compatibility-shim changes.
- Blast-radius review on July 4, 2026, ahead of the v1.3 retrain: added "Implementation
  notes (2026-07-04)" and Open Decisions 3-4 below; no normalization-policy changes. The
  `docs/work_plan.md` section-2 bug list was re-verified the same day (one entry demoted to
  latent after a reachability check).

Scope
- Unify name normalization for first/middle/last across data preparation, modeling, subblocking, and auxiliary datasets (name counts, name tuples, ORCID prefix counts).
- Ensure training-time and inference-time normalization are identical, including upstream normalized-name behavior.

Decided (from issue history)
- Apostrophes: canonical fields should remove apostrophes globally (no dual stream).
- Chinese given names: upstream data preparation handles language-aware compounds; hyphenated given names should stay together.
- Spaced given names (ruled 2026-07-04, per issue #39): dash-bound tokens stay together,
  but separate space tokens after a dashed compound still spill into middle; when the raw
  first has no dash-like character, tokens after the first also spill into middle.
  Cross-variant compatibility (`Jo` / `Jo Ann` / `JoAnn`) is a compare-time concern
  (`same_prefix_tokens`), not canonicalization identity.
- Apostrophe-like marks (ruled 2026-07-04): all apostrophe-like characters, including backtick,
  are removed from name fields (never treated as separators).
- Count keys (ruled 2026-07-04): canonical count keys use canonical fields after missing and
  informativeness gating (the compact-join shims are retired); the informative threshold is
  string-level `len(key) > 1` for first-dependent keys. Multi-initial compounds such as `j p`
  and joined initials such as `jp` are intentionally informative; a lone initial such as `j`
  is not. Keys with a missing component are null, never sentinel counts. The null-key change
  is a feature-value change for missing-first examples because legacy still looks up
  `last_first_initial` from the last name alone; empty-last/null-last are already pinned to
  null last-dependent keys in the live code. This is gated by the retrain re-baseline eval,
  like the work_plan section-2 fixes.
- Dash semantics: canonical behavior should not assign different semantic meaning to ASCII hyphen versus Unicode
  dash-like characters. Any current ASCII/non-ASCII split must be treated as a measured legacy-compatibility repair,
  not as the desired canonical policy.
- First-name compatibility checks should use multi-token prefix logic (`same_prefix_tokens`), not single-token-only rules.
- Canonical surname storage:
  - Persist canonical last names as normalized, space-separated tokens (e.g., `ou yang`, `de la cruz`).
  - Treat hyphen/space variants equivalently during canonicalization.
  - Use compact surname projection (remove spaces/hyphens) only where required (block key and legacy-compatible count keys during transition).
- Surname particles/prefixes:
  - Preserve particles in canonical surname form (`de la`, `van der`, `bin`, etc. are not dropped or rewritten).
- Artifact regeneration is required after normalization policy changes:
  - Name counts.
  - Name tuples.
  - ORCID prefix counts.
- Retraining requirement: production retraining data must go through the same normalization path as production inference.

Canonicalization Pipeline (target `canonical_v2`)
1. Inputs are raw `first`, `middle`, and `last`; `None` is treated as missing/empty.
2. Normalize Unicode spacing before tokenization. NBSP is whitespace. Soft hyphen (`U+00AD`)
   and zero-width joiner (`U+200D`) are invisible formatting controls and are deleted, not
   treated as dash separators.
3. Map every D3 apostrophe-like code point to ASCII apostrophe before transliteration; after
   transliteration, delete ASCII apostrophe and backtick rather than converting them to spaces.
4. Map every configured dash-like character to a dash separator. The canonical dash set is
   exactly `-\u2010\u2011\u2012\u2013\u2014\u2212\ufe58\ufe63\uff0d`; soft hyphen is deliberately
   outside this set.
5. Transliterate with `text-unidecode`, lowercase, replace remaining non-letter/non-whitespace
   characters with spaces, and collapse repeated whitespace.
6. Drop at most one leading title-prefix token from the normalized first field.
   `md` is retained as a common South Asian given-name abbreviation, not dropped as a
   title prefix.
7. Split first/middle after normalization:
   - If normalized first has no dash-bound group, keep the first token as first and spill later
     first tokens into middle ahead of existing middle tokens.
   - If normalized first starts with a dash-bound group, keep that group in first as spaced
     tokens; separate space tokens after that group still spill to middle (`Anne-Marie Claire`
     -> first `anne marie`, middle `claire`).
8. Normalize last independently and preserve normalized spaces in canonical surname form.
   Suffix/postnominal-like tokens already present in `last` are retained; suffix stripping is
   outside `canonical_v2`.
9. Build count keys from canonical fields after gating:
   - `first`: canonical first when first is present and `len(first) > 1`, else null.
   - `last`: canonical last when last is present, else null.
   - `first_last`: `<first> <last>` when first is informative and last is present, else null.
   - `last_first_initial`: `<last> <first[0]>` when first and last are present, else null.

Compare-Time First-Name Compatibility
- `same_prefix_tokens` is a symmetric predicate over already-normalized canonical first fields.
  It is a compare/backoff contract, not canonicalization identity.
- For each aligned token pair up to the shorter token list, one token must be an exact prefix
  of the other. Extra tokens on the longer side are allowed. Callers must not treat an empty
  first field as positive name evidence.
- Middle and last names do not participate in this predicate; they are handled by separate
  fields/features.

Truth table pinned by `tests/fixtures/canonical_name_examples.json`:

| A | B | Compatible? | Reason |
|---|---|---|---|
| `jo` | `joann` | yes | single-token prefix |
| `jon` | `john` | no | no exact prefix relation |
| `j p` | `jean pierre` | yes | both aligned tokens are prefixes |
| `j p` | `john paul` | yes | prefix compatibility, not semantic identity |
| `yu zhong` | `y z` | yes | initials are aligned prefixes |
| `yu` | `yusuf` | yes | short token is a prefix of longer token |
| `john david` | `j f` | no | second aligned token mismatches |
| `yu` | `wei` | no | no aligned prefix relation |
| `` | `alice` | no | empty first is missing evidence |
| `` | `` | no | missing evidence on both sides is still not compatibility |

Fixture Metadata
- `family` is a reviewer-facing category label.
- `equivalence_group` means all members must share identical canonical first/middle/last fields.
- Per-case `decisions` are primary review highlights, not exhaustive dependency labels. The
  policy registry is the source of truth for global applicability.

Open Decisions (all ruled 2026-07-09; kept with their rulings as the decision record)
1) Compatibility-mode decommission window
   - RULED 2026-07-09: moot. Open Decision 4 adopts the single-mode cutover, so there is no
     runtime compatibility mode to decommission.
2) Threshold tightening
   - RULED 2026-07-09: keep the current thresholds (pairwise AUC delta <= 0.001, F1 <= 0.005,
     clustering B3 <= 0.005; runtime/RSS <= 10%). They gate no-op alignment/refactor comparisons
     on unchanged inputs — the retrain release gate is end-metric non-regression (clarified
     2026-07-04) — so tightening buys no protection for the retrain and adds flake risk from
     benign nondeterminism. Revisit only if a real regression slips under them.
3) Benchmark training-data names (added 2026-07-04)
   - RULED 2026-07-09: re-export canonical names re-joined by signature id, at minimum for the
     benchmark datasets used in production training. Rationale: read-time renormalization runs the
     same code either way — the difference is the input distribution. Upstream data preparation
     performs language-aware compound handling (see Decided) that `canonical_v2` applied to legacy
     raw strings cannot reconstruct, and the benchmark datasets are the pairwise training data, so
     the distribution gap would land in the shipped model exactly on the name-compound features
     this migration targets.
   - Execution notes (decision only — the re-join/re-export tooling is deliberately NOT written
     yet, per 2026-07-09 ruling): join by signature id, log the join rate loudly, and add
     canonical name columns alongside the raw fields rather than replacing them so historical
     comparisons survive. Optional cheap pre-gate before building the tool: canonicalize benchmark
     raw names and diff against upstream-canonical names for overlapping signature ids; if
     divergence is genuinely negligible, the re-export can be deferred.
4) Single-mode cutover vs compatibility window (added 2026-07-04)
   - RULED 2026-07-09: single-mode, per the recommendation. Drop the dual-mode
     `legacy_compat`/`canonical_v2` runtime contract; a release accepts only `canonical_v2`
     artifacts and fails fast on code/artifact mismatch. Rollback = redeploying the previous
     package + artifact set. Artifacts, models, and code already ship as one checksummed release
     unit, and "old code + canonical_v2" is unsupported either way; a runtime compatibility mode
     doubles the cutover test matrix for a rollback path that package/artifact pinning already
     provides. Implement the fail-fast via `normalization_version` in the `name_counts_index/`
     manifest and Arrow dataset manifests asserted against the model feature contract (see
     "Implementation notes (2026-07-04)"), which also permanently discharges the
     hardcoded-`InitialChar` concern.

Rust Alignment Decisions (effective February 20, 2026; refreshed 2026-05-23)
1) Canonical cutover contract
   - Rust ingestion paths must stay compatible with current Python + compatibility-shim behavior until canonical artifacts are regenerated and versioned.
   - Switch Rust and Python to canonical-only behavior only after the same canonical artifact gates pass.
2) Version-compatibility contract
   - `normalization_version=legacy_compat`:
     - allows current code paths + legacy artifacts + compatibility shims.
   - `normalization_version=canonical_v2`:
     - requires regenerated canonical artifacts for name counts, name tuples, and ORCID prefix counts.
     - must fail fast on code/artifact version mismatch unless an explicit temporary compatibility override is enabled.
   - old code + `canonical_v2` artifacts is unsupported.
3) Removal contract for compatibility shims
   - Do not remove `_canonicalize_last_for_counts`, `_lasts_equivalent_for_constraint`,
     name-tuple compatibility probing, or ORCID first-token fallback until canonical artifacts are validated in rollout.
4) Retraining contract
   - Before enabling canonical mode by default, production retraining and production inference must use the same canonical normalization path.
5) Rust coupling
   - Any Rust ingestion change that affects normalized names, name-count keys, ORCID fallbacks, or block keys must be treated as a policy-sensitive change, not a pure performance refactor.

Implementation notes (2026-07-04 blast-radius review)
- `normalization_version` enforcement is net-new: no code path or manifest carries it today. The
  model bundle already records a versioned feature contract
  (`s2and/data/production_model_v1.21/clusterer.json` ->
  `feature_contract.name_counts_last_first_initial_semantics`); the recommended mechanism is to
  extend that same contract shape into the `name_counts_index/` manifest and the Arrow dataset
  manifests (add `normalization_version` alongside `name_counts_last_first_initial_semantics`),
  then assert model contract == artifact contract at load in `s2and.arrow_inputs` and in the Rust
  readers (the `schema_version` gate in `s2and_rust/src/name_counts.rs` is the natural fail-fast
  hook). This also discharges the latent hardcoded-`InitialChar` item in `docs/work_plan.md`.
- Name tuples can be regenerated deterministically: re-normalize
  `s2and/data/s2and_unnormalized_filtered_name_tuples.txt` through the canonical normalizer
  instead of re-running the archived hmni/LLM pipeline. Regenerated tuples must be emitted
  symmetrically (or Rust must symmetrize on insert): `insert_name_tuple_alias` in
  `s2and_rust/src/ingest_dataset.rs` is directional and relies on the shipped file listing both
  directions of each pair.
- Count generators are internal doc-stubs (`pys2`) built on legacy normalization. Regenerating
  `name_counts.pickle` and `first_k_letter_counts_from_orcid.json` requires rewriting both
  scripts to import the canonical routine before running them on internal infrastructure;
  `generate_orcid_name_prefix_counts.py::normalize_names` is a divergent inline copy of the
  legacy splitter and must not survive the rewrite. `name_counts_index/` is a pure
  re-serialization of the pickle and is regenerated afterwards (note: the checked-in index
  manifest predates the current writer and lacks its `fingerprint` field; a clean regeneration
  adds it).
- Additional cutover sites beyond "References in code" below:
  - Inline duplicate of the name-tuple probing logic in `s2and_rust/src/rust_featurizer.rs`
    (constraint scoring, ~lines 345-351); remove together with the `first_names_name_compatible`
    probing forms.
  - `author_info_first_normalized` (single-token) is a legacy cached first-token field:
    `ANDData.preprocess_signatures` still reads/materializes it, but feature, constraint,
    subblocking, and model-scoring paths consume `author_info_first_normalized_without_apostrophe`
    instead. Remove it with the dual-field unification.

Current State (post-hyphen pass)
- Given-name canonicalization currently preserves hyphenated Chinese given names:
  - `s2and.text.split_first_middle_hyphen_aware`.
- Generic text normalization treats all punctuation/dash-like characters as separators after transliteration; this is
  shared by generic name/text features, affiliation/coauthor evidence, titles, venues, and Rust compatibility helpers.
- ORCID normalization accepts ASCII and Unicode dash-like separators and emits canonical ASCII-hyphenated ORCIDs; compact
  ORCID keys remove those hyphens afterward.
- Backward-compat shims exist for artifacts built with legacy normalization:
  - Name counts (first): when raw first had a hyphen, join spaces in canonical first for count keys (e.g., `qi xin` -> `qixin`).
  - Name counts (last): join spaces in canonical last for compound/hyphenated surnames (e.g., `ou yang` -> `ouyang`).
    - Helper: `_canonicalize_last_for_counts(...)`.
  - Constraints: last-name disallow uses space-insensitive comparison (`ou yang` == `ouyang`).
    - Helper: `_lasts_equivalent_for_constraint(...)`.
  - Subblocking: ORCID prefix map lookup has a first-token fallback for multi-token first names.
  - Name tuples in constraints and incremental new-name guarding: shared helper
    `first_names_name_compatible(...)` probes exact, joined, and first-token forms for compatibility with legacy tuples.
- Subblocking first/middle keys have an additional measured legacy-compatibility repair:
  - Canonical first/middle fields keep dash-like given names together.
  - Current subblocking quality is recovered by keeping ASCII-hyphen compounds together while spilling non-ASCII dash
    compounds into first + middle for subblocking keys only.
  - This is not semantically desirable, but it matches current legacy artifacts and restored measured quality on the
    active 20260525 Arrow artifacts with sparse graph fallback:
    - `s_lee`, `maximum_size=2500`: keep-all-dash recall `0.978647821860`; current repair recall
      `0.983113309912` versus historical graph `0.983118072979`.
    - `s_park`, `maximum_size=2500`: keep-all-dash recall `0.973201405109`; current repair recall
      `0.979665201080`, matching the historical graph value.
    - `h_wang`, `maximum_size=5000`: current sparse graph repair recall `0.914999198749`, above the historical graph
      floor `0.911296989543`.
  - Uniform single-key alternatives were tested and were worse on the active artifacts:
    - Keep all dashed compounds together regressed `s_lee` and `s_park`.
    - Spill all dashed compounds increased single-letter first-name signatures, fallback work, and regressed
      `s_lee`/`h_wang`.
    - Local-count adaptive key choice also regressed `s_park`: splitting a dashed compound whenever the split view was
      locally larger gave recall `0.971676243404`; requiring the split view to be at least 2x larger gave
      `0.973458134081`, still below the `0.979665201080` gate.
  - The clean replacement should be alias-aware or canonical-artifact-based, not another single-key dash heuristic.

Fix during the blocked canonical migration (real-data findings)
- These are intentionally deferred from `legacy_compat` unless called out elsewhere as a compatibility repair. Fix them
  when artifacts, caches, and production models can move together under a versioned normalization contract.
- Title/text feature normalization is too destructive for some paper fields:
  - Real titles with formulas, identifiers, and enumerated parts collapse because `normalize_text(...)` drops digits and
    punctuation (`Co3O4`, `H2O2`, `CCDC 619488`, `Part 1`/`Part 2`).
  - Python locations to audit/change under a versioned feature contract:
    - Generic normalizer: `s2and/text.py::normalize_text`.
    - Paper preprocessing: `s2and/data.py::preprocess_paper_1`.
    - Incremental query/summary title and venue terms:
      `s2and/incremental_linking/query_adapter.py::_normalize_term_set`.
    - Any training/reference feature code that consumes normalized titles or title n-grams.
  - Rust locations to audit/change in the same release:
    - Generic compatibility normalizer: `s2and_rust/src/text_compat.rs::normalize_text_compat_from_map`.
    - Paper preprocessing and raw Arrow/JSON feature extraction paths that normalize titles, venues, journals,
      paper authors, or reference details before hashing/feature construction.
  - Do not change global `normalize_text(...)` in legacy mode. Introduce field-specific canonical title/venue
    normalization only with cache/version bumps and production-model validation.
- Name canonicalization needs a single versioned first/middle/last policy:
  - Python locations:
    - `s2and/text.py::split_first_middle_hyphen_aware` or its canonical replacement.
    - `s2and/data.py::ANDData.preprocess_signatures` and `ANDData._compute_signature_name_counts`.
    - `s2and/data.py::_canonicalize_last_for_counts` and `_lasts_equivalent_for_constraint`.
    - `s2and/text.py::first_names_name_compatible`.
    - `s2and/subblocking.py::signature_name_parts_for_subblocking`.
    - Pairwise/incremental consumers of `author_info_first_normalized`,
      `author_info_first_normalized_without_apostrophe`, and middle/last normalized fields.
  - Rust locations:
    - `s2and_rust/src/text_compat.rs::split_first_middle_hyphen_aware_compat`.
    - `s2and_rust/src/ingest_dataset.rs::build_name_counts_data_from_artifact`.
    - `s2and_rust/src/ingest_dataset.rs::canonical_last_for_counts`.
    - `s2and_rust/src/name_counts.rs::NameCountsLastFirstInitialSemantics` (Rust now supports both
      `legacy_full_first_token` and `initial_char` semantics; `InitialChar` is the default and matches
      existing canonical artifacts).
    - Rust constraint/name-tuple helpers and pairwise/incremental feature extraction paths that consume normalized
      first/middle/last values.
  - Compatibility repairs inside `legacy_compat` may keep current behavior correct, but canonical-only semantics must
    wait for regenerated name counts, name tuples, and ORCID prefix counts.
- Subblocking dash handling should not permanently encode ASCII/non-ASCII semantics:
  - Current repair is acceptable only as a localized `legacy_compat` quality repair.
  - The failed local-count adaptive key experiment shows that subblocking cannot safely choose one key from nearby
    spelling counts alone; large split cohorts can be semantically broader and noisier than the dashed compound cohort.
  - A cleaner near-term experiment can still be done before full canonical cutover if it keeps canonical dash semantics
    uniform while emitting compatibility aliases for subblocking merge/graph evidence.
  - Candidate design: one canonical key for all dash-like compounds, plus split aliases used only for merge candidates,
    prefix-count lookup, graph/co-location evidence, or a generated policy artifact when capacity constraints are
    satisfied.
  - Required evidence before replacing the current repair: full `s_lee`, `s_park`, and `h_wang` subblocking metrics must
    meet or beat the current repair; telemetry must not materially increase fallback invocations/signatures or final
    subblock fragmentation.
- `preprocess=False` is semantically misleading:
  - Today Python `s2and/data.py::preprocess_paper_1(..., preprocess=False)` still normalizes titles and authors,
    builds title word n-grams, and computes language for signature papers, while leaving venue/journal and some
    character n-gram fields raw/unset.
  - Rust stage/from-JSON paths intentionally mirror that behavior for parity.
  - During migration, replace the boolean with explicit modes such as `raw`, `minimal_legacy`, and `full`, or keep the
    legacy mode name explicit. Tests should assert exactly which fields are normalized in each mode.
- Subblock-token fallback parsing is case/punctuation preserving:
  - Python: `s2and/incremental_linking_training/query_support.py::_subblock_tokens`.
  - Rust: `s2and_rust/src/lib.rs::subblock_tokens_from_key`.
  - Generated current indexes appear to feed normalized keys, so this is not an observed generated-data failure.
    Canonical migration should either normalize parsed fallback tokens in both languages or fail fast on raw keys.
- Missing/non-informative text values collapse to empty strings:
  - `normalize_text(None)`, empty strings, digit-only strings, and punctuation-only strings can all become `""`.
  - During canonical migration, distinguish true missingness from normalized-empty nonmissing values where that matters
    for paper titles, venues, journals, and affiliation evidence. Any schema/cache change must be versioned.
- Source identifiers are not text:
  - `source_author_ids`, MAG IDs, DBLP suffixes, ACM IDs, and ORCIDs must never use `normalize_text(...)`.
  - Python locations carrying source IDs: `s2and/incremental_linking/feature_block_contract.py`,
    `scripts/arrow_conversion_helpers.py`, and
    `s2and/incremental_linking/feature_block_arrow.py`.
  - Rust raw Arrow/JSON contracts should preserve source IDs verbatim unless an identifier-specific canonicalizer is
    explicitly selected.

Target End State
- One canonical normalization path for first/middle/last consumed by all codepaths.
- No semantic distinction between `author_info_first_normalized` and `author_info_first_normalized_without_apostrophe`.
- Canonical last names are stored in spaced normalized form, with compact projections derived only for specific downstream keys.
- No runtime compatibility shims for legacy artifacts.
- All generated artifacts are built from the same canonical normalization logic.
- Field-specific text canonicalizers are explicit; title/venue/journal/source-ID behavior is not implicitly inherited
  from person-name normalization.

Migration Plan (phased, verifiable)
1) Lock policy and examples
   - Status 2026-07-04: the frozen table exists at `tests/fixtures/canonical_name_examples.json`
     (89 cases, decisions D1-D8 ruled), enforced by `tests/test_canonical_name_examples.py`
     (legacy pins run now; canonical contract activates when the step-2 functions land).
   - Resolve all Open Decisions above. (Done 2026-07-09; rulings recorded inline in the Open
     Decisions section.)
   - Freeze a canonical example table covering:
     - `Jo Ann`, `Jo-Ann`, `JoAnn`.
     - `Yu Zhong`, `Yu-Zhong`, `YuZhong`, `Y. Z.`.
     - ASCII and Unicode dash-equivalent forms: `Sang-Min`, every configured `NAME_DASH_CHARS`
       Unicode dash spelling for `Sang-Min`, `Sang Min`; `Qi-Xin`, `Qi<U+2010>Xin`, `Qi Xin`.
     - Apostrophe-like forms (`O'Brien`, ``O`Brien``, curly apostrophes, spacing acute,
       okina/modifier apostrophe, saltillo, primes, U+FE4D, and fullwidth apostrophe).
     - Multi-initial cases (`H. G.`-style), close-dotted initials, space-separated initials,
       joined initials, and dashed/spaced `J. P.` variants.
     - Missing/null first, middle, and last fields.
     - Apostrophe-as-space and joined variants (`O Brien`/`OBrien`, `D Angelo`/`DAngelo`).
     - Invisible formatting (`NBSP`, `U+00AD` soft hyphen, `U+200D` zero-width joiner).
     - Suffix/postnominal leakage in `last` (`Smith Jr.`, `Doe PhD`).
     - Surname dash/space variants (`Ou-Yang`, `Ou Yang`, `Ouyang`) and particle surnames,
       including dashed particles and joined particle spellings that remain single tokens.
     - Joined/spaced particle surname aliases (`de Souza`/`DeSouza`, `La Salle`/`LaSalle`).
     - A combined dash-plus-apostrophe case proving the policies compose.
   - Output: explicit normalization invariants used by tests and artifact builders.

2) Implement unified canonicalization
   - Status 2026-07-09: the canonicalization routine landed as
     `s2and.text.canonicalize_name_parts` (fields) + `s2and.text.canonical_name_count_keys`
     (gated count keys), pure functions with no live consumers. The canonical-contract
     layer of `tests/test_canonical_name_examples.py` now runs against them (all 89 cases).
     The D3 apostrophe-like set is `s2and.text.NAME_APOSTROPHE_LIKE_CHARS`; soft hyphen and
     zero-width joiner are deleted pre-tokenization per the pipeline spec.
   - Provide one canonicalization routine for first/middle/last (extend or replace `split_first_middle_hyphen_aware`). (Done, above.)
   - Remove dual-read usage of `author_info_*_normalized*` fields in featurizer/subblocking/constraints and standardize on canonical fields.
     (Cutover work — moves with regenerated artifacts + the v1.3 retrain, not before.)
   - Keep migration-scoped feature/version switch only if needed for safe rollout.

3) Regenerate artifacts with canonical logic
   - Regenerate name counts (`first`, `last`, `first_last`, `last_first_initial`).
   - Regenerate name tuples aligned with canonical forms.
   - Regenerate `s2and/data/first_k_letter_counts_from_orcid.json` using canonical first names (no token fallback).
   - Record reproducibility metadata: source snapshot, script/version hash, generation date.

4) Cut over and remove compatibility code
   - Remove `_canonicalize_last_for_counts`.
   - Remove `_lasts_equivalent_for_constraint`.
   - Remove name-tuple compatibility probing (joined/first-token fallback) from
     `first_names_name_compatible(...)`.
   - Remove subblocking first-token ORCID count probe.
   - Remove inference-only block compaction workaround once blocks are canonical everywhere.

5) Validate, benchmark, and roll out
   - Pairwise and clustering evaluation on representative datasets; compare to pinned baseline.
   - Re-baseline the pinned quality gates on regenerated artifacts: the `s_lee`/`s_park`/`h_wang`
     recall values above and the `eval_prod_models.py` docstring B3 baselines were measured on
     legacy-name artifacts and do not transfer to a new-name bundle. Protocol: score old code +
     old bundle and new code + new bundle each against its own gold labels and compare deltas;
     additionally run new code on the old bundle once to separate code-driven from data-driven
     movement.
   - Subblocking checks: size distribution, merge behavior, ORCID co-location sanity checks, and dash-variant alias
     behavior on `s_lee`, `s_park`, and `h_wang`.
   - Performance checks: runtime and memory for preprocessing/subblocking/featurization.
   - Cache/version bump as needed (featurizer cache and artifact versioning).

6) Rust canonical alignment track
   - Audit Rust ingestion paths against the frozen canonical example table before cutover.
   - Update Rust helpers or constructor policies only as needed for `canonical_v2` artifacts.
   - Verify parity against Python outputs while compatibility shims are still enabled, then add no-shim canonical tests before enabling canonical mode.

Required Evidence / Exit Criteria
- Behavior:
  - Targeted pytest coverage for canonical examples and no-shim paths.
  - Existing transitional tests replaced or updated for the end state.
- Quality:
  - No-regression thresholds for pairwise and clustering metrics are met.
- Quality thresholds (adopted for Rust alignment):
  - Pairwise: `AUC delta <= 0.001`, `F1 delta <= 0.005` versus pinned baseline.
  - Clustering: `B3 delta <= 0.005` versus pinned baseline.
  - Threshold scope (clarified 2026-07-04): these deltas gate no-op alignment/refactor
    comparisons on unchanged inputs. The retrain release gate is end-metric non-regression
    versus the shipped production model on the same eval sets; per-column drift tolerances do
    not apply to an intentional feature-changing retrain.
- Runtime:
  - No unexpected slowdown beyond agreed threshold.
- Runtime thresholds (adopted for Rust alignment):
  - Subblocking/preprocess runtime regression `<=10%` versus pinned baseline on the active benchmark protocol.
  - Peak RSS regression `<=10%` unless explicitly accepted for a release candidate.
- Data integrity:
  - Artifact generation logs include counts, key cardinalities, and basic spot checks.
- Versioning integrity:
  - Every regenerated artifact includes `normalization_version` metadata and generation provenance.
  - Code/artifact mismatch behavior is validated (fail-fast by default).

Compatibility/Rollback Notes
- Use explicit artifact normalization versioning during transition.
- Prefer fail-fast on code/artifact version mismatch unless a temporary compatibility flag is intentionally enabled.
- Decommission compatibility mode after one validated release window.
- Rust rollout note:
  - Treat any remaining Rust canonical-cutover work as a separate release action from legacy compatibility-shim removal.

References in code (as of this migration doc)
- Given-name canonicalization: `s2and.text.split_first_middle_hyphen_aware`.
- Rust compatibility implementation: `s2and_rust/src/text_compat.rs::split_first_middle_hyphen_aware_compat`.
- Subblocking legacy-compat first/middle key materialization:
  `s2and/subblocking.py::signature_name_parts_for_subblocking` and
  `s2and_rust/src/subblocking.rs::normalize_subblocking_signature_rows`.
- Surname count shim: `_canonicalize_last_for_counts` in `s2and/data.py`.
- Last-name constraint shim: `_lasts_equivalent_for_constraint` in `s2and/data.py`.
- Constraint and incremental new-name tuple fallback logic (exact/joined/first-token forms):
  `first_names_name_compatible(...)` in `s2and/text.py`, consumed by `ANDData.get_constraint`
  and incremental clustering guards.
- ORCID prefix fallback in subblocking: lookup path in `s2and/subblocking.py` during merge-pair scoring.
Tests (current)
- `tests/test_surname_hyphen_aware.py`
  - Transitional regression coverage for surname count canonicalization, last-name constraint equivalence,
    and name-tuple compatibility forms.
- `tests/test_cluster_incremental.py`
  - Transitional regression coverage that incremental new-name guarding accepts the same legacy
    name-tuple compatibility forms as constraints.

Tests (required for end state)
- Canonical first-name equivalence cases from the frozen example table.
- Canonical surname policy tests for spaced storage + compact projection sites.
- Tests proving removal of compatibility fallbacks does not break expected behavior with regenerated artifacts.

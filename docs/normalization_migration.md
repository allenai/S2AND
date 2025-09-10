Normalization Unification Migration Plan

Scope
- Unify name normalization for first/middle/last across data preparation, modeling, subblocking, and auxiliary datasets (name counts, name tuples, ORCID prefix counts).

Current State (post-Sinonym hyphen pass)
- Canonical first/middle preserve hyphenated Chinese given names:
  - Implemented via `s2and.text.split_first_middle_hyphen_aware`.
- Backward-compat shims for artifacts built with legacy normalization (to be removed):
  - Name counts (first): when raw first had a hyphen, join spaces in canonical first for count keys (e.g., "qi xin" → "qixin").
  - Name counts (last): join spaces in canonical last for compound/hyphenated surnames (e.g., "ou yang" → "ouyang").
    - Helpers: `_canonicalize_last_for_counts(...)`.
  - Constraints: last-name disallow uses a space-insensitive comparison ("ou yang" == "ouyang").
    - Helper: `_lasts_equivalent_for_constraint(...)`.
  - Subblocking: temporary ORCID prefix map probe uses the first token from canonical first when multi-token.
  - Sinonym overwrite gating (optional, off by default unless invoked):
    - Compute allowlist per normalized name using multi-author evidence priority:
      - If multi-author evidence exists: overwrite when flips x satisfy `x >= min_ratio * y` (not-flips).
      - Else (single-author only): overwrite when any flip evidence exists (a > 0).
  - Blocks on Sinonym overwrite (inference-only): recompute `author_info_block` as `first_initial + compact_surname` where
    compact_surname removes spaces/hyphens (e.g., `q ouyang`) to keep compound surnames atomic for blocking.

Target State
- Single, unified normalization for names (apostrophes always stripped; hyphen/space variants treated consistently; Sinonym for Chinese names keeping given-name tokens together; surname handling consistent with given names).
- Remove the distinction between `author_info_first_normalized` and `author_info_first_normalized_without_apostrophe`; use a single canonical field consumed everywhere (features, constraints, counts, tuples, subblocking).
- No special-case shims for counts or constraints; artifacts regenerated to match canonicalization.

Steps
1) Finalize normalization policy
   - Apostrophes: always remove (no replacement with spaces in canonical fields).
   - Hyphen/compound names: treat hyphen and space variants equivalently in canonicalization. For Chinese given names, keep tokens together per Sinonym; for surnames, adopt consistent joining/preservation policy (see 4).
   - Language-agnostic defaults: ensure non-Chinese names remain unaffected other than consistent punctuation handling.

2) Implement unified canonicalization
   - Provide a single canonicalization path for first/middle/last (extends `split_first_middle_hyphen_aware` to surnames or replaces with a unified routine).
   - Replace all reads/writes of `author_info_*_normalized*` with the unified canonical fields.

3) Regenerate external artifacts
   - Name counts: regenerate with unified canonicalization (first, last, first_last, last_first_initial).
   - Name tuples: regenerate with canonical forms aligned to the unified logic.
   - ORCID prefix counts: regenerate `first_k_letter_counts_from_orcid.json` using canonical first names (no first-token fallback).

4) Blocks and downstream features
   - Define block computation on canonical names (first-initial + surname). Decide whether canonical surname should be joined (preferred) or space-separated; update `compute_block` and Sinonym overwrite code accordingly.
   - Remove inference-only surname compaction once blocks universally use canonicalization.

5) Remove compatibility shims
   - Drop `_canonicalize_last_for_counts`, `_lasts_equivalent_for_constraint`, and related TODOs.
   - Remove the subblocking ORCID first-token probe.

6) Validation
   - Run pairwise and clustering evaluations on representative datasets; compare against baseline.
   - Inspect subblocks and merges; verify Chinese/Western hyphenated cases perform as expected.
   - Rebuild caches as needed (featurizer cache keyed by version may need a bump).

Rollback/Compat Notes
- Keep a feature flag or version switch temporarily to load and operate on legacy artifacts during transition.

References in code (as of this migration doc)
- Given-name canonicalization: `s2and.text.split_first_middle_hyphen_aware`.
- Surname compat for counts: `_canonicalize_last_for_counts` in `s2and/data.py`.
- Last-name constraint shim: `_lasts_equivalent_for_constraint` in `s2and/data.py`.
- ORCID prefix fallback in subblocking: comment and lookup in `s2and/subblocking.py` near the counts probe.
- Sinonym overwrite gating and application: `compute_sinonym_overwrite_allowlist`, `apply_sinonym_overwrites` in `s2and/data.py`.

Tests added for regression
- `tests/test_surname_hyphen_aware.py`
  - Validates surname count canonicalization, last-name constraint equivalence, and block compaction behavior under Sinonym overwrites.


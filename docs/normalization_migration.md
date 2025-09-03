Normalization Unification Migration Plan

Scope
- Unify name normalization for first/middle/last across data preparation, modeling, subblocking, and auxiliary datasets (name counts, name tuples, ORCID prefix counts).

Current State (post-hyphen fix)
- Canonical fields in runtime (used by featurizer/model/subblocking) preserve hyphenated first names:
  - Implemented via `s2and.text.split_first_middle_hyphen_aware`.
- Legacy fields for counts/tuples remain single-token:
  - `author_info_first_normalized` stays single-token for compatibility with existing name counts and name tuples.
- ORCID prefix map compatibility fallback:
  - Subblocking probes `FIRST_K_LETTER_COUNTS` using the first token when canonical first contains spaces.

Target State
- Single, unified normalization for names (apostrophes always stripped; hyphen variants normalized; Sinonym wired for Chinese names to keep given names together).
- Remove the distinction between `author_info_first_normalized` and `author_info_first_normalized_without_apostrophe` throughout the codebase.

Steps
1) Decide normalization policy
   - Always strip apostrophes to nothing (handle typographic variants).
   - Normalize hyphen/dash variants consistently.
   - For Chinese names, use Sinonym to keep given-name tokens together; confirm no regression on prod model.

2) Implement unified normalizer
   - Update `s2and.text.normalize_text` and/or replace usages with a single canonical path.
   - Deprecate `special_case_apostrophes` and `split_first_middle_hyphen_aware` once a single path exists.

3) Regenerate data artifacts with the new normalization
   - Name counts: rerun `get_name_counts.py`.
   - Name tuples: write/adjust a script to use `s2and_unnormalized_filtered_name_tuples.txt` from raw tuples using the new normalization.
   - ORCID prefix counts: rewrite `scripts/get_orcid_name_prefix_counts.py` to call the unified logic; regenerate `data/first_k_letter_counts_from_orcid.json`.

4) Code cleanup and renames
   - Replace usages of `author_info_first_normalized_without_apostrophe` with the unified canonical field.
   - Remove `author_info_first_normalized` or alias it to the canonical field (depending on migration strategy).
   - Remove the temporary first-token fallback in `s2and/subblocking.py` for ORCID lookups.

5) Validation
   - Run clustering metrics and pairwise evaluation on representative datasets.
   - Check subblock sizes/distributions and merge logs for anomalies.
   - Spot-check Chinese and Western hyphenated names for expected behavior.

Rollback/Compat Notes
- Keep a feature flag or version switch if needed to load legacy datasets during transition.


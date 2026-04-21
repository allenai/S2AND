# Normalization Unification Migration Plan

Execution status (2026-03-02)
- Blocked: normalization work is on hold until the required data/artifacts are ready.
- Keep this plan separate from the active execution plan in `docs/work_plan.md`.
- When unblocked, landing Bundle 5 (artifact format unification) first reduces churn so Phase 3 regeneration
  can target MessagePack/Safetensors once instead of regenerating twice.

Status
- Draft updated from issue notes through August 31, 2025.
- Rust Phase 1B alignment update added on February 20, 2026.
- Reviewed on February 24, 2026 alongside big-block execution planning; no normalization-policy changes in that workstream.

Scope
- Unify name normalization for first/middle/last across data preparation, modeling, subblocking, and auxiliary datasets (name counts, name tuples, ORCID prefix counts).
- Ensure training-time and inference-time normalization are identical, including Sinonym-dependent behavior.

Decided (from issue history)
- Apostrophes: canonical fields should remove apostrophes globally (no dual stream).
- Chinese given names: Sinonym-aware handling is part of canonicalization; hyphenated given names should stay together.
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
- Retraining requirement: production retraining data must go through the same normalization + Sinonym path as production inference.

Open Decisions (remaining before migration freeze)
1) Compatibility-mode decommission window
   - finalize the exact number of releases/runs required before removing compatibility mode.
2) Threshold tightening
   - decide whether adopted Rust-alignment thresholds should be tightened for release-grade datasets.

Rust Alignment Decisions (effective February 20, 2026)
1) Two-stage rollout contract
   - Stage A (compatibility-preserving): Rust ports of normalization helpers must match current Python + compatibility-shim behavior.
   - Stage B (canonical cutover): switch to canonical-only behavior only after canonical artifacts are regenerated and versioned.
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
   - Before enabling canonical mode by default, production retraining and production inference must use the same canonical normalization + Sinonym path.
5) Rust coupling
   - Rust `from_json_paths` migration in must treat normalization helper ports as policy-sensitive changes, not pure performance refactors.

Current State (post-Sinonym hyphen pass)
- Given-name canonicalization currently preserves hyphenated Chinese given names:
  - `s2and.text.split_first_middle_hyphen_aware`.
- Backward-compat shims exist for artifacts built with legacy normalization:
  - Name counts (first): when raw first had a hyphen, join spaces in canonical first for count keys (e.g., `qi xin` -> `qixin`).
  - Name counts (last): join spaces in canonical last for compound/hyphenated surnames (e.g., `ou yang` -> `ouyang`).
    - Helper: `_canonicalize_last_for_counts(...)`.
  - Constraints: last-name disallow uses space-insensitive comparison (`ou yang` == `ouyang`).
    - Helper: `_lasts_equivalent_for_constraint(...)`.
  - Subblocking: ORCID prefix map lookup has a first-token fallback for multi-token first names.
  - Name tuples in constraints: alias logic probes exact, joined, and first-token forms for compatibility with legacy tuples.
  - Sinonym overwrite block recomputation compacts surnames for blocking (`q ouyang`) when overwriting blocks.

Target End State
- One canonical normalization path for first/middle/last consumed by all codepaths.
- No semantic distinction between `author_info_first_normalized` and `author_info_first_normalized_without_apostrophe`.
- Canonical last names are stored in spaced normalized form, with compact projections derived only for specific downstream keys.
- No runtime compatibility shims for legacy artifacts.
- All generated artifacts are built from the same canonical normalization logic.

Migration Plan (phased, verifiable)
1) Lock policy and examples
   - Resolve all Open Decisions above.
   - Freeze a canonical example table covering:
     - `Jo Ann`, `Jo-Ann`, `JoAnn`.
     - `Yu Zhong`, `Yu-Zhong`, `YuZhong`, `Y. Z.`.
     - Apostrophe-like forms (`O'Brien`, ``O`Brien``, curly apostrophes).
     - Multi-initial cases (`H. G.`-style).
   - Output: explicit normalization invariants used by tests and artifact builders.

2) Implement unified canonicalization
   - Provide one canonicalization routine for first/middle/last (extend or replace `split_first_middle_hyphen_aware`).
   - Remove dual-read usage of `author_info_*_normalized*` fields in featurizer/subblocking/constraints and standardize on canonical fields.
   - Keep migration-scoped feature/version switch only if needed for safe rollout.

3) Regenerate artifacts with canonical logic
   - Regenerate name counts (`first`, `last`, `first_last`, `last_first_initial`).
   - Regenerate name tuples aligned with canonical forms.
   - Regenerate `s2and/data/first_k_letter_counts_from_orcid.json` using canonical first names (no token fallback).
   - Record reproducibility metadata: source snapshot, script/version hash, generation date.

4) Cut over and remove compatibility code
   - Remove `_canonicalize_last_for_counts`.
   - Remove `_lasts_equivalent_for_constraint`.
   - Remove name-tuple compatibility probing (joined/first-token fallback) in constraints.
   - Remove subblocking first-token ORCID count probe.
   - Remove inference-only block compaction workaround once blocks are canonical everywhere.

5) Validate, benchmark, and roll out
   - Pairwise and clustering evaluation on representative datasets; compare to pinned baseline.
   - Subblocking checks: size distribution, merge behavior, and ORCID co-location sanity checks.
   - Performance checks: runtime and memory for preprocessing/subblocking/featurization.
   - Cache/version bump as needed (featurizer cache and artifact versioning).

6) Rust ingestion alignment track (required for Phase 1B)
   - Port `normalize_text`, `split_first_middle_hyphen_aware`, and `compute_block` to Rust in compatibility-preserving mode first.
   - Verify parity against Python outputs while compatibility shims are still enabled.
   - After canonical artifacts/regeneration/retraining gates pass, enable canonical mode and remove compatibility shims.

Required Evidence / Exit Criteria
- Behavior:
  - Targeted pytest coverage for canonical examples and no-shim paths.
  - Existing transitional tests replaced or updated for the end state.
- Quality:
  - No-regression thresholds for pairwise and clustering metrics are met.
- Quality thresholds (adopted for Rust alignment):
  - Pairwise: `AUC delta <= 0.001`, `F1 delta <= 0.005` versus pinned baseline.
  - Clustering: `B3 delta <= 0.005` versus pinned baseline.
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
  - Treat Stage A (compatibility-preserving Rust helper ports) and Stage B (canonical cutover) as separate release actions.

References in code (as of this migration doc)
- Given-name canonicalization: `s2and.text.split_first_middle_hyphen_aware`.
- Surname count shim: `_canonicalize_last_for_counts` in `s2and/data.py`.
- Last-name constraint shim: `_lasts_equivalent_for_constraint` in `s2and/data.py`.
- Constraint tuple fallback logic (exact/joined/first-token forms): `ANDData.get_constraint` in `s2and/data.py`.
- ORCID prefix fallback in subblocking: lookup path in `s2and/subblocking.py` during merge-pair scoring.
- Sinonym overwrite gating/application: `compute_sinonym_overwrite_allowlist`, `apply_sinonym_overwrites` in `s2and/data.py`.

Tests (current)
- `tests/test_surname_hyphen_aware.py`
  - Transitional regression coverage for surname count canonicalization, last-name constraint equivalence, and block compaction behavior under Sinonym overwrites.

Tests (required for end state)
- Canonical first-name equivalence cases from the frozen example table.
- Canonical surname policy tests for spaced storage + compact projection sites.
- Tests proving removal of compatibility fallbacks does not break expected behavior with regenerated artifacts.

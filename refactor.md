# S2AND Single-Letter / Joint-Safe-Link Pipeline ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Refactor Proposal

Audit scope: retrieval, row/feature/label generation, joint-safe-link "official stack",
train/eval/calibration, and core constraint/feature behavior in `s2and/data.py` and
`s2and/feature_port.py`. This is a proposal ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â no code changes yet.

---

## 1. Executive summary

The pipeline mostly works and has good test coverage, but it has accumulated four
classes of debt that are now causing real correctness risk for the ORCID-labels-but-no-ORCID-features
product requirement:

1. **One real, still-live ORCID leak into features ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â through both Python and Rust constraint paths.**
   [s2and/data.py:1467-1470](s2and/data.py#L1467-L1470) makes `ANDData.get_constraint()`
   return `low_value` (the "must-link, distanceÃƒÂ¢Ã¢â‚¬Â°Ã‹â€ 0" constraint value) when both
   signatures share an ORCID. The Rust constraint engine has the *same* short-circuit at
   [s2and_rust/src/lib.rs:3413-3417](s2and_rust/src/lib.rs#L3413-L3417). The reranker's
   constraint backend uses the Rust path by default
   ([s2and/model.py:970-986](s2and/model.py#L970-L986),
   [build_single_letter_reranker_dataset.py:1480](scripts/build_single_letter_reranker_dataset.py#L1480),
   [rebuild_joint_safe_link_official_stack.py:1101](scripts/rebuild_joint_safe_link_official_stack.py#L1101)).
   The constraint label flows through `_predict_and_combine` at
   [s2and/model.py:484-488](s2and/model.py#L484-L488) (`predictions[not_predict_flag] = labels[not_predict_flag] + LARGE_INTEGER`)
   and is aggregated into `min_distance`, `mean_distance`, and the top-k distance
   features at [single_letter_reranker_utils.py:2511-2516](scripts/single_letter_reranker_utils.py#L2511-L2516).
   Because ANDData query/candidate signatures retain raw ORCID even when
   `orcid_enabled=False` is set at the *retrieval* layer, any candidate signature
   that happens to share an ORCID with the query gets a forced near-0 distance.
   **Fixing this requires a backend-level ORCID-suppression flag with parity in
   both Python and Rust ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a Python-side wrapper alone is bypassed by the Rust path.**

2. **Two parallel pipelines that mostly do the same thing.**
   `build_single_letter_reranker_dataset.py` (2,299 lines) and
   `rebuild_joint_safe_link_official_stack.py` (3,489 lines) re-implement the same
   end-to-end flow (load dataset ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ retrieve ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ score pairwise ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ assemble rows ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ write
   CSV) with subtle behavioral differences. Each has its own
   `_raw_similarity_features_by_component` ([build:649](scripts/build_single_letter_reranker_dataset.py#L649),
   [rebuild:1290](scripts/rebuild_joint_safe_link_official_stack.py#L1290)) and its own
   prepared-group struct. Bug fixes have to be ported by hand and have already drifted.

3. **Bypass *predicate* is correct, bypass *scope* is still positive-keyed.**
   `_query_case_allows_seed_constraint_bypass` ([single_letter_reranker_utils.py:1151-1158](scripts/single_letter_reranker_utils.py#L1151-L1158))
   correctly gates on `query_in_seed_before_holdout`/LOO/self-containing predicates ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â
   that part of the previously-known bug is fixed. But the call sites still pass
   `eligible_component_keys=positive_component_keys` ([build:1286](scripts/build_single_letter_reranker_dataset.py#L1286),
   [rebuild:1930](scripts/rebuild_joint_safe_link_official_stack.py#L1930)),
   which means the bypass *only ever fires on label-positive components*. For an LOO
   query, that is consistent with intent; for "self-containing positive removed via
   LOO" cases on negatives or near-positives, this couples bypass scope back to labels
   in a subtle way that's hard to reason about.

4. **Ranker training and official gate calibration use disconnected surfaces.**
   `eval_single_letter_ranker.py` does not consume the bundle's calibration surface.
   The official stack does use `classic_gate_source` for gate calibration, but that
   is a separate path with potentially different filters and feature columns, and it
   is not schema-stamped against the trained ranker's feature schema. Any future
   isotonic/Platt layer in the ranker would need an explicit shared schema contract
   or it could silently fit on a non-aligned surface.

There are also several MEDIUM issues ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â divergent default hybrid weights in three
places, single-letter name features returning all-NaN while the constraint side
accepts them, and `runpy.run_path` + `sys.path` shenanigans that hide import errors.

The **good news**: there is no ORCID feature directly in the trained feature surface
(no preset references ORCID, no `retrieval_top1_rank` exists in the codebase, no gold
cluster id appears in any preset). `resolve_feature_columns` at
[single_letter_reranker_utils.py:2910-2912](scripts/single_letter_reranker_utils.py#L2910-L2912)
also rejects any unknown numeric column at training time, so accidental ORCID-as-feature
inclusion would crash, not silently propagate. The leak is structural ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â through the
constraint backend ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â not through a feature column.

The fix is bounded but **not** trivial: it requires an ORCID-suppression flag plumbed
through the constraint backend in *both* Python and Rust, with cached-featurizer policy tests
and parity tests. The h_wang ORCID label channel is preserved separately via
`positive_component_keys` ([build_single_letter_reranker_dataset.py:903-905](scripts/build_single_letter_reranker_dataset.py#L903-L905) ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢
[single_letter_reranker_utils.py:2402-2407](scripts/single_letter_reranker_utils.py#L2402-L2407)),
so suppressing ORCID at the feature-constraint layer does not erase positive labels.

**Recommended first implementation phase** is Phase 1 below: add a backend-level
`suppress_orcid` flag to the constraint engine (Python + Rust parity), route the
reranker's constraint backend through it, and verify with a regression test that
ORCID positive label counts are unchanged.

---

## 2. Bug report

Severity scale:
- **Blocker** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â silently wrong labels, leakage, or model corruption.
- **High** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â silently wrong features or breaks reproducibility.
- **Medium** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â drift, brittleness, observability.
- **Low** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â code smell, maintenance debt.

| # | Severity | Area | Where | What | Effect |
|---|---|---|---|---|---|
| B1 | **Blocker** | ConstraintÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢Feature leak (Python *and* Rust) | [s2and/data.py:1467-1470](s2and/data.py#L1467-L1470); [s2and_rust/src/lib.rs:3413-3417](s2and_rust/src/lib.rs#L3413-L3417); routed via [s2and/model.py:970-986](s2and/model.py#L970-L986) and `_predict_and_combine` at [s2and/model.py:484-488](s2and/model.py#L484-L488) | Both Python `get_constraint()` and the Rust constraint engine return `low_value` when two signatures share an ORCID. The Rust path is the default for the reranker's constraint backend (built from raw dataset at [build_single_letter_reranker_dataset.py:1480](scripts/build_single_letter_reranker_dataset.py#L1480) and [rebuild_joint_safe_link_official_stack.py:1101](scripts/rebuild_joint_safe_link_official_stack.py#L1101)). The constraint label is converted to a distance via `predictions = labels + LARGE_INTEGER` and aggregated into `min_distance` / `mean_distance` / `topk_distance_*` features at [single_letter_reranker_utils.py:2511-2516](scripts/single_letter_reranker_utils.py#L2511-L2516). | Same-ORCID candidate components show feature-perfect distances. The model learns "this looks easy"; at inference (no query ORCID) it is not. Calibration is biased by the same effect. **Fix must land in Python *and* Rust with explicit cached-featurizer policy tests.** |
| B2 | **Blocker** | Pipeline forking | [build_single_letter_reranker_dataset.py](scripts/build_single_letter_reranker_dataset.py) vs [rebuild_joint_safe_link_official_stack.py](scripts/rebuild_joint_safe_link_official_stack.py) | Two implementations of the same end-to-end row-generation flow (`_raw_similarity_features_by_component`, `_prepared_group`, `_rebuild_group`). Fix in one path does not propagate to the other. Promote/strict-bundle scripts ([promote_name_compat_eval_rows.py](scripts/promote_name_compat_eval_rows.py), [assemble_strict_name_compat_surface_bundle.py](scripts/assemble_strict_name_compat_surface_bundle.py)) overlay a *third* surface. | Strict rows can have features that don't match the trained model's surface; behavior varies by which entry point ran. The "old strict rows missing raw metadata similarity" bug was an instance of this. |
| H1 | High | Train/eval split | [eval_single_letter_ranker.py:813-828](scripts/eval_single_letter_ranker.py#L813-L828) | Inner hyperparameter split is `GroupShuffleSplit` on `base_group_id = "{dataset}:{query_source}:{query_id}"`. For h_wang, the same physical author can produce many `query_id`s that share an ORCID and shared candidate components; these end up split across train/validation. | Hyperparameter validation is optimistic; selected hyperparameters do not generalize. |
| H2 | High | Calibration | [eval_single_letter_ranker.py](scripts/eval_single_letter_ranker.py); [assemble_strict_name_compat_surface_bundle.py:696-708](scripts/assemble_strict_name_compat_surface_bundle.py#L696-L708); [joint_safe_link_official_stack.py:2211-2237](scripts/joint_safe_link_official_stack.py#L2211-L2237) | Bundle defines `classic_gate_source_path` and the official stack uses it for classic gate calibration/eval, but the ranker trainer never reads that surface. There is no shared held-out calibration fold, no ranker/calibrator schema digest, and no calibration step in `eval_single_letter_ranker.py`. | The existing gate calibration can drift from the ranker surface. If a ranker-level calibrator is added without re-aligning surfaces, it could silently fit on a different feature schema or on training data. |
| H3 | High | Constraint bypass scope | [build_single_letter_reranker_dataset.py:1286](scripts/build_single_letter_reranker_dataset.py#L1286), [rebuild_joint_safe_link_official_stack.py:1930](scripts/rebuild_joint_safe_link_official_stack.py#L1930) | `seed_constraint_bypass_component_keys(eligible_component_keys=positive_component_keys, ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦)`. The predicate is correct (LOO/self/seed), but the scope is "only positives". Combined with B1 this means: ORCID positives that fall under LOO get bypass, but ORCID negatives caused by `low_value` short-circuit don't. | Asymmetric: positives are correctly held out, but ORCID-induced false-perfect features on negatives stand. |
| H5 | High | Raw similarity zeroing | [single_letter_reranker_utils.py:2421-2426](scripts/single_letter_reranker_utils.py#L2421-L2426) | `raw_similarity_features = raw_similarity_features_by_component.get(str(component_key), {})`. If the dict is empty (e.g. caller forgot to compute), per-row features quietly default to 0.0, which is a strong "no overlap" signal. | Silent feature degradation. The previously-noted "strict rows zeroing useful evidence" bug was caused by exactly this code path being hit with empty input. |
| M11 | Medium | ORCID metadata persisted in row CSVs (was H4) | [single_letter_reranker_utils.py:2471-2479](scripts/single_letter_reranker_utils.py#L2471-L2479) | Phase 2 renames row/query-group CSV metadata to `_audit_normalized_orcid`, `_audit_orcid_group_size`, `_audit_orcid_group_size_bucket`. **`resolve_feature_columns` at [2910-2912](scripts/single_letter_reranker_utils.py#L2910-L2912) raises on any column not in `NUMERIC_FEATURE_COLUMNS`, so accidental inclusion would crash, not silently propagate.** | Hygiene only. Future-bug-magnet if the trainer's column-discovery logic is ever loosened. |
| M2 | Medium | Default weights split-brain | [eval_cluster_retrieval.py:482-487](scripts/eval_cluster_retrieval.py#L482-L487), [s2and_rust/src/lib.rs:242-249](s2and_rust/src/lib.rs#L242-L249), [single_letter_retrieval_utils.py:169-176](scripts/single_letter_retrieval_utils.py#L169-L176) | Three separate magic-number weight tuples for the hybrid centroid scorer. | Easy to leave a path on stale weights when tuning. |
| M3 | Medium | Single-letter feature/constraint divergence | [s2and/text.py:431](s2and/text.py#L431) returns NaN when either name has length ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¤1; [s2and/data.py:1479](s2and/data.py#L1479) accepts ("A", "Alice") via prefix match | Same pair ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ constraint-allowed but features = NaN. | Featurizer reports "no info" while constraint reports "compatible". Single-letter rows lose evidence the constraint engine considers usable. |
| M4 | Medium | `min_distance` self-containing confound | [single_letter_reranker_utils.py:1099, 2511](scripts/single_letter_reranker_utils.py#L1099) | When a candidate component contains the query signature itself (LOO residual), pairwise distance to self ÃƒÂ¢Ã¢â‚¬Â°Ã‹â€  0 unless filtered out. The current `_drop_self_containing_rows` filter runs in promote_*, not in build_single_letter_reranker_dataset_train_path. | Train rows can include self-containing candidates with `min_distanceÃƒÂ¢Ã¢â‚¬Â°Ã‹â€ 0`, biasing the model. |
| M5 | Medium | `runpy.run_path` masking | [build_joint_safe_link_official_stack.py:13-25](scripts/build_joint_safe_link_official_stack.py#L13-L25) | Wrapper does `sys.path.insert` + `runpy.run_path` for the rebuild script. Import errors surface as cryptic runpy traceback. | Slow root-cause analysis; encourages forking instead of importing. |
| M6 | Medium | Mid-rebuild KeyError on missing decisions | [rebuild_joint_safe_link_official_stack.py:631, 736](scripts/rebuild_joint_safe_link_official_stack.py#L631) | `KeyError(f"S2AND relabel decision missing for active query_group_id={query_group_id!r}")` raised mid-stream after spool DB is partially populated. | Restart loop, partial-bundle risk if not noticed. |
| M7 | Medium | Validator silent fallback | [validate_joint_safe_link_official_stack.py:653](scripts/validate_joint_safe_link_official_stack.py#L653) | `_summarize_active_feature_coverage(summarize_unmatched_rows_as=ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦)` falls back to summarizing the entire file when the dataset filter matches 0 rows. | Misspelled dataset name ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ validator passes silently on unrelated data. |
| M9 | Medium | Silent Rust/Python name-compat fallback | [single_letter_reranker_utils.py:963-975](scripts/single_letter_reranker_utils.py#L963-L975) | Rust selector failure silently falls back to Python (unless `S2AND_STRICT_RUST_NAME_COMPAT=1`). | Different candidate sets in CI vs prod; hard to audit which path ran. |
| M10 | Medium | Year handling brittleness | [promote_name_compat_eval_rows.py:484-496](scripts/promote_name_compat_eval_rows.py#L484-L496); year_compat in single_letter_reranker_utils | Blank `candidate_year_min/max` coerce to NaN; `_year_compatibility` returns 0.0, biasing toward "incompatible". | Subtle bias on candidates with missing year metadata. |
| L1 | Low | `orcid_enabled` default | [eval_cluster_retrieval.py:265](scripts/eval_cluster_retrieval.py#L265) | Helper-function default is `True`. **The CLI default is `--orcid-mode disabled` at [eval_cluster_retrieval.py:1259](scripts/eval_cluster_retrieval.py#L1259), so production runs are safe.** Risk is only a future caller that imports the helper directly without overriding. | Tighten helper default to match CLI default. |
| L2 | Low | Magic constants in scoring | `-0.25` middle-initial penalty, `Ãƒâ€šÃ‚Â±10` year window, `0.42/0.23/0.12/0.05/0.07` weights. | Not exported as constants; live in code in 2-3 places each. | Debug/tune friction. |
| L3 | Low | Dual import path | [build_single_letter_reranker_dataset.py:20-100](scripts/build_single_letter_reranker_dataset.py#L20-L100) | `try: import scripts.foo as foo / except: import foo as foo`. | Direct script execution and `python -m` produce different import graphs. |
| L4 | Low | Compatibility shims | [s2and/data.py:1471-1488](s2and/data.py#L1471-L1488) | Two TODOs ("hyphen/space-insensitive shim", "revisit once we re-extract name_tuples") that are constraint-correctness shims for legacy data. | Fine, but should be tracked. |

**Removed from this audit after review:**
- ~~H4 (ORCID metadata persisted)~~ ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ downgraded to **M11**: trainer's `resolve_feature_columns` rejects unknown columns, so accidental feature inclusion crashes rather than silently propagating.
- ~~H6 (retrieval ORCID asymmetry)~~ ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ downgraded to **L1**: CLI default is already `--orcid-mode disabled`. The helper-function default is the only remaining concern.
- ~~M1 (Python `len(str)` vs Rust `chars().count()`)~~ ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ **withdrawn**. Python 3 `len(str)` *is* Unicode scalar count, same semantics as Rust `chars().count()`. Verified: `python -c "print(len('ÃƒÆ’Ã‚Â©'))"` ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ `1` for the precomposed scalar. The original audit claim was based on a Python-2 era understanding.
- ~~M8 (deferred n-gram crash risk)~~ ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ **withdrawn**. [s2and/feature_port.py:774-805](s2and/feature_port.py#L774-L805) has explicit retry-via-`materialize_signature_ngrams_python` on the deferred-ngrams build failure, so the originally-claimed unhandled crash path no longer exists.

### Cheap verifications

For each major claim, here is a check the reviewer can run.

| Claim | Check |
|---|---|
| B1 (Python path) | Construct two synthetic signatures with `author_info_orcid='0000'` each, otherwise dissimilar. Call `dataset.get_constraint(s1, s2)` and assert the return value equals `low_value`. |
| B1 (Rust path) | Same pair, same dataset. Build the Rust constraint backend via `_build_incremental_constraint_backend(dataset, ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦)`. Call `get_constraints_matrix_indexed_rust` on the (s1,s2) pair and assert it also returns `low_value`. Then run the pair through `compute_query_cluster_stats` and confirm `min_distance` is the constraint short-circuit value, not the featurizer-derived distance. |
| B1 (full) | After applying the Phase-1 fix, repeat both checks with `suppress_orcid=True` on the backend; expect the constraint to fall through to last-name / first-initial / name-tuple logic instead of short-circuiting on ORCID. |
| B2 | `diff <(grep -n "^def \|^class " scripts/build_single_letter_reranker_dataset.py) <(grep -n "^def \|^class " scripts/rebuild_joint_safe_link_official_stack.py)` to surface the parallel APIs. |
| H1 | In `_fit_ranker_for_split`, dump `inner_train_ids` and `inner_validation_ids` and assert no two `query_id`s in opposite splits share `_audit_normalized_orcid`. |
| H2 | `python -c "import json; b=json.load(open('.../bundle.json')); print(b.get('classic_gate_source_path'))"` then `grep classic_gate_source eval_single_letter_ranker.py` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â current grep should return nothing. |
| H5 | Construct a `make_candidate_rows` call with `raw_similarity_features_by_component={}`; confirm the four `raw_max_*_jaccard` columns in output rows are 0.0. |
| M11 | `grep -rn "normalized_orcid\|orcid_group_size" scripts/ tests/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â list of consumers (currently 8 files). Each must be updated atomically when the rename happens. |
| M3 | `python -c "from s2and.text import name_text_features; print(name_text_features('A','Alice'))"` should return all-NaN; `dataset.get_constraint(...)` on the same pair returns None (allowed). |
| M9 | `S2AND_STRICT_RUST_NAME_COMPAT=0 uv run pytest tests/test_rust_hybrid_centroid_retriever.py -q -k name_compat` and compare candidate lists when forcing Python fallback. |

---

## 3. Proposed architecture

### 3.1 Stage contracts

The pipeline should have five stages, each with a small frozen contract. Naming
deliberately uses `Plan`/`View`/`Row`/`Frame` to make the surface boundaries explicit.

```
                  ÃƒÂ¢Ã¢â‚¬ÂÃ…â€™ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â
                  ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡ ANDData (ground truth, including ORCID)                       ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡
                  ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‹Å“
                                 ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡ ORCID-suppressed constraint backend (Python+Rust)
                                 ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¼
            ÃƒÂ¢Ã¢â‚¬ÂÃ…â€™ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â
   1. PLAN  ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡ QueryPlan: signature_id, query_view, masked QueryFeatures   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡  (no labels)
            ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‹Å“
                           ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¼
            ÃƒÂ¢Ã¢â‚¬ÂÃ…â€™ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â
   2. RETR  ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡ RetrievalResult: ordered list[ComponentKey, score, rank,    ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡  (Rust)
            ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   strategy_branch ÃƒÂ¢Ã‹â€ Ã‹â€  {same_block,name_compat,global_backfill}ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡
            ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‹Å“
                           ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¼
            ÃƒÂ¢Ã¢â‚¬ÂÃ…â€™ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â
   3. PAIR  ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡ PairwiseStats: per-component min/mean/topk distance,        ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡  (Rust+Python)
            ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   require_pair_count, disallow_pair_count                   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡  via suppress_orcid backend
            ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‹Å“
                           ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¼
            ÃƒÂ¢Ã¢â‚¬ÂÃ…â€™ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â
   4. LABEL ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡ LabelDecision: positive_component_keys (from gold + ORCID), ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡  (Python, label-only
            ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   bypass_component_keys (LOO/seed), self_containing_keys    ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ANDData read)
            ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‹Å“
                           ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¼
            ÃƒÂ¢Ã¢â‚¬ÂÃ…â€™ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â
   5. ROW   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡ RerankerRow: feature columns + label + audit metadata       ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡
            ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‚Â¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ‹Å“
                           ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¼
                    parquet/csv frames ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ trainer / calibrator / evaluator
```

Hard rules:

- **Stage 3 (PAIR) must read from a constraint backend constructed with `suppress_orcid=True`.**
  That backend is the single enforcement point for the no-query-ORCID-into-features
  requirement.
- **Stage 4 (LABEL) reads from raw ANDData**, including ORCID, gold clusters, seed
  metadata. This is the *only* stage allowed to look at labels. Positive labels for
  ORCID datasets (e.g. h_wang) are derived from `seed_cluster_counts_by_orcid` *before*
  any constraint resolution, then carried through `positive_component_keys`
  ([build:903-905](scripts/build_single_letter_reranker_dataset.py#L903-L905)) ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢
  row label ([utils:2402-2407](scripts/single_letter_reranker_utils.py#L2402-L2407)).
- **No stage may pass label evidence into the feature surface.** The trainer can
  only ingest columns declared in a `FeatureSchema` registry.

### 3.2 The ORCID-suppression primitive (Python + Rust)

The fix has to land at the constraint backend, not at a Python wrapper, because the
default path resolves constraints via the Rust featurizer
([s2and/model.py:970-986](s2and/model.py#L970-L986)).

```python
# Python side: s2and/data.py
class ANDData:
    def get_constraint(
        self,
        sid_1: str,
        sid_2: str,
        *,
        suppress_orcid: bool = False,                      # NEW
        dont_merge_cluster_seeds: bool = ...,
        incremental_dont_use_cluster_seeds: bool = ...,
    ) -> float | None:
        ...
        # at line 1469, gate the short-circuit:
        # elif (not suppress_orcid) and orcid_1 is not None and orcid_2 is not None and orcid_1 == orcid_2:
        #     return low_value
        ...
```

```rust
// Rust side: s2and_rust/src/lib.rs
// Prefer a per-call flag on every constraint API rather than storing policy on
// RustFeaturizer. The same cached featurizer can then serve both policy states.
fn constraint_value_from_records(
    ...,
    suppress_orcid: bool,                                  // NEW
) -> Option<f64> {
    ...
    // at line 3413, gate the short-circuit:
    // if !suppress_orcid && o1 == o2 { return Some(low_value); }
}
```

```python
# Construction site: s2and/model.py and s2and/feature_port.py
def _build_incremental_constraint_backend(
    dataset, *, suppress_orcid: bool = False, ...,         # NEW
):
    ...
    # Python backend object stores the policy and forwards it per constraint call.
```

```python
# Caller: scripts/reranker_dataset/pairwise.py (was build_single_letter_reranker_dataset.py:1480
# and rebuild_joint_safe_link_official_stack.py:1101)
constraint_backend = _build_incremental_constraint_backend(
    dataset,
    suppress_orcid=True,                                   # the ONE policy decision
    use_default_constraints_as_supervision=...,
    runtime_context=runtime_context,
    use_cache=clusterer.use_cache,
)
```

Critical implementation details:

- **The backend is per-dataset, not per-query.** The simplification "suppress ORCID
  *symmetrically* for the entire reranker dataset build" is acceptable because:
  - Positive labels are derived *before* the backend is constructed
    (via `positive_component_keys` from `seed_cluster_counts_by_orcid`), so label
    fidelity is preserved.
  - The model is for the no-query-ORCID case; no candidate pair in the feature
    surface should benefit from ORCID equality.
  - Per-pair query-side-only redaction would require threading the query signature
    id through the Rust constraint API, which is a larger surface change.

- **Cache semantics are mandatory.** The current Rust featurizer cache is keyed by
  dataset object in memory and by dataset/artifact metadata on disk; it is not keyed
  by constraint policy. Therefore the preferred implementation is a **per-call**
  `suppress_orcid` argument on `get_constraint`, `get_constraints_matrix`,
  `get_constraints_matrix_indexed`, and `get_constraints_block_upper_triangle_indexed`.
  The Python `_IncrementalConstraintBackend` carries the policy and forwards it on
  every call. If an implementation instead stores `suppress_orcid` on `RustFeaturizer`,
  then both the in-memory cache key and disk metadata in `s2and/feature_port.py`
  must include the flag, and persisted Rust artifacts need serde defaults / a schema
  bump so old artifacts cannot mask the new behavior.

- **Parity tests cover both paths.** The Phase 1 PR ships unit tests against
  Python `get_constraint()` and against `get_constraints_matrix_indexed_rust()` to
  prevent the two paths from drifting in the future.

### 3.3 Single row-generation engine

Replace the parallel `build_single_letter_reranker_dataset.py` and the row-generating
core of `rebuild_joint_safe_link_official_stack.py` with one engine:

```
scripts/reranker_dataset/
    __init__.py
    plan.py        # Stage 1: QueryPlan construction from ANDData (+ holdout policy)
    retrieve.py    # Stage 2: thin Python wrapper on Rust retrieval
    pairwise.py    # Stage 3: PairwiseStats via suppress_orcid=True backend
    labels.py      # Stage 4: LabelDecision (the only stage that touches labels)
    rows.py        # Stage 5: RerankerRow assembly + FeatureSchema enforcement
    schema.py      # FeatureSchema, FeaturePreset (single source of truth)
    bundle.py      # write/read the bundle artifact (replaces assemble_*)
    cli.py         # one CLI entry point with subcommands
```

`rebuild_joint_safe_link_official_stack.py`'s job collapses to:
- read source bundle config,
- call `cli.py build` per dataset,
- apply the dataset-contract filter ledger,
- write spool/manifests.

### 3.4 Where Rust owns logic

Rust should be authoritative for:
- All retrieval scoring (not fully true today; Phase 3a makes this a real migration
  task).
- Name-compatibility prefix/alias check (currently still has a Python re-impl in
  [single_letter_reranker_utils.py:916-930](scripts/single_letter_reranker_utils.py#L916-L930)).
- Default hybrid weights (single source of truth as Rust constant; Python imports it).
- Year window threshold, middle-initial penalty (Rust constants).
- First-name length semantics (`py_len`, Rust-side `chars().count()`).

Python should retain:
- Stage 1 (plan), Stage 4 (labels), Stage 5 (row assembly) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â all involve
  bundle / contract / scheduling logic that does not need to be fast.
- The trainer (`eval_single_letter_ranker`) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â sklearn/lightgbm world.

### 3.5 Calibration becomes first-class

The trainer in Stage 5 produces:
1. The fitted ranker.
2. A held-out calibration fold (additional `GroupShuffleSplit` on ORCID/`query_id`,
   not just `base_group_id`) with the *same feature schema* as training.
3. An isotonic / Platt calibrator fit only on that fold, persisted alongside the
   ranker with the `FeatureSchema` digest stamped in.
4. An eval pass that asserts schema-digest match between calibrator, ranker, and
   eval rows, and crashes loudly on mismatch.

The bundle's `classic_gate_source_path` either becomes the shared held-out
calibration fold consumed by the trainer, or remains explicitly stamped as a
classic-gate-only surface with its own schema digest. No unstamped third surface.

### 3.6 Feature schema registry

One module owns:
- `FEATURE_PRESETS` (today: 15 entries at [single_letter_reranker_utils.py:377-393](scripts/single_letter_reranker_utils.py#L377-L393))
- `FEATURE_PRESET_DIGEST` (sha256 of (preset, column list, derivation versions))
- The contract that **active features must come from this registry**. Any column
  written to rows that is not in any preset is "audit-only" and gets a standard
  `_audit_` prefix.

The trainer already does the right thing here: `resolve_feature_columns` at
[single_letter_reranker_utils.py:2910-2912](scripts/single_letter_reranker_utils.py#L2910-L2912)
raises `ValueError` on any column not in `NUMERIC_FEATURE_COLUMNS`. So accidental
ORCID-as-feature inclusion crashes loudly today. The `_audit_` prefix would be
defense-in-depth, not a strict requirement.

### 3.7 Speed and repeat-work elimination

The refactor should make later dataset generation cheaper, not just cleaner.

- **Persistent pairwise-stats cache.** Persist `PairwiseStats` / query-to-component
  distance aggregates keyed by dataset artifact digest, model/featurizer digest,
  `suppress_orcid`, query signature id, ordered candidate signature ids, constraint
  flags, feature preset, and prediction contract version. This avoids recomputing
  the same query-to-candidate distances and model predictions every time a new
  train/calibration/eval slice is generated from the same source data.
- **Content-addressed reusable artifacts.** Persist component summaries, retrieval
  subblock indexes, raw-paper token caches, raw-similarity inputs, and Rust
  featurizer artifacts under content-addressed keys. Dataset builds should reuse
  these artifacts across training, calibration, eval, and future regenerated
  bundles when the underlying ANDData/artifact digests are unchanged.
- **Collapse row engines before adding new feature work.** A speed fix in one row
  engine is not a real speed fix while `build_single_letter_reranker_dataset.py`,
  `rebuild_joint_safe_link_official_stack.py`, and promotion/strict-surface scripts
  can each recompute or zero a different surface.
- **Move retrieval filtering/scoring into one Rust call.** Python should pass the
  query view, candidate strategy, hard-filter policy, and rank budget to Rust, then
  receive ranked component ids plus telemetry. This removes repeated Python loops
  around scoring variants and makes name-compat/global-backfill behavior auditable
  in one place.
- **Keep the multi-query pair-batching path.** The batched path is the performance
  path. If no production caller needs the single-query stats API after Phase 4, make
  it a test-only compatibility wrapper or delete it and retire tests that only pin
  single-query-vs-batched equivalence.

---

## 4. Migration phases

Each phase is independently shippable, has a rollback, and a verification gate.

### Phase 1 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â `suppress_orcid` flag on the constraint backend (the B1 fix)

**Goal:** Eliminate ORCID-into-features leak with parity in Python and Rust paths.
**This is the only correctness fix in this PR.** No rename, no refactor ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â those are
follow-on phases.

- **Files touched:**
  - `s2and/data.py`: add a `suppress_orcid: bool = False` keyword to `ANDData.get_constraint`.
    Gate the short-circuit at [s2and/data.py:1467-1470](s2and/data.py#L1467-L1470) on
    `not suppress_orcid`.
  - `s2and_rust/src/lib.rs`: add a per-call `suppress_orcid: bool` argument to the
    Rust constraint APIs and to the internal `constraint_value_from_records` path.
    Gate the short-circuit at [s2and_rust/src/lib.rs:3413-3417](s2and_rust/src/lib.rs#L3413-L3417)
    on `!suppress_orcid`. Thread the flag through `get_constraint`,
    `get_constraints_matrix`, `get_constraints_matrix_indexed`, and
    `get_constraints_block_upper_triangle_indexed`.
  - `s2and/feature_port.py` and `s2and/model.py`: add `suppress_orcid` to
    `_IncrementalConstraintBackend`, thread it through the Python wrappers around
    the Rust API, and pass it through `_resolve_constraint_labels_batch` at
    [s2and/model.py:925-1010](s2and/model.py#L925-L1010).
  - `s2and/feature_port.py`: add a regression test that the same cached
    `RustFeaturizer` object can return different constraint values for
    `suppress_orcid=False` and `suppress_orcid=True`. If the implementation stores
    the policy on `RustFeaturizer` instead of passing it per call, then the in-memory
    cache key and disk metadata must include the flag.
  - `scripts/single_letter_reranker_utils.py` and the two row-engine call sites:
    pass `suppress_orcid=True` when constructing the reranker's constraint backend
    at [build:1480](scripts/build_single_letter_reranker_dataset.py#L1480) and
    [rebuild:1101](scripts/rebuild_joint_safe_link_official_stack.py#L1101).
  - `tests/test_feature_safe_view.py` (new): parity tests for both paths, plus a
    regression test that ORCID positive label counts are unchanged.

- **Behavioral change:** All pairwise distances computed for the reranker no longer
  short-circuit to `low_value` on ORCID equality. Other constraints (last name,
  first initial, name tuples, middle conflicts) still apply. Positive labels for
  h_wang are unchanged because they come from `positive_component_keys` *before*
  constraint resolution.

- **Scope decision (open question 1 in Ãƒâ€šÃ‚Â§6 below):** the flag suppresses ORCID
  symmetrically for the entire reranker dataset build. Per-pair "query side only"
  redaction would require threading the query signature id through the Rust
  constraint API, a substantially larger surface change. The symmetric flag is
  adequate because the model is for the no-query-ORCID case generally, and label
  derivation does not go through this backend.

- **Rollback:** the call site can be reverted to `suppress_orcid=False` to restore
  the old behavior. The Python and Rust default for the flag stays `False`, so the
  rest of S2AND (the labeled-data clusterer) is unaffected. No env-var rollback shim
  needed.

- **Verification:**
  1. **Python parity test.** Build an `ANDData` with two signatures that share an
     ORCID. `dataset.get_constraint(s1, s2)` returns `low_value`;
     `dataset.get_constraint(s1, s2, suppress_orcid=True)` falls through to the
     last-name / first-initial / name-tuple chain.
  2. **Rust parity test.** Same pair. Reuse the same Rust featurizer and call
     `get_constraints_matrix_indexed_rust` once with `suppress_orcid=False` and
     once with `suppress_orcid=True`; assert the same difference as Python. Also
     assert that a cached featurizer does not pin the first policy state. If a
     featurizer-level policy is chosen instead, assert the cache key differs between
     the two flag states.
  3. **End-to-end snapshot test.** Take a 50-query slice from a labeled dataset
     where some queries have ORCID. Regenerate rows under old behavior and under
     `suppress_orcid=True`. Diff `min_distance` distribution per row; every changed
     row must be attributable to at least one query/candidate same-ORCID pair, and
     every same-ORCID pair must avoid the ORCID sentinel unless a non-ORCID require
     rule independently applies.
  4. **Label-preservation regression test.** On the same slice, count rows with
     `label==1` per query before and after the fix. Counts must be identical
     (positive label channel goes through `positive_component_keys`, not constraints).
  5. **Tiny-fixture training smoke.** Train on a tiny fixture; eval AUC should not
     collapse. Some metrics may shift modestly because ORCID-equal pairs no longer
     show feature-perfect distances.

### Phase 2 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â ORCID metadata column rename (M11) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â separate ask-first migration

**Status:** Implemented after explicit approval. This is a serialization-contract
change: generated row/query-group CSVs and supported `query_set.json` metadata use
`_audit_normalized_orcid`, `_audit_orcid_group_size`, and
`_audit_orcid_group_size_bucket`. Source-domain ORCID normalization and internal
`RerankerQueryCase` fields keep their existing names.

- **Goal:** Rename to `_audit_*` prefix to signal "audit metadata, not features."
- **Files touched:** row/query-group schema, row writers/readers, supported query-set
  metadata readers, rebuild readers, and tests. Pinned/generated bundle artifacts are
  not rewritten by this code change.
- **Behavioral change:** None for the model. Bundle CSVs change column names.
- **Rollback:** Rename back; coordinate across all eight consumers.
- **Verification:** `grep -nE "(normalized_orcid|orcid_group_size)" scripts/ tests/ s2and/`
  ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ no hits outside the renamed sites; `grep _audit_normalized_orcid` ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ present in
  the same set.
- **Implementation note:** legacy names still appear in source-domain/internal query-case
  code and in negative assertions; generated row/query-group/query-set artifact
  columns use `_audit_*`.
- **Defense-in-depth note:** the trainer's `resolve_feature_columns` already rejects
  unknown numeric columns ([single_letter_reranker_utils.py:2910-2912](scripts/single_letter_reranker_utils.py#L2910-L2912)),
  so this rename is hygiene, not a correctness fix.

### Phase 3 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Unify default scoring constants (M2)

**Goal:** One source of truth for hybrid weights, year threshold, middle-initial
penalty.

- **Files touched:** `s2and_rust/src/lib.rs` (export constants), Python wrappers in
  `single_letter_retrieval_utils.py` and `eval_cluster_retrieval.py` import from Rust.
- **Behavioral change:** Default-weight sites all read the same numbers; no value
  changes (the three current tuples should be identical at the default site, and any
  policy override sites stay explicit).
- **Rollback:** Per-call-site reverts; constants are additive.
- **Verification:** Diff retrieval candidate sets on a held slice; expect zero
  change (all sites used the same numbers; we are deduplicating the *source* only).

### Phase 3a - Rust-owned retrieval scoring surface

**Goal:** Make "Rust owns retrieval scoring" true, not aspirational.

- **Files touched:** `s2and_rust/src/lib.rs`, `scripts/single_letter_retrieval_utils.py`,
  `scripts/single_letter_reranker_utils.py`, `scripts/eval_cluster_retrieval.py`,
  and Rust retrieval tests.
- **Behavioral change:** Python no longer contains authoritative hybrid-centroid
  scoring, hard-filter ranking variants, name-compat/global-backfill scoring, or
  duplicate default constants. Python builds query plans and reads Rust-ranked
  component ids, scores, ranks, and telemetry.
- **Rollback:** Keep a temporary `--retrieval-engine=python|rust` or equivalent
  only while parity is being proven. Delete the Python engine once candidate-set
  parity and telemetry checks pass.
- **Verification:**
  1. Tiny fixture: Python legacy vs Rust engine candidate ids/scores/ranks match
     for full, initial-only, name-compat, and global-backfill strategies.
  2. Real 200-query slice: compare recall@K and candidate-count distributions;
     expected delta is zero unless an intentionally approved bug fix is included.
  3. Telemetry contract: Rust returns same-subblock/global-backfill counts and
     hard-filter counts for every query.
  4. Delete or quarantine Python-only scoring tests after the Rust contract tests
     cover the same behavior.

### Phase 3b - Persistent reusable artifacts for speed

**Goal:** Eliminate repeat work as more datasets and bundles are generated.

- **Files touched:** new artifact/cache module under `scripts/reranker_dataset/`,
  `single_letter_reranker_utils.py` during migration, and bundle metadata.
- **Behavioral change:** Row generation reuses content-addressed component
  summaries, retrieval subblock indexes, raw token caches, raw-similarity inputs,
  Rust featurizers, and pairwise-stats aggregates when their input digests match.
- **Rollback:** Caches are optional and guarded by explicit `--reuse-artifacts` /
  `--no-reuse-artifacts` controls; disabling cache returns to recomputation.
- **Verification:**
  1. Cold vs warm tiny-fixture run: same rows byte-for-byte, warm run logs cache hits.
  2. Cache invalidation test: changing dataset digest, model digest,
     `suppress_orcid`, feature preset, constraint flags, or candidate signature ids
     forces recomputation.
  3. Warm real-slice smoke: report wall-clock delta and cache-hit counts.

### Phase 4 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Single row engine (B2 collapse)

**Goal:** Replace parallel implementations with one. This is the largest phase but
is mechanical.

- **Files touched:** new `scripts/reranker_dataset/`, refactor
  `rebuild_joint_safe_link_official_stack.py` to depend on it, deprecate
  `build_single_letter_reranker_dataset.py` to a thin wrapper (for one release),
  delete the wrapper after Phase 5.
- **Behavioral change:** The bridge mode must be bit-identical for current bundles,
  modulo normalized feature column ordering. After parity is proven, the canonical
  unified engine may intentionally differ for approved fixes (ORCID suppression,
  bypass scope, missing-year handling, self-containment filtering). Those deltas
  must be recorded in a migration manifest, not hidden in the refactor.
- **Rollback:** Both old scripts kept side-by-side until parity is proven; flip a
  `--engine=legacy|unified` flag.
- **Verification:**
  1. Generate a tiny labeled-dataset bundle with both engines in bridge mode;
     row-by-row diff on all columns. Allowed delta: 0.
  2. Run the existing `tests/test_joint_safe_link_official_stack.py` and
     `tests/test_single_letter_reranker.py` against unified engine.
  3. Validator (`validate_joint_safe_link_official_stack.py`) passes with same
     report in bridge mode; approved canonical-mode behavior changes have explicit
     migration-manifest entries and focused regression tests.

### Phase 5 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Calibration as first-class (H2)

**Goal:** Ranker training and official gate calibration share an explicit held-out
surface contract, and any ranker-level calibrator persists a `FeatureSchema` digest.

- **Files touched:** `eval_single_letter_ranker.py`, `bundle.py` (Phase 4 module),
  `validate_joint_safe_link_official_stack.py` (validate calibrator schema).
- **Behavioral change:** The existing `classic_gate_source` path is either promoted
  into the shared held-out calibration contract or explicitly kept as a classic-gate
  only surface. Any new ranker-level artifacts (`calibrator.joblib`,
  `feature_schema.json`) live next to the ranker, and eval asserts schema match.
- **Rollback:** Calibrator is optional; ranker still produces uncalibrated scores.
- **Verification:**
  1. Pretend-corrupt the feature schema digest; eval crashes loudly.
  2. Train calibrator on training data instead of held-out ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ reliability diagram
     shows worse Brier; document the regression so it's caught in CI.
  3. Compare calibrated probabilities to current bundle's `classic_gate_source`
     metrics, and assert the classic-gate surface and ranker calibration surface
     either share a schema digest or are explicitly stamped as separate surfaces.

### Phase 6 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Bypass scope cleanup (H3)

**Goal:** Decouple `seed_constraint_bypass_component_keys` scope from
`positive_component_keys`. The eligible scope should be "any component that has
a seed connection to the query", regardless of label.

- **Files touched:** `scripts/single_letter_reranker_utils.py`, both call sites
  (build / rebuild).
- **Behavioral change:** Bypass may now apply to negative components too. In
  practice this affects only LOO setups; unit tests will pin the new behavior.
- **Rollback:** Pass `eligible_component_keys=ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦positive_component_keys` again.
- **Verification:** New unit test: LOO query where the only seed connection is to
  a label-negative component; assert bypass fires under new logic, doesn't under
  old.

### Phase 7 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Train/eval split tightening (H1)

**Goal:** GroupShuffleSplit on `(dataset, _audit_normalized_orcid OR query_id)` instead of
`base_group_id` only.

- **Files touched:** `eval_single_letter_ranker.py`.
- **Behavioral change:** Inner-validation set is smaller for h_wang because
  same-ORCID queries are kept together.
- **Rollback:** Trivial revert.
- **Verification:** Hyperopt re-run on tiny fixture; report old-vs-new AUC. Expect
  validation AUC to drop (the previous number was optimistic). Primary eval metric
  on held-out test set should not change much.

### Phase 8 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Cleanup & deletions

After all of the above stabilizes:
- Delete `build_joint_safe_link_official_stack.py` (the runpy wrapper).
- Delete the legacy `build_single_letter_reranker_dataset.py` once Phase 4 unified
  engine is canonical.
- Delete `assemble_strict_name_compat_surface_bundle.py` and
  `promote_name_compat_eval_rows.py`; their behavior is subsumed by the unified
  engine + bundle module.
- Delete or demote the single-query pairwise stats API once all callers use the
  multi-query batched path.
- Delete legacy tests listed in Ã‚Â§5.8 only after their replacement contract tests
  have landed.
- Quarantine columns now redundantly named `retrieval_top1_*` if any survive
  (currently `retrieval_top1_rank` doesn't exist; verified by grep).

### Status of old scripts after migration

| Script | Disposition |
|---|---|
| `build_single_letter_reranker_dataset.py` | **Delete after Phase 4.** Replaced by `scripts/reranker_dataset/`. |
| `build_joint_safe_link_official_stack.py` | **Delete in Phase 8.** Was just a `runpy.run_path` wrapper. |
| `rebuild_joint_safe_link_official_stack.py` | **Keep**, but rewritten on top of the unified engine. ~3,000 lines should drop to ~1,000. |
| `joint_safe_link_official_stack.py` | **Keep** (config / decisions library). |
| `joint_safe_link_dataset_contract.py` | **Keep** (pure helpers). |
| `compile_joint_safe_link_dataset_contract.py` | **Keep** (entry point for ledger refresh). |
| `validate_joint_safe_link_official_stack.py` | **Keep**, extend with schema-digest checks. |
| `run_joint_safe_link_official_classic.py` | **Keep** (replay/verification). |
| `sync_joint_safe_link_official_bundle_metadata.py` | **Keep**. |
| `joint_safe_link_initial_only_rereview.py` | **Keep** (decision library). |
| `assemble_strict_name_compat_surface_bundle.py` | **Delete in Phase 8** (subsumed by bundle module). |
| `promote_name_compat_eval_rows.py` | **Delete in Phase 8**. |
| `single_letter_retrieval_utils.py` | **Keep**, but slim down to a thin wrapper over Rust. |
| `single_letter_reranker_utils.py` | **Keep**, but split into the modules under `scripts/reranker_dataset/`. |
| `eval_single_letter_ranker.py` | **Keep**, extend with calibration. |
| `eval_cluster_retrieval.py` | **Keep** (retrieval pilot tool). |
| `giant_block_cluster_retrieval_task.py` | **Keep** (uses retrieval utils). |

### Status of generated columns

| Column | Disposition |
|---|---|
| `normalized_orcid` | **Renamed** to `_audit_normalized_orcid` in row/query-group/query-set artifacts (Phase 2). |
| `orcid_group_size`, `orcid_group_size_bucket` | **Renamed** to `_audit_orcid_group_size`, `_audit_orcid_group_size_bucket` in row/query-group/query-set artifacts (Phase 2). |
| `min_distance`, `mean_distance`, top-k distances | **Keep**, but recomputed under `suppress_orcid=True` constraint backend (Phase 1). |
| `raw_max_*_jaccard` | **Keep**. Engine must always populate them; Phase 4 enforces. |
| `retrieval_top1_rank` | **Confirmed absent** ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â no action. |
| `candidate_year_min/max` | **Keep**, but `_year_compatibility` must explicitly handle missing-year as a separate signal column rather than collapsing to 0.0 (M10). |

---

## 5. Verification matrix

### 5.1 Unit / contract tests

| Test | What it pins | New or existing |
|---|---|---|
| `test_get_constraint_suppress_orcid_python` | `suppress_orcid=True` makes Python `ANDData.get_constraint` skip the ORCID short-circuit; default behavior unchanged. | New (Phase 1) |
| `test_get_constraint_suppress_orcid_rust_parity` | Rust constraint backend behaves identically under both flag values. | New (Phase 1) |
| `test_cached_rust_featurizer_respects_suppress_orcid_per_call` | Reusing a cached Rust featurizer does not pin the first `suppress_orcid` policy state. | New (Phase 1) |
| `test_orcid_positive_label_count_unchanged` | On a fixture with ORCID positives, total positive label count is identical with and without `suppress_orcid=True`. | New (Phase 1) |
| `test_other_constraints_intact_under_suppress_orcid` | Last-name mismatch, first-initial mismatch, and middle-conflict pairs still return `high_value` under `suppress_orcid=True`. | New (Phase 1) |
| `test_audit_columns_not_in_any_preset` | No `FeaturePreset` references an `_audit_*` column. | New (Phase 2) |
| `test_feature_schema_digest_stable` | Digest changes iff column list changes. | New (Phase 5) |
| `test_calibrator_schema_digest_match` | Eval crashes on mismatch. | New (Phase 5) |
| `test_unified_engine_byte_parity` | Tiny fixture under unified engine == legacy on all columns. | New (Phase 4, deleted in Phase 8) |
| `test_seed_bypass_scope_negative_components` | Under H3 fix, bypass eligible scope is independent of label sign. | New (Phase 6) |
| `test_split_no_orcid_overlap` | Inner-train and inner-validation share no `_audit_normalized_orcid` for h_wang. | New (Phase 7) |

### 5.2 Tiny-fixture integration tests

- One labeled dataset (e.g. `qian` or its tiny test slice), 20 queries, full pipeline
  end-to-end (plan ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ retrieve ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ pair ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ label ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ row ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ train ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ eval). Wall time
  budget < 2 minutes. Asserts row schema, label distribution, bound on
  `min_distance==0` row count.

### 5.3 Small real-slice smoke tests (require approval)

- 200 h_wang queries, full pipeline. Compare AUC and `cluster_b3_f1` against a
  pinned baseline. Tolerance Ãƒâ€šÃ‚Â±0.5 absolute on AUC.
- Same on a `s_park` slice for single-letter coverage.

### 5.4 Leakage checks (CI-level, every PR)

- For every row in train/eval CSV: `is_orcid_label_positive` is allowed; no row may
  have an `_audit_*` column referenced by any `FEATURE_PRESETS` member.
- For every fitted ranker: feature names must intersect with `FEATURE_PRESETS`
  exactly; extra/missing ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ fail (already enforced today by `resolve_feature_columns`).
- For every (query, candidate) pair where both signatures share an ORCID: recompute
  constraints with and without `suppress_orcid=True`. With suppression enabled, the
  pair must not return the ORCID short-circuit sentinel unless a non-ORCID require
  rule independently applies (for example an explicit cluster seed require). Row-level
  `min_distance` changes must be attributable to these pair-level differences.

### 5.5 Feature audits (CI-level)

- No NaN columns in row CSVs (or NaN-only allowed in explicitly-quarantined
  audit columns).
- No zero-variance columns in the active feature surface.
- All-finite assertion on training matrix.
- Coverage check: every active feature is non-zero on at least 5% of rows.

### 5.6 Retrieval audits

- Recall@K against gold cluster: per-dataset, with and without ORCID-on retrieval.
  Document the gap, expect ORCID-off to be lower (this is expected; it's the model's
  actual operating regime).
- Same-subblock vs global-backfill telemetry: `pct_of_top1_from_subblock`,
  `pct_global_backfill_used`. Trend over time.
- Candidate-count distribution per query: median, p99, max.

### 5.7 Train/eval/calibration checks

- Inner split policy is stable across train and eval; assert via stamped split JSON.
- No train rows in eval set (`base_group_id` set intersection = ÃƒÂ¢Ã‹â€ Ã¢â‚¬Â¦).
- Calibrator was fit on rows whose `(dataset, query_id)` does not appear in train.
- Calibrator's reliability diagram has expected-vs-actual gap < 0.05 in at least
  the top three quintiles.

### 5.8 Tests to keep vs deprecate during the refactor

Deprecate tests only after a replacement contract test exists and has passed on a
tiny fixture.

**Keep / strengthen:**
- Rust/Python constraint parity tests, including ORCID suppression and cached
  featurizer policy behavior.
- Feature-schema tests (`resolve_feature_columns`, feature preset membership,
  schema digest, no active `_audit_*` features).
- Self-containment validation tests and residual/required-positive exceptions.
- Calibration split/schema tests once Phase 5 exists.
- Batched pairwise scoring tests, since this is the performance path.

**Deprecate after replacement:**
- `test_unified_engine_byte_parity`: keep only during Phase 4 bridge mode; delete
  once the unified engine is canonical and approved migration deltas are pinned by
  focused tests.
- `test_promote_name_compat_*` tests in `tests/test_joint_safe_link_official_stack.py`:
  retire after `promote_name_compat_eval_rows.py` is deleted and equivalent
  label-decision / row-engine tests exist.
- `assemble_strict_name_compat_surface_bundle.py` tests: retire after bundle
  assembly moves into `scripts/reranker_dataset/bundle.py` and schema-digest tests
  cover the same behavior.
- Legacy CLI parser/default tests for `build_single_letter_reranker_dataset.py`:
  retire when it becomes a thin wrapper or is deleted.
- Single-query-vs-batched stats equivalence tests: retire if the single-query API
  is removed or made a test-only compatibility wrapper. Keep direct tests for the
  batched path.

---

## 6. Open questions

1. **Symmetric vs per-pair `suppress_orcid`.** Current proposal: the flag suppresses
   ORCID *symmetrically* on the constraint backend for the entire reranker dataset
   build (both sides of every pair, all pairs). Per-pair "query-side only" redaction
   would require threading the query signature id through the Rust constraint API,
   a substantially larger surface change. Symmetric is simpler, and arguably more
   conservative for the no-query-ORCID model. Confirm symmetric is acceptable.

2. ~~**ORCID-as-label preservation.**~~ **Resolved.** h_wang positives are derived
   from `seed_cluster_counts_by_orcid` *before* the constraint backend is built,
   then carried through `positive_component_keys` ([build_single_letter_reranker_dataset.py:903-905](scripts/build_single_letter_reranker_dataset.py#L903-L905))
   to the row label at [single_letter_reranker_utils.py:2402-2407](scripts/single_letter_reranker_utils.py#L2402-L2407).
   `suppress_orcid=True` on the constraint backend does *not* erase this label
   channel. Phase 1 ships with a regression test
   (`test_orcid_positive_label_count_unchanged`) that pins this behavior.

3. **Calibration on what fold?** Held-out from training (cleanest, but smaller
   train) vs cross-validation in-fold (messier, but more train data). The current
   bundle suggests an explicit `classic_gate_source` of manually-corrected rows;
   should those be reused as the calibration fold, or replaced entirely?

4. **Single-letter feature/constraint divergence (M3).** Should the featurizer
   compute non-NaN features for `("A", "Alice")` pairs, or should the constraint
   side return `high_value` to match the featurizer? Either direction works; the
   current asymmetry is the bug.

5. **Rust ownership of name-compat alias logic.** The Python re-implementation
   exists as a fallback. Is the goal "make Rust authoritative and remove Python
   fallback (with strict-fallback flag)" or "keep Python as a debuggable
   reference and add a parity test"? My proposal is the former for correctness;
   confirm.

6. **Time/cost budget.** Phase 4 alone is ~2 weeks of careful work to land
   without behavioral drift, and Phases 3a/3b add real Rust-retrieval and artifact
   reuse work. Is the team OK with a longer migration in exchange for the speed and
   correctness guarantees, or should we ship Phase 1 plus Phase 7 as a hotfix while
   scheduling Phase 2, 3a, 3b, and 4 separately?

7. **Should we delete `build_joint_safe_link_official_stack.py` immediately
   (Phase 0)?** It's a 25-line `runpy.run_path` wrapper that adds no value
   beyond hardcoded args. Dropping it now is independent of Phase 1.

---

## 7. Recommended first implementation phase

**Phase 1 alone, shipped as a single focused PR.**

Why Phase 1 is on its own (revised after review):

- Phase 1 is the only correctness fix in this batch ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â it eliminates B1, the one
  true blocker. The fix is bigger than originally estimated: a `suppress_orcid`
  flag with parity in Python and Rust, threaded through the constraint backend
  and constraint API, plus cache-policy tests on both paths. Estimate ~300-500
  lines of code + tests, not "30 lines + one call site".
- Phase 2 (column rename to `_audit_*`) is **not bundled in**. It is a
  serialization-contract change touching eight files, including bundle artifacts
  consumed downstream. Per CLAUDE.md "Ask-first triggers" it needs explicit
  approval before scheduling. The trainer's
  [`resolve_feature_columns`](scripts/single_letter_reranker_utils.py#L2910-L2912)
  already rejects unknown numeric columns, so the Phase 2 rename is hygiene, not
  a correctness fix ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â it can wait without leaving the model exposed.
- Verification fits on tiny fixtures plus a 200-query h_wang slice; no need for
  full-pipeline runs.

Concrete plan for the Phase 1 PR:

1. Add `suppress_orcid: bool = False` keyword to `ANDData.get_constraint` in
   `s2and/data.py`. Gate the short-circuit at
   [s2and/data.py:1467-1470](s2and/data.py#L1467-L1470). Default behavior unchanged.
2. Add a per-call `suppress_orcid: bool = false` argument to the Rust constraint
   APIs and internal constraint resolver. Gate the short-circuit at
   [s2and_rust/src/lib.rs:3413-3417](s2and_rust/src/lib.rs#L3413-L3417). Prefer
   this over a `RustFeaturizer` field so the existing cached featurizer can serve
   both policy states.
3. Thread the flag through the Python wrappers around the Rust constraint API
   (`get_constraints_matrix_indexed_rust`, `_resolve_constraint_labels_batch` at
   [s2and/model.py:925-1010](s2and/model.py#L925-L1010), and the
   `_build_incremental_constraint_backend` factory).
4. At the two row-engine call sites
   ([build_single_letter_reranker_dataset.py:1480](scripts/build_single_letter_reranker_dataset.py#L1480),
   [rebuild_joint_safe_link_official_stack.py:1101](scripts/rebuild_joint_safe_link_official_stack.py#L1101)),
   pass `suppress_orcid=True`.
5. Add `tests/test_feature_safe_view.py` with the five unit tests in Ãƒâ€šÃ‚Â§5.1
   (Python parity, Rust parity, cached-featurizer policy behavior, label-count
   preservation, other-constraints-intact).
6. Run the existing test suite and the Rust parity tests; expect no regressions
   in the labeled-data clusterer (which uses the default `suppress_orcid=False`).
7. Regenerate a tiny labeled-dataset bundle and a 200-query h_wang slice; report
   the diff in `min_distance` distribution and AUC, plus per-query positive
   label counts (must be unchanged).
8. Schedule Phase 2 (column rename) as a follow-on ask-first PR; coordinate the
   eight-file rename atomically with bundle artifact updates.

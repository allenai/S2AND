# Test suite audit — cut list (2026-07-20)

Six parallel domain audits of all 104 test files (~1,188 tests, ~40k lines), cross-referenced
against production code. Verdict: the suite is healthier than it looks — the bulk is boundary
validation, crash-safety, and Python↔Rust parity. Immediately cuttable: **~150 test functions,
~4,000–4,300 lines (~10%)**. Another **~1,900 lines** unlock conditionally (dead-code deletion +
post-migration teardown).

## Implementation status (2026-07-20)

The safe-now cleanup is applied. Relative to the audited working-tree baseline:

- test modules: 96 -> 90
- test function definitions: 1,188 -> 1,042 (-146)
- collected pytest cases: 1,761 -> 1,283 (-478, 27%)
- test Python lines: 38,910 -> 36,525 (-2,385)

Verification after the cleanup:

- `uv run ruff check .`: passed
- `git diff --check`: passed
- `uv run pytest -q`: 1,281 passed, 2 skipped

Tier 2 dead production surfaces and Tier 4 migration teardown remain intentionally deferred. The
independent linker row-feature oracle, single-read artifact integrity test, CI path check, and explicit
Arrow disallow merge contract were retained after source tracing showed they protect distinct behavior.

Theatre concentrates in five genres:
1. **Tombstones** — tests asserting removed flags/helpers/subcommands *stay* removed.
2. **Mock-the-code-under-test** — collaborator mocked, test asserts the mock got called / kwargs echoed.
3. **Constant change-detectors** — argparse defaults verbatim, `CONSTANT == value`, log-text shape.
4. **Vacuous assertions** — checks that cannot fail (`count >= 1`, testing behind a module-level skip).
5. **"Doesn't crash" duplicates** — MP/smoke permutations subsumed by a real consistency test.

---

## Tier 1 — whole-file deletions (high confidence, ~245 lines)

| File | Lines | Reason |
|---|---|---|
| `tests/test_stress_rebuild_cmd.py` | 28 | Two argparse change-detectors; one is a tombstone for removed `--build-path`. |
| `tests/test_plotting_utils.py` | 57 | `s2and/plotting_utils.py` reachable only from `scripts/archive/`; main test mocks `plot_box` and asserts bin constants. Delete code + test. |
| `tests/test_make_inventors_hf_specter_embeddings.py` | 19 | Tests argparse `choices`/`required` of a one-off scratch script. |
| `tests/test_featurizer_reuse_cmd.py` | 16 | Single ValueError guard in a dev profiling script. |
| `tests/test_featurizer_pool_mode_defaults.py` | 64 | Mocks `UniversalPool`, asserts mock received `use_threads=...`; real MP coverage lives in `test_featurizer.py`. |
| `tests/test_name_count_semantics.py` | 60 | MERGE into `test_data.py::test_preprocessing_name_counts_use_single_character_initial` (this variant's `"sattar abdul"` decoy is the strongest bit — move it there). |

## Tier 2 — package deals: dead production code + its tests (decision needed)

- **SHAP / claims / facet eval** (~300–400 test lines + library code): `pairwise_eval`, `claims_eval`,
  `facet_eval` in `s2and/eval.py` and all of `s2and/shap_utils.py` (incl. VotingClassifier /
  CalibratedClassifierCV unwrap paths) have **zero callers outside `scripts/archive/`**.
  Delete `TestShapIntegration` (tests/test_eval.py:97–423, keep only
  `test_pairwise_eval_validates_fitted_feature_names` L269), the claims-eval trio (L469–536),
  `test_shap_values_restore_lightgbm_booster_params` (L512), and the dead library surface together.
- **Unused seed-score/broadcast modes** (~180 test lines + model.py code): config modes `min`,
  `mean_min_hybrid`, `never`, `top1_consensus` are never set by the shipped v1.21 bundle
  (`clusterer.json` uses `mean`/`always`). Deleting the modes removes
  tests/test_cluster_incremental.py:1837, 2509, 2576.
- **`tests/test_clean_linker_dataset_bundles.py`** (229 lines) + script: one-off migration tool for the
  20260525 bundle ("removes weak `unlabeled_singleton_orcid` rows after manual label repair").
  Delete both once the bundle migration is done.
- **`tests/test_sanitize_arrow_replay_bundle.py`** (91 lines) + script: one-off metadata rewriter for the
  legacy 20260513 replay bundle; delete once the sanitized bundle is published.

## Tier 3 — per-test trims by file

### Incremental linking (~500 lines)

**tests/test_cluster.py** — cut `test_clusterer_rejects_legacy_positional_use_cache_argument` (L214, tombstone on Python's own keyword-only enforcement).

**tests/test_cluster_incremental.py** — delete:
- `test_predict_incremental_old_helper_name_is_not_exposed` (L442) — `hasattr` tombstone.
- `test_seed_cluster_count_matches_anddata_cluster_count` (L891) — tests a one-line dict comprehension.
- `test_predict_incremental_dont_use_cluster_seeds_flag` (L1564) — asserts default kwarg == explicit `False`; never exercises `True`.
- `test_packaged_incremental_name_tuples_are_immutable` (L896) — mocked loader identity passthrough; real canonicalization asserted at L500.
- `test_promoted_incremental_orcid_fanout_skips_seed_scan_without_query_orcids` (L478) — internal call-order pin via patched helper.
- `test_predict_from_arrow_paths_passes_disallow_sidecar_to_rust_featurizer_once` (L1448) — both callees mocked; "once" never actually asserted.

Merge:
- Telemetry-merge quintet (L1480, L1507, L1522, L1536, L1552) → 2 tests.
- L2199 → fold into L2242 (seeds+altered+disallows from arrow).
- L2334 + L2347 (one boolean predicate) → 1 test.
- L574 (explicit seed mapping) → fold into L500.
- L1404 (single-kwarg mock assert) → fold its assert into L1372.

**tests/test_incremental_linking_runtime.py** — delete:
- `test_signature_id_to_index_map_returns_zero_indexed_map_from_featurizer` (L1883) — tests `dict(enumerate(...))`.
- `test_naturalize_incremental_clusters_maps_split_ids` (L1889) — one-line `.get` comprehension; covered end-to-end at L1896.
- `test_compact_artifact_scoring_forwards_num_threads` (L659) — kwarg echoed by stub.
- `test_private_production_slice_uses_explicit_retrieval_top_k` (L1997) — kwarg-forwarding mock assert.
- Optional: trim `test_fused_pairwise_model_uses_configurable_nan_policies` (L855) from 3 → 2 cases.

**tests/test_incremental_linking_artifact.py** — delete `test_retrieval_stack_contract_records_constraint_decision_policy` (L353, constants-dict change detector; digest test L329 already binds it) and `test_concurrent_identical_artifact_publication_has_one_winner` (L254, same race as L277). Trim `test_metadata_v4_rejects_every_deleted_top_level_field` (L361) from 16 params → 1.

**tests/test_query_adapter.py** — optional: merge `test_orcid_enabled_false_suppresses_populated_orcid_field` (L218) into L303 (~45 lines, low priority — it carries migration rationale).

### Arrow ingest (~450 lines)

**tests/test_raw_block_candidate_plan_arrow.py** — merge L1960→L1899 (same ownership contract), delete duplicate-id middle-path test (L927; covered at L680 and L102), merge the two `initial_view_keeps_full_first_token` tests (L2056, L2078). Post-migration: delete legacy-key tombstone L2251.

**tests/test_convert_s2and_mini_to_arrow.py** — delete `test_name_counts_subcommand_is_validation_only` (L138, wiring assert + removed-subcommand tombstone) and `..._accepts_canonical_key_for_specter2_file` (L567, subsumed by release-layout fixture test). Merge the two parser-requires-explicit-dataset tests (L102, L119) and the two discovers-datasets dispatch tests (L149, L478). Trim L885 (integer-id rejection; same `validate_arrow_schema` funnel) and L603 from 4 params → 2.

**tests/test_arrow_inputs.py** — merges: L270→L297 (no-rehash superset), L371+L389 (immutability/copy of same dataclass), L746→L343 (same projection). Post-migration: legacy-path tests L424, L440, L575 become deletable.

**tests/test_arrow_training_ingestion.py** — delete `test_arrow_ingestion_rejects_specter2_alias_paths` (L463; validator tested at its home boundary, wiring proven at L639). Trim physical-type rejection params 3→1 per loader (L129, L176); drop the `""` case at L293.

**tests/test_arrow_release_layout.py** — delete `test_validate_release_root_accepts_canonical_keys_for_specter2_files` (L241; strictly subsumed by L184 on the identical fixture).

**tests/test_arrow_batch_lookup_index.py** — trim `test_write_reuse_rejects_corrupt_batch_index` (L72) 3 params → 1 (validator's corruption modes already tested at L29–69).

**tests/test_arrow_production_boundary.py** — delete `test_arrow_production_builder_calls_only_arrow_constructor` (L64) plus the `ArrowOnlyRustFeaturizer`/autouse-mock scaffolding (L17–47); move the int→str signature-id coercion assert into a real-extension test. Keep the four reject-unindexed/reject-rust-context contracts.

### Featurization (~500 lines)

**tests/test_featurizer.py** — delete `test_featurizer_with_feature_subset_ok` (L456, "subset" is character-identical to the full list), `test_bound_dataset_is_available_in_workers` (L528), `test_multiprocessing_fallback_to_single_thread` (L584), `test_spawn_context_compatibility` (L601) — all subsumed by the MP consistency test (L497); merge L551 into L497 as a config. Trim L70 to the one non-trivial assert.

**tests/test_feature_port_cache.py** — delete `test_increment_rust_featurizer_build_count_is_thread_safe` (L704, telemetry-counter stress) and `test_rust_featurizer_build_runs_per_dataset_with_and_without_json_paths` (L826, subsumed by L138). Merge the "python-side mutation doesn't rekey" trio (L296, L315, L327) into one parametrized test.

**tests/test_feature_port_parity.py** — delete `test_rust_extension_available` (L204; module-level skip makes it unfailable). Keep everything else — this is a load-bearing parity gate.

**tests/test_feature_snapshot_cache.py** — merge corrupt/truncated-zip tests (L350, L596); delete `test_snapshot_publication_uses_uncompressed_npz` (L371, pins `ZIP_STORED` with no perf evidence).

**tests/test_feature_block.py** — parametrize the three `*_rejects_integer_id_columns` tests (L751, L838, L973) into one; trim language-reliability params 5→3 (L93) folding in the accept-boundaries test (L148).

**tests/test_linker_pairwise_aggregates.py** — merge nan-policy mock pair (L115+L167) into one; delete one of L429/L493 (same pair-enumeration check); parametrize the `_localize_row_indices` trio (L627, L636, L645).

**tests/test_linker_feature_assembly.py** — delete `test_promoted_linker_feature_columns_are_promoted_53_without_rank_fractions` (L51; count subsumed by JSON-match test, move the `_rank_fraction` assert there).

**tests/test_feature_safe_view.py** — merge L110 into L130 (per-call test already asserts the same values plus cache non-latching).

### Production / eval (~750 lines, before Tier-2 SHAP deal)

**tests/test_production_model.py** — merge L1007 into L788 (carry the derived-fields-absent assert); trim L385 params 8→4; trim the 10-value type loops at L502/L534 to ~3 values.

**tests/test_promoted_train_calibrate_eval_helpers.py** — delete `test_arrow_rust_materialization_skips_tables_empty_after_dataset_filter` (L933; 6 monkeypatches, duplicated scaffold in the CLI file). Merge the six `_classic_feature_matrix` tests (L49, L478–L517) into 2.

**tests/test_promoted_linker_training_cli.py** — delete:
- `test_promoted_training_requires_explicit_artifacts_and_defaults_to_arrow_rust_source` (L47) — argparse-defaults change detector + removed-flags tombstone.
- `test_promoted_training_uses_extracted_training_helpers` (L400) — `inspect.getsource` grep for deleted imports; pure tombstone.
- `test_selected_row_positions_rejects_non_positive_limit_rows` (L127) — redundant with L68.
- `test_materialization_selects_source_tables_from_featureless_assets` (L228) — 8 monkeypatches, dict-keys loop.
- `test_run_uses_explicit_precomputed_promoted_bundle` (L870) — full-stub plumbing; the real subprocess CLI flow test covers it.

**tests/test_eval_prod_models.py** — delete defaults change-detectors L92, L97, L109, and L73 (stubbed resolver, error-string formatting). Trim L136 to the ValueError case (merge with L148); merge L156 into L170. Parametrize the four argv-rejection tests (L698–L731) into one. Keep the `resolve_arrow_dataset_paths` family and the two skipped LFS tests (tied to cutover checklist).

**tests/test_eval.py** — Tier-2 SHAP/claims deal above. Keep `TestB3AndF1` (hand-computed oracle) and the fitted-feature-names test.

**tests/test_promoted_arrow_rust_materializer.py** — merge `test_load_target_accepts_current_supported_promoted_features` (L56) into the reject tests.

**tests/test_eps_sweep.py** — delete `test_eps_sweep_cli_rejects_removed_no_choice_flags` (L79, tombstone); optional cut L68 (CLI default verbatim).

**tests/test_promoted_incremental_arrow_profile_cmd.py** — delete `..._is_canonical_command` (L11, tombstone) and `test_run_uses_direct_arrow_api_and_forwards_batching_threshold` (L58, 10 monkeypatches on a non-production tool); merge L51 into L44.

**tests/test_train_pairwise_script.py** — delete `test_feature_cache_dir_defaults_to_none` (L91); merge the two backend-env subprocess tests (L41, L51).

### Rust parity / subblocking / compare CLIs (~1,150 lines)

**tests/test_rust_distance_matrix_blockwise.py** (misnamed: everything mocks the Rust side; this is Python orchestration coverage) — delete L407 (copy-paste of L310 + mock trap), L557 (second change-detector on the same micro-optimization as L489), L945 (kwargs on mocked predict; real Arrow disallow coverage at L829). Merge the three `rejects precomputed dists` tests (L715, L732, L748) into one parametrized test; merge L255 into L205.

**tests/test_subblocking_telemetry.py** — delete `test_make_subblocks_with_telemetry_uses_python_implementation` (L116, asserts a fake was called), one of the three capacity-skip topologies (L497), and the warning-text assert in L240.

**tests/test_subblocking_merge_candidates.py** — delete `test_arrow_graph_subblocking_accepts_canonical_key_for_specter2_file` (L609; re-runs the L185 happy path with a renamed file).

**tests/test_rust_batch_chunking.py** — delete `test_rust_batch_uses_same_process_featurizer_without_cache_flag` (L315; final assert is `>= 1`, cannot fail) and `test_rust_batch_prediction_matches_observed_real_workload` (L353; live-RSS assert around a mocked zero-returning featurizer — flake bait; memory-prediction accuracy belongs in profiling evidence). Merge L202 into L165 (indexed API is now the only API).

**tests/test_compare_cmd.py** — delete L98 (12 monkeypatched seams, verbatim kwargs dicts, ~140 lines), L34, L15. Keep the vacuous-comparison guard (L242), stderr surfacing (L273), and cost guardrails.

**tests/test_compare_full_predict_arrow_parity.py** — delete parser change-detectors L120, L158, L180 (L180 is a removed-flag tombstone) and L107 (trivial dict conversion). Keep the real-data parity test (L71) and the verifier-strictness meta-tests.

**tests/test_compare_graph_subblocking_arrow_quality.py** — delete flag-default echo tests L33, L48, L74; keep the cost guardrail (L26) and the Arrow-loading test (L119).

**tests/test_largest_block_cmd.py** — delete L88 (mock call order), L57+L75 (flag-matrix), L119 (flag plumbing); merge L154+L162. Keep the vacuous-result guards (L130, L142) and the unbounded-run guardrail (L108).

**tests/test_compare_python_vs_rust.py** — delete L13 (name-list constant detector), L132 (default constant verbatim), L139 (argparse trivia); optionally merge L74 into L57.

**tests/test_rust_suite_common.py** — delete `test_collect_rust_extension_identity_uses_loaded_native_module` (L39, fake module fields echoed back).

**tests/test_extract_big_block_dataset.py** — if the script stays: keep the chunk-boundary parser test (L116) and the end-to-end writer test (L146); drop the pretty/minified params elsewhere and L132. If the extraction was one-time, delete file + script.

### Infra / misc (~450 lines beyond the Tier-1/2 files)

**tests/test_regression_fixes.py** — delete dead helper `_run_make_subblocks_with_fixed_first_pass` (L391–413, never called) and the bench-script parity test (L48, benchmark machinery). Merge the two `sync_rust_cluster_seeds` tests, dropping the ~15 telemetry-counter asserts.

**tests/test_data.py** — in `test_split_pairs_within_blocks` (L236) drop the code-generated golden tuples (keep count/balance/all-pairs asserts + one determinism pin); merge L578 into `test_initialization`.

**tests/test_memory_budget.py** — delete L87 (subsumed by L96); parametrize L109 into L77; drop the first half of L190–205 (verbatim duplicate of L133); merge L618 into L603.

**tests/test_generate_orcid_name_prefix_counts.py** — delete `test_import_is_side_effect_free_without_internal_pys2` (L31, subsumed by every `_load_module()` call) and `test_legacy_orcid_counts_remain_excluded_until_regenerated` (L148, pyproject string tombstone; policy enforced against the wheel in test_release_workflow.py). Trim the SQL-substring test (L37) to the ordering asserts.

**tests/test_name_tuple_artifact.py** — delete `test_loader_reads_data_and_metadata_once` (L87, read-count micro perf) and `test_packaged_artifact_cache_is_immutable_and_avoids_rehashing` (L61, tests lru_cache itself; keep only the frozenset assert).

**tests/test_normalization_version_contract.py** — delete `test_package_has_one_normalization_version` (L42, constant == its value).

**tests/test_thread_config.py** — delete `test_classic_training_default_n_jobs_matches_production_cli` (L66, `DEFAULT_CLASSIC_N_JOBS == 20`).

**tests/test_preprocess_papers_parallel_defaults.py** — reduce `test_preprocess_papers_parallel_linux_uses_pool` (L49) to its one real assert (`use_threads is False`) or delete.

**tests/test_logging_defaults.py** — delete `test_rust_suite_file_logging_preserves_existing_logger_level` (L9, logging-config change detector).

**tests/test_memory_calibration.py** — optional: fold the two `ignores_other_stages` tests (L74, L113) into their siblings.

**tests/test_run_ci_locally.py** — when the command-echo tests next churn, trim them to the env-hygiene asserts (`S2AND_BACKEND` must not leak). The dedicated Rust parity file list and its path-existence test were removed when CI stopped running those files twice.

## Tier 4 — post-migration teardown checklist (~450+ lines, don't cut yet)

When the migration compare suite retires, these become whole-file deletions:
`test_compare_cmd.py` (remnants), `test_largest_block_cmd.py`, `test_compare_python_vs_rust.py`,
`test_transfer_mini_cmd.py`, `test_rust_suite_common.py`, `test_compare_graph_subblocking_arrow_quality.py`,
plus the legacy-path/legacy-key tests flagged above in `test_arrow_inputs.py` and
`test_raw_block_candidate_plan_arrow.py`.

## Do not touch (crown jewels)

- `test_rust_lightgbm_booster_parity.py` — bit-exact independent Rust evaluator vs Python lightgbm, adversarial threshold inputs, coverage meta-asserts.
- `test_rust_signature_preprocess.py`, `test_feature_port_parity.py` (minus L204), the Rust-native subblocking tests in `test_subblocking_telemetry.py`, the real-data report test in `test_compare_full_predict_arrow_parity.py` (L71), the 4,096-pair accumulator oracle in `test_incremental_linking_runtime.py` (L1141) — the real Python↔Rust safety net.
- `test_canonical_name_examples.py`, `test_generate_name_counts_script.py`, and the six `name_counts_*` files — they partition cleanly by layer (writer/manifest/provenance/binding/crash-window/semantics) with no real overlap; migration contract, untouchable until the migration lands.
- `test_production_model_cli_flow.py` — the real end-to-end subprocess flow that makes the stub-heavy CLI tests deletable.
- `tests/linker_row_feature_reference.py` — genuine independent pure-Python oracle for the Rust row features; underused (one 3-row consumer at test_linker_runtime_batch.py:581) — widening its inputs is worth more than any deletion.
- Compact link-or-abstain policy suite (test_incremental_linking_runtime.py:1214–1702), raw-plan boundary validation, artifact tamper/atomic-publish tests, `test_mp.py` cancellation, `test_smoke_installed_incremental_arrow.py` (looks trivial, runs a real synthetic bundle end-to-end).

## CI observations

- CI runs the **entire** `tests/` suite per push (`scripts/run_ci_locally.py`, `--cov-fail-under=40`).
- Four parity files run **twice** per job (once Rust-required, again in the full sweep with
  `S2AND_BACKEND=python`): test_name_counts_manifest, test_feature_port_parity,
  test_rust_signature_preprocess, test_rust_batch_chunking. Intentional (two backends) but worth
  confirming the second run isn't pure duplication for the files that skip without Rust.
- The `heavy` pytest marker declared in pyproject.toml is used by zero tests — remove the marker or
  the declaration.

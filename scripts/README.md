# Scripts

Find runnable entry points under `scripts/`.
Follow the [release process](../docs/release.md) for production runs.

## Quick reference

### Production artifacts

| Script | What it does |
|---|---|
| `pairwise/train_pairwise.py` | Release-only training of the pairwise half of a native `production_model_vX.Y/` bundle |
| `pairwise/prepare_pairwise_inputs.py` | Prepare corrected benchmark splits (`benchmarks`), generate omission variants (`augmentation`), and attach vectors and verify populations (`finalize`) |
| `pairwise/package_pairwise_inputs.py` | Copy prepared pairwise sources and Arrow data into one portable directory |
| `release/package_corrected_datasets.py` | Package the final corrected records, pairwise populations, and complete incremental tasks with one common index |
| `release/write_release_bundles.py` | Write the published `inference`, `dataset`, and optional `source` (`--skip-source-bundle`) bundles from the prepared snapshot, scanning for plaintext emails by default |
| `linker/train_linker_and_finalize.py` | One-fit complete-bundle linker finalization, reload, and evaluation |
| `pairwise/release_pairwise.py` | Validation-only EPS calibration (`calibrate-eps`) and outer-test evaluation (`evaluate-heldout`) |
| `release/rebind_name_counts_index.py` | Bind a copy of a complete model to a rebuilt `name_counts_index/` without refitting |
| `intake/build_identity_evidence.py` | Join and audit Gold ORCID labels against a frozen Arrow snapshot |
| `intake/assign_block_roles.py` | Deterministically partition production blocks into ORCID-disjoint 1.5 training, tuning, and test pools |
| `intake/gate_public_snapshot.py` | Intake-once S2AG paper filtering, abstract redaction, email scrubbing, and `population_corrections.json` generation |
| `judging/review_packets.py` | Blind review packet builder, embedded ORCID control injector, and pre-launch validator; `judged_bank.py` reuses its renderer |
| `judging/decision_capture.py` | Record one final linker decision per signature during `sweep_eps_on_linking_gold --capture-dir` |
| `judging/judged_bank.py` | [Assemble blind author-pair packages, merge frozen verdicts, and check planted controls](../docs/judged_bank.md) |
| `linker/linker_source_bundle.py` | Assemble and preflight the fresh linker-source bundle and public-data root from reviewed inputs. Use only when the accepted support or replay Arrow roots change; see [data preparation](intake/README.md) |
| `intake/generate_canonical_name_tuples.py` | Deterministically generate the canonical tuple text file from the reviewed source artifact |
| `intake/counts/generate_author_counts_from_athena.py` | Export author-table name counts from Athena and build an immutable native `name_counts_index/` (AI2 staff only); see [count producers](intake/counts/README.md) |
| `intake/counts/generate_name_counts_from_parquet.py` | Build a canonical binary `name_counts_index/` from reviewed weighted raw-name Parquet files. Invoke with `python -m`. |
| `intake/counts/generate_orcid_name_prefix_counts.py` | Convert one reviewed, guardrailed CSV export into canonical ORCID prefix-count JSON bound to the packaged tuple artifact; invoke with `python -m` |

### 1.5 training data (stage 0 to linker)

The [1.5 runbook](../docs/v15_training_design.md) gives the order and commands.

| Script | What it does |
|---|---|
| `intake/requests_to_datasets.py` | Convert production Arrow request bundles into a training snapshot |
| `intake/build_freeze_input.py` | Build the role allocator input from exported signature evidence and source strata |
| `intake/attach_manual_labels.py` | Re-attach corpus-keyed manual labels to a verified stage-0 snapshot |
| `intake/build_v14_training_inventory.py` | Build the 1.4 training inventory from private, corpus-keyed ORCID labels |
| `intake/check_v14_training_overlap.py` | Check the sealed test role against the 1.4 training signature inventory |
| `pairwise/mine_pairwise_pairs.py` | Mine 1.4 errors and random controls from frozen `fit_pool` blocks |
| `linker/build_linker_inputs.py` | Build held-out request lists and ORCID-hidden 1.4 assignments |
| `linker/build_linker_task_tables.py` | Freeze top-25 linker fitting windows and blind-review gate windows |
| `linker/blind_gate_metrics.py` | Score query outcomes on fully judged `gate_thresh` windows (older JSONL review path) |
| `eps/orcid_b3.py` | Keep giant-block ORCID references outside prediction and score known identities |

### Tutorials

| Script | What it does |
|---|---|
| `tutorial_for_predicting_with_the_prod_model.py` | Predict clusters for every block of one Arrow or JSON dataset with a complete production model, or link one block's newer signatures to profiles seeded from its older labeled signatures; prints a summary and can write the clusters to JSON |

The script needs the `inference` bundle and an Arrow dataset, such as a benchmark from the `dataset` bundle.
Use [the data guide](../docs/data.md) to download them.

### Dataset creation & preprocessing

| Script | What it does |
|---|---|
| `intake/convert_to_arrow.py` | Convert service JSON, benchmarks, and linker replay inputs into bounded Arrow runtime artifacts with current raw-planner batch indexes; validate name counts/datasets and write checksum-bound collection manifests |
| `release/verification/analyze_giant_block_subblocking.py` | Sweep subblocking thresholds on a giant Arrow block and write preservation metrics, plots, and tables. Runs the native subblocker that `Clusterer.predict_from_arrow` uses, with the packaged ORCID prefix counts. |

### Evaluation and verification

| Script | What it does |
|---|---|
| `eps/sweep_eps_on_linking_gold.py` | Giant-block release EPS curves with hidden-ORCID identity B3; reuse native distances and linkage trees |
| `eps/plot_b3_eps_sweeps.py` | Plot completed giant-block B3 curves for manual EPS review |
| `eps/apply_reviewed_eps.py` | Apply the explicit reviewed EPS to a fresh complete model and preserve all trained weights |
| `release/verification/validate_local_arrow_release.py` | Non-network local Arrow release-root smoke; checks manifests, checksum fields, required files, batch-index paths, replay bundle manifests, and `name_counts_index` targets without scanning large Arrow tables |
| `release/verification/verify_production_model_distributions.py` | Verify wheel/sdist bytes, require canonical tuple/ORCID assets and an exact `s2and` package inventory, and reject packaged default/model paths |
| `release/verification/smoke_installed_incremental_arrow.py` | Installed-wheel synthetic canonical Arrow/linker smoke; the release runbook additionally requires one smoke with the real v1.3 bundle |
| `release/verification/evaluate_incremental_gold.py` | Freeze bounded gold identity cases before scores; test positive and author-absent profiles through the complete model's public Arrow API |
| `release/verification/compare_full_predict_arrow_parity.py` | Build a manifest-bound Arrow artifact with current raw-planner indexes and a generated bounded (or supplied) canonical name-count index, then compare Python/`ANDData` full predict against direct Arrow/Rust full predict |
| `release/verification/profile_promoted_incremental_arrow.py` | Produce the release performance report from a bounded promoted incremental Arrow workload |
| `release/verification/smoke_installed_rust_api.py` | Check that the installed `s2and._s2and_rust` extension exposes the expected API |
| `release/verification/verify_rust_distribution_licenses.py` | Release CI check that Rust wheels and source distributions keep their license notices |

### CI & release

| Script | What it does |
|---|---|
| `dev/run_ci_locally.py` | Shared hosted/local CI policy. With no argument it runs both jobs; pass `lint` or `typecheck-and-test` to run one job. |
| `dev/sync_version.py` | Sync `VERSION` into Python/Rust manifests and lockfiles |
| `dev/run_bounded.py` | Run one long model job with RAM, deadline, `--threads`, `--min-free-gib`, and expected-output limits and durable run records; see [bounded model runs](../docs/development.md#bounded-model-runs) |
| `dev/sync_feature_schema.py` | Regenerate the Rust feature-column constants from `s2and/feature_schema.py`; CI runs it with `--check` |

### Shared helpers (`common/`)

Code that several stages import. Put new shared helpers here rather than importing another stage's private helpers.

| Module | What it holds |
|---|---|
| `common/arrow_conversion_helpers.py` | Convert `ANDData`-like inputs directly to Arrow |
| `common/arrow_roots.py` | Read, copy and rebind Arrow collection roots; check their name-count index binding |
| `common/block_roles.py` | Read frozen block roles and apply their ORCID scoring policy |
| `common/distance_policy.py` | `DISTANCE_POLICY_VERSION`, which EPS curves and linker task tables record |
| `common/email_redaction.py` | Hash and scrub emails, and scan trees for plaintext emails |
| `common/identity_ledger.py` | Read the verified identity evidence for exported block signatures |
| `common/linker_seeds.py` | Read frozen pre-attachment seed membership |
| `common/stage0_invariants.py` | Check invariants shared by stage-0 producers and consumers |
| `common/initial_query_sources.py` | Verify derived query origins against the source signature tables |
| `common/orcid_references.py` | `EXCLUDED_ORCID_GROUPS` and the external ORCID reference-label reader |
| `common/release_root_checks.py` | `validate_release_root`, the local Arrow release-root check; `release/verification/validate_local_arrow_release.py` is its command line |
| `common/plan_datasets.py` | Load model-plan datasets from the published Arrow data |
| `common/unknown_signatures.py` | Read and validate unknown-signature lists beside source labels |

## Moved files

Look up a former path to find its current location.

| Former path | Current location |
| --- | --- |
| `scripts/analyze_giant_block_subblocking.py` | [release/verification/analyze_giant_block_subblocking.py](release/verification/analyze_giant_block_subblocking.py) |
| `scripts/arrow_conversion_helpers.py` | [common/arrow_conversion_helpers.py](common/arrow_conversion_helpers.py) |
| `scripts/convert_to_arrow.py` | [intake/convert_to_arrow.py](intake/convert_to_arrow.py) |
| `scripts/eps_sweep/__init__.py` | [eps/__init__.py](eps/__init__.py) |
| `scripts/eps_sweep/apply_reviewed_eps.py` | [eps/apply_reviewed_eps.py](eps/apply_reviewed_eps.py) |
| `scripts/eps_sweep/common.py` | [eps/sweep_curves.py](eps/sweep_curves.py) |
| `scripts/eps_sweep/orcid_b3.py` | [eps/orcid_b3.py](eps/orcid_b3.py) |
| `scripts/eps_sweep/plot_b3_eps_sweeps.py` | [eps/plot_b3_eps_sweeps.py](eps/plot_b3_eps_sweeps.py) |
| `scripts/eps_sweep/sweep_eps_on_linking_gold.py` | [eps/sweep_eps_on_linking_gold.py](eps/sweep_eps_on_linking_gold.py) |
| `scripts/production/README.md` | [release/README.md](release/README.md) |
| `scripts/production/__init__.py` | [common/__init__.py](common/__init__.py) |
| `scripts/production/counts/README.md` | [intake/counts/README.md](intake/counts/README.md) |
| `scripts/production/counts/__init__.py` | [intake/counts/__init__.py](intake/counts/__init__.py) |
| `scripts/production/counts/_run_support.py` | [intake/counts/_run_support.py](intake/counts/_run_support.py) |
| `scripts/production/counts/generate_author_counts_from_athena.py` | [intake/counts/generate_author_counts_from_athena.py](intake/counts/generate_author_counts_from_athena.py) |
| `scripts/production/counts/generate_name_counts_from_parquet.py` | [intake/counts/generate_name_counts_from_parquet.py](intake/counts/generate_name_counts_from_parquet.py) |
| `scripts/production/counts/generate_orcid_name_prefix_counts.py` | [intake/counts/generate_orcid_name_prefix_counts.py](intake/counts/generate_orcid_name_prefix_counts.py) |
| `scripts/production/data/README.md` | [intake/README.md](intake/README.md) |
| `scripts/production/data/_pairwise_augmentation.py` | [pairwise/_pairwise_augmentation.py](pairwise/_pairwise_augmentation.py) |
| `scripts/production/data/_pairwise_embeddings.py` | [pairwise/_pairwise_embeddings.py](pairwise/_pairwise_embeddings.py) |
| `scripts/production/data/assign_block_roles.py` | [intake/assign_block_roles.py](intake/assign_block_roles.py) |
| `scripts/production/data/block_roles.py` | [common/block_roles.py](common/block_roles.py) |
| `scripts/production/data/build_linker_task_tables.py` | [linker/build_linker_task_tables.py](linker/build_linker_task_tables.py) |
| `scripts/production/data/check_v14_training_overlap.py` | [intake/check_v14_training_overlap.py](intake/check_v14_training_overlap.py) |
| `scripts/production/data/gate_public_snapshot.py` | [intake/gate_public_snapshot.py](intake/gate_public_snapshot.py) |
| `scripts/production/data/mine_pairwise_pairs.py` | [pairwise/mine_pairwise_pairs.py](pairwise/mine_pairwise_pairs.py) |
| `scripts/production/data/package_corrected_datasets.py` | [release/package_corrected_datasets.py](release/package_corrected_datasets.py) |
| `scripts/production/data/package_pairwise_inputs.py` | [pairwise/package_pairwise_inputs.py](pairwise/package_pairwise_inputs.py) |
| `scripts/production/data/prepare_pairwise_inputs.py` | [pairwise/prepare_pairwise_inputs.py](pairwise/prepare_pairwise_inputs.py) |
| `scripts/production/data/review_packets.py` | [judging/review_packets.py](judging/review_packets.py) |
| `scripts/production/data/write_release_bundles.py` | [release/write_release_bundles.py](release/write_release_bundles.py) |
| `scripts/production/generate_canonical_name_tuples.py` | [intake/generate_canonical_name_tuples.py](intake/generate_canonical_name_tuples.py) |
| `scripts/production/model/__init__.py` | [pairwise/__init__.py](pairwise/__init__.py) |
| `scripts/production/model/augmented_leakage.py` | [pairwise/augmented_leakage.py](pairwise/augmented_leakage.py) |
| `scripts/production/model/initial_query_sources.py` | [common/initial_query_sources.py](common/initial_query_sources.py) |
| `scripts/production/model/linker_source_bundle.py` | [linker/linker_source_bundle.py](linker/linker_source_bundle.py) |
| `scripts/production/model/pairwise_incumbent_h13.json` | [pairwise/pairwise_incumbent_h13.json](pairwise/pairwise_incumbent_h13.json) |
| `scripts/production/model/plan_datasets.py` | [common/plan_datasets.py](common/plan_datasets.py) |
| `scripts/production/model/rebind_name_counts_index.py` | [release/rebind_name_counts_index.py](release/rebind_name_counts_index.py) |
| `scripts/production/model/release_pairwise.py` | [pairwise/release_pairwise.py](pairwise/release_pairwise.py) |
| `scripts/production/model/train_linker_and_finalize.py` | [linker/train_linker_and_finalize.py](linker/train_linker_and_finalize.py) |
| `scripts/production/model/train_pairwise.py` | [pairwise/train_pairwise.py](pairwise/train_pairwise.py) |
| `scripts/production/model/unknown_signatures.py` | [common/unknown_signatures.py](common/unknown_signatures.py) |
| `scripts/run_bounded.py` | [dev/run_bounded.py](dev/run_bounded.py) |
| `scripts/run_ci_locally.py` | [dev/run_ci_locally.py](dev/run_ci_locally.py) |
| `scripts/sync_feature_schema.py` | [dev/sync_feature_schema.py](dev/sync_feature_schema.py) |
| `scripts/sync_version.py` | [dev/sync_version.py](dev/sync_version.py) |
| `scripts/verification/compare_full_predict_arrow_parity.py` | [release/verification/compare_full_predict_arrow_parity.py](release/verification/compare_full_predict_arrow_parity.py) |
| `scripts/verification/evaluate_incremental_gold.py` | [release/verification/evaluate_incremental_gold.py](release/verification/evaluate_incremental_gold.py) |
| `scripts/verification/profile_promoted_incremental_arrow.py` | [release/verification/profile_promoted_incremental_arrow.py](release/verification/profile_promoted_incremental_arrow.py) |
| `scripts/verification/smoke_installed_incremental_arrow.py` | [release/verification/smoke_installed_incremental_arrow.py](release/verification/smoke_installed_incremental_arrow.py) |
| `scripts/verification/smoke_installed_rust_api.py` | [release/verification/smoke_installed_rust_api.py](release/verification/smoke_installed_rust_api.py) |
| `scripts/verification/validate_local_arrow_release.py` | [release/verification/validate_local_arrow_release.py](release/verification/validate_local_arrow_release.py) |
| `scripts/verification/verify_production_model_distributions.py` | [release/verification/verify_production_model_distributions.py](release/verification/verify_production_model_distributions.py) |
| `scripts/verification/verify_rust_distribution_licenses.py` | [release/verification/verify_rust_distribution_licenses.py](release/verification/verify_rust_distribution_licenses.py) |

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
| `intake/prepare_identity_labels.py` | Join and audit external Gold ORCID `(paper_id, position)` labels onto Arrow block signatures |
| `intake/assign_block_roles.py` | Deterministically partition production blocks into ORCID-disjoint 1.5 training, tuning, and test pools |
| `intake/gate_public_snapshot.py` | Intake-once S2AG paper filtering, abstract redaction, email scrubbing, and `population_corrections.json` generation |
| `judging/review_packets.py` | Leak-proof blind Sol 6.2 review packet builder, embedded ORCID control injector, and pre-launch validator |
| `intake/hash_release_emails.py` | Hash structured emails, scrub free-text emails at intake, and scan staged release bundles for plaintext emails |
| `linker/linker_source_bundle.py` | Assemble and preflight the fresh linker-source bundle and public-data root from reviewed inputs. Use only when the accepted support or replay Arrow roots change; see [data preparation](intake/README.md) |
| `intake/generate_canonical_name_tuples.py` | Deterministically generate the canonical tuple text file from the reviewed source artifact |
| `intake/counts/generate_author_counts_from_athena.py` | Export author-table name counts from Athena and build an immutable native `name_counts_index/` (AI2 staff only); see [count producers](intake/counts/README.md) |
| `intake/counts/generate_name_counts_from_parquet.py` | Build a canonical binary `name_counts_index/` from reviewed weighted raw-name Parquet files. Invoke with `python -m`. |
| `intake/counts/generate_orcid_name_prefix_counts.py` | Convert one reviewed, guardrailed CSV export into canonical ORCID prefix-count JSON bound to the packaged tuple artifact; invoke with `python -m` |

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

## Moved files

Paths start at the repository's `scripts/` directory. Join the former folder with the file name to look up an old path.
A dot means the file was directly in `scripts/`.

| File | Former folder | Current location |
| --- | --- | --- |
| `analyze_giant_block_subblocking.py` | `.` | [release/verification/analyze_giant_block_subblocking.py](release/verification/analyze_giant_block_subblocking.py) |
| `arrow_conversion_helpers.py` | `.` | [common/arrow_conversion_helpers.py](common/arrow_conversion_helpers.py) |
| `convert_to_arrow.py` | `.` | [intake/convert_to_arrow.py](intake/convert_to_arrow.py) |
| `__init__.py` | `eps_sweep` | [eps/__init__.py](eps/__init__.py) |
| `apply_reviewed_eps.py` | `eps_sweep` | [eps/apply_reviewed_eps.py](eps/apply_reviewed_eps.py) |
| `common.py` | `eps_sweep` | [eps/common.py](eps/common.py) |
| `orcid_b3.py` | `eps_sweep` | [eps/orcid_b3.py](eps/orcid_b3.py) |
| `plot_b3_eps_sweeps.py` | `eps_sweep` | [eps/plot_b3_eps_sweeps.py](eps/plot_b3_eps_sweeps.py) |
| `sweep_eps_on_linking_gold.py` | `eps_sweep` | [eps/sweep_eps_on_linking_gold.py](eps/sweep_eps_on_linking_gold.py) |
| `README.md` | `production` | [release/README.md](release/README.md) |
| `__init__.py` | `production` | [common/__init__.py](common/__init__.py) |
| `README.md` | `production/counts` | [intake/counts/README.md](intake/counts/README.md) |
| `__init__.py` | `production/counts` | [intake/counts/__init__.py](intake/counts/__init__.py) |
| `_run_support.py` | `production/counts` | [intake/counts/_run_support.py](intake/counts/_run_support.py) |
| `generate_author_counts_from_athena.py` | `production/counts` | [intake/counts/generate_author_counts_from_athena.py](intake/counts/generate_author_counts_from_athena.py) |
| `generate_name_counts_from_parquet.py` | `production/counts` | [intake/counts/generate_name_counts_from_parquet.py](intake/counts/generate_name_counts_from_parquet.py) |
| `generate_orcid_name_prefix_counts.py` | `production/counts` | [intake/counts/generate_orcid_name_prefix_counts.py](intake/counts/generate_orcid_name_prefix_counts.py) |
| `README.md` | `production/data` | [intake/README.md](intake/README.md) |
| `_pairwise_augmentation.py` | `production/data` | [pairwise/_pairwise_augmentation.py](pairwise/_pairwise_augmentation.py) |
| `_pairwise_embeddings.py` | `production/data` | [pairwise/_pairwise_embeddings.py](pairwise/_pairwise_embeddings.py) |
| `assign_block_roles.py` | `production/data` | [intake/assign_block_roles.py](intake/assign_block_roles.py) |
| `block_roles.py` | `production/data` | [common/block_roles.py](common/block_roles.py) |
| `build_linker_task_tables.py` | `production/data` | [linker/build_linker_task_tables.py](linker/build_linker_task_tables.py) |
| `check_v14_training_overlap.py` | `production/data` | [intake/check_v14_training_overlap.py](intake/check_v14_training_overlap.py) |
| `gate_public_snapshot.py` | `production/data` | [intake/gate_public_snapshot.py](intake/gate_public_snapshot.py) |
| `hash_release_emails.py` | `production/data` | [intake/hash_release_emails.py](intake/hash_release_emails.py) |
| `mine_pairwise_pairs.py` | `production/data` | [pairwise/mine_pairwise_pairs.py](pairwise/mine_pairwise_pairs.py) |
| `package_corrected_datasets.py` | `production/data` | [release/package_corrected_datasets.py](release/package_corrected_datasets.py) |
| `package_pairwise_inputs.py` | `production/data` | [pairwise/package_pairwise_inputs.py](pairwise/package_pairwise_inputs.py) |
| `prepare_identity_labels.py` | `production/data` | [intake/prepare_identity_labels.py](intake/prepare_identity_labels.py) |
| `prepare_pairwise_inputs.py` | `production/data` | [pairwise/prepare_pairwise_inputs.py](pairwise/prepare_pairwise_inputs.py) |
| `review_packets.py` | `production/data` | [judging/review_packets.py](judging/review_packets.py) |
| `write_release_bundles.py` | `production/data` | [release/write_release_bundles.py](release/write_release_bundles.py) |
| `generate_canonical_name_tuples.py` | `production` | [intake/generate_canonical_name_tuples.py](intake/generate_canonical_name_tuples.py) |
| `__init__.py` | `production/model` | [pairwise/__init__.py](pairwise/__init__.py) |
| `augmented_leakage.py` | `production/model` | [pairwise/augmented_leakage.py](pairwise/augmented_leakage.py) |
| `initial_query_sources.py` | `production/model` | [common/initial_query_sources.py](common/initial_query_sources.py) |
| `linker_source_bundle.py` | `production/model` | [linker/linker_source_bundle.py](linker/linker_source_bundle.py) |
| `pairwise_incumbent_h13.json` | `production/model` | [pairwise/pairwise_incumbent_h13.json](pairwise/pairwise_incumbent_h13.json) |
| `plan_datasets.py` | `production/model` | [common/plan_datasets.py](common/plan_datasets.py) |
| `rebind_name_counts_index.py` | `production/model` | [release/rebind_name_counts_index.py](release/rebind_name_counts_index.py) |
| `release_pairwise.py` | `production/model` | [pairwise/release_pairwise.py](pairwise/release_pairwise.py) |
| `train_linker_and_finalize.py` | `production/model` | [linker/train_linker_and_finalize.py](linker/train_linker_and_finalize.py) |
| `train_pairwise.py` | `production/model` | [pairwise/train_pairwise.py](pairwise/train_pairwise.py) |
| `unknown_signatures.py` | `production/model` | [common/unknown_signatures.py](common/unknown_signatures.py) |
| `run_bounded.py` | `.` | [dev/run_bounded.py](dev/run_bounded.py) |
| `run_ci_locally.py` | `.` | [dev/run_ci_locally.py](dev/run_ci_locally.py) |
| `sync_feature_schema.py` | `.` | [dev/sync_feature_schema.py](dev/sync_feature_schema.py) |
| `sync_version.py` | `.` | [dev/sync_version.py](dev/sync_version.py) |
| `compare_full_predict_arrow_parity.py` | `verification` | [release/verification/compare_full_predict_arrow_parity.py](release/verification/compare_full_predict_arrow_parity.py) |
| `evaluate_incremental_gold.py` | `verification` | [release/verification/evaluate_incremental_gold.py](release/verification/evaluate_incremental_gold.py) |
| `profile_promoted_incremental_arrow.py` | `verification` | [release/verification/profile_promoted_incremental_arrow.py](release/verification/profile_promoted_incremental_arrow.py) |
| `smoke_installed_incremental_arrow.py` | `verification` | [release/verification/smoke_installed_incremental_arrow.py](release/verification/smoke_installed_incremental_arrow.py) |
| `smoke_installed_rust_api.py` | `verification` | [release/verification/smoke_installed_rust_api.py](release/verification/smoke_installed_rust_api.py) |
| `validate_local_arrow_release.py` | `verification` | [release/verification/validate_local_arrow_release.py](release/verification/validate_local_arrow_release.py) |
| `verify_production_model_distributions.py` | `verification` | [release/verification/verify_production_model_distributions.py](release/verification/verify_production_model_distributions.py) |
| `verify_rust_distribution_licenses.py` | `verification` | [release/verification/verify_rust_distribution_licenses.py](release/verification/verify_rust_distribution_licenses.py) |

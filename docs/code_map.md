# S2AND code map

## Production Arrow requests

| Step | Start here |
| --- | --- |
| Open and validate the Arrow inputs | [`arrow_inputs.py`](../s2and/arrow_inputs.py) |
| Load the bundle and bind the name-count index | [`production_model.py`](../s2and/production_model.py) |
| Bulk · Split oversized blocks | [`subblocking.py`](../s2and/subblocking.py) |
| Bulk · Score pairs and cluster by average linkage | [`model.py`](../s2and/model.py) |
| Bulk · Apply enabled cluster corrections inside each block | [`block_corrections.py`](../s2and/block_corrections.py) |
| Bulk · Reunite no-middle-name pieces across subblocks | [`model.py`](../s2and/model.py) (`_reunite_no_middle_dead_ends`) |
| Bulk · Attach initial-only groups | [`model.py`](../s2and/model.py) |
| Bulk · Join clusters across subblocks by ORCID | [`orcid_join.py`](../s2and/orcid_join.py) |
| Incremental · Set up required seed memberships and disallows | [`model.py`](../s2and/model.py) (`Clusterer.predict_incremental_from_arrow`), [`seed_merging.py`](../s2and/seed_merging.py), then [`incremental_linking/production.py`](../s2and/incremental_linking/production.py) |
| Incremental · Retrieve candidate seed profiles | [`raw_candidate_planner.rs`](../s2and_rust/src/raw_candidate_planner.rs) |
| Incremental · Score and rank candidates | [`incremental_linking/runtime.py`](../s2and/incremental_linking/runtime.py) |
| Incremental · Gate the best link or abstain | [`incremental_linking/logistic_gate.py`](../s2and/incremental_linking/logistic_gate.py) |
| Incremental · Complete the result, including residual clustering | [`incremental_linking/completion.py`](../s2and/incremental_linking/completion.py) |

## Training pipeline

| Stage | Scripts | Output |
| --- | --- | --- |
| Intake | [`convert_to_arrow.py`](../scripts/intake/convert_to_arrow.py), [`prepare_identity_labels.py`](../scripts/intake/prepare_identity_labels.py), [`assign_block_roles.py`](../scripts/intake/assign_block_roles.py) | Arrow datasets, verified identity labels, and frozen block roles |
| Judging | [`review_packets.py`](../scripts/judging/review_packets.py) | Blind review packets, control keys, and judge scores |
| Pairwise | [`prepare_pairwise_inputs.py`](../scripts/pairwise/prepare_pairwise_inputs.py), [`package_pairwise_inputs.py`](../scripts/pairwise/package_pairwise_inputs.py), [`train_pairwise.py`](../scripts/pairwise/train_pairwise.py), [`release_pairwise.py`](../scripts/pairwise/release_pairwise.py) | Portable pairwise inputs, both boosters, benchmark EPS, and held-out scores |
| Linker | [`build_linker_task_tables.py`](../scripts/linker/build_linker_task_tables.py), [`linker_source_bundle.py`](../scripts/linker/linker_source_bundle.py), [`train_linker_and_finalize.py`](../scripts/linker/train_linker_and_finalize.py) | Frozen task tables, a complete model, and linker evaluation |
| Clustering threshold (EPS) | [`sweep_eps_on_linking_gold.py`](../scripts/eps/sweep_eps_on_linking_gold.py), [`plot_b3_eps_sweeps.py`](../scripts/eps/plot_b3_eps_sweeps.py), [`apply_reviewed_eps.py`](../scripts/eps/apply_reviewed_eps.py) | ORCID B-cubed curves and a complete bundle with the reviewed threshold |
| Release (runs before pairwise training; see release Stage 2 in [`release.md`](release.md)) | [`package_corrected_datasets.py`](../scripts/release/package_corrected_datasets.py), [`write_release_bundles.py`](../scripts/release/write_release_bundles.py) | A corrected snapshot and the published `inference`, `dataset`, and optional `source` bundles |

## Where is it?

| Looking for | Start here |
| --- | --- |
| Training data and local assets | [`s2and/data.py`](../s2and/data.py) holds `ANDData`. [`s2and/data/`](../s2and/data/) holds assets and downloaded bundles. |
| Arrow table schema | [`arrow_schema_contract.json`](../s2and/arrow_schema_contract.json) |
| Batch index readers and writers | [`incremental_linking/feature_block_arrow.py`](../s2and/incremental_linking/feature_block_arrow.py) |
| Complete bundle manifest validation | [`production_bundle.py`](../s2and/production_bundle.py) |
| Ordered pairwise feature columns | [`feature_schema.py`](../s2and/feature_schema.py) |
| Native pairwise feature calculations | [`features.rs`](../s2and_rust/src/features.rs) |
| Native LightGBM scoring | [`lightgbm_booster.rs`](../s2and_rust/src/lightgbm_booster.rs) |
| Pair constraints | [`constraints.rs`](../s2and_rust/src/constraints.rs) |
| Canonical name and surname rules | [`text.py`](../s2and/text.py) |
| Name-count index and producers | [`name_counts_index.py`](../s2and/name_counts_index.py), [intake count commands](../scripts/intake/counts/README.md) |
| Name-conflict repair | [`name_conflict_repair.py`](../s2and/name_conflict_repair.py) |
| Retained seed memberships | [`seed_merging.py`](../s2and/seed_merging.py) |
| Linker ranker and gate fitting | [`incremental_linking_training/classic.py`](../s2and/incremental_linking_training/classic.py) |
| Thread and memory budgets | [`thread_config.py`](../s2and/thread_config.py), [`memory_budget.py`](../s2and/memory_budget.py) |
| Local and hosted CI policy | [`run_ci_locally.py`](../scripts/dev/run_ci_locally.py) |

## Public entry points

The service imports `ArrowDataset` from `s2and.arrow_inputs`, `validate_arrow_schema` and `validate_arrow_file_schema` from `s2and.arrow_schema`, `read_cluster_seeds_arrow`, `read_cluster_seed_disallows_arrow`, and `write_raw_arrow_batch_lookup_indexes` from `s2and.incremental_linking.feature_block`, `load_production_model` from `s2and.production_model`, and `Clusterer` from `s2and.model`.
Use `ArrowDataset.open` for published inputs, `ArrowDataset.open_trusted` for trusted caller-owned inputs, and `Clusterer.predict_from_arrow` or `Clusterer.predict_incremental_from_arrow` to predict.
The [inference guide](production_inference.md) and [training guide](training.md) describe the public Python API.
The [Arrow specification](rust/arrow_dataset_spec.md) defines the wire formats, and the [script catalog](../scripts/README.md) maps published commands to their current paths.

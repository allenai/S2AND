# Scripts

This catalog describes implemented entry points in the current worktree. It
does not authorize expensive production work or imply that the
v1.3 release work is complete. Release operators must follow
[../docs/release.md](../docs/release.md).

## Quick reference

### Production artifacts

| Script | What it does |
|---|---|
| `production/model/train_pairwise.py` | Release-only training of the pairwise half of a native `production_model_vX.Y/` bundle |
| `production/model/train_linker_and_finalize.py` | One-fit complete-bundle linker finalization, reload, and evaluation |
| `production/model/release_pairwise.py` | Validation-only EPS calibration plus measurement components and `evaluate-release` aggregation into one report |
| `production/model/linker_source_bundle.py` | Assemble and preflight the fresh linker-source bundle and public-data root from reviewed inputs (release Stage 2) |
| `production/generate_canonical_name_tuples.py` | Deterministically generate the canonical tuple text file from the reviewed source artifact |
| `production/counts/generate_name_counts.py` | Convert one reviewed, guardrailed CSV export into an immutable native `name_counts_index/`; invoke with `python -m` |
| `production/counts/generate_orcid_name_prefix_counts.py` | Convert one reviewed, guardrailed CSV export into canonical ORCID prefix-count JSON bound to the packaged tuple artifact; invoke with `python -m` |

### Tutorials

| Script | What it does |
|---|---|
| `tutorial_for_predicting_with_the_prod_model.py` | Guide to predicting with a complete production bundle, Arrow input by default; JSON fixtures are opt-in |
| `tutorial.ipynb` | Notebook walkthrough of the S2AND pipeline on the legacy JSON benchmark files |

> **Update at v1.3 release:** both tutorials need artifacts that are not
> published yet. The script needs a complete v1.3 bundle; the notebook needs a
> public-format-1 name-count index. Update this note once they run against the
> published data.

### Dataset creation & preprocessing

| Script | What it does |
|---|---|
| `make_inventors_s2and_subset.py` | Create inventors S2AND subset (defaults to a local ignored output path) |
| `make_inventors_split_and_histograms.py` | Split inventors data and generate histograms (defaults to a local ignored output path) |
| `make_inventors_hf_specter_embeddings.py` | Generate one inventors SPECTER embedding set per invocation (`--model specter` or `--model specter2`; defaults to a local ignored output path) |
| `extract_big_block_dataset.py` | Stream a monolithic big-block export into `ANDData`-friendly `signatures.json`, `papers.json`, `specter.pickle`, `cluster_seeds.json`, `altered_cluster_signatures.txt`, and `meta.json`; supports both pretty-printed and minified JSON exports |
| `convert_to_arrow.py` | Join canonical names into the benchmarks and the augmented training set; convert service JSON, benchmarks, and linker replay inputs into bounded Arrow runtime artifacts with current raw-planner batch indexes; validate name counts/datasets and write checksum-bound collection manifests |
| `analyze_giant_block_subblocking.py` | Sweep subblocking thresholds on an extracted giant block and write preservation metrics, plots, and tables. Uses the same graph fallback as `Clusterer`. Needs the ORCID prefix-count files (see [subblocking](../docs/subblocking.md)). |
| `bench_preprocess_phases.py` | Benchmark preprocessing phases (papers, signatures) across serial / threads / processes |

### Evaluation and verification

| Script | What it does |
|---|---|
| `eval_prod_models.py` | Evaluate an explicitly supplied SPECTER2 bundle on full, inventors_s2and, or mini datasets. Requires an explicit `--arrow-data-root` or `--json-data-root` (JSON also needs `--name-counts-index-root` and `--name-tuples-path`). Non-training evaluation uses the test blocks recorded in the bundle, rejects `--seed`, and auto-uses Arrow when complete artifacts exist; SPECTER1 is an explicit `--train` research comparison only. |
| `eps_sweep/sweep_eps_on_linking_gold.py` | Research EPS sweep over linking gold; it is not the validation-only pairwise-stage selector/finalizer used by the release |
| `verification/validate_local_arrow_release.py` | Non-network local Arrow release-root smoke; checks manifests, checksum fields, required files, batch-index paths, replay bundle manifests, and `name_counts_index` targets without scanning large Arrow tables |
| `verification/verify_production_model_distributions.py` | Verify wheel/sdist bytes, require canonical tuple/ORCID assets, and reject packaged default/model paths |
| `verification/smoke_installed_incremental_arrow.py` | Installed-wheel synthetic canonical Arrow/linker smoke; the release runbook additionally requires one smoke with the real v1.3 bundle |
| `verification/compare_full_predict_arrow_parity.py` | Build a manifest-bound Arrow artifact with current raw-planner indexes and a generated bounded (or supplied) canonical name-count index, then compare Python/`ANDData` full predict against direct Arrow/Rust full predict |
| `verification/profile_promoted_incremental_arrow.py` | Produce the release performance report from a bounded promoted incremental Arrow workload |
| `verification/compare_graph_subblocking_arrow_quality.py` | Produce the release subblocking-quality report by comparing Python and Rust graph subblocking on the frozen Arrow and component inputs |
| `verification/smoke_installed_rust_api.py` | CI smoke that checks the installed `s2and-rust` extension exposes the expected API |

### CI & release

| Script | What it does |
|---|---|
| `run_ci_locally.py` | Shared hosted/local CI policy. With no argument it runs both jobs; pass `lint` or `typecheck-and-test` to run one job. |
| `sync_version.py` | Sync `VERSION` into Python/Rust manifests and lockfiles |
| `sync_feature_schema.py` | Regenerate the Rust feature-column constants from `s2and/feature_schema.py`; CI runs it with `--check` |

### Archived historical artifacts

Files under `archive/` are retained for reproducibility, data lineage, or as
behavioral specifications. They are not supported current entrypoints. Read the
warning at the top of each file before use; several require private data, refer
to legacy APIs, execute work at import time, or can call paid services. Any
attempt to rerun one needs an explicit bounded migration plan first.

| Script | Status | Why it is retained |
|---|---|---|
| `archive/LLM_based_filtering_of_name_tuples.py` | Provenance only; do not run | Records the paid/nondeterministic curation pipeline that produced the raw source consumed by canonical name-tuple generation |
| `archive/blog_post_eval.py` | Historical private-data workflow | Preserves the blog and claims-evaluation recipe |
| `archive/make_augmentation_dataset_a.py` | Provenance only; incompatible as written | Records pair selection and title-only embedding inputs for the augmented training data |
| `archive/make_augmentation_dataset_b.py` | Provenance only; incompatible as written | Records the exact feature-corruption and translation policy for the augmented training data |
| `archive/make_claims_dataset.py` | Historical private-data workflow | Records how block-local Semantic Scholar corrections datasets were constructed |
| `archive/make_s2and_mini_dataset.py` | Legacy JSON/pickle recipe | Records the source-selection policy for the mini datasets still used in evaluation and conversion |
| `archive/sota.py` | Historical private-data workflow | Preserves post-paper SOTA splits, metrics, and multi-seed aggregation fixes |
| `archive/transform_all_datasets.py` | Legacy dry-run utility | Preserves the old-schema conversion and curated dataset corrections |

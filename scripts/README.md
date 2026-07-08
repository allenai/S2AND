# Scripts

## Quick reference

### Rust profiling & parity

| Script | What it does | Key output |
|---|---|---|
| `rust_suite.py compare` | Featurize random pairs on one dataset, compare Python vs Rust outputs | Feature parity report, runtime speedup, RSS reduction |
| `rust_suite.py transfer-mini` | Smoke-scale KISTI transfer run by default; pass the full preset for the historical 3-dataset reduced-scale run | Per-stage timing, peak RSS, clustering quality (python vs rust) |
| `rust_suite.py prod-inference` | Run Arrow `predict_from_arrow_paths` inference with the pre-trained prod model + cProfile; legacy JSON/ANDData baselines are opt-in | Function-level hotspots, latency, RSS, clustering metrics |
| `rust_suite.py featurizer-reuse` | Repeated production-model predictions through Arrow by default; `--input-format json` keeps the legacy same-object vs re-instantiated `ANDData` cache check | Per-iteration timing, RSS, Arrow telemetry or legacy featurizer cache counts |
| `rust_suite.py largest-block` | Profile one large block; `--mode single --backend rust --input-format arrow` uses Arrow `predict_from_arrow_paths`, while compare/constraint parity remain JSON reference workflows | Partition diff (digest + per-signature), latency, RSS; optional `--quality-check` + JSON-only `--constraint-sample` |
| `rust_suite.py promoted-incremental-arrow-profile` | Arrow-only promoted Rust `predict_incremental` profiling against the canonical `s2and_and_big_blocks_linker_dataset_20260525` bundle | Per-run wall time, p50 latency, peak RSS, promoted incremental telemetry, Arrow planner/summary timings |
| `rust_suite.py stress-rebuild` | Repeat Arrow Rust featurizer construction to stress lifecycle stability | Per-iteration elapsed + RSS peaks, RSS growth fraction, failure payloads |
| `rust_suite.py calibrate-phase-a` | Calibrate memory estimates for phase-A accumulator from memory telemetry JSONL | Per-entry byte overhead percentiles |
| `rust_suite.py calibrate-rust-batch` | Calibrate memory estimates for Rust batch persistent overhead from memory telemetry JSONL | Per-row byte overhead percentiles |

### Production artifacts

| Script | What it does |
|---|---|
| `production/model/train_pairwise.py` | Train the pairwise half of a native `production_model_vX.Y/` bundle |
| `production/model/train_linker_and_finalize.py` | Train the promoted incremental linker and finalize the production model bundle |
| `production/model/linker_train_calibrate_eval.py` | Low-level promoted linker replay implementation used by the finalization wrapper |
| `production/counts/generate_name_counts.py` | Documentation for how production name-count metadata was collected (internal data) |
| `production/counts/generate_orcid_name_prefix_counts.py` | Documentation for how ORCID prefix counts were collected (internal data) |

### Paper experiments & tutorials

| Script | What it does |
|---|---|
| `transfer_experiment_seed_paper.py` | Main script to reproduce all paper experiments |
| `tutorial_for_predicting_with_the_prod_model.py` | Guide to using the released production model with Arrow input by default; JSON fixtures remain opt-in |
| `tutorial.ipynb` | Notebook walkthrough of the S2AND pipeline |

### Dataset creation & preprocessing

| Script | What it does |
|---|---|
| `make_inventors_s2and_subset.py` | Create inventors S2AND subset (defaults to a local ignored output path) |
| `make_inventors_split_and_histograms.py` | Split inventors data and generate histograms (defaults to a local ignored output path) |
| `make_inventors_hf_specter_embeddings.py` | Generate SPECTER embeddings for inventors dataset (defaults to a local ignored output path) |
| `extract_big_block_dataset.py` | Convert a monolithic big-block export into `ANDData`-friendly `signatures.json`, `papers.json`, and `specter.pickle` files; supports both pretty-printed and minified JSON exports |
| `convert_to_arrow.py` | Convert service JSON, benchmark datasets, linker replay raw inputs, and shared name-counts into bounded Arrow runtime artifacts with current raw-planner batch-index sidecars (`S2ABI002`); subcommands are `service-json`, `benchmark`, `linker-replay`, `name-counts-index`, and `validate` |
| `analyze_giant_block_subblocking.py` | Sweep subblocking thresholds on an extracted giant block and write preservation metrics, plots, and tables |
| `bench_preprocess_phases.py` | Benchmark preprocessing phases (papers, signatures) across serial / threads / processes |

### Testing

| Script | What it does |
|---|---|
| `eval_prod_models.py` | Evaluate production models (SPECTER1 vs SPECTER2) on full, inventors_s2and, or mini datasets; non-training evals use Arrow automatically when complete Arrow artifacts exist |
| `verification/validate_local_arrow_release.py` | Non-network local Arrow release-root smoke; checks manifests, checksum fields, required files, batch-index paths, replay bundle manifests, and `name_counts_index` targets without scanning large Arrow tables |
| `verification/compare_full_predict_arrow_parity.py` | Build a bounded Arrow parity artifact, including current raw-planner batch-index sidecars, and compare incumbent full predict against direct Arrow/Rust full predict |
| `verification/compare_existing_arrow_anddata_feature_parity.py` | Compare Rust feature matrices from existing raw `ANDData` JSON/pickle inputs against existing Arrow release bundles |

### CI & release

| Script | What it does |
|---|---|
| `run_ci_locally.py` | Run CI locally with parity to `.github/workflows/main.yaml`: version sync check, lint job, `py-only` + `rust-enabled` matrix lanes, Rust parity tests, ty, pytest |
| `sync_version.py` | Sync VERSION file into pyproject.toml + Cargo.toml |

### Archived

Scripts in `archive/` are historical and generally not intended to be rerun.

| Script | What it does |
|---|---|
| `archive/transfer_experiment_internal.py` | Full-scale transfer experiment with unreleased datasets (supports Rust backend) |
| `archive/make_augmentation_dataset_a.py` | Create augmentation dataset step 1 (unreleased data) |
| `archive/make_augmentation_dataset_b.py` | Create augmentation dataset step 2 (unreleased data) |
| `archive/test_s2aff.py` | Test S2 affiliation matching (internal) |
| `archive/sota.py` | Historical state-of-the-art results table script from the paper |
| `archive/make_s2and_mini_dataset.py` | Historical mini-dataset creation utility |
| `archive/make_s2and_name_tuples.py` | Historical name tuples creation (superseded; don't rerun) |
| `archive/LLM_based_filtering_of_name_tuples.py` | Historical Gemini-based name-tuple filtering helper (costs money to re-run) |
| `archive/paper_experiments.sh` | Historical paper-era command set; reproduce from the `s2and_paper` branch instead of current `main` |
| `archive/find_largest_block.py` | Scan dataset signature files and find the single largest block |
| `archive/blog_post_eval.py` | Min edit distance numbers for blog post (Python-only legacy) |
| `archive/claims_cluster_eval.py` | Evaluate on S2 corrections data (Python-only legacy) |
| `archive/transform_all_datasets.py` | Transform old dataset format to final |
| `archive/make_claims_dataset.py` | Create S2 corrections evaluation dataset (internal data) |
| `archive/test_specter2.ipynb` | Historical notebook comparing SPECTER1 and SPECTER2 embeddings |

## Notes

**`transfer_experiment_seed_paper.py`**: Uses `main_data_dir` from `s2and/data/path_config.json` (or set the `S2AND_PATH_CONFIG` env var to point elsewhere). For one-shot large runs, leave `--use_cache` off unless you expect to rerun the same workload and reuse cached pair features. With `--use_cache`, S2AND writes the SQLite-backed pair-feature cache, and a loaded pair-feature cache is also kept in process memory, so it can add IO and RAM pressure when the cache will not be reused.

**`production/model/linker_train_calibrate_eval.py`**: Defaults to safe smoke/materialization behavior unless `--run-full` is passed. Full runs can be expensive; use `--limit-rows`, `--tables`, or `--datasets` with `--materialize-only` for bounded checks.

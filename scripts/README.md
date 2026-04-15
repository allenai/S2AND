# Scripts

## Quick reference

### Rust profiling & parity

| Script | What it does | Key output |
|---|---|---|
| `rust_suite.py compare` | Featurize random pairs on one dataset, compare Python vs Rust outputs | Feature parity report, runtime speedup, RSS reduction |
| `rust_suite.py transfer-mini` | Train union model across 3 datasets at reduced scale (~30 min) | Per-stage timing, peak RSS, clustering quality (python vs rust) |
| `rust_suite.py prod-inference` | Run inference with pre-trained prod model + cProfile | Function-level hotspots, latency, RSS, clustering metrics |
| `rust_suite.py featurizer-reuse` | Repeated KISTI predictions, same-object vs re-instantiated | Featurizer cache hit rate, per-iteration timing, RSS |
| `rust_suite.py largest-block` | Profile Python vs Rust on one large block | Partition diff (digest + per-signature), latency, RSS; optional `--quality-check` + `--constraint-sample` |
| `rust_suite.py big-block-incremental` | Compare incremental baseline vs phase-split incremental behavior on giant-block subsets | Runtime delta, peak RSS delta, cluster-equivalence / partition-diff, `phase_b_mode` telemetry |
| `rust_suite.py stress-rebuild` | Repeat Rust featurizer construction (`from_json_paths` / `from_dataset`) to stress lifecycle stability | Per-iteration elapsed + RSS peaks, RSS growth fraction, failure payloads |
| `rust_suite.py measure-counter-data` | Measure CounterData memory contribution to Rust featurizer | Disk and in-memory size with vs without CounterData fields |
| `rust_suite.py calibrate-phase-a` | Calibrate memory estimates for phase-A accumulator from log files | Per-entry byte overhead percentiles |
| `rust_suite.py calibrate-rust-batch` | Calibrate memory estimates for Rust batch persistent overhead from log files | Per-row byte overhead percentiles |

### Rust utilities

| Script | What it does |
|---|---|
| `export_name_counts_for_rust.py` | Convert Python name-count pickle to Rust JSON artifact with normalization metadata (defaults output to `scratch/`) |

### Paper experiments & tutorials

| Script | What it does |
|---|---|
| `transfer_experiment_seed_paper.py` | Main script to reproduce all paper experiments |
| `tutorial_for_predicting_with_the_prod_model.py` | Guide to using the released production model (supports `--use-rust`) |
| `tutorial.ipynb` | Notebook walkthrough of the S2AND pipeline |

### Dataset creation & preprocessing

| Script | What it does |
|---|---|
| `full_model_dump.py` | Train and dump a full model on all datasets (includes unreleased data) |
| `make_inventors_s2and_subset.py` | Create inventors S2AND subset (defaults output to `scratch/inventors_s2and`) |
| `make_inventors_split_and_histograms.py` | Split inventors data and generate histograms (defaults output to `scratch/inventors`) |
| `make_inventors_hf_specter_embeddings.py` | Generate SPECTER embeddings for inventors dataset (defaults to `scratch/inventors_s2and`) |
| `extract_big_block_dataset.py` | Convert a monolithic big-block export into `ANDData`-friendly `signatures.json`, `papers.json`, and `specter.pickle` files |
| `bench_preprocess_phases.py` | Benchmark preprocessing phases (papers, signatures) across serial / threads / processes |
| `get_name_counts.py` | Documentation for how name counts metadata was collected (internal data) |
| `get_orcid_name_prefix_counts.py` | Documentation for how ORCID prefix counts were collected (internal data) |

### Testing

| Script | What it does |
|---|---|
| `eval_prod_models.py` | Evaluate production models (SPECTER1 vs SPECTER2) on inventors_s2and or mini datasets |

### CI & release

| Script | What it does |
|---|---|
| `run_ci_locally.py` | Run CI locally with parity to `.github/workflows/main.yaml`: lint job, `py-only` + `rust-enabled` matrix lanes, Rust parity tests, ty, pytest |
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
| `archive/LLM_based_filtering_of_name_tuples.py` | Filter name tuples using Gemini 2.5 Pro (costs money to re-run) |
| `archive/find_largest_block.py` | Scan dataset signature files and find the single largest block |
| `archive/blog_post_eval.py` | Min edit distance numbers for blog post (Python-only legacy) |
| `archive/claims_cluster_eval.py` | Evaluate on S2 corrections data (Python-only legacy) |
| `archive/transform_all_datasets.py` | Transform old dataset format to final |
| `archive/make_claims_dataset.py` | Create S2 corrections evaluation dataset (internal data) |

## Notes

**`transfer_experiment_seed_paper.py`**: Uses `main_data_dir` from `s2and/data/path_config.json` (or set the `S2AND_PATH_CONFIG` env var to point elsewhere). If you have limited RAM, don't use `--use_cache` — it's slower without the cache but won't try to fit all feature data into memory.

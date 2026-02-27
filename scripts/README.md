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

### Rust utilities

| Script | What it does |
|---|---|
| `export_name_counts_for_rust.py` | Convert Python name-count pickle to Rust JSON artifact with normalization metadata |

### Paper experiments & tutorials

| Script | What it does |
|---|---|
| `transfer_experiment_seed_paper.py` | Main script to reproduce all paper experiments |
| `custom_block_transfer_experiment_seed_paper.py` | Transfer experiment variant with custom blocking |
| `sota.py` | Compute state-of-the-art results table from the paper |
| `tutorial_for_predicting_with_the_prod_model.py` | Guide to using the released production model (supports `--use-rust`) |
| `tutorial.ipynb` | Notebook walkthrough of the S2AND pipeline |

### Dataset creation & preprocessing

| Script | What it does |
|---|---|
| `full_model_dump.py` | Train and dump a full model on all datasets (includes unreleased data) |
| `make_s2and_mini_dataset.py` | Create a smaller dataset for faster iteration (skips medline) |
| `make_s2and_name_tuples.py` | Create name tuples file of known aliases |
| `make_triplets.py` | Generate training triplets |
| `make_inventors_s2and_subset.py` | Create inventors S2AND subset |
| `make_inventors_split_and_histograms.py` | Split inventors data and generate histograms |
| `generate_inventors_hf_specter_embeddings.py` | Generate SPECTER embeddings for inventors dataset |
| `make_classification_style.py` | Convert data to classification-style format |
| `LLM_based_filtering_of_name_tuples.py` | Filter name tuples using Gemini 2.5 Pro (costs money to re-run) |
| `get_name_counts.py` | Documentation for how name counts metadata was collected (internal data) |
| `get_orcid_name_prefix_counts.py` | Documentation for how ORCID prefix counts were collected (internal data) |
| `bench_paper_preprocess_pool.py` | Benchmark threads vs processes for paper preprocessing |

### Testing

| Script | What it does |
|---|---|
| `test_specter2.py` | Test SPECTER2 embedding integration |
| `test_inventors_s2and.py` | Test inventors S2AND dataset loading and eval |

### CI & release

| Script | What it does |
|---|---|
| `run_ci_locally.py` | Run full CI locally: Rust extension compile (`maturin develop`), ruff, ty, pytest |
| `sync_version.py` | Sync VERSION file into pyproject.toml + Cargo.toml |

### Internal / archived

| Script | What it does |
|---|---|
| `internal/transfer_experiment_internal.py` | Full-scale transfer experiment with unreleased datasets (supports Rust backend) |
| `internal/make_augmentation_dataset_a.py` | Create augmentation dataset step 1 (unreleased data) |
| `internal/make_augmentation_dataset_b.py` | Create augmentation dataset step 2 (unreleased data) |
| `internal/test_s2aff.py` | Test S2 affiliation matching (internal) |
| `archive/blog_post_eval.py` | Min edit distance numbers for blog post (Python-only legacy) |
| `archive/claims_cluster_eval.py` | Evaluate on S2 corrections data (Python-only legacy) |
| `archive/transform_all_datasets.py` | Transform old dataset format to final |
| `archive/make_claims_dataset.py` | Create S2 corrections evaluation dataset (internal data) |

## Notes

**`transfer_experiment_seed_paper.py`**: Assumes S2AND data is in `<code root>/data/`. If not, modify `"main_data_dir"` in `data/path_config.json`. If you have limited RAM, don't use `--use_cache` — it's slower without the cache but won't try to fit all feature data into memory.


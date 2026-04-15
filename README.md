# S2AND
This repository provides access to the S2AND dataset and S2AND reference model described in the paper [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421) by Shivashankar Subramanian, Daniel King, Doug Downey, Sergey Feldman.

The reference model is live on semanticscholar.org, and the trained model is available now as part of the data download (see below).

---

## Table of Contents
- [Installation](#installation)
- [Data](#data)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Using the Production Model](#using-the-production-model)
- [Training a Model](#training-a-model)
- [Predicting with a Saved Model](#predicting-with-a-saved-model)
- [Advanced Topics](#advanced-topics)
- [Development](#development)
- [Reproducibility](#reproducibility)
- [Licensing](#licensing)
- [Citation](#citation)

---

## Installation

### Prerequisites (one-time)
Clone the repo.

Install `uv` using the official guide:
- [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/)

Install Rust (needed to build the native extension from source):
- [Rust installation docs](https://www.rust-lang.org/tools/install)

If you are building the Rust extension, install OS prerequisites:

```bash
# Ubuntu / Debian / WSL2
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libgomp1
```

```powershell
# Windows (one-time)
# Install Visual Studio Build Tools with the "Desktop development with C++" workload.
```

Verify toolchain availability:

```bash
uv --version
rustc --version
cargo --version
```

WSL notes:
- Some Ubuntu images do not provide a `python` alias by default; use `python3` for system Python commands.
- On PEP 668-managed systems, `python3 -m pip install --user ...` may fail with `externally-managed-environment`; use one of the official `uv` install methods above.

### Setup

1. From repo root:

```bash
# create the project venv (uv defaults to .venv if you don't give a name)
# use Python 3.11.x (fasttext doesn't support 3.12+ here)
uv venv --python 3.11.13
```

2. Activate the venv (choose one):

```bash
# macOS / Linux (bash / zsh)
source .venv/bin/activate

# Windows PowerShell
. .venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

3. Runtime install (end users, pick one):

```bash
# default runtime is Python (`auto` resolves to Python).
uv pip install s2and
# optional: Rust-enabled runtime when extension wheels are available.
uv pip install "s2and[rust]"
```

4. Developer install (repo checkout):

```bash
# prefer uv --active so uv uses your activated environment
uv sync --active --extra dev
```

5. (Recommended) Build/install the Rust extension into the active venv:

```bash
# requires Rust toolchain on PATH (rustc/cargo)
# if Rust was just installed via rustup in this shell:
# source "$HOME/.cargo/env"
uv run --active --no-project maturin develop -m s2and_rust/Cargo.toml
```

Notes:
- This installs the native module into site-packages so imports use the compiled extension.
- If you don't want an editable install, you can `uv pip install .` instead of `uv sync`, then run the
  `maturin develop` step above.
- Once wheels are published, you can install the native extension via extras:
  `uv pip install "s2and[rust]"`.
- On WSL with repo paths mounted from Windows (for example, `/mnt/c/...`), `uv` may warn about failed hardlinks.
  To suppress this and avoid repeated warnings, set `UV_LINK_MODE=copy` before `uv sync` / `uv pip install`.

---

## Data
To obtain the S2AND dataset, run the following command after the package is installed (from inside the `S2AND` directory).
Expected download size is **~50.4 GiB**.

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release data/
```

Note that this software package comes with tools specifically designed to access and model the dataset.

If you only need the production model (without the full dataset), you can download just the model pickle:

```bash
aws s3 cp --no-sign-request s3://ai2-s2-research-public/s2and-release/production_model_v1.1.pickle data/
```

## Configuration
Modify the config file at `data/path_config.json`. This file should look like this
```
{
    "main_data_dir": "absolute path to wherever you downloaded the data to",
    "internal_data_dir": "ignore this one unless you work at AI2"
}
```
As the dummy file says, `main_data_dir` should be set to the location of wherever you downloaded the data to, and
`internal_data_dir` can be ignored, as it is used for some scripts that rely on unreleased data, internal to Semantic Scholar.

---

## Quick Start

Run a bundled example with the `tests/qian` fixture. You still need the production model pickle — download it first if you haven't already (see [Data](#data)):

```bash
uv run --no-project python scripts/tutorial_for_predicting_with_the_prod_model.py \
  --use-rust 1 \
  --dataset qian \
  --data-root tests \
  --load-name-counts 0
```

Run the same tutorial on `data/s2and_mini` (after downloading the full dataset):

```bash
uv run --no-project python scripts/tutorial_for_predicting_with_the_prod_model.py --use-rust 1 --dataset qian
```

When running scripts from the repo, prefer `uv run --no-project` so the installed packages (including the Rust extension)
resolve from site-packages. Avoid setting `PYTHONPATH` to the repo root, which can shadow the compiled module.

---

## Using the Production Model

We provide trained production models in the S3 bucket along with the datasets:

| Model file | Status | Embeddings | Uses reference features? |
|---|---|---|---|
| **`production_model_v1.2.pickle`** | **Current** (used on Semantic Scholar website and API) | SPECTER2 [PRX] | No |
| `production_model_v1.1.pickle` | Previous | SPECTER1 | No |
| `production_model_v1.0.pickle` | Deprecated | SPECTER1 | Yes |

To see a full example, see `scripts/tutorial_for_predicting_with_the_prod_model.py`. You can also use it on your own data, as long as it is formatted the same way as the S2AND data. SPECTER embeddings for papers are available via the [Semantic Scholar API](https://api.semanticscholar.org/) (use the `embedding.specter_v2` field for v1.2, or `embedding.specter_v1` for v1.1).

### What "does not use reference features" means

The production models v1.1 and v1.2 are trained with `compute_reference_features=False`. This means they do **not** use any features derived from a paper's bibliography (cited references). Specifically, the following six features are disabled and filled with NaN at inference time:

- `references_authors_overlap` — overlap of author names across referenced papers
- `references_titles_overlap` — overlap of titles of referenced papers
- `references_venues_overlap` — overlap of venues/journals of referenced papers
- `references_author_blocks_jaccard` — Jaccard similarity of author blocks from references
- `references_self_citation` — whether one paper cites the other
- `references_overlap` — Jaccard similarity of referenced paper IDs

**What you can leave out of your input data** when using these models:

In **`papers.json`**, the `references` field can be set to `null` or omitted entirely. It is only used to compute the six reference features above. Example minimal paper entry:

```json
{
  "paper_id": 12345,
  "title": "My Paper Title",
  "abstract": "Optional but recommended for the has_abstract feature.",
  "year": 2023,
  "venue": "Conference Name",
  "journal_name": "Journal Name",
  "authors": [
    {"position": 0, "author_name": "Jane Smith"},
    {"position": 1, "author_name": "John Doe"}
  ],
  "references": null
}
```

In **`signatures.json`**, all fields are still needed regardless of whether reference features are used. No signature fields relate to references. Example minimal signature entry:

```json
{
  "signature_id": "0",
  "paper_id": 12345,
  "author_info": {
    "position": 0,
    "block": "j smith",
    "first": "Jane",
    "middle": null,
    "last": "Smith",
    "suffix": null,
    "email": null,
    "affiliations": ["University of Example"]
  }
}
```

> **Note:** The deprecated v1.0 model *does* use reference features, so if you use that model you must populate the `references` field with a list of cited paper IDs.

### Name-count semantics compatibility

S2AND currently supports two runtime semantics for the name-count feature key used by
`last_first_initial_count_min`:

- `legacy_full_first_token`: key is `<last> <first_token>` (historical behavior).
- `initial_char`: key is `<last> <first[0]>` (current intended semantics).

Model compatibility rules:

- `production_model_v1.1.pickle` and `production_model_v1.2.pickle` were trained with
  `legacy_full_first_token`.
- In `ANDData(..., mode="inference")`, prediction automatically applies the semantics expected by
  the loaded model via `clusterer.feature_contract["name_counts_last_first_initial_semantics"]`
  (with `featurizer_version` fallback for older artifacts).
- Do not mix model artifacts and feature semantics without retraining, because this changes model
  inputs and can materially change clustering output.

---

## Training a Model

Once you have downloaded the datasets, you can go ahead and load up one of them:

```python
from os.path import join
from s2and.data import ANDData

dataset_name = "pubmed"
parent_dir = f"data/{dataset_name}"
dataset = ANDData(
    signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
    papers=join(parent_dir, f"{dataset_name}_papers.json"),
    mode="train",
    specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
    clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
    block_type="s2",
    train_pairs_size=100000,
    val_pairs_size=10000,
    test_pairs_size=10000,
    name=dataset_name,
    n_jobs=8,
)
```

This may take a few minutes - there is a lot of text pre-processing to do.

The first step in the S2AND pipeline is to specify a featurizer and then train a binary classifier
that tries to guess whether two signatures are referring to the same person.

We'll do hyperparameter selection with the validation set and then get the test area under ROC curve.

Here's how to do all that:

```python
from s2and.model import PairwiseModeler
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.eval import cluster_eval, pairwise_eval

featurization_info = FeaturizationInfo()
# the cache will make it faster to train multiple times - it stores the features on disk for you
train, val, test = featurize(dataset, featurization_info, n_jobs=8, use_cache=True)
X_train, y_train = train
X_val, y_val = val
X_test, y_test = test

# calibration fits isotonic regression after the binary classifier is fit
# monotone constraints help the LightGBM classifier behave sensibly
pairwise_model = PairwiseModeler(
    n_iter=25, calibrate=True, monotone_constraints=featurization_info.lightgbm_monotone_constraints
)
# this does hyperparameter selection, which is why we need to pass in the validation set.
pairwise_model.fit(X_train, y_train, X_val, y_val)

# this will also dump a lot of useful plots (ROC, PR, SHAP) to the figs_path
pairwise_metrics = pairwise_eval(X_test, y_test, pairwise_model.classifier, figs_path='figs/', title='example')
print(pairwise_metrics)
```

The second stage in the S2AND pipeline is to tune hyperparameters for the clusterer on the validation data
and then evaluate the full clustering pipeline on the test blocks.

We use agglomerative clustering as implemented in `fastcluster` with average linkage.
There is only one hyperparameter to tune.

```python
from s2and.model import Clusterer, FastCluster
from hyperopt import hp

clusterer = Clusterer(
    featurization_info,
    pairwise_model,
    cluster_model=FastCluster(linkage="average"),
    search_space={"eps": hp.uniform("eps", 0, 1)},
    n_iter=25,
    n_jobs=8,
)
clusterer.fit(dataset)

# the metrics_per_signature are there so we can break out the facets if needed
metrics, metrics_per_signature = cluster_eval(dataset, clusterer)
print(metrics)
```

For a fuller example, please see the transfer script: `scripts/transfer_experiment_seed_paper.py`.

## Predicting with a Saved Model
Assuming you have a clusterer already fit, you can dump the model to disk like so
```python
import pickle

with open("saved_model.pkl", "wb") as _pkl_file:
    pickle.dump(clusterer, _pkl_file)
```

You can then reload it, load a new dataset, and run prediction
```python
import pickle

with open("saved_model.pkl", "rb") as _pkl_file:
    clusterer = pickle.load(_pkl_file)

anddata = ANDData(
    signatures=signatures,
    papers=papers,
    specter_embeddings=paper_embeddings,
    name="your_name_here",
    mode="inference",
    block_type="s2",
)
pred_clusters, pred_distance_matrices = clusterer.predict(anddata.get_blocks(), anddata)
# pred_distance_matrices can be None when using memory-optimized fused clustering
```

---

## Advanced Topics

### Rust featurizer (runtime backend)
S2AND backend selection is controlled by `S2AND_BACKEND`:
- `auto` (default) — uses Rust when available and capable, otherwise Python
- `rust` — strict Rust mode; fails fast on Rust-stage errors
- `python` — Python-only path; zero Rust calls

For the full list of environment variables, see [docs/environment.md](docs/environment.md).

Install contract:
- `uv pip install s2and`: Python-only runtime.
- `uv pip install "s2and[rust]"`: Rust-enabled runtime.
- Full runtime contract: [docs/rust/runtime.md](docs/rust/runtime.md).

Notes:
- Rust batch mode uses Rayon internally for parallelism; Python process pools are not used.
- When Rust is enabled, signature n-gram Counters may be deferred and computed natively during Rust featurizer construction.
- If a Python code path needs eager n-gram Counters, call `ANDData.materialize_signature_ngrams_python()`.

### Cache policy
- Default: `use_cache=False` (no caching).
- `use_cache=True`: enables Python pair-feature cache and Rust featurizer cache.
- Cache root: `S2AND_CACHE` env var (defaults to `~/.s2and`).
- Enable caching explicitly when you want cached reruns; disable it when validating feature changes or running one-shot experiments.

Pre-warm once at server start:

```python
from s2and.feature_port import warm_rust_featurizer
warm_rust_featurizer(dataset, use_cache=True)
```

### Large-scale inference with subblocking

For processing massive blocks (hundreds of thousands of signatures), use the Rust backend with
subblocking to keep memory bounded. This is the recommended production setup.

#### Standard prediction with subblocking

Use `predict()` with `batching_threshold` to automatically split large blocks into manageable subblocks:

```python
import os

# 1. Force Rust backend (set before importing s2and modules)
os.environ["S2AND_BACKEND"] = "rust"

from s2and.data import ANDData
from s2and.feature_port import warm_rust_featurizer
from s2and.serialization import load_pickle_with_verified_label_encoder_compat

# 2. Load the production model
clusterer = load_pickle_with_verified_label_encoder_compat(
    "data/production_model_v1.2.pickle"
)["clusterer"]
clusterer.use_cache = False  # disable caching for one-shot inference
clusterer.n_jobs = 8

# 3. Load your dataset in inference mode
dataset = ANDData(
    signatures="path/to/signatures.json",
    papers="path/to/papers.json",
    specter_embeddings="path/to/specter.pickle",
    mode="inference",
    block_type="s2",
    n_jobs=8,
    name="my_dataset",
)

# 4. (Optional) Pre-warm Rust featurizer to reduce cold-start latency
warm_rust_featurizer(dataset, use_cache=False)

# 5. Predict clusters with subblocking for large blocks
pred_clusters, _ = clusterer.predict(
    dataset.get_blocks(),
    dataset,
    batching_threshold=5000,  # blocks larger than this are split into subblocks
    desired_memory_use=5000 * 5000,  # memory budget in signature-pairs (25M pairs here)
)

# pred_clusters is a dict mapping signature_id -> list of cluster member signature_ids
print(f"Total clusters: {len(pred_clusters)}")
```

Key parameters for `predict()`:
- `batching_threshold`: blocks larger than this are split via `make_subblocks()` before clustering
- `desired_memory_use`: memory budget in signature-pair units; controls chunk sizing for subblocked incremental paths (default: `batching_threshold²`)

#### Incremental prediction (adding new signatures to existing clusters)

Use `predict_incremental()` when you have existing clusters (`cluster_seeds`) and want to assign
new signatures without reclustering everything:

```python
# Load dataset with existing cluster seeds
dataset = ANDData(
    signatures="path/to/signatures.json",
    papers="path/to/papers.json",
    specter_embeddings="path/to/specter.pickle",
    mode="inference",
    block_type="s2",
    n_jobs=8,
    name="my_dataset",
    cluster_seeds={
        "require": {
            "block_key": {("sig1", "sig2"): 1.0, ...},  # pairs that must cluster together
        },
        "disallow": {
            "block_key": {("sig3", "sig4"), ...},  # pairs that must NOT cluster together
        },
    },
)

# Run incremental prediction on one block
blocks = dataset.get_blocks()
block_key = "j smith"  # target block
block_signatures = blocks[block_key]

result = clusterer.predict_incremental(
    block_signatures,
    dataset,
    batching_threshold=5000,         # subblock size cap for phase-split mode
    total_ram_bytes=32 * 1024**3,    # explicit RAM budget (32 GB)
    max_chunk_pairs=50_000_000,      # Phase A chunk cap (None=use default, 0=unlimited)
)

clusters = result["clusters"]
phase_b_mode = result.get("phase_b_mode", "N/A")

# phase_b_mode indicates how phase-split handled memory:
# - "exact": ran Phase B globally (monolithic-equivalent behavior)
# - "subblock_local": ran Phase B per-subblock (memory-bounded approximation)
print(f"Clusters: {len(set(clusters.values()))}, mode={phase_b_mode}")
```

There is also a `predict_incremental` function on the `Clusterer`, that allows prediction for just a small set of *new* signatures. When instantiating `ANDData`, you can pass in `cluster_seeds`, which will be used instead of model predictions for those signatures. If you call `predict_incremental`, the full distance matrix will not be created, and the new signatures will simply be assigned to the cluster they have the lowest average distance to, as long as it is below the model's `eps`, or separately reclustered with the other unassigned signatures, if not within `eps` of any existing cluster.

For very large incremental blocks, phase-split mode is used automatically when subblocking is active (i.e., when `batching_threshold` is set and the block exceeds it). Phase-split subblocks Phase A and then:
- runs Phase B globally when it fits budget (`phase_b_mode="exact"`),
- auto-falls back to subblock-local B/C/D when over budget (`phase_b_mode="subblock_local"`).

`predict_incremental` returns a payload with:
- `clusters`
- `phase_b_mode`
- `phase_b_budget_bytes`
- `phase_b_required_bytes`

RAM policy:
- Preferred: pass `total_ram_bytes=<int>` directly to `predict_incremental`.
- If omitted, runtime auto-detects RAM (cgroup first, then host probes) and applies a `0.8` safety factor before deriving budgets.

For detailed subblocking behavior, see `docs/subclustering.md`.

### Controlling RAM usage

S2AND provides two primary knobs for controlling peak memory consumption, plus several secondary knobs that interact with them.

#### `total_ram_bytes` — explicit RAM budget (bytes)

Pass this to `predict_incremental()` or `many_pairs_featurize()` to tell S2AND how much physical RAM is available. The system uses it to derive chunk sizes, accumulator limits, and Rust batch plans that stay within budget.

```python
result = clusterer.predict_incremental(
    block_signatures,
    dataset,
    total_ram_bytes=16 * 1024**3,  # 16 GiB
)
```

If omitted, the runtime auto-detects RAM (cgroup limits first, then host probes) and applies two sequential reductions: first a **0.8× safety factor** on the detected total, then a **10% safety margin** (plus current RSS) is subtracted to compute the usable budget. Together these mean the effective budget is roughly 72% of detected RAM minus current process memory. You can always override with an explicit value — useful in containers where cgroup detection may return the host's total RAM instead of the container's limit.

#### `max_chunk_pairs` — Phase A chunk size cap (pairs)

Pass this to `predict()` or `predict_incremental()` to set an explicit hard cap on the Phase A chunk size (the number of signature pairs processed in one featurization batch). This overrides the default `PHASE_A_MAX_CHUNK_PAIRS_DEFAULT = 100_000_000`.

```python
# With predict_incremental
result = clusterer.predict_incremental(
    block_signatures,
    dataset,
    total_ram_bytes=200 * 1024**3,
    max_chunk_pairs=50_000_000,
)

# With predict (when batching is enabled, affects single-letter subblocks)
pred_clusters, _ = clusterer.predict(
    block_dict,
    dataset,
    batching_threshold=5000,
    total_ram_bytes=200 * 1024**3,
    max_chunk_pairs=50_000_000,
)
```

**Common values**:
- `None` (default): Uses `PHASE_A_MAX_CHUNK_PAIRS_DEFAULT = 100_000_000`
- `100_000_000`: 100M pairs
- `50_000_000`: 50M pairs
- `10_000_000`: 10M pairs
- `0`: Unlimited

The actual chunk size is the minimum of:
1. Memory-budget-derived limit (always present — from explicit `total_ram_bytes` or auto-detected RAM)
2. `max_chunk_pairs` cap (`PHASE_A_MAX_CHUNK_PAIRS_DEFAULT` = 100M when `None`, caller value when > 0, no cap when `0`)

Use this when:
- You want a hard cap regardless of detected RAM
- You want to disable the cap (`0`) and rely purely on memory budget calculations

#### `train_pairs_size` — number of training tuples

Controls how many signature pairs are sampled for training the pairwise classifier (in `ANDData`). Each pair produces a feature vector held in memory, so this directly determines the size of the training feature matrix.

```python
dataset = ANDData(
    ...,
    mode="train",
    train_pairs_size=100000,   # default: 30000
    val_pairs_size=10000,
    test_pairs_size=10000,
)
```

Lowering `train_pairs_size` reduces peak RAM during training at the cost of potentially fewer training examples. Raising it increases memory usage proportionally.

#### How they interact

| Knob | Phase | What it controls |
|---|---|---|
| `train_pairs_size` | Training | Number of sampled pairs → size of feature matrix in RAM |
| `total_ram_bytes` | Inference (incremental / Rust batch) | Chunk sizes and accumulator limits for memory-bounded featurization |
| `max_chunk_pairs` | Inference (`predict` + `predict_incremental`) | Hard cap on Phase A chunk size; overrides `PHASE_A_MAX_CHUNK_PAIRS_DEFAULT` (100M) |
| `batch_size` (on `Clusterer`, default `1_000_000`) | Inference (standard predict) | Max pairs featurized per chunk; lower = less peak RAM but slower |
| `n_jobs` (on `ANDData` / `Clusterer`) | Both | Parallelism level; more jobs = more concurrent memory |
| `batching_threshold` (on `predict` / `predict_incremental`) | Inference | Block-size cap before subblocking kicks in; controls per-block pair count |
| `desired_memory_use` (on `predict`) | Inference | Memory budget in signature-pair units for subblocked paths (default: `batching_threshold²`) |

**During training**, `train_pairs_size` is the main lever. `total_ram_bytes` is not used during training — the feature matrix is built in one shot from the sampled pairs.

**During inference**, `total_ram_bytes` is the primary lever. When set, the runtime derives:
- **Chunk pairs**: how many pairs to featurize per chunk (bounded by available bytes ÷ bytes-per-pair).
- **Accumulator limits**: how many entries the incremental Phase A accumulator can hold before early-stopping.
- **Rust batch plan**: chunk sizing for the Rust featurization backend.

If you are running out of memory during inference, try (in order):
1. Set `total_ram_bytes` to a value smaller than your actual RAM (e.g., 50–75% of physical RAM).
2. Lower `batch_size` on the `Clusterer` (e.g., `clusterer.batch_size = 100_000`).
3. Lower `n_jobs` to reduce parallel memory pressure.
4. Use `batching_threshold` to force subblocking on large blocks.

If you are running out of memory during training:
1. Lower `train_pairs_size` (e.g., from `100000` to `30000`).
2. Lower `n_jobs`.

### Profiling

```bash
S2AND_BACKEND=rust uv run --no-project python scripts/rust_suite.py prod-inference \
  --dataset-name qian \
  --data-root tests \
  --n-jobs 4
```

Benchmark baseline ownership:
- Active Rust runtime gate baselines and promotion rules: `docs/rust/baselines.md`

---

## Development

### Running tests

```bash
uv run --no-project pytest tests/
```

To run the entire CI suite mimicking the GH Actions:
```bash
uv run python scripts/run_ci_locally.py
```
`scripts/run_ci_locally.py` mirrors `.github/workflows/main.yaml` by running:
- lint job (`ruff check` + `ruff format --check`)
- `typecheck-and-test` matrix lanes (`py-only`, then `rust-enabled`)
- Rust parity guardrail tests in the `rust-enabled` lane

By default, local `ty` checks use `--python-version 3.11 --python-platform linux` to match GitHub Linux runners.
To override platform emulation locally, set `S2AND_CI_TY_PLATFORM` (for example, `windows`).

To run CI checks locally without Rust extension compilation (faster iteration):
```bash
uv sync --active --extra dev --frozen
uv run --active --no-project ruff format --check s2and scripts/*.py
uv run --active --no-project ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global
uv run --active --no-project ty check scripts/*.py --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global --ignore unresolved-reference --ignore unresolved-attribute
# macOS/Linux:
PYTHONPATH=. uv run --active --no-project pytest tests/ --cov=s2and --cov-report=term-missing --cov-fail-under=40
# Windows PowerShell:
$env:PYTHONPATH='.'; uv run --active --no-project pytest tests/ --cov=s2and --cov-report=term-missing --cov-fail-under=40
```

### Version bumping
Versioning is centralized in the `VERSION` file (single source of truth). When you update it, we sync the Python/Rust
manifests and regenerate lockfiles.

One-time setup for hooks (recommended):
```bash
git config core.hooksPath .githooks
```

Workflow:
```bash
# 1) edit VERSION
echo 0.48.0 > VERSION

# 2) sync manifests
uv run python scripts/sync_version.py

# 3) regenerate lockfiles
uv sync --extra dev
uv run --active --no-project cargo generate-lockfile --manifest-path s2and_rust/Cargo.toml
```

Notes:
- The pre-commit hook only runs when `VERSION` is staged and will auto-sync + regenerate lockfiles if needed.
- `uv.lock` and `s2and_rust/Cargo.lock` are generated files and will contain the version after syncing.

### Docs

- Index (start here): `docs/README.md`
- Next steps: `docs/work_plan.md`
- Backlog: `docs/work_plan.md` (Backlog section)

---

## Reproducibility
The experiments in the paper were run with the python (3.7.9) package versions in `paper_experiments_env.txt`, in the branch `s2and_paper`.

To install, run:
```bash
git checkout s2and_paper
pip install pip==21.0.0
pip install -r paper_experiments_env.txt --use-feature=fast-deps --use-deprecated=legacy-resolver
```

Then, rerunning `scripts/paper_experiments.sh` on the branch `s2and_paper` should produce the same numbers as in the paper (we will update here if this becomes not true).

Our trained, released models are in the `s3` folder referenced above, and are called `production_model.pickle` (the original paper-era model, which does not compute reference features; see [Using the Production Model](#using-the-production-model) for the current versioned models) and `full_union_seed_*.pickle` (models trained during benchmark experiments). They can be loaded the same way as in the section above called "[Predicting with a Saved Model](#predicting-with-a-saved-model)", except that the pickled object is a *dictionary*, with a `clusterer` key. *Important*: these pickles will only run on the branch `s2and_paper` and not on main.

## Licensing
The code in this repo is released under the Apache 2.0 license. The dataset is released under ODC-BY (included in S3 bucket with the data). We would also like to acknowledge that some of the affiliations data comes directly from the Microsoft Academic Graph (https://aka.ms/msracad).

## Citation

If you use S2AND in your research, please cite [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421).

```
@inproceedings{subramanian2021s2and,
      title={{S}2{AND}: {A} {B}enchmark and {E}valuation {S}ystem for {A}uthor {N}ame {D}isambiguation},
      author={Subramanian, Shivashankar and King, Daniel and Downey, Doug and Feldman, Sergey},
      year={2021},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      booktitle = {{JCDL} '21: Proceedings of the {ACM/IEEE} Joint Conference on Digital Libraries in 2021},
      series = {JCDL '21}
}
```

S2AND is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.

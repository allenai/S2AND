# S2AND

S2AND provides the S2AND author-name-disambiguation benchmark datasets and the reference model described in the paper [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421) by Shivashankar Subramanian, Daniel King, Doug Downey, and Sergey Feldman.

The repository supports both Python-only use and a Rust-accelerated runtime for the expensive inference and featurization paths.

## What S2AND Provides

- The S2AND datasets used for author name disambiguation research.
- Versioned production model artifacts used by Semantic Scholar.
- Training, evaluation, and inference APIs in Python.
- An optional Rust backend for faster runtime on supported installs.

## Choose a Workflow

| Use case | Start here | Details |
| --- | --- | --- |
| Run the released model on your own data | [Quick Start](#quick-start) | [docs/production_inference.md](docs/production_inference.md) |
| Download the benchmark datasets | [Download Data or Model](#download-data-or-model) | [docs/data.md](docs/data.md) |
| Train or evaluate a model | [Training and Evaluation Essentials](#training-and-evaluation-essentials) | [docs/training.md](docs/training.md) |
| Build a production release bundle | `scripts/production/` | [docs/production_inference.md](docs/production_inference.md) |
| Operate Rust-backed large-scale inference | [Runtime and Scaling](#runtime-and-scaling) | [docs/rust/runtime.md](docs/rust/runtime.md), [docs/subclustering.md](docs/subclustering.md), [docs/threading.md](docs/threading.md) |
| Work on the repo itself | [Development](#development) | [docs/development.md](docs/development.md) |

## Install

S2AND currently targets Python 3.11.x.

Package install:

```bash
uv pip install s2and
uv pip install "s2and[rust]"
```

Both package installs include the production model files as package data. You do
not need Git LFS or a separate model download when installing from PyPI.

Repo checkout:

```bash
git lfs install
git lfs pull
uv venv --python 3.11.13
# activate the environment, then:
uv sync --active --extra dev
uv run --active --no-project maturin develop -m s2and_rust/Cargo.toml
```

Source checkouts use Git LFS for versioned model artifacts, including the bundled
production model directory and legacy production pickle files. Run `git lfs pull`
after cloning and after switching branches that change model artifacts. Small
pointer files in `s2and/data/production_model_*` mean the LFS files were not
hydrated.

The Rust build step is optional and only needed when you want the native extension from source. For OS prerequisites, activation commands, WSL notes, and install variants, see [docs/install.md](docs/install.md).

## Download Data or Model

Full dataset download:

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release s2and/data/
```

Expected size is about `55.5 GiB`.

The current production model bundle is checked into `s2and/data/production_model_v1.21/`
and is included in package data. You do not need a separate model download for
prediction.

Starting with S2AND `0.50.0`, production releases are native
`production_model_vX.Y/` directories tracked through Git LFS, not pickle files.
Release bundles are built with `scripts/production/model/train_pairwise.py`
followed by `scripts/production/model/train_linker_and_finalize.py`; the final
bundle includes linker artifacts when production inference needs them.

## Configuration

Modify the config file at `s2and/data/path_config.json` (or set the `S2AND_PATH_CONFIG` env var to point elsewhere). This file should look like this:

```json
{
  "main_data_dir": "absolute path to your downloaded S2AND data",
  "internal_data_dir": ""
}
```

More on dataset layout, config, and model-only usage: [docs/data.md](docs/data.md).

## Quick Start

This uses the bundled `tests/qian` fixture, so you do not need the full S2AND dataset or a model download:

```bash
uv run --no-project python scripts/tutorial_for_predicting_with_the_prod_model.py \
  --use-rust 1 \
  --dataset qian \
  --data-root tests \
  --load-name-counts 0
```

When running repo scripts, prefer `uv run --no-project` so imports resolve from the installed packages and compiled extension in `site-packages`. Avoid setting `PYTHONPATH` to the repo root for scripts because it can shadow the compiled module. Test commands may still intentionally exercise the checkout source tree.

## Production Inference Essentials

### Which model to use

| Model artifact | Release line | Repo storage | Included in PyPI install? | Linker artifact | Loader | Embeddings | Uses reference features? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `production_model_v1.21/` | Current, starting with `0.50.0` | Directory bundle in Git LFS | Yes | Bundled in `incremental_linker/` | `load_production_model(...)` | SPECTER2 PRX | No |
| `production_model_v1.2.pickle` | Legacy, pre-`0.50.0` | Pickle in Git LFS | Yes | Not bundled | Legacy pickle loader only | SPECTER2 PRX | No |
| `production_model_v1.1.pickle` | Legacy, pre-`0.50.0` | Pickle in Git LFS | Yes | Not bundled | Legacy pickle loader only | SPECTER1 | No |
| `production_model_v1.0.pickle` | Deprecated, pre-`0.50.0` | Pickle in Git LFS | Yes | Not bundled | Legacy pickle loader only | SPECTER1 | Yes |

Key points:

- `production_model_v1.21/` is the current recommended model. It bundles the v1.2 pairwise model and the promoted Rust incremental linker.
- Starting with S2AND `0.50.0`, production model releases are directory bundles named `production_model_vX.Y/`; new production releases should not be published as pickle files.
- Git LFS is only a source-checkout concern. Published `s2and` wheels and sdists include the hydrated model files.
- Use directory bundles for workflows that need a linker model. The legacy `v1.0`, `v1.1`, and `v1.2` pickle artifacts contain only the legacy pickled model state and do not bundle `incremental_linker/` artifacts.
- Models `v1.1`, `v1.2`, and `v1.21` were trained with `compute_reference_features=False`.
- For `v1.1`, `v1.2`, and `v1.21`, `papers.references` can be omitted or set to `null`.
- `v1.0` still requires reference features and is kept only for backward compatibility.

Minimal input shape for `v1.1`, `v1.2`, and `v1.21`:

```json
{
  "paper_id": 12345,
  "title": "My Paper Title",
  "abstract": "Optional but useful.",
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

Minimal prediction example:

```python
from s2and.data import ANDData
from s2and.production_model import load_production_model

clusterer = load_production_model("s2and/data/production_model_v1.21")

dataset = ANDData(
    signatures="path/to/signatures.json",
    papers="path/to/papers.json",
    specter_embeddings="path/to/specter_embeddings.pkl",
    mode="inference",
    block_type="s2",
    n_jobs=8,
    name="my_dataset",
)

pred_clusters, _ = clusterer.predict(dataset.get_blocks(), dataset)
```

SPECTER embeddings can be sourced from the Semantic Scholar API. Use `embedding.specter_v2` with `v1.21`/`v1.2` and `embedding.specter_v1` with `v1.1`.

Inference-mode `ANDData(..., mode="inference")` automatically applies the name-count semantics expected by the loaded model artifact. Full inference details, large-block examples, and compatibility notes are in [docs/production_inference.md](docs/production_inference.md).

## Training and Evaluation Essentials

Minimal training flow:

```python
from os.path import join

from hyperopt import hp

from s2and.data import ANDData
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.model import Clusterer, FastCluster, PairwiseModeler

dataset_name = "pubmed"
parent_dir = f"s2and/data/{dataset_name}"

dataset = ANDData(
    signatures=join(parent_dir, f"{dataset_name}_signatures.json"),
    papers=join(parent_dir, f"{dataset_name}_papers.json"),
    clusters=join(parent_dir, f"{dataset_name}_clusters.json"),
    specter_embeddings=join(parent_dir, f"{dataset_name}_specter.pickle"),
    mode="train",
    block_type="s2",
    train_pairs_size=100000,
    val_pairs_size=10000,
    test_pairs_size=10000,
    n_jobs=8,
    name=dataset_name,
)

featurization_info = FeaturizationInfo()
train, val, test = featurize(dataset, featurization_info, n_jobs=8, use_cache=True)
X_train, y_train = train
X_val, y_val = val

pairwise_model = PairwiseModeler(
    n_iter=25,
    calibrate=True,
    monotone_constraints=featurization_info.lightgbm_monotone_constraints,
)
pairwise_model.fit(X_train, y_train, X_val, y_val)

clusterer = Clusterer(
    featurization_info,
    pairwise_model,
    cluster_model=FastCluster(linkage="average"),
    search_space={"eps": hp.uniform("eps", 0, 1)},
    n_iter=25,
    n_jobs=8,
)
clusterer.fit(dataset)
```

For evaluation, model serialization, and fuller scripts such as `scripts/transfer_experiment_seed_paper.py`, see [docs/training.md](docs/training.md).

## Runtime and Scaling

Runtime controls:

- `S2AND_BACKEND=auto` is the default. It uses Rust when available and capable, otherwise Python.
- `S2AND_BACKEND=rust` is strict Rust mode and fails fast on Rust-stage errors.
- `S2AND_BACKEND=python` disables Rust entirely.

Cache behavior:

- `use_cache=False` skips persistent pair-feature caching and Rust featurizer disk-cache reads and writes.
- `use_cache=True` enables the SQLite-backed pair-feature cache and Rust featurizer disk cache under `S2AND_CACHE`.
- Same-process Rust featurizer reuse remains available even when `use_cache=False`.

Large blocks:

- `predict(..., batching_threshold=...)` uses subblocking to keep full-block work bounded.
- `predict_incremental(..., batching_threshold=...)` uses promoted Rust query batching when the Rust backend is active and cluster seeds are available. The Python fallback rejects `batching_threshold`; pass `None` or use the promoted Rust route.
- Incremental results still include `phase_b_mode`; current supported routes report `exact`.
- `total_ram_bytes` is the main memory-control knob for large inference jobs.

Concurrency:

- Treat `n_jobs` as the main concurrency knob for a run.
- Set thread-related environment variables before importing heavy compute libraries.

Details:

- Runtime contract: [docs/rust/runtime.md](docs/rust/runtime.md)
- Cache semantics: [docs/caching.md](docs/caching.md)
- Threading guidance: [docs/threading.md](docs/threading.md)
- Subblocking and memory tradeoffs: [docs/subclustering.md](docs/subclustering.md)
- Environment variables: [docs/environment.md](docs/environment.md)

## Documentation Map

- Install and setup: [docs/install.md](docs/install.md)
- Data download and config: [docs/data.md](docs/data.md)
- Production inference: [docs/production_inference.md](docs/production_inference.md)
- Training and saved-model workflows: [docs/training.md](docs/training.md)
- Development workflow: [docs/development.md](docs/development.md)
- Paper-era reproducibility notes: [docs/reproducibility.md](docs/reproducibility.md)
- Docs index: [docs/README.md](docs/README.md)

## Development

Canonical commands:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format .
uv run ty check s2and
```

To run the entire CI suite mimicking the GH Actions:
```bash
uv run python scripts/run_ci_locally.py
```
`scripts/run_ci_locally.py` mirrors `.github/workflows/main.yaml` by running:
- lint job (`scripts/sync_version.py --check`, `ruff check`, and `ruff format --check`)
- `typecheck-and-test` matrix lanes (`py-only`, then `rust-enabled`)
- Rust parity guardrail tests in the `rust-enabled` lane

The runner passes `-ra` to pytest so skip reasons are printed by lane. Rust-only tests may skip in `py-only` because
that lane intentionally omits the `rust` extra and forces `S2AND_BACKEND=python`; they must run in `rust-enabled` after
the local extension is built with `maturin develop`.

By default, local `ty` checks use `--python-version 3.11 --python-platform linux` to match GitHub Linux runners.
To override platform emulation locally, set `S2AND_CI_TY_PLATFORM` (for example, `windows`).

To run CI checks locally without Rust extension compilation (faster iteration):
```bash
uv sync --active --extra dev --frozen
uv run --active --no-project ruff format --check s2and scripts/*.py
uv run --active --no-project ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global
uv run --active --no-project ty check scripts/*.py --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global --ignore unresolved-reference --ignore unresolved-attribute
uv run --active --no-project pytest tests/ --cov=s2and --cov-report=term-missing --cov-fail-under=40
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
echo 0.50.2 > VERSION

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

The original paper-era environment and scripts live on the `s2and_paper` branch. See [docs/reproducibility.md](docs/reproducibility.md) for the current guidance and compatibility notes for old released artifacts.

## Licensing

The code in this repo is released under the Apache 2.0 license. The dataset is released under ODC-BY. Some affiliation data comes directly from the Microsoft Academic Graph.

## Citation

If you use S2AND in your research, please cite [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421).

```text
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

S2AND is an open-source project developed by the Allen Institute for Artificial Intelligence (AI2).

# S2AND

S2AND provides the S2AND author-name-disambiguation benchmark datasets and the reference model described in the paper [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421) by Shivashankar Subramanian, Daniel King, Doug Downey, and Sergey Feldman.

> **Release status (2026-07-24):** this development branch contains the
> canonical-v2 code migration, but it is not yet a complete production release.
> Canonical artifacts and the model bundle `production_model_v1.3` still need to
> be generated, retrained, evaluated, and published. The package manifests
> currently say `0.60.0`; whether the coordinated package release keeps that
> version or becomes `1.3.0` is an explicit open decision. Release operators
> must use [docs/release.md](docs/release.md).

As of this version, S2AND requires the `s2and-rust` extension at install time.
Explicit classic Python routes still exist for selected `ANDData` stages, but
they are not silent fallbacks from Arrow/Rust APIs. Production model loading
and the maintained large-scale runtime require the Rust package.

## What S2AND Provides

- The S2AND datasets used for author name disambiguation research.
- Versioned production model artifacts used by Semantic Scholar.
- Training, evaluation, and inference APIs in Python.
- A required Rust extension for production model scoring and maintained large-scale runtime paths.

## Choose a Workflow

| Use case | Start here | Details |
| --- | --- | --- |
| Run the released model on your own data | [Quick Start](#quick-start) | [docs/production_inference.md](docs/production_inference.md) |
| Download the benchmark datasets | [Download Data or Model](#download-data-or-model) | [docs/data.md](docs/data.md) |
| Train or evaluate a model | [Training and Evaluation Essentials](#training-and-evaluation-essentials) | [docs/training.md](docs/training.md) |
| Build a production release bundle | `scripts/production/` | [docs/production_inference.md](docs/production_inference.md) |
| Operate the v1.3 retrain and release | [v1.3 release runbook](docs/release.md) | Follow its stages, approvals, and release contract |
| Operate Rust-backed large-scale inference | [Runtime and Scaling](#runtime-and-scaling) | [docs/rust/runtime.md](docs/rust/runtime.md), [docs/subblocking.md](docs/subblocking.md), [docs/threading.md](docs/threading.md) |
| Work on the repo itself | [Development](#development) | [docs/development.md](docs/development.md) |

## Install

S2AND supports Python 3.11, 3.12, and 3.13.

Package install:

```bash
uv pip install s2and
```

That command installs the latest package available from the configured index;
it does not install this unreleased worktree. Use the repo-checkout flow below
when validating canonical-v2 before publication.

The base install includes the exactly matched `s2and-rust` runtime. During the
canonical-v2 cutover, no default production model is packaged; inference callers
must provide an explicit compatible bundle.

Repo checkout:

```bash
git lfs install
git lfs pull --include "tests/fixtures/arrow/pubmed_specter2/**"
uv venv --python 3.11
# activate the environment, then:
uv sync --active --extra dev
uv run --active --no-project maturin develop -m s2and_rust/Cargo.toml
```

Source checkouts use Git LFS for Arrow test fixtures. Run `git lfs pull` after
cloning and after switching branches that change those fixtures.

The Rust build step is required for source checkouts unless you are using an
already-built compatible `s2and-rust` wheel. For OS prerequisites, activation
commands, WSL notes, and install variants, see [docs/install.md](docs/install.md).

## Download Data or Model

> **Canonical-v2 migration status (2026-07-24):** this branch contains the
> canonical-v2 code cutover but does not yet contain a compatible production
> model or canonical count artifacts. No default model is distributed by this
> branch. Use the previous published S2AND release for working v1.21 inference
> until canonical v1.3 is trained, validated, and packaged. See
> [docs/release.md](docs/release.md).

Rust/Arrow dataset download:

The AWS CLI is not a runtime dependency; `uvx` installs and runs it in an
isolated environment for this command.

```bash
uvx --from awscli aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release-arrow s2and/data/
```

Expected size is about `10.1 GiB`; use a narrower S3 prefix when only one
dataset is needed. The checked-in v1.21 directory remains an explicit historical
source artifact only. It is not packaged or loadable on this branch.

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

This migration branch has no compatible default production model. Training and
evaluation remain available below; production inference requires an explicit
complete canonical bundle once one has passed the release gates.

## Production Inference Essentials

`load_production_model(path)` accepts one explicit complete native bundle. It
does not discover a default, load pickle models, or accept pairwise-only staging
directories. Classic `ANDData` prediction is the Python route; explicit
methods that take an open `ArrowDataset` are the Rust route.

Full inference and bundle publication details are in
[docs/production_inference.md](docs/production_inference.md).

## Training and Evaluation Essentials

The example below is a small research/API example, not the production release
protocol: it materializes test features and permits immediate evaluation.
During the v1.3 release, test identities and scores remain sealed until the
one-shot gates in Stages 7 and 8 of the
[release runbook](docs/release.md).

Minimal training flow:

```python
import json
from pathlib import Path

from hyperopt import hp

from s2and.arrow_inputs import ArrowDataset
from s2and.arrow_training import build_training_anddata_from_arrow
from s2and.consts import NORMALIZATION_VERSION
from s2and.featurizer import FeaturizationInfo, featurize
from s2and.model import Clusterer, FastCluster, PairwiseModeler

bundle_dir = Path("/path/to/canonical_arrow_training_bundle/pubmed")
manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
manifest_paths = manifest["paths"]
with ArrowDataset.open(
    bundle_dir,
    require_specter=True,
    require_name_counts_index=True,
    expected_normalization_version=NORMALIZATION_VERSION,
) as arrow_dataset:
    dataset = build_training_anddata_from_arrow(
        arrow_dataset,
        "pubmed",
        clusters=str((bundle_dir / manifest_paths["clusters"]).resolve()),
        train_pairs_size=1000,
        val_pairs_size=200,
        test_pairs_size=200,
        n_jobs=4,
    )

    featurization_info = FeaturizationInfo()
    train, val, test = featurize(dataset, featurization_info, n_jobs=4)
    X_train, y_train, _ = train
    X_val, y_val, _ = val

    pairwise_model = PairwiseModeler(
        n_iter=25,
        monotone_constraints=featurization_info.lightgbm_monotone_constraints,
    )
    pairwise_model.fit(X_train, y_train, X_val, y_val)

    clusterer = Clusterer(
        featurization_info,
        pairwise_model,
        cluster_model=FastCluster(linkage="average"),
        search_space={"eps": hp.uniform("eps", 0, 1)},
        n_iter=25,
        n_jobs=4,
    )
    clusterer.fit(dataset)
```

Point `bundle_dir` at a manifest-backed canonical Arrow dataset root.
`ArrowDataset.open(...)` validates its tables, batch indexes, checksums,
normalization version, and name-count index once and retains the opened files
for the handle's lifetime. This migration branch does not bundle a full
training root.

For evaluation and native-bundle publication guidance, see
[docs/training.md](docs/training.md).

## Runtime and Scaling

Runtime controls:

- Unset `S2AND_BACKEND` means Python; the only accepted values are `python` and
  `rust`.
- Rust mode requires the exact `s2and-rust` version pinned by the project
  metadata and fails explicitly if it is missing or different.
- Public prediction routes are method-based: `ANDData` methods use Python and
  `predict_from_arrow(..., arrow_dataset)` and
  `predict_incremental_from_arrow(..., arrow_dataset)` use Rust.

Cache behavior:

- Production inference has no persistent cache; Rust featurizers are not
  serialized to disk, and same-process Rust featurizer reuse is the only
  inference-time reuse mechanism.
- Repeated training experiments can opt into the featurized-split snapshot
  cache (`train_pairwise.py --feature-cache-dir`). See `docs/caching.md`.

Large blocks:

- `predict(..., batching_threshold=...)` uses subblocking to keep full-block work bounded.
- `predict_incremental(...)` is the Python `ANDData` route and does not accept
  query batching. Use `predict_incremental_from_arrow(...,
  batching_threshold=...)` for promoted Rust query batching.
- Incremental results still include `phase_b_mode`; current supported routes report `exact`.
- `total_ram_bytes` is the main memory-control knob for large inference jobs.

Concurrency:

- Treat `n_jobs` as the main concurrency knob for a run.
- Set thread-related environment variables before importing heavy compute libraries.

Details:

- Runtime contract: [docs/rust/runtime.md](docs/rust/runtime.md)
- Cache semantics: [docs/caching.md](docs/caching.md)
- Threading guidance: [docs/threading.md](docs/threading.md)
- Subblocking and memory tradeoffs: [docs/subblocking.md](docs/subblocking.md)
- Environment variables: [docs/environment.md](docs/environment.md)

## Documentation Map

- Install and setup: [docs/install.md](docs/install.md)
- Data download and config: [docs/data.md](docs/data.md)
- Production inference: [docs/production_inference.md](docs/production_inference.md)
- Training and saved-model workflows: [docs/training.md](docs/training.md)
- Development workflow: [docs/development.md](docs/development.md)
- Paper-era reproducibility notes: [docs/reproducibility.md](docs/reproducibility.md)
- v1.3 retrain and release runbook: [docs/release.md](docs/release.md)
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

See [docs/development.md](docs/development.md#local-ci-mirror) for the shared
hosted/local job policy and individual-job commands.

To run static CI checks locally without Rust extension compilation (faster iteration):
```bash
uv sync --active --extra dev --frozen --no-install-package s2and-rust
uv run --active --no-project ruff format --check s2and scripts/*.py
uv run --active --no-project ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global
uv run --active --no-project ty check scripts/*.py --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global --ignore unresolved-reference --ignore unresolved-attribute
```

The full pytest suite is not a no-native check; build the required extension first or use `scripts/run_ci_locally.py`.

### Version bumping
Versioning is centralized in the `VERSION` file (single source of truth). When you update it, we sync the Python/Rust
manifests and regenerate lockfiles.

One-time setup for hooks (recommended):
```bash
git config core.hooksPath .githooks
```

Workflow:
```bash
# 1) edit VERSION to the new semantic version

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
- v1.3 release operator runbook: `docs/release.md`

---

## Reproducibility

The original paper-era environment and scripts live on the `s2and_paper` branch. See [docs/reproducibility.md](docs/reproducibility.md) for the current guidance and compatibility notes for old released artifacts.

## Licensing

Package metadata currently declares the Python package license as MIT, while the
root `LICENSE` file is CC-BY-4.0. The released dataset is under ODC-BY. Some
affiliation data comes directly from the Microsoft Academic Graph.

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

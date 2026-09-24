# S2AND

S2AND provides the S2AND author-name-disambiguation benchmark datasets and the reference model described in the paper [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421) by Shivashankar Subramanian, Daniel King, Doug Downey, and Sergey Feldman.

> **Update at v1.3 release:** this development branch contains the simplified
> `1.0.0` runtime, but it is not yet a complete production release. Canonical
> artifacts and the model bundle `production_model_v1.3` still need to be
> generated, retrained, evaluated, and published, so the model-dependent and
> data-dependent workflows below do not work yet. The coordinated Python and
> Rust package version is fixed at `1.0.0`; the production model and public-data
> release version is `1.3`, and independently readable Arrow/name-count data
> uses public format `1`. Release operators must use
> [docs/release.md](docs/release.md). This README is also the PyPI description,
> so every "Update at v1.3 release" note must be resolved before publication.

S2AND requires the `s2and-rust` extension, which is installed with the package.
Classic `ANDData` methods run in Python and methods that take an open
`ArrowDataset` run in Rust; neither falls back to the other. Production model
loading and the large-scale runtime require the Rust package.

## What S2AND Provides

- The S2AND datasets used for author name disambiguation research.
- Versioned production model artifacts used by Semantic Scholar.
- Training, evaluation, and inference APIs in Python.
- A required Rust extension for production model scoring and large-scale runtime paths.

## Choose a Workflow

| Use case | Start here | Details |
| --- | --- | --- |
| Run the released model on your own data | [Quick Start](#quick-start) | [docs/production_inference.md](docs/production_inference.md) |
| Download the benchmark datasets | [Download Data](#download-data) | [docs/data.md](docs/data.md) |
| Train or evaluate a model | [Training and Evaluation](#training-and-evaluation) | [docs/training.md](docs/training.md) |
| Run large-scale inference | [Runtime and Scaling](#runtime-and-scaling) | [docs/subblocking.md](docs/subblocking.md), [docs/threading.md](docs/threading.md) |
| Train and publish the v1.3 production release | [Release runbook](docs/release.md) | [Production command reference](scripts/production/README.md) |
| Work on the repo itself | [Development](#development) | [docs/development.md](docs/development.md) |

All guides are indexed in [docs/README.md](docs/README.md), which also has a
glossary of S2AND terms.

## Install

S2AND supports Python 3.11, 3.12, and 3.13.

Package install:

```bash
uv pip install s2and
```

This installs the latest version on the configured package index. Starting with
`1.0.0`, that install includes the exactly matched `s2and-rust` runtime.

> **Update at v1.3 release:** until `s2and==1.0.0` is published, this command
> installs an earlier release in which `s2and-rust` is an optional extra. Use a
> source checkout to validate the unreleased `1.0.0` worktree.

Repo checkout (requires Rust; see [docs/install.md](docs/install.md) for OS
prerequisites, environment activation, and WSL notes):

```bash
git lfs install
git lfs pull
uv sync --extra dev
```

`uv sync` builds the matching Rust extension from `s2and_rust/`. Repeat
`git lfs pull` after switching to a branch that changes the LFS-managed
LightGBM test fixtures. If a Rust change is not picked up, see
[build recovery](docs/install.md#build-the-rust-extension-from-source).

## Download Data

> **Update at v1.3 release:** no compatible production model or public-format-1
> data has been published yet. The Arrow release below is the historical,
> pre-format-1 release: this branch's `ArrowDataset.open(...)` and release
> validation reject it, so it is useful only for migration and source-data work.
> Use the previous published S2AND release for working v1.21 inference. At
> release, replace this note and the command below with the published v1.3 data
> location and size. See [docs/release.md](docs/release.md) and
> [docs/data.md](docs/data.md#dataset-download).

The AWS CLI is not a runtime dependency; `uvx` installs and runs it in an
isolated environment for this command:

```bash
uvx --from awscli aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release-arrow s2and/data/
```

Expected size is about `10.1 GiB`; use a narrower S3 prefix when only one
dataset is needed. The prior v1.21 model is available only from its compatible
release and Git history; it is not packaged or loadable on this branch.

Classic `ANDData` finds its default name-count index at
`<main_data_dir>/name_counts_index`, where `main_data_dir` comes from
`s2and/data/path_config.json` (or the file named by `S2AND_PATH_CONFIG`). Arrow
routes and the production scripts take explicit paths instead. See
[docs/data.md](docs/data.md#configuring-s2anddatapath_configjson).

## Quick Start

> **Update at v1.3 release:** replace this note with a short load-and-predict
> example that uses the published v1.3 bundle and data. Until then there is no
> compatible production model, so production inference cannot run; training
> and evaluation are described below.

`load_production_model(path)` accepts one explicit complete native bundle. It
does not discover a default or accept pairwise-only training stages. Full
inference details, including the Python and Rust routes, are in
[docs/production_inference.md](docs/production_inference.md).

## Training and Evaluation

[docs/training.md](docs/training.md) walks through building a Rust-backed
training dataset from a canonical Arrow root, featurizing pairs, training the
pairwise model, fitting the clusterer, and evaluating both. Those examples are
research/API examples: they materialize test features for immediate
evaluation. The v1.3 release instead keeps test identities and scores sealed
until the one-shot evaluation in Stage 5 of the
[release runbook](docs/release.md).

## Runtime and Scaling

- **Routes:** `ANDData` methods use Python; `predict_from_arrow(...,
  arrow_dataset)` and `predict_incremental_from_arrow(..., arrow_dataset)` use
  Rust. No environment variable changes these routes, and Rust routes fail
  explicitly if `s2and-rust` is missing or a different version. See
  [execution routes](docs/production_inference.md#explicit-execution-routes).
- **Large blocks:** `batching_threshold` on full-block prediction enables
  subblocking ([docs/subblocking.md](docs/subblocking.md)); on
  `predict_incremental_from_arrow` it caps the queries per linker batch.
  `total_ram_bytes` is the main memory-control knob.

  > **Update at v1.3 release:** subblocking loads the packaged ORCID
  > prefix-count files, which Stage 1 of the release generates. Until they are
  > copied into `s2and/data/`, any call with `batching_threshold` set raises
  > `FileNotFoundError`.
- **Concurrency:** `n_jobs` is the main concurrency knob. A loaded production
  bundle starts with `n_jobs=1`; set `clusterer.n_jobs` before predicting. Set
  thread-related environment variables before importing heavy compute
  libraries ([docs/threading.md](docs/threading.md),
  [docs/environment.md](docs/environment.md)).

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format .
uv run ty check s2and
uv run --no-project python scripts/run_ci_locally.py  # shared local/hosted CI
```

The full test suite needs the native extension that `uv sync --extra dev`
builds. [docs/development.md](docs/development.md) covers the CI jobs, the
static-check fast path without a native build, native build troubleshooting,
and version bumping (`VERSION` is the source of truth for both packages).

## Reproducibility

The original paper experiments used the `s2and_paper` branch with Python
`3.7.9`. Start in that branch with an isolated environment:

```bash
git checkout s2and_paper
uv venv --python 3.7.9
```

Install the pinned environment from that branch's `paper_experiments_env.txt`,
then rerun its paper experiment commands. The selected artifacts retained in
`scripts/archive/` are historical references, not supported current entrypoints
or substitutes for the paper branch; see the
[archive catalog](scripts/README.md#archived-historical-artifacts).

Use paper-era seed artifacts such as `full_union_seed_*.pickle` with the paper
branch. Some historical model pickles contain a dictionary with a `clusterer`
key rather than a bare clusterer. Current S2AND neither distributes nor loads
production-model pickles. Reproduce later historical releases with their
compatible release or Git history; current model loading and publication follow
the [native-bundle contract](docs/production_inference.md#complete-model-bundles)
and [release runbook](docs/release.md).

## Licensing

S2AND code is licensed under the MIT License ([LICENSE](LICENSE)). S2AND data,
including the benchmark datasets and the data files in this repository, is
licensed under the Creative Commons Attribution 4.0 International License
([LICENSE-DATA](LICENSE-DATA)). Data releases published before v1.3 keep the
license they were published with. Some affiliation data comes directly from the
Microsoft Academic Graph.

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

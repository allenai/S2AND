# S2AND

S2AND provides author-name-disambiguation datasets and the reference model from [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421).
The model predicts author clusters from paper and author records.

This branch prepares Python/Rust runtime `1.0.0`, model release `1.4`, and data release `1.4`.

Use [the release process](docs/release.md) for current preparation, fit, evaluation, and publication steps.

## Choose a workflow

| Goal | Required input | Guide |
| --- | --- | --- |
| Run the production model | Your records and vectors, plus the inference bundle | [Production inference](docs/production_inference.md) |
| Use corrected research data | The inference bundle and the `dataset` bundle | [Data and models](docs/data.md) |
| Reproduce the production training process | The `1.3.1` inference bundle and `dataset` bundle | [Reproduce the 1.3.1 model](docs/reproduce.md) |
| Correct data or release another model | All three S3 bundles and maintained scripts | [Release process](docs/release.md) |
| Train or evaluate with the API | Manifest-backed dataset and labels | [Training and evaluation](docs/training.md) |
| Use current name normalization and small alias artifacts | Source checkout; optional count-index download | [Canonical names](docs/canonical_names.md#access-the-current-rules-and-artifacts) |
| Work on the repository | Source checkout and development environment | [Development](docs/development.md) |

The model, the name-count index, and the data download from public S3 bundles: `inference` and `dataset`.
See [Download data](#download-data).
Small name tuples and ORCID prefix counts ship in normal Git and the Python wheel.
Release `1.3` also published a `source` bundle with the full corrected records; `1.4` does not.

## Install

S2AND supports Python 3.11, 3.12, and 3.13.
The `1.0.0` package includes its native extension in the same wheel.
Until publication, use a source checkout:

```bash
git clone --branch canonical-v2-migration https://github.com/allenai/S2AND.git
cd S2AND
uv sync
```

`uv sync` builds the native extension from `s2and_rust/`.
`uv sync --extra dev` also installs the test and lint tools.
See [installation prerequisites](docs/install.md) for operating-system dependencies and build recovery.

## Download data

Sync the bundles into `s2and/data/`, the default data folder.
The commands use the AWS CLI and need no AWS account.

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.4/inference/ s2and/data/
# Training, evaluation, and the quick start examples also need:
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and/1.4/dataset/ s2and/data/
```

The inference bundle holds the complete model and the name-count index (3.0 GB).
`dataset` holds the benchmark Arrow data and labels (12.3 GB).
The [data guide](docs/data.md#choose-the-required-data) states which bundles each user needs.
Git ignores the downloaded files.

## Quick start

After you [download](#download-data) the inference bundle and the `dataset` bundle, cluster every block of a released benchmark.
For `qian` the model puts 6,379 signatures into 1,283 clusters:

```bash
uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
    --model-path s2and/data/production_model_v1.4 \
    --arrow-dataset s2and/data/benchmarks/qian --n-jobs 8
```

Incremental prediction links new signatures to existing author profiles.
In this example, labeled `j wang` signatures on papers before 2015 seed 498 profiles.
The block's 303 other signatures arrive as new papers; 233 of them join a profile:

```bash
uv run python scripts/tutorial_for_predicting_with_the_prod_model.py \
    --model-path s2and/data/production_model_v1.4 \
    --arrow-dataset s2and/data/benchmarks/aminer \
    --incremental-block "j wang" \
    --seed-clusters s2and/data/benchmarks/aminer/aminer_clusters.json \
    --seed-before-year 2015 --n-jobs 8
```

In your own code, load the complete model from an explicit path:

```python
from s2and.arrow_inputs import ArrowDataset
from s2and.production_model import load_production_model

clusterer = load_production_model('s2and/data/production_model_v1.4', n_jobs=8)
blocks = {'j smith': ['signature_1', 'signature_2']}
with ArrowDataset.open('/path/to/your_arrow_dataset') as dataset:
    clusters, distances = clusterer.predict_from_arrow(
        blocks,
        dataset,
        total_ram_bytes=32 * 1024**3,
    )
```

Replace the dataset path and signature IDs with your validated Arrow dataset.
`blocks` maps each S2 author block to its signature IDs.
The tutorial's `arrow_blocks` function builds that mapping from `signatures.arrow`.
The loader rejects pickles and pairwise-only stages.
The [inference guide](docs/production_inference.md) describes raw inputs, embeddings, constraints, incremental calls, and bundle compatibility.

## Training and evaluation

- Reproduce the production model on published data with the [reproduction guide](docs/reproduce.md) and [production commands](scripts/production/README.md#pairwise-model).
- Run smaller experiments and inspect metrics through the Python API with the [training and evaluation guide](docs/training.md).

## Runtime and scaling

- `Clusterer.predict` on `ANDData` uses Python. `predict_from_arrow` (full-block and subblocked) and `predict_incremental_from_arrow` use Arrow/Rust.
- `batching_threshold` enables subblocks for full prediction and caps incremental query batches.
- `total_ram_bytes` controls the major memory allocation bounds.
- `n_jobs` controls concurrency; model load defaults to one job unless you specify another value.

Set thread environment variables before heavy-library imports.
See [subblocking](docs/subblocking.md), [threading](docs/threading.md), and [environment settings](docs/environment.md).

## Development

Run the full CI checks with `uv run --no-project python scripts/run_ci_locally.py`.

The tests require the native extension from the development setup.
See [the development guide](docs/development.md) for focused checks and native build recovery.

## Reproducibility

The original paper experiments use the `s2and_paper` branch and Python `3.7.9`.
Use that branch's pinned environment and experiment commands for those results.

See the [reproduction guide](docs/reproduce.md) and the [release process](docs/release.md) for current models and release artifacts.

## Licensing

S2AND code uses the MIT License ([LICENSE](LICENSE)).
S2AND data uses the Creative Commons Attribution 4.0 International License ([LICENSE-DATA](LICENSE-DATA)).
Earlier data releases retain their published licenses.
Some affiliation data comes from the Microsoft Academic Graph.

## Citation

Cite [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421) when you use S2AND in research.

```text
@inproceedings{subramanian2021s2and,
  title={{S}2{AND}: {A} {B}enchmark and {E}valuation {S}ystem for {A}uthor {N}ame {D}isambiguation},
  author={Subramanian, Shivashankar and King, Daniel and Downey, Doug and Feldman, Sergey},
  year={2021},
  publisher={Association for Computing Machinery},
  address={New York, NY, USA},
  booktitle={{JCDL} '21: Proceedings of the {ACM/IEEE} Joint Conference on Digital Libraries in 2021},
  series={JCDL '21}
}
```

The Allen Institute for Artificial Intelligence develops S2AND.

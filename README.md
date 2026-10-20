# S2AND

S2AND provides author-name-disambiguation datasets and the reference model from [S2AND: A Benchmark and Evaluation System for Author Name Disambiguation](https://api.semanticscholar.org/CorpusID:232233421).
The model predicts author clusters from paper and author records.

This branch prepares Python/Rust runtime `1.0.0` and model/data release `1.3`.

> **Update at v1.3 release:** The complete model and the `1.3` data bundles are not yet published.

Use [the release process](docs/release.md) for current preparation, fit, evaluation, and publication steps.

## Choose a workflow

| Goal | Required input | Guide |
| --- | --- | --- |
| Run the production model | Common model and name-count index; your records and vectors | [Production inference](docs/production_inference.md) |
| Use corrected research data | The `inference` and `dataset` S3 bundles | [Data and models](docs/data.md) |
| Reproduce the production training process | The `inference` and `dataset` S3 bundles | [Reproduce the 1.3 model](docs/reproduce.md) |
| Correct data or release another model | All three S3 bundles and maintained scripts | [Release process](docs/release.md) |
| Train or evaluate with the API | Manifest-backed dataset and labels | [Training and evaluation](docs/training.md) |
| Use current name normalization and small alias artifacts | Source checkout; optional count-index download | [Canonical names](docs/canonical_names.md#access-the-current-rules-and-artifacts) |
| Work on the repository | Source checkout and development environment | [Development](docs/development.md) |

Common large assets use Git LFS.
Small name tuples and ORCID prefix counts ship in normal Git and the Python wheel.
Each release also publishes `inference`, `dataset`, and `source` S3 bundles; see [Download data](#download-data).
The `source` bundle holds the full corrected records, the starting point for future corrections.

## Install

S2AND supports Python 3.11, 3.12, and 3.13.
The `1.0.0` package requires the exactly matched `s2and-rust` package.
Until publication, use a source checkout:

```bash
git lfs install
git lfs pull
uv sync --extra dev
```

`uv sync` builds the native extension from `s2and_rust/`.
See [installation prerequisites](docs/install.md) for operating-system dependencies and build recovery.
Pull LFS assets after a branch or release-ref change that affects those files.

## Download data

> **Update at v1.3 release:** The compatible release locations remain unpublished.

The release notes will provide the common Git ref, model path, S3 prefix, and hashes.
Each release publishes three S3 bundles: `inference`, `dataset`, and `source`.
The [data guide](docs/data.md#choose-the-required-data) states which bundles each user needs.
Historical Arrow prefixes do not satisfy the current format-1 contract.

Inference users sync `inference` and supply their own data.
Training and evaluation users also sync `dataset` into the same folder.
Classic `ANDData` selects that index through [path configuration](docs/data.md#configuring-s2anddatapath_configjson).
Arrow and production script calls use explicit paths.

## Quick start

Load the complete model from an explicit path:

```python
from s2and.arrow_inputs import ArrowDataset
from s2and.production_model import load_production_model

clusterer = load_production_model('/path/to/production_model_v1.3', n_jobs=8)
blocks = {'j smith': ['signature_1', 'signature_2']}
with ArrowDataset.open('/path/to/your_arrow_dataset') as dataset:
    clusters, distances = clusterer.predict_from_arrow(
        blocks,
        dataset,
        total_ram_bytes=32 * 1024**3,
    )
```

Replace the paths and signature IDs with your complete model and validated dataset.
The loader rejects pickles and pairwise-only stages.
The [inference guide](docs/production_inference.md) describes raw inputs, embeddings, constraints, incremental calls, and bundle compatibility.

## Training and evaluation

The [reproduction guide](docs/reproduce.md) runs the production training process on the published data.
The production pairwise trainer reads `pairwise/model_plan.json` from the synced `dataset` bundle.
It uses the dataset names that the plan declares.
The [production commands](scripts/production/README.md#pairwise-model) describe pairwise fit, validation-only EPS selection, and incremental finalization.
A historical baseline is not a prerequisite for a pairwise fit.
Final release acceptance remains a separate step.

The [research API guide](docs/training.md) provides smaller experiments and direct metric inspection.
These examples do not replace the release test policy.

## Runtime and scaling

- `ANDData` prediction uses Python; Arrow prediction uses Rust.
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
Retired historical scripts remain in Git history; see [the script catalog](scripts/README.md#retired-historical-scripts).

The current process starts from a complete corrected snapshot.
It preserves exact pair and split identities and uses versioned native models.
See [the release process](docs/release.md) for compatible artifacts and update rules.

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

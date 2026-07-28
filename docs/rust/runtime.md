# Rust Runtime Contract

Status date: 2026-07-27

S2AND has two explicit execution routes. Classic `ANDData` workflows use
Python. Arrow-native training and prediction use the pinned Rust extension.
The runtime does not discover native methods or switch routes after a failure.

## Installation and version

The exact `s2and-rust` version pinned by the S2AND project metadata is a
required dependency installed by a normal `uv sync`.

When Rust is requested, S2AND imports `s2and_rust` and requires its
`__version__` to equal the runtime's required version. A missing extension or a
different version is an error. Supporting another native version requires
updating the pinned dependency and S2AND together.

## Backend selection

`S2AND_BACKEND` accepts only `python` or `rust`. If it is unset, a runtime
context without an explicit backend uses `python`. Invalid values raise
`ValueError`; requesting Rust with a missing or mismatched extension raises
`RuntimeError`.

Public dataset and prediction APIs select their own route:

| API | Route |
|---|---|
| `ANDData(...)` | Python |
| `Clusterer.predict(...)` | Python |
| `Clusterer.predict_incremental(...)` | Python |
| `build_training_anddata_from_arrow(...)` | Rust |
| `Clusterer.predict_from_arrow(...)` | Rust |
| `Clusterer.predict_incremental_from_arrow(...)` | Rust |

Passing a Rust runtime context to a classic prediction API, or a Python
runtime context to an Arrow prediction API, is an error. There is no automatic
backend and no silent fallback between implementations.

## Arrow contract

`build_training_anddata_from_arrow(...)` is the one Rust-training dataset
constructor. It takes one `ArrowDataset.open(root)` handle, validates the
training profile, and builds a fully initialized dataset that retains the
handle. Rust reads embeddings and name counts from it; classic `ANDData` does
not acquire an Arrow route after construction.

The `predict_from_arrow(...)` and `predict_incremental_from_arrow(...)` methods
take the same open handle and construct the Rust featurizer directly. Missing
model-required artifacts, invalid Arrow schemas, or native execution errors are
surfaced to the caller.
Model roots must record `generated_by_runtime` equal to the installed package
version. Arrow and name-count manifests must declare public format `1`.
Missing or legacy declarations are not executable runtime modes.

An open Arrow root is immutable. Reuse its validated owning handle across
requests; publish changed content at a new root and open a new handle rather
than mutating live files.

## Threading

`n_jobs` is passed as `num_threads` to maintained Rust batch APIs. LightGBM
estimators use their configured thread count. See [../threading.md](../threading.md)
for avoiding nested Rayon, OpenMP, and BLAS parallelism.

## Verification

Build the pinned extension and run the focused routing tests:

```bash
uv run maturin develop -m s2and_rust/Cargo.toml --release
uv run pytest -q tests/test_runtime.py tests/test_data.py tests/test_arrow_training_ingestion.py
```

The release performance check is the bounded promoted-incremental Arrow
profiler documented in [baselines.md](baselines.md). Feature parity,
subblocking quality, packaging, and installed-runtime checks use their focused
verification scripts instead of a shared benchmark dispatcher.

# Rust Runtime Contract

Status date: 2026-07-10

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
| `Clusterer.predict_from_arrow_paths(...)` | Rust |
| `Clusterer.predict_incremental_from_arrow_paths(...)` | Rust |

Passing a Rust runtime context to a classic prediction API, or a Python
runtime context to an Arrow prediction API, is an error. There is no automatic
backend and no silent fallback between implementations.

## Arrow contract

`build_training_anddata_from_arrow(...)` is the one Rust-training dataset
constructor. It validates the training artifact profile, builds a fully
initialized dataset, and binds one immutable `dataset.arrow_paths` mapping as
the path authority. Rust reads embeddings and name counts from that mapping;
classic `ANDData` does not acquire an Arrow route after construction.

The `*_from_arrow_paths` methods validate the prediction artifact profile and
construct the Rust featurizer directly. Missing model-required artifacts,
invalid Arrow schemas, or native execution errors are surfaced to the caller.
Models and artifacts must explicitly declare the package's
`canonical_v2` normalization contract. Missing or legacy declarations are not
executable runtime modes.

Arrow generations are treated as immutable. Process-local Rust featurizer
reuse is keyed by the validated artifact generation and build settings.
Replace an artifact by publishing a new generation rather than mutating a
live path.

Pair featurization selects a signature-index layout adaptively
(`select_signature_index_layout`): dense direct indexing when the referenced
index range is proportionate to the pair workload, and a compact remapped
layout when indices are sparse. The retired always-dense layout allocated
one slot per index up to `max_index`, so a request touching two signatures in
a ten-million-signature corpus paid the full dense allocation; the compact
layout bounds live memory by the distinct signatures actually referenced.
Layout selection is covered by unit tests in `rust_featurizer.rs`.

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

The broader Rust quality, latency, and memory checks are the canonical
`scripts/rust_suite.py` workflows documented in [baselines.md](baselines.md):

```bash
uv run python scripts/rust_suite.py compare ...
uv run python scripts/rust_suite.py transfer-mini ...
uv run python scripts/rust_suite.py stress-rebuild ...
```

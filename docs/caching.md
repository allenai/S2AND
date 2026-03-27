# Caching

This document describes every cache-like mechanism in S2AND and how it relates to the public
`use_cache` flag.

## Public API

`use_cache` is the single public cache control on the main APIs:

- `featurize(..., use_cache=...)`
- `many_pairs_featurize(..., use_cache=...)`
- `Clusterer.use_cache`
- `warm_rust_featurizer(..., use_cache=...)`

Public semantics:

- `use_cache=True`: read and write the persistent pair-feature cache and the Rust featurizer disk
  cache.
- `use_cache=False`: skip persistent pair-feature cache reads/writes and skip Rust featurizer
  disk-cache reads/writes.

Important nuance:

- `use_cache` does not disable same-process Rust featurizer reuse.
- `use_cache` does not disable the artifact download cache used by `s2and.file_cache.cached_path`.

## Cache Inventory

| Layer | Controlled by `use_cache` | Purpose | Default location |
| --- | --- | --- | --- |
| Pair-feature cache | Yes | Reuse computed pairwise feature rows across repeated featurization/prediction | `<S2AND_CACHE>/<dataset>/<featurizer_version>/pair_features.sqlite3` |
| Rust featurizer disk cache | Yes | Reuse serialized Rust featurizers across process restarts | `<S2AND_CACHE>/rust_featurizer/*.bin` plus `.meta.json` |
| Rust featurizer in-memory reuse | No | Reuse an already-built Rust featurizer within the current Python process | memory only |
| Artifact download cache | No | Avoid re-downloading remote artifacts fetched through `cached_path()` | `<S2AND_CACHE>/artifacts` |

`S2AND_CACHE` defaults to `~/.s2and`.

## Pair-Feature Cache

The pair-feature cache stores full feature rows keyed by the internal signature-pair cache key.
Its path is derived from:

- dataset name
- featurizer version

Current on-disk layout:

```text
<S2AND_CACHE>/
  <dataset_name>/
    <featurizer_version>/
      pair_features.sqlite3
```

The SQLite database stores:

- one row per cached pair
- the full `NUM_FEATURES` feature vector as a float64 blob
- cache metadata such as schema version and `features_to_use`

Operational behavior:

- ordinary writes are incremental, so write cost scales with newly computed rows instead of the
  total cache size
- the cache is only consulted when `use_cache=True`
- if `use_cache=False`, pair features are computed and returned normally but are not read from or
  written to the persistent cache
- once loaded, the cache payload is memoized in process memory so repeated calls in the same
  process do not re-read SQLite; large cache-enabled runs can therefore still consume substantial
  RAM

### Legacy JSON Compatibility

Older S2AND versions wrote pair features to:

```text
<S2AND_CACHE>/<dataset>/<featurizer_version>/all_features.json
```

Current code still reads that file for compatibility. If a legacy JSON cache is loaded and the
cache is later written, S2AND migrates those entries into `pair_features.sqlite3`. After that, the
SQLite database is the authoritative persistent cache.

## Rust Featurizer Caches

The Rust featurizer has two distinct reuse mechanisms.

### Same-Process In-Memory Reuse

When the same `ANDData` object is reused inside one Python process, S2AND keeps the built Rust
featurizer in memory and reuses it on later calls. This is always enabled.

Implications:

- `warm_rust_featurizer(dataset, use_cache=False)` is still useful
- `use_cache=False` does not force a rebuild if the same dataset object already has a live cached
  Rust featurizer
- `S2AND_RUST_FEATURIZER_MAX_INMEM` controls how many in-memory Rust featurizers are retained

### Rust Disk Cache

When `use_cache=True`, S2AND also reads and writes a serialized Rust featurizer on disk:

```text
<S2AND_CACHE>/
  rust_featurizer/
    <derived-cache-key>.bin
    <derived-cache-key>.bin.meta.json
```

The cache key includes dataset identity, featurizer version, cache schema version, and build-path
metadata. This cache is useful when you create fresh Python processes that repeatedly rebuild the
same Rust featurizer.

## Artifact Download Cache

`s2and.file_cache.cached_path()` stores downloaded remote artifacts under:

```text
<S2AND_CACHE>/artifacts
```

This cache is separate from `use_cache`. It is an input-artifact cache, not a featurization cache.

## Interaction with Rust Batch Featurization

Rust batch featurization can sometimes emit only the selected feature columns needed downstream.
Persistent pair-feature caching needs the full feature row, so:

- `use_cache=False` allows the selected-feature fast path when the rest of the runtime conditions
  allow it
- `use_cache=True` materializes full feature rows so they can be written into the pair-feature cache

This is an internal optimization detail, but it explains why persistent caching can add some extra
work even when the cache backend itself is fast.

## Recommended Usage

- Repeated training or repeated inference on the same dataset or pair set: use `use_cache=True`
- One-shot experiments, one-pass offline jobs, or feature-development work: use `use_cache=False`
- Long-lived services that want lower cold-start latency in a single process:
  call `warm_rust_featurizer(dataset, use_cache=False)` during startup
- Services that also want reuse across process restarts:
  call `warm_rust_featurizer(dataset, use_cache=True)`

If a job will not revisit the same pair set, `use_cache=False` is usually the right choice because
it avoids unnecessary persistent writes.

## Clearing Caches

To force a rebuild, delete the relevant cache paths under `S2AND_CACHE`:

- pair-feature cache: `<S2AND_CACHE>/<dataset>/<featurizer_version>/`
- Rust disk cache: `<S2AND_CACHE>/rust_featurizer/`
- artifact cache: `<S2AND_CACHE>/artifacts/`

You can delete one layer without affecting the others.

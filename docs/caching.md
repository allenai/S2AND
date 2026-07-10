# Caching

This document describes every cache-like mechanism in S2AND and how it relates to the public
`use_cache` flag.

## Public API

`use_cache` is the public control for the persistent pair-feature cache on the main pair-featurization APIs:

- `featurize(..., use_cache=...)`
- `many_pairs_featurize(..., use_cache=...)`
- `Clusterer.use_cache`

Public semantics:

- `use_cache=True`: read and write the persistent pair-feature cache.
- `use_cache=False`: skip persistent pair-feature cache reads/writes.

Important nuance:

- `use_cache` does not disable same-process Rust featurizer reuse.
- `use_cache` does not disable the artifact download cache used by `s2and.file_cache.cached_path`.
- Direct Arrow/Rust production prediction paths bypass the persistent pair-feature SQLite cache; `use_cache` only affects
  prediction paths that materialize pair features through the Python cache-aware featurization layer.

## Cache Inventory

| Layer | Controlled by `use_cache` | Purpose | Default location |
| --- | --- | --- | --- |
| Pair-feature cache | Yes | Reuse computed pairwise feature rows across repeated featurization/prediction | `<S2AND_CACHE>/<dataset>/<featurizer_version>/pair_features.sqlite3` |
| Rust featurizer in-memory reuse | No | Reuse an already-built Rust featurizer within the current Python process | memory only |
| Direct Arrow/Rust prediction inputs | No | Read request/runtime Arrow artifacts directly without pair-feature SQLite caching | request or bundle artifact paths |
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

## Rust Featurizer Caches

The Rust featurizer has two distinct reuse mechanisms.

### Same-Process In-Memory Reuse

When the same Arrow-attached dataset object is reused inside one Python process,
S2AND keeps the built Rust featurizer in memory and reuses it on later calls.
Rust construction is Arrow-only; `warm_rust_featurizer(dataset)` therefore
requires validated attached Arrow artifacts and is not a generic JSON/`ANDData`
warm path.

Current implications:

- `use_cache=False` does not force a rebuild if the same dataset object already has a live cached
  Rust featurizer
- Rust featurizers are not serialized to disk; process restarts rebuild them from the dataset
- `evict_rust_featurizer(dataset)` evicts one dataset and
  `clear_rust_featurizer_cache()` clears the process cache
- published Arrow/count artifacts are immutable content-addressed generations
- the cache key binds the exact normalized path set, full artifact-generation
  inventory, non-seed settings, and seed version
- filesystem change watches invalidate a cached generation before reuse;
  same-size rewrites and restored mtimes do not preserve a stale cache hit
- request-local sidecars are deliberately excluded from the immutable
  generation and are consumed separately under the request boundary

The verified-generation hot-cache check is fixed request/build overhead, not a
pair-loop operation. The current local benchmark is 12.76 microseconds per hit
(78.4k checks/s), including the exact material binding and mutation watch, with
no retained duplicate artifact payload.

## Artifact Download Cache

`s2and.file_cache.cached_path()` stores downloaded remote artifacts under:

```text
<S2AND_CACHE>/artifacts
```

This cache is separate from `use_cache`. It is an input-artifact cache, not a featurization cache.

Remote artifacts are keyed by URL plus the server validator. Validators are namespaced as
`etag:<value>` or `last-modified:<value>` before hashing.

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
- Long-lived Arrow-attached services that want lower cold-start latency in a
  single process may call `warm_rust_featurizer(dataset)` during startup
- Production Arrow services should keep Arrow artifacts local and call
  `Clusterer.predict_from_arrow_paths(...)` or Arrow-routed
  `Clusterer.predict(...)`; `warm_rust_featurizer(dataset)` is not the Arrow
  production warmup API
If a job will not revisit the same pair set, `use_cache=False` is usually the right choice because
it avoids unnecessary persistent writes.

## Clearing Caches

To force a rebuild, delete the relevant cache paths under `S2AND_CACHE`:

- pair-feature cache: `<S2AND_CACHE>/<dataset>/<featurizer_version>/`
- artifact cache: `<S2AND_CACHE>/artifacts/`

You can delete one layer without affecting the others.

For process-local Rust reuse, call `evict_rust_featurizer(dataset)` or
`clear_rust_featurizer_cache()`; deleting disk cache directories does not evict
live Rust objects.

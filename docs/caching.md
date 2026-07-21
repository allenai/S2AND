# Caching

The core S2AND library has one persistent cache — the training-time feature
snapshot cache — plus a small set of in-memory reuse mechanisms. Production
inference never reads or writes any disk cache. Standalone analysis scripts
may manage their own output reuse, such as the epsilon-sweep distance files.

## Feature snapshot cache (training only)

`train_pairwise.py --feature-cache-dir PATH` opts into snapshot reuse for
repeated pairwise training experiments; there is no implicit default location.
The training command owns dataset construction and immediately featurizes it,
so mutable `ANDData` objects are never accepted as a reusable cache boundary.
The cache stores the *output* of featurization, not per-pair state:

- One uncompressed NPZ file per train/val/test split at
  `<cache_dir>/<split>_<key-prefix>.npz`, holding the exact `X`, `y`,
  and (when configured) `nameless_X` matrices.
- The key is a SHA-256 over: a snapshot schema version, the content hashes of
  the feature-input files (signatures, papers, SPECTER), the verified name-counts
  binding when name-count features are selected, the name-tuples digest, the
  normalization version, both featurizer configurations (version + feature
  groups), `nan_value`,
  and the hash of the exact resolved pair lists.
- There is no invalidation logic: any input change produces a different key
  and a fresh snapshot. Old snapshots are dead files; delete the directory
  whenever you like.
- Identity is fail-closed: source hashes are captured privately while the
  cache-enabled training dataset is parsed, consumed once, and never exposed
  for reuse. Missing file identity or required name-count provenance raises.
- Snapshots are write-once (published under a short file lock; never
  overwritten) and strictly validated on load (`allow_pickle=False`, exact
  members, dtypes, shapes). A corrupt snapshot raises with its path — delete
  the file and rerun.
- Loading a snapshot reproduces the originally written matrices bit-for-bit.

The snapshot boundary currently supports classic file-backed `ANDData`
training, which uses the Python feature backend. Arrow-backed Rust feature
generation does not read or write these disk snapshots, so there is no shared
Python/Rust snapshot contract to compare. Rust's reuse is the separate
in-process mechanism described below.

Prefer an unsynced local directory for `cache_dir`; snapshots are intentionally
uncompressed to keep cold insertion cheap and can be tens of MB per split.
Cloud-sync churn is pure overhead.

`Clusterer.fit` has no pair-level cache: its validation-block featurization
recomputes each run (measured ~2.3 s per 6.4k pairs on the Python backend;
`fit` logs `stage=fit_val_dists` telemetry with pair counts and seconds so the
cost can be re-evaluated on full retrains).

## In-memory reuse (not caches on disk)

- **Rust featurizer in-process reuse**: an already-built Rust featurizer is
  reused for the same Arrow-attached dataset object within one process, keyed
  by validated generation ID, path set, and seed versions. Process restarts
  rebuild it. `evict_rust_featurizer(dataset)` / `clear_rust_featurizer_cache()`
  manage it explicitly; `warm_rust_featurizer(dataset)` prebuilds it for
  lower cold-start latency in long-lived Arrow services.
- **Name-counts index open dedup**: one shared immutable `NameCountsIndex`
  handle per `(path, manifest sha256)` within a process.
- Assorted per-request/per-call memoization inside incremental linking; all
  ephemeral, none persisted.

## History

The per-pair SQLite cache (`~/.s2and/<dataset>/<version>/pair_features.sqlite3`)
and the `use_cache` flag were removed after the 2026-07-20 benchmark
(`scratch/feature_cache_benchmark/`): its binding did not cover dataset file
contents (stale hits on changed inputs), it forced full-row materialization
and disabled the fused Rust block path, and the snapshot cache replaces its
only real use. Delete any leftover `~/.s2and` dataset directories to reclaim
space.

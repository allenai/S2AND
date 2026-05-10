# Production Inference

This document collects the operational details for using the released S2AND production models.

## Which pairwise model to use

| Model file | Status | Embeddings | Uses reference features? |
| --- | --- | --- | --- |
| `production_model_v1.2.pickle` | Current | SPECTER2 PRX | No |
| `production_model_v1.1.pickle` | Previous | SPECTER1 | No |
| `production_model_v1.0.pickle` | Deprecated | SPECTER1 | Yes |

Recommended default:

- Use `production_model_v1.2.pickle` unless you have a specific compatibility reason to stay on `v1.1`.

Embedding source:

- For `v1.2`, use `embedding.specter_v2` from the Semantic Scholar API.
- For `v1.1`, use `embedding.specter_v1`.

## Promoted incremental linker artifact

The promoted incremental linker artifact is versioned with the pairwise model it
depends on:

| Artifact | Status | Depends on | Format |
| --- | --- | --- | --- |
| `production_incremental_linker_v1.2/` | Current | `production_model_v1.2.pickle` | `booster.lgb` + `metadata.json` |

This is intentionally not a pickle. The directory artifact stores the LightGBM
booster separately from metadata that validates the 70-feature schema,
production contract, retrieval contract, gate thresholds, required Rust
capabilities, and a prediction fixture at load time. It also ships
`training_target.json`, the portable target spec used by replay scripts for the
released 70-feature model.

This artifact is the promoted incremental linker for Rust-backed
`Clusterer.predict_incremental(...)`. It is not intended to reproduce the
legacy incremental output. When Rust mode is selected and the extension plus
artifact pass validation, the target behavior is to use this promoted
retrieval/linker/gate path because it has shown better runtime and quality than
the long-standing legacy implementation.

## Reference-feature behavior

Models `v1.1` and `v1.2` were trained with `compute_reference_features=False`. That means they do not use features derived from cited references.

The disabled reference-derived features are:

- `references_authors_overlap`
- `references_titles_overlap`
- `references_venues_overlap`
- `references_author_blocks_jaccard`
- `references_self_citation`
- `references_overlap`

Practical consequence:

- For `v1.1` and `v1.2`, `papers.references` can be omitted or set to `null`.
- Signature fields are still required as usual.

If you use `v1.0`, you must provide the paper-reference lists needed for those features.

## Minimal input contract

Minimal paper entry for `v1.1` and `v1.2`:

```json
{
  "paper_id": 12345,
  "title": "My Paper Title",
  "abstract": "Optional but recommended for the has_abstract feature.",
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

Minimal signature entry:

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

## Name-count semantics compatibility

S2AND supports two runtime semantics for the name-count feature key used by `last_first_initial_count_min`:

- `legacy_full_first_token`: key is `<last> <first_token>`
- `initial_char`: key is `<last> <first[0]>`

Compatibility rules:

- `production_model_v1.1.pickle` and `production_model_v1.2.pickle` use `initial_char` with
  `s2and/data/name_counts.pickle`; that pickle stores keys like `smith j`, not `smith john`.
- In `ANDData(..., mode="inference")`, prediction automatically applies the semantics expected by the loaded model via the stored feature contract.
- Do not mix model artifacts and feature semantics without retraining.

## Minimal prediction flow

```python
from s2and.data import ANDData
from s2and.serialization import load_pickle_with_verified_label_encoder_compat

clusterer = load_pickle_with_verified_label_encoder_compat(
    "s2and/data/production_model_v1.2.pickle"
)["clusterer"]

dataset = ANDData(
    signatures="path/to/signatures.json",
    papers="path/to/papers.json",
    specter_embeddings="path/to/specter_embeddings.pkl",
    mode="inference",
    block_type="s2",
    n_jobs=8,
    name="my_dataset",
)

pred_clusters, pred_distance_matrices = clusterer.predict(dataset.get_blocks(), dataset)
```

`pred_distance_matrices` may be `None` when using memory-optimized fused clustering paths.

## Caching

Public cache control:

- `Clusterer.use_cache`
- `featurize(..., use_cache=...)`
- `many_pairs_featurize(..., use_cache=...)`
- `warm_rust_featurizer(..., use_cache=...)`

Semantics:

- `use_cache=True` enables the persistent pair-feature SQLite cache and the Rust featurizer disk cache.
- `use_cache=False` skips those persistent cache reads and writes.
- Same-process Rust featurizer reuse still stays enabled even when `use_cache=False`.

Recommended defaults:

- Repeated inference on the same dataset or pair set: `use_cache=True`
- One-shot jobs and experiments: `use_cache=False`

Full cache details: [caching.md](caching.md)

## Rust backend

`S2AND_BACKEND` controls runtime backend selection:

- `auto`: use Rust when available and capable, otherwise Python
- `rust`: strict Rust mode
- `python`: Python-only mode

Install contract:

- `uv pip install s2and`: Python-only runtime
- `uv pip install "s2and[rust]"`: Rust-enabled runtime when wheels are available

Full runtime contract: [rust/runtime.md](rust/runtime.md)

## Large blocks and incremental inference

For standard full-block prediction, subblocking keeps peak memory bounded. For
the promoted Rust incremental target, query batching should provide the memory
bound for the promoted retrieval/linker/gate path. The legacy incremental
implementation remains a fallback or compatibility mode, not the output target.

Standard large-block prediction:

```python
pred_clusters, _ = clusterer.predict(
    dataset.get_blocks(),
    dataset,
    batching_threshold=5000,
    desired_memory_use=5000 * 5000,
)
```

Incremental prediction with explicit RAM budget:

```python
result = clusterer.predict_incremental(
    block_signatures,
    dataset,
    batching_threshold=5000,
    total_ram_bytes=32 * 1024**3,
    max_chunk_pairs=50_000_000,
)

clusters = result["clusters"]
phase_b_mode = result["phase_b_mode"]
```

`phase_b_mode` tells you which path ran:

- `exact`: Phase B ran globally and is intended to match monolithic semantics.
- `subblock_local`: runtime fell back to approximate per-subblock behavior to stay within budget.

### Rust promoted incremental target

The target behavior is that `Clusterer.predict_incremental(...)` uses the
promoted Rust linker by default when `S2AND_BACKEND` selects Rust and the
extension has the required promoted-incremental capabilities. Legacy output
parity is not a release goal; the promoted path intentionally uses different
retrieval, linker, and margin-gate decisions.

`S2AND_BACKEND=rust` and `S2AND_BACKEND=auto` now route `predict_incremental`
through the promoted linker when backend resolution selects Rust. The temporary
`incremental_linker_private=True` switch remains available for focused
experiments. Promoted query batching is available: `batching_threshold` caps
the number of unassigned query signatures per promoted linker batch, while
`total_ram_bytes` derives the default batch size when the caller does not pass a
cap. The first meaningful promoted batch recalibrates rows/pairs per query for
remaining batches, and telemetry records predicted/observed RSS deltas.

Supporting docs:

- Subblocking behavior and tradeoffs: [subclustering.md](subclustering.md)
- Threading guidance: [threading.md](threading.md)
- Environment variables: [environment.md](environment.md)

## Warm-starting the Rust featurizer

For long-lived services, you can pre-warm once at startup:

```python
from s2and.feature_port import warm_rust_featurizer

warm_rust_featurizer(dataset, use_cache=True)
```

Use `use_cache=False` if you only want same-process warmup and do not want persistent disk-cache writes.

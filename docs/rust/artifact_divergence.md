# Rust Artifact Divergence and Format Migration Plan

Status date: 2026-03-02

Tracks artifact-level divergences between Python and Rust paths and the planned
format migration. See also: `docs/work_plan.md` (Bundle 5) for execution scheduling.

---

## Format selection rationale

All datasets are loaded entirely into memory — there is no partial or columnar read
pattern in any current code path. This means Parquet's primary advantages (column
pruning, predicate pushdown) do not apply.

| Format | Use case | Rationale |
|---|---|---|
| **MessagePack** | Dict-structured artifacts (signatures, papers, clusters, cluster_seeds, name_counts, orcid constraints) | Serializes existing Python dict shape with no schema translation; Python (`msgpack`) and Rust (`rmp-serde`) load it natively; also works in Scala via Jackson MessagePack; smaller than JSON, faster than pickle. |
| **Safetensors** | Embedding matrices (specter) | Designed for safe, fast, keyed tensor loading; Python (`safetensors`) and Rust (`safetensors` crate) are first-class; eliminates the current hidden Python-FFI pickle dependency in the Rust ingest path. |
| **LightGBM native text** | Trained models | Cross-language (Python, Rust via `lightgbm-rs`, Scala via `LightGBM4j`); not pickle-dependent. |
| **Plain text CSV** | name_tuples | Already shared by both paths; no change needed. |

### Formats rejected

| Format | Reason |
|---|---|
| **Parquet** | Columnar advantages (column pruning, pushdown) don't apply when loading entire datasets. Adds schema-translation overhead for no benefit. |
| **Pickle** | Python-only. Rust must call `pickle.load` via PyO3 FFI. Ruled out as a target for all types. |
| **JSON** (for dict artifacts) | Acceptable as intermediate but MessagePack is strictly better: binary, smaller, faster. |
| **JSON** (for embeddings) | No native float32 type — text-encoding bloats size 3.7x (qian specter2: ~183 MB binary vs ~684 MB JSON). For inventors_s2and (8.9M × 768): ~27 GB binary vs ~100 GB JSON — impractical. Plus float32→float64 precision loss. |
| **NPZ** | Fast for numpy in Python, but no ergonomic Rust loader. Safetensors has first-class Rust support. |

---

## Format decision table

| Artifact | Current Python | Current Rust | Target | Both natively? |
|---|---|---|---|---|
| `signatures.json` | JSON | JSON (serde_json) | MessagePack | Yes |
| `papers.json` | JSON | JSON (serde_json) | MessagePack | Yes |
| `clusters.json` | JSON | not loaded natively | MessagePack | Yes |
| `cluster_seeds.json` | JSON | JSON (serde_json) | MessagePack | Yes |
| `orcid_s2_constraints.json` | JSON | not loaded natively | MessagePack | Yes |
| `name_counts.pickle` | pickle | `name_counts_rust.json` (JSON) | MessagePack (single file) | **Yes — eliminates divergence** |
| `*_specter.pickle` / `*_specter2.pkl` | pickle | pickle via Python FFI | Safetensors | **Yes — eliminates FFI dependency** |
| `first_k_letter_counts_from_orcid.json` | JSON | not loaded by Rust | JSON (keep as-is) | Python-only for now |
| `lid.176.bin` (FastText) | FastText binary | FastText binary (Rust crate) | keep as-is | Yes |
| `name_tuples` txt | plain text | plain text (`fs::read_to_string`) | keep as-is | Yes |
| Trained model (LightGBM) | pickle (Clusterer) | not loaded by Rust | LightGBM native text | Ready for multi-language |
| `RustFeaturizer` disk cache | bincode via PyO3 | bincode (native) | keep as-is | Rust-only by design |
| Train/val/test pairs | CSV | not used by Rust | CSV (keep as-is) | Python-only |

---

## Current divergence map

| Area | Current divergence | Why it exists | Mitigation | Resolution target |
|---|---|---|---|---|
| `name_counts` artifact | Python loads `name_counts.pickle`; Rust native JSON ingest expects `name_counts_rust.json` shape | Rust `from_json_paths` cannot consume Python pickle | Keep comparator runs on `name_tuples="filtered"` and track name-count source telemetry | **Resolved when format migration lands**: single `name_counts.msgpack` read by both; `name_counts_rust.json` and `export_name_counts_for_rust.py` become obsolete |
| `name_tuples` source | Python supports both `s2and_name_tuples_filtered.txt` and full `s2and_name_tuples.txt`; Rust ingest defaults to filtered file | Historical compatibility with older experiments | Force `name_tuples="filtered"` in parity/perf gates | Single default variant at runtime; non-default as offline experiment-only |
| ORCID first-k counts normalization | `first_k_letter_counts_from_orcid.json` generated with legacy normalization; runtime has compatibility lookup | Artifact predates current normalization | Keep compatibility shim during migration | Regenerate under current normalization; remove compatibility fallback |
| Specter embedding load path | Python loads pickle natively; Rust `from_json_paths` calls Python `pickle.load` via PyO3 FFI | Specter was always pickle; no Rust-native pickle reader | Acceptable FFI cost during migration (one-time load) | **Resolved when format migration lands**: Safetensors loaded natively by both; hidden Python dependency eliminated |

---

## Deferred unification backlog

After format migration milestones:

1. **`name_counts`**: collapse into one canonical format used by both paths.
   Target: `name_counts.msgpack` replaces `name_counts.pickle` and `name_counts_rust.json`.
   Then delete `scripts/export_name_counts_for_rust.py` and the `_rust_name_counts_artifact_path()` /
   dual-path logic in `feature_port.py`.

2. **`name_tuples`**: collapse to one default variant.

3. **ORCID first-k counts**: regenerate with current normalization; delete compatibility shims.

4. **Specter embeddings**: replace pickle with Safetensors for all specter artifacts.
   Rust `from_json_paths` then loads via the `safetensors` crate instead of calling `pickle.load`
   through PyO3.

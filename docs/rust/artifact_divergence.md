# Rust Artifact Divergence: Temporary Operating Mode and Deferred Unification

Status date: 2026-02-28

See also:
- Current execution bundles: `docs/work_plan.md` (Bundle 5: staged dual-read + converters + fixtures; defer large artifact regen while evidence is local-only).

## Why this doc exists
Tracks artifact-level divergences between Python and Rust paths and the format migration plan.

---

## Artifact format analysis

### Format selection rationale

All datasets are always loaded entirely into memory — there is no partial/columnar read pattern in
any current code path. This means Parquet's primary advantages (column pruning, predicate pushdown,
Spark-native analytics) do not apply. The format decision simplifies accordingly:

- **MessagePack** for all dict-structured artifacts (signatures, papers, clusters, cluster_seeds,
  name_counts, orcid constraints). Rationale: serializes the existing Python dict shape directly
  with no schema translation; both Python (`msgpack`) and Rust (`rmp-serde`) load it natively;
  works in Scala via Jackson MessagePack; smaller than JSON, faster than pickle.
- **Safetensors** for embedding matrices (specter). Rationale: designed specifically for safe,
  fast, keyed tensor loading; Python (`safetensors` lib) and Rust (`safetensors` crate) are
  first-class; eliminates the current hidden Python-FFI pickle dependency in the Rust ingest path.
- **LightGBM native text format** for trained models. Rationale: cross-language (Python, Rust via
  `lightgbm-rs`, Scala via `LightGBM4j`); not pickle-dependent; currently Python-only but format
  is ready for other consumers when needed.
- **Plain text CSV** for name_tuples — already shared by both paths, no change needed.

### Format decision table (per artifact type)

| Artifact | Current Python format | Current Rust format | Target format | Shared natively? |
| --- | --- | --- | --- | --- |
| `signatures.json` | JSON | JSON (serde_json) | MessagePack | Yes |
| `papers.json` | JSON | JSON (serde_json) | MessagePack | Yes |
| `clusters.json` | JSON | not loaded natively | MessagePack | Yes |
| `cluster_seeds.json` | JSON | JSON (serde_json) | MessagePack | Yes |
| `orcid_s2_constraints.json` | JSON | not loaded natively | MessagePack | Yes |
| `name_counts.pickle` | pickle | `name_counts_rust.json` (JSON) | MessagePack (single file) | **Yes — eliminates divergence** |
| `*_specter.pickle` / `*_specter2.pkl` | pickle | pickle via Python FFI | Safetensors | **Yes — eliminates FFI dependency** |
| `first_k_letter_counts_from_orcid.json` | JSON | not loaded by Rust | JSON (keep as-is) | Python-only for now |
| `lid.176.bin` (FastText) | FastText binary | FastText binary (Rust crate) | keep as-is | Yes |
| `name_tuples` txt | plain text | plain text (fs::read_to_string) | keep as-is | Yes |
| Trained model (LightGBM) | pickle (Clusterer) | not loaded by Rust (Python-only for now) | LightGBM native text | Ready for multi-language when needed |
| `RustFeaturizer` disk cache | bincode via PyO3 | bincode (native) | keep as-is | Rust-only by design |
| Train/val/test pairs | CSV | not used by Rust | CSV (keep as-is) | Python-only |

### Formats rejected and why

**Parquet**: columnar format; only useful when reading a subset of columns or rows. Since every
code path loads the entire dataset, columnar layout adds schema-translation overhead for no
benefit. Ruled out for all artifact types.

**Pickle**: Python-only. Cannot be read natively by Rust without shelling out to Python via PyO3
(which is what the current specter path does). Ruled out as a target format.

**JSON** (as target for dict-structured data): human-readable but slow and large. Already used as
an intermediate for `name_counts_rust.json`; MessagePack is strictly better for a binary artifact.

**JSON** (as target for embeddings): explicitly ruled out despite `maybe_load_specter` accepting a
plain dict (meaning the in-memory representation is format-agnostic and no pickle is technically
required). The problem is on-disk size and parse speed. JSON has no native float32 type — values
must be text-encoded. For qian specter2 (59,432 × 768): binary float32 ≈ 183 MB vs JSON ≈ 684 MB
(3.7× larger), plus slow text float parsing and a float32→float64 precision round-trip. For
inventors_s2and (8,913,392 × 768): binary ≈ 27 GB vs JSON ≈ 100 GB — completely impractical.
Safetensors stores float32 exactly, is mmap-able, and has first-class Python and Rust support.

**NPZ** (`np.savez`): fast for numpy in Python, but no ergonomic Rust loader. Ruled out in favor
of Safetensors which has first-class Rust support.

---

## Current divergence map

| Area | Current divergence | Why it exists now | Mitigation in place | Resolution target |
| --- | --- | --- | --- | --- |
| `name_counts` artifact | Python path loads `name_counts.pickle`; Rust native JSON ingest expects `name_counts_rust.json` shape (`first_dict`, `last_dict`, `first_last_dict`, `last_first_initial_dict`) | Rust `from_json_paths` cannot directly consume Python pickle | Keep comparator runs on `name_tuples="filtered"` and track name-count source telemetry to detect drift | **Resolved when format migration lands**: single `name_counts.msgpack` file read by both paths; `name_counts_rust.json` and `export_name_counts_for_rust.py` become obsolete |
| `name_tuples` source | Python supports both `s2and_name_tuples_filtered.txt` and full `s2and_name_tuples.txt` (via `name_tuples=None`); Rust JSON ingest defaults to filtered file unless explicit path override | Historical behavior and backwards compatibility with older experiments | For parity/perf gates, force `name_tuples="filtered"` (already done in comparator); do not use full tuples in migration benchmarks | Single default tuple variant for runtime; keep non-default variants as offline experiment-only inputs |
| ORCID first-k counts normalization | `first_k_letter_counts_from_orcid.json` was generated with legacy normalization; runtime has compatibility lookup behavior | Artifact predates current normalization behavior | Keep compatibility shim during active migration; treat as known temporary compatibility code | Regenerate artifact under current normalization and remove compatibility fallback |
| Specter embedding load path | Python loads pickle natively; Rust `from_json_paths` calls Python's `pickle.load` via PyO3 FFI — not a true native Rust load | Specter was always pickle; no Rust-native pickle reader existed | Acceptable during migration since FFI cost is a one-time load | **Resolved when format migration lands**: Safetensors loaded natively by both Python and Rust with no FFI; hidden Python dependency in Rust ingest path eliminated |

---

## Deferred unification backlog (after migration milestones)

1. Collapse `name_counts` into one canonical format + loader used by both Python and Rust paths.
   Target: `name_counts.msgpack` replaces both `name_counts.pickle` and `name_counts_rust.json`.
   Then delete `scripts/export_name_counts_for_rust.py` and the `_rust_name_counts_artifact_path()` /
   dual-path logic in `feature_port.py`.
2. Collapse runtime `name_tuples` behavior to one default variant.
3. Regenerate ORCID first-k counts with current normalization and delete compatibility shims.
4. Remove hidden Python FFI dependency in Rust specter load path.
   Target: Safetensors replaces pickle for all specter artifacts; Rust `from_json_paths` loads
   specter natively via the `safetensors` crate instead of calling Python's `pickle.load` through PyO3.

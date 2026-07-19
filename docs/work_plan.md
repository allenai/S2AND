# Canonical-v2 Release Work Plan

Status date: 2026-07-10

This is the authoritative status for the in-process canonical-v2 pull request.
It replaces the earlier defect narrative with the disposition of every audited
item, the evidence already collected, and the work that genuinely cannot start
until the new names, datasets, and model artifacts exist.

## Current outcome

The artifact-independent implementation is nearly complete, but the branch is
not a releasable production package yet:

- Python and Rust implement one `canonical_v2` normalization contract and
  `FEATURIZER_VERSION = 10`.
- Production Rust inference enters through strictly validated Arrow artifacts.
- Pairwise and promoted-linker bundles are immutable, provenance-bound release
  units published from validated sibling staging with one directory rename.
- Promoted query-disallow decisions are request-global and deterministic across
  input order and batch/window size. Cross-batch conflicts use complete
  single-query rescoring. RAM limits are re-read and enforced after each major
  allocation.
- Name-count, Arrow-index, model, linker, and mutable-sidecar provenance is
  content-verified. The new ORCID prefix-count path requires a versioned
  canonical generation and has no legacy JSON fallback.
- Production pairwise contracts bind the exact canonical name-tuple and ORCID
  prefix-count data hashes. Training records them; export and load compare them
  to the installed artifacts; the linker binding covers them through the
  ordered feature-contract digest.
- Models using global name counts bind the exact generation, source identity,
  source snapshot, and selected-row digest at every Python, Arrow, and
  prebuilt-Rust-featurizer boundary before feature work. The Rust binding is
  runtime-only and does not change serialized artifacts.
- Python `ANDData` now opens the same validated, shared, memory-mapped
  `name_counts_index` as Rust and materializes four count scalars per signature
  in bounded batches. The dictionary-injection API, package pickle loader, and
  process-wide Python count dictionaries have been removed. The v1
  `pickle_sha256` field remains historical source-lineage metadata only; no
  runtime path opens that pickle.
- Production Arrow bundles require a canonical full-digest generation
  inventory. Request-local sidecars remain outside that immutable identity and
  cannot poison generation-cache reuse.
- No production model or default declaration is packaged during the cutover.
  The historical v1.21 source bundle is not loadable by canonical-v2, and
  `load_production_model(path)` requires an explicit complete canonical bundle.

The repository owner controls merging. The release risk is therefore not an
automatic merge; it is deliberately publishing a mixed or insufficiently
measured artifact set.

## Status vocabulary

- **Done:** code, focused regression tests, and applicable local evidence are
  complete without the missing release data.
- **Artifact:** code is complete, but the real canonical artifact must be
  generated and validated.
- **Retrain:** the behavior/feature contract changed and needs v1.3 training and
  quality re-baselining.
- **Measure:** synthetic/local evidence exists, but release-scale runtime and
  peak-RSS evidence still must be captured.
- **Held:** intentionally not changed because another explicit requirement
  would be violated.

## External release gates

These are the only tasks blocked on the pending names/datasets/artifacts:

1. Generate the canonical versioned `name_counts` generation and its binary
   `name_counts_index` from the reviewed source snapshot.
2. Generate the canonical versioned ORCID prefix-count generation and publish
   its pointer manifest.
3. Audit nullable `signatures.author_position` in every intended release
   dataset, then either repair the source rows or activate the non-null schema.
4. Re-export benchmark names by signature-ID join and report join/divergence
   counts.
5. Retrain and bundle v1.3 pairwise and promoted-linker models from exactly
   those immutable artifacts.
6. Run pairwise, clustering, subblocking, parity, quality, throughput, and
   peak-RSS gates on the release candidate.
7. Package the complete v1.3 bundle and run the installed-wheel pairwise and
   incremental smoke against its explicit path.

The generation contract and acceptance thresholds are in
[normalization_migration_blocked.md](normalization_migration_blocked.md).

## Audit disposition: release and artifact integrity

| ID | Status | Implemented disposition or remaining dependency |
|---|---|---|
| A1 | Done + Artifact + Retrain | Public production loading requires an explicit complete native bundle path. No model or default declaration is distributed during the cutover; packaging v1.3 waits for all gates. |
| A2 | Done + Artifact | Clean-built Python/Rust wheels installed outside the source tree pass a synthetic public Arrow incremental smoke. The real explicit-bundle smoke awaits v1.3. |
| A3 | Done | Rust publishes first; Python publication depends on the exact-version Rust availability probe and cannot use a Python-only escape hatch. |
| A4 | Done | Wheel/sdist validation accepts the cutover distribution with no model, validates any future packaged bundle without silently selecting it at runtime, and rejects duplicate or undeclared `production_model_v*` assets. It also rejects the obsolete unversioned ORCID JSON. |
| A5 | Done + Artifact | Name counts carry normalization, generation, source-row digest, and v1 source-pickle identity; provenance propagates into indexes/models. Python and Rust runtime lookup use only the verified binary index. The real generation is pending. |
| A6 | Done | Public Arrow training/featurizer construction requires an explicit expected normalization version; absence and mismatch fail before feature work. |
| A7 | Done | Bundle export requires source normalization provenance and cannot relabel a legacy model canonical. |
| A8 | Done + Retrain | Linker metadata is bound to the pairwise boosters, ordered feature contract, normalization, and featurizer version; finalization and load both enforce it. |
| A9 | Done + Artifact + Retrain | ORCID counts load lazily from a contained immutable generation with verified manifest, metadata/data checksums, normalization, pair semantics, source identity, and cardinality. No unversioned fallback remains. Production contracts bind the exact data SHA-256, and export/load compare it to the packaged priors. The real generation is pending. |
| A10 | Done + Artifact | Dataset/count/model/linker authorities compare exact versions and content identities before feature work, including already-built Rust featurizers. Every present Arrow/count-index input has SHA/size coverage. Final cross-artifact equality needs the real release unit. |
| A11 | Done | Immutable Arrow generations are full-digest validated once and then trusted by exact generation identity. Same-path mutation is unsupported; changed content is published as a new generation. |
| A12 | Done | Pairwise and complete-bundle writers use validated sibling staging. Finalization copies the pairwise source into a complete staging tree and renames it once to a nonexistent final path; no bundle is completed in place. |
| A13 | Done + Retrain | Missing, nonfinite, or regressed metrics fail before promotion. `--allow-metric-drift` is rejected up front with promotion flags. Real gates await v1.3. |
| A14 | Done + Artifact + Retrain | Omitted `name_tuples` selects a strict metadata/SHA/cardinality/semantics-verified canonical artifact; only an explicit empty set disables aliases. Production contracts bind its exact data SHA-256, while Python and Rust retain one strict validation policy rather than parallel identity APIs. Same-path regeneration is offline/fail-closed, not crash-atomic, until a generation-pointer schema is approved. |
| A15 | Done | Missing, unreadable, or invalid tuple artifacts fail in Python and Rust instead of becoming an empty alias policy. |
| A16 | Done | Bundle schemas require exact checksummed runtime files and reject traversal, absolute paths, symlink escapes, duplicates, and undeclared files. |
| A17 | Done + Retrain | Pairwise metadata, clusterer config, and actual Rust booster feature counts must agree before prediction. |
| A18 | Done + Retrain | Bundle/clusterer schema, model family, finite ranges, linkage, epsilon, hybrid weights, batch sizes, and iterations are strictly validated. |
| A19 | Done + Retrain | Canonical production load rejects either booster on featurizer-version drift instead of warning. |
| A20 | Done + Retrain | The v1.0-v1.2 pickles are removed. v1.21 remains only as an explicit historical source/parity artifact and is not packaged or accepted by the canonical loader. |
| A21 | Done + Artifact | Synthetic canonical bundles cover writer, loader, finalizer, classifier, and corruption tests. The real explicit v1.3 integration gate awaits the artifact. |

## Audit disposition: runtime correctness, parity, and state

| ID | Status | Implemented disposition or remaining dependency |
|---|---|---|
| B1 | Done + Retrain | Initial decisions are scored in bounded batches and globally ordered by require/score/signature ID. A conflicted lower-priority query is rebuilt and rescored from its complete single-query plan with the winner excluded; compact query replay state is gone. Outcome telemetry is request-invariant. |
| B2 | Done + Measure | Query/window limits refresh after planner, featurizer, and scorer allocations; oversized queued work is discarded, re-sliced, and re-planned before scoring. Planning uses the loaded model's exact final/pairwise/aggregate widths (currently 53/35/18), not stale generic constants. Release-scale evidence remains. |
| B3 | Done | Featurizer reuse is keyed by exact normalized material paths, validated full-digest generation, non-seed settings, and seed version. The validated immutable generation is trusted without filesystem watchers; request-local sidecars remain outside its identity. |
| B4 | Done | Seed require/disallow sidecars are parsed once into request-local state with no process-global parsed cache. Altered-presplit cache identity still uses stable full-file digests for mutable disallow and altered-profile inputs. |
| B5 | Done | Name-count generation writes/fsyncs a published generation, serializes pointer replacement, and never restores stale manifest text after a competing writer succeeds. |
| B6 | Done + Artifact + Retrain | ORCID prefix pairs are canonical unordered keys in generation and lookup; reverse input/subblock order is equivalent. |
| B7 | Done | Alias pairs are unordered/deduplicated in Python and Rust, including custom one-direction inputs. |
| B8 | Done + Artifact | Empty canonical names are rejected with metrics; rows are deterministic; a reviewed explicit per-ORCID bound fails before quadratic expansion and never truncates. |
| B9 | Done + Retrain | Full-name/query-author facets use canonical fields only; empty canonical fields cannot fall back to raw text. |
| B10 | Done | Python and Rust subblocking reject duplicate IDs after string coercion and use explicit runtime invariants rather than correctness `assert`s. |
| B11 | Done + Retrain | Only one `@` with nonempty whitespace-free local/domain parts is valid; malformed email yields missing evidence in both runtimes. |
| B12 | Done + Retrain | Python CLD2 uses explicit plain-text mode and matches the Rust input policy. |
| B13 | Done | Rust excludes the query only from local10 evidence, matching Python same-paper behavior. |
| B14 | Held for dataset audit | Making `signatures.author_position` non-null before inspecting the pending release datasets could reject legitimate rows without a repair plan. No silent coercion was added. |
| B15 | Done | Python and Rust reject duplicate `(paper_id, position)`, empty author names, and dangling paper-author references before filtering/feature work. |
| B16 | Held for throughput | True zero-Rust `backend="python"` scoring measured about 27% slower. Explicit Python still uses the native scorer; telemetry reports the actual scorer. This will not change without a faster Python path or owner acceptance. |
| B17 | Done + Retrain | Maintained deterministic parity is `1e-6`, exact for discrete/count/boolean fields; the old global `1e-3` and language exception are gone. |
| B18 | Done + Retrain | Title normalization preserves Unicode letters and digits across Python, retrieval, staged Rust, and raw Arrow paths. |
| B19 | Done + Retrain | Incremental six-decimal rounding is explicit ties-to-even before float32 conversion in both runtimes. |
| B20 | Done | Attached paths are absolute; manifest-relative paths resolve only against the manifest directory and survive CWD changes. |
| B21 | Done + Retrain | Stored language evidence is a complete `(predicted_language, is_reliable, reliability)` triple. Reliability must be finite in `[0,1]`; unreliable rows are exactly zero. Python and Rust reject partial or malformed triples and preserve pair order. |
| B22 | Done | Raw candidate planning has two constructible modes: strict declared query signatures and explicit automatic queries. The empty-sidecar/boolean-bypass state is gone. |
| B23 | Done | Numeric telemetry aggregation is field-specific. Unknown numeric fields cannot silently become counters and corrupt request-level measurements. |
| B24 | Done + Artifact | `ANDData` accepts only a path or shared `NameCountsIndex` handle. The mutable dictionary seam and pickle loader are deleted. Python preprocessing deduplicates each 2,048-signature key batch before one four-column native lookup and stores only the four scalar results per signature. Historical full-index KISTI throughput and RSS pass the real preprocessing gate below; canonical-artifact confirmation remains. |

## Audit disposition: scale, operations, packaging, and documentation

| ID | Status | Implemented disposition or remaining dependency |
|---|---|---|
| C1 | Done + Measure | Rust chooses dense borrowed lookup for common dense indices and compact remapping only when cheaper; pair representations are not duplicated unnecessarily. Release-scale evidence remains. |
| C2 | Done + Measure | Arrow training iterates mmap record batches directly into final objects, rejects cross-batch duplicates, and releases temporary batches. A true Rust-native training representation remains a future architecture option, not a claim of this PR. |
| C3 | Done + Measure | Name-count writing uses bounded buffers/runs, disk preflight, immutable generation publication, and run/buffer/temp-byte telemetry. Real multi-million-key evidence awaits canonical counts. |
| C4 | Done + Measure | Native scoring retains float32 input, plans persistent output plus bounded scratch, and chunks under pressure without probability drift. The specialized scorer is faster locally and immutable scorer state is shared across clusterer deep copies. Real release-candidate profiling remains. |
| C5 | Done + Artifact | Count scripts are import-safe and require explicit source snapshot/output; warehouse access needs `--run-full`; fixture, limit, and dry-run modes are available. |
| C6 | Done | Name-count and ORCID outputs use staged immutable generations, validation/fsync, a publication lock, and pointer-last commit with failure/race regressions. ORCID count JSON uses deterministic native serialization with byte-identical SHA/output and lower peak RSS. |
| C7 | Done | The parent S2AND package remains Python 3.11.x as declared. The separately distributed Rust extension builds and exercises CPython 3.11-3.13 wheels; cp310 is removed. |
| C8 | Done | `.gitattributes` establishes LF defaults for Python/JSON/Markdown and `git diff --check` is a release gate. Two pre-existing CRLF scripts have explicit byte-preserving exceptions to avoid full-file review churn. |
| C9 | Retrain | Run the pinned Sinonym/fastText/reference-feature ablation and quality comparison on v1.3; do not restore features without measured gate failure. |
| C10 | Done | README, data/environment/caching/production/Rust/release docs and this ledger describe the current single-mode, artifact-blocked state. |
| C11 | Done | Release commands obey the repository `uv` contract; executable raw `python`/`pip` workflow calls are removed. |
| C12 | Done | Rust CI fails hard if the native module or required ABI is absent. Windows/macOS jobs exercise the wheels they built. Version-changing PRs build every distribution; the `force-build` label opts into the combined clean installed-wheel Arrow smoke, which remains mandatory before Python publication. |

## Performance and memory evidence collected so far

These are local/synthetic gates, not substitutes for the final release workload.
No measured key junction increases peak RAM by more than 10%.

| Junction | Throughput/wall result | Peak/live RAM result | Disposition |
|---|---:|---:|---|
| Validated-generation hot-cache check | 12.76 us/hit (78.4k/s), +4.48 us for exact path/generation binding | No additional retained artifact copy | Accepted fixed build/request overhead; no pair-loop work |
| Arrow training ingestion | +0.95% throughput | -24.2% peak RSS | Accepted; realistic dataset still required |
| Name-count Arrow writing | +30.6% throughput | -74.8% peak RSS | Accepted |
| Name-count binary index, 500k records | +50.6% throughput | -15.8% peak incremental RSS | Accepted; exact output SHA |
| Name-count binary index, 1.1M external sort | +32.9% throughput | -16.5% peak incremental RSS | Accepted; exact output SHA |
| Native float32 scorer, 200k x 33 | +34.9% throughput | -73.8% per-call RSS | Accepted; exact probability parity |
| Native float32 scorer, 100k x 53 | +26.8% throughput | -74.2% per-call RSS | Accepted; exact probability parity |
| Retained native scorer/model | No scoring loss; scorer is shared across deep copies | +7.0% to +7.39% retained model RSS | Accepted below 10%; deep copies no longer duplicate model vectors |
| Promoted request artifact reuse | Removes per-request reload/hash/model parse | One artifact retained once | Accepted |
| ORCID generator, 40k rows/230,754 pairs | +3.6% throughput | -18.3% peak RSS | Accepted; spawned-process race also passes |
| ORCID JSON publication, 300k records | +713% throughput (0.894M -> 7.264M records/s) | -13.5% incremental peak RSS (11.19 -> 9.68 MB) | Accepted; byte-for-byte and SHA identical |
| Historical Python name-count dictionaries, 35.4M entries | Cold unpickle 17.61 s; matched KISTI batch lookup 5.198M cells/s | +3.390 GiB retained; +3.885 GiB peak | Removed baseline |
| Python boundary over full historical mmap index | Native open 0.00083 s; exact KISTI output 1.860M cells/s; digest matches dictionaries | +0.52 MB at open; +197.4 MB benchmark peak | Lookup-only microbenchmark is slower, so lookup stays outside pair loops and uses bounded Python-side deduplication |
| Full KISTI Python signature preprocessing, 40,383 signatures | Median 3.089 s without counts vs 3.308 s with index (+7.1%); best-run delta +9.0%; warm-dict estimate narrows median delta to about +6% | Replaces the 3.390 GiB retained dictionaries; batch temporaries are capped at 2,048 rows | Accepted below 10% at the real junction; rerun on canonical artifact |
| Exact name-count binding, Python / Arrow / prebuilt Rust | 17.5 / 111 / 19.2 us per request | 793 B / 11.2 KB / 1,039 B transient; roughly 250-350 B retained in Rust | Accepted fixed boundary work; zero pair-loop work |
| Canonical name-tuple cache | Subsequent loads ~0-0.002 ms vs 39-41 ms uncached | One immutable artifact retained | Accepted |
| Dense Rust signature lookup, 100k signatures/2M pairs | 10.3x faster | -9.1% peak RSS; live layout unchanged | Accepted |
| Sparse Rust signature lookup, 10M index range | 6.6x faster | -95.2% peak RSS; -99.5% live layout | Accepted |
| True Python scorer rollback | -27.0% throughput | No compensating benefit established | Rejected under B16 |

The historical promoted-incremental profile on the local PubMed `r agarwal`
fixture remains a useful baseline: mmap-backed name-count reads changed p50 wall
2.18 s -> 2.01 s and peak RSS 3.84 GB -> 3.02 GB. It predates canonical-v2
and must be rerun on v1.3.

## Verification already required for this code state

Before handoff, run from the repository root:

```powershell
uv run --no-sync pytest -q
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global
$env:PYO3_PYTHON=(Resolve-Path '.venv\Scripts\python.exe').Path
uv run --no-sync cargo fmt --manifest-path s2and_rust/Cargo.toml -- --check
uv run --no-sync cargo test --manifest-path s2and_rust/Cargo.toml --lib --no-default-features
git diff --check
```

The release workflow additionally builds wheel/sdist plus the Rust wheel,
installs them outside the source checkout, verifies distribution contents, and
runs the public synthetic Arrow incremental smoke.

Latest local verification for this working tree; packaging rows record the most
recent completed release-gate run and must be repeated for the canonical bundle:

- Python: 1,562 passed, 3 skipped in 166.62 s, plus 6/6 focused name-count
  writer tests after the final type-only cleanup.
- Rust: 84 passed, 4 ignored reproducible benchmark-only tests.
- Ruff lint and repository formatting: clean.
- CI-scoped `ty` check for `s2and`: clean.
- The latest clean wheel/sdist content verification passed; neither archive contains the
  obsolete unversioned ORCID runtime file.
- The latest clean Python 3.11 install verified the built Python and Windows Rust
  wheels, passed the fail-hard ABI smoke, and completed the public promoted
  Arrow smoke with 3 signatures, 2 clusters, and `query_view=raw_arrow`.

## Verification after the new artifacts arrive

Do not run the warehouse generation, full retrain, or release-scale profile
without the repository owner's explicit approval. First run bounded fixtures,
then capture the full command, source snapshot/generation IDs, logs, record/key
counts, quality metrics, wall time, scorer calls, and peak RSS.

The release is done only when:

1. all eight external release gates above are complete;
2. every artifact identity agrees across dataset, counts, tuples, ORCID priors,
   pairwise model, and linker;
3. no production-model test depends on a legacy or implicit default;
4. the installed exact wheels load the explicit complete candidate and run
   real pairwise and incremental fixtures;
5. v1.3 quality, parity, throughput, and peak-RSS gates pass; and
6. publication consumes only that already-validated immutable release unit.

## Retained watchlist and standing decisions

- Compact-linker retrieved-candidate scoring still raises `NotImplementedError`
  for `partial_supervision`. The production request path does not reach it; the
  intentional failure remains pinned by a regression test.
- Correctness and provenance override compatibility. Do not relabel a legacy
  artifact or introduce fallback-heavy dual normalization modes.
- `s2and.arrow_inputs` is the production validation authority.
- Full scans are explicit compatibility/test opt-ins, never silent fallbacks.
- Large generation, retraining, or paid/internal-query jobs require a tiny
  fixture first and explicit owner approval.

The audit also rechecked and rejected false positives around probability
orientation, native LightGBM empty/NaN arithmetic, FNV batch over-selection,
all-zero SPECTER rows, canonical tuple symmetry, space-insensitive surname
comparison, and the deliberate missing-first compatibility behavior.

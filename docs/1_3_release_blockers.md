# S2AND v1.3 implementation blockers

Status date: 2026-07-26

This ledger tracks code, data-producer, and workflow work that must land before
the corresponding command in [1_3_release_todo.md](1_3_release_todo.md) is
run. It is not an operator checklist. The runbook owns execution order; this
file owns blocker status and the smallest evidence needed to close each blocker.

## Fixed v1.3 scope decisions

- Keep the current persisted formats for this release:
  - `s2and_name_tuples_v3`;
  - `name_counts_index_v2` with `name_counts_provenance_v3`;
  - `orcid_prefix_counts_v2`; and
  - `s2and_production_model_bundle_v5`.
- Reject format cleanup, compatibility readers, and schema migrations from the
  v1.3 critical path. A later format change needs its own proposal and tests.
- Publish `production_model_v1.3` as an immutable external bundle. Do not add a
  packaged default model or change `load_production_model(None)` for v1.3.
- Preserve the exact evaluated linker artifact and
  `candidate_target.json`. Complete-bundle assembly validates and copies those
  bytes without fitting, materializing, or rewriting artifact metadata.
- Do not add `evaluation_start.json`, runtime `promotion.json`,
  `release_inputs.json`, or mandatory launch/process/completion triples.
- Keep release evidence separate from inference-time bundle validation.

## Status meanings

- **Closed**: no remaining v1.3 work.
- **CI**: outstanding work that closes in ordinary regression coverage, not a
  separate release authority or a claim that the test already exists.
- **Partial**: useful behavior exists, but the named closure evidence is
  incomplete.
- **Open**: implementation or real release evidence is still missing.
- **Merged**: the stable ID remains, but another blocker supplies its closure.

## Stable blocker ledger

| ID | Status | Required resolution | Smallest closure evidence |
|---|---|---|---|
| B01 | Open | Choose one unused synchronized Python/Rust package version and record the final model/data/package matrix before the release commit is frozen. | `uv run --no-project python scripts/sync_version.py --check`, package-index nonexistence evidence, and reviewed version matrix. |
| B02 | Closed | Retain tracked `scripts/production/counts/_run_support.py`. | Clean-checkout imports of both count producers pass. |
| B03 | Closed | Retain the module-only count CLI; do not add another bootstrap path. | Subprocess `--help` and bounded fixture runs pass from the repository root. |
| B04 | Closed | Hash the Rust/package tree into both native cache keys, exclude generated build outputs, and do not restore an old cached virtualenv through a broad fallback key. Clean isolated release-wheel construction belongs to B26. | The workflow contract test covers both exact cache keys and the no-fallback virtualenv rule. |
| B05 | Open | Run v1.21 in its compatible frozen runtime on the same frozen independent-gold manifests. Historical commit `e54c6ba` explicitly overrides versions `1.2`/`1.21` to runtime EPS `0.65`; the current canonical loader and stale stored threshold are not the baseline authority. Isolate its legacy path config and frozen name-count artifact from the canonical-v2 `S2AND_PATH_CONFIG`, with no ambient download fallback. Retain model/runtime/data identity, effective EPS, predictions, and only metrics used by the v1.3 quality policy. | Reproducible baseline report from the reviewed historical code/artifact/data identity; observed post-load EPS is `0.65`. |
| B06 | Open | Deterministically join canonical benchmark names into the exact fields consumed by `ANDData`, preserve raw names in one audit map, and inventory the result. | A fixture verifies joined values after loading, Arrow conversion, and pairwise featurization. The report records source/target counts, duplicate/missing IDs, joined/unjoined counts and rate, per-field divergence, representative differences, source snapshot, and output digest. |
| B07 | Open | Generate deterministic linker assignments at `base_group_id` granularity. | Fixed-input output is deterministic and no base identity appears in more than one split. |
| B08 | Closed | Retain one real-shaped optional-evaluation fixture covering `s_park_eval_path`, `s_lee_eval_path`, and a flattened `extra_eval_paths.<slug>` entry. Actual release evidence comes from B10. | Fixture proves source resolution, normalized model metadata, and consumer loading. |
| B09 | Open | Regenerate the published/downloaded linker replay Arrow bundle from frozen raw JSON, SPECTER2 inputs, and the selected name-count generation. | Deep validation passes and every manifest carries the required normalization and generation bindings. |
| B10 | Partial | Add one supported `validate-source-bundle` command that validates the B19 manifest, support files, required tables, assignments, Arrow manifests, and name-count bindings. | Mutation of any declared source member fails; the final release source bundle passes. |
| B11 | Closed | Retain the one digest-bound `preflight-training-inputs` boundary. It validates fixed-pair schemas, labels, duplicates, and unordered overlap plus random-block signature isolation, then emits one plan containing train/validation paths and only sealed test manifest/member digests. Full training accepts that plan instead of discovering a data root. | Focused tests cover success, invalid labels, duplicate/reversed pairs, fixed/random test overlap, wrong plan identity, and the mocked `ANDData` boundary receives no test path. |
| B12 | Closed | Replace the two-step release EPS branch with one validation-only calibration command driven by a frozen calibration spec. It always emits one fresh calibrated bundle and one report; Stage 6 alone produces v1.3 candidate cluster-test scores. | Tests cover wrong-input failure before matrix work, deterministic selection/tie-break, changed and unchanged EPS, byte preservation, and reload. |
| B13 | Partial | Fit on train/calibration, serialize the learned payload, then evaluate that exact payload on the frozen linker population. Atomically emit identity-only `candidate.json` plus `linker_evaluation_report.json`; neither makes the aggregate release decision. The replay schema's metrics in `candidate_target.json` are explicitly derived from, and must exactly equal, the linker report. | No test row is opened before all bindings validate; retries reuse the learned payload rather than refitting; candidate artifacts and predictions are fully inventoried; candidate-target/report metric equality is enforced. |
| B14 | Closed | The production README already places the seed target outside fresh output paths. Retain that behavior. | CLI/help smoke and the existing path-separation check pass. |
| B15 | Partial | Use an external immutable model bundle for v1.3; no packaged default. Tighten the `release_candidate` distribution phase so it forbids `default_production_model.json` and every packaged `production_model_v*` directory instead of accepting a declared default. Make repository-state package tests phase-neutral so the Stage 1 ORCID promotion does not require a behavior/test edit; synthetic tests cover both phase contracts and the explicit verifier invocation owns actual-tree phase evidence. | Focused source tests reject a default declaration, wheel/sdist tests reject default/model members, the exact final tree passes `--phase release_candidate`, and explicit-path model loading remains documented. |
| B16 | Partial | Retain the clean-installed `release-candidate` smoke that validates exact candidate data/model/name-count manifests, selects the configured name-count path, and exercises real pairwise and incremental Rust-backed predictions. The authoritative workflow still must download and run it against the final artifacts. | Exact wheel/sdist plus candidate artifacts pass; a wrong model or name-count digest fails. |
| B17 | Open | Add one no-overwrite data/model publisher with dry-run support, manifest-last publication, remote verification, and explicit abandoned-prefix recovery after partial failure. | Bounded namespace rehearsal passes; wrong/preexisting bytes fail; an interrupted prefix is never reused. |
| B18 | Merged into B19 | Build the final data-root manifest once after benchmark and replay trees are complete. Do not create register-versus-refresh transition state. | B19 assembly covers benchmark datasets, nested replay data, name counts, and required root helpers. |
| B19 | Closed | Add one `assemble-source-bundle`/final-root assembly path with a manifest inventorying every consumed non-Arrow member and nested Arrow manifest by logical role, size, and SHA-256. | B10 validates the result; equal-size/equal-mtime mutation fails. |
| B20 | Open | Add one no-training `assemble-complete-bundle` command around the existing v5 finalizer. It validates candidate/policy/source/pairwise bindings and copies exact evaluated bytes regardless of metric outcome; assembly grants no approval. The quality report verifies candidate-to-complete-bundle continuity. | Tests prove no fit/materialization call occurs, candidate artifact/target and pairwise members are byte-identical, and the v5 complete bundle reloads. |
| B21 | Closed | Retain the synthetic pairwise writer/reloader and missing/default-branch Python/Rust parity case as an ordinary regression test. | The focused test is non-vacuous for both written-and-reloaded boosters. Release evidence comes from real-model parity and B16. |
| B22 | Open | Maintain a deterministic, manifest-backed bounded pairwise smoke root containing one clustered dataset and bounded fixed-pair train/validation/test files. | Manifest proves actual row counts and the smoke remains cheap. |
| B23 | Open | Emit one pairwise training report with validation metrics, trials, selected parameters/objectives, train/validation identities, selected name-count-index path/manifest, and sealed pair/cluster test digests, with no candidate test results. | Full report is finite and complete; the run-specific `S2AND_PATH_CONFIG` resolves the reviewed v2/v3 name-count generation; trainer has no test paths or test metrics. |
| B24 | Closed | Retain generic missing/altered declared-member failures for both wheel and sdist. | Parameterized archive tests pass. The release gate separately verifies every exact workflow-built member. |
| B25 | Open | After EPS is frozen, always regenerate/finalize candidate members, then build the B19 inventory. Remove the independence-proof branch. | The final source manifest is produced only after the calibrated pairwise manifest is fixed. |
| B26 | Open | Add one authoritative workflow that consumes one immutable B34 evidence archive by URL and SHA-256, reads the exact commit/version matrix from the frozen quality policy, validates quality/rollback/data/model evidence, builds once, runs real smokes, generates checksums, pauses for protected approval, and publishes those exact bytes. | Tampered evidence, failed quality/rollback/smoke, wrong commit, remote mismatch, or distribution digest drift makes publication unreachable. |
| B27 | Open | Bind each warehouse query to independently immutable snapshot/export evidence rather than a caller-supplied label. | Provenance records exact SQL, query ID, source identity/evidence, result digest, and produced artifact digest. |
| B28 | Open | Choose and implement the replacement count-source/query path. `pys2` is retired and is not a v1.3 release dependency. | The reviewed replacement runs a bounded extraction for both producers and records its tool/source identity; neither producer invokes `pys2`. |
| B29 | Merged into B16 | Validate the destination name-count generation through the configured production selector during the real installed smoke. | B16 asserts resolved path and manifest SHA-256; no additional release authority is created. |
| B30 | Closed | Keep training unable to resolve test rows. The sealed pair/cluster evaluators validate policy, complete model, and population before scoring and atomically write bound reports with finite/range/averaging checks. Retain validation-before-score ordering and no standalone unblind/start receipt. | Wrong policy, model, or population bindings fail before scoring. Tests also cover finite probabilities in `[0,1]`, one main/nameless average, strict `> 0.5` F1, and exact-input retry behavior without lifecycle records. |
| B31 | Partial | Add manifest-bound release reports for subblocking, complete-model Python/Rust parity, and performance/RSS. | Each command rejects a wrong expected input digest and atomically writes a report binding the exact model, data, fixture, or workload it measured. |
| B32 | Open | Add one deterministic quality-policy validator and quality-report producer that applies the frozen gates once. | Weakened policy, missing or non-finite evidence, and mismatched bindings fail; a passing fixture produces the same report on repeat. |
| B33 | Open | Add one validator for the operator-authored `rollback_report.json`. | Missing or mismatched previous-release identity, restore commands, smoke result, or candidate-manifest binding fails; a complete fixture passes. |
| B34 | Open | Add one evidence-archive packager for the fixed logical allowlist and publish the ZIP through B17's no-overwrite upload path. | Missing, duplicate, unexpected, path-traversing, or digest-mismatched members fail; a bounded archive uploads once and verifies by SHA-256. |

## Closure rule

A new producer, evaluator, validator, assembler, smoke, or workflow is closed
only when:

1. its supported CLI is visible in `--help`;
2. focused wrong-input and success-path tests pass;
3. the operator runbook contains the exact command actually implemented; and
4. any required real release output has been reviewed and its digest retained.

Do not close a blocker with a speculative command in the runbook.

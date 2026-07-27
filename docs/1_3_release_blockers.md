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
- The calibrated pairwise bundle requires a fresh Stage 5 linker fit whose
  metadata binds that exact pairwise feature contract and both booster digests.
  The release command serializes a complete v5 bundle, reloads it, and evaluates
  only that exact bundle without a second fit.
- Do not add `evaluation_start.json`, runtime `promotion.json`,
  `release_inputs.json`, or mandatory launch/process/completion triples.
- The only semantic release authorities are `release_spec.json`, the final data
  manifest, the complete-v5 manifest, one evaluation report, and
  `SHA256SUMS`. A transport index is not a sixth authority.

## Status meanings

- **Closed**: no remaining v1.3 work.
- **Partial**: useful behavior exists, but the named closure evidence is
  incomplete.
- **Open**: implementation or real release evidence is still missing.

## Active blockers

| ID | Status | Remaining resolution | Smallest closure evidence |
|---|---|---|---|
| B01 | Open | Choose one unused synchronized Python/Rust package version and record the final model/data/package matrix before the release commit is frozen. | `uv run --no-project python scripts/sync_version.py --check`, package-index nonexistence evidence, and reviewed version matrix. |
| B05 | Partial | The frozen historical environment now loads the v1.21 bundle from commit `e54c6ba` and observes the runtime override EPS `0.65`. Still run it on the release-spec populations, isolated from canonical-v2 path config and ambient downloads. | The final evaluation report binds the historical code/model/data identities and records observed post-load EPS `0.65`. |
| B06 | Open | Deterministically join canonical benchmark names into the exact fields consumed by `ANDData`, preserve raw names in one audit map, and inventory the result. | A fixture verifies values after load, Arrow conversion, and pairwise featurization; the report binds counts, divergences, examples, source snapshot, and output digest. |
| B07 | Open | Generate deterministic linker assignments at `base_group_id` granularity. | Fixed-input output is deterministic and no base identity crosses splits. |
| B09 | Open | Regenerate the published/downloaded linker replay Arrow bundle from frozen raw JSON, SPECTER2 inputs, and the selected name-count generation. | Deep validation passes and every manifest carries the required normalization and generation bindings. |
| B10 | Partial | Add one supported `validate-source-bundle` command for the B19 manifest, support files, tables, assignments, Arrow manifests, and name-count bindings. | Mutation of any declared source member fails; the final source bundle passes. |
| B13 | Partial | Fit one fresh linker on train/calibration using the final calibrated pairwise bundle, serialize it into a complete v5 bundle, reload that exact bundle, and only then evaluate it. | Validation precedes held-out feature materialization; the linker binding covers the pairwise feature contract, both booster digests, and embedded training target. |
| B15 | Partial | Keep v1.3 as an immutable external model bundle and make release distributions reject every packaged default/model directory. | The fixed distribution verifier requires canonical tuple/ORCID assets, forbids packaged model paths, and explicit-path loading remains documented. |
| B16 | Partial | Retain the clean-installed real-model smoke and run it in the authoritative workflow against the final wheel/sdist, data, model, and selected name-count generation. | Exact final artifacts pass; a wrong model or name-count digest fails. |
| B17 | Open | Add one no-overwrite data/model publisher with dry-run support, remote verification, and abandoned-prefix recovery. | Bounded rehearsal passes; wrong/preexisting bytes fail; an interrupted prefix is never reused. |
| B23 | Open | Retain pairwise validation/search provenance, train/validation and name-count identities, and sealed test digests in complete-model reproducibility members, without test paths or results. | The complete-v5 manifest covers the finite training provenance and the trainer cannot resolve test rows. |
| B25 | Open | After EPS is frozen, always regenerate/finalize linker source members and then build the B19 inventory. | The final data manifest is produced only after the calibrated pairwise manifest is fixed. |
| B26 | Open | Complete the release workflow around one pinned evidence manifest: stage its nine fixed members, rerun `evaluate-release`, build once, run clean-installed smokes and rollback rehearsal, write `SHA256SUMS`, pause for protected approval, and publish those exact bytes. | Tampered inputs, failed evaluation/rollback/smoke, wrong commit, or checksum drift makes publication unreachable. |
| B27 | Open | Bind each warehouse query to independently immutable snapshot/export evidence rather than a caller-supplied label. | Provenance records exact SQL, query ID, source identity/evidence, result digest, and artifact digest. |
| B28 | Open | Choose and implement the replacement count-source/query path; `pys2` is not a v1.3 dependency. | A bounded extraction for both producers records tool/source identity and neither invokes `pys2`. |
| B32 | Partial | Complete the component measurements consumed by the deterministic `evaluate-release` aggregator. It applies subblocking, complete-model Python/Rust parity, performance/RSS, and all gates from the release spec, data manifest, and complete-model manifest, then writes one report. Runtime remains a relative hard gate; RSS records raw repeated measurements and enforces the frozen absolute ceiling. | Wrong authority digests, weakened gates, or missing/non-finite measurements fail; fixed inputs produce one deterministic `s2and_release_evaluation_report_v1` `evaluation_report.json`. |

## Closed history

| ID | Status | Retained result |
|---|---|---|
| B02 | Closed | The tracked count-producer run support imports from a clean checkout. |
| B03 | Closed | Count production has one module CLI; help and bounded fixture runs pass. |
| B04 | Closed | Native cache keys bind the Rust/package tree, exclude build outputs, and have no broad virtualenv fallback; B26 owns clean release builds. |
| B08 | Closed | A real-shaped optional-evaluation fixture covers named and flattened extra datasets. |
| B11 | Closed | One digest-bound training preflight validates schemas, overlap, and isolation and withholds test paths. |
| B12 | Closed | One validation-only EPS command applies the release-spec rule and emits a fresh calibrated pairwise bundle. |
| B14 | Closed | Linker seed targets remain outside fresh output paths. |
| B19 | Closed | One source-bundle assembler builds the final data-root inventory after benchmark and replay trees are complete and binds all consumed members and nested Arrow manifests by role, size, and digest. |
| B21 | Closed | The non-vacuous pairwise writer/reloader and Python/Rust default-branch case remains ordinary regression coverage. |
| B24 | Closed | Wheel and sdist validation rejects missing or altered declared members. |
| B30 | Closed | Training cannot resolve sealed test rows, and evaluation validates all authority bindings before scoring. |

## Closure rule

A new producer, evaluator, validator, assembler, smoke, or workflow is closed
only when:

1. its supported CLI is visible in `--help`;
2. focused wrong-input and success-path tests pass;
3. the operator runbook contains the exact command actually implemented; and
4. any required real release output has been reviewed and its digest retained.

Do not close a blocker with a speculative command in the runbook.

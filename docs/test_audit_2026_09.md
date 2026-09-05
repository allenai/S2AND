# Test audit — September 2026

## Scope and approach

The starting Python suite collected 1,429 cases across 102 modules. The native
source inventory contained 120 tests across 17 of 24 Rust source files. Review
combined a suite-wide inventory/search for weak assertions, dependency skips,
cross-test imports, and global state with independent reviews of algorithmic
contracts, persistence/Arrow boundaries, release/CLI behavior, and infrastructure.
It was a risk-prioritized audit, not a claim that every possible input or branch
has been exhausted.

All work used bounded local fixtures. No production training, paid API calls,
large data scans, or package publication ran. Existing checksum, schema, label
independence, holdout, numerical parity, and import-boundary tests were retained:
they protect real contracts.

## Removed or replaced weak evidence

| Previous evidence | Decision and replacement |
| --- | --- |
| Two AST checks of archived augmentation scripts | Deleted. The scripts are explicitly unsupported and the tests never ran their incompatible behavior. |
| Exact source text of the release ref guard | Execute the real guard for main, feature, lookalike, tag, PR, and empty refs; inspect publication prerequisites semantically. |
| Hook substrings and an extracted shell loop | Execute the complete hook with bounded git/uv doubles; check skip, sync, failed sync, CRLF, and paths containing spaces. |
| AST import scan | Import all incremental runtime modules in a fresh process with forbidden direct/transitive imports blocked. |
| Tutorial mocks that returned the expected metrics or ignored the named threshold | Real scoring on each selected split, full CLI forwarding, missing-versus-corrupt Arrow behavior, and resource cleanup. |
| FastCluster configuration dictionary | Actual sklearn clone, refit, threshold/linkage behavior, exact partitions, and preserved input. |
| Handwritten simulation of singleton ID allocation | Deleted. Existing tests already call real completion and exercise collision avoidance. |
| Rust source-fingerprint simulation | Deleted; exercise the actual native file hasher from Python on empty/known data and around its 1 MiB read boundary. |
| Fixed completion order after a thread barrier | Corrected to compare the complete set of builds; retained rendezvous, termination, and error assertions. Both legal schedules now satisfy the contract. |

The audit found no remaining unconditional-pass tests, blanket
`pytest.raises(Exception)` assertions, AST/source-inspection-only tests, or
imports from another collected test module. This is structural evidence, not a
claim that every remaining test is maximally strong.

## Added critical behavioral coverage

- **Publication and resource ownership:** failed manifest writes/replacements
  preserve the old bytes, remove temporary files, and permit retry; four partial
  Arrow-open failures close acquired descriptors; rejected profile/reader
  operations release leases; exceptional publisher exits release process locks.
- **Artifact identity:** replacing a model changes predictions, the cached
  booster, and its digest. Duplicate paper keys select every relevant Arrow batch.
  Empty and multi-buffer native hashing checks both fingerprint and SHA-256.
- **Inference and clustering:** real, independently trained tiny LightGBM models
  exercise Python/native main and nameless combinations, bounded chunks, class-0
  distances, constrained rows, and original ordering. Completion covers transitive
  bridges; rank decoding checks integer boundaries in very large pair populations
  without allocating those populations.
- **Evaluation:** hand-calculated ensemble metrics verify averaging, a strict
  `> 0.5` decision, AUROC/AP and macro metrics, plus empty, single-class, and
  misaligned populations.
- **CI and CLI failures:** prerequisite failures stop later gates; native-build
  retries are bounded and preserve the final underlying error; shell execution
  uses native Git Bash on Windows instead of the WSL launcher.

**Confirmed production bug, fixed:** `write_arrow_artifact_manifest` recorded the
temporary path only after writing. A disk-full/write exception left a partial
temporary file. Recording the path before the write lets the existing cleanup
run. The change alters no schema or public API. A real temporary-file fault probe
fails with the original function and passes with the fix.

## Infrastructure changes

- Shared raw Arrow, classic training, logistic gate, and serialized model fixtures
  no longer require importing collected test modules. Responsibilities are listed
  in [tests/README.md](../tests/README.md).
- One autouse fixture owns the Python backend default and restores global Python
  and NumPy RNG state. Duplicate backend setup/teardown was removed. Native tests
  continue to route explicitly; absence-of-environment tests remove the variable.
- Required Rust/PyArrow dependencies now fail collection or execution when broken;
  the optional-runtime helper and its environment switch were removed.
- Strict pytest configuration and marker validation expose configuration mistakes;
  skips are reported by default. Legitimate platform and Bash availability skips
  remain explicit.
- Coverage now records branches and uses one 80% combined floor in
  `pyproject.toml`, replacing the runner's 40% statement-only override. The measured
  combined result is about 85%, leaving platform/version headroom.
- PR CI adds Windows/Python 3.11 alongside Ubuntu/Python 3.11–3.13. Previously, the
  Windows CRT lock and open-file cleanup regression tests never ran in PR CI.

## First-pass verification record

First-pass result: **1,492 Python tests passed**, no skipped cases, and six subtests
passed in **154.20 seconds** on Windows/Python 3.11. This is 63 net additional
collected cases after removals, replacements, and parameterization. Coverage was
**88.71% statements**, **76.34% branches**, and **85.35% combined**, satisfying the
shared 80% floor. Ruff passed for `s2and`, `scripts`, and `tests`; formatting passed
for all 49 changed/new Python files; configured ty checks for `s2and` passed.
The six warnings came from Windows physical-core discovery and SHAP's existing
LightGBM return-shape notice. No warnings were hidden to obtain a passing run.

The full-suite and final checks are recorded under `scratch/test-audit-*`.
Commands used a workspace uv cache and isolated pytest temporary directories
because pre-existing user cache/temp directories were inaccessible in this
session. That environment problem was not counted as a product failure.

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) 'scratch/uv-cache'
$env:COVERAGE_FILE = Join-Path (Get-Location) 'scratch/test-audit-final.coverage'
uv run --no-sync pytest -q --basetemp=scratch/test-audit-final2-tmp -o cache_dir=scratch/test-audit-pytest-cache --durations=25 --cov=s2and --cov-report=term:skip-covered --cov-report=json:scratch/test-audit-final-coverage.json --junitxml=scratch/test-audit-final.xml
uv run --no-sync ruff check s2and scripts tests
uv run --no-sync ty check s2and --ignore unresolved-import --ignore unused-type-ignore-comment --ignore possibly-missing-attribute --ignore unresolved-global --python-version 3.11 --python-platform linux
```

The runtime-isolation probe ran 77 real tests with an inherited Rust backend and
checked environment and global RNG states outside each complete test protocol.
All passed. Separate mutation probes rejected eight plausible faults: ignored
linkage, lost transitive bridge, shifted pair rank, probability used as distance,
leaked manifest temporary file, disabled release ref guard, validation bypass,
and damaged CRLF staging. The scheduler probe demonstrated that the old cache
test rejected a legal execution order.

Reproducible local probes (ignored scratch artifacts):

```powershell
uv run --no-sync python -m scratch.test_audit_runtime_isolation
uv run --no-sync python -m scratch.test_audit_algorithm_mutations
uv run --no-sync python -m scratch.test_audit_manifest_mutation
uv run --no-sync python -m scratch.test_value_mutations
uv run --no-sync python -m scratch.test_value_scheduler_probe
```

## First-pass verification limits

Verification ran locally on Windows/Python 3.11 with the installed matching native
extension. The newly configured hosted matrix has not run in this session.
Native production code was unchanged; the native suite was inspected and one
test deleted, but the remaining 119 native unit tests were not recompiled or
rerun. Targeted rustfmt and the real Python/native hashing integration passed.

Statement/branch coverage still leaves paths unexercised, especially query
feature adaptation, training preflight, platform memory detection, and private
Rayon-cache eviction/concurrency. These are coverage risks rather than confirmed
defects. Production-scale accuracy, memory limits, and artifact-release gates
still require their separate bounded/approved workloads. The fault probes are
targeted examples, not systematic mutation coverage of the repository.

## Second pass: deeper test reduction

The second pass compared overlapping assertions and fixtures against a saved
snapshot of the first pass. It removed copied algorithms, mock-only wrapper
checks, repeated lifecycle scenarios, and Cartesian combinations with no distinct
interaction. Counts include helpers and parameterized cases; they do not treat
moving setup into a helper as deletion.

| Measure | Before second pass | Current checkout | Change |
| --- | ---: | ---: | ---: |
| Python test/support lines | 39,392 | 35,235 | −4,157 (10.6%) |
| Test functions | 1,092 | 983 | −109 |
| Collected Python cases | 1,492 | 1,369 | −123 |
| Python test/support files | 109 | 106 | −3 |
| Native Rust test declarations | 119 | 119 | 0 |

Git's text diff against the saved snapshot reports **6,597 deleted lines and
2,440 added lines** in Python tests/support. Gross figures include code moves.
These are checkout-to-snapshot totals: separate work added subblocking/report
tests and changed production code during this pass. Those edits were preserved;
they are included in the current suite counts, not claimed as this audit's work.

| Redundant evidence or setup | Retained protection |
| --- | --- |
| A test-side native aggregation implementation and fabricated memory plans | Real native execution, actual planner limits, and independent dense reductions for sparse rows, chunk accumulation, empty rows, and NaN policies |
| Legacy clustering wrapper and repeated constant-model training | Native/Python partition and distance contracts, plus concise FastCluster/DBSCAN precomputed-input cases |
| Mock native scorer and separate copy/pickle/load scenarios | Real trained-model predictions across a complete artifact lifecycle, with independent LightGBM expected values |
| Repeated Arrow retention, conversion, and ORCID fixtures | Exact converted rows, retained-reader lifecycle, and cross-backend partitions/telemetry |
| Repeated data/query setup and broad feature-selection enumeration | Small domain builders, lazy dataset setup, explicit feature-group boundaries, real query/retriever construction, and frozen expected columns |
| Four scattered name-count modules | One component module covering bounded writing, atomic publication, manifest rejection, identity, and cache lifetime |

Review and coverage comparison recovered useful assertions that the initial
reduction had missed: constructor-default feature groups, real booster cache
fingerprints, fully connected completion, implicit native loading, and competing
subblock packing strategies. These were folded into concise scenarios. All
previously executed lines/branches in unchanged modules were recovered except
the unconditional direct-`ArrowDataset()` rejection and early destructor return
on that partially constructed object. Those two lines and one branch were
deliberately dropped with the construction-only check; real open/close/failure
ownership tests remain. Concurrent source-line moves were compared separately.

Final verification: **1,369 passed**, **six subtests passed**, **no skips**, in
**142.30 seconds** on Windows/Python 3.11. Coverage is **88.92% statements**,
**76.79% branches**, and **85.63% combined**, versus 88.71%, 76.34%, and 85.35%
before this pass. This timing is one run, not a performance claim. Ruff lint,
format checks for all 106 test/support Python files, and configured ty checks
passed. Eighteen targeted fault injections were detected, including disabled
ORCID repair, wrong native predictions, holdout leakage, incorrect row mapping,
ignored packing strategy, missing default features, and shifted pair offsets.

The first full run encountered a concurrent report-path/test edit; its updated
module passed before the final rerun. Source hashes around the final run found
only a subblocking test-formatting change; that module subsequently passed all
18 cases again. Native integration used the installed extension. This pass did
not rebuild the separately edited Rust source or rerun its 119 native unit tests.

Reproduction and detailed removed-to-retained mappings are saved under the
ignored `scratch/test-parsimony/` directory: `before/`, `after/`, `reduction.json`,
`git-numstat.txt`, `final.log`, `final-results.xml`, `final-coverage.json`, coverage
comparisons, and the algorithm/Arrow/value reports and fault probes.

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) 'scratch/uv-cache'
$env:COVERAGE_FILE = Join-Path (Get-Location) 'scratch/test-parsimony/verified.coverage'
uv run --no-sync pytest -q --basetemp=scratch/test-parsimony/full-tmp2 -o cache_dir=scratch/test-audit-pytest-cache --durations=15 --cov=s2and --cov-report=term:skip-covered --cov-report=json:scratch/test-parsimony/final-coverage.json --junitxml=scratch/test-parsimony/final-results.xml
uv run --no-sync python scratch/test-parsimony/measure_reduction.py
```

# S2AND 1.0.0 / model-data v1.3 release

Status date: 2026-07-27

This is the sole release policy and dependency-ordered operator sequence for
the `s2and==1.0.0` and `s2and-rust==1.0.0` packages with production model and
public-data version `1.3`.
[scripts/production/README.md](../scripts/production/README.md) is the command
reference.

| Version axis | Fixed release version |
|---|---|
| Python package (`s2and`) | `1.0.0` |
| Rust package (`s2and-rust`) | `1.0.0` |
| Production model | `1.3` |
| Public data | `1.3` |

## Current external inputs

Four reviewed external inputs are not in this repository:

| Input | Required shape | Role |
|---|---|---|
| Canonical benchmark names | One JSON file per benchmark, with exactly one row per dataset `signature_id` and `first`, `middle`, `last` | Required before Stage 2 can regenerate benchmarks |
| ORCID names | Query-ordered CSV with `raw_orcid`, `orcid`, `first_name`, and `middle` | Required before Stage 1 can build package assets |
| Name counts | Query-ordered CSV with `first_name`, `last_name`, and `count` | Ordinary Stage 1 release input; its producer is already complete |
| v1.21 baseline record | Reviewed cluster B3, pairwise aggregate/per-dataset AUROC and macro-F1, and performance p50 for the frozen populations/workload, plus source/model/data/environment/command identities | Ordinary Stage 0 input copied as scalars into `release.json`; the repository has no cross-version baseline runner |

Do not replace the canonical or ORCID input with a fixture or relabel a legacy
artifact. The canonical join rejects duplicate, missing, and extra IDs. The
ORCID producer requires reviewed row and expansion bounds.

The first two are the remaining implementation prerequisites carried into this
runbook. The name counts, historical baseline, and other benchmark/linker data
are ordinary release inputs.

Two former implementation gaps are already closed:

- the reviewed
  `combined_query_split_assignments_base_group_seed13.csv` has 23,572 rows,
  SHA-256
  `b67b7ba7a5258b99d71f624ae12a2b5c6f938ba207215d1b7ebf63b791eadc64`
  and zero `base_group_id` overlap across splits. It changed only the 394
  leaking rows from source SHA-256
  `9ecc005c0734cd64627bef84fd320a5d0f4e281ecebf483e1f6728dce42d597b`;
  and
- pairwise training records finite selected validation AUROCs for both
  boosters and fails if either is non-finite.

## Fixed decisions

- Model and public data version: `1.3`.
- Normalization: `canonical_v2`; featurizer contract: `10`.
- The release model is one complete external v5 bundle, never a packaged default.
- Runtime data consists of canonical tuple text, ORCID JSON with its tuple
  dependency, and the direct-file name-count v3 index, without compatibility
  readers.
- Python and `s2and-rust` remain separate distributions and both support
  Python 3.11, 3.12, and 3.13.
- EPS is selected from validation data only.
- Pairwise training precedes EPS selection. Linker inputs are materialized from
  the final pairwise boosters and feature contract, and the linker is fit once.
- An EPS-only change does not invalidate linker features or the linker fit.
- The complete bundle is serialized, reloaded, and evaluated without a second fit.
- The final package commit is the exact commit dispatched for release.
- Local CI is exactly
  `uv run --no-project python scripts/run_ci_locally.py`.
- Publication is the no-input `.github/workflows/release-rust.yml` workflow on
  `main`; its protected `pypi` environment publishes Rust before Python.

## Operating rules

Use a fresh local, unsynchronized run root outside Google Drive. Never reuse
stage outputs. Initially create only the root and `release.json`;
`prepare-run` creates the other entries:

```text
D:\s2and-v1.3-YYYYMMDD-attempt-N\
  release.json
  model_plan.json
  evaluation_plan.json
  stages\
  reports\
  final\
```

For every warehouse query, conversion, training run, and performance run:

1. run a bounded fixture first;
2. review the full command, cost, RAM, threads, and expected runtime;
3. run long work detached or through a scheduler with durable logs; and
4. retain native outputs, logs, and reviewed count or metric summaries.

Freeze one source commit before full production work. If behavior changes,
commit again and rerun every affected downstream stage.

Before using full inputs, rehearse the release boundaries from a clean checkout:

```powershell
uv run pytest -q `
  tests/test_convert_s2and_mini_to_arrow.py `
  tests/test_generate_name_counts_script.py `
  tests/test_generate_orcid_name_prefix_counts.py `
  tests/test_real_tiny_trainers.py `
  tests/test_release_pairwise.py `
  tests/test_smoke_installed_incremental_arrow.py
```

This bounded rehearsal must pass without warehouse access. It exercises the
canonical join, both count producers, real tiny trainers, three-plan
orchestration and gates, and the installed-artifact smoke contract.

## Stage 0: freeze external choices and source

- [ ] Supply and review all four external inputs listed above.
- [ ] Confirm the synchronized Python/Rust package version remains `1.0.0` and
      the package/model/data matrix above is unchanged. Verify it through the
      [version-bump workflow](development.md#version-bumping), then confirm
      both package versions are unused:

```powershell
$Version = (Get-Content -Raw VERSION).Trim()
foreach ($Project in @('s2and', 's2and-rust')) {
  $Status = curl.exe `
    --silent `
    --output NUL `
    --write-out '%{http_code}' `
    "https://pypi.org/pypi/$Project/$Version/json"
  if ($LASTEXITCODE -ne 0 -or $Status -ne '404') {
    throw "$Project==$Version is not unused on PyPI"
  }
}
```

- [ ] Freeze the evaluation populations, metrics, gates, and logical
      performance workload. Verify that the reviewed v1.21 baseline record was
      measured at compatible commit
      `e54c6ba9c0e3ca4c2b5a40dcaa9a55c2c771d87d`, observed runtime EPS `0.65`,
      and matches those exact populations/workload. Retain its raw results,
      commands, environment, threads, and machine identity. This historical
      analysis is the sole pre-Stage-5 held-out prediction; its results cannot
      change the frozen choices or v1.3 behavior.
- [ ] Commit the implementation, workflow, metadata, and version changes needed
      for full production work.
- [ ] Run `uv lock --check`, then the executable local CI authority:

```powershell
uv run --no-project python scripts/run_ci_locally.py
```

- [ ] Record this production source commit. Any later behavior change
      invalidates every affected downstream result.

## Stage 1: generate runtime data

- [ ] Use a fresh artifact workspace outside the still-uncreated run
      directory.
- [ ] Generate canonical name tuples; review curated accept/reject accounting
      and load the output through the production loader.
- [ ] Export warehouse rows through any working authenticated client. Review
      the SQL, required columns, row bounds, and spot checks documented in the
      [count command reference](../scripts/production/README.md#count-artifacts)
      before feeding rows to the pure count transformations.
- [ ] Run bounded count fixtures, then convert the reviewed full exports into
      fresh paths.
- [ ] Review counts, missing values, normalization, and loader round trips.
- [ ] Copy the final tuple and both ORCID runtime files to their declared
      package-data paths. Keep the large name-count v3 index external.
- [ ] Build the local wheel and sdist and run
      `verify_production_model_distributions.py`. It must find the tuple and
      ORCID assets and reject a packaged default model. Write this preliminary
      build outside the run directory.

## Stage 2: build training and evaluation data

- [ ] Join the reviewed canonical benchmark-name export into exactly the
      fields consumed by `ANDData` using the
      [Arrow command reference](../scripts/production/README.md#canonical-benchmark-names);
      retain raw names only for reviewed spot checks. Stage these source files
      outside the still-empty run directory.
- [ ] Review and freeze `release.json` using the
      [exact shape and example](../scripts/production/README.md#pairwise-model):
      training/validation and held-out inputs, validation-only EPS grid and
      floors, reviewed historical baselines, gates, and the exact performance
      Arrow root and workload.
- [ ] Put the reviewed `release.json` alone in a fresh run root and run
      `release_pairwise.py prepare-run`.
- [ ] Inspect the resulting `model_plan.json` and `evaluation_plan.json`.
      Confirm the model plan contains training and validation identities only,
      with no held-out path or reference.
- [ ] Convert one benchmark and one linker-replay fixture to Arrow outside the
      run root, then convert the full reviewed datasets into the prepared
      run's fresh `stages/` roots.
- [ ] Regenerate linker-replay Arrow from reviewed raw JSON, SPECTER2 inputs,
      and the final name counts; validate it with production loaders.
- [ ] Stage only the reviewed linker support files and the frozen assignment
      CSV/summary, verify their declared hashes, and run
      `linker_source_bundle.py` once. It assembles and preflights the fresh
      linker-source and public-data roots; do not carry a legacy `datasets/`
      directory into the support root.

## Stage 3: train the pairwise model

- [ ] Run bounded training fixtures, then reviewed full training with durable
      logs and the frozen plan.
- [ ] Review the complete training report: search, selection, resource use,
      and finite selected validation metrics.
- [ ] Confirm the pairwise-only v5 bundle reloads and both boosters round-trip
      through Rust.
- [ ] Confirm training had no held-out prediction, metric, path, or lookup capability.

## Stage 4: select EPS and freeze linker inputs

- [ ] Calibrate EPS against validation inputs and the frozen model plan.
      Confirm only clusterer configuration and bundle metadata changed.
- [ ] Freeze the already assembled linker-source root, public-data root, final
      pairwise bundle, name-count index, reviewed 53-feature linker target,
      compute resources, and held-out populations before fitting.

A pairwise-booster, feature-contract, linker-source, or linker-target change
requires fresh linker features and a fresh fit. An EPS-only change does not.

## Stage 5: fit, evaluate, and approve

For the v1.3 candidate, held-out access before this stage is limited to
identity and overlap preflight. The frozen v1.21 baseline described in Stage 0
is the only historical exception. Do not materialize v1.3 held-out features,
predictions, or metrics until the model, populations, metrics, baselines,
thresholds, and performance workload are frozen.

- [ ] Run `train_linker_and_finalize.py` once, using the final calibrated
      pairwise bundle and the frozen inputs. The command materializes linker
      features, fits once, serializes and reloads the complete v5 bundle, then
      evaluates the linker source bundle's frozen evaluation split. Retain
      `linker_evaluation_report.json`; this split is separate from the
      pairwise/cluster populations in `evaluation_plan.json`.
- [ ] Confirm the complete bundle metadata binds the feature contract, both
      boosters, and the exact linker target.
- [ ] Produce pairwise, cluster, parity, and performance reports from the
      reloaded complete bundle. Produce subblocking quality from the frozen
      Arrow and candidate-component inputs; it does not consume the model.
- [ ] Confirm every model-dependent held-out metric came from that reloaded
      bundle without a second fit.
- [ ] Put those five numeric reports in `reports/`, then run
      `release_pairwise.py evaluate-release` with the frozen
      `evaluation_plan.json`.
- [ ] Require every numeric gate to pass and the performance workload to match
      the frozen performance Arrow root and workload exactly.
- [ ] Retain component reports for diagnosis and the aggregate decision.

- [ ] After final package assets and release metadata are present, commit the
      release tree, rerun `run_ci_locally.py` and version checks, and record
      this as the final package commit.
- [ ] From that exact commit, build the distributions, install them into an
      empty environment, and run pairwise and incremental prediction against
      the real external complete model, public data root, and name-count index.
      Use `$DistRoot = "$RunRoot\stages\local-dist"` and follow the
      [distribution build](../scripts/production/README.md#distribution-and-arrow-checks)
      and
      [installed release smoke command](../scripts/production/README.md#installed-release-smoke).
      The workflow's synthetic installed smoke does not replace this.
      A behavior-changing difference from the production source commit
      invalidates every affected production result.

## Stage 6: publish packages

Push the exact final commit to `main`, then confirm the remote ref before
dispatching the no-input workflow:

```powershell
$ReleaseCommit = (git rev-parse HEAD).Trim()
git fetch origin main
if ((git rev-parse origin/main).Trim() -ne $ReleaseCommit) {
  throw "origin/main is not the reviewed release commit"
}
gh workflow run release-rust.yml --ref main
gh run list `
  --workflow release-rust.yml `
  --event workflow_dispatch `
  --branch main `
  --commit $ReleaseCommit `
  --limit 1 `
  --json databaseId,headSha,status,conclusion,url
gh run watch REVIEWED_RUN_ID --exit-status
```

Before approving the protected `pypi` environment, verify the selected run's
`headSha` is `$ReleaseCommit`, record its ID/URL, and rerun the two PyPI 404
checks from Stage 0. A published Rust version followed by an already-existing
Python version would be an irreversible partial release.

The workflow must:

1. reject any non-`main` ref, then run tests and version checks;
2. build and verify Python distributions and Windows, macOS, manylinux, and
   musllinux Rust artifacts;
3. clean-install both distributions and run the synthetic installed bulk and
   incremental Arrow smoke;
4. publish the already-built Rust distributions through the PyPI environment;
5. wait with bounded retries for that exact Rust version to install; and
6. publish the already-built Python distributions.

## Stage 7: finish

- [ ] Publish the reviewed model and public data to immutable owner-selected
      locations, download them again, verify their manifest identities, and
      record the immutable URLs and hashes.
- [ ] Install the exact public Python and Rust versions together in a fresh
      environment and rerun the
      [real bulk and incremental smoke](../scripts/production/README.md#installed-release-smoke)
      against those downloaded model/data bytes, with
      `uv pip install ... "s2and==$Version" "s2and-rust==$Version"`.
- [ ] Tag the exact final commit and verify the tag before pushing it:

```powershell
$Version = (Get-Content -Raw VERSION).Trim()
$Tag = "v$Version"
git tag -a $Tag $ReleaseCommit -m "S2AND $Version"
if ((git rev-list -n 1 $Tag) -ne $ReleaseCommit) {
  throw "release tag does not name the final commit"
}
git push origin $Tag
gh release create $Tag `
  --verify-tag `
  --title "S2AND $Version" `
  --notes-file REVIEWED_RELEASE_NOTES.md
```

The reviewed notes contain the package/model/data version matrix, immutable
model and data URLs plus manifest identities, evaluation decision, and known
format breaks.

## Invalidation rules

- A normalization, tuple, count, ORCID, benchmark, assignment, or Arrow change
  invalidates every downstream result that consumed it.
- A pairwise booster, feature-contract, linker-source, or linker-target change
  invalidates linker features and the linker fit. An EPS-only change does not.
- A linker source or target change invalidates the complete model and its
  evaluation.
- A population, metric, baseline, threshold, performance-root, or
  performance-workload change
  invalidates its report. If changed after seeing a held-out result, evaluation
  requires a genuinely untouched holdout; rerunning the same holdout is not
  valid confirmation.
- A package-content or version change requires rebuilding distributions and
  rerunning both synthetic and real installed-artifact checks.
- Any behavior-changing source commit requires rerunning every affected stage
  and publishing only from the newly verified commit.

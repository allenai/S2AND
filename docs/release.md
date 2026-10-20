# S2AND release process

This guide defines the production release process.
The Python and Rust package version is `1.0.0`.
The next model and research snapshot version is `1.3`.
Public Arrow and name-count data use format `1`.

The [data guide](data.md#download) lists the release `1.3` populations.
Fresh author-table counts, A, define the production count authority.
The selected rebuild uses current normalization and the current packaged tuple and ORCID assets.
See the [A mainline record](release_evidence/author_counts_mainline_20261017.md) for its status.
A production release still requires final acceptance and publication.
The S3 prefixes stay empty until acceptance and publication.

## Release inputs

Use one final corrected snapshot as the input to each release.
It contains the output of all accepted additions and corrections.
It preserves full paper text, current labels, uncertainty masks, embeddings, producer sidecars, and exact split identities.
It also contains augmented records and complete incremental task definitions.
Do not reconstruct accepted corrections from historical review archives.

The current benchmark masters retain full text where available.
The seven giant replay sources lack full abstract text.
Preserve their available current records and vectors; a new abstract-based encoding needs another source of full abstracts.

The [data guide](data.md) defines the audience, artifact layout, and update rules.
The [production command reference](../scripts/production/README.md) contains executable commands.
Keep source snapshots and fit outputs in separate directories.

## Fixed decisions

- Keep one common binary name-count index and complete production model in versioned Git LFS assets.
- Build the common index from author-table rows. Archive mention counts, M, as a research alternative.
- Keep small name tuples and ORCID prefix counts in normal Git and the Python wheel.
- Publish three S3 bundles, `inference`, `dataset`, and `source`, with one copy of each file.
- Train every release model from the published `inference` and `dataset` bundles.
- Use supported formats and schemas for dataset compatibility.
- Bind models to feature-contract versions, ordered feature names, and exact required artifact hashes.
- Treat the model's generating runtime version as provenance.
- Require matched installed Python and Rust package versions for their internal interfaces.
- Reject old unversioned model stages; rebuild or explicitly export them under the current contract.
- Select pairwise hyperparameters and EPS from development data.
- Review giant-block EPS curves before the final operating-point decision.
- Keep test metrics separate from model selection.
- Recompute incremental features after a pairwise booster or feature-contract change.
- Retain incremental features after an EPS-only change when their other inputs stay fixed.
- Publish only a complete serialized model after evaluation through that same model.

## Operating rules

Start expensive stages with a tiny real sample.
Use an explicit workload bound and the approved CPU and memory limits.
Keep logs, a process ID, and a reproducible command for long jobs.
Preserve source files and failed output evidence.
Use a fresh output path for each changed dataset or model stage.
Follow the active repository instructions for long jobs.

Count changes can require a warehouse export.
Dataset refreshes can require local SPECTER2 encoder files.
These inputs are required only when their current artifacts change.
A historical baseline is not a prerequisite for snapshot preparation or a pairwise fit.

## Stage 0: choose the input snapshot

1. Select the final corrected master records and accepted incremental dataset.
2. Select the exact common name-count index and packaged tuple/ORCID assets.
3. Declare fit datasets, split identities, pair lists, and the EPS policy.
4. Verify the declared workload and resource bounds.
5. Keep the selected snapshot immutable for the run.

The selected local input path is `D:/local-unsynced/s2and_ca_mainline_20261017/snapshot`.
The snapshot passes relocation checks.
It excludes `inventors_s2and` from corrected records, Arrow data, pairwise plans, and incremental inputs.

## Stage 1: generate runtime data

Reuse unchanged common assets.
Regenerate a common asset only when its source data or semantics change.
Regenerate A after a canonical name-rule change, even when the raw Athena export stays unchanged.
Only AI2 staff can run the Athena export and rebuild A. Expect this for every future refresh.
Recompute pairwise and linker features after a count-authority change.
The ORCID producer preserves the surname rule and excludes the 14 reviewed groups.
Run the focused producer checks before a bounded rebuild.
See [the count commands](../scripts/production/README.md#count-artifacts).

Do not put the large index into the Python wheel.
Verify that the wheel contains the canonical tuple and both ORCID runtime files.
See [distribution checks](../scripts/production/README.md#distribution-and-arrow-checks).

## Stage 2: build training and evaluation data

Apply new corrections directly to a copy of the previous corrected snapshot.
Preserve useful record fields and current labels.
Regenerate augmentation from the accepted records and declared source identities.
Reuse SPECTER2 vectors only when their input text remains unchanged.
Encode changed informative text and informative text without a source vector.
Retain the declared encoder model, adapter, tokenizer, pooling, and truncation recipe.

Package the corrected records, pairwise inputs, benchmark projections, and complete incremental inputs with `package_corrected_datasets.py`.
The producer writes an internal snapshot with portable paths to one shared count index.
Validate the assembled snapshot after relocation.
Check record counts, gold coverage, masks, pair endpoints, split assignments, and native Arrow opens.
See [the preparation commands](../scripts/production/data/README.md).

Freeze identity-gold cases against the final snapshot with `evaluate_incremental_gold.py prepare`.
The cases bind the published benchmark files, so evaluation needs only the `dataset` bundle.
Freeze them again after any change to the benchmark Arrow data or the count index.

Write the published bundles with `write_release_bundles.py`.
The command proves that the benchmark Arrow data rebuilds the snapshot's prepared populations.
It then writes the dataset model plan.
See [release bundles](../scripts/production/README.md#release-bundles).

The model plan declares the fit population rather than a fixed seven-source list.
The current release declares seven non-patent benchmarks and non-patent augmentation.
Unknown signatures remain in inference context but do not supply supervised labels.
Training and external evaluation must satisfy the declared identity overlap rules.

## Stage 3: train the pairwise model

Sync the `inference` and `dataset` bundles into one root.
Run `train_pairwise.py` with that root's model plan and common index.
Use a separate run output and matrix workspace.
The trainer searches main and nameless models from the declared datasets.
Pass a known configuration with `--incumbent-parameters` to let it compete with each search winner.
It writes an immutable pairwise-only stage with EPS marked `pending`.
It does not require an evaluation baseline or a historical release record.

Review finite validation metrics, source counts, pair counts, and resource use.
Verify both boosters after serialization.
Check the actual trained boosters' native scoring plans:

```powershell
$env:S2AND_PAIRWISE_MODEL_DIR = '<pairwise-stage>/pairwise'
uv run cargo test --manifest-path s2and_rust/Cargo.toml --lib --no-default-features `
  release_bundle_uses_native_fast_plans -- --ignored --nocapture
Remove-Item Env:S2AND_PAIRWISE_MODEL_DIR
```

A failed native fast-plan check requires representative latency measurements and a candidate review.
Do not silently change the scientific search space to satisfy this check.

## Stage 4: select EPS and freeze linker inputs

Calibrate EPS from validation data with `release_pairwise.py calibrate-eps`.
Calibration writes a fresh calibrated sibling and preserves both boosters.
Verify that only clusterer configuration and bundle metadata change.
This benchmark selection supplies the initial EPS for giant-block review.

Freeze the calibrated pairwise model, corrected incremental source, common assets, linker target, and fit resources.
Recheck safe-action labels if query evidence or profile memberships change.
A pairwise model change requires new incremental features; it does not automatically invalidate manual judgments.

## Stage 5: fit, evaluate, and approve

Freeze evaluation populations, metrics, thresholds, and representative performance workloads before candidate scores are available.
Distinguish independent release tests from inherited regression tests.
Compare old and new models on the same population when a matched comparison is required.
Do not compare historical rates from different data or split populations as a matched baseline.

Run `train_linker_and_finalize.py` once against the calibrated model and the synced data root.
The command computes fresh features, fits the linker, writes a complete sibling, reloads it, and evaluates its declared test split.
The classic booster fits train rows.
Both calibration splits select gate features and fit gate coefficients; they serve as dev populations.
Retain existing manual dev labels after verified query-paper exposure exclusions.

Complete [giant-block EPS review](eps_calibration.md) before final release evaluation.
The release owner selects the final EPS there; that guide also records the `1.3` decision.
Apply the reviewed EPS to a fresh complete model without another fit.
Freeze that package before final acceptance tests.

Prior development includes the inherited test population, so its stratified scores provide regression evidence.
Full-source diagnostics can also include gate-calibration queries.
Use [evaluate_incremental_gold.py](../scripts/verification/evaluate_incremental_gold.py) for frozen fresh identity cases through the complete model.
Identity gold differs from manual safe-action labels.
Keep these reports separate.

The public loader requires the two pairwise prediction fixtures and complete incremental artifact.
Evaluate the outer test, the identity-gold cases, parity, subblocking quality, and representative performance.
Model-dependent release metrics must use the serialized complete model without a second fit.
The [acceptance checks](../scripts/production/README.md#acceptance-checks) list the commands.

The release owner compares the reports with the declared thresholds before publication.
Retain the reports, including those of a failed check.
A metric, population, threshold, or workload change after test inspection requires a new evaluation decision.
A changed selection policy requires untouched test data for confirmation.

## Stage 6: publish packages

Run required project checks from the final source commit.
Build both distributions from that commit.
Verify wheel and source archive contents.
Install the candidate distributions in an empty environment outside the checkout.
Run bulk and incremental prediction with the real complete model and combined data root.
The workflow's synthetic smoke does not replace this check.
See [installed release smoke](../scripts/production/README.md#installed-release-smoke).

Publish through `.github/workflows/release-rust.yml` from the reviewed `main` commit.
Its validate job runs `scripts/run_ci_locally.py`, the same checks as pull-request CI.
Verify the workflow commit before approval of its protected PyPI environment.
Confirm both package versions are absent before the first irreversible publish.
The workflow publishes Rust first and then the matching Python package.
Do not dispatch publication before model and data acceptance are complete.

## Stage 7: finish

Publish the common large assets to Git LFS under a versioned release path.
Release `1.3` uses `artifacts/common/v1.3` for the index and accepted complete model.
Write the bundles again with the accepted model, then publish each bundle to its immutable S3 prefix.
Release `1.3` uses `s3://ai2-s2-research-public/s2and/1.3/` with `inference/`, `dataset/`, and `source/`.
Sync `inference` and `dataset` into one fresh local root.
Verify hashes and repeat installed bulk and incremental smoke checks.
Run one public step from the [reproduction guide](reproduce.md) on that root.

Record the package, model, research snapshot, and format versions in the release notes.
Record the common Git ref, asset paths, S3 prefix, manifest hashes, and evaluation decision.
Replace unpublished placeholders only with verified download locations.
Resolve every `Update at v1.3 release` note (`git grep -n "Update at v1.3 release"`).
Tag the exact reviewed source commit.

## Invalidation rules

An input change invalidates the downstream results that consume it.
A pairwise booster or feature-contract change invalidates incremental features and the linker fit.
An EPS-only change does not invalidate unchanged incremental features.
A source-label or profile-membership change requires checks of affected task labels and features.
A package change requires new distributions and installed artifact checks.
A runtime implementation change requires new affected evaluation evidence, even when the artifact contract version remains compatible.

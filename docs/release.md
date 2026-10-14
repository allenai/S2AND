# S2AND release process

This guide defines the production release process.
The Python and Rust package version is `1.0.0`.
The next model and research snapshot version is `1.3`.
Public Arrow and name-count data use format `1`.

Release `1.3` excludes patent data.
Pairwise inputs contain eight sources: seven non-patent benchmarks and the existing non-patent augmented dataset.
Incremental inputs contain the 13 inherited non-patent datasets.
The patent-free complete candidate passes fit, evaluation, and installed Arrow inference checks.
The [candidate record](release_evidence/production_candidate_20261013.md) gives model hashes, input populations, evaluation results, and remaining limits.
A production release still requires final acceptance and publication.
No final release model or S3 location is declared here before acceptance and publication.

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
- Keep small name tuples and ORCID prefix counts in normal Git and the Python wheel.
- Publish corrected research data to S3 without a second model or count index.
- Use supported formats and schemas for dataset compatibility.
- Bind models to feature-contract versions, ordered feature names, and exact required artifact hashes.
- Treat the model's generating runtime version as provenance.
- Require matched installed Python and Rust package versions for their internal interfaces.
- Reject old unversioned model stages; rebuild or explicitly export them under the current contract.
- Select pairwise hyperparameters and EPS from development data.
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

The selected local input path is `D:/local-unsynced/s2and_incremental_fresh_20261014/ready/complete_datasets_no_patents`.
The snapshot passes relocation checks.
It excludes `inventors_s2and` from corrected records, Arrow data, pairwise plans, and incremental inputs.

## Stage 1: generate runtime data

Reuse unchanged common assets.
Regenerate a common asset only when its source data or semantics change.
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
The producer writes `pairwise/model_plan.json` and portable paths to one shared count index.
Validate the assembled snapshot after relocation.
Check record counts, gold coverage, masks, pair endpoints, split assignments, and native Arrow opens.
See [the preparation commands](../scripts/production/data/README.md).

Dataset manifests declare the fit population rather than a fixed seven-source list.
The current release declares seven non-patent benchmarks and non-patent augmentation.
Unknown signatures remain in inference context but do not supply supervised labels.
Training and external evaluation must satisfy the declared identity overlap rules.

## Stage 3: train the pairwise model

Run `train_pairwise.py` with the snapshot's model plan, fit Arrow root, and common index.
Use a separate run output and matrix workspace.
The trainer searches main and nameless models from the declared datasets.
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

Freeze the calibrated pairwise model, corrected incremental source, common assets, linker target, and fit resources.
Recheck safe-action labels if query evidence or profile memberships change.
A pairwise model change requires new incremental features; it does not automatically invalidate manual judgments.

## Stage 5: fit, evaluate, and approve

Freeze evaluation populations, metrics, thresholds, and representative performance workloads before candidate scores are available.
Distinguish independent release tests from inherited regression tests.
Compare old and new models on the same population when a matched comparison is required.
Do not compare historical rates from different data or split populations as a matched baseline.

Run `train_linker_and_finalize.py` once against the calibrated model and snapshot.
The command computes fresh features, fits the linker, writes a complete sibling, reloads it, and evaluates its declared test split.
The classic booster fits train rows.
Both calibration splits select gate features and fit gate coefficients; they serve as dev populations.
Retain existing manual dev labels after verified query-paper exposure exclusions.

Prior development includes the inherited test population, so its stratified scores provide regression evidence.
Full-source diagnostics can also include gate-calibration queries.
Use [evaluate_incremental_gold.py](../scripts/verification/evaluate_incremental_gold.py) for frozen fresh identity cases through the complete model.
Identity gold differs from manual safe-action labels.
Keep these reports separate.

The public loader requires the two pairwise prediction fixtures and complete incremental artifact.
Evaluate pairwise, cluster, linker, parity, subblocking quality, and representative performance through the final contract.
Model-dependent release metrics must use the serialized complete model without a second fit.
Use the prepared evaluation plan and run binding when the existing gate tools require them.
Their baseline record belongs to promotion, not the input or fit stages.

Require the declared gates to pass before publication.
Retain reports for failed gates.
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
Verify the workflow commit before approval of its protected PyPI environment.
Confirm both package versions are absent before the first irreversible publish.
The workflow publishes Rust first and then the matching Python package.
Do not dispatch publication before model and data acceptance are complete.

## Stage 7: finish

Publish the common large assets to Git LFS under a versioned release path.
Release `1.3` uses `artifacts/common/v1.3` for the index and accepted complete model.
Publish the research snapshot to an immutable S3 prefix without common assets.
Download both parts into one local root.
Verify hashes and repeat installed bulk and incremental smoke checks.

Record the package, model, research snapshot, and format versions in the release notes.
Record the common Git ref, asset paths, S3 prefix, manifest hashes, and evaluation decision.
Replace unpublished placeholders only with verified download locations.
Tag the exact reviewed source commit.

## Invalidation rules

An input change invalidates the downstream results that consume it.
A pairwise booster or feature-contract change invalidates incremental features and the linker fit.
An EPS-only change does not invalidate unchanged incremental features.
A source-label or profile-membership change requires checks of affected task labels and features.
A package change requires new distributions and installed artifact checks.
A runtime implementation change requires new affected evaluation evidence, even when the artifact contract version remains compatible.

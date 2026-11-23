# S2AND release process

Read [`VERSION`](../VERSION) for the package version, currently `1.0.0b14`.
The target final package version is `1.0.0`.
The current model version is `1.4`, on research snapshot `1.4`.
Public Arrow and name-count data use format `1`.

The `1.3`, `1.3.1`, `1.3.2`, and `1.4` prefixes are published on Amazon S3.
See the [data guide](data.md#download) for download locations.
Keep each new release's S3 prefix empty until acceptance and publication.
Fresh author-table counts, A, define the production count authority.
See the [A mainline record](release_evidence/author_counts_mainline_20261017.md) for the rebuild evidence.

## Release inputs

Use one final corrected snapshot as the input to each release.
It contains the output of all accepted additions and corrections.
It preserves full paper text, current labels, uncertainty masks, embeddings, producer sidecars, and exact split identities.
It also contains augmented records and complete incremental task definitions.
Do not reconstruct accepted corrections from historical review archives.

The current benchmark masters retain full text where available.
The seven giant replay sources lack full abstract text.
Preserve their available current records and vectors; a new abstract-based encoding needs another source of full abstracts.

See the [data guide](data.md) for artifact layout and required bundles.
The [production command reference](../scripts/production/README.md) contains executable commands.
Keep source snapshots and fit outputs in separate directories.

## Fixed decisions

- Publish the binary name-count index and complete model only in the `inference` bundle.
- Track name tuples and Open Researcher and Contributor ID (ORCID) prefix counts in Git.
- Keep the ORCID prefix counts required by supported models.
- Use the newest packaged ORCID counts in new training contracts.
- Preserve abstract presence and existing SPECTER2 vectors when removing restricted text.
- Use supported formats and schemas for dataset compatibility.
- Bind models to feature-contract versions, ordered feature names, and exact required artifact hashes.
- Record the model's generating runtime version as provenance.
- Rebuild unversioned model stages or export them explicitly under the current contract.
- Train every release model from the published `inference` and `dataset` bundles.
- Select hyperparameters and EPS on development data. Keep test metrics out of model selection.
- Rebuild distributions and rerun installed checks after a package change.
- Rerun affected evaluations after a runtime change, even when the contract stays compatible.

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
3. Declare fit datasets, split identities, pair lists, and the clustering threshold (`EPS`) policy.
4. Verify the declared workload and resource bounds.
5. Keep the selected snapshot immutable for the run.

Before publication, check each paper corpus ID against the production Semantic Scholar Academic Graph (S2AG).
Use the same production snapshot's abstract-sharing gate for source text.
Check augmented paper origins and initial-query source mappings too.
Remove excluded papers, signatures, vectors, cluster memberships, pairs, and task references.
Refresh batch indexes, manifests, provenance, and split counts after filtering.
Keep incremental tasks when their queries and usable profile members survive.
Remove candidate rows with no usable non-query members.
Remove query groups whose query disappears or whose positive support disappears entirely.
Record the removed rows and query groups.
Do not claim that filtered training populations reproduce an earlier model fit.
When a filter removes records from data that a released model was fit on, publish `pairwise/population_corrections.json` beside the model plan.
It pairs each benchmark's original development-population fingerprint with the filtered one.
Check first that every filtered block is a subset of the same original block.

The selected local input path is `D:/local-unsynced/s2and_ca_mainline_20261017/snapshot`.
The snapshot passes relocation checks.
It excludes `inventors_s2and` from corrected records, Arrow data, pairwise plans, and incremental inputs.

## Stage 1: generate runtime data

Reuse unchanged common assets.
Regenerate a common asset only when its source data or semantics change.
Regenerate A after a canonical name-rule change, even when the raw Athena export stays unchanged.
Only AI2 staff can run the Athena export and rebuild A. Expect this for every future refresh.
Recompute pairwise and linker features after a count-authority change.
Exclude the 19 mixed-identity ORCID groups listed in `EXCLUDED_ORCID_GROUPS`.
Keep the rule requiring matching, nonempty canonical surnames.
Run the focused producer checks before a bounded rebuild.
See [the count commands](../scripts/production/README.md#count-artifacts).

Verify that the wheel contains the canonical tuple file and `s2and_surname_variants_canonical.txt`.
Check for the ORCID counts and their tuple-dependency manifest in both `s2and/data/` and `s2and/data/orcid_prefix_counts_v1_4/`.
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
Search main and nameless models on the declared development data.
Pass `--incumbent-parameters` to let a known configuration compete with each search winner.
Use `--fixed-parameters` instead to refit both heads directly.
The trainer writes an immutable pairwise-only stage with EPS marked `pending`.

Review finite validation metrics, source counts, pair counts, and resource use.
Verify both boosters after serialization.
Check the actual trained boosters' native scoring plans:

```powershell
$env:S2AND_PAIRWISE_MODEL_DIR = '<pairwise-stage>/pairwise'
uv run cargo test --manifest-path s2and_rust/Cargo.toml --lib --no-default-features `
  release_bundle_uses_native_fast_plans -- --ignored --nocapture
Remove-Item Env:S2AND_PAIRWISE_MODEL_DIR
```

Cap `num_leaves` at 256 (`MAX_FAST_PLAN_LEAVES`) in the search, incumbent parameters, and fixed parameters.
Larger trees fall back to slower native scoring.
A failed native fast-plan check still requires representative latency measurements and a candidate review.

## Stage 4: select EPS and freeze linker inputs

Calibrate EPS from validation data with `release_pairwise.py calibrate-eps`.
Calibration also decides whether the bundle uses the close-pair bonus.
It scores every EPS for each entry of `CLOSE_PAIR_CANDIDATES` and keeps the best pair; ties go to plain linkage.
Calibration and held-out B-cubed (B3) scores retain full precision.
By default the best signature-weighted F1 wins.

Set `precision_f1_tolerance` in the model plan's `eps` section to lean toward precision.
Calibration then selects the most precise setting whose F1 is within that tolerance of the best.
With the close-pair bonus, F1 is nearly flat over a wide range of EPS, so a small tolerance buys a clear gain in precision.
Packaging requires `--precision-f1-tolerance` and records the supplied value, including zero.
Calibration treats a legacy plan without the field as zero. If a close-pair bonus wins at zero tolerance,
calibration requires `--allow-zero-precision-f1-tolerance` to write the selected settings.
The rule needs no earlier model, so a dataset update or a new pairwise fit does not change how it works.
Calibration writes a fresh calibrated sibling and preserves both boosters.
Verify that only clusterer configuration and bundle metadata change.
This benchmark selection supplies the initial EPS for giant-block review.

Freeze the calibrated pairwise model, corrected incremental source, common assets, linker target, and fit resources.
Recheck safe-action labels if query evidence or profile memberships change.
See the [data update rules](data.md#update-rules) for changes that require new artifacts, fits, or evaluations.

## Stage 5: fit, evaluate, and approve

Freeze evaluation populations, metrics, thresholds, and representative performance workloads before candidate scores are available.
Distinguish independent release tests from inherited regression tests.
Compare old and new models on the same population when a matched comparison is required.
Do not compare historical rates from different data or split populations as a matched baseline.

Run `train_linker_and_finalize.py` once against the calibrated model and the synced data root.
The command computes fresh features, fits the linker, writes a complete sibling, reloads it, and evaluates its declared test split.

The fixed classic booster uses 53 features and 675 trees and fits training rows.
Before computing features, the trainer reads complete holdout query and base identities from the source tables.
It needs only the identity columns, even when it computes features for calibration rows only.
An overlap excludes the whole training query and preserves each retained query's candidate context.
The trainer checks the same identities again before fitting and records the early exclusion counts.
It uses origin maps to remove original queries from candidate profiles.

Use `calibration_fit` and `calibration_check` as development splits for gate feature selection and coefficient fitting.
Exclude verified query-paper exposure before gate calibration. Retain the remaining manual development labels.
Keep the exposure annotations for pairwise training, EPS selection, and paper overlap.

Complete [giant-block EPS review](eps_calibration.md) before final release evaluation.
The release owner selects the final EPS there; that guide also records the `1.3` decision.
Apply the reviewed EPS to a fresh complete model without another fit.
Freeze that package before final acceptance tests.

Read `evaluation_summary.stratified_eval_test_split.overall.test` as regression evidence because earlier development used that population.
Full-source diagnostics can also include gate-calibration queries.
Score frozen identity-gold cases through the complete model with [evaluate_incremental_gold.py](../scripts/verification/evaluate_incremental_gold.py).
Report those scores separately from the [manual safe-action label](data.md#incremental-tasks) scores.

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
Build the unified source archive and platform wheels from that commit.
Verify wheel and source archive contents.
Install the candidate distributions in an empty environment outside the checkout.
Run bulk and incremental prediction with the real complete model and combined data root.
The workflow's synthetic smoke does not replace this check.
See [installed release smoke](../scripts/production/README.md#installed-release-smoke).

Publish through `.github/workflows/release-rust.yml`.
A push to `main` that changes `VERSION` starts it; a manual dispatch from the reviewed `main` commit does too.
Its validate job runs `scripts/run_ci_locally.py`, the same checks as pull-request CI.
A push whose version PyPI already has releases nothing; a manual run fails on that version.
Verify the workflow commit before approval of its protected PyPI environment.
The workflow publishes one source archive and 15 platform/interpreter wheels.
Hold that approval until model and data acceptance are complete.

### Beta pre-releases

A beta lets integrators test before acceptance. It never replaces Stage 6.

1. Set `VERSION` to a PEP 440 pre-release such as `1.0.0b1`.
   Run `uv run python scripts/sync_version.py`; Cargo files receive `1.0.0-beta.1`.
2. Push the branch and dispatch `release-rust.yml` on it.
   A pre-release builds and smoke-tests on any branch; the publish jobs run only on `main`.
   The validate job fails when PyPI already has the version.
3. Download the run's `release-distributions` artifact.
   Confirm every file carries the beta version.
4. Confirm that `https://pypi.org/pypi/s2and/<beta version>/json` still returns 404.
   PyPI adds new files to an existing release, so a reused version would mix two builds.
   Upload the unified `s2and` distributions with `uv publish` and an API token.
5. Install `s2and==<beta version>` in an empty environment and run one real prediction.

Installers skip pre-releases unless the requirement names one, so `pip install s2and` stays on the last final release.
Set `VERSION` back to the final version before Stage 6.

## Stage 7: finish

Write the bundles again with the accepted model, then publish each bundle to its immutable S3 prefix.
Release `1.3` uses `s3://ai2-s2-research-public/s2and/1.3/` with `inference/`, `dataset/`, and `source/`.
Model release `1.3.1` changes only linkage settings, so it publishes `inference/` and `dataset/` under `s2and/1.3.1/` and reuses the `1.3` `source/` bundle.
Its `dataset/` is the `1.3` bundle with the `1.3.1` model plan.
Model release `1.3.2` rebinds the `1.3.1` model to a rebuilt name-count index with
`scripts/production/model/rebind_name_counts_index.py`, so it publishes only `inference/` under `s2and/1.3.2/`.
Release `1.4` publishes `inference/` and `dataset/` under `s2and/1.4/` and no `source/` bundle.
Reuse the dataset bundle for model-only updates with unchanged data.
Sync `inference` and `dataset` into one fresh local root.
Verify hashes and repeat installed bulk and incremental smoke checks.
Run one public step from the [reproduction guide](reproduce.md) on that root.

Record the package, model, research snapshot, and format versions in the release notes.
Record the common Git ref, asset paths, S3 prefix, manifest hashes, and evaluation decision.
Replace unpublished placeholders only with verified download locations.
Resolve every `Update at v1.3 release` note (`git grep -n "Update at v1.3 release"`).
Tag the exact reviewed source commit.

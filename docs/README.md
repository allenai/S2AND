# S2AND documentation

The guides describe the current runtime and the corrected dataset release process.
The compatible model and research download locations remain unpublished.
The [release process](release.md) states the remaining fit, evaluation, and publication steps.
The [project README](../README.md) provides the installation and workflow overview.

## Start with your goal

| Goal | Guide |
| --- | --- |
| Run the model on your own records | [Production inference](production_inference.md) |
| Obtain the common assets and corrected research snapshot | [Data and models](data.md) |
| Reproduce the production training process on public data | [Reproduce the 1.3 model](reproduce.md) |
| Correct data, add a dataset, or release another model | [Release process](release.md) |
| Prepare a new corrected snapshot | [Data preparation commands](../scripts/production/data/README.md) |
| Fit the production model | [Production commands](../scripts/production/README.md) |
| Run a research API experiment | [Training and evaluation](training.md) |

One complete corrected snapshot supports research and the next correction cycle.
Common large assets use Git LFS; research data uses S3 without a second common asset copy.
Small name tuples and ORCID counts remain in normal Git and the wheel.
The [data guide](data.md) defines the shared local layout and update rules.

## Runtime guides

- [Install and setup](install.md)
- [Large blocks and subblocking](subblocking.md)
- [Thread and memory controls](threading.md)
- [Environment variables](environment.md)
- [Canonical name rules](canonical_names.md)
- [Public Arrow and name-count formats](rust/arrow_dataset_spec.md)

## Contributor guides

- [Development and local CI](development.md)
- [Test guide](../tests/README.md)
- [Native package overview](../s2and_rust/README.md)
- [Script catalog](../scripts/README.md)
- [Release notes](release_notes.md)

## Historical notes

Dated notes in `rust/profiling` describe their named runs.
They are not current preparation commands or mandatory inputs to a new correction cycle.
The corrected snapshot already contains accepted corrections.
Use historical records only for questions about those prior decisions or measurements.
The original paper process belongs to the [paper branch](../README.md#reproducibility).

## Glossary

| Term | Meaning |
| --- | --- |
| Signature | One author position on a paper |
| Block | Signatures with the same canonical author block key |
| Subblock | A bounded partition of a large block |
| Cluster or profile | Signatures assigned to one author |
| Query | A requested signature outside the supplied existing profile memberships |
| Seeds | Required profile memberships and disallowed pair relations supplied with a request |
| Initial-only view | A task whose query evidence contains an initial instead of a full first name |
| Pairwise model | Main and nameless LightGBM boosters that score signature pairs |
| EPS | The average-linkage distance threshold selected from validation data |
| Incremental linker | The 53-feature link-or-abstain model and gate |
| Safe-action label | Whether the supplied evidence supports attachment to the complete candidate profile |
| Corrected snapshot | Complete current records, accepted labels, masks, vectors, and exact task/split definitions |
| Common assets | Complete model and count index reused by inference and research |
| Runtime Arrow projection | Fast runtime tables derived from corrected records; full abstract text remains in the corrected source |
| Batch index | A `*_batch_index.bin` sidecar that maps lookup keys to Arrow record batches |
| Pairwise-only stage | Both pairwise boosters with pending or calibrated EPS; not a complete inference model |
| Complete model | Calibrated pairwise stage plus incremental artifact, required fixtures, and target |
| Format version | Supported serialization/schema contract, separate from the Python package version |
| Feature-contract version | Supported feature semantics plus ordered feature names |
| Generating runtime | Model provenance; not an exact model-loader package-version lock |

# Reproducibility

This document covers the paper-era environment and compatibility notes for older released artifacts.

## Paper-era branch

The original paper experiments were run on the `s2and_paper` branch with a Python `3.7.9` environment captured in `paper_experiments_env.txt`.

If you need to reproduce the paper-era setup:

```bash
git checkout s2and_paper
uv venv --python 3.7.9
```

Then install the pinned environment from `paper_experiments_env.txt` inside that isolated environment and rerun the paper experiment command set from the `s2and_paper` branch. Current `main` retains selected provenance artifacts under `scripts/archive/`, but they are not substitutes for the paper-era branch or supported current entrypoints. See [scripts/README.md](../scripts/README.md#archived-historical-artifacts) for their individual status.

## Paper-era released artifacts

Paper-era seed artifacts such as:

- `full_union_seed_*.pickle`

are legacy artifacts for reproducing the original paper setup and should be used
from the `s2and_paper` branch, not current `main`.

Some historical model pickles used by the paper-era branch stored a dictionary
with a `clusterer` key rather than a bare clusterer object. The current branch
does not distribute or load production-model pickles; use the paper-era branch
and its released artifacts when reproducing those runs.

## Previous release and canonical migration branch

The checked-in native v1.21 bundle was the previous release default:

- `production_model_v1.21/`

The obsolete v1.0-v1.2 production pickles have been removed. Canonical-v2 also
rejects the v1.21 source bundle because its normalization contract is legacy.
No compatible model or default is distributed on this branch; current
evaluation requires an explicit model bundle path. After v1.3 is trained and
validated, B15 still requires an explicit choice between an external bundle
and a packaged default.

The v1.21 bundle includes the previous promoted incremental linker under
`incremental_linker/`. Its replay target is tracked separately at
`production_model_v1.21/reproducibility/incremental_linker_training_target.json`;
replay scripts should not depend on machine-local analysis artifacts.

Promoted-linker replay always materializes a fresh Arrow/Rust feature bundle
from the source bundle, target JSON, and pairwise model. The destination must be
new, so a replay cannot silently consume stale feature tables. A bounded
`--materialize-only` smoke is necessary, but it does not authorize a full v1.3
replay: B13/B20 still require retention and reviewed no-retraining promotion of
the exact evaluated candidate. Follow [1_3_release_todo.md](1_3_release_todo.md)
before any full command, and record all source identities, runtime, peak RSS,
predictions, and bundle hashes in the durable release record.

See [production_inference.md](production_inference.md) for the current inference contract.

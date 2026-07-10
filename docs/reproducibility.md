# Reproducibility

This document covers the paper-era environment and compatibility notes for older released artifacts.

## Paper-era branch

The original paper experiments were run on the `s2and_paper` branch with a Python `3.7.9` environment captured in `paper_experiments_env.txt`.

If you need to reproduce the paper-era setup:

```bash
git checkout s2and_paper
uv venv --python 3.7.9
```

Then install the pinned environment from `paper_experiments_env.txt` inside that isolated environment and rerun the paper experiment command set from the `s2and_paper` branch. The repository keeps a reference copy at `scripts/archive/paper_experiments.sh`, but that file is historical and not the supported entrypoint for current development.

## Paper-era released artifacts

Paper-era seed artifacts such as:

- `full_union_seed_*.pickle`

are legacy artifacts for reproducing the original paper setup and should be used
from the `s2and_paper` branch, not current `main`.

Some historical model pickles used by the paper-era branch stored a dictionary
with a `clusterer` key rather than a bare clusterer object. Current `main`
compatibility artifacts use versioned names such as `production_model_v1.2.pickle`
and are covered below.

## Previous release and canonical migration branch

The checked-in native v1.21 bundle was the previous release default:

- `production_model_v1.21/`

Legacy pickles are still present as historical/parity inputs:

- `production_model_v1.2.pickle`
- `production_model_v1.1.pickle`

Canonical-v2 rejects all of these artifacts because their normalization
contract is legacy. No compatible default exists on this branch until v1.3 is
trained and validated. Do not use their presence in package data as evidence
that current inference is supported.

The v1.21 bundle includes the previous promoted incremental linker under
`incremental_linker/`. Its replay target is tracked separately at
`production_model_v1.21/reproducibility/incremental_linker_training_target.json`;
replay scripts should not depend on machine-local analysis artifacts.

For repeated promoted-linker replay, materialized feature bundles can be reused
only through the explicit `precomputed-promoted` mode. The bundle must be
portable and validated against the replay target JSON. Feature-table metadata in
the reusable input bundle must use bundle-relative paths; finalized production
artifact audit metadata may still record historical scratch/provenance paths,
but replay must not depend on them.

See [production_inference.md](production_inference.md) for the current inference contract.

# Reproducibility

This document covers the paper-era environment and compatibility notes for older released artifacts.

## Paper-era branch

The original paper experiments were run on the `s2and_paper` branch with a Python `3.7.9` environment captured in `paper_experiments_env.txt`.

If you need to reproduce the paper-era setup:

```bash
git checkout s2and_paper
uv venv --python 3.7.9
```

Then install the pinned environment from `paper_experiments_env.txt` inside that isolated environment and rerun `scripts/paper_experiments.sh` from the `s2and_paper` branch.

## Old released model artifacts

Older released pickles such as:

- `production_model.pickle`
- `full_union_seed_*.pickle`

are paper-era artifacts and only run on the `s2and_paper` branch, not on `main`.

Those pickles store a dictionary with a `clusterer` key rather than a bare clusterer object.

## Current branch

For current work on `main`, prefer the versioned released models:

- `production_model_v1.2.pickle`
- `production_model_v1.1.pickle`

See [production_inference.md](production_inference.md) for the current inference contract.

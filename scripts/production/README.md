# Production Release Artifacts

Scripts in this directory create or validate checked-in production artifacts
under `s2and/data/`. The model release artifact is a native
`production_model_vX.Y/` directory. Do not create production pickles.

Examples below use `X.Y` as the target production bundle version.

## 1. Train Pairwise

```powershell
uv run python scripts/production/model/train_pairwise.py `
  --production-version X.Y `
  --output-dir scratch/pairwise_stage/production_model_vX.Y `
  --run-full
```

This writes the pairwise-only bundle stage:

```text
production_model_vX.Y/
  clusterer.json
  manifest.json
  pairwise/
    main.lgb
    nameless.lgb
    main_prediction_fixture.json
    nameless_prediction_fixture.json
  reproducibility/
    pairwise_training_config.json
    pairwise_training_summary.json
```

This stage is loadable for training/finalization, but it is not a complete
runtime production model until the linker is added.

## 2. Train Linker And Finalize

```powershell
uv run python scripts/production/model/train_linker_and_finalize.py `
  --production-bundle-version X.Y `
  --target-json scratch/production_linker_vX.Y/incremental_linker_training_target.json `
  --pairwise-model-path scratch/pairwise_stage/production_model_vX.Y `
  --save-production-bundle-to s2and/data/production_model_vX.Y `
  --linker-artifact-version vX.Y `
  --output-dir scratch/production_linker_vX.Y `
  --run-full
```

The destination must not exist. The command trains the linker under
`scratch/production_linker_vX.Y/production_incremental_linker/`, assembles and
validates a complete sibling staging directory, then publishes the bundle with
one directory rename. The pairwise stage remains unchanged. The final bundle
contains:

```text
production_model_vX.Y/
  incremental_linker/
    booster.lgb
    metadata.json
  reproducibility/
    incremental_linker_training_target.json
  manifest.json
```

Linker training has one feature path: it materializes a fresh Arrow/Rust
feature bundle under the requested output directory, then trains from that
bundle. The feature-bundle destination must not already exist. Use
`--materialize-only` with `--limit-rows`, `--tables`, or `--datasets` for a
bounded smoke run before approving an unbounded `--run-full` job.

After this step, users load the model with:

```python
from s2and.production_model import load_production_model

clusterer = load_production_model("s2and/data/production_model_vX.Y")
```

There is no implicit default model. Runtime callers must pass the complete
bundle path; pairwise-only stages are accepted only by internal training code.

## Arrow Release Validation

For local release-root smoke checks that do not touch S3 or scan large Arrow
tables, run:

```powershell
uv run python scripts/verification/validate_local_arrow_release.py `
  --release-root s2and/data
```

This verifies manifest checksums, required local files, raw-planner batch-index
paths, replay-bundle manifest references, and `name_counts_index/manifest.json`
targets. Use `scripts/convert_to_arrow.py validate --dataset-dir ...` for
deeper per-dataset Arrow schema/table validation.

## Count Artifacts

The `counts/` scripts document production count artifacts:

- `counts/generate_name_counts.py` writes a provenance-bound immutable
  `name_counts_index/` into a previously absent target. It requires an explicit
  source snapshot, verifies selected-row content, and supports bounded fixture
  runs before any authorized warehouse run. The writer builds the complete
  directory in a temporary sibling and publishes it with one rename. Python
  and Rust runtime paths share that verified mmap index; neither unpickles nor
  retains the full dictionaries. Models compare the exact generation/source
  binding before feature work. Regeneration uses a new output directory; there
  is no in-place overwrite mode.
- `counts/generate_orcid_name_prefix_counts.py` writes canonical unordered
  ORCID prefix pairs directly to `first_k_letter_counts_from_orcid.json` with
  an adjacent `.meta.json` sidecar. The runtime verifies the normalization and
  pair semantics plus the exact data SHA-256; it never accepts the data file
  without its sidecar. Producer-only source provenance, parameters, and metrics
  are written separately to `first_k_letter_counts_from_orcid.generation.json`
  so they do not expand the runtime metadata contract. `--max-names-per-orcid`
  is checked before quadratic pair expansion. Install its JSON serializer with
  `uv sync --extra orcid-counts` before running the producer.

Both scripts are import-safe without the internal warehouse package. Start
with `--help`, `--dry-run`, or a small fixture: `--fixture-input` for
`generate_name_counts.py` and `--input-json` for
`generate_orcid_name_prefix_counts.py`. A full internal query requires
`--run-full`, an explicit `--output-dir`, and a reviewed `--source-snapshot-id`;
it is not part of the local verification suite.

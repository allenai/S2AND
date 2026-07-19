# Pair-source ablation

This study asks one question: does adding balanced linker-derived training
pairs improve held-out clustering without harming any public gold domain?

The maintained runner does not extract pairs or featurize raw datasets. It
accepts one prepared directory:

```text
training/catalog.parquet
training/feature_schema.json
training/main.npy
training/nameless.npy
evaluation/<domain>/main.npy
evaluation/<domain>/nameless.npy
evaluation/<domain>/labels.npy
b3/<gold-domain>/main.npy
b3/<gold-domain>/nameless.npy
b3/<gold-domain>/staged_labels.npy
b3/<gold-domain>/pair_offsets.npy
b3/<gold-domain>/signature_offsets.npy
b3/<gold-domain>/signature_ids.npy
b3/<gold-domain>/gold_cluster_ids.npy
```

The catalog has exactly `source_domain`, `source_family`, `pair1`, `pair2`,
and `label`. Families are `base` and `linker`; labels are integer `0` or `1`;
pairs are already canonical (`pair1 < pair2`). Training feature rows align
with catalog rows. `feature_schema.json` records the current S2AND featurizer
and normalization versions plus the exact ordered main/nameless feature
groups. Feature NPY files are C-contiguous float32/float64 arrays. Evaluation
arrays align within each domain. B3 feature
rows use SciPy condensed pair order within every `signature_offsets` block;
`pair_offsets` gives the corresponding feature-row boundaries.

The baseline uses every non-held-out `base` row. An additive arm appends a
deterministic, class-balanced prefix of linker rows from either `all13` or
`big7`. It never replaces or resamples baseline rows.

Run a bounded fresh study by naming every fold explicitly:

```powershell
uv run python scripts/run_pair_source_ablation.py `
  --prepared-dir scratch/pair_ablation_prepared `
  --donor-model-dir s2and/data/production_model_v1.21/pairwise `
  --output-dir scratch/pair_ablation_run `
  --domain pubmed --domain qian `
  --source-set big7 --dose 1250 --n-jobs 8
```

Each fold writes one flat JSON result beneath
`results/<seed>/<arm>/<domain>.json` and its two models beneath the matching
`models/` path. Results bind the prepared input and donor/training recipe,
selected training-pair digests, base and linker row counts, pairwise
AUROC/AUPRC, and held-out B3.

Analyze one or more fresh result trees with:

```powershell
uv run python scripts/analyze_additive_linker_dose_ablation.py `
  scratch/pair_ablation_run/results
```

Candidate and baseline cells are paired by seed and held-out domain. A release
decision requires all seven public gold domains and all three inclusive gates:

- mean paired B3 delta at least `0`;
- fifth-percentile paired B3 delta at least `-0.002`;
- worst-domain mean paired B3 delta at least `-0.010`.

Proxy and pair-only domains never enter B3 gates. There is deliberately no
resume path, cache migration, artifact-version framework, ranking manifest,
or historical result reader. The output path must not exist; start failed or
changed studies at a new path.

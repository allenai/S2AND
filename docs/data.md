# Data and Models

This document covers dataset download, model-only download, and `path_config.json`.

## Full dataset download

Download the full S2AND release into `s2and/data/`:

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release s2and/data/
```

Expected size is about `50.4 GiB`.

The release includes dataset files plus released model artifacts.

## Model-only download

If you only want the current pairwise production model, download just the pickle:

```bash
aws s3 cp --no-sign-request s3://ai2-s2-research-public/s2and-release/production_model_v1.2.pickle s2and/data/
```

This is enough for the quick-start path that uses the bundled `tests/qian` fixture.

The promoted incremental linker is a directory artifact, not a pickle:

```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/s2and-release/production_incremental_linker_v1.2 s2and/data/production_incremental_linker_v1.2
```

It is used with `production_model_v1.2.pickle`.
The directory includes `training_target.json`, which is the portable training
target spec used by replay scripts.

## Configuring `s2and/data/path_config.json`

Some scripts look up the main data root through `s2and/data/path_config.json`
or the `S2AND_PATH_CONFIG` environment variable. This config points at the
downloaded benchmark dataset root; it is separate from the package data checked
in under `s2and/data/`.

Example:

```json
{
  "main_data_dir": "absolute path to your downloaded S2AND data",
  "internal_data_dir": ""
}
```

Guidance:

- Set `main_data_dir` to the directory containing your downloaded S2AND datasets.
- `internal_data_dir` is only relevant for internal AI2 workflows and can be left empty.
- If your data lives in this repo's `s2and/data/` directory, the default placeholder config already resolves there.

## Dataset file expectations

Most workflows use the standard S2AND JSON files for:

- signatures
- papers
- clusters
- optional cluster seeds
- SPECTER embeddings

The tutorial script supports both:

- mini-dataset naming such as `<dataset>_papers.json`
- plain fixture naming such as `papers.json`

See [production_inference.md](production_inference.md) for the minimal inference input contract, and [training.md](training.md) for training-mode dataset requirements.

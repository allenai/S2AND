# Global name counts from Parquet

Use `generate_name_counts_from_parquet` for large reviewed name-count exports.
The producer sums plain mention counts across the complete source population.
It applies the current canonical name rules and retains counts above one.
It preserves the existing four-file binary index format.

Each Parquet object must contain `first_name: string`, `last_name: string`, and `mention_count: int64`.
Additional columns are permitted.
Null name fields follow the current canonical name rules.
Mention counts must be positive integers.

Supply a reviewed JSON manifest with a nonempty `files` list.
Each entry requires `path`, `bytes`, `rows`, and `sha256`.
Paths can be absolute or relative to the manifest directory.
If the manifest contains `status`, its value must be `complete`.

```json
{
  "files": [
    {
      "path": "part-000.parquet",
      "bytes": 12345,
      "rows": 100,
      "sha256": "<reviewed SHA-256>"
    }
  ]
}
```

Set exact row and mention bounds before the run.
Use a new output directory.
This example uses the reviewed September 25 export population.

```powershell
uv run --no-sync --with 'duckdb>=1.5,<2' python -m scripts.production.counts.generate_name_counts_from_parquet `
  --input-manifest path/to/reviewed/manifest.json `
  --output-dir path/to/new/count-publication `
  --expected-source-rows 89695371 `
  --expected-source-mentions 699054083 `
  --threads 4 `
  --memory-limit 12GB
```

The producer checks all source SHA-256 values before the database stage.
DuckDB stores aggregate tables and sort checkpoints on disk.
The native batch hash function and existing binary writer produce the index files.
The native loader verifies the index before publication.
`run.json` contains source identities, source populations, per-mapping counts, progress, and completion status.
Keep stdout and stderr logs for long runs.

Add `--resume` to the same command after a failed or interrupted run.
Resume requires the same source manifest, expected populations, and producer code.
You can change the thread limit and DuckDB memory limit.
Resume reuses completed tables and files.
A completed resume preserves published files and timestamps.

## September 25 source policy

The Athena source contains 699,054,142 author mentions.
Its export excludes 59 blank-surname mentions.
The complete surname-eligible export contains 89,695,371 rows and 699,054,083 mentions.
The source README records this reconciliation and the export SQL.
The earlier CSV contains a citation join error.
Use the validated Parquet source for this release.

The October 12 release build uses the scratch predecessor of this producer.
Its index contains 45,474,003 retained keys across four mappings.
All 75,085 research M query keys match the complete index.
The durable producer has byte parity with the existing production writer on focused Parquet tests.
The release evidence file records the source and artifact hashes.

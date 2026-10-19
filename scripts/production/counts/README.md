# Author-table name counts

These producers read internal Semantic Scholar exports, so only AI2 staff can run them.
Public users download the released index and prefix artifacts instead.
Plan a staff rebuild for every future count refresh.

## ORCID prefix evidence

Use `generate_orcid_name_prefix_counts` for the final ORCID prefix artifact.
The producer excludes 14 mixed-identity ORCID groups from the September 25 manual review.
`EXCLUDED_ORCID_GROUPS` contains the reviewed identifiers.
The producer records excluded group and row counts in its result log.
It preserves raw source data and requires matching, nonempty canonical surnames for all retained ORCID evidence.
It retains the EVENT threshold of 10 and the curated alias threshold of 2.
The exclusion does not affect curated aliases or the separate runtime same-ORCID subblocking rule.
The artifact and manifest formats remain unchanged.

```powershell
uv run --no-sync python -m scripts.production.counts.generate_orcid_name_prefix_counts `
  --input-csv path/to/reviewed/sorted-orcid-names.csv `
  --guardrails-json path/to/reviewed/guardrails.json `
  --output-dir path/to/new/prefix-publication
```

The CSV requires `raw_orcid,orcid,first_name,middle,last_name` in this order.
Sort its rows by canonical ORCID.
Supply exact source bounds and reviewed name-expansion bounds.

## Global name-count index

Use fresh author-table counts, A, for production.
One source author row supplies one contribution.
This quantity is an author-table row count. It is not an independently verified count of people.
The retired mention-count study remains in [the archive](../../../docs/archive/name_count_experiment_20261009_results.md).
The separate ORCID prefix artifact retains its current policy.

### Athena author-row export

Use `generate_author_counts_from_athena` to export and build A.
The `author-counts` extra requires Boto3 1.40 or later for the data-manifest API.
The source query preserves all raw name groups, including null names:

```sql
SELECT first_name, last_name, count(*) AS mention_count
FROM content_ext.authors
GROUP BY first_name, last_name
```

Do not add a paper join or a surname filter.
Do not normalize names in SQL.
Preserve raw case because the initials rule uses the surname's original case.
The `mention_count` column retains the shared producer's historical field name.
For A, its value is the number of source author rows in that raw name group.

Select an AWS profile or use the standard credential chain.
Use an Athena workgroup with `BytesScannedCutoffPerQuery` at or below the declared scan limit.
Reserve a new, empty S3 prefix for each export.
The [Athena UNLOAD contract](https://docs.aws.amazon.com/athena/latest/ug/unload.html) defines the empty-prefix requirement.

Inspect the plan before a query:

```powershell
uv sync --extra author-counts
uv run --no-sync python -m scripts.production.counts.generate_author_counts_from_athena `
  --profile <aws-profile> `
  --region us-west-2 `
  --workgroup <bounded-workgroup> `
  --query-output s3://<bucket>/athena-results/ `
  --unload-prefix s3://<bucket>/author-counts/<new-export>/ `
  --output-dir <fresh-output> `
  --expected-author-rows <source-author-row-count> `
  --expected-raw-groups <raw-first-last-group-count> `
  --max-scan-bytes 15000000000 `
  --max-query-seconds 900 `
  --threads 4 --memory-limit 12GB
```

The plan prints source SQL, population-audit SQL, resource limits, and a scan-price estimate.
It submits no query and needs no AWS credentials.
The scan-price input is configurable. The workgroup cutoff supplies the service-side cancellation threshold.
Confirm population counts against the selected source snapshot.
Add `--execute` to the same command to start the export and local build.

The October 15 source has 192,718,689 author rows in 84,128,144 raw name groups.
These figures identify that snapshot. Do not assume they describe a later refresh.
The producer verifies the downloaded author-row mass and optional raw-group count.
It uses only the query's exact data manifest, with file, schema, size, and SHA-256 checks.
The query ID, SQL, token, source manifest, and completion status remain under the output directory.

Add `--execute --resume` to recover the same export.
Resume preserves its idempotency token and query ID.
Complete local exports need no AWS SDK or credentials.
A changed normalizer or source plan requires a fresh output directory.
The published index is `<fresh-output>/canonical/name_counts_index`.

### Canonical key contract

Use `generate_name_counts_from_parquet` for large verified author-row exports.
It applies the current canonical name rules and retains counts above one.
It preserves the existing four-file binary index format.
The producer canonicalizes each distinct first name and each distinct last name once.
The first-name map keeps the canonical first with and without the D9 initials rule.
The last-name map keeps the D9 guard, and the join selects the applicable first-name result.

Each Parquet object must contain `first_name: string`, `last_name: string`, and `mention_count: int64`.
Additional columns are permitted.
Null name fields follow the current canonical name rules.
Weights must be positive integers.
For A, each weight counts author-table rows in one raw name group.
An informative first name has more than one character.
Only informative first names contribute `first` and `first_last` keys.
A nonempty surname contributes `last`.
Both name parts must exist for `last_first_initial`, which uses the first character of the canonical first name.
An ineligible lookup produces a missing feature.
An eligible key absent from the index returns a count of one.
Aggregate all contributions to each canonical key before the count-above-one filter.

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

Set exact raw-group and author-row bounds before the run.
Use a new output directory.
This example uses the verified October 15 author-table population.

```powershell
uv run --no-sync --with 'duckdb>=1.5,<2' python -m scripts.production.counts.generate_name_counts_from_parquet `
  --input-manifest path/to/reviewed/manifest.json `
  --output-dir path/to/new/count-publication `
  --expected-source-rows 84128144 `
  --expected-source-mentions 192718689 `
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
Resume requires the same source manifest, expected populations, producer code, and normalizer source.
You can change the thread limit and DuckDB memory limit.
Resume reuses completed tables and files.
A completed resume preserves published files and timestamps.

For A, `--expected-source-rows` counts raw name groups.
The retained `--expected-source-mentions` option counts author-table rows.
The Parquet producer publishes `<fresh-output>/name_counts_index`.
The earlier M source policy remains in the archived study.

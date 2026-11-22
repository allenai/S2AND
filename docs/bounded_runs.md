# Bounded model runs

## Plan the run

Detect available logical CPUs from process affinity before each CPU-intensive launch.
Reserve two logical CPUs and share the remaining thread budget across concurrent jobs.
Apply a lower total limit when the user requests one.
For a 12-thread total limit, allocate four threads to each of three workers.
Apply each allocation to model prediction, OpenMP, and Rayon.
Keep secondary Basic Linear Algebra Subprograms (BLAS) pools at one thread.
Use disjoint CPU affinity leases to enforce allocations across library pools.
Record the detected capacity and allocations.

Run one serial milestone at a time.
Use a bounded representative sample before an expensive milestone.
Record its command, inputs, resource limits, expected duration, and required outputs.

For comparisons, complete one production model before queuing more variants.
Strictly reload the complete bundle.
Run a bounded public prediction on one representative full block.
Verify its outputs before starting the next variant.
For the giant-block count study, use the full Silva block at epsilon (EPS) values of 0, 0.55, and 1.

Before a threshold sweep, identify work independent of the threshold.
Verify exact outputs before reusing that work across cuts.
Keep threshold-dependent constraints, temporary seeds, attachment, and residual prediction in the public production path.
Cache pristine arrays before use.
Bind the cache to fixed inputs, models, counts, source, native code, and resource settings.
Measure cold and warm runs to estimate the full runtime.

## Check the checkout

Run these Git checks in order before an expensive launch.
Give each command its own time and memory limit.

```bash
uv run --no-sync python scripts/run_bounded.py --output-dir scratch/git-head-check --max-seconds 15 --max-rss-gib 1 -- git rev-parse --verify "HEAD^{commit}"
uv run --no-sync python scripts/run_bounded.py --output-dir scratch/git-tree-check --max-seconds 15 --max-rss-gib 1 -- git cat-file -e "HEAD^{tree}"
uv run --no-sync python scripts/run_bounded.py --output-dir scratch/git-ref-check --max-seconds 15 --max-rss-gib 1 -- git for-each-ref --sort=-committerdate "--format=%(refname)%00%(objectname)%00%(*objectname)"
```

These checks cover the HEAD commit, root tree, and ref enumeration.
Stop the launch if a check fails or exceeds its limit.
Preserve the checkout for a separate Git diagnosis.
Keep follow-up checks bounded.

Keep active checkouts and their Git directories outside cloud-sync folders.
Transfer commits with Git push and fetch.
Cloud sync can cause missing objects and broken refs. [Git FAQ](https://git-scm.com/docs/gitfaq#_transfers)
Leave the checkout in place while jobs use its files or environment.

## Launch and verify

Wrap each milestone with `scripts/run_bounded.py`.
Set `--output-dir` to a fresh directory and set `--max-rss-gib`.
Select `--max-seconds SECONDS` or `--no-deadline`.
Use `--no-deadline` when the approved workload has no wall deadline.
This records `max_seconds` as `null` and keeps the memory limit and process ownership active.

Repeat `--expected-output PATH` for each required output file.
Use paths that do not exist before launch.
Reserve the runner's own records for execution evidence.
Put the child command after `--`.
The runner checks file presence and records file size.
Validate artifact contents separately.

Preserve `run.json`, `events.jsonl`, `stdout.log`, and `stderr.log`.
Confirm completion with `result.json` and validated outputs.
Require `status` to be `complete` and `command_exit_code` to be zero.
Preserve partial outputs and worker logs after failure.

Use `supervisor_pid` in `run.json` to identify the actual supervisor.
Outer `uv` and environment launcher process IDs can differ.
Use one timed query per progress check.
When a run exceeds its limit, stop only its owned process tree.
The runner samples resources every 0.1 seconds, so it can miss short memory peaks.

## Process ownership and telemetry

On Windows, a Job Object owns the supervisor and descendants from launch.
When the supervisor dies, its private handle closes and stops remaining descendants.
On POSIX, the runner owns the command's process group.
A descendant can start a session outside that group.
Hard supervisor death can leave group members alive.

The run stays open while an owned descendant remains.
The Microsoft Visual C++ linker, `link.exe`, can leave its telemetry process `vctip.exe` running.
Ten seconds after the command exits, the runner checks the remaining Windows Job Object members.
Cleanup runs only when the remaining members are Visual Studio `vctip.exe` processes or the console host (`conhost.exe`).
The runner stops the telemetry processes and lists them under `telemetry_cleanup` in `result.json`.
The console host then exits on its own.

On Windows, an open reader can block atomic replacement if it omits delete sharing.
Prefer immutable records and append-only telemetry.
For replaceable status files, allow delete sharing and close readers promptly.
Report optional telemetry failures while the worker continues.
Fail the milestone if it cannot publish a required artifact.

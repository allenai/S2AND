# Installation

This document covers the fuller install and setup options for S2AND.

## Requirements

- Python `3.11`, `3.12`, or `3.13`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Git LFS for the LightGBM test fixtures in a source checkout
- Rust, if you are working from a source checkout or building the native extension from source:
  [`rustup`](https://www.rust-lang.org/tools/install)

As of this version, `s2and-rust` is a required runtime dependency. Package
installs get the matching wheel when one is available; source checkouts should
build the local extension.

If you are building the Rust extension from source, install OS prerequisites first.

Ubuntu / Debian / WSL2:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libgomp1
```

Windows:

- Install Visual Studio Build Tools with the `Desktop development with C++` workload.

Toolchain sanity check:

```bash
uv --version
rustc --version
cargo --version
```

## Package install

Runtime install:

```bash
uv pip install s2and
```

This installs the latest version available from the configured package index.
The `1.0.0` packages are coordinated `s2and==1.0.0` and `s2and-rust==1.0.0`
with model/public-data version `1.3`.

> **Update at v1.3 release:** until `1.0.0` is published, this command installs
> an earlier release rather than this worktree. Use the checkout flow below for
> pre-release validation.

## Repo checkout

Hydrate the LFS-managed LightGBM parity fixtures after cloning and after
switching to a branch that changes them:

```bash
git lfs install
git lfs pull
```

Create and activate a supported Python environment (3.11, 3.12, or 3.13):

```bash
uv venv --python 3.11
```

Activation examples:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
. .venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

Install repo dependencies:

```bash
uv sync --active --extra dev
```

If you do not want to activate the environment first, `uv sync --extra dev` also works and will use the project environment.

The `dev` extra includes everything the tests need. Outside it, the
`orcid-counts` extra installs `orjson`, which the ORCID prefix-count producer
(`scripts/production/counts/generate_orcid_name_prefix_counts.py`) requires.

## Build the Rust extension from source

`uv sync --extra dev` builds `s2and-rust` from `s2and_rust/` and installs it
into the project environment, and syncing `uv run` commands rebuild it after
ordinary Rust edits. The test suite stops before collection if the installed
extension was built from different Rust sources than the checkout.

If a rebuild is missed, force a refresh:

```bash
uv sync --extra dev --reinstall-package s2and-rust
```

If a forced reinstall still reports a mismatch (for example after edits that
preserve timestamps, or after switching checkouts that share
`CARGO_TARGET_DIR`), clear the local crates' cached release artifacts and
repeat the forced refresh:

```bash
uv run --no-sync cargo clean -p s2and_rust -p cld2 --manifest-path s2and_rust/Cargo.toml --release
```

Notes:

- If you just installed Rust with `rustup` in the current shell, load its environment first if needed.
- Keep the Python and Rust packages from the same checkout. Do not rely on a
  public `s2and-rust` wheel for unreleased source behavior.

How rebuild detection and the source digest work is described in
[native build troubleshooting](development.md#native-build-troubleshooting).

## Running repo scripts

When running scripts from a repo checkout, prefer:

```bash
uv run --no-project python path/to/script.py
```

This keeps imports pointed at the installed package and compiled extension in `site-packages`. Avoid setting `PYTHONPATH` to the repo root, which can shadow the compiled module.

## WSL notes

- Some Ubuntu images do not provide a `python` alias by default. Use `python3` for system-Python commands when needed.
- On repo paths mounted from Windows, `uv` may warn about failed hardlinks. To suppress that and avoid repeated warnings, set `UV_LINK_MODE=copy` before `uv sync` or `uv pip install`.

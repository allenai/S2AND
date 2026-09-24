# Installation

This document covers the fuller install and setup options for S2AND.

## Requirements

- Python `3.11`, `3.12`, or `3.13`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Git LFS for the Arrow fixtures in a source checkout
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

This installs the latest version available from the configured package index,
not the unreleased `1.0.0` worktree. That worktree targets coordinated
`s2and==1.0.0` and `s2and-rust==1.0.0` packages with model/public-data version
`1.3`; use the checkout flow for pre-release validation.

## Repo checkout

Hydrate the LFS-managed realistic LightGBM parity fixture after cloning and
after switching to a branch that changes it:

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

## Build the Rust extension from source

`uv sync --extra dev` builds `s2and-rust` from `s2and_rust/` and installs it
into the project environment. Syncing `uv run` uses timestamps to detect
ordinary Rust build-input edits. Cache keys and the test freshness digest cover
the root Cargo manifest, lockfile and build script, the vendored CLD2 manifest,
and Rust source files under `src/` and `vendor/cld2/src/`. Editor swap files,
build outputs and the unused vendored lockfile do not change the source digest
or invalidate uv's cached wheel. Cargo still watches source directories to
detect new modules when a native build runs.

uv's file-glob keys track the newest matching timestamp, not file contents or
the complete file inventory. Deleting an older file, adding a backdated file,
or changing content while preserving timestamps can therefore miss a rebuild.
The test guard compares a content digest and stops before collection if the
installed extension's embedded source snapshot differs. Force a refresh with:

```bash
uv sync --extra dev --reinstall-package s2and-rust
```

Notes:

- The test suite imports the installed package and stops before collection when
  its source snapshot differs from the checkout, for example after `uv run --no-sync`
  or a direct interpreter call following a Rust change. Use the forced refresh
  above if an ordinary sync does not rebuild it.
- `uv run --active --no-project maturin develop -m s2and_rust/Cargo.toml` still
  installs an editable build; the next syncing `uv sync` or `uv run` replaces it
  with the locked build.
- Cargo also uses timestamps. After edits that preserve timestamps, or if a
  forced reinstall still reports a mismatch after switching checkouts that
  share `CARGO_TARGET_DIR`, clear the local crates' cached release artifacts with
  `uv run --no-sync cargo clean -p s2and_rust -p cld2 --manifest-path s2and_rust/Cargo.toml --release`,
  then repeat the forced refresh above. The digest records inputs seen by the
  build script; it does not prove that Cargo recompiled every dependency.
- Participating native input paths must be valid UTF-8 and source directories
  must not be symlinks or junctions. Unsupported paths fail with a diagnostic
  instead of producing different Python and Rust digests.
- If you just installed Rust with `rustup` in the current shell, load its environment first if needed.
- Keep the Python and Rust packages from the same checkout. Do not rely on a
  public `s2and-rust` wheel for unreleased source behavior.

## Running repo scripts

When running scripts from a repo checkout, prefer:

```bash
uv run --no-project python path/to/script.py
```

This keeps imports pointed at the installed package and compiled extension in `site-packages`. Avoid setting `PYTHONPATH` to the repo root, which can shadow the compiled module.

## WSL notes

- Some Ubuntu images do not provide a `python` alias by default. Use `python3` for system-Python commands when needed.
- On repo paths mounted from Windows, `uv` may warn about failed hardlinks. To suppress that and avoid repeated warnings, set `UV_LINK_MODE=copy` before `uv sync` or `uv pip install`.

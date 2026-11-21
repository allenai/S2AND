# Local CLD2 patch

## Source provenance

Use the published crates.io `cld2-sys` 1.0.2 archive as the upstream source.

- Archive: <https://static.crates.io/crates/cld2-sys/cld2-sys-1.0.2.crate>
- SHA-256: `bfbb85e425a2c4287874ade9656bd7114f28978e7165dc600ae233467c0b808f`
- Registry checksum: <https://index.crates.io/cl/d2/cld2-sys>, version `1.0.2`.
- Exact source changes: [`S2AND.patch`](S2AND.patch), relative to the extracted crate root.

Keep the generated tables byte-identical to this archive.
Keep the source vendored so builds need no additional CLD2 source download.
Mark generated tables in `.gitattributes` without hiding handwritten detection code or this patch.

## Local changes

Make `prior_lang`, `prior_unreliable`, and `gDisplayPiece` thread-local.
Preserve debug-function references and per-call resets.
Use compiler thread-local storage to preserve the generated tables' C++03 conversion behavior.
Retain the patch's whitespace changes to reproduce the vendored source exactly.

Use the static scoring tables and detection flags of zero through the Rust wrapper.
These settings exclude dynamic-table mutations and debug output buffers.
Keep the Rust version-string lock because the C++ version function writes a shared buffer.

Preserve the upstream source licenses in `cld2/LICENSE` and the crate metadata.
Test serial-versus-parallel language results and complete request partitions before retaining this experiment.

## Audit or reproduce

Download the archive and verify its SHA-256 against the registry checksum before extracting it.
Extract it beneath `scratch/cld2-vendor-audit/` from the repository root.
Apply the patch to that fresh extraction:

```sh
git apply --check --directory=scratch/cld2-vendor-audit/cld2-sys-1.0.2 s2and_rust/vendor/cld2-sys/S2AND.patch
git apply --directory=scratch/cld2-vendor-audit/cld2-sys-1.0.2 s2and_rust/vendor/cld2-sys/S2AND.patch
git diff --no-index -- scratch/cld2-vendor-audit/cld2-sys-1.0.2 s2and_rust/vendor/cld2-sys
```

Expect only these file-list differences after applying the patch:

- Omit upstream `.gitignore` and `Cargo.toml.orig` from the vendored copy.
- Add local `S2AND_PATCHES.md` and `S2AND.patch` to the vendored copy.

Require byte-identical contents for every remaining file, including all generated tables.
Expect the final diff command to exit with status 1 for those metadata differences.

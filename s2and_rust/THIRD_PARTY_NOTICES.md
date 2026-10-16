# Third-party notices

The s2and-rust extension includes Compact Language Detector 2 (CLD2), through
the `cld2` and `cld2-sys` Rust crates.

The original CLD2 library is Copyright 2013 Google Inc. and is distributed under
the Apache License, Version 2.0. The same license covers the derived portions
of the `cld2-sys` wrapper. See `vendor/cld2/LICENSE-APACHE.txt`.

The Rust wrapper's original code is dedicated to the public domain under the
Unlicense. See `vendor/cld2/UNLICENSE.txt`.

The vendored `cld2` Rust wrapper in this distribution includes S2AND changes
to expose the language results used by its featurizer. Its upstream project is
<https://github.com/emk/rust-cld2>; CLD2's upstream project is
<https://github.com/CLD2Owners/cld2>.

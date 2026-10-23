# Third-party notices

The s2and native extension (`s2and._s2and_rust`) includes Compact Language Detector 2 (CLD2), through
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

The s2and wheels for x86_64 Linux (glibc), macOS and Windows include mimalloc,
through the `mimalloc` and `libmimalloc-sys` Rust crates, as their memory
allocator. Its upstream project is <https://github.com/microsoft/mimalloc>.
mimalloc is distributed under the MIT License:

```text
MIT License

Copyright (c) 2018-2025 Microsoft Corporation, Daan Leijen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The vendored cld2-sys 1.0.2 source makes detection debug state thread-local.
See vendor/cld2-sys/S2AND_PATCHES.md for the patch and supported detection settings.

The s2and musllinux wheels bundle the GCC C++ runtime, `libstdc++` and `libgcc_s`,
in `s2and.libs/`, because CLD2 is C++ and musllinux does not provide that runtime.
These libraries are part of GCC <https://gcc.gnu.org/> and are distributed under the
GNU General Public License version 3 with the GCC Runtime Library Exception, which
permits this redistribution.

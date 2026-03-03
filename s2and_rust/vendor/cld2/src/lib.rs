//! **DEPRECATED in favor of [whatlang][],** which is native Rust and
//! smaller. If you have a compelling use-case for this code, please open
//! an issue.
//!
//! [whatlang]: https://crates.io/crates/whatlang
//!
//! Detect the language of a string using the [cld2 library][cld2] from the
//! Chromium project.
//!
//! ```
//! use cld2::{detect_language, Format, Reliable, Lang};
//!
//! let text = "It is an ancient Mariner,
//! And he stoppeth one of three.
//! 'By thy long grey beard and glittering eye,
//! Now wherefore stopp'st thou me?";
//!
//! assert_eq!((Some(Lang("en")), Reliable),
//!            detect_language(text, Format::Text));
//! ```
//!
//! This library wraps the `cld2-sys` library, which provides a raw
//! interface to cld2.  The only major feature which isn't yet wrapped is
//! the `ResultChunk` interface, because it tends to give fairly imprecise
//! answers&mdash;it wouldn't make a very good multi-lingual spellchecker
//! component, for example.  As always, pull requests are eagerly welcome!
//!
//! WARNING: We assume that nobody tries to change the loaded `cld2` data
//! tables or calls the C++ function `CLD2::DetectLanguageVersion` behind
//! our backs.  These configuration and debugging APIs in `cld2` are not
//! thread safe.
//!
//! For more information, see the [GitHub project][github] for this
//! library.
//!
//! [cld2]: https://code.google.com/p/cld2/
//! [github]: https://github.com/emk/rust-cld2

#![deny(missing_docs)]

extern crate cld2_sys as ffi;
#[macro_use]
extern crate lazy_static;
extern crate libc;

pub use types::*;
pub use detection::*;

mod types;
mod language;
mod detection;

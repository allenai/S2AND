//! Data types used in our public API.

use ffi::Encoding;
pub use self::Reliability::{Reliable, Unreliable};

/// Possible data formats.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Format {
    /// Process the text as-is.
    Text, 
    /// Try to stip HTML tags and expand entities.
    Html
}

/// Is the output of the language decoder reliable?
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum Reliability {
    /// The decoder is reasonably confident about this guess.
    Reliable,
    /// The decoder does not have high confidence in this guess.
    Unreliable
}

impl Reliability {
    /// Construct from a boolean value.
    pub fn from_bool(is_reliable: bool) -> Reliability {
        if is_reliable { Reliable } else { Unreliable }
    }
}

/// A language code, normally two letters for common languages.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub struct Lang(pub &'static str);

/// Hints to the decoder, which it will use to make better guesses.
///
/// ```
/// use std::default::Default;
/// use cld2::Hints;
///
/// // Specify just one hint.
/// let hints = Hints{content_language: Some("en"),
///                   .. Default::default()};
/// assert_eq!(Some("en"), hints.content_language);
/// assert_eq!(None, hints.tld);
/// ```
#[derive(Debug, Default)]
pub struct Hints<'a> {
    /// A value from an HTTP Content-Language header.  The value "fr,en"
    /// will bias the decoder towards French and English.
    pub content_language: Option<&'a str>,

    /// The top-level domain associated with this text.  The value "fr"
    /// will bias the decoder towards French.
    pub tld: Option<&'a str>,

    /// EXPERIMENTAL: The original encoding of the text, before it was
    /// converted to UTF-8.  See `Encoding` for legal values.
    pub encoding: Option<Encoding>,

    /// An extra language hint.
    pub language: Option<Lang>
}

/// Detailed information about how well the input text matched a specific
/// language.
#[derive(Clone, Copy)]
pub struct LanguageScore {
    /// The language matched.
    pub language: Option<Lang>,

    /// The percentage of the text which appears to be in this language.
    /// Between 0 and 100.
    pub percent: u8,

    /// Scores near 1.0 indicate a "normal" text for this language.  Scores
    /// further away from 1.0 indicate strange or atypical texts.
    pub normalized_score: f64
}

/// Detailed language detection results.
///
/// Note: Do not rely on this struct containing only the fields listed
/// below.  It may gain extra fields in the future.
#[derive(Clone, Copy)]
pub struct DetectionResult {
    /// The language detected.
    pub language: Option<Lang>,

    /// The scores for the top 3 candidate languages.
    pub scores: [LanguageScore; 3],

    /// The number of bytes of actual text found, excluding tags, etc.
    pub text_bytes: i32,

    /// Is this guess reliable?
    pub reliability: Reliability,

    /// A private field to keep the user from being able to construct
    /// instances directly, so we can extend this struct without breaking
    /// the API.  There's probably a better way to do this.
    _dummy: ()
}

impl DetectionResult {
    /// EXPERIMENTAL: Create a new DetectionResult.  You generally don't
    /// need to call this directly.
    pub fn new(language: Option<Lang>, scores: [LanguageScore; 3],
               text_bytes: i32, reliability: Reliability) -> DetectionResult {
        DetectionResult{language: language, scores: scores,
                        text_bytes: text_bytes, reliability: reliability,
                        _dummy: ()}
    }
}

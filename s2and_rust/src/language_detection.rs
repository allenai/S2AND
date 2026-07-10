use cld2::{
    detect_language_ext as cld2_detect_language_ext, Format as Cld2Format,
    Reliability as Cld2Reliability,
};
use pyo3::prelude::*;
use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

/// Python `str.isalpha` counts only general-category Letter (L*) characters,
/// while Rust's `char::is_alphabetic` is the wider derived `Alphabetic`
/// property (L* plus Nl plus `Other_Alphabetic`, e.g. Indic combining vowel
/// signs). Keep this precheck aligned with Python `s2and.text.detect_language`.
pub(crate) fn is_python_alpha(ch: char) -> bool {
    ch.general_category_group() == GeneralCategoryGroup::Letter
}

pub(crate) fn python_alpha_count(text: &str) -> usize {
    text.chars().filter(|ch| is_python_alpha(*ch)).count()
}

pub(crate) struct LanguageDetectorCompat;

pub(crate) struct LanguageDetectionAudit {
    pub(crate) predicted_language: String,
    pub(crate) is_reliable: bool,
    pub(crate) is_english: bool,
    pub(crate) language_reliability: f64,
}

impl LanguageDetectorCompat {
    pub(crate) fn new(_py: Python<'_>) -> PyResult<Self> {
        Ok(Self)
    }

    pub(crate) fn detect(&self, text: &str) -> PyResult<(bool, bool, String, f64)> {
        let audit = self.audit(text)?;
        Ok((
            audit.is_reliable,
            audit.is_english,
            audit.predicted_language,
            audit.language_reliability,
        ))
    }

    pub(crate) fn audit(&self, text: &str) -> PyResult<LanguageDetectionAudit> {
        if text.split_whitespace().count() <= 1 || python_alpha_count(text) == 0 {
            return Ok(unknown_language_detection());
        }

        let cld2_result = cld2_detect_language_ext(text, Cld2Format::Text, &Default::default());
        let top_score = cld2_result.scores[0];
        let predicted_language = match top_score.language {
            Some(lang) if lang.0 != "un" => lang.0.to_string(),
            _ => return Ok(unknown_language_detection()),
        };

        let is_reliable = cld2_result.reliability == Cld2Reliability::Reliable;
        let language_reliability = if is_reliable {
            top_score.percent as f64 / 100.0
        } else {
            0.0
        };
        Ok(LanguageDetectionAudit {
            is_english: predicted_language == "en",
            predicted_language,
            is_reliable,
            language_reliability,
        })
    }
}

fn unknown_language_detection() -> LanguageDetectionAudit {
    LanguageDetectionAudit {
        predicted_language: "un".to_string(),
        is_reliable: false,
        is_english: false,
        language_reliability: 0.0,
    }
}

#[cfg(test)]
mod alpha_gate_tests {
    use super::{is_python_alpha, python_alpha_count, LanguageDetectorCompat};
    use pyo3::Python;

    #[test]
    fn letter_categories_are_python_alpha() {
        for ch in ['a', 'Z', '\u{00E9}', '\u{6C49}', '\u{02B0}', '\u{01C5}'] {
            assert!(is_python_alpha(ch), "{ch:?}");
        }
    }

    #[test]
    fn other_alphabetic_and_letter_number_are_not_python_alpha() {
        // U+093F DEVANAGARI VOWEL SIGN I (Mn, Other_Alphabetic) and
        // U+2160 ROMAN NUMERAL ONE (Nl) are `Alphabetic` in Rust but
        // isalpha() == False in Python.
        for ch in ['\u{093F}', '\u{2160}'] {
            assert!(ch.is_alphabetic(), "{ch:?} lost its Alphabetic property");
            assert!(!is_python_alpha(ch), "{ch:?}");
        }
    }

    #[test]
    fn text_of_only_combining_marks_counts_zero_alpha() {
        assert_eq!(python_alpha_count("\u{093F}\u{0941} \u{093F}"), 0);
    }

    #[test]
    fn markup_title_uses_plain_text_cld2_mode_matching_python() {
        #[cfg(windows)]
        if let Some(python_home) = option_env!("S2AND_RUST_PYTHONHOME") {
            std::env::set_var("PYTHONHOME", python_home);
        }
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let detector = LanguageDetectorCompat::new(py).expect("detector");
            let result = detector
                .audit(
                    "<div>This is a detailed English research title about neural systems and scientific evaluation.</div>",
                )
                .expect("detection");
            assert_eq!(result.predicted_language, "en");
            assert!(result.is_reliable);
            assert!(result.is_english);
            assert_eq!(result.language_reliability, 0.98);
        });
    }
}

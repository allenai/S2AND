use cld2::{
    detect_language_ext as cld2_detect_language_ext, Format as Cld2Format,
    Reliability as Cld2Reliability,
};
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

fn is_pycld2_rejected_control(ch: char) -> bool {
    matches!(
        ch,
        '\u{0000}'..='\u{0008}'
            | '\u{000B}'
            | '\u{000E}'..='\u{001F}'
            | '\u{007F}'..='\u{009F}'
    )
}

pub(crate) struct LanguageDetectionAudit {
    pub(crate) predicted_language: String,
    pub(crate) language_reliability: f64,
}

pub(crate) fn detect_language_compat(text: &str) -> LanguageDetectionAudit {
    // pycld2 rejects these C0/C1 controls while Rust cld2 accepts them.
    // Python catches that binding error and returns "un"/0.0, so reject the
    // same inputs before applying Rust's word and alphabetic gates.
    if text.chars().any(is_pycld2_rejected_control) {
        return unknown_language_detection();
    }

    if text.split_whitespace().count() <= 1 || python_alpha_count(text) == 0 {
        return unknown_language_detection();
    }

    let cld2_result = cld2_detect_language_ext(text, Cld2Format::Text, &Default::default());
    let top_score = cld2_result.scores[0];
    let predicted_language = match top_score.language {
        Some(lang) if lang.0 != "un" => lang.0.to_string(),
        _ => return unknown_language_detection(),
    };

    let is_reliable = cld2_result.reliability == Cld2Reliability::Reliable;
    let language_reliability = if is_reliable {
        top_score.percent as f64 / 100.0
    } else {
        0.0
    };
    LanguageDetectionAudit {
        predicted_language,
        language_reliability,
    }
}

fn unknown_language_detection() -> LanguageDetectionAudit {
    LanguageDetectionAudit {
        predicted_language: "un".to_string(),
        language_reliability: 0.0,
    }
}

#[cfg(test)]
mod alpha_gate_tests {
    use super::{
        detect_language_compat, is_pycld2_rejected_control, is_python_alpha, python_alpha_count,
    };

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
    fn pycld2_rejected_control_set_is_exact() {
        let observed = (0..=0x009F)
            .filter_map(char::from_u32)
            .filter(|ch| is_pycld2_rejected_control(*ch))
            .collect::<Vec<_>>();
        let expected = (0..=0x0008)
            .chain([0x000B])
            .chain(0x000E..=0x001F)
            .chain(0x007F..=0x009F)
            .filter_map(char::from_u32)
            .collect::<Vec<_>>();

        assert_eq!(observed, expected);
        assert_eq!(observed.len(), 61);
    }

    #[test]
    fn pycld2_accepted_ascii_whitespace_is_not_rejected() {
        for ch in ['\t', '\n', '\u{000C}', '\r'] {
            assert!(!is_pycld2_rejected_control(ch), "{ch:?}");
        }
    }

    #[test]
    fn pycld2_rejected_controls_return_unknown() {
        for ch in ['\u{0000}', '\u{001C}', '\u{007F}', '\u{0080}', '\u{009F}'] {
            let result = detect_language_compat(&format!(
                "This is a detailed English research title {ch} about neural systems and scientific evaluation."
            ));
            assert_eq!(result.predicted_language, "un", "{ch:?}");
            assert_eq!(result.language_reliability, 0.0, "{ch:?}");
        }
    }

    #[test]
    fn markup_title_uses_plain_text_cld2_mode_matching_python() {
        let result = detect_language_compat(
            "<div>This is a detailed English research title about neural systems and scientific evaluation.</div>",
        );
        assert_eq!(result.predicted_language, "en");
        assert_eq!(result.language_reliability, 0.98);
    }
}

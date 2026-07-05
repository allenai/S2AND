use cld2::{detect_language_ext as cld2_detect_language_ext, Format as Cld2Format};
use fasttext::FastText;
use pyo3::prelude::*;

fn parse_fasttext_label(label: &str) -> String {
    label.rsplit("__").next().unwrap_or(label).to_string()
}

/// Reconcile the fastText and cld2 predictions into a final call. Mirrors the
/// Python `s2and.text.reconcile_detected_languages`: a prediction is reliable
/// only when BOTH detectors return a concrete language (neither the
/// `"un_ft"`/`"un_2"` unknown sentinel) AND they agree. Any non-agreement --
/// disagreement, or only one detector responding -- collapses to
/// `("un", false)`, preserving the invariant `is_reliable <=> language != "un"`.
pub(crate) fn reconcile_detected_languages(
    predicted_language_ft: &str,
    predicted_language_2: &str,
) -> (String, bool) {
    let ft_known = predicted_language_ft != "un_ft";
    let cld2_known = predicted_language_2 != "un_2";
    if ft_known && cld2_known && predicted_language_ft == predicted_language_2 {
        (predicted_language_2.to_string(), true)
    } else {
        ("un".to_string(), false)
    }
}

fn resolve_fasttext_model_path(py: Python<'_>) -> PyResult<String> {
    let consts = py.import("s2and.consts")?;
    let fasttext_path = consts.getattr("FASTTEXT_PATH")?;
    let file_cache = py.import("s2and.file_cache")?;
    let cached_path = file_cache.getattr("cached_path")?;
    cached_path.call1((&fasttext_path,))?.extract()
}

fn python_fasttext_loading_enabled(py: Python<'_>) -> bool {
    if let Ok(value) = std::env::var("S2AND_SKIP_FASTTEXT") {
        if matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes"
        ) {
            return false;
        }
    }
    py.import("s2and.text")
        .and_then(|text_module| text_module.getattr("fasttext_loading_enabled"))
        .and_then(|enabled_fn| enabled_fn.call0())
        .and_then(|enabled| enabled.extract::<bool>())
        .unwrap_or(true)
}

pub(crate) struct LanguageDetectorCompat {
    fasttext: Option<FastText>,
}

pub(crate) struct LanguageDetectionAudit {
    pub(crate) predicted_language: String,
    pub(crate) is_reliable: bool,
    pub(crate) is_english: bool,
}

impl LanguageDetectorCompat {
    pub(crate) fn new(py: Python<'_>) -> PyResult<Self> {
        if !python_fasttext_loading_enabled(py) {
            return Ok(Self { fasttext: None });
        }
        let model_path = resolve_fasttext_model_path(py)?;
        let mut model = FastText::new();
        model.load_model(&model_path).map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "s2and_rust: failed to load fastText language model at '{model_path}' ({err})"
            ))
        })?;
        Ok(Self {
            fasttext: Some(model),
        })
    }

    pub(crate) fn detect(&self, text: &str) -> PyResult<(bool, bool, String)> {
        let audit = self.audit(text)?;
        Ok((
            audit.is_reliable,
            audit.is_english,
            audit.predicted_language,
        ))
    }

    pub(crate) fn audit(&self, text: &str) -> PyResult<LanguageDetectionAudit> {
        if text.split_whitespace().count() <= 1 {
            return Ok(LanguageDetectionAudit {
                predicted_language: "un".to_string(),
                is_reliable: false,
                is_english: false,
            });
        }

        let mut alpha_count = 0usize;
        let mut uppercase_count = 0usize;
        for ch in text.chars() {
            if ch.is_alphabetic() {
                alpha_count += 1;
                if ch.is_uppercase() {
                    uppercase_count += 1;
                }
            }
        }
        if alpha_count == 0 {
            return Ok(LanguageDetectionAudit {
                predicted_language: "un".to_string(),
                is_reliable: false,
                is_english: false,
            });
        }

        let predicted_language_ft = if let Some(fasttext_model) = &self.fasttext {
            let uppercase_ratio = uppercase_count as f64 / alpha_count as f64;
            let mut fasttext_input = text.replace('\n', " ");
            if uppercase_ratio > 0.9 {
                fasttext_input = fasttext_input.to_lowercase();
            }
            match fasttext_model.predict(&fasttext_input, 1, 0.0) {
                Ok(predictions) => predictions
                    .first()
                    .map(|prediction| parse_fasttext_label(&prediction.label))
                    .unwrap_or_else(|| "un_ft".to_string()),
                Err(err) => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "s2and_rust: fastText language prediction failed ({err})"
                    )));
                }
            }
        } else {
            "un_ft".to_string()
        };

        let cld2_result = cld2_detect_language_ext(text, Cld2Format::Text, &Default::default());
        let mut predicted_language_2 = match cld2_result.scores[0].language {
            Some(lang) => lang.0.to_string(),
            None => "un_2".to_string(),
        };
        if predicted_language_2 == "un" {
            predicted_language_2 = "un_2".to_string();
        }

        let (predicted_language, is_reliable) =
            reconcile_detected_languages(&predicted_language_ft, &predicted_language_2);

        let is_english = predicted_language == "en";
        Ok(LanguageDetectionAudit {
            predicted_language,
            is_reliable,
            is_english,
        })
    }
}

#[cfg(test)]
mod reconcile_tests {
    use super::reconcile_detected_languages;

    // Mirrors the Python reconcile_detected_languages tests in tests/test_text.py
    // so the two implementations are pinned to the same truth table.
    #[test]
    fn agreement_is_reliable() {
        assert_eq!(
            reconcile_detected_languages("en", "en"),
            ("en".to_string(), true)
        );
        assert_eq!(
            reconcile_detected_languages("fr", "fr"),
            ("fr".to_string(), true)
        );
    }

    #[test]
    fn disagreement_collapses_to_unknown() {
        assert_eq!(
            reconcile_detected_languages("en", "fr"),
            ("un".to_string(), false)
        );
    }

    #[test]
    fn single_detector_is_not_reliable() {
        // Only cld2 responded (fastText unknown, e.g. disabled).
        assert_eq!(
            reconcile_detected_languages("un_ft", "en"),
            ("un".to_string(), false)
        );
        // Only fastText responded (cld2 unknown/failed).
        assert_eq!(
            reconcile_detected_languages("en", "un_2"),
            ("un".to_string(), false)
        );
    }

    #[test]
    fn both_unknown_is_not_reliable() {
        assert_eq!(
            reconcile_detected_languages("un_ft", "un_2"),
            ("un".to_string(), false)
        );
    }
}

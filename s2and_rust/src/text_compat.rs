use crate::{HashMap, HashSet};
use pyo3::prelude::PyResult;
use unicode_normalization::UnicodeNormalization;
use unicode_properties::{GeneralCategory, GeneralCategoryGroup, UnicodeGeneralCategory};

pub(crate) fn ensure_unidecode_for_text(
    text: &str,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    if text.is_ascii() {
        return Ok(());
    }
    for ch in text.chars() {
        if ch.is_ascii() || unidecode_char_map.contains_key(&ch) {
            continue;
        }
        unidecode_char_map.insert(ch, text_unidecode_char(ch).to_string());
    }
    Ok(())
}

pub(crate) fn text_unidecode_char(ch: char) -> &'static str {
    match ch as u32 {
        0x0080 | 0x0082 | 0x0083 | 0x0084 | 0x0085 | 0x0086 | 0x0087 | 0x0088 | 0x0089 | 0x008A
        | 0x008B | 0x008C | 0x008E | 0x0091 | 0x0092 | 0x0093 | 0x0094 | 0x0095 | 0x0096
        | 0x0097 | 0x0098 | 0x0099 | 0x009A | 0x009B | 0x009C | 0x009E | 0x009F | 0x02E5
        | 0x02E6 | 0x02E7 | 0x02E8 | 0x02E9 | 0x02EA | 0x02EB | 0xFDF0 | 0xFDF1 | 0xFDF2
        | 0xFDF3 | 0xFDF4 | 0xFDF5 | 0xFDF6 | 0xFDF7 | 0xFDF8 | 0xFDF9 | 0xFDFA | 0xFDFB => "",
        0x02FF | 0x03FF | 0x04FF | 0x05FF | 0x06FF | 0x07FF | 0x09FF | 0x0AFF | 0x0BFF | 0x0CFF
        | 0x0DFF | 0x0EFF | 0x0FFF | 0x10FF | 0x11FF | 0x13FF | 0x16FF | 0x17FF | 0x18FF
        | 0x1EFF | 0x1FFF | 0x20FF | 0x21FF | 0x22FF | 0x23FF | 0x24FF | 0x25FF | 0x26FF
        | 0x27FF | 0x2EFF | 0x2FFF | 0x30FF | 0x31FF | 0x32FF | 0x33FF | 0x4DFF | 0x9FFF
        | 0xA4FF | 0xD7FF | 0xFAFF | 0xFDFF => "[?] ",
        0x25F4 | 0x25F5 | 0x25F6 | 0x25F7 => "#",
        0x02EF | 0x02F0 | 0x02F1 | 0x02F2 | 0x02F3 | 0x02F4 | 0x02F5 | 0x02F6 | 0x02F7 | 0x02F8
        | 0x02F9 | 0x02FA | 0x02FB | 0x02FC | 0x02FD | 0x02FE | 0x03F4 | 0x03F5 | 0x03F6
        | 0x03F7 | 0x03F8 | 0x03F9 | 0x03FC | 0x03FD | 0x03FE | 0x0AF0 | 0x0AF1 | 0x0AF9
        | 0x13F5 | 0x13F8 | 0x13F9 | 0x13FA | 0x13FB | 0x13FC | 0x13FD | 0x1EFA | 0x1EFB
        | 0x1EFC | 0x1EFD | 0x1EFE | 0x25F8 | 0x25F9 | 0x25FA | 0x25FB | 0x25FC | 0x25FD
        | 0x25FE | 0xFDFC | 0xFDFD => "[?]",
        _ => unidecode::unidecode_char(ch),
    }
}

pub(crate) fn normalize_ascii_text_compat(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut prev_space = true;
    for byte in text.bytes() {
        let lowered = byte.to_ascii_lowercase();
        if lowered.is_ascii_alphabetic() {
            normalized.push(lowered as char);
            prev_space = false;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

pub(crate) fn normalize_ascii_title_compat(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut prev_space = true;
    for byte in text.bytes() {
        let lowered = byte.to_ascii_lowercase();
        if lowered.is_ascii_alphanumeric() {
            normalized.push(lowered as char);
            prev_space = false;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

#[cfg(test)]
pub(crate) fn normalize_text_compat_native(text: &str) -> String {
    normalize_text_compat_with_map(text, None)
}

pub(crate) fn normalize_text_compat_from_map(
    text: &str,
    unidecode_char_map: &HashMap<char, String>,
) -> String {
    normalize_text_compat_with_map(text, Some(unidecode_char_map))
}

pub(crate) fn normalize_title_compat_from_map(
    text: &str,
    unidecode_char_map: &HashMap<char, String>,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    if text.is_ascii() {
        return normalize_ascii_title_compat(text);
    }

    let mut transliterated = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            transliterated.push(ch.to_ascii_lowercase());
            continue;
        }
        let mapped = unidecode_char_map
            .get(&ch)
            .map(String::as_str)
            .unwrap_or_else(|| text_unidecode_char(ch));
        for mapped_ch in mapped.chars() {
            transliterated.push(mapped_ch.to_ascii_lowercase());
        }
    }
    normalize_ascii_title_compat(&transliterated)
}

fn normalize_text_compat_with_map(
    text: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    if text.is_ascii() {
        return normalize_ascii_text_compat(text);
    }

    let mut transliterated = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            transliterated.push(ch.to_ascii_lowercase());
            continue;
        }
        let mapped = unidecode_char_map
            .and_then(|char_map| char_map.get(&ch).map(String::as_str))
            .unwrap_or_else(|| text_unidecode_char(ch));
        for mapped_ch in mapped.chars() {
            transliterated.push(mapped_ch.to_ascii_lowercase());
        }
    }

    let mut normalized = String::with_capacity(transliterated.len());
    let mut prev_space = true;
    for ch in transliterated.chars() {
        if ch.is_ascii_alphabetic() {
            normalized.push(ch);
            prev_space = false;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

fn is_name_dash(ch: char) -> bool {
    matches!(
        ch,
        '-' | '\u{2010}'
            | '\u{2011}'
            | '\u{2012}'
            | '\u{2013}'
            | '\u{2014}'
            | '\u{2212}'
            | '\u{FE58}'
            | '\u{FE63}'
            | '\u{FF0D}'
    )
}

// D3 apostrophe-like marks (mirrors Python `NAME_APOSTROPHE_LIKE_CHARS` in
// s2and/text/canonical_names.py): ASCII apostrophe, backtick, spacing acute, curly quotes,
// modifier letters (okina/apostrophe), primes, saltillo, U+FE4D (classified with
// apostrophe-like marks by issue #39 despite its Unicode name), fullwidth
// apostrophe.
fn is_name_apostrophe_like(ch: char) -> bool {
    matches!(
        ch,
        '\'' | '`'
            | '\u{00B4}'
            | '\u{2018}'
            | '\u{2019}'
            | '\u{02BB}'
            | '\u{02BC}'
            | '\u{2032}'
            | '\u{2035}'
            | '\u{A78C}'
            | '\u{FE4D}'
            | '\u{FF07}'
    )
}

// Invisible formatting controls deleted before tokenization (mirrors Python
// `_NAME_INVISIBLE_FORMAT_CHARS`): soft hyphen (not a dash separator) and
// zero-width joiner.
fn is_name_invisible_format(ch: char) -> bool {
    matches!(ch, '\u{00AD}' | '\u{200D}')
}

/// Match Python `str.isspace()` where it is broader than Rust's Unicode
/// `char::is_whitespace()`.
pub(crate) fn is_python_whitespace_compat(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '\u{001C}' | '\u{001D}' | '\u{001E}' | '\u{001F}')
}

// D10 (mirrors Python `_GEORGIAN_ROMANIZATION`): scholar's Georgian table for
// Mkhedruli U+10D0..=U+10F6, so names such as ბერიძე give "beridze" in both
// systems. Mtavruli capitals (U+1C90..=U+1CB6) give the same letters in
// uppercase, which keeps the D9 case guard.
#[rustfmt::skip]
const GEORGIAN_ROMANIZATION: [&str; 39] = [
    "a", "b", "g", "d", "e", "v", "z", "t", "i", "k", "l", "m", "n", "o", "p", "zh",
    "r", "s", "t", "u", "p", "k", "gh", "q", "sh", "ch", "ts", "dz", "ts", "ch", "kh",
    "j", "h", "e", "y", "v", "kh", "o", "f",
];
const GEORGIAN_MKHEDRULI_START: u32 = 0x10D0;
const GEORGIAN_MTAVRULI_START: u32 = 0x1C90;
// D10 (mirrors Python `_ARMENIAN_ROMANIZATION`): scholar's Eastern Armenian
// table for the small letters U+0561..=U+0586. `romanize_armenian` holds the
// context rules.
#[rustfmt::skip]
const ARMENIAN_ROMANIZATION: [&str; 38] = [
    "a", "b", "g", "d", "e", "z", "e", "y", "t", "zh", "i", "l", "kh", "ts", "k", "h",
    "dz", "gh", "ch", "m", "y", "n", "sh", "o", "ch", "p", "j", "r", "s", "v", "t",
    "r", "ts", "v", "p", "k", "o", "f",
];
const ARMENIAN_SMALL_START: u32 = 0x0561;
// Armenian capitals sit 0x30 below their small letters.
const ARMENIAN_CAPITAL_START: u32 = 0x0531;
// The ligature և is ե plus ւ under NFKC, which scholar applies first.
const ARMENIAN_LIGATURE_ECH_YIWN: char = '\u{0587}';
const ARMENIAN_ECH: char = '\u{0565}';
const ARMENIAN_VO: char = '\u{0578}';
const ARMENIAN_VEW: char = '\u{057E}';
const ARMENIAN_YIWN: char = '\u{0582}';

/// Return the D10 Georgian romanization of `ch` and whether it is a capital.
fn georgian_romanization(ch: char) -> Option<(&'static str, bool)> {
    let code = ch as u32;
    if let Some(roman) = code
        .checked_sub(GEORGIAN_MKHEDRULI_START)
        .and_then(|index| GEORGIAN_ROMANIZATION.get(index as usize))
    {
        return Some((roman, false));
    }
    code.checked_sub(GEORGIAN_MTAVRULI_START)
        .and_then(|index| GEORGIAN_ROMANIZATION.get(index as usize))
        .map(|roman| (*roman, true))
}

/// Return the small form of an Armenian letter in the D10 table, or `None`.
fn armenian_small_letter(ch: char) -> Option<char> {
    let code = ch as u32;
    let small = if (ARMENIAN_CAPITAL_START..ARMENIAN_CAPITAL_START + 38).contains(&code) {
        code + (ARMENIAN_SMALL_START - ARMENIAN_CAPITAL_START)
    } else {
        code
    };
    (ARMENIAN_SMALL_START..ARMENIAN_SMALL_START + 38)
        .contains(&small)
        .then(|| char::from_u32(small).expect("Armenian small letter"))
}

/// Match Python `_RE_ARMENIAN_LETTER`: the capitals, small letters, and ligatures.
fn is_armenian_letter_or_ligature(ch: char) -> bool {
    matches!(ch, '\u{0531}'..='\u{0556}' | '\u{0561}'..='\u{0587}' | '\u{FB13}'..='\u{FB17}')
}

/// The two letters of an Armenian ligature, as NFKC splits it (mirrors Python
/// `_ARMENIAN_LIGATURES`). Scholar applies NFKC first.
fn armenian_ligature_letters(ch: char) -> Option<[char; 2]> {
    match ch {
        ARMENIAN_LIGATURE_ECH_YIWN => Some([ARMENIAN_ECH, ARMENIAN_YIWN]),
        '\u{FB13}' => Some(['\u{0574}', '\u{0576}']),
        '\u{FB14}' => Some(['\u{0574}', '\u{0565}']),
        '\u{FB15}' => Some(['\u{0574}', '\u{056B}']),
        '\u{FB16}' => Some(['\u{057E}', '\u{0576}']),
        '\u{FB17}' => Some(['\u{0574}', '\u{056D}']),
        _ => None,
    }
}

/// Scholar deletes these characters before it reads an Armenian letter's
/// neighbors: punctuation, nonspacing marks, and format, control,
/// private-use, and surrogate characters (mirrors Python
/// `_ARMENIAN_CONTEXT_SKIPPED_CATEGORIES`).
fn skipped_in_armenian_context(ch: char) -> bool {
    matches!(
        ch.general_category(),
        GeneralCategory::NonspacingMark
            | GeneralCategory::Format
            | GeneralCategory::Control
            | GeneralCategory::PrivateUse
            | GeneralCategory::Surrogate
    ) || ch.general_category_group() == GeneralCategoryGroup::Punctuation
}

/// Romanize the Armenian letters of `text` with scholar's table (D10; mirrors
/// Python `_romanize_armenian`). ու is "u". A word-initial ե is "ye", and a
/// word-initial ո is "vo" unless վ follows. As in scholar, the neighbors of a
/// letter skip punctuation, nonspacing marks, and format and control
/// characters, and a word starts after a character that is not a letter
/// (Unicode category L). A capital gives a capitalized romanization.
fn romanize_armenian(text: &str) -> String {
    let mut chars: Vec<char> = Vec::with_capacity(text.len());
    for ch in text.chars() {
        match armenian_ligature_letters(ch) {
            Some(letters) => chars.extend(letters),
            None => chars.push(ch),
        }
    }
    let folded: Vec<char> = chars
        .iter()
        .map(|&ch| armenian_small_letter(ch).unwrap_or(ch))
        .collect();
    // The nearest kept neighbor on each side; `None` at either end of the text.
    let kept: Vec<bool> = chars
        .iter()
        .map(|&ch| !skipped_in_armenian_context(ch))
        .collect();
    let mut previous_kept: Vec<Option<char>> = vec![None; chars.len()];
    let mut last = None;
    for index in 0..chars.len() {
        previous_kept[index] = last;
        if kept[index] {
            last = Some(folded[index]);
        }
    }
    let mut following_kept: Vec<Option<char>> = vec![None; chars.len()];
    last = None;
    for index in (0..chars.len()).rev() {
        following_kept[index] = last;
        if kept[index] {
            last = Some(folded[index]);
        }
    }
    let mut out = String::with_capacity(text.len());
    for (index, &ch) in chars.iter().enumerate() {
        let Some(letter) = armenian_small_letter(ch) else {
            out.push(ch);
            continue;
        };
        let previous = previous_kept[index];
        let following = following_kept[index];
        let word_start = previous
            .is_none_or(|before| before.general_category_group() != GeneralCategoryGroup::Letter);
        let roman = if letter == ARMENIAN_VO && following == Some(ARMENIAN_YIWN) {
            "u"
        } else if letter == ARMENIAN_YIWN && previous == Some(ARMENIAN_VO) {
            ""
        } else if letter == ARMENIAN_ECH && word_start {
            "ye"
        } else if letter == ARMENIAN_VO && word_start && following != Some(ARMENIAN_VEW) {
            "vo"
        } else {
            ARMENIAN_ROMANIZATION[(letter as u32 - ARMENIAN_SMALL_START) as usize]
        };
        let mut bytes = roman.bytes();
        if ch != letter {
            if let Some(head) = bytes.next() {
                out.push(head.to_ascii_uppercase() as char);
            }
        }
        out.extend(bytes.map(char::from));
    }
    out
}

/// canonical_v2 pre-translation on raw code points, before transliteration
/// (mirrors Python `_canonical_name_pretranslate`): delete invisible format
/// controls, unify apostrophe-like marks to ASCII apostrophe, unify dash-like
/// characters to ASCII hyphen, and romanize Georgian and Armenian (D10).
fn canonical_name_pretranslate(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut has_armenian = false;
    for ch in raw.chars() {
        if ch.is_ascii() {
            // Most characters are ASCII; of those only the backtick changes.
            out.push(if ch == '`' { '\'' } else { ch });
            continue;
        }
        if is_name_invisible_format(ch) {
            continue;
        }
        if is_name_apostrophe_like(ch) {
            out.push('\'');
        } else if is_name_dash(ch) {
            out.push('-');
        } else if let Some((roman, capital)) = georgian_romanization(ch) {
            if capital {
                out.extend(roman.bytes().map(|byte| byte.to_ascii_uppercase() as char));
            } else {
                out.push_str(roman);
            }
        } else {
            has_armenian |= is_armenian_letter_or_ligature(ch);
            out.push(ch);
        }
    }
    if has_armenian {
        return romanize_armenian(&out);
    }
    out
}

/// canonical_v2 token normalization of a pre-translated string (mirrors Python
/// `_canonical_name_tokens`): transliterate, lowercase, delete apostrophes and
/// backticks post-transliteration (no token boundary), and split on everything
/// that is not an ASCII letter. Dash binding is decided by the caller before
/// this runs.
fn canonical_name_tokens(
    pretranslated: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    let consume = |mapped_ch: char, current: &mut String, tokens: &mut Vec<String>| {
        let lowered = mapped_ch.to_ascii_lowercase();
        if lowered.is_ascii_lowercase() {
            current.push(lowered);
        } else if lowered == '\'' || lowered == '`' {
            // Deleted, not a separator: O'Brien -> obrien.
        } else if !current.is_empty() {
            tokens.push(std::mem::take(current));
        }
    };
    for ch in pretranslated.chars() {
        if ch.is_ascii() {
            consume(ch, &mut current, &mut tokens);
            continue;
        }
        let mapped = unidecode_char_map
            .and_then(|char_map| char_map.get(&ch).map(String::as_str))
            .unwrap_or_else(|| text_unidecode_char(ch));
        for mapped_ch in mapped.chars() {
            consume(mapped_ch, &mut current, &mut tokens);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// Canonicalize a whole name string to spaced canonical_v2 tokens (mirrors
/// Python `canonicalize_name_text`). This is the normalization for canonical
/// middle and last fields and for whole-string artifact keys.
pub(crate) fn canonicalize_name_text_compat(
    raw: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> String {
    canonical_name_tokens(&canonical_name_pretranslate(raw), unidecode_char_map).join(" ")
}

// Han, kana and hangul code points (mirrors Python `_CJK_NAME_RANGES`).
// Transliteration writes each Han character as one pinyin syllable, so the
// syllables of two different given names often match by prefix ("zhong xi"
// and "zhong xiang"). The name rule compares such given names whole.
const HAN_NAME_RANGES: [(u32, u32); 4] = [
    (0x3400, 0x4DBF),   // CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   // CJK Unified Ideographs
    (0xF900, 0xFAFF),   // CJK Compatibility Ideographs
    (0x20000, 0x323AF), // CJK Unified Ideographs Extensions B-H and Compatibility Supplement
];
const OTHER_CJK_NAME_RANGES: [(u32, u32); 8] = [
    (0x1100, 0x11FF), // Hangul Jamo
    (0x3005, 0x3007), // Ideographic iteration mark, closing mark and number zero
    (0x3040, 0x30FF), // Hiragana and Katakana
    (0x3130, 0x318F), // Hangul Compatibility Jamo
    (0x31F0, 0x31FF), // Katakana Phonetic Extensions
    (0xA960, 0xA97F), // Hangul Jamo Extended-A
    (0xAC00, 0xD7FF), // Hangul Syllables and Hangul Jamo Extended-B
    (0xFF66, 0xFFDC), // Halfwidth Katakana and Halfwidth Hangul
];

// Script tags of a whole CJK given name (mirrors Python `_cjk_script_tag`). Pinyin of a
// Han name and the romanization of its hangul or kana form differ, so only names in one
// script are compared whole.
const HANGUL_NAME_RANGES: [(u32, u32); 5] = [
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7FF),
    (0xFFA0, 0xFFDC),
];
const KANA_NAME_RANGES: [(u32, u32); 3] = [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)];

/// Write halfwidth katakana and spacing voicing marks as their composable forms (mirrors
/// Python `_widen_kana`). NFC alone keeps them apart from the fullwidth and voiced kana.
fn widen_kana(ch: char) -> Vec<char> {
    match ch {
        '\u{ff61}'..='\u{ff9f}' => std::iter::once(ch).nfkc().collect(),
        '\u{309b}' => vec!['\u{3099}'],
        '\u{309c}' => vec!['\u{309a}'],
        _ => vec![ch],
    }
}

fn cjk_script_tag(given: &str) -> char {
    let in_ranges = |ranges: &[(u32, u32)]| {
        given.chars().any(|ch| {
            ranges
                .iter()
                .any(|&(start, end)| (start..=end).contains(&(ch as u32)))
        })
    };
    if in_ranges(&HANGUL_NAME_RANGES) {
        'G'
    } else if in_ranges(&KANA_NAME_RANGES) {
        'K'
    } else {
        'H'
    }
}

fn is_cjk_name_character(ch: char) -> bool {
    let code = ch as u32;
    HAN_NAME_RANGES
        .iter()
        .chain(&OTHER_CJK_NAME_RANGES)
        .any(|&(start, end)| start <= code && code <= end)
}

/// Whether a raw given name is written in Han, kana or hangul only (mirrors
/// Python `is_cjk_given_name`). The given name is the raw first and middle
/// fields together. It needs one CJK character, and its other characters must
/// transliterate to no letter.
fn is_cjk_given_name(
    first_raw: &str,
    middle_raw: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> bool {
    let given = [first_raw, middle_raw];
    if given.iter().all(|part| part.is_ascii())
        || !given
            .iter()
            .any(|part| part.chars().any(is_cjk_name_character))
    {
        return false;
    }
    let other: String = given
        .iter()
        .flat_map(|part| part.chars().chain(std::iter::once(' ')))
        .map(|ch| if is_cjk_name_character(ch) { ' ' } else { ch })
        .collect();
    canonical_name_tokens(&canonical_name_pretranslate(&other), unidecode_char_map).is_empty()
}

/// Prepare the whole CJK given name for constraints: a script tag, then the joined
/// letters. An empty string is incomplete evidence. None marks a name that is not CJK.
/// Constraints compare two values whole only when both are complete and share a tag.
pub(crate) fn cjk_given_name_for_constraint(
    first_raw: &str,
    middle_raw: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> Option<String> {
    if !is_cjk_given_name(first_raw, middle_raw, unidecode_char_map) {
        return None;
    }

    let raw = canonical_name_pretranslate(&format!("{first_raw}{middle_raw}"));
    let mut previous = None;
    let mut given = String::from(cjk_script_tag(&raw));
    for mut ch in raw
        .chars()
        .filter(|&ch| !is_python_whitespace_compat(ch))
        .flat_map(widen_kana)
        .nfc()
    {
        if ch == '々' {
            let Some(ideograph) = previous.filter(|&previous| {
                HAN_NAME_RANGES
                    .iter()
                    .any(|&(start, end)| (start..=end).contains(&(previous as u32)))
            }) else {
                return Some(String::new());
            };
            ch = ideograph;
        }
        let mapped = unidecode_char_map
            .and_then(|char_map| char_map.get(&ch).map(String::as_str))
            .unwrap_or_else(|| text_unidecode_char(ch));
        let start = given.len();
        given.extend(
            mapped
                .chars()
                .filter(char::is_ascii_alphabetic)
                .map(|letter| letter.to_ascii_lowercase()),
        );
        // A partial transliteration can make equivalent names look different. Any non-ASCII
        // character without letters marks the name incomplete, so the ordinary rules apply.
        // Unicode categories are not used: Python and Rust ship different Unicode versions.
        if given.len() == start && !ch.is_ascii() {
            return Some(String::new());
        }
        previous = Some(ch);
    }
    Some(given)
}

/// Joined-initials reading of a raw given name for constraints (mirrors Python
/// `joined_initials_for_constraint`): "Ka" with no middle name reads "Jka" (K. A.), and
/// "MArfan" reads "Wm arfan" (M. Arfan). It only matters when another name spells exactly
/// these initials; see `constraints::joined_initials_match`.
pub(crate) fn joined_initials_for_constraint(first_raw: &str, middle_raw: &str) -> Option<String> {
    let first = first_raw.trim_matches(is_python_whitespace_compat);
    if !middle_raw
        .trim_matches(is_python_whitespace_compat)
        .is_empty()
    {
        return None;
    }
    let bytes = first.as_bytes();
    let all = |range: &[u8], test: fn(&u8) -> bool| range.iter().all(test);
    if (2..=3).contains(&bytes.len())
        && bytes[0].is_ascii_uppercase()
        && all(&bytes[1..], u8::is_ascii_lowercase)
    {
        let lowered = first.to_ascii_lowercase();
        // ED and MD stay given names, as in the canonical initials parser.
        if MIXED_CASE_NON_INITIALS.contains(&lowered.as_str())
            || crate::name_normalization::NAME_PREFIXES.contains(&lowered.as_str())
            || JOINED_INITIALS_GIVEN_NAMES.contains(&first.to_ascii_uppercase().as_str())
        {
            return None;
        }
        return Some(format!("J{lowered}"));
    }
    if bytes.len() >= 4
        && bytes[0].is_ascii_uppercase()
        && bytes[1].is_ascii_uppercase()
        && all(&bytes[2..], u8::is_ascii_lowercase)
    {
        let lowered = first.to_ascii_lowercase();
        return Some(format!("W{} {}", &lowered[..1], &lowered[1..]));
    }
    None
}

/// "S" plus the middle initials of a name whose middle is joined capitals (mirrors Python
/// `spelled_middle_initials_for_constraint`): "Johanna IP" gives "Sip". Canonical names keep
/// "IP" as one token. Only the last word of the middle counts, the token that
/// `spell_middle_tokens` splits, so "A IP" gives "Sip" and "IP A" gives None.
/// The first name must pass the D9 letter-case test that last names use,
/// since an all-capitals name carries no case; it transliterates first, so "Лев IP" qualifies.
pub(crate) fn spelled_middle_initials_for_constraint(
    first_raw: &str,
    middle_raw: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> Option<String> {
    let mut first = first_raw.to_string();
    let mut middle = middle_raw
        .trim_matches(is_python_whitespace_compat)
        .to_string();
    if middle.is_empty() {
        // A raw first field can hold the middle too: "Johanna IP".
        let words: Vec<&str> = first_raw
            .split(is_python_whitespace_compat)
            .filter(|word| !word.is_empty())
            .collect();
        if words.len() < 2 {
            return None;
        }
        first = words[..words.len() - 1].join(" ");
        middle = words[words.len() - 1].to_string();
    }
    if !last_name_allows_joined_initials(&first, unidecode_char_map) {
        return None;
    }
    let last_word = middle
        .split(is_python_whitespace_compat)
        .rfind(|word| !word.is_empty())?;
    joined_initials(last_word, crate::name_normalization::name_prefixes())
        .filter(|initials| !MIDDLE_NON_INITIALS.contains(&initials.as_str()))
        .map(|initials| format!("S{initials}"))
}

// A middle name of these letters is a suffix, a roman numeral or the surname prefix "Mc",
// not initials, in any letter case (mirrors Python `_MIDDLE_NON_INITIALS`).
const MIDDLE_NON_INITIALS: [&str; 9] = ["jr", "sr", "ii", "iii", "iv", "vi", "vii", "viii", "mc"];

/// The CJK given-name value, else a joined-initials value (mirrors Python
/// `given_name_for_constraint`). CJK given names are not ASCII, joined-initials readings
/// have no middle name, and spelled middle initials need one, so the values never overlap.
pub(crate) fn given_name_for_constraint(
    first_raw: &str,
    middle_raw: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> Option<String> {
    cjk_given_name_for_constraint(first_raw, middle_raw, unidecode_char_map)
        .or_else(|| joined_initials_for_constraint(first_raw, middle_raw))
        .or_else(|| {
            spelled_middle_initials_for_constraint(first_raw, middle_raw, unidecode_char_map)
        })
}

// D9 (mirrors Python `_JOINED_INITIALS_GIVEN_NAMES`): ED is a frequent given
// name, and MD abbreviates Muhammad (D7).
const JOINED_INITIALS_GIVEN_NAMES: [&str; 2] = ["ED", "MD"];
// With a final period, these capital pairs abbreviate one given name, for
// example "PH." (Philippe), "GY." (György), and "YU." (Yuri). Mirrors Python
// `_DOTTED_GIVEN_NAME_ABBREVIATIONS`.
const DOTTED_GIVEN_NAME_ABBREVIATIONS: [&str; 15] = [
    "CH", "SH", "ZH", "KH", "TS", "YU", "YA", "YE", "GY", "NY", "SZ", "ZS", "CS", "PH", "TH",
];
// The D9 mixed-case branch rejects a vowel or y, because these letters can
// make a short name, such as "Ty" or "Bo".
const MIXED_CASE_INITIALS_VOWELS: &[u8] = b"aeiouy";
// These mixed-case strings are not initials (mirrors Python
// `_MIXED_CASE_NON_INITIALS`). The 2026-10-16 review of mention counts and
// gold clusters found that each one is a given-name abbreviation, a title, a
// suffix, a real name, or a company word.
#[rustfmt::skip]
const MIXED_CASE_NON_INITIALS: [&str; 35] = [
    // Abbreviations of one given name, for example "Md" (Muhammad).
    "ch", "chr", "cs", "dzh", "fr", "gh", "kh", "khr", "lj", "ll", "md", "mhd", "ph", "sh", "sk",
    "st", "sw", "sz", "th", "vl", "wm", "zh", "zs",
    // Titles before the given name, for example "Mst" (Mosammat).
    "ds", "hjh", "lt", "mst", "rr", "smt",
    // Suffixes: "Jr" (Junior) and "Sr" (Senior).
    "jr", "sr",
    // Real names: "Ng" and "Lv" are surnames, and "Tsz" is a given name.
    "lv", "ng", "tsz",
    // A company word ("Ltd").
    "ltd",
];

/// D9 letter-case guard for one raw last name (mirrors Python
/// `last_name_allows_joined_initials`). Letter case comes from the canonical
/// transliteration before lowercasing: ASCII `a-z` is lowercase and `A-Z` is
/// uppercase, so `á` counts as lowercase. The guard passes when there is at
/// least one lowercase letter and no more uppercase than lowercase letters.
pub(crate) fn last_name_allows_joined_initials(
    last_raw: &str,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> bool {
    let mut lowercase_count = 0usize;
    let mut uppercase_count = 0usize;
    let mut ascii_buffer = [0u8; 4];
    for ch in canonical_name_pretranslate(last_raw).chars() {
        let mapped: &str = if ch.is_ascii() {
            ch.encode_utf8(&mut ascii_buffer)
        } else {
            unidecode_char_map
                .and_then(|char_map| char_map.get(&ch).map(String::as_str))
                .unwrap_or_else(|| text_unidecode_char(ch))
        };
        for byte in mapped.bytes() {
            if byte.is_ascii_lowercase() {
                lowercase_count += 1;
            } else if byte.is_ascii_uppercase() {
                uppercase_count += 1;
            }
        }
    }
    lowercase_count > 0 && lowercase_count >= uppercase_count
}

/// Lowercase D9 initials of a raw first name, or `None` (mirrors Python
/// `_joined_initials`). This applies every D9 condition except the last-name
/// guard.
fn joined_initials(first_raw: &str, name_prefixes: &HashSet<String>) -> Option<String> {
    let mut letters = String::with_capacity(3);
    let mut has_period = false;
    for ch in first_raw.chars() {
        if is_python_whitespace_compat(ch) {
            continue;
        }
        if ch == '.' {
            has_period = true;
            continue;
        }
        // Both branches need 2-3 ASCII letters, so stop early on a long first.
        if !ch.is_ascii_alphabetic() || letters.len() == 3 {
            return None;
        }
        letters.push(ch);
    }
    if letters.len() < 2 {
        return None;
    }
    let bytes = letters.as_bytes();
    let initials = if bytes.iter().all(u8::is_ascii_uppercase) {
        if JOINED_INITIALS_GIVEN_NAMES.contains(&letters.as_str()) {
            return None;
        }
        if letters.len() == 2
            && DOTTED_GIVEN_NAME_ABBREVIATIONS.contains(&letters.as_str())
            && first_raw
                .trim_matches(is_python_whitespace_compat)
                .ends_with('.')
        {
            return None;
        }
        letters.to_ascii_lowercase()
    } else if !has_period
        && bytes[0].is_ascii_uppercase()
        && bytes[1..].iter().all(u8::is_ascii_lowercase)
    {
        // The mixed-case branch keeps periods in its match, so "Ch." fails it.
        let initials = letters.to_ascii_lowercase();
        if initials
            .bytes()
            .any(|byte| MIXED_CASE_INITIALS_VOWELS.contains(&byte))
            || MIXED_CASE_NON_INITIALS.contains(&initials.as_str())
        {
            return None;
        }
        initials
    } else {
        return None;
    };
    // Titles keep the D7 drop: "DR" and "Dr" are not the initials "d r".
    (!name_prefixes.contains(&initials)).then_some(initials)
}

/// Canonicalize raw first/middle/last per the canonical_v2 pipeline (mirrors
/// Python `canonicalize_name_parts`). Returns (first, middle, last).
///
/// - At most one leading title-prefix token is dropped from first (D7).
/// - First/middle split (D1): a leading dash-bound group stays together in
///   first as spaced tokens; otherwise the first token stays and later tokens
///   spill into middle ahead of existing middle tokens. Space tokens after a
///   dash-bound group still spill.
/// - Joined initials (D9): a first of 2-3 ASCII capitals after whitespace and
///   period removal becomes a first initial plus middle initials when the
///   last name passes `last_name_allows_joined_initials`. A first of one
///   capital and 1-2 lowercase letters with no vowel, no y, and no period,
///   such as "Jh", changes the same way unless it is a listed non-initial.
pub(crate) fn canonicalize_name_parts_compat(
    first_raw: &str,
    middle_raw: &str,
    last_raw: &str,
    name_prefixes: &HashSet<String>,
    unidecode_char_map: Option<&HashMap<char, String>>,
) -> (String, String, String) {
    let middle_text = canonicalize_name_text_compat(middle_raw, unidecode_char_map);
    let last = canonicalize_name_text_compat(last_raw, unidecode_char_map);
    if let Some(initials) = joined_initials(first_raw, name_prefixes) {
        if last_name_allows_joined_initials(last_raw, unidecode_char_map) {
            let mut middle_tokens: Vec<String> = initials[1..].chars().map(String::from).collect();
            if !middle_text.is_empty() {
                middle_tokens.push(middle_text);
            }
            return (initials[..1].to_string(), middle_tokens.join(" "), last);
        }
    }

    let first_clean = canonical_name_pretranslate(first_raw);

    // Whitespace chunks of the raw first field, each normalized to tokens and
    // tagged with whether a dash bound it together.
    let mut flattened: Vec<(String, usize)> = Vec::new();
    let mut dash_bound: Vec<bool> = Vec::new();
    for (group_index, chunk) in first_clean
        .split(is_python_whitespace_compat)
        .filter(|chunk| !chunk.is_empty())
        .enumerate()
    {
        dash_bound.push(chunk.contains('-'));
        for token in canonical_name_tokens(&chunk.replace('-', " "), unidecode_char_map) {
            flattened.push((token, group_index));
        }
    }

    if let Some((token, _group)) = flattened.first() {
        if name_prefixes.contains(token) {
            flattened.remove(0);
        }
    }

    let (first_field_tokens, spilled_tokens): (Vec<String>, Vec<String>) = if flattened.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        let lead_group = flattened[0].1;
        if dash_bound[lead_group] {
            // partition preserves order within both halves.
            let (kept, spilled): (Vec<_>, Vec<_>) = flattened
                .into_iter()
                .partition(|(_token, group)| *group == lead_group);
            (
                kept.into_iter().map(|(token, _)| token).collect(),
                spilled.into_iter().map(|(token, _)| token).collect(),
            )
        } else {
            let mut iter = flattened.into_iter().map(|(token, _)| token);
            let head = iter.next().expect("flattened is non-empty");
            (vec![head], iter.collect())
        }
    };

    let first = first_field_tokens.join(" ");
    let middle = if spilled_tokens.is_empty() {
        middle_text
    } else if middle_text.is_empty() {
        spilled_tokens.join(" ")
    } else {
        format!("{} {}", spilled_tokens.join(" "), middle_text)
    };
    (first, middle, last)
}

/// Canonical_v2 count keys built from canonical fields after gating (mirrors
/// Python `canonical_name_count_keys`, D6/D8). A `None` key means no lookup
/// (NaN feature), never a sentinel count. `first` and `first_last` require an
/// informative first (string length > 1); `last_first_initial` requires first
/// and last present and stays initial-char semantics.
pub(crate) struct CanonicalNameCountKeys {
    pub(crate) first: Option<String>,
    pub(crate) last: Option<String>,
    pub(crate) first_last: Option<String>,
    pub(crate) last_first_initial: Option<String>,
}

pub(crate) fn canonical_name_count_keys_compat(first: &str, last: &str) -> CanonicalNameCountKeys {
    let first_informative = crate::py_len(first) > 1;
    let last_present = !last.is_empty();
    CanonicalNameCountKeys {
        first: first_informative.then(|| first.to_string()),
        last: last_present.then(|| last.to_string()),
        first_last: (first_informative && last_present).then(|| format!("{first} {last}")),
        last_first_initial: (!first.is_empty() && last_present).then(|| {
            let first_initial = first.chars().next().expect("first is non-empty");
            format!("{last} {first_initial}")
        }),
    }
}

pub(crate) fn compute_block_compat(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let name_parts: Vec<&str> = name.split(' ').collect();
    if name_parts.len() == 1 {
        return name_parts[0].to_string();
    }
    let Some(first_initial) = name_parts[0].chars().next() else {
        return String::new();
    };
    format!("{} {}", first_initial, name_parts[name_parts.len() - 1])
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_name_count_keys_compat, canonicalize_name_parts_compat,
        cjk_given_name_for_constraint, given_name_for_constraint, is_cjk_given_name,
        normalize_text_compat_native, text_unidecode_char, MIXED_CASE_NON_INITIALS,
    };

    #[test]
    fn spelled_middle_initials_need_joined_capitals_and_a_cased_first_name() {
        let value = |first: &str, middle: &str| given_name_for_constraint(first, middle, None);
        assert_eq!(value("Johanna", "IP").as_deref(), Some("Sip"));
        assert_eq!(value("Johanna", "I.P.").as_deref(), Some("Sip"));
        assert_eq!(value("Johanna", "Ip"), None);
        assert_eq!(value("Johanna", "I. Petra"), None);
        assert_eq!(value("JOHANNA", "IP"), None);
        assert_eq!(value("Лев", "IP").as_deref(), Some("Sip"));
        assert_eq!(value("Johanna IP", "").as_deref(), Some("Sip"));
        assert_eq!(value("Johanna", "A IP").as_deref(), Some("Sip"));
        assert_eq!(value("Johanna", "A. IP").as_deref(), Some("Sip"));
        assert_eq!(value("Johanna", "IP A"), None);
        assert_eq!(value("Jeffrey", "Jm").as_deref(), Some("Sjm"));
        assert_eq!(value("Jeffrey", "Mc"), None);
        assert_eq!(value("John", "III"), None);
        assert_eq!(value("John", "JR"), None);
        assert_eq!(value("Edward", "ED"), None);
        assert_eq!(value("Ka", "").as_deref(), Some("Jka"));
    }

    #[test]
    fn cjk_given_name_values_carry_a_script_tag_or_mark_incomplete_names() {
        let value = |first: &str| cjk_given_name_for_constraint(first, "", None);
        assert_eq!(value("重喜").as_deref(), Some("Hzhongxi"));
        assert_eq!(value("奈々").as_deref(), value("奈奈").as_deref());
        assert_eq!(value("ひろし").as_deref(), Some("Khirosi"));
        assert_eq!(value("ヒロシ").as_deref(), Some("Khirosi"));
        assert_eq!(value("민수").as_deref(), Some("Gminsu"));
        // Halfwidth katakana and spacing voicing marks compose like their fullwidth forms.
        assert_eq!(value("ｶﾞｸ").as_deref(), Some("Kgaku"));
        assert_eq!(value("しか゛").as_deref(), value("しが").as_deref());
        // A long-vowel mark has no letters, so the name keeps the ordinary rules.
        assert_eq!(value("ピーター").as_deref(), Some(""));
        assert_eq!(value("Wei"), None);
    }

    #[test]
    fn cjk_given_names_are_han_kana_or_hangul_only() {
        for (first, middle) in [
            ("重喜", ""),
            ("孝", ""),
            ("重", "喜"),
            ("ひろし", ""),
            ("ヒロシ", ""),
            ("민수", ""),
            ("민 수", ""),
            ("佐々木", ""),
            ("重喜.", " "),
        ] {
            assert!(
                is_cjk_given_name(first, middle, None),
                "{first:?} {middle:?}"
            );
        }
        for (first, middle) in [
            ("", ""),
            ("Wei", ""),
            ("重喜", "Shigeki"),
            ("Shigeki 重喜", ""),
            ("Ж", ""),
        ] {
            assert!(
                !is_cjk_given_name(first, middle, None),
                "{first:?} {middle:?}"
            );
        }
    }
    use crate::HashSet;
    use std::path::Path;

    #[test]
    fn text_unidecode_char_matches_python_text_unidecode_compat_overrides() {
        assert_eq!(text_unidecode_char('\u{0080}'), "");
        assert_eq!(text_unidecode_char('\u{02EF}'), "[?]");
        assert_eq!(text_unidecode_char('\u{02FF}'), "[?] ");
        assert_eq!(text_unidecode_char('\u{25F4}'), "#");
        assert_eq!(text_unidecode_char('\u{FDFD}'), "[?]");
    }

    // Mirrors `NAME_PREFIXES` in s2and/text/normalize.py. Note "md" is deliberately NOT a
    // prefix (D7): it is a common South Asian given-name abbreviation.
    fn python_name_prefixes() -> HashSet<String> {
        [
            "dr",
            "prof",
            "professor",
            "mr",
            "miss",
            "mrs",
            "ms",
            "mx",
            "sir",
            "phd",
            "doctor",
        ]
        .into_iter()
        .map(str::to_string)
        .collect()
    }

    fn fixture_string(value: &serde_json::Value) -> String {
        // Fixture inputs use null for missing fields; canonicalization treats
        // null and empty identically.
        value.as_str().unwrap_or("").to_string()
    }

    #[test]
    fn canonical_name_fixture_matches_python_reference_for_all_cases() {
        let fixture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("fixtures")
            .join("canonical_name_examples.json");
        let fixture_text =
            std::fs::read_to_string(&fixture_path).expect("read canonical_name_examples.json");
        let fixture: serde_json::Value =
            serde_json::from_str(&fixture_text).expect("parse canonical_name_examples.json");
        let cases = fixture["cases"].as_array().expect("fixture cases array");
        assert!(!cases.is_empty(), "fixture has no cases");

        let name_prefixes = python_name_prefixes();
        for case in cases {
            let case_id = case["id"].as_str().expect("case id");
            let input = &case["input"];
            let canonical = &case["canonical"];
            // Native transliteration table (no Python-supplied unidecode map):
            // the crate's `text_unidecode_char` carries the python parity
            // overrides, so the fixture must pass without a supplied map.
            let (first, middle, last) = canonicalize_name_parts_compat(
                &fixture_string(&input["first"]),
                &fixture_string(&input["middle"]),
                &fixture_string(&input["last"]),
                &name_prefixes,
                None,
            );
            assert_eq!(
                first,
                canonical["first"].as_str().expect("canonical first"),
                "case {case_id}: first"
            );
            assert_eq!(
                middle,
                canonical["middle"].as_str().expect("canonical middle"),
                "case {case_id}: middle"
            );
            assert_eq!(
                last,
                canonical["last"].as_str().expect("canonical last"),
                "case {case_id}: last"
            );

            let expected_keys = &canonical["count_keys"];
            let keys = canonical_name_count_keys_compat(&first, &last);
            for (name, actual) in [
                ("first", &keys.first),
                ("last", &keys.last),
                ("first_last", &keys.first_last),
                ("last_first_initial", &keys.last_first_initial),
            ] {
                let expected = expected_keys[name].as_str();
                assert_eq!(
                    actual.as_deref(),
                    expected,
                    "case {case_id}: count key {name}"
                );
            }
        }
    }

    #[test]
    fn canonical_first_splitting_matches_python_whitespace() {
        let name_prefixes = python_name_prefixes();
        for separator in ['\u{001c}', '\u{001d}', '\u{001e}', '\u{001f}'] {
            let raw = format!("Anne-Marie{separator}Louise");
            let (first, middle, last) =
                canonicalize_name_parts_compat(&raw, "", "", &name_prefixes, None);
            assert_eq!(first, "anne marie", "separator U+{:04X}", separator as u32);
            assert_eq!(middle, "louise", "separator U+{:04X}", separator as u32);
            assert_eq!(last, "");
        }

        for raw in [
            "Anne-Marie\u{001c}\u{001d}\u{001e}\u{001f}Louise",
            "Anne-Marie\u{00a0}Louise",
        ] {
            let (first, middle, last) =
                canonicalize_name_parts_compat(raw, "", "", &name_prefixes, None);
            assert_eq!(first, "anne marie", "raw input {raw:?}");
            assert_eq!(middle, "louise", "raw input {raw:?}");
            assert_eq!(last, "");
        }
    }

    #[test]
    fn mixed_case_non_initials_match_python_source() {
        let python_source = include_str!("../../s2and/text/canonical_names.py");
        let literal = python_source
            .split_once("_MIXED_CASE_NON_INITIALS = frozenset(")
            .expect("Python non-initials set")
            .1
            .split_once("\n)\n")
            .expect("Python non-initials set end")
            .0;
        let python: HashSet<&str> = literal
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .flat_map(|line| line.split('"').skip(1).step_by(2))
            .collect();
        let rust: HashSet<&str> = MIXED_CASE_NON_INITIALS.into_iter().collect();
        assert_eq!(rust.len(), MIXED_CASE_NON_INITIALS.len());
        assert_eq!(rust, python);
    }

    #[test]
    fn normalize_text_compat_preserves_python_boundaries_for_overrides() {
        assert_eq!(
            normalize_text_compat_native("a\u{0080}b \u{03F4}c \u{02FF}d"),
            "ab c d",
        );
    }
}

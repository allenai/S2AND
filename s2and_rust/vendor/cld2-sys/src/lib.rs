//! Unsafe, low-level wrapper around cld2, the "compact language detector"
//! based on Chromium's code, plus a very thin C wrapper layer.  Normally
//! you won't want to use this library directly unless you're writing
//! your own cld2 wrapper library.
//!
//! If you need access to APIs which are not currently wrapped, please feel
//! free to send pull requests!

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

extern crate libc;

pub use encodings::*;
pub use languages::*;
pub use flags::*;
pub use wrapper::*;

mod encodings;
mod languages;
mod flags;
mod wrapper;

// Just a single placeholder test in case somebody runs 'cargo test' in
// this library's directory, and not in the main library's directory.  This
// is not intended to be comprehensive, but please add regression tests for
// any bugs.
#[test]
fn test_detection() {
    let english = "
It is an ancient Mariner,
And he stoppeth one of three.
'By thy long grey beard and glittering eye,
Now wherefore stopp'st thou me?

'The Bridegroom's doors are opened wide,
And I am next of kin;
The guests are met, the feast is set:
May'st hear the merry din.'

He holds him with his skinny hand,
'There was a ship,' quoth he.
'Hold off! unhand me, grey-beard loon!'
Eftsoons his hand dropt he.
";

    let mut is_reliable: bool = false;
    let language = unsafe {
        CLD2_DetectLanguage(english.as_ptr() as *const i8,
                            english.len() as libc::c_int,
                            true, &mut is_reliable)
    };
    assert_eq!(Language::ENGLISH, language);
    assert_eq!(true, is_reliable);
}

// This particular API has extra wrapper code, so we want to test it.
#[test]
fn test_result_chunks() {
    use libc::{c_int, c_double};
    use std::slice::from_raw_parts;
    use std::iter::repeat;

    let mixed = "
It is an ancient Mariner,
And he stoppeth one of three.
'By thy long grey beard and glittering eye,
Now wherefore stopp'st thou me?

Sur le pont d'Avignon,
L'on y danse, l'on y danse,
Sur le pont d'Avignon
L'on y danse tous en rond.

Les belles dames font comme ça
Et puis encore comme ça.
Les messieurs font comme ça
Et puis encore comme ça.
";

    let hints = CLDHints{content_language_hint: std::ptr::null(),
                         tld_hint: std::ptr::null(),
                         encoding_hint: Encoding::UNKNOWN_ENCODING as c_int,
                         language_hint: Language::UNKNOWN_LANGUAGE};
    let mut language3: Vec<Language> =
        repeat(Language::UNKNOWN_LANGUAGE).take(3).collect();
    let mut percent3: Vec<c_int> = repeat(0).take(3).collect();
    let mut normalized_score3: Vec<c_double> = repeat(0.0).take(3).collect();
    let mut text_bytes: c_int = 0;
    let mut is_reliable: bool = false;

    let chunks = unsafe { CLD2_ResultChunkVector_new() };

    let language = unsafe {
        CLD2_ExtDetectLanguageSummary4(mixed.as_ptr() as *const i8,
                                       mixed.len() as c_int,
                                       true, &hints, 0,
                                       language3.as_mut_ptr(),
                                       percent3.as_mut_ptr(),
                                       normalized_score3.as_mut_ptr(),
                                       chunks,
                                       &mut text_bytes, &mut is_reliable)
    };
    assert_eq!(Language::FRENCH, language);

    unsafe {
        let data = CLD2_ResultChunkVector_data(chunks as *const ResultChunks);
        let size = CLD2_ResultChunkVector_size(chunks as *const ResultChunks);
        let slice: &[ResultChunk] = from_raw_parts(data, size as usize);
        //println!("Chunks: {}", slice);
        let mut found_mariner = false;
        let mut found_comme_ca = false;
        for chunk in slice.iter() {
            let text =
                &mixed[chunk.offset as usize..
                       chunk.offset as usize + chunk.bytes as usize];

            if chunk.lang1 == Language::ENGLISH as u16
                && text.contains("ancient Mariner")
            {
                found_mariner = true;
            }

            if chunk.lang1 == Language::FRENCH as u16
                && text.contains("comme ça")
            {
                found_comme_ca = true;
            }
        }
        assert!(found_mariner);
        assert!(found_comme_ca);
    };

    unsafe { CLD2_ResultChunkVector_delete(chunks); }
}

#[test]
fn test_language_names() {
    use std::ffi::{CString, CStr};
    use std::str::from_utf8;

    let code = unsafe { 
        let char_ptr = CLD2_LanguageCode(Language::ENGLISH);
        let bytes = CStr::from_ptr(char_ptr).to_bytes();
        from_utf8(bytes).unwrap().to_string()
    };
    assert_eq!("en", code);

    let language = unsafe {
        let c_str = CString::new("fr".as_bytes()).unwrap();
        CLD2_GetLanguageFromName(c_str.as_ptr())
    };
    assert_eq!(Language::FRENCH, language);
}

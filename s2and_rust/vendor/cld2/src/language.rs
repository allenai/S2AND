//! A semi-reasonable interface to the language IDs used by cld2.  We could
//! theoretically export this if somebody needed it.

#![allow(missing_docs)]

use libc::c_char;
use std::mem::transmute;
use std::ffi::{CString, CStr};
use std::str::from_utf8;
use ffi::{CLD2_GetLanguageFromName, CLD2_LanguageName,
          CLD2_LanguageCode, CLD2_LanguageDeclaredName};
pub use ffi::Language as LanguageId;

use types::Lang;

unsafe fn from_static_c_str<'a>(raw: &'a *const c_char) -> &'static str {
    let ptr: *const c_char = transmute(*raw);
    from_utf8(CStr::from_ptr(ptr).to_bytes()).unwrap()
}

pub trait LanguageIdExt {
    fn from_name(name: &str) -> Self;
    fn name(&self) -> &'static str;
    fn code(&self) -> &'static str;
    fn declared_name(&self) -> &'static str;
    fn is_unknown(&self) -> bool;
    fn is_known(&self) -> bool;
    fn to_lang(&self) -> Option<Lang>;
}

impl LanguageIdExt for LanguageId {
    fn from_name(name: &str) -> LanguageId {
        unsafe {
            let c_name = CString::new(name.as_bytes()).unwrap();
            CLD2_GetLanguageFromName(c_name.as_ptr())
        }
    }

    fn name(&self) -> &'static str {
        unsafe { from_static_c_str(&CLD2_LanguageName(*self)) }
    }

    fn code(&self) -> &'static str {
        unsafe { from_static_c_str(&CLD2_LanguageCode(*self)) }
    }

    fn declared_name(&self) -> &'static str {
        unsafe { from_static_c_str(&CLD2_LanguageDeclaredName(*self)) }
    }

    fn is_unknown(&self) -> bool { *self == LanguageId::UNKNOWN_LANGUAGE }
    fn is_known(&self) -> bool { *self != LanguageId::UNKNOWN_LANGUAGE }

    fn to_lang(&self) -> Option<Lang> {
        if self.is_known() {
            Some(Lang(self.code()))
        } else {
            None
        }
    }
}

#[test]
fn test_language_ext() {
    let lang = LanguageIdExt::from_name("en");
    assert_eq!(LanguageId::ENGLISH, lang);
    assert_eq!("ENGLISH", lang.name());
    assert_eq!("en", lang.code());
    assert_eq!("ENGLISH", lang.declared_name());

    assert_eq!(true, LanguageId::ENGLISH.is_known());
    assert_eq!(false, LanguageId::ENGLISH.is_unknown());
    assert_eq!(false, LanguageId::UNKNOWN_LANGUAGE.is_known());
    assert_eq!(true, LanguageId::UNKNOWN_LANGUAGE.is_unknown());

    // This is a bit yucky.
    assert_eq!("un", LanguageId::UNKNOWN_LANGUAGE.code());
}

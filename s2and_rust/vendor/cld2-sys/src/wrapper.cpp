#include "wrapper.h"

ResultChunks *CLD2_ResultChunkVector_new() {
    return static_cast<ResultChunks*>(new CLD2::ResultChunkVector());
}

const ResultChunk *CLD2_ResultChunkVector_data(const ResultChunks *chunks) {
    const CLD2::ResultChunkVector *vec =
        static_cast<const CLD2::ResultChunkVector *>(chunks);
    return static_cast<const ResultChunk*>(static_cast<const void*>(&(*vec)[0]));
}

size_t CLD2_ResultChunkVector_size(const ResultChunks *chunks) {
    const CLD2::ResultChunkVector *vec =
        static_cast<const CLD2::ResultChunkVector *>(chunks);
    return vec->size();
}

void CLD2_ResultChunkVector_delete(ResultChunks *chunks) {
    CLD2::ResultChunkVector *vec =
        static_cast<CLD2::ResultChunkVector *>(chunks);
    delete vec;
}

const char* CLD2_LanguageName(Language lang) {
    return CLD2::LanguageName(lang);
}

const char* CLD2_LanguageCode(Language lang) {
    return CLD2::LanguageCode(lang);
}

const char* CLD2_LanguageDeclaredName(Language lang) {
    return CLD2::LanguageDeclaredName(lang);
}

Language CLD2_GetLanguageFromName(const char* src) {
    return CLD2::GetLanguageFromName(src);
}

Language CLD2_DetectLanguage(const char* buffer,
                             int buffer_length,
                             bool is_plain_text,
                             bool* is_reliable)
{
    return CLD2::DetectLanguage(buffer, buffer_length, is_plain_text,
                                is_reliable);
}

Language CLD2_DetectLanguageSummary(const char* buffer,
                                    int buffer_length,
                                    bool is_plain_text,
                                    Language* language3,
                                    int* percent3,
                                    int* text_bytes,
                                    bool* is_reliable)
{
    return CLD2::DetectLanguageSummary(buffer, buffer_length, is_plain_text,
                                       language3, percent3, text_bytes,
                                       is_reliable);
}

Language CLD2_DetectLanguageSummary2(const char* buffer,
                                     int buffer_length,
                                     bool is_plain_text,
                                     const char* tld_hint,
                                     int encoding_hint,
                                     Language language_hint,
                                     Language* language3,
                                     int* percent3,
                                     int* text_bytes,
                                     bool* is_reliable)
{
    return CLD2::DetectLanguageSummary(buffer, buffer_length, is_plain_text,
                                       tld_hint, encoding_hint, language_hint,
                                       language3, percent3, text_bytes,
                                       is_reliable);
}

Language CLD2_ExtDetectLanguageSummary(const char* buffer,
                                       int buffer_length,
                                       bool is_plain_text,
                                       Language* language3,
                                       int* percent3,
                                       int* text_bytes,
                                       bool* is_reliable)
{
    return CLD2::ExtDetectLanguageSummary(buffer, buffer_length, is_plain_text,
                                          language3, percent3, text_bytes,
                                          is_reliable);
}

Language CLD2_ExtDetectLanguageSummary2(const char* buffer,
                                        int buffer_length,
                                        bool is_plain_text,
                                        const char* tld_hint,
                                        int encoding_hint,
                                        Language language_hint,
                                        Language* language3,
                                        int* percent3,
                                        int* text_bytes,
                                        bool* is_reliable)
{
    return CLD2::ExtDetectLanguageSummary(buffer, buffer_length, is_plain_text,
                                          tld_hint, encoding_hint,
                                          language_hint, language3, percent3,
                                          text_bytes, is_reliable);
}

Language CLD2_ExtDetectLanguageSummary3(const char* buffer,
                                        int buffer_length,
                                        bool is_plain_text,
                                        const char* tld_hint,
                                        int encoding_hint,
                                        Language language_hint,
                                        Language* language3,
                                        int* percent3,
                                        double* normalized_score3,
                                        int* text_bytes,
                                        bool* is_reliable)
{
    return CLD2::ExtDetectLanguageSummary(buffer, buffer_length, is_plain_text,
                                          tld_hint, encoding_hint, language_hint,
                                          language3, percent3, normalized_score3,
                                          text_bytes, is_reliable);
}

Language CLD2_ExtDetectLanguageSummary4(const char* buffer,
                                        int buffer_length,
                                        bool is_plain_text,
                                        const CLDHints* cld_hints,
                                        int flags,
                                        Language* language3,
                                        int* percent3,
                                        double* normalized_score3,
                                        ResultChunks *resultchunkvector,
                                        int* text_bytes,
                                        bool* is_reliable)
{
    const CLD2::CLDHints *hints =
        static_cast<const CLD2::CLDHints *>(static_cast<const void*>(cld_hints));
    CLD2::ResultChunkVector *vec =
        static_cast<CLD2::ResultChunkVector *>(resultchunkvector);
    return CLD2::ExtDetectLanguageSummary(buffer, buffer_length, is_plain_text,
                                          hints, flags, language3, percent3,
                                          normalized_score3, vec, text_bytes,
                                          is_reliable);
}

const char* CLD2_DetectLanguageVersion() {
    return CLD2::DetectLanguageVersion();
}

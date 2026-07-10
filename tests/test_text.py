import random
import unittest
from collections import Counter
from typing import Any, cast

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from s2and.consts import NUMPY_NAN
from s2and.data import NameCounts
from s2and.text import (
    canonicalize_name_parts,
    compute_block,
    cosine_sim,
    counter_jaccard,
    detect_language,
    diff,
    email_prefix_suffix,
    equal,
    equal_initial,
    equal_middle,
    first_names_name_compatible,
    get_text_ngrams,
    get_text_ngrams_words,
    jaccard,
    name_counts,
    name_text_features,
    normalize_orcid,
    normalize_orcid_compact,
    normalize_text,
    normalize_title,
    same_prefix_tokens,
)


class TestClusterer(unittest.TestCase):
    def test_normalize_text(self):
        assert "" == normalize_text(None)
        assert "" == normalize_text("")
        assert "text" == normalize_text("TeXt")
        assert "te han zi xt" == normalize_text("te'漢字xt")
        assert "text" == normalize_text("te'xt", True)
        assert "a b" == normalize_text("A1 B-2")

    def test_normalize_title_preserves_identifying_digits(self):
        assert normalize_title("Part 1: Co3O4 in 2025") == "part 1 co3o4 in 2025"
        assert normalize_title("PART-1 / Co3O4") == "part 1 co3o4"
        assert normalize_title(None) == ""

    def test_normalize_orcid_canonicalizes_common_forms(self):
        assert normalize_orcid(" https://orcid.org/0000-0002-1825-0097 ") == "0000-0002-1825-0097"
        assert normalize_orcid("ORCID: 000000021825009x") == "0000-0002-1825-009X"
        assert normalize_orcid_compact("ORCID: 000000021825009x") == "000000021825009X"
        for dash in "-\u2010\u2011\u2012\u2013\u2014\u2212\ufe58\ufe63\uff0d":
            assert normalize_orcid(dash.join(["0000", "0002", "1825", "0097"])) == "0000-0002-1825-0097"
        assert normalize_orcid("https://orcid.org/0000\u20100002\u20101825\u20100097") == "0000-0002-1825-0097"
        assert normalize_orcid("s000-0000-1879-1075X") is None
        assert normalize_orcid("0000-0002-1825") is None
        # Non-ASCII (Unicode) digits are rejected, matching the Rust
        # is_ascii_digit() behavior; only [0-9] count as ORCID digits.
        assert normalize_orcid("٠٠٠٠-٠٠٠٢-1825-009X") is None
        assert normalize_orcid("００００-0002-1825-0097") is None

    def test_canonical_first_treats_unicode_dashes_as_hyphens(self):
        assert canonicalize_name_parts("Amin-ul-Haq", None, None)[:2] == ("amin ul haq", "")
        assert canonicalize_name_parts("Arif\u2010ullah", None, None)[:2] == ("arif ullah", "")
        assert canonicalize_name_parts("Hua\uff0dli", None, None)[:2] == ("hua li", "")

    def test_canonical_first_preserves_md_as_given_name(self):
        assert canonicalize_name_parts("Md Karim", None, None)[:2] == ("md", "karim")
        assert canonicalize_name_parts("Md", None, None)[:2] == ("md", "")
        assert canonicalize_name_parts("Dr Md Karim", None, None)[:2] == ("md", "karim")

    def test_name_similarity_features(self):
        assert [NUMPY_NAN] * 4 == name_text_features("", cast(Any, None))
        assert [0.0, 0.0, 0.0, 1.0] == name_text_features("text", "text")
        assert all([s >= 0.0 and s <= 1.0 for s in name_text_features("textual", "txt")])
        assert all([s >= 0.0 and s <= 1.0 for s in name_text_features("a", "alice")])

    def test_cosine_sim(self):
        random_vec_1 = np.array([random.uniform(-1000, 1000) for i in range(1000)])
        random_vec_2 = np.array([random.uniform(-1000, 1000) for i in range(1000)])
        self.assertAlmostEqual(
            cosine_sim(random_vec_1, random_vec_2),
            cosine_similarity(random_vec_1.reshape(1, -1), random_vec_2.reshape(1, -1))[0][0],
        )
        assert cosine_sim(np.zeros(1000), random_vec_2) == 0

    def test_get_text_ngrams(self):
        assert Counter() == get_text_ngrams(None)
        assert Counter() == get_text_ngrams("the")
        assert Counter(
            [
                "hell",
                "ello",
                "hel",
                "ell",
                "llo",
                "he",
                "el",
                "ll",
                "lo",
                "wor",
                "wo",
                "or",
            ]
        ) == get_text_ngrams("hello wor")
        assert Counter(
            [
                "hell",
                "ello",
                "hel",
                "ell",
                "llo",
                "he",
                "el",
                "ll",
                "lo",
                "wor",
                "wo",
                "or",
                "h",
                "e",
                "l",
                "l",
                "o",
                "w",
                "o",
                "r",
            ]
        ) == get_text_ngrams("hello wor", use_unigrams=True)

    def test_get_text_ngrams_words(self):
        assert Counter() == get_text_ngrams_words(None)
        assert Counter() == get_text_ngrams_words("the")
        assert Counter(
            [
                "quick green fox",
                "green fox jumped",
                "quick green",
                "green fox",
                "fox jumped",
                "quick",
                "green",
                "fox",
                "jumped",
            ]
        ) == get_text_ngrams_words("the quick green fox jumped")
        assert get_text_ngrams_words("part 1", drop_short_tokens=False)["1"] == 1

    def test_equal(self):
        assert np.isnan(equal(None, None))
        assert np.isnan(equal("", ""))
        assert np.isnan(equal("-", "text"))
        assert 1 == equal("text", "text")
        assert 0 == equal("text", "hi")
        # Whitespace-only inputs are empty after stripping -> default (NaN),
        # not a spurious "" == "" match.
        assert np.isnan(equal(" ", "  "))
        assert np.isnan(equal(" ", "text"))

    def test_equal_middle(self):
        assert np.isnan(equal_middle(None, None))
        assert np.isnan(equal_middle("", ""))
        assert np.isnan(equal_middle("a", ""))
        assert 0 == equal_middle("a", "b")
        assert 1 == equal_middle("a", "a")
        assert 1 == equal_middle("a", "as")
        assert 0 == equal_middle("as", "af")
        assert 1 == equal_middle("as", "as")
        # Multi-token middle: a single initial matches ANY token's initial,
        # not just the first token's.
        assert 1 == equal_middle("l", "james lee")
        assert 1 == equal_middle("james lee", "l")
        assert 1 == equal_middle("j", "james lee")
        assert 0 == equal_middle("k", "james lee")
        assert 1 == equal_middle("a j", "j")

    def test_email_prefix_suffix(self):
        assert email_prefix_suffix("jsmith@mit.edu") == ("jsmith", "mit.edu")
        # Leading/trailing dots are stripped and case lowered; internal dots kept.
        assert email_prefix_suffix("J.Smith@MIT.EDU.") == ("j.smith", "mit.edu")
        for malformed in (
            "jsmith",
            "a@b@c",
            "@",
            "a@",
            "@b",
            "a b@c",
            "a@b c",
            " a@b",
            "a@b ",
            "a\u00a0@b",
            ".@b",
        ):
            assert email_prefix_suffix(malformed) == (None, None)

    def test_first_name_aliases_are_order_independent(self):
        directed_aliases = {("qi xin", "qadir")}
        assert first_names_name_compatible("qi xin", "qadir", directed_aliases)
        assert first_names_name_compatible("qadir", "qi xin", directed_aliases)

    def test_get_text_ngrams_short_token_filter_decoupled_from_stopwords(self):
        # Reference-author ngrams pass stopwords=None but still drop short tokens.
        with_none = get_text_ngrams("li wu abcd", stopwords=None)
        assert "li" not in with_none
        assert "ab" in with_none
        # With stopwords None vs an (empty) set the result is now identical:
        # the two filters are independent.
        assert with_none == get_text_ngrams("li wu abcd", stopwords=set())
        # Coauthor ngrams opt out so short romanized names still contribute.
        coauthor_style = get_text_ngrams("li wu abcd", stopwords=None, drop_short_tokens=False)
        assert "li" in coauthor_style
        assert "wu" in coauthor_style
        assert "ab" in coauthor_style
        # Providing real stopwords still removes those words too.
        assert "ab" not in get_text_ngrams("abcd efgh", stopwords={"abcd"})

    def test_same_prefix_tokens_empty_is_not_positive_evidence(self):
        assert not same_prefix_tokens("", "alice")
        assert not same_prefix_tokens("", "")
        assert first_names_name_compatible("", "alice", set()) is True

    def test_equal_initial(self):
        assert np.isnan(equal_initial(None, None))
        assert np.isnan(equal_initial("", ""))
        assert np.isnan(equal_initial("a", ""))
        assert 0 == equal_initial("a", "b")
        assert 1 == equal_initial("a", "a")
        assert 1 == equal_initial("a", "as")

    def test_counter_jaccard(self):
        assert np.isnan(counter_jaccard(Counter(), Counter()))
        self.assertAlmostEqual(4 / 6, counter_jaccard(Counter([1, 2, 3, 4, 5]), Counter([1, 2, 3, 4, 6])))
        self.assertAlmostEqual(4 / 7, counter_jaccard(Counter([1, 2, 3, 4, 5, 5]), Counter([1, 2, 3, 4, 6])))

    def test_jaccard(self):
        assert np.isnan(jaccard(set(), set()))
        self.assertAlmostEqual(4 / 6, jaccard({1, 2, 3, 4, 5}, {1, 2, 3, 4, 6}))
        self.assertAlmostEqual(4 / 6, jaccard({1, 2, 3, 4, 5}, {1, 2, 3, 4, 6}))

    def test_compute_block(self):
        assert "" == compute_block("")
        assert "text" == compute_block("text")
        assert "t text" == compute_block("tony text")

    def test_diff(self):
        assert np.isnan(diff(None, None))
        assert 5 == diff(10, 5)
        assert 5 == diff(5, 10)

    def test_name_counts(self):
        nc1 = NameCounts(first=5, first_last=100, last=10, last_first_initial=200)
        nc2 = NameCounts(first=4, first_last=99, last=11, last_first_initial=201)
        assert [4, 99, 10, 200, 5, 100] == name_counts(nc1, nc2)

    def test_detect_language_uses_cld2_reliable_confidence(self):
        import s2and.text as text_module

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                text_module.cld2,
                "detect",
                lambda _text, **kwargs: (True, kwargs, [("ENGLISH", "en", 92, 0.0)]),
            )

            detection = detect_language("hello world")

        assert detection.is_reliable is True
        assert detection.is_english is True
        assert detection.predicted_language == "en"
        assert detection.language_reliability == pytest.approx(0.92)

    def test_detect_language_keeps_unreliable_known_language_with_zero_reliability(self):
        import s2and.text as text_module

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                text_module.cld2,
                "detect",
                lambda _text, **kwargs: (False, kwargs, [("FRENCH", "fr", 82, 0.0)]),
            )

            detection = detect_language("bonjour monde")

        assert detection.is_reliable is False
        assert detection.is_english is False
        assert detection.predicted_language == "fr"
        assert detection.language_reliability == 0.0

    def test_detect_language_pins_plain_text_mode(self):
        import s2and.text as text_module

        captured: dict[str, Any] = {}

        def fake_detect(_text: str, **kwargs: Any):
            captured.update(kwargs)
            return True, None, [("ENGLISH", "en", 99, 0.0)]

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(text_module.cld2, "detect", fake_detect)
            detect_language("<b>hello world</b>")

        assert captured == {"isPlainText": True}

    def test_detect_language_returns_unknown_for_combining_marks_only_text(self):
        # str.isalpha counts general-category Letter (L*) characters only, so a
        # text whose alphabetic-looking characters are all combining vowel signs
        # (Mn, Other_Alphabetic) hits the zero-isalpha early exit. This is the
        # Python reference behavior mirrored by the Rust tests in
        # s2and_rust/src/language_detection.rs.
        detection = detect_language("\u093f\u0941 \u093f")
        assert detection.is_reliable is False
        assert detection.is_english is False
        assert detection.predicted_language == "un"
        assert detection.language_reliability == 0.0


def test_cld2_unexpected_error_propagates(monkeypatch):
    import s2and.text as text_module

    def _raise_type_error(_text: str, **_kwargs: Any):
        raise TypeError("bad cld2 state")

    monkeypatch.setattr(text_module.cld2, "detect", _raise_type_error)

    with pytest.raises(TypeError, match="bad cld2 state"):
        detect_language("hello world")

import random
import threading
import time
import unittest
from collections import Counter
from typing import Any, cast

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from s2and.consts import NUMPY_NAN
from s2and.data import NameCounts
from s2and.text import (
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
    reconcile_detected_languages,
    same_prefix_tokens,
    split_first_middle_hyphen_aware,
)


class TestClusterer(unittest.TestCase):
    def test_normalize_text(self):
        assert "" == normalize_text(None)
        assert "" == normalize_text("")
        assert "text" == normalize_text("TeXt")
        assert "te han zi xt" == normalize_text("te'漢字xt")
        assert "text" == normalize_text("te'xt", True)
        assert "a b" == normalize_text("A1 B-2")

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

    def test_split_first_middle_treats_unicode_dashes_as_hyphens(self):
        assert split_first_middle_hyphen_aware("Amin-ul-Haq", None) == ("amin ul haq", "")
        assert split_first_middle_hyphen_aware("Arif\u2010ullah", None) == ("arif ullah", "")
        assert split_first_middle_hyphen_aware("Hua\uff0dli", None) == ("hua li", "")

    def test_split_first_middle_preserves_md_as_given_name(self):
        assert split_first_middle_hyphen_aware("Md Karim", None) == ("md", "karim")
        assert split_first_middle_hyphen_aware("Md", None) == ("md", "")
        assert split_first_middle_hyphen_aware("Dr Md Karim", None) == ("md", "karim")

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
        # No "@" -> whole string is prefix, suffix is missing (None), so two
        # malformed emails never match on a shared sentinel suffix.
        assert email_prefix_suffix("jsmith") == ("jsmith", None)
        assert email_prefix_suffix("a@b@c") == ("ab", "c")

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

    def test_detect_language_unreliable_when_fasttext_disabled(self):
        # Reliability now requires fastText AND cld2 to agree. The suite disables
        # fastText (see conftest), so the two detectors cannot agree and
        # detection collapses to unreliable / "un" regardless of the text. (The
        # real-model agreement path is covered by the reconcile unit tests below.)
        text = "Genetic behavior of resistance to the beet cyst as a way to enchant"
        is_reliable, is_english, predicted_language = detect_language(text)
        assert is_reliable is False
        assert is_english is False
        assert predicted_language == "un"


def test_fasttext_model_lazy_load_is_thread_safe(monkeypatch):
    import s2and.text as text_module

    fake_model = object()
    load_calls = {"count": 0}
    load_calls_lock = threading.Lock()
    start_event = threading.Event()
    outputs: list[object | None] = []

    def _fake_load_model(_path: str):
        with load_calls_lock:
            load_calls["count"] += 1
        time.sleep(0.05)
        return fake_model

    def _worker() -> None:
        start_event.wait(timeout=2.0)
        outputs.append(text_module._get_fasttext_model())

    monkeypatch.setattr(text_module.fasttext, "load_model", _fake_load_model)
    monkeypatch.setattr(text_module, "cached_path", lambda path: path)
    monkeypatch.setattr(text_module, "FASTTEXT_PATH", "dummy_model_path.bin")
    text_module.set_fasttext_loading_enabled(True)
    text_module._FASTTEXT_MODEL = None
    text_module._FASTTEXT_MODEL_INITIALIZED = False

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    start_event.set()
    for thread in threads:
        thread.join(timeout=3.0)

    assert load_calls["count"] == 1
    assert len(outputs) == 8
    assert all(model is fake_model for model in outputs)


def test_fasttext_skip_overrides_cached_model():
    import s2and.text as text_module

    text_module_any = cast(Any, text_module)
    text_module_any.set_fasttext_loading_enabled(True)
    text_module_any._FASTTEXT_MODEL = object()
    text_module_any._FASTTEXT_MODEL_INITIALIZED = True
    text_module_any.set_fasttext_loading_enabled(False)

    assert text_module_any._get_fasttext_model() is None
    assert text_module_any._FASTTEXT_MODEL is None


@pytest.mark.parametrize("skip_value", ["1", " 1 ", "TRUE ", " yes"])
def test_fasttext_skip_env_prevents_loading(monkeypatch, skip_value):
    import s2and.text as text_module

    text_module_any = cast(Any, text_module)
    load_calls = {"count": 0}

    def _fake_load_model(_path: str):
        load_calls["count"] += 1
        return object()

    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", skip_value)
    monkeypatch.setattr(text_module.fasttext, "load_model", _fake_load_model)
    monkeypatch.setattr(text_module, "cached_path", lambda path: path)
    text_module_any.set_fasttext_loading_enabled(True)
    text_module_any._FASTTEXT_MODEL = object()
    text_module_any._FASTTEXT_MODEL_INITIALIZED = True

    assert text_module_any._get_fasttext_model() is None
    assert text_module_any._FASTTEXT_MODEL is None
    assert load_calls["count"] == 0


def test_fasttext_can_reenable_after_skip_env(monkeypatch):
    import s2and.text as text_module

    fake_model = object()
    load_calls = {"count": 0}

    def _fake_load_model(_path: str):
        load_calls["count"] += 1
        return fake_model

    monkeypatch.setattr(text_module.fasttext, "load_model", _fake_load_model)
    monkeypatch.setattr(text_module, "cached_path", lambda path: path)
    monkeypatch.setattr(text_module, "FASTTEXT_PATH", "dummy_model_path.bin")
    text_module.set_fasttext_loading_enabled(True)
    text_module._FASTTEXT_MODEL = None
    text_module._FASTTEXT_MODEL_INITIALIZED = False

    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")
    assert text_module._get_fasttext_model() is None

    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "0")
    text_module.set_fasttext_loading_enabled(True)

    assert text_module._get_fasttext_model() is fake_model
    assert load_calls["count"] == 1


def test_fasttext_enable_preserves_loaded_model(monkeypatch):
    import s2and.text as text_module

    fake_model = object()
    load_calls = {"count": 0}

    def _fake_load_model(_path: str):
        load_calls["count"] += 1
        return fake_model

    monkeypatch.setattr(text_module.fasttext, "load_model", _fake_load_model)
    monkeypatch.setattr(text_module, "cached_path", lambda path: path)
    monkeypatch.setattr(text_module, "FASTTEXT_PATH", "dummy_model_path.bin")
    text_module.set_fasttext_loading_enabled(True)
    text_module._FASTTEXT_MODEL = None
    text_module._FASTTEXT_MODEL_INITIALIZED = False

    assert text_module._get_fasttext_model() is fake_model
    text_module.set_fasttext_loading_enabled(True)

    assert text_module._get_fasttext_model() is fake_model
    assert load_calls["count"] == 1


def test_fasttext_failed_load_raises(monkeypatch):
    import s2and.text as text_module

    load_calls = {"count": 0}

    def _raise_os_error(_path: str):
        load_calls["count"] += 1
        raise OSError("missing model")

    monkeypatch.setattr(text_module.fasttext, "load_model", _raise_os_error)
    monkeypatch.setattr(text_module, "cached_path", lambda path: path)
    monkeypatch.delenv("S2AND_SKIP_FASTTEXT", raising=False)
    text_module.set_fasttext_loading_enabled(True)
    text_module._FASTTEXT_MODEL = None
    text_module._FASTTEXT_MODEL_INITIALIZED = False
    text_module._FASTTEXT_LOAD_FAILED = False

    # fastText is mandatory in production: a load failure must raise, not
    # silently degrade to a cld2-only path.
    with pytest.raises(RuntimeError, match="fastText language model is required"):
        text_module._get_fasttext_model()
    # The failure is not cached as "initialized", so a later call re-attempts
    # the load and raises again (rather than returning a silent None).
    with pytest.raises(RuntimeError, match="fastText language model is required"):
        text_module._get_fasttext_model()
    assert load_calls["count"] == 2


def test_fasttext_unexpected_load_error_propagates(monkeypatch):
    import s2and.text as text_module

    def _raise_type_error(_path: str):
        raise TypeError("bad monkeypatch")

    monkeypatch.setattr(text_module.fasttext, "load_model", _raise_type_error)
    monkeypatch.setattr(text_module, "cached_path", lambda path: path)
    monkeypatch.delenv("S2AND_SKIP_FASTTEXT", raising=False)
    text_module.set_fasttext_loading_enabled(True)
    text_module._FASTTEXT_MODEL = None
    text_module._FASTTEXT_MODEL_INITIALIZED = False

    with pytest.raises(TypeError, match="bad monkeypatch"):
        text_module._get_fasttext_model()


def test_cld2_unexpected_error_propagates(monkeypatch):
    import s2and.text as text_module

    monkeypatch.setenv("S2AND_SKIP_FASTTEXT", "1")

    def _raise_type_error(_text: str):
        raise TypeError("bad cld2 state")

    monkeypatch.setattr(text_module.cld2, "detect", _raise_type_error)

    with pytest.raises(TypeError, match="bad cld2 state"):
        detect_language("hello world")


# reconcile_detected_languages truth table (mirrored by the Rust unit tests in
# s2and_rust/src/language_detection.rs::reconcile_tests). is_reliable is True
# only when both detectors return a concrete language AND agree.
def test_reconcile_detected_languages_agreement_is_reliable():
    assert reconcile_detected_languages("en", "en") == ("en", True)
    assert reconcile_detected_languages("fr", "fr") == ("fr", True)


def test_reconcile_detected_languages_disagreement_is_unknown():
    assert reconcile_detected_languages("en", "fr") == ("un", False)


def test_reconcile_detected_languages_single_detector_is_unknown():
    # Only cld2 responded (fastText unknown, e.g. disabled during tests).
    assert reconcile_detected_languages("un_ft", "en") == ("un", False)
    # Only fastText responded (cld2 unknown/failed).
    assert reconcile_detected_languages("en", "un_2") == ("un", False)


def test_reconcile_detected_languages_both_unknown_is_unknown():
    assert reconcile_detected_languages("un_ft", "un_2") == ("un", False)

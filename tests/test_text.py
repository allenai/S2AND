import unittest
import random
import numpy as np
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity

from s2and.text import normalize_text, name_text_features, cosine_sim, get_text_ngrams, get_text_ngrams_words, equal, equal_middle, equal_initial, counter_jaccard, jaccard, compute_block, diff, name_counts, detect_language
from s2and.consts import NUMPY_NAN
from s2and.data import NameCounts


class TestClusterer(unittest.TestCase):
    def test_normalize_text(self):
        assert "" == normalize_text(None)
        assert "" == normalize_text("")
        assert "text" == normalize_text("TeXt")
        assert "te han zi xt" == normalize_text("te'漢字xt")
        assert "text" == normalize_text("te'xt", True)

    def test_name_similarity_features(self):
        a = [NUMPY_NAN] * 4
        b = name_text_features("", None)
        assert [NUMPY_NAN] * 4 == name_text_features("", None)
        assert [0.0, 0.0, 0.0, 1.0] == name_text_features("text", "text")
        assert all([s >= 0.0 and s <= 1.0 for s in name_text_features("textual", "txt")])

    def test_cosine_sim(self):
        random_vec_1 = np.array([random.uniform(-1000, 1000) for i in range(1000)])
        random_vec_2 = np.array([random.uniform(-1000, 1000) for i in range(1000)])
        self.assertAlmostEqual(
            cosine_sim(random_vec_1, random_vec_2),
            cosine_similarity(random_vec_1.reshape(1, -1), random_vec_2.reshape(1, -1))[0][0],
        )
        assert cosine_sim([0] * 1000, random_vec_2) == 0

    def test_get_text_ngrams(self):
        assert Counter() == get_text_ngrams(None)
        assert Counter() == get_text_ngrams("the")
        assert Counter(["hell", "ello", "hel", "ell", "llo", "he", "el", "ll", "lo", "wor", "wo", "or"]) == get_text_ngrams(
            "hello wor"
        )
        assert Counter(
            ["hell", "ello", "hel", "ell", "llo", "he", "el", "ll", "lo", "wor", "wo", "or", "h", "e", "l", "l", "o", "w", "o", "r"]
        ) == get_text_ngrams("hello wor", use_unigrams=True)

    def test_get_text_ngrams_words(self):
        assert Counter() == get_text_ngrams_words(None)
        assert Counter() == get_text_ngrams_words("the")
        assert Counter(["quick green fox", "green fox jumped", "quick green", "green fox", "fox jumped", "quick", "green", "fox", "jumped"]) == get_text_ngrams_words("the quick green fox jumped")

    def test_equal(self):
        assert np.isnan(equal(None, None))
        assert np.isnan(equal("", ""))
        assert np.isnan(equal("-", "text"))
        assert 1 == equal("text", "text")
        assert 0 == equal("text", "hi")

    def test_equal_middle(self):
        assert np.isnan(equal_middle(None, None))
        assert np.isnan(equal_middle("", ""))
        assert np.isnan(equal_middle("a", ""))
        assert 0 == equal_middle("a", "b")
        assert 1 == equal_middle("a", "a")
        assert 1 == equal_middle("a", "as")
        assert 0 == equal_middle("as", "af")
        assert 1 == equal_middle("as", "as")

    def test_equal_initial(self):
        assert np.isnan(equal_initial(None, None))
        assert np.isnan(equal_initial("", ""))
        assert np.isnan(equal_initial("a", ""))
        assert 0 == equal_initial("a", "b")
        assert 1 == equal_initial("a", "a")
        assert 1 == equal_initial("a", "as")

    def test_counter_jaccard(self):
        assert np.isnan(counter_jaccard(Counter(), Counter()))
        self.assertAlmostEqual(4/6, counter_jaccard(Counter([1,2,3,4,5]), Counter([1,2,3,4,6])))
        self.assertAlmostEqual(4/7, counter_jaccard(Counter([1,2,3,4,5,5]), Counter([1,2,3,4,6])))


    def test_jaccard(self):
        assert np.isnan(jaccard({}, {}))
        self.assertAlmostEqual(4/6, jaccard({1,2,3,4,5}, {1,2,3,4,6}))
        self.assertAlmostEqual(4/6, jaccard({1,2,3,4,5,5}, {1,2,3,4,6}))

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

    def test_detect_language(self):
        is_reliable, is_english, predicted_language = detect_language("Genetic behavior of resistance to the beet cyst as a way to enchant")
        assert is_reliable is True
        assert is_english is True
        assert predicted_language == 'en'
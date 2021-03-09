import unittest
import pytest

from s2and.data import ANDData
from s2and.s2_funcs import (
    year_gap_is_small,
    affiliation_fuzzy_match,
    trusted_ids_are_compatible,
    emails_match_exactly,
    trusted_ids_match_exactly,
    names_are_compatible,
)


class TestData(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.dummy_dataset = ANDData(
            "tests/dummy/signatures.json",
            "tests/dummy/papers.json",
            clusters="tests/dummy/clusters.json",
            name="dummy",
            load_name_counts=False,
        )

    def test_year_gap_is_small(self):
        predicted = year_gap_is_small(["0", "1"], ["4", "3"], self.dummy_dataset)
        expected = True
        assert predicted == expected

        predicted = year_gap_is_small(["0"], ["6"], self.dummy_dataset)
        expected = False
        assert predicted == expected

    def test_affiliation_fuzzy_match(self):
        predicted = affiliation_fuzzy_match(["0", "1"], ["4", "3"], self.dummy_dataset)
        expected = 0.25
        assert predicted == expected

        predicted = affiliation_fuzzy_match(["0"], ["6"], self.dummy_dataset)
        expected = 1 / 3
        assert predicted == expected

    def test_trusted_ids_are_compatible(self):
        predicted = trusted_ids_are_compatible(["0", "1", "2"], ["3", "4"], self.dummy_dataset)
        expected = True
        assert predicted == expected

        predicted = trusted_ids_are_compatible(["0", "2"], ["3", "5"], self.dummy_dataset)
        expected = False
        assert predicted == expected

    def test_trusted_ids_match_exactly(self):
        predicted = trusted_ids_match_exactly(["2"], ["3"], self.dummy_dataset)
        expected = True
        assert predicted == expected

        predicted = trusted_ids_match_exactly(["2", "1"], ["3"], self.dummy_dataset)
        expected = False
        assert predicted == expected

        predicted = trusted_ids_match_exactly(["0"], ["2"], self.dummy_dataset)
        expected = False
        assert predicted == expected

    def test_emails_match_exactly(self):
        predicted = emails_match_exactly(["4"], ["5"], self.dummy_dataset)
        expected = True
        assert predicted == expected

        predicted = emails_match_exactly(["5"], ["6"], self.dummy_dataset)
        expected = False
        assert predicted == expected

    def test_names_are_compatible(self):
        predicted = names_are_compatible(["1"], ["2"], self.dummy_dataset)
        expected = True
        assert predicted == expected

        predicted = names_are_compatible(["0"], ["3"], self.dummy_dataset)
        expected = False
        assert predicted == expected

        predicted = names_are_compatible(["6"], ["7"], self.dummy_dataset)
        expected = True
        assert predicted == expected

        predicted = names_are_compatible(["6"], ["8"], self.dummy_dataset)
        expected = False
        assert predicted == expected

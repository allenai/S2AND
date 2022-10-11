import unittest

from s2and.eval import b3_precision_recall_fscore
from s2and.consts import ORPHAN_CLUSTER_KEY


class TestClusterer(unittest.TestCase):
    def test_b3_precision_recall_score(self):
        cluster_homogeneity_1 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7, 9, 12, 13, 14], "line": [8], "purple": [10], "spotted": [11]},
            {"1": [1, 2, 3, 4], "2": [5, 6, 7], "3": [8, 9, 10, 11, 12, 13, 14]},
        )
        self.assertAlmostEqual(cluster_homogeneity_1[0], 0.599)
        self.assertAlmostEqual(cluster_homogeneity_1[1], 0.695)
        self.assertAlmostEqual(cluster_homogeneity_1[2], 0.643)

        cluster_homogeneity_2 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7, 9, 12, 13, 14], "line": [8], "purple": [10], "spotted": [11]},
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9, 10, 11, 12, 13, 14]},
        )
        self.assertAlmostEqual(cluster_homogeneity_2[0], 0.694)
        self.assertAlmostEqual(cluster_homogeneity_2[1], 0.695)
        self.assertAlmostEqual(cluster_homogeneity_2[2], 0.695)

        size_v_quantity_1 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7], "line1": [8, 9], "line2": [10, 11], "line3": [12, 13]},
            {"1": [1, 2, 3, 4, 5], "2": [6], "3": [7], "4": [8], "5": [9], "6": [10], "7": [11], "8": [12], "9": [13]},
        )
        self.assertAlmostEqual(size_v_quantity_1[0], 1)
        self.assertAlmostEqual(size_v_quantity_1[1], 0.692)
        self.assertAlmostEqual(size_v_quantity_1[2], 0.818)

        size_v_quantity_2 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7], "line1": [8, 9], "line2": [10, 11], "line3": [12, 13]},
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9], "5": [10, 11], "6": [12, 13]},
        )
        self.assertAlmostEqual(size_v_quantity_2[0], 1)
        self.assertAlmostEqual(size_v_quantity_2[1], 0.877)
        self.assertAlmostEqual(size_v_quantity_2[2], 0.934)
        
        size_v_quantity_3 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7], "line1": [8, 9], "line2": [10, 11], "line3": [12, 13], 'x_' + ORPHAN_CLUSTER_KEY: [14]},
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9], "5": [10, 11], "6": [12, 13], "7": [14]},
        )
        # nothing changes in the output since the orphan goes to its own cluster
        self.assertEqual(size_v_quantity_3, size_v_quantity_2)
        
        size_v_quantity_4 = b3_precision_recall_fscore(
            {"dark": [1, 2, 3, 4, 5], "light": [6, 7], "line1": [8, 9], "line2": [10, 11], "line3": [12, 13], 'x_' + ORPHAN_CLUSTER_KEY: [14]},
            {"1": [1, 2, 3, 4], "2": [5], "3": [6, 7], "4": [8, 9], "5": [10, 11], "6": [12, 13, 14]},
        )
        # doesn't matter where the orphan goes, we ignore it during B3 construction
        self.assertAlmostEqual(size_v_quantity_4, size_v_quantity_3)
    
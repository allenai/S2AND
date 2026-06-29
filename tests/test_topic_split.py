import unittest
from types import SimpleNamespace

import numpy as np

from s2and.topic_split import coauthor_corroborated_split


def _make_dataset(group_a_coauthors, group_b_coauthors, *, with_embeddings=True, dim=8):
    """Build a stub dataset with two embedding-separated topic groups (10 sigs each)."""
    rng = np.random.RandomState(0)
    signatures = {}
    embeddings = {}
    # Two well-separated topic clouds: one around axis 0, the other around axis 1.
    for i in range(10):
        sid = f"a{i}"
        vec = np.zeros(dim)
        vec[0] = 5.0 + rng.rand()
        vec[2:] = rng.rand(dim - 2) * 0.01
        signatures[sid] = SimpleNamespace(paper_id=i, author_info_coauthor_blocks=set(group_a_coauthors))
        embeddings[str(i)] = vec.tolist()
    for i in range(10, 20):
        sid = f"b{i}"
        vec = np.zeros(dim)
        vec[1] = 5.0 + rng.rand()
        vec[2:] = rng.rand(dim - 2) * 0.01
        signatures[sid] = SimpleNamespace(paper_id=i, author_info_coauthor_blocks=set(group_b_coauthors))
        embeddings[str(i)] = vec.tolist()
    return SimpleNamespace(
        signatures=signatures,
        specter_embeddings=(embeddings if with_embeddings else None),
    )

ALL_SIGS = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10, 20)]


class TestCoauthorCorroboratedSplit(unittest.TestCase):
    def test_splits_when_topics_distinct_and_coauthors_disjoint(self):
        ds = _make_dataset({"x dis", "y joint"}, {"z far", "w other"})
        out = coauthor_corroborated_split({"c0": list(ALL_SIGS)}, ds)
        self.assertEqual(len(out), 2)
        # each output cluster is internally one topic group (all 'a' or all 'b')
        for sigs in out.values():
            prefixes = {s[0] for s in sigs}
            self.assertEqual(len(prefixes), 1)

    def test_no_split_when_coauthors_overlap(self):
        shared = {"s shared", "t shared", "u shared", "v shared"}
        ds = _make_dataset(shared, shared)
        out = coauthor_corroborated_split({"c0": list(ALL_SIGS)}, ds)
        self.assertEqual(len(out), 1)
        self.assertEqual(sorted(out["c0"]), sorted(ALL_SIGS))

    def test_no_split_when_below_min_embedded(self):
        ds = _make_dataset({"x"}, {"z"})
        small = ALL_SIGS[:6]
        out = coauthor_corroborated_split({"c0": small}, ds, min_embedded=10)
        self.assertEqual(len(out), 1)

    def test_passthrough_when_no_embeddings(self):
        ds = _make_dataset({"x"}, {"z"}, with_embeddings=False)
        clusters = {"c0": list(ALL_SIGS)}
        out = coauthor_corroborated_split(clusters, ds)
        self.assertEqual(out, clusters)

    def test_never_drops_signatures(self):
        ds = _make_dataset({"x dis"}, {"z far"})
        out = coauthor_corroborated_split({"c0": list(ALL_SIGS)}, ds)
        recovered = [s for sigs in out.values() for s in sigs]
        self.assertEqual(sorted(recovered), sorted(ALL_SIGS))


if __name__ == "__main__":
    unittest.main()

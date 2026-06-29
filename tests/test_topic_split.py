import unittest
from types import SimpleNamespace

import numpy as np

from s2and.topic_split import coauthor_corroborated_split


def _make_dataset(group_a_coauthors, group_b_coauthors, *, with_embeddings=True, dim=8):
    """Two embedding-separated topic groups (10 sigs each) with given coauthor sets."""
    rng = np.random.RandomState(0)
    signatures = {}
    embeddings = {}
    for i in range(10):
        vec = np.zeros(dim)
        vec[0] = 5.0 + rng.rand()
        vec[2:] = rng.rand(dim - 2) * 0.01
        signatures[f"a{i}"] = SimpleNamespace(paper_id=i, author_info_coauthors=set(group_a_coauthors))
        embeddings[str(i)] = vec.tolist()
    for i in range(10, 20):
        vec = np.zeros(dim)
        vec[1] = 5.0 + rng.rand()
        vec[2:] = rng.rand(dim - 2) * 0.01
        signatures[f"b{i}"] = SimpleNamespace(paper_id=i, author_info_coauthors=set(group_b_coauthors))
        embeddings[str(i)] = vec.tolist()
    return SimpleNamespace(signatures=signatures, specter_embeddings=(embeddings if with_embeddings else None))


ALL_SIGS = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10, 20)]


class TestCoauthorCorroboratedSplit(unittest.TestCase):
    def test_splits_when_topics_distinct_and_coauthors_disjoint(self):
        ds = _make_dataset({"alice smith", "bob jones"}, {"carol white", "dan brown"})
        out = coauthor_corroborated_split({"c0": list(ALL_SIGS)}, ds)
        self.assertEqual(len(out), 2)
        for sigs in out.values():
            self.assertEqual(len({s[0] for s in sigs}), 1)  # each side is one topic group

    def test_no_split_when_coauthors_shared(self):
        shared = {"alice smith", "bob jones"}
        ds = _make_dataset(shared, shared)
        out = coauthor_corroborated_split({"c0": list(ALL_SIGS)}, ds)
        self.assertEqual(len(out), 1)

    def test_single_shared_coauthor_does_not_remerge(self):
        # remerge requires >= 2 shared; a single common-name collision must not glue them
        ds = _make_dataset({"j lee", "alice smith"}, {"j lee", "carol white"})
        out = coauthor_corroborated_split({"c0": list(ALL_SIGS)}, ds)
        self.assertEqual(len(out), 2)

    def test_no_split_below_min_cluster(self):
        ds = _make_dataset({"alice"}, {"carol"})
        out = coauthor_corroborated_split({"c0": ALL_SIGS[:6]}, ds, min_cluster=8)
        self.assertEqual(len(out), 1)

    def test_passthrough_when_no_embeddings(self):
        ds = _make_dataset({"alice"}, {"carol"}, with_embeddings=False)
        clusters = {"c0": list(ALL_SIGS)}
        self.assertEqual(coauthor_corroborated_split(clusters, ds), clusters)

    def test_never_drops_signatures(self):
        ds = _make_dataset({"alice smith", "bob jones"}, {"carol white", "dan brown"})
        out = coauthor_corroborated_split({"c0": list(ALL_SIGS)}, ds)
        recovered = [s for sigs in out.values() for s in sigs]
        self.assertEqual(sorted(recovered), sorted(ALL_SIGS))


if __name__ == "__main__":
    unittest.main()

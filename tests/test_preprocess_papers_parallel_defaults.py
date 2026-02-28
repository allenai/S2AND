from __future__ import annotations

from s2and import data
from s2and.data import Author, Paper, preprocess_papers_parallel
from s2and.text import normalize_text


def _make_paper(*, paper_id: int, title: str, venue: str, journal: str, references: list[int] | None) -> Paper:
    return Paper(
        title=title,
        has_abstract=False,
        in_signatures=False,  # keep tests light; avoids language detection + extra ngram features
        is_english=None,
        is_reliable=None,
        predicted_language=None,
        title_ngrams_words=None,
        authors=[Author(author_name="John Smith", position=0)],
        venue=venue,
        journal_name=journal,
        title_ngrams_chars=None,
        venue_ngrams=None,
        journal_ngrams=None,
        reference_details=None,
        year=2020,
        references=references,
        paper_id=paper_id,
    )


def test_preprocess_papers_parallel_windows_defaults_to_serial(monkeypatch):
    monkeypatch.setattr(data.platform, "system", lambda: "Windows")

    def _unexpected_pool_use(*_args, **_kwargs):
        raise AssertionError("UniversalPool should not be instantiated on Windows by default")

    monkeypatch.setattr(data, "UniversalPool", _unexpected_pool_use)

    papers = {
        "1": _make_paper(paper_id=1, title="Some Title", venue="My Venue", journal="My Journal", references=None),
        "2": _make_paper(paper_id=2, title="Another Title", venue="Venue 2", journal="Journal 2", references=None),
    }

    out = preprocess_papers_parallel(papers, n_jobs=8, preprocess=False)

    assert out["1"].title == normalize_text("Some Title")
    assert out["1"].venue == "My Venue"
    assert out["1"].journal_name == "My Journal"


def test_preprocess_papers_parallel_linux_uses_pool_only_for_stage_1(monkeypatch):
    monkeypatch.setattr(data.platform, "system", lambda: "Linux")

    class FakeUniversalPool:
        init_calls = 0
        imap_calls = 0

        def __init__(self, processes: int | None = None, use_threads: bool | None = None):
            type(self).init_calls += 1
            self.processes = processes
            self.use_threads = use_threads

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap(self, func, iterable, chunksize=1, max_prefetch=4):
            type(self).imap_calls += 1
            for item in iterable:
                yield func(item)

    monkeypatch.setattr(data, "UniversalPool", FakeUniversalPool)

    papers = {
        "1": _make_paper(paper_id=1, title="Paper 1", venue="Venue 1", journal="Journal 1", references=[2]),
        "2": _make_paper(paper_id=2, title="Paper 2", venue="Venue 2", journal="Journal 2", references=None),
    }

    out = preprocess_papers_parallel(papers, n_jobs=2, preprocess=True, compute_reference_features=True)

    assert FakeUniversalPool.init_calls == 1
    assert FakeUniversalPool.imap_calls == 1
    assert out["1"].reference_details is not None
    assert out["2"].reference_details is not None

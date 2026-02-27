import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from s2and.data import Author, Paper, preprocess_paper_1
from s2and.text import normalize_text


def test_preprocess_paper_1_spawn_respects_preprocess_flag():
    # Using spawn context reproduces Windows-style multiprocessing semantics on any OS.
    ctx = mp.get_context("spawn")

    paper = Paper(
        title="Some Title",
        has_abstract=False,
        in_signatures=False,  # keep this test light; avoids language detection + ngram features
        is_english=None,
        is_reliable=None,
        predicted_language=None,
        title_ngrams_words=None,
        authors=[Author(author_name="John Smith", position=0)],
        venue="My Venue",
        journal_name="My Journal",
        title_ngrams_chars=None,
        venue_ngrams=None,
        journal_ngrams=None,
        reference_details=None,
        year=2020,
        references=None,
        paper_id=1,
    )
    item = ("1", paper)

    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        _, out_false = ex.submit(preprocess_paper_1, item, preprocess=False).result(timeout=30)
        _, out_true = ex.submit(preprocess_paper_1, item, preprocess=True).result(timeout=30)

    assert out_false.venue == "My Venue"
    assert out_false.journal_name == "My Journal"

    assert out_true.venue == normalize_text("My Venue")
    assert out_true.journal_name == normalize_text("My Journal")

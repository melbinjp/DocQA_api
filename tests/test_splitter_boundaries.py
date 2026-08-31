"""The splitter's job is to not cut a fact in half.

The old 500-character window did exactly that, and it cost correct answers:
Table 3 of *Attention Is All You Need* puts its column headers and its row
values far enough apart that no 500-character chunk held both, so the API said
it had no information about the big model's parameter count while the number
sat in the document.
"""

from utils.splitter import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP,
    split_pages,
    split_text,
)


def test_chunks_are_large_enough_to_hold_a_table():
    """The regression that started this. 500 was too small; guard the floor."""
    assert DEFAULT_MAX_CHARS >= 1000
    assert DEFAULT_OVERLAP < DEFAULT_MAX_CHARS


def test_a_short_document_is_one_chunk():
    assert split_text("Short enough to stay whole.") == ["Short enough to stay whole."]


def test_empty_and_whitespace_produce_nothing():
    assert split_text("") == []
    assert split_text("   \n\n  \t ") == []


def test_chunks_respect_the_hard_limit():
    text = " ".join(f"word{i}" for i in range(4000))
    for chunk in split_text(text, max_chars=300, overlap=50):
        assert len(chunk) <= 300


def test_chunks_start_on_a_word_not_mid_token():
    text = " ".join(f"token{i}" for i in range(2000))
    chunks = split_text(text, max_chars=200, overlap=40)
    for chunk in chunks[1:]:
        assert chunk.startswith("token"), f"chunk begins mid-word: {chunk[:30]!r}"


def test_sentences_are_preferred_as_boundaries():
    sentence = "The quick brown fox jumped over the lazy dog. "
    chunks = split_text(sentence * 20, max_chars=300, overlap=50)
    # Every chunk but the last should end where a sentence ended.
    for chunk in chunks[:-1]:
        assert chunk.endswith("."), f"cut mid-sentence: {chunk[-40:]!r}"


def test_a_fact_split_by_a_boundary_survives_whole_somewhere():
    """What overlap is actually for."""
    filler = "padding text here. " * 40
    fact = "The big model has 213 million parameters and the base model has 65."
    text = filler + fact + " " + filler
    chunks = split_text(text, max_chars=400, overlap=120)
    assert any(fact in c for c in chunks), "the fact was cut and appears in no chunk whole"


def test_no_unbounded_growth_on_text_with_no_breaks():
    """One long run of characters must still terminate, and must still advance."""
    chunks = split_text("x" * 5000, max_chars=400, overlap=100)
    assert len(chunks) < 40
    assert "".join(chunks).count("x") >= 5000


def test_paragraph_breaks_are_kept_for_use_as_boundaries():
    """The old splitter flattened every newline, throwing away the best break."""
    chunks = split_text("First para.\n\nSecond para.", max_chars=DEFAULT_MAX_CHARS)
    assert "\n\n" in chunks[0]


def test_pages_do_not_bleed_into_each_other():
    pages = [(1, "Page one text. " * 30), (2, "Page two text. " * 30)]
    out = split_pages(pages, max_chars=200, overlap=40)
    for item in out:
        if "one" in item["text"]:
            assert item["page"] == 1
        if "two" in item["text"]:
            assert item["page"] == 2
    assert {i["page"] for i in out} == {1, 2}


def test_overlap_at_or_above_max_is_rejected_rather_than_looping():
    import pytest

    with pytest.raises(ValueError):
        split_text("some text", max_chars=100, overlap=100)

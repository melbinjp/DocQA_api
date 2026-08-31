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


# --- tables are one fact, and must not be cut into numbers with no names ---

TABLE = (
    "|model|layers|params|\n"
    "|---|---|---|\n"
    "|base|6|65|\n"
    "|big|6|213|\n"
)


def test_a_small_table_stays_in_one_chunk():
    """The regression: `params` and `213` ended up in different chunks, and the
    answer became 'the provided text does not contain information'."""
    out = split_pages([(9, "Some prose about results.\n" + TABLE)])
    table_chunks = [c["text"] for c in out if "|---|" in c["text"] or c["text"].lstrip().startswith("|")]
    assert len(table_chunks) == 1
    whole = table_chunks[0]
    assert "params" in whole and "213" in whole and "65" in whole


def test_a_table_too_big_for_one_chunk_repeats_its_header():
    rows = "".join(f"|row{i}|6|{i}|\n" for i in range(200))
    out = split_pages([(1, "|model|layers|params|\n|---|---|---|\n" + rows)], max_chars=400)
    table_chunks = [c["text"] for c in out if "|---|" in c["text"] or c["text"].lstrip().startswith("|")]
    assert len(table_chunks) > 1, "this table should have needed splitting"
    for chunk in table_chunks:
        assert "params" in chunk, "a row block lost its header, so its numbers are unnamed"


def test_prose_around_a_table_is_still_chunked_normally():
    prose = "This is a sentence about the results. " * 60
    out = split_pages([(3, prose + "\n" + TABLE + "\n" + prose)], max_chars=500)
    assert len([c for c in out if not "|---|" in c["text"] or c["text"].lstrip().startswith("|")]) > 1
    assert len([c for c in out if "|---|" in c["text"] or c["text"].lstrip().startswith("|")]) == 1
    assert {c["page"] for c in out} == {3}


def test_a_page_that_is_only_a_table_still_produces_a_chunk():
    out = split_pages([(2, TABLE)])
    assert len(out) == 1 and "213" in out[0]["text"] and out[0]["page"] == 2


def test_a_table_caption_is_not_severed_from_its_grid():
    """The caption is the only natural language a grid has.

    Measured on page 9 of Attention Is All You Need: with the caption filed as
    prose, the chunk holding `base 65` and `big 213` began `|Col1|train<br>N d d
    h` and was never retrieved at all for "how does the parameter count of the
    big model compare to the base model". The caption says "Unlisted values are
    identical to those of the base model", which is the language the question is
    asked in.
    """
    page = (
        "Some earlier prose that belongs to the page.\n"
        "\n"
        "Table 3: Variations on the architecture, against the base model.\n"
        "Columns: model, layers, params.\n"
        "|model|layers|params|\n"
        "|---|---|---|\n"
        "|base|6|65|\n"
        "|big|6|213|\n"
    )
    out = split_pages([(9, page)])
    holding = [c["text"] for c in out if "213" in c["text"]]
    assert len(holding) == 1
    chunk = holding[0]
    assert "Table 3" in chunk, "the caption was cut away from the grid"
    assert "base model" in chunk
    assert "params" in chunk and "65" in chunk


def test_a_long_paragraph_is_not_swallowed_as_a_caption():
    """Only the lines directly above a grid attach, and only a few of them."""
    prose = "\n".join(f"Sentence number {i} of ordinary prose." for i in range(20))
    out = split_pages([(1, prose + "\n|a|b|\n|---|---|\n|1|2|\n")], max_chars=2000)
    table = [c["text"] for c in out if "|---|" in c["text"]][0]
    assert "Sentence number 0" not in table, "the whole paragraph was pulled into the table"

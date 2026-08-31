"""Pages survive extraction, so a citation can name one.

The API returned `text`, `score`, `doc_id` and `source` and no page, because
`load_source` handed back one flat string: PyMuPDF's own per-page output was
joined with newlines before anything downstream could see it. Page identity was
destroyed at the first step, so no amount of work later could recover it.

The second half is the extractor order. MarkItDown ran first for PDFs and
PyMuPDF only when MarkItDown returned *nothing*, so a PDF that MarkItDown read
badly rather than not at all never reached the fallback. Measured on arXiv
2005.11401, MarkItDown returned the whole two-column document with the spaces
removed. Silent mangling is worse than failure: retrieval degrades, the answer
often still looks right, and the quoted source shown to a user is unreadable.
"""

import fitz
import pytest

from utils.loaders import load_source_pages
from utils.splitter import split_pages


def _pdf(pages: list[str]) -> bytes:
    """A real PDF, built with the same library the loader reads it back with."""
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 720), body, fontsize=11)
    raw = doc.tobytes()
    doc.close()
    return raw


def test_each_page_comes_back_separately_and_numbered_from_one():
    raw = _pdf(["Rent is payable fortnightly in advance.",
                "The bond is four weeks rent.",
                "Pets require written consent."])
    pages = load_source_pages(raw, "pdf")

    assert [n for n, _ in pages] == [1, 2, 3]
    assert "fortnightly" in pages[0][1]
    assert "bond" in pages[1][1]
    assert "consent" in pages[2][1]


def test_the_words_keep_their_spaces():
    """The regression that made citations unreadable rather than merely absent."""
    raw = _pdf(["The landlord must give sixty days written notice."])
    _, text = load_source_pages(raw, "pdf")[0]

    assert "written notice" in text
    assert not [w for w in text.split() if len(w) > 25], text


def test_every_chunk_knows_which_page_it_came_from():
    raw = _pdf(["Clause one about rent.", "Clause two about the bond."])
    chunks = split_pages(load_source_pages(raw, "pdf"))

    assert chunks, "the fixture produced no chunks"
    assert {c["page"] for c in chunks} == {1, 2}
    for c in chunks:
        assert c["page"] is not None
        assert c["text"].strip()


def test_a_chunk_never_straddles_a_page_boundary():
    """Splitting per page is the point. Over one joined string a chunk could carry
    text from two pages and be attributed to whichever came first, which is a
    citation that is confidently wrong."""
    raw = _pdf(["AAA " * 60, "BBB " * 60])
    chunks = split_pages(load_source_pages(raw, "pdf"))

    for c in chunks:
        assert not ("AAA" in c["text"] and "BBB" in c["text"]), c


def test_a_format_without_pages_says_so_rather_than_guessing():
    pages = load_source_pages(b"Plain text has no pages.", "txt")
    assert [n for n, _ in pages] == [None]
    assert split_pages(pages)[0]["page"] is None

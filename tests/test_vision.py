"""Selecting pages to look at, and folding what was read back in.

The rules that matter here are about not losing or corrupting the document:
a transcription is what a model believes a page says, and it must never quietly
replace what the PDF actually says. The exception is a page with no text, where
there is nothing to protect and without the transcription the page is gone.
"""
import fitz
import pytest

from utils.vision import (
    LOW_TEXT_THRESHOLD,
    MAX_VISION_PAGES,
    is_transcript,
    merge,
    pages_for_vision,
    wrap_transcript,
)


def make_pdf(page_texts):
    """`insert_text` draws one unwrapped line that runs off the page edge, so
    only the part inside the page is extractable. A text box wraps, which is
    what a page of real prose looks like."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def test_the_fixture_really_makes_a_dense_page():
    """Guard the fixture itself: an earlier version silently produced 97
    characters and made the dense-page test meaningless."""
    raw = make_pdf(["word " * 200])
    doc = fitz.open(stream=raw, filetype="pdf")
    assert len(doc[0].get_text().strip()) > LOW_TEXT_THRESHOLD
    doc.close()


def test_a_page_with_no_text_is_selected():
    raw = make_pdf(["", "x" * 400])
    chosen = {n: reason for n, _, reason in pages_for_vision(raw)}
    assert chosen.get(1) == "no-text"


def test_a_dense_page_with_nothing_on_it_is_left_alone():
    raw = make_pdf(["word " * 200])
    assert pages_for_vision(raw) == []


def test_pages_with_no_text_are_prioritised_over_the_rest():
    raw = make_pdf([""] * 3 + ["word " * 200] * 3)
    chosen = pages_for_vision(raw, max_pages=2)
    assert [reason for _, _, reason in chosen] == ["no-text", "no-text"]


def test_selection_is_capped():
    raw = make_pdf([""] * (MAX_VISION_PAGES + 6))
    assert len(pages_for_vision(raw)) == MAX_VISION_PAGES


def test_rendered_pages_come_back_as_images_in_page_order():
    raw = make_pdf(["", "", ""])
    chosen = pages_for_vision(raw)
    assert [n for n, _, _ in chosen] == [1, 2, 3]
    for _, png, _ in chosen:
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_bytes_that_are_not_a_pdf_select_nothing_rather_than_raising():
    assert pages_for_vision(b"not a pdf at all") == []
    assert pages_for_vision(b"") == []


def test_a_transcript_is_marked_as_one():
    assert is_transcript(wrap_transcript("|a|b|"))
    assert not is_transcript("ordinary extracted text")
    assert wrap_transcript("   ") == ""


def test_merging_keeps_the_documents_own_text():
    pages = [(1, "The real extracted text.")]
    out = dict(merge(pages, {1: "|col|col|"}))
    assert "The real extracted text." in out[1], "extraction was overwritten"
    assert "|col|col|" in out[1]


def test_a_page_that_had_no_text_is_created_from_the_transcript():
    """The scanned case. Without this the page does not exist at all."""
    out = dict(merge([], {3: "Transcribed from the image."}))
    assert out[3].endswith("Transcribed from the image.")
    assert is_transcript(out[3])


def test_an_empty_transcript_changes_nothing():
    pages = [(1, "Original.")]
    assert merge(pages, {1: "   "}) == [(1, "Original.")]
    assert merge(pages, {}) == [(1, "Original.")]


def test_pages_come_back_in_order_after_merging():
    out = merge([(5, "five"), (1, "one")], {3: "three"})
    assert [n for n, _ in out] == [1, 3, 5]


def test_the_low_text_threshold_is_small_enough_not_to_catch_real_pages():
    """A page of genuine prose must not be mistaken for a scan."""
    assert LOW_TEXT_THRESHOLD < 400

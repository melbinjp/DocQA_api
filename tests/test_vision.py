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
    for _, image, _ in chosen:
        assert image[:2] == b"\xff\xd8", "not a JPEG"
        # Keeps a page small enough that several in one ingest is not
        # itself the failure.
        assert len(image) < 2_000_000, f"{len(image)} bytes is too big to send"


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


@pytest.mark.asyncio
async def test_no_targets_returns_an_empty_result_and_no_errors():
    from utils.vision import transcribe_pages
    assert await transcribe_pages(None, "m", []) == ({}, [])


@pytest.mark.asyncio
async def test_a_failing_page_reports_why_instead_of_vanishing():
    """The first version swallowed these, every page failed, and the only
    symptom was a document that would not ingest for no stated reason."""
    from utils.vision import transcribe_pages

    class Boom:
        class aio:
            class models:
                @staticmethod
                async def generate_content(**kwargs):
                    raise RuntimeError("model refused the image")

    out, errors = await transcribe_pages(Boom(), "m", [(1, b"jpegbytes", "no-text")])
    assert out == {}
    assert errors and "model refused the image" in errors[0]


@pytest.mark.asyncio
async def test_the_reader_walks_the_model_ladder_on_failure():
    """A quota error on the first model is a queue, not a dead end.

    Measured 2026-09-01: every page came back "429 RESOURCE_EXHAUSTED" from the
    primary model while ordinary questions were still being answered fine on the
    same key, because the answering path falls back and the reader did not.
    """
    from utils.vision import transcribe_pages

    tried = []

    class Ladder:
        class aio:
            class models:
                @staticmethod
                async def generate_content(model=None, **kwargs):
                    tried.append(model)
                    if model == "first":
                        raise RuntimeError("429 RESOURCE_EXHAUSTED")
                    class R:
                        text = "transcribed by the second model"
                    return R()

    out, errors = await transcribe_pages(
        Ladder(), ["first", "second"], [(1, b"img", "no-text")], timeout=5
    )
    assert tried == ["first", "second"]
    assert out == {1: "transcribed by the second model"}
    assert errors == []


@pytest.mark.asyncio
async def test_a_single_model_name_still_works():
    from utils.vision import transcribe_pages

    class Ok:
        class aio:
            class models:
                @staticmethod
                async def generate_content(**kwargs):
                    class R:
                        text = "read"
                    return R()

    out, _ = await transcribe_pages(Ok(), "only-model", [(2, b"img", "no-text")])
    assert out == {2: "read"}


def test_by_default_only_scanned_pages_are_sent():
    """Tables and figures answered 16 of 16 from text, so spending a request on
    them buys nothing on the evidence available."""
    raw = make_pdf(["", "word " * 200])
    reasons = [r for _, _, r in pages_for_vision(raw)]
    assert reasons == ["no-text"]


def test_every_chunk_of_a_transcript_is_marked_not_just_the_first():
    """Marking the page and then splitting it marks one chunk in four.

    Measured on the live Space: a scanned page became four chunks and only the
    first carried "[Read from the page image]", so three of them were
    indistinguishable from the document's own words. The guarantee is that a
    reader can always tell, which means the marker belongs on the chunk.
    """
    from utils.splitter import split_pages

    long_transcript = ("The transcribed sentence repeats. " * 200)
    chunks = split_pages([(1, long_transcript)])
    assert len(chunks) > 1, "this fixture needs to produce several chunks"

    marked = [wrap_transcript(c["text"]) for c in chunks]
    assert all(is_transcript(m) for m in marked)
    assert all(m.count("[Read from the page image]") == 1 for m in marked)


def test_marking_a_chunk_does_not_change_what_it_says():
    body = "Row: base 6 512 2048 8 64 64 0.1 0.1 100K 4.92 25.8 65"
    assert body in wrap_transcript(body)


# --- a document is a scan, or it is not; bare pages inside a text document
#     are figures, and figures already answer from their captions ---

def test_a_text_document_with_a_few_bare_pages_is_not_a_scan():
    """The regression that made a 75-page paper stop ingesting at all.

    GPT-3 has figure pages with almost no text, so a per-page rule sent three of
    them to be read. Three pages against a three-model ladder with a long
    timeout on each rung turned a thirty second ingest into a 504.
    """
    raw = make_pdf(["word " * 200] * 8 + [""] * 2)
    assert pages_for_vision(raw) == []


def test_a_document_that_really_is_scanned_still_qualifies():
    raw = make_pdf([""] * 6 + ["word " * 200] * 2)
    assert len(pages_for_vision(raw)) == 6


def test_the_all_scope_ignores_the_document_level_gate():
    from utils.vision import SCOPE_ALL
    raw = make_pdf(["word " * 200] * 8 + [""] * 2)
    assert len(pages_for_vision(raw, scope=SCOPE_ALL)) == 2


@pytest.mark.asyncio
async def test_the_reader_gives_up_at_its_budget_rather_than_hanging():
    """Ingest happens inside one HTTP request. A request that never returns is
    worse than a document that is merely harder to search."""
    import asyncio
    from utils.vision import transcribe_pages

    class Slow:
        class aio:
            class models:
                @staticmethod
                async def generate_content(**kwargs):
                    await asyncio.sleep(30)

    out, errors = await transcribe_pages(
        Slow(), "m", [(1, b"img", "no-text")], timeout=20, budget=0.4
    )
    assert out == {}
    assert errors and "budget" in errors[0]

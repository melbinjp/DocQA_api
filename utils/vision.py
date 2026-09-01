"""Reading a page by looking at it, when reading its text is not enough.

Two things a text extractor cannot do, both measured on the live Space:

A scanned PDF fails outright. Rendering two pages of a paper to images and
putting only those images in a PDF produces a file with a zero-character text
layer, and ingesting it returns `400 Extraction resulted in no content`. That is
every photographed contract, scanned invoice and old report rejected at the
door, which is most of what a document arrives as outside a research library.

A chart is invisible. Questions about figures currently succeed by reading the
caption and the prose around it, never the picture, so anything the caption does
not say is unavailable. The same is true of any table `find_tables` mangles: on
page 9 of *Attention Is All You Need* it collapses fourteen columns into three.

So the page is rendered and shown to the model that is already answering
questions here, and what it reads is added to the text. Added, not substituted:
extracted text is what the PDF actually says, a transcription is what a model
believes it says, and the second must never quietly overwrite the first. The one
exception is a page with no text at all, where there is nothing to overwrite.
"""

import asyncio
import io

# Rendering resolution. 140 DPI keeps 8pt table digits legible and keeps a page
# small. Measured on page 9 of Attention Is All You Need: 341KB as PNG at 170
# DPI, 243KB as JPEG at 140. Both are modest, so size was never the reason the
# first version of this returned nothing on every page, and an earlier comment
# here claiming 8MB pages was wrong.
RENDER_DPI = 140

# JPEG rather than PNG, because a rendered page is a photograph of text and JPEG
# is the format for that. A third off, at a quality difference no reader sees.
RENDER_QUALITY = 82
RENDER_MIME = "image/jpeg"

# A page with less text than this, relative to nothing at all, is either scanned
# or almost entirely picture. 180 characters is roughly two lines: a page number
# and a running header, which is what a scanned page's text layer usually holds.
LOW_TEXT_THRESHOLD = 180

# Pages sent to the model per document, most valuable first. Every page is a
# request, so this is the ceiling on both wall-clock and quota for one ingest.
MAX_VISION_PAGES = 25

# Which pages are worth spending a request on.
#
# "scanned" is the default because that is where measurement found a real
# failure: a PDF with no text layer was rejected outright, while tables and
# figures already answered 16 of 16 questions from extracted text, captions and
# surrounding prose. Looking at every table and figure costs a request each and
# buys, on the evidence so far, nothing. "all" turns that on for anyone whose
# documents are worse than these papers, which many are.
SCOPE_SCANNED = "scanned"
SCOPE_ALL = "all"

# Concurrent vision requests. Enough to keep a long document moving, low enough
# not to trip rate limits on a free tier.
VISION_CONCURRENCY = 4

VISION_PROMPT = (
    "Transcribe what is printed on this page, exactly as it appears.\n\n"
    "Tables: reproduce every table as a markdown table. Keep every column "
    "separate and every row on its own line, and copy the numbers exactly. "
    "If a column has a header, put it in the header row. Getting the columns "
    "right matters more than anything else here.\n\n"
    "Charts, plots and diagrams: state the title, the axis labels and their "
    "ranges, the name of each series, and the values or the shape of each "
    "curve. If two curves cross or one is consistently above another, say so.\n\n"
    "Body text: transcribe it plainly.\n\n"
    "Do not summarise, do not explain, and do not add anything that is not "
    "visible on the page. If you cannot read something, leave it out rather "
    "than guessing at it."
)

# What the transcription is wrapped in, so a reader and the answering model can
# both tell it apart from the document's own text.
_HEADER = "[Read from the page image]"


def wrap_transcript(text: str) -> str:
    """Mark a transcription as one, so it is never mistaken for extracted text."""
    body = (text or "").strip()
    return f"{_HEADER}\n{body}" if body else ""


def is_transcript(text: str) -> bool:
    return text.lstrip().startswith(_HEADER)


def _page_needs_vision(page, text: str) -> str | None:
    """Why this page is worth looking at, or None."""
    stripped = (text or "").strip()
    if len(stripped) < LOW_TEXT_THRESHOLD:
        # Scanned, or a full-page figure. Nothing else can recover it.
        return "no-text"
    try:
        if page.get_images(full=False):
            return "image"
    except Exception:
        pass
    try:
        if page.find_tables().tables:
            return "table"
    except Exception:
        pass
    return None


# Most valuable first: a page with no text is unreadable without this, a table
# is readable but often wrong, a figure is merely invisible.
_PRIORITY = {"no-text": 0, "table": 1, "image": 2}


def pages_for_vision(raw: bytes, max_pages: int = MAX_VISION_PAGES,
                     scope: str = SCOPE_SCANNED):
    """`[(page_number, png_bytes, reason), ...]`, one-based, best first.

    Returns [] rather than raising if the bytes are not a PDF this can open;
    vision is an addition to ingestion and must never be the reason it fails.
    """
    try:
        import fitz
    except Exception:
        return []

    doc = None
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
        candidates = []
        for number in range(doc.page_count):
            page = doc[number]
            try:
                text = page.get_text()
            except Exception:
                text = ""
            reason = _page_needs_vision(page, text)
            if reason and (scope == SCOPE_ALL or reason == "no-text"):
                candidates.append((number, reason))

        candidates.sort(key=lambda c: (_PRIORITY.get(c[1], 9), c[0]))
        chosen = candidates[:max_pages]
        chosen.sort(key=lambda c: c[0])

        out = []
        for number, reason in chosen:
            try:
                pixmap = doc[number].get_pixmap(dpi=RENDER_DPI)
                out.append((number + 1, pixmap.tobytes("jpeg", jpg_quality=RENDER_QUALITY), reason))
            except Exception:
                continue
        return out
    except Exception:
        return []
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


async def transcribe_pages(client, model, targets, timeout: float = 90.0,
                           concurrency: int = VISION_CONCURRENCY):
    """Look at each rendered page. Returns `({page_number: transcript}, errors)`.

    A page that fails is simply absent from the result. One unreadable page must
    not cost the document.

    The errors come back rather than being swallowed because the first version of
    this swallowed them, every page failed, and the only symptom was a document
    that would not ingest for no stated reason. A failure nobody can see is worse
    than the failure itself.
    """
    if not targets:
        return {}, []
    from google.genai import types

    # A quota error on the first model is not a failure, it is a queue. The
    # answering path has walked a ladder of models on 429 since before this
    # existed; the reader had none and died on the first one. Measured
    # 2026-09-01: every page came back "429 RESOURCE_EXHAUSTED" while ordinary
    # questions were still being answered fine on the same key.
    models = [model] if isinstance(model, str) else list(model)

    semaphore = asyncio.Semaphore(concurrency)

    async def attempt(page_number: int, png: bytes, model_name: str):
        return await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_bytes(data=png, mime_type=RENDER_MIME),
                                    types.Part.from_text(text=VISION_PROMPT),
                                ],
                            )
                        ],
                    ),
                    timeout=timeout,
                )

    async def one(page_number: int, png: bytes):
        async with semaphore:
            last_error = None
            for position, model_name in enumerate(models):
                try:
                    response = await attempt(page_number, png, model_name)
                    return page_number, (response.text or "").strip(), None
                except Exception as e:
                    last_error = f"{type(e).__name__}: {str(e)[:160]}"
                    # Back off before the next model, so a burst of pages does
                    # not walk the whole ladder at once and exhaust it too.
                    if position + 1 < len(models):
                        await asyncio.sleep(1.0 * (position + 1))
            return page_number, "", last_error

    results = await asyncio.gather(
        *(one(number, png) for number, png, _ in targets),
        return_exceptions=True,
    )

    out, errors = {}, []
    for item in results:
        if isinstance(item, Exception):
            errors.append(f"{type(item).__name__}: {str(item)[:160]}")
            continue
        page_number, text, error = item
        if text:
            out[page_number] = text
        elif error:
            errors.append(error)
    return out, errors


def merge(pages, transcripts: dict):
    """Fold transcripts into `[(page, text), ...]`, keeping both.

    A page that had text keeps it and gains the transcription underneath. A page
    that had none, which is what a scanned page looks like, is created from the
    transcription: there is nothing to protect and without it the page does not
    exist at all.
    """
    merged = {number: text for number, text in pages}
    for number, transcript in transcripts.items():
        wrapped = wrap_transcript(transcript)
        if not wrapped:
            continue
        existing = merged.get(number, "").strip()
        merged[number] = f"{existing}\n\n{wrapped}" if existing else wrapped
    return [(number, merged[number]) for number in sorted(merged)
            if merged[number] and merged[number].strip()]

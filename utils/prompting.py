"""Turning retrieved chunks into the text the model is asked to read.

Kept out of `app.py` so it can be tested without importing the embedding model.
"""


def short_source(source: str) -> str:
    """A source a reader would recognise: a filename, not a URL with a query."""
    if not source:
        return "document"
    name = source.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
    return name or source


def label_chunk(chunk: dict) -> str:
    """One retrieved excerpt, headed by the document and page it came from."""
    name = short_source(chunk.get("source") or "")
    page = chunk.get("page")
    where = f"{name}, page {page}" if page is not None else name
    return f"[From {where}]\n{chunk['text']}"


# The manifest has a total budget, split across whatever is loaded, rather than
# a fixed size per document.
#
# It was 260 characters each, which on an academic paper is the title and then a
# wall of author names. Measured 2026-09-01 on the GPT-3 paper: 260 characters
# ends inside the author list and "175 billion parameters" is at character 1266.
# Asked which of two models has more parameters, the answer had BERT's 340
# million from a retrieved chunk and nothing for GPT-3, five times out of five,
# because the question embeds like benchmark prose and pulls result pages rather
# than the abstract. The manifest is what should answer that and was too short.
#
# Raising it to a fixed 900 was my own arithmetic error: 900 is less than 1266,
# so the fix could not reach the fact that motivated it. A budget divided among
# the documents gets that right without a number that is wrong at some other
# document count: two documents get 2000 characters each and see a full opening
# chunk, eight get 500 and still clear an author block.
MANIFEST_TOTAL_CHARS = 4000

# No document gets less than this, however many are loaded.
MANIFEST_MIN_CHARS = 300

# And none needs more than one chunk's worth, since that is all there is.
MANIFEST_MAX_CHARS = 1600


def manifest_budget(document_count: int) -> int:
    """Characters of opening to show for each of `document_count` documents."""
    if document_count <= 0:
        return MANIFEST_MIN_CHARS
    share = MANIFEST_TOTAL_CHARS // document_count
    return max(MANIFEST_MIN_CHARS, min(MANIFEST_MAX_CHARS, share))


def build_manifest(documents) -> str:
    """A list of what is loaded, for questions about the corpus rather than in it.

    `documents` is an iterable of `(source, first_chunk_text)`.

    Retrieval answers questions asked *of* a document. It cannot answer one
    asked *about* the set of them, because the best-matching chunks are not the
    ones that say what each document is. Measured 2026-09-01 with eight
    documents loaded: asked which of the loaded papers was about image
    recognition, retrieval returned ResNet's reference-list page, which is
    denser in the phrase "image recognition" than the paper's own abstract, and
    the answer listed seven of ResNet's citations instead of naming ResNet.
    Telling the model not to mistake a bibliography for a document did not help,
    because a bibliography was all it had been given.

    So the model is always shown what is loaded, whatever retrieval returns.
    """
    documents = list(documents)
    budget = manifest_budget(len(documents))
    lines = []
    for source, opening in documents:
        snippet = " ".join((opening or "").split())[:budget]
        name = short_source(source)
        lines.append(f"- {name}: {snippet}" if snippet else f"- {name}")
    if not lines:
        return ""
    return "Documents loaded in this session:\n" + "\n".join(lines)

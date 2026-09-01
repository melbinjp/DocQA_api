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


# How much of a document's opening to show in the manifest.
#
# This was 260 characters, which on an academic paper is the title followed by a
# wall of author names and nothing about what the paper does. Measured
# 2026-09-01 on the GPT-3 paper: 260 characters ends inside the author list, and
# "175 billion parameters" sits at character 1266. Asked which of two models has
# more parameters, the answer had BERT's 340 million from a retrieved chunk and
# nothing for GPT-3, because the question is about comparing documents and the
# retriever matched benchmark prose instead of the abstract. The manifest exists
# for exactly that kind of question and was too short to answer it.
#
# 900 characters reaches into the abstract on a paper and still costs less than
# one retrieved chunk per document.
MANIFEST_SNIPPET_CHARS = 900


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
    lines = []
    for source, opening in documents:
        snippet = " ".join((opening or "").split())[:MANIFEST_SNIPPET_CHARS]
        name = short_source(source)
        lines.append(f"- {name}: {snippet}" if snippet else f"- {name}")
    if not lines:
        return ""
    return "Documents loaded in this session:\n" + "\n".join(lines)

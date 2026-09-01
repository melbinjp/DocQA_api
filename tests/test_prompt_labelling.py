"""Every excerpt in the prompt says where it came from.

Without this the model reads eight chunks from eight documents as one
undifferentiated wall of text. Measured 2026-09-01: asked which of the loaded
papers was about image recognition, it answered with three entries out of
ResNet's bibliography rather than naming ResNet, because a chunk of reference
list and a chunk of a paper are indistinguishable when neither says what it is.
"""
from utils.prompting import label_chunk, short_source


def test_a_url_becomes_a_readable_name():
    assert short_source("https://arxiv.org/pdf/1512.03385") == "1512.03385"
    assert short_source("https://example.com/docs/report.pdf?v=2") == "report.pdf"
    assert short_source("https://example.com/guide/") == "guide"


def test_an_empty_source_does_not_produce_an_empty_label():
    assert short_source("") == "document"


def test_a_labelled_chunk_names_its_document_and_page():
    out = label_chunk({"text": "Residual nets are easier to optimize.",
                        "source": "https://arxiv.org/pdf/1512.03385", "page": 1})
    assert out.startswith("[From 1512.03385, page 1]")
    assert "Residual nets are easier to optimize." in out


def test_a_pageless_chunk_still_names_its_document():
    out = label_chunk({"text": "Some HTML page text.",
                        "source": "https://example.com/article.html", "page": None})
    assert out.startswith("[From article.html]")
    assert "page" not in out.split("\n")[0]


def test_the_text_is_never_altered_only_prefixed():
    text = "Exact text with |pipes| and 213 numbers."
    out = label_chunk({"text": text, "source": "a.pdf", "page": 9})
    assert out.endswith(text)


# --- the manifest: what is loaded, regardless of what retrieval returned ---

from utils.prompting import build_manifest


def test_the_manifest_names_each_document_and_what_it_is():
    m = build_manifest([
        ("https://arxiv.org/pdf/1512.03385", "Deep Residual Learning for Image Recognition. We present a residual learning framework."),
        ("https://arxiv.org/pdf/1810.04805", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."),
    ])
    assert "1512.03385" in m and "Residual" in m
    assert "1810.04805" in m and "BERT" in m


def test_the_manifest_is_short_enough_not_to_become_a_second_context():
    """Bounded by the snippet size rather than a magic number, so raising the
    snippet does not silently break the intent this guards."""
    from utils.prompting import MANIFEST_SNIPPET_CHARS

    m = build_manifest([("a.pdf", "x" * 5000)])
    assert len(m) < MANIFEST_SNIPPET_CHARS + 60


def test_an_empty_session_has_no_manifest():
    assert build_manifest([]) == ""


def test_a_document_with_no_text_still_appears():
    m = build_manifest([("only-name.pdf", "")])
    assert "only-name.pdf" in m


def test_the_manifest_reaches_past_an_author_list():
    """260 characters on an academic paper is the title and then names.

    Measured on the GPT-3 paper: the first 260 characters end inside the author
    block, and the fact that identifies the paper, "175 billion parameters", is
    at character 1266. A manifest that says who wrote something but not what it
    is cannot answer a question about which document covers what.
    """
    from utils.prompting import MANIFEST_SNIPPET_CHARS

    opening = (
        "Language Models are Few-Shot Learners "
        + "Author Name " * 40
        + "Abstract Recent work has demonstrated substantial gains. "
        + "Here we train GPT-3, with 175 billion parameters."
    )
    m = build_manifest([("https://arxiv.org/pdf/2005.14165", opening)])
    assert "Few-Shot Learners" in m
    assert MANIFEST_SNIPPET_CHARS >= 700, "too short to clear an author block"


def test_the_manifest_still_costs_less_than_a_retrieved_chunk():
    """It is a list of what is loaded, not a second context."""
    from utils.prompting import MANIFEST_SNIPPET_CHARS
    from utils.splitter import DEFAULT_MAX_CHARS

    assert MANIFEST_SNIPPET_CHARS < DEFAULT_MAX_CHARS
    m = build_manifest([(f"doc{i}.pdf", "x" * 5000) for i in range(8)])
    assert len(m) < 8 * (MANIFEST_SNIPPET_CHARS + 40)

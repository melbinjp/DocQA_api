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

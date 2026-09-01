"""A vector must never name the wrong chunk.

Once a chunk owns several vectors, `self.chunks[vector_id]` is wrong and quietly
so: it would return real text, from the wrong place, with a page number
attached. That is worse than returning nothing, which is why the mapping is
tested with fake embeddings rather than trusted to review.
"""
import asyncio

import numpy as np
import pytest

from rag_session import RAGSession


class FakeModel:
    """Embeds by a marker in the text, so retrieval order is predictable."""

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, convert_to_numpy=True):
        out = []
        for t in texts:
            if "APPLE" in t:
                out.append([1.0, 0.0, 0.0])
            elif "BANANA" in t:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return np.array(out, dtype="float32")


def make_session():
    session = RAGSession(source="doc.pdf", embedding_model=FakeModel())
    chunks = ["chunk zero, mentions APPLE somewhere inside",
              "chunk one, mentions BANANA somewhere inside"]
    pages = [4, 9]
    # Chunk 0 owns three vectors, chunk 1 owns two.
    vector_parents = [0, 0, 0, 1, 1]
    texts = ["whole zero APPLE", "window zero a", "window zero b APPLE",
             "whole one BANANA", "window one BANANA"]
    session.ingest(chunks, FakeModel().encode(texts), pages, vector_parents)
    return session


def test_a_vector_resolves_to_its_own_chunk_and_page():
    s = make_session()
    r = asyncio.run(s.query("APPLE", k=2))
    assert r[0]["text"].startswith("chunk zero")
    assert r[0]["page"] == 4


def test_the_other_chunk_is_not_confused_with_it():
    s = make_session()
    r = asyncio.run(s.query("BANANA", k=2))
    assert r[0]["text"].startswith("chunk one")
    assert r[0]["page"] == 9


def test_several_matching_windows_produce_one_result_not_several():
    """Chunk zero owns two APPLE vectors. It is still one answer."""
    s = make_session()
    r = asyncio.run(s.query("APPLE", k=5))
    texts = [x["text"] for x in r]
    assert len(texts) == len(set(texts)), f"duplicated chunk in results: {texts}"


def test_a_wrong_length_mapping_is_refused_rather_than_guessed():
    s = RAGSession(source="d", embedding_model=FakeModel())
    with pytest.raises(ValueError):
        s.ingest(["a", "b"], FakeModel().encode(["x", "y", "z"]), [1, 2], [0, 1])


def test_the_default_mapping_is_one_to_one():
    """A caller that knows nothing about windows still gets sane behaviour."""
    s = RAGSession(source="d", embedding_model=FakeModel())
    s.ingest(["only APPLE chunk"], FakeModel().encode(["only APPLE chunk"]), [7])
    r = asyncio.run(s.query("APPLE", k=1))
    assert r[0]["text"] == "only APPLE chunk" and r[0]["page"] == 7


def test_a_second_ingest_does_not_point_at_the_first_documents_chunks():
    """Parent indices arrive relative to the call; they must be offset."""
    s = make_session()
    s.ingest(["chunk two, mentions CHERRY"], FakeModel().encode(["CHERRY here"]),
             [11], [0])
    r = asyncio.run(s.query("CHERRY", k=1))
    assert r[0]["text"].startswith("chunk two")
    assert r[0]["page"] == 11

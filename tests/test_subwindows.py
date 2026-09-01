"""A long chunk is indexed several times over, and still cites once.

The RAG paper says "Each Wikipedia article is split into disjoint 100-word
chunks, to make a total of 21M documents", inside a 1435-character chunk that
opens "To estimate the probability of an hypothesis y we run an additional
forward pass for each document z". The chunk's vector is about marginalising
over documents, so three differently worded questions about the size of the
index all failed to retrieve it. The chunk was correct, indexed and unreachable.

Shrinking chunks would fix that and undo the largest accuracy win in this repo,
so a chunk is indexed whole and again window by window instead. What must never
happen is a vector pointing at the wrong chunk, because that is a citation
naming a page the text did not come from.
"""
import pytest

from utils.splitter import (
    SUBWINDOW_CHARS,
    SUBWINDOW_MIN_CHUNK,
    index_entries,
    subwindows,
)


def test_a_short_chunk_gets_no_windows():
    assert subwindows("short text") == []
    assert subwindows("x" * (SUBWINDOW_CHARS - 1)) == []


def test_windows_cover_the_whole_chunk():
    text = "".join(f"{i:04d} " for i in range(400))  # 2000 chars, unique markers
    windows = subwindows(text)
    assert len(windows) > 1
    joined = " ".join(windows)
    for marker in ("0000", "0199", "0399"):
        assert marker in joined, f"{marker} appears in no window"


def test_windows_overlap_so_a_fact_on_a_seam_survives_whole():
    text = "a" * 500 + " THE BURIED FACT IS HERE " + "b" * 900
    assert any("THE BURIED FACT IS HERE" in w for w in subwindows(text))


def test_every_entry_points_at_a_real_chunk():
    chunks = [{"text": "x" * 1500, "page": 1}, {"text": "short", "page": 2}]
    entries = index_entries(chunks)
    for text, parent in entries:
        assert 0 <= parent < len(chunks)
        assert text.strip()


def test_a_long_chunk_owns_several_vectors_and_a_short_one_owns_itself():
    chunks = [{"text": "x" * 1500, "page": 1}, {"text": "short", "page": 2}]
    parents = [p for _, p in index_entries(chunks)]
    assert parents.count(0) > 1, "the long chunk was not windowed"
    assert parents.count(1) == 1, "a short chunk should not be windowed"


def test_a_chunk_just_under_the_threshold_is_not_windowed():
    chunks = [{"text": "y" * (SUBWINDOW_MIN_CHUNK - 1), "page": 1}]
    assert len(index_entries(chunks)) == 1


def test_a_table_is_still_indexed_only_by_its_caption():
    chunks = [{"text": "|a|b|\n|---|---|\n|1|2|" + "x" * 2000, "page": 9,
               "embed_text": "Table 3: the caption."}]
    entries = index_entries(chunks)
    assert len(entries) == 1, "a table must not be windowed into its own grid"
    assert entries[0][0] == "Table 3: the caption."


def test_the_first_entry_of_a_prose_chunk_is_the_whole_chunk():
    """The whole-chunk vector carries the chunk's overall topic, which is what
    broad questions match. Windows are an addition to it, not a replacement."""
    chunks = [{"text": "z" * 1500, "page": 1}]
    entries = index_entries(chunks)
    assert entries[0][0] == "z" * 1500


def test_entry_count_is_bounded():
    """Each vector costs an embedding at ingest, so the multiplier must stay
    small enough that a 200-chunk document still finishes inside the timeout."""
    chunks = [{"text": "w" * 1500, "page": 1} for _ in range(100)]
    assert len(index_entries(chunks)) <= 100 * 6


def test_a_very_large_document_is_not_windowed():
    """Losing a document to a timeout is worse than one sentence being harder
    to find, so past the cap the windows are dropped."""
    from utils.splitter import MAX_CHUNKS_FOR_WINDOWING
    chunks = [{"text": "w" * 1500, "page": 1}
              for _ in range(MAX_CHUNKS_FOR_WINDOWING + 1)]
    assert len(index_entries(chunks)) == len(chunks)


def test_a_document_just_inside_the_cap_is_still_windowed():
    from utils.splitter import MAX_CHUNKS_FOR_WINDOWING
    chunks = [{"text": "w" * 1500, "page": 1}
              for _ in range(MAX_CHUNKS_FOR_WINDOWING)]
    assert len(index_entries(chunks)) > len(chunks)

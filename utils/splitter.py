"""Splitting a document into the pieces that get embedded and retrieved.

The chunk size is the single biggest lever on whether an answer is right, and
the old value of 500 characters was costing correct answers. Measured on the
live Space against *Attention Is All You Need*, 2026-09-01:

    "How many parameters does the big model have compared to the base model?"

The paper answers this in Table 3: base 65, big 213, in a column headed
`params x10^6`. At 500 characters the header block and the row values landed in
different chunks, so no retrieved chunk ever contained both, and the API
answered "the provided context does not contain information comparing the
parameter count". Handing the same page to the same model as one document, with
nothing else changed, produced "the base model has 65 million parameters, and
the big model has 213 million". The retrieval was not wrong. The chunk was too
small to hold the fact.

500 characters is roughly three sentences. Any table, any equation with its
explanation, any argument that takes a paragraph to make, is cut in half by it.
The receiving model has a context window measured in the hundreds of thousands
of tokens, so the frugality bought nothing.

Boundaries are chosen at paragraph breaks, then sentence ends, then whitespace,
in that order, rather than at a fixed offset. A chunk that begins mid-sentence
embeds badly: the first clause is a fragment the model cannot place, and it
drags the vector away from what the chunk is actually about.
"""
import re
from typing import List

# Large enough to hold a small table or a full argument, small enough that
# retrieving eight of them is still a focused prompt rather than the document.
DEFAULT_MAX_CHARS = 1500

# Enough that a fact split across a boundary survives whole in one of the two
# neighbours. At 500/100 a fact needed to be under 100 characters to be safe.
DEFAULT_OVERLAP = 250

# How far back from the hard limit we will look for a clean break before giving
# up and cutting at the limit. A quarter of the chunk keeps chunks near target
# size while still finding a boundary in ordinary prose.
_LOOKBACK_RATIO = 0.25


def _best_break(text: str, start: int, end: int) -> int:
    """The latest clean boundary in `text[start:end]`, or `end` if there is none.

    Paragraph, then sentence, then word. Only boundaries in the last
    `_LOOKBACK_RATIO` of the window are considered, so one long unbroken run
    cannot drag a chunk down to a fraction of the target size.
    """
    window = end - start
    floor = start + int(window * (1 - _LOOKBACK_RATIO))
    segment = text[start:end]

    para = segment.rfind("\n\n")
    if para != -1 and start + para > floor:
        return start + para + 2

    # A sentence end followed by a space. Rules out "3.5" and "Fig." reasonably
    # well by requiring the character before the stop not to be a digit.
    for match in reversed(list(re.finditer(r"(?<!\d)[.!?]\s", segment))):
        pos = start + match.end()
        if pos > floor:
            return pos

    space = segment.rfind(" ")
    if space != -1 and start + space > floor:
        return start + space + 1

    return end


def _snap_forward(text: str, pos: int, limit: int) -> int:
    """Move `pos` forward to the start of the next word, if one is close enough.

    Bounded by `limit` so a run of text with no whitespace cannot push the next
    chunk past the end of the one before it and open a gap in coverage. If
    nothing is found, `pos` is returned unchanged, which keeps the loop
    advancing.
    """
    if pos <= 0 or pos >= len(text):
        return pos
    if text[pos - 1].isspace():
        return pos
    space = text.find(" ", pos, limit)
    if space == -1:
        newline = text.find("\n", pos, limit)
        if newline == -1:
            return pos
        space = newline
    return space + 1


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS,
               overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """Split `text` into overlapping chunks that start and end cleanly."""
    # Collapse runs of spaces and tabs, but keep paragraph breaks: they are the
    # best boundary available and squashing them throws that away. This is a
    # change from the old behaviour, which flattened everything to single spaces.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text).strip()

    if not text:
        return []
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars, or splitting never advances")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end if hard_end == len(text) else _best_break(text, start, hard_end)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        # Advance from the boundary actually used, not from the hard limit, or a
        # short chunk would be followed by a gap the size of what it gave up.
        #
        # Then snap forward to a word boundary. Choosing a clean end is only half
        # the job: stepping back by a fixed overlap lands wherever it lands, and
        # a chunk opening `100 token101 token102` starts on the tail of a word
        # that means nothing on its own and pulls the embedding off the topic.
        nxt = max(end - overlap, start + 1)
        nxt = _snap_forward(text, nxt, end)
        start = nxt
    return chunks


# A table's caption and column list are written above the grid and do not begin
# with a pipe, so a plain run-splitter files them as prose and cuts them away
# from the rows they name. That is the bug this guards: at most this many
# non-blank lines directly above a grid are pulled into the table's block.
_MAX_PREAMBLE_LINES = 3

# And short. A caption is a sentence or two; a paragraph that happens to sit
# directly above a grid with no blank line between them is not a caption, and
# without this bound a whole page of prose gets swallowed into the table block.
_MAX_PREAMBLE_CHARS = 500


def _table_blocks(text: str):
    """Split `text` into `(is_table, segment)` runs.

    A markdown table is a contiguous run of lines beginning with a pipe, plus
    the caption lines immediately above it. Runs of ordinary prose come back
    untouched.
    """
    lines = text.split("\n")
    runs, current, current_is_table = [], [], None
    for line in lines:
        is_table = line.lstrip().startswith("|")
        if current_is_table is None:
            current_is_table = is_table
        if is_table != current_is_table:
            runs.append([current_is_table, current])
            current, current_is_table = [], is_table
        current.append(line)
    if current:
        runs.append([bool(current_is_table), current])

    # Move the caption off the end of the prose run and onto the table it
    # belongs to. "Off the end" is the non-blank lines after the last blank
    # line, which is exactly the caption and column list as they are emitted.
    for i in range(1, len(runs)):
        if not runs[i][0] or runs[i - 1][0]:
            continue
        prose = runs[i - 1][1]
        cut = len(prose)
        while cut > 0 and prose[cut - 1].strip():
            cut -= 1
        attached = prose[cut:]
        if (0 < len(attached) <= _MAX_PREAMBLE_LINES
                and len("\n".join(attached)) <= _MAX_PREAMBLE_CHARS):
            runs[i - 1][1] = prose[:cut]
            runs[i][1] = attached + runs[i][1]

    return [(is_table, "\n".join(body)) for is_table, body in runs if "".join(body).strip()]


def _split_table(table: str, max_chars: int) -> List[str]:
    """Chunk a markdown table without orphaning its rows from its header.

    A table is one fact spread over a grid, and cutting it anywhere destroys the
    only thing that makes the numbers mean anything. Measured on page 9 of
    *Attention Is All You Need*: rendering the table as a grid was not enough on
    its own, because the generic splitter then cut it into a chunk holding
    `params` and a separate 117-character chunk holding
    `|big|...|**4.33**<br>**26.4**<br>213|`. Asked for the big model's parameter
    count, retrieval returned the row without the header, which is a number with
    no name attached, and the answer was still "the provided text does not
    contain information".

    So the table is kept whole where it fits, and where it does not, the header
    rows are repeated at the top of every piece.
    """
    if len(table) <= max_chars:
        return [table]

    rows = [r for r in table.split("\n") if r.strip()]
    # Anything before the first pipe is the caption and column list. It repeats
    # with the header, because a block of rows is unreadable without both.
    lead = 0
    while lead < len(rows) and not rows[lead].lstrip().startswith("|"):
        lead += 1
    # A markdown table's first two lines are the header and its separator.
    header = rows[:lead + 2] if len(rows) > lead + 2 else rows[:lead]
    body = rows[len(header):]
    header_text = "\n".join(header)

    chunks, current = [], list(header)
    current_len = len(header_text)
    for row in body:
        if current_len + len(row) + 1 > max_chars and len(current) > len(header):
            chunks.append("\n".join(current))
            current = list(header)
            current_len = len(header_text)
        current.append(row)
        current_len += len(row) + 1
    if len(current) > len(header):
        chunks.append("\n".join(current))
    return chunks or [table]


def split_pages(pages, max_chars: int = DEFAULT_MAX_CHARS,
                overlap: int = DEFAULT_OVERLAP) -> List[dict]:
    """Split `[(page, text), ...]` into chunks that remember their page.

    Splitting per page rather than over one concatenated string is what makes a
    citation possible at all: once the pages are joined, no chunk can say where it
    came from. It also stops a chunk straddling a page boundary and being
    attributed to whichever page happened to come first.
    """
    out: List[dict] = []
    for page, text in pages:
        for is_table, segment in _table_blocks(text):
            if not segment.strip():
                continue
            pieces = (_split_table(segment, max_chars) if is_table
                      else split_text(segment, max_chars=max_chars, overlap=overlap))
            for chunk in pieces:
                if chunk.strip():
                    out.append({"text": chunk, "page": page})
    return out

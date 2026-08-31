"""Minimal document loaders for the DocQA application."""

import os
import re
import tempfile
from markitdown import MarkItDown
from .exceptions import DocumentLoaderError

def _page_text_with_tables(page) -> str:
    """Page text, with any table also rendered as a grid underneath it.

    `get_text()` reads a table column by column, so every header arrives first
    and the numbers follow as a bare stream. Page 9 of *Attention Is All You
    Need* extracts as `N dmodel dff h dk dv ... base 6 512 2048 8 64 64 ... 213`.
    Nothing joins `h` to its value, so asking what changes at 32 attention heads
    retrieves a chunk of loose numbers that means nothing to the embedding and
    nothing to the model reading it. Measured 2026-09-01: that question was
    answered "not listed in Table 3" from the raw text and answered with the
    right figures from the same table rendered as a grid.

    The prose is kept as well as the grid, not replaced by it. Table detection
    on a borderless academic table is approximate, and it merged fourteen
    columns into three here, so the grid is an addition that can help rather
    than a parse that has to be right.
    """
    text = page.get_text()
    try:
        tables = page.find_tables().tables
    except Exception:
        return text
    if not tables:
        return text

    captions = _table_captions(text)

    rendered = []
    for position, table in enumerate(tables):
        try:
            markdown = table.to_markdown()
        except Exception:
            continue
        if not markdown or not markdown.strip():
            continue
        # A grid on its own is unretrievable, which is not the same as being
        # badly ranked. Measured 2026-09-01: the Table 3 chunk held its header,
        # `base ... 65` and `big ... 213`, and neither the dense nor the lexical
        # retriever put it in the top eight for "how does the parameter count of
        # the big model compare to the base model". It was never retrieved at
        # all. The chunk began `|Col1|train<br>N d d h`, which has no sentence in
        # it for an embedding to match and too few words for BM25 to weigh.
        #
        # Its caption reads "Table 3: Variations on the Transformer
        # architecture. Unlisted values are identical to those of the base
        # model." That is the language the question is asked in, and it is
        # sitting in the prose a few lines above. Prepending it, with the column
        # names spelled out, gives the grid something to be found by.
        preamble = []
        if position < len(captions):
            preamble.append(captions[position])
        columns = _column_names(markdown)
        if columns:
            preamble.append(f"Columns: {columns}.")
        block = "\n".join(preamble + [markdown.strip()]) if preamble else markdown.strip()
        rendered.append(block)

    if not rendered:
        return text
    return text + "\n\n" + "\n\n".join(rendered)


# "Table 3: Variations on ..." or "Table 3. Variations on ...", to the end of
# the sentence after it. Figure captions are deliberately not matched: a figure
# has no grid to attach them to.
_CAPTION = re.compile(r"^(Table\s+\d+[:.][\s\S]{0,400})", re.MULTILINE)


def _table_captions(text: str) -> list[str]:
    """Captions on the page, in the order they appear.

    Paired with tables by position, which is the order `find_tables` returns
    them in. Mispairing is possible on a page with two tables and one caption,
    which is why the caption is added to the text rather than used to label it:
    a wrong caption sitting above the right numbers is a retrieval hint that
    misses, not a citation that lies.
    """
    out = []
    for match in _CAPTION.finditer(text):
        caption = " ".join(match.group(1).split())
        # Trim to a sentence end so the caption does not trail off into the
        # body text underneath it. Whole caption if there is no full stop.
        end = caption.rfind(". ")
        if end > 40:
            caption = caption[:end + 1]
        out.append(caption)
    return out


def _column_names(markdown: str) -> str:
    """The header cells of a markdown table, as a plain comma-separated list."""
    first = markdown.strip().split("\n", 1)[0]
    cells = [c.replace("<br>", " ").strip() for c in first.split("|")]
    cells = [" ".join(c.split()) for c in cells if c.strip() and not set(c.strip()) <= {"-"}]
    return ", ".join(cells)


def load_source_pages(raw: bytes, ext: str) -> list[tuple[int | None, str]]:
    """Extract text while keeping the page it came from.

    Returns `[(page_number, text), ...]`, one-based, with `None` for formats that
    have no pages. `load_source` flattens this and is kept for callers that do not
    need the page.

    **PyMuPDF is tried first for PDFs, not second.** It used to be a fallback that
    only ran when MarkItDown returned nothing at all, so a PDF MarkItDown read
    badly rather than not at all never reached it.

    Observed on the deployed Space, 2026-08-31: querying arXiv 2005.11401 returned
    source text with the spaces removed, `nenthelps toguidethegeneration`. The same
    PDF, the same pinned `markitdown==0.1.6`, extracts cleanly on a developer
    machine. The difference is that `markitdown[pdf]` pulls **pdfminer.six, which
    this project does not pin**, so the image resolves its own version at build
    time and the same document extracts differently in development and production.

    That is the real defect, and it is worse than a bad extractor: text quality
    varies by deployment and nothing downstream can tell. PyMuPDF is a pinned
    direct dependency and is deterministic here, so it goes first. Extraction that
    silently mangles is worse than extraction that fails, because a mangled chunk
    still embeds, still retrieves, and still gets quoted back to a user.
    """
    ext = ext.lower().strip('.')

    if ext == 'pdf':
        try:
            import fitz
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                doc = fitz.open(tmp_path)
                pages = [(i + 1, _page_text_with_tables(page)) for i, page in enumerate(doc)]
                doc.close()
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
            if any(text and text.strip() for _, text in pages):
                return [(n, t) for n, t in pages if t and t.strip()]
        except Exception:
            # Fall through to the generic path below rather than failing here.
            pass

    text = load_source(raw, ext)
    return [(None, text)]


def load_source(raw: bytes, ext: str) -> str:
    """
    Extracts text content from a raw byte stream using MarkItDown (for office documents)
    or BeautifulSoup (for HTML/URLs).

    Args:
        raw: The raw bytes of the file.
        ext: The file extension (e.g., '.txt', 'pdf', '.html').

    Returns:
        The extracted text content.

    Raises:
        DocumentLoaderError: If there's an error during parsing.
    """
    ext = ext.lower().strip('.')
    
    # Fast-path for plain text and markdown files
    if ext in ['md', 'txt', 'text']:
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('utf-8', errors='ignore')

    # Fast-path for HTML/URLs using BeautifulSoup to prevent timeouts on large webpages
    if ext in ['url', 'html', 'htm']:
        try:
            from bs4 import BeautifulSoup
            html = raw.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')
            # Decompose tags that do not contain main reading content
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript']):
                tag.decompose()
            text_content = soup.get_text(' ', strip=True)
            if not text_content or not text_content.strip():
                raise DocumentLoaderError("HTML text extraction resulted in no content.")
            return text_content
        except Exception as e:
            if isinstance(e, DocumentLoaderError):
                raise
            raise DocumentLoaderError(f"Failed to parse HTML content: {e}") from e

    md = MarkItDown()

    # MarkItDown prefers working with files, so we write to a temp file
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        text_content = ""
        try:
            result = md.convert(tmp_path)
            text_content = result.text_content or ""
        except Exception as e:
            if ext == "pdf":
                # Suppress error and let PyMuPDF handle fallback below
                pass
            else:
                raise DocumentLoaderError(f"Failed to load content with extension '{ext}': {e}") from e

        # If PDF extraction returned no content or failed, run PyMuPDF fallback
        if ext == "pdf" and (not text_content or not text_content.strip()):
            try:
                import fitz
                doc = fitz.open(tmp_path)
                texts = [page.get_text() for page in doc]
                text_content = "\n".join(texts)
            except Exception as pdf_err:
                raise DocumentLoaderError(
                    f"PDF extraction failed using both MarkItDown and PyMuPDF fallback. "
                    f"PyMuPDF error: {pdf_err}"
                )

        if not text_content or not text_content.strip():
            raise DocumentLoaderError("Extraction resulted in no content.")
        return text_content
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

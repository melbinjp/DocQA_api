"""Fetching a document from a URL the user pasted.

Kept out of `app.py` so it can be tested without importing the embedding model.
"""

import asyncio
import ipaddress
import pathlib
import socket
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException


def is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.hostname:
            return False
        ip = socket.gethostbyname(parsed.hostname)
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_reserved:
            return False
        return True
    except Exception:
        return False


async def is_safe_url_async(url: str) -> bool:
    """`is_safe_url` off the event loop.

    `socket.gethostbyname` blocks. This is a single-replica async server and the
    redirect loop below calls this once per hop, so on the loop it would stall
    every other request for the length of a DNS lookup.
    """
    return await asyncio.to_thread(is_safe_url, url)


# Content types we can extract pages or structured text from, mapped to the
# extension `load_source_pages` dispatches on. Anything not here is treated as a
# web page and read through Jina Reader instead.
_CONTENT_TYPE_EXT = {
    "application/pdf": "pdf",
    "application/x-pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/msword": "doc",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/csv": "csv",
}

_HTML_TYPES = {"text/html", "application/xhtml+xml"}

_MAX_REDIRECTS = 5


async def _fetch_following_redirects(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """GET `url`, following up to `_MAX_REDIRECTS` hops, checking every one.

    `is_safe_url` only validates the URL the user typed. Turning on
    `follow_redirects` would therefore hand us an SSRF primitive: a public host
    that answers 302 with `Location: http://169.254.169.254/` would be fetched
    with no check at all. So redirects are followed by hand and each hop is
    re-resolved and re-checked before it is requested.
    """
    current = url
    for _ in range(_MAX_REDIRECTS):
        response = await client.get(current, timeout=30.0, follow_redirects=False)
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise HTTPException(
                    status_code=400,
                    detail="That URL redirected without saying where to. Try the address it resolves to.",
                )
            current = str(response.url.join(location))
            if not await is_safe_url_async(current):
                raise HTTPException(
                    status_code=400,
                    detail="That URL redirects somewhere this service will not fetch.",
                )
            continue
        return response
    raise HTTPException(status_code=400, detail=f"That URL redirected more than {_MAX_REDIRECTS} times.")


async def fetch_url_document(client: httpx.AsyncClient, url: str) -> tuple[bytes, str]:
    """Fetch `url` and return `(bytes, extension)` ready for `load_source_pages`.

    Direct fetch comes first, deliberately. Jina Reader returns Markdown, which
    has no pages, so routing a PDF through it threw away the page numbers this
    app exists to cite: every source came back `"page": null`. Fetching the PDF
    itself hands PyMuPDF real bytes and the citation gets a page.

    Jina Reader is still the right tool for an HTML page, and still the fallback
    when a direct fetch is refused, which happens often enough (bot walls answer
    a plain GET with an interstitial).
    """
    direct_error = None
    try:
        response = await _fetch_following_redirects(client, url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        ext = _CONTENT_TYPE_EXT.get(content_type)
        if ext is None:
            # Servers mislabel documents often enough that the path is worth a look.
            suffix = pathlib.Path(urlparse(url).path).suffix.lower().lstrip(".")
            if suffix in {"pdf", "docx", "pptx", "xlsx", "doc", "txt", "md", "csv"}:
                ext = suffix
        if ext:
            return response.content, ext
        if content_type in _HTML_TYPES or not content_type:
            # Read the page ourselves. `load_source` strips script, style, nav,
            # header and footer with BeautifulSoup, which is enough for a
            # readable article and does not depend on anyone else being up.
            #
            # Jina Reader used to be the only route for HTML, and on 2026-09-01
            # it started refusing this Space: every HTML URL came back 502,
            # "Could not read that URL (content-type text/html)", while the same
            # Jina request from a laptop answered 200. A third party that can
            # take out the whole HTML path is not a dependency worth having for
            # something BeautifulSoup already does in-process, and it also means
            # every URL a user pastes stops being forwarded to a third party.
            return response.content, "html"
        direct_error = f"content-type {content_type or 'unknown'}"
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        # Never echo the upstream body; it is arbitrary remote HTML.
        direct_error = f"HTTP {e.response.status_code}"
    except httpx.RequestError as e:
        direct_error = type(e).__name__

    try:
        response = await client.get(f"https://r.jina.ai/{url}", timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        return response.content, "md"
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=f"Could not read that URL ({direct_error}). Download the file and upload it instead.",
        )

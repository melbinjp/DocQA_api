"""URL ingestion: page fidelity, and the redirect guard.

Two things are being pinned here.

The first is that a PDF fetched from a URL reaches the extractor as a PDF. It
used to go through Jina Reader, which returns Markdown, so every source from a
URL came back `"page": null` and the one thing this app is for did not work on
the one input a stranger is most likely to try.

The second is that following redirects does not become an SSRF hole. `is_safe_url`
validates the URL the user typed and nothing else, so a public host answering
302 with `Location: http://169.254.169.254/` would otherwise be fetched with no
check at all. `test_redirect_to_private_address_is_refused` fails if the per-hop
check is dropped; it is the test that has to stay red under that mutation.
"""

from unittest.mock import patch

import httpx
import pytest
from fastapi import HTTPException

from utils import url_fetch as app_module
from utils.url_fetch import _MAX_REDIRECTS, fetch_url_document

PDF_BYTES = b"%PDF-1.4 fake but plausible"


class FakeResponse:
    def __init__(self, status_code, headers=None, content=b"", url=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self.url = httpx.URL(url)

    @property
    def text(self):
        return self.content.decode(errors="replace")

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom", request=httpx.Request("GET", self.url), response=self
            )


class FakeClient:
    """Answers from a {url: FakeResponse} map and records what was asked for."""

    def __init__(self, responses):
        self.responses = responses
        self.requested = []

    async def get(self, url, timeout=None, follow_redirects=False):
        self.requested.append(str(url))
        try:
            return self.responses[str(url)]
        except KeyError:  # pragma: no cover - a miss means the test is wrong
            raise AssertionError(f"unexpected request to {url}")


@pytest.fixture
def allow_all_hosts():
    with patch.object(app_module, "is_safe_url", return_value=True):
        yield


@pytest.fixture
def block_private_hosts():
    def predicate(url):
        return "169.254.169.254" not in url and "localhost" not in url

    with patch.object(app_module, "is_safe_url", side_effect=predicate):
        yield


@pytest.mark.asyncio
async def test_pdf_url_reaches_the_extractor_as_a_pdf(allow_all_hosts):
    """The whole point: a PDF stays a PDF, so it can carry page numbers."""
    client = FakeClient(
        {
            "https://example.com/paper.pdf": FakeResponse(
                200, {"content-type": "application/pdf"}, PDF_BYTES,
                "https://example.com/paper.pdf",
            )
        }
    )

    content, ext = await fetch_url_document(client, "https://example.com/paper.pdf")

    assert ext == "pdf"
    assert content == PDF_BYTES
    assert not any("r.jina.ai" in u for u in client.requested), (
        "a PDF must not be laundered through Jina Reader; that is what dropped the pages"
    )


@pytest.mark.asyncio
async def test_pdf_served_without_a_content_type_is_still_a_pdf(allow_all_hosts):
    client = FakeClient(
        {
            "https://example.com/paper.pdf": FakeResponse(
                200, {"content-type": "application/octet-stream"}, PDF_BYTES,
                "https://example.com/paper.pdf",
            )
        }
    )

    _, ext = await fetch_url_document(client, "https://example.com/paper.pdf")

    assert ext == "pdf"


@pytest.mark.asyncio
async def test_redirect_to_a_document_is_followed(allow_all_hosts):
    client = FakeClient(
        {
            "https://gov.example/doc": FakeResponse(
                301, {"location": "https://cdn.example/doc.pdf"}, b"",
                "https://gov.example/doc",
            ),
            "https://cdn.example/doc.pdf": FakeResponse(
                200, {"content-type": "application/pdf"}, PDF_BYTES,
                "https://cdn.example/doc.pdf",
            ),
        }
    )

    content, ext = await fetch_url_document(client, "https://gov.example/doc")

    assert (content, ext) == (PDF_BYTES, "pdf")


@pytest.mark.asyncio
async def test_redirect_to_private_address_is_refused(block_private_hosts):
    """The SSRF guard. Delete the per-hop check and this must go red."""
    client = FakeClient(
        {
            "https://public.example/x": FakeResponse(
                302, {"location": "http://169.254.169.254/latest/meta-data/"}, b"",
                "https://public.example/x",
            ),
            "http://169.254.169.254/latest/meta-data/": FakeResponse(
                200, {"content-type": "text/plain"}, b"iam-credentials",
                "http://169.254.169.254/latest/meta-data/",
            ),
        }
    )

    with pytest.raises(HTTPException) as excinfo:
        await fetch_url_document(client, "https://public.example/x")

    assert excinfo.value.status_code == 400
    assert "http://169.254.169.254/latest/meta-data/" not in client.requested, (
        "the link-local address was actually fetched"
    )


@pytest.mark.asyncio
async def test_redirect_loop_stops(allow_all_hosts):
    client = FakeClient(
        {
            "https://loop.example/": FakeResponse(
                302, {"location": "https://loop.example/"}, b"", "https://loop.example/"
            )
        }
    )

    with pytest.raises(HTTPException) as excinfo:
        await fetch_url_document(client, "https://loop.example/")

    assert excinfo.value.status_code == 400
    assert len(client.requested) <= _MAX_REDIRECTS + 1


@pytest.mark.asyncio
async def test_redirect_without_a_location_is_a_clear_error(allow_all_hosts):
    client = FakeClient(
        {"https://example.com/x": FakeResponse(301, {}, b"", "https://example.com/x")}
    )

    with pytest.raises(HTTPException) as excinfo:
        await fetch_url_document(client, "https://example.com/x")

    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_upstream_error_body_is_never_echoed(allow_all_hosts):
    """The 404 page of a stranger's server is not ours to repeat back."""
    secret_ish = "<html>internal-hostname-and-stack-trace</html>"
    client = FakeClient(
        {
            "https://example.com/missing": FakeResponse(
                404, {"content-type": "text/html"}, secret_ish.encode(),
                "https://example.com/missing",
            ),
            "https://r.jina.ai/https://example.com/missing": FakeResponse(
                500, {}, b"", "https://r.jina.ai/https://example.com/missing"
            ),
        }
    )

    with pytest.raises(HTTPException) as excinfo:
        await fetch_url_document(client, "https://example.com/missing")

    assert "internal-hostname" not in excinfo.value.detail
    assert "404" in excinfo.value.detail


@pytest.mark.asyncio
async def test_html_page_falls_back_to_the_reader(allow_all_hosts):
    client = FakeClient(
        {
            "https://example.com/article": FakeResponse(
                200, {"content-type": "text/html"}, b"<html>...</html>",
                "https://example.com/article",
            ),
            "https://r.jina.ai/https://example.com/article": FakeResponse(
                200, {"content-type": "text/plain"}, b"# Article\n\ntext",
                "https://r.jina.ai/https://example.com/article",
            ),
        }
    )

    content, ext = await fetch_url_document(client, "https://example.com/article")

    assert ext == "md"
    assert b"# Article" in content

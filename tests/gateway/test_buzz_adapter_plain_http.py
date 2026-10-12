"""Inbound Buzz attachments from a relay served over plain http.

A relay on a private network often has no certificate. Its own files are
accepted over http; every other origin stays https-only, and the size,
SHA-256, redirect, credential and fragment checks apply to both.
"""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import CachedMedia
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

BuzzAdapter = load_plugin_adapter("buzz").BuzzAdapter

SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
OTHER_PUBKEY = "a" * 64
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"


def _make_adapter(relay_url, extra=None):
    cfg = PlatformConfig(enabled=True, extra={"relay_url": relay_url, **(extra or {})})
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._display_name = "Chip"
    adapter._private_key = "nsec1test"
    adapter.set_authorization_check(lambda *_args: True)
    return adapter


def _metadata(url, payload=b"x"):
    return {
        "url": url,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "filename": "file.bin",
        "mime_type": "application/octet-stream",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relay_url", "url"),
    [
        ("http://test.relay:3000", "http://test.relay:3000/media/verified.bin"),
        ("http://test.relay", "http://test.relay/media/verified.bin"),
        ("http://Test.Relay.:3000", "http://test.relay:3000/media/verified.bin"),
    ],
)
async def test_download_caches_the_plain_http_relay_s_own_file(monkeypatch, relay_url, url):
    import httpx

    payload = b"verified Buzz attachment"
    real_async_client = httpx.AsyncClient
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=payload, headers={"content-length": str(len(payload))})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_async_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    cached = await _make_adapter(relay_url)._download_attachment(_metadata(url, payload))

    assert cached is not None
    assert Path(cached.path).read_bytes() == payload
    assert requested == [url]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("relay_url", "extra", "url"),
    [
        ("http://test.relay:3000", {}, "http://elsewhere.example:3000/media/file.bin"),
        ("http://test.relay:3000", {}, "http://test.relay:3001/media/file.bin"),
        ("http://test.relay:3000", {}, "http://test.relay/media/file.bin"),
        ("http://test.relay:3000", {}, "https://test.relay:3000/media/file.bin"),
        (
            "http://test.relay:3000",
            {"attachment_hosts": ["elsewhere.example:3000"]},
            "http://elsewhere.example:3000/media/file.bin",
        ),
        ("https://test.relay", {}, "http://test.relay/media/file.bin"),
        ("https://test.relay", {}, "http://test.relay:443/media/file.bin"),
    ],
)
async def test_download_trusts_plain_http_only_for_a_plain_http_relay_s_own_origin(
    monkeypatch, relay_url, extra, url
):
    import httpx

    def must_not_create_client(**kwargs):
        raise AssertionError("network client must not be created")

    monkeypatch.setattr(httpx, "AsyncClient", must_not_create_client)

    assert await _make_adapter(relay_url, extra)._download_attachment(_metadata(url)) is None


def test_imeta_accepts_plain_http_urls_only_when_told_the_relay_is_plain_http():
    event = {
        "tags": [
            ["imeta", "url http://test.relay:3000/media/file.bin", "x " + "a" * 64, "size 12", "filename file.bin"],
        ]
    }

    assert BuzzAdapter._imeta_attachments(event) == []
    assert BuzzAdapter._parse_imeta_attachments(event) == ([], 1)
    accepted, rejected = BuzzAdapter._parse_imeta_attachments(event, plain_http=True)
    assert [item["url"] for item in accepted] == ["http://test.relay:3000/media/file.bin"]
    assert rejected == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(("relay_url", "downloads"), [("http://test.relay:3000", 1), ("https://test.relay", 0)])
async def test_plain_http_attachment_is_downloaded_only_for_a_plain_http_relay(relay_url, downloads):
    adapter = _make_adapter(relay_url)
    attachment = CachedMedia(
        path="/agent/cache/doc_attachment.bin",
        media_type="application/octet-stream",
        kind="document",
        display_name="attachment.bin",
    )
    adapter._download_attachment = AsyncMock(return_value=attachment)
    adapter._user_names[OTHER_PUBKEY] = "Other"
    adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
    dispatched = []

    async def capture(**kwargs):
        dispatched.append(kwargs)

    adapter._dispatch_message = capture
    event = {
        "id": "plain-http-attachment",
        "pubkey": OTHER_PUBKEY,
        "content": "@Chip inspect this",
        "created_at": 1000,
        "kind": 9,
        "tags": [
            ["h", CHANNEL],
            [
                "imeta",
                "url http://test.relay:3000/media/file.bin",
                "m application/octet-stream",
                "x " + "a" * 64,
                "size 12",
                "filename attachment.bin",
            ],
        ],
    }

    await adapter._handle_event(CHANNEL, adapter._channel_state[CHANNEL], event)

    assert adapter._download_attachment.await_count == downloads
    assert dispatched[-1]["media_urls"] == ([attachment.path] if downloads else [])

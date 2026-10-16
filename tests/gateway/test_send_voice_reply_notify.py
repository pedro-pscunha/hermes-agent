"""Regression test for issue #27970 Bug 2.

The auto Telegram voice reply (``GatewayRunner._send_voice_reply``) is the
final response of a turn. It must mark its metadata as ``notify=True`` so
adapters that gate push notifications (Telegram's "important" mode) deliver
it as a normal push instead of a silent message — mirroring the existing
final-text path in ``gateway/platforms/base.py``.
"""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import FinalReplyAnchor, MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_event(thread_id=None):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
        thread_id=thread_id,
    )
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )


def _runner_with_adapter(send_voice_mock):
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        send_voice=send_voice_mock,
        is_in_voice_channel=lambda *_a, **_k: False,
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner


def _fake_tts_call(monkeypatch, audio_bytes=b"\x00" * 32):
    """Patch the TTS tool so it writes a real file at the requested path."""

    def _fake_text_to_speech_tool(*, text, output_path, **_kwargs):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(audio_bytes)
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        _fake_text_to_speech_tool,
    )
    monkeypatch.setattr(
        "tools.tts_text_normalize._strip_markdown_for_tts",
        lambda text: text,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("queued_anchor", [None, FinalReplyAnchor("m2"), FinalReplyAnchor(None)])
@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.SLACK])
async def test_voice_reply_marks_existing_thread_metadata_without_mutation(monkeypatch, tmp_path, queued_anchor, platform):
    """When thread metadata exists (Telegram DM-topic), notify=True is added without mutating the source dict."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    _fake_tts_call(monkeypatch)

    send_voice = AsyncMock()
    runner = _runner_with_adapter(send_voice)
    # Use a DM topic source so _thread_metadata_for_source returns a non-None dict.
    event = _make_event(thread_id="17585")
    event.source.platform = platform
    runner.adapters[platform] = runner.adapters[Platform.TELEGRAM]
    event.source.scope_id = "workspace"
    event.source.parent_chat_id = "parent-channel"
    event.source.message_id = "source-m1"
    event._final_reply_anchor = queued_anchor
    source_meta_snapshot = runner._thread_metadata_for_source(
        event.source, runner._reply_anchor_for_event(event)
    )
    assert source_meta_snapshot is not None
    snapshot_copy = dict(source_meta_snapshot)

    await runner._send_voice_reply(event, "Hello there.")

    send_voice.assert_awaited_once()
    kwargs = send_voice.await_args.kwargs
    assert kwargs["metadata"].get("notify") is True
    expected = "m1" if queued_anchor is None else queued_anchor.reply_to
    assert kwargs["reply_to"] == expected
    anchor_key = "telegram_reply_to_message_id" if platform == Platform.TELEGRAM else "message_id"
    assert kwargs["metadata"].get(anchor_key) == expected
    # All pre-existing thread keys are preserved.
    for k, v in snapshot_copy.items():
        if k != anchor_key:
            assert kwargs["metadata"].get(k) == v
    # The freshly-computed source-side metadata must NOT have been mutated
    # (would otherwise leak notify=True into the typing-indicator state).
    fresh = runner._thread_metadata_for_source(
        event.source, runner._reply_anchor_for_event(event)
    )
    assert "notify" not in fresh

# -*- coding: utf-8 -*-
"""Unit tests for Enterprise WeChat Stream bot adapter."""

import time
from unittest.mock import AsyncMock, patch

from bot.models import BotResponse, ChatType
from bot.platforms.wecom_stream import (
    TTLMessageDeduplicator,
    WeComStreamClient,
    WeComStreamHandler,
    WeComStreamReplyClient,
)


class _DummyReplyClient(WeComStreamReplyClient):
    def __init__(self, max_bytes: int = 3500):
        self.calls = []
        super().__init__(self.calls.append, max_bytes=max_bytes, interval_seconds=0)


def _payload(
    content: str = "/chat hello",
    *,
    msgid: str = "msg-1",
    chattype: str = "group",
    msgtype: str = "text",
):
    body = {
        "msgid": msgid,
        "msgtype": msgtype,
        "chattype": chattype,
        "from_userid": "user-1",
        "chatid": "chat-1",
    }
    if msgtype == "text":
        body["text"] = {"content": content}
    return {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": "req-1"},
        "body": body,
    }


def test_subscribe_and_heartbeat_payloads():
    subscribe = WeComStreamReplyClient.build_subscribe_payload(
        "bot-1",
        "secret-1",
        req_id="req-subscribe",
    )
    assert subscribe == {
        "cmd": "aibot_subscribe",
        "headers": {"req_id": "req-subscribe"},
        "body": {"bot_id": "bot-1", "secret": "secret-1"},
    }

    ping = WeComStreamReplyClient.build_ping_payload(req_id="req-ping")
    assert ping == {"cmd": "ping", "headers": {"req_id": "req-ping"}}


def test_stream_reply_payload_and_chunked_final():
    client = _DummyReplyClient(max_bytes=1000)

    payload = WeComStreamReplyClient.build_stream_reply_payload(
        req_id="req-1",
        stream_id="stream-1",
        content="hello",
        finish=False,
    )
    assert payload["cmd"] == "aibot_respond_msg"
    assert payload["headers"]["req_id"] == "req-1"
    assert payload["body"]["msgtype"] == "stream"
    assert payload["body"]["stream"] == {
        "id": "stream-1",
        "finish": False,
        "content": "hello",
    }

    client.send_final(req_id="req-1", stream_id="stream-1", content="A" * 3000)
    assert len(client.calls) >= 2
    assert all(call["headers"]["req_id"] == "req-1" for call in client.calls)
    assert [call["body"]["stream"]["id"] for call in client.calls[:2]] == ["stream-1", "stream-1-2"]
    assert all(call["body"]["stream"]["finish"] is True for call in client.calls)


def test_parse_text_message_rewrites_mentions_and_private_plain_text():
    handler = WeComStreamHandler(lambda message: BotResponse.text_response("ok"), _DummyReplyClient())
    try:
        group_message = handler.parse_callback(_payload("@股票助手 帮我看一下茅台"))
        assert group_message is not None
        assert group_message.platform == "wecom"
        assert group_message.chat_type == ChatType.GROUP
        assert group_message.content == "/chat 帮我看一下茅台"
        assert group_message.mentioned is True

        private_message = handler.parse_callback(_payload("帮我分析一下 600519", chattype="single"))
        assert private_message is not None
        assert private_message.chat_type == ChatType.PRIVATE
        assert private_message.content == "/chat 帮我分析一下 600519"

        trailing_mention = handler.parse_callback(_payload("帮我看一下 @股票助手 000021"))
        assert trailing_mention is not None
        assert trailing_mention.content == "/chat 帮我看一下 000021"
        assert trailing_mention.mentioned is True
    finally:
        handler.shutdown(wait=True)


def test_commands_are_not_rewritten():
    handler = WeComStreamHandler(lambda message: BotResponse.text_response("ok"), _DummyReplyClient())
    try:
        ask_message = handler.parse_callback(_payload("/ask 600519", chattype="single"))
        chat_message = handler.parse_callback(_payload("/chat hello", chattype="group"))
        assert ask_message is not None
        assert chat_message is not None
        assert ask_message.content == "/ask 600519"
        assert chat_message.content == "/chat hello"
    finally:
        handler.shutdown(wait=True)


def test_non_text_message_gets_controlled_reply():
    reply_client = _DummyReplyClient()
    handled = []
    handler = WeComStreamHandler(lambda message: handled.append(message), reply_client)
    try:
        handler.handle_payload(_payload(msgtype="image"))
        assert handled == []
        assert len(reply_client.calls) == 2
        assert reply_client.calls[0]["body"]["stream"]["finish"] is False
        assert reply_client.calls[1]["body"]["stream"]["finish"] is True
        assert "当前仅支持文本消息" in reply_client.calls[1]["body"]["stream"]["content"]
    finally:
        handler.shutdown(wait=True)


def test_duplicate_msgid_is_skipped():
    reply_client = _DummyReplyClient()
    handled = []

    def on_message(message):
        handled.append(message.message_id)
        return BotResponse.text_response("ok")

    handler = WeComStreamHandler(on_message, reply_client)
    try:
        handler.handle_payload(_payload("/chat hello", msgid="dup-1"))
        handler.handle_payload(_payload("/chat hello again", msgid="dup-1"))

        deadline = time.time() + 1
        while len(handled) < 1 and time.time() < deadline:
            time.sleep(0.01)

        assert handled == ["dup-1"]
        assert len([call for call in reply_client.calls if call["body"]["stream"]["finish"] is False]) == 1
    finally:
        handler.shutdown(wait=True)


def test_ttl_deduplicator_expires_message_ids():
    dedup = TTLMessageDeduplicator(ttl_seconds=1)
    assert dedup.add("msg-1", now=100.0) is True
    assert dedup.add("msg-1", now=100.5) is False
    assert dedup.add("msg-1", now=102.0) is True


def test_client_websocket_connect_disables_proxy(monkeypatch):
    client = WeComStreamClient(bot_id="bot-1", client_secret="secret-1", url="wss://example.invalid")

    class _FakeWebSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def send(self, _payload):
            return None

    with patch("bot.platforms.wecom_stream.websockets.connect", return_value=_FakeWebSocket()) as connect_mock, \
         patch.object(client, "_heartbeat_loop", new=AsyncMock()):
        monkeypatch.setattr(client, "_running", True)
        import asyncio

        asyncio.run(client._run_once())

    assert connect_mock.call_args.kwargs["proxy"] is None
    assert connect_mock.call_args.kwargs["ping_interval"] is None

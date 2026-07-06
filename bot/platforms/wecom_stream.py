# -*- coding: utf-8 -*-
"""Enterprise WeChat long-connection bot adapter."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Optional

from bot.models import BotMessage, BotResponse, ChatType
from src.formatters import chunk_content_by_max_bytes

logger = logging.getLogger(__name__)

try:
    import websockets

    WECOM_STREAM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by startup branch
    websockets = None
    WECOM_STREAM_AVAILABLE = False


WECOM_STREAM_URL = "wss://openws.work.weixin.qq.com"
WECOM_HEARTBEAT_SECONDS = 30
WECOM_REPLY_MAX_BYTES = 3500
WECOM_REPLY_INTERVAL_SECONDS = 1.0
_NON_TEXT_REPLY = "当前仅支持文本消息，请使用 /chat 提问"
_PROCESSING_REPLY = "收到，正在分析..."


class TTLMessageDeduplicator:
    """In-process TTL deduplicator for platform message IDs."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 4096) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, message_id: str, now: Optional[float] = None) -> bool:
        """Return True when message_id is new; False when it is a duplicate."""
        if not message_id:
            return True

        current = time.time() if now is None else now
        expires_before = current - self.ttl_seconds

        with self._lock:
            expired = [key for key, ts in self._seen.items() if ts < expires_before]
            for key in expired:
                self._seen.pop(key, None)

            if message_id in self._seen:
                return False

            if len(self._seen) >= self.max_size:
                oldest = sorted(self._seen.items(), key=lambda item: item[1])
                for key, _ts in oldest[: max(1, self.max_size // 10)]:
                    self._seen.pop(key, None)

            self._seen[message_id] = current
            return True


class WeComStreamReplyClient:
    """Send aibot_respond_msg payloads through an active WebSocket."""

    def __init__(
        self,
        send_json: Callable[[dict[str, Any]], Any],
        *,
        max_bytes: int = WECOM_REPLY_MAX_BYTES,
        interval_seconds: float = WECOM_REPLY_INTERVAL_SECONDS,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._send_json = send_json
        self._max_bytes = max_bytes
        self._interval_seconds = interval_seconds
        self._loop = loop

    @staticmethod
    def build_subscribe_payload(bot_id: str, secret: str, req_id: Optional[str] = None) -> dict[str, Any]:
        return {
            "cmd": "aibot_subscribe",
            "headers": {"req_id": req_id or uuid.uuid4().hex},
            "body": {"bot_id": bot_id, "secret": secret},
        }

    @staticmethod
    def build_ping_payload(req_id: Optional[str] = None) -> dict[str, Any]:
        return {"cmd": "ping", "headers": {"req_id": req_id or uuid.uuid4().hex}}

    @staticmethod
    def build_stream_reply_payload(
        *,
        req_id: str,
        stream_id: str,
        content: str,
        finish: bool,
    ) -> dict[str, Any]:
        return {
            "cmd": "aibot_respond_msg",
            "headers": {"req_id": req_id},
            "body": {
                "msgtype": "stream",
                "stream": {
                    "id": stream_id,
                    "finish": finish,
                    "content": content,
                },
            },
        }

    def _send_payload(self, payload: dict[str, Any]) -> None:
        result = self._send_json(payload)
        if not inspect.isawaitable(result):
            return

        if self._loop is None:
            asyncio.run(result)
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            self._loop.create_task(result)
            return

        future = asyncio.run_coroutine_threadsafe(result, self._loop)
        future.result(timeout=30)

    def send_stream_reply(
        self,
        *,
        req_id: str,
        stream_id: str,
        content: str,
        finish: bool,
    ) -> None:
        payload = self.build_stream_reply_payload(
            req_id=req_id,
            stream_id=stream_id,
            content=content,
            finish=finish,
        )
        self._send_payload(payload)

    def send_processing(self, *, req_id: str, stream_id: str) -> None:
        self.send_stream_reply(
            req_id=req_id,
            stream_id=stream_id,
            content=_PROCESSING_REPLY,
            finish=False,
        )

    def send_final(self, *, req_id: str, stream_id: str, content: str) -> None:
        text = content or ""
        chunks = chunk_content_by_max_bytes(text, self._max_bytes, add_page_marker=True)
        if not chunks:
            chunks = [""]

        for index, chunk in enumerate(chunks):
            # WeCom stream messages are safest when one stream id represents
            # one visible message bubble. Reusing the same id for every chunk
            # can make later chunks replace earlier content in the client.
            current_stream_id = stream_id if index == 0 else f"{stream_id}-{index + 1}"
            self.send_stream_reply(
                req_id=req_id,
                stream_id=current_stream_id,
                content=chunk,
                finish=True,
            )
            if index < len(chunks) - 1 and self._interval_seconds > 0:
                time.sleep(self._interval_seconds)


class WeComStreamHandler:
    """Convert WeCom callbacks into BotMessage and serialize work per conversation."""

    def __init__(
        self,
        on_message: Callable[[BotMessage], Any],
        reply_client: WeComStreamReplyClient,
        *,
        deduplicator: Optional[TTLMessageDeduplicator] = None,
    ) -> None:
        self._on_message = on_message
        self._reply_client = reply_client
        self._deduplicator = deduplicator or TTLMessageDeduplicator()
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="wecom-msg")
        self._pending_messages: dict[str, deque[BotMessage]] = {}
        self._active_conversations: set[str] = set()
        self._queue_lock = threading.Lock()
        self._shutdown = False

    @staticmethod
    def _get_nested(data: dict[str, Any], *keys: str) -> Any:
        current: Any = data
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    @staticmethod
    def _strip_bot_mention(text: str) -> tuple[str, bool]:
        raw = (text or "").strip()
        stripped = re.sub(r"^@[\S]+\s*", "", raw).strip()
        if stripped != raw or raw.startswith("@"):
            return stripped, True

        # Some clients place the mention after leading text. Treat any @token
        # as a group-chat mention and remove it before routing to /chat.
        stripped = re.sub(r"@[\S]+\s*", "", raw).strip()
        return stripped, stripped != raw

    @classmethod
    def _rewrite_content(cls, *, raw_content: str, chat_type: ChatType) -> tuple[str, bool]:
        content, mentioned = cls._strip_bot_mention(raw_content)
        if content.startswith("/"):
            return content, mentioned

        if chat_type == ChatType.PRIVATE:
            return f"/chat {content}".strip(), True

        if chat_type == ChatType.GROUP and mentioned:
            return f"/chat {content}".strip(), True

        return content, mentioned

    @staticmethod
    def _conversation_key(bot_message: BotMessage) -> str:
        return bot_message.chat_id or bot_message.user_id or bot_message.message_id or "unknown"

    @staticmethod
    def _message_req_id(payload: dict[str, Any]) -> str:
        return (
            str(WeComStreamHandler._get_nested(payload, "headers", "req_id") or "")
            or str(WeComStreamHandler._get_nested(payload, "header", "req_id") or "")
            or uuid.uuid4().hex
        )

    @staticmethod
    def _message_id(payload: dict[str, Any]) -> str:
        body = payload.get("body") if isinstance(payload.get("body"), dict) else payload
        return str(
            body.get("msgid")
            or body.get("msg_id")
            or body.get("message_id")
            or WeComStreamHandler._get_nested(payload, "headers", "msgid")
            or ""
        )

    def parse_callback(self, payload: dict[str, Any]) -> Optional[BotMessage]:
        """Parse aibot_msg_callback payload into BotMessage."""
        cmd = str(payload.get("cmd") or "")
        if cmd and cmd != "aibot_msg_callback":
            return None

        body = payload.get("body") if isinstance(payload.get("body"), dict) else payload
        message_id = self._message_id(payload)
        msgtype = str(body.get("msgtype") or body.get("msg_type") or "").lower()
        if msgtype and msgtype != "text":
            return BotMessage(
                platform="wecom",
                message_id=message_id,
                user_id=str(body.get("from_userid") or body.get("from_user_id") or body.get("userid") or ""),
                user_name=str(body.get("from_username") or body.get("user_name") or ""),
                chat_id=str(body.get("chatid") or body.get("chat_id") or body.get("conversation_id") or ""),
                chat_type=self._parse_chat_type(body.get("chattype") or body.get("chat_type")),
                content="",
                raw_content="",
                mentioned=True,
                timestamp=datetime.now(),
                raw_data=payload,
            )

        text = body.get("text") if isinstance(body.get("text"), dict) else {}
        raw_content = str(text.get("content") or body.get("content") or "")
        chat_type = self._parse_chat_type(body.get("chattype") or body.get("chat_type"))
        content, mentioned = self._rewrite_content(raw_content=raw_content, chat_type=chat_type)

        return BotMessage(
            platform="wecom",
            message_id=message_id,
            user_id=str(body.get("from_userid") or body.get("from_user_id") or body.get("userid") or ""),
            user_name=str(body.get("from_username") or body.get("user_name") or ""),
            chat_id=str(body.get("chatid") or body.get("chat_id") or body.get("conversation_id") or ""),
            chat_type=chat_type,
            content=content,
            raw_content=raw_content,
            mentioned=mentioned,
            mentions=[],
            timestamp=datetime.now(),
            raw_data=payload,
        )

    @staticmethod
    def _parse_chat_type(value: Any) -> ChatType:
        normalized = str(value or "").lower()
        if normalized == "group":
            return ChatType.GROUP
        if normalized in {"single", "private", "p2p"}:
            return ChatType.PRIVATE
        return ChatType.UNKNOWN

    def handle_payload(self, payload: dict[str, Any]) -> None:
        message = self.parse_callback(payload)
        if message is None:
            return

        req_id = self._message_req_id(payload)
        stream_id = uuid.uuid4().hex
        message.raw_data["_wecom_req_id"] = req_id
        message.raw_data["_wecom_stream_id"] = stream_id

        if not self._deduplicator.add(message.message_id):
            logger.info("[WeCom Stream] Duplicate message skipped: msg_id=%s", message.message_id)
            return

        logger.info(
            "[WeCom Stream] Incoming message: msg_id=%s user_id=%s chat_id=%s chat_type=%s mentioned=%s content=%s",
            message.message_id,
            message.user_id,
            message.chat_id,
            getattr(message.chat_type, "value", message.chat_type),
            message.mentioned,
            (message.content or "")[:120].replace("\n", " "),
        )
        self._reply_client.send_processing(req_id=req_id, stream_id=stream_id)
        if not message.content:
            self._reply_client.send_final(
                req_id=req_id,
                stream_id=stream_id,
                content=_NON_TEXT_REPLY,
            )
            return

        self._enqueue_message(message)

    def _enqueue_message(self, bot_message: BotMessage) -> None:
        if self._shutdown:
            logger.debug("[WeCom Stream] Handler already stopped, dropping message")
            return

        conversation_key = self._conversation_key(bot_message)
        should_start_worker = False
        with self._queue_lock:
            self._pending_messages.setdefault(conversation_key, deque()).append(bot_message)
            if conversation_key not in self._active_conversations:
                self._active_conversations.add(conversation_key)
                should_start_worker = True

        if should_start_worker:
            try:
                self._executor.submit(self._drain_conversation, conversation_key)
            except RuntimeError as exc:
                with self._queue_lock:
                    self._active_conversations.discard(conversation_key)
                    self._pending_messages.pop(conversation_key, None)
                logger.error("[WeCom Stream] 无法启动消息处理线程: %s", exc)

    def _drain_conversation(self, conversation_key: str) -> None:
        while True:
            with self._queue_lock:
                queue = self._pending_messages.get(conversation_key)
                if not queue:
                    self._pending_messages.pop(conversation_key, None)
                    self._active_conversations.discard(conversation_key)
                    return
                bot_message = queue.popleft()

            self._process_message(bot_message)

    def _process_message(self, bot_message: BotMessage) -> None:
        req_id = str(bot_message.raw_data.get("_wecom_req_id") or uuid.uuid4().hex)
        stream_id = str(bot_message.raw_data.get("_wecom_stream_id") or uuid.uuid4().hex)
        try:
            response = self._on_message(bot_message)
            if inspect.isawaitable(response):
                response = asyncio.run(response)

            text = response.text if isinstance(response, BotResponse) else ""
            self._reply_client.send_final(req_id=req_id, stream_id=stream_id, content=text)
        except Exception as exc:
            logger.error("[WeCom Stream] 消息处理失败: %s", exc)
            logger.exception(exc)
            self._reply_client.send_final(
                req_id=req_id,
                stream_id=stream_id,
                content=f"命令执行失败: {str(exc)[:100]}",
            )

    def shutdown(self, wait: bool = False) -> None:
        self._shutdown = True
        with self._queue_lock:
            self._pending_messages.clear()
            self._active_conversations.clear()
        self._executor.shutdown(wait=wait)


def _sanitize_control_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a log-safe copy of a WeCom control payload."""
    if not isinstance(payload, dict):
        return {}

    sensitive_keys = {"secret", "token", "access_token", "client_secret"}

    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                if str(key).lower() in sensitive_keys:
                    cleaned[key] = "***"
                else:
                    cleaned[key] = _sanitize(item)
            return cleaned
        if isinstance(value, list):
            return [_sanitize(item) for item in value]
        return value

    return _sanitize(payload)


class WeComStreamClient:
    """Manage the WeCom WebSocket lifecycle."""

    def __init__(
        self,
        *,
        bot_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        client_id: Optional[str] = None,
        url: str = WECOM_STREAM_URL,
    ) -> None:
        if not WECOM_STREAM_AVAILABLE:
            raise ImportError("websockets 未安装，请运行: pip install websockets")

        from src.config import get_config

        config = get_config()
        resolved_bot_id = (bot_id or getattr(config, "wecom_stream_bot_id", None) or "").strip()
        resolved_client_id = (client_id or getattr(config, "wecom_stream_client_id", None) or "").strip()
        self._client_secret = (
            client_secret or getattr(config, "wecom_stream_client_secret", None) or ""
        ).strip()
        if not resolved_bot_id and resolved_client_id:
            resolved_bot_id = resolved_client_id
            logger.warning(
                "[WeCom Stream] WECOM_STREAM_CLIENT_ID is used as BotID fallback; "
                "please migrate to WECOM_STREAM_BOT_ID."
            )

        self._bot_id = resolved_bot_id
        self._url = url
        if not self._bot_id or not self._client_secret:
            raise ValueError("企业微信长连接需要配置 WECOM_STREAM_BOT_ID 和 WECOM_STREAM_CLIENT_SECRET")

        self._background_thread: Optional[threading.Thread] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._websocket: Any = None
        self._handler: Optional[WeComStreamHandler] = None

    def _create_message_handler(self) -> Callable[[BotMessage], Any]:
        async def handle_message(message: BotMessage) -> BotResponse:
            from bot.dispatcher import get_dispatcher

            dispatcher = get_dispatcher()
            return await dispatcher.dispatch_async(message)

        return handle_message

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._websocket is None:
            raise RuntimeError("WeCom WebSocket is not connected")
        await self._websocket.send(json.dumps(payload, ensure_ascii=False))

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(WECOM_HEARTBEAT_SECONDS)
            await self._send_json(WeComStreamReplyClient.build_ping_payload())

    async def _run_once(self) -> None:
        async with websockets.connect(self._url, proxy=None, ping_interval=None) as websocket:
            self._websocket = websocket
            reply_client = WeComStreamReplyClient(self._send_json, loop=asyncio.get_running_loop())
            self._handler = WeComStreamHandler(self._create_message_handler(), reply_client)
            await self._send_json(
                WeComStreamReplyClient.build_subscribe_payload(self._bot_id, self._client_secret)
            )
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            try:
                async for raw_message in websocket:
                    try:
                        payload = json.loads(raw_message)
                    except json.JSONDecodeError:
                        logger.warning("[WeCom Stream] 忽略无法解析的 WebSocket 消息")
                        continue
                    if payload.get("cmd") == "aibot_msg_callback":
                        self._handler.handle_payload(payload)
                    else:
                        logger.info(
                            "[WeCom Stream] Control message received: %s",
                            json.dumps(_sanitize_control_payload(payload), ensure_ascii=False),
                        )
            finally:
                close_code = getattr(websocket, "close_code", None)
                close_reason = getattr(websocket, "close_reason", None)
                if close_code or close_reason:
                    logger.warning(
                        "[WeCom Stream] WebSocket closed: code=%s reason=%s",
                        close_code,
                        close_reason,
                    )
                heartbeat_task.cancel()
                if self._handler:
                    self._handler.shutdown(wait=False)
                self._websocket = None

    async def _run_forever(self) -> None:
        backoff = 1
        while self._running:
            try:
                await self._run_once()
                backoff = 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("[WeCom Stream] 运行异常: %s", exc)
                if self._running:
                    logger.info("[WeCom Stream] %s 秒后重连...", backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 60)

    def start(self) -> None:
        logger.info("[WeCom Stream] 正在启动...")
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run_forever())

    def start_background(self) -> None:
        if self._background_thread and self._background_thread.is_alive():
            logger.warning("[WeCom Stream] 客户端已在运行")
            return

        self._running = True
        self._background_thread = threading.Thread(
            target=self.start,
            daemon=True,
            name="WeComStreamClient",
        )
        self._background_thread.start()
        logger.info("[WeCom Stream] 后台客户端已启动")

    def stop(self) -> None:
        self._running = False
        logger.info("[WeCom Stream] 客户端已停止")

    @property
    def is_running(self) -> bool:
        return self._running


_stream_client: Optional[WeComStreamClient] = None


def get_wecom_stream_client() -> Optional[WeComStreamClient]:
    global _stream_client
    if _stream_client is None and WECOM_STREAM_AVAILABLE:
        try:
            _stream_client = WeComStreamClient()
        except (ImportError, ValueError) as exc:
            logger.warning("[WeCom Stream] 无法创建客户端: %s", exc)
            return None
    return _stream_client


def start_wecom_stream_background() -> bool:
    client = get_wecom_stream_client()
    if client:
        client.start_background()
        return True
    return False

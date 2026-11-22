"""Native JSON WebSocket protocol, never the older binary S2S API.

https://docs.volcengine.com/docs/doubao-voice/end-to-end-real-time-voice-full-duplex-version
Input is mono PCM16LE / 16 kHz. Output is explicitly ``pcm_s16le`` / 24 kHz;
the provider's plain ``pcm`` output format means Float32 and is incompatible.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

MAX_EVENT_BYTES = 4 * 1024 * 1024

DEEP_AGENT_TOOL = {
    "type": "function",
    "name": "delegate_deep_agent",
    "description": (
        "实际完成车险续保/加保销售中的业务查询与办理：沿用已选客户资料，查询产品、"
        "使用calc_premium按真实参数报价、核实保单，客户明确同意且写入开关允许后出单。"
        "客户资料、保费、折扣、保单与风险必须依赖对应工具结果，不能凭记忆补造。"
        "缺少车辆类别、座位数或关键承保条件先澄清，已确认条件不重复问。"
        "询价、犹豫和附和不是出单同意，操作员开场指令也不是客户同意；客户拒绝后停止推销。"
        "完整任务、当前客户和已确认条件放入question；同一任务条件改变时先查真实旧任务id，"
        "用replaces_task_id原子替换未交付版本，不先取消。独立事项和已交付后的跟进不替换。"
        "报价或保单业务前先自然回应一句，再立即调用；不要只承诺而不执行。"
        "纯思考返回completed和实际结果，实际工具执行才转后台；task_started仅代表受理，"
        "没有任何客户资料或业务结果，不能据此声称报价、成交或继续编造。"
        "省略先回应会返回native_reply_required；此时先简短承接再用新call_id重试。"
        "普通即时交流和必要澄清直接语音回复；选客由操作员拨号入口静默完成，不能反复换客。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "完整用户任务、已知条件和必要上下文"},
            "title": {"type": "string", "maxLength": 24,
                      "description": "可选的简短任务标题，概括任务，供多任务列表和完成提醒使用"},
            "replaces_task_id": {
                "type": "string", "minLength": 1, "maxLength": 128,
                "description": (
                    "可选的真实旧任务 ID：同一交付物补充或更改条件时，原子替换未交付版本"
                ),
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

TASK_CONTROL_TOOL = {
    "type": "function",
    "name": "manage_background_tasks",
    "description": (
        "读取公开对话记忆，或查询、读取、取消后台任务。"
        "独立播报未必在你的原生历史中；后续问题依赖此前答复的对象、条件或价格时，"
        "先用 history 读取已公开对话，再续说、比较、回答或委派，不能凭旧记忆换对象。"
        "即使原生上下文完全没有这些对象，也先读 history；不要未读记忆就问用户指什么。"
        "history 不启动任务，只返回已展示并开始播报的答复，不含未播报的后台结果；"
        "不能搭配 selector 或 task_ids。历史价格不代表当前售价，新增事实仍需核实。"
        "用户询问进度时用 list，主动要求读取或重听任务结果时用 get。"
        "get 会登记本轮向用户交付的任务结果，避免之后重复通知。"
        "用户改聊别的话题或打断播报不等于取消任务。"
        "用户明确取消时才调用 cancel；取消刚才那个可 selector=latest，全部取消用 all。"
        "同一任务补充条件需替换旧版本时，用 delegate_deep_agent 的 replaces_task_id，"
        "不拆成先取消再委派两个调用。"
        "指定任务不清楚时先 list，根据真实标题和 id 选择，仍有歧义则询问用户。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["history", "list", "get", "cancel"]},
            "selector": {"type": "string", "enum": ["latest", "all"]},
            "task_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


class SeeduplexProtocolError(RuntimeError):
    """A malformed upstream event; messages never include upstream payloads."""


def _safe_code(value: Any) -> int | str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        return value
    return None


class SeeduplexProviderError(RuntimeError):
    """Safe service failure, retaining only a validated code and HTTP status."""

    def __init__(self, *, code: Any = None, http_status: int | None = None):
        self.code = _safe_code(code)
        self.status_code = http_status
        suffix = f" (code {self.code})" if self.code is not None else ""
        super().__init__("Seeduplex upstream request failed" + suffix)


def decode_event(raw: str | bytes) -> dict[str, Any]:
    if not isinstance(raw, str) or len(raw.encode()) > MAX_EVENT_BYTES:
        raise SeeduplexProtocolError("Seeduplex expected a bounded JSON text frame")
    try:
        event = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise SeeduplexProtocolError("Seeduplex returned malformed JSON") from exc
    if (
        not isinstance(event, dict)
        or not isinstance(event.get("type"), str)
        or not 1 <= len(event["type"]) <= 128
    ):
        raise SeeduplexProtocolError("Seeduplex event requires a type")
    if event["type"] in ("error", "conversation.item.input_audio_transcription.failed"):
        error = event.get("error", {})
        code = error.get("code") if isinstance(error, dict) else None
        raise SeeduplexProviderError(code=code or event.get("status_code") or event.get("code"))
    return event


def decode_audio(event: dict[str, Any]) -> bytes:
    """Decode only native output audio, rejecting corrupt or partial PCM samples."""
    if event.get("type") != "response.output_audio.delta":
        raise SeeduplexProtocolError("Seeduplex event is not output audio")
    encoded = event.get("delta")
    if not isinstance(encoded, str) or len(encoded) > MAX_EVENT_BYTES:
        raise SeeduplexProtocolError("Seeduplex output audio must be bounded Base64")
    try:
        audio = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SeeduplexProtocolError("Seeduplex returned invalid Base64 audio") from exc
    if len(audio) % 2:
        raise SeeduplexProtocolError("Seeduplex returned a partial Int16 sample")
    return audio


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


def parse_tool_calls(event: dict[str, Any]) -> list[ToolCall]:
    """Parse complete calls only; the application decides which tools may execute."""
    if event.get("type") != "response.function_call_arguments.done":
        raise SeeduplexProtocolError("Seeduplex event is not a completed tool call")
    items = event.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 16:
        raise SeeduplexProtocolError("Seeduplex tool calls must be a bounded array")
    calls = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise SeeduplexProtocolError("Seeduplex tool call must be an object")
        call_id, name, arguments = item.get("call_id"), item.get("name"), item.get("arguments")
        if (
            not isinstance(call_id, str)
            or not 1 <= len(call_id) <= 256
            or call_id in seen
            or not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", name)
            or not isinstance(arguments, str)
            or len(arguments) > 16000
        ):
            raise SeeduplexProtocolError("Seeduplex tool call has invalid fields")
        try:
            parsed = json.loads(arguments)
        except ValueError as exc:
            raise SeeduplexProtocolError("Seeduplex tool arguments are not JSON") from exc
        if not isinstance(parsed, dict):
            raise SeeduplexProtocolError("Seeduplex tool arguments must be an object")
        seen.add(call_id)
        calls.append(ToolCall(call_id, name, parsed))
    return calls


def _secret(value: Any) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value or "")


def auth_headers(settings: Any) -> dict[str, str]:
    """Prefer Speech-console API Key; retain the documented legacy auth option."""
    if key := _secret(getattr(settings, "seeduplex_api_key", "")):
        return {"X-Api-Key": key}
    app_id = _secret(getattr(settings, "seeduplex_app_id", ""))
    access = _secret(getattr(settings, "seeduplex_access_key", ""))
    app_key = _secret(getattr(settings, "seeduplex_app_key", ""))
    if not app_id or not access or not app_key:
        raise ValueError(
            "Configure SEEDUPLEX_API_KEY or "
            "SEEDUPLEX_APP_ID/SEEDUPLEX_ACCESS_KEY/SEEDUPLEX_APP_KEY"
        )
    return {
        "X-Api-App-Id": app_id,
        "X-Api-Access-Key": access,
        "X-Api-App-Key": app_key,
        "X-Api-Resource-Id": getattr(settings, "seeduplex_resource_id", "volc.speech.dialog"),
        "X-Api-Request-Id": str(uuid4()),
    }


class SeeduplexStream:
    """One native conversation. Only one receive iterator may read the socket."""

    def __init__(self, connection: Any, *, timeout: float, session: dict[str, Any]):
        self._connection = connection
        self._timeout = timeout
        self._send_lock = asyncio.Lock()
        self._reading = False
        self._closing = False
        self._closed = asyncio.Event()
        self._muted = False
        self.session_id = ""
        self._session = session

    async def _send(self, kind: str, **fields: Any) -> None:
        async with self._send_lock:
            if self._closing and kind != "session.close":
                raise SeeduplexProtocolError("Seeduplex session is closing")
            payload = {"type": kind, "event_id": str(uuid4()), **fields}
            try:
                await self._connection.send(json.dumps(payload, ensure_ascii=False))
            except ConnectionClosed as exc:
                raise SeeduplexProviderError() from exc

    async def send_audio(self, audio: bytes) -> None:
        if not isinstance(audio, bytes) or len(audio) % 2 or len(audio) > 64000:
            raise ValueError("Seeduplex input must be bounded mono Int16 PCM")
        if not audio:
            return
        if self._muted:
            await self.set_muted(False)
        await self._send("input_audio_buffer.append", audio=base64.b64encode(audio).decode("ascii"))

    async def commit_audio(self) -> None:
        """Optional explicit endpoint; continuous mode normally lets the model decide."""
        await self._send("input_audio_buffer.commit")

    async def set_muted(self, muted: bool) -> None:
        if not isinstance(muted, bool):
            raise ValueError("Seeduplex mute state must be boolean")
        if muted == self._muted:
            return
        await self._send("input_audio_mute.commit" if muted else "input_audio_unmute.commit")
        self._muted = muted

    async def cancel(self) -> None:
        await self._send("response.cancel")

    async def update_instructions(self, instructions: str) -> None:
        if (
            not isinstance(instructions, str)
            or not instructions.strip()
            or len(instructions) > 16000
        ):
            raise ValueError("Seeduplex instructions must be bounded nonempty text")
        session = {**self._session, "instructions": instructions}
        await self._send("session.update", session=session, extension={})
        self._session = session

    async def replace_text(self, text: str, *, final: bool = False) -> None:
        if not isinstance(text, str) or len(text) > 16000:
            raise ValueError("Seeduplex replacement text must be bounded text")
        await self._send(
            "speech_text_buffer.replacement.commit"
            if final
            else "speech_text_buffer.replacement.append",
            text=text,
        )

    async def send_text(self, text: str) -> None:
        """Speak supplied text; this is NOT an undocumented native text-query API."""
        await self.replace_text(text, final=True)

    async def greet(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Seeduplex greeting must be bounded nonempty text")
        await self._send("speech_text_buffer.commit", text=text)

    async def send_tool_result(self, call_id: str, text: str) -> None:
        await self.send_tool_results([{"call_id": call_id, "text": text}])

    async def send_tool_results(self, results: list[dict[str, str]]) -> None:
        if not isinstance(results, list) or not 1 <= len(results) <= 16:
            raise ValueError("Seeduplex tool results must be a bounded nonempty list")
        items = []
        seen = set()
        for result in results:
            call_id, text = result.get("call_id"), result.get("text")
            if (
                not isinstance(call_id, str)
                or not 1 <= len(call_id) <= 256
                or call_id in seen
                or not isinstance(text, str)
                or len(text) > 64000
            ):
                raise ValueError("Seeduplex tool result has invalid fields")
            seen.add(call_id)
            items.append(
                {
                    "call_id": call_id,
                    "role": "tool",
                    "content": [{"type": "input_text", "text": text}],
                }
            )
        await self._send("conversation.item.create", items=items)

    async def add_history(self, exchanges: list[tuple[str, str]]) -> None:
        """Inject only completed, paired public Q&A, at most 20 exchanges per request."""
        if not isinstance(exchanges, list) or len(exchanges) > 20:
            raise ValueError("Seeduplex history accepts at most 20 completed exchanges")
        items = []
        for pair in exchanges:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Seeduplex history requires paired questions and answers")
            for role, text in zip(("user", "assistant"), pair, strict=True):
                if not isinstance(text, str) or not text.strip() or len(text) > 16000:
                    raise ValueError("Seeduplex history content must be bounded nonempty text")
                items.append(
                    {
                        "id": str(uuid4()),
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
        if items:
            await self._send("conversation.item.create", items=items)

    async def add_assistant_answer(self, text: str) -> None:
        """Complete an ASR user turn already present in the native conversation."""
        if not isinstance(text, str) or not text.strip() or len(text) > 16000:
            raise ValueError("Seeduplex history answer must be bounded nonempty text")
        await self._send(
            "conversation.item.create",
            items=[
                {
                    "id": str(uuid4()),
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": text}],
                }
            ],
        )

    async def receive(self) -> AsyncIterator[dict[str, Any]]:
        if self._reading:
            raise SeeduplexProtocolError("Seeduplex permits only one event consumer")
        self._reading = True
        try:
            while not self._closed.is_set():
                try:
                    event = decode_event(await self._connection.recv())
                except ConnectionClosed as exc:
                    if self._closing:
                        return
                    raise SeeduplexProviderError() from exc
                if event["type"] == "session.closed":
                    self._closed.set()
                    yield event
                    return
                if event["type"] == "response.output_audio.delta":
                    decode_audio(event)  # Fail before malformed audio reaches the application.
                yield event
        finally:
            self._reading = False

    async def close(self) -> None:
        if self._closing or self._closed.is_set():
            return
        self._closing = True
        # Context-manager exit still closes the socket when the service fails to ack.
        with contextlib.suppress(Exception):
            async with asyncio.timeout(min(self._timeout, 3)):
                await self._send("session.close")
                if self._reading:
                    await self._closed.wait()
                else:
                    while not self._closed.is_set():
                        event = decode_event(await self._connection.recv())
                        if event["type"] == "session.closed":
                            self._closed.set()


class SeeduplexClient:
    def __init__(self, settings: Any):
        self.settings = settings

    async def speech(self, text: str) -> AsyncIterator[bytes]:
        """Speak prepared task results through Seeduplex, with isolated event ownership.

        The documented greeting event accepts supplied speech text even without
        an active user-audio response. Replacement events require that response
        and cannot implement keyboard input or recovery on their own.
        """
        if not isinstance(text, str) or not text.strip() or len(text) > 64000:
            raise ValueError("Seeduplex speech requires bounded nonempty text")
        async with self.stream(
            instructions="你是中文语音助手，播报给定的任务结果。", tools=[]
        ) as stream:
            await stream.set_muted(True)
            async with contextlib.aclosing(stream.receive()) as events:
                for offset in range(0, len(text), 4000):
                    await stream.greet(text[offset : offset + 4000])
                    response_id = None
                    started = False
                    produced = False
                    while True:
                        async with asyncio.timeout(self.settings.seeduplex_timeout):
                            event = await anext(events)
                        kind = event["type"]
                        if kind == "response.output_audio.started":
                            started = True
                            response_id = event.get("response_id") or None
                        elif kind == "response.output_audio.delta" and started:
                            if audio := decode_audio(event):
                                produced = True
                                yield audio
                        elif kind == "response.output_audio.done" and (
                            started and (not response_id or event.get("response_id") == response_id)
                        ):
                            if not produced:
                                raise SeeduplexProtocolError("Seeduplex speech returned no PCM")
                            break

    @asynccontextmanager
    async def stream(
        self, *, instructions: str, tools: list[dict[str, Any]] | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> AsyncIterator[SeeduplexStream]:
        settings = self.settings
        timeout = settings.seeduplex_timeout
        missing = [
            name.upper() for name in ("seeduplex_url", "seeduplex_model", "seeduplex_speaker")
            if not isinstance(getattr(settings, name, None), str)
            or not getattr(settings, name).strip()
        ]
        if missing:
            raise ValueError("Configure " + ", ".join(missing))
        headers = auth_headers(settings)
        model = settings.seeduplex_model
        request = {
            "model": model,
            "instructions": instructions,
            "audio": {
                "input": {"format": {"type": "pcm", "rate": 16000}},
                "output": {
                    "format": {"type": "pcm_s16le", "rate": 24000},
                    "voice": settings.seeduplex_speaker,
                    "speed": 0,
                    "loudness": 0,
                },
            },
            "tools": [DEEP_AGENT_TOOL] if tools is None else tools,
        }
        extension = {}
        if history is not None and (not isinstance(history, list) or len(history) > 20):
            raise ValueError("Seeduplex initial history accepts at most 20 exchanges")
        if history:
            dialog_context = []
            for pair in history:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("Seeduplex initial history requires paired Q&A")
                for role, text in zip(("user", "assistant"), pair, strict=True):
                    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
                        raise ValueError("Seeduplex initial history must be bounded nonempty text")
                    dialog_context.append({"role": role, "text": text})
            extension = {"dialog": {"dialog_context": dialog_context}}
        try:
            async with connect(
                settings.seeduplex_url,
                additional_headers=headers,
                open_timeout=timeout,
                close_timeout=3,
                max_size=MAX_EVENT_BYTES,
                ping_interval=20,
                proxy=None,
            ) as connection:
                stream = SeeduplexStream(connection, timeout=timeout, session=request)
                try:
                    await stream._send("session.create", session=request, extension=extension)
                    async with asyncio.timeout(timeout):
                        created = decode_event(await connection.recv())
                    if created["type"] != "session.created":
                        raise SeeduplexProtocolError("Seeduplex expected session.created")
                    session = created.get("session", {})
                    if isinstance(session, dict) and isinstance(session.get("id"), str):
                        stream.session_id = session["id"]
                    yield stream
                finally:
                    await stream.close()
        except InvalidStatus as exc:
            response = exc.response
            code = response.headers.get("X-Api-Status-Code")
            raise SeeduplexProviderError(code=code, http_status=response.status_code) from None
        except ConnectionClosed:
            raise SeeduplexProviderError() from None

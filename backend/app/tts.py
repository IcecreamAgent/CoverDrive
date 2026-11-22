"""豆包语音合成 2.0（Volcengine bidirectional TTS）流式适配器。

自包含实现，协议与官方 demo（protocols.py）一致：
- 二进制帧：4 字节头 + 可选 event/session_id/connect_id + 4 字节长度前缀载荷。
- 生命周期：StartConnection → StartSession → TaskRequest(text) → FinishSession → FinishConnection。
- 下行：TTSResponse（AUDIO_ONLY_SERVER）逐块产出 PCM，SessionFinished 结束本轮。

端点 wss://openspeech.bytedance.com/api/v3/tts/bidirection，X-Api-Key 鉴权。
"""
import asyncio
import io
import json
import logging
import ssl
import struct
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp
import numpy as np
import soundfile as sf

from .config import settings

logger = logging.getLogger(__name__)

URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
RECEIVE_TIMEOUT = 60


# ── 协议常量（与官方 protocols.py 一致）──────────────────────────────
class MsgType:
    FULL_CLIENT_REQUEST = 0b0001
    AUDIO_ONLY_CLIENT = 0b0010
    FULL_SERVER_RESPONSE = 0b1001
    AUDIO_ONLY_SERVER = 0b1011
    ERROR = 0b1111


class Flags:
    NO_SEQ = 0b0000
    POSITIVE_SEQ = 0b0001
    LAST_NO_SEQ = 0b0010
    NEGATIVE_SEQ = 0b0011
    WITH_EVENT = 0b0100


class Serialization:
    RAW = 0b0000
    JSON = 0b0001


class Compression:
    NONE = 0b0000
    GZIP = 0b0001


class Event:
    START_CONNECTION = 1
    FINISH_CONNECTION = 2
    CONNECTION_STARTED = 50
    CONNECTION_FAILED = 51
    CONNECTION_FINISHED = 52
    START_SESSION = 100
    CANCEL_SESSION = 101
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_CANCELED = 151
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    TASK_REQUEST = 200
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352


# 无 session_id 的事件（连接级）
_NO_SESSION_EVENTS = {
    Event.START_CONNECTION,
    Event.FINISH_CONNECTION,
    Event.CONNECTION_STARTED,
    Event.CONNECTION_FAILED,
    Event.CONNECTION_FINISHED,
}
# 需要读 connect_id 的服务器事件
_CONNECT_EVENTS = {
    Event.CONNECTION_STARTED,
    Event.CONNECTION_FAILED,
    Event.CONNECTION_FINISHED,
}


# ── 帧编解码 ──────────────────────────────────────────────────────────
def _marshal(event: int, session_id: str, payload: bytes) -> bytes:
    """FullClientRequest + WithEvent 客户端消息。"""
    buf = bytearray()
    buf.append(0x11)  # version=1, header_size=1（4 字节头）
    buf.append((MsgType.FULL_CLIENT_REQUEST << 4) | Flags.WITH_EVENT)
    buf.append((Serialization.JSON << 4) | Compression.NONE)
    buf.append(0)
    buf.extend(struct.pack(">i", event))
    if event not in _NO_SESSION_EVENTS:
        sid = session_id.encode("utf-8")
        buf.extend(struct.pack(">I", len(sid)))
        buf.extend(sid)
    buf.extend(struct.pack(">I", len(payload)))
    buf.extend(payload)
    return bytes(buf)


@dataclass
class _ServerMsg:
    type: int = 0
    flag: int = 0
    event: int = 0
    session_id: str = ""
    connect_id: str = ""
    error_code: int = 0
    payload: bytes = b""


def _unmarshal(data: bytes) -> _ServerMsg:
    msg_type = (data[1] >> 4) & 0x0F
    flag = data[1] & 0x0F
    offset = 4
    m = _ServerMsg(type=msg_type, flag=flag)
    if msg_type == MsgType.ERROR:
        m.error_code = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
    elif msg_type in (MsgType.FULL_SERVER_RESPONSE, MsgType.AUDIO_ONLY_SERVER) and flag in (
        Flags.POSITIVE_SEQ,
        Flags.NEGATIVE_SEQ,
    ):
        offset += 4  # sequence
    if flag == Flags.WITH_EVENT:
        m.event = struct.unpack(">i", data[offset : offset + 4])[0]
        offset += 4
        if m.event not in _NO_SESSION_EVENTS:
            size = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            m.session_id = data[offset : offset + size].decode("utf-8", "replace")
            offset += size
        if m.event in _CONNECT_EVENTS:
            size = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            m.connect_id = data[offset : offset + size].decode("utf-8", "replace")
            offset += size
    size = struct.unpack(">I", data[offset : offset + 4])[0]
    offset += 4
    m.payload = data[offset : offset + size]
    return m


def _payload(event: int, text: str = "") -> bytes:
    """构造 JSON 载荷（namespace=BidirectionalTTS）。"""
    req_params = {
        "speaker": settings.tts_speaker,
        "audio_params": {
            "format": settings.tts_format,
            "sample_rate": settings.tts_sample_rate,
            "speech_rate": settings.tts_speech_rate,
            "loudness_rate": settings.tts_loudness_rate,
        },
    }
    if text:
        req_params["text"] = text
    body = {
        "user": {"uid": "voice-agent"},
        "event": event,
        "namespace": "BidirectionalTTS",
        "req_params": req_params,
    }
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


# ── 流式合成 ──────────────────────────────────────────────────────────
class TtsStream:
    """一条 WebSocket 连接，顺序复用多个合成 session（每句一个 session）。

    open() 建立连接并预热（隐藏 TLS/WS 握手）；每句 begin_session() 开新 session →
    feed → chunks 收 PCM。会话结束后连接仍在，可继续 begin_session() 下一句，
    省去每句重连的开销。
    """

    def __init__(self) -> None:
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_id = ""  # 由 begin_session() 生成
        self._closed = False

    @property
    def sample_rate(self) -> int:
        return settings.tts_sample_rate

    async def open(self) -> None:
        if not settings.tts_api_key:
            raise RuntimeError("未配置 TTS_API_KEY")
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        headers = {
            "X-Api-Key": settings.tts_api_key,
            "X-Api-Resource-Id": settings.tts_resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(URL, headers=headers, ssl=ssl_ctx)
            await self._send(Event.START_CONNECTION, "", b"{}")
            await self._wait_event(Event.CONNECTION_STARTED)
        except BaseException:
            await self.close()
            raise

    async def begin_session(self) -> None:
        """在当前连接上开启下一个合成 session（复用连接，跳过重连）。"""
        self._session_id = uuid.uuid4().hex
        await self._send(Event.START_SESSION, self._session_id, _payload(Event.START_SESSION))
        await self._wait_event(Event.SESSION_STARTED)

    async def _send(self, event: int, session_id: str, payload: bytes) -> None:
        if self._ws is None:
            raise RuntimeError("TTS 连接未建立")
        await self._ws.send_bytes(_marshal(event, session_id, payload))

    async def _recv(self) -> _ServerMsg:
        while True:
            raw = await asyncio.wait_for(self._ws.receive(), timeout=RECEIVE_TIMEOUT)
            if raw.type == aiohttp.WSMsgType.BINARY:
                return _unmarshal(raw.data)
            if raw.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f"TTS WebSocket 错误: {raw.data}")
            if raw.type == aiohttp.WSMsgType.CLOSED:
                raise RuntimeError("TTS WebSocket 已关闭")
            # TEXT / PING / PONG 忽略

    def _raise_if_error(self, m: _ServerMsg) -> None:
        if m.type == MsgType.ERROR:
            raise RuntimeError(f"TTS 错误 code={m.error_code}: {m.payload[:200]!r}")
        if m.event == Event.SESSION_FAILED:
            raise RuntimeError(f"TTS session 失败: {m.payload[:200]!r}")
        if m.event == Event.CONNECTION_FAILED:
            raise RuntimeError(f"TTS 连接失败: {m.payload[:200]!r}")

    async def _wait_event(self, event: int) -> None:
        while True:
            m = await self._recv()
            self._raise_if_error(m)
            if m.event == event:
                return

    async def feed(self, text: str, is_last: bool) -> None:
        text = (text or "").strip()
        if not text:
            if is_last:
                await self._send(Event.FINISH_SESSION, self._session_id, b"{}")
            return
        await self._send(Event.TASK_REQUEST, self._session_id, _payload(Event.TASK_REQUEST, text))
        if is_last:
            await self._send(Event.FINISH_SESSION, self._session_id, b"{}")

    async def chunks(self) -> AsyncIterator[bytes]:
        """逐块产出 PCM，直到 SessionFinished。"""
        while True:
            m = await self._recv()
            self._raise_if_error(m)
            if m.type == MsgType.AUDIO_ONLY_SERVER and m.payload:
                yield m.payload
            if m.event == Event.SESSION_FINISHED:
                return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ws and not self._ws.closed:
            try:
                await self._send(Event.FINISH_CONNECTION, "", b"{}")
            except Exception:
                pass
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "TtsStream":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()


# ── 兼容旧接口（整段合成，/api/voice 兜底）──────────────────────────
async def synthesize(text: str) -> bytes:
    """整段文本合成，返回 WAV 字节。"""
    pcm = bytearray()
    async with TtsStream() as stream:
        await stream.begin_session()
        await stream.feed(text, is_last=True)
        async for chunk in stream.chunks():
            pcm.extend(chunk)
    wav = io.BytesIO()
    sf.write(
        wav,
        np.frombuffer(bytes(pcm), dtype=np.int16),
        samplerate=settings.tts_sample_rate,
        subtype="PCM_16",
        format="WAV",
    )
    return wav.getvalue()

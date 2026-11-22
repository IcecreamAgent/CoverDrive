"""火山 SAUC WebSocket ASR 流式适配器。

自包含实现（不再动态 import 示例脚本），协议与官方 demo 保持一致：
- 输入：16kHz / mono / int16 小端裸 PCM，边收边切 100ms 段发送。
- 输出：实时中间结果（`result.text`）与最终结果。

端点使用 `bigmodel_async`（豆包流式 ASR，`model_name=bigmodel`），默认开启二遍识别并为
每轮连接注入受限的对话上下文。中间响应会回传实时识别文本，最终包给出完整转写。
鉴权优先用 `X-Api-Key`（新控制台 API Key），
未配置 `asr_api_key` 时回退 `X-Api-App-Key` + `X-Api-Access-Key`（旧控制台无签名）。
"""
import gzip
import json
import logging
import struct
import uuid
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import aiohttp

from .config import settings

logger = logging.getLogger(__name__)

URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
DEFAULT_SAMPLE_RATE = 16000
SEGMENT_DURATION_MS = 100


class AsrContextHistory:
    """Bounded per-call dialog context formatted for Doubao ASR.

    The provider expects newest entries first. Keeping a small local window is
    intentional: recent wording and domain terms help recognition, while long
    transcripts can bias unrelated later utterances.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_turns: int = 4,
        max_chars: int = 600,
        domain_hint: str = "",
    ) -> None:
        self.enabled = enabled
        self.max_messages = max(0, max_turns) * 2
        self.max_chars = max(0, max_chars)
        self.domain_hint = " ".join(domain_hint.split())
        self._messages: list[tuple[str, str]] = []

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join((text or "").split())

    def add(self, role: str, text: str) -> None:
        text = self._clean(text)
        if not text or not self.enabled:
            return
        if role == "坐席" and self._messages and self._messages[-1][0] == role:
            previous = self._messages[-1][1]
            self._messages[-1] = (role, previous + text)
        else:
            self._messages.append((role, text))
        if self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def context_data(self) -> list[dict[str, str]]:
        if not self.enabled or self.max_chars <= 0:
            return []
        domain = f"业务场景：{self.domain_hint}" if self.domain_hint else ""
        # Reserve room for the stable domain hint so insurance terminology is
        # still available when recent dialog is relatively long.
        dialog_budget = max(0, self.max_chars - len(domain))
        items: list[dict[str, str]] = []
        used = 0
        for role, text in reversed(self._messages):
            entry = f"{role}：{text}"
            remaining = dialog_budget - used
            if remaining <= 0:
                break
            if len(entry) > remaining:
                entry = entry[:remaining]
            if entry:
                items.append({"text": entry})
                used += len(entry)
            if used >= dialog_budget:
                break
        if domain and len(domain) <= self.max_chars - used:
            items.append({"text": domain})
        return items


# ── 协议常量 ────────────────────────────────────────────────────────
class MessageType:
    CLIENT_FULL_REQUEST = 0b0001
    CLIENT_AUDIO_ONLY_REQUEST = 0b0010
    SERVER_FULL_RESPONSE = 0b1001
    SERVER_ERROR_RESPONSE = 0b1111


class Flags:
    NO_SEQUENCE = 0b0000
    POS_SEQUENCE = 0b0001
    NEG_SEQUENCE = 0b0010
    NEG_WITH_SEQUENCE = 0b0011


class Serialization:
    NO_SERIALIZATION = 0b0000
    JSON = 0b0001


class Compression:
    GZIP = 0b0001


@dataclass
class AsrResponse:
    code: int = 0
    is_last_package: bool = False
    payload_msg: Optional[dict] = None

    @property
    def text(self) -> str:
        msg = self.payload_msg or {}
        if not isinstance(msg, dict):
            return ""
        result = msg.get("result", {})
        if not isinstance(result, dict):
            return ""

        # `result_type=full` should make result.text cumulative. Prefer the
        # utterance list when present because it is the authoritative set of
        # ASR segments and avoids losing an earlier segment if text contains
        # only the latest VAD sentence in an intermediate/final response.
        utterances = result.get("utterances") or []
        parts = [
            item.get("text", "").strip()
            for item in utterances
            if isinstance(item, dict) and item.get("text", "").strip()
        ]
        if parts:
            text = parts[0]
            for part in parts[1:]:
                # Chinese punctuation needs no separator. Preserve word
                # boundaries when two English/alphanumeric segments meet.
                separator = " " if (
                    text[-1:].isascii() and text[-1:].isalnum()
                    and part[:1].isascii() and part[:1].isalnum()
                ) else ""
                text += separator + part
            return text
        return result.get("text", "") or ""


def _gzip(data: bytes) -> bytes:
    return gzip.compress(data)


def _gunzip(data: bytes) -> bytes:
    return gzip.decompress(data)


def _full_client_request(seq: int, context_data: list[dict[str, str]] | None = None) -> bytes:
    header = bytearray()
    header.append(0x11)  # Protocol V1, header size 1
    header.append((MessageType.CLIENT_FULL_REQUEST << 4) | Flags.POS_SEQUENCE)
    header.append((Serialization.JSON << 4) | Compression.GZIP)
    header.append(0x00)

    request = {
        "model_name": "bigmodel",
        "enable_itn": True,
        "enable_punc": True,
        "enable_ddc": True,
        "show_utterances": True,
        "result_type": "full",
        "enable_nonstream": settings.asr_two_pass_enabled,
    }
    if context_data:
        request["corpus"] = {
            # The API field is a JSON string rather than a nested object.
            "context": json.dumps(
                {"context_type": "dialog_ctx", "context_data": context_data},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        }
    payload = {
        "user": {"uid": "voice-agent"},
        "audio": {"format": "wav", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1},
        "request": request,
    }
    compressed = _gzip(json.dumps(payload).encode("utf-8"))
    req = bytearray()
    req.extend(header)
    req.extend(struct.pack(">i", seq))
    req.extend(struct.pack(">I", len(compressed)))
    req.extend(compressed)
    return bytes(req)


def _audio_only_request(seq: int, segment: bytes, is_last: bool) -> bytes:
    header = bytearray()
    header.append(0x11)
    if is_last:
        header.append((MessageType.CLIENT_AUDIO_ONLY_REQUEST << 4) | Flags.NEG_WITH_SEQUENCE)
    else:
        header.append((MessageType.CLIENT_AUDIO_ONLY_REQUEST << 4) | Flags.POS_SEQUENCE)
    header.append((Serialization.NO_SERIALIZATION << 4) | Compression.GZIP)
    header.append(0x00)

    compressed = _gzip(segment)
    req = bytearray()
    req.extend(header)
    req.extend(struct.pack(">i", -seq if is_last else seq))
    req.extend(struct.pack(">I", len(compressed)))
    req.extend(compressed)
    return bytes(req)


def _parse_response(msg: bytes) -> AsrResponse:
    response = AsrResponse()
    header_size = msg[0] & 0x0F
    message_type = msg[1] >> 4
    flags = msg[1] & 0x0F
    serialization = msg[2] >> 4
    compression = msg[2] & 0x0F

    payload = msg[header_size * 4:]

    if flags & 0x01:
        payload = payload[4:]  # 跳过 payload_sequence
    if flags & 0x02:
        response.is_last_package = True
    if flags & 0x04:
        payload = payload[4:]  # 跳过 event

    if message_type == MessageType.SERVER_FULL_RESPONSE:
        payload = payload[4:]  # 跳过 payload_size
    elif message_type == MessageType.SERVER_ERROR_RESPONSE:
        response.code = struct.unpack(">i", payload[:4])[0]
        payload = payload[8:]

    if not payload:
        return response

    if compression == Compression.GZIP:
        try:
            payload = _gunzip(payload)
        except Exception:
            logger.error("Failed to decompress ASR payload", exc_info=True)
            return response

    if serialization == Serialization.JSON:
        try:
            response.payload_msg = json.loads(payload.decode("utf-8"))
        except Exception:
            logger.error("Failed to parse ASR payload", exc_info=True)
    return response


# ── 流式客户端 ────────────────────────────────────────────────────────
@dataclass
class AsrStreamResult:
    text: str = ""
    is_final: bool = False


class StreamingAsrClient:
    """接收裸 PCM，边送边识别，产出中间/最终文本。"""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        context_data: list[dict[str, str]] | None = None,
    ):
        self.sample_rate = sample_rate
        self.context_data = context_data or []
        self.seq = 1
        self._segment_size = sample_rate * 2 * SEGMENT_DURATION_MS // 1000  # int16 mono
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._buffer = b""
        self._finalized = False
        self._header_sent = False

    @property
    def is_open(self) -> bool:
        """连接仍可写；预热连接可能被服务端因空闲主动关闭。"""
        return self._ws is not None and not self._ws.closed and not self._finalized

    def _wav_header(self) -> bytes:
        """构造 44 字节 WAV 头（PCM 16k mono 16bit，大小字段用占位值）。

        服务端按请求体 `format: "wav"` 解析音频，只读取 fmt 块，
        忽略 RIFF/data 大小字段（流式下总长未知），故用 0xFFFFFFFF 占位。
        """
        byte_rate = self.sample_rate * 1 * 16 // 8
        header = b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, self.sample_rate, byte_rate, 2, 16)
        header += b"data" + struct.pack("<I", 0xFFFFFFFF)
        return header

    async def _connect(self) -> None:
        import ssl

        headers = {
            "X-Api-Resource-Id": settings.asr_resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        if settings.asr_api_key:
            headers["X-Api-Key"] = settings.asr_api_key
        else:
            headers["X-Api-Access-Key"] = settings.asr_access_key
            headers["X-Api-App-Key"] = settings.asr_app_key
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(URL, headers=headers, ssl=ssl_ctx)
        # 发送 full client request
        await self._ws.send_bytes(_full_client_request(self.seq, self.context_data))
        self.seq += 1
        # 等待首个响应（确认协议握手）
        first = await self._ws.receive()
        if first.type == aiohttp.WSMsgType.BINARY:
            logger.info("ASR handshake response received")
        elif first.type == aiohttp.WSMsgType.ERROR:
            raise RuntimeError(f"ASR 握手失败: {first.data}")
        else:
            raise RuntimeError(f"ASR 握手收到异常消息类型: {first.type}")

    async def feed(self, pcm: bytes) -> None:
        """喂入一段裸 PCM（16k mono int16，不带 WAV 头），内部缓冲按 100ms 切段。

        首次调用前自动前置 44 字节 WAV 头（服务端按 format:"wav" 解析，裸 PCM 会被忽略）。
        始终在缓冲里保留一段（约 100ms），由 finalize 作为末段发送，
        避免音频长度恰为整段倍数时发出"空末段"导致服务端返回空转写。
        """
        if not self._header_sent:
            self._buffer += self._wav_header()
            self._header_sent = True
        self._buffer += pcm
        while len(self._buffer) >= self._segment_size * 2 and not self._finalized:
            segment = self._buffer[: self._segment_size]
            self._buffer = self._buffer[self._segment_size:]
            await self._ws.send_bytes(_audio_only_request(self.seq, segment, is_last=False))
            self.seq += 1

    async def finalize(self) -> None:
        """结束音频流：把保留的末段标记 last 发送（末段非空）。"""
        if self._finalized:
            return
        self._finalized = True
        tail = self._buffer
        self._buffer = b""
        # 剩余超过一段时才继续拆，保证标记 is_last 的末段非空
        while len(tail) > self._segment_size:
            segment = tail[: self._segment_size]
            tail = tail[self._segment_size:]
            await self._ws.send_bytes(_audio_only_request(self.seq, segment, is_last=False))
            self.seq += 1
        await self._ws.send_bytes(_audio_only_request(self.seq, tail, is_last=True))

    async def results(self) -> AsyncIterator[AsrStreamResult]:
        """持续产出识别结果，直到服务端标记 is_last_package。"""
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                response = _parse_response(msg.data)
                text = response.text
                if text:
                    yield AsrStreamResult(text=text, is_final=response.is_last_package)
                if response.is_last_package or response.code != 0:
                    break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"ASR WebSocket error: {msg.data}")
                break
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                break

    async def close(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()


def _strip_wav_header(audio: bytes) -> bytes:
    """若输入是 WAV（带 RIFF 头），剥离头部返回裸 PCM；否则原样返回。"""
    if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        pos = 12
        while pos < len(audio) - 8:
            subchunk_id = audio[pos:pos + 4]
            subchunk_size = struct.unpack("<I", audio[pos + 4:pos + 8])[0]
            if subchunk_id == b"data":
                return audio[pos + 8:pos + 8 + subchunk_size]
            pos += 8 + subchunk_size
    return audio


# ── 兼容旧接口（整段上传 /api/voice 兜底用）──────────────────────────
async def transcribe(audio: bytes, filename: str = "audio.wav") -> str:
    """整段音频转写（保留给 /api/voice 兜底）。

    注意：仅支持 16k PCM 或带 RIFF 头的 WAV；webm 等压缩格式需先转码。
    """
    if not settings.asr_app_key or not settings.asr_access_key:
        raise RuntimeError("未配置 ASR_APP_KEY / ASR_ACCESS_KEY")
    client = StreamingAsrClient()
    try:
        await client._connect()
        await client.feed(_strip_wav_header(audio))
        await client.finalize()
        last = ""
        async for result in client.results():
            if result.text:
                last = result.text
        return last
    finally:
        await client.close()

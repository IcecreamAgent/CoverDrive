"""Volcengine streaming ASR over the v3 binary WebSocket protocol.

Protocol reference: https://www.volcengine.com/docs/6561/1354869
Audio is raw little-endian signed 16-bit PCM, 16 kHz, mono. Response
``text`` values are cumulative; ``is_final`` marks the final protocol packet,
not a VAD utterance boundary inside a recording.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed


class ASRError(RuntimeError):
    """Recognition failed or the upstream stream ended unexpectedly."""


class ASRProtocolError(ASRError):
    """An upstream binary frame doesn't match the documented protocol."""


@dataclass(frozen=True, slots=True)
class ASRResult:
    text: str
    is_final: bool
    utterance_definite: bool = False


@dataclass(frozen=True, slots=True)
class _Response:
    payload: dict[str, Any]
    is_final: bool


def _secret(value: Any) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


def _encode_request(
    message_type: int, payload: bytes, sequence: int, *, final: bool = False
) -> bytes:
    """Encode client metadata (type 1) or client audio (type 2)."""
    if sequence <= 0:
        raise ValueError("ASR sequence must be positive before final encoding")
    compressed = gzip.compress(payload)
    flags = 3 if final else 1
    serialization = 1 if message_type == 1 else 0
    header = bytes((0x11, (message_type << 4) | flags, (serialization << 4) | 1, 0))
    return (
        header + struct.pack(">iI", -sequence if final else sequence, len(compressed)) + compressed
    )


def _decode_response(frame: bytes) -> _Response:
    if len(frame) < 4 or frame[0] >> 4 != 1:
        raise ASRProtocolError("Invalid ASR protocol version or truncated header")
    header_size = (frame[0] & 0x0F) * 4
    if header_size < 4 or len(frame) < header_size:
        raise ASRProtocolError("Invalid ASR header length")
    message_type, flags = frame[1] >> 4, frame[1] & 0x0F
    serialization, compression = frame[2] >> 4, frame[2] & 0x0F
    position = header_size

    def read_int(*, signed: bool = False) -> int:
        nonlocal position
        if len(frame) < position + 4:
            raise ASRProtocolError("Truncated ASR integer field")
        value = int.from_bytes(frame[position : position + 4], "big", signed=signed)
        position += 4
        return value

    sequence = read_int(signed=True) if flags & 1 else 0
    # Event-bearing frames are supported by the official demo parser too.
    if flags & 4:
        read_int(signed=True)
    if message_type not in (9, 15):
        raise ASRProtocolError(f"Unsupported ASR response type: {message_type}")
    error_code = read_int() if message_type == 15 else None
    payload_size = read_int()
    if len(frame) - position != payload_size:
        raise ASRProtocolError("ASR payload length does not match frame length")
    payload = frame[position:]
    if compression == 1:
        try:
            payload = gzip.decompress(payload)
        except (OSError, EOFError) as error:
            raise ASRProtocolError("Invalid gzip ASR payload") from error
    elif compression != 0:
        raise ASRProtocolError(f"Unsupported ASR compression: {compression}")
    if error_code is not None:
        # Don't expose the upstream payload, which can contain request metadata.
        raise ASRError(f"ASR service returned error code {error_code}")
    if not payload:
        data: dict[str, Any] = {}
    elif serialization == 1:
        try:
            data = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ASRProtocolError("Invalid JSON ASR payload") from error
        if not isinstance(data, dict):
            raise ASRProtocolError("ASR response JSON must be an object")
    else:
        raise ASRProtocolError(f"Unsupported ASR serialization: {serialization}")
    return _Response(data, bool(flags & 2) or sequence < 0)


class ASRStream:
    def __init__(self, connection: ClientConnection, *, timeout: float):
        self._connection = connection
        self._timeout = timeout
        self._sequence = 2  # Metadata request is sequence 1.
        self._final_sent = False
        self._send_lock = asyncio.Lock()

    async def send_audio(self, audio: bytes, *, final: bool = False) -> None:
        if len(audio) % 2:
            raise ValueError("ASR PCM must contain complete Int16 samples")
        async with self._send_lock:
            if self._final_sent:
                raise ASRError("Audio was sent after the final ASR packet")
            await self._connection.send(_encode_request(2, audio, self._sequence, final=final))
            self._sequence += 1
            self._final_sent = final

    async def receive(self) -> AsyncIterator[ASRResult]:
        last_text = ""
        last_definite = False
        while True:
            try:
                async with asyncio.timeout(self._timeout):
                    frame = await self._connection.recv()
            except TimeoutError as error:
                raise ASRError("Timed out waiting for ASR recognition") from error
            except ConnectionClosed as error:
                raise ASRError("ASR connection closed before the final result") from error
            if not isinstance(frame, bytes):
                raise ASRProtocolError("ASR service returned a text WebSocket frame")
            response = _decode_response(frame)
            result = response.payload.get("result", {})
            if isinstance(result, list):
                parts = [item for item in result if isinstance(item, dict)]
                texts = [item.get("text", "") for item in parts]
                if any(not isinstance(text, str) for text in texts):
                    raise ASRProtocolError("ASR recognition text must be a string")
                text = "".join(texts)
                latest_result = parts[-1] if parts else {}
            elif isinstance(result, dict):
                text = result.get("text", "")
                latest_result = result
            else:
                raise ASRProtocolError("Invalid ASR recognition result")
            if not isinstance(text, str):
                raise ASRProtocolError("ASR recognition text must be a string")
            text = text or last_text
            # A VAD utterance can become definite without any new characters.
            # This is useful endpoint evidence, independently of stream finality.
            utterances = latest_result.get("utterances")
            definite = None
            if isinstance(utterances, list) and utterances and isinstance(utterances[-1], dict):
                value = utterances[-1].get("definite")
                if isinstance(value, bool):
                    definite = value
            if definite is None:
                # Some final/ack packets omit result metadata.  Preserve prior
                # evidence only while the cumulative transcript is unchanged.
                definite = last_definite if text == last_text else False
            if text != last_text or definite != last_definite or response.is_final:
                yield ASRResult(
                    text=text,
                    is_final=response.is_final,
                    utterance_definite=definite,
                )
            last_text = text
            last_definite = definite
            if response.is_final:
                return


class ASRClient:
    def __init__(self, settings: Any):
        self._settings = settings

    @asynccontextmanager
    async def stream(self) -> AsyncIterator[ASRStream]:
        settings = self._settings
        if settings.asr_sample_rate != 16000:
            raise ValueError("The ASR service requires 16000 Hz audio")
        request_id = str(uuid4())
        headers = {
            "X-Api-Resource-Id": settings.asr_resource_id,
            "X-Api-Connect-Id": request_id,
            "X-Api-Request-Id": request_id,
        }
        api_key = _secret(settings.asr_api_key)
        if api_key:
            headers["X-Api-Key"] = api_key
        else:
            app_key = _secret(settings.asr_app_key)
            access_key = _secret(settings.asr_access_key)
            if not app_key or not access_key:
                raise ASRError("Configure ASR_APP_KEY and ASR_ACCESS_KEY, or ASR_API_KEY")
            headers.update({"X-Api-App-Key": app_key, "X-Api-Access-Key": access_key})
        metadata = {
            "user": {"uid": request_id},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": settings.asr_sample_rate,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "show_utterances": True,
                "result_type": "full",
            },
        }
        timeout = getattr(settings, "asr_timeout", 30.0)
        async with connect(
            settings.asr_url,
            additional_headers=headers,
            open_timeout=timeout,
            close_timeout=5,
            max_size=8 * 1024 * 1024,
            compression=None,
        ) as connection:
            await connection.send(_encode_request(1, json.dumps(metadata).encode(), 1))
            yield ASRStream(connection, timeout=timeout)

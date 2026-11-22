"""Volcengine V3 bidirectional TTS, returning 24 kHz mono signed Int16 PCM.

Wire format follows the official WebSocket V3 documentation and Python demo:
https://www.volcengine.com/docs/6561/1329505
"""

from __future__ import annotations

import asyncio
import gzip
import json
import struct
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from websockets.asyncio.client import connect


class Event(IntEnum):
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
    SENTENCE_START = 350
    SENTENCE_END = 351
    AUDIO = 352


class TTSProtocolError(RuntimeError):
    """Malformed or out-of-order upstream frame."""


class TTSProviderError(RuntimeError):
    """Upstream rejected the connection or synthesis request."""


@dataclass(frozen=True)
class Message:
    message_type: int
    event: int | None
    payload: bytes
    session_id: str = ""
    connection_id: str = ""
    error_code: int | None = None


def encode_request(
    event: Event, payload: dict[str, Any] | None = None, session_id: str = ""
) -> bytes:
    """Encode a JSON client event; connection requests omit the ID field."""
    body = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":")).encode()
    frame = bytearray(b"\x11\x14\x10\x00")
    frame.extend(struct.pack(">i", event))
    if event not in (Event.START_CONNECTION, Event.FINISH_CONNECTION):
        if not session_id:
            raise ValueError("TTS session event requires a session ID")
        identifier = session_id.encode()
        frame.extend(struct.pack(">I", len(identifier)))
        frame.extend(identifier)
    frame.extend(struct.pack(">I", len(body)))
    frame.extend(body)
    return bytes(frame)


def decode_message(data: bytes) -> Message:
    """Decode a bounded server frame, checking all lengths before reading."""
    if not isinstance(data, bytes) or len(data) < 4:
        raise TTSProtocolError("TTS expected a binary frame with a four-byte header")
    version, header_words = data[0] >> 4, data[0] & 15
    if version != 1 or header_words < 1 or len(data) < header_words * 4:
        raise TTSProtocolError("TTS unsupported or truncated protocol header")
    message_type, flags = data[1] >> 4, data[1] & 15
    serialization, compression = data[2] >> 4, data[2] & 15
    if message_type not in (9, 11, 15):
        raise TTSProtocolError("TTS unsupported server message type")
    if serialization not in (0, 1) or compression not in (0, 1):
        raise TTSProtocolError("TTS unsupported serialization or compression")
    offset = header_words * 4

    def take(length: int) -> bytes:
        nonlocal offset
        if length > len(data) - offset:
            raise TTSProtocolError("TTS truncated frame")
        value = data[offset : offset + length]
        offset += length
        return value

    def uint32() -> int:
        return struct.unpack(">I", take(4))[0]

    error_code = uint32() if message_type == 15 else None
    if flags & 3 in (1, 3) and message_type != 15:
        take(4)  # Optional sequence number, before the event.
    event = struct.unpack(">i", take(4))[0] if flags & 4 else None
    session_id = connection_id = ""
    if event is not None:
        try:
            if event in (50, 51, 52):
                connection_id = take(uint32()).decode()
            elif event not in (1, 2):
                session_id = take(uint32()).decode()
        except UnicodeDecodeError as exc:
            raise TTSProtocolError("TTS invalid identifier encoding") from exc
    payload = take(uint32())
    if offset != len(data):
        raise TTSProtocolError("TTS trailing bytes in frame")
    if compression == 1:
        try:
            payload = gzip.decompress(payload)
        except (OSError, EOFError) as exc:
            raise TTSProtocolError("TTS invalid gzip payload") from exc
    return Message(message_type, event, payload, session_id, connection_id, error_code)


def _check_error(message: Message) -> None:
    if message.message_type == 15:
        raise TTSProviderError(f"TTS upstream error (code {message.error_code})")
    if message.event in (Event.CONNECTION_FAILED, Event.SESSION_FAILED):
        code: Any = "unknown"
        try:
            code = json.loads(message.payload).get("status_code", "unknown")
        except (ValueError, AttributeError):
            pass
        # Do not forward upstream payloads; they can contain credentials or input.
        if not isinstance(code, int):
            code = "unknown"
        raise TTSProviderError(f"TTS upstream rejected event {message.event} (code {code})")
    if message.event == Event.SESSION_CANCELED:
        raise TTSProviderError("TTS upstream canceled the session")


class TTSClient:
    """Each synthesis owns its socket so cancellation immediately releases it."""

    def __init__(self, settings: Any):
        self.settings = settings

    async def _receive(self, websocket: Any) -> Message:
        raw = await asyncio.wait_for(websocket.recv(), self.settings.tts_timeout)
        message = decode_message(raw)
        _check_error(message)
        return message

    async def _expect(self, websocket: Any, event: Event, session_id: str = "") -> Message:
        message = await self._receive(websocket)
        if message.message_type != 9 or message.event != event:
            raise TTSProtocolError(f"TTS expected event {int(event)}")
        if session_id and message.session_id != session_id:
            raise TTSProtocolError("TTS response session ID does not match")
        return message

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        settings = self.settings
        if settings.tts_format != "pcm" or settings.tts_sample_rate != 24000:
            raise ValueError("TTS output must be 24000 Hz mono Int16 PCM")
        api_key = settings.tts_api_key
        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()
        if not api_key:
            raise TTSProviderError("TTS_API_KEY is required")
        headers = {
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": settings.tts_resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        session_id = str(uuid.uuid4())
        request = {
            "user": {"uid": session_id},
            "namespace": "BidirectionalTTS",
            "req_params": {
                "speaker": settings.tts_speaker,
                "audio_params": {
                    "format": "pcm",
                    "sample_rate": 24000,
                    "speech_rate": settings.tts_speech_rate,
                    "loudness_rate": settings.tts_loudness_rate,
                },
            },
        }
        async with connect(
            settings.tts_url,
            additional_headers=headers,
            open_timeout=settings.tts_timeout,
            close_timeout=3,
            max_size=4 * 1024 * 1024,
            ping_interval=20,
            proxy=None,
        ) as websocket:
            await websocket.send(encode_request(Event.START_CONNECTION))
            await self._expect(websocket, Event.CONNECTION_STARTED)
            await websocket.send(
                encode_request(
                    Event.START_SESSION,
                    {**request, "event": int(Event.START_SESSION)},
                    session_id,
                )
            )
            await self._expect(websocket, Event.SESSION_STARTED, session_id)
            await websocket.send(
                encode_request(
                    Event.TASK_REQUEST,
                    {
                        **request,
                        "event": int(Event.TASK_REQUEST),
                        "req_params": {**request["req_params"], "text": text},
                    },
                    session_id,
                )
            )
            await websocket.send(encode_request(Event.FINISH_SESSION, session_id=session_id))
            received_audio = False
            tail = b""
            # Deadline also bounds streams that keep sending metadata without audio.
            async with asyncio.timeout(settings.tts_timeout):
                while True:
                    message = await self._receive(websocket)
                    if message.session_id != session_id:
                        raise TTSProtocolError("TTS response session ID does not match")
                    if message.event == Event.SESSION_FINISHED:
                        break
                    if message.message_type == 11 and message.event == Event.AUDIO:
                        audio = tail + message.payload
                        aligned_size = len(audio) - len(audio) % 2
                        tail = audio[aligned_size:]
                        if aligned_size:
                            received_audio = True
                            yield audio[:aligned_size]
                    elif message.message_type == 11:
                        raise TTSProtocolError("TTS unexpected audio event")
            if tail:
                raise TTSProtocolError("TTS returned a partial Int16 sample")
            if not received_audio:
                raise TTSProviderError("TTS completed without audio")
            await websocket.send(encode_request(Event.FINISH_CONNECTION))
            await self._expect(websocket, Event.CONNECTION_FINISHED)

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI
from app import main, speech
from app.asr import AsrStreamResult
from test_speech import FakeTts

# The historical pipeline remains testable, but is not a public application route.
legacy_app = FastAPI()
legacy_app.add_api_websocket_route("/ws/voice/legacy", main.voice_socket)


class FakeVad:
    def __init__(self, **kwargs):
        pass

    def reset(self):
        pass

    def process(self, pcm):
        return {b"start": ["speech_start"], b"end": ["speech_end"]}.get(pcm, [])


class FakeAsr:
    def __init__(self, context_data=None):
        self.context_data = context_data or []
        self.is_open = True
        self.done = asyncio.Event()

    async def _connect(self):
        pass

    async def feed(self, pcm):
        pass

    async def finalize(self):
        self.done.set()

    async def results(self):
        yield AsrStreamResult(text="我要投保", is_final=False)
        await self.done.wait()
        yield AsrStreamResult(text="我要投保吗？先不要办理。", is_final=True)

    async def close(self):
        self.is_open = False


class VoiceSocketTests(unittest.TestCase):
    def test_final_transcript_used_before_agent_and_streamed_audio(self):
        agent_inputs = []

        async def agent(text, thread):
            agent_inputs.append(text)
            yield {"type": "agent_delta", "text": "明白。"}
            yield {"type": "sentence", "text": "明白。"}

        patches = [
            patch.object(main, "init_agent", AsyncMock()),
            patch.object(main, "warm_fillers", AsyncMock()),
            patch.object(main, "SileroVad", FakeVad),
            patch.object(main, "StreamingAsrClient", FakeAsr),
            patch.object(main, "run_agent_stream", agent),
            patch.object(main.settings, "proactive_opening", False),
            patch.object(speech, "TtsStream", FakeTts),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        with TestClient(legacy_app) as client:
            with client.websocket_connect("/ws/voice/legacy") as ws:
                ws.send_bytes(b"start")
                ws.send_bytes(b"end")
                audio = []
                final = None
                while True:
                    event = ws.receive()
                    if event.get("bytes"):
                        audio.append(event["bytes"])
                    elif event.get("text"):
                        import json
                        payload = json.loads(event["text"])
                        if payload["type"] == "asr_final":
                            final = payload["text"]
                        if payload["type"] == "done":
                            break
                self.assertEqual(final, "我要投保吗？先不要办理。")
                self.assertEqual(agent_inputs, [final])
                self.assertEqual(audio, ["明白。".encode()])
                ws.send_text("playback_done")

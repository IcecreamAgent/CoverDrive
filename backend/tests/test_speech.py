import asyncio
import time
import unittest
from unittest.mock import patch

from app import speech


class FakeTts:
    instances = []

    def __init__(self):
        self.received = []
        self.audio = asyncio.Queue()
        self.closed = False
        self.session_count = 0
        self.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def begin_session(self):
        self.session_count += 1

    async def feed(self, text, is_last):
        self.received.append((text, is_last))
        await self.audio.put(None if is_last else text.encode())

    async def chunks(self):
        while (chunk := await self.audio.get()) is not None:
            yield chunk


class FailingReceiveTts(FakeTts):
    async def chunks(self):
        raise RuntimeError("TTS receive failed")
        yield  # pragma: no cover - keep this an async generator


class SpeechTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeTts.instances.clear()
        self.patches = [
            patch.object(speech, "TtsStream", FakeTts),
            patch.object(speech, "filler_audio", {speech.TOOL_PROGRESS_TEXT["calc_premium"]: b"filler"}),
            patch.object(speech.settings, "latency_filler_delay_ms", 20),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.queue = asyncio.Queue()
        self.sent = []
        self.timings = {"t0": time.perf_counter()}
        self.first_audio = asyncio.Event()
        self.progress = speech.ToolProgress()
        self.response_started = asyncio.Event()

    def tool(self, kind, name="calc_premium", run_id="one"):
        self.progress.update({"type": kind, "name": name, "run_id": run_id})

    async def send(self, data):
        self.sent.append(data)
        self.first_audio.set()

    def start(self, opening=False):
        return asyncio.create_task(speech.stream_speech(
            self.queue, self.send, self.timings, asyncio.Event(),
            progress=None if opening else self.progress, response_started=self.response_started,
        ))

    async def test_audio_before_generation_finishes_and_no_filler_after_body(self):
        self.tool("tool_start")
        task = self.start()
        await self.queue.put("第一句")
        await asyncio.wait_for(self.first_audio.wait(), 1)
        self.assertFalse(task.done())
        await asyncio.sleep(.04)
        await self.queue.put("第二句")
        await self.queue.put(None)
        await task
        self.assertEqual(self.sent, ["第一句".encode(), "第二句".encode()])
        self.assertEqual(len(FakeTts.instances), 1)
        self.assertTrue(FakeTts.instances[0].closed)

    async def test_tool_boundary_finishes_session_and_second_segment_restarts_it(self):
        task = self.start()
        await self.queue.put("调用前回复。")
        await self.queue.put(speech.SEGMENT_END)
        await asyncio.wait_for(self.first_audio.wait(), 1)
        await asyncio.sleep(.03)  # Simulate MCP/LLM gap with no active TTS session.
        self.assertFalse(task.done())
        await self.queue.put("调用后结果。")
        await self.queue.put(speech.SEGMENT_END)
        await self.queue.put(None)
        await task
        self.assertEqual(self.sent, ["调用前回复。".encode(), "调用后结果。".encode()])
        self.assertEqual(len(FakeTts.instances), 2)
        self.assertEqual([instance.session_count for instance in FakeTts.instances], [1, 1])
        self.assertTrue(all(instance.closed for instance in FakeTts.instances))

    async def test_slow_response_gets_one_cached_filler(self):
        self.tool("tool_start")
        task = self.start()
        await asyncio.wait_for(self.first_audio.wait(), 1)
        await asyncio.sleep(.04)
        await self.queue.put("正文")
        await self.queue.put(None)
        await task
        self.assertEqual(self.sent, [b"filler", "正文".encode()])
        self.assertLess(self.timings["t_filler_audio"], self.timings["t_body_audio"])

    async def test_slow_normal_response_waits_silently_for_body(self):
        task = self.start()
        await asyncio.sleep(.04)
        self.assertEqual(self.sent, [])
        self.assertEqual(FakeTts.instances, [])
        await self.queue.put("正文")
        await self.queue.put(None)
        await task
        self.assertEqual(self.sent, ["正文".encode()])

    async def test_barge_in_cancels_children_and_closes_connection(self):
        task = self.start()
        await self.queue.put("正文")
        await asyncio.wait_for(self.first_audio.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await asyncio.sleep(.03)
        self.assertEqual(self.sent, ["正文".encode()])
        self.assertTrue(FakeTts.instances[0].closed)
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        self.assertEqual(pending, [])

    async def test_receive_failure_propagates_immediately_and_closes_connection(self):
        with patch.object(speech, "TtsStream", FailingReceiveTts):
            task = self.start()
            await self.queue.put("正文")
            with self.assertRaisesRegex(RuntimeError, "TTS receive failed"):
                await asyncio.wait_for(task, .1)

        self.assertTrue(FailingReceiveTts.instances[-1].closed)
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        self.assertEqual(pending, [])

    async def test_waiting_for_first_sentence_times_out_without_opening_tts(self):
        with patch.object(speech.settings, "voice_response_timeout_s", .02):
            task = self.start()
            with self.assertRaises(TimeoutError):
                await task

        self.assertEqual(FakeTts.instances, [])
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        self.assertEqual(pending, [])

    async def test_opening_does_not_get_filler(self):
        self.tool("tool_start")
        task = self.start(opening=True)
        await asyncio.sleep(.04)
        await self.queue.put(None)
        await task
        self.assertEqual(self.sent, [])

    async def assert_silent_until_done(self):
        task = self.start()
        await asyncio.sleep(.04)
        await self.queue.put(None)
        await task
        self.assertEqual(self.sent, [])

    async def test_slow_normal_answer_stays_silent(self):
        await self.assert_silent_until_done()

    async def test_completed_tool_stays_silent(self):
        self.tool("tool_start")
        task = self.start()
        await asyncio.sleep(.005)
        self.tool("tool_end")
        await asyncio.sleep(.04)
        await self.queue.put(None)
        await task
        self.assertEqual(self.sent, [])

    async def test_unknown_and_silent_tools_stay_silent(self):
        self.tool("tool_start", "get_expiring_customers")
        self.tool("tool_start", "unknown")
        await self.assert_silent_until_done()

    async def test_text_ready_suppresses_notice_even_before_audio(self):
        self.tool("tool_start")
        self.response_started.set()
        with patch.object(speech, "filler_audio", {
            speech.TOOL_PROGRESS_TEXT["calc_premium"]: b"filler",
        }):
            await self.assert_silent_until_done()

    async def test_parallel_calls_track_each_invocation(self):
        self.tool("tool_start", run_id="one")
        self.tool("tool_start", run_id="two")
        self.tool("tool_end", run_id="one")
        self.assertEqual(set(self.progress.pending), {"two"})
        task = self.start()
        await asyncio.wait_for(self.first_audio.wait(), 1)
        self.tool("tool_end", run_id="two")
        await self.queue.put(None)
        await task
        self.assertEqual(self.sent, [b"filler"])


if __name__ == "__main__":
    unittest.main()

"""Low-latency speech with cached progress notices and segmented TTS sessions."""
import asyncio
import logging
import time

from .config import settings
from .tts import TtsStream

logger = logging.getLogger(__name__)
TOOL_PROGRESS_TEXT = {
    "calc_premium": "我正在核算报价，请稍等一下。",
    "create_policy": "我正在处理您的保单，请稍等一下。",
}
# These are fixed public tool-progress phrases, never customer data.
FILLERS = tuple(TOOL_PROGRESS_TEXT.values())
filler_audio: dict[str, bytes] = {}
SEGMENT_END = object()


class ToolProgress:
    """Track each real tool invocation; completed and unknown tools stay silent."""

    def __init__(self):
        self.pending = {}
        self.changed = asyncio.Event()

    def update(self, event):
        key = event.get("run_id") or event["name"]
        if event["type"] == "tool_start":
            if event["name"] in TOOL_PROGRESS_TEXT:
                self.pending[key] = (time.perf_counter(), TOOL_PROGRESS_TEXT[event["name"]])
        else:
            self.pending.pop(key, None)
        self.changed.set()

    def oldest(self):
        return min(self.pending.values(), default=None)


async def warm_fillers():
    if not settings.latency_filler_enabled:
        return
    # Cache only public, fixed phrases; never cache customer audio/text.
    for text in FILLERS:
        try:
            async with asyncio.timeout(12):
                async with TtsStream() as stream:
                    await stream.begin_session()
                    await stream.feed(text, is_last=True)
                    chunks = [chunk async for chunk in stream.chunks()]
                    if chunks:
                        filler_audio[text] = b"".join(chunks)
                        logger.info("承接语预热成功: %s (%d bytes)", text, len(filler_audio[text]))
        except Exception as exc:
            logger.warning("承接语预热失败（不阻塞正文）: %r", exc)


async def stream_speech(queue, send_audio, timings, finished, *, progress=None, response_started=None):
    """Synthesize each conversational segment in its own session.

    Wait for the first sentence before opening TTS so a slow LLM cannot leave an
    idle WebSocket to be closed by the provider. A tool boundary closes the
    connection; the post-tool segment gets a fresh one.
    Cached progress audio and body audio share a lock to preserve ordering.
    """
    audio_lock = asyncio.Lock()
    body_started = asyncio.Event()

    async def emit(chunk, *, notice=None):
        async with audio_lock:
            if notice and (
                body_started.is_set() or finished.is_set()
                or (response_started is not None and response_started.is_set())
            ):
                return
            if notice == "tool" and not progress.pending:
                return
            now = time.perf_counter()
            timings.setdefault("t_first_audio", now)
            if notice == "tool":
                timings["t_filler_audio"] = now
            else:
                body_started.set()
                timings.setdefault("t_body_audio", now)
            await send_audio(chunk)

    async def filler():
        if progress is None or not settings.latency_filler_enabled:
            return

        # Normal model latency stays silent. Only known long-running business
        # tools may emit a pre-synthesized progress notice.
        while not body_started.is_set() and not finished.is_set():
            progress.changed.clear()
            oldest = progress.oldest()
            if oldest is None:
                await progress.changed.wait()
                continue
            started, text = oldest
            delay = started + settings.latency_filler_delay_ms / 1000 - time.perf_counter()
            if delay > 0:
                try:
                    await asyncio.wait_for(progress.changed.wait(), timeout=delay)
                    continue
                except TimeoutError:
                    continue  # Recheck tool completion before speaking.
            pcm = filler_audio.get(text)
            if pcm:
                await emit(pcm, notice="tool")
            return  # At most one progress notice per user turn.

    async def run_segment(stream, first_sentence):
        """Return True when the whole pipeline ends, False at a segment boundary."""
        async def feed_segment():
            item = first_sentence
            while True:
                if item is SEGMENT_END:
                    await stream.feed("", is_last=True)
                    return False
                if item is None:
                    await stream.feed("", is_last=True)
                    return True
                await stream.feed(item, is_last=False)
                item = await queue.get()

        async def receive_segment():
            async for chunk in stream.chunks():
                await emit(chunk)

        feed_task = asyncio.create_task(feed_segment())
        receive_task = asyncio.create_task(receive_segment())
        try:
            done, pending = await asyncio.wait(
                {feed_task, receive_task}, return_when=asyncio.FIRST_EXCEPTION
            )
            for task in done:
                if task.cancelled():
                    raise asyncio.CancelledError
                error = task.exception()
                if error is not None:
                    for other in pending:
                        other.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    raise error
            # FIRST_EXCEPTION waits for both tasks when neither raises. The
            # receiver therefore fully drains audio after feed sends is_last.
            return feed_task.result()
        finally:
            for task in (feed_task, receive_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(feed_task, receive_task, return_exceptions=True)

    filler_task = asyncio.create_task(filler())
    try:
        async with asyncio.timeout(settings.voice_response_timeout_s):
            while True:
                first = await queue.get()
                if first is None:
                    break
                if first is SEGMENT_END:
                    continue
                async with TtsStream() as stream:
                    await stream.begin_session()
                    if await run_segment(stream, first):
                        break
    finally:
        filler_task.cancel()
        await asyncio.gather(filler_task, return_exceptions=True)

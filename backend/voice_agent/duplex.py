"""Continuous microphone fan-out, semantic endpointing, and reversible barge-in.

One PCM stream feeds WebRTC VAD/energy, streaming ASR, and bounded semantic
classification. No classifier is allowed to stop an actively speaking user.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from voice_agent.endpointing import AcousticFrame, EndpointDetector
from voice_agent.semantics import SemanticAnalyzer, SemanticResult, local_semantics

if TYPE_CHECKING:
    from voice_agent.session import VoiceSession


@dataclass(eq=False)
class Utterance:
    turn_id: str
    candidate: bool
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=150))
    task: asyncio.Task | None = None
    semantic_task: asyncio.Task | None = None
    latest: str = ""
    semantic: SemanticResult | None = None
    revision: int = 0
    semantic_calls: int = 0
    awaiting_new_text: bool = False
    announced: bool = False
    finalizing: bool = False
    superseded: bool = False
    canceled: bool = False
    reason: str = ""
    prefix_source: Utterance | None = None
    result: asyncio.Future | None = None


def normalized(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).lower()


class DuplexInput:
    def __init__(self, session: VoiceSession):
        self.session = session
        self.settings = session.settings
        self.detector = EndpointDetector(self.settings)
        self.semantic = SemanticAnalyzer(self.settings)
        self.enabled = False
        self.playback_active = False
        self.current: Utterance | None = None
        self.pending: Utterance | None = None
        self.utterances: set[Utterance] = set()
        self.tasks: set[asyncio.Task] = set()

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)

        def completed(done: asyncio.Task) -> None:
            self.tasks.discard(done)
            if not done.cancelled() and (exc := done.exception()) is not None:
                self.session.log_failure("continuous", exc)

        task.add_done_callback(completed)
        return task

    @property
    def assistant_speaking(self) -> bool:
        task = self.session.response_task
        return self.playback_active or (task is not None and not task.done())

    async def enable(self) -> None:
        if self.enabled:
            return
        if missing := self.settings.missing():
            await self.session.send_global(
                "error", stage="config", message="请在 .env 配置：" + ", ".join(missing)
            )
            return
        await self.session.cancel_recording()
        self.detector.reset()
        self.enabled = True
        await self.session.send_global("listening", enabled=True, mode="automatic")

    async def disable(self, *, notify: bool = True) -> None:
        self.enabled = False
        await self.abort_input()
        if notify:
            await self.session.send_global("listening", enabled=False)

    async def abort_input(self, *, notify: bool = True) -> None:
        """Release recognition candidates, preserving an existing assistant reply."""
        from voice_agent.session import cancel_task

        candidate = any(item.candidate and not item.announced for item in self.utterances)
        for task in list(self.tasks):
            await cancel_task(task)
        self.tasks.clear()
        self.utterances.clear()
        self.current = self.pending = None
        self.detector.reset()
        if candidate and notify:
            await self.session.send_global("barge_in_rejected", reason="input_closed")

    async def close(self) -> None:
        self.enabled = False
        await self.abort_input(notify=False)

    async def feed(self, pcm: bytes) -> None:
        if not self.enabled:
            return
        if len(pcm) % 2 or len(pcm) > 64_000:
            await self.disable()
            await self.session.send_global(
                "error", stage="asr", message="音频必须是单声道 Int16 PCM，单帧最大 64 KB。"
            )
            return
        # Analyze each transport frame in order, including when a client batches audio.
        for offset in range(0, len(pcm), self.detector.frame_bytes):
            for frame in self.detector.feed(pcm[offset : offset + self.detector.frame_bytes]):
                if frame.started and self.current is None:
                    await self._begin(frame)
                elif self.current is not None and not self.current.finalizing:
                    item = self.current
                    if frame.resumed:
                        item.revision += 1
                        item.semantic = None
                        item.awaiting_new_text = True
                        if item.semantic_task:
                            item.semantic_task.cancel()
                    try:
                        item.queue.put_nowait(frame.pcm)
                    except asyncio.QueueFull:
                        await self.abort_input()
                        await self.session.send_global(
                            "error", stage="asr", message="语音网络阻塞，请稍后重试。"
                        )
                        return
                if self.current is not None:
                    decision = self.detector.decision()
                    if decision.end:
                        await self._end(self.current, decision.reason)

    async def _begin(self, frame: AcousticFrame) -> None:
        item = Utterance(str(uuid4()), candidate=self.assistant_speaking)
        item.result = asyncio.get_running_loop().create_future()
        if self.pending is not None and not self.pending.canceled:
            # Speech resumed while final ASR was in flight: combine it before answering.
            self.pending.superseded = True
            item.prefix_source = self.pending
        self.current = item
        self.utterances.add(item)
        if item.candidate:
            await self.session.send_global("barge_in_candidate")
        else:
            await self._announce(item)
        for offset in range(0, len(frame.pre_roll), self.detector.frame_bytes):
            item.queue.put_nowait(frame.pre_roll[offset : offset + self.detector.frame_bytes])
        item.task = self._spawn(self._recognize(item))

    async def _announce(self, item: Utterance) -> None:
        if item.announced or item.canceled:
            return
        if item.candidate:
            await self.session.cancel_response()
            agent = getattr(self.session, "agent", None)
            if agent is not None:
                agent.note_interruption()
            self.playback_active = False
            await self.session.send_global("barge_in_confirmed")
        item.announced = True
        self.session.turn_id = item.turn_id
        await self.session.send_for(
            item.turn_id,
            "speech_start",
            barge_in=item.candidate,
            continuation_of=item.prefix_source.turn_id if item.prefix_source else None,
        )
        await self.session.send_for(item.turn_id, "state", stage="asr")

    def _is_echo(self, text: str) -> bool:
        clean = normalized(text)
        spoken = normalized(self.session.answer_text)
        return len(clean) >= 8 and clean in spoken

    def _could_be_echo(self, text: str) -> bool:
        clean = normalized(text)
        return len(clean) >= 2 and clean in normalized(self.session.answer_text)

    async def _transcript(self, item: Utterance, text: str, *, definite: bool) -> None:
        if item.canceled or item.superseded:
            return
        changed = text != item.latest
        item.latest = text
        if changed:
            item.revision += 1
            item.awaiting_new_text = False
        local = local_semantics(text, assistant_speaking=item.candidate)
        if (changed or item.semantic is None) and not item.awaiting_new_text:
            item.semantic = local
        if self.current is item:
            self.detector.update_transcript(text, definite=definite)
            if not item.awaiting_new_text and item.semantic is not None:
                evidence = item.semantic
                kind = evidence.kind if evidence.confidence >= 0.75 else "unknown"
                self.detector.set_semantic(kind, text=text)
        meaningful = bool(normalized(text)) and (
            local.kind == "cancel" or len(normalized(text)) >= 2
        )
        if item.candidate and not item.announced and meaningful:
            if local.kind != "backchannel" and not self._could_be_echo(text):
                await self._announce(item)
        if item.announced:
            await self.session.send_for(item.turn_id, "transcript", text=text, is_final=False)
        if changed and local.kind == "complete" and self.settings.semantic_enabled:
            if item.semantic_task:
                item.semantic_task.cancel()
            if item.semantic_calls < self.settings.semantic_max_calls:
                item.semantic_task = self._spawn(self._classify(item, text, item.revision))

    async def _classify(self, item: Utterance, text: str, revision: int) -> None:
        await asyncio.sleep(0.18)  # Coalesce frequent partial revisions without blocking audio.
        if item.canceled or item.superseded or revision != item.revision:
            return
        item.semantic_calls += 1
        result = await self.semantic.analyze(text, assistant_speaking=item.candidate)
        if item.canceled or item.superseded or revision != item.revision or item.latest != text:
            return
        item.semantic = result
        if self.current is item:
            kind = result.kind if result.confidence >= 0.75 else "unknown"
            self.detector.set_semantic(kind, text=text)

    async def _end(self, item: Utterance, reason: str) -> None:
        if item.finalizing:
            return
        item.finalizing = True
        item.reason = reason
        if item.announced:
            await self.session.send_for(
                item.turn_id,
                "speech_end",
                reason=reason,
                signals={
                    "silence_ms": self.detector.silence_ms,
                    "speech_ms": self.detector.speech_ms,
                    "asr_definite": self.detector.definite,
                    "semantic": self.detector.semantic,
                },
            )
        try:
            item.queue.put_nowait(None)
        except asyncio.QueueFull:
            await self.abort_input()
            await self.session.send_global(
                "error", stage="asr", message="语音网络阻塞，请稍后重试。"
            )
            return
        self.current = None
        self.pending = item
        self.detector.reset()

    async def _recognize(self, item: Utterance) -> None:
        from voice_agent.session import cancel_task

        receiver = sender = None
        final = ""
        try:
            async with asyncio.timeout(
                self.settings.max_recording_seconds + self.settings.asr_timeout
            ):
                async with self.session.asr.stream() as stream:

                    async def collect() -> str:
                        latest = ""
                        async for result in stream.receive():
                            latest = result.text or latest
                            await self._transcript(
                                item, latest, definite=getattr(result, "utterance_definite", False)
                            )
                            if result.is_final:
                                return latest
                        raise RuntimeError("ASR ended before a final transcript")

                    async def transmit() -> None:
                        while True:
                            audio = await item.queue.get()
                            if audio is None:
                                await stream.send_audio(b"", final=True)
                                break
                            await stream.send_audio(audio)

                    receiver = asyncio.create_task(collect())
                    sender = asyncio.create_task(transmit())
                    done, _ = await asyncio.wait(
                        [receiver, sender], return_when=asyncio.FIRST_COMPLETED
                    )
                    if receiver in done:
                        final = receiver.result()
                        if self.current is item:
                            await self._end(item, "asr_final")
                    else:
                        sender.result()
                        final = await asyncio.wait_for(receiver, self.settings.asr_timeout)
            if item.prefix_source and item.prefix_source.result:
                prefix = await asyncio.wait_for(
                    asyncio.shield(item.prefix_source.result), self.settings.asr_timeout
                )
                if prefix and not final.startswith(prefix):
                    final = prefix.rstrip("。！？!?.") + "，" + final
            if item.result and not item.result.done():
                item.result.set_result(final)
            if not item.superseded and not item.canceled:
                await self._complete(item, final.strip())
        except asyncio.CancelledError:
            item.canceled = True
            raise
        except Exception as exc:
            if item.candidate and not item.announced:
                await self.session.send_global("barge_in_rejected", reason="recognition_failed")
            if not item.superseded:
                await self.session.send_global(
                    "error", stage="asr", message="语音识别暂时不可用，请检查网络后重试。"
                )
                # Only a safe exception type is logged by the session helper.
                self.session.log_failure("asr", exc)
        finally:
            await cancel_task(sender)
            await cancel_task(receiver)
            await cancel_task(item.semantic_task)
            if item.result and not item.result.done():
                item.result.set_result("")
            if self.current is item:
                self.current = None
                self.detector.reset()
            if self.pending is item:
                self.pending = None
            self.utterances.discard(item)

    async def _complete(self, item: Utterance, text: str) -> None:
        local = local_semantics(text, assistant_speaking=item.candidate)
        if item.semantic is not None and item.semantic.confidence >= 0.75:
            local = item.semantic
        rejected = not text or item.reason == "too_short"
        if item.candidate and not item.announced:
            rejected = rejected or local.kind == "backchannel" or self._is_echo(text)
            if rejected:
                await self.session.send_global("barge_in_rejected", reason="backchannel_or_noise")
                return
        if rejected:
            if item.announced:
                await self.session.send_for(item.turn_id, "state", stage="idle")
            return
        await self._announce(item)
        await self.session.send_for(item.turn_id, "transcript", text=text, is_final=True)
        if local.kind == "cancel":
            await self.session.cancel_response()
            self.playback_active = False
            await self.session.send_for(item.turn_id, "state", stage="idle")
            return
        if len(text) > self.settings.max_text_length:
            await self.session.send_for(
                item.turn_id, "error", stage="asr", message="语音内容过长，请分段提问。"
            )
            return
        # A new utterance may have superseded this one while announcing/canceling.
        if not item.superseded and not item.canceled and self.enabled:
            await self.session.cancel_response()
            self.session.turn_id = item.turn_id
            self.session.response_task = asyncio.create_task(self.session.respond(text))

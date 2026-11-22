"""WebSocket lifecycle, recording, interruption, and ordered ASR → agent → TTS flow."""

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect

from voice_agent.agent import AgentSession
from voice_agent.config import Settings
from voice_agent.duplex import DuplexInput
from voice_agent.execution_trace import ExecutionTrace
from voice_agent.providers.asr import ASRClient
from voice_agent.providers.tts import TTSClient
from voice_agent.routing import REASON_LABELS, RouteDecision, acknowledgement_for
from voice_agent.sentences import SentenceBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _WaitFeedback:
    text: str
    superseded: asyncio.Event


async def cancel_task(task: asyncio.Task | None) -> None:
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


class VoiceSession:
    def __init__(self, websocket: WebSocket, settings: Settings, *, speech_clients: bool = True):
        self.ws = websocket
        self.settings = settings
        self.agent: AgentSession | None = None
        self.asr = ASRClient(settings) if speech_clients else None
        self.tts = TTSClient(settings) if speech_clients else None
        self.send_lock = asyncio.Lock()
        self.recording_task: asyncio.Task | None = None
        self.response_task: asyncio.Task | None = None
        self.audio_queue: asyncio.Queue | None = None
        self.accepting_audio = False
        self.recorded_bytes = 0
        self.turn_id: str | None = None
        self.answer_text = ""
        self.duplex = DuplexInput(self)
        self._last_feedback_at: float | None = None
        self.execution_trace = ExecutionTrace(settings)

    async def _send_message(self, event_type: str, data: dict, *, trace_event=True) -> None:
        # Control messages and their companion traces share the same public
        # role labels; neither should expose deployment-specific model names.
        if "model" in data:
            stage = "deep" if data.get("route") == "deep" else "fast"
            data = {**data, "model": self.execution_trace.model_label(stage)}
        async with self.send_lock:
            event = self.execution_trace.observe(
                event_type, {"turn_id": self.turn_id, **data}
            ) if trace_event else None
            if event is not None:
                await self.ws.send_json({"type": "execution_event", "event": event})
            await self.ws.send_json({"type": event_type, **data})

    async def trace(self, turn_id, stage, name, label, status="info", detail="", **metadata):
        async with self.send_lock:
            event = self.execution_trace.event(
                turn_id, stage, name, label, status, detail, **metadata
            )
            if event is not None:
                await self.ws.send_json({"type": "execution_event", "event": event})

    async def trace_handoff(self, turn_id: str, **metadata) -> None:
        async with self.send_lock:
            event = self.execution_trace.handoff(turn_id, **metadata)
            await self.ws.send_json({"type": "execution_event", "event": event})

    async def send(self, event_type: str, *, trace_event=True, **data) -> None:
        if self.turn_id is not None:
            data["turn_id"] = self.turn_id
        await self._send_message(event_type, data, trace_event=trace_event)

    async def send_global(self, event_type: str, **data) -> None:
        await self._send_message(event_type, data)

    async def send_for(self, turn_id: str, event_type: str, *, trace_event=True, **data) -> None:
        await self._send_message(
            event_type, {"turn_id": turn_id, **data}, trace_event=trace_event
        )

    def log_failure(self, stage: str, exc: Exception) -> None:
        logger.warning("%s failed (%s)", stage, type(exc).__name__)

    async def cancel_recording(self) -> None:
        self.accepting_audio = False
        await cancel_task(self.recording_task)
        self.recording_task = None
        self.audio_queue = None

    async def cancel_response(self) -> None:
        await cancel_task(self.response_task)
        self.response_task = None

    async def error(self, stage: str, message: str, exc: Exception | None = None) -> None:
        # Upstream exception bodies can contain auth headers; log only exception type.
        if exc is not None:
            logger.warning("%s failed (%s)", stage, type(exc).__name__)
            if stage == "agent" and getattr(exc, "code", None) == "InvalidEndpoint.ClosedEndpoint":
                message = "LLM 模型端点已关闭，请在 Ark 开通该模型或更新 LLM_MODEL_ID 后重启。"
        await self.send("error", stage=stage, message=message)

    async def interrupt(self, *, notify: bool = True) -> None:
        await self.duplex.abort_input(notify=notify)
        self.duplex.playback_active = False
        await self.cancel_recording()
        await self.cancel_response()
        if notify:
            await self.send("interrupted")
            await self.send("state", stage="idle")

    async def start_recording(self, turn_id: str | None = None) -> None:
        await self.interrupt(notify=False)
        self.turn_id = turn_id
        missing = self.settings.missing()
        if missing:
            await self.error("config", "请在 .env 配置：" + ", ".join(missing))
            return
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.recorded_bytes = 0
        self.accepting_audio = True
        await self.send("state", stage="asr")
        self.recording_task = asyncio.create_task(self.recognize(self.audio_queue))

    async def recognize(self, queue: asyncio.Queue) -> None:
        text = ""
        try:
            async with asyncio.timeout(
                self.settings.max_recording_seconds + self.settings.asr_timeout
            ):
                async with self.asr.stream() as stream:

                    async def collect() -> str:
                        latest = ""
                        async for result in stream.receive():
                            latest = result.text or latest
                            await self.send("transcript", text=latest, is_final=result.is_final)
                            if result.is_final:
                                return latest
                        raise RuntimeError("ASR ended without a final result")

                    async def transmit() -> None:
                        while True:
                            audio = await queue.get()
                            if audio is None:
                                await stream.send_audio(b"", final=True)
                                break
                            await stream.send_audio(audio)

                    receiver = asyncio.create_task(collect())
                    sender = asyncio.create_task(transmit())
                    try:
                        done, _ = await asyncio.wait(
                            [receiver, sender], return_when=asyncio.FIRST_COMPLETED
                        )
                        # Observe upstream failure immediately, even while the microphone is open.
                        if receiver in done:
                            text = receiver.result()
                        else:
                            sender.result()
                            text = await asyncio.wait_for(receiver, self.settings.asr_timeout)
                    finally:
                        await cancel_task(sender)
                        await cancel_task(receiver)
            if not text.strip():
                await self.error("asr", "没有识别到语音，请靠近麦克风后重试。")
                await self.send("state", stage="idle")
            elif len(text) > self.settings.max_text_length:
                await self.error("asr", "识别内容超过长度限制，请分段提问。")
                await self.send("state", stage="idle")
            else:
                self.response_task = asyncio.create_task(self.respond(text.strip()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.error("asr", "语音识别失败或超时，请检查 ASR 凭证、资源授权及网络。", exc)
            await self.send("state", stage="idle")
        finally:
            self.accepting_audio = False
            self.audio_queue = None

    def _feedback_available(self) -> bool:
        return (
            self._last_feedback_at is None
            or time.monotonic() - self._last_feedback_at
            >= self.settings.slow_feedback_cooldown_ms / 1000
        )

    async def _delayed_feedback(
        self, queue: asyncio.Queue, text: str, superseded: asyncio.Event
    ) -> None:
        await asyncio.sleep(self.settings.slow_feedback_delay_ms / 1000)
        if not superseded.is_set() and self._feedback_available():
            await queue.put(_WaitFeedback(text, superseded))

    async def _speech_audio(self, text: str, feedback: _WaitFeedback | None):
        async with contextlib.aclosing(self.tts.synthesize(text)) as stream:
            if feedback is not None:

                async def first_pcm():
                    async for pcm in stream:
                        if pcm:
                            return pcm
                    return None

                first = asyncio.create_task(first_pcm())
                superseded = asyncio.create_task(feedback.superseded.wait())
                try:
                    await asyncio.wait([first, superseded], return_when=asyncio.FIRST_COMPLETED)
                    if feedback.superseded.is_set():
                        return
                    if pcm := first.result():
                        yield pcm
                finally:
                    await cancel_task(first)
                    await cancel_task(superseded)
            async for pcm in stream:
                yield pcm

    async def speak(self, queue: asyncio.Queue) -> None:
        started = False
        failed = False
        try:
            while True:
                sentence = await queue.get()
                if sentence is None:
                    break
                if failed:
                    continue  # Preserve text generation even if TTS is unavailable.
                feedback = sentence if isinstance(sentence, _WaitFeedback) else None
                if feedback is not None:
                    if feedback.superseded.is_set() or not self._feedback_available():
                        continue
                    sentence = feedback.text
                feedback_sent = False
                try:
                    if not started:
                        await self.send("state", stage="tts")
                        await self.send("audio_start", sample_rate=24000, channels=1, format="pcm")
                        started = True
                    async with contextlib.aclosing(
                        self._speech_audio(sentence, feedback)
                    ) as stream:
                        async for pcm in stream:
                            if pcm:
                                if feedback is not None and not feedback_sent:
                                    # Results may arrive while feedback TTS is still connecting.
                                    if feedback.superseded.is_set():
                                        break
                                    self._last_feedback_at = time.monotonic()
                                    feedback_sent = True
                                async with self.send_lock:
                                    event = self.execution_trace.pcm(self.turn_id)
                                    if event is not None:
                                        await self.ws.send_json({
                                            "type": "execution_event", "event": event,
                                        })
                                    await self.ws.send_bytes(pcm)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    failed = True
                    await self.error(
                        "tts", "语音合成失败，请检查 TTS 凭证和音色授权；文字回复仍会继续。", exc
                    )
        finally:
            if started:
                await self.send("audio_end")

    async def respond(
        self,
        text: str,
        *,
        force_deep: bool = False,
        acknowledgement: str = "",
        route_reason: str = "none",
        decision: RouteDecision | None = None,
        acknowledge: bool = True,
        prepared_answer: str | None = None,
    ) -> str | None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        speaker = asyncio.create_task(self.speak(queue))
        feedback_task: asyncio.Task | None = None
        feedback_superseded = asyncio.Event()
        completed_text = None
        try:
            async with asyncio.timeout(self.settings.turn_timeout):
                if self.agent is None and prepared_answer is None:
                    self.agent = AgentSession(self.settings)
                if (
                    prepared_answer is None
                    and decision is None
                    and hasattr(self.agent, "plan_route")
                ):
                    decision = await self.agent.plan_route(text)
                elif (
                    prepared_answer is None
                    and decision is None
                    and hasattr(self.agent, "route_decision")
                ):
                    decision = self.agent.route_decision(text)
                if decision is not None and decision.escalate:
                    force_deep = True
                    route_reason = decision.reason
                if force_deep:
                    await self.send(
                        "agent_route",
                        route="deep",
                        provider="deepagents",
                        model="任务 LLM",
                        reason=route_reason,
                        source=decision.source if decision else "native",
                        speech_provider="tts",
                    )
                    await self.send(
                        "agent_progress",
                        phase="working",
                        message=REASON_LABELS.get(route_reason, "需要深入处理")
                        + "，正在深入回答，可以随时打断",
                    )
                    if acknowledge:
                        if not acknowledgement:
                            acknowledgement = acknowledgement_for(route_reason)
                        feedback_task = asyncio.create_task(
                            self._delayed_feedback(queue, acknowledgement, feedback_superseded)
                        )
                await self.send("state", stage="agent")
                self.answer_text = ""
                buffer = SentenceBuffer()
                answer = []
                route_reported = force_deep
                options = {"force_deep": True} if force_deep else {}
                if decision is not None and hasattr(self.agent, "plan_route"):
                    options["decision"] = decision

                async def existing_result():
                    yield prepared_answer

                stream = (
                    existing_result()
                    if prepared_answer is not None
                    else self.agent.stream(text, **options)
                )
                async for delta in stream:
                    if not delta:
                        continue
                    if delta.strip():
                        feedback_superseded.set()
                        await cancel_task(feedback_task)
                        feedback_task = None
                    if not route_reported:
                        route = getattr(self.agent, "last_route", None)
                        if route in ("fast", "deep"):
                            await self.send(
                                "agent_route",
                                route=route,
                                provider="mini" if route == "fast" else "deepagents",
                                reason=getattr(self.agent, "last_route_reason", "model_selected"),
                                source=decision.source if decision else "native",
                            )
                        route_reported = True
                    answer.append(delta)
                    self.answer_text += delta
                    await self.send("assistant_delta", text=delta)
                    for sentence in buffer.feed(delta):
                        await queue.put(sentence)
                if tail := buffer.flush():
                    await queue.put(tail)
                if not answer:
                    raise RuntimeError("Agent returned no text")
                if force_deep:
                    await self.send("agent_progress", phase="answering", message="正在播报深入回答")
                await self.send("assistant_done", text="".join(answer))
                await queue.put(None)
                await speaker
                completed_text = "".join(answer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.error(
                "agent", "回复生成失败或超时，请检查 LLM 模型、凭证及网络后重试。", exc
            )
        finally:
            feedback_superseded.set()
            await cancel_task(feedback_task)
            await cancel_task(speaker)
        if force_deep:
            await self.send("agent_progress", phase="completed", message="")
        await self.send("state", stage="idle")
        return completed_text

    async def handle_audio(self, audio: bytes) -> None:
        if self.duplex.enabled:
            await self.duplex.feed(audio)
            return
        if not self.accepting_audio or self.audio_queue is None:
            return
        if len(audio) % 2 or len(audio) > 64_000:
            await self.interrupt()
            await self.error("asr", "音频帧必须是单声道 Int16 PCM，单帧最大 64 KB。")
            return
        self.recorded_bytes += len(audio)
        if self.recorded_bytes > self.settings.max_recording_seconds * 16000 * 2:
            await self.interrupt()
            await self.error("asr", "录音超过时长限制，请分段提问。")
            return
        try:
            self.audio_queue.put_nowait(audio)
        except asyncio.QueueFull:
            await self.interrupt()
            await self.error("asr", "音频发送过快或网络阻塞，请重试。")

    async def run(self) -> None:
        await self.send(
            "ready", input_sample_rate=16000, output_sample_rate=24000, automatic_endpointing=True
        )
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await self.handle_audio(message["bytes"])
                    continue
                raw = message.get("text", "")
                if len(raw) > self.settings.max_text_length * 4 + 1024:
                    await self.error("protocol", "消息过长。")
                    continue
                try:
                    event = json.loads(raw)
                    if not isinstance(event, dict):
                        raise ValueError
                except (ValueError, TypeError):
                    await self.error("protocol", "控制消息必须是 JSON 对象。")
                    continue
                kind = event.get("type")
                turn_id = event.get("turn_id")
                if turn_id is not None and (not isinstance(turn_id, str) or len(turn_id) > 80):
                    await self.error("protocol", "turn_id 必须是长度不超过 80 的字符串。")
                    continue
                if kind == "listen":
                    await self.duplex.enable()
                elif kind == "unlisten":
                    await self.duplex.disable()
                elif kind == "playback":
                    if turn_id == self.turn_id and isinstance(event.get("playing"), bool):
                        self.duplex.playback_active = event["playing"]
                elif kind == "start":
                    if self.duplex.enabled:
                        await self.duplex.disable()
                    await self.start_recording(turn_id)
                elif kind == "stop":
                    if self.accepting_audio and self.audio_queue is not None:
                        self.accepting_audio = False
                        try:
                            self.audio_queue.put_nowait(None)
                        except asyncio.QueueFull:
                            await self.interrupt()
                            await self.error("asr", "音频队列已满，请重试。")
                elif kind == "interrupt":
                    await self.interrupt()
                elif kind == "text":
                    await self.interrupt(notify=False)
                    self.turn_id = turn_id
                    text = event.get("text", "")
                    if not isinstance(text, str) or not text.strip():
                        await self.error("protocol", "请输入有效文字。")
                    elif len(text) > self.settings.max_text_length:
                        await self.error("protocol", "文字超过长度限制。")
                    elif missing := self.settings.missing(voice=False):
                        await self.error("config", "请在 .env 配置：" + ", ".join(missing))
                    else:
                        self.response_task = asyncio.create_task(self.respond(text.strip()))
                elif kind == "ping":
                    await self.send("pong")
                else:
                    await self.error("protocol", "未知控制消息。")
        except WebSocketDisconnect:
            pass
        finally:
            await self.duplex.close()
            await self.interrupt(notify=False)

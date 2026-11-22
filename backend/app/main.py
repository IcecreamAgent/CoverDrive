import asyncio
import base64
import json
import logging
import time
import uuid
import anyio
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .asr import transcribe, StreamingAsrClient, AsrContextHistory
from .agent import run_agent, run_agent_stream, init_agent, OUTBOUND_OPENING
from .tts import synthesize
from .vad import SileroVad, SAMPLE_RATE
from .speech import stream_speech, warm_fillers, ToolProgress, SEGMENT_END
from voice_agent.config import get_settings as get_voice_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Native voice creates no ASR/TTS client and makes no external calls at startup.
    try:
        yield
    finally:
        _text_sessions.clear()


app = FastAPI(title="Voice Agent API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str
    session_id: str | None = None


_text_sessions = {}


def _check_http_auth(request: Request):
    import secrets
    token = get_voice_settings().voice_auth_token
    if token and not secrets.compare_digest(
        request.headers.get("authorization", "").encode(), ("Bearer " + token).encode()
    ):
        raise HTTPException(401, "请先提供访问令牌。")


@app.get("/api/health")
async def health():
    config = get_voice_settings()
    native = config.voice_provider == "seeduplex"
    return {
        "ok": True, "status": "ok", "service": "voice-agent",
        "configured": not config.missing(), "missing": config.missing(),
        "auth_required": bool(config.voice_auth_token),
        "voice_provider": config.voice_provider,
        "voice_output_provider": "seeduplex" if native else "tts",
        "decision_provider": "seeduplex" if native else "cascade",
        "voice_orchestration": "native_tool_only" if native else "cascade",
        "seeduplex_model": "端到端语音" if native else None,
        "llm_model": "任务 LLM",
        "fast_llm_enabled": False, "semantic_router_enabled": False,
        "standalone_tts_enabled": not native,
        "background_tasks_enabled": native and config.background_tasks_enabled,
        "input_sample_rate": 16000, "output_sample_rate": 24000,
        "voice_mode": "automatic", "insurance_writes_enabled": config.insurance_writes_enabled,
    }


@app.post("/api/text")
async def text_chat(request: TextRequest, http: Request):
    _check_http_auth(http)
    config = get_voice_settings()
    if not request.text.strip():
        raise HTTPException(400, "text 不能为空")
    if len(request.text) > config.max_text_length or (request.session_id and len(request.session_id) > 80):
        raise HTTPException(400, "文字或会话标识超过长度限制。")
    if config.missing(voice=False):
        raise HTTPException(503, "请配置：" + ", ".join(config.missing(voice=False)))
    from voice_agent.agent import AgentSession
    key = request.session_id or uuid.uuid4().hex
    if key not in _text_sessions:
        if len(_text_sessions) >= 30:
            expired = next((k for k, (_, lock) in _text_sessions.items() if not lock.locked()), None)
            if expired is None:
                raise HTTPException(503, "会话繁忙，请稍后重试。")
            _text_sessions.pop(expired)
        _text_sessions[key] = (AgentSession(config, task_only=True), asyncio.Lock())
    agent, lock = _text_sessions[key]
    async with lock:
        agent.authorization.observe_final(request.text.strip())
        agent._authorization_version = agent.authorization.version
        agent.graph = None
        try:
            async with asyncio.timeout(config.turn_timeout):
                _ = [part async for part in agent.stream(request.text.strip(), force_deep=True)]
            if not agent.final_answer:
                raise RuntimeError("No complete answer")
            return {"transcript": request.text, "answer": agent.final_answer, "events": []}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("text task failed (%s)", type(exc).__name__)
            raise HTTPException(502, "任务处理失败或超时，请重试。") from None


@app.post("/api/voice")
async def voice_chat(audio: UploadFile = File(...)):
    await audio.close()
    raise HTTPException(410, "整段音频接口已停用，请使用 /ws/voice 原生流式语音。")


@app.websocket("/ws/voice")
async def native_voice_socket(socket: WebSocket):
    import secrets
    config = get_voice_settings()
    origin = socket.headers.get("origin")
    if origin and origin not in config.allowed_origins:
        await socket.close(code=1008, reason="Origin not allowed")
        return
    await socket.accept()
    if config.voice_auth_token:
        try:
            auth = await asyncio.wait_for(socket.receive_json(), timeout=10)
            if (not isinstance(auth, dict) or auth.get("type") != "auth"
                or not isinstance(auth.get("token"), str)
                or not secrets.compare_digest(auth["token"], config.voice_auth_token)):
                await socket.close(code=1008, reason="Authentication required")
                return
        except (TimeoutError, ValueError, WebSocketDisconnect):
            await socket.close(code=1008, reason="Authentication required")
            return
    if config.voice_provider == "seeduplex":
        from voice_agent.realtime import RealtimeVoiceSession
        await RealtimeVoiceSession(socket, config).run()
    else:
        # Explicit configuration only; native errors never fall back to cascade.
        from voice_agent.session import VoiceSession
        await VoiceSession(socket, config).run()


async def voice_socket(socket: WebSocket):
    """电话式全双工语音：后端用 VAD 全权接管轮次切换。

    前端全程连续上送 16kHz int16 裸 PCM；后端 VAD 判定开口 / 说完：
    - idle：等待开口（VAD 监听 + 滚动缓冲句首）。
    - capturing：用户正在说话（喂 ASR），说完 finalize 转写后跑 Agent→TTS。
    - speaking：Agent 播报中（VAD 继续监听），开口即打断并另起一轮。

    打断 / 轮次切换由后端驱动；前端只负责收音、播 TTS、渲染文字，并在
    播报真正结束时回传 ``playback_done``，让后端把 speaking 落回 idle——
    这样用户在「播报尾巴还没放完」时开口，也能被正确识别为打断。
    """
    await socket.accept()
    thread_id = str(uuid.uuid4())  # 一次通话 = 一个会话线程，承载上下文记忆

    vad = SileroVad(
        positive=settings.vad_positive,
        negative=settings.vad_negative,
        min_speech_frames=settings.vad_min_speech_frames,
        min_energy=settings.vad_min_energy,
        energy_margin_db=settings.vad_energy_margin_db,
        redemption_frames=settings.vad_redemption_frames,
    )
    asr_client: StreamingAsrClient | None = None
    recognize_task: asyncio.Task | None = None
    asr_prepare_task: asyncio.Task | None = None
    pipeline_task: asyncio.Task | None = None  # Agent→TTS 后台流水线，可被打断取消
    final_text: str = ""
    latest_asr_text: str = ""
    stable_asr_text: str = ""
    stable_asr_repeats = 0
    final_event = asyncio.Event()
    state = "idle"  # idle | capturing | speaking
    asr_context = AsrContextHistory(
        enabled=settings.asr_context_enabled,
        max_turns=settings.asr_context_turns,
        max_chars=settings.asr_context_max_chars,
        domain_hint=settings.asr_context_domain_hint,
    )

    # 滚动缓冲：保存最近 ~1s 的 PCM，开口时补喂给 ASR 以恢复句首
    pre_speech = bytearray()
    PRE_SPEECH_MAX = SAMPLE_RATE * 2  # 1s @16kHz int16 = 32000 字节

    async def recognize_loop():
        """ASR 识别：实时推中间结果，最终文本写入 final_text 并置位 final_event。"""
        nonlocal final_text, latest_asr_text, stable_asr_text, stable_asr_repeats
        last_text = ""
        async for result in asr_client.results():
            if result.text:
                last_text = result.text
                if result.text == latest_asr_text:
                    stable_asr_repeats += 1
                else:
                    stable_asr_repeats = 1
                latest_asr_text = result.text
                if stable_asr_repeats >= 2:
                    stable_asr_text = result.text
                if result.is_final:
                    final_text = result.text
                else:
                    await socket.send_json({"type": "asr_partial", "text": result.text})
        # 若服务端未在 is_last_package 里给出最终文本，用最后一次中间结果兜底
        if not final_text:
            final_text = last_text
        final_event.set()

    async def prepare_asr():
        """提前建立下一轮 ASR，避免把 WS 握手放在用户说完后的关键路径。"""
        nonlocal asr_client, recognize_task, final_text, latest_asr_text, stable_asr_text, stable_asr_repeats, final_event
        if asr_client is not None:
            return
        final_text = ""
        latest_asr_text = ""
        stable_asr_text = ""
        stable_asr_repeats = 0
        final_event = asyncio.Event()
        asr_client = StreamingAsrClient(context_data=asr_context.context_data())
        try:
            await asr_client._connect()
            recognize_task = asyncio.create_task(recognize_loop())
        except BaseException:
            await asr_client.close()
            asr_client = None
            raise

    async def open_asr(pre_pcm: bytes = b""):
        """取用预热的 ASR 流；没有预热成功时才同步建连。"""
        nonlocal asr_client, recognize_task, final_text, final_event, asr_prepare_task
        if asr_prepare_task is not None:
            task = asr_prepare_task
            asr_prepare_task = None
            try:
                await task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"ASR 预热失败，改为按轮建连：{exc}")
        # 预热连接可能因空闲超时关闭；识别任务结束也说明这条流不可继续使用。
        if (
            asr_client is None
            or not asr_client.is_open
            or (recognize_task is not None and recognize_task.done())
        ):
            if asr_client is not None:
                await asr_client.close()
            asr_client = None
            recognize_task = None
            await prepare_asr()
        # 预热流可能在本轮开始前已存在；只补喂句首，不重复建连。
        if pre_pcm:
            await asr_client.feed(pre_pcm)

    async def schedule_asr_prepare():
        """在本轮结束后立刻准备下一轮 ASR。"""
        nonlocal asr_prepare_task
        if asr_client is None and (asr_prepare_task is None or asr_prepare_task.done()):
            asr_prepare_task = asyncio.create_task(prepare_asr())

    async def _legacy_open_asr(pre_pcm: bytes = b""):
        """兼容保留：旧式强制重建逻辑，仅用于异常恢复。"""
        nonlocal asr_client, recognize_task, final_text, final_event
        if asr_client is not None:
            await asr_client.close()
        final_text = ""
        final_event = asyncio.Event()
        asr_client = StreamingAsrClient(context_data=asr_context.context_data())
        await asr_client._connect()
        recognize_task = asyncio.create_task(recognize_loop())
        if pre_pcm:
            await asr_client.feed(pre_pcm)

    async def finalize_asr() -> str:
        """结束当前识别流并取回最终文本。"""
        nonlocal asr_client, recognize_task
        if asr_client is None:
            return ""
        try:
            async with asyncio.timeout(settings.asr_final_timeout_s):
                await asr_client.finalize()
                await final_event.wait()
            return final_text
        except TimeoutError:
            logger.warning("ASR final 超时，请用户重试，避免用不完整文本执行业务工具")
            return ""
        finally:
            if recognize_task:
                recognize_task.cancel()
                await asyncio.gather(recognize_task, return_exceptions=True)
                recognize_task = None
            await asr_client.close()
            asr_client = None

    async def consume_agent(
        sentence_queue: asyncio.Queue,
        text: str,
        timings: dict,
        first_sentence: asyncio.Event,
        progress: ToolProgress,
    ):
        """消费 agent 事件：文字增量直推前端，句子入队供 TTS。

        客户端在回复途中挂断时 send_json 会抛错，这里静默收尾并保证
        None 哨兵一定入队，否则 tts_loop 会永久等待导致 handler 卡住。
        """
        try:
            async for event in run_agent_stream(text, thread_id):
                etype = event["type"]
                try:
                    if etype == "model_text_started":
                        timings.setdefault("t_first_model_delta", time.perf_counter())
                    elif etype == "agent_delta":
                        first_sentence.set()  # Do not interject once answer text is available.
                        if "t_first_token" not in timings:
                            timings["t_first_token"] = time.perf_counter()
                        await socket.send_json({"type": "agent_delta", "text": event["text"]})
                    elif etype == "sentence":
                        asr_context.add("坐席", event["text"])
                        first_sentence.set()
                        if "t_first_sentence" not in timings:
                            timings["t_first_sentence"] = time.perf_counter()
                        await sentence_queue.put(event["text"])
                    elif etype == "segment_end":
                        await sentence_queue.put(SEGMENT_END)
                        await socket.send_json({"type": "agent_segment_end"})
                    elif etype in ("tool_start", "tool_end"):
                        progress.update(event)
                        await socket.send_json({"type": "tool_event", "event": event})
                except Exception:
                    logger.info("Agent 发送中断，客户端可能已断开")
                    break
        except Exception as exc:
            logger.warning(f"Agent 流式运行失败：{exc}")
            raise
        finally:
            await sentence_queue.put(None)  # 结束哨兵


    async def run_pipeline(text: str, timings: dict):
        """跑一轮 Agent→TTS 流水线；播报内容全部回传后发 ``done``。"""
        nonlocal state
        sentence_queue: asyncio.Queue = asyncio.Queue()
        first_sentence = asyncio.Event()
        progress = ToolProgress()
        finished = asyncio.Event()

        async def produce():
            try:
                await consume_agent(sentence_queue, text, timings, first_sentence, progress)
            finally:
                finished.set()

        try:
            async with asyncio.timeout(settings.voice_response_timeout_s):
                async with asyncio.TaskGroup() as group:
                    group.create_task(produce())
                    group.create_task(stream_speech(
                        sentence_queue, socket.send_bytes, timings, finished,
                        progress=progress if text != OUTBOUND_OPENING else None,
                        response_started=first_sentence,
                    ))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("语音流水线失败")
            state = "idle"
            try:
                await socket.send_json({"type": "error", "message": "回复暂时失败，请再试一次。"})
            except (WebSocketDisconnect, RuntimeError):
                pass
            return
        await socket.send_json({"type": "done", "text": text})
        t0 = timings["t0"]
        parts = []
        for key, label in (
            ("t_asr_provisional", "ASR准最终"),
            ("t_asr", "ASR"),
            ("t_first_model_delta", "LLM原始首文本块"),
            ("t_first_token", "LLM首个可播文本"),
            ("t_first_sentence", "首句"),
            ("t_first_audio", "TTS首音频"),
            ("t_filler_audio", "承接语"),
            ("t_body_audio", "正文首音频"),
        ):
            if key in timings:
                parts.append(f"{label}={int((timings[key] - t0) * 1000)}ms")
        logger.info("链路延迟分解: " + " ".join(parts))

    async def handle_speech_start():
        """用户开口：播报中即打断；随后另起一段聆听（补喂句首）。"""
        nonlocal state, pipeline_task
        barged = state == "speaking"
        if barged:
            # Stop browser playback immediately, before network cleanup/ASR handshake.
            await socket.send_json({"type": "barge_in"})
        if barged and pipeline_task is not None and not pipeline_task.done():
            pipeline_task.cancel()
            try:
                await pipeline_task
            except asyncio.CancelledError:
                pass
            pipeline_task = None
        pre = bytes(pre_speech)
        pre_speech.clear()
        await open_asr(pre)
        state = "capturing"

    async def handle_speech_end():
        """用户说完：finalize ASR 取文本，跑 Agent→TTS 流水线。"""
        nonlocal state, pipeline_task
        timings = {"t0": time.perf_counter()}
        timings["t_speech_end"] = timings["t0"]
        vad.reset()
        pre_speech.clear()
        state = "speaking"
        await socket.send_json({"type": "speech_end"})

        async def finish_turn():
            nonlocal state
            try:
                # Await final text before any business tools, but keep receiving
                # microphone frames so the user can interrupt during ASR too.
                text = (await finalize_asr()).strip()
                timings["t_asr"] = time.perf_counter()
                if not text:
                    await schedule_asr_prepare()
                    await socket.send_json({"type": "error", "message": "没有识别到清晰语音，请靠近麦克风再试。"})
                    state = "idle"
                    return
                asr_context.add("客户", text)
                await socket.send_json({"type": "asr_final", "text": text})
                await run_pipeline(text, timings)
                # Build the next ASR connection after the response is known,
                # typically while its buffered audio is still playing. If the
                # customer barges in earlier, open_asr builds it immediately
                # from the partial response accumulated above.
                await schedule_asr_prepare()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("语音轮次失败")
                state = "idle"
                try:
                    await socket.send_json({"type": "error", "message": "语音连接暂时异常，请重试。"})
                except (WebSocketDisconnect, RuntimeError):
                    pass

        pipeline_task = asyncio.create_task(finish_turn())

    if settings.proactive_opening:
        # 外呼模式：接通即由 agent 主动开场推销。先告知前端进入「思考」态以接收文字与 TTS，
        # 再以开场触发词启动首轮 Agent→TTS 流水线（客户开口会走既有打断流程）。
        await socket.send_json({"type": "asr_final", "text": ""})
        state = "speaking"
        pipeline_task = asyncio.create_task(run_pipeline(OUTBOUND_OPENING, {"t0": time.perf_counter()}))

    # 通话建立后即预热第一轮 ASR；外呼开场期间也为客户打断准备好连接。
    asr_prepare_task = asyncio.create_task(prepare_asr())

    try:
        while True:
            message = await socket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            # 前端播报真正结束 → 后端从 speaking 落回 idle
            if message.get("text") == "playback_done":
                if state == "speaking" and (pipeline_task is None or pipeline_task.done()):
                    vad.reset()
                    pre_speech.clear()
                    state = "idle"
                continue

            pcm = message.get("bytes")
            if not pcm:
                continue  # 忽略其它文本消息

            # VAD 全状态运行；capturing 额外喂 ASR，其余状态喂滚动缓冲
            events = vad.process(pcm)
            if state == "capturing":
                await asr_client.feed(pcm)
            else:
                pre_speech.extend(pcm)
                if len(pre_speech) > PRE_SPEECH_MAX:
                    del pre_speech[: len(pre_speech) - PRE_SPEECH_MAX]

            for event in events:
                if event == "speech_start":
                    await handle_speech_start()
                elif event == "speech_end":
                    await handle_speech_end()
    except WebSocketDisconnect:
        pass
    finally:
        # ASGI may cancel the request scope on disconnect. Still join tasks and
        # close their HTTP sessions instead of abandoning network resources.
        with anyio.CancelScope(shield=True):
            tasks = [task for task in (pipeline_task, asr_prepare_task) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if recognize_task:
                recognize_task.cancel()
                await asyncio.gather(recognize_task, return_exceptions=True)
            if asr_client is not None:
                await asr_client.close()

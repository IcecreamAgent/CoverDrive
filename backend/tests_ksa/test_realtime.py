"""Native voice adapter behavior at the browser and provider protocol boundaries."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import voice_agent.realtime as realtime
import voice_agent.session as cascade
from config_fixture import configured_settings


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, event):
        await self.outgoing.put(event)

    async def send_bytes(self, data):
        await self.outgoing.put(data)

    async def control(self, event):
        await self.incoming.put({"type": "websocket.receive", "text": json.dumps(event)})

    async def until(self, predicate):
        events = []
        async with asyncio.timeout(2):
            while True:
                event = await self.outgoing.get()
                events.append(event)
                if predicate(event):
                    return events

    def drain(self):
        events = []
        while not self.outgoing.empty():
            events.append(self.outgoing.get_nowait())
        return events


def is_event(kind, **values):
    return lambda event: (
        isinstance(event, dict)
        and event.get("type") == kind
        and all(event.get(key) == value for key, value in values.items())
    )


def settings(**values):
    return configured_settings(
        _env_file=None,
        **{
            "voice_provider": "seeduplex",
            "llm_api_key": "fake-llm-key",
            "seeduplex_app_id": "fake-app-id",
            "seeduplex_access_key": "fake-access-key",
            "voice_auth_token": "",
            "router_enabled": False,
            **values,
        },
    )


@pytest.fixture
async def native(monkeypatch, request):
    state = SimpleNamespace(
        opened=0,
        closed=0,
        muted=[],
        cancels=0,
        audio=[],
        history=[],
        initial_history=[],
        answer_history=[],
        results=[],
        calls=[],
        contexts=[],
        agent_results={},
        speech=[],
        remembered=[],
        instructions="",
        tools=[],
        hints=[],
        sequence=[],
        pause=False,
        uses_tools=True,
        planner_calls=[],
        delivery_contexts=[],
        delivery_plans=[],
        planner_pause=False,
        planner_entered=asyncio.Event(),
        planner_released=asyncio.Event(),
        entered=asyncio.Event(),
        released=asyncio.Event(),
        canceled=asyncio.Event(),
    )

    class Stream:
        def __init__(self):
            self.events = asyncio.Queue()

        async def receive(self):
            while True:
                event = await self.events.get()
                if isinstance(event, BaseException):
                    raise event
                yield event

        async def set_muted(self, muted):
            state.muted.append(muted)

        async def send_audio(self, data):
            state.audio.append(data)

        async def commit_audio(self):
            state.audio.append("committed")

        async def cancel(self):
            state.cancels += 1

        async def update_instructions(self, instructions):
            state.sequence.append("instructions")
            state.hints.append(instructions)

        async def add_history(self, exchanges):
            state.sequence.append("history")
            state.history.extend(exchanges)

        async def add_assistant_answer(self, text):
            state.answer_history.append(text)

        async def send_tool_results(self, results):
            state.results.append(results)

    stream = Stream()

    class Client:
        def __init__(self, config):
            pass

        async def speech(self, text):
            state.sequence.append("speech")
            state.speech.append(text)
            yield b"\x03\x00\x04\x00"

        @asynccontextmanager
        async def stream(self, *, instructions, tools, history=None):
            state.opened += 1
            state.instructions = instructions
            state.tools = tools
            state.initial_history.append(list(history or []))
            try:
                yield stream
            finally:
                state.closed += 1

    class Agent:
        def __init__(self, config, *, task_only=False):
            assert task_only, "Seeduplex delegates directly to a task executor"
            self.on_tool_start = None

        def remember_exchange(self, question, answer):
            state.remembered.append((question, answer))

        def note_interruption(self):
            pass

        async def stream(self, text, *, force_deep=False, context=""):
            state.calls.append((text, force_deep))
            state.contexts.append(context)
            state.entered.set()
            try:
                if (
                    getattr(request, "param", False) and state.uses_tools
                    and self.on_tool_start is not None
                ):
                    await self.on_tool_start()
                if state.pause:
                    await state.released.wait()
                if text in state.agent_results:
                    yield state.agent_results[text]
                else:
                    yield "真实工具"
                    yield "结果。"
            except asyncio.CancelledError:
                state.canceled.set()
                raise

    class Planner:
        def __init__(self, config):
            pass

        async def plan(self, tasks, history, current_question="", **context):
            state.planner_calls.append((tuple(tasks), list(history), current_question))
            state.delivery_contexts.append(context)
            state.planner_entered.set()
            if state.planner_pause:
                await state.planner_released.wait()
            if state.delivery_plans:
                return state.delivery_plans.pop(0)
            first = tasks[0]
            return SimpleNamespace(
                task_ids=(first.task_id,), text=first.result or first.error,
            )

    class NoCascadeASR:
        def __init__(self, config):
            raise AssertionError("Native mode must not construct independent ASR")

        def stream(self):
            raise AssertionError("Native speech must not open cascade ASR")

    class NoCascadeTTS:
        def __init__(self, config):
            raise AssertionError("Native mode must not construct independent TTS")

        async def synthesize(self, text):
            raise AssertionError("Native speech must not invoke separate TTS")
            yield b""  # pragma: no cover

    monkeypatch.setattr(realtime, "SeeduplexClient", Client)
    monkeypatch.setattr(realtime, "AgentSession", Agent)
    monkeypatch.setattr(realtime, "DeliveryPlanner", Planner, raising=False)
    monkeypatch.setattr(cascade, "ASRClient", NoCascadeASR)
    monkeypatch.setattr(cascade, "TTSClient", NoCascadeTTS)
    socket = Socket()
    instance = realtime.RealtimeVoiceSession(
        socket, settings(background_tasks_enabled=getattr(request, "param", False))
    )
    await instance.enable_native()
    socket.drain()
    try:
        yield socket, instance, stream, state
    finally:
        await instance.close_native()
        await instance.cancel_response()
        await instance.duplex.close()


async def utterance(instance, item="q1", text="二加二等于几？"):
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": item,
            "round_id": 1,
        }
    )
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item,
            "text": text,
        }
    )
    return instance.native_turn


async def output(instance, item="q1", response="r1", text="四。"):
    await instance.handle_native_event(
        {
            "type": "response.output_text.delta",
            "question_id": item,
            "response_id": response,
            "delta": text,
        }
    )


async def audio_start(instance, item="q1", response="r1"):
    await instance.handle_native_event(
        {
            "type": "response.output_audio.started",
            "question_id": item,
            "response_id": response,
            "tts_type": "default",
        }
    )


async def audio_chunk(instance, pcm=b"\x01\x00\x02\x00"):
    await instance.handle_native_event(
        {"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()}
    )


async def finish(instance, item="q1", response="r1", text="四。"):
    await instance.handle_native_event(
        {
            "type": "response.output_text.done",
            "question_id": item,
            "response_id": response,
            "text": text,
        }
    )
    await instance.handle_native_event(
        {"type": "response.output_audio.done", "question_id": item, "response_id": response}
    )
    await instance.handle_native_event({"type": "response.done", "response": {"usage": {}}})


def tool_event(item="q1", calls=None):
    return {
        "type": "response.function_call_arguments.done",
        "question_id": item,
        "items": calls
        or [
            {
                "call_id": "call-1",
                "name": "delegate_deep_agent",
                "arguments": json.dumps({"question": "上海现在几点？"}),
            }
        ],
    }


async def test_native_asr_revisions_replace_text_and_final_text_is_authoritative(native):
    socket, instance, _, _ = native
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "q1",
            "round_id": 1,
        }
    )
    for text in ("二", "二加", "二加二等于几", "2+2 等于几？"):
        await instance.handle_native_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": "q1",
                "delta": text,
            }
        )
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "q1",
            "text": "2+2 等于几？",
        }
    )
    events = socket.drain()
    transcripts = [e for e in events if is_event("transcript")(e)]
    assert [e["text"] for e in transcripts] == [
        "二",
        "二加",
        "二加二等于几",
        "2+2 等于几？",
        "2+2 等于几？",
    ]
    assert transcripts[-1]["is_final"] is True
    assert len([e for e in events if is_event("speech_start")(e)]) == 1
    assert len([e for e in events if is_event("speech_end", reason="seeduplex_native")(e)]) == 1
    assert any(is_event("agent_route", route="fast", provider="seeduplex")(e) for e in events)


async def test_native_audio_automatically_replies_without_cascade_or_manual_stop(native):
    socket, instance, _, state = native
    await instance.handle_audio(bytes(640))
    turn = await utterance(instance)
    await output(instance, text="四")
    await output(instance, text="。")
    await audio_start(instance)
    await audio_chunk(instance)
    await finish(instance)
    events = socket.drain()
    assert state.audio == [bytes(640)]
    assert state.calls == []
    assert b"\x01\x00\x02\x00" in events
    header = next(e for e in events if is_event("audio_start")(e))
    assert {k: header[k] for k in ("sample_rate", "channels", "format")} == {
        "sample_rate": 24000,
        "channels": 1,
        "format": "pcm",
    }
    assert all(
        e.get("turn_id") == turn.turn_id
        for e in events
        if isinstance(e, dict) and e["type"] not in ("listening", "execution_event")
    )
    assert len([e for e in events if is_event("audio_end")(e)]) == 1
    assert len([e for e in events if is_event("assistant_done", text="四。")(e)]) == 1
    assert instance.public_history == [("二加二等于几？", "四。")]
    assert state.closed == 0


async def test_native_tool_calls_force_deep_and_return_single_batch_without_tts(native):
    socket, instance, _, state = native
    await utterance(instance, text="请帮我核实两件事情")
    await audio_start(instance)
    await audio_chunk(instance)
    socket.drain()
    calls = [
        {
            "call_id": "a",
            "name": "delegate_deep_agent",
            "arguments": json.dumps({"question": "上海现在几点？"}),
        },
        {
            "call_id": "b",
            "name": "delegate_deep_agent",
            "arguments": json.dumps({"question": "规划下一步任务"}),
        },
    ]
    await instance.handle_native_event(tool_event(calls=calls))
    await instance.tool_task
    assert state.calls == [("上海现在几点？", True), ("规划下一步任务", True)]
    assert state.results == [
        [{"call_id": "a", "text": "真实工具结果。"}, {"call_id": "b", "text": "真实工具结果。"}]
    ]
    events = socket.drain()
    assert any(is_event("agent_route", route="deep")(e) for e in events)
    assert not any(is_event("assistant_delta")(e) or is_event("audio_start")(e) for e in events)
    await output(instance, text="工具已返回。")
    await audio_start(instance)
    await audio_chunk(instance)
    await finish(instance, text="工具已返回。")
    assert instance.public_history == [("请帮我核实两件事情", "工具已返回。")]


@pytest.mark.parametrize("completion_order", [("text", "audio"), ("audio", "text")])
async def test_native_confirmation_and_final_answer_share_one_continuing_response(
    native, completion_order
):
    socket, instance, _, state = native
    question = "请帮我核实上海时间"
    turn = await utterance(instance, text=question)
    socket.drain()
    acknowledgement = "我帮你核实一下。"
    answer = "上海当前时间是工具返回的时间。"
    await output(instance, text=acknowledgement)
    await audio_start(instance)
    ack_pcm = b"\x01\x00\x02\x00"
    await audio_chunk(instance, pcm=ack_pcm)

    state.pause = True
    await instance.handle_native_event(tool_event())
    await state.entered.wait()
    pending = socket.drain()
    assert ack_pcm in pending
    assert any(is_event("assistant_delta", text=acknowledgement)(e) for e in pending)
    assert any(is_event("agent_progress", phase="working")(e) for e in pending)
    assert not any(
        is_event("assistant_done")(e)
        or is_event("audio_end")(e)
        or is_event("state", stage="idle")(e)
        for e in pending
    )
    assert not instance.tool_task.done()
    assert not turn.finished
    assert state.results == []
    assert instance.public_history == []

    state.released.set()
    await instance.tool_task
    assert state.results == [[{"call_id": "call-1", "text": "真实工具结果。"}]]
    after_result = socket.drain()
    assert any(is_event("agent_progress", phase="answering")(e) for e in after_result)
    assert not turn.finished
    assert instance.public_history == []

    await output(instance, text=answer)
    final_pcm = b"\x03\x00\x04\x00"
    await audio_chunk(instance, pcm=final_pcm)
    continuation = socket.drain()
    assert final_pcm in continuation
    assert turn.answer == acknowledgement + answer
    assert not any(is_event("audio_start")(e) for e in continuation)

    completed = []
    for index, kind in enumerate(completion_order):
        event = {
            "type": f"response.output_{kind}.done",
            "question_id": "q1",
            "response_id": "r1",
        }
        if kind == "text":
            event["text"] = acknowledgement + answer
        await instance.handle_native_event(event)
        completed.extend(socket.drain())
        if index == 0:
            assert not turn.finished
            assert instance.public_history == []
            assert not any(is_event("state", stage="idle")(e) for e in completed)
    all_events = pending + after_result + continuation + completed
    assert turn.finished
    assert instance.public_history == [(question, acknowledgement + answer)]
    assert len([e for e in all_events if is_event("audio_start")(e)]) == 1
    assert len([e for e in all_events if is_event("audio_end")(e)]) == 1
    assert len([e for e in all_events if is_event("state", stage="idle")(e)]) == 1
    assert len([e for e in all_events if is_event("assistant_done")(e)]) == 1
    assert any(is_event("agent_progress", phase="completed")(e) for e in completed)
    assert state.remembered == []


async def test_barge_in_during_confirmed_deep_wait_discards_acknowledgement_and_result(native):
    socket, instance, _, state = native
    first = await utterance(instance, text="请帮我分析一个复杂问题")
    await output(instance, text="我先核实相关信息。")
    await audio_start(instance)
    await audio_chunk(instance)
    state.pause = True
    await instance.handle_native_event(tool_event())
    await state.entered.wait()
    task = instance.tool_task
    socket.drain()

    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "q2",
            "round_id": 2,
        }
    )
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "q2",
            "delta": "改一下，先回答新问题",
        }
    )
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "q2",
            "text": "改一下，先回答新问题",
        }
    )
    assert task.done()
    assert state.canceled.is_set()
    assert first.canceled
    assert state.results == []
    assert instance.public_history == []
    interrupted = socket.drain()
    assert any(is_event("barge_in_confirmed")(e) for e in interrupted)

    await audio_chunk(instance, pcm=b"\x05\x00\x06\x00")
    await output(instance, text="旧任务已完成，迟到的确认语或正文。")
    await audio_start(instance)
    await audio_chunk(instance, pcm=b"\x07\x00\x08\x00")
    await finish(instance, text="旧任务最终结果。")
    state.released.set()
    assert socket.drain() == []
    assert instance.native_turn.answer == ""
    assert not instance.native_turn.canceled
    assert instance.public_history == []


async def test_unknown_tools_return_safe_data_and_never_execute_an_agent(native):
    _, instance, _, state = native
    await utterance(instance)
    calls = [
        {
            "call_id": "unknown",
            "name": "execute_shell",
            "arguments": json.dumps({"question": "print private secret"}),
        }
    ]
    await instance.handle_native_event(tool_event(calls=calls))
    await instance.tool_task
    assert state.calls == []
    result = state.results[0][0]
    assert result["call_id"] == "unknown"
    assert json.loads(result["text"]) == {"error": "工具或参数不可用"}
    assert "private" not in result["text"]


async def test_backchannel_suspends_then_rejects_without_starting_a_new_turn(native):
    socket, instance, _, state = native
    turn = await utterance(instance)
    await output(instance)
    await audio_start(instance)
    socket.drain()
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "ack",
            "round_id": 2,
        }
    )
    assert any(is_event("barge_in_candidate")(e) for e in socket.drain())
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "ack",
            "delta": "嗯",
        }
    )
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "ack",
            "text": "嗯，好的。",
        }
    )
    events = socket.drain()
    assert any(is_event("barge_in_rejected", reason="backchannel")(e) for e in events)
    assert not any(is_event("speech_start")(e) for e in events)
    assert instance.native_turn is turn
    assert not turn.canceled
    assert state.cancels == 0


async def test_meaningful_barge_in_cancels_deep_work_and_never_returns_stale_result(native):
    socket, instance, _, state = native
    first = await utterance(instance, text="请帮我核实一个事情")
    await audio_start(instance)
    await audio_chunk(instance)
    state.pause = True
    await instance.handle_native_event(tool_event())
    await state.entered.wait()
    old_task = instance.tool_task
    socket.drain()
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "q2",
            "round_id": 2,
        }
    )
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "q2",
            "delta": "换个问题",
        }
    )
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "q2",
            "text": "换个问题，二加二等于几？",
        }
    )
    state.released.set()
    assert old_task.done()
    assert state.canceled.is_set()
    assert first.canceled
    assert instance.native_turn is not first
    assert state.results == []
    events = socket.drain()
    assert any(is_event("barge_in_confirmed")(e) for e in events)
    assert any(is_event("speech_start", barge_in=True)(e) for e in events)
    socket.drain()
    await output(instance, item="q1", response="late-r", text="过期结果")
    await audio_start(instance, item="q1", response="late-r")
    await audio_chunk(instance)
    assert socket.drain() == []


async def test_uncorrelated_audio_and_late_old_response_never_attach_to_new_turn(native):
    socket, instance, _, _ = native
    await utterance(instance)
    await output(instance)
    await audio_start(instance)
    await instance.interrupt(notify=False)
    await utterance(instance, item="q2", text="新问题")
    socket.drain()
    await audio_chunk(instance)
    await instance.handle_native_event(
        {"type": "response.output_text.delta", "response_id": "r1", "delta": "旧回答"}
    )
    await instance.handle_native_event(
        {"type": "response.output_audio.started", "response_id": "r1"}
    )
    await audio_chunk(instance)
    await instance.handle_native_event({"type": "response.canceled"})
    assert socket.drain() == []
    assert not instance.native_turn.canceled
    assert instance.native_turn.answer == ""


async def test_explicit_unknown_question_cannot_claim_the_current_turn(native):
    socket, instance, _, _ = native
    await utterance(instance)
    socket.drain()
    await output(instance, item="unknown-question", response="unknown-response", text="未归属回答")
    await audio_start(instance, item="unknown-question", response="unknown-response")
    await audio_chunk(instance)
    assert socket.drain() == []
    assert instance.native_turn.answer == ""


async def test_unscoped_late_usage_done_cannot_finish_a_new_unanswered_turn(native):
    socket, instance, _, _ = native
    await utterance(instance)
    await output(instance)
    await audio_start(instance)
    await instance.interrupt(notify=False)
    current = await utterance(instance, item="q2", text="新问题")
    socket.drain()
    await instance.handle_native_event(
        {"type": "response.done", "response": {"usage": {"total_tokens": 30}}}
    )
    assert not current.finished
    assert not any(is_event("state", stage="idle")(e) for e in socket.drain())


async def test_ignored_old_asr_delta_cannot_confirm_another_candidates_barge_in(native):
    socket, instance, _, _ = native
    old = await utterance(instance)
    await output(instance)
    await audio_start(instance)
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "q2",
            "round_id": 2,
        }
    )
    socket.drain()
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "q1",
            "delta": "旧问题的延迟识别",
        }
    )
    assert instance.native_turn is old
    assert not old.canceled
    assert not any(is_event("barge_in_confirmed")(e) for e in socket.drain())


async def test_unlisten_mutes_capture_but_existing_native_answer_can_finish(native):
    socket, instance, _, state = native
    turn = await utterance(instance)
    await instance.disable_native()
    assert state.muted[-1] is True
    await instance.handle_audio(bytes(640))
    assert state.audio == []
    await instance.handle_native_event(
        {"type": "conversation.item.input_audio_transcription.started", "item_id": "muted-new"}
    )
    assert instance.native_turn is turn
    await output(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    await finish(instance)
    assert any(is_event("assistant_done", text="四。")(e) for e in socket.drain())
    assert turn.finished
    assert not turn.canceled


async def test_native_enable_reuses_connection_and_close_cancels_tasks(native):
    _, instance, _, state = native
    await instance.enable_native()
    assert state.opened == 1
    await utterance(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    state.pause = True
    await instance.handle_native_event(tool_event())
    await state.entered.wait()
    task = instance.tool_task
    await instance.close_native()
    assert state.closed == 1
    assert task.done()
    assert state.canceled.is_set()
    assert instance.native_task is None
    assert instance.native is None


async def test_receiver_failure_reports_safe_error_and_releases_pending_tool(native):
    socket, instance, stream, state = native
    await utterance(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    state.pause = True
    await instance.handle_native_event(tool_event())
    await state.entered.wait()
    await stream.events.put(RuntimeError("private upstream key details"))
    events = await socket.until(is_event("listening", enabled=False))
    assert any(is_event("error", stage="seeduplex")(e) for e in events)
    assert "private" not in str(events)
    assert state.canceled.is_set()
    assert not instance.listening


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        "null",
        '{"type":"listen","turn_id":17}',
        '{"type":"text","text":17}',
        '{"type":"unsupported"}',
    ],
)
async def test_malformed_browser_messages_are_reported_without_ending_session(native, raw):
    socket, instance, _, _ = native
    task = asyncio.create_task(instance.run())
    try:
        await socket.until(is_event("ready"))
        await socket.incoming.put({"type": "websocket.receive", "text": raw})
        events = await socket.until(is_event("error", stage="protocol"))
        assert events[-1]["message"]
        await socket.control({"type": "ping"})
        await socket.until(is_event("pong"))
        assert not task.done()
    finally:
        await socket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(task, 2)


@pytest.mark.parametrize("audio", [b"\x00", bytes(64002)])
async def test_invalid_browser_audio_never_reaches_provider(native, audio):
    socket, instance, _, state = native
    await instance.handle_audio(audio)
    assert state.audio == []
    assert any(is_event("error", stage="protocol")(e) for e in socket.drain())


async def test_completed_turn_rejects_late_text_and_audio_restarts(native):
    socket, instance, _, _ = native
    turn = await utterance(instance)
    await output(instance)
    await audio_start(instance)
    await finish(instance)
    assert turn.finished
    socket.drain()
    await output(instance, text="完成后迟到的旧文字")
    await audio_start(instance)
    await audio_chunk(instance)
    assert socket.drain() == []
    assert turn.answer == "四。"


async def test_audio_completion_waits_for_authoritative_text_before_committing_history(native):
    socket, instance, _, _ = native
    turn = await utterance(instance)
    await output(instance, text="四")
    await audio_start(instance)
    await audio_chunk(instance)
    socket.drain()
    await instance.handle_native_event(
        {"type": "response.output_audio.done", "question_id": "q1", "response_id": "r1"}
    )
    assert not turn.finished
    assert instance.public_history == []
    assert not any(is_event("state", stage="idle")(e) for e in socket.drain())
    await instance.handle_native_event(
        {
            "type": "response.output_text.done",
            "question_id": "q1",
            "response_id": "r1",
            "text": "四。",
        }
    )
    assert turn.finished
    assert instance.public_history == [("二加二等于几？", "四。")]
    assert len([e for e in socket.drain() if is_event("assistant_done", text="四。")(e)]) == 1


async def test_native_fast_exchange_is_given_to_deep_agent_as_public_history(native):
    _, instance, _, state = native
    await utterance(instance)
    await output(instance)
    await audio_start(instance)
    await finish(instance)
    await utterance(instance, item="q2", text="结合刚才的结果帮我规划下一步")
    await audio_start(instance, item="q2", response="r2")
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event(item="q2"))
    await instance.tool_task
    assert state.remembered == [("二加二等于几？", "四。")]
    assert state.calls == [("上海现在几点？", True)]
    assert state.results[0][0]["call_id"] == "call-1"


def stub_server_answer(monkeypatch):
    """Pause task execution; speech remains owned by the native client."""
    state = SimpleNamespace(
        calls=[],
        entered=asyncio.Event(),
        released=asyncio.Event(),
        canceled=asyncio.Event(),
        answer="深度回答补充了延迟的原因和可执行的排查步骤。",
    )

    async def speech(self, text):
        state.calls.append(text)
        state.entered.set()
        try:
            await state.released.wait()
            yield b"\x03\x00\x04\x00"
        except asyncio.CancelledError:
            state.canceled.set()
            raise

    monkeypatch.setattr(realtime.SeeduplexClient, "speech", speech)
    return state


async def complete_initial_native_answer(instance):
    question = "语音系统为什么总是出现延迟？"
    answer = "可能与网络和服务处理时间有关。"
    await utterance(instance, text=question)
    await output(instance, text=answer)
    await audio_start(instance)
    await finish(instance, text=answer)
    return question, answer


@pytest.mark.parametrize(
    "question",
    ["今天天气这个词用英语怎么说？", "刚才没听清，请再说一遍。"],
)
async def test_new_topic_or_playback_request_does_not_force_assisted(
    native, monkeypatch, question
):
    _, instance, _, state = native
    server = stub_server_answer(monkeypatch)
    await complete_initial_native_answer(instance)
    turn = await utterance(instance, item="q2", text=question)
    assert not turn.assisted
    assert server.calls == []
    assert state.cancels == 0


@pytest.mark.parametrize("empty_item_id", ["", "empty-upstream-start"])
async def test_empty_native_candidate_resumes_output_and_cannot_block_the_next_question(
    native, monkeypatch, empty_item_id
):
    socket, instance, _, _ = native
    server = stub_server_answer(monkeypatch)
    question, previous = await complete_initial_native_answer(instance)
    forced = await utterance(instance, item="q2", text=question)
    forced.deep = True
    forced.deep_result = server.answer
    await instance._recover_result(forced)
    async with asyncio.timeout(2):
        await server.entered.wait()
    pending = instance.response_task
    socket.drain()

    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": empty_item_id,
        }
    )
    assert instance.candidate
    events = await socket.until(is_event("barge_in_rejected", reason="no_words_yet"))
    assert any(is_event("barge_in_candidate")(event) for event in events)
    assert not any(is_event("barge_in_confirmed")(event) for event in events)
    assert instance.native_turn is forced
    assert not forced.canceled
    assert not pending.done()
    assert not server.canceled.is_set()
    assert instance.candidate  # Late words can still provide a real interruption.

    server.released.set()
    await pending
    assert forced.finished
    assert instance.public_history == [(question, previous), (question, server.answer)]
    socket.drain()

    next_question = "二加二等于几？"
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "q3",
        }
    )
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "q3",
            "delta": "二加二",
        }
    )
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "q3",
            "text": next_question,
        }
    )
    current = instance.native_turn
    assert current is not forced
    assert not current.assisted
    assert current.question == next_question
    assert not instance.candidate
    events = socket.drain()
    transcripts = [event for event in events if is_event("transcript")(event)]
    assert [(event["text"], event["is_final"]) for event in transcripts] == [
        ("二加二", False),
        (next_question, True),
    ]
    assert any(is_event("speech_start", turn_id=current.turn_id)(event) for event in events)
    await output(instance, item="q3", response="r3", text="四。")
    await audio_start(instance, item="q3", response="r3")
    await audio_chunk(instance)
    await finish(instance, item="q3", response="r3", text="四。")
    assert current.finished
    assert instance.public_history[-1] == (next_question, "四。")
    assert any(is_event("assistant_done", text="四。")(event) for event in socket.drain())


async def test_late_words_after_empty_candidate_timeout_still_confirm_barge_in(native, monkeypatch):
    socket, instance, _, _ = native
    server = stub_server_answer(monkeypatch)
    question, previous = await complete_initial_native_answer(instance)
    old = await utterance(instance, item="q2", text=question)
    old.deep = True
    old.deep_result = server.answer
    await instance._recover_result(old)
    async with asyncio.timeout(2):
        await server.entered.wait()
    pending = instance.response_task
    socket.drain()

    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "late-words",
        }
    )
    await socket.until(is_event("barge_in_rejected", reason="no_words_yet"))
    assert instance.native_turn is old
    assert not old.canceled
    assert not pending.done()

    next_question = "换个问题，二加二等于几？"
    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "late-words",
            "delta": next_question,
        }
    )
    assert pending.done()
    assert server.canceled.is_set()
    assert old.canceled
    current = instance.native_turn
    assert current is not old
    assert current.question == next_question
    assert not instance.candidate
    events = socket.drain()
    assert any(is_event("barge_in_confirmed")(event) for event in events)
    assert any(is_event("speech_start", barge_in=True)(event) for event in events)
    assert any(is_event("transcript", text=next_question)(event) for event in events)
    assert instance.public_history == [(question, previous)]

    await instance.handle_native_event(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "late-words",
            "text": next_question,
        }
    )
    assert not current.assisted
    await output(instance, item="late-words", response="r3", text="四。")
    await audio_start(instance, item="late-words", response="r3")
    await audio_chunk(instance)
    await finish(instance, item="late-words", response="r3", text="四。")
    assert current.finished
    assert instance.public_history == [(question, previous), (next_question, "四。")]


async def test_native_result_timeout_uses_existing_deep_result_without_executing_again(native):
    socket, instance, _, state = native
    instance.settings.native_result_timeout = 0.02
    turn = await utterance(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await instance.tool_task
    watchdog = turn.result_task
    await watchdog
    await instance.response_task
    assert state.calls == [("上海现在几点？", True)]
    assert state.speech == ["真实工具结果。"]
    assert turn.fallback_reason == "native_result_timeout"
    assert turn.finished
    assert instance.public_history == [("二加二等于几？", "真实工具结果。")]
    assert any(
        is_event("agent_recovery", reason="native_result_timeout")(event)
        for event in socket.drain()
    )


async def test_continuing_native_audio_extends_result_watchdog(native):
    _, instance, _, state = native
    instance.settings.native_result_timeout = 0.04
    turn = await utterance(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await instance.tool_task
    await output(instance, text="结果已经出来。")
    await audio_start(instance)
    for _ in range(4):
        await audio_chunk(instance)
        await asyncio.sleep(0.015)
    assert not turn.assisted
    assert state.cancels == 0
    await finish(instance, text="结果已经出来。")
    assert turn.finished
    assert not turn.assisted


async def test_interruption_after_tool_return_cancels_native_result_watchdog(native):
    _, instance, _, state = native
    instance.settings.native_result_timeout = 0.04
    turn = await utterance(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await instance.tool_task
    watchdog = turn.result_task
    await instance.interrupt()
    await asyncio.sleep(0.06)
    assert watchdog.cancelled()
    assert not turn.assisted
    assert len(state.calls) == 1


async def test_interrupt_while_fallback_is_canceling_native_never_starts_a_stale_answer(
    native, monkeypatch
):
    _, instance, stream, _ = native
    server = stub_server_answer(monkeypatch)
    old = await utterance(instance)
    entered, released = asyncio.Event(), asyncio.Event()

    async def delayed_cancel():
        entered.set()
        await released.wait()

    monkeypatch.setattr(stream, "cancel", delayed_cancel)
    old.deep = True
    old.deep_result = "已有工具结果。"
    fallback = asyncio.create_task(instance._recover_result(old))
    await entered.wait()
    await utterance(instance, item="q2", text="三加四等于几？")
    assert old.canceled
    released.set()
    await fallback
    assert server.calls == []
    assert instance.native_turn is not old


async def pending_playback_result(native):
    socket, instance, _, state = native
    instance.playback_barriers = True
    turn = await utterance(instance, text="上海现在几点？")
    await output(instance, text="我查一下当前时间。")
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await state.entered.wait()
    events = await socket.until(is_event("playback_barrier"))
    return turn, events[-1]


async def test_ready_deep_result_waits_for_browser_playback_instead_of_cutting_speech(native):
    socket, instance, _, state = native
    turn, barrier = await pending_playback_result(native)
    assert state.calls  # The slow task ran while the acknowledgement was still playing.
    assert turn.deep_result == "真实工具结果。"
    assert not instance.tool_task.done()
    assert state.results == []
    for wrong in (
        {"turn_id": "old-turn", "barrier_id": barrier["barrier_id"]},
        {"turn_id": turn.turn_id, "barrier_id": "old-marker"},
    ):
        instance.confirm_playback(wrong)
        assert not turn.playback_drained.is_set()
    instance.confirm_playback(barrier)
    await asyncio.wait_for(instance.tool_task, 1)
    assert len(state.results) == 1
    assert any(is_event("agent_progress", phase="answering")(e) for e in socket.drain())


async def test_late_prefix_pcm_requires_another_playback_confirmation(native):
    socket, instance, _, state = native
    turn, first = await pending_playback_result(native)
    await audio_chunk(instance)
    instance.confirm_playback(first)
    second = (await socket.until(is_event("playback_barrier")))[-1]
    assert second["barrier_id"] != first["barrier_id"]
    assert not instance.tool_task.done()
    assert state.results == []
    instance.confirm_playback(first)
    assert not turn.playback_drained.is_set()
    instance.confirm_playback(second)
    await asyncio.wait_for(instance.tool_task, 1)
    assert len(state.results) == 1


async def test_slow_result_ready_before_first_prefix_pcm_waits_for_that_audio(native):
    socket, instance, _, state = native
    instance.playback_barriers = True
    await utterance(instance, text="上海现在几点？")
    await output(instance, text="我查一下当前时间。")
    await audio_start(instance)
    await instance.handle_native_event(tool_event())
    await state.entered.wait()
    await socket.until(is_event("agent_route", route="deep"))
    assert state.results == []
    await audio_chunk(instance)
    barrier = (await socket.until(is_event("playback_barrier")))[-1]
    assert state.results == []
    instance.confirm_playback(barrier)
    await asyncio.wait_for(instance.tool_task, 1)
    assert len(state.results) == 1


async def test_interrupt_during_playback_wait_discards_prepared_result(native):
    _, instance, _, state = native
    turn, barrier = await pending_playback_result(native)
    pending = instance.tool_task
    await instance.interrupt()
    instance.confirm_playback(barrier)
    assert pending.done()
    assert turn.canceled
    assert not turn.playback_drained.is_set()
    assert state.results == []


async def test_browser_playback_acknowledgement_is_handled_without_blocking_input(native):
    socket, instance, _, state = native
    turn, barrier = await pending_playback_result(native)
    runner = asyncio.create_task(instance.run())
    try:
        await socket.control({"type": "ping"})
        await socket.until(is_event("pong"))
        await socket.control({"type": "playback", "turn_id": turn.turn_id, "playing": False})
        await socket.control({"type": "ping"})
        await socket.until(is_event("pong"))
        assert state.results == []  # A queue gap is not a fence acknowledgement.
        await socket.control({**barrier, "type": "playback_drained"})
        await asyncio.wait_for(instance.tool_task, 1)
        assert len(state.results) == 1
    finally:
        await socket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(runner, 2)


async def test_keyboard_task_uses_deepagents_and_only_seeduplex_speech(native):
    socket, instance, _, state = native
    instance.turn_id = "typed-task"
    await instance.respond("帮我分析一下")
    assert state.calls == [("帮我分析一下", True)]
    assert state.speech == ["真实工具结果。"]
    assert instance.asr is None and instance.tts is None
    assert state.history == [("帮我分析一下", "真实工具结果。")]
    routes = [event for event in socket.drain() if is_event("agent_route")(event)]
    assert routes and all(event["speech_provider"] == "seeduplex" for event in routes)
    assert instance.public_history == [("帮我分析一下", "真实工具结果。")]


async def test_seeduplex_speech_failure_has_no_independent_tts_fallback(native, monkeypatch):
    socket, instance, _, _ = native

    async def fail_speech(text):
        raise RuntimeError("private credential must not be exposed")
        yield b""  # pragma: no cover

    monkeypatch.setattr(instance.native_client, "speech", fail_speech)
    await instance.respond("帮我分析一下")
    events = socket.drain()
    assert any(is_event("error", stage="seeduplex")(event) for event in events)
    assert not any(is_event("assistant_done")(event) for event in events)
    assert not any(is_event("agent_fallback")(event) for event in events)
    assert "private credential" not in str(events)
    assert instance.tts is None and instance.public_history == []


async def test_interrupt_cancels_seeduplex_assisted_speech_and_never_commits(native, monkeypatch):
    _, instance, _, _ = native
    entered, canceled = asyncio.Event(), asyncio.Event()

    async def paused_speech(text):
        entered.set()
        try:
            await asyncio.Event().wait()
            yield b""  # pragma: no cover
        finally:
            canceled.set()

    monkeypatch.setattr(instance.native_client, "speech", paused_speech)
    instance.response_task = asyncio.create_task(instance.respond("帮我分析一下"))
    pending = instance.response_task
    await entered.wait()
    await instance.interrupt()
    assert pending.done() and canceled.is_set()
    assert instance.native_turn.canceled
    assert instance.public_history == []


async def test_assisted_result_waits_for_existing_prefix_playback(native):
    socket, instance, _, state = native
    instance.playback_barriers = True
    turn = await utterance(instance)
    await output(instance, text="我查一下。")
    await audio_start(instance)
    await audio_chunk(instance)
    turn.deep = True
    turn.deep_result = "复用的任务结果。"
    await instance._recover_result(turn)
    barrier = (await socket.until(is_event("playback_barrier")))[-1]
    assert state.speech == [] and state.calls == []
    instance.confirm_playback(barrier)
    await instance.response_task
    assert state.speech == ["复用的任务结果。"]
    assert state.calls == []


@pytest.mark.parametrize(
    "question",
    ["请深入分析语音系统延迟的原因。", "语音系统为什么总是出现延迟？", "五个人，两小时，推理的。"],
)
async def test_only_native_tool_calls_can_start_deep_work_even_with_cascade_routing_enabled(
    native, question
):
    socket, instance, _, state = native
    instance.settings.fast_llm_enabled = True
    instance.settings.router_enabled = True
    instance.public_history = [("语音系统为什么总是出现延迟？", "可能与网络有关。")]
    turn = await utterance(instance, text=question)
    await asyncio.sleep(0.05)
    assert state.calls == [] and state.hints == []
    assert not turn.deep and not turn.assisted
    assert instance.agent is None and instance.tool_task is None
    assert not any(is_event("agent_recovery")(event) for event in socket.drain())
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await instance.tool_task
    assert turn.deep and not turn.assisted
    assert len(state.calls) == 1 and state.calls[0][1] is True
    assert state.hints == []


@pytest.mark.parametrize("answer", ["四。", "希望面向哪些人，需要多大规模？", "我给你梳理一下。"])
async def test_native_final_answer_commits_without_external_semantic_review(native, answer):
    socket, instance, _, state = native
    instance.settings.fast_llm_enabled = True
    instance.settings.router_enabled = True
    turn = await utterance(instance)
    await output(instance, text=answer)
    await audio_start(instance)
    await audio_chunk(instance)
    await finish(instance, text=answer)
    assert turn.finished and not turn.deep and not turn.assisted
    assert instance.public_history == [("二加二等于几？", answer)]
    assert state.calls == [] and state.hints == [] and state.cancels == 0
    assert any(is_event("assistant_done", text=answer)(event) for event in socket.drain())


async def test_technical_recovery_cannot_invent_a_missing_delegation(native):
    socket, instance, _, state = native
    turn = await utterance(instance, text="帮我深入比较几种方案")
    await instance._recover_result(turn)
    assert not turn.assisted and not turn.deep
    assert state.calls == [] and state.speech == [] and state.cancels == 0
    assert not any(is_event("agent_recovery")(event) for event in socket.drain())


async def test_result_recovery_filters_late_native_audio_and_never_executes_task_twice(
    native, monkeypatch
):
    socket, instance, _, state = native
    speech = stub_server_answer(monkeypatch)
    turn = await utterance(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await instance.tool_task
    await instance._recover_result(turn)
    await speech.entered.wait()
    socket.drain()
    await output(instance, text="迟到的原生答案。")
    await audio_start(instance)
    await audio_chunk(instance)
    await finish(instance, text="迟到的原生答案。")
    await instance.handle_native_event(tool_event())
    assert socket.drain() == []
    assert len(state.calls) == 1 and len(state.results) == 1
    assert not turn.finished and instance.public_history == []
    speech.released.set()
    await instance.response_task
    assert speech.calls == ["真实工具结果。"]
    assert turn.finished and instance.public_history == [("二加二等于几？", "真实工具结果。")]



@pytest.mark.parametrize("prefix", ["我核实一下。", "我先核实关键条件，再给你一个可执行的方案。"])
async def test_native_voice_reply_is_forwarded_before_any_optional_delegation(native, prefix):
    socket, instance, _, state = native
    turn = await utterance(instance, text="请帮我核实上海时间")
    socket.drain()
    await output(instance, text=prefix)
    await audio_start(instance)
    await audio_chunk(instance)
    events = socket.drain()
    assert any(is_event("assistant_delta", text=prefix)(e) for e in events)
    assert b"\x01\x00\x02\x00" in events
    assert state.calls == [] and state.speech == []
    assert instance.agent is None

    # Seeduplex alone decides to delegate. Execution starts while its own
    # first reply is still in the browser's playback queue.
    instance.playback_barriers = True
    state.pause = True
    await instance.handle_native_event(tool_event())
    await state.entered.wait()
    assert state.calls == [("上海现在几点？", True)]
    assert state.speech == [] and state.results == []
    state.released.set()
    barrier = (await socket.until(is_event("playback_barrier")))[-1]
    assert turn.deep_result == "真实工具结果。" and state.results == []
    instance.confirm_playback(barrier)
    await instance.tool_task
    assert len(state.results) == 1 and state.speech == []


@pytest.mark.parametrize("scoped", [True, False])
@pytest.mark.parametrize("existing_agent", [True, False])
async def test_tool_after_native_spoken_segment_done_continues_same_turn_without_duplicate_history(
    native, scoped, existing_agent
):
    socket, instance, _, state = native
    if existing_agent:
        instance._agent()
    turn = await utterance(instance, text="请帮我核实上海时间")
    prefix = "我核实一下。"
    await output(instance, text=prefix)
    await audio_start(instance)
    await audio_chunk(instance)
    await finish(instance, text=prefix)
    assert turn.finished and instance.public_history == [(turn.question, prefix)]
    assert state.remembered == []

    instance.playback_barriers = True
    event = tool_event()
    if not scoped:
        event.pop("question_id")
    await instance.handle_native_event(event)
    barrier = (await socket.until(is_event("playback_barrier")))[-1]
    assert not turn.finished and turn is instance.native_turn
    assert instance.public_history == [] and state.remembered == []
    assert state.calls == [("上海现在几点？", True)] and state.speech == []
    instance.confirm_playback(barrier)
    await instance.tool_task
    assert len(state.results) == 1
    await output(instance, response="r2", text="上海当前时间是工具返回的时间。")
    await audio_start(instance, response="r2")
    await audio_chunk(instance)
    await finish(instance, response="r2", text="上海当前时间是工具返回的时间。")
    assert turn.finished and turn.deep
    assert turn.answer == prefix + "上海当前时间是工具返回的时间。"
    assert instance.public_history == [(turn.question, turn.answer)]
    assert state.remembered == [] and state.speech == []


async def test_completed_delegation_ignores_duplicate_tool_packet_without_reopening(native):
    socket, instance, _, state = native
    turn = await utterance(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await instance.tool_task
    await output(instance, text="核实结果。")
    await audio_start(instance)
    await finish(instance, text="核实结果。")
    assert turn.finished
    socket.drain()
    await instance.handle_native_event(tool_event())
    assert turn.finished and len(state.calls) == 1 and len(state.results) == 1
    assert socket.drain() == []
    assert instance.public_history == [(turn.question, "核实结果。")]


async def test_late_tool_from_previous_completed_question_cannot_delegate_new_question(native):
    _, instance, _, state = native
    old = await utterance(instance)
    await output(instance, text="我核实一下。")
    await audio_start(instance)
    await finish(instance, text="我核实一下。")
    current = await utterance(instance, item="q2", text="先不用了，说个笑话")
    await instance.handle_native_event(tool_event())
    assert instance.native_turn is current and old.finished
    assert not current.deep and state.calls == [] and state.speech == []


async def test_interrupted_completed_prefix_cannot_restart_task(native):
    _, instance, _, state = native
    turn = await utterance(instance)
    await output(instance, text="我核实一下。")
    await audio_start(instance)
    await finish(instance, text="我核实一下。")
    await instance.interrupt()
    await instance.handle_native_event(tool_event())
    assert turn.canceled and state.calls == [] and state.speech == []


async def test_existing_executor_gets_completed_fast_history_once_at_next_user_turn(native):
    _, instance, _, state = native
    instance._agent()
    await utterance(instance)
    await output(instance)
    await audio_start(instance)
    await finish(instance)
    assert state.remembered == []
    await utterance(instance, item="q2", text="请帮我核实上海时间")
    assert state.remembered == [("二加二等于几？", "四。")]
    await audio_start(instance, item="q2", response="r2")
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event(item="q2"))
    await instance.tool_task
    assert state.remembered == [("二加二等于几？", "四。")]


async def test_duplicate_native_tool_packet_before_executor_is_scheduled_runs_only_once(native):
    _, instance, _, state = native
    turn = await utterance(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    first_task = instance.tool_task
    assert turn.pending_tool_calls == {"call-1"} and state.calls == []
    # No yielding to the new executor task between these two receive handlers.
    await instance.handle_native_event(tool_event())
    assert instance.tool_task is first_task
    await first_task
    assert state.calls == [("上海现在几点？", True)] and len(state.results) == 1
    assert not turn.pending_tool_calls and not turn.canceled


async def test_unscoped_tool_after_new_asr_requests_native_reply_before_execution(native):
    _, instance, _, state = native
    await utterance(instance)
    await output(instance, text="我核实一下。")
    await audio_start(instance)
    await finish(instance, text="我核实一下。")
    current = await utterance(instance, item="q2", text="先回答三加四")
    event = tool_event()
    event.pop("question_id")
    await instance.handle_native_event(event)
    assert not current.deep and state.calls == []
    assert instance.tool_task is None and state.speech == []
    assert len(state.results) == 1
    assert json.loads(state.results[0][0]["text"])["error"]["code"] == "native_reply_required"
    await output(instance, item="q2", response="r2", text="七。")
    await audio_start(instance, item="q2", response="r2")
    await finish(instance, item="q2", response="r2", text="七。")
    assert current.finished and current.answer == "七。"


async def test_unscoped_first_tool_packet_requests_native_reply_without_being_dropped(native):
    _, instance, _, state = native
    turn = await utterance(instance)
    event = tool_event()
    event.pop("question_id")
    await instance.handle_native_event(event)
    assert not turn.deep and state.calls == [] and instance.tool_task is None
    assert state.speech == [] and instance.agent is None
    assert len(state.results) == 1 and state.results[0][0]["call_id"] == "call-1"
    assert json.loads(state.results[0][0]["text"])["error"]["code"] == "native_reply_required"


@pytest.mark.parametrize("completion_order", [("text", "audio"), ("audio", "text")])
async def test_out_of_order_tool_survives_empty_completion_then_native_reply_and_retry(
    native, completion_order
):
    socket, instance, _, state = native
    turn = await utterance(instance, text="请帮我核实上海时间")
    socket.drain()
    await instance.handle_native_event(tool_event())
    assert state.calls == [] and state.speech == [] and instance.agent is None
    assert instance.tool_task is None and turn.result_task is not None
    assert not turn.deep and not turn.deep_result and not turn.finished
    assert len(state.results) == 1 and state.results[0][0]["call_id"] == "call-1"
    assert json.loads(state.results[0][0]["text"])["error"]["code"] == "native_reply_required"
    assert turn.reply_request_calls == {"call-1"} and turn.reply_request_attempts == 1
    # Empty completion packets from the rejected tool attempt can precede
    # the model's own reply. They must not close the current user turn.
    for kind in completion_order:
        event = {
            "type": f"response.output_{kind}.done",
            "question_id": "q1",
            "response_id": "r1",
        }
        if kind == "text":
            event["text"] = ""
        await instance.handle_native_event(event)
        assert not turn.finished and turn.awaiting_result
    await instance.handle_native_event({"type": "response.done", "response": {"usage": {}}})
    assert not turn.finished and instance.public_history == []
    assert not any(
        is_event("assistant_delta")(event)
        or is_event("audio_start")(event)
        or is_event("assistant_done")(event)
        or is_event("audio_end")(event)
        or is_event("agent_route", route="deep")(event)
        or isinstance(event, bytes)
        for event in socket.drain()
    )

    # The model supplies its own first reply on the original connection,
    # then retries with a new call ID while that speech is still playing.
    instance.playback_barriers = True
    await output(instance, response="r2", text="我核实一下。")
    await audio_start(instance, response="r2")
    await audio_chunk(instance)
    assert turn.answer == "我核实一下。" and turn.pcm_version == 1
    assert not turn.awaiting_result and not turn.finished
    retry = tool_event()
    retry["items"][0]["call_id"] = "call-2"
    await instance.handle_native_event(retry)
    barrier = (await socket.until(is_event("playback_barrier")))[-1]
    assert state.calls == [("上海现在几点？", True)] and len(state.results) == 1
    assert turn.deep_result == "真实工具结果。" and state.speech == []
    instance.confirm_playback(barrier)
    await instance.tool_task
    assert state.results[1] == [{"call_id": "call-2", "text": "真实工具结果。"}]
    await output(instance, response="r3", text="已核实当前时间。")
    await audio_start(instance, response="r3")
    await audio_chunk(instance)
    await finish(instance, response="r3", text="已核实当前时间。")
    assert turn.finished and not turn.canceled
    assert instance.public_history == [(turn.question, "我核实一下。已核实当前时间。")]
    assert state.speech == []


async def test_audio_start_without_text_or_pcm_still_requests_native_reply(native):
    _, instance, _, state = native
    turn = await utterance(instance)
    await audio_start(instance)
    await instance.handle_native_event(tool_event())
    assert turn.audio_started and not turn.answer and turn.pcm_version == 0
    assert not turn.deep and state.calls == [] and instance.tool_task is None
    assert len(state.results) == 1
    assert json.loads(state.results[0][0]["text"])["error"]["code"] == "native_reply_required"


async def test_duplicate_reply_correction_packet_returns_only_one_result(native):
    _, instance, _, state = native
    turn = await utterance(instance)
    await instance.handle_native_event(tool_event())
    await instance.handle_native_event(tool_event())
    assert len(state.results) == 1 and state.calls == [] and instance.tool_task is None
    assert turn.reply_request_calls == {"call-1"} and turn.reply_request_attempts == 1
    # A rejected call stays rejected even after native speech arrives; only
    # a new call ID may execute the task.
    await output(instance, text="我核实一下。")
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    assert len(state.results) == 1 and state.calls == [] and instance.tool_task is None
    assert turn.reply_request_attempts == 1 and not turn.deep


async def test_native_reply_corrections_are_bounded_before_canceling_turn(native):
    socket, instance, _, state = native
    turn = await utterance(instance)
    socket.drain()
    for call_id in ("call-1", "call-2"):
        event = tool_event()
        event["items"][0]["call_id"] = call_id
        await instance.handle_native_event(event)
    assert len(state.results) == 2 and state.calls == [] and state.cancels == 0
    assert all(
        json.loads(batch[0]["text"])["error"]["code"] == "native_reply_required"
        for batch in state.results
    )
    third = tool_event()
    third["items"][0]["call_id"] = "call-3"
    await instance.handle_native_event(third)
    assert turn.canceled and state.cancels == 1
    assert turn.reply_request_attempts == 2 and len(state.results) == 2
    assert state.calls == [] and state.speech == [] and instance.agent is None
    assert instance.tool_task is None
    events = socket.drain()
    assert any(is_event("error", stage="native_response")(event) for event in events)
    assert any(is_event("state", stage="idle")(event) for event in events)
    await instance.handle_native_event(third)
    assert state.cancels == 1 and len(state.results) == 2


@pytest.mark.parametrize(
    "name, question",
    [("execute_shell", "任意任务"), ("delegate_deep_agent", ""), ("delegate_deep_agent", None)],
)
async def test_invalid_tool_calls_need_no_native_prefix_or_reply_correction(native, name, question):
    _, instance, _, state = native
    turn = await utterance(instance)
    event = tool_event(
        calls=[{
            "call_id": "invalid-call",
            "name": name,
            "arguments": json.dumps({"question": question}),
        }]
    )
    await instance.handle_native_event(event)
    await instance.tool_task
    assert not turn.answer and turn.pcm_version == 0 and not turn.deep
    assert state.calls == [] and state.speech == [] and instance.agent is None
    assert not turn.reply_request_calls and turn.reply_request_attempts == 0
    assert len(state.results) == 1
    assert json.loads(state.results[0][0]["text"]) == {"error": "工具或参数不可用"}

"""Background task execution and safe result announcements in native voice mode."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_realtime import (
    audio_chunk,
    audio_start,
    finish,
    is_event,
    output,
    tool_event,
    utterance,
)
from test_realtime import (
    native as native,
)

pytestmark = pytest.mark.parametrize("native", [True], indirect=True)


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():  # noqa: ASYNC110 - wait for observable session state in tests
            await asyncio.sleep(0.005)


def call(call_id, question):
    return {
        "call_id": call_id,
        "name": "delegate_deep_agent",
        "arguments": json.dumps({"question": question}),
    }


async def dispatch(instance, calls=None, *, item="q1"):
    await instance.handle_native_event(tool_event(item=item, calls=calls))
    await asyncio.wait_for(asyncio.shield(instance.tool_task), 0.5)


async def pending(native, *, calls=None):
    _, instance, _, state = native
    instance.settings.background_notify_delay = 0.1
    state.pause = True
    turn = await utterance(instance, text="帮我查舟山六百左右且免费停车的民宿")
    await output(instance, text="我帮你查一下。")
    await audio_start(instance)
    await audio_chunk(instance)
    await dispatch(instance, calls)
    await asyncio.wait_for(state.entered.wait(), 1)
    return turn, list(instance.background.records.values())


async def manage(native, *, item, call_id="get-result", **arguments):
    _, instance, _, state = native
    await dispatch(instance, [{
        "call_id": call_id, "name": "manage_background_tasks",
        "arguments": json.dumps(arguments),
    }], item=item)
    return json.loads(state.results[-1][0]["text"])


async def test_receipt_is_immediate_and_does_not_wait_for_or_route_the_task_again(
    native, monkeypatch
):
    socket, instance, _, state = native

    async def forbidden_recovery(turn):
        raise AssertionError("A background job must not enter foreground result recovery")

    monkeypatch.setattr(instance, "_recover_result", forbidden_recovery)
    turn, (job,) = await pending(native)
    receipt = json.loads(state.results[0][0]["text"])
    assert receipt["status"] == "task_started"
    assert receipt["task_id"] == job.task_id
    assert job.status == "running" and not job.worker.done()
    assert state.calls == [("上海现在几点？", True)]
    assert instance.agent is None and turn.result_task is None
    assert not turn.awaiting_result and not turn.deep_result
    assert state.speech == []
    assert not any(is_event("task_announcement")(event) for event in socket.drain())


async def test_new_user_question_interrupts_foreground_but_not_background_worker(native):
    _, instance, _, state = native
    original, (job,) = await pending(native)
    current = await utterance(instance, item="q2", text="那你叫什么名字？")
    assert current is not original and original.canceled
    assert job.status == "running" and not job.worker.done()
    assert not state.canceled.is_set()
    await output(instance, item="q2", response="r2", text="我是语音助手。")
    await audio_start(instance, item="q2", response="r2")
    await audio_chunk(instance)
    await finish(instance, item="q2", response="r2", text="我是语音助手。")
    assert current.finished and job.status == "running"
    assert instance.public_history == [("那你叫什么名字？", "我是语音助手。")]
    assert len(state.results) == 1 and len(state.calls) == 1


async def test_multiple_native_calls_have_independent_workers_and_deduplicate_packets(native):
    _, instance, _, state = native
    calls = [call("lodging", "查免费停车的舟山民宿"), call("weather", "查舟山周末天气")]
    _, jobs = await pending(native, calls=calls)
    await eventually(lambda: len(state.calls) == 2)
    assert len(jobs) == 2
    assert jobs[0].task_id != jobs[1].task_id
    assert jobs[0].worker is not jobs[1].worker
    assert jobs[0].agent is not jobs[1].agent
    receipts = [json.loads(result["text"]) for result in state.results[0]]
    assert [result["status"] for result in receipts] == ["task_started", "task_started"]
    assert {result["task_id"] for result in receipts} == {job.task_id for job in jobs}
    await instance.handle_native_event(tool_event(calls=calls))
    assert len(instance.background.records) == 2 and len(state.results) == 1
    state.released.set()
    await asyncio.gather(*(job.worker for job in jobs))
    assert all(job.status == "ready" for job in jobs)
    assert all(job.result == "真实工具结果。" for job in jobs)
    assert len(state.calls) == 2 and state.speech == []


async def test_reasoning_without_actual_tool_use_returns_inline_and_creates_no_background_task(
    native,
):
    socket, instance, _, state = native
    state.uses_tools = False
    turn = await utterance(instance, text="解释一下这两个方案的取舍")
    await output(instance, text="我来比较下关键取舍。")
    await audio_start(instance)
    await audio_chunk(instance)
    await dispatch(instance, [call("reasoning-only", "解释一下这两个方案的取舍")])
    result = json.loads(state.results[-1][0]["text"])
    assert result["status"] == "completed" and result["result"] == "真实工具结果。"
    assert instance.background.records == {} and instance.background.active_tasks() == []
    assert turn.awaiting_result and turn.deep
    assert state.speech == [] and state.planner_calls == []
    assert not any(is_event("task_update")(event) for event in socket.drain())
    await output(instance, response="r1-result", text="方案一更省时，方案二更灵活。")
    await audio_start(instance, response="r1-result")
    await audio_chunk(instance)
    await finish(instance, response="r1-result", text="方案一更省时，方案二更灵活。")
    assert turn.finished and instance.background.records == {}


@pytest.mark.parametrize("input_mode", ["voice", "keyboard"])
async def test_ready_results_are_not_shared_with_voice_or_later_delegations(native, input_mode):
    _, instance, _, state = native
    weather = "舟山周末有雨，沿海风力较大；来源：气象台，11月10日。"
    state.agent_results["查舟山周末天气"] = weather
    _, (job,) = await pending(native, calls=[call("weather", "查舟山周末天气")])
    hints_before = list(state.hints)
    state.released.set()
    await job.worker
    assert job.status == "ready" and job.result == weather
    assert state.hints == hints_before
    assert weather not in instance._native_instructions()
    assert job.task_id not in instance._native_instructions()
    assert all(weather not in answer for _, answer in state.history + instance.public_history)
    assert not state.speech and not state.planner_calls
    question = "从上海去舟山自驾两天，帮我规划一下"
    state.uses_tools = False
    state.pause = False
    if input_mode == "voice":
        await utterance(instance, item="q2", text=question)
        await output(instance, item="q2", response="r2", text="我来安排两天行程。")
        await audio_start(instance, item="q2", response="r2")
        await audio_chunk(instance)
        await dispatch(instance, [call("itinerary", question)], item="q2")
        result = json.loads(state.results[-1][0]["text"])
        assert result == {"status": "completed", "result": "真实工具结果。", "error": ""}
    else:
        await instance.respond(question)
    if input_mode == "voice":
        assert json.loads(state.contexts[-1])["original_user_request"] == question
    else:
        assert state.contexts[-1] == ""
    assert weather not in state.contexts[-1]
    assert all(weather not in answer for _, answer in state.remembered)
    assert len(instance.background.records) == 1
    assert job.status == "ready" and job.task_id not in instance.announced_tasks


async def test_connecting_with_ready_results_does_not_include_them_in_native_instructions(native):
    socket, previous, _, state = native
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    instance = type(previous)(socket, previous.settings)
    instance.background.records[job.task_id] = job
    instance.user_speaking = True
    try:
        await instance._connect_native()
        assert job.result not in state.instructions and job.task_id not in state.instructions
        assert instance.background.records[job.task_id] is job
        assert job.status == "ready" and not state.speech
    finally:
        await instance.close_native()


async def test_new_user_question_cancels_reasoning_that_has_not_started_a_tool(native):
    socket, instance, _, state = native
    state.uses_tools = False
    state.pause = True
    previous = await utterance(instance, text="仔细比较一下两个方案")
    await output(instance, text="我来比较下关键取舍。")
    await instance.handle_native_event(tool_event(calls=[call("reasoning", "比较两个方案")]))
    worker = instance.tool_task
    await asyncio.wait_for(state.entered.wait(), 1)
    assert instance.background.records == {} and not worker.done()
    current = await utterance(instance, item="q2", text="先不比较了，你叫什么名字")
    assert previous.canceled and current is instance.native_turn
    assert worker.done() and state.canceled.is_set()
    assert instance.background.records == {} and not state.results
    assert not any(is_event("task_update")(event) for event in socket.drain())


async def test_delivery_planner_can_defer_until_a_later_conversation_turn(native):
    _, instance, _, state = native
    _, (job,) = await pending(native)
    state.delivery_plans.append(SimpleNamespace(task_ids=(), text=""))
    state.released.set()
    await job.worker
    await finish(instance, text="我帮你查一下。")
    await asyncio.wait_for(state.planner_entered.wait(), 1)
    await asyncio.sleep(0.25)
    assert len(state.planner_calls) == 1 and not state.speech
    assert job.status == "ready" and job.task_id not in instance.announced_tasks
    await utterance(instance, item="q2", text="好的，那先说说你的名字")
    await output(instance, item="q2", response="r2", text="我是语音助手。")
    await finish(instance, item="q2", response="r2", text="我是语音助手。")
    await eventually(lambda: job.status == "completed")
    assert len(state.planner_calls) == 2 and state.speech == [job.result]
    assert state.planner_calls[-1][2] == "好的，那先说说你的名字"


async def test_new_question_during_delivery_planning_discards_the_stale_speech_plan(native):
    _, instance, _, state = native
    _, (job,) = await pending(native)
    state.planner_pause = True
    state.released.set()
    await job.worker
    await finish(instance, text="我帮你查一下。")
    await asyncio.wait_for(state.planner_entered.wait(), 1)
    current = await utterance(instance, item="q2", text="先等等，我现在有个别的问题")
    state.planner_released.set()
    await asyncio.sleep(0.22)
    assert instance.native_turn is current and not current.canceled
    assert job.status == "ready" and job.task_id not in instance.announced_tasks
    assert not state.speech and job.announcement_attempts == 0
    await output(instance, item="q2", response="r2", text="你说，我在听。")
    await finish(instance, item="q2", response="r2", text="你说，我在听。")
    await eventually(lambda: job.status == "completed")
    assert len(state.speech) == 1


async def test_delivery_can_merge_selected_tasks_and_only_consume_the_selected_results(native):
    socket, instance, _, state = native
    _, jobs = await pending(native, calls=[
        call("weather", "查舟山天气"), call("hotel", "查舟山民宿"),
        call("road", "查舟山自驾路况"),
    ])
    await eventually(lambda: len(state.calls) == 3)
    selected = (jobs[0].task_id, jobs[2].task_id)
    text = "周末舟山有雨，自驾走跨海大桥时注意横风，行程先安排室内活动。"
    state.delivery_plans.extend([
        SimpleNamespace(task_ids=selected, text=text),
        SimpleNamespace(task_ids=(), text=""),
    ])
    state.released.set()
    await asyncio.gather(*(job.worker for job in jobs))
    await finish(instance, text="我帮你查一下。")
    events = await socket.until(is_event("task_announcement"))
    assert events[-1]["task_ids"] == list(selected)
    await eventually(lambda: all(jobs[index].status == "completed" for index in (0, 2)))
    await asyncio.sleep(0.22)
    assert jobs[1].status == "ready" and jobs[1].task_id not in instance.announced_tasks
    assert instance.announced_tasks == set(selected)
    assert state.speech == [text]


async def test_ready_result_waits_for_foreground_completion_and_user_speech(native):
    socket, instance, _, state = native
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    assert job.status == "ready"
    await asyncio.sleep(0.22)
    assert state.speech == []  # The acknowledgement segment is still unfinished.
    await finish(instance, text="我帮你查一下。")
    instance.user_speaking = True
    await asyncio.sleep(0.22)
    assert state.speech == [] and job.status == "ready"
    assert not any(is_event("task_announcement")(event) for event in socket.drain())
    instance.user_speaking = False
    instance.background.ready.set()
    await eventually(lambda: job.status == "completed")
    assert state.speech == ["真实工具结果。"]


async def test_notification_waits_for_prefix_and_result_playback_barriers(native):
    socket, instance, _, state = native
    instance.playback_barriers = True
    prefix, (job,) = await pending(native)
    state.released.set()
    await job.worker
    await finish(instance, text="我帮你查一下。")
    events = await socket.until(is_event("playback_barrier", turn_id=prefix.turn_id))
    prefix_barrier = events[-1]
    assert prefix.finished and job.status == "ready" and not state.speech
    assert not any(is_event("task_announcement")(event) for event in events)
    instance.confirm_playback({**prefix_barrier, "barrier_id": "wrong-barrier"})
    await asyncio.sleep(0.03)
    assert job.status == "ready" and not state.speech
    instance.confirm_playback(prefix_barrier)
    events = await socket.until(is_event("task_announcement", task_id=job.task_id))
    announcement = events[-1]
    assert announcement["turn_id"] != prefix.turn_id
    events = await socket.until(
        is_event("playback_barrier", turn_id=announcement["turn_id"])
    )
    result_barrier = events[-1]
    assert job.status == "announcing"
    assert job.task_id not in instance.announced_tasks
    assert len(state.calls) == 1 and len(state.results) == 1
    instance.confirm_playback(result_barrier)
    await eventually(lambda: job.status == "completed")
    assert job.task_id in instance.announced_tasks
    assert len(state.results) == 1  # A tool receipt is never replaced by a late result batch.
    assert len(instance.public_history) == 2
    assert instance.public_history[-1] == (job.question, job.result)
    memory = await instance._manage_background({"action": "history"})
    assert memory["exchanges"][-1] == {"user": job.question, "assistant": job.result}
    assert state.history == state.answer_history == []


@pytest.mark.parametrize("partial", [True, False])
async def test_confirmed_barge_in_stops_upstream_output_but_preserves_work(native, partial):
    _, instance, _, state = native
    _, (job,) = await pending(native)
    await instance.handle_native_event({
        "type": "conversation.item.input_audio_transcription.started", "item_id": "q2",
    })
    if partial:
        await instance.handle_native_event({
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": "q2", "delta": "那你叫什么名字",
        })
    await instance.handle_native_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "q2", "text": "那你叫什么名字？",
    })
    # When the first event already completes ASR, upstream may have begun the
    # new answer; an unscoped cancel must never kill that newer response.
    assert state.cancels == (1 if partial else 0)
    assert job.status == "running" and not state.canceled.is_set()
    assert instance.native_turn.question == "那你叫什么名字？"
    state.released.set()
    await job.worker
    assert job.status == "ready" and not job.worker.cancelled()


@pytest.mark.parametrize("action", ["barge_in", "stop_answer"])
async def test_user_interrupt_consumes_started_notification_only_and_never_replays_it(
    native, action
):
    _, instance, _, state = native
    speech_started, speech_canceled = asyncio.Event(), asyncio.Event()
    blocked = asyncio.Event()
    attempts = 0

    async def speech(text):
        nonlocal attempts
        attempts += 1
        state.speech.append(text)
        yield b"\x03\x00\x04\x00"
        if attempts == 1:
            speech_started.set()
            try:
                await blocked.wait()
            except asyncio.CancelledError:
                speech_canceled.set()
                raise

    instance.native_client.speech = speech
    _, (job, other) = await pending(
        native, calls=[call("one", "查民宿"), call("two", "查天气")]
    )
    await eventually(lambda: len(state.calls) == 2)
    state.released.set()
    await asyncio.gather(job.worker, other.worker)
    await finish(instance, text="我帮你查一下。")
    await asyncio.wait_for(speech_started.wait(), 1)
    assert job.status == "announcing"
    if action == "stop_answer":
        await instance.interrupt()
    await utterance(instance, item="q2", text="先说说你叫什么名字？")
    assert speech_canceled.is_set()
    assert job.status == "completed" and job.result == "真实工具结果。"
    assert job.worker.done() and not job.worker.cancelled()
    assert job.task_id in instance.announced_tasks
    assert other.status == "ready" and other.task_id not in instance.announced_tasks
    assert len(state.calls) == 2 and len(state.results) == 1
    await asyncio.sleep(0.22)
    assert attempts == 1  # A result does not interrupt the new foreground question.
    await output(instance, item="q2", response="r2", text="我是语音助手。")
    await finish(instance, item="q2", response="r2", text="我是语音助手。")
    await eventually(lambda: other.status == "completed")
    assert attempts == 2 and len(state.calls) == 2
    assert job.announcement_attempts == other.announcement_attempts == 1
    assert all("有结果了" not in text for text in state.speech)


async def test_user_barge_in_before_notification_pcm_preserves_result_for_later_announcement(
    native,
):
    _, instance, _, state = native
    synthesis_started, blocked = asyncio.Event(), asyncio.Event()
    attempts = 0

    async def speech(text):
        nonlocal attempts
        attempts += 1
        state.speech.append(text)
        if attempts == 1:
            synthesis_started.set()
            await blocked.wait()
        yield b"\x03\x00\x04\x00"

    instance.native_client.speech = speech
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    await finish(instance, text="我帮你查一下。")
    await asyncio.wait_for(synthesis_started.wait(), 1)
    assert instance.native_turn.pcm_version == 0
    await utterance(instance, item="q2", text="先说一下你的名字")
    assert job.status == "ready" and job.task_id not in instance.announced_tasks
    assert job.result == "真实工具结果。" and len(state.calls) == 1
    await output(instance, item="q2", response="r2", text="我是语音助手。")
    await finish(instance, item="q2", response="r2", text="我是语音助手。")
    await eventually(lambda: job.status == "completed")
    assert attempts == 2 and len(state.calls) == 1


@pytest.mark.parametrize("after_pcm", [False, True])
async def test_technical_synthesis_failure_preserves_ready_result_even_after_pcm(native, after_pcm):
    _, instance, _, state = native

    async def speech(text):
        state.speech.append(text)
        if after_pcm:
            yield b"\x03\x00\x04\x00"
        raise RuntimeError("temporary synthesis failure")

    instance.native_client.speech = speech
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    await finish(instance, text="我帮你查一下。")
    await eventually(lambda: job.announcement_attempts == 1 and instance.notification_job is None)
    assert job.status == "ready" and job.task_id not in instance.announced_tasks
    assert job.result == "真实工具结果。" and not instance.native_turn.user_interrupted
    await asyncio.sleep(0.22)
    assert len(state.speech) == 1 and len(state.calls) == 1


async def test_explicit_interrupt_stops_audio_without_canceling_independent_tasks(native):
    _, instance, _, state = native
    _, (job,) = await pending(native)
    await instance.interrupt()
    assert job.status == "running" and not job.worker.done()
    assert not state.canceled.is_set()
    state.released.set()
    await job.worker
    assert job.result == "真实工具结果。"


async def test_native_manage_tool_lists_and_cancels_selected_tasks(native):
    _, instance, _, state = native
    _, jobs = await pending(native, calls=[call("one", "查民宿"), call("two", "查天气")])
    await eventually(lambda: len(state.calls) == 2)
    await utterance(instance, item="q2", text="查一下后台任务，然后取消最新任务")

    async def manage(call_id, **arguments):
        await dispatch(instance, [{
            "call_id": call_id, "name": "manage_background_tasks",
            "arguments": json.dumps(arguments),
        }], item="q2")
        return json.loads(state.results[-1][0]["text"])

    listed = await manage("list", action="list")
    assert {task["id"] for task in listed["tasks"]} == {job.task_id for job in jobs}
    assert all(task["status"] == "running" for task in listed["tasks"])
    invalid = await manage("invalid", action="cancel", selector="all", task_ids=[jobs[0].task_id])
    assert "error" in invalid and all(job.status == "running" for job in jobs)
    canceled = await manage("latest", action="cancel", selector="latest")
    assert canceled == {"canceled_task_ids": [jobs[1].task_id]}
    assert jobs[1].status == "canceled" and jobs[1].worker.done()
    assert jobs[0].status == "running" and not jobs[0].worker.done()
    unknown = await manage("unknown", action="cancel", task_ids=["missing-task"])
    assert unknown["tasks"] == [] and jobs[0].status == "running"
    canceled = await manage("all", action="cancel", selector="all")
    assert canceled == {"canceled_task_ids": [jobs[0].task_id]}
    assert all(job.status == "canceled" and job.worker.done() for job in jobs)


async def test_canceling_a_notification_keeps_other_tasks_and_new_foreground_alive(native):
    socket, instance, _, state = native
    instance.playback_barriers = True
    original, jobs = await pending(
        native, calls=[call("one", "查民宿"), call("two", "查天气")]
    )
    await eventually(lambda: len(state.calls) == 2)
    state.released.set()
    await asyncio.gather(*(job.worker for job in jobs))
    await finish(instance, text="我帮你查一下。")
    barrier = (await socket.until(is_event("playback_barrier", turn_id=original.turn_id)))[-1]
    instance.confirm_playback(barrier)
    await socket.until(is_event("task_announcement", task_id=jobs[0].task_id))
    await socket.until(is_event("playback_barrier"))
    assert jobs[0].status == "announcing"
    assert await instance._cancel_background(jobs[0].task_id)
    assert jobs[0].status == "canceled" and jobs[1].status == "ready"
    assert jobs[0].task_id not in instance.announced_tasks
    current = await utterance(instance, item="q2", text="现在先聊别的")
    assert not current.canceled
    assert jobs[1].status == "ready" and len(state.calls) == 2


async def test_close_cancels_background_workers_and_notification_scheduler(native):
    _, instance, _, state = native
    _, jobs = await pending(native, calls=[call("one", "查民宿"), call("two", "查天气")])
    await eventually(lambda: len(state.calls) == 2)
    scheduler = instance.background_loop
    assert scheduler is not None and not scheduler.done()
    await instance.close_native()
    assert scheduler.done() and instance.background_loop is None
    assert instance.notification_task is None and instance.background.records == {}
    assert all(job.worker.done() and job.status == "canceled" for job in jobs)
    assert state.canceled.is_set()
    await instance.close_native()  # Closing an already closed voice session is safe.


async def test_barge_in_during_announcing_state_publish_cannot_strand_a_task(native):
    _, instance, _, state = native
    publishing, blocked = asyncio.Event(), asyncio.Event()
    publish = instance.background.publish

    async def slow_publish(job):
        await publish(job)
        if job.status == "announcing":
            publishing.set()
            await blocked.wait()

    instance.background.publish = slow_publish
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    await finish(instance, text="我帮你查一下。")
    await asyncio.wait_for(publishing.wait(), 1)
    await utterance(instance, item="q2", text="先告诉我你叫什么名字")
    assert job.status == "ready" and job.result == "真实工具结果。"
    assert instance.notification_job is None
    assert job.worker.done() and not job.worker.cancelled()


async def test_heard_notification_memory_does_not_depend_on_provider_history_injection(native):
    socket, instance, stream, state = native

    async def failing_history(text):
        raise AssertionError("Public memory must be read through the native tool")

    stream.add_assistant_answer = failing_history
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    await finish(instance, text="我帮你查一下。")
    await socket.until(is_event("task_announcement", task_id=job.task_id))
    await eventually(lambda: instance.notification_job is None)
    assert len(state.speech) == 1 and job.task_id in instance.announced_tasks
    assert job.status == "completed"
    assert any("真实工具结果" in answer for _, answer in instance.public_history)
    memory = await instance._manage_background({"action": "history"})
    assert memory["exchanges"][-1] == {"user": job.question, "assistant": job.result}
    assert state.history == state.answer_history == []


async def test_requested_ready_result_is_consumed_only_after_main_voice_playback(native):
    socket, instance, _, state = native
    instance.playback_barriers = True
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    current = await utterance(instance, item="q2", text="刚才查的民宿有结果了吗？")
    result = await manage(native, item="q2", action="get", selector="latest")
    assert result["tasks"][0]["result"] == "真实工具结果。"
    assert current.background_results == {job.task_id}
    await output(instance, item="q2", response="r2", text="查到了，真实工具结果。")
    await audio_start(instance, item="q2", response="r2")
    await audio_chunk(instance)
    await finish(instance, item="q2", response="r2", text="查到了，真实工具结果。")
    events = await socket.until(is_event("playback_barrier", turn_id=current.turn_id))
    assert job.status == "ready" and job.task_id not in instance.announced_tasks
    assert state.speech == []
    instance.confirm_playback(events[-1])
    await eventually(lambda: job.status == "completed")
    assert job.task_id in instance.announced_tasks and current.background_results == set()
    assert job.announcement_attempts == 0 and state.speech == []
    assert not any(is_event("task_announcement")(event) for event in socket.drain())


@pytest.mark.parametrize("prefix_pcm", [False, True])
async def test_get_result_with_finished_text_but_no_result_pcm_stays_ready(native, prefix_pcm):
    _, instance, _, state = native
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    asked = await utterance(instance, item="q2", text="上一个任务结果呢？")
    if prefix_pcm:
        await output(instance, item="q2", response="r2-prefix", text="我看一下。")
        await audio_start(instance, item="q2", response="r2-prefix")
        await audio_chunk(instance)
    await manage(native, item="q2", action="get", selector="latest")
    state.delivery_plans.append(SimpleNamespace(task_ids=(), text=""))
    await output(instance, item="q2", response="r2-result", text="真实工具结果。")
    await audio_start(instance, item="q2", response="r2-result")
    # Upstream can complete a text/audio segment without delivering PCM.
    await finish(instance, item="q2", response="r2-result", text="真实工具结果。")
    await asyncio.wait_for(state.planner_entered.wait(), 1)
    assert asked.finished and asked.pcm_version == (1 if prefix_pcm else 0)
    assert job.status == "ready" and job.task_id not in instance.announced_tasks
    assert not state.speech and job.announcement_attempts == 0


async def test_two_inline_delegations_keep_both_results_for_native_speech_recovery(native):
    _, instance, _, state = native
    state.uses_tools = False
    state.agent_results.update({"分析方案A": "方案A更省时。", "分析方案B": "方案B更灵活。"})
    turn = await utterance(instance, text="分别分析两个方案")
    await output(instance, text="我来比较两个方案。")
    await audio_start(instance)
    await audio_chunk(instance)
    await dispatch(instance, [call("inline-a", "分析方案A"), call("inline-b", "分析方案B")])
    results = [json.loads(result["text"]) for result in state.results[-1]]
    assert [result["result"] for result in results] == ["方案A更省时。", "方案B更灵活。"]
    assert "方案A更省时。" in turn.deep_result and "方案B更灵活。" in turn.deep_result
    await instance._recover_result(turn)
    await instance.response_task
    assert len(state.speech) == 1
    assert "方案A更省时。" in state.speech[0] and "方案B更灵活。" in state.speech[0]
    assert instance.background.records == {} and len(state.calls) == 2


@pytest.mark.parametrize("action", ["barge_in", "stop_answer"])
async def test_requested_result_interrupted_after_result_pcm_is_not_announced_again(native, action):
    socket, instance, _, state = native
    instance.playback_barriers = True
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    asked = await utterance(instance, item="q2", text="刚才的任务结果呢？")
    await manage(native, item="q2", action="get", selector="latest")
    await output(instance, item="q2", response="r2", text="真实工具结果。")
    await audio_start(instance, item="q2", response="r2")
    await audio_chunk(instance)
    await finish(instance, item="q2", response="r2", text="真实工具结果。")
    await socket.until(is_event("playback_barrier", turn_id=asked.turn_id))
    instance.duplex.playback_active = True
    if action == "stop_answer":
        await instance.interrupt()
    await utterance(instance, item="q3", text="等一下，先说你叫什么名字？")
    instance.duplex.playback_active = False  # Browser discarded the interrupted PCM queue.
    assert asked.canceled and job.status == "completed"
    assert job.task_id in instance.announced_tasks
    await output(instance, item="q3", response="r3", text="我是语音助手。")
    await finish(instance, item="q3", response="r3", text="我是语音助手。")
    await asyncio.sleep(0.22)
    assert job.status == "completed" and job.announcement_attempts == 0
    assert state.speech == [] and len(state.calls) == 1
    assert not any(is_event("task_announcement")(event) for event in socket.drain())


@pytest.mark.parametrize("result_text", [False, True])
async def test_requested_result_without_new_result_pcm_stays_ready_on_barge_in(native, result_text):
    socket, instance, _, state = native
    instance.playback_barriers = True
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    asked = await utterance(instance, item="q2", text="帮我看一下上一个任务的结果")
    await output(instance, item="q2", response="r2", text="我看一下。")
    await audio_start(instance, item="q2", response="r2")
    await audio_chunk(instance)
    await manage(native, item="q2", action="get", selector="latest")
    assert asked.background_result_pcm_version == asked.pcm_version == 1
    if result_text:
        await output(instance, item="q2", response="r2-result", text="真实工具结果。")
    await utterance(instance, item="q3", text="先说说你叫什么名字？")
    assert asked.canceled and job.status == "ready"
    assert job.task_id not in instance.announced_tasks
    await output(instance, item="q3", response="r3", text="我是语音助手。")
    await finish(instance, item="q3", response="r3", text="我是语音助手。")
    await socket.until(is_event("task_announcement", task_id=job.task_id))
    barrier = (await socket.until(is_event("playback_barrier")))[-1]
    instance.confirm_playback(barrier)
    await eventually(lambda: job.status == "completed")
    assert len(state.speech) == 1 and len(state.calls) == 1


@pytest.mark.parametrize("action", ["barge_in", "stop_answer"])
async def test_late_prefix_pcm_after_get_cannot_consume_a_result_that_has_not_started(
    native, action
):
    _, instance, _, state = native
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    asked = await utterance(instance, item="q2", text="上一个任务的结果呢？")
    await output(instance, item="q2", response="r2-prefix", text="我看一下。")
    await audio_start(instance, item="q2", response="r2-prefix")
    assert asked.pcm_version == 0
    await manage(native, item="q2", action="get", selector="latest")
    await audio_chunk(instance)  # Acknowledgement PCM arrives late, after the tool result.
    await output(instance, item="q2", response="r2-result", text="真实工具结果。")
    assert not asked.awaiting_result
    assert asked.pcm_version == asked.background_result_pcm_version == 1
    assert asked.audio_response_id == "r2-prefix"
    assert asked.background_result_response_id == "r2-result"
    if action == "stop_answer":
        await instance.interrupt()
    await utterance(instance, item="q3", text="先说说你的名字吧")
    assert asked.canceled and job.status == "ready"
    assert job.task_id not in instance.announced_tasks
    assert job.result == "真实工具结果。"
    await output(instance, item="q3", response="r3", text="我是语音助手。")
    await finish(instance, item="q3", response="r3", text="我是语音助手。")
    await eventually(lambda: job.status == "completed")
    assert job.announcement_attempts == 1 and len(state.speech) == 1
    assert len(state.calls) == 1


@pytest.mark.parametrize("action", ["barge_in", "stop_answer"])
async def test_result_pcm_before_result_text_is_consumed_using_its_response_id(native, action):
    socket, instance, _, state = native
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    asked = await utterance(instance, item="q2", text="上一个任务的结果是什么？")
    await output(instance, item="q2", response="r2-prefix", text="我看一下。")
    await audio_start(instance, item="q2", response="r2-prefix")
    await manage(native, item="q2", action="get", selector="latest")
    await audio_chunk(instance)  # Ignore this late prefix when the next audio segment begins.
    await audio_start(instance, item="q2", response="r2-result")
    await audio_chunk(instance)  # The result audio can precede its text delta.
    await output(instance, item="q2", response="r2-result", text="真实工具结果。")
    assert asked.audio_response_id == asked.background_result_response_id == "r2-result"
    assert asked.pcm_version == 2 and asked.background_result_pcm_version == 1
    if action == "stop_answer":
        await instance.interrupt()
    await utterance(instance, item="q3", text="先说说你的名字吧")
    assert asked.canceled and job.status == "completed"
    assert job.task_id in instance.announced_tasks
    await output(instance, item="q3", response="r3", text="我是语音助手。")
    await finish(instance, item="q3", response="r3", text="我是语音助手。")
    await asyncio.sleep(0.22)
    assert job.announcement_attempts == 0 and state.speech == []
    assert not any(is_event("task_announcement")(event) for event in socket.drain())
    assert len(state.calls) == 1


async def test_get_latest_can_retrieve_an_already_completed_task_without_resubmitting(native):
    _, instance, _, state = native
    _, (job,) = await pending(native)
    state.released.set()
    await job.worker
    await finish(instance, text="我帮你查一下。")
    await eventually(lambda: job.status == "completed")
    current = await utterance(instance, item="q2", text="再说一下刚才那个结果")
    result = await manage(native, item="q2", action="get", selector="latest")
    assert result["tasks"] == [{
        **instance._task_info(job, include_trace=False), "result": "真实工具结果。",
    }]
    assert current.background_results == set()
    assert len(state.calls) == 1 and len(instance.background.records) == 1


async def test_browser_trace_updates_include_results_without_resending_native_context(native):
    socket, instance, _, state = native
    _, (job,) = await pending(native)
    socket.drain()
    hints_before = len(state.hints)
    await job.agent.on_tool_event({
        "id": "search-run", "name": "global_search", "status": "running",
        "input": {"query": "舟山天气", "api_key": "secret-api-key"},
    })
    await job.agent.on_tool_event({
        "id": "search-run", "status": "completed", "output": {"summary": "明天有雨"},
    })
    assert len(state.hints) == hints_before
    events = [event["task"] for event in socket.drain() if is_event("task_update")(event)]
    assert len(events) == 2
    assert events[0]["calls"][0]["status"] == "running"
    assert events[1]["calls"][0]["status"] == "completed"
    assert events[1]["question"] == job.question and events[1]["result"] == ""
    assert events[1]["created_at"] <= events[1]["started_at"] <= events[1]["updated_at"]
    assert events[1]["finished_at"] is None
    assert "secret-api-key" not in json.dumps(events)
    state.agent_results[job.question] = "完整结果" * 2000
    state.released.set()
    await job.worker
    events = [event["task"] for event in socket.drain() if is_event("task_update")(event)]
    result = next(event for event in events if event["status"] == "ready")
    assert result["result"] == job.result == "完整结果" * 2000
    assert result["started_at"] <= result["finished_at"] <= result["updated_at"]
    assert len(state.hints) == hints_before


async def test_keyboard_task_connects_native_without_enabling_microphone(native):
    socket, previous, _, state = native
    await previous.close_native()
    socket.drain()
    instance = type(previous)(socket, previous.settings)
    instance.settings.background_notify_delay = 0.1
    state.pause = True
    opened = state.opened
    assert instance.native is None and not instance.listening
    session = asyncio.create_task(instance.run())
    try:
        await socket.control({"type": "text", "text": "帮我查免费停车的民宿", "turn_id": "typed"})
        events = await socket.until(is_event("assistant_done", turn_id="typed"))
        await eventually(lambda: instance.response_task.done())
        assert instance.response_task.exception() is None
        assert state.opened == opened + 1 and instance.native is not None
        assert not instance.listening and state.muted[-1] is True
        assert not any(is_event("listening", enabled=True)(event) for event in events)
        job = next(iter(instance.background.records.values()))
        await asyncio.wait_for(state.entered.wait(), 1)
        state.released.set()
        await job.worker
        await socket.until(is_event("task_announcement", task_id=job.task_id))
        await eventually(lambda: job.status == "completed")
        assert not instance.listening and len(state.speech) == 1
    finally:
        await socket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(session, 1)


async def test_keyboard_capacity_error_finishes_response_and_preserves_existing_task(native):
    socket, instance, _, state = native
    _, (job,) = await pending(native)
    instance.background.max_active = 1
    socket.drain()
    session = asyncio.create_task(instance.run())
    try:
        await socket.control({"type": "text", "text": "还要查天气", "turn_id": "over-capacity"})
        await socket.until(is_event("error", stage="agent", turn_id="over-capacity"))
        await eventually(lambda: instance.response_task.done())
        assert instance.response_task.exception() is None
        assert instance.native_turn.finished and not job.worker.done()
        assert job.status == "running" and len(instance.background.records) == 1
        assert len(state.calls) == 2  # Capacity is checked when the real tool starts.
        await socket.control({"type": "ping"})
        await socket.until(is_event("pong"))
        assert not session.done()
    finally:
        await socket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(session, 1)


@pytest.mark.parametrize("capacity_full", [False, True])
async def test_keyboard_reasoning_answers_inline_even_when_background_capacity_is_full(
    native, capacity_full
):
    socket, instance, _, state = native
    job = None
    if capacity_full:
        _, (job,) = await pending(native)
        instance.background.max_active = 1
    state.uses_tools = False
    state.pause = False
    socket.drain()
    session = asyncio.create_task(instance.run())
    try:
        await socket.control({
            "type": "text", "text": "比较一下两个方案的取舍", "turn_id": "typed-reasoning",
        })
        events = await socket.until(is_event("assistant_done", turn_id="typed-reasoning"))
        await eventually(lambda: instance.response_task.done())
        assert instance.response_task.exception() is None
        assert events[-1]["text"] == "真实工具结果。"
        assert state.speech == ["真实工具结果。"]
        assert state.planner_calls == []
        assert not any(is_event("task_update")(event) for event in events)
        assert not any(is_event("task_announcement")(event) for event in events)
        assert len(instance.background.records) == (1 if capacity_full else 0)
        if job is not None:
            assert job.status == "running" and not job.worker.done()
        assert instance.public_history[-1] == ("比较一下两个方案的取舍", "真实工具结果。")
    finally:
        await socket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(session, 1)

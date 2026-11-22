"""Public speech memory is read through native tools, never through unspoken results."""

import asyncio
import json

import pytest
from test_realtime import audio_chunk, audio_start, finish, output, tool_event, utterance
from test_realtime import native as native
from test_realtime_background import call, dispatch, eventually, manage

from voice_agent.background import BackgroundTask

pytestmark = pytest.mark.parametrize("native", [True], indirect=True)

QUESTION = "比较北辰R与青鹭S两种方案，预算4000到6000，给出已有证据支持的结论"
ANSWER = "预算4000到6000时，北辰R维护省心，青鹭S更适合空间受限的场景。"
PREFIX = "我来比较已有条件。"
PCM = b"\x03\x00\x04\x00"


async def ready_result(native):
    _, instance, _, state = native
    instance.settings.background_notify_delay = 0.1
    state.pause = True
    state.agent_results[QUESTION] = ANSWER
    prefix = await utterance(instance, text=QUESTION)
    await output(instance, text=PREFIX)
    await audio_start(instance)
    await audio_chunk(instance)
    await dispatch(instance, [call("compare-public-options", QUESTION)])
    job = next(iter(instance.background.records.values()))
    state.released.set()
    await job.worker
    await finish(instance, text=PREFIX)
    return prefix, job


def matching_history(instance):
    return [pair for pair in instance.public_history if ANSWER in pair[1]]


def forbid_user_history(monkeypatch, stream):
    async def forbidden(exchanges):
        raise AssertionError("A result must not invent another native user question")

    monkeypatch.setattr(stream, "add_history", forbidden)


@pytest.mark.parametrize("action", ["barge_in", "stop_answer"])
async def test_interrupted_public_result_preserves_options_for_followup_delegation(
    native, monkeypatch, action,
):
    _, instance, stream, state = native
    first_pcm_sent, blocked = asyncio.Event(), asyncio.Event()

    async def speech(text):
        state.speech.append(text)
        yield PCM
        first_pcm_sent.set()
        await blocked.wait()

    monkeypatch.setattr(instance.native_client, "speech", speech)
    forbid_user_history(monkeypatch, stream)
    _, job = await ready_result(native)
    await asyncio.wait_for(first_pcm_sent.wait(), 1)
    announcing = instance.native_turn

    if action == "stop_answer":
        await instance.interrupt()
    await utterance(instance, item="followup", text="那你继续说，具体比较这两个方案")

    assert announcing.canceled and job.status == "completed"
    assert job.task_id in instance.announced_tasks
    assert announcing.public_delivery_text == ANSWER
    history = matching_history(instance)
    assert len(history) == 1
    assert history[0][1] != ANSWER  # Its partial-public status must remain explicit.
    assert state.answer_history == state.history == []

    memory = await manage(native, item="followup", action="history", call_id="read-interrupted")
    assert memory["status"] == "public_history"
    assert [item["assistant"] for item in memory["exchanges"] if ANSWER in item["assistant"]] == [
        history[0][1]
    ]

    state.pause = False
    state.uses_tools = False
    await output(instance, item="followup", response="r-followup", text="我接着比较。")
    await audio_start(instance, item="followup", response="r-followup")
    await audio_chunk(instance)
    await dispatch(instance, [call("deepen-public-options", "继续比较北辰R和青鹭S")],
                   item="followup")

    assert any(ANSWER in answer for _, answer in state.remembered)
    assert json.loads(state.contexts[-1]) == {
        "original_user_request": "那你继续说，具体比较这两个方案",
        "source": "final_user_input", "origin_turn_id": instance.native_turn.turn_id,
    }  # Only final user input; task results stay in the public ledger.
    assert ANSWER not in state.contexts[-1]
    assert len(matching_history(instance)) == 1
    assert not state.answer_history and state.speech == [ANSWER]


async def test_interruption_before_first_result_pcm_does_not_share_the_prepared_result(
    native, monkeypatch,
):
    _, instance, stream, state = native
    synthesis_entered, blocked = asyncio.Event(), asyncio.Event()

    async def speech(text):
        state.speech.append(text)
        synthesis_entered.set()
        await blocked.wait()
        yield PCM

    monkeypatch.setattr(instance.native_client, "speech", speech)
    forbid_user_history(monkeypatch, stream)
    _, job = await ready_result(native)
    await asyncio.wait_for(synthesis_entered.wait(), 1)
    announcing = instance.native_turn
    assert announcing.pcm_version == 0
    memory = await instance._manage_background({"action": "history"})
    assert ANSWER not in json.dumps(memory, ensure_ascii=False)

    await utterance(instance, item="followup", text="先等等，我再补充一个条件")

    assert announcing.canceled and job.status == "ready"
    assert not announcing.answer and announcing.committed_exchange is None
    assert not matching_history(instance) and not state.answer_history
    assert job.task_id not in instance.announced_tasks and state.history == []


async def test_finished_public_result_replaces_its_partial_marker_without_duplicate_memory(
    native, monkeypatch,
):
    _, instance, stream, state = native
    first_pcm_sent, release = asyncio.Event(), asyncio.Event()

    async def speech(text):
        state.speech.append(text)
        yield PCM
        first_pcm_sent.set()
        await release.wait()
        yield PCM

    monkeypatch.setattr(instance.native_client, "speech", speech)
    forbid_user_history(monkeypatch, stream)
    _, job = await ready_result(native)
    await asyncio.wait_for(first_pcm_sent.wait(), 1)
    announcing = instance.native_turn
    partial = matching_history(instance)
    assert len(partial) == 1 and partial[0][1] != ANSWER
    assert announcing.public_delivery_text == ANSWER
    partial_memory = await instance._manage_background({"action": "history"})
    assert [item["assistant"] for item in partial_memory["exchanges"]
            if ANSWER in item["assistant"]] == [partial[0][1]]
    assert not state.answer_history

    release.set()
    await eventually(lambda: job.status == "completed" and instance.notification_job is None)

    assert matching_history(instance) == [(QUESTION, ANSWER)]
    memory = await instance._manage_background({"action": "history"})
    assert [item for item in memory["exchanges"] if ANSWER in item["assistant"]] == [
        {"user": QUESTION, "assistant": ANSWER}
    ]
    assert state.history == state.answer_history == []


@pytest.mark.parametrize("interrupted", [False, True])
async def test_public_memory_read_works_without_assistant_injection_or_reconnection(
    native, monkeypatch, interrupted,
):
    _, instance, stream, state = native
    first_pcm_sent, release = asyncio.Event(), asyncio.Event()

    async def sync(text):
        raise AssertionError("Public memory must be read through the native tool result")

    async def speech(text):
        state.speech.append(text)
        yield PCM
        first_pcm_sent.set()
        if interrupted:
            await release.wait()

    monkeypatch.setattr(stream, "add_assistant_answer", sync)
    monkeypatch.setattr(instance.native_client, "speech", speech)
    forbid_user_history(monkeypatch, stream)
    _, job = await ready_result(native)
    await asyncio.wait_for(first_pcm_sent.wait(), 1)
    announcing = instance.native_turn
    if interrupted:
        await instance.interrupt()
    else:
        await eventually(lambda: job.status == "completed" and instance.notification_job is None)

    assert len(matching_history(instance)) == 1 and not state.answer_history

    following = await utterance(instance, item="followup", text="这两个方案分别适合什么条件？")
    memory = await manage(native, item="followup", action="history", call_id="read-public")

    assert following is instance.native_turn and following is not announcing
    assert any(ANSWER in item["assistant"] for item in memory["exchanges"])
    assert not following.deep and not following.deep_result and following.result_task is not None
    assert following.awaiting_result and not following.finished
    assert state.opened == 1 and state.history == state.answer_history == []
    assert len(matching_history(instance)) == 1
    assert job.status == "completed" and state.speech == [ANSWER]


async def test_history_after_completed_acknowledgement_keeps_native_continuation_and_fast_label(
    native,
):
    socket, instance, _, state = native
    instance.public_history.append((QUESTION, ANSWER))
    following = await utterance(instance, text="它们分别适合什么条件？")
    await output(instance, text="我接着比较这两个。")
    await audio_start(instance)
    await audio_chunk(instance)
    await finish(instance, text="我接着比较这两个。")
    assert following.finished
    socket.drain()

    memory = await manage(native, item="q1", action="history", call_id="after-ack-history")

    assert memory["status"] == "public_history"
    assert ANSWER in json.dumps(memory, ensure_ascii=False)
    assert not following.finished and following.awaiting_result
    assert not following.deep and not following.deep_result and following.result_task is not None
    assert not following.background_results
    assert instance.background.records == {} and state.calls == state.speech == []

    continuation = "北辰R适合省心维护，青鹭S适合空间受限。"
    await output(instance, response="r-context", text=continuation)
    await audio_start(instance, response="r-context")
    await audio_chunk(instance)
    await finish(instance, response="r-context", text=continuation)

    assert following.finished and not following.awaiting_result
    assert continuation in following.answer
    assert continuation in instance.public_history[-1][1]
    events = [event for event in socket.drain() if isinstance(event, dict)]
    assert any(event.get("type") == "assistant_done" and continuation in event["text"]
               for event in events)
    assert not any(event.get("type") == "agent_route" and event.get("route") == "deep"
                   for event in events)
    assert not any(event.get("type") in ("task_update", "task_announcement") for event in events)
    assert not any(event.get("type") == "execution_event"
                   and event["event"].get("name") in ("result_returned", "deep_started")
                   for event in events)


async def test_memory_does_not_expose_ready_or_completed_but_unpublished_task_results(native):
    _, instance, _, state = native
    instance.public_history.append(("比较已经播过的方案", ANSWER))
    for task_id, status in (("unspoken-ready", "ready"), ("unspoken-inline", "completed")):
        instance.background.records[task_id] = BackgroundTask(
            origin_turn_id="unpublished-turn", call_id=task_id, task_id=task_id,
            question="秘密条件", title="未公开结果", status=status,
            result=f"{task_id}的秘密报价是9876元。",
        )
    turn = await utterance(instance, text="刚才那两个方案是什么？")

    memory = await manage(native, item="q1", action="history", call_id="only-public")

    assert memory["exchanges"] == [{"user": "比较已经播过的方案", "assistant": ANSWER}]
    assert "秘密" not in json.dumps(memory, ensure_ascii=False)
    assert "9876" not in json.dumps(memory, ensure_ascii=False)
    assert turn.background_results == set() and not turn.deep
    assert not instance.announced_tasks and not state.speech and not state.calls
    assert [task.status for task in instance.background.records.values()] == ["ready", "completed"]


@pytest.mark.parametrize("extra", [
    {"selector": "latest"}, {"selector": "all"}, {"task_ids": ["some-task"]},
])
async def test_history_rejects_task_selectors_without_reading_or_changing_tasks(native, extra):
    _, instance, _, state = native
    turn = await utterance(instance, text="你刚才提到了什么？")
    before = list(instance.public_history)
    memory = await manage(native, item="q1", action="history", **extra)
    assert "error" in memory and "exchanges" not in memory
    assert instance.public_history == before and not turn.background_results
    assert state.calls == state.speech == [] and instance.background.records == {}


async def test_canceled_memory_read_cannot_return_into_the_new_user_turn(native, monkeypatch):
    _, instance, _, state = native
    instance.public_history.append((QUESTION, ANSWER))
    previous = await utterance(instance, text="继续比较这两款。")
    original_manage = instance._manage_background
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(arguments, turn=None):
        memory = await original_manage(arguments, turn)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # A late service may finish despite cancellation. The turn guard,
            # rather than this fake's cancellation, must block its old return.
            return memory
        return memory

    monkeypatch.setattr(instance, "_manage_background", delayed)
    await instance.handle_native_event(tool_event(calls=[{
        "call_id": "slow-memory", "name": "manage_background_tasks",
        "arguments": json.dumps({"action": "history"}),
    }]))
    previous_worker = instance.tool_task
    await asyncio.wait_for(entered.wait(), 1)

    following = await utterance(instance, item="followup", text="先不比较了，你叫什么名字？")

    assert previous.canceled and previous_worker.done()
    assert following is instance.native_turn and not following.canceled
    assert state.results == [] and state.calls == []
    await output(instance, item="followup", response="r-followup", text="我是语音助手。")
    await audio_start(instance, item="followup", response="r-followup")
    await audio_chunk(instance)
    await finish(instance, item="followup", response="r-followup", text="我是语音助手。")
    assert following.finished and not following.deep


async def test_memory_returns_old_price_with_original_limitations_without_refreshing_it(native):
    _, instance, _, state = native
    quotation = "北辰R在2026年10月活动价4599元，来源：媒体报道，当前价格尚未核实。"
    instance.public_history.append(("北辰R以前多少钱？", quotation))
    await utterance(instance, text="它现在是多少钱？")

    memory = await manage(native, item="q1", action="history", call_id="old-quotation")

    assert memory["exchanges"] == [{"user": "北辰R以前多少钱？", "assistant": quotation}]
    assert instance.public_history == [("北辰R以前多少钱？", quotation)]
    assert state.calls == state.speech == [] and not instance.background.records
    assert "不是新的请求或事实证据" in memory["note"]


async def test_memory_keeps_latest_complete_answer_and_reports_omitted_oversize_entry(native):
    _, instance, _, _ = native
    instance.public_history.extend([("早前长答", "旧资料" * 12000), (QUESTION, ANSWER)])
    memory = await instance._manage_background({"action": "history"})
    assert memory["exchanges"] == [{"user": QUESTION, "assistant": ANSWER}]
    assert memory["omitted_exchanges"] == 1
    assert len(json.dumps(memory, ensure_ascii=False).encode()) < 33000

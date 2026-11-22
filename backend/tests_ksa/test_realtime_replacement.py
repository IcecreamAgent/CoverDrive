"""Native model task replacement across turns, using explicit tool arguments only."""

import asyncio
import json

import pytest
from test_realtime import audio_chunk, audio_start, output, utterance
from test_realtime import native as native
from test_realtime_background import dispatch, manage

pytestmark = pytest.mark.parametrize("native", [True], indirect=True)


def delegation(call_id, question, **arguments):
    return {
        "call_id": call_id,
        "name": "delegate_deep_agent",
        "arguments": json.dumps({"question": question, **arguments}, ensure_ascii=False),
    }


async def acknowledge(instance, *, item):
    await output(instance, item=item, response="response-" + item, text="我来按条件处理。")
    await audio_start(instance, item=item, response="response-" + item)
    await audio_chunk(instance)


async def start_search(native, *, item, call_id, question, **arguments):
    _, instance, _, state = native
    await utterance(instance, item=item, text=question)
    await acknowledge(instance, item=item)
    await dispatch(instance, [delegation(call_id, question, **arguments)], item=item)
    receipt = json.loads(state.results[-1][0]["text"])
    assert receipt["status"] == "task_started"
    return instance.background.records[receipt["task_id"]]


async def existing_tasks(native, *, ready):
    _, instance, _, state = native
    instance.settings.background_notify_delay = 0.1
    state.pause = True
    weather = await start_search(
        native, item="weather-turn", call_id="weather-call", question="查询上海今天的天气",
        title="上海天气",
    )
    question = "推荐扫地机器人，尚未给出预算；" + "优先比较清洁能力和日常维护。" * 180
    state.agent_results[question] = "这是旧版不含预算的搜索结果，不得进入后续任务。"
    old = await start_search(
        native, item="robot-turn", call_id="robot-call", question=question, title="扫地机推荐",
    )
    if ready:
        state.released.set()
        await asyncio.gather(weather.worker, old.worker)
        state.released.clear()
        assert old.status == weather.status == "ready"
    else:
        assert old.status == weather.status == "running"
    return weather, old


@pytest.mark.parametrize("ready", [False, True])
async def test_followup_lists_metadata_and_atomically_replaces_only_the_selected_task(
    native, ready,
):
    socket, instance, _, state = native
    weather, old = await existing_tasks(native, ready=ready)
    old.calls.append({
        "id": "old-search", "status": "completed", "name": "global_search",
        "output": "原始工具详情不应发给语音模型",
    })
    await utterance(instance, item="budget-turn", text="预算四千到六千，按这个预算找")
    listed = await manage(native, item="budget-turn", call_id="inspect-tasks", action="list")
    by_id = {task["id"]: task for task in listed["tasks"]}
    assert set(by_id) == {weather.task_id, old.task_id}
    assert by_id[old.task_id]["question"] == old.question[:2000]
    assert len(by_id[old.task_id]["question"]) == 2000
    assert by_id[old.task_id]["title"] == "扫地机推荐"
    assert all("result" not in task and "calls" not in task for task in listed["tasks"])
    assert "原始工具详情" not in json.dumps(listed, ensure_ascii=False)
    if old.result:
        assert old.result not in json.dumps(listed, ensure_ascii=False)
    await acknowledge(instance, item="budget-turn")
    complete_question = "推荐价格4000到6000元的扫地机器人，比较清洁能力和日常维护"
    await dispatch(instance, [delegation(
        "budget-call", complete_question, title="预算版扫地机推荐", replaces_task_id=old.task_id,
    )], item="budget-turn")
    receipt = json.loads(state.results[-1][0]["text"])
    assert receipt["status"] == "task_started" and receipt["result"] is None
    assert receipt["replaced_task_id"] == old.task_id
    new = instance.background.records[receipt["task_id"]]
    assert new.question == complete_question and new.origin_turn_id == instance.native_turn.turn_id
    assert old.status == "canceled" and old.replaced_by_task_id == new.task_id
    assert new.replaces_task_id == old.task_id and new.status == "running"
    assert weather.status == ("ready" if ready else "running")
    assert not weather.replaces_task_id and not weather.replaced_by_task_id
    assert state.calls.count((old.question, True)) == 1
    assert state.calls.count((complete_question, True)) == 1
    assert json.loads(state.contexts[-1]) == {
        "original_user_request": instance.native_turn.question,
        "source": "final_user_input", "origin_turn_id": instance.native_turn.turn_id,
    }
    assert old.result not in state.contexts[-1] if old.result else True
    assert weather.result not in state.contexts[-1] if weather.result else True
    assert state.remembered == []
    assert not state.speech
    traces = [
        event["event"] for event in socket.drain()
        if isinstance(event, dict) and event.get("type") == "execution_event"
    ]
    swaps = [event for event in traces if event["name"] == "task_replaced"]
    assert len(swaps) == 1 and swaps[0]["task_id"] == new.task_id
    assert swaps[0]["replaced_task_id"] == old.task_id
    state.released.set()
    await asyncio.gather(weather.worker, old.worker, new.worker, return_exceptions=True)
    assert new.status == "ready" and old.status == "canceled"
    assert old.task_id not in instance.announced_tasks


@pytest.mark.parametrize("target_state", ["missing", "completed"])
async def test_replacement_rejection_preserves_original_and_returns_actionable_receipt(
    native, target_state,
):
    socket, instance, _, state = native
    weather, old = await existing_tasks(native, ready=target_state == "completed")
    if target_state == "completed":
        await instance.background.update(old, "completed")
        target_id = old.task_id
    else:
        target_id = "task-from-another-session"
    old_status, weather_status, old_result = old.status, weather.status, old.result
    original_calls = list(state.calls)
    await utterance(instance, item="followup-turn", text="预算四千到六千，重新找")
    await acknowledge(instance, item="followup-turn")
    await dispatch(instance, [delegation(
        "rejected-call", "按4000到6000元预算推荐扫地机器人", replaces_task_id=target_id,
    )], item="followup-turn")
    receipt = json.loads(state.results[-1][0]["text"])
    assert set(receipt) == {"error"}
    assert "list" in receipt["error"] and "校验失败不会取消旧任务" in receipt["error"]
    assert old.status == old_status and old.result == old_result and not old.replaced_by_task_id
    assert weather.status == weather_status and not weather.replaced_by_task_id
    assert len(instance.background.records) == 2 and state.calls == original_calls
    events = socket.drain()
    assert not any(isinstance(event, dict) and event.get("type") == "error" for event in events)
    assert any(
        isinstance(event, dict) and event.get("type") == "execution_event"
        and event["event"]["name"] == "delegation_rejected" for event in events
    )


@pytest.mark.parametrize("invalid", [None, 42, "", "x" * 129])
async def test_malformed_replacement_id_never_cancels_an_existing_task(native, invalid):
    _, instance, _, state = native
    _, old = await existing_tasks(native, ready=False)
    await utterance(instance, item="invalid-turn", text="按新预算找")
    await acknowledge(instance, item="invalid-turn")
    await dispatch(instance, [delegation(
        "invalid-call", "按新预算推荐扫地机器人", replaces_task_id=invalid,
    )], item="invalid-turn")
    receipt = json.loads(state.results[-1][0]["text"])
    assert receipt == {"error": "工具或参数不可用"}
    assert old.status == "running" and not old.replaced_by_task_id
    assert len(instance.background.records) == 2 and len(state.calls) == 2


@pytest.mark.parametrize("ready", [False, True])
async def test_replacement_that_needs_no_tools_returns_inline_without_creating_background_job(
    native, ready,
):
    _, instance, _, state = native
    weather, old = await existing_tasks(native, ready=ready)
    state.uses_tools = False
    state.pause = False
    question = "预算4000到6000元，直接根据已知型号比较扫地机器人的选购标准，无需再搜索"
    state.agent_results[question] = "这个预算优先选清洁和避障稳定的型号，再比较维护成本。"
    await utterance(instance, item="inline-turn", text=question)
    await acknowledge(instance, item="inline-turn")
    await dispatch(instance, [delegation(
        "inline-replacement", question, replaces_task_id=old.task_id,
    )], item="inline-turn")
    receipt = json.loads(state.results[-1][0]["text"])
    assert receipt == {
        "status": "completed", "result": state.agent_results[question], "error": "",
        "replaced_task_id": old.task_id,
    }
    inline = instance.background.by_call["inline-replacement"]
    assert inline.status == "completed" and not inline.promoted.is_set()
    assert inline.task_id not in instance.background.records
    assert old.status == "canceled" and old.replaced_by_task_id == inline.task_id
    assert weather.status == ("ready" if ready else "running") and not weather.replaced_by_task_id
    assert instance.native_turn.awaiting_result
    assert instance.native_turn.deep_result == inline.result
    assert len(instance.background.records) == 2 and not state.speech
    assert json.loads(state.contexts[-1]) == {
        "original_user_request": instance.native_turn.question,
        "source": "final_user_input", "origin_turn_id": instance.native_turn.turn_id,
    }
    assert old.result not in state.contexts[-1] if old.result else True
    assert weather.result not in state.contexts[-1] if weather.result else True
    assert state.remembered == []

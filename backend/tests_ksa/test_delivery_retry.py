"""Completed results recover during silence without rerunning task execution."""

import asyncio

import pytest
from test_realtime import finish, output, utterance
from test_realtime import native as native
from test_realtime_background import eventually, pending

from voice_agent.delivery import DeliveryPlan

pytestmark = pytest.mark.parametrize("native", [True], indirect=True)


def deferred_reasons(socket):
    return [
        event["event"]["reason"]
        for event in socket.drain()
        if isinstance(event, dict)
        and event.get("type") == "execution_event"
        and event["event"].get("name") == "delivery_deferred"
    ]


async def ready_for_delivery(native, monkeypatch):
    _, instance, _, state = native
    monkeypatch.setattr(instance, "_delivery_retry_delay", lambda attempts: 0.03)
    _, (job,) = await pending(native)
    generation = instance.foreground_generation
    state.released.set()
    await job.worker
    await finish(instance, text="我帮你查一下。")
    return job, generation


async def test_deferred_result_is_spoken_without_another_utterance_or_task_execution(
    native, monkeypatch
):
    socket, instance, _, state = native
    state.delivery_plans.append(DeliveryPlan())
    job, generation = await ready_for_delivery(native, monkeypatch)

    await eventually(lambda: job.status == "completed")

    assert instance.foreground_generation == generation
    assert len(state.calls) == 1 and len(state.results) == 1
    assert len(state.planner_calls) == 2
    assert state.speech == [job.result]
    assert state.delivery_contexts[-1]["deferrals"] == {job.task_id: 1}
    assert state.delivery_contexts[-1]["idle_seconds"] > 0
    assert deferred_reasons(socket) == ["deferred"]


@pytest.mark.parametrize("failure", ["exception", "invalid_response"])
async def test_planning_failure_recovers_while_user_remains_silent(
    native, monkeypatch, failure
):
    socket, instance, _, state = native
    original_plan = instance.delivery_planner.plan
    attempts = 0

    async def first_plan_fails(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "exception":
                raise RuntimeError("temporary delivery planner failure")
            return DeliveryPlan(outcome="invalid_response")
        return await original_plan(*args, **kwargs)

    monkeypatch.setattr(instance.delivery_planner, "plan", first_plan_fails)
    job, generation = await ready_for_delivery(native, monkeypatch)

    await eventually(lambda: job.status == "completed")

    assert attempts == 2
    assert instance.foreground_generation == generation
    assert len(state.calls) == 1 and len(state.results) == 1
    assert state.speech == [job.result]
    expected = "planning_failed" if failure == "exception" else "invalid_response"
    assert deferred_reasons(socket) == [expected]


async def test_retry_waits_for_new_user_question_and_foreground_answer_to_finish(
    native, monkeypatch
):
    _, instance, _, state = native
    state.delivery_plans.append(DeliveryPlan())
    job, _ = await ready_for_delivery(native, monkeypatch)
    await eventually(lambda: job.task_id in instance.deferred_tasks)

    current = await utterance(instance, item="q2", text="等一下，你叫什么名字？")
    await output(instance, item="q2", response="r2", text="我是语音助手。")
    await asyncio.sleep(0.3)

    assert not current.finished
    assert not state.speech and job.status == "ready"
    assert len(state.calls) == 1

    await finish(instance, item="q2", response="r2", text="我是语音助手。")
    await eventually(lambda: job.status == "completed")

    assert state.speech == [job.result]
    assert len(state.calls) == 1
    assert state.planner_calls[-1][2] == "等一下，你叫什么名字？"


async def test_canceling_deferred_result_prevents_its_automatic_retry(native, monkeypatch):
    _, instance, _, state = native
    state.delivery_plans.append(DeliveryPlan())
    job, generation = await ready_for_delivery(native, monkeypatch)
    await eventually(lambda: job.task_id in instance.deferred_tasks)

    assert await instance._cancel_background(job.task_id)
    await asyncio.sleep(0.35)

    assert job.status == "canceled"
    assert instance.foreground_generation == generation
    assert len(state.planner_calls) == 1 and len(state.calls) == 1
    assert not state.speech
    assert job.task_id not in instance.announced_tasks


@pytest.mark.parametrize("after_pcm", [False, True])
async def test_failed_speech_recovers_without_user_prompt_or_task_rerun(
    native, monkeypatch, after_pcm
):
    socket, instance, _, state = native
    attempts = 0

    async def speech(text):
        nonlocal attempts
        attempts += 1
        state.speech.append(text)
        if attempts == 1:
            if after_pcm:
                yield b"\x03\x00\x04\x00"
            raise RuntimeError("temporary speech failure")
        yield b"\x03\x00\x04\x00"

    monkeypatch.setattr(instance.native_client, "speech", speech)
    job, generation = await ready_for_delivery(native, monkeypatch)

    await eventually(lambda: job.status == "completed")

    assert instance.foreground_generation == generation
    assert attempts == job.announcement_attempts == 2
    assert len(state.calls) == 1 and len(state.results) == 1
    assert state.speech == [job.result, job.result]
    assert job.task_id in instance.announced_tasks
    assert deferred_reasons(socket) == ["speech_not_delivered"]

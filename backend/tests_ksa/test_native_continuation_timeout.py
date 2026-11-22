"""Provider correction/continuation silence must terminate without business retries."""
import asyncio
import json

import pytest

from test_realtime import native as native, utterance, output, audio_start, audio_chunk, finish, tool_event, is_event


async def reject_without_prefix(instance):
    await instance.handle_native_event(tool_event())
    return instance.native_turn.result_task


@pytest.mark.parametrize('native', [False, True], indirect=True)
async def test_rejected_delegation_silence_ends_thinking_and_allows_next_turn(native):
    socket, instance, _, state = native
    instance.settings.native_result_timeout = 0.02
    turn = await utterance(instance, text='可以的。')
    timer = await reject_without_prefix(instance)
    assert timer is not None and not timer.done()
    await asyncio.wait_for(timer, 1)
    events = socket.drain()
    assert turn.canceled and turn.result_task is None
    assert any(is_event('error', stage='native_response')(event) for event in events)
    assert any(is_event('state', stage='idle')(event) for event in events)
    assert not any(is_event('assistant_done')(event) for event in events)
    assert state.calls == state.speech == [] and state.cancels == 1
    assert instance.background.records == {} and instance.insurance_authorization.final_text == instance.native_turn.question
    assert instance.public_history == []
    fresh = await utterance(instance, item='retry', text='刚才没听到回复')
    await output(instance, item='retry', response='retry-response', text='请确认是否办理。')
    await audio_start(instance, item='retry', response='retry-response')
    await finish(instance, item='retry', response='retry-response', text='请确认是否办理。')
    assert fresh.finished and not fresh.canceled


async def test_correction_progress_extends_wait_and_success_cleans_timer(native):
    _, instance, _, state = native
    instance.settings.native_result_timeout = 0.08
    turn = await utterance(instance)
    timer = await reject_without_prefix(instance)
    await asyncio.sleep(0.04)
    await output(instance, response='reply', text='请确认')
    await asyncio.sleep(0.05)
    assert not turn.canceled and not timer.done(), 'new output must extend the wait'
    await audio_start(instance, response='reply')
    await audio_chunk(instance)
    await finish(instance, response='reply', text='请确认是否办理。')
    await asyncio.gather(timer, return_exceptions=True)
    assert turn.finished and turn.result_task is None and state.cancels == 0


@pytest.mark.parametrize('action', ['new_turn', 'interrupt', 'close'])
async def test_retired_correction_timer_cannot_cancel_other_turns_or_speak(native, action):
    socket, instance, _, state = native
    instance.settings.native_result_timeout = 0.03
    previous = await utterance(instance)
    timer = await reject_without_prefix(instance)
    if action == 'new_turn':
        current = await utterance(instance, item='new', text='我补充一个问题')
        assert current is not previous
    elif action == 'interrupt':
        await instance.interrupt()
    else:
        await instance.close_native()
    cancels = state.cancels
    socket.drain()
    await asyncio.sleep(0.06)
    assert timer.done() and previous.result_task is None
    assert state.cancels == cancels and state.calls == state.speech == []
    assert not any(is_event('error')(event) for event in socket.drain())


@pytest.mark.parametrize('failure', ['send_error', 'send_timeout'])
async def test_correction_send_failure_is_recoverable_without_business_execution(native, monkeypatch, failure):
    socket, instance, stream, state = native
    instance.settings.native_result_timeout = 0.02
    async def fail(results):
        if failure == 'send_timeout':
            await asyncio.Event().wait()
        raise RuntimeError('isolated-private-provider-detail')
    monkeypatch.setattr(stream, 'send_tool_results', fail)
    turn = await utterance(instance)
    await asyncio.wait_for(instance.handle_native_event(tool_event()), 1)
    events = socket.drain()
    assert turn.canceled and turn.result_task is None
    assert any(is_event('error', stage='native_response')(event) for event in events)
    assert 'isolated-private-provider-detail' not in json.dumps(events)
    assert any(is_event('state', stage='idle')(event) for event in events)
    assert state.calls == state.speech == []


async def test_memory_tool_continuation_silence_has_same_bounded_recovery(native):
    socket, instance, _, state = native
    instance.settings.native_result_timeout = 0.02
    instance.settings.background_tasks_enabled = True
    turn = await utterance(instance, text='那个方案可以的')
    await instance.handle_native_event(tool_event(calls=[{
        'call_id': 'read-history', 'name': 'manage_background_tasks',
        'arguments': json.dumps({'action': 'history'}),
    }]))
    await instance.tool_task
    timer = turn.result_task
    assert timer is not None
    await asyncio.wait_for(timer, 1)
    assert turn.canceled and instance.insurance_authorization.final_text == instance.native_turn.question
    assert json.loads(state.results[-1][0]['text'])['status'] == 'public_history'
    assert any(is_event('error', stage='native_response')(event) for event in socket.drain())
    assert state.calls == state.speech == []


@pytest.mark.parametrize('native', [True], indirect=True)
async def test_correction_timeout_preserves_existing_independent_business_task(native):
    from test_realtime_background import pending
    _, instance, _, state = native
    _, (job,) = await pending(native)
    instance.settings.native_result_timeout = 0.02
    await utterance(instance, item='q2', text='那个方案可以的')
    event = tool_event(item='q2')
    event['items'][0]['call_id'] = 'new-rejected-call'
    await instance.handle_native_event(event)
    timer = instance.native_turn.result_task
    await asyncio.wait_for(timer, 1)
    assert job.status == 'running' and not job.worker.done()
    assert len(state.calls) == 1 and not state.canceled.is_set()
    state.released.set()
    await job.worker
    assert job.status == 'ready', 'native failure must not rerun or cancel accepted background work'

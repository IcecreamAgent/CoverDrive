"""Public traces follow real execution without changing routing or revealing prompts."""

import asyncio
import json

import pytest
from test_realtime import audio_chunk, audio_start, finish, output, tool_event, utterance
from test_realtime import native as native

from config_fixture import configured_settings
from voice_agent.execution_trace import ExecutionTrace


def trace_events(socket):
    return [item['event'] for item in socket.drain()
            if isinstance(item, dict) and item.get('type') == 'execution_event']


def names(events):
    return [event['name'] for event in events]


async def test_native_fast_milestones_are_once_per_turn_and_contain_no_content(native):
    socket, instance, _, state = native
    turn = await utterance(instance, text='私人问题不应进入诊断记录')
    await output(instance, text='私密回答前半段')
    await output(instance, text='后半段')
    await audio_start(instance)
    await audio_chunk(instance)
    await audio_chunk(instance)
    await finish(instance)
    events = trace_events(socket)
    assert names(events).count('first_text') == names(events).count('first_audio') == 1
    assert names(events).index('input_final') < names(events).index('route_fast')
    assert names(events).index('route_fast') < names(events).index('first_text')
    assert not any(event['stage'] == 'deep' for event in events)
    assert not any(event['stage'] == 'voice' for event in events)
    assert all(event['turn_id'] == turn.turn_id for event in events)
    assert [event['sequence'] for event in events] == sorted(event['sequence'] for event in events)
    assert all(isinstance(event['at'], int) and event['at'] > 0 for event in events)
    assert len({event['id'] for event in events}) == len(events)
    assert '私密' not in json.dumps(events, ensure_ascii=False)
    assert '私人问题' not in json.dumps(events, ensure_ascii=False)
    assert state.calls == []


@pytest.mark.parametrize('native', [False, True], indirect=True)
async def test_fast_deep_result_return_and_real_voice_resume(native):
    socket, instance, _, state = native
    state.uses_tools = False
    turn = await utterance(instance)
    await output(instance, text='我来核实。')
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await asyncio.wait_for(instance.tool_task, 1)
    await output(instance, response='r2', text='核实后的结果。')
    await audio_start(instance, response='r2')
    await audio_chunk(instance)
    await finish(instance, response='r2', text='核实后的结果。')
    events = trace_events(socket)
    for first, second in [('route_fast', 'delegation_requested'),
                          ('delegation_requested', 'delegation_accepted'),
                          ('deep_started', 'deep_result_ready'),
                          ('deep_result_ready', 'result_returned'),
                          ('result_returned', 'voice_resumed')]:
        assert names(events).index(first) < names(events).index(second)
    first_texts = [event for event in events if event['name'] == 'first_text']
    assert [(event['stage'], event['generation']) for event in first_texts] == [
        ('fast', 0), ('voice', 1),
    ]
    assert [event['generation'] for event in events if event['name'] == 'first_audio'] == [0, 1]
    assert turn.finished and not instance.background.records


async def test_protocol_correction_is_visible_and_does_not_claim_deep_started(native):
    socket, instance, _, state = native
    await utterance(instance)
    await instance.handle_native_event(tool_event())
    events = trace_events(socket)
    assert names(events)[-2:] == ['delegation_requested', 'reply_required']
    assert events[-1]['call_id'] == 'call-1'
    assert 'deep_started' not in names(events)
    assert state.calls == []


@pytest.mark.parametrize('native', [True], indirect=True)
async def test_background_events_keep_origin_when_foreground_is_interrupted(native):
    socket, instance, _, state = native
    state.pause = True
    original = await utterance(instance)
    await output(instance, text='我查一下。')
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await asyncio.wait_for(instance.tool_task, 1)
    job = next(iter(instance.background.records.values()))
    initial = trace_events(socket)
    assert 'background_started' in names(initial)
    current = await utterance(instance, item='q2', text='另一个问题')
    state.released.set()
    await job.worker
    events = trace_events(socket)
    ready = next(event for event in events if event['name'] == 'task_ready')
    assert ready['turn_id'] == original.turn_id != current.turn_id
    assert ready['task_id'] == job.task_id
    assert any(event['name'] == 'interrupted' and event['turn_id'] == original.turn_id
               for event in events)
    assert job.status == 'ready' and not state.canceled.is_set()


@pytest.mark.parametrize('native', [True], indirect=True)
async def test_keyboard_entry_has_no_fast_trace_or_synthetic_first_text(native):
    socket, instance, _, state = native
    state.pause = True
    instance.turn_id = 'keyboard-input'
    await instance.respond('请查天气')
    events = trace_events(socket)
    assert 'input_text' in names(events) and 'route_deep' in names(events)
    assert 'deep_started' in names(events) and 'background_started' in names(events)
    assert 'route_fast' not in names(events) and 'first_text' not in names(events)
    assert all(event['stage'] != 'fast' for event in events)
    state.released.set()


@pytest.mark.parametrize('native', [True], indirect=True)
async def test_inline_keyboard_result_hands_off_to_voice_without_rerunning_deep(native):
    socket, instance, _, state = native
    state.uses_tools = False
    instance.turn_id = 'keyboard-inline'
    await instance.respond('解释方案差异')
    events = trace_events(socket)
    assert names(events).count('deep_started') == 1
    assert names(events).count('deep_result_ready') == 1
    assert names(events).count('result_returned') == 1
    assert 'first_audio' in names(events) and 'route_fast' not in names(events)
    assert len(state.calls) == 1 and not instance.background.records


async def test_late_native_prefix_remains_fast_while_deep_runs(native):
    socket, instance, _, state = native
    state.pause = True
    await utterance(instance)
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await asyncio.wait_for(state.entered.wait(), 1)
    await output(instance, text='这段承接文字晚于委派到达。')
    events = trace_events(socket)
    first_text = next(event for event in events if event['name'] == 'first_text')
    assert first_text['stage'] == 'fast'
    assert names(events).index('route_deep') < names(events).index('first_text')
    state.released.set()
    await asyncio.wait_for(instance.tool_task, 1)


async def test_late_prefix_pcm_after_return_is_not_claimed_as_result_audio(native):
    socket, instance, _, _ = native
    await utterance(instance)
    await output(instance, text='稍等。')
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await asyncio.wait_for(instance.tool_task, 1)
    trace_events(socket)
    await audio_chunk(instance)  # Still belongs to r1, before new result output starts.
    assert not any(event['name'] == 'first_audio' for event in trace_events(socket))
    await output(instance, response='r2', text='结果。')
    await audio_start(instance, response='r2')
    await audio_chunk(instance)
    assert any(event['name'] == 'first_audio' for event in trace_events(socket))


def test_trace_state_and_dedup_are_bounded():
    trace = ExecutionTrace(configured_settings(_env_file=None))
    for index in range(1200):
        trace.turn(str(index))
        trace.event(str(index), 'input', 'input_started', '输入', once='one')
    assert len(trace.turns) <= 64 and len(trace.seen) <= 1024


@pytest.mark.parametrize('native', [True], indirect=True)
async def test_inline_queue_wait_is_visible_before_execution_begins(native):
    socket, instance, _, state = native
    state.uses_tools = False
    instance.background._semaphore = asyncio.Semaphore(0)
    pending = asyncio.create_task(instance.background.start_delegation(
        '解释方案', 'queued-turn', 'queued-call', [],
    ))
    try:
        observed = await socket.until(lambda event: isinstance(event, dict)
                                      and event.get('type') == 'execution_event'
                                      and event['event']['name'] == 'deep_queued')
        queued = observed[-1]['event']
        assert queued['turn_id'] == 'queued-turn' and queued['status'] == 'waiting'
        assert not state.calls and not pending.done() and not instance.background.records
        instance.background._semaphore.release()
        task = await asyncio.wait_for(pending, 1)
        events = trace_events(socket)
        started = next(event for event in events if event['name'] == 'deep_started')
        ready = next(event for event in events if event['name'] == 'deep_result_ready')
        assert queued['at'] <= started['at'] <= ready['at']
        assert started['task_id'] == ready['task_id'] == task.task_id
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def test_websocket_route_and_trace_metadata_do_not_expose_configured_models_or_keys(native):
    socket, instance, _, _ = native
    private_values = {
        instance.settings.llm_model_id, instance.settings.seeduplex_model,
        instance.settings.llm_api_key, instance.settings.seeduplex_access_key,
    }
    await utterance(instance)
    await output(instance, text="我来核实。")
    await audio_start(instance)
    await audio_chunk(instance)
    await instance.handle_native_event(tool_event())
    await asyncio.wait_for(instance.tool_task, 1)
    outgoing = socket.drain()
    assert any(isinstance(item, dict) and item.get("type") == "agent_route" for item in outgoing)
    assert any(isinstance(item, dict) and item.get("type") == "execution_event" for item in outgoing)
    serialized = json.dumps([item for item in outgoing if isinstance(item, dict)], ensure_ascii=False)
    assert "端到端语音" in serialized and "任务 LLM" in serialized
    for private_value in private_values:
        assert private_value and private_value not in serialized

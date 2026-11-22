"""Model boundaries keep final customer intent separate from delegated wording."""
import json

import pytest

from test_realtime import native as native, utterance, output, audio_start, audio_chunk, tool_event
from voice_agent import outbound
from voice_agent.insurance import customer_fact_context


@pytest.mark.parametrize('background_enabled', [False, True])
async def test_sync_and_background_delegation_keep_original_final_request(native, background_enabled):
    _, instance, _, state = native
    instance.settings.background_tasks_enabled = background_enabled
    original = '我去年买了多少钱呢？'
    task_text = '查询这辆车的保险费用'
    turn = await utterance(instance, text=original)
    await output(instance, text='我帮您核实记录。')
    await audio_start(instance)
    await audio_chunk(instance)
    event = tool_event(calls=[{'call_id': 'original-intent', 'name': 'delegate_deep_agent',
                              'arguments': json.dumps({'question': task_text})}])
    await instance.handle_native_event(event)
    await instance.tool_task
    assert state.calls == [(task_text, True)]
    context = json.loads(state.contexts[-1])
    assert context == {'original_user_request': original, 'source': 'final_user_input',
                       'origin_turn_id': turn.turn_id}
    assert instance.insurance_authorization.final_text == original
    if instance.settings.background_tasks_enabled:
        record = instance.background.by_call['original-intent']
        assert record.context == state.contexts[-1]


async def test_partial_asr_cannot_become_original_final_request(native):
    _, instance, _, _ = native
    await instance.handle_native_event({'type': 'conversation.item.input_audio_transcription.started', 'item_id': 'partial'})
    await instance.handle_native_event({'type': 'conversation.item.input_audio_transcription.delta',
                                        'item_id': 'partial', 'delta': '部分识别还会修订'})
    turn = instance.native_turn
    assert instance._delegation_context(turn) == ''
    assert instance.insurance_authorization.version == 0


async def test_opening_and_reconnect_share_complete_selected_tool_fact(native, monkeypatch):
    _, instance, _, state = native
    customer = {'id': 101, '姓名': '隔离客户甲', '车辆类别': '家用轿车', '座位数': 5,
                '车型': '隔离车型', '年度保费(元)': 4200, '险种': '交强险+商业险',
                '到期日期': '2026-12-18', '新车购置价(元)': 180000}

    async def query(arguments):
        return {'客户': [customer], '数量': 1}

    async def catalog():
        from types import SimpleNamespace
        return [SimpleNamespace(name='get_expiring_customers', ainvoke=query)]

    monkeypatch.setattr(outbound, 'load_business_tools', catalog)
    await instance.start_outbound()
    await instance.response_task
    facts = instance.insurance_authorization.customer_facts
    assert json.loads(facts[0]['result'])['客户'] == [customer]
    shared_context = customer_fact_context(facts)
    assert shared_context in state.hints[-1]
    assert shared_context in instance._native_instructions(), 'reconnect must retain the selected facts'
    assert instance.insurance_authorization.version == 0

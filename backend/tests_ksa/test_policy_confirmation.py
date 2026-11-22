"""Policy confirmation is a tool transaction, never a native model claim."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from test_realtime import native as native, utterance, output, audio_start, audio_chunk, finish
from voice_agent import agent
from config_fixture import configured_settings
from voice_agent.insurance import InsuranceAuthorization, guarded_tools


CONFIRM = '同意按交强险加三者险 100 万这个方案办理续保。'
POLICY = {'id': 501, 'policy_no': 'POL-ISOLATED-501', 'status': '已承保',
          'total_premium': 2130.0, 'effective_date': '2026-11-19'}


@pytest.mark.parametrize('text', [CONFIRM, '那就帮我办上吧。', '行，就这个。',
    '我同意投保吗？', '我不要车损险，其他照旧办。', '暂时不办。'])
def test_final_tracker_preserves_words_without_classifying_business_intent(text):
    auth = InsuranceAuthorization()
    auth.observe_final(text)
    assert auth.final_text == text and auth.version == 1
    assert not hasattr(auth, 'consent') and not hasattr(auth, 'refused')


class Args(BaseModel):
    status: str


def protected(auth, result=POLICY, error=None):
    async def create_policy(status):
        if error:
            raise error
        return result
    tool = StructuredTool.from_function(coroutine=create_policy, name='create_policy',
        description='Isolated policy fixture', args_schema=Args)
    return guarded_tools([tool], SimpleNamespace(insurance_writes_enabled=True), auth)[0]


@pytest.mark.parametrize('result', [POLICY, json.dumps(POLICY),
    [{'type': 'text', 'text': json.dumps(POLICY)}],
    ToolMessage(content=json.dumps(POLICY), tool_call_id='fixture')])
async def test_receipt_uses_only_valid_structured_policy_tool_data(result):
    auth = InsuranceAuthorization()
    auth.observe_final(CONFIRM)
    await protected(auth, result).ainvoke({'status': '已生效'})
    answer = auth.policy_confirmation_result(auth.version)
    assert '已出单' in answer and POLICY['policy_no'] in answer and '2130.0元' in answer
    assert '2026-11-19' in answer and '发送' not in answer


@pytest.mark.parametrize('result', [None, '已成功出单，放心吧', {'status': '报价'},
    {**POLICY, 'total_premium': float('nan')}, {**POLICY, 'policy_no': None},
    {'error': 'isolated-error'}, ToolMessage(content='failure', status='error', tool_call_id='fixture')])
async def test_failed_incomplete_or_narrative_tool_output_cannot_confirm_policy(result):
    auth = InsuranceAuthorization()
    auth.observe_final(CONFIRM)
    await protected(auth, result).ainvoke({'status': '已生效'})
    answer = auth.policy_confirmation_result(auth.version)
    assert '已出单，保单号' not in answer
    assert '无法确认' in answer or '未能完成' in answer


@pytest.mark.parametrize('failure', [TimeoutError(), RuntimeError('isolated-provider-error')])
async def test_unknown_write_cannot_be_retried_by_same_model_turn(failure):
    auth = InsuranceAuthorization()
    auth.observe_final(CONFIRM)
    tool = protected(auth, error=failure)
    with pytest.raises(type(failure)):
        await tool.ainvoke({'status': '已生效'})
    assert '无法确认' in auth.policy_confirmation_result(auth.version)
    assert '不重复办理' in await tool.ainvoke({'status': '已生效'})


async def test_canceled_write_remains_unknown_without_success_or_retry():
    auth = InsuranceAuthorization()
    auth.observe_final(CONFIRM)
    entered = asyncio.Event()
    async def create_policy(status):
        entered.set()
        await asyncio.Event().wait()
    original = StructuredTool.from_function(coroutine=create_policy, name='create_policy',
        description='Blocked isolated fixture', args_schema=Args)
    tool = guarded_tools([original], SimpleNamespace(insurance_writes_enabled=True), auth)[0]
    task = asyncio.create_task(tool.ainvoke({'status': '已生效'}))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert '无法确认' in auth.policy_confirmation_result(auth.version)
    assert '不重复办理' in await tool.ainvoke({'status': '已生效'})


async def test_actual_policy_attempt_uses_receipt_instead_of_model_fabricated_fields(monkeypatch):
    auth = InsuranceAuthorization()
    auth.observe_final(CONFIRM)
    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
    responses = [AIMessage(content='', tool_calls=[{'name':'create_policy',
        'args':{'status':'已生效'}, 'id':'fixture-call'}])]
    responses.append(AIMessage(content='已经办理成功，电子保单已经发送，保费1元。'))
    monkeypatch.setattr(agent, 'ChatOpenAI', lambda **kwargs: ToolModel(responses=responses))
    async def graph(self):
        return agent.create_deep_agent(model=self._model, tools=[protected(auth)],
            system_prompt='Only use isolated fixture data.')
    monkeypatch.setattr(agent.AgentSession, '_build_graph', graph)
    session = agent.AgentSession(configured_settings(_env_file=None, llm_api_key='fake'),
        task_only=True, authorization=auth)
    parts = [p async for p in session.stream(CONFIRM)]
    assert parts == [auth.policy_confirmation_result(auth.version)]
    assert session.final_answer == parts[0]
    assert '已经发送' not in parts[0] and '保费1元' not in parts[0]
    assert '已出单' in parts[0]
    assert session._state['messages'][-1].content == parts[0]


@pytest.mark.parametrize('native', [False, True], indirect=True)
@pytest.mark.parametrize('text', [CONFIRM, '那就帮我办上吧。', '我不要车损险，其他照旧办。'])
async def test_natural_input_uses_model_delegation_without_server_consent_route(native, text):
    socket, instance, _, state = native
    turn = await utterance(instance, text=text)
    assert not turn.assisted and not turn.deep
    assert state.calls == [] and state.cancels == 0
    assert instance.insurance_authorization.final_text == text
    await output(instance, text='我来办理。')
    await audio_start(instance)
    await audio_chunk(instance)
    from test_realtime import tool_event
    await instance.handle_native_event(tool_event(calls=[{'call_id':'model-policy',
        'name':'delegate_deep_agent','arguments':json.dumps({'question':text})}]))
    await asyncio.wait_for(instance.tool_task,1)
    assert state.calls == [(text,True)]
    assert instance.insurance_authorization.version == 1
    # Duplicate final and FC packets must not submit another model task.
    await instance.handle_native_event({'type':'conversation.item.input_audio_transcription.completed',
        'item_id':'q1','text':text})
    await instance.handle_native_event(tool_event(calls=[{'call_id':'model-policy',
        'name':'delegate_deep_agent','arguments':json.dumps({'question':text})}]))
    assert len(state.calls) == 1


@pytest.mark.parametrize('committed', [False, True])
async def test_post_tool_model_failure_keeps_committed_or_unknown_receipt(monkeypatch, committed):
    auth = InsuranceAuthorization()
    auth.observe_final(CONFIRM)
    class Graph:
        async def astream(self, state, **kwargs):
            tool = protected(auth, error=None if committed else TimeoutError())
            await tool.ainvoke({'status':'已生效'})
            raise RuntimeError('isolated-post-tool-model-failure')
            yield  # pragma: no cover
    session = agent.AgentSession(configured_settings(_env_file=None, llm_api_key='fake'),
        task_only=True, authorization=auth)
    session.graph = Graph()
    answer = ''.join([p async for p in session.stream(CONFIRM)])
    assert answer == session.final_answer
    assert ('已出单，保单号' in answer) == committed
    assert POLICY['policy_no'] in answer if committed else '无法确认' in answer


async def test_selected_customer_cannot_be_replaced_by_model_generated_identity():
    auth = InsuranceAuthorization()
    auth.observe_final(CONFIRM)
    auth.selected_customer={'姓名':'隔离客户','电话':'00000000991','车牌号':'测T00991'}
    class IdentityArgs(BaseModel):
        customer_name: str
        customer_phone: str
        plate: str
    calls=[]
    async def create_policy(**arguments):
        calls.append(arguments)
        return POLICY
    original=StructuredTool.from_function(coroutine=create_policy, name='create_policy',
        description='Isolated identity fixture', args_schema=IdentityArgs)
    tool=guarded_tools([original], SimpleNamespace(insurance_writes_enabled=True),auth)[0]
    result=await tool.ainvoke({'customer_name':'其他客户','customer_phone':'00000000882','plate':'测T00882'})
    assert '不一致' in result and calls == [] and auth.policy_results == {}
    await tool.ainvoke({'customer_name':'隔离客户','customer_phone':'00000000991','plate':'测T00991'})
    assert len(calls)==1 and POLICY['policy_no'] in auth.policy_confirmation_result(auth.version)


@pytest.mark.parametrize('native', [False, True], indirect=True)
async def test_interrupting_model_delegation_keeps_existing_cancellation_contract(native):
    socket, instance, _, state=native
    state.pause=True
    first=await utterance(instance,text='那就帮我办上吧。')
    await output(instance,text='我来办理。')
    from test_realtime import tool_event
    await instance.handle_native_event(tool_event(calls=[{'call_id':'model-policy-interrupt',
        'name':'delegate_deep_agent','arguments':json.dumps({'question':first.question})}]))
    await asyncio.wait_for(state.entered.wait(),1)
    if instance.settings.background_tasks_enabled:
        await asyncio.wait_for(instance.tool_task,1)
    await instance.interrupt()
    assert first.canceled
    socket.drain()
    await output(instance,text='已出单成功。')
    await audio_start(instance)
    await audio_chunk(instance)
    assert not any(isinstance(e,bytes) for e in socket.drain())
    if instance.settings.background_tasks_enabled:
        record=instance.background.by_call['model-policy-interrupt']
        assert record.status in {'queued','running'}
        state.released.set()
        await asyncio.wait_for(record.worker,1)
        assert record.status=='ready'
    else:
        assert state.canceled.is_set()


async def test_final_words_do_not_force_policy_response_or_block_normal_streaming():
    from langchain_core.messages import AIMessageChunk
    auth=InsuranceAuthorization()
    auth.observe_final('我同意投保，请出单。')
    class Graph:
        async def astream(self,state,**kwargs):
            yield 'messages', (AIMessageChunk(content='请先确认'),{})
            yield 'messages', (AIMessageChunk(content='要保哪辆车？'),{})
            yield 'values', {'messages':[*state['messages'],AIMessage(content='请先确认要保哪辆车？')]}
    session=agent.AgentSession(configured_settings(_env_file=None,llm_api_key='fake'),task_only=True,authorization=auth)
    session.graph=Graph()
    parts=[p async for p in session.stream(auth.final_text)]
    assert parts == ['请先确认','要保哪辆车？']
    assert session.final_answer=='请先确认要保哪辆车？'
    assert auth.policy_attempts==set() and auth.policy_results=={}


@pytest.mark.parametrize('text',['行，帮我续上。','我不要车损险，其他照旧办。'])
async def test_colloquial_model_tool_call_generates_policy_without_consent_flag(monkeypatch,text):
    auth=InsuranceAuthorization()
    auth.observe_final(text)
    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self,tools,**kwargs):
            return self
    model=ToolModel(responses=[AIMessage(content='',tool_calls=[{'name':'create_policy',
        'args':{'status':'已生效'},'id':'colloquial-policy'}]),AIMessage(content='根据工具结果答复。')])
    monkeypatch.setattr(agent,'ChatOpenAI',lambda **kwargs:model)
    async def graph(self):
        return agent.create_deep_agent(model=self._model,tools=[protected(auth)],
            system_prompt='Use isolated tool results.')
    monkeypatch.setattr(agent.AgentSession,'_build_graph',graph)
    session=agent.AgentSession(configured_settings(_env_file=None,llm_api_key='fake'),task_only=True,authorization=auth)
    parts=[p async for p in session.stream(text)]
    assert parts==[auth.policy_confirmation_result(auth.version)]
    assert POLICY['policy_no'] in session.final_answer
    assert not hasattr(auth,'consent')

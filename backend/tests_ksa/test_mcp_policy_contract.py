"""Exercise the installed MCP adapter, including status and artifact semantics."""
import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import CallToolResult, TextContent, Tool

from voice_agent.insurance import InsuranceAuthorization, guarded_tools


POLICY = {'id': 801, 'policy_no': 'POL-ISOLATED-801', 'status': '已承保',
          'total_premium': 3660.5, 'effective_date': '2026-11-19'}
CUSTOMER = {'id': 801, '姓名': '隔离测试', '电话': '00000000801', '车牌号': '测T00801'}
IDENTITY = {'customer_name': CUSTOMER['姓名'], 'customer_phone': CUSTOMER['电话'],
            'plate': CUSTOMER['车牌号']}
COVERAGE = [{'code': 'cie'}, {'code': 'vehicle_damage'},
            {'code': 'third_party', 'amount': 30}, {'code': 'wading'}]
PARAMETERS = {'type': 'object', 'properties': {
    'coverage_items': {'type': 'array', 'items': {'type': 'object'}},
    'customer_name': {'type': 'string'}, 'customer_phone': {'type': 'string'},
    'plate': {'type': 'string'}, 'customer_id': {'type': 'integer'},
    'status': {'type': 'string'}, 'policy_id': {'type': 'string'},
}}
ENABLED = SimpleNamespace(insurance_writes_enabled=True)


def packet(data=None, *, error=None, structured=None):
    return CallToolResult(isError=error is not None,
        content=[] if data is None and error is None else [TextContent(type='text',
            text=error if error is not None else json.dumps(data, ensure_ascii=False))],
        structuredContent=structured)


def mcp_tool(name, responses, calls):
    class Session:
        async def call_tool(self, tool_name, tool_args, **kwargs):
            calls.append((tool_name, deepcopy(tool_args)))
            response = responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response
    return convert_mcp_tool_to_langchain_tool(Session(), Tool(name=name,
        description='Isolated MCP tool; coverage_items accepts string names or dictionaries.',
        inputSchema=PARAMETERS))


def authorization():
    auth = InsuranceAuthorization()
    auth.selected_customer = CUSTOMER.copy()
    auth.observe_final('可以，就按刚才的方案办。')
    return auth


async def test_mcp_validation_error_is_an_outer_error_and_not_a_policy_receipt():
    auth, calls = authorization(), []
    original = mcp_tool('create_policy', [packet(error=(
        'Error executing tool create_policy: 1 validation error for create_policyArguments\n'
        'coverage_items.0\n  Input should be a valid dictionary'))], calls)
    protected = guarded_tools([original], ENABLED, auth)[0]
    result = await protected.ainvoke({'type': 'tool_call', 'id': 'outer-call',
        'name': 'create_policy', 'args': {**IDENTITY, 'coverage_items': COVERAGE}})
    assert isinstance(result, ToolMessage) and result.tool_call_id == 'outer-call'
    assert result.status == 'error'
    assert auth.policy_results[auth.version]['outcome'] == 'not_submitted'
    assert '未提交' in auth.policy_confirmation_result(auth.version)
    assert '已出单，保单号' not in auth.policy_confirmation_result(auth.version)
    assert len(calls) == 1


async def test_mcp_definite_argument_rejection_can_be_corrected_in_same_turn():
    auth, calls = authorization(), []
    original = mcp_tool('create_policy', [packet(error=(
        'Error executing tool create_policy: 2 validation errors for create_policyArguments\ncoverage_items')),
        packet(POLICY)], calls)
    protected = guarded_tools([original], ENABLED, auth)[0]
    await protected.ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    await protected.ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    assert len(calls) == 2
    assert POLICY['policy_no'] in auth.policy_confirmation_result(auth.version)
    again = await protected.ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    assert '不重复办理' in again and len(calls) == 2


async def test_string_coverage_entries_are_structured_without_changing_coverage_or_limits():
    auth, calls = authorization(), []
    original = mcp_tool('create_policy', [packet(POLICY)], calls)
    protected = guarded_tools([original], ENABLED, auth)[0]
    requested = ['交强险', '车损险', {'code': 'third_party', 'amount': 30}, 'wading']
    await protected.ainvoke({**IDENTITY, 'coverage_items': requested})
    assert calls[0][1]['coverage_items'] == [
        {'name': '交强险'}, {'name': '车损险'}, {'code': 'third_party', 'amount': 30}, {'name': 'wading'}]
    assert requested[0] == '交强险'
    assert POLICY['policy_no'] in auth.policy_confirmation_result(auth.version)


@pytest.mark.parametrize('name', ['calc_premium', 'get_policy', 'create_policy'])
async def test_mcp_unstructured_error_text_never_becomes_success_or_customer_facts(name):
    auth, calls = authorization(), []
    original = mcp_tool(name, [packet(error=f'Error executing tool {name}: isolated provider failure')], calls)
    protected = guarded_tools([original], ENABLED, auth)[0]
    result = await protected.ainvoke({'type': 'tool_call', 'id': 'outer-failure', 'name': name,
        'args': {**IDENTITY, 'coverage_items': COVERAGE}})
    assert result.status == 'error'
    assert 'isolated provider failure' not in result.content
    assert auth.customer_facts == []
    if name == 'create_policy':
        assert '已出单' not in auth.policy_confirmation_result(auth.version)
        await protected.ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
        assert len(calls) == 1  # Only definitive argument rejection permits correction.


@pytest.mark.parametrize('failure', [TimeoutError(), RuntimeError('isolated transport failure')])
async def test_transport_failure_keeps_unknown_reservation_without_duplicate_submission(failure):
    auth, calls = authorization(), []
    original = mcp_tool('create_policy', [failure], calls)
    protected = guarded_tools([original], ENABLED, auth)[0]
    with pytest.raises(type(failure)):
        await protected.ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    assert auth.policy_results[auth.version]['outcome'] == 'unknown'
    assert '不重复办理' in await protected.ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    assert len(calls) == 1


async def test_mcp_structured_artifact_can_supply_receipt_without_text_content():
    auth, calls = authorization(), []
    original = mcp_tool('create_policy', [packet(structured=POLICY)], calls)
    protected = guarded_tools([original], ENABLED, auth)[0]
    await protected.ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    assert POLICY['policy_no'] in auth.policy_confirmation_result(auth.version)


@pytest.mark.parametrize('outcome', [None, 'unknown', 'not_submitted', 'failed', 'invalid'])
async def test_issued_customer_mark_requires_real_valid_policy_for_this_customer(outcome):
    auth, calls = authorization(), []
    if outcome:
        auth.policy_results[auth.version] = {'outcome': outcome, 'customer_id': CUSTOMER['id'], 'policy': {}}
    original = mcp_tool('mark_customer_contacted', [packet({'updated': True})], calls)
    protected = guarded_tools([original], ENABLED, auth)[0]
    result = await protected.ainvoke({'type': 'tool_call', 'id': 'false-mark',
        'name': 'mark_customer_contacted', 'args': {'customer_id': CUSTOMER['id'], 'status': '已出单'}})
    assert result.status == 'error' and calls == []
    assert '真实保单' in result.content


async def test_successful_policy_allows_same_customer_mark_and_only_one_submission():
    auth, calls = authorization(), []
    tools = guarded_tools([mcp_tool('create_policy', [packet(POLICY)], calls),
        mcp_tool('mark_customer_contacted', [packet({'updated': True})], calls)], ENABLED, auth)
    await tools[0].ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    await tools[1].ainvoke({'customer_id': CUSTOMER['id'], 'status': '已出单'})
    assert [name for name, _ in calls] == ['create_policy', 'mark_customer_contacted']
    assert POLICY['policy_no'] in auth.policy_confirmation_result(auth.version)
    other = await tools[1].ainvoke({'customer_id': CUSTOMER['id']+1, 'status': '已出单'})
    assert '不一致' in other and len(calls) == 2


async def test_nested_adapter_callbacks_do_not_report_a_failed_call_as_returned():
    from voice_agent.agent import _ToolStartCallback
    auth, calls, events = authorization(), [], []
    async def observe(event):
        events.append(event)
    original = mcp_tool('create_policy', [packet(error=(
        'Error executing tool create_policy: 1 validation error for create_policyArguments\ncoverage_items'))], calls)
    protected = guarded_tools([original], ENABLED, auth)[0]
    await protected.ainvoke({'type': 'tool_call', 'id': 'outer-trace', 'name': 'create_policy',
        'args': {**IDENTITY, 'coverage_items': COVERAGE}}, config={'callbacks': [_ToolStartCallback(None, observe)]})
    assert [event['status'] for event in events] == ['running', 'failed']


async def test_quote_display_name_is_resolved_from_actual_tool_data_across_tasks():
    auth, calls = authorization(), []
    quote = {'total_premium': 3660.5, 'items': [
        {'product_code': 'wading', 'product_name': '发动机涉水损失险', 'premium': 95.5}]}
    quote_tool = mcp_tool('calc_premium', [packet(quote)], calls)
    await guarded_tools([quote_tool], ENABLED, auth)[0].ainvoke({'coverage_items': [{'code': 'wading'}]})
    # A new isolated task uses the same verified alias map, never a guessed code.
    policy_tool = mcp_tool('create_policy', [packet(POLICY)], calls)
    await guarded_tools([policy_tool], ENABLED, auth)[0].ainvoke({**IDENTITY,
        'coverage_items': ['发动机涉水损失险', {'险种': '三者险', '保额': 30}]})
    assert calls[-1][1]['coverage_items'] == [{'name': 'wading'}, {'险种': '三者险', '保额': 30}]
    assert auth.customer_facts == []
    assert InsuranceAuthorization().coverage_aliases == {}


async def test_unknown_product_rejected_before_backend_submission_can_be_corrected():
    auth, calls = authorization(), []
    original = mcp_tool('create_policy', [packet(error='Error executing tool create_policy: 未知险种：isolated-invalid-name'),
        packet(POLICY)], calls)
    protected = guarded_tools([original], ENABLED, auth)[0]
    failure = await protected.ainvoke({**IDENTITY, 'coverage_items': [{'name': 'isolated-invalid-name'}]})
    assert '未提交' in failure
    await protected.ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    assert len(calls) == 2 and POLICY['policy_no'] in auth.policy_confirmation_result(auth.version)


async def test_structured_only_mcp_receipt_is_preserved_and_used():
    auth, calls = authorization(), []
    original = mcp_tool('create_policy', [packet(structured=POLICY)], calls)
    await guarded_tools([original], ENABLED, auth)[0].ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    assert POLICY['policy_no'] in auth.policy_confirmation_result(auth.version)


async def test_failed_submission_cannot_mark_customer_as_issued():
    auth, calls = authorization(), []
    tools = guarded_tools([
        mcp_tool('create_policy', [packet(error='Error executing tool create_policy: 1 validation error for create_policyArguments\ncoverage_items')], calls),
        mcp_tool('mark_customer_contacted', [packet({'status': '已出单'})], calls),
    ], ENABLED, auth)
    await tools[0].ainvoke({**IDENTITY, 'coverage_items': COVERAGE})
    failure = await tools[1].ainvoke({'customer_id': CUSTOMER['id'], 'status': '已出单'})
    assert '真实保单回执' in failure
    assert [name for name, _ in calls] == ['create_policy']


async def test_graph_corrects_mcp_arguments_without_false_issue_tag_or_fabricated_receipt(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from voice_agent import agent
    from config_fixture import configured_settings
    auth, calls = authorization(), []
    tools = [mcp_tool('create_policy', [packet(error=(
        'Error executing tool create_policy: 1 validation error for create_policyArguments\ncoverage_items')),
        packet(POLICY)], calls), mcp_tool('mark_customer_contacted', [packet({'status': '已出单'})], calls)]
    def fc(name, args, call_id):
        return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': call_id}])
    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
    model = Model(responses=[
        fc('create_policy', {**IDENTITY, 'coverage_items': COVERAGE}, 'invalid-call'),
        fc('mark_customer_contacted', {'customer_id': CUSTOMER['id'], 'status': '已出单'}, 'wrong-mark'),
        fc('create_policy', {**IDENTITY, 'coverage_items': COVERAGE}, 'corrected-call'),
        fc('mark_customer_contacted', {'customer_id': CUSTOMER['id'], 'status': '已出单'}, 'actual-mark'),
        AIMessage(content='已经发送电子保单，保费一元。'),
    ])
    monkeypatch.setattr(agent, 'ChatOpenAI', lambda **kwargs: model)
    async def catalog():
        return tools
    monkeypatch.setattr(agent, 'load_business_tools', catalog)
    session = agent.AgentSession(configured_settings(_env_file=None, llm_api_key='fake', insurance_writes_enabled=True),
        task_only=True, authorization=auth)
    result = ''.join([part async for part in session.stream(auth.final_text)])
    assert result == session.final_answer == auth.policy_confirmation_result(auth.version)
    assert POLICY['policy_no'] in result and '3660.5元' in result
    assert '已经发送' not in result
    assert [name for name, _ in calls] == ['create_policy', 'create_policy', 'mark_customer_contacted']
    messages = {m.tool_call_id: m for m in session._state['messages'] if isinstance(m, ToolMessage)}
    assert messages['invalid-call'].status == messages['wrong-mark'].status == 'error'
    assert messages['corrected-call'].status == messages['actual-mark'].status == 'success'


async def test_structured_only_quote_remains_visible_to_execution_model():
    auth, calls = authorization(), []
    quote = {'total_premium': 3660.5, 'items': [{'product_code': 'wading', 'product_name': '涉水险'}]}
    original = mcp_tool('calc_premium', [packet(structured=quote)], calls)
    result = await guarded_tools([original], ENABLED, auth)[0].ainvoke({'coverage_items': COVERAGE})
    assert json.loads(result) == quote
    assert auth.coverage_aliases['涉水险'] == 'wading'


@pytest.mark.parametrize('same_customer', [False, True])
async def test_renewal_receipt_does_not_authorize_issue_mark_for_different_customer(same_customer):
    auth, calls = authorization(), []
    policy = {**POLICY, 'customer': {'name': CUSTOMER['姓名'],
        'phone': CUSTOMER['电话'] if same_customer else '00000000802'},
        'vehicle': {'plate_no': CUSTOMER['车牌号']}}
    tools = guarded_tools([mcp_tool('renew_policy', [packet(policy)], calls),
        mcp_tool('mark_customer_contacted', [packet({'updated': True})], calls)], ENABLED, auth)
    await tools[0].ainvoke({'policy_id': 'ISOLATED-OLD-POLICY'})
    await tools[1].ainvoke({'customer_id': CUSTOMER['id'], 'status': '已出单'})
    assert [name for name, _ in calls] == (['renew_policy', 'mark_customer_contacted'] if same_customer else ['renew_policy'])

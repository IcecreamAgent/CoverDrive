"""Sales entry and customer facts at real production boundaries, all providers isolated."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from test_realtime import audio_start, native as native, output, utterance
import voice_agent.realtime as realtime
from voice_agent import outbound


CUSTOMER = {
    "id": 101, "姓名": "隔离客户甲", "车型": "隔离车型", "到期日期": "2026-12-01",
    "车辆类别": "家用轿车", "座位数": 5,
}
OTHER_CUSTOMER = {"id": 102, "姓名": "不应选择的客户乙", "车型": "另一车型"}
GREETING = "您好，请问是隔离客户甲吗？"
FACT = {"tool": "get_expiring_customers", "arguments": "{}", "result": json.dumps({"客户": [CUSTOMER], "数量": 1}, ensure_ascii=False)}


def business_catalog(monkeypatch, result, *, error=None):
    calls = []

    async def select(arguments):
        calls.append(("get_expiring_customers", arguments))
        if error is not None:
            raise error
        return result

    async def write(arguments):
        calls.append(("create_policy", arguments))
        raise AssertionError("A dial/opening cannot create a policy or mark a customer")

    async def catalog():
        return [SimpleNamespace(name="create_policy", ainvoke=write),
                SimpleNamespace(name="mark_customer_contacted", ainvoke=write),
                SimpleNamespace(name="get_expiring_customers", ainvoke=select)]

    monkeypatch.setattr(outbound, "load_business_tools", catalog)
    return calls


def wrapped_result(kind, payload):
    text = json.dumps(payload, ensure_ascii=False)
    if kind == "dict":
        return payload
    if kind == "json":
        return text
    if kind == "content":
        return [{"type": "text", "text": text}]
    return ToolMessage(content=[{"type": "text", "text": text}], tool_call_id="isolated-select")


@pytest.mark.parametrize("kind", ["dict", "json", "content", "tool_message"])
async def test_opening_uses_one_real_selected_customer_and_only_the_readonly_tool(monkeypatch, kind):
    calls = business_catalog(monkeypatch, wrapped_result(kind, {
        "客户": [CUSTOMER, OTHER_CUSTOMER], "数量": 2,
    }))
    greeting, facts = await outbound.prepare_outbound_customer()
    assert calls == [("get_expiring_customers", {
        "days": 30, "only_uncontacted": True, "random_pick": False, "limit": 1,
    })]
    assert greeting == f'您好，请问是{CUSTOMER["姓名"]}吗？'
    assert greeting.count(CUSTOMER["姓名"]) == 1
    assert facts["customer_salutation"] == "您"
    assert "车险" not in greeting and "续保" not in greeting and "方便" not in greeting
    assert OTHER_CUSTOMER["姓名"] not in greeting
    encoded_facts = json.dumps(facts, ensure_ascii=False)
    assert CUSTOMER["姓名"] in encoded_facts
    assert OTHER_CUSTOMER["姓名"] not in encoded_facts
    assert "已出单" not in greeting and "已生效" not in greeting
    assert facts["customer_name"] == CUSTOMER["姓名"]
    context = outbound.native_opening_context(facts["customer_name"], greeting)
    assert CUSTOMER["姓名"] in context and greeting in context
    assert '"sales_stage": "awaiting_identity_reply"' in context
    assert "尚未自我介绍或询问是否方便" in context and "不得从姓名猜测" in context


@pytest.mark.parametrize("context_kind", ["opening", "native", "deep"])
def test_prompt_contract_keeps_identity_availability_and_consent_separate(context_kind):
    """Inspect supplied instructions, not a scripted model's claimed understanding."""
    from voice_agent.insurance import SYSTEM_PROMPT

    context = {
        "opening": outbound.native_opening_context(CUSTOMER["姓名"], GREETING),
        "native": realtime.INSTRUCTIONS,
        "deep": SYSTEM_PROMPT,
    }[context_kind]
    assert "客户确认本人后才说“您好，我是保险公司的车险服务专员，想跟您聊一下车险续保的事，现在方便吗？”" in context
    assert "然后等待回应" in context
    assert "之后客户确认方便才了解续保需求" in context
    assert "确认本人不等于确认方便，更不等于同意投保或任何业务写入" in context
    assert "客户否认身份时简短道歉并结束沟通，不换客推销，不透露车辆、保费等客户资料" in context
    assert "含糊回应或反问不算身份确认" in context
    assert "明确拒绝时停止推销" in context
    assert "不是永久阶段" in context and "实际对话和公开历史为准" in context
    assert "已经喊出姓名并完成自我介绍" not in context
    assert "awaiting_availability_reply" not in context


@pytest.mark.parametrize("playback_active", [False, True])
@pytest.mark.parametrize("reply", ["是的", "嗯", "不是本人", "你是哪位", "不需要，别再联系"])
async def test_final_identity_reply_reaches_native_turn_without_reselecting_or_business_write(native, monkeypatch, reply, playback_active):
    """Protocol evidence only: the provider, not this fixture, chooses the next response."""
    socket, instance, _, state = native
    calls = business_catalog(monkeypatch, {"客户": [CUSTOMER]})
    await instance.start_outbound()
    await asyncio.wait_for(instance.response_task, 2)
    opening = instance.native_turn
    instance._update_playback(opening.turn_id, playback_active)
    socket.drain()

    turn = await utterance(instance, item="identity-reply", text=reply)

    assert turn.question == reply
    assert instance.insurance_authorization.final_text == reply
    assert instance.insurance_authorization.version == 1
    assert instance.insurance_authorization.selected_customer == CUSTOMER
    assert state.speech == [GREETING] and state.calls == []
    assert [name for name, _ in calls] == ["get_expiring_customers"]
    events = socket.drain()
    assert any(isinstance(event, dict) and event.get("type") == "transcript"
               and event.get("text") == reply and event.get("is_final") for event in events)
    assert not any(isinstance(event, dict) and event.get("type") == "barge_in_rejected"
                   and event.get("reason") == "backchannel" for event in events)
    # Provider output with this input's question identity remains routable.
    # This sentinel proves protocol mapping, not real model comprehension.
    await instance.handle_native_event({"type": "response.output_text.delta",
                                       "question_id": "identity-reply", "response_id": "identity-response",
                                       "delta": "provider-output-sentinel"})
    assert instance.native_turn.answer == "provider-output-sentinel"
    assert not instance.turns["identity-reply"].canceled


async def test_opening_partial_ack_waits_for_final_input_without_canceling_its_upstream_response(native, monkeypatch):
    socket, instance, _, state = native
    business_catalog(monkeypatch, {"客户": [CUSTOMER]})
    await instance.start_outbound()
    await asyncio.wait_for(instance.response_task, 2)
    opening = instance.native_turn
    instance._update_playback(opening.turn_id, True)
    socket.drain()
    await instance.handle_native_event({"type": "conversation.item.input_audio_transcription.started",
                                       "item_id": "identity-reply"})
    await instance.handle_native_event({"type": "conversation.item.input_audio_transcription.delta",
                                       "item_id": "identity-reply", "delta": "是的"})
    assert instance.native_turn is opening and instance.candidate
    assert instance.insurance_authorization.version == 0
    assert not any(isinstance(event, dict) and event.get("type") == "transcript" for event in socket.drain())

    await instance.handle_native_event({"type": "conversation.item.input_audio_transcription.completed",
                                       "item_id": "identity-reply", "text": "是的"})

    assert instance.native_turn is not opening
    assert instance.insurance_authorization.final_text == "是的"
    assert instance.insurance_authorization.version == 1
    assert state.cancels == 0, "final ASR already owns the provider's new response"
    assert any(isinstance(event, dict) and event.get("type") == "transcript"
               and event.get("text") == "是的" and event.get("is_final") for event in socket.drain())


@pytest.mark.parametrize("canceled", [False, True])
@pytest.mark.parametrize("candidate_id", ["", "fresh-reply"])
async def test_opening_reply_never_revives_retained_late_input_mapping(native, monkeypatch, canceled, candidate_id):
    socket, instance, _, state = native
    business_catalog(monkeypatch, {"客户": [CUSTOMER]})
    await instance.start_outbound()
    await asyncio.wait_for(instance.response_task, 2)
    opening = instance.native_turn
    retained = realtime.NativeTurn(canceled=canceled)
    retained.retired.set()
    instance.turns["retired-item"] = retained
    instance._update_playback(opening.turn_id, True)
    socket.drain()
    await instance.handle_native_event({"type": "conversation.item.input_audio_transcription.started",
                                       "item_id": candidate_id})
    await instance.handle_native_event({"type": "conversation.item.input_audio_transcription.completed",
                                       "item_id": "retired-item", "text": "是的"})

    assert instance.native_turn is opening and not opening.canceled
    assert instance.turns["retired-item"] is retained
    assert instance.insurance_authorization.version == 0
    assert not any(isinstance(event, dict) and event.get("type") == "transcript" for event in socket.drain())
    assert state.cancels == 0
    # The unrelated late packet does not poison the current candidate's input.
    await instance.handle_native_event({"type": "conversation.item.input_audio_transcription.completed",
                                       "item_id": "fresh-reply", "text": "是的"})
    assert instance.native_turn.question == "是的"
    assert instance.insurance_authorization.version == 1
    assert instance.turns["retired-item"] is retained


@pytest.mark.parametrize("action", ["interrupt", "reconnect"])
async def test_canceled_or_reconnected_opening_has_no_fast_reply_exemption(native, monkeypatch, action):
    socket, instance, _, state = native
    business_catalog(monkeypatch, {"客户": [CUSTOMER]})
    await instance.start_outbound()
    await asyncio.wait_for(instance.response_task, 2)
    opening = instance.native_turn
    if action == "reconnect":
        await instance.close_native()
        await instance.handle_native_event({"type": "conversation.item.input_audio_transcription.completed",
                                           "item_id": "closed-input", "text": "是的"})
        assert instance.insurance_authorization.version == 0
        await instance.enable_native()
    else:
        await instance.interrupt()
    assert opening.canceled
    instance._update_playback(opening.turn_id, True)
    cancellations = state.cancels
    socket.drain()

    await utterance(instance, item="late-identity-reply", text="是的")

    assert instance.native_turn is opening and opening.canceled
    assert instance.insurance_authorization.version == 0
    events = socket.drain()
    assert any(isinstance(event, dict) and event.get("type") == "barge_in_rejected"
               and event.get("reason") == "backchannel" for event in events)
    assert not any(isinstance(event, dict) and event.get("type") == "transcript" for event in events)
    assert state.cancels == cancellations


async def test_opening_exception_expires_after_first_input_and_keeps_late_items_retired(native, monkeypatch):
    socket, instance, _, state = native
    business_catalog(monkeypatch, {"客户": [CUSTOMER]})
    await instance.start_outbound()
    await asyncio.wait_for(instance.response_task, 2)
    instance._update_playback(instance.native_turn.turn_id, True)
    identity_reply = await utterance(instance, item="identity-reply", text="是的")
    # Ordinary subsequent output uses the existing backchannel rule; this
    # fixture does not attempt to judge the provider's identity semantics.
    await output(instance, item="identity-reply", text="后续普通陈述。")
    await audio_start(instance, item="identity-reply")
    instance._update_playback(identity_reply.turn_id, True)
    socket.drain()
    await utterance(instance, item="ordinary-ack", text="嗯")
    assert instance.native_turn is identity_reply
    assert instance.turns["ordinary-ack"].canceled
    assert instance.insurance_authorization.version == 1
    events = socket.drain()
    assert any(isinstance(event, dict) and event.get("type") == "barge_in_rejected"
               and event.get("reason") == "backchannel" for event in events)
    assert not any(isinstance(event, dict) and event.get("type") == "transcript" for event in events)

    newer = await utterance(instance, item="next-question", text="请先介绍一下服务。")
    socket.drain()
    await instance.handle_native_event({"type": "conversation.item.input_audio_transcription.completed",
                                       "item_id": "identity-reply", "text": "是的"})
    await output(instance, item="identity-reply", response="late-output", text="迟到内容。")
    assert instance.native_turn is newer and newer.question == "请先介绍一下服务。"
    assert instance.insurance_authorization.version == 2
    assert socket.drain() == []
    assert state.cancels == 0


@pytest.mark.parametrize("exchanges", [
    [],
    [("是我", "您好，我是保险公司的车险服务专员，想跟您聊一下车险续保的事，现在方便吗？")],
    [("是我", "您好，我是保险公司的车险服务专员，想跟您聊一下车险续保的事，现在方便吗？"),
     ("现在方便", "您这次续保有什么保障调整需求？")],
    [("不是本人", "抱歉打扰了，再见。")],
])
@pytest.mark.parametrize("reconnect", [False, True])
async def test_reopening_native_preserves_recorded_sales_progress_without_permanent_initial_stage(native, monkeypatch, exchanges, reconnect):
    """Recorded histories exercise reconstruction; no fake semantic response is generated."""
    _, instance, _, state = native
    calls = business_catalog(monkeypatch, {"客户": [CUSTOMER]})
    await instance.start_outbound()
    await asyncio.wait_for(instance.response_task, 2)
    instance.public_history.extend(exchanges)
    expected_history = [(outbound.OUTBOUND_CONTEXT, GREETING), *exchanges]
    assert instance.public_history == expected_history

    if reconnect:
        await instance.close_native()
    else:
        await instance.disable_native()
    await instance.enable_native()
    await instance.start_outbound()

    if reconnect:
        assert state.initial_history[-1] == expected_history
        assert '"sales_stage": "awaiting_identity_reply"' not in state.instructions
        assert "实际对话和公开历史为准，不重复已完成的身份确认、自我介绍或方便确认" in state.instructions
    else:
        assert state.opened == 1
        assert "这是首句时的初始状态，不是永久阶段" in state.hints[-1]
        assert instance.public_history == expected_history
    assert state.speech == [GREETING]
    assert instance.insurance_authorization.selected_customer == CUSTOMER
    assert [name for name, _ in calls] == ["get_expiring_customers"]


@pytest.mark.parametrize("result", [
    {"客户": [], "数量": 0}, {}, {"客户": None}, {"客户": "不是名单"},
    {"客户": [{"id": 101}]}, {"客户": [{"id": None, "姓名": "隔离客户甲"}]},
    {"客户": [{"id": True, "姓名": "隔离客户甲"}]},
    {"客户": [{"id": 101, "姓名": ""}]},
    {"error": "isolated-private-provider-detail"},
    [{"type": "text", "text": '{"error":"isolated-private-provider-detail"}'}],
    ToolMessage(content="isolated-private-provider-detail", status="error", tool_call_id="bad"),
    "unparseable-isolated-private-provider-detail",
])
async def test_empty_error_and_malformed_customer_results_fail_closed(monkeypatch, result):
    business_catalog(monkeypatch, result)
    with pytest.raises(outbound.OutboundUnavailable) as failure:
        await outbound.prepare_outbound_customer()
    assert "isolated-private-provider-detail" not in str(failure.value)
    assert CUSTOMER["姓名"] not in str(failure.value)


async def test_provider_exception_does_not_expose_private_details_or_make_up_customer(monkeypatch):
    calls = business_catalog(monkeypatch, None, error=RuntimeError("isolated-private-provider-detail"))
    with pytest.raises(outbound.OutboundUnavailable) as failure:
        await outbound.prepare_outbound_customer()
    assert "isolated-private-provider-detail" not in str(failure.value)
    assert [name for name, _ in calls] == ["get_expiring_customers"]


async def test_missing_customer_tool_does_not_fall_back_to_a_different_capability(monkeypatch):
    async def catalog():
        return []
    monkeypatch.setattr(outbound, "load_business_tools", catalog)
    with pytest.raises(outbound.OutboundUnavailable):
        await outbound.prepare_outbound_customer()


async def test_dial_starts_assistant_only_sales_opening_without_customer_consent_or_model_task(native, monkeypatch):
    socket, instance, _, state = native
    queries = []

    async def prepare():
        queries.append("selected")
        return GREETING, {**FACT, "customer_name": CUSTOMER["姓名"]}

    monkeypatch.setattr(realtime, "prepare_outbound_customer", prepare)
    await instance.start_outbound()
    task = instance.response_task
    assert task is not None
    await asyncio.wait_for(task, 2)
    events = socket.drain()
    opening = next(event for event in events if isinstance(event, dict) and event.get("type") == "outbound_opening")
    assert opening["turn_id"] == instance.native_turn.turn_id
    assert any(isinstance(event, bytes) for event in events)
    assert not any(isinstance(event, dict) and event.get("type") == "transcript" for event in events)
    assert state.speech == [GREETING] and state.calls == [] and queries == ["selected"]
    assert instance.insurance_authorization.version == 0
    assert instance.insurance_authorization.final_text == ""
    assert CUSTOMER["姓名"] in json.dumps(instance.insurance_authorization.customer_facts, ensure_ascii=False)
    assert instance.native_turn.finished and instance.native_turn.answer == GREETING
    assert state.sequence[:3] == ["instructions", "history", "speech"]
    assert len(state.hints) == 1 and CUSTOMER["姓名"] in state.hints[0]
    assert "客户确认本人后才说" in state.hints[0]
    assert state.history == [(outbound.OUTBOUND_CONTEXT, GREETING)]
    assert instance.native_turn.native_history_seeded

    await instance.start_outbound()
    await instance.disable_native()
    await instance.enable_native()
    await instance.start_outbound()
    assert queries == ["selected"], "closing and reopening input must not select a new sales customer"


@pytest.mark.parametrize("condition", ["not_listening", "previous_dialogue"])
async def test_sales_opening_cannot_take_over_an_existing_or_closed_dialogue(native, monkeypatch, condition):
    socket, instance, _, state = native
    queries = []
    async def prepare():
        queries.append("unexpected")
        return GREETING, {**FACT, "customer_name": CUSTOMER["姓名"]}
    monkeypatch.setattr(realtime, "prepare_outbound_customer", prepare)
    if condition == "not_listening":
        await instance.disable_native()
    else:
        instance.public_history.append(("已有问题", "已有答复"))
    await instance.start_outbound()
    await asyncio.sleep(0)
    assert not queries and not state.speech
    assert not any(isinstance(event, dict) and event.get("type") == "outbound_opening" for event in socket.drain())


async def test_overlapping_dial_requests_reserve_the_opening_before_first_await(native, monkeypatch):
    _, instance, _, state = native
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def prepare():
        calls.append("select")
        entered.set()
        await release.wait()
        return GREETING, {**FACT, "customer_name": CUSTOMER["姓名"]}
    monkeypatch.setattr(realtime, "prepare_outbound_customer", prepare)
    await asyncio.gather(instance.start_outbound(), instance.start_outbound())
    await asyncio.wait_for(entered.wait(), 1)
    assert calls == ["select"]
    release.set()
    await asyncio.wait_for(instance.response_task, 2)
    assert state.speech == [GREETING]


async def test_opening_failure_stays_recoverable_without_fictional_customer_or_speech(native, monkeypatch):
    socket, instance, _, state = native
    async def prepare():
        raise RuntimeError("isolated-private-provider-detail")
    monkeypatch.setattr(realtime, "prepare_outbound_customer", prepare)
    await instance.start_outbound()
    await asyncio.wait_for(instance.response_task, 2)
    events = socket.drain()
    failures = [event for event in events if isinstance(event, dict) and event.get("type") == "error"]
    assert failures and failures[-1]["stage"] == "insurance"
    assert "isolated-private-provider-detail" not in json.dumps(events, ensure_ascii=False)
    assert not state.speech and not instance.insurance_authorization.customer_facts
    assert instance.insurance_authorization.version == 0
    assert any(isinstance(event, dict) and event.get("type") == "state" and event.get("stage") == "idle" for event in events)


async def test_selection_timeout_cancels_the_provider_and_never_broadcasts_an_opening(native, monkeypatch):
    socket, instance, _, state = native
    entered, canceled = asyncio.Event(), asyncio.Event()
    instance.settings.insurance_opening_timeout = .02
    async def prepare():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()
    monkeypatch.setattr(realtime, "prepare_outbound_customer", prepare)
    await instance.start_outbound()
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.wait_for(instance.response_task, 1)
    assert canceled.is_set() and not state.speech
    assert not instance.insurance_authorization.customer_facts
    assert any(isinstance(event, dict) and event.get("type") == "error" and event.get("stage") == "insurance" for event in socket.drain())


@pytest.mark.parametrize("action", ["interrupt", "close_native", "socket_disconnect", "user_speech"])
async def test_new_speech_interrupt_or_disconnect_cancels_selection_and_drops_old_greeting(native, monkeypatch, action):
    socket, instance, _, state = native
    entered, canceled = asyncio.Event(), asyncio.Event()
    async def prepare():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()
    monkeypatch.setattr(realtime, "prepare_outbound_customer", prepare)
    running = asyncio.create_task(instance.run()) if action == "socket_disconnect" else None
    await instance.start_outbound()
    task = instance.response_task
    await asyncio.wait_for(entered.wait(), 1)
    if action == "socket_disconnect":
        await socket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(running, 1)
    elif action == "user_speech":
        await utterance(instance, item="new-customer-turn", text="先等一下，我不续保了。")
    else:
        await getattr(instance, action)()
    await asyncio.wait_for(canceled.wait(), 1)
    assert task.done() and not state.speech
    assert not instance.insurance_authorization.customer_facts
    events = socket.drain()
    assert not any(isinstance(event, bytes) for event in events)
    assert not any(isinstance(event, dict) and event.get("type") == "assistant_done" for event in events)
    if action == "user_speech":
        assert instance.insurance_authorization.version == 1
        assert instance.insurance_authorization.final_text == "先等一下，我不续保了。"


async def test_isolated_sales_model_clarifies_quotes_respects_refusal_and_then_calls_policy(monkeypatch):
    """A scripted model exercises the real graph and business guards, not vendor behavior."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel
    from voice_agent import agent
    from config_fixture import configured_settings
    from voice_agent.insurance import InsuranceAuthorization

    class QuoteArguments(BaseModel):
        customer_id: int
        vehicle_category: str
        seat_count: int

    class PolicyArguments(BaseModel):
        customer_id: int
        quote_id: str
        status: str

    quote = {"quote_id": "isolated-quote-101", "total_premium": 3478.26, "currency": "CNY"}
    policy = {"policy_id": "isolated-policy-101", "policy_no": "isolated-policy-101",
              "quote_id": quote["quote_id"], "status": "已承保", "total_premium": quote["total_premium"]}
    calls = []

    async def calc_premium(customer_id, vehicle_category, seat_count):
        calls.append(("calc_premium", {"customer_id": customer_id,
                                      "vehicle_category": vehicle_category, "seat_count": seat_count}))
        return quote

    async def create_policy(customer_id, quote_id, status):
        calls.append(("create_policy", {"customer_id": customer_id, "quote_id": quote_id, "status": status}))
        return policy

    tools = [
        StructuredTool.from_function(coroutine=calc_premium, name="calc_premium",
                                     description="Calculate a quote in an isolated fixture.", args_schema=QuoteArguments),
        StructuredTool.from_function(coroutine=create_policy, name="create_policy",
                                     description="Create a policy in an isolated fixture.", args_schema=PolicyArguments),
    ]

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    policy_call = {"name": "create_policy", "args": {
        "customer_id": 101, "quote_id": quote["quote_id"], "status": "已生效",
    }}
    model = ToolModel(responses=[
        AIMessage(content="请确认您的车辆类别和座位数。"),
        AIMessage(content="", tool_calls=[{"name": "calc_premium", "args": {
            "customer_id": 101, "vehicle_category": "家用轿车", "seat_count": 5,
        }, "id": "quote-call"}]),
        AIMessage(content=f"本次工具报价合计{quote['total_premium']}元。"),
        AIMessage(content="本次尚未出单，您可以先考虑保障方案。"),
        AIMessage(content="", tool_calls=[{**policy_call, "id": "confirmed-policy-call"}]),
        AIMessage(content=f"工具已返回生效保单{policy['policy_id']}。"),
    ])

    async def catalog():
        return tools

    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kwargs: model)
    monkeypatch.setattr(agent, "load_business_tools", catalog)
    authorization = InsuranceAuthorization()
    authorization.customer_facts = [FACT]
    config = configured_settings(_env_file=None, llm_api_key="isolated-fake-key", insurance_writes_enabled=True)
    session = agent.AgentSession(config, task_only=True, authorization=authorization)

    async def final_customer_turn(text):
        authorization.observe_final(text)
        session._authorization_version = authorization.version
        session.graph = None
        _ = [part async for part in session.stream(text, force_deep=True)]
        return session.final_answer

    answer = await final_customer_turn("可以，续保需要确认哪些信息？")
    assert "车辆类别" in answer and "座位数" in answer and calls == []
    answer = await final_customer_turn("家用轿车，五座，请给我报价。")
    assert str(quote["total_premium"]) in answer
    assert calls == [("calc_premium", {"customer_id": 101, "vehicle_category": "家用轿车", "seat_count": 5})]
    assert authorization.final_text == "家用轿车，五座，请给我报价。"

    answer = await final_customer_turn("有点贵，我要再考虑一下，不要出单。")
    assert "尚未出单" in answer and len(calls) == 1
    assert authorization.final_text == "有点贵，我要再考虑一下，不要出单。"
    answer = await final_customer_turn("我同意投保，请出单。")
    assert policy["policy_id"] in answer
    assert authorization.final_text == "我同意投保，请出单。"
    assert calls[-1] == ("create_policy", {
        "customer_id": 101, "quote_id": quote["quote_id"], "status": "已生效",
    })
    assert len(calls) == 2
    results = [message for message in session._state["messages"]
               if isinstance(message, ToolMessage) and message.tool_call_id == "confirmed-policy-call"]
    assert len(results) == 1 and results[0].status == "success"
    assert json.loads(results[0].content) == policy
    assert authorization.customer_facts == [FACT], "quote/policy results must not replace verified customer identity"


@pytest.mark.parametrize("outbound_flag", [False, True])
async def test_socket_listen_requires_explicit_outbound_flag_before_selecting_customer(native, monkeypatch, outbound_flag):
    socket, instance, _, state = native
    queries = []
    async def prepare():
        queries.append("select")
        return GREETING, {**FACT, "customer_name": CUSTOMER["姓名"]}
    monkeypatch.setattr(realtime, "prepare_outbound_customer", prepare)
    running = asyncio.create_task(instance.run())
    try:
        await socket.until(lambda event: isinstance(event, dict) and event.get("type") == "ready")
        await socket.control({"type": "listen", "outbound": outbound_flag})
        if outbound_flag:
            events = await socket.until(lambda event: isinstance(event, dict) and event.get("type") == "assistant_done")
            assert queries == ["select"] and state.speech == [GREETING]
            assert any(isinstance(event, dict) and event.get("type") == "outbound_opening" for event in events)
        else:
            # FIFO pong proves the preceding listen control has been handled.
            await socket.control({"type": "ping"})
            events = await socket.until(lambda event: isinstance(event, dict) and event.get("type") == "pong")
            assert queries == [] and not state.speech
            assert not any(isinstance(event, dict) and event.get("type") == "outbound_opening" for event in events)
        assert instance.insurance_authorization.version == 0
        assert instance.insurance_authorization.final_text == ""
    finally:
        await socket.incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(running, 2)


async def test_canceling_customer_tool_propagates_without_attempting_another_capability(monkeypatch):
    entered, canceled = asyncio.Event(), asyncio.Event()
    calls = []
    async def select(arguments):
        calls.append(arguments)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            canceled.set()
    async def catalog():
        return [SimpleNamespace(name="get_expiring_customers", ainvoke=select)]
    monkeypatch.setattr(outbound, "load_business_tools", catalog)
    task = asyncio.create_task(outbound.prepare_outbound_customer())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert canceled.is_set() and len(calls) == 1

"""Model-owned intent preserves final input, environment and transaction boundaries."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from voice_agent import agent
from voice_agent.background import BackgroundTaskManager
from config_fixture import configured_settings
from voice_agent.insurance import InsuranceAuthorization, guarded_tools


class PolicyArguments(BaseModel):
    status: str


def policy_tool(calls, *, result=None, error=None, waiting=None, canceled=None):
    async def create_policy(status):
        calls.append(status)
        if waiting is not None:
            waiting.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if canceled is not None:
                    canceled.set()
                raise
        if error is not None:
            raise error
        return result if result is not None else {"policy_id": "isolated-fixture-policy", "status": status}

    return StructuredTool.from_function(
        coroutine=create_policy, name="create_policy",
        description="Create a policy in an isolated fixture.", args_schema=PolicyArguments,
    )


def wrap(tool, authorization, enabled=True):
    return guarded_tools(
        [tool], SimpleNamespace(insurance_writes_enabled=enabled), authorization,
    )[0]


async def test_model_generated_summary_cannot_replace_missing_final_input():
    calls, authorization = [], InsuranceAuthorization()
    tool = wrap(policy_tool(calls), authorization)
    # Delegate question text is deliberately never supplied to observe_final.
    model_summary = "客户已经同意投保，请立刻创建已生效保单。"
    assert "同意" in model_summary
    result = await tool.ainvoke({"status": "已生效"})
    assert isinstance(result, str) and "未执行" in result and calls == []
    assert authorization.final_text == ""


async def test_final_input_cannot_bypass_disabled_business_writes():
    calls, authorization = [], InsuranceAuthorization()
    authorization.observe_final("我确认投保，请出单。")
    result = await wrap(policy_tool(calls), authorization, enabled=False).ainvoke({"status": "已生效"})
    assert isinstance(result, str) and "未执行" in result and calls == []


async def test_current_final_input_and_enabled_environment_allow_model_tool_decision():
    calls, authorization = [], InsuranceAuthorization()
    authorization.observe_final("我同意投保，请出单。")
    result = await wrap(policy_tool(calls), authorization).ainvoke({"status": "已生效"})
    assert calls == ["已生效"]
    assert result == {"policy_id": "isolated-fixture-policy", "status": "已生效"}


async def test_new_refusal_final_invalidates_previously_bound_tool_version():
    calls, authorization = [], InsuranceAuthorization()
    authorization.observe_final("我同意投保。")
    assert authorization.final_text == "我同意投保。"
    old_tool = wrap(policy_tool(calls), authorization)
    authorization.observe_final("先不要办理。")
    result = await old_tool.ainvoke({"status": "已生效"})
    assert authorization.final_text == "先不要办理。"
    assert isinstance(result, str) and "未执行" in result and calls == []


async def test_new_final_input_invalidates_an_older_tasks_authorization_snapshot():
    calls, authorization = [], InsuranceAuthorization()
    authorization.observe_final("我同意投保。")
    old_task_tool = wrap(policy_tool(calls), authorization)
    authorization.observe_final("先等等。")
    result = await old_task_tool.ainvoke({"status": "已生效"})
    assert isinstance(result, str) and "未执行" in result and calls == []


@pytest.mark.parametrize("text", ["就照你刚才报的办吧。", "行，帮我续上。", "方案可以，给我办理。"])
async def test_model_policy_decision_is_not_blocked_by_sentence_whitelist(text):
    calls, authorization = [], InsuranceAuthorization()
    authorization.observe_final(text)
    result = await wrap(policy_tool(calls), authorization).ainvoke({"status": "已生效"})
    assert authorization.final_text == text
    assert result == {"policy_id": "isolated-fixture-policy", "status": "已生效"}
    assert calls == ["已生效"]


def test_final_input_tracker_does_not_classify_consent_or_refusal():
    authorization = InsuranceAuthorization()
    authorization.observe_final("不要续保了，请停止推销。")
    authorization.observe_final("嗯。")
    assert authorization.version == 2 and authorization.final_text == "嗯。"
    assert not hasattr(authorization, "consent") and not hasattr(authorization, "refused")


@pytest.mark.parametrize("failure", [
    {"error": "isolated-provider-sensitive-error"},
    '{"error":"isolated-provider-sensitive-error"}',
    [{"type": "text", "text": '{"error":"isolated-provider-sensitive-error"}'}],
    ToolMessage(content="isolated-provider-sensitive-error", tool_call_id="upstream", status="error"),
])
async def test_real_tool_error_result_remains_an_error_without_policy_success(failure):
    calls, authorization = [], InsuranceAuthorization()
    authorization.observe_final("我同意投保。")
    events = []

    async def observe(event):
        events.append(event)

    result = await wrap(policy_tool(calls, result=failure), authorization).ainvoke({
        "type": "tool_call", "id": "failed-policy-call", "name": "create_policy",
        "args": {"status": "已生效"},
    }, config={"callbacks": [agent._ToolStartCallback(None, observe)]})
    assert isinstance(result, ToolMessage) and result.status == "error"
    assert "没有有效结果" in result.content
    assert "policy_id" not in result.content and "isolated-provider-sensitive-error" not in result.content
    assert calls == ["已生效"]
    assert events[-1]["status"] == "failed" and events[-1].get("output", "") == ""
    assert authorization.customer_facts == []


async def test_provider_exception_propagates_without_fabricating_success():
    calls, authorization = [], InsuranceAuthorization()
    authorization.observe_final("我同意投保。")
    with pytest.raises(RuntimeError, match="isolated failure"):
        await wrap(policy_tool(calls, error=RuntimeError("isolated failure")), authorization).ainvoke({"status": "已生效"})
    assert calls == ["已生效"]


async def test_cancellation_reaches_inflight_tool_without_a_success_result():
    calls, authorization = [], InsuranceAuthorization()
    authorization.observe_final("我同意投保。")
    waiting, canceled = asyncio.Event(), asyncio.Event()
    tool = wrap(policy_tool(calls, waiting=waiting, canceled=canceled), authorization)
    task = asyncio.create_task(tool.ainvoke({"status": "已生效"}))
    await asyncio.wait_for(waiting.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert canceled.is_set() and calls == ["已生效"]


def test_unrelated_tools_are_not_exposed_to_the_insurance_agent():
    async def unrelated(status):
        raise AssertionError("Unrelated capability must not execute")

    tool = StructuredTool.from_function(
        coroutine=unrelated, name="unrelated_external_write",
        description="Unavailable capability.", args_schema=PolicyArguments,
    )
    assert guarded_tools([tool], SimpleNamespace(insurance_writes_enabled=True), InsuranceAuthorization()) == []


@pytest.mark.parametrize("previous_final", ["请先介绍方案。", "不要投保，请停止推销。"])
async def test_queued_task_cannot_borrow_a_later_turns_final_input(monkeypatch, previous_final):
    """Reserve authorization before queueing, not when an execution slot opens."""
    calls, observed_results = [], []
    authorization = InsuranceAuthorization()
    authorization.observe_final(previous_final)
    reserved_version = authorization.version
    original = policy_tool(calls)
    settings = configured_settings(
        _env_file=None, llm_api_key="fake-key", insurance_tools_enabled=False,
        insurance_writes_enabled=True, background_max_concurrency=1,
    )

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def task_model(**kwargs):
        return ToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "create_policy", "args": {"status": "已生效"}, "id": "fixture-policy-call",
            }]),
            AIMessage(content="隔离工具执行已结束，以工具返回状态为准。"),
        ])

    async def graph_for_policy(self):
        protected = guarded_tools(
            [original], settings, self.authorization,
            expected_version=self._authorization_version,
        )[0]

        async def observe_protected(**arguments):
            result = await protected.ainvoke(arguments)
            observed_results.append(result)
            return result

        instrumented = StructuredTool.from_function(
            coroutine=observe_protected, name=protected.name,
            description=protected.description, args_schema=protected.args_schema,
        )
        return agent.create_deep_agent(
            model=self._model, tools=[instrumented], system_prompt="仅调用本地隔离出单工具。",
        )

    monkeypatch.setattr(agent, "ChatOpenAI", task_model)
    monkeypatch.setattr(agent.AgentSession, "_build_graph", graph_for_policy)

    def factory(config, *, task_only):
        return agent.AgentSession(config, task_only=task_only, authorization=authorization)

    async def publish(record):
        pass

    manager = BackgroundTaskManager(
        settings, publish, agent_factory=factory, authorization=authorization,
    )
    manager._semaphore = asyncio.Semaphore(0)
    try:
        old = await manager.submit("旧任务试图创建生效保单", "turn-old", "call-old", [])
        await asyncio.sleep(0)
        assert old.authorization_version == reserved_version
        assert old.agent is None and old.status == "queued"

        authorization.observe_final("我同意投保，请出单。")
        assert authorization.final_text == "我同意投保，请出单。" and authorization.version > reserved_version
        manager._semaphore.release()
        await asyncio.wait_for(old.worker, 2)

        assert old.status == "ready"
        assert old.agent._authorization_version == reserved_version
        assert calls == []
        assert len(observed_results) == 1 and isinstance(observed_results[0], str)
        assert "未执行" in observed_results[0]
        assert "policy_id" not in observed_results[0]

        # A separately submitted current-turn task uses its own final input.
        current = await manager.submit("本轮明确同意出单", "turn-current", "call-current", [])
        await asyncio.wait_for(current.worker, 2)
        assert current.authorization_version == authorization.version
        assert current.status == "ready" and calls == ["已生效"]
        assert observed_results[-1] == {
            "policy_id": "isolated-fixture-policy", "status": "已生效",
        }
    finally:
        await manager.close()


@pytest.mark.parametrize("writes_enabled,final", [
    (False, "我同意投保。"), (False, "就照刚才的办。"), (True, ""),
])
async def test_blocked_tool_call_has_error_status_and_failed_execution_callback(writes_enabled, final):
    calls, events = [], []
    authorization = InsuranceAuthorization()
    authorization.observe_final(final)
    protected = wrap(policy_tool(calls), authorization, enabled=writes_enabled)

    async def observe(event):
        events.append(event)

    callback = agent._ToolStartCallback(None, observe)
    result = await protected.ainvoke({
        "type": "tool_call", "id": "blocked-policy-call", "name": "create_policy",
        "args": {"status": "已生效"},
    }, config={"callbacks": [callback]})
    assert isinstance(result, ToolMessage)
    assert result.status == "error" and "未执行" in result.content
    assert result.tool_call_id == "blocked-policy-call" and calls == []
    assert events[0]["status"] == "running"
    assert events[-1]["status"] == "failed"
    assert events[-1].get("output", "") == ""

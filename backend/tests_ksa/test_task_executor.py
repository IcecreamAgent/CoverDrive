"""Transactional task execution after adapting KSA to insurance capabilities."""

import asyncio
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from voice_agent import agent
from config_fixture import configured_settings


def executor(monkeypatch):
    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kwargs: object())
    return agent.AgentSession(
        configured_settings(_env_file=None, llm_api_key="fake-key", insurance_tools_enabled=False),
        task_only=True,
    )


async def test_only_the_committed_final_answer_is_a_task_result(monkeypatch):
    instance = executor(monkeypatch)

    class Graph:
        async def astream(self, state, **kwargs):
            yield "messages", (AIMessageChunk(content="准备核对工具结果。"), {})
            yield "messages", (AIMessageChunk(content=[{"type": "text", "text": "实际结果。"}]), {})
            yield "values", {"messages": [
                *state["messages"],
                AIMessage(content="中间规划。", tool_calls=[{"name": "calc_premium", "args": {}, "id": "quote"}]),
                ToolMessage(content="来自隔离工具的原文。", tool_call_id="quote"),
                AIMessage(content="实际结果。"),
            ]}

    instance.graph = Graph()
    chunks = [part async for part in instance.stream("核对方案", force_deep=True, context="不可信上下文资料")]
    assert chunks == ["准备核对工具结果。", "实际结果。"]
    assert instance.final_answer == "实际结果。"
    assert instance._state["messages"][-1].content == "实际结果。"
    assert not any("不可信上下文资料" in str(message.content) for message in instance._state["messages"])
    assert instance.router is None and instance._fast_model is None


async def test_interrupted_graph_cannot_commit_partial_text_or_tool_state(monkeypatch):
    instance = executor(monkeypatch)
    instance.remember_exchange("先前问题", "先前公开答案")
    previous = deepcopy(instance._state)
    entered = asyncio.Event()

    class Graph:
        async def astream(self, state, **kwargs):
            yield "messages", (AIMessageChunk(content="尚未完成的规划。"), {})
            entered.set()
            await asyncio.Event().wait()

    instance.graph = Graph()

    async def consume():
        return [part async for part in instance.stream("新任务", force_deep=True)]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert instance._state == previous
    assert instance.final_answer == ""


async def test_tool_cycle_without_a_final_answer_never_reuses_an_old_answer(monkeypatch):
    instance = executor(monkeypatch)
    instance.remember_exchange("旧问题", "旧答案")
    prior_state = deepcopy(instance._state)

    class Graph:
        async def astream(self, state, **kwargs):
            yield "messages", (AIMessageChunk(content="我准备计算。"), {})
            yield "values", {"messages": [
                *state["messages"],
                AIMessage(content="我准备计算。", tool_calls=[{"name": "calc_premium", "args": {}, "id": "incomplete"}]),
                ToolMessage(content="尚未形成最终答案", tool_call_id="incomplete"),
            ]}

    instance.graph = Graph()
    assert [part async for part in instance.stream("新报价请求", force_deep=True)] == ["我准备计算。"]
    assert instance.final_answer == ""
    assert instance._state == prior_state


@pytest.mark.parametrize("context", [None, "x" * 16001])
async def test_invalid_context_fails_before_any_graph_execution(monkeypatch, context):
    instance = executor(monkeypatch)
    with pytest.raises(ValueError, match="bounded"):
        _ = [part async for part in instance.stream("任务", context=context)]
    assert instance.graph is None


async def test_disabled_insurance_tools_do_not_connect_to_mcp(monkeypatch):
    instance = executor(monkeypatch)
    captured = {}

    def build(**values):
        captured.update(values)
        return object()

    async def never_load_business_tools():
        raise AssertionError("Disabled MCP tools must not trigger discovery")

    monkeypatch.setattr(agent, "create_deep_agent", build)
    monkeypatch.setattr(agent, "load_business_tools", never_load_business_tools)
    graph = await instance._build_graph()
    assert graph is not None
    assert [tool.name for tool in captured["tools"]] == ["get_current_time"]
    assert captured["model"] is instance._model
    assert "不得" in captured["system_prompt"]


def test_task_model_uses_reference_streaming_and_thinking_parameters(monkeypatch):
    captured = {}

    def model(**values):
        captured.update(values)
        return object()

    monkeypatch.setattr(agent, "ChatOpenAI", model)
    config = configured_settings(_env_file=None, llm_api_key="fake-key", insurance_tools_enabled=False)
    instance = agent.AgentSession(config, task_only=True)
    assert instance.router is None and instance._fast_model is None
    assert captured["streaming"] is True
    assert captured["use_responses_api"] is False
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["model"] == config.llm_model_id

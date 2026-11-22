"""Independent task state, admission, cancellation and session cleanup."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool

from voice_agent import agent
from voice_agent.background import MAX_TRACE_CALLS, BackgroundTaskManager, _trace_text
from config_fixture import configured_settings


def settings(**values):
    return SimpleNamespace(
        **{
            "background_max_tasks": 8,
            "background_max_concurrency": 3,
            "turn_timeout": 2,
            "max_text_length": 4000,
            **values,
        }
    )


class Agents:
    def __init__(self):
        self.instances = []
        self.entered = asyncio.Queue()
        self.running = set()
        self.canceled = set()
        self.gates = {}
        self.errors = {}
        self.ignore_cancel = set()
        self.empty = set()
        self.tools = set()
        self.tool_gates = {}

    def gate(self, question):
        return self.gates.setdefault(question, asyncio.Event())

    def tool_gate(self, question):
        return self.tool_gates.setdefault(question, asyncio.Event())

    def __call__(self, config, *, task_only):
        assert task_only
        owner = self

        class Agent:
            def __init__(self):
                self.history = []
                self.question = None
                self.on_tool_start = None
                self.context = ""

            def remember_exchange(self, question, answer):
                self.history.append((question, answer))

            async def stream(self, question, *, force_deep, context=""):
                assert force_deep
                self.question = question
                self.context = context
                owner.running.add(question)
                owner.entered.put_nowait(question)
                try:
                    if question in owner.tools:
                        await owner.tool_gate(question).wait()
                        if self.on_tool_start is not None:
                            await self.on_tool_start()
                    try:
                        await owner.gate(question).wait()
                    except asyncio.CancelledError:
                        owner.canceled.add(question)
                        if question not in owner.ignore_cancel:
                            raise
                    if question in owner.errors:
                        raise owner.errors[question]
                    if question not in owner.empty:
                        yield question
                        yield "的实际结果。"
                finally:
                    owner.running.discard(question)

        instance = Agent()
        self.instances.append(instance)
        return instance


class Updates:
    def __init__(self):
        self.events = []

    async def __call__(self, record):
        self.events.append((record.task_id, record.status, record.result, record.error))


async def entered(agents):
    return await asyncio.wait_for(agents.entered.get(), 1)


async def submitted(manager, name="任务一", **values):
    return await manager.submit(name, "turn-1", "call-" + name, [], **values)


async def test_deep_reasoning_without_tools_returns_inline_and_never_publishes_a_task():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    caller = asyncio.create_task(manager.start_delegation(
        "按已知天气规划行程", "turn-1", "call-1", [("前文", "答复")],
        context="舟山明天有雨，来源：当地气象台。",
    ))
    assert await entered(agents) == "按已知天气规划行程"
    assert manager.records == {} and manager.active_tasks() == []
    assert updates.events == [] and not caller.done()
    agents.gate("按已知天气规划行程").set()
    record = await caller
    assert record.status == "completed" and not record.promoted.is_set()
    assert record.result == "按已知天气规划行程的实际结果。"
    assert record.agent.history == [("前文", "答复")]
    assert record.agent.context == "舟山明天有雨，来源：当地气象台。"
    assert manager.records == {} and updates.events == [] and not manager.ready.is_set()
    await manager.close()


async def test_real_tool_start_promotes_the_same_worker_and_survives_foreground_cancellation():
    agents, updates = Agents(), Updates()
    agents.tools.add("查询天气")
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)

    async def foreground():
        record = await manager.start_delegation("查询天气", "turn-1", "call-1", [])
        await asyncio.Event().wait()
        return record

    caller = asyncio.create_task(foreground())
    assert await entered(agents) == "查询天气"
    record = manager.by_call["call-1"]
    worker = record.worker
    assert manager.records == {} and updates.events == []
    agents.tool_gate("查询天气").set()
    await asyncio.wait_for(record.promoted.wait(), 1)
    assert manager.records == {record.task_id: record}
    assert record.status == "running"
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert record.worker is worker and not worker.done()
    assert "查询天气" not in agents.canceled
    agents.gate("查询天气").set()
    await worker
    assert record.status == "ready" and record.result == "查询天气的实际结果。"
    assert [event[1] for event in updates.events] == ["running", "ready"]
    await manager.close()


async def test_cancel_before_any_tool_starts_cancels_hidden_reasoning_without_notification():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    caller = asyncio.create_task(manager.start_delegation("深入解释", "turn-1", "call-1", []))
    await entered(agents)
    record = manager.by_call["call-1"]
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert record.worker.done() and record.status == "canceled"
    assert "深入解释" in agents.canceled
    assert manager.records == {} and updates.events == [] and not manager.ready.is_set()
    await manager.close()


async def test_duplicate_inline_delegations_share_one_worker_and_cancel_only_last_waiter():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    first = asyncio.create_task(manager.start_delegation("深入解释", "turn-1", "same", []))
    second = asyncio.create_task(manager.start_delegation("深入解释", "turn-1", "same", []))
    await entered(agents)
    record = manager.by_call["same"]
    assert record.waiters == 2 and len(agents.instances) == 1
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert record.waiters == 1 and not record.worker.done()
    agents.gate("深入解释").set()
    assert await second is record
    assert record.status == "completed" and record.waiters == 0
    assert await manager.start_delegation("重复包", "turn-1", "same", []) is record
    assert len(agents.instances) == 1 and updates.events == []
    await manager.close()


async def test_background_limit_does_not_block_inline_answer_but_prevents_tool_promotion():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_tasks=1), updates, agent_factory=agents
    )
    existing = await submitted(manager, "原后台任务")
    await entered(agents)
    inline = asyncio.create_task(manager.start_delegation("直接深入回答", "turn-2", "call-2", []))
    await entered(agents)
    agents.gate("直接深入回答").set()
    answer = await inline
    assert answer.status == "completed" and answer.task_id not in manager.records
    agents.tools.add("新查询")
    query = asyncio.create_task(manager.start_delegation("新查询", "turn-3", "call-3", []))
    await entered(agents)
    agents.tool_gate("新查询").set()
    rejected = await query
    assert rejected.status == "failed" and not rejected.promoted.is_set()
    assert rejected.error and manager.records == {existing.task_id: existing}
    assert all(event[0] == existing.task_id for event in updates.events)
    await manager.close()


async def test_close_cancels_hidden_delegations_and_releases_their_waiters():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    caller = asyncio.create_task(manager.start_delegation("深入解释", "turn-1", "call-1", []))
    await entered(agents)
    record = manager.by_call["call-1"]
    await manager.close()
    assert await caller is record
    assert record.worker.done() and record.status == "canceled"
    assert manager.records == {} and manager.by_call == {} and updates.events == []


@pytest.mark.parametrize("mode", ["failure", "empty", "timeout"])
async def test_failure_before_tools_returns_inline_without_queueing_a_notification(mode):
    agents, updates = Agents(), Updates()
    if mode == "failure":
        agents.errors["纯推理"] = RuntimeError("private provider detail")
    elif mode == "empty":
        agents.empty.add("纯推理")
    manager = BackgroundTaskManager(settings(turn_timeout=0.03), updates, agent_factory=agents)
    caller = asyncio.create_task(manager.start_delegation("纯推理", "turn-1", "call-1", []))
    await entered(agents)
    if mode != "timeout":
        agents.gate("纯推理").set()
    record = await caller
    assert record.status == "failed" and record.error == "深度处理失败，请重试。"
    assert not record.result and not record.promoted.is_set()
    assert not manager.records and not updates.events and not manager.ready.is_set()
    await manager.close()


async def test_real_deepagents_tool_execution_promotes_on_the_session_loop(monkeypatch):
    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    release, started = asyncio.Event(), asyncio.Event()

    @tool
    async def search_probe() -> str:
        """Return deterministic weather data after an external-tool boundary."""
        started.set()
        await release.wait()
        return "舟山明天有雨，来源：本地测试气象台。"

    model = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "search_probe", "args": {}, "id": "probe-1",
        }]),
        AIMessage(content="舟山明天有雨，行程优先安排室内活动。"),
    ])
    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kwargs: model)

    async def graph_for_probe(self):
        return agent.create_deep_agent(
            model=model, tools=[search_probe], system_prompt="只返回本地隔离工具的实际结果。",
        )

    monkeypatch.setattr(agent.AgentSession, "_build_graph", graph_for_probe)
    updates = Updates()
    manager = BackgroundTaskManager(
        configured_settings(_env_file=None), updates, agent_factory=agent.AgentSession
    )
    record = await asyncio.wait_for(
        manager.start_delegation("查询舟山天气", "turn-1", "call-1", []), 1
    )
    assert record.promoted.is_set() and manager.records[record.task_id] is record
    await asyncio.wait_for(started.wait(), 1)
    assert record.status == "running" and not record.worker.done()
    release.set()
    await record.worker
    assert record.status == "ready" and record.result == "舟山明天有雨，行程优先安排室内活动。"
    assert [event[1] for event in updates.events] == ["running", "running", "running", "ready"]
    assert len(record.calls) == 1 and record.calls[0]["status"] == "completed"
    assert record.calls[0]["name"] == "search_probe"
    assert record.calls[0]["output"] == "舟山明天有雨，来源：本地测试气象台。"
    await manager.close()


async def test_tasks_run_with_separate_agents_and_history_snapshots():
    agents, updates, ready = Agents(), Updates(), asyncio.Event()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents, ready=ready)
    history = [["已完成问题", "已完成回答"]]
    first = await manager.submit("找民宿", "turn-one", "call-one", history, title="免费停车民宿")
    second = await manager.submit("查天气", "turn-two", "call-two", history)
    history[0][1] = "提交后变动不应污染任务"
    history.append(["后来问题", "后来回答"])
    assert await entered(agents) == "找民宿"
    assert await entered(agents) == "查天气"
    assert first.agent is not second.agent
    assert first.agent.history == second.agent.history == [("已完成问题", "已完成回答")]
    assert first.origin_turn_id == "turn-one" and second.origin_turn_id == "turn-two"
    assert first.title == "免费停车民宿" and second.title == "查天气"
    assert len(first.task_id) == 12 and int(first.task_id, 16)
    agents.gate("查天气").set()
    await second.worker
    assert second.status == "ready" and first.status == "running"
    agents.gate("找民宿").set()
    await first.worker
    assert manager.ready is ready and ready.is_set()
    assert first.result == "找民宿的实际结果。" and second.result == "查天气的实际结果。"
    assert manager.active_tasks() == [first, second]
    for record in (first, second):
        assert [event[1] for event in updates.events if event[0] == record.task_id] == [
            "queued", "running", "ready",
        ]
    await manager.close()


async def test_duplicate_calls_reserve_only_one_worker_even_when_capacity_is_full():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_tasks=1), updates, agent_factory=agents
    )
    records = await asyncio.gather(*[
        manager.submit("同一个任务", "turn-1", "same-call", []) for _ in range(3)
    ])
    assert records[0] is records[1] is records[2]
    assert manager.by_call["same-call"] is records[0]
    assert len(manager.records) == 1
    await entered(agents)
    assert len(agents.instances) == 1
    with pytest.raises(ValueError, match="上限"):
        await submitted(manager, "第二个任务")
    await manager.close()


async def test_concurrency_is_bounded_but_ready_results_keep_their_admission_slot():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_tasks=3, background_max_concurrency=1), updates,
        agent_factory=agents,
    )
    records = [await submitted(manager, name) for name in ("一", "二", "三")]
    assert await entered(agents) == "一"
    assert [record.status for record in records] == ["running", "queued", "queued"]
    assert agents.running == {"一"}
    agents.gate("一").set()
    await records[0].worker
    assert await entered(agents) == "二"
    assert records[0].status == "ready" and agents.running == {"二"}
    await manager.update(records[0], "announcing")
    with pytest.raises(ValueError, match="上限"):
        await submitted(manager, "四")
    await manager.update(records[0], "completed")
    fourth = await submitted(manager, "四")
    assert fourth.status == "queued"
    await manager.close()
    assert not agents.running


async def test_canceling_a_running_task_awaits_cleanup_without_canceling_other_work():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    first, second = await submitted(manager, "一"), await submitted(manager, "二")
    await entered(agents)
    await entered(agents)
    assert await manager.cancel(first.task_id)
    assert first.status == "canceled" and first.worker.done()
    assert "一" in agents.canceled and agents.running == {"二"}
    assert not await manager.cancel(first.task_id)
    assert not await manager.cancel("unknown")
    agents.gate("二").set()
    await second.worker
    assert second.status == "ready"
    await manager.close()


async def test_canceling_queued_work_never_constructs_an_agent():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_concurrency=1), updates, agent_factory=agents
    )
    first, queued = await submitted(manager, "运行"), await submitted(manager, "排队")
    await entered(agents)
    assert queued.status == "queued"
    assert await manager.cancel(queued.task_id)
    assert queued.worker.done() and queued.agent is None
    agents.gate("运行").set()
    await first.worker
    assert len(agents.instances) == 1
    await manager.close()


async def test_cancel_wins_even_if_agent_yields_a_result_after_cancellation():
    agents, updates = Agents(), Updates()
    agents.ignore_cancel.add("取消竞态")
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    record = await submitted(manager, "取消竞态")
    await entered(agents)
    assert await manager.cancel(record.task_id)
    assert record.worker.done() and record.status == "canceled" and not record.result
    await manager.update(record, "ready")
    await manager.update(record, "completed")
    assert record.status == "canceled"
    assert not any(event[1] == "ready" for event in updates.events)
    await manager.close()


async def test_a_ready_result_can_be_canceled_before_announcement():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    record = await submitted(manager)
    await entered(agents)
    agents.gate(record.question).set()
    await record.worker
    assert record.status == "ready"
    assert await manager.cancel(record.task_id)
    assert record.status == "canceled" and manager.active_tasks() == []
    await manager.update(record, "announcing")
    assert record.status == "canceled"
    await manager.close()


@pytest.mark.parametrize("status", ["queued", "running", "ready", "announcing"])
async def test_replacement_invalidates_only_the_selected_undelivered_version(status):
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    weather = await submitted(manager, "上海天气")
    old = await submitted(manager, "推荐扫地机器人")
    if status != "queued":
        await entered(agents)
        await entered(agents)
        if status in {"ready", "announcing"}:
            agents.gate(old.question).set()
            await old.worker
        if status == "announcing":
            await manager.update(old, status)
    assert old.status == status
    new = await submitted(
        manager, "推荐4000到6000元扫地机器人", replaces_task_id=old.task_id,
    )
    # submit performs the complete swap before yielding to cancellation publish.
    assert old.status == "canceled" and old.replaced_by_task_id == new.task_id
    assert new.replaces_task_id == old.task_id and new.status == "queued"
    assert weather.status in {"queued", "running"} and not weather.replaced_by_task_id
    agents.gate(new.question).set()
    await new.worker
    assert new.status == "ready" and old.status == "canceled"
    await manager.update(old, "announcing")
    assert old.status == "canceled"
    assert not new.context and new.agent.history == []
    await manager.close()


async def test_replacement_uses_the_old_slot_when_background_capacity_is_full():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_tasks=1), updates, agent_factory=agents,
    )
    old = await submitted(manager, "原推荐")
    await entered(agents)
    new = await submitted(manager, "预算版推荐", replaces_task_id=old.task_id)
    assert manager.active_tasks() == [new] and old.status == "canceled"
    await manager.close()


async def test_hidden_replacement_keeps_its_slot_until_tool_promotion_despite_competing_work():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_tasks=2), updates, agent_factory=agents,
    )
    weather = await submitted(manager, "独立天气")
    old = await submitted(manager, "原推荐")
    await entered(agents)
    await entered(agents)
    agents.tools.update({"其他查询", "预算版推荐"})
    competitor = asyncio.create_task(manager.start_delegation(
        "其他查询", "turn-2", "competing-call", [],
    ))
    assert await entered(agents) == "其他查询"
    replacing = asyncio.create_task(manager.start_delegation(
        "预算版推荐", "turn-3", "replacement-call", [], replaces_task_id=old.task_id,
    ))
    assert await entered(agents) == "预算版推荐"
    successor = manager.by_call["replacement-call"]
    assert old.status == "canceled" and not successor.promoted.is_set()
    assert manager.active_tasks() == [weather]
    # Both direct submission and a previously admitted hidden reasoning call
    # must respect the slot transferred to the replacement, before it promotes.
    with pytest.raises(ValueError, match="上限"):
        await submitted(manager, "直接新增查询")
    agents.tool_gate("其他查询").set()
    rejected = await competitor
    assert rejected.status == "failed" and not rejected.promoted.is_set()
    assert manager.active_tasks() == [weather] and successor.status == "running"
    agents.tool_gate("预算版推荐").set()
    assert await replacing is successor
    assert successor.promoted.is_set() and manager.active_tasks() == [weather, successor]
    agents.gate("预算版推荐").set()
    await successor.worker
    assert successor.status == "ready" and weather.status == "running"
    assert old.status == "canceled"
    await manager.close()


@pytest.mark.parametrize("outcome", ["completed", "failed", "canceled"])
async def test_hidden_replacement_releases_its_reserved_slot_when_it_finishes_without_tools(
    outcome,
):
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_tasks=1), updates, agent_factory=agents,
    )
    old = await submitted(manager, "原查询")
    await entered(agents)
    replacing = asyncio.create_task(manager.start_delegation(
        "纯思考新版", "turn-2", "replacement-call", [], replaces_task_id=old.task_id,
    ))
    assert await entered(agents) == "纯思考新版"
    successor = manager.by_call["replacement-call"]
    assert old.status == "canceled" and not successor.promoted.is_set()
    with pytest.raises(ValueError, match="上限"):
        await submitted(manager, "预约尚未释放")
    if outcome == "canceled":
        replacing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replacing
    else:
        if outcome == "failed":
            agents.errors["纯思考新版"] = RuntimeError("test failure")
        agents.gate("纯思考新版").set()
        assert await replacing is successor
    assert successor.status == outcome and not successor.promoted.is_set()
    available = await submitted(manager, "预约释放后的查询")
    assert manager.active_tasks() == [available] and available.status == "queued"
    await manager.close()


async def test_replacement_hidden_capacity_rejection_preserves_the_original_task():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_tasks=1), updates, agent_factory=agents,
    )
    old = await submitted(manager, "原查询")
    await entered(agents)
    inline = asyncio.create_task(manager.start_delegation("独立推理", "turn-2", "inline", []))
    await entered(agents)
    with pytest.raises(ValueError, match="上限"):
        await manager.start_delegation(
            "新条件查询", "turn-3", "replacement", [], replaces_task_id=old.task_id,
        )
    assert old.status == "running" and not old.replaced_by_task_id
    assert "replacement" not in manager.by_call
    agents.gate("独立推理").set()
    await inline
    await manager.close()


async def test_replacement_invalid_history_snapshot_preserves_the_original_task():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    old = await submitted(manager, "原查询")
    await entered(agents)

    def broken_history():
        yield ("之前", "答复")
        raise ValueError("invalid history snapshot")

    with pytest.raises(ValueError, match="history"):
        await manager.submit(
            "新查询", "turn-2", "replacement", broken_history(), replaces_task_id=old.task_id,
        )
    assert old.status == "running" and not old.replaced_by_task_id
    assert manager.active_tasks() == [old] and "replacement" not in manager.by_call
    await manager.close()


@pytest.mark.parametrize("invalid", [
    {"question": ""}, {"question": 42}, {"question": "问" * 4001},
    {"title": None}, {"title": "题" * 81}, {"context": 42}, {"context": "文" * 16001},
    {"replaces_task_id": 42}, {"replaces_task_id": None},
    {"replaces_task_id": "x" * 129}, {"replaces_task_id": "other-session-task"},
])
async def test_replacement_validation_failure_preserves_the_original_task(invalid):
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    old = await submitted(manager, "原推荐")
    await entered(agents)
    request = dict(
        question="预算版推荐", origin_turn_id="turn-2", call_id="new-call", history=[],
        replaces_task_id=old.task_id,
    )
    request.update(invalid)
    with pytest.raises(ValueError):
        await manager.submit(**request)
    assert old.status == "running" and not old.replaced_by_task_id
    assert manager.active_tasks() == [old] and "new-call" not in manager.by_call
    assert not old.worker.done() and "原推荐" not in agents.canceled
    await manager.close()


@pytest.mark.parametrize("terminal", ["completed", "failed", "canceled"])
async def test_replacement_rejects_terminal_targets_and_preserves_delivered_history(terminal):
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    old = await submitted(manager, "已经处理的推荐")
    await entered(agents)
    agents.gate(old.question).set()
    await old.worker
    await manager.update(old, terminal)
    old_result = old.result
    with pytest.raises(ValueError, match="尚未交付"):
        await submitted(manager, "新版推荐", replaces_task_id=old.task_id)
    assert old.status == terminal and old.result == old_result
    assert not old.replaced_by_task_id and len(manager.records) == 1
    await manager.close()


async def test_replacement_replay_is_idempotent_even_if_it_names_another_target():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    old, independent = await submitted(manager, "原推荐"), await submitted(manager, "天气")
    new = await submitted(manager, "预算版推荐", replaces_task_id=old.task_id)
    assert await submitted(
        manager, "预算版推荐", replaces_task_id=independent.task_id,
    ) is new
    agents.gate(new.question).set()
    await new.worker
    assert len(agents.instances) <= 3 and len(manager.by_call) == 3
    assert independent.status == "running" and not independent.replaced_by_task_id
    assert new.replaces_task_id == old.task_id and old.replaced_by_task_id == new.task_id
    await manager.close()


async def test_replacement_prevents_a_late_cancellation_ignoring_result_from_reviving():
    agents, updates = Agents(), Updates()
    agents.ignore_cancel.add("旧查询")
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    old = await submitted(manager, "旧查询")
    await entered(agents)
    new = await submitted(manager, "新条件查询", replaces_task_id=old.task_id)
    agents.gate(new.question).set()
    await new.worker
    assert old.worker.done() and old.status == "canceled" and not old.result
    assert "旧查询" in agents.canceled
    assert not any(row[0] == old.task_id and row[1] == "ready" for row in updates.events)
    assert new.status == "ready"
    await manager.close()


async def test_replacement_publishes_and_cleans_up_independently_of_the_foreground():
    agents, updates = Agents(), Updates()
    publishing, release = asyncio.Event(), asyncio.Event()

    async def publish(record):
        await updates(record)
        if record.status == "canceled":
            publishing.set()
            await release.wait()

    manager = BackgroundTaskManager(settings(), publish, agent_factory=agents)
    old = await submitted(manager, "原查询")
    await entered(agents)
    new = await submitted(manager, "新查询", replaces_task_id=old.task_id)
    await asyncio.wait_for(publishing.wait(), 1)
    assert old.status == "canceled" and new.task_id in manager.records
    agents.gate(new.question).set()
    await new.worker
    release.set()
    assert any(row[0] == old.task_id and row[1] == "canceled" for row in updates.events)
    await manager.close()


@pytest.mark.parametrize("uses_tools", [False, True])
async def test_replacement_keeps_inline_reasoning_inline_and_promotes_only_real_tools(uses_tools):
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_tasks=1), updates, agent_factory=agents,
    )
    old = await submitted(manager, "原查询")
    await entered(agents)
    question = "按补充条件重新处理"
    if uses_tools:
        agents.tools.add(question)
        agents.tool_gate(question).set()
    caller = asyncio.create_task(manager.start_delegation(
        question, "turn-2", "new-call", [], replaces_task_id=old.task_id,
    ))
    assert await entered(agents) == question
    assert old.status == "canceled"
    new = manager.by_call["new-call"]
    assert new.promoted.is_set() == uses_tools
    agents.gate(question).set()
    assert await caller is new
    await new.worker
    assert new.status == ("ready" if uses_tools else "completed")
    assert (new.task_id in manager.records) == uses_tools
    assert not new.context and new.agent.history == []
    await manager.close()


async def test_cancel_while_ready_update_is_being_published_cannot_revive_result():
    agents = Agents()
    ready_publish, publish_release = asyncio.Event(), asyncio.Event()

    async def publish(record):
        if record.status == "ready":
            ready_publish.set()
            await publish_release.wait()

    manager = BackgroundTaskManager(settings(), publish, agent_factory=agents)
    record = await submitted(manager)
    await entered(agents)
    agents.gate(record.question).set()
    await asyncio.wait_for(ready_publish.wait(), 1)
    assert await manager.cancel(record.task_id)
    publish_release.set()
    assert record.status == "canceled" and record.worker.done()
    await manager.close()


@pytest.mark.parametrize("mode", ["failure", "empty", "timeout"])
async def test_execution_failures_are_ready_for_notification_and_keep_error_details_private(
    caplog, mode
):
    agents, updates = Agents(), Updates()
    agents.errors["故障"] = RuntimeError("secret-provider-access-token")
    if mode == "empty":
        agents.errors.clear()
        agents.empty.add("故障")
    manager = BackgroundTaskManager(settings(turn_timeout=0.03), updates, agent_factory=agents)
    record = await submitted(manager, "故障")
    await entered(agents)
    if mode != "timeout":
        agents.gate("故障").set()
    await record.worker
    assert record.status == "failed" and record.error == "后台任务处理失败，请重试。"
    assert manager.ready.is_set() and manager.active_tasks() == []
    assert not record.result
    assert "secret-provider-access-token" not in caplog.text
    assert "secret-provider-access-token" not in str(updates.events)
    await manager.close()


async def test_queue_wait_and_delivery_do_not_consume_execution_timeout():
    agents, updates = Agents(), Updates()
    config = settings(background_max_concurrency=1, turn_timeout=0.04)
    manager = BackgroundTaskManager(config, updates, agent_factory=agents)
    first, second = await submitted(manager, "一"), await submitted(manager, "二")
    assert await entered(agents) == "一"
    # The first worker times out; the queued worker receives a fresh timeout.
    await first.worker
    assert first.status == "failed"
    assert await entered(agents) == "二"
    agents.gate("二").set()
    await second.worker
    assert second.status == "ready"
    await asyncio.sleep(0.06)
    assert second.status == "ready" and second.result
    await manager.close()


async def test_new_foreground_caller_cancellation_cannot_strand_a_submitted_worker():
    agents = Agents()
    queued_publish, release_publish = asyncio.Event(), asyncio.Event()

    async def publish(record):
        if record.status == "queued":
            queued_publish.set()
            await release_publish.wait()

    manager = BackgroundTaskManager(settings(), publish, agent_factory=agents)

    async def foreground():
        await submitted(manager)
        await asyncio.Event().wait()

    caller = asyncio.create_task(foreground())
    await asyncio.wait_for(queued_publish.wait(), 1)
    record = next(iter(manager.records.values()))
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert record.worker is not None and not record.worker.done()
    release_publish.set()
    await entered(agents)
    agents.gate(record.question).set()
    await record.worker
    assert record.status == "ready"
    await manager.close()


async def test_terminal_retention_preserves_active_tasks_and_recent_completion_order():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    old_active = await submitted(manager, "很早开始的任务")
    await entered(agents)
    completed = []
    for index in range(32):
        record = await submitted(manager, f"任务{index}")
        await entered(agents)
        agents.gate(record.question).set()
        await record.worker
        await manager.update(record, "completed")
        completed.append(record)
    assert old_active.task_id in manager.records and old_active.status == "running"
    assert len(manager.records) == 31
    assert [r for r in manager.records.values() if r.status == "completed"] == completed[-30:]
    assert completed[0].call_id not in manager.by_call
    agents.gate(old_active.question).set()
    await old_active.worker
    await manager.update(old_active, "completed")
    assert old_active.task_id in manager.records
    assert completed[2].task_id not in manager.records
    assert len(manager.records) == 30
    await manager.close()


async def test_close_cancels_running_and_queued_workers_then_clears_the_session():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        settings(background_max_concurrency=1), updates, agent_factory=agents
    )
    first, second = await submitted(manager, "一"), await submitted(manager, "二")
    await entered(agents)
    await manager.close()
    assert first.status == second.status == "canceled"
    assert first.worker.done() and second.worker.done()
    assert not agents.running
    assert not manager.records and not manager.by_call and not manager.ready.is_set()
    with pytest.raises(ValueError, match="已关闭"):
        await submitted(manager, "关闭后的任务")
    await manager.close()


async def test_close_also_releases_failed_workers_waiting_to_publish():
    agents = Agents()
    failed_publish = asyncio.Event()
    agents.errors["故障"] = RuntimeError("private detail")

    async def publish(record):
        if record.status == "failed":
            failed_publish.set()
            await asyncio.Event().wait()

    manager = BackgroundTaskManager(settings(), publish, agent_factory=agents)
    record = await submitted(manager, "故障")
    await entered(agents)
    agents.gate("故障").set()
    await asyncio.wait_for(failed_publish.wait(), 1)
    assert record.status == "failed" and not record.worker.done()
    await manager.close()
    assert record.worker.done() and not manager.records


async def test_publish_failure_does_not_change_execution_outcome_or_log_private_body(caplog):
    agents = Agents()

    async def publish(record):
        raise RuntimeError("private websocket credential")

    manager = BackgroundTaskManager(settings(), publish, agent_factory=agents)
    record = await submitted(manager)
    await entered(agents)
    agents.gate(record.question).set()
    await record.worker
    assert record.status == "ready" and record.result
    assert "private websocket credential" not in caplog.text
    await manager.close()


async def test_manager_defaults_are_available_with_older_settings_objects():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(
        SimpleNamespace(turn_timeout=2, max_text_length=4000), updates, agent_factory=agents
    )
    assert manager.max_active == 8 and manager.concurrency == 3
    await manager.close()


async def test_background_result_uses_deep_agent_final_answer_without_search_progress(monkeypatch):
    class SearchGraph:
        async def astream(self, state, **kwargs):
            yield "messages", (AIMessageChunk(content="I'll search for that now."), {})
            yield "messages", (AIMessageChunk(content="正在比对停车条件。"), {})
            yield "messages", (AIMessageChunk(content="找到两家支持免费停车的舟山民宿。"), {})
            yield "values", {"messages": [
                *state["messages"],
                AIMessage(content="I'll search for that now.", tool_calls=[{
                    "name": "global_search", "args": {}, "id": "search-1",
                }]),
                ToolMessage(content="原始查询摘要", tool_call_id="search-1"),
                AIMessage(content="正在比对停车条件。"),
                AIMessage(content="找到两家支持免费停车的舟山民宿。"),
            ]}

    monkeypatch.setattr(agent, "ChatOpenAI", lambda **kwargs: object())

    async def graph_without_live_capabilities(self):
        return SearchGraph()

    monkeypatch.setattr(agent.AgentSession, "_build_graph", graph_without_live_capabilities)
    updates = Updates()
    manager = BackgroundTaskManager(
        configured_settings(_env_file=None), updates, agent_factory=agent.AgentSession
    )
    record = await submitted(manager, "查免费停车的舟山民宿")
    await record.worker
    assert record.status == "ready"
    assert record.result == "找到两家支持免费停车的舟山民宿。"
    assert record.result == record.agent.final_answer
    assert "search" not in record.result and "正在" not in record.result
    assert updates.events[-1][2] == record.result
    await manager.close()


async def test_missing_final_answer_does_not_fall_back_to_progress_chunks():
    class IncompleteAgent:
        final_answer = ""

        def __init__(self, config, *, task_only):
            assert task_only

        async def stream(self, question, *, force_deep):
            yield "I'll search for that now."

    manager = BackgroundTaskManager(settings(), Updates(), agent_factory=IncompleteAgent)
    record = await submitted(manager)
    await record.worker
    assert record.status == "failed" and record.result == ""
    assert record.error == "后台任务处理失败，请重试。"
    await manager.close()


async def test_tool_progress_tracks_parallel_calls_and_preserves_execution_timing():
    agents, updates, progress = Agents(), Updates(), []

    async def publish_progress(record):
        progress.append(deepcopy(record.calls))

    manager = BackgroundTaskManager(
        settings(), updates, agent_factory=agents, publish_progress=publish_progress,
    )
    record = await submitted(manager)
    await entered(agents)
    assert record.created_at <= record.started_at <= record.updated_at
    assert record.finished_at is None
    event = record.agent.on_tool_event
    await asyncio.gather(*[
        event({"id": name, "name": "search", "status": "running", "input": {"query": name}})
        for name in ("one", "two")
    ])
    await event({"id": "two", "status": "completed", "output": "第二个先完成"})
    assert [call["status"] for call in record.calls] == ["running", "completed"]
    assert progress[0][0]["status"] == "running"
    await event({"id": "one", "status": "failed", "error": "secret-provider-token"})
    assert record.calls[0]["error"] == "调用失败，请查看任务结果或重试。"
    assert "secret-provider-token" not in str(progress)
    assert [row[1] for row in updates.events] == ["queued", "running"]
    agents.gate(record.question).set()
    await record.worker
    assert record.started_at <= record.finished_at <= record.updated_at
    finished = record.finished_at
    await manager.update(record, "announcing")
    await manager.update(record, "completed")
    assert record.finished_at == finished
    assert all(call["started_at"] <= call["finished_at"] <= finished for call in record.calls)
    await manager.close()


@pytest.mark.parametrize("failure", [False, True])
async def test_task_cancel_or_failure_settles_inflight_tools_and_ignores_late_updates(failure):
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    record = await submitted(manager)
    await entered(agents)
    event = record.agent.on_tool_event
    await event({"id": "one", "name": "search", "status": "running", "input": "舟山天气"})
    if failure:
        agents.errors[record.question] = RuntimeError("provider-secret")
        agents.gate(record.question).set()
        await record.worker
    else:
        await manager.cancel(record.task_id)
    terminal = "failed" if failure else "canceled"
    assert record.status == record.calls[0]["status"] == terminal
    assert record.finished_at == record.calls[0]["finished_at"]
    snapshot = deepcopy(record.calls)
    await event({"id": "one", "status": "completed", "output": "过期结果"})
    await event({"id": "two", "name": "search", "status": "running", "input": "迟到调用"})
    assert record.calls == snapshot
    await manager.close()


async def test_trace_retention_and_each_payload_are_bounded_without_hiding_truncation():
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    record = await submitted(manager)
    await entered(agents)
    for index in range(MAX_TRACE_CALLS + 3):
        await record.agent.on_tool_event({
            "id": str(index), "name": "search", "status": "running", "input": "查" * 5000,
        })
        await record.agent.on_tool_event({
            "id": str(index), "status": "completed", "output": {"data": "结果" * 5000},
        })
    assert len(record.calls) == MAX_TRACE_CALLS and record.calls_truncated
    assert all(len(call["input"]) <= 4000 and len(call["output"]) <= 4000
               for call in record.calls)
    assert all(call["input_truncated"] and call["output_truncated"] for call in record.calls)
    await manager.close()


@pytest.mark.parametrize("payload", [
    {"query": "天气", "headers": {"Authorization": "Bearer hidden-value", "X-Api-Key": "key-value"},
     "nested": [{"access_token": "token-value", "password": "pass-value"}]},
    '{"query":"天气","token":"token-value","nested":{"secret":"hidden-value"}}',
    "Authorization: Bearer hidden-value; api_key=key-value&token=token-value password=pass-value",
    "https://user:pass-value@example.com/search?token=token-value&query=天气",
])
def test_trace_redacts_nested_credentials_json_text_headers_and_urls(payload):
    text, truncated = _trace_text(payload)
    assert not truncated
    for secret in ("hidden-value", "key-value", "token-value", "pass-value"):
        assert secret not in text
    assert "[已隐藏]" in text


def test_trace_does_not_serialize_unknown_object_configuration():
    value = SimpleNamespace(api_key="secret-provider-key")
    assert _trace_text(value) == ("[非文本结果]", False)


@pytest.mark.parametrize(("name", "output", "status", "error"), [
    ("global_search", {"error": {"code": "unavailable", "message": "private-provider-detail"}},
     "failed", "搜索服务暂时不可用，当前信息尚未核实。"),
    ("global_search", json.dumps({"error": {"code": "timeout", "message": "private-detail"}}),
     "failed", "联网搜索超时，当前信息尚未核实。"),
    ("global_search", {"query": "舟山天气", "results": []}, "completed", ""),
    ("global_search", {"error": {"code": "unknown_error"}}, "completed", ""),
    ("other_tool", {"error": {"code": "timeout"}}, "completed", ""),
])
async def test_search_error_envelopes_only_change_trace_not_task_execution(
    name, output, status, error,
):
    agents, updates = Agents(), Updates()
    manager = BackgroundTaskManager(settings(), updates, agent_factory=agents)
    record = await submitted(manager)
    await entered(agents)
    await record.agent.on_tool_event({
        "id": "run-1", "name": name, "status": "running", "input": {"query": "舟山天气"},
    })
    await record.agent.on_tool_event({"id": "run-1", "status": "completed", "output": output})
    call, = record.calls
    assert call["status"] == status and call["error"] == error
    assert call["finished_at"] >= call["started_at"]
    assert "private-" not in str(call)
    assert record.status == "running" and record.error == ""
    agents.gate(record.question).set()
    await record.worker
    assert record.status == "ready" and record.result == f"{record.question}的实际结果。"
    assert call["status"] == status
    await manager.close()

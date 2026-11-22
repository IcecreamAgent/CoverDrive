"""Session-scoped task execution that survives foreground voice interruptions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from voice_agent.agent import AgentSession
from voice_agent.config import Settings

logger = logging.getLogger(__name__)

BackgroundStatus = Literal[
    "queued", "running", "ready", "announcing", "completed", "canceled", "failed"
]
ACTIVE_STATUSES = frozenset({"queued", "running", "ready", "announcing"})
TERMINAL_STATUSES = frozenset({"completed", "canceled", "failed"})
ALL_STATUSES = ACTIVE_STATUSES | TERMINAL_STATUSES
MAX_TRACE_CALLS = 40
MAX_TRACE_TEXT = 4000
_SEARCH_TRACE_ERRORS = {
    "not_configured": "联网搜索未配置，无法核实网络信息。",
    "timeout": "联网搜索超时，当前信息尚未核实。",
    "auth_failed": "搜索服务鉴权失败，请检查服务端搜索凭证。",
    "rate_limited": "搜索服务暂时限流，请稍后重试。",
    "unavailable": "搜索服务暂时不可用，当前信息尚未核实。",
    "invalid_response": "搜索服务返回了无效数据，当前信息尚未核实。",
}
_CREDENTIAL_KEY = re.compile(
    r"(?:authorization|cookie|password|passwd|secret|token|api[_-]?key|access[_-]?key|"
    r"private[_-]?key|credential)", re.IGNORECASE,
)
_CREDENTIAL_VALUE = re.compile(
    r"(?i)((?:authorization|cookie|password|passwd|secret|token|api[_-]?key|"
    r"access[_-]?key|private[_-]?key|credential)[\w-]{0,40}[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;&}]+)",
)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _trace_text(value: object) -> tuple[str, bool]:
    """Bound public tool data, removing credential fields before serialization."""
    truncated = False
    budget = 200

    def clean(item: object, depth: int = 0) -> object:
        nonlocal truncated, budget
        budget -= 1
        if depth > 6 or budget < 0:
            truncated = True
            return "…"
        if isinstance(item, dict):
            result = {}
            for index, (key, child) in enumerate(item.items()):
                if index >= 40 or budget <= 0:
                    truncated = True
                    break
                key = str(key)
                truncated = truncated or len(key) > 100
                result[key[:100]] = (
                    "[已隐藏]" if _CREDENTIAL_KEY.search(key) else clean(child, depth + 1)
                )
            return result
        if isinstance(item, (list, tuple)):
            result = []
            for index, child in enumerate(item):
                if index >= 40 or budget <= 0:
                    truncated = True
                    break
                result.append(clean(child, depth + 1))
            return result
        if isinstance(item, str):
            item = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", "[已隐藏]", item)
            item = _CREDENTIAL_VALUE.sub(r"\1[已隐藏]", item)
            item = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[已隐藏]@", item)
            if len(item) > MAX_TRACE_TEXT:
                truncated = True
                item = item[:MAX_TRACE_TEXT]
            return item
        if item is None or isinstance(item, (bool, int, float)):
            return item
        # Do not expose object reprs, which may embed clients or configuration.
        return "[非文本结果]"

    if isinstance(value, str) and len(value) <= 100_000:
        try:
            value = json.loads(value)
        except (ValueError, RecursionError):
            pass
    value = clean(value)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return text[:MAX_TRACE_TEXT], truncated or len(text) > MAX_TRACE_TEXT


@dataclass(eq=False)
class BackgroundTask:
    origin_turn_id: str
    call_id: str
    question: str
    title: str
    task_id: str = field(default_factory=lambda: uuid4().hex[:12])
    status: BackgroundStatus = "queued"
    result: str = ""
    error: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)
    context: str = ""
    worker: asyncio.Task | None = field(default=None, repr=False)
    agent: AgentSession | None = field(default=None, repr=False)
    announcement_attempts: int = 0
    promoted: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    waiters: int = field(default=0, repr=False)
    created_at: int = field(default_factory=_now_ms)
    started_at: int | None = None
    finished_at: int | None = None
    updated_at: int = field(init=False)
    calls: list[dict] = field(default_factory=list)
    calls_truncated: bool = False
    replaces_task_id: str = ""
    replaced_by_task_id: str = ""
    authorization_version: int | None = None

    def __post_init__(self) -> None:
        self.updated_at = self.created_at


class BackgroundTaskManager:
    """Own task workers separately from the current input/output turn.

    State changes reserve their place before any await. Each worker receives an
    independent AgentSession and a conversation snapshot, never mutable graph
    state shared with another task. The caller owns notification scheduling.
    """

    def __init__(
        self,
        settings: Settings,
        publish: Callable[[BackgroundTask], Awaitable[None]],
        *,
        ready: asyncio.Event | None = None,
        agent_factory: Callable[..., AgentSession] | None = None,
        publish_progress: Callable[[BackgroundTask], Awaitable[None]] | None = None,
        publish_execution: Callable[[BackgroundTask], Awaitable[None]] | None = None,
        authorization=None,
    ):
        self.settings = settings
        self.publish = publish
        self.publish_progress = publish_progress or publish
        self.publish_execution = publish_execution
        self.ready = ready if ready is not None else asyncio.Event()
        self.agent_factory = agent_factory if agent_factory is not None else AgentSession
        self.authorization = authorization
        self.max_active = getattr(settings, "background_max_tasks", 8)
        self.concurrency = getattr(settings, "background_max_concurrency", 3)
        self.records: dict[str, BackgroundTask] = {}
        self.by_call: dict[str, BackgroundTask] = {}
        self._terminal_order: list[str] = []
        self._inline_terminal_order: list[str] = []
        self._workers: set[asyncio.Task] = set()
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._closed = False

    def active_tasks(self) -> list[BackgroundTask]:
        return [record for record in self.records.values() if record.status in ACTIVE_STATUSES]

    def _reserved_background_slots(self, *, excluding: BackgroundTask | None = None) -> int:
        # Replacing an admitted task transfers its slot immediately. Otherwise
        # another hidden reasoning call could promote before this replacement
        # reaches its first tool, losing both the old task and its successor.
        return sum(
            record is not excluding and bool(record.replaces_task_id)
            and record.task_id not in self.records and record.status in ACTIVE_STATUSES
            for record in self.by_call.values()
        )

    async def _publish(self, record: BackgroundTask, *, progress: bool = False) -> None:
        try:
            await self._trace_execution(record)
            await (self.publish_progress if progress else self.publish)(record)
        except Exception as exc:
            # A lost browser connection must not leak its exception or silently
            # turn a completed task into an execution failure.
            logger.warning("background_publish failed (%s)", type(exc).__name__)

    async def _trace_execution(self, record: BackgroundTask) -> None:
        if self.publish_execution is not None:
            try:
                await self.publish_execution(record)
            except Exception as exc:
                logger.warning("execution_trace failed (%s)", type(exc).__name__)

    async def _tool_event(self, record: BackgroundTask, event: dict) -> None:
        if self._closed or record.status != "running" or not record.promoted.is_set():
            return
        now = _now_ms()
        call = next((item for item in record.calls if item["id"] == event["id"]), None)
        if event["status"] == "running":
            if call is not None:
                return
            if len(record.calls) >= MAX_TRACE_CALLS:
                if not record.calls_truncated:
                    record.calls_truncated = True
                    record.updated_at = now
                    await self._publish(record, progress=True)
                return
            input_text, input_truncated = _trace_text(event.get("input", ""))
            call = {
                "id": event["id"], "name": str(event.get("name", "tool"))[:100],
                "status": "running", "started_at": now, "finished_at": None,
                "input": input_text, "output": "", "error": "",
                "input_truncated": input_truncated, "output_truncated": False,
            }
            record.calls.append(call)
        elif call is not None and call["status"] == "running":
            call["status"] = event["status"]
            call["finished_at"] = now
            if event["status"] == "failed":
                call["error"] = "调用失败，请查看任务结果或重试。"
            else:
                output = event.get("output", "")
                if call["name"] == "global_search":
                    envelope = output
                    if isinstance(envelope, str) and len(envelope) <= 100_000:
                        with contextlib.suppress(ValueError, RecursionError):
                            envelope = json.loads(envelope)
                    error = envelope.get("error") if isinstance(envelope, dict) else None
                    code = error.get("code") if isinstance(error, dict) else None
                    if isinstance(code, str) and code in _SEARCH_TRACE_ERRORS:
                        # global_search returns documented errors as normal tool
                        # values. Reflect that failure only in the trace, using
                        # known messages rather than any provider error body.
                        call["status"] = "failed"
                        call["error"] = _SEARCH_TRACE_ERRORS[code]
                        output = {"error": {"code": code, "message": call["error"]}}
                call["output"], call["output_truncated"] = _trace_text(output)
        else:
            return
        record.updated_at = now
        await self._publish(record, progress=True)

    def _prune(self) -> None:
        while len(self._terminal_order) > 30:
            task_id = self._terminal_order.pop(0)
            record = self.records.pop(task_id, None)
            if record is None:
                continue
            if self.by_call.get(record.call_id) is record:
                self.by_call.pop(record.call_id, None)
        while len(self._inline_terminal_order) > 30:
            call_id = self._inline_terminal_order.pop(0)
            record = self.by_call.get(call_id)
            if record is not None and record.task_id not in self.records:
                self.by_call.pop(call_id, None)

    def _reserve(
        self,
        question: str,
        origin_turn_id: str,
        call_id: str,
        history: Iterable[tuple[str, str]],
        title: str,
        context: str,
        *,
        background: bool,
        replaces_task_id: str = "",
    ) -> BackgroundTask:
        """Validate and reserve a call without yielding to another submission."""
        if self._closed:
            raise ValueError("后台任务管理器已关闭")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("后台任务需要有效调用 ID")
        if call_id in self.by_call:
            return self.by_call[call_id]
        if not isinstance(question, str) or not question.strip():
            raise ValueError("后台任务内容不能为空")
        if len(question) > self.settings.max_text_length:
            raise ValueError("后台任务内容超过长度限制")
        if not isinstance(title, str) or len(title) > 80:
            raise ValueError("后台任务标题无效")
        if not isinstance(context, str) or len(context) > 16000:
            raise ValueError("任务上下文超过长度限制")
        if not isinstance(replaces_task_id, str) or len(replaces_task_id) > 128:
            raise ValueError("被替换任务 ID 无效")
        replaced = self.records.get(replaces_task_id) if replaces_task_id else None
        if replaces_task_id and (replaced is None or replaced.status not in ACTIVE_STATUSES):
            raise ValueError("只能替换当前会话中尚未交付的有效任务，请先查询任务列表")
        hidden_active = sum(
            record.task_id not in self.records and record.status in ACTIVE_STATUSES
            for record in self.by_call.values()
        )
        occupied = len(self.active_tasks()) + self._reserved_background_slots()
        if (
            background and occupied - int(replaced is not None) >= self.max_active
            or not background and hidden_active >= self.max_active
        ):
            raise ValueError("后台任务数量已达上限，请等待已有任务完成或取消任务")
        record = BackgroundTask(
            origin_turn_id=origin_turn_id,
            call_id=call_id,
            question=question.strip(),
            title=(title.strip() or question.strip()[:24]).rstrip("。！？?、， "),
            history=deepcopy(list(history)),
            authorization_version=(self.authorization.version if self.authorization is not None else None),
            context=context,
            replaces_task_id=replaces_task_id,
        )
        reserved_ids = {item.task_id for item in self.by_call.values()}
        while record.task_id in reserved_ids:
            record.task_id = uuid4().hex[:12]
        if background:
            self.records[record.task_id] = record
            record.promoted.set()
        self.by_call[call_id] = record
        if replaced is not None:
            # Everything above, including the new history snapshot, must validate
            # before invalidating the old result. No await separates the two
            # records, so neither delivery nor another call can see half a swap.
            replaced.replaced_by_task_id = record.task_id
            if replaced.worker is not None and not replaced.worker.done():
                replaced.worker.cancel()
            self._transition(replaced, "canceled")
            cleanup = asyncio.create_task(
                self._finish_replacement(replaced), name=f"replaced-{replaced.task_id}"
            )
            self._workers.add(cleanup)
            cleanup.add_done_callback(self._workers.discard)
        record.worker = asyncio.create_task(
            self._execute(record), name=f"delegation-{record.task_id}"
        )
        self._workers.add(record.worker)
        record.worker.add_done_callback(self._workers.discard)
        return record

    async def submit(
        self,
        question: str,
        origin_turn_id: str,
        call_id: str,
        history: Iterable[tuple[str, str]],
        title: str = "",
        *,
        context: str = "",
        replaces_task_id: str = "",
    ) -> BackgroundTask:
        # Reserving and starting synchronously makes duplicate submissions and
        # limits safe. Publishing belongs to the independent worker, so a
        # foreground caller being interrupted cannot strand a queued record.
        return self._reserve(
            question, origin_turn_id, call_id, history, title, context,
            background=True, replaces_task_id=replaces_task_id,
        )

    async def start_delegation(
        self,
        question: str,
        origin_turn_id: str,
        call_id: str,
        history: Iterable[tuple[str, str]],
        title: str = "",
        *,
        context: str = "",
        replaces_task_id: str = "",
    ) -> BackgroundTask:
        """Wait inline until a real tool starts or the final answer is available.

        A pure reasoning answer never appears in the background task list.
        Once promoted, its worker belongs to the session and survives caller
        cancellation; before that boundary it belongs to the foreground turn.
        """
        record = self._reserve(
            question, origin_turn_id, call_id, history, title, context,
            background=False, replaces_task_id=replaces_task_id,
        )
        promoted = asyncio.create_task(record.promoted.wait())
        record.waiters += 1
        try:
            await asyncio.wait((record.worker, promoted), return_when=asyncio.FIRST_COMPLETED)
            return record
        except asyncio.CancelledError:
            if not record.promoted.is_set() and record.waiters == 1:
                await self._cancel_record(record)
            raise
        finally:
            record.waiters -= 1
            promoted.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await promoted

    async def _promote(self, record: BackgroundTask) -> None:
        if self._closed or record.status == "canceled":
            raise asyncio.CancelledError
        if record.promoted.is_set():
            return
        occupied = len(self.active_tasks()) + self._reserved_background_slots(excluding=record)
        if occupied >= self.max_active:
            raise ValueError("后台任务数量已达上限，请等待已有任务完成或取消任务")
        # Register synchronously before publishing: cancellation can now stop
        # foreground speech without canceling this real tool execution.
        self.records[record.task_id] = record
        record.promoted.set()
        await self._publish(record)

    def _transition(self, record: BackgroundTask, status: BackgroundStatus) -> bool:
        """Commit state synchronously; publishing may subsequently yield."""
        if status not in ALL_STATUSES:
            raise ValueError("无效的后台任务状态")
        if record.status == "canceled" or record.status == status:
            return False
        record.status = status
        record.updated_at = _now_ms()
        if status == "running" and record.started_at is None:
            record.started_at = record.updated_at
        if status in {"ready", "completed", "failed", "canceled"}:
            if record.finished_at is None:
                record.finished_at = record.updated_at
            for call in record.calls:
                if call["status"] == "running":
                    call["status"] = status if status in {"failed", "canceled"} else "completed"
                    call["finished_at"] = record.updated_at
                    if status == "failed":
                        call["error"] = "任务处理失败，调用已结束。"
        if not record.promoted.is_set():
            if status in TERMINAL_STATUSES:
                self._inline_terminal_order.append(record.call_id)
                self._prune()
            return True
        if record.task_id in self._terminal_order:
            self._terminal_order.remove(record.task_id)
        if status in TERMINAL_STATUSES:
            self._terminal_order.append(record.task_id)
        if status in ("ready", "failed"):
            self.ready.set()
        self._prune()
        return True

    async def _publish_update(self, record: BackgroundTask) -> None:
        if record.promoted.is_set():
            await self._publish(record)
        else:
            await self._trace_execution(record)

    async def update(self, record: BackgroundTask, status: BackgroundStatus) -> None:
        if self._transition(record, status):
            await self._publish_update(record)

    async def _finish_replacement(self, record: BackgroundTask) -> None:
        # A foreground interruption must not strand cancellation notification
        # or worker cleanup after the atomic swap has already succeeded.
        await self._publish_update(record)
        if record.worker is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await record.worker

    async def _execute(self, record: BackgroundTask) -> None:
        try:
            if record.promoted.is_set():
                await self._publish(record)
            else:
                await self._trace_execution(record)
            # Queue waiting and ready-result delivery are not execution time.
            async with self._semaphore:
                if self._closed or record.status == "canceled":
                    return
                await self.update(record, "running")
                if self._closed or record.status == "canceled":
                    return
                async with asyncio.timeout(self.settings.turn_timeout):
                    record.agent = self.agent_factory(self.settings, task_only=True)
                    if record.authorization_version is not None and hasattr(record.agent, "_authorization_version"):
                        record.agent._authorization_version = record.authorization_version
                    if not record.promoted.is_set():
                        record.agent.on_tool_start = lambda: self._promote(record)
                    record.agent.on_tool_event = lambda event: self._tool_event(record, event)
                    for question, answer in record.history:
                        record.agent.remember_exchange(question, answer)
                    parts = []
                    options = {"force_deep": True}
                    if record.context:
                        options["context"] = record.context
                    async for part in record.agent.stream(record.question, **options):
                        if self._closed or record.status == "canceled":
                            return
                        parts.append(part)
                    final_answer = getattr(record.agent, "final_answer", None)
                    result = final_answer if isinstance(final_answer, str) else "".join(parts)
                    if not result.strip():
                        raise RuntimeError("Empty background task result")
                if self._closed or record.status == "canceled":
                    return
                record.result = result
                await self.update(record, "ready" if record.promoted.is_set() else "completed")
        except asyncio.CancelledError:
            if record.status != "canceled":
                await self.update(record, "canceled")
            raise
        except Exception as exc:
            if not self._closed and record.status != "canceled":
                logger.warning("background_execute failed (%s)", type(exc).__name__)
                record.error = (
                    "后台任务处理失败，请重试。" if record.promoted.is_set()
                    else "深度处理失败，请重试。"
                )
                await self.update(record, "failed")

    async def _cancel_record(self, record: BackgroundTask) -> None:
        worker = record.worker
        if worker is not None and worker is not asyncio.current_task() and not worker.done():
            worker.cancel()
        await self.update(record, "canceled")
        if worker is not None and worker is not asyncio.current_task():
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    async def cancel(self, task_id: str) -> bool:
        record = self.records.get(task_id)
        if record is None or record.status not in ACTIVE_STATUSES:
            return False
        # The canceled state is set before update yields, so a late generated
        # result cannot revive the task while the cancel notification is sent.
        await self._cancel_record(record)
        return True

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(
            *(self._cancel_record(record) for record in list(self.by_call.values())
              if record.status in ACTIVE_STATUSES)
        )
        # A failed/completed worker may still be awaiting its final UI publish.
        workers = [worker for worker in self._workers if worker is not asyncio.current_task()]
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        self.records.clear()
        self.by_call.clear()
        self._terminal_order.clear()
        self._inline_terminal_order.clear()
        self._workers.clear()
        self.ready.clear()

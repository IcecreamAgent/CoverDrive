"""Bounded, public execution milestones; never model reasoning or audio payloads."""

from collections import OrderedDict
from time import time_ns

from voice_agent.routing import REASON_LABELS


class ExecutionTrace:
    def __init__(self, settings):
        self.settings = settings
        self.sequence = 0
        self.turns: OrderedDict[str, dict] = OrderedDict()
        self.seen: OrderedDict[tuple, None] = OrderedDict()
        self.task_states: OrderedDict[str, str] = OrderedDict()

    def model_label(self, stage: str) -> str:
        """Publish a model's role, never its configured deployment identifier."""
        if stage == "deep":
            return "任务 LLM"
        return "端到端语音" if self.settings.voice_provider == "seeduplex" else "快速 LLM"

    def turn(self, turn_id: str | None) -> dict:
        key = turn_id or ""
        if key not in self.turns:
            self.turns[key] = {"stage": "fast", "generation": 0, "returned": False}
        self.turns.move_to_end(key)
        while len(self.turns) > 64:
            self.turns.popitem(last=False)
        return self.turns[key]

    def event(
        self, turn_id, stage, name, label, status="info", detail="", *, once=None, **metadata
    ) -> dict | None:
        if once is not None:
            key = (turn_id or "", name, once)
            if key in self.seen:
                return None
            self.seen[key] = None
            while len(self.seen) > 1024:
                self.seen.popitem(last=False)
        self.sequence += 1
        if "model" in metadata:
            metadata["model"] = self.model_label(stage)
        return {
            "id": str(self.sequence), "sequence": self.sequence,
            "at": metadata.pop("at", time_ns() // 1_000_000), "turn_id": turn_id or "",
            "stage": stage, "name": name, "label": label, "status": status, "detail": detail,
            **{key: value for key, value in metadata.items() if value is not None},
        }

    def observe(self, event_type: str, data: dict) -> dict | None:
        turn_id = data.get("turn_id")
        state = self.turn(turn_id)
        native = self.settings.voice_provider == "seeduplex"
        if event_type == "speech_start":
            return self.event(turn_id, "input", "input_started", "开始接收语音", "running",
                              "由服务端检测到发言开始。", once="input")
        if event_type == "transcript" and data.get("is_final"):
            return self.event(turn_id, "input", "input_final", "语音输入已确认", "completed",
                              "识别结束，等待模型响应。", once="input")
        if event_type == "agent_route":
            route, source = data.get("route"), data.get("source")
            if source in {"background", "native_recovery"}:
                return None  # These replay results; they are not another model execution.
            if route not in {"fast", "deep"}:
                return None
            stage = "deep" if route == "deep" else "fast"
            state["stage"] = stage
            reason = data.get("reason", "model_selected")
            detail = REASON_LABELS.get(reason, {
                "native_delegation": "语音模型发起真实工具委派；服务端未获得模型内部选择理由。",
                "keyboard_task": "键盘任务直接交给 Deep Agents，不经过原生语音快答。",
                "model_selected": "依据实际模型路由事件记录。",
            }.get(reason, "依据实际模型路由事件记录。"))
            if source == "keyboard":
                detail = "键盘任务直接交给 Deep Agents，不经过原生语音快答。"
            model = self.model_label(stage)
            return self.event(
                turn_id, stage, "route_" + stage,
                "进入慢模型" if stage == "deep" else "进入语音快模型" if native else "进入快模型",
                "running", detail, model=model, reason=reason, source=source,
                provider=data.get("provider"), once=(stage, source, state["generation"]),
            )
        if event_type == "assistant_delta" and data.get("text"):
            stage = state["stage"]
            return self.event(
                turn_id, stage, "first_text", "收到首段文字", "info",
                "服务端收到首段可见文字；不代表生成或播放完成。",
                generation=state["generation"], once=state["generation"],
            )
        if event_type == "assistant_done":
            return self.event(turn_id, state["stage"], "text_completed", "本段文字已完成",
                              "completed", "语音可能仍在生成或播放。",
                              generation=state["generation"], once=state["generation"])
        if event_type == "state" and data.get("stage") == "idle":
            return self.event(turn_id, "system", "foreground_completed", "前台处理已结束",
                              "completed", "可以继续对话；独立后台任务和浏览器播放可能仍在继续。",
                              once=state["generation"])
        if event_type in {"interrupted", "barge_in_confirmed"}:
            return self.event(turn_id, "system", "interrupted", "前台输出已打断", "canceled",
                              "只停止前台输出；已提交的后台任务继续执行。", once="interrupted")
        if event_type == "agent_progress":
            phase = data.get("phase")
            if phase == "waiting_playback":
                return self.event(turn_id, "voice", "playback_wait", "等待前段播放完成",
                                  "waiting", "结果已就绪，等待浏览器音频队列排空。",
                                  once=state["generation"])
            if phase == "result_ready":
                return self.event(turn_id, "deep", "deep_result_ready", "慢模型结果已生成",
                                  "completed", "执行结果已就绪，尚未等同于用户听到结果。",
                                  once=state["generation"])
        if event_type == "error":
            return self.event(turn_id, "system", "error", "处理出现错误", "failed",
                              "请查看页面错误提示。", source=data.get("stage"))
        return None

    def handoff(self, turn_id, *, task_id=None, call_id=None, detail="") -> dict:
        state = self.turn(turn_id)
        state["stage"] = "voice"
        state["generation"] += 1
        state["returned"] = True
        return self.event(turn_id, "voice", "result_returned", "慢模型 → 语音模型", "running",
                          detail or "已有结果已交给 Seeduplex 接续回答，不会重新执行任务。",
                          task_id=task_id, call_id=call_id,
                          generation=state["generation"], model="端到端语音")

    def pcm(self, turn_id) -> dict | None:
        state = self.turn(turn_id)
        stage = "fast" if (
            self.settings.voice_provider == "seeduplex" and not state["returned"]
        ) else "voice"
        return self.event(turn_id, stage, "first_audio", "收到首块语音", "info",
                          "服务端正在向浏览器发送首块 PCM；不代表浏览器已经播放。",
                          generation=state["generation"], once=state["generation"],
                          model="端到端语音"
                          if self.settings.voice_provider == "seeduplex" else None,
                          provider="seeduplex" if self.settings.voice_provider == "seeduplex"
                          else "tts")

    def task(self, task) -> list[dict]:
        previous = self.task_states.get(task.task_id)
        self.task_states[task.task_id] = task.status
        self.task_states.move_to_end(task.task_id)
        while len(self.task_states) > 128:
            self.task_states.popitem(last=False)
        result = []
        if previous != task.status:
            common = {"task_id": task.task_id, "call_id": task.call_id}
            if task.status == "queued":
                result.append(self.event(
                    task.origin_turn_id, "deep", "deep_queued", "等待慢模型执行名额", "waiting",
                    "执行器等待并发名额；不代表任务已经登记为后台任务。",
                    at=task.created_at, model="任务 LLM", **common,
                ))
            elif task.status == "running":
                result.append(self.event(
                    task.origin_turn_id, "deep", "deep_started", "慢模型开始执行", "running",
                    "已取得并发执行名额；排队时间不计入执行耗时。",
                    at=task.started_at, model="任务 LLM", **common,
                ))
            elif task.status in {"ready", "completed"} and previous in {"queued", "running"}:
                result.append(self.event(
                    task.origin_turn_id, "deep", "deep_result_ready", "慢模型结果已生成",
                    "completed", "最终结果已生成。", at=task.finished_at, **common,
                ))
            labels = {
                "queued": ("任务等待执行", "waiting"), "running": ("任务执行中", "running"),
                "ready": ("结果待播报", "waiting"), "announcing": ("开始播报任务结果", "running"),
                "completed": ("任务交付结束", "completed"), "canceled": ("任务已取消", "canceled"),
                "failed": ("任务执行失败", "failed"),
            }
            if task.promoted.is_set() or task.status in {"failed", "canceled"}:
                label, status = labels[task.status]
                result.append(self.event(task.origin_turn_id, "deep", "task_" + task.status,
                                         label, status, at=task.updated_at, **common))
        for call in task.calls:
            status = call["status"]
            label = {"running": "开始工具调用", "completed": "工具已返回",
                     "failed": "工具调用失败", "canceled": "工具调用已取消"}[status]
            result.append(self.event(
                task.origin_turn_id, "tool", "tool_started" if status == "running"
                else "tool_" + status, label, status,
                call["name"], task_id=task.task_id, call_id=call["id"],
                at=call["started_at"] if status == "running" else call["finished_at"],
                once=(call["id"], status),
            ))
        return [item for item in result if item is not None]

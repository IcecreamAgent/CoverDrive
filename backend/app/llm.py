"""LangChain ChatModel adapter for the internal SenseTime/Volc chat API."""
import json
from typing import Any, Sequence
import httpx
from pydantic import ConfigDict, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from .config import settings

SYSTEM = """你是一个中文语音助手。回答要自然、简洁、适合朗读，不使用 Markdown。需要完成实际任务时调用工具，工具返回后再给用户最终答复。"""


def _schema(item: Any) -> dict:
    if isinstance(item, dict):
        return item
    if isinstance(item, BaseTool):
        params = item.args_schema.model_json_schema() if item.args_schema else {"type": "object", "properties": {}}
        return {"name": item.name, "description": item.description, "parameters": params}
    if hasattr(item, "model_json_schema"):
        return {"name": item.__name__, "description": item.__doc__ or "", "parameters": item.model_json_schema()}
    return {"name": getattr(item, "__name__", "tool"), "description": getattr(item, "__doc__", ""), "parameters": {"type": "object", "properties": {}}}


def _message_content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _to_api_messages(messages: list[BaseMessage]) -> list[dict]:
    converted = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, ToolMessage):
            converted.append({"role": "user", "content": [{"type": "text", "text": f"<tool_response>工具返回：{_message_content(message)}</tool_response>"}]})
        elif isinstance(message, HumanMessage):
            converted.append({"role": "user", "content": [{"type": "text", "text": _message_content(message)}]})
        else:
            converted.append({"role": "assistant", "content": [{"type": "text", "text": _message_content(message)}]})
    return converted


def _parse_response(data: dict) -> tuple[str, list[dict]]:
    root = data.get("result", data)
    choices = root.get("choices", []) if isinstance(root, dict) else []
    message = choices[0].get("message", {}) if choices else root.get("message", {}) if isinstance(root, dict) else {}
    content = message.get("content", "") or root.get("text", "") if isinstance(root, dict) else ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    calls = message.get("tool_calls") or root.get("tool_calls", []) if isinstance(root, dict) else []
    if not calls and isinstance(root, dict) and root.get("function_call"):
        calls = [{"function": root["function_call"]}]
    normalized = []
    for index, call in enumerate(calls or []):
        fn = call.get("function", call)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args or "{}")
        normalized.append({"name": fn.get("name"), "args": args, "id": call.get("id", f"call_{index}")})
    return str(content), normalized


class InternalChatModel(BaseChatModel):
    """让 Deep Agents 可以使用现有内部 LLM，而无需改造服务端 API。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_id: str = Field(default_factory=lambda: settings.llm_model_id)
    bound_tools: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "internal-vchat"

    @property
    def _identifying_params(self) -> dict:
        return {"model_id": self.model_id, "endpoint": settings.llm_base_url}

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any):
        return self.model_copy(update={"bound_tools": [_schema(tool) for tool in tools]})

    def _payload(self, messages: list[BaseMessage]) -> dict:
        return {
            "model_id": self.model_id,
            "thinking_config": {"reasoning_effort": "none"},
            "system_instruction": {"content": SYSTEM},
            "messages": _to_api_messages(messages),
            "tools": self.bound_tools,
            "generation_config": {"max_tokens": 1200},
        }

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        return headers

    def _make_result(self, data: dict) -> ChatResult:
        text, calls = _parse_response(data)
        message = AIMessage(content=text, tool_calls=[{"name": c["name"], "args": c["args"], "id": c["id"], "type": "tool_call"} for c in calls])
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> ChatResult:
        with httpx.Client(timeout=90) as client:
            response = client.post(settings.llm_base_url, headers=self._headers(), json=self._payload(messages))
            response.raise_for_status()
            return self._make_result(response.json())

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> ChatResult:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(settings.llm_base_url, headers=self._headers(), json=self._payload(messages))
            response.raise_for_status()
            return self._make_result(response.json())

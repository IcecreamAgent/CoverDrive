import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app import agent as agent_module


def model_start():
    return {"event": "on_chat_model_start", "data": {}}


def text_chunk(text):
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": SimpleNamespace(content=text, tool_call_chunks=[])},
    }


def responses_text_chunk(text):
    return {
        "event": "on_chat_model_stream",
        "data": {
            "chunk": SimpleNamespace(
                content=[{"type": "text", "text": text, "index": 0}],
                tool_call_chunks=[],
            )
        },
    }


def tool_chunk(name):
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": SimpleNamespace(content="", tool_call_chunks=[{"name": name}])},
    }


def model_end(*tool_names):
    return {
        "event": "on_chat_model_end",
        "data": {"output": SimpleNamespace(tool_calls=[{"name": name} for name in tool_names])},
    }


def tool_event(kind, name, run_id="run-1"):
    data = {"input": {}} if kind == "on_tool_start" else {"output": "ok"}
    return {"event": kind, "name": name, "run_id": run_id, "data": data}


class FakeAgent:
    def __init__(self, events):
        self.events = events

    async def astream_events(self, *args, **kwargs):
        for event in self.events:
            yield event


class FakeInvokeAgent:
    def __init__(self, messages):
        self.messages = messages

    async def ainvoke(self, *args, **kwargs):
        return {"messages": self.messages}


class LlmConfigurationTests(unittest.TestCase):
    def test_build_model_uses_configured_responses_api_settings(self):
        with (
            patch.object(
                agent_module.settings,
                "llm_base_url",
                "https://llm.example.test/v1",
            ),
            patch.object(
                agent_module.settings,
                "llm_model_id",
                "fake-legacy-model",
            ),
            patch.object(agent_module.settings, "ark_api_key", "test-ark-key"),
            patch.object(agent_module.settings, "llm_api_key", "legacy-key"),
            patch.object(agent_module, "ChatOpenAI") as chat_model,
        ):
            agent_module._build_model()

        chat_model.assert_called_once_with(
            model="fake-legacy-model",
            api_key="test-ark-key",
            base_url="https://llm.example.test/v1",
            timeout=90,
            max_retries=2,
            use_responses_api=True,
        )

    def test_build_model_falls_back_to_generic_llm_api_key(self):
        with (
            patch.object(agent_module.settings, "ark_api_key", ""),
            patch.object(agent_module.settings, "llm_api_key", "generic-key"),
            patch.object(agent_module, "ChatOpenAI") as chat_model,
        ):
            agent_module._build_model()

        self.assertEqual(chat_model.call_args.kwargs["api_key"], "generic-key")

    def test_build_agent_only_passes_explicit_voice_tools(self):
        allowed = SimpleNamespace(name="calc_premium")
        built = object()
        with (
            patch.object(agent_module, "_build_model", return_value="model") as build_model,
            patch.object(agent_module, "InMemorySaver", return_value="memory"),
            patch.object(agent_module, "create_agent", return_value=built) as create,
        ):
            result = agent_module._build_agent([allowed])

        self.assertIs(result, built)
        build_model.assert_called_once_with()
        create.assert_called_once_with(
            model="model",
            tools=[allowed],
            system_prompt=agent_module.SYSTEM_PROMPT,
            checkpointer="memory",
        )


class AgentStreamTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, source_events, text="test"):
        with (
            patch.object(agent_module, "agent", FakeAgent(source_events)),
            patch.object(agent_module, "USER_TOOL_NAMES", {
                "get_expiring_customers", "mark_customer_contacted", "calc_premium"
            }),
        ):
            return [event async for event in agent_module.run_agent_stream(text, "thread")]

    async def test_init_agent_filters_unrelated_mcp_tools(self):
        allowed = SimpleNamespace(name="get_expiring_customers")
        another_allowed = SimpleNamespace(name="calc_premium")
        feishu_tool = SimpleNamespace(name="get_project_list")
        low_level_lookup = SimpleNamespace(name="lookup_ncd")
        rebuilt_agent = object()

        with (
            patch.object(
                agent_module,
                "load_mcp_tools",
                new=AsyncMock(return_value=[allowed, feishu_tool, low_level_lookup, another_allowed]),
            ),
            patch.object(agent_module, "_build_agent", return_value=rebuilt_agent) as build_agent,
            patch.object(agent_module, "agent"),
            patch.object(agent_module, "USER_TOOL_NAMES", set()),
        ):
            await agent_module.init_agent()
            installed = build_agent.call_args.args[0]
            installed_names = {tool.name for tool in installed}
            self.assertIn("get_expiring_customers", installed_names)
            self.assertIn("calc_premium", installed_names)
            self.assertNotIn("get_project_list", installed_names)
            self.assertNotIn("lookup_ncd", installed_names)
            self.assertIs(agent_module.agent, rebuilt_agent)
            self.assertEqual(agent_module.USER_TOOL_NAMES, installed_names)

    async def test_real_lightweight_agent_preserves_stream_and_thread_memory(self):
        model = GenericFakeChatModel(messages=iter([
            AIMessage(content="第一轮。"),
            AIMessage(content="第二轮。"),
        ]))
        with patch.object(agent_module, "_build_model", return_value=model):
            lightweight = agent_module._build_agent([])

        with patch.object(agent_module, "agent", lightweight):
            first = [event async for event in agent_module.run_agent_stream("你好", "same-thread")]
            second = [event async for event in agent_module.run_agent_stream("继续", "same-thread")]

        self.assertEqual(self.spoken_segments(first), ["第一轮。"])
        self.assertEqual(self.spoken_segments(second), ["第二轮。"])
        snapshot = lightweight.get_state({"configurable": {"thread_id": "same-thread"}})
        human_messages = [message for message in snapshot.values["messages"] if message.type == "human"]
        self.assertEqual([message.content for message in human_messages], ["你好", "继续"])

    @staticmethod
    def spoken_segments(events):
        segments = []
        current = []
        for event in events:
            if event["type"] == "agent_delta":
                current.append(event["text"])
            elif event["type"] == "segment_end":
                segments.append("".join(current))
                current = []
        return segments

    async def test_silent_tool_discards_internal_narration(self):
        events = await self.collect([
            model_start(),
            text_chunk("选到这位客户了，我先把他标记一下。"),
            tool_chunk("get_expiring_customers"),
            model_end("get_expiring_customers"),
            tool_event("on_tool_start", "get_expiring_customers"),
            tool_event("on_tool_end", "get_expiring_customers"),
            model_start(),
            text_chunk("您好，我是保险公司的车险服务专员。"),
            model_end(),
        ], text=agent_module.OUTBOUND_OPENING)
        self.assertEqual(self.spoken_segments(events), ["您好，我是保险公司的车险服务专员。"])
        self.assertNotIn("选到这位客户了", "".join(e.get("text", "") for e in events))

    async def test_responses_api_structured_text_blocks_are_streamed(self):
        answer = "您好，我是保险公司的车险服务专员。"
        events = await self.collect([
            model_start(), responses_text_chunk(answer), model_end(),
        ])

        self.assertEqual(self.spoken_segments(events), [answer])
        event_types = [event["type"] for event in events]
        self.assertEqual(event_types.count("model_text_started"), 1)
        self.assertLess(event_types.index("model_text_started"), event_types.index("agent_delta"))

    async def test_responses_api_structured_text_is_returned_by_non_streaming_path(self):
        answer = "您好，我是保险公司的车险服务专员。"
        message = AIMessage(content=[{"type": "text", "text": answer}])

        with patch.object(agent_module, "agent", FakeInvokeAgent([message])):
            result = await agent_module.run_agent("test", "thread")

        self.assertEqual(result["answer"], answer)

    async def test_business_tool_has_separate_before_and_after_segments(self):
        before = "好的，家用SUV、5座，我给您算一下今年的报价。"
        after = "周先生，今年的报价已经出来了。"
        events = await self.collect([
            model_start(), text_chunk(before), tool_chunk("calc_premium"), model_end("calc_premium"),
            tool_event("on_tool_start", "calc_premium"),
            tool_event("on_tool_end", "calc_premium"),
            model_start(), text_chunk(after), model_end(),
        ])
        self.assertEqual(self.spoken_segments(events), [before, after])
        types = [event["type"] for event in events]
        first_end = types.index("segment_end")
        tool_start = types.index("tool_start")
        tool_end = types.index("tool_end")
        second_end = len(types) - 1 - types[::-1].index("segment_end")
        self.assertLess(first_end, tool_start)
        self.assertLess(tool_end, second_end)


if __name__ == "__main__":
    unittest.main()

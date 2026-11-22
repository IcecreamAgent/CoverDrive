import gzip
import json
import struct
import unittest

from app.asr import AsrContextHistory, AsrResponse, _full_client_request


class AsrResultTests(unittest.TestCase):
    @staticmethod
    def request_payload(context_data=None):
        packet = _full_client_request(1, context_data)
        payload_size = struct.unpack(">I", packet[8:12])[0]
        return json.loads(gzip.decompress(packet[12:12 + payload_size]))

    def test_request_explicitly_asks_for_full_cumulative_result(self):
        payload = self.request_payload()
        request = payload["request"]
        self.assertEqual(request["result_type"], "full")
        self.assertTrue(request["show_utterances"])
        self.assertTrue(request["enable_nonstream"])

    def test_context_is_serialized_as_provider_json_string(self):
        context_data = [{"text": "客户：我想续保"}, {"text": "业务场景：车险续保"}]
        payload = self.request_payload(context_data)
        context = payload["request"]["corpus"]["context"]
        self.assertIsInstance(context, str)
        self.assertEqual(json.loads(context), {
            "context_type": "dialog_ctx",
            "context_data": context_data,
        })

    def test_empty_context_does_not_send_corpus(self):
        self.assertNotIn("corpus", self.request_payload([])["request"])

    def test_multiple_utterances_are_combined(self):
        response = AsrResponse(payload_msg={
            "result": {
                "text": "我想了解续保。",
                "utterances": [
                    {"text": "你好，", "definite": True},
                    {"text": "我想了解续保。", "definite": False},
                ],
            }
        })
        self.assertEqual(response.text, "你好，我想了解续保。")

    def test_alphanumeric_utterances_keep_word_boundary(self):
        response = AsrResponse(payload_msg={
            "result": {"utterances": [{"text": "hello"}, {"text": "world"}]}
        })
        self.assertEqual(response.text, "hello world")

    def test_chinese_utterances_do_not_gain_spaces(self):
        response = AsrResponse(payload_msg={
            "result": {"utterances": [{"text": "你好"}, {"text": "我想续保"}]}
        })
        self.assertEqual(response.text, "你好我想续保")

    def test_text_fallback_without_utterances(self):
        response = AsrResponse(payload_msg={"result": {"text": "完整文本"}})
        self.assertEqual(response.text, "完整文本")


class AsrContextHistoryTests(unittest.TestCase):
    def test_recent_dialog_is_newest_first_with_domain_hint_last(self):
        history = AsrContextHistory(
            max_turns=2,
            max_chars=200,
            domain_hint="车险 续保",
        )
        history.add("客户", "第一轮")
        history.add("坐席", "第一答复。")
        history.add("坐席", "补充内容。")
        history.add("客户", "第二轮")
        self.assertEqual(history.context_data(), [
            {"text": "客户：第二轮"},
            {"text": "坐席：第一答复。补充内容。"},
            {"text": "客户：第一轮"},
            {"text": "业务场景：车险 续保"},
        ])

    def test_history_keeps_only_configured_rounds(self):
        history = AsrContextHistory(max_turns=1, max_chars=100)
        history.add("客户", "旧问题")
        history.add("坐席", "旧回答")
        history.add("客户", "新问题")
        history.add("坐席", "新回答")
        self.assertEqual(history.context_data(), [
            {"text": "坐席：新回答"},
            {"text": "客户：新问题"},
        ])

    def test_context_respects_character_budget(self):
        history = AsrContextHistory(max_turns=2, max_chars=20, domain_hint="车险")
        history.add("客户", "这是一段很长很长很长的客户文本")
        context = history.context_data()
        self.assertLessEqual(sum(len(item["text"]) for item in context), 20)
        self.assertEqual(context[-1], {"text": "业务场景：车险"})

    def test_disabled_context_is_empty(self):
        history = AsrContextHistory(enabled=False, domain_hint="车险")
        history.add("客户", "续保")
        self.assertEqual(history.context_data(), [])


if __name__ == "__main__":
    unittest.main()

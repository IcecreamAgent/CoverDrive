"""Native Seeduplex JSON protocol and cancellation regressions."""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from voice_agent.providers import seeduplex


def settings(**overrides):
    return SimpleNamespace(
        **(
            {
                "seeduplex_url": "wss://speech.example.test/duplex",
                "seeduplex_api_key": "test-new-speech-key",
                "seeduplex_app_id": "test-app-id",
                "seeduplex_access_key": "test-legacy-access-key",
                "seeduplex_app_key": "fake-native-app-key",
                "seeduplex_resource_id": "volc.speech.dialog",
                "seeduplex_model": "fake-duplex-model",
                "seeduplex_speaker": "fake-speaker",
                "seeduplex_timeout": 0.2,
            }
            | overrides
        )
    )


class FakeSocket:
    def __init__(self, *, create_error=False, hang=False):
        self.sent = []
        self.queue = asyncio.Queue()
        self.closed = False
        self.create_error = create_error
        self.hang = hang

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.closed = True

    async def send(self, data):
        assert isinstance(data, str)  # No binary frame from the older protocol.
        event = json.loads(data)
        self.sent.append(event)
        if self.hang:
            return
        if event["type"] == "session.create":
            if self.create_error:
                self.push(
                    {"type": "error", "error": {"code": 45000030, "message": "private-key-here"}}
                )
            else:
                self.push({"type": "session.created", "session": {"id": "native-session"}})
        elif event["type"] == "session.close":
            self.push({"type": "session.closed"})

    async def recv(self):
        return await self.queue.get()

    def push(self, event):
        self.queue.put_nowait(json.dumps(event, ensure_ascii=False))


def patch_connect(monkeypatch, socket):
    captured = {}

    def connect(url, **kwargs):
        captured.update(url=url, **kwargs)
        return socket

    monkeypatch.setattr(seeduplex, "connect", connect)
    return captured


async def test_public_history_is_initialized_in_the_documented_dialog_extension(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    history = [("预算4000到6000，比较两个方案", "北辰R47为4599元，青鹭S82为5299元。")]
    async with seeduplex.SeeduplexClient(settings()).stream(
        instructions="沿用已公开内容。", history=history,
    ):
        pass
    assert socket.sent[0]["extension"] == {"dialog": {"dialog_context": [
        {"role": "user", "text": history[0][0]},
        {"role": "assistant", "text": history[0][1]},
    ]}}
    assert "extension" not in socket.sent[0]["session"]
    assert not any(event["type"] == "conversation.item.create" for event in socket.sent)


@pytest.mark.parametrize("history", [{}, "history", [("q",)], [("q", "")], [("q", 7)],
                                     [("q", "a")] * 21])
async def test_invalid_initial_history_is_rejected_before_connect(monkeypatch, history):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    with pytest.raises(ValueError):
        async with seeduplex.SeeduplexClient(settings()).stream(
            instructions="简短回答。", history=history,
        ):
            pass
    assert not socket.sent


async def test_native_session_uses_s16le_output_and_only_new_auth_header(monkeypatch):
    socket = FakeSocket()
    captured = patch_connect(monkeypatch, socket)
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="简短回答。") as stream:
        assert stream.session_id == "native-session"
        await stream.send_audio(b"\x01\x00" * 320)
    assert socket.closed
    assert captured["additional_headers"] == {"X-Api-Key": "test-new-speech-key"}
    assert captured["proxy"] is None
    event = socket.sent[0]
    assert event["type"] == "session.create"
    assert event["extension"] == {}
    session = event["session"]
    assert session["model"] == "fake-duplex-model"
    assert session["audio"]["input"]["format"] == {"type": "pcm", "rate": 16000}
    assert session["audio"]["output"]["format"] == {"type": "pcm_s16le", "rate": 24000}
    assert session["audio"]["output"]["voice"] == "fake-speaker"
    assert session["tools"] == [seeduplex.DEEP_AGENT_TOOL]
    assert "function" not in session["tools"][0]  # Native flat schema, not Chat API nesting.
    sent_audio = socket.sent[1]
    assert sent_audio["type"] == "input_audio_buffer.append"
    assert base64.b64decode(sent_audio["audio"]) == b"\x01\x00" * 320
    assert socket.sent[-1]["type"] == "session.close"


def test_legacy_auth_uses_user_configured_app_credentials():
    headers = seeduplex.auth_headers(settings(seeduplex_api_key=""))
    assert headers["X-Api-App-Id"] == "test-app-id"
    assert headers["X-Api-Access-Key"] == "test-legacy-access-key"
    assert headers["X-Api-App-Key"] == "fake-native-app-key"
    assert headers["X-Api-Resource-Id"] == "volc.speech.dialog"
    assert headers["X-Api-Request-Id"]
    assert "X-Api-Key" not in headers


def test_auth_never_implicitly_reuses_other_service_keys():
    value = settings(seeduplex_api_key="", seeduplex_app_id="", seeduplex_access_key="")
    value.tts_api_key = "unrelated-tts-key"
    value.asr_access_key = "unrelated-asr-key"
    with pytest.raises(ValueError, match="SEEDUPLEX_API_KEY"):
        seeduplex.auth_headers(value)


async def test_muting_cancellation_and_text_replacement_wire_events(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    async with seeduplex.SeeduplexClient(settings()).stream(
        instructions="test", tools=[]
    ) as stream:
        await stream.set_muted(True)
        await stream.set_muted(True)  # Idempotent.
        await stream.send_audio(b"\x00\x00")  # Unmute before audio resumes.
        await stream.commit_audio()
        await stream.cancel()
        await stream.replace_text("第一句。")
        await stream.replace_text("", final=True)
        await stream.send_text("直接朗读。")
        await stream.greet("你好。")
    assert [x["type"] for x in socket.sent] == [
        "session.create",
        "input_audio_mute.commit",
        "input_audio_unmute.commit",
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "response.cancel",
        "speech_text_buffer.replacement.append",
        "speech_text_buffer.replacement.commit",
        "speech_text_buffer.replacement.commit",
        "speech_text_buffer.commit",
        "session.close",
    ]


@pytest.mark.parametrize("length", [10, 4001])
async def test_prepared_task_speech_uses_native_pcm_without_fake_tool_ids(monkeypatch, length):
    class SpeechSocket(FakeSocket):
        async def send(self, data):
            await super().send(data)
            if json.loads(data)["type"] == "speech_text_buffer.commit":
                response = str(len(self.sent))
                self.push(
                    {
                        "type": "response.output_audio.started",
                        "response_id": response if length > 10 else "",
                    }
                )
                self.push({"type": "response.output_audio.delta", "delta": "AQACAA=="})
                self.push({"type": "response.output_audio.done", "response_id": response})

    socket = SpeechSocket()
    patch_connect(monkeypatch, socket)
    text = "答" * length
    chunks = [pcm async for pcm in seeduplex.SeeduplexClient(settings()).speech(text)]
    assert chunks == [b"\x01\x00\x02\x00"] * ((length + 3999) // 4000)
    assert socket.sent[0]["session"]["tools"] == []
    spoken = [
        event["text"] for event in socket.sent if event["type"] == "speech_text_buffer.commit"
    ]
    assert "".join(spoken) == text
    assert all(len(part) <= 4000 for part in spoken)
    assert all(event["type"] != "conversation.item.create" for event in socket.sent)
    assert socket.closed


async def test_conversation_and_prepared_result_share_voice_parameters_and_service_style(monkeypatch):
    class SpeechSocket(FakeSocket):
        async def send(self, data):
            await super().send(data)
            if json.loads(data)["type"] == "speech_text_buffer.commit":
                self.push({"type": "response.output_audio.started", "response_id": "result"})
                self.push({"type": "response.output_audio.delta", "delta": "AQACAA=="})
                self.push({"type": "response.output_audio.done", "response_id": "result"})

    conversation, result = FakeSocket(), SpeechSocket()
    sockets = iter([conversation, result])
    monkeypatch.setattr(seeduplex, "connect", lambda *_args, **_kwargs: next(sockets))
    configured = settings()
    configured.tts_speaker = "unrelated-standalone-tts-voice"
    configured.tts_loudness_rate = 100
    client = seeduplex.SeeduplexClient(configured)
    async with client.stream(instructions="先确认身份，再了解续保需求。"):
        pass
    prepared = "按已确认方案，合计4254.49元。"
    assert [pcm async for pcm in client.speech(prepared)] == [b"\x01\x00\x02\x00"]

    native_request, result_request = [socket.sent[0]["session"] for socket in (conversation, result)]
    assert native_request["model"] == result_request["model"]
    assert native_request["audio"] == result_request["audio"]
    assert result_request["audio"]["output"] == {
        "format": {"type": "pcm_s16le", "rate": 24000},
        "voice": configured.seeduplex_speaker, "speed": 0, "loudness": 0,
    }
    for request in (native_request, result_request):
        assert request["instructions"].startswith(seeduplex.VOICE_STYLE_INSTRUCTIONS + "\n\n")
        assert request["instructions"].count(seeduplex.VOICE_STYLE_INSTRUCTIONS) == 1
    assert "先确认身份" in native_request["instructions"]
    assert "不改写" in result_request["instructions"] and "新的业务内容" in result_request["instructions"]
    assert native_request["tools"] == [seeduplex.DEEP_AGENT_TOOL]
    assert result_request["tools"] == []
    assert [event["text"] for event in result.sent if event["type"] == "speech_text_buffer.commit"] == [prepared]
    assert all(socket.closed and socket.sent[-1]["type"] == "session.close"
               for socket in (conversation, result))


@pytest.mark.parametrize("failure", ["provider_error", "empty_audio", "timeout"])
async def test_prepared_result_failure_keeps_style_and_closes_its_connection(monkeypatch, failure):
    class FailedSpeechSocket(FakeSocket):
        async def send(self, data):
            await super().send(data)
            if json.loads(data)["type"] != "speech_text_buffer.commit":
                return
            if failure == "provider_error":
                self.push({"type": "error", "error": {"code": "failed", "message": "private-body"}})
            elif failure == "empty_audio":
                self.push({"type": "response.output_audio.started", "response_id": "empty"})
                self.push({"type": "response.output_audio.done", "response_id": "empty"})

    socket = FailedSpeechSocket()
    patch_connect(monkeypatch, socket)
    expected = {"provider_error": seeduplex.SeeduplexProviderError,
                "empty_audio": seeduplex.SeeduplexProtocolError, "timeout": TimeoutError}[failure]
    with pytest.raises(expected) as caught:
        _ = [pcm async for pcm in seeduplex.SeeduplexClient(settings(seeduplex_timeout=0.01)).speech("准确播报结果。")]
    assert "private-body" not in str(caught.value)
    assert socket.sent[0]["session"]["instructions"].startswith(seeduplex.VOICE_STYLE_INSTRUCTIONS)
    assert socket.closed and socket.sent[-1]["type"] == "session.close"


async def test_prepared_speech_cancellation_releases_the_native_connection(monkeypatch):
    entered = asyncio.Event()

    class PausedSpeechSocket(FakeSocket):
        async def send(self, data):
            await super().send(data)
            if json.loads(data)["type"] == "speech_text_buffer.commit":
                entered.set()

    socket = PausedSpeechSocket()
    patch_connect(monkeypatch, socket)
    generator = seeduplex.SeeduplexClient(settings()).speech("一条任务结果。")
    task = asyncio.create_task(anext(generator))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert socket.sent[0]["session"]["instructions"].startswith(seeduplex.VOICE_STYLE_INSTRUCTIONS)
    assert socket.closed
    assert socket.sent[-1]["type"] == "session.close"


async def test_tool_results_are_grouped_and_call_ids_are_preserved(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="test") as stream:
        await stream.send_tool_results(
            [
                {"call_id": "call_a", "text": "任务一结果"},
                {"call_id": "call_b", "text": "任务二结果"},
            ]
        )
    assert socket.sent[1]["items"] == [
        {
            "call_id": "call_a",
            "role": "tool",
            "content": [{"type": "input_text", "text": "任务一结果"}],
        },
        {
            "call_id": "call_b",
            "role": "tool",
            "content": [{"type": "input_text", "text": "任务二结果"}],
        },
    ]


async def test_supervision_updates_instructions_without_changing_voice_or_tools(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="原提示") as stream:
        await stream.update_instructions("当前任务需要工具，结果由你讲解。")
        await stream.update_instructions("原提示")
    original, first, second = socket.sent[:3]
    assert first["type"] == second["type"] == "session.update"
    assert first["session"]["instructions"] == (
        seeduplex.VOICE_STYLE_INSTRUCTIONS + "\n\n当前任务需要工具，结果由你讲解。"
    )
    assert second["session"] == original["session"]
    for key in ("model", "audio", "tools"):
        assert first["session"][key] == original["session"][key]


async def test_context_injection_only_accepts_completed_pairs(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="test") as stream:
        await stream.add_history([("二加二？", "四。")])
        with pytest.raises(ValueError):
            await stream.add_history([("没有回答的用户发言", "")])
    items = socket.sent[1]["items"]
    assert [item["role"] for item in items] == ["user", "assistant"]
    assert [item["content"][0]["text"] for item in items] == ["二加二？", "四。"]
    assert all(item["type"] == "message" and item["id"] for item in items)


async def test_assistant_answer_completes_existing_user_without_inventing_another(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    answers = ["四。", "答" * 16000]
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="test") as stream:
        for answer in answers:
            await stream.add_assistant_answer(answer)
    injected = socket.sent[1:-1]
    assert len(injected) == len(answers)
    assert all(event["type"] == "conversation.item.create" for event in injected)
    items = [item for event in injected for item in event["items"]]
    assert [item["role"] for item in items] == ["assistant", "assistant"]
    assert [item["content"] for item in items] == [
        [{"type": "input_text", "text": answer}] for answer in answers
    ]
    assert all(item["type"] == "message" and item["id"] for item in items)
    assert len({item["id"] for item in items}) == len(items)


@pytest.mark.parametrize("answer", [None, 4, [], "", " \n\t", "答" * 16001])
async def test_invalid_assistant_answer_is_rejected_before_sending(monkeypatch, answer):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="test") as stream:
        with pytest.raises(ValueError, match="bounded nonempty text"):
            await stream.add_assistant_answer(answer)
        assert [event["type"] for event in socket.sent] == ["session.create"]


async def test_receive_retains_native_ids_and_accepts_audio_without_response_id(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    events = [
        {"type": "conversation.item.input_audio_transcription.started", "item_id": "i1"},
        {
            "type": "response.output_text.delta",
            "response_id": "r1",
            "question_id": "q1",
            "delta": "四。",
        },
        {"type": "response.output_audio.delta", "delta": "AQACAA=="},
    ]
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="test") as stream:
        for event in events:
            socket.push(event)
        received = stream.receive()
        assert [await anext(received) for _ in events] == events
        await received.aclose()
    assert seeduplex.decode_audio(events[-1]) == b"\x01\x00\x02\x00"


@pytest.mark.parametrize("raw", [b"{}", "not json", "[]", "{}", '{"type":3}'])
def test_invalid_json_frames_are_rejected(raw):
    with pytest.raises(seeduplex.SeeduplexProtocolError):
        seeduplex.decode_event(raw)


@pytest.mark.parametrize("audio", [None, "!!", "AQ==", "非ASCII"])
def test_invalid_base64_or_partial_int16_is_rejected(audio):
    with pytest.raises(seeduplex.SeeduplexProtocolError):
        seeduplex.decode_audio({"type": "response.output_audio.delta", "delta": audio})


@pytest.mark.parametrize("arguments", [
    {"question": "请规划行程"},
    {"question": "改成免费停车的民宿", "replaces_task_id": "task-existing"},
])
def test_tool_arguments_are_parsed_but_never_executed(arguments):
    event = {
        "type": "response.function_call_arguments.done",
        "items": [
            {
                "call_id": "call_native",
                "name": "delegate_deep_agent",
                "arguments": json.dumps(arguments),
            }
        ],
    }
    assert seeduplex.parse_tool_calls(event) == [
        seeduplex.ToolCall("call_native", "delegate_deep_agent", arguments)
    ]
    event["items"][0]["arguments"] = "[]"
    with pytest.raises(seeduplex.SeeduplexProtocolError):
        seeduplex.parse_tool_calls(event)


def test_upstream_error_payloads_never_appear_in_safe_exception():
    with pytest.raises(seeduplex.SeeduplexProviderError) as caught:
        seeduplex.decode_event(
            '{"type":"error","error":{"code":45000030,"message":"secret authentication body"}}'
        )
    assert caught.value.code == 45000030
    assert "secret" not in str(caught.value)
    assert "authentication" not in str(caught.value)


async def test_handshake_error_and_timeout_release_socket(monkeypatch):
    socket = FakeSocket(create_error=True)
    patch_connect(monkeypatch, socket)
    with pytest.raises(seeduplex.SeeduplexProviderError) as caught:
        async with seeduplex.SeeduplexClient(settings()).stream(instructions="test"):
            raise AssertionError("Rejected session cannot be yielded")
    assert caught.value.code == 45000030
    assert "private" not in str(caught.value)
    assert socket.closed
    socket = FakeSocket(hang=True)
    patch_connect(monkeypatch, socket)
    with pytest.raises(TimeoutError):
        async with seeduplex.SeeduplexClient(settings(seeduplex_timeout=0.01)).stream(
            instructions="test"
        ):
            raise AssertionError("Missing handshake cannot be yielded")
    assert socket.closed


async def test_cancelled_receive_closes_session_and_releases_socket(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    entered = asyncio.Event()

    async def run():
        async with seeduplex.SeeduplexClient(settings()).stream(instructions="test") as stream:
            entered.set()
            async for _event in stream.receive():
                pass

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert socket.closed
    assert socket.sent[-1]["type"] == "session.close"


async def test_concurrent_receive_is_rejected(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="test") as stream:
        consumer = stream.receive()
        waiting = asyncio.create_task(anext(consumer))
        await asyncio.sleep(0)
        with pytest.raises(seeduplex.SeeduplexProtocolError, match="one event consumer"):
            await anext(stream.receive())
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting


async def test_close_ack_uses_active_reader_without_concurrent_recv(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="test") as stream:

        async def consume():
            return [event async for event in stream.receive()]

        consumer = asyncio.create_task(consume())
        await asyncio.sleep(0)
        await stream.close()
        assert await consumer == [{"type": "session.closed"}]


async def test_invalid_audio_fails_before_wire_send(monkeypatch):
    socket = FakeSocket()
    patch_connect(monkeypatch, socket)
    async with seeduplex.SeeduplexClient(settings()).stream(instructions="test") as stream:
        with pytest.raises(ValueError):
            await stream.send_audio(b"\x00")
        with pytest.raises(ValueError):
            await stream.send_audio(bytes(64002))
        assert len(socket.sent) == 1


async def test_unconfigured_native_model_fails_before_opening_provider(monkeypatch):
    def forbidden_connect(*args, **kwargs):
        raise AssertionError("Missing model must not connect to the provider")

    monkeypatch.setattr(seeduplex, "connect", forbidden_connect)
    with pytest.raises(ValueError, match="SEEDUPLEX_MODEL"):
        async with seeduplex.SeeduplexClient(settings(seeduplex_model="")).stream(instructions="test"):
            pass



def test_legacy_auth_never_supplies_an_implicit_app_key():
    with pytest.raises(ValueError, match="SEEDUPLEX_APP_KEY"):
        seeduplex.auth_headers(settings(seeduplex_api_key="", seeduplex_app_key=""))

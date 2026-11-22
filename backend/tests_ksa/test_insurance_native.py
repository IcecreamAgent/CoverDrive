"""Project rules at the copied native protocol boundary, using no real services."""

import asyncio
import json

from test_realtime import native as native
from test_realtime import output, tool_event
from test_realtime import Socket

from config_fixture import configured_settings
from voice_agent.realtime import RealtimeVoiceSession


def delegation(question, call_id="insurance-call"):
    return tool_event(calls=[{
        "call_id": call_id,
        "name": "delegate_deep_agent",
        "arguments": json.dumps({"question": question}, ensure_ascii=False),
    }])


def test_native_is_the_default_without_an_implicit_cascade_fallback(monkeypatch):
    monkeypatch.delenv("VOICE_PROVIDER", raising=False)
    assert configured_settings(_env_file=None).voice_provider == "seeduplex"


async def test_partial_insurance_input_cannot_start_a_business_delegation(native):
    socket, instance, _, state = native
    await instance.handle_native_event({
        "type": "conversation.item.input_audio_transcription.started",
        "item_id": "q1", "round_id": 1,
    })
    await instance.handle_native_event({
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": "q1", "delta": "我要投保",
    })
    await output(instance, text="我来核对方案。")
    await instance.handle_native_event(delegation("请立即创建生效保单"))
    await asyncio.sleep(0)
    assert state.calls == []
    assert not instance.background.records and not instance.background.by_call
    assert not instance.native_turn.ended
    assert instance.insurance_authorization.version == 0
    assert instance.insurance_authorization.final_text == ""
    transcript = [event for event in socket.drain()
                  if isinstance(event, dict) and event.get("type") == "transcript"]
    assert transcript[-1]["text"] == "我要投保"
    assert transcript[-1]["is_final"] is False

    final = "我要投保吗？先不要办理。"
    await instance.handle_native_event({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "q1", "text": final,
    })
    assert instance.native_turn.ended
    assert instance.native_turn.question == final
    assert instance.insurance_authorization.version == 1
    assert instance.insurance_authorization.final_text == final
    assert state.calls == []
    assert not instance.background.records and not instance.background.by_call
    transcript = [event for event in socket.drain()
                  if isinstance(event, dict) and event.get("type") == "transcript"]
    assert transcript[-1]["text"] == final
    assert transcript[-1]["is_final"] is True


async def test_missing_native_credentials_reports_configuration_error_without_cascade(native):
    socket, instance, _, state = native
    await instance.close_native()
    instance.settings.seeduplex_api_key = ""
    instance.settings.seeduplex_app_id = ""
    instance.settings.seeduplex_access_key = ""
    opened = state.opened
    await instance.enable_native()
    assert state.opened == opened
    errors = [event for event in socket.drain()
              if isinstance(event, dict) and event.get("type") == "error"]
    assert errors and errors[-1]["stage"] == "config"
    assert "SEEDUPLEX_APP_ID" in errors[-1]["message"]
    assert "SEEDUPLEX_ACCESS_KEY" in errors[-1]["message"]
    assert state.calls == []


async def test_missing_llm_key_constructs_session_and_reports_configuration_error_without_connecting():
    config = configured_settings(
        _env_file=None, llm_api_key="", seeduplex_app_id="fake-app-id",
        seeduplex_access_key="fake-access-key", insurance_tools_enabled=False,
    )
    socket = Socket()
    instance = RealtimeVoiceSession(socket, config)
    try:
        await instance.enable_native()
        assert instance.native is None and not instance.listening
        errors = [event for event in socket.drain()
                  if isinstance(event, dict) and event.get("type") == "error"]
        assert errors and errors[-1]["stage"] == "config"
        assert "LLM_API_KEY" in errors[-1]["message"]
    finally:
        await instance.close_native()
        await instance.cancel_response()
        await instance.duplex.close()

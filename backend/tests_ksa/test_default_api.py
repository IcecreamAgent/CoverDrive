"""Production entry points use native sessions and task executors, with no live services."""

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import main
from voice_agent import agent
from config_fixture import configured_settings
from voice_agent.insurance import InsuranceAuthorization


def receive_control(websocket):
    """Execution telemetry may precede its corresponding control message."""
    for _ in range(8):
        event = websocket.receive_json()
        if event["type"] != "execution_event":
            return event
    raise AssertionError("Expected a control message after bounded telemetry")


@pytest.fixture
def config(monkeypatch):
    isolated = configured_settings(
        _env_file=None, llm_api_key="fake-key", voice_auth_token="isolated-access-token",
        seeduplex_app_id="fake-app", seeduplex_access_key="fake-access",
        insurance_tools_enabled=False, insurance_writes_enabled=False,
    )
    monkeypatch.setattr(main, "get_voice_settings", lambda: isolated)
    monkeypatch.setattr(main, "_text_sessions", {})

    async def forbidden_legacy(*args, **kwargs):
        raise AssertionError("Default API must not invoke a historical cascade implementation")

    for name in ("run_agent", "run_agent_stream", "transcribe", "synthesize", "init_agent", "warm_fillers"):
        monkeypatch.setattr(main, name, forbidden_legacy)
    return isolated


def test_text_api_requires_access_token_before_constructing_an_executor(config, monkeypatch):
    def forbidden_executor(*args, **kwargs):
        raise AssertionError("Unauthenticated request must not construct a task executor")

    monkeypatch.setattr(agent, "AgentSession", forbidden_executor)
    with TestClient(main.app) as client:
        response = client.post("/api/text", json={"text": "请介绍方案"})
        assert response.status_code == 401
        assert "访问令牌" in response.json()["detail"]


def test_text_api_uses_new_task_executor_and_returns_only_complete_final_answer(config, monkeypatch):
    calls = []

    class Executor:
        def __init__(self, settings, *, task_only):
            assert settings is config and task_only
            self.authorization = InsuranceAuthorization()
            self._authorization_version = -1
            self.graph = object()
            self.final_answer = ""

        async def stream(self, text, *, force_deep):
            assert force_deep and self.graph is None
            assert self._authorization_version == self.authorization.version
            calls.append((text, self.authorization.final_text))
            yield "尚未完成的内部进度。"
            self.final_answer = "隔离任务的完整最终答案。"
            yield self.final_answer

    monkeypatch.setattr(agent, "AgentSession", Executor)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/text", json={"text": "我同意投保，请出单。", "session_id": "isolated-http-session"},
            headers={"Authorization": "Bearer " + config.voice_auth_token},
        )
    assert response.status_code == 200
    assert response.json() == {
        "transcript": "我同意投保，请出单。", "answer": "隔离任务的完整最终答案。", "events": [],
    }
    assert calls == [("我同意投保，请出单。", "我同意投保，请出单。")]


def test_text_api_without_complete_final_answer_fails_closed(config, monkeypatch):
    class IncompleteExecutor:
        def __init__(self, settings, *, task_only):
            self.authorization = InsuranceAuthorization()
            self.final_answer = ""

        async def stream(self, text, *, force_deep):
            yield "只有中间进度，没有完成。"

    monkeypatch.setattr(agent, "AgentSession", IncompleteExecutor)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/text", json={"text": "核对方案"},
            headers={"Authorization": "Bearer " + config.voice_auth_token},
        )
    assert response.status_code == 502
    assert "只有中间进度" not in response.text


def test_whole_audio_api_is_gone_without_calling_legacy_recognition_or_agent(config):
    with TestClient(main.app) as client:
        response = client.post("/api/voice", files={"audio": ("fixture.wav", b"isolated-audio", "audio/wav")})
    assert response.status_code == 410
    assert "/ws/voice" in response.json()["detail"]


def test_production_routes_do_not_expose_the_unguarded_historical_pipeline(config):
    paths = [getattr(route, "path", None) for route in main.app.routes]
    assert paths.count("/ws/voice") == 1
    assert "/ws/voice/legacy" not in paths


@pytest.mark.parametrize("auth", [
    {"type": "auth", "token": "wrong-isolated-token"},
    {"type": "listen"},
    {"type": "auth", "token": 17},
])
def test_native_websocket_rejects_invalid_access_token_before_ready(config, auth):
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/voice") as websocket:
            websocket.send_json(auth)
            with pytest.raises(WebSocketDisconnect) as error:
                websocket.receive_json()
            assert error.value.code == 1008


def test_native_ready_and_missing_configuration_error_keep_control_channel_recoverable(config):
    config.llm_api_key = ""
    with TestClient(main.app) as client:
        with client.websocket_connect("/ws/voice") as websocket:
            websocket.send_json({"type": "auth", "token": config.voice_auth_token})
            ready = websocket.receive_json()
            assert ready["type"] == "ready" and ready["voice_provider"] == "seeduplex"
            for private_value in (config.llm_model_id, config.seeduplex_model,
                                  config.seeduplex_access_key, config.voice_auth_token):
                assert private_value not in json.dumps(ready)
            websocket.send_json({"type": "listen", "playback_barriers": True})
            failure = receive_control(websocket)
            assert failure["type"] == "error" and failure["stage"] == "config"
            assert "LLM_API_KEY" in failure["message"]
            websocket.send_json({"type": "ping"})
            assert receive_control(websocket)["type"] == "pong"
            websocket.send_json({"type": "listen"})
            assert receive_control(websocket)["stage"] == "config"


def test_health_reports_native_orchestration_and_missing_configuration(config):
    config.llm_api_key = ""
    with TestClient(main.app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    health = response.json()
    assert health["voice_provider"] == health["decision_provider"] == "seeduplex"
    assert health["voice_orchestration"] == "native_tool_only"
    assert health["standalone_tts_enabled"] is False
    assert health["fast_llm_enabled"] is False and health["semantic_router_enabled"] is False
    assert health["configured"] is False and "LLM_API_KEY" in health["missing"]

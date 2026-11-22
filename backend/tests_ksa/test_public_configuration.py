"""Public checkout requires user-owned services and never publishes their identities."""

import pytest
from fastapi.testclient import TestClient

from app import main
from config_fixture import configured_settings
from voice_agent.config import Settings


@pytest.fixture
def empty_configuration(monkeypatch):
    # These tests must exercise shipped defaults, not the developer's shell.
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    return Settings(_env_file=None)


def test_shipped_defaults_require_user_owned_models_and_keys(empty_configuration):
    config = empty_configuration
    for field in (
        "llm_base_url", "llm_model_id", "llm_api_key", "fast_llm_model_id",
        "seeduplex_url", "seeduplex_model", "seeduplex_speaker",
        "seeduplex_api_key", "seeduplex_app_id", "seeduplex_access_key", "seeduplex_app_key",
    ):
        assert getattr(config, field) == "", field
    assert set(config.missing(voice=False)) == {"LLM_BASE_URL", "LLM_MODEL_ID", "LLM_API_KEY"}
    assert {
        "LLM_BASE_URL", "LLM_MODEL_ID", "LLM_API_KEY",
        "SEEDUPLEX_URL", "SEEDUPLEX_MODEL", "SEEDUPLEX_SPEAKER",
        "SEEDUPLEX_APP_ID", "SEEDUPLEX_ACCESS_KEY", "SEEDUPLEX_APP_KEY",
    } <= set(config.missing())
    assert config.insurance_writes_enabled is False


@pytest.mark.parametrize("authentication", [
    {"seeduplex_api_key": "fake-user-native-key", "seeduplex_app_key": ""},
    {"seeduplex_app_id": "fake-user-app", "seeduplex_access_key": "fake-user-access"},
])
def test_user_owned_model_and_authentication_configuration_is_complete(empty_configuration, authentication):
    config = configured_settings(llm_api_key="fake-user-llm-key", **authentication)
    assert config.missing(voice=False) == []
    assert config.missing() == []


@pytest.mark.parametrize("field,expected", [
    ("llm_base_url", "LLM_BASE_URL"),
    ("llm_model_id", "LLM_MODEL_ID"),
    ("llm_api_key", "LLM_API_KEY"),
    ("seeduplex_url", "SEEDUPLEX_URL"),
    ("seeduplex_model", "SEEDUPLEX_MODEL"),
    ("seeduplex_speaker", "SEEDUPLEX_SPEAKER"),
])
def test_each_required_model_field_is_reported_when_missing(empty_configuration, field, expected):
    options = {"llm_api_key": "fake-user-llm-key", "seeduplex_api_key": "fake-user-native-key"}
    options[field] = ""
    config = configured_settings(**options)
    assert expected in config.missing()


def test_legacy_native_authentication_requires_its_own_app_key(empty_configuration):
    config = configured_settings(
        llm_api_key="fake-user-llm-key", seeduplex_api_key="", seeduplex_app_id="fake-user-app",
        seeduplex_access_key="fake-user-access", seeduplex_app_key="",
    )
    assert config.missing() == ["SEEDUPLEX_APP_KEY"]


def test_unconfigured_text_request_fails_before_constructing_model(empty_configuration, monkeypatch):
    monkeypatch.setattr(main, "get_voice_settings", lambda: empty_configuration)

    def forbidden_executor(*args, **kwargs):
        raise AssertionError("Incomplete configuration must not construct an external model")

    from voice_agent import agent
    monkeypatch.setattr(agent, "AgentSession", forbidden_executor)
    with TestClient(main.app) as client:
        response = client.post("/api/text", json={"text": "测试配置"})
    assert response.status_code == 503
    assert all(name in response.json()["detail"] for name in (
        "LLM_BASE_URL", "LLM_MODEL_ID", "LLM_API_KEY",
    ))


def test_health_exposes_configuration_status_without_model_ids_or_keys(empty_configuration, monkeypatch):
    config = configured_settings(
        llm_api_key="fake-private-llm-key-sentinel", llm_model_id="fake-private-task-model-sentinel",
        seeduplex_api_key="fake-private-voice-key-sentinel",
        seeduplex_model="fake-private-voice-model-sentinel", insurance_tools_enabled=False,
    )
    monkeypatch.setattr(main, "get_voice_settings", lambda: config)
    with TestClient(main.app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["configured"] is True
    for secret in (config.llm_api_key, config.llm_model_id, config.seeduplex_api_key, config.seeduplex_model):
        assert secret not in response.text

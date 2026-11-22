"""Explicit fake configuration for isolated tests; never process-wide credentials."""

from voice_agent.config import Settings


def configured_settings(**overrides):
    return Settings(**{
        "_env_file": None,
        "llm_base_url": "https://llm.example.test/v1",
        "llm_model_id": "fake-task-model",
        "seeduplex_url": "wss://speech.example.test/duplex",
        "seeduplex_model": "fake-duplex-model",
        "seeduplex_speaker": "fake-speaker",
        "seeduplex_app_key": "fake-native-app-key",
        **overrides,
    })

"""Environment configuration. Secrets must never appear in logs or API responses."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = ""
    llm_model_id: str = Field(default="", repr=False)
    llm_api_key: str = Field(default="", repr=False)
    llm_disable_thinking: bool = True
    llm_timeout: float = Field(default=45, gt=0)
    llm_max_tokens: int = Field(default=1024, ge=1, le=8192)
    search_api_key: str = Field(default="", repr=False)
    search_url: str = "https://open.feedcoopapi.com/search_api/global_search"
    search_timeout: float = Field(default=8, gt=0, le=30)
    search_max_results: int = Field(default=10, ge=1, le=20)
    search_max_snippet_length: int = Field(default=1000, ge=100, le=2000)
    optional_skills_dir: str = str(Path.home() / ".local/share/ksa-deepagents/skills")
    skill_install_timeout: float = Field(default=300, gt=0, le=1800)
    skill_call_timeout: float = Field(default=60, gt=0, le=300)
    fast_llm_enabled: bool = False
    fast_llm_base_url: str = ""
    fast_llm_model_id: str = Field(default="", repr=False)
    fast_llm_api_key: str = Field(default="", repr=False)
    fast_llm_service_tier: Literal["auto", "default", "fast"] = "fast"
    fast_llm_timeout: float = Field(default=10, gt=0, le=45)
    fast_llm_max_tokens: int = Field(default=256, ge=32, le=1024)
    router_enabled: bool = True
    router_timeout: float = Field(default=2.0, gt=0, le=5)
    router_prefetch_ms: int = Field(default=200, ge=50, le=1000)
    slow_feedback_delay_ms: int = Field(default=1000, ge=0, le=10000)
    slow_feedback_cooldown_ms: int = Field(default=30000, ge=0, le=300000)
    voice_provider: Literal["cascade", "seeduplex"] = "seeduplex"
    insurance_opening_timeout: float = Field(default=8.0, gt=0, le=30)
    insurance_tools_enabled: bool = True
    insurance_writes_enabled: bool = False
    seeduplex_url: str = ""
    seeduplex_model: str = Field(default="", repr=False)
    seeduplex_api_key: str = Field(default="", repr=False)
    seeduplex_app_id: str = Field(default="", repr=False)
    seeduplex_access_key: str = Field(default="", repr=False)
    seeduplex_app_key: str = Field(default="", repr=False)
    seeduplex_resource_id: str = "volc.speech.dialog"
    seeduplex_speaker: str = ""
    seeduplex_timeout: float = Field(default=15, gt=0, le=60)
    native_result_timeout: float = Field(default=8.0, gt=0, le=30)
    background_tasks_enabled: bool = True
    background_max_tasks: int = Field(default=8, ge=1, le=20)
    background_max_concurrency: int = Field(default=3, ge=1, le=8)
    background_notify_delay: float = Field(default=0.6, ge=0.1, le=5)
    asr_url: str = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
    asr_api_key: str = Field(default="", repr=False)
    asr_app_key: str = Field(default="", repr=False)
    asr_access_key: str = Field(default="", repr=False)
    asr_resource_id: str = "volc.seedasr.sauc.duration"
    asr_sample_rate: int = Field(default=16000, ge=16000, le=16000)
    asr_timeout: float = Field(default=30, gt=0)
    tts_url: str = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
    tts_api_key: str = Field(default="", repr=False)
    tts_resource_id: str = "seed-tts-2.0"
    tts_speaker: str = ""
    tts_format: Literal["pcm"] = "pcm"
    tts_sample_rate: int = Field(default=24000, ge=24000, le=24000)
    tts_speech_rate: int = Field(default=0, ge=-50, le=100)
    tts_loudness_rate: int = Field(default=0, ge=-50, le=100)
    tts_timeout: float = Field(default=30, gt=0)
    max_recording_seconds: int = Field(default=120, ge=1, le=600)
    max_text_length: int = Field(default=4000, ge=1, le=16000)
    turn_timeout: float = Field(default=180, gt=0)
    voice_idle_timeout_seconds: float = Field(default=60, gt=0, le=60)
    vad_start_ms: int = Field(default=120, ge=60, le=500)
    vad_min_speech_ms: int = Field(default=180, ge=100, le=1000)
    vad_preroll_ms: int = Field(default=300, ge=120, le=1000)
    endpoint_complete_ms: int = Field(default=600, ge=300, le=3000)
    endpoint_definite_ms: int = Field(default=1000, ge=500, le=4000)
    endpoint_neutral_ms: int = Field(default=1400, ge=600, le=5000)
    endpoint_incomplete_ms: int = Field(default=2400, ge=1000, le=8000)
    endpoint_stability_ms: int = Field(default=200, ge=100, le=1500)
    endpoint_no_text_ms: int = Field(default=2000, ge=500, le=5000)
    semantic_enabled: bool = True
    semantic_timeout: float = Field(default=1.2, gt=0, le=5)
    semantic_model_id: str = Field(default="", repr=False)
    semantic_max_calls: int = Field(default=3, ge=0, le=10)
    voice_auth_token: str = Field(default="", repr=False)
    allowed_origins: list[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8766",
        "http://127.0.0.1:8766",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @model_validator(mode="after")
    def validate_urls(self) -> "Settings":
        if self.llm_base_url and not self.llm_base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("LLM_BASE_URL must use HTTPS (or local HTTP)")
        if self.fast_llm_base_url and not self.fast_llm_base_url.startswith(
            ("https://", "http://localhost", "http://127.0.0.1")
        ):
            raise ValueError("FAST_LLM_BASE_URL must use HTTPS (or local HTTP)")
        search_url = urlsplit(self.search_url)
        if (
            search_url.scheme != "https"
            or not search_url.hostname
            or search_url.username is not None
            or search_url.password is not None
        ):
            raise ValueError("SEARCH_URL must use HTTPS without embedded credentials")
        if not all(
            not url or url.startswith("wss://")
            for url in [self.asr_url, self.tts_url, self.seeduplex_url]
        ):
            raise ValueError("Speech service URLs must use WSS")
        return self

    def missing(self, *, voice: bool = True) -> list[str]:
        # Blank configuration must leave health/configuration feedback available,
        # while every business entry point fails closed before an upstream call.
        fields = []
        for name in ("llm_base_url", "llm_model_id", "llm_api_key"):
            if not getattr(self, name).strip():
                fields.append(name.upper())
        if voice and self.voice_provider == "seeduplex":
            for name in ("seeduplex_url", "seeduplex_model", "seeduplex_speaker"):
                if not getattr(self, name).strip():
                    fields.append(name.upper())
            if not self.seeduplex_api_key.strip():
                for name in ("seeduplex_app_id", "seeduplex_access_key", "seeduplex_app_key"):
                    if not getattr(self, name).strip():
                        fields.append(name.upper())
            return fields
        if voice:
            for name in ("asr_url", "tts_url"):
                if not getattr(self, name).strip():
                    fields.append(name.upper())
            if not self.asr_api_key.strip():
                if not self.asr_app_key.strip():
                    fields.append("ASR_APP_KEY")
                if not self.asr_access_key.strip():
                    fields.append("ASR_ACCESS_KEY")
            if not self.tts_api_key.strip():
                fields.append("TTS_API_KEY")
            if not self.tts_speaker.strip():
                fields.append("TTS_SPEAKER")
        return fields


@lru_cache
def get_settings() -> Settings:
    return Settings()

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 兼容组件；具体服务地址、模型与凭据由使用者配置。
    llm_base_url: str = ""
    llm_model_id: str = ""
    ark_api_key: str = Field(default="", repr=False)
    llm_api_key: str = Field(default="", repr=False)  # 兼容通用环境变量；ARK_API_KEY 优先
    asr_api_key: str = Field(default="", repr=False)  # 豆包 ASR 的 API Key（X-Api-Key 新鉴权）；留空则回退 App Key/Access Key
    asr_app_key: str = Field(default="", repr=False)
    asr_access_key: str = Field(default="", repr=False)
    asr_resource_id: str = "volc.seedasr.sauc.duration"
    asr_two_pass_enabled: bool = True  # 流式实时出字 + 停说后非流式模型复核最终结果
    asr_context_enabled: bool = True
    asr_context_turns: int = 4
    asr_context_max_chars: int = 600  # 保守控制在官方 800 token 限额内
    asr_context_domain_hint: str = (
        "车险续保电话，可能涉及客户姓名、车牌、车型、交强险、商业险、车损险、"
        "第三者责任险、保额、保费、出险次数、无赔款优待、报价、投保和退保。"
    )
    tts_api_key: str = Field(default="", repr=False)  # 豆包 TTS 2.0 的 API Key（X-Api-Key）
    tts_resource_id: str = "seed-tts-2.0"
    tts_speaker: str = ""
    tts_format: str = "pcm"  # 流式输出 pcm，前端按 16-bit int16 播放
    tts_sample_rate: int = 24000
    tts_speech_rate: int = 0  # 语速 [-50,100]，0 为正常
    tts_loudness_rate: int = 0  # 音量 [-50,100]，0 为正常
    frontend_origin: str = "http://localhost:5173"
    feishu_mcp_url: str = "https://project.feishu.cn/mcp_server/v1"
    feishu_mcp_token: str = Field(default="", repr=False)
    insurance_mcp_url: str = "http://127.0.0.1:8010/mcp"  # 保险业务 MCP（报价/退保/折扣/评分）
    proactive_opening: bool = True  # 外呼模式：通话接通后 agent 主动开场推销（否则等客户先开口）
    vad_min_energy: float = 0.008  # 提高绝对能量门槛，过滤较远的旁人/背景声
    vad_energy_margin_db: float = 14.0  # 相对噪声底提高能量门槛，减少旁人触发打断
    vad_positive: float = 0.70  # 提高人声确认阈值，减少误打断
    vad_negative: float = 0.25  # 人声结束阈值
    vad_min_speech_frames: int = 3  # 连续人声帧数，约 288ms 后触发开口/打断
    vad_redemption_frames: int = 1  # 后端 VAD 静音判停帧数（每帧 ~96ms），电话场景低延迟模式
    latency_filler_enabled: bool = True
    latency_filler_delay_ms: int = 1200  # 已知业务工具持续执行超过阈值才提示一次
    asr_final_timeout_s: float = 1.8  # 给二遍识别留余量；超时不使用未确认文本执行业务
    voice_response_timeout_s: float = 45
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")


settings = Settings()

"""后端 Silero VAD：用 ONNX 模型判断「是否人声」，驱动电话式对话的轮次切换。

模型：Silero VAD（legacy，LSTM 2×64），窗口 1536 样本 @16kHz（96ms/帧）。
逐帧概率经迟滞状态机后产出两个事件：

- ``speech_start``：连续 `min_speech_frames` 帧「概率 ≥ positive 且能量 ≥ 自适应门限」，
  判定「用户开口」（用于打断：播报途中检测到人声 → 打断智能体）。
  能量门限 = max(绝对下限, 噪声底 + margin dB)，用于拒绝远处/底噪——它们可能
  被 VAD 判成人声，但不够响。
- ``speech_end``：开口后静音达到 `redemption_frames` 帧，判定「说完了」
  （用于端点：聆听途中用户停止说话 → 交给 ASR 转写）。

输入是 16kHz / mono / int16 小端裸 PCM（与 ASR 喂入的数据同源），
内部先缓冲再切成 1536 样本的帧，按需把 int16 转成 [-1, 1] 的 float32。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1536  # legacy 模型窗口，96ms/帧
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"


class SileroVad:
    """Silero VAD ONNX 推理 + 迟滞状态机。"""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        positive: float = 0.7,  # 置信度阈值：提高以拒绝远处/低置信度人声，减少误打断
        negative: float = 0.35,
        min_speech_frames: int = 5,  # ~480ms 连续人声才算「开口」，过滤短促远处杂音
        redemption_frames: int = 7,  # ~672ms 静音判定「说完了」
        min_energy: float = 0.01,  # 绝对能量下限（线性 RMS）：噪声底极低时兜底，防自噪声触发
        energy_margin_db: float = 15.0,  # 相对噪声底高出多少 dB 才算「够响」，自适应门限的关键
    ):
        self._sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._sr = np.array([SAMPLE_RATE], dtype=np.int64)
        self.positive = positive
        self.negative = negative
        self.min_speech_frames = min_speech_frames
        self.redemption_frames = redemption_frames
        self.min_energy = min_energy
        self.energy_margin_db = energy_margin_db
        self._noise_floor: float | None = None  # 跟踪的噪声底（线性 RMS），跨轮持续不随 reset 清零
        self._noise_alpha = 0.05  # 噪声底 EMA 系数（约 2s 时间常数）
        self._buf = bytearray()
        self.reset()

    def reset(self) -> None:
        """复位 LSTM 状态与迟滞状态机（每轮结束 / 打断后调用）。"""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._speaking = False
        self._speech_frames = 0
        self._redemption = 0
        self._start_fired = False
        self._buf.clear()

    def process(self, pcm: bytes) -> list[str]:
        """喂入一段 int16 裸 PCM，返回期间发生的事件列表。

        返回值元素为 ``"speech_start"`` 或 ``"speech_end"``（可能同时多个，也可能为空）。
        """
        self._buf.extend(pcm)
        events: list[str] = []
        frame_len = FRAME_SAMPLES * 2
        while len(self._buf) >= frame_len:
            raw = bytes(self._buf[:frame_len])
            del self._buf[:frame_len]
            frame = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(frame * frame)))
            event = self._step(self._infer(frame), rms)
            if event:
                events.append(event)
        return events

    def _infer(self, frame: np.ndarray) -> float:
        """单帧推理，返回人声概率（0~1），并推进 LSTM 状态。"""
        output, hn, cn = self._sess.run(
            ["output", "hn", "cn"],
            {
                "input": frame.reshape(1, FRAME_SAMPLES).astype(np.float32),
                "sr": self._sr,
                "h": self._h,
                "c": self._c,
            },
        )
        self._h, self._c = hn, cn
        return float(output[0, 0])

    def _energy_threshold(self) -> float:
        """当前生效的能量门限 = max(绝对下限, 噪声底 + margin dB)。

        噪声底实时跟踪「不够响」的帧（静音 / 底噪 / 远处声），本人语音（响）不参与，
        因此门限自动贴合环境：安静房间门槛低、嘈杂房间门槛高，无需手调绝对音量。
        """
        if self._noise_floor is None:
            return self.min_energy
        margin = self._noise_floor * (10.0 ** (self.energy_margin_db / 20.0))
        return max(self.min_energy, margin)

    def _step(self, prob: float, rms: float) -> str | None:
        """迟滞状态机：把单帧（VAD 概率 + 能量）折算成事件。

        判定为「语音」需同时满足：VAD 概率 ≥ positive 且 能量 ≥ 自适应门限。
        能量门限相对噪声底计算，用于拒绝远处声——它们可能被 VAD 判成人声，但不够响。
        """
        loud = rms >= self._energy_threshold()

        # 只有「不够响」的帧才纳入噪声底统计；响的（本人语音）不参与，避免门限被自己抬高
        if not loud:
            if self._noise_floor is None:
                self._noise_floor = rms
            else:
                self._noise_floor += self._noise_alpha * (rms - self._noise_floor)

        if prob >= self.positive and loud:
            self._speech_frames += 1
            self._redemption = 0
            self._speaking = True
        elif self._speaking and (prob < self.negative or not loud):
            self._redemption += 1

        # 连续人声达到阈值 → 开口
        if not self._start_fired and self._speech_frames >= self.min_speech_frames:
            self._start_fired = True
            return "speech_start"

        # 开口后静音达到阈值 → 说完了
        if self._start_fired and self._redemption >= self.redemption_frames:
            self.reset()
            return "speech_end"

        # 短噪声（始终没到开口阈值）后静音 → 复位，避免状态残留污染下一轮
        if not self._start_fired and self._speaking and self._redemption >= self.min_speech_frames:
            self.reset()

        return None

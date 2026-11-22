"""Audio, ASR, and semantic evidence for automatic end-of-turn decisions.

The same 16 kHz PCM frames used by ASR are examined here.  Audio time, rather
than wall time, drives the detector so network scheduling cannot turn a pause
into an endpoint.  Semantic classification is supplied by the caller; this
module does no network I/O.
"""

from __future__ import annotations

import math
import sys
from array import array
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

SemanticOutcome = Literal["complete", "incomplete", "backchannel", "cancel", "unknown"]
SEMANTIC_OUTCOMES = {"complete", "incomplete", "backchannel", "cancel", "unknown"}


@dataclass(frozen=True, slots=True)
class AcousticFrame:
    """One analyzed 20 ms frame; onset pre-roll already includes ``pcm``."""

    pcm: bytes
    voiced: bool
    started: bool
    resumed: bool
    speech_ms: int
    silence_ms: int
    elapsed_ms: int
    pre_roll: bytes = b""


@dataclass(frozen=True, slots=True)
class EndpointDecision:
    end: bool
    reason: str
    discard: bool = False


class EndpointDetector:
    """WebRTC VAD + adaptive energy gate + ASR stability + semantic evidence.

    ``classifier`` can be injected for deterministic tests.  It receives a
    complete 640-byte PCM frame and the integer sample rate.  With no override,
    the production classifier is WebRTC VAD mode 2.
    """

    sample_rate = 16000
    frame_ms = 20
    frame_bytes = 640

    def __init__(
        self,
        settings: Any = None,
        classifier: Callable[[bytes, int], bool] | None = None,
    ):
        self.settings = settings
        if classifier is None:
            import webrtcvad

            self._vad = webrtcvad.Vad(self._setting("vad_mode", 2))
            classifier = self._vad.is_speech
        self._classifier = classifier
        self.start_ms = self._setting("vad_start_ms", 120)
        self.min_speech_ms = self._setting("vad_min_speech_ms", 180)
        self.preroll_ms = self._setting("vad_preroll_ms", 300)
        self.complete_ms = self._setting("endpoint_complete_ms", 600)
        self.definite_ms = self._setting("endpoint_definite_ms", 1000)
        self.neutral_ms = self._setting("endpoint_neutral_ms", 1400)
        self.incomplete_ms = self._setting("endpoint_incomplete_ms", 2400)
        self.stability_ms = self._setting("endpoint_stability_ms", 200)
        self.no_text_ms = self._setting("endpoint_no_text_ms", 2000)
        self.short_silence_ms = self._setting("endpoint_short_ms", 400)
        self.max_duration_ms = self._setting("max_recording_seconds", 120) * 1000
        self.energy_floor = float(self._setting("vad_energy_floor", 0.006))
        self.reset()

    def _setting(self, name: str, default: Any) -> Any:
        return getattr(self.settings, name, default)

    def reset(self) -> None:
        """Clear all turn evidence and retain no audio from the previous turn."""
        self._buffer = bytearray()
        self._preroll: deque[bytes] = deque(maxlen=max(1, math.ceil(self.preroll_ms / 20)))
        self.audio_ms = 0
        self.active = False
        self.speech_ms = 0
        self.silence_ms = 0
        self.elapsed_ms = 0
        self._onset_ms = 0
        self._turn_start_ms = 0
        self.noise_floor = 0.0015
        self.text = ""
        self.definite = False
        self.semantic: SemanticOutcome = "unknown"
        self._text_changed_ms = 0
        self._last_voice_ms = 0

    @staticmethod
    def _rms(pcm: bytes) -> float:
        samples = array("h")
        samples.frombytes(pcm)
        if sys.byteorder != "little":
            samples.byteswap()
        return math.sqrt(sum(sample * sample for sample in samples) / len(samples)) / 32768

    def feed(self, pcm: bytes) -> list[AcousticFrame]:
        """Accept arbitrary complete Int16 chunks and return complete 20 ms frames.

        A transport chunk may contain several frames or part of a frame.  Callers
        should fan out the returned frames, rather than sending the original
        chunk again, when attaching a new ASR stream at a speech onset.
        """
        if len(pcm) % 2:
            raise ValueError("PCM audio must contain complete Int16 samples")
        self._buffer.extend(pcm)
        frames = []
        while len(self._buffer) >= self.frame_bytes:
            frame = bytes(self._buffer[: self.frame_bytes])
            del self._buffer[: self.frame_bytes]
            frames.append(self._analyze(frame))
        return frames

    def _analyze(self, pcm: bytes) -> AcousticFrame:
        self.audio_ms += self.frame_ms
        self._preroll.append(pcm)
        rms = self._rms(pcm)
        vad_voice = bool(self._classifier(pcm, self.sample_rate))
        gate = max(self.energy_floor, self.noise_floor * 2.4)
        # Relax the amplitude gate after onset, preserving quieter word endings.
        voiced = vad_voice and rms >= gate * (0.65 if self.active else 1.0)
        if not voiced and not self.active:
            # Estimate background from non-speech only.  Bound its adaptation so
            # a transient click cannot raise the floor enough to hide speech.
            estimate = min(rms, 0.02)
            self.noise_floor = max(0.0003, self.noise_floor * 0.95 + estimate * 0.05)

        started = resumed = False
        if not self.active:
            self._onset_ms = self._onset_ms + self.frame_ms if voiced else 0
            if self._onset_ms >= self.start_ms:
                self.active = started = True
                self.speech_ms = self._onset_ms
                self._turn_start_ms = self.audio_ms - self._onset_ms
                self._last_voice_ms = self.audio_ms
                self.silence_ms = 0
        elif voiced:
            resumed = self.silence_ms > 0
            self.speech_ms += self.frame_ms
            self.silence_ms = 0
            self._last_voice_ms = self.audio_ms
            if resumed:
                self.definite = False
                self.semantic = "unknown"
        else:
            self.silence_ms += self.frame_ms
        if self.active:
            self.elapsed_ms = self.audio_ms - self._turn_start_ms
        return AcousticFrame(
            pcm=pcm,
            voiced=voiced,
            started=started,
            resumed=resumed,
            speech_ms=self.speech_ms,
            silence_ms=self.silence_ms,
            elapsed_ms=self.elapsed_ms,
            pre_roll=b"".join(self._preroll) if started else b"",
        )

    def update_transcript(self, text: str, *, definite: bool = False) -> bool:
        """Update cumulative ASR text; return whether its wording changed."""
        text = text.strip()
        changed = text != self.text
        if changed:
            self.text = text
            self._text_changed_ms = self.audio_ms
            self.semantic = "unknown"
            self.definite = definite
        else:
            self.definite = self.definite or definite
        return changed

    def set_semantic(self, outcome: SemanticOutcome, *, text: str | None = None) -> bool:
        """Accept a result only for the latest transcript revision."""
        if outcome not in SEMANTIC_OUTCOMES:
            raise ValueError("Unknown semantic endpoint outcome")
        if text is not None and text.strip() != self.text:
            return False
        self.semantic = outcome
        return True

    @property
    def transcript_stable(self) -> bool:
        # Also wait after resumed speech: old ASR text must never be used as an
        # immediate endpoint for words that have not arrived from ASR yet.
        last_change = max(self._text_changed_ms, self._last_voice_ms)
        return bool(self.text) and self.audio_ms - last_change >= self.stability_ms

    def decision(self) -> EndpointDecision:
        """Fuse all paths without letting one semantic guess override live voice."""
        if not self.active:
            return EndpointDecision(False, "waiting_for_speech")
        if self.elapsed_ms >= self.max_duration_ms:
            return EndpointDecision(True, "max_duration", discard=not bool(self.text))
        if self.speech_ms < self.min_speech_ms:
            if self.silence_ms >= self.short_silence_ms:
                return EndpointDecision(True, "too_short", discard=True)
            return EndpointDecision(False, "minimum_speech")
        if not self.text:
            if self.silence_ms >= self.no_text_ms:
                return EndpointDecision(True, "no_transcript", discard=True)
            return EndpointDecision(False, "waiting_for_transcript")
        if self.silence_ms < self.complete_ms:
            return EndpointDecision(False, "speech_or_brief_pause")
        if not self.transcript_stable:
            return EndpointDecision(False, "transcript_changing")
        if self.semantic in {"complete", "cancel", "backchannel"}:
            return EndpointDecision(True, f"semantic_{self.semantic}")
        if self.semantic == "incomplete":
            if self.silence_ms >= self.incomplete_ms:
                return EndpointDecision(True, "incomplete_hard_wait")
            return EndpointDecision(False, "semantic_incomplete")
        if self.definite and self.silence_ms >= self.definite_ms:
            return EndpointDecision(True, "asr_definite")
        if self.silence_ms >= self.neutral_ms:
            return EndpointDecision(True, "silence_fallback")
        return EndpointDecision(False, "waiting_for_evidence")

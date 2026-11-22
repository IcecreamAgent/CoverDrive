// 低延迟 PCM 播放器：主线程只投递 Int16 PCM，音频线程负责连续消费。
class PcmPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inputRate = 24000;
    this.samples = new Float32Array(0);
    this.position = 0;
    this.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === 'clear') {
        this.samples = new Float32Array(0);
        this.position = 0;
        return;
      }
      if (message.type === 'pcm' && message.buffer) {
        const input = new Int16Array(message.buffer);
        const next = new Float32Array(this.samples.length + input.length);
        next.set(this.samples);
        for (let i = 0; i < input.length; i++) next[this.samples.length + i] = input[i] / 32768;
        this.samples = next;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const channel = output[0];
    // PCM 来自 TTS 的 24kHz，AudioWorklet 输出是 AudioContext 采样率（常见为 48kHz）。
    // 线性插值保持原始时长和音调，避免把 24kHz 样本错误地当成 48kHz 播放。
    const step = this.inputRate / sampleRate;
    for (let i = 0; i < channel.length; i++) {
      const index = Math.floor(this.position);
      if (index + 1 >= this.samples.length) {
        channel[i] = 0;
      } else {
        const frac = this.position - index;
        channel[i] = this.samples[index] + (this.samples[index + 1] - this.samples[index]) * frac;
        this.position += step;
      }
    }
    // 保留插值所需的前一个采样，避免每个网络包之间出现断点。
    const consumed = Math.max(0, Math.floor(this.position) - 1);
    if (consumed > 0) {
      this.samples = this.samples.slice(consumed);
      this.position -= consumed;
    }
    return true;
  }
}

registerProcessor('pcm-playback', PcmPlaybackProcessor);

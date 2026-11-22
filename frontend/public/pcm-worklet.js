// 录音 AudioWorklet：把 AudioContext 原生采样率的 Float32 重采样为 16kHz int16 PCM，
// 按 ~20ms 分块回传，减少句尾等待凑包（支持 44.1k/48k 等输入率）。
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sourceRate = sampleRate;
    this.targetRate = 16000;
    this.step = this.sourceRate / this.targetRate; // 每产出 1 个输出样本需消耗的输入样本数
    this.phase = 0;                                  // 当前在输入流中的分数位置
    this.leftover = new Float32Array(0);             // 尚未消费的输入样本
    this.outBuf = new Int16Array(320);               // 20ms @ 16k
    this.outLen = 0;
  }

  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (channel && channel.length) {
      const buf = new Float32Array(this.leftover.length + channel.length);
      buf.set(this.leftover, 0);
      buf.set(channel, this.leftover.length);
      this.leftover = buf;
    }

    const n = this.leftover.length;
    while (this.phase < n - 1) {
      const idx = Math.floor(this.phase);
      const frac = this.phase - idx;
      const sample = this.leftover[idx] + (this.leftover[idx + 1] - this.leftover[idx]) * frac;
      this.outBuf[this.outLen++] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
      this.phase += this.step;
      if (this.outLen === this.outBuf.length) {
        const chunk = this.outBuf.slice(0); // 复制出独立 ArrayBuffer
        this.port.postMessage(chunk.buffer, [chunk.buffer]);
        this.outLen = 0;
      }
    }

    const consumed = Math.floor(this.phase);
    if (consumed > 0) {
      this.leftover = this.leftover.slice(consumed);
      this.phase -= consumed;
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);

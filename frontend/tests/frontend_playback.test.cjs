const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Run the actual browser handlers with controllable Web Audio source endings.
function browser() {
  const sent = [];
  const sources = [];
  const element = () => ({
    value: "", dataset: {}, style: {},
    classList: {toggle() {}, add() {}, remove() {}},
    addEventListener() {}, setAttribute() {}, querySelectorAll: () => [],
  });
  const audio = {
    state: "running", currentTime: 0, destination: {},
    createBuffer: (_, samples, rate) => ({
      duration: samples / rate, getChannelData: () => new Float32Array(samples),
    }),
    createBufferSource() {
      const source = {connect() {}, disconnect() {}, start() {}, stop() {}};
      sources.push(source);
      return source;
    },
    suspend() { this.state = "suspended"; return Promise.resolve(); },
    resume() { this.state = "running"; return Promise.resolve(); },
  };
  const context = vm.createContext({
    document: {getElementById: element, querySelector: element, querySelectorAll: () => []},
    window: {addEventListener() {}}, WebSocket: {OPEN: 1},
    ArrayBuffer, Uint8Array, Float32Array, DataView,
    setTimeout, clearTimeout, setInterval, clearInterval,
    testAudio: audio, testSocket: {readyState: 1, send: raw => sent.push(JSON.parse(raw))},
  });
  const script = fs.readFileSync(path.join(__dirname, "../src/ksa.js"), "utf8");
  vm.runInContext(script.replace("void initialize();", `
    connected = true; socket = testSocket; outputContext = testAudio;
    activeTurnId = "turn-1"; audioTurnId = "turn-1"; busy = true;
    globalThis.api = {playAudio, handleMessage, stopPlayback, pausePlayback, resumePlayback};
  `), context);
  return {
    ...context.api, sent, sources, audio,
    barrier(id, turn = "turn-1") {
      context.api.handleMessage({data: JSON.stringify({
        type: "playback_barrier", turn_id: turn, barrier_id: id,
      })});
    },
    drained: () => sent.filter(event => event.type === "playback_drained"),
    pcm: () => context.api.playAudio(new ArrayBuffer(4800)),
  };
}

test("acknowledge only after every preceding audio source ends", () => {
  const page = browser();
  page.pcm(); page.pcm();
  page.barrier("marker-1");
  assert.deepEqual(page.drained(), []);
  page.sources[1].onended();
  assert.deepEqual(page.drained(), []);
  page.sources[0].onended();
  assert.deepEqual(page.drained(), [{
    type: "playback_drained", turn_id: "turn-1", barrier_id: "marker-1",
  }]);
});

test("an already drained queue acknowledges immediately and ignores old turns", () => {
  const page = browser();
  page.barrier("old", "old-turn");
  assert.deepEqual(page.drained(), []);
  page.barrier("now");
  assert.equal(page.drained()[0].barrier_id, "now");
});

test("barge-in discards a pending fence without claiming the audio was heard", () => {
  const page = browser();
  page.pcm(); page.barrier("pending");
  page.stopPlayback();
  assert.equal(page.sources[0].onended, null);
  assert.deepEqual(page.drained(), []);
});

test("new PCM is covered by a subsequent fence", () => {
  const page = browser();
  page.pcm(); page.barrier("first");
  page.pcm(); page.barrier("second");
  page.sources[0].onended();
  assert.deepEqual(page.drained().map(event => event.barrier_id), ["first"]);
  page.sources[1].onended();
  assert.deepEqual(page.drained().map(event => event.barrier_id), ["first", "second"]);
});

test("a temporary playback pause does not release the waiting result", async () => {
  const page = browser();
  page.pcm(); page.barrier("paused");
  page.pausePlayback();
  await Promise.resolve();
  assert.equal(page.audio.state, "suspended");
  assert.deepEqual(page.drained(), []);
  page.resumePlayback();
  assert.deepEqual(page.drained(), []);
  page.sources[0].onended();
  assert.equal(page.drained()[0].barrier_id, "paused");
});

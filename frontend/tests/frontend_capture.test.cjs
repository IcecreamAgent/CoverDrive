const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

// Exercise the migrated production handlers without requesting a real microphone.
function browser({withTask = true} = {}) {
  const sent = [], contexts = [], nodes = [], revoked = [], elements = new Map(), timers = new Map();
  let timerId = 0;
  const element = () => ({
    value: '', dataset: {}, style: {}, hidden: false, disabled: false, attributes: {}, listeners: {},
    classList: {toggle() {}, add() {}, remove() {}},
    addEventListener(name, callback) { this.listeners[name] = callback; },
    setAttribute(name, value) { this.attributes[name] = value; }, querySelector: () => element(),
    querySelectorAll: () => [], focus() { this.focused = true; },
  });
  const get = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  let microphone = async () => stream;
  let loadModule = async () => {};
  const track = {enabled: true, stopped: false, stop() { this.stopped = true; }};
  const stream = {getTracks: () => [track]};
  const node = () => ({connect() {}, disconnect() { this.disconnected = true; }});
  class Context {
    constructor() {
      this.state = 'running';
      this.destination = {};
      this.audioWorklet = {addModule: (...args) => loadModule(...args)};
      contexts.push(this);
    }
    createMediaStreamSource() { return node(); }
    createGain() { return {...node(), gain: {value: 1}}; }
    async resume() { this.state = 'running'; }
    async close() { this.state = 'closed'; }
  }
  class Worklet {
    constructor() {
      Object.assign(this, node());
      this.port = {onmessage: null, postMessage() {}, close() { this.closed = true; }};
      nodes.push(this);
    }
  }
  const makeSocket = () => ({
    readyState: 1, bufferedAmount: 0,
    send: raw => sent.push(typeof raw === 'string' ? JSON.parse(raw) : raw),
    close() { this.readyState = 3; },
  });
  const socket = makeSocket();
  class Socket {
    static OPEN = 1;
    constructor() { return makeSocket(); }
  }
  const output = {state: 'running', resume: async () => {}};
  const context = vm.createContext({
    document: {getElementById: get, querySelector: element, querySelectorAll: () => []},
    window: {addEventListener() {}, isSecureContext: true, AudioContext: Context,
      location: {protocol: 'http:', host: '127.0.0.1:5173'}},
    navigator: {mediaDevices: {getUserMedia: (...args) => microphone(...args)}},
    WebSocket: Socket, AudioWorkletNode: Worklet,
    URL: {createObjectURL: () => 'blob:capture', revokeObjectURL: url => revoked.push(url)},
    Blob, ArrayBuffer, Uint8Array, Float32Array, DataView,
    setTimeout: (callback, delay) => { const id = ++timerId; timers.set(id, {callback, delay}); return id; },
    clearTimeout: id => timers.delete(id), setInterval: () => 1, clearInterval() {},
    testSocket: socket, testAudio: output, withTask,
  });
  const script = fs.readFileSync(path.join(__dirname, '../src/ksa.js'), 'utf8');
  vm.runInContext(script.replace('void initialize();', `
    connected = true; socket = testSocket; outputContext = testAudio; busy = true; playbackActive = true;
    if (withTask) taskRecords.set('background-1', {data: {status: 'running'}});
    globalThis.api = {startListening, stopListening, toggleMicrophone, handleMessage,
      connectForTest: () => { connected = false; socket = null; connect(); },
      currentSocket: () => socket,
      addPlaybackForTest: source => { audioSources.add(source); playbackActive = true; },
      state: () => ({listening, busy, playbackActive, microphoneMuted, captureStarting, captureStopping, capture,
        connected, connecting, idleTimedOut, taskStatus: taskRecords.get('background-1')?.data.status})};
  `), context);
  return {
    ...context.api, sent, track, contexts, nodes, revoked, get, socket, timers,
    event: message => context.api.handleMessage({data: JSON.stringify(message)}),
    useMicrophone: fn => { microphone = fn; },
    useModuleLoader: fn => { loadModule = fn; },
  };
}

test('closing microphone releases capture and preserves current reply and background work', async () => {
  const page = browser();
  await page.startListening();
  assert.equal(page.state().listening, true);
  assert.equal(page.sent[0].type, 'listen');
  assert.equal(page.sent[0].playback_barriers, true);
  assert.equal(page.sent[0].outbound, true);
  assert.equal(page.get('microphone-label').textContent, '挂断');
  const onAudio = page.nodes[0].port.onmessage;
  const pcm = new ArrayBuffer(640);
  onAudio({data: {type: 'audio', buffer: pcm}});
  assert.equal(page.sent[1], pcm);
  await page.stopListening();
  assert.equal(page.state().listening, false);
  assert.equal(page.state().busy, true);
  assert.equal(page.track.stopped, true);
  assert.equal(page.contexts[0].state, 'closed');
  assert.equal(page.nodes[0].port.closed, true);
  assert.deepEqual(page.revoked, ['blob:capture']);
  assert.equal(page.sent.at(-1).type, 'unlisten');
  assert.ok(!page.sent.some(event => ['interrupt', 'cancel_task', 'cancel_all_tasks'].includes(event.type)));
  const count = page.sent.length;
  onAudio({data: {type: 'audio', buffer: pcm}});
  assert.equal(page.sent.length, count, 'late capture callbacks must not send audio');
});

test('microphone permission refusal exposes retry and leaves text controls available', async () => {
  const page = browser();
  page.get('message-input').value = '权限拒绝后仍可发送文字';
  page.useMicrophone(async () => { throw Object.assign(new Error('denied'), {name: 'NotAllowedError'}); });
  await page.startListening();
  assert.equal(page.state().listening, false);
  assert.equal(page.state().captureStarting, false);
  assert.equal(page.get('microphone-button').disabled, false);
  assert.equal(page.get('send-button').disabled, false);
  assert.equal(page.get('mute-button').hidden, true);
  assert.match(page.get('error-text').textContent, /允许麦克风权限/);
  assert.equal(page.sent.length, 0);
  page.useMicrophone(async () => ({getTracks: () => [page.track]}));
  await page.startListening();
  assert.equal(page.state().listening, true);
  await page.stopListening();
});

test('closing input while permission is pending stops the late stream without listening again', async () => {
  const page = browser();
  let grant;
  page.useMicrophone(() => new Promise(resolve => { grant = resolve; }));
  const starting = page.startListening();
  await Promise.resolve();
  await page.stopListening();
  grant({getTracks: () => [page.track]});
  await starting;
  assert.equal(page.track.stopped, true);
  assert.equal(page.state().listening, false);
  assert.equal(page.state().capture, null);
  assert.ok(!page.sent.some(event => event.type === 'listen'));
});

test('worklet initialization failure releases microphone, context, and blob URL', async () => {
  const page = browser();
  page.useModuleLoader(async () => { throw new Error('worklet unavailable'); });
  await page.startListening();
  assert.equal(page.track.stopped, true);
  assert.equal(page.contexts[0].state, 'closed');
  assert.deepEqual(page.revoked, ['blob:capture']);
  assert.equal(page.state().capture, null);
  assert.equal(page.state().captureStarting, false);
  assert.match(page.get('error-text').textContent, /worklet unavailable/);
});

test('upload backpressure closes input and does not cancel independent tasks', async () => {
  const page = browser();
  await page.startListening();
  page.socket.bufferedAmount = 1024 * 1024 + 1;
  page.nodes[0].port.onmessage({data: {type: 'audio', buffer: new ArrayBuffer(640)}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(page.state().listening, false);
  assert.equal(page.track.stopped, true);
  assert.equal(page.state().busy, true);
  assert.match(page.get('error-text').textContent, /网络发送较慢/);
  assert.ok(!page.sent.some(event => event.type === 'cancel_task'));
});


test('muting blocks queued microphone PCM while preserving cadence, reply playback, and independent work', async () => {
  const page = browser();
  await page.startListening();
  const onAudio = page.nodes[0].port.onmessage;
  const pcm = new Uint8Array(640).fill(127).buffer;
  onAudio({data: {type: 'audio', buffer: pcm}});
  assert.equal(page.sent.at(-1), pcm);
  const sentBeforeMute = page.sent.length;

  page.toggleMicrophone();
  assert.equal(page.state().microphoneMuted, true);
  assert.equal(page.track.enabled, false);
  assert.equal(page.get('mute-button').attributes['aria-pressed'], 'true');
  assert.equal(page.get('mute-label').textContent, '取消静音');
  assert.equal(page.get('mic-status').hidden, false);
  assert.equal(page.get('mute-button').hidden, false);
  assert.equal(page.sent.length, sentBeforeMute, 'mute must not change the server listener or task lifecycle');

  onAudio({data: {type: 'audio', buffer: pcm}});
  onAudio({data: {type: 'audio', buffer: pcm}});
  assert.equal(page.sent.length, sentBeforeMute + 2, 'keep one silent frame for each capture frame');
  for (const frame of page.sent.slice(sentBeforeMute)) {
    assert.equal(frame.byteLength, pcm.byteLength);
    assert.notEqual(frame, pcm);
    assert.ok(new Uint8Array(frame).every(value => value === 0), 'even queued pre-mute sound must be removed');
  }
  assert.ok(new Uint8Array(pcm).every(value => value === 127), 'do not mutate the worklet-owned input');
  assert.equal(page.state().listening, true);
  assert.equal(page.state().busy, true);
  assert.equal(page.state().playbackActive, true);
  assert.equal(page.state().taskStatus, 'running');
  assert.equal(page.contexts[0].state, 'running');
  assert.equal(page.track.stopped, false);
  assert.ok(!page.sent.some(event => ['unlisten', 'interrupt', 'cancel_task', 'cancel_all_tasks'].includes(event.type)));

  page.toggleMicrophone();
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(page.track.enabled, true);
  assert.equal(page.get('mute-button').attributes['aria-pressed'], 'false');
  assert.equal(page.get('mute-label').textContent, '静音');
  assert.equal(page.get('mic-status').hidden, true);
  onAudio({data: {type: 'audio', buffer: pcm}});
  assert.equal(page.sent.at(-1), pcm, 'unmute resumes the original microphone frame');
  await page.stopListening();
});

test('hanging up resets mute and a new call starts with an enabled microphone', async () => {
  const page = browser();
  await page.startListening();
  page.toggleMicrophone();
  const oldAudio = page.nodes[0].port.onmessage;
  await page.stopListening();
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(page.get('microphone-label').textContent, '拨号');
  assert.equal(page.get('mute-button').hidden, true);
  assert.equal(page.get('mic-status').hidden, true);
  const sentAfterHangup = page.sent.length;
  oldAudio({data: {type: 'audio', buffer: new Uint8Array(640).fill(127).buffer}});
  assert.equal(page.sent.length, sentAfterHangup);

  const nextTrack = {enabled: true, stopped: false, stop() { this.stopped = true; }};
  page.useMicrophone(async () => ({getTracks: () => [nextTrack]}));
  await page.startListening();
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(nextTrack.enabled, true);
  assert.equal(page.get('mute-label').textContent, '静音');
  assert.equal(page.get('mute-button').hidden, false);
  const pcm = new Uint8Array(640).fill(91).buffer;
  page.nodes[1].port.onmessage({data: {type: 'audio', buffer: pcm}});
  assert.equal(page.sent.at(-1), pcm);
  await page.stopListening();
});

test('mute is inert without an active call and cannot affect a pending microphone grant', async () => {
  const page = browser();
  page.toggleMicrophone();
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(page.sent.length, 0);
  let grant;
  page.useMicrophone(() => new Promise(resolve => { grant = resolve; }));
  const starting = page.startListening();
  await Promise.resolve();
  page.toggleMicrophone();
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(page.track.enabled, true);
  await page.stopListening();
  grant({getTracks: () => [page.track]});
  await starting;
  assert.equal(page.state().listening, false);
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(page.track.stopped, true);
  assert.ok(!page.sent.some(event => event.type === 'listen'));
});


test('the actual dial and mute button handlers start, mute, unmute, and hang up independently', async () => {
  const page = browser();
  assert.equal(page.sent.length, 0, 'loading the page must not start an outbound conversation');
  page.get('microphone-button').listeners.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(page.state().listening, true);
  assert.equal(page.sent.filter(event => event.type === 'listen').length, 1);
  assert.equal(page.sent.find(event => event.type === 'listen').outbound, true);
  page.get('mute-button').listeners.click();
  assert.equal(page.state().microphoneMuted, true);
  assert.equal(page.track.enabled, false);
  assert.equal(page.state().playbackActive, true);
  assert.equal(page.state().taskStatus, 'running');
  page.get('mute-button').listeners.click();
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(page.track.enabled, true);
  page.get('microphone-button').listeners.click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(page.state().listening, false);
  assert.equal(page.track.stopped, true);
  assert.equal(page.sent.at(-1).type, 'unlisten');
});

test('a real socket close handler clears mute and rejects late capture data', async () => {
  const page = browser({withTask: false});
  page.connectForTest();
  const connection = page.currentSocket();
  connection.onmessage({data: JSON.stringify({type: 'ready'})});
  await page.startListening();
  page.get('mute-button').listeners.click();
  const oldAudio = page.nodes[0].port.onmessage;
  connection.onclose({code: 1008});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(page.state().listening, false);
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(page.state().capture, null);
  assert.equal(page.track.stopped, true);
  assert.equal(page.get('mute-button').hidden, true);
  assert.equal(page.get('mic-status').hidden, true);
  assert.equal(page.get('microphone-button').disabled, true);
  const count = page.sent.length;
  oldAudio({data: {type: 'audio', buffer: new Uint8Array(640).fill(127).buffer}});
  assert.equal(page.sent.length, count);
});

test('the clear conversation button clears mute and releases capture before the new session is ready', async () => {
  const page = browser({withTask: false});
  await page.startListening();
  page.get('mute-button').listeners.click();
  const oldAudio = page.nodes[0].port.onmessage;
  page.get('clear-button').listeners.click();
  page.currentSocket().onmessage({data: JSON.stringify({type: 'ready'})});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(page.state().listening, false);
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(page.state().capture, null);
  assert.equal(page.track.stopped, true);
  assert.equal(page.get('mute-button').hidden, true);
  assert.equal(page.get('mic-status').hidden, true);
  assert.equal(page.get('microphone-label').textContent, '拨号');
  assert.equal(page.get('message-input').focused, true);
  const count = page.sent.length;
  oldAudio({data: {type: 'audio', buffer: new Uint8Array(640).fill(127).buffer}});
  assert.equal(page.sent.length, count);
});

test('idle warning stays visible during output and clears only when the server confirms user activity', async () => {
  const page = browser({withTask: false});
  await page.startListening();
  const before = page.sent.length;
  page.event({type: 'idle_warning', turn_id: 'another-turn', message: '10秒后将自动挂断；请说话或发送文字继续。'});
  assert.equal(page.get('idle-notice').hidden, false);
  assert.match(page.get('idle-notice').textContent, /10秒后/);
  assert.equal(page.state().listening, true);
  assert.equal(page.state().playbackActive, true);
  assert.equal(page.track.stopped, false);
  assert.equal(page.sent.length, before, 'a reminder must not generate speech, model work, or control messages');
  page.event({type: 'idle_resumed'});
  assert.equal(page.get('idle-notice').hidden, true);
  await page.stopListening();
});

test('idle timeout closes capture, queued output and socket without reconnecting or losing its reason', async () => {
  const page = browser({withTask: false});
  page.connectForTest();
  const connection = page.currentSocket();
  connection.onmessage({data: JSON.stringify({type: 'ready'})});
  const lateClose = connection.onclose;
  await page.startListening();
  page.toggleMicrophone();
  const lateAudio = page.nodes[0].port.onmessage;
  const source = {onended() {}, stop() { this.stopped = true; }, disconnect() { this.disconnected = true; }};
  page.addPlaybackForTest(source);
  connection.onmessage({data: JSON.stringify({type: 'idle_timeout', turn_id: 'another-turn', message: '已连续1分钟未检测到你说话，通话已自动挂断。'})});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(page.currentSocket(), null);
  assert.equal(connection.readyState, 3);
  assert.equal(page.state().connected, false);
  assert.equal(page.state().listening, false);
  assert.equal(page.state().busy, false);
  assert.equal(page.state().playbackActive, false);
  assert.equal(page.state().capture, null);
  assert.equal(page.state().captureStopping, false);
  assert.equal(page.track.stopped, true);
  assert.equal(page.contexts[0].state, 'closed');
  assert.equal(page.nodes[0].port.closed, true);
  assert.equal(source.stopped, true);
  assert.equal(source.disconnected, true);
  assert.equal(source.onended, null);
  assert.equal(page.get('idle-notice').hidden, false);
  assert.match(page.get('idle-notice').textContent, /自动挂断.*拨号/);
  assert.equal(page.get('microphone-label').textContent, '拨号');
  assert.equal(page.get('microphone-button').disabled, false);
  assert.equal(page.get('mute-button').hidden, true);
  assert.equal(page.get('call-status').hidden, true);
  lateClose({code: 1000});
  page.event({type: 'idle_resumed'});
  assert.equal(page.timers.size, 0, 'the retired socket must not schedule reconnects');
  assert.equal(page.get('idle-notice').hidden, false, 'late lifecycle events cannot erase the end reason');
  const sent = page.sent.length;
  lateAudio({data: {type: 'audio', buffer: new ArrayBuffer(640)}});
  assert.equal(page.sent.length, sent);
});

test('dial after idle timeout starts a fresh connection and clears the reason only once capture restarts', async () => {
  const page = browser({withTask: false});
  await page.startListening();
  page.event({type: 'idle_timeout'});
  await new Promise(resolve => setImmediate(resolve));
  const sent = page.sent.length;
  page.get('microphone-button').listeners.click();
  assert.equal(page.state().connecting, true);
  assert.equal(page.state().listening, false);
  assert.equal(page.get('microphone-button').disabled, true);
  assert.equal(page.get('idle-notice').hidden, false);
  assert.equal(page.sent.length, sent);
  const connection = page.currentSocket();
  page.useMicrophone(async () => ({getTracks: () => [{enabled: true, stop() {}}]}));
  connection.onmessage({data: JSON.stringify({type: 'ready'})});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(page.state().connected, true);
  assert.equal(page.state().listening, true);
  assert.equal(page.state().microphoneMuted, false);
  assert.equal(page.state().idleTimedOut, false);
  assert.equal(page.get('idle-notice').hidden, true);
  assert.equal(page.sent.filter(event => event.type === 'listen').length, 2);
  await page.stopListening();
});

test('an idle close reason ends the session without reconnecting when the timeout message is lost', async () => {
  const page = browser({withTask: false});
  page.connectForTest();
  const connection = page.currentSocket();
  connection.onmessage({data: JSON.stringify({type: 'ready'})});
  await page.startListening();
  connection.onclose({code: 1000, reason: 'user_inactive'});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(page.currentSocket(), null);
  assert.equal(page.state().idleTimedOut, true);
  assert.equal(page.state().listening, false);
  assert.equal(page.state().capture, null);
  assert.equal(page.state().playbackActive, false);
  assert.equal(page.track.stopped, true);
  assert.equal(page.timers.size, 0);
  assert.equal(page.get('idle-notice').hidden, false);
  assert.match(page.get('idle-notice').textContent, /1分钟.*自动挂断.*拨号/);
  assert.equal(page.get('connection-label').textContent, '通话已自动结束');
  assert.equal(page.get('microphone-button').disabled, false);
});

test('idle timeout retires a pending permission grant without starting late audio', async () => {
  const page = browser({withTask: false});
  let grant;
  page.useMicrophone(() => new Promise(resolve => { grant = resolve; }));
  const starting = page.startListening();
  await Promise.resolve();
  page.event({type: 'idle_timeout'});
  grant({getTracks: () => [page.track]});
  await starting;
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(page.track.stopped, true);
  assert.equal(page.state().capture, null);
  assert.equal(page.state().listening, false);
  assert.equal(page.state().captureStarting, false);
  assert.ok(!page.sent.some(event => event.type === 'listen'));
  assert.match(page.get('idle-notice').textContent, /自动挂断/);
  assert.equal(page.get('microphone-button').disabled, false);
});

test('manually ending input clears a warning and retains the connected session', async () => {
  const page = browser({withTask: false});
  await page.startListening();
  page.event({type: 'idle_warning'});
  await page.stopListening();
  assert.equal(page.get('idle-notice').hidden, true);
  assert.equal(page.state().connected, true);
  assert.equal(page.state().idleTimedOut, false);
  assert.equal(page.sent.at(-1).type, 'unlisten');
});

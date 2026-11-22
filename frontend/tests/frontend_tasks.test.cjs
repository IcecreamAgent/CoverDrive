const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Exercise background-result and audio events with a minimal persistent DOM.
class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.listeners = new Map();
    this.className = "";
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.parentElement = null;
    this._text = "";
    this.open = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
    this.tabIndex = ["BUTTON", "SELECT", "INPUT", "TEXTAREA", "SUMMARY"].includes(this.tagName) ? 0 : -1;
    this.classList = {
      contains: name => this.className.split(/\s+/).includes(name),
      add: name => {
        if (!this.classList.contains(name)) this.className = `${this.className} ${name}`.trim();
      },
      remove: name => { this.className = this.className.split(/\s+/).filter(x => x !== name).join(" "); },
      toggle: (name, enabled) => {
        if (enabled ?? !this.classList.contains(name)) this.classList.add(name);
        else this.classList.remove(name);
      },
    };
  }
  set textContent(value) {
    this._text = String(value);
    for (const child of this.children) child.parentElement = null;
    this.children = [];
  }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  append(...children) {
    for (const child of children) {
      child.remove();
      this.children.push(child);
      child.parentElement = this;
    }
  }
  prepend(...children) {
    for (const child of children.reverse()) {
      child.remove();
      this.children.unshift(child);
      child.parentElement = this;
    }
  }
  insertBefore(child, before) {
    if (!before) { this.append(child); return child; }
    child.remove();
    const index = this.children.indexOf(before);
    if (index < 0) throw new Error("insertBefore reference is not a child");
    this.children.splice(index, 0, child);
    child.parentElement = this;
    return child;
  }
  replaceChildren(...children) { this.textContent = ""; this.append(...children); }
  focus() {
    this.focused = true;
    if (this.ownerDocument) this.ownerDocument.activeElement = this;
  }
  contains(element) { return this === element || this.children.some(child => child.contains(element)); }
  get isConnected() { return this === this.ownerDocument?.body || Boolean(this.parentElement?.isConnected); }
  scrollIntoView() { this.scrolledIntoView = true; }
  getAttribute(name) { return this.attributes[name] ?? null; }
  getClientRects() {
    for (let current = this; current; current = current.parentElement) {
      if (current.hidden) return [];
    }
    return [{}];
  }
  removeAttribute(name) { delete this.attributes[name]; }
  remove() {
    if (this.parentElement) {
      this.parentElement.children = this.parentElement.children.filter(child => child !== this);
      this.parentElement = null;
    }
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "tabindex") this.tabIndex = Number(value);
  }
  addEventListener(name, callback) { this.listeners.set(name, callback); }
  trigger(name, options = {}) {
    if (!this.disabled) return this.listeners.get(name)?.({
      target: this, currentTarget: this, preventDefault() {}, stopPropagation() {}, ...options,
    });
  }
  querySelectorAll(selector) {
    const selectors = selector.split(",").map(value => value.trim());
    const matches = child => selectors.some(value => {
      if (value.startsWith(".")) return child.classList.contains(value.slice(1));
      const attribute = value.match(/^(\w+)?\[([\w-]+)\]$/);
      if (attribute) return (!attribute[1] || child.tagName === attribute[1].toUpperCase())
        && child.getAttribute(attribute[2]) !== null;
      return child.tagName === value.toUpperCase();
    });
    return this.children.flatMap(child => [
      ...(matches(child) ? [child] : []), ...child.querySelectorAll(selector),
    ]);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

function browser({narrow = false, skills, preview, outbound = false} = {}) {
  const elements = new Map();
  const documentListeners = new Map(), windowListeners = new Map(), mediaListeners = new Set();
  const document = {
    activeElement: null,
    addEventListener: (name, callback) => documentListeners.set(name, callback),
  };
  const createElement = tag => {
    const element = new Element(tag);
    element.ownerDocument = document;
    return element;
  };
  document.body = createElement("body");
  const media = {
    get matches() { return narrow; },
    addEventListener: (_, callback) => mediaListeners.add(callback),
    addListener: callback => mediaListeners.add(callback),
  };
  const get = id => {
    if (!elements.has(id)) {
      const button = /^(inspector-(toggle|close|backdrop|tab-.*)|trace-(copy|toggle)|task-(jump|filter-.*))$/.test(id);
      const element = createElement(id === "trace-turn-select" ? "select" : button ? "button" : "div");
      if (["trace-fast", "trace-deep", "trace-voice"].includes(id)) {
        const state = new Element("span");
        state.className = "trace-lane-state";
        element.append(state);
      }
      if (id === "trace-turn-select") {
        const option = new Element("option");
        option.value = "all";
        option.textContent = "全部轮次";
        element.append(option);
        element.value = "all";
      }
      elements.set(id, element);
      document.body.append(element);
    }
    return elements.get(id);
  };
  // Mirror the drawer's relevant static structure so focus and hidden ancestors behave like a browser.
  const inspector = get("inspector-panel");
  inspector.hidden = true;
  inspector.setAttribute("tabindex", "-1");
  get("inspector-toggle").setAttribute("aria-expanded", "false");
  get("inspector-backdrop").hidden = true;
  get("workspace-grid").dataset.inspectorOpen = "false";
  inspector.append(get("inspector-close"));
  for (const name of ["trace", "tasks", "system"]) {
    const tab = get(`inspector-tab-${name}`), view = get(`inspector-view-${name}`);
    tab.setAttribute("aria-selected", String(name === "trace"));
    tab.setAttribute("tabindex", name === "trace" ? "0" : "-1");
    view.setAttribute("tabindex", "0");
    view.hidden = name !== "trace";
    inspector.append(tab);
  }
  inspector.append(get("inspector-body"));
  for (const name of ["trace", "tasks", "system"]) get("inspector-body").append(get(`inspector-view-${name}`));
  get("inspector-view-trace").append(get("trace-turn-select"), get("trace-copy"), get("trace-toggle"), get("trace-content"));
  get("trace-content").append(get("trace-list"));
  get("inspector-view-tasks").append(get("task-filter-all"), get("task-filter-active"), get("task-list"));
  const background = ["sidebar", "topbar", "page-title", "conversation-card", "page-footer"].map(className => {
    const element = createElement("section");
    element.className = className;
    document.body.append(element);
    return element;
  });
  const sent = [], sources = [], copied = [], timeouts = new Map(), intervals = new Map();
  let timerId = 0, now = 10000;
  class ClockDate extends Date { static now() { return now; } }
  const clipboard = {writeText: async text => { copied.push(text); }};
  class FakeWebSocket {
    static OPEN = 1;
    constructor() { this.readyState = 1; }
    send(raw) { sent.push(JSON.parse(raw)); }
    close() { this.readyState = 3; }
  }
  const audio = {
    state: "running", currentTime: 0, destination: {},
    createBuffer: (_, samples, rate) => ({
      duration: samples / rate, getChannelData: () => new Float32Array(samples),
    }),
    createBufferSource() {
      const source = {
        stopped: false, connect() {}, disconnect() {}, start() {}, stop() { this.stopped = true; },
      };
      sources.push(source);
      return source;
    },
    suspend() { this.state = "suspended"; return Promise.resolve(); },
    resume() { this.state = "running"; return Promise.resolve(); },
  };
  Object.assign(document, {
      getElementById: get, querySelector: () => new Element(),
      querySelectorAll: selector => document.body.querySelectorAll(selector),
      createElement,
      createTextNode: text => { const node = createElement("#text"); node.textContent = text; return node; },
  });
  const context = vm.createContext({
    document,
    window: {
      KsaSkills: skills, KsaXlsxPreview: preview,
      addEventListener: (name, callback) => windowListeners.set(name, callback),
      matchMedia: query => query.includes("prefers-reduced-motion") ? {matches: false} : media,
      location: {protocol: "http:", host: "localhost"},
    },
    navigator: {clipboard}, Date: ClockDate,
    WebSocket: FakeWebSocket, performance: {now: () => 1000},
    ArrayBuffer, Uint8Array, Float32Array, DataView,
    setTimeout: (callback, delay) => { const id = ++timerId; timeouts.set(id, {callback, delay}); return id; },
    clearTimeout: id => timeouts.delete(id),
    setInterval: (callback, delay) => { const id = ++timerId; intervals.set(id, {callback, delay}); return id; },
    clearInterval: id => intervals.delete(id),
    testAudio: audio, testOutbound: outbound,
    testSocket: {readyState: 1, send: raw => sent.push(JSON.parse(raw)), close() {}},
  });
  const script = fs.readFileSync(path.join(__dirname, "../src/ksa.js"), "utf8");
  vm.runInContext(script.replace("void initialize();", `
    connected = true; socket = testSocket; outputContext = testAudio; listening = true;
    activeTurnId = "current-turn"; audioTurnId = "current-turn"; busy = true; userSpeaking = true;
    outboundOpeningPending = testOutbound;
    globalThis.api = {
      handleMessage, disconnectCurrentSocket,
      state: () => ({activeTurnId, busy, userSpeaking, turnCount, assistantText, activeInput}),
    };
  `), context);
  return {
    ...context.api, get, sent, sources, copied, timeouts, intervals, clipboard,
    document, background,
    windowEvent: (name, event = {}) => windowListeners.get(name)?.(event),
    key: (key, extra = {}) => {
      const event = {key, preventDefault() {}, ...extra};
      documentListeners.get("keydown")?.(event);
      windowListeners.get("keydown")?.(event);
    },
    setNarrow(value) {
      narrow = value;
      for (const callback of mediaListeners) callback({matches: value});
    },
    setNow: value => { now = value; },
    dom: () => JSON.stringify([...elements].map(([id, element]) => [
      id, element.textContent, element.hidden, element.children.length,
    ])),
    event: message => context.api.handleMessage({data: JSON.stringify(message)}),
    taskUpdate(id, status = "running", extra = {}) {
      this.event({type: "task_update", turn_id: "old-turn", task: {
        id, title: `任务 ${id}`, status, origin_turn_id: "old-turn", ...extra,
      }});
    },
    pcm: () => context.api.handleMessage({data: new ArrayBuffer(4800)}),
    cards: () => get("task-list").querySelectorAll(".task-card"),
    card: id => get("task-list").querySelectorAll(".task-card").find(card => card.dataset.taskId === id),
    trace(event, envelope = {}) {
      this.event({type: "execution_event", event, ...envelope});
    },
    traceRows: () => get("trace-list").querySelectorAll(".trace-event"),
  };
}

function executionEvent(sequence, extra = {}) {
  return {
    id: `event-${sequence}`, sequence, at: 1000 + sequence * 100,
    turn_id: "turn-a", stage: "fast", name: "route_fast",
    label: "快模型开始回应", status: "running", detail: "原生语音直接回应", ...extra,
  };
}

test("the inspector stays collapsed while collecting events and preserves records and audio across reopening", () => {
  const page = browser();
  const panel = page.get("inspector-panel"), toggle = page.get("inspector-toggle");
  assert.equal(panel.hidden, true);
  page.pcm();
  const state = JSON.stringify(page.state());
  page.trace(executionEvent(1));
  page.taskUpdate("background");
  const row = page.traceRows()[0], card = page.card("background");
  row.querySelector(".trace-event-debug").open = true;
  assert.equal(panel.hidden, true);
  assert.equal(toggle.getAttribute("aria-expanded"), "false");
  assert.match(page.get("inspector-badge").textContent, /1 进行中/);
  toggle.focus();
  toggle.trigger("click");
  assert.equal(panel.hidden, false);
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  assert.equal(page.get("inspector-backdrop").hidden, true);
  page.get("inspector-close").trigger("click");
  assert.equal(panel.hidden, true);
  assert.equal(page.document.activeElement, toggle);
  page.trace(executionEvent(2, {stage: "deep", label: "慢模型处理中"}));
  page.taskUpdate("background", "completed", {result: "返回的结果"});
  assert.equal(panel.hidden, true);
  toggle.trigger("click");
  assert.equal(page.traceRows().length, 2);
  assert.equal(page.traceRows()[0], row);
  assert.equal(row.querySelector(".trace-event-debug").open, true);
  assert.equal(page.card("background"), card);
  assert.equal(card.querySelector(".task-result-text").textContent, "返回的结果");
  assert.equal(page.sources[0].stopped, false);
  assert.equal(JSON.stringify(page.state()), state);
  assert.deepEqual(page.sent, [{type: "playback", playing: true, turn_id: "current-turn"}]);
});

test("the task shortcut opens results and Escape returns focus to the invoking shortcut", () => {
  const page = browser();
  page.taskUpdate("background");
  const shortcut = page.get("task-jump");
  shortcut.focus();
  shortcut.trigger("click");
  assert.equal(page.get("inspector-panel").hidden, false);
  assert.equal(page.get("inspector-view-tasks").hidden, false);
  assert.equal(page.get("inspector-view-trace").hidden, true);
  assert.equal(page.document.activeElement, page.get("inspector-tab-tasks"));
  page.key("Escape", {isComposing: true});
  assert.equal(page.get("inspector-panel").hidden, false);
  page.key("Escape");
  assert.equal(page.get("inspector-panel").hidden, true);
  assert.equal(page.document.activeElement, shortcut);
  page.get("inspector-toggle").trigger("click");
  assert.equal(page.get("inspector-view-tasks").hidden, false);
  assert.equal(page.sent.length, 0);
});

test("inspector tabs support wrapping arrow navigation and Home and End keys", () => {
  const page = browser();
  page.get("inspector-toggle").trigger("click");
  const expectSelected = name => {
    for (const view of ["trace", "tasks", "system"]) {
      const selected = name === view, tab = page.get(`inspector-tab-${view}`);
      assert.equal(tab.getAttribute("aria-selected"), String(selected));
      assert.equal(tab.tabIndex, selected ? 0 : -1);
      assert.equal(page.get(`inspector-view-${view}`).hidden, !selected);
    }
    assert.equal(page.document.activeElement, page.get(`inspector-tab-${name}`));
  };
  expectSelected("trace");
  page.get("inspector-tab-trace").trigger("keydown", {key: "ArrowLeft"});
  expectSelected("system");
  page.get("inspector-tab-system").trigger("keydown", {key: "ArrowRight"});
  expectSelected("trace");
  page.get("inspector-tab-trace").trigger("keydown", {key: "ArrowRight"});
  expectSelected("tasks");
  page.get("inspector-tab-tasks").trigger("keydown", {key: "End"});
  expectSelected("system");
  page.get("inspector-tab-system").trigger("keydown", {key: "Home"});
  expectSelected("trace");
  assert.equal(page.sent.length, 0);
});

test("a narrow inspector traps focus in the visible tab and backdrop close restores the page", () => {
  const page = browser({narrow: true});
  const panel = page.get("inspector-panel"), toggle = page.get("inspector-toggle");
  toggle.focus();
  toggle.trigger("click");
  assert.equal(panel.getAttribute("role"), "dialog");
  assert.equal(panel.getAttribute("aria-modal"), "true");
  assert.equal(page.get("inspector-backdrop").hidden, false);
  assert.equal(page.document.body.classList.contains("inspector-overlay-open"), true);
  assert.ok(page.background.every(element => element.inert));
  page.get("inspector-close").focus();
  page.key("Tab", {shiftKey: true});
  assert.equal(page.document.activeElement, page.get("trace-toggle"));
  page.key("Tab");
  assert.equal(page.document.activeElement, page.get("inspector-close"));
  page.get("inspector-tab-tasks").trigger("click");
  page.get("inspector-close").focus();
  page.key("Tab", {shiftKey: true});
  assert.equal(page.document.activeElement, page.get("task-filter-active"));
  page.get("inspector-backdrop").trigger("click");
  assert.equal(panel.hidden, true);
  assert.equal(panel.getAttribute("aria-modal"), null);
  assert.equal(page.get("inspector-backdrop").hidden, true);
  assert.equal(page.document.body.classList.contains("inspector-overlay-open"), false);
  assert.ok(page.background.every(element => !element.inert));
  assert.equal(page.document.activeElement, toggle);
});

test("resizing an open inspector switches modal behavior without losing the selected tab or records", () => {
  const page = browser();
  page.trace(executionEvent(1));
  const row = page.traceRows()[0];
  page.get("inspector-toggle").trigger("click");
  page.get("inspector-tab-system").trigger("click");
  page.get("message-input").focus();
  page.setNarrow(true);
  assert.equal(page.document.activeElement, page.get("inspector-tab-system"));
  assert.equal(page.get("inspector-panel").getAttribute("aria-modal"), "true");
  assert.ok(page.background.every(element => element.inert));
  page.setNarrow(false);
  assert.equal(page.get("inspector-panel").hidden, false);
  assert.equal(page.get("inspector-panel").getAttribute("role"), "complementary");
  assert.equal(page.get("inspector-panel").getAttribute("aria-modal"), null);
  assert.equal(page.get("inspector-view-system").hidden, false);
  assert.equal(page.get("inspector-backdrop").hidden, true);
  assert.ok(page.background.every(element => !element.inert));
  assert.equal(page.document.body.classList.contains("inspector-overlay-open"), false);
  assert.equal(page.traceRows()[0], row);
});

test("new trace events follow the inspector only when its visible timeline is near the bottom", () => {
  const page = browser();
  const body = page.get("inspector-body");
  body.clientHeight = 400;
  Object.defineProperty(body, "scrollHeight", {get: () => 1000 + page.traceRows().length * 100});
  body.scrollTop = 590;
  page.trace(executionEvent(1));
  assert.equal(body.scrollTop, 590, "a closed inspector must not move when events arrive");
  page.get("inspector-toggle").trigger("click");
  body.scrollTop = 120;
  page.trace(executionEvent(2));
  assert.equal(body.scrollTop, 120, "reading earlier records must not jump to the newest event");
  body.scrollTop = body.scrollHeight - body.clientHeight - 20;
  page.trace(executionEvent(3));
  assert.equal(body.scrollTop, body.scrollHeight, "following uses the expanded sidebar scroll container");
  const select = page.get("trace-turn-select");
  select.value = "turn-a";
  select.trigger("change");
  body.scrollTop = body.scrollHeight - body.clientHeight;
  const filteredPosition = body.scrollTop;
  page.trace(executionEvent(4, {turn_id: "turn-b"}));
  assert.equal(body.scrollTop, filteredPosition, "filtered-out events must not move the current timeline");
  page.get("trace-toggle").trigger("click");
  body.scrollTop = body.scrollHeight - body.clientHeight;
  const collapsedPosition = body.scrollTop;
  page.trace(executionEvent(5));
  assert.equal(body.scrollTop, collapsedPosition, "a folded timeline must not claim the sidebar scroll position");
  page.get("inspector-tab-tasks").trigger("click");
  body.scrollTop = body.scrollHeight - body.clientHeight;
  const taskPosition = body.scrollTop;
  page.trace(executionEvent(6));
  assert.equal(body.scrollTop, taskPosition, "new trace events must not move the task results tab");
});

test("switching inspector tabs and reopening retain each tab's reading position", () => {
  const page = browser();
  const body = page.get("inspector-body");
  page.get("inspector-toggle").trigger("click");
  body.scrollTop = 230;
  page.get("inspector-tab-tasks").trigger("click");
  assert.equal(body.scrollTop, 0);
  body.scrollTop = 410;
  page.get("inspector-tab-system").trigger("click");
  assert.equal(body.scrollTop, 0);
  body.scrollTop = 70;
  page.get("inspector-tab-trace").trigger("click");
  assert.equal(body.scrollTop, 230);
  page.get("inspector-tab-tasks").trigger("click");
  assert.equal(body.scrollTop, 410);
  page.get("inspector-close").trigger("click");
  body.scrollTop = 0;
  page.trace(executionEvent(1));
  page.get("inspector-toggle").trigger("click");
  assert.equal(page.get("inspector-view-tasks").hidden, false);
  assert.equal(body.scrollTop, 410);
  page.get("inspector-tab-system").trigger("click");
  assert.equal(body.scrollTop, 70);
});

test("execution events from previous turns stay visible without taking over conversation or audio", () => {
  const page = browser();
  page.pcm();
  const state = JSON.stringify(page.state());
  const conversation = page.get("messages").textContent;
  page.trace(executionEvent(1), {turn_id: "old-turn"});
  page.trace(executionEvent(2, {stage: "deep", name: "route_deep", label: "进入慢模型",
    task_id: "task-a", call_id: "delegate-a", reason: "native_delegation"}), {turn_id: "old-turn"});
  assert.equal(page.traceRows().length, 2);
  assert.deepEqual(page.traceRows().map(row => row.dataset.stage), ["fast", "deep"]);
  assert.equal(JSON.stringify(page.state()), state);
  assert.equal(page.get("messages").textContent, conversation);
  assert.equal(page.sources[0].stopped, false);
  assert.deepEqual(page.sent, [{type: "playback", playing: true, turn_id: "current-turn"}]);
});

test("trace events sort by server sequence and deduplicate while retaining expanded details", () => {
  const page = browser();
  page.trace(executionEvent(3));
  const original = page.traceRows()[0];
  original.querySelector(".trace-event-debug").open = true;
  page.trace(executionEvent(1, {at: 5000}));
  page.trace(executionEvent(2, {at: 4000}));
  page.trace(executionEvent(3, {label: "重复事件不覆盖已有记录"}));
  page.trace(executionEvent(3, {id: "different-id-same-sequence"}));
  assert.deepEqual(page.traceRows().map(row => row.dataset.eventId), ["event-1", "event-2", "event-3"]);
  assert.equal(page.traceRows()[2], original);
  assert.equal(original.querySelector(".trace-event-debug").open, true);
  assert.equal(original.querySelector(".trace-event-label").textContent, "快模型开始回应");
  assert.match(page.traceRows()[1].querySelector(".trace-event-detail").textContent, /补记事件发生时间/);
});

test("turn filtering isolates lanes while unrelated background events continue arriving", () => {
  const page = browser();
  page.trace(executionEvent(1));
  page.trace(executionEvent(2, {stage: "deep", name: "deep_started", label: "A 慢模型执行", task_id: "task-a"}));
  page.trace(executionEvent(3, {turn_id: "turn-b", label: "B 快模型回应"}));
  const select = page.get("trace-turn-select");
  select.value = "turn-b";
  select.trigger("change");
  page.trace(executionEvent(4, {stage: "deep", name: "deep_result_ready", label: "A 慢模型完成",
    status: "completed", task_id: "task-a"}));
  page.trace(executionEvent(5, {turn_id: "", stage: "system", name: "session_notice", label: "会话提示", status: "info"}));
  assert.deepEqual(page.traceRows().filter(row => !row.hidden).map(row => row.dataset.eventId), ["event-3", "event-5"]);
  assert.equal(page.get("trace-deep").dataset.status, "idle");
  assert.equal(page.get("trace-fast").querySelector(".trace-lane-state").textContent, "B 快模型回应");
  select.value = "turn-a";
  select.trigger("change");
  assert.equal(page.get("trace-deep").dataset.status, "completed");
  assert.equal(page.get("trace-deep").querySelector(".trace-lane-state").textContent, "A 慢模型完成");
  assert.equal(select.children.filter(option => option.value === "turn-a").length, 1);
  select.value = "all";
  select.trigger("change");
  assert.match(page.get("trace-deep").querySelector(".trace-lane-state").textContent, /^第 1 轮 · A 慢模型完成$/);
  assert.match(page.get("trace-fast").querySelector(".trace-lane-state").textContent, /^第 2 轮 · B 快模型回应$/);
});

test("keyboard deep-only execution never invents a fast-model milestone", () => {
  const page = browser();
  page.trace(executionEvent(1, {stage: "input", name: "input_text", label: "收到键盘输入", status: "completed"}));
  page.trace(executionEvent(2, {stage: "deep", name: "route_deep", label: "直接进入慢模型", source: "keyboard"}));
  page.trace(executionEvent(3, {stage: "voice", name: "result_returned", label: "慢模型 → 语音模型"}));
  assert.equal(page.traceRows().some(row => row.dataset.stage === "fast"), false);
  assert.equal(page.get("trace-fast").dataset.status, "idle");
  assert.equal(page.get("trace-fast").querySelector(".trace-lane-state").textContent, "本轮未经过快模型");
  assert.equal(page.get("trace-voice").querySelector(".trace-lane-state").textContent, "第 1 轮 · 慢模型 → 语音模型");
});

test("selecting an origin turn follows its result announcement without mixing unrelated speech", async () => {
  const page = browser();
  page.trace(executionEvent(1, {stage: "deep", name: "deep_started", task_id: "task-a"}));
  page.trace(executionEvent(2, {turn_id: "turn-b", label: "另一轮快模型回复"}));
  page.trace(executionEvent(3, {stage: "voice", name: "announcement_started", label: "开始播报后台结果",
    task_id: "task-a", announcement_turn_id: "announcement-a"}));
  page.trace(executionEvent(4, {turn_id: "announcement-a", stage: "voice", name: "first_audio", label: "结果语音首包", status: "info"}));
  page.trace(executionEvent(5, {turn_id: "turn-b", stage: "voice", name: "first_audio", label: "另一轮语音首包", status: "info"}));
  page.get("trace-turn-select").value = "turn-a";
  page.get("trace-turn-select").trigger("change");
  assert.deepEqual(page.traceRows().filter(row => !row.hidden).map(row => row.dataset.eventId), ["event-1", "event-3", "event-4"]);
  assert.equal(page.get("trace-voice").querySelector(".trace-lane-state").textContent, "结果语音首包");
  await page.get("trace-copy").trigger("click");
  const exported = JSON.parse(page.copied.at(-1));
  assert.equal(exported.events.find(event => event.id === "event-3").announcement_turn_id, "announcement-a");
  assert.equal(exported.events.at(-1).turn_id, "announcement-a");
});

test("trace metrics use matching task milestones and never estimate missing completions", () => {
  const page = browser();
  page.trace(executionEvent(1, {at: 1000, stage: "input", name: "input_final", label: "输入已确认", status: "completed"}));
  page.trace(executionEvent(2, {at: 1250, name: "first_text", label: "首段文字", status: "info"}));
  page.trace(executionEvent(3, {at: 2200, stage: "voice", name: "first_audio", label: "首块语音", status: "info"}));
  page.trace(executionEvent(4, {at: 1500, stage: "deep", name: "deep_started", task_id: "task-a"}));
  page.trace(executionEvent(5, {at: 2500, stage: "deep", name: "deep_started", task_id: "task-b"}));
  page.trace(executionEvent(6, {at: 5000, stage: "deep", name: "deep_result_ready", status: "completed", task_id: "task-b"}));
  assert.match(page.get("trace-metrics").textContent, /首段文字 250 ms/);
  assert.match(page.get("trace-metrics").textContent, /首包音频（服务端） 1\.20 s/);
  assert.match(page.get("trace-metrics").textContent, /最近慢任务 2\.50 s/);
  page.trace(executionEvent(7, {turn_id: "turn-b", stage: "deep", name: "deep_started", task_id: "task-c"}));
  assert.match(page.get("trace-metrics").textContent, /等待对应的完成事件，耗时不作估算/);
});

test("debug JSON export preserves selected correlations and renders untrusted text literally", async () => {
  const page = browser();
  const hostile = '<img src=x onerror="alert(1)">';
  page.trace(executionEvent(1, {stage: "deep", name: "route_deep", label: hostile, detail: hostile,
    task_id: "task-a", call_id: "delegate-a", model: "model-a", source: "native", reason: "native_delegation", generation: 2,
    unexpected_secret: "not exported"}));
  page.trace(executionEvent(2, {turn_id: "turn-b"}));
  page.get("trace-turn-select").value = "turn-a";
  page.get("trace-turn-select").trigger("change");
  await page.get("trace-copy").trigger("click");
  const exported = JSON.parse(page.copied.at(-1));
  assert.equal(exported.clock, "server_unix_ms");
  assert.equal(exported.filter, "turn-a");
  assert.equal(exported.connection, "connected");
  assert.equal(exported.events.length, 1);
  assert.deepEqual(Object.fromEntries(["turn_id", "task_id", "call_id", "model", "source", "reason", "generation"].map(key => [key, exported.events[0][key]])), {
    turn_id: "turn-a", task_id: "task-a", call_id: "delegate-a", model: "model-a", source: "native", reason: "native_delegation", generation: 2,
  });
  assert.equal(exported.events[0].unexpected_secret, undefined);
  assert.equal(page.traceRows()[0].querySelector(".trace-event-label").textContent, hostile);
  assert.equal(page.get("trace-list").querySelectorAll("img").length, 0);
  assert.match(page.get("trace-notice").textContent, /已复制 1 条调试记录/);
  page.clipboard.writeText = async () => { throw new Error("denied"); };
  await assert.doesNotReject(() => page.get("trace-copy").trigger("click"));
  assert.match(page.get("trace-notice").textContent, /手动复制/);
});

test("trace history is capped and stale replay cannot reintroduce evicted events", async () => {
  const page = browser();
  for (let i = 1; i <= 205; i += 1) page.trace(executionEvent(i, {turn_id: i === 1 ? "old-turn" : "turn-a"}));
  assert.equal(page.traceRows().length, 200);
  assert.equal(page.traceRows()[0].dataset.eventId, "event-6");
  page.trace(executionEvent(1, {id: "late-old-record"}));
  assert.equal(page.traceRows().length, 200);
  assert.equal(page.get("trace-turn-select").children.some(option => option.value === "old-turn"), false);
  await page.get("trace-copy").trigger("click");
  const exported = JSON.parse(page.copied.at(-1));
  assert.equal(exported.events.length, 200);
  assert.equal(exported.discarded_events, 5);
  assert.match(page.get("trace-summary").textContent, /已省略较早 5 条/);
});

test("disconnect preserves trace evidence and reports waiting instead of fabricated completion", async () => {
  const page = browser();
  page.trace(executionEvent(1, {stage: "deep", name: "deep_started", label: "慢模型处理中"}));
  page.disconnectCurrentSocket();
  assert.equal(page.traceRows().length, 1);
  assert.equal(page.traceRows()[0].dataset.status, "running");
  assert.equal(page.get("trace-deep").dataset.status, "waiting");
  assert.match(page.get("trace-summary").textContent, /连接已断开/);
  assert.match(page.get("trace-deep").querySelector(".trace-lane-state").textContent, /已断开.*慢模型处理中/);
  await page.get("trace-copy").trigger("click");
  assert.equal(JSON.parse(page.copied.at(-1)).connection, "disconnected");
  page.event({type: "ready"});
  assert.equal(page.traceRows().length, 0);
  assert.equal(page.get("trace-copy").disabled, true);
  assert.equal(page.get("trace-empty").hidden, false);
  page.trace(executionEvent(1));
  assert.equal(page.traceRows().length, 1);
  assert.doesNotMatch(page.get("trace-summary").textContent, /连接已断开/);
  page.get("clear-button").trigger("click");
  assert.equal(page.traceRows().length, 0);
  assert.equal(page.get("trace-turn-select").value, "all");
  assert.equal(page.get("trace-notice").textContent, "");
});

test("collapsed trace continues receiving events without reopening itself", () => {
  const page = browser();
  page.get("trace-toggle").trigger("click");
  page.trace(executionEvent(1));
  assert.equal(page.get("trace-content").hidden, true);
  assert.equal(page.get("trace-toggle").attributes["aria-expanded"], "false");
  assert.equal(page.traceRows().length, 1);
  page.get("trace-toggle").trigger("click");
  assert.equal(page.get("trace-content").hidden, false);
  assert.equal(page.get("trace-toggle").attributes["aria-expanded"], "true");
});

test("malformed trace records cannot break subsequent valid telemetry", () => {
  const page = browser();
  for (const event of [null, [], {}, executionEvent(1, {id: ""}), executionEvent(1, {sequence: 1.5}),
    executionEvent(1, {at: -1}), executionEvent(1, {stage: "arbitrary"}), executionEvent(1, {status: "success"}),
    executionEvent(1, {turn_id: {unexpected: "object"}}), executionEvent(1, {label: null})]) {
    assert.doesNotThrow(() => page.trace(event));
  }
  assert.equal(page.traceRows().length, 0);
  page.trace(executionEvent(1));
  assert.equal(page.traceRows().length, 1);
});

test("session task updates render every status without changing the current conversation or queued audio", () => {
  const page = browser();
  page.pcm();
  const before = JSON.stringify(page.state());
  const conversation = page.get("messages").textContent;
  for (const status of ["queued", "running", "ready", "announcing", "completed", "canceled", "failed"]) {
    page.taskUpdate(status, status, {error: status === "failed" ? "搜索超时" : ""});
    assert.equal(page.card(status)?.dataset.status, status);
  }
  assert.equal(page.cards().length, 7);
  assert.equal(JSON.stringify(page.state()), before);
  assert.equal(page.get("messages").textContent, conversation);
  assert.equal(page.sources[0].stopped, false);
  assert.deepEqual(page.sent, [{type: "playback", playing: true, turn_id: "current-turn"}]);
});

test("ready results and real tool traces are readable before speech and render hostile content as text", async () => {
  const page = browser();
  const hostile = '<img src=x onerror="alert(1)">';
  page.taskUpdate("search", "ready", {
    title: hostile, question: "查询附近酒店", result: `最终结果 ${hostile}`,
    created_at: 1000, started_at: 2000, finished_at: 4000, updated_at: 4000,
    calls: [{id: "call-1", name: "global_search", status: "completed", started_at: 2100,
      finished_at: 3100, input: '{"query":"附近酒店"}', output: `工具返回 ${hostile}`, error: ""}],
  });
  const card = page.card("search");
  assert.equal(card.querySelector(".task-title").textContent, hostile);
  assert.equal(card.querySelector(".task-result-text").textContent, `最终结果 ${hostile}`);
  assert.equal(card.querySelectorAll(".task-call").length, 1);
  assert.match(card.textContent, /global_search/);
  assert.match(card.textContent, /附近酒店/);
  assert.ok(card.textContent.includes(`工具返回 ${hostile}`));
  assert.equal(card.querySelectorAll("img").length, 0);
  assert.equal(page.get("messages").children.length, 0);
  await card.querySelector(".task-copy").trigger("click");
  assert.deepEqual(page.copied, [`最终结果 ${hostile}`]);
});

test("canceling one task waits for its server update and preserves foreground audio", () => {
  const page = browser();
  page.taskUpdate("lodging");
  page.pcm();
  const before = JSON.stringify(page.state());
  const button = page.card("lodging").querySelector(".task-cancel");
  button.trigger("click");
  button.trigger("click");
  assert.deepEqual(page.sent.filter(message => message.type === "cancel_task"), [
    {type: "cancel_task", task_id: "lodging"},
  ]);
  assert.equal(page.sent.some(message => message.type === "interrupt"), false);
  assert.equal(page.card("lodging").dataset.status, "running");
  assert.equal(JSON.stringify(page.state()), before);
  assert.equal(page.sources[0].stopped, false);
  page.taskUpdate("lodging", "canceled");
  assert.equal(page.card("lodging").dataset.status, "canceled");
  assert.equal(page.card("lodging").querySelector(".task-cancel").hidden, true);
});

test("a cancel timeout allows retry without claiming the task was canceled", () => {
  const page = browser();
  page.taskUpdate("slow");
  const card = page.card("slow");
  const button = card.querySelector(".task-cancel");
  const notice = card.querySelector(".task-actions").querySelector(".task-note");
  button.trigger("click");
  assert.equal(button.disabled, true);
  const [timerId, timer] = [...page.timeouts].find(([, timer]) => timer.delay === 8000);
  page.timeouts.delete(timerId);
  timer.callback();
  assert.equal(button.disabled, false);
  assert.equal(card.dataset.status, "running");
  assert.equal(card.querySelector(".task-status").textContent, "执行中");
  assert.equal(notice.attributes.role, "status");
  assert.match(notice.textContent, /尚未收到取消确认/);
  button.trigger("click");
  assert.equal(page.sent.filter(message => message.type === "cancel_task").length, 2);
  assert.equal(button.disabled, true);
  page.taskUpdate("slow", "canceled");
  assert.equal(card.dataset.status, "canceled");
  assert.equal(page.timeouts.size, 0);
});

test("older task snapshots cannot undo progress and canceled tasks cannot be revived", () => {
  const page = browser();
  page.taskUpdate("search", "running", {title: "正在查询", updated_at: 3000});
  page.taskUpdate("search", "queued", {title: "过期排队快照", updated_at: 2000});
  const card = page.card("search");
  assert.equal(card.dataset.status, "running");
  assert.equal(card.querySelector(".task-title").textContent, "正在查询");
  page.taskUpdate("search", "canceled", {updated_at: 4000});
  for (const status of ["running", "ready", "completed"]) {
    page.taskUpdate("search", status, {updated_at: 5000, result: "迟到的结果"});
    assert.equal(card.dataset.status, "canceled");
    assert.equal(card.querySelector(".task-result-text").textContent, "");
    assert.equal(card.querySelector(".task-cancel").hidden, true);
  }
});

test("task and tool elapsed times stop at their execution finish even while other work keeps ticking", () => {
  const page = browser();
  page.setNow(5000);
  const call = {id: "call-1", name: "global_search", status: "running", started_at: 2000};
  page.taskUpdate("search", "running", {
    created_at: 500, started_at: 1000, calls: [call],
  });
  const card = page.card("search");
  const meta = card.querySelector(".task-meta");
  const toolTime = card.querySelector(".task-call-time");
  assert.match(meta.textContent, /已执行 4\.0 秒$/);
  assert.equal(toolTime.textContent, "3.0 秒");
  page.setNow(8000);
  for (const timer of page.intervals.values()) timer.callback();
  assert.match(meta.textContent, /已执行 7\.0 秒$/);
  assert.equal(toolTime.textContent, "6.0 秒");
  page.setNow(10000);
  page.taskUpdate("search", "ready", {finished_at: 10000, result: "查询结果",
    calls: [{...call, status: "completed", finished_at: 8500, output: "已返回"}],
  });
  assert.match(meta.textContent, /耗时 9\.0 秒$/);
  assert.equal(toolTime.textContent, "6.5 秒");
  page.taskUpdate("other", "running", {started_at: 10000});
  assert.equal(page.intervals.size, 1);
  page.setNow(20000);
  for (const timer of page.intervals.values()) timer.callback();
  page.taskUpdate("search", "announcing");
  assert.match(meta.textContent, /耗时 9\.0 秒$/);
  assert.equal(toolTime.textContent, "6.5 秒");
  assert.match(page.card("other").querySelector(".task-meta").textContent, /已执行 10 秒$/);
});

test("clipboard denial gives an accessible manual-copy notice and leaves the result readable", async () => {
  const page = browser();
  page.clipboard.writeText = async () => { throw new Error("Clipboard permission denied"); };
  page.taskUpdate("answer", "ready", {result: "仍可选中并复制的结果"});
  const card = page.card("answer");
  await assert.doesNotReject(() => card.querySelector(".task-copy").trigger("click"));
  const notice = card.querySelector(".task-actions").querySelector(".task-note");
  assert.equal(notice.attributes.role, "status");
  assert.match(notice.textContent, /无法自动复制，请选中返回结果后复制/);
  assert.equal(card.querySelector(".task-result-text").textContent, "仍可选中并复制的结果");
  assert.equal(card.dataset.status, "ready");
  assert.deepEqual(page.copied, []);
});

test("task filters preserve cards and details while subsequent updates refresh visible status", () => {
  const page = browser();
  page.taskUpdate("running");
  page.taskUpdate("done", "completed", {result: "已完成的结果"});
  const running = page.card("running");
  running.open = true;
  page.get("task-filter-active").trigger("click");
  assert.equal(running.hidden, false);
  assert.equal(page.card("done").hidden, true);
  page.taskUpdate("running", "ready", {result: "新结果"});
  assert.equal(page.card("running"), running);
  assert.equal(running.open, true);
  page.get("task-filter-all").trigger("click");
  assert.equal(page.card("done").hidden, false);
});

test("disconnect retains a truthful snapshot and the next session clears task records and timers", () => {
  const page = browser();
  page.taskUpdate("active");
  page.taskUpdate("done", "completed", {result: "已返回"});
  page.disconnectCurrentSocket();
  assert.equal(page.card("active").dataset.status, "disconnected");
  assert.equal(page.card("done").dataset.status, "completed");
  assert.equal(page.card("done").querySelector(".task-result-text").textContent, "已返回");
  assert.equal(page.intervals.size, 0);
  page.event({type: "ready", output_sample_rate: 24000});
  assert.equal(page.cards().length, 0);
  assert.equal(page.get("task-empty").hidden, false);
  page.taskUpdate("fresh");
  assert.equal(page.cards().length, 1);
});

test("clearing the conversation immediately clears tasks even before a new connection is ready", () => {
  const page = browser();
  page.taskUpdate("active");
  page.taskUpdate("done", "completed", {result: "旧结果"});
  page.get("clear-button").trigger("click");
  assert.equal(page.cards().length, 0);
  assert.equal(page.get("task-empty").hidden, false);
  assert.equal(page.intervals.size, 0);
  assert.equal(page.state().turnCount, 0);
});

test("terminal history stays bounded without evicting active tasks", () => {
  const page = browser();
  page.taskUpdate("active");
  for (let i = 0; i < 40; i += 1) page.taskUpdate(`done-${i}`, "completed", {result: `结果 ${i}`});
  assert.ok(page.card("active"));
  assert.ok(page.card("done-39"));
  assert.equal(page.card("done-0"), undefined);
  assert.equal(page.cards().length, 31);
});

test("malformed task events are ignored without breaking the next valid event", () => {
  const page = browser();
  for (const task of [null, [], {}, {id: ""}, {id: 2, status: "running"},
    {id: "bad", status: "unexpected"}, {id: "bad", status: null}]) {
    assert.doesNotThrow(() => page.event({type: "task_update", task}));
  }
  assert.equal(page.cards().length, 0);
  page.taskUpdate("valid");
  assert.equal(page.cards().length, 1);
});

test("barge-in stops current audio without canceling background work", () => {
  const page = browser();
  page.taskUpdate("lodging");
  page.pcm();
  page.event({type: "barge_in_confirmed"});
  assert.equal(page.sources[0].stopped, true);
  assert.equal(page.get("recording-caption").textContent, "当前播报已停止，继续说话即可");
  assert.equal(page.sent.some(message => message.type.startsWith("cancel_")), false);
});

test("a background result creates only an assistant turn and owns its audio", () => {
  const page = browser();
  page.taskUpdate("lodging", "announcing");
  page.event({type: "task_announcement", task_id: "lodging", turn_id: "notice-1", title: "民宿搜索"});
  assert.equal(page.state().activeTurnId, "notice-1");
  assert.equal(page.state().activeInput, "task");
  assert.equal(page.state().turnCount, 0);
  assert.equal(page.get("messages").children.some(row => row.classList.contains("user")), false);
  assert.equal(page.get("messages").children.length, 1);
  assert.equal(page.get("model-route").textContent, "Deep Agents 协作");
  page.event({type: "assistant_delta", turn_id: "old-turn", text: "旧回复"});
  assert.equal(page.state().assistantText, "");
  page.pcm();
  assert.equal(page.sources.length, 0);
  page.event({type: "audio_start", turn_id: "notice-1", format: "pcm", sample_rate: 24000});
  page.event({type: "assistant_delta", turn_id: "notice-1", text: "这家民宿提供免费停车，房价需要按入住日期确认。"});
  page.pcm();
  assert.equal(page.sources.length, 1);
  assert.equal(page.state().assistantText, "这家民宿提供免费停车，房价需要按入住日期确认。");
  const before = JSON.stringify(page.state());
  const conversation = page.get("messages").textContent;
  page.taskUpdate("lodging", "completed");
  assert.equal(JSON.stringify(page.state()), before);
  assert.equal(page.get("messages").textContent, conversation);
  assert.equal(page.card("lodging").dataset.status, "completed");
  assert.equal(page.sources[0].stopped, false);
});

test("interrupting a background result stops its audio and keeps the conversation usable", () => {
  const page = browser();
  page.event({type: "task_announcement", task_id: "lodging", turn_id: "notice-1", title: "民宿搜索"});
  page.event({type: "audio_start", turn_id: "notice-1", format: "pcm", sample_rate: 24000});
  page.event({type: "assistant_delta", turn_id: "notice-1", text: "这家民宿提供免费停车。"});
  page.pcm();
  page.event({type: "barge_in_confirmed"});
  assert.equal(page.sources[0].stopped, true);
  assert.equal(page.state().activeTurnId, null);
  page.event({type: "speech_start", turn_id: "next-turn"});
  assert.equal(page.state().activeTurnId, "next-turn");
  assert.equal(page.state().userSpeaking, true);
  assert.equal(page.sent.some(message => message.type.startsWith("cancel_")), false);
});


test("a dial's outbound opening creates only a sales assistant turn and keeps background work independent", () => {
  const page = browser({outbound: true});
  page.taskUpdate("quote", "running");
  page.event({type: "outbound_opening", turn_id: "opening-1"});
  assert.equal(page.state().activeTurnId, "opening-1");
  assert.equal(page.state().activeInput, "outbound");
  assert.equal(page.state().turnCount, 0);
  assert.equal(page.state().busy, true);
  assert.match(page.get("recording-caption").textContent, /车险沟通/);
  assert.equal(page.get("messages").children.length, 0, "do not invent a customer question");
  page.event({type: "assistant_delta", turn_id: "opening-1", text: "您好，我是保险公司的车险服务专员，请问方便沟通续保吗？"});
  assert.equal(page.get("messages").children.length, 1);
  assert.equal(page.get("messages").querySelectorAll(".user").length, 0);
  assert.match(page.get("messages").textContent, /车险坐席/);
  page.event({type: "audio_start", turn_id: "opening-1", format: "pcm", sample_rate: 24000});
  page.pcm();
  assert.equal(page.sources.length, 1);
  assert.equal(page.sources[0].stopped, false);
  const state = JSON.stringify(page.state());
  page.event({type: "outbound_opening", turn_id: "duplicate-opening"});
  page.event({type: "assistant_delta", turn_id: "old-turn", text: "旧开场"});
  assert.equal(JSON.stringify(page.state()), state);
  assert.equal(page.state().turnCount, 0);
  assert.equal(page.card("quote").dataset.status, "running");
  assert.equal(page.sent.some(event => event.type.startsWith("cancel_")), false);
});

test("unsolicited or malformed outbound events cannot replace the current conversation", () => {
  const page = browser();
  const state = JSON.stringify(page.state());
  page.event({type: "outbound_opening", turn_id: "not-requested"});
  assert.equal(JSON.stringify(page.state()), state);
  const pending = browser({outbound: true});
  for (const turn_id of [undefined, null, "", 8, "x".repeat(81)]) {
    pending.event({type: "outbound_opening", turn_id});
    assert.equal(pending.state().activeTurnId, "current-turn");
  }
  pending.event({type: "outbound_opening", turn_id: "valid-opening"});
  assert.equal(pending.state().activeTurnId, "valid-opening");
});

test("user speech or an interruption retires a pending opening before its late event arrives", () => {
  for (const prior of [
    {type: "speech_start", turn_id: "customer-turn"},
    {type: "barge_in_confirmed"},
    {type: "task_announcement", task_id: "quote", turn_id: "quote-result"},
  ]) {
    const page = browser({outbound: true});
    page.event(prior);
    const state = JSON.stringify(page.state());
    page.event({type: "outbound_opening", turn_id: "late-opening"});
    assert.equal(JSON.stringify(page.state()), state);
    assert.equal(page.get("messages").querySelectorAll(".user").length, 0);
  }
});

test("an interrupted sales opening drops its old audio and continues with final customer input", () => {
  const page = browser({outbound: true});
  page.event({type: "outbound_opening", turn_id: "opening-1"});
  page.event({type: "assistant_delta", turn_id: "opening-1", text: "您好，我是保险公司的车险服务专员。"});
  page.event({type: "audio_start", turn_id: "opening-1", format: "pcm", sample_rate: 24000});
  page.pcm();
  page.event({type: "barge_in_confirmed"});
  assert.equal(page.sources[0].stopped, true);
  page.event({type: "speech_start", turn_id: "customer-turn"});
  page.event({type: "transcript", turn_id: "customer-turn", text: "我想了解续保方案", is_final: true});
  assert.equal(page.state().turnCount, 1);
  assert.equal(page.get("messages").querySelectorAll(".user").length, 1);
  const state = JSON.stringify(page.state());
  page.event({type: "audio_start", turn_id: "opening-1", format: "pcm", sample_rate: 24000});
  page.pcm();
  assert.equal(page.sources.length, 1, "late opening audio must not enter the new customer turn");
  assert.equal(JSON.stringify(page.state()), state);
});

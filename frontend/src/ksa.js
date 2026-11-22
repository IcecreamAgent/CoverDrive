// Native voice and task lifecycle adapted from KSA; Voice Agent keeps its original page layout.
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    messages: $("messages"), welcome: $("welcome"), input: $("message-input"), send: $("send-button"),
    mic: $("microphone-button"), micLabel: $("microphone-label"),
    mute: $("mute-button"), muteLabel: $("mute-label"), micStatus: $("mic-status"), callStatus: $("call-status"), interrupt: $("interrupt-button"),
    recordingBar: $("recording-bar"), recordingTime: $("recording-time"), recordingCaption: $("recording-caption"),
    connection: $("connection-label"), connectionDot: $("connection-dot"), error: $("error-banner"),
    errorText: $("error-text"), counter: $("turn-counter"), authForm: $("auth-form"), token: $("auth-token"),
    modelRoute: $("model-route"), voiceProvider: $("voice-provider"),
    taskList: $("task-list"), taskEmpty: $("task-empty"), taskCount: $("task-count"),
    taskSummary: $("task-summary"), taskAll: $("task-filter-all"), taskActive: $("task-filter-active"),
    taskEmptyTitle: $("task-empty-title"), taskEmptyDescription: $("task-empty-description"),
    taskJump: $("task-jump"),
    traceList: $("trace-list"), traceEmpty: $("trace-empty"), traceSelect: $("trace-turn-select"),
    traceCopy: $("trace-copy"), traceToggle: $("trace-toggle"), traceContent: $("trace-content"),
    traceSummary: $("trace-summary"), traceNotice: $("trace-notice"), traceMode: $("trace-mode"),
    traceModels: $("trace-models"), traceMetrics: $("trace-metrics"),
    inspector: $("inspector-panel"), inspectorToggle: $("inspector-toggle"),
    inspectorClose: $("inspector-close"), inspectorBackdrop: $("inspector-backdrop"),
    inspectorBadge: $("inspector-badge"), inspectorBody: $("inspector-body"), workspace: $("workspace-grid"),
  };
  const stageDefaults = { asr: "等待语音输入", agent: "准备好为你思考", tts: "等待生成回应" };
  let socket = null;
  let connected = false;
  let connecting = false;
  let reconnectTimer = null;
  let reconnectDelay = 1000;
  let authRequired = false;
  let authToken = "";
  let leaving = false;
  let listening = false;
  let microphoneMuted = false;
  let listenerId = null;
  let outboundOpeningPending = false;
  let outboundOpeningSeen = false;
  let userSpeaking = false;
  let playbackPaused = false;
  let playbackReportTurn = null;
  let captureStarting = false;
  let captureStopping = false;
  let captureGeneration = 0;
  let capture = null;
  let recordTimer = null;
  let recordStarted = 0;
  let busy = false;
  let playbackActive = false;
  let userMessage = null;
  let assistantMessage = null;
  let assistantText = "";
  let activeRoute = null;
  let activeInput = "voice";
  let activeTurnId = null;
  let audioTurnId = null;
  let turnCount = 0;
  let outputSampleRate = 24000;
  let outputContext = null;
  let nextAudioTime = 0;
  let audioSources = new Set();
  const playbackBarriers = new Map();
  let audioByteRemainder = null;
  let agentProgress = null;
  let agentProgressTimer = null;
  const taskRecords = new Map();
  const taskStatuses = {
    queued: "排队中", running: "执行中", ready: "结果已返回", announcing: "正在播报",
    completed: "已完成", canceled: "已取消", failed: "执行失败", disconnected: "连接已断开",
  };
  const activeTaskStatuses = new Set(["queued", "running", "ready", "announcing"]);
  let taskFilter = "all";
  let taskTimer = null;
  const traceEvents = [];
  const traceViews = new Map();
  const traceTurns = new Map();
  const traceStageNames = {input:"输入", fast:"快模型", deep:"慢模型", tool:"工具", voice:"语音接续", system:"会话"};
  const traceStatuses = new Set(["running", "completed", "waiting", "failed", "canceled", "info"]);
  let traceFilter = "all";
  let traceTurnCount = 0;
  let traceDiscarded = 0;
  let traceDisconnected = false;
  let inspectorOpen = false;
  let inspectorView = "trace";
  let inspectorOpener = null;
  const inspectorScrollPositions = {trace:0, tasks:0, system:0};
  const inspectorViews = ["trace", "tasks", "system"];
  const inspectorMedia = window.matchMedia?.("(max-width: 1099px)") || {matches:false};

  function updateInspectorBadge() {
    const active = [...taskRecords.values()].filter(record => activeTaskStatuses.has(record.data.status)).length;
    ui.inspectorBadge.textContent = active ? `${active} 进行中`
      : traceEvents.length ? `${traceEvents.length} 条` : taskRecords.size ? `${taskRecords.size} 任务` : "";
    ui.inspectorBadge.hidden = !ui.inspectorBadge.textContent;
    ui.inspectorBadge.dataset.active = String(active > 0);
    ui.inspectorToggle.setAttribute("aria-expanded", String(inspectorOpen));
    ui.inspectorToggle.setAttribute("aria-label", `${inspectorOpen ? "收起" : "展开"}运行详情${active ? `，${active} 个任务进行中` : ""}`);
    ui.inspectorToggle.title = `执行过程、任务结果与系统状态${traceEvents.length ? ` · ${traceEvents.length} 条执行事件` : ""}`;
  }
  function selectInspectorView(view, focus = true) {
    if (!inspectorViews.includes(view)) return;
    if (!ui.inspector.hidden) inspectorScrollPositions[inspectorView] = ui.inspectorBody.scrollTop;
    inspectorView = view;
    for (const name of inspectorViews) {
      const selected = name === view;
      const tab = $(`inspector-tab-${name}`);
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      $(`inspector-view-${name}`).hidden = !selected;
    }
    ui.inspectorBody.scrollTop = inspectorScrollPositions[view];
    if (focus) $(`inspector-tab-${view}`).focus({preventScroll:true});
  }
  function syncInspectorLayout() {
    const overlay = inspectorOpen && inspectorMedia.matches;
    ui.workspace.dataset.inspectorOpen = String(inspectorOpen);
    ui.inspector.hidden = !inspectorOpen;
    ui.inspector.dataset.overlay = String(overlay);
    ui.inspector.setAttribute("role", overlay ? "dialog" : "complementary");
    if (overlay) ui.inspector.setAttribute("aria-modal", "true");
    else ui.inspector.removeAttribute("aria-modal");
    ui.inspectorBackdrop.hidden = !overlay;
    document.body?.classList.toggle("inspector-overlay-open", overlay);
    for (const element of document.querySelectorAll(".sidebar, .topbar, .page-title, .conversation-card, .page-footer")) {
      element.inert = overlay;
    }
    if (overlay && !ui.inspector.contains(document.activeElement)) {
      $(`inspector-tab-${inspectorView}`).focus({preventScroll:true});
    }
    updateInspectorBadge();
  }
  function openInspector(view = inspectorView) {
    if (!inspectorOpen) inspectorOpener = document.activeElement;
    inspectorOpen = true;
    selectInspectorView(view, false);
    syncInspectorLayout();
    ui.inspectorBody.scrollTop = inspectorScrollPositions[view];
    $(`inspector-tab-${inspectorView}`).focus({preventScroll:true});
  }
  function closeInspector() {
    if (!inspectorOpen) return;
    inspectorScrollPositions[inspectorView] = ui.inspectorBody.scrollTop;
    inspectorOpen = false;
    syncInspectorLayout();
    const opener = inspectorOpener?.isConnected && inspectorOpener.getClientRects().length && !inspectorOpener.disabled
      ? inspectorOpener : ui.inspectorToggle;
    opener.focus({preventScroll:true});
    inspectorOpener = null;
  }
  function handleInspectorKeydown(event) {
    if (!inspectorOpen || event.isComposing) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeInspector();
      return;
    }
    if (event.key !== "Tab" || !inspectorMedia.matches) return;
    const focusable = [...ui.inspector.querySelectorAll("button, select, input, textarea, a[href], summary, [tabindex]")]
      .filter(element => !element.disabled && element.tabIndex !== -1 && element.getClientRects().length);
    if (!focusable.length) { event.preventDefault(); ui.inspector.focus(); return; }
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !ui.inspector.contains(document.activeElement))) {
      event.preventDefault(); last.focus();
    } else if (!event.shiftKey && (document.activeElement === last || !ui.inspector.contains(document.activeElement))) {
      event.preventDefault(); first.focus();
    }
  }

  function traceMilliseconds(value) {
    return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
  }
  function relatedTraceTurns(turnId) {
    return new Set([turnId, ...traceEvents.filter(event => event.turn_id === turnId && event.announcement_turn_id)
      .map(event => event.announcement_turn_id)]);
  }
  function visibleTraceEvents() {
    const turns = relatedTraceTurns(traceFilter);
    return traceEvents.filter(event => traceFilter === "all" || !event.turn_id || turns.has(event.turn_id));
  }
  function updateTraceConfiguration(config) {
    const native = config.voice_provider === "seeduplex";
    if (!config.voice_provider) return;
    ui.traceMode.textContent = native ? "原生语音 · 自主委派" : "三段式 · 模型路由";
    const fast = native ? `Seeduplex ${config.seeduplex_model || ""}`.trim()
      : config.fast_llm_enabled ? config.fast_llm_model || "Mini" : "未启用";
    ui.traceModels.textContent = `快模型：${fast} · 慢模型：${config.llm_model || "Deep Agents"}${native ? " · 键盘输入直接进入慢模型" : ""}`;
  }
  function updateTraceMetrics(events) {
    let turnId = traceFilter !== "all" ? traceFilter : [...events].reverse().find(event => event.turn_id)?.turn_id;
    if (traceFilter === "all") turnId = [...events].reverse().find(event => event.announcement_turn_id === turnId)?.turn_id || turnId;
    const turns = relatedTraceTurns(turnId);
    const turnEvents = events.filter(event => turns.has(event.turn_id));
    if (!turnEvents.length) { ui.traceMetrics.textContent = "收到执行事件后显示关键耗时。"; return; }
    const input = turnEvents.find(event => ["input_final", "input_text"].includes(event.name));
    const firstText = turnEvents.find(event => event.name === "first_text");
    const firstAudio = turnEvents.find(event => event.name === "first_audio");
    const segments = [`第 ${traceTurns.get(turnId)?.number || "—"} 轮`];
    for (const [label, milestone] of [["首段文字", firstText], ["首包音频（服务端）", firstAudio]]) {
      if (input && milestone && milestone.at >= input.at) segments.push(`${label} ${traceMilliseconds(milestone.at - input.at)}`);
    }
    const result = [...turnEvents].reverse().find(event => event.name === "deep_result_ready");
    const start = result && turnEvents.find(event => event.name === "deep_started"
      && (result.task_id ? event.task_id === result.task_id : event.call_id === result.call_id));
    if (start && result.at >= start.at) segments.push(`最近慢任务 ${traceMilliseconds(result.at - start.at)}`);
    if (segments.length === 1) segments.push("等待对应的完成事件，耗时不作估算");
    ui.traceMetrics.textContent = segments.join(" · ");
  }
  function renderExecutionTrace() {
    const events = visibleTraceEvents();
    const visibleIds = new Set(events.map(event => event.id));
    const previous = new Map();
    for (const event of traceEvents) {
      const view = traceViews.get(event.id);
      view.row.hidden = !visibleIds.has(event.id);
      const turn = traceTurns.get(event.turn_id);
      const baseline = turn?.base || event.at;
      const absolute = new Date(event.at).toLocaleTimeString("zh-CN", {hour12:false, hour:"2-digit", minute:"2-digit", second:"2-digit", fractionalSecondDigits:3});
      view.time.textContent = `${absolute}\n+${traceMilliseconds(Math.max(0, event.at - baseline))}`;
      view.lane.textContent = `${traceStageNames[event.stage]} · ${turn ? `第 ${turn.number} 轮` : "会话事件"}`;
      const prior = previous.get(event.turn_id);
      const gap = prior ? event.at - prior.at : null;
      view.detail.textContent = [event.detail, gap !== null && gap >= 0 ? `距同轮上条 ${traceMilliseconds(gap)}` : gap < 0 ? "补记事件发生时间" : ""].filter(Boolean).join(" · ");
      previous.set(event.turn_id, event);
    }
    ui.traceEmpty.hidden = events.length > 0;
    ui.traceCopy.disabled = !events.length;
    ui.traceSummary.textContent = `${traceDisconnected ? "连接已断开 · " : ""}${events.length} 条事件${traceDiscarded ? ` · 已省略较早 ${traceDiscarded} 条` : ""}`;
    for (const stage of ["fast", "deep", "voice"]) {
      const lane = $(`trace-${stage}`);
      const event = [...events].reverse().find(item => item.stage === stage);
      lane.dataset.status = event ? (traceDisconnected && ["running", "waiting"].includes(event.status) ? "waiting" : event.status) : "idle";
      const label = lane.querySelector(".trace-lane-state");
      if (!label) continue;
      label.textContent = event
        ? `${traceDisconnected ? "已断开 · " : ""}${traceFilter === "all" && event.turn_id ? `第 ${traceTurns.get(event.turn_id)?.number} 轮 · ` : ""}${event.label}`
        : stage === "fast" && events.some(item => item.name === "input_text") ? "本轮未经过快模型" : "尚无事件";
    }
    updateTraceMetrics(events);
    updateInspectorBadge();
  }
  function receiveExecutionEvent(incoming) {
    if (!incoming || typeof incoming !== "object" || typeof incoming.id !== "string" || !incoming.id || incoming.id.length > 128
      || !Number.isSafeInteger(incoming.sequence) || incoming.sequence < 0 || !taskTimestamp(incoming.at)
      || !Object.hasOwn(traceStageNames, incoming.stage) || !traceStatuses.has(incoming.status)
      || typeof incoming.name !== "string" || !incoming.name || typeof incoming.label !== "string"
      || (incoming.turn_id != null && typeof incoming.turn_id !== "string")) return;
    if (traceViews.has(incoming.id) || traceEvents.some(event => event.sequence === incoming.sequence)) return;
    // Records older than the retained window must not reappear on delayed delivery.
    if (traceDiscarded && traceEvents.length && incoming.sequence < traceEvents[0].sequence) return;
    const event = {
      id:incoming.id, sequence:incoming.sequence, at:incoming.at, turn_id:(incoming.turn_id || "").slice(0, 128),
      stage:incoming.stage, name:incoming.name.slice(0, 100), label:incoming.label.slice(0, 160),
      status:incoming.status, detail:typeof incoming.detail === "string" ? incoming.detail.slice(0, 1200) : "",
    };
    for (const key of ["task_id", "call_id", "announcement_turn_id", "model", "provider", "source", "reason"]) {
      if (typeof incoming[key] === "string") event[key] = incoming[key].slice(0, 240);
    }
    if (Number.isSafeInteger(incoming.generation)) event.generation = incoming.generation;
    if (event.turn_id) {
      let turn = traceTurns.get(event.turn_id);
      if (!turn) {
        const option = taskElement("option", "");
        option.value = event.turn_id;
        turn = {number:++traceTurnCount, base:event.at, option};
        option.textContent = `第 ${turn.number} 轮 · ${event.turn_id.slice(-8)}`;
        traceTurns.set(event.turn_id, turn);
        ui.traceSelect.append(option);
      } else turn.base = Math.min(turn.base, event.at);
    }
    const shouldFollow = inspectorOpen && inspectorView === "trace" && !ui.traceContent.hidden
      && ui.inspectorBody.scrollHeight - ui.inspectorBody.scrollTop - ui.inspectorBody.clientHeight < 60;
    const row = taskElement("li", "trace-event");
    row.dataset.eventId = event.id;
    row.dataset.stage = event.stage;
    row.dataset.status = event.status;
    const time = taskElement("span", "trace-event-time");
    const main = taskElement("div", "trace-event-main");
    const heading = taskElement("div", "trace-event-heading");
    const label = taskElement("span", "trace-event-label", event.label);
    const lane = taskElement("span", "trace-event-lane");
    heading.append(label, lane);
    const detail = taskElement("p", "trace-event-detail");
    const debug = taskElement("details", "trace-event-debug");
    debug.append(taskElement("summary", "", "关联信息"), taskElement("pre", "trace-event-data", JSON.stringify(event, null, 2)));
    main.append(heading, detail, debug);
    row.append(time, main);
    const before = traceEvents.findIndex(item => item.sequence > event.sequence);
    if (before < 0) { traceEvents.push(event); ui.traceList.append(row); }
    else { ui.traceList.insertBefore(row, traceViews.get(traceEvents[before].id).row); traceEvents.splice(before, 0, event); }
    traceViews.set(event.id, {row, time, lane, detail});
    while (traceEvents.length > 200) {
      const removed = traceEvents.shift();
      traceViews.get(removed.id).row.remove();
      traceViews.delete(removed.id);
      traceDiscarded += 1;
    }
    const retainedTurns = new Set(traceEvents.map(item => item.turn_id));
    for (const [id, turn] of traceTurns) {
      if (!retainedTurns.has(id)) {
        turn.option.remove();
        traceTurns.delete(id);
        if (traceFilter === id) { traceFilter = "all"; ui.traceSelect.value = "all"; }
      }
    }
    renderExecutionTrace();
    if (shouldFollow && !row.hidden) ui.inspectorBody.scrollTop = ui.inspectorBody.scrollHeight;
  }
  function resetExecutionTrace() {
    for (const view of traceViews.values()) view.row.remove();
    for (const turn of traceTurns.values()) turn.option.remove();
    traceEvents.length = 0;
    traceViews.clear();
    traceTurns.clear();
    traceTurnCount = 0;
    traceDiscarded = 0;
    traceFilter = "all";
    traceDisconnected = false;
    ui.traceSelect.value = "all";
    ui.traceNotice.textContent = "";
    renderExecutionTrace();
  }
  function disconnectExecutionTrace() {
    if (!traceEvents.length || traceDisconnected) return;
    traceDisconnected = true;
    ui.traceNotice.textContent = "连接已断开，保留断开前的追踪记录；重连后开始新会话。";
    renderExecutionTrace();
  }

  function taskElement(tag, className, text) {
    const element = document.createElement(tag);
    element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function taskTimestamp(value) { return Number.isFinite(value) && value > 0 ? value : null; }
  function taskDuration(start, end = Date.now()) {
    if (!start) return "";
    const seconds = Math.max(0, (end - start) / 1000);
    return seconds < 60 ? `${seconds.toFixed(seconds < 10 ? 1 : 0)} 秒`
      : `${Math.floor(seconds / 60)} 分 ${Math.floor(seconds % 60)} 秒`;
  }
  function taskPayload(value) {
    if (!value) return "（空）";
    try { return JSON.stringify(JSON.parse(value), null, 2); }
    catch (_) { return value; }
  }
  function updateTaskTimes() {
    for (const {data, view} of taskRecords.values()) {
      const created = new Date(data.created_at).toLocaleTimeString("zh-CN", {hour:"2-digit", minute:"2-digit", hour12:false});
      const elapsed = taskDuration(data.started_at || data.created_at, data.finished_at || Date.now());
      const label = data.status === "queued" ? "已等待" : data.finished_at ? "耗时" : "已执行";
      view.meta.textContent = `${created} · ${label} ${elapsed}`;
      for (const call of data.calls) {
        const callView = view.calls.get(call.id);
        if (callView) callView.time.textContent = taskDuration(call.started_at, call.finished_at || data.finished_at || Date.now());
      }
    }
  }
  function refreshTaskPanel() {
    let active = 0, returned = 0, failed = 0, visible = 0;
    for (const record of taskRecords.values()) {
      const isActive = activeTaskStatuses.has(record.data.status);
      if (isActive) active += 1;
      if (record.data.result) returned += 1;
      if (record.data.status === "failed") failed += 1;
      record.view.card.hidden = taskFilter === "active" && !isActive;
      if (!record.view.card.hidden) visible += 1;
    }
    ui.taskCount.textContent = String(taskRecords.size);
    ui.taskJump.hidden = !taskRecords.size;
    ui.taskJump.textContent = `任务 ${taskRecords.size}${active ? ` · ${active} 进行中` : " · 查看详情"}`;
    ui.taskSummary.textContent = taskRecords.size
      ? [`${active} 个进行中`, `${returned} 个结果已返回`, failed ? `${failed} 个失败` : ""].filter(Boolean).join(" · ")
      : "执行进度和返回结果将在这里同步";
    ui.taskEmpty.hidden = visible > 0;
    ui.taskEmptyTitle.textContent = taskRecords.size ? "暂无进行中的任务" : "等待任务开始";
    ui.taskEmptyDescription.textContent = taskRecords.size ? "可切换「全部」查看已返回的结果。"
      : "需要调用工具时，执行过程会自动显示在这里。";
    ui.taskAll.setAttribute("aria-pressed", String(taskFilter === "all"));
    ui.taskActive.setAttribute("aria-pressed", String(taskFilter === "active"));
    const ticking = [...taskRecords.values()].some(({data}) => ["queued", "running"].includes(data.status));
    if (ticking && !taskTimer) taskTimer = setInterval(updateTaskTimes, 1000);
    if (!ticking) { clearInterval(taskTimer); taskTimer = null; }
    updateTaskTimes();
    updateInspectorBadge();
  }
  function setTaskFilter(filter) {
    taskFilter = filter;
    refreshTaskPanel();
  }
  function resetTasks() {
    for (const record of taskRecords.values()) {
      clearTimeout(record.cancelTimer);
      record.view.card.remove();
    }
    taskRecords.clear();
    clearInterval(taskTimer);
    taskTimer = null;
    setTaskFilter("all");
  }
  function disconnectTasks() {
    for (const record of taskRecords.values()) {
      clearTimeout(record.cancelTimer);
      record.cancelPending = false;
      if (activeTaskStatuses.has(record.data.status)) {
        record.data = {...record.data, status:"disconnected", finished_at:record.data.finished_at || Date.now(),
          calls:record.data.calls.map(call => call.status === "running"
            ? {...call, status:"disconnected", finished_at:Date.now()} : call)};
      }
      renderTask(record);
    }
    refreshTaskPanel();
  }
  function createTaskView(record) {
    const card = taskElement("details", "task-card");
    card.dataset.taskId = record.data.id;
    card.open = true;
    const summary = taskElement("summary", "task-card-summary");
    const icon = taskElement("span", "task-status-icon");
    icon.setAttribute("aria-hidden", "true");
    const heading = taskElement("span", "task-heading");
    const title = taskElement("span", "task-title");
    const meta = taskElement("span", "task-meta");
    heading.append(title, meta);
    const status = taskElement("span", "task-status");
    summary.append(icon, heading, status);
    const body = taskElement("div", "task-body");
    const description = taskElement("p", "task-description");
    const steps = taskElement("ol", "task-steps");
    steps.setAttribute("aria-label", "任务执行阶段");
    ["已接收", "执行中", "返回结果"].forEach(label => steps.append(taskElement("li", "", label)));
    const callList = taskElement("ol", "task-call-list");
    callList.setAttribute("aria-label", "接口调用过程");
    const traceNote = taskElement("p", "task-note");
    const result = taskElement("section", "task-result");
    result.append(taskElement("h4", "task-result-heading", "返回结果"));
    const resultText = taskElement("pre", "task-result-text");
    result.append(resultText);
    const error = taskElement("p", "task-error");
    const note = taskElement("p", "task-note");
    const actions = taskElement("div", "task-actions");
    const copy = taskElement("button", "task-copy", "复制结果");
    copy.type = "button";
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(record.data.result);
        actionNote.textContent = "结果已复制";
      } catch (_) { actionNote.textContent = "无法自动复制，请选中返回结果后复制。"; }
    });
    const cancel = taskElement("button", "task-cancel", "取消任务");
    cancel.type = "button";
    cancel.addEventListener("click", () => {
      if (record.cancelPending || !activeTaskStatuses.has(record.data.status)) return;
      if (!send({type:"cancel_task", task_id:record.data.id})) {
        actionNote.textContent = "连接已断开，无法取消任务。";
        return;
      }
      record.cancelPending = true;
      actionNote.textContent = "正在请求取消…";
      renderTask(record);
      record.cancelTimer = setTimeout(() => {
        record.cancelPending = false;
        actionNote.textContent = "尚未收到取消确认，请稍后重试。";
        renderTask(record);
      }, 8000);
    });
    const actionNote = taskElement("span", "task-note");
    actionNote.setAttribute("role", "status");
    actions.append(copy, cancel, actionNote);
    body.append(description, steps, callList, traceNote, result, error, note, actions);
    card.append(summary, body);
    ui.taskList.prepend(card);
    return {card, icon, title, meta, status, description, steps, callList, traceNote,
      result, resultText, error, note, copy, cancel, actionNote, calls:new Map()};
  }
  function renderTaskCall(call, view) {
    let item = view.calls.get(call.id);
    if (!item) {
      const row = taskElement("li", "task-call");
      const details = taskElement("details", "");
      const summary = taskElement("summary", "task-call-summary");
      const name = taskElement("span", "task-call-name");
      const status = taskElement("span", "task-call-status");
      const time = taskElement("span", "task-call-time");
      summary.append(name, status, time);
      const body = taskElement("div", "task-call-body");
      const input = taskElement("pre", "task-payload");
      const output = taskElement("pre", "task-payload");
      const inputLabel = taskElement("div", "task-payload-label");
      const outputLabel = taskElement("div", "task-payload-label");
      body.append(inputLabel, input, outputLabel, output);
      details.append(summary, body);
      row.append(details);
      view.callList.append(row);
      item = {row, name, status, time, input, output, inputLabel, outputLabel};
      view.calls.set(call.id, item);
    }
    const names = {global_search:"联网搜索", get_current_time:"查询时间", write_todos:"规划任务", task:"子任务协作"};
    item.row.dataset.status = call.status;
    item.name.textContent = names[call.name] ? `${names[call.name]} · ${call.name}` : call.name;
    item.status.textContent = {running:"调用中", completed:"已返回", failed:"调用失败", canceled:"已取消", disconnected:"已断开"}[call.status];
    item.inputLabel.textContent = `请求参数${call.input_truncated ? "（内容已截取）" : ""}`;
    item.outputLabel.textContent = `接口返回${call.output_truncated ? "（内容已截取）" : ""}`;
    item.input.textContent = taskPayload(call.input);
    item.output.textContent = call.error || (call.status === "running" ? "等待接口返回…"
      : call.status === "canceled" ? "调用已取消" : call.status === "disconnected" ? "连接已断开，未收到返回" : taskPayload(call.output));
  }
  function renderTask(record) {
    const {data, view} = record;
    view.card.dataset.status = data.status;
    view.title.textContent = data.title;
    view.status.textContent = taskStatuses[data.status];
    view.icon.textContent = {queued:"·", running:"", ready:"✓", announcing:"↗", completed:"✓", canceled:"−", failed:"!", disconnected:"!"}[data.status];
    view.description.textContent = data.question;
    view.description.hidden = !data.question || data.question === data.title;
    const returned = Boolean(data.result) || ["ready", "announcing", "completed"].includes(data.status);
    const stopped = ["failed", "canceled", "disconnected"].includes(data.status);
    const current = data.status === "queued" || (!data.started_at && stopped) ? 0 : 1;
    [...view.steps.children].forEach((step, index) => {
      step.dataset.state = returned || index < current ? "done" : index === current ? (stopped ? "stopped" : "current") : "pending";
      if (step.dataset.state === "current") step.setAttribute("aria-current", "step");
      else step.removeAttribute("aria-current");
    });
    for (const call of data.calls) renderTaskCall(call, view);
    const ids = new Set(data.calls.map(call => call.id));
    for (const [id, callView] of view.calls) {
      if (!ids.has(id)) { callView.row.remove(); view.calls.delete(id); }
    }
    view.callList.hidden = !data.calls.length;
    view.traceNote.hidden = !data.calls_truncated && data.calls.length > 0;
    view.traceNote.textContent = data.calls_truncated ? "调用记录较多，仅展示前 40 条。" : "等待接口调用记录";
    if (stopped && !data.calls.length) view.traceNote.hidden = true;
    view.result.hidden = !data.result;
    if (view.resultText.textContent !== data.result) view.resultText.textContent = data.result;
    view.error.hidden = data.status !== "failed";
    view.error.textContent = data.error || "任务执行失败，请重新提交。";
    view.note.textContent = {
      queued:"任务正在等待执行，可以继续对话。", running:"任务正在执行，可以继续对话。",
      ready:"结果已返回，等待合适时机播报。", announcing:"结果已返回，正在语音播报。",
      completed:"任务已完成，可随时查看和复制结果。", canceled:"任务已取消，不会继续执行或播报。",
      failed:"可根据错误信息调整问题后重新提交。", disconnected:"连接已断开，当前记录为断开前的快照。重连后开启新会话。",
    }[data.status];
    view.cancel.hidden = !activeTaskStatuses.has(data.status);
    view.cancel.disabled = !connected || record.cancelPending;
    view.cancel.textContent = record.cancelPending ? "正在取消…" : "取消任务";
    view.copy.hidden = !data.result;
  }
  function receiveTaskUpdate(task) {
    if (!task || typeof task !== "object" || typeof task.id !== "string" || !task.id || task.id.length > 128
      || !Object.hasOwn(taskStatuses, task.status) || task.status === "disconnected") return;
    let record = taskRecords.get(task.id);
    const previous = record?.data;
    const updated = taskTimestamp(task.updated_at);
    if (updated && previous?.updated_at && updated < previous.updated_at) return;
    if (previous?.status === "canceled" && task.status !== "canceled") return;
    const text = key => typeof task[key] === "string" ? task[key] : previous?.[key] || "";
    const calls = Array.isArray(task.calls) ? task.calls.slice(0, 40).filter(call => call && typeof call.id === "string"
      && typeof call.name === "string" && ["running", "completed", "failed", "canceled"].includes(call.status)).map(call => ({
      id:call.id, name:call.name, status:call.status, started_at:taskTimestamp(call.started_at),
      finished_at:taskTimestamp(call.finished_at), input:typeof call.input === "string" ? call.input : "",
      output:typeof call.output === "string" ? call.output : "", error:typeof call.error === "string" ? call.error : "",
      input_truncated:Boolean(call.input_truncated), output_truncated:Boolean(call.output_truncated),
    })) : previous?.calls || [];
    const stopped = ["completed", "canceled", "failed"].includes(task.status);
    const data = {
      id:task.id, title:text("title") || "后台任务", question:text("question"), result:text("result"), error:text("error"),
      status:task.status, created_at:taskTimestamp(task.created_at) || previous?.created_at || Date.now(),
      started_at:taskTimestamp(task.started_at) || previous?.started_at || (task.status === "running" ? Date.now() : null),
      finished_at:taskTimestamp(task.finished_at) || previous?.finished_at || (stopped || task.status === "ready" ? Date.now() : null),
      updated_at:updated || previous?.updated_at, calls, calls_truncated:Boolean(task.calls_truncated),
    };
    if (!record) {
      record = {data, view:null, cancelPending:false, cancelTimer:null};
      record.view = createTaskView(record);
      taskRecords.set(task.id, record);
    } else record.data = data;
    if (!activeTaskStatuses.has(data.status)) {
      clearTimeout(record.cancelTimer);
      record.cancelPending = false;
      record.view.actionNote.textContent = "";
    }
    renderTask(record);
    // Mirror the server's bounded terminal history, without evicting live tasks.
    const terminal = [...taskRecords.values()].filter(item => !activeTaskStatuses.has(item.data.status));
    while (terminal.length > 30) {
      const old = terminal.shift();
      clearTimeout(old.cancelTimer);
      old.view.card.remove();
      taskRecords.delete(old.data.id);
    }
    refreshTaskPanel();
  }

  function showError(message) {
    clearAgentProgress();
    ui.errorText.textContent = message;
    ui.error.hidden = false;
  }
  function hideError() { ui.error.hidden = true; }
  function scrollMessages() { ui.messages.scrollTop = ui.messages.scrollHeight; }
  function setConnection(label, state) {
    ui.connection.textContent = label;
    ui.connectionDot.className = `status-dot ${state || ""}`;
    updateControls();
  }
  function updateControls() {
    ui.send.disabled = !connected || !ui.input.value.trim() || captureStarting || captureStopping;
    ui.mic.disabled = !connected || captureStarting || captureStopping;
    ui.mic.classList.toggle("is-recording", listening);
    ui.mic.setAttribute("aria-pressed", String(listening));
    ui.micLabel.textContent = captureStarting ? "准备麦克风…" : captureStopping ? "挂断中…" : listening ? "挂断" : "拨号";
    ui.mic.setAttribute("aria-label", listening ? "挂断语音输入" : "拨号，开启语音输入");
    ui.mute.hidden = !listening;
    ui.mute.disabled = !connected || !listening || captureStarting || captureStopping;
    ui.mute.classList.toggle("muted", microphoneMuted);
    ui.mute.setAttribute("aria-pressed", String(microphoneMuted));
    ui.mute.setAttribute("aria-label", microphoneMuted ? "取消麦克风静音" : "关闭麦克风");
    ui.muteLabel.textContent = microphoneMuted ? "取消静音" : "静音";
    ui.micStatus.hidden = !listening || !microphoneMuted;
    ui.callStatus.hidden = !listening;
    ui.interrupt.hidden = !busy && !playbackActive && !agentProgress;
    ui.recordingBar.hidden = !listening || microphoneMuted;
    ui.recordingBar.dataset.speaking = String(userSpeaking);
    ui.recordingBar.dataset.muted = String(microphoneMuted);
  }
  function toggleMicrophone() {
    if (!connected || !listening || !capture || captureStarting || captureStopping) return;
    microphoneMuted = !microphoneMuted;
    capture.stream.getTracks().forEach(track => { track.enabled = !microphoneMuted; });
    // Keep receiving the current reply and running tasks while the input is silent.
    if (microphoneMuted) resumePlayback();
    updateControls();
  }
  function setStage(stage) {
    if (listening) ui.recordingCaption.textContent = ({
      asr:"正在聆听，说完后自动回复", agent:"正在思考…", tts:"正在回复…",
    })[stage] || "语音已开启";
    for (const name of Object.keys(stageDefaults)) {
      $(`stage-${name}`).dataset.active = String(name === stage);
      $(`status-${name}`).textContent = name === stage ? ({asr:"正在聆听与识别…",agent:"正在思考并生成回答…",tts:"正在合成与播放语音…"})[name] : stageDefaults[name];
    }
  }
  function updateProviderInfo(message) {
    const provider = message.voice_provider || message.realtime_provider || message.speech_provider;
    const name = {seeduplex:"原生语音",cascade:"三段式语音"}[provider];
    if (!name) return;
    ui.voiceProvider.textContent = `语音配置：${name}`;
    ui.voiceProvider.hidden = false;
    const native = provider === "seeduplex";
    document.querySelector(".pipeline-description").textContent = native
      ? "原生语音负责续保沟通，车险工具负责客户查询、报价与办理。"
      : "听见停顿，也理解你的话是否说完。";
    $("stage-asr").querySelector("p").textContent = native ? "Seeduplex 连续音频理解" : "豆包流式语音识别";
    $("stage-agent").querySelector("p").textContent = native ? "Seeduplex / Deep Agents" : "LangChain Deepagents";
    $("stage-tts").querySelector("p").textContent = native ? "Seeduplex 原生语音输出" : "豆包双向流式语音合成";
    const connectors = document.querySelectorAll(".pipeline-connector");
    connectors[0].lastElementChild.textContent = native ? "音频 → 理解" : "语音 → 文字";
    connectors[1].lastElementChild.textContent = native ? "直接生成语音" : "文字 → 回应";
  }
  function routeInfo(message) {
    if (message.route === "deep") {
      const reasons = {
        repeated_question:"同一问题再次询问", dissatisfied_followup:"前答未解决疑问",
        topic_followup:"同主题继续追问", explicit_depth:"明确要求深入回答",
        comparison:"比较与权衡", planning:"规划或多步推理", multiple_constraints:"多项约束",
        professional_advice:"需要核实的专业建议", realtime_information:"需要真实时间工具",
        complex_task:"需要完成较复杂的任务", task_continuation:"条件已补齐，继续执行任务",
        tool_required:"需要工具或外部信息", uncertainty:"需要进一步核实或推理",
        native_delegation:"语音模型调用任务工具",
      };
      const source = {semantic:"语义路由",fallback:"路由超时或不可用，规则兜底",local:"本地规则",native:"原生语义委派",native_recovery:"原生协作接续"}[message.source];
      const nativeSpeech = message.speech_provider === "seeduplex";
      const label = nativeSpeech ? "Deep Agents 协作" : "Deep Agents";
      const recoveryReasons = {
        native_result_timeout:"重新接续语音播报",
      };
      const speech = nativeSpeech ? "由语音模型接续播报" : "";
      const input = activeInput === "text" ? "键盘提问" : "";
      return {kind:"deep", label, reason:[input, source, reasons[message.reason] || recoveryReasons[message.reason] || "根据问题语义转入深入处理", speech].filter(Boolean).join("：")};
    }
    if (message.provider === "seeduplex" || message.route === "seeduplex") {
      const reason = message.reason === "clarification" ? "补充任务的关键条件" : "直接语音回答";
      const source = {semantic:"语义路由",fallback:"路由超时或不可用，规则兜底",local:"本地规则"}[message.source];
      return {kind:"speech", label:"语音快模型", reason:[source, reason].filter(Boolean).join("：")};
    }
    if (message.route === "fast") return {kind:"fast", label:activeInput === "text" ? "Mini · 键盘提问" : "Mini · 快模型"};
    return null;
  }
  function showRoute(route) {
    ui.modelRoute.dataset.route = route?.kind || "waiting";
    ui.modelRoute.textContent = route?.label || "等待本轮选择模型";
    ui.modelRoute.title = route?.reason || "";
    if (!assistantMessage || !route) return;
    let badge = assistantMessage.label.querySelector(".message-route");
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "message-route";
      assistantMessage.label.append(badge);
    }
    badge.dataset.route = route.kind;
    badge.textContent = route.label;
    badge.title = route.reason || "";
  }
  function send(message) {
    if (!connected || !socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(message));
    return true;
  }
  function addMessage(role, text = "", partial = false) {
    ui.welcome.hidden = true;
    const row = document.createElement("div");
    row.className = `message ${role}${partial ? " partial" : ""}`;
    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "你" : "✦";
    avatar.setAttribute("aria-hidden", "true");
    const body = document.createElement("div");
    body.className = "message-body";
    const label = document.createElement("div");
    label.className = "message-label";
    label.append(document.createTextNode(role === "user" ? "你" : "车险坐席"));
    const time = document.createElement("span");
    time.className = "message-time";
    time.textContent = new Date().toLocaleTimeString("zh-CN", {hour:"2-digit",minute:"2-digit",hour12:false});
    label.append(time);
    const content = document.createElement("div");
    content.className = "message-text";
    content.textContent = text;
    body.append(label, content);
    row.append(avatar, body);
    ui.messages.append(row);
    scrollMessages();
    return {row, content, label};
  }
  function newId() { return window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`; }
  function beginTurn(turnId = newId(), input = "voice") {
    if (input !== "outbound") outboundOpeningPending = false;
    clearAgentProgress();
    activeTurnId = turnId;
    userMessage = null;
    assistantMessage = null;
    assistantText = "";
    activeRoute = null;
    activeInput = input;
    showRoute(null);
  }
  function ensureAssistant() {
    if (!assistantMessage) {
      assistantMessage = addMessage("assistant");
      const dots = document.createElement("span");
      dots.className = "typing-indicator";
      dots.setAttribute("aria-label", "正在思考");
      for (let i = 0; i < 3; i++) dots.append(document.createElement("i"));
      assistantMessage.content.append(dots);
      if (activeRoute) showRoute(activeRoute);
    }
  }
  function countTurn() {
    turnCount += 1;
    ui.counter.textContent = `${turnCount} 轮对话`;
  }
  function clearPendingAssistant() {
    clearAgentProgress();
    if (assistantMessage && !assistantText) assistantMessage.row.remove();
  }

  function clearAgentProgress() {
    clearInterval(agentProgressTimer);
    agentProgressTimer = null;
    agentProgress?.row.remove();
    agentProgress = null;
  }
  function showAgentProgress(message) {
    if (!activeTurnId || !["working", "waiting_playback", "answering", "completed"].includes(message.phase)) return;
    if (message.phase === "completed") { clearAgentProgress(); return; }
    ensureAssistant();
    if (!agentProgress || agentProgress.turnId !== activeTurnId) {
      clearAgentProgress();
      const row = document.createElement("div");
      row.className = "assistant-progress";
      const label = document.createElement("span");
      label.className = "assistant-progress-label";
      label.setAttribute("role", "status");
      const elapsed = document.createElement("span");
      elapsed.className = "assistant-progress-elapsed";
      elapsed.setAttribute("aria-hidden", "true");
      row.append(label, elapsed);
      assistantMessage.content.parentElement.append(row);
      agentProgress = {row, label, elapsed, turnId:activeTurnId, started:performance.now()};
    }
    const progress = agentProgress;
    progress.row.dataset.phase = message.phase;
    progress.label.textContent = message.message || (message.phase === "working"
      ? "正在深入处理，可以随时补充或打断" : "结果已返回，正在组织语音");
    clearInterval(agentProgressTimer);
    agentProgressTimer = null;
    if (message.phase === "working") {
      const updateElapsed = () => {
        if (agentProgress !== progress || activeTurnId !== progress.turnId) { clearAgentProgress(); return; }
        progress.elapsed.textContent = `已处理 ${Math.floor((performance.now() - progress.started) / 1000)} 秒`;
      };
      updateElapsed();
      agentProgressTimer = setInterval(updateElapsed, 1000);
    } else progress.elapsed.textContent = "";
    busy = true;
    updateControls();
    scrollMessages();
  }

  async function ensureOutputAudio() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) throw new Error("当前浏览器不支持音频播放，请使用最新版 Chrome、Edge 或 Safari。");
    if (!outputContext || outputContext.state === "closed") {
      try { outputContext = new AudioContext({sampleRate: 24000}); }
      catch (_) { outputContext = new AudioContext(); }
      nextAudioTime = 0;
    }
    if (outputContext.state === "suspended" && !playbackPaused) await outputContext.resume();
    return outputContext;
  }
  function reportPlayback(playing, turnId = playbackReportTurn) {
    if (playing) {
      if (playbackReportTurn === turnId) return;
      playbackReportTurn = turnId;
    } else {
      if (!playbackReportTurn) return;
      turnId = playbackReportTurn;
      playbackReportTurn = null;
    }
    if (turnId) send({type:"playback", playing, turn_id:turnId});
  }
  function pausePlayback() {
    playbackPaused = true;
    ui.recordingCaption.textContent = "听到你在说话，正在确认…";
    if (outputContext?.state === "running") {
      void outputContext.suspend().then(() => {
        if (!playbackPaused && outputContext?.state === "suspended") return outputContext.resume();
      }).catch(() => {});
    }
  }
  function resumePlayback() {
    playbackPaused = false;
    if (outputContext?.state === "suspended") void outputContext.resume().catch(() => {});
  }
  function settlePlaybackBarriers() {
    for (const [barrierId, barrier] of playbackBarriers) {
      if (barrier.turnId !== activeTurnId) { playbackBarriers.delete(barrierId); continue; }
      for (const source of barrier.sources) {
        if (!audioSources.has(source)) barrier.sources.delete(source);
      }
      if (!barrier.sources.size) {
        playbackBarriers.delete(barrierId);
        send({type:"playback_drained", turn_id:barrier.turnId, barrier_id:barrierId});
      }
    }
  }
  function stopPlayback() {
    // Discarding audio on interruption must never acknowledge it as fully heard.
    playbackBarriers.clear();
    reportPlayback(false);
    audioTurnId = null;
    for (const source of audioSources) {
      source.onended = null;
      try { source.stop(); } catch (_) { /* The source may have already ended. */ }
      source.disconnect();
    }
    audioSources.clear();
    nextAudioTime = 0;
    audioByteRemainder = null;
    playbackActive = false;
    resumePlayback();
    updateControls();
  }
  function playAudio(bytes) {
    if (!outputContext || (outputContext.state !== "running" && !playbackPaused)) {
      showError("浏览器暂停了语音播放。点击麦克风或发送消息后可恢复声音。");
      return;
    }
    let data = new Uint8Array(bytes);
    if (audioByteRemainder !== null) {
      const joined = new Uint8Array(data.length + 1);
      joined[0] = audioByteRemainder;
      joined.set(data, 1);
      data = joined;
      audioByteRemainder = null;
    }
    if (data.length % 2) {
      audioByteRemainder = data[data.length - 1];
      data = data.subarray(0, data.length - 1);
    }
    if (!data.length) return;
    const buffer = outputContext.createBuffer(1, data.byteLength / 2, outputSampleRate);
    const samples = buffer.getChannelData(0);
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;
    const source = outputContext.createBufferSource();
    source.buffer = buffer;
    source.connect(outputContext.destination);
    audioSources.add(source);
    playbackActive = true;
    reportPlayback(true, audioTurnId || activeTurnId);
    source.onended = () => {
      audioSources.delete(source);
      source.disconnect();
      settlePlaybackBarriers();
      if (!audioSources.size) {
        playbackActive = false;
        reportPlayback(false);
        if (!busy) setStage(listening ? "asr" : "idle");
        updateControls();
      }
    };
    nextAudioTime = Math.max(outputContext.currentTime + 0.035, nextAudioTime);
    source.start(nextAudioTime);
    nextAudioTime += buffer.duration;
    updateControls();
  }

  const captureProcessor = `
    class PcmCapture extends AudioWorkletProcessor {
      constructor() {
        super();
        this.ratio = sampleRate / 16000;
        this.remaining = this.ratio;
        this.sum = 0;
        this.samples = [];
        this.active = true;
        this.port.onmessage = (event) => {
          if (event.data === 'flush') {
            this.active = false;
            if (this.samples.length) this.emit();
            this.port.postMessage({type: 'flushed'});
          }
        };
      }
      emit() {
        const buffer = new ArrayBuffer(this.samples.length * 2);
        const view = new DataView(buffer);
        for (let i = 0; i < this.samples.length; i++) {
          const sample = Math.max(-1, Math.min(1, this.samples[i]));
          view.setInt16(i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
        }
        this.samples = [];
        this.port.postMessage({type: 'audio', buffer}, [buffer]);
      }
      process(inputs) {
        const channels = inputs[0];
        if (!this.active || !channels || !channels.length) return true;
        const length = channels[0].length;
        for (let i = 0; i < length; i++) {
          let sample = 0;
          for (let c = 0; c < channels.length; c++) sample += channels[c][i];
          sample /= channels.length;
          let width = 1;
          while (width > 1e-9) {
            const weight = Math.min(width, this.remaining);
            this.sum += sample * weight;
            this.remaining -= weight;
            width -= weight;
            if (this.remaining < 1e-9) {
              this.samples.push(this.sum / this.ratio);
              this.sum = 0;
              this.remaining = this.ratio;
              if (this.samples.length === 320) this.emit();
            }
          }
        }
        return true;
      }
    }
    registerProcessor('pcm-capture', PcmCapture);
  `;

  async function closeCapture(flush = false) {
    const current = capture;
    capture = null;
    outboundOpeningPending = false;
    microphoneMuted = false;
    clearInterval(recordTimer);
    if (!current) return;
    current.source.disconnect();
    current.stream.getTracks().forEach((track) => track.stop());
    if (flush) {
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, 250);
        current.flushed = () => { clearTimeout(timer); resolve(); };
        current.node.port.postMessage("flush");
      });
    }
    current.node.disconnect();
    current.silent.disconnect();
    current.node.port.onmessage = null;
    current.node.port.close();
    if (current.context.state !== "closed") await current.context.close();
    URL.revokeObjectURL(current.moduleUrl);
  }

  async function startListening() {
    if (!connected || captureStarting || captureStopping || listening) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      showError("麦克风需要安全环境。请使用 http://localhost 或 HTTPS 访问此工作台。");
      return;
    }
    const generation = ++captureGeneration;
    microphoneMuted = false;
    captureStarting = true;
    updateControls();
    hideError();
    let stream = null;
    let context = null;
    let moduleUrl = null;
    try {
      await ensureOutputAudio();
      stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount:1, echoCancellation:true, noiseSuppression:true, autoGainControl:true}});
      if (generation !== captureGeneration || !connected) { stream.getTracks().forEach((track) => track.stop()); return; }
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      context = new AudioContext();
      if (!context.audioWorklet) throw new Error("当前浏览器不支持 AudioWorklet，请升级浏览器后重试。");
      moduleUrl = URL.createObjectURL(new Blob([captureProcessor], {type:"application/javascript"}));
      await context.audioWorklet.addModule(moduleUrl);
      await context.resume();
      if (generation !== captureGeneration || !connected) {
        stream.getTracks().forEach((track) => track.stop());
        await context.close();
        URL.revokeObjectURL(moduleUrl);
        return;
      }
      listenerId = newId();
      const source = context.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(context, "pcm-capture");
      const silent = context.createGain();
      silent.gain.value = 0;
      const current = {stream, context, source, node, silent, moduleUrl, flushed:null};
      node.port.onmessage = (event) => {
        if (event.data.type === "audio" && socket?.readyState === WebSocket.OPEN && connected && (listening && capture === current)) {
          if (socket.bufferedAmount > 1024 * 1024) {
            showError("网络发送较慢，已关闭语音，请连接稳定后重新开启。");
            void stopListening();
            return;
          }
          // Silence even already queued microphone samples; preserve stream cadence.
          socket.send(microphoneMuted ? new ArrayBuffer(event.data.buffer.byteLength) : event.data.buffer);
        } else if (event.data.type === "flushed") current.flushed?.();
      };
      capture = current;
      listening = true;
      outboundOpeningPending = !outboundOpeningSeen;
      if (!send({type:"listen",turn_id:listenerId,playback_barriers:true,outbound:true})) outboundOpeningPending = false;
      recordStarted = Date.now();
      ui.recordingTime.textContent = "00:00";
      ui.recordingCaption.textContent = "语音已开启，停顿后自动回复";
      source.connect(node);
      node.connect(silent);
      silent.connect(context.destination);
      if (!busy && !playbackActive) setStage("asr");
      recordTimer = setInterval(() => {
        const seconds = Math.floor((Date.now() - recordStarted) / 1000);
        ui.recordingTime.textContent = `${String(Math.floor(seconds / 60)).padStart(2,"0")}:${String(seconds % 60).padStart(2,"0")}`;
      }, 250);
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      if (context && context.state !== "closed") await context.close().catch(() => {});
      if (moduleUrl) URL.revokeObjectURL(moduleUrl);
      const messages = {NotAllowedError:"无法使用麦克风，请在浏览器地址栏中允许麦克风权限。",NotFoundError:"未找到麦克风，请连接设备后重试。",NotReadableError:"无法打开麦克风，设备可能正在被其他应用使用。"};
      showError(messages[error.name] || error.message || "麦克风初始化失败，请重试。");
    } finally {
      captureStarting = false;
      updateControls();
    }
  }

  async function stopListening() {
    if ((!listening && !captureStarting) || captureStopping) return;
    captureGeneration += 1;
    captureStopping = true;
    listening = false;
    microphoneMuted = false;
    userSpeaking = false;
    listenerId = null;
    send({type:"unlisten"});
    resumePlayback();
    updateControls();
    try { await closeCapture(false); }
    catch (error) { showError(error.message || "关闭麦克风时出现问题，请重试。"); }
    finally {
      captureStopping = false;
      if (!busy && !playbackActive) setStage("idle");
      updateControls();
    }
  }

  function interruptResponse(sendInterrupt = true) {
    outboundOpeningPending = false;
    if (sendInterrupt) send({type:"interrupt"});
    activeTurnId = null;
    userSpeaking = false;
    if (assistantMessage && assistantText && busy) assistantMessage.row.classList.add("interrupted");
    clearPendingAssistant();
    busy = false;
    stopPlayback();
    if (listening) ui.recordingCaption.textContent = "当前播报已停止，继续说话即可";
    if (!busy) setStage(listening ? "asr" : "idle");
    updateControls();
  }

  async function submitText(text) {
    text = text.trim();
    if (!text || !connected || captureStarting || captureStopping) return;
    try { await ensureOutputAudio(); }
    catch (error) { showError(error.message); }
    if (!connected || captureStarting || captureStopping) return;
    interruptResponse(false);
    hideError();
    beginTurn(undefined, "text");
    userMessage = addMessage("user", text);
    countTurn();
    ensureAssistant();
    if (send({type:"text",text,turn_id:activeTurnId})) {
      busy = true;
      setStage("agent");
      ui.input.value = "";
      resizeInput();
    }
    updateControls();
  }

  function handleMessage(event) {
    if (event.data instanceof ArrayBuffer) {
      if (activeTurnId && audioTurnId === activeTurnId) playAudio(event.data);
      return;
    }
    let message;
    try { message = JSON.parse(event.data); }
    catch (_) { showError("收到无法解析的服务端消息，请重新连接。"); return; }
    if (!message || typeof message !== "object") return;
    if (message.type === "execution_event") { receiveExecutionEvent(message.event); return; }
    // Session events must be processed before the current-turn filter. A speech
    // candidate keeps the old route until ASR confirms an actual interruption.
    // Tasks belong to the connection, so switching conversation turns must not hide their updates.
    if (message.type === "task_update") { receiveTaskUpdate(message.task); return; }
    if (message.type === "outbound_opening") {
      if (!connected || !listening || !outboundOpeningPending || outboundOpeningSeen
        || typeof message.turn_id !== "string" || !message.turn_id || message.turn_id.length > 80) return;
      // A dial starts an assistant-only sales opening. User input, hangup or
      // interruption retires this pending request before a late event can win.
      outboundOpeningPending = false;
      outboundOpeningSeen = true;
      beginTurn(message.turn_id, "outbound");
      busy = true;
      setStage("agent");
      ui.recordingCaption.textContent = "正在准备本轮车险沟通…";
      updateControls();
      return;
    }
    if (message.type === "task_announcement") {
      if (typeof message.turn_id !== "string" || !message.turn_id
        || typeof message.task_id !== "string" || !message.task_id) return;
      // The server schedules this only after user speech and queued audio drain.
      // It is a new assistant notification, never a synthetic user question.
      beginTurn(message.turn_id, "task");
      busy = true;
      activeRoute = {kind:"deep", label:"Deep Agents 协作", reason:"后台任务结果"};
      ensureAssistant();
      showRoute(activeRoute);
      setStage("agent");
      updateControls();
      return;
    }
    if (message.type === "listening") {
      updateProviderInfo(message);
      if (message.enabled && listening && capture) {
        ui.recordingCaption.textContent = "语音已开启，停顿后自动回复";
      } else if (!message.enabled && (listening || captureStarting)) {
        captureGeneration += 1;
        listening = false;
        userSpeaking = false;
        listenerId = null;
        resumePlayback();
        void closeCapture(false).catch(() => {});
      }
      updateControls();
      return;
    }
    if (message.type === "barge_in_candidate") {
      if (listening) pausePlayback();
      return;
    }
    if (message.type === "barge_in_rejected") {
      resumePlayback();
      ui.recordingCaption.textContent = "继续说话即可，支持直接插话";
      return;
    }
    if (message.type === "barge_in_confirmed") {
      outboundOpeningPending = false;
      if (assistantMessage && assistantText && (busy || playbackActive)) {
        assistantMessage.row.classList.add("interrupted");
      }
      clearPendingAssistant();
      stopPlayback();
      activeTurnId = null;
      busy = false;
      setStage(listening ? "asr" : "idle");
      ui.recordingCaption.textContent = "当前播报已停止，继续说话即可";
      updateControls();
      return;
    }
    if (message.type === "speech_start") {
      if (!listening || typeof message.turn_id !== "string") return;
      stopPlayback();
      clearPendingAssistant();
      beginTurn(message.turn_id);
      busy = false;
      userSpeaking = true;
      setStage("asr");
      ui.recordingCaption.textContent = "正在听你说，停顿后自动提交…";
      updateControls();
      return;
    }
    if (message.turn_id !== undefined && message.turn_id !== activeTurnId) return;
    switch (message.type) {
      case "ready":
        outboundOpeningPending = false;
        outboundOpeningSeen = false;
        resetTasks();
        resetExecutionTrace();
        connected = true;
        connecting = false;
        reconnectDelay = 1000;
        outputSampleRate = message.output_sample_rate || 24000;
        updateProviderInfo(message);
        ui.authForm.hidden = true;
        setConnection("服务已连接", "connected");
        break;
      case "agent_route": {
        if (!activeTurnId) break;
        const route = routeInfo(message);
        if (!route) break;
        activeRoute = route;
        ensureAssistant();
        showRoute(route);
        break;
      }
      case "agent_recovery":
        if (!activeTurnId) break;
        activeRoute = routeInfo({...message, route:"deep", speech_provider:"seeduplex"});
        ensureAssistant();
        showRoute(activeRoute);
        break;
      case "agent_progress":
        if (message.turn_id === activeTurnId) showAgentProgress(message);
        break;
      case "state":
        if (message.stage === "agent" || message.stage === "idle") userSpeaking = false;
        setStage(message.stage);
        if (message.stage === "agent") { busy = true; ensureAssistant(); }
        if (message.stage === "tts") busy = true;
        if (message.stage === "idle") {
          clearAgentProgress();
          busy = false;
          if (playbackActive) setStage("tts");
          else if (listening) {
            setStage("asr");
            ui.recordingCaption.textContent = "继续说话即可，支持直接插话";
          }
        }
        updateControls();
        break;
      case "speech_end":
        userSpeaking = false;
        busy = true;
        ui.recordingCaption.textContent = "已自动识别停顿，正在准备回答…";
        updateControls();
        break;
      case "transcript":
        if (!message.text && !userMessage) break;
        if (!userMessage) userMessage = addMessage("user", message.text || "", !message.is_final);
        else userMessage.content.textContent = message.text || "";
        userMessage.row.classList.toggle("partial", !message.is_final);
        ui.recordingCaption.textContent = message.is_final ? "已自动提交，正在准备回答…" : message.text ? "正在听你说，停顿后自动提交…" : "正在聆听…";
        if (message.is_final && !userMessage.row.dataset.counted) {
          userMessage.row.dataset.counted = "true";
          countTurn();
        }
        scrollMessages();
        break;
      case "assistant_delta":
        ensureAssistant();
        assistantText = message.replace ? (message.text || "") : assistantText + (message.text || "");
        assistantMessage.content.textContent = assistantText;
        scrollMessages();
        break;
      case "assistant_done":
        ensureAssistant();
        assistantText = message.text ?? assistantText;
        assistantMessage.content.textContent = assistantText;
        scrollMessages();
        break;
      case "audio_start":
        if (message.format && message.format !== "pcm") { showError("服务端音频格式不兼容，需要 PCM16。"); break; }
        audioTurnId = message.turn_id ?? activeTurnId;
        outputSampleRate = message.sample_rate || 24000;
        audioByteRemainder = null;
        setStage("tts");
        break;
      case "audio_end":
        audioTurnId = null;
        busy = false;
        if (!playbackActive && !busy) setStage(listening ? "asr" : "idle");
        updateControls();
        break;
      case "playback_barrier":
        if (typeof message.barrier_id !== "string" || !activeTurnId) break;
        playbackBarriers.set(message.barrier_id, {
          turnId:activeTurnId, sources:new Set(audioSources),
        });
        settlePlaybackBarriers();
        break;
      case "interrupted":
        // Playback is also stopped immediately on the client, before this acknowledgment arrives.
        stopPlayback();
        busy = false;
        clearPendingAssistant();
        if (!busy) setStage(listening ? "asr" : "idle");
        updateControls();
        break;
      case "error":
        showError(message.message || "服务暂时无法完成请求，请重试。");
        userSpeaking = false;
        busy = message.stage === "tts" && busy;
        if (!busy) clearPendingAssistant();
        stopPlayback();
        if (["auth", "config", "protocol"].includes(message.stage) && (listening || captureStarting)) {
          void stopListening();
        }
        if (message.stage === "auth") { authRequired = true; ui.authForm.hidden = false; }
        setStage(busy ? "agent" : listening ? "asr" : "idle");
        updateControls();
        break;
      case "pong": break;
      default: break;
    }
  }

  function disconnectCurrentSocket() {
    clearAgentProgress();
    disconnectTasks();
    disconnectExecutionTrace();
    if (!socket) return;
    socket.onopen = socket.onmessage = socket.onerror = socket.onclose = null;
    socket.close();
    socket = null;
    connected = false;
    connecting = false;
    activeTurnId = null;
  }
  function connect() {
    if (leaving || connecting || connected) return;
    clearTimeout(reconnectTimer);
    if (authRequired && !authToken) {
      ui.authForm.hidden = false;
      setConnection("等待访问令牌", "disconnected");
      return;
    }
    disconnectCurrentSocket();
    connecting = true;
    setConnection("正在连接", "");
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const current = new WebSocket(`${protocol}//${window.location.host}/ws/voice`);
    socket = current;
    current.binaryType = "arraybuffer";
    const readyTimeout = setTimeout(() => {
      if (!connected && current === socket) {
        showError("连接服务超时，请检查后端服务是否已启动。");
        current.close();
      }
    }, 15000);
    current.onopen = () => {
      if (authRequired && authToken) current.send(JSON.stringify({type:"auth", token:authToken}));
    };
    current.onmessage = (event) => {
      if (current !== socket) return;
      if (typeof event.data === "string") {
        try { if (JSON.parse(event.data).type === "ready") clearTimeout(readyTimeout); } catch (_) { /* Handled below. */ }
      }
      handleMessage(event);
    };
    current.onerror = () => {
      if (current === socket) { clearAgentProgress(); setConnection("连接异常", "disconnected"); }
    };
    current.onclose = (event) => {
      clearTimeout(readyTimeout);
      if (current !== socket) return;
      const wasListening = listening || captureStarting;
      captureGeneration += 1;
      listening = false;
      userSpeaking = false;
      connected = false;
      connecting = false;
      busy = false;
      disconnectTasks();
      disconnectExecutionTrace();
      clearPendingAssistant();
      void closeCapture(false).catch(() => {});
      stopPlayback();
      setStage("idle");
      if (event.code === 1008 || event.code === 4401 || event.code === 4403) {
        setConnection("连接被拒绝", "disconnected");
        if (authRequired) {
          authToken = "";
          ui.authForm.hidden = false;
          showError("访问令牌无效或已过期，请重新输入。");
        } else showError("服务拒绝了连接，请检查服务端访问来源和认证配置。");
      } else if (!leaving) {
        setConnection("连接断开，重连中", "disconnected");
        if (wasListening) showError("连接已断开，麦克风已关闭。重连后可再次开启语音。");
        reconnectTimer = setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 10000);
      }
      updateControls();
    };
  }

  function resizeInput() {
    ui.input.style.height = "auto";
    ui.input.style.height = `${Math.min(ui.input.scrollHeight, 120)}px`;
    updateControls();
  }
  ui.input.addEventListener("input", resizeInput);
  ui.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      void submitText(ui.input.value);
    }
  });
  $("message-form").addEventListener("submit", (event) => { event.preventDefault(); void submitText(ui.input.value); });
  ui.mic.addEventListener("click", () => { void (listening ? stopListening() : startListening()); });
  ui.mute.addEventListener("click", toggleMicrophone);
  ui.interrupt.addEventListener("click", () => interruptResponse());
  ui.taskAll.addEventListener("click", () => setTaskFilter("all"));
  ui.taskActive.addEventListener("click", () => setTaskFilter("active"));
  ui.taskJump.addEventListener("click", () => {
    openInspector("tasks");
  });
  ui.inspectorToggle.addEventListener("click", () => inspectorOpen ? closeInspector() : openInspector());
  ui.inspectorClose.addEventListener("click", closeInspector);
  ui.inspectorBackdrop.addEventListener("click", closeInspector);
  for (const [index, view] of inspectorViews.entries()) {
    const tab = $(`inspector-tab-${view}`);
    tab.addEventListener("click", () => selectInspectorView(view));
    tab.addEventListener("keydown", event => {
      const next = {ArrowRight:(index + 1) % 3, ArrowLeft:(index + 2) % 3, Home:0, End:2}[event.key];
      if (next === undefined) return;
      event.preventDefault();
      selectInspectorView(inspectorViews[next]);
    });
  }
  window.addEventListener("keydown", handleInspectorKeydown);
  inspectorMedia.addEventListener?.("change", syncInspectorLayout);
  ui.traceSelect.addEventListener("change", () => {
    traceFilter = ui.traceSelect.value;
    renderExecutionTrace();
  });
  ui.traceToggle.addEventListener("click", () => {
    ui.traceContent.hidden = !ui.traceContent.hidden;
    ui.traceToggle.setAttribute("aria-expanded", String(!ui.traceContent.hidden));
    ui.traceToggle.textContent = ui.traceContent.hidden ? "展开时间线" : "折叠时间线";
  });
  ui.traceCopy.addEventListener("click", async () => {
    const events = visibleTraceEvents();
    if (!events.length) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify({version:1, clock:"server_unix_ms",
        connection:traceDisconnected ? "disconnected" : "connected", filter:traceFilter,
        discarded_events:traceDiscarded, events}, null, 2));
      ui.traceNotice.textContent = `已复制 ${events.length} 条调试记录（包含时间、切换原因和关联 ID）。`;
    } catch (_) { ui.traceNotice.textContent = "无法自动复制，可展开事件的关联信息后手动复制。"; }
  });
  $("error-dismiss").addEventListener("click", hideError);
  $("connection-button").addEventListener("click", () => {
    if (connected) return;
    clearTimeout(reconnectTimer);
    disconnectCurrentSocket();
    connect();
  });
  ui.authForm.addEventListener("submit", (event) => {
    event.preventDefault();
    authToken = ui.token.value.trim();
    ui.token.value = "";
    if (!authToken) { ui.token.focus(); return; }
    disconnectCurrentSocket();
    hideError();
    connect();
  });
  document.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => { void submitText(button.dataset.prompt); });
  });
  $("clear-button").addEventListener("click", () => {
    captureGeneration += 1;
    listening = false;
    userSpeaking = false;
    void closeCapture(false).catch(() => {});
    interruptResponse();
    disconnectCurrentSocket();
    resetTasks();
    resetExecutionTrace();
    connect();
    for (const message of ui.messages.querySelectorAll(".message")) message.remove();
    ui.welcome.hidden = false;
    turnCount = 0;
    ui.counter.textContent = "0 轮对话";
    beginTurn();
    hideError();
    ui.input.focus();
  });
  window.addEventListener("pagehide", () => {
    leaving = true;
    clearTimeout(reconnectTimer);
    captureGeneration += 1;
    listening = false;
    userSpeaking = false;
    void closeCapture(false).catch(() => {});
    stopPlayback();
    disconnectCurrentSocket();
    if (outputContext?.state !== "closed") void outputContext?.close();
  });
  window.addEventListener("pageshow", (event) => { if (event.persisted) { leaving = false; connect(); } });

  async function initialize() {
    try {
      const response = await fetch("/api/health", {cache:"no-store"});
      if (response.ok) {
        const health = await response.json();
        authRequired = Boolean(health.auth_required);
        updateProviderInfo(health);
        updateTraceConfiguration(health);
        if (health.configured === false && health.missing?.length) showError("服务端配置尚未完成，请填写本地 .env 中的语音服务与模型配置。");
      }
    } catch (_) { /* WebSocket connection reports service availability. */ }
    connect();
  }
  void initialize();
})();

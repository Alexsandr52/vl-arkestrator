"use strict";

const CSRF = document.querySelector('meta[name="csrf"]').content;
const $ = (id) => document.getElementById(id);
let lastSeq = 0;
let lastState = null;
let failures = 0;

const PHASE_RU = {
  OFF: "Сессия выключена", STARTING: "Запуск сессии", IDLE: "Жду команду",
  BACKSTEP: "Шаг назад", CAPTURE_COMMANDER: "Запоминаю вас", SEARCH: "Ищу предмет",
  APPROACH_TABLE: "Подхожу к столу", GRASP: "Беру предмет", RETURN: "Возвращаюсь",
  FIND_PERSON: "Ищу вас", APPROACH_PERSON: "Подхожу к вам", HANDOVER: "Передаю предмет",
  RECOVERING: "Восстановление", SAFE_HOLD: "Безопасное удержание", RELEASE_READY: "Готов отдать руки",
  ENDING: "Отдаю руки", TEACH: "Teach",
};

async function post(path, body, headers = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "X-Arkestrator-CSRF": CSRF, ...headers },
    body,
  });
  let data = {};
  try { data = await response.json(); } catch (_) { /* empty */ }
  if (!response.ok && data.ok === undefined) data = { ok: false, message: `HTTP ${response.status}` };
  return data;
}

async function action(name, extra = {}) {
  const data = await post("/api/action", JSON.stringify({ action: name, ...extra }), { "Content-Type": "application/json" });
  showReply(data);
  return data;
}

function showReply(data) {
  const el = $("reply");
  el.textContent = data.message || (data.ok ? "OK" : "Ошибка");
  el.className = "reply " + (data.ok ? "ok" : "err");
  if (data.heard !== undefined) $("heard").textContent = `Услышал: «${data.heard}»`;
}

// ---------------------------------------------------------------- rendering
function setText(el, text) { if (el.textContent !== text) el.textContent = text; }

function headRow(titles) {
  const tr = document.createElement("tr");
  for (const title of titles) { const th = document.createElement("th"); th.textContent = title; tr.appendChild(th); }
  return tr;
}

function render(payload) {
  const s = payload.state;
  lastState = s;
  const phase = s.phase || "OFF";
  const badge = $("phase");
  setText(badge, PHASE_RU[phase] || phase);
  badge.className = "badge " + (phase === "IDLE" ? "idle" : phase === "SAFE_HOLD" ? "hold" : phase === "OFF" ? "off" : "busy");
  setText($("session"), s.session_active ? "руки удерживаются оркестратором" : "руки у заводского контроллера");

  const allowed = new Set(s.allowed || []);
  $("btn-start").disabled = !allowed.has("start_session");
  $("btn-cancel").disabled = !allowed.has("cancel");
  $("btn-release").disabled = !allowed.has("release");
  $("btn-stop").disabled = !s.session_active;
  $("btn-talk").disabled = !allowed.has("command") && !allowed.has("release") && !allowed.has("cancel");
  $("btn-prepare").disabled = !allowed.has("prepare_release");
  $("btn-end").disabled = !allowed.has("end_session") || !s.release_token;
  setText($("release-token"), s.release_token ? `Удержание готово, токен ${s.release_token.slice(0, 8)}…` : "");
  if (s.release_token) $("end-box").open = true;

  const info = $("task-info");
  const rows = [];
  if (s.task) rows.push(["Команда", `«${s.task.text}»`], ["Профиль", s.task.profile + (s.task.attributes.length ? ` (${s.task.attributes.join(", ")})` : "")]);
  if (s.stage_details && Object.keys(s.stage_details).length) rows.push(["Этап", JSON.stringify(s.stage_details)]);
  rows.push(["В руке", s.holding || "—"], ["У стола", s.near_table ? "да (поворот только после шага назад)" : "нет"]);
  if (s.awaiting_release) rows.push(["Жду", "«отпусти» (голосом или кнопкой)"]);
  if (s.last_heard) rows.push(["Последняя фраза", `«${s.last_heard}»`]);
  info.replaceChildren(...rows.flatMap(([k, v]) => { const dt = document.createElement("dt"); dt.textContent = k; const dd = document.createElement("dd"); dd.textContent = v; return [dt, dd]; }));

  const attention = $("attention");
  attention.hidden = !s.attention;
  setText(attention, s.attention || "");

  $("said").replaceChildren(...(s.said || []).slice().reverse().map((t) => { const li = document.createElement("li"); li.textContent = t; return li; }));

  renderProfiles(s, allowed);
  renderHistory(s.history || []);
  renderVla(s.vla || {});
  renderMap(s, payload.health || {});
  renderEvents(payload.events || []);
}

function renderProfiles(s, allowed) {
  const table = $("profiles");
  const head = headRow(["Профиль", "Готов", "Поза у стола (м)", ""]);
  const rows = Object.entries(s.profiles || {}).map(([name, p]) => {
    const tr = document.createElement("tr");
    const cells = [p.display, p.ready ? "да" : p.why_not, p.taught ? p.taught.map((v) => v.toFixed(2)).join(", ") : "не обучено"];
    for (const text of cells) { const td = document.createElement("td"); td.textContent = text; tr.appendChild(td); }
    const td = document.createElement("td");
    const button = document.createElement("button");
    button.textContent = "Teach";
    button.disabled = !allowed.has("teach");
    button.title = "Поставьте робота у стола как при записи датасета (пультом, до сессии) и нажмите";
    button.addEventListener("click", () => action("teach", { profile: name }));
    td.appendChild(button);
    tr.appendChild(td);
    return tr;
  });
  table.replaceChildren(head, ...rows);
}

function renderHistory(history) {
  const table = $("history");
  const head = headRow(["Команда", "Итог", "Попыток", "с"]);
  const rows = history.slice().reverse().map((h) => {
    const tr = document.createElement("tr");
    for (const text of [h.text, h.outcome, String(h.grasp_attempts), String(h.seconds)]) {
      const td = document.createElement("td"); td.textContent = text; tr.appendChild(td);
    }
    return tr;
  });
  table.replaceChildren(head, ...rows);
}

function renderVla(vla) {
  const items = Object.entries(vla).map(([name, status]) => { const li = document.createElement("li"); li.textContent = `${name}: ${status}`; return li; });
  if (!items.length) { const li = document.createElement("li"); li.textContent = "не загружены"; items.push(li); }
  $("vla").replaceChildren(...items);
}

// World (x forward, y left) → SVG (x right, y down): robot's initial heading points up.
const toSvg = ([x, y]) => [-y, -x];
const SVG_NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs) { const el = document.createElementNS(SVG_NS, tag); for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v); return el; }

function renderMap(s, health) {
  const map = $("map");
  const points = [];
  const children = [];
  const world = health.world;
  if (world) {
    for (const o of world.objects) { const [x, y] = toSvg(o.xy); points.push([x, y]); children.push(svg("rect", { x: x - 0.06, y: y - 0.06, width: 0.12, height: 0.12, fill: "#9ca3af" })); }
    for (const p of world.persons) { const [x, y] = toSvg(p.xy); points.push([x, y]); children.push(svg("circle", { cx: x, cy: y, r: 0.15, fill: "none", stroke: "#9ca3af", "stroke-width": 0.03 })); }
  }
  const path = (s.waypoints || []).map((w) => toSvg(w));
  if (path.length > 1) children.push(svg("polyline", { points: path.map((p) => p.join(",")).join(" "), fill: "none", stroke: "#8b5cf6", "stroke-width": 0.03 }));
  points.push(...path);
  if (s.object_fix) { const [x, y] = toSvg(s.object_fix); points.push([x, y]); children.push(svg("circle", { cx: x, cy: y, r: 0.08, fill: "#b45309" })); }
  if (s.commander) { const [x, y] = toSvg(s.commander); points.push([x, y]); children.push(svg("circle", { cx: x, cy: y, r: 0.1, fill: "#15803d" })); }
  if (s.pose) {
    const [x, y] = toSvg(s.pose);
    points.push([x, y]);
    const deg = (-s.pose[2] * 180) / Math.PI;
    const robot = svg("polygon", { points: "0,-0.22 0.13,0.15 -0.13,0.15", fill: "#2563eb", transform: `translate(${x} ${y}) rotate(${deg})` });
    children.push(robot);
  }
  if (points.length) {
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
    const minX = Math.min(...xs) - 0.8, maxX = Math.max(...xs) + 0.8, minY = Math.min(...ys) - 0.8, maxY = Math.max(...ys) + 0.8;
    const size = Math.max(maxX - minX, maxY - minY, 3);
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    map.setAttribute("viewBox", `${cx - size / 2} ${cy - size / 2} ${size} ${size}`);
  }
  map.replaceChildren(...children);
}

function renderEvents(events) {
  if (!events.length) return;
  const list = $("events");
  for (const e of events) {
    lastSeq = Math.max(lastSeq, e.seq);
    const li = document.createElement("li");
    const t = document.createElement("span"); t.className = "t"; t.textContent = e.t.toFixed(1);
    const k = document.createElement("span"); k.className = "k"; k.textContent = e.kind;
    const rest = { ...e }; delete rest.seq; delete rest.t; delete rest.wall; delete rest.kind; delete rest.trace;
    const body = document.createElement("span"); body.textContent = Object.keys(rest).length ? JSON.stringify(rest) : "";
    li.append(t, k, body);
    list.prepend(li);
  }
  while (list.children.length > 300) list.lastChild.remove();
}

async function poll() {
  try {
    const response = await fetch(`/api/state?after=${lastSeq}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    failures = 0;
    setText($("link"), "");
  } catch (err) {
    failures += 1;
    if (failures > 2) setText($("link"), "нет связи с демоном");
  } finally {
    setTimeout(poll, 300);
  }
}

// ---------------------------------------------------------------- push-to-talk
let recorder = null;

async function startTalk() {
  if (recorder) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    const context = new AudioContext();
    await context.audioWorklet.addModule("/static/recorder-worklet.js");
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, "recorder");
    const chunks = [];
    node.port.onmessage = (event) => chunks.push(event.data);
    source.connect(node);
    recorder = { stream, context, node, chunks, started: performance.now() };
    recorder.timer = setTimeout(stopTalk, 15000);
    $("btn-talk").classList.add("recording");
    setText($("btn-talk"), "🎙 Говорите… отпустите, чтобы отправить");
  } catch (err) {
    showReply({ ok: false, message: "Микрофон недоступен: " + err.message + " (откройте панель через localhost/SSH-туннель)" });
  }
}

async function stopTalk() {
  if (!recorder) return;
  const r = recorder;
  recorder = null;
  clearTimeout(r.timer);
  $("btn-talk").classList.remove("recording");
  setText($("btn-talk"), "🎙 Удерживайте и говорите");
  r.node.disconnect();
  r.stream.getTracks().forEach((t) => t.stop());
  const rate = r.context.sampleRate;
  await r.context.close();
  if (performance.now() - r.started < 300) return;
  const samples = concat(r.chunks);
  const wav = encodeWav(downsample(samples, rate, 16000), 16000);
  showReply({ ok: true, message: "Распознаю…" });
  showReply(await post("/api/voice", wav, { "Content-Type": "audio/wav" }));
}

function concat(chunks) {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) { out.set(c, offset); offset += c.length; }
  return out;
}

function downsample(samples, from, to) {
  if (from === to) return samples;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(samples.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio), end = Math.min(samples.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += samples[j];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)); };
  write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); write(8, "WAVE");
  write(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// ---------------------------------------------------------------- wiring
$("btn-start").addEventListener("click", () => action("start_session"));
$("btn-stop").addEventListener("click", () => action("stop"));
$("btn-cancel").addEventListener("click", () => action("cancel"));
$("btn-release").addEventListener("click", () => action("release"));
$("btn-prepare").addEventListener("click", () => {
  $("chk-path").checked = false;
  $("chk-object").checked = false;
  action("prepare_release");
});
$("btn-end").addEventListener("click", () => action("end_session", {
  token: lastState && lastState.release_token,
  arm_path_clear: $("chk-path").checked,
  object_supported_or_absent: $("chk-object").checked,
}).then((data) => { if (data.ok) { $("chk-path").checked = false; $("chk-object").checked = false; } }));
$("cmd-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = $("cmd").value.trim();
  if (!text) return;
  action("command", { text }).then((data) => { if (data.ok) $("cmd").value = ""; });
});
const talk = $("btn-talk");
talk.addEventListener("pointerdown", (event) => { event.preventDefault(); startTalk(); });
for (const type of ["pointerup", "pointerleave", "pointercancel"]) talk.addEventListener(type, () => stopTalk());

const camera = $("camera");
camera.addEventListener("load", () => { camera.hidden = false; });
camera.addEventListener("error", () => { camera.hidden = true; setTimeout(() => { camera.src = "/stream.mjpg?" + Date.now(); }, 5000); });
camera.src = "/stream.mjpg";

poll();

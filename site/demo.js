const NS = "http://www.w3.org/2000/svg";
const WIDTH = 800;
const HEIGHT = 372;
const MARGIN = { top: 32, right: 12, bottom: 40, left: 54 };
const Y_STRETCH = 14;
const STEP_MS = 110;
const HOLD_MS = 1400;

const MODELS = [
  { test: /argon/, label: "Google DeepMind · Gemini 4 Argon", color: "#1A73E8" },
  { test: /gemini/, label: "Google DeepMind · Gemini 3.1 Pro", color: "#174EA6" },
  { test: /flint|recursive/, label: "Recursive · Flint", color: "#574D80" },
];
const OTHER_MODEL = { label: "Other", color: "#8A8478" };

const BASELINES = [
  { name: "One-hot VICReg", validation_auroc: 0.733, validation_auprc: 0.721 },
  { name: "Raw ESMC VICReg", validation_auroc: 0.809, validation_auprc: 0.803 },
];

const plots = [
  { svg: document.querySelector("#chart"), metric: "validation_auroc", title: "Validation AUROC", yLabel: "AUROC (Exponential)" },
  { svg: document.querySelector("#auprc"), metric: "validation_auprc", title: "Validation AUPRC", yLabel: "AUPRC (Exponential)" },
];

const playButton = document.querySelector("#play");
const stopButton = document.querySelector("#stop");
const restartButton = document.querySelector("#restart");
const recordButton = document.querySelector("#record");
const progress = document.querySelector("#progress");
const notesButton = document.querySelector("#notes");
const legend = document.querySelector("#legend");

const state = {
  sessions: [],
  layouts: [],
  step: 0,
  maxStep: 1,
  timer: null,
  hidden: new Set(),
  notes: true,
  recording: false,
  keepVideo: false,
};

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function modelStyle(model) {
  const name = String(model || "").toLowerCase();
  return MODELS.find((entry) => entry.test.test(name)) || OTHER_MODEL;
}

function svgEl(name, attrs) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

function experimentNumber(experiment, index) {
  const parsed = Number.parseInt(experiment.id, 10);
  return Number.isFinite(parsed) ? parsed : index + 1;
}

function metricValue(experiment, metric) {
  const value = experiment[metric];
  return experiment.status === "valid" && typeof value === "number" ? value : null;
}

function yUnits(value) {
  return Math.exp(Y_STRETCH * value);
}

function valueFromUnits(units) {
  return Math.log(units) / Y_STRETCH;
}

function buildLayout(metric) {
  const points = [];
  let maxX = 1;
  const values = BASELINES.map((baseline) => baseline[metric]).filter((value) => typeof value === "number");
  state.sessions.forEach((session) => {
    if (state.hidden.has(session.id)) return;
    session.experiments.forEach((experiment, index) => {
      const score = metricValue(experiment, metric);
      if (score === null) return;
      const x = experimentNumber(experiment, index);
      maxX = Math.max(maxX, x);
      values.push(score);
      points.push({ x, y: score, session, experiment, index });
    });
  });
  let yMin = Math.min(...values);
  let yMax = Math.max(...values);
  const low = yUnits(yMin);
  const high = yUnits(yMax);
  const span = Math.max(high - low, high * 0.04);
  const pad = span * 0.08;
  yMin = valueFromUnits(Math.max(low - pad, low * 0.92));
  yMax = Math.min(1, valueFromUnits(high + pad));
  const innerW = WIDTH - MARGIN.left - MARGIN.right;
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
  const xMax = Math.max(2, maxX);
  const yLow = yUnits(yMin);
  const yHigh = yUnits(yMax);
  return {
    points,
    yMin,
    yMax,
    xMax,
    xOf: (x) => MARGIN.left + ((x - 1) / (xMax - 1)) * innerW,
    yOf: (y) => MARGIN.top + (1 - (yUnits(y) - yLow) / (yHigh - yLow)) * innerH,
  };
}

function improvementPath(session, metric) {
  const path = [];
  let best = null;
  session.experiments.forEach((experiment, index) => {
    if (experimentNumber(experiment, index) > state.step) return;
    const score = metricValue(experiment, metric);
    if (score === null || (best !== null && score <= best)) return;
    path.push({ x: experimentNumber(experiment, index), y: score });
    best = score;
  });
  return path;
}

function pathFrom(points, layout) {
  return points.map((point, index) => {
    const command = index === 0 ? "M" : "L";
    return `${command}${layout.xOf(point.x).toFixed(2)} ${layout.yOf(point.y).toFixed(2)}`;
  }).join(" ");
}

function drawFrame(plot, layout) {
  const chart = plot.svg;
  chart.replaceChildren();
  chart.append(svgEl("rect", { x: 0, y: 0, width: WIDTH, height: HEIGHT, fill: "transparent" }));
  const title = svgEl("text", { class: "chart-title", x: WIDTH / 2, y: 16, "text-anchor": "middle" });
  title.textContent = plot.title;
  chart.append(title);
  const yAxis = svgEl("text", {
    x: 14,
    y: (MARGIN.top + HEIGHT - MARGIN.bottom) / 2,
    "text-anchor": "middle",
    transform: `rotate(-90 14 ${(MARGIN.top + HEIGHT - MARGIN.bottom) / 2})`,
  });
  yAxis.textContent = plot.yLabel;
  chart.append(yAxis);

  for (let index = 0; index <= 5; index += 1) {
    const units = yUnits(layout.yMin) + (yUnits(layout.yMax) - yUnits(layout.yMin)) * (index / 5);
    const tick = valueFromUnits(units);
    const y = layout.yOf(tick);
    chart.append(svgEl("line", { class: "grid", x1: MARGIN.left, x2: WIDTH - MARGIN.right, y1: y, y2: y }));
    const label = svgEl("text", { x: MARGIN.left - 8, y: y + 3, "text-anchor": "end" });
    label.textContent = tick.toFixed(3);
    chart.append(label);
  }

  const xStep = layout.xMax > 40 ? 10 : layout.xMax > 16 ? 5 : 1;
  const xTicks = [1];
  for (let x = xStep; x <= layout.xMax; x += xStep) {
    if (x !== 1) xTicks.push(x);
  }
  for (const x of xTicks) {
    const label = svgEl("text", { x: layout.xOf(x), y: HEIGHT - 18, "text-anchor": "middle" });
    label.textContent = String(x);
    chart.append(label);
  }
  const xAxis = svgEl("text", { x: WIDTH / 2, y: HEIGHT - 3, "text-anchor": "middle" });
  xAxis.textContent = "Experiment #";
  chart.append(xAxis);

  const byLevel = new Map();
  for (const baseline of BASELINES) {
    const score = baseline[plot.metric];
    if (typeof score !== "number") continue;
    const key = score.toFixed(3);
    if (!byLevel.has(key)) byLevel.set(key, []);
    byLevel.get(key).push(baseline);
  }
  for (const [key, rows] of byLevel) {
    const y = layout.yOf(Number(key));
    chart.append(svgEl("line", {
      class: "baseline",
      x1: MARGIN.left,
      x2: WIDTH - MARGIN.right,
      y1: y,
      y2: y,
    }));
    rows.forEach((row, index) => {
      const label = svgEl("text", {
        class: "baseline-label",
        x: WIDTH - MARGIN.right,
        y: y - 8 - (rows.length - 1 - index) * 16,
        "text-anchor": "end",
      });
      label.textContent = `${row.name} ${row[plot.metric].toFixed(3)}`;
      chart.append(label);
    });
  }

  state.sessions.forEach((session) => {
    if (state.hidden.has(session.id)) return;
    const color = modelStyle(session.model).color;
    const path = improvementPath(session, plot.metric);
    if (path.length > 1) {
      chart.append(svgEl("path", { class: "series", d: pathFrom(path, layout), stroke: color }));
    }
    let best = null;
    session.experiments.forEach((experiment, index) => {
      const x = experimentNumber(experiment, index);
      if (x > state.step) return;
      const score = metricValue(experiment, plot.metric);
      if (score === null) return;
      const improved = best !== null && score > best;
      if (best === null || score > best) best = score;
      const circle = svgEl("circle", {
        class: `point${improved ? "" : " is-open"}`,
        cx: layout.xOf(x),
        cy: layout.yOf(score),
        r: improved ? 5 : 3,
        fill: improved ? color : "none",
      });
      circle.style.setProperty("--point", color);
      chart.append(circle);
    });
  });
}

let audioCtx = null;

function audio() {
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context) return null;
  if (!audioCtx) audioCtx = new Context();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

const PENTATONIC = [0, 2, 4, 7, 9];

function tone(frequency, delay, duration, level, type) {
  const ctx = audio();
  if (!ctx || ctx.state !== "running") return;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const start = ctx.currentTime + delay;
  osc.type = type;
  osc.frequency.value = frequency;
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(level, start + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  osc.connect(gain);
  gain.connect(ctx.destination);
  if (recordGain) gain.connect(recordGain);
  osc.start(start);
  osc.stop(start + duration + 0.02);
}

function pitchFor(score) {
  const placed = Math.min(1, Math.max(0, (score - 0.5) / 0.34));
  const steps = Math.round(placed * 14);
  const octave = Math.floor(steps / PENTATONIC.length);
  const midi = 60 + octave * 12 + PENTATONIC[steps % PENTATONIC.length];
  return 440 * 2 ** ((midi - 69) / 12);
}

function shepard(score, duration, level) {
  const frequency = pitchFor(score);
  tone(frequency, 0, duration, level, "triangle");
  tone(frequency / 2, 0, duration, level * 0.28, "sine");
  tone(frequency * 2, 0.02, duration * 0.7, level * 0.12, "sine");
}

function scoreAtStep() {
  let score = null;
  for (const session of state.sessions) {
    if (state.hidden.has(session.id)) continue;
    for (let index = 0; index < session.experiments.length; index += 1) {
      const experiment = session.experiments[index];
      if (experimentNumber(experiment, index) !== state.step) continue;
      const value = metricValue(experiment, "validation_auroc");
      if (value !== null) score = score === null ? value : Math.max(score, value);
    }
  }
  return score;
}

function playStepSound(finished) {
  const bests = newBests();
  if (bests.length) {
    shepard(Math.max(...bests.map((hit) => hit.score)), 0.2, 0.04);
    return;
  }
  const score = scoreAtStep();
  if (score !== null) {
    shepard(score, 0.055, 0.016);
    return;
  }
  if (finished) {
    let best = null;
    for (const session of state.sessions) {
      if (state.hidden.has(session.id)) continue;
      for (let index = 0; index < session.experiments.length; index += 1) {
        const experiment = session.experiments[index];
        if (experimentNumber(experiment, index) > state.step) continue;
        const value = metricValue(experiment, "validation_auroc");
        if (value !== null) best = best === null ? value : Math.max(best, value);
      }
    }
    if (best !== null) shepard(best, 0.32, 0.03);
  }
}

function newBests() {
  const found = [];
  for (const session of state.sessions) {
    if (state.hidden.has(session.id)) continue;
    let best = null;
    for (let index = 0; index < session.experiments.length; index += 1) {
      const experiment = session.experiments[index];
      const x = experimentNumber(experiment, index);
      if (x > state.step) continue;
      const score = metricValue(experiment, "validation_auroc");
      if (score === null) continue;
      const improved = best !== null && score > best;
      if (best === null || score > best) best = score;
      if (improved && x === state.step) found.push({ session, experiment, score, x });
    }
  }
  return found;
}

function renderNote() {
  const figure = plots[0].svg.parentElement;
  figure.querySelectorAll(".chart-tooltip").forEach((node) => node.remove());
  if (!state.notes) return;
  const hits = newBests();
  if (!hits.length) return;
  const layout = state.layouts[0];
  const svg = plots[0].svg;
  const scaleX = svg.clientWidth / WIDTH;
  const scaleY = svg.clientHeight / HEIGHT;
  for (const hit of hits) {
    const tip = document.createElement("div");
    tip.className = "chart-tooltip";
    const title = hit.experiment.title || hit.experiment.change || "Untitled";
    tip.innerHTML = `<span class="tip-score">${esc(hit.experiment.id)} · ${hit.score.toFixed(3)}</span><span class="tip-title">${esc(title)}</span>`;
    figure.append(tip);
    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    let x = layout.xOf(hit.x) * scaleX + 12;
    let y = layout.yOf(hit.score) * scaleY - height - 8;
    if (y < 8) y = layout.yOf(hit.score) * scaleY + 12;
    x = Math.min(Math.max(8, x), Math.max(8, svg.clientWidth - width - 8));
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
}

function renderLegend() {
  legend.replaceChildren();
  const counts = {};
  for (const session of state.sessions) {
    const label = modelStyle(session.model).label;
    counts[label] = (counts[label] || 0) + 1;
  }
  for (const session of state.sessions) {
    const style = modelStyle(session.model);
    const hidden = state.hidden.has(session.id);
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    if (hidden) button.classList.add("is-off");
    button.title = hidden ? "Show on charts" : "Hide on charts";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.color = style.color;
    const name = counts[style.label] > 1 ? `${style.label} · ${session.id}` : style.label;
    button.append(swatch, document.createTextNode(name));
    button.addEventListener("click", () => {
      if (hidden) state.hidden.delete(session.id);
      else state.hidden.add(session.id);
      renderLegend();
      refreshLayouts();
      render();
    });
    item.append(button);
    legend.append(item);
  }
}

function enabledLimit() {
  let max = 0;
  for (const session of state.sessions) {
    if (state.hidden.has(session.id)) continue;
    session.experiments.forEach((experiment, index) => {
      const scored = metricValue(experiment, "validation_auroc") !== null
        || metricValue(experiment, "validation_auprc") !== null;
      if (!scored) return;
      max = Math.max(max, experimentNumber(experiment, index));
    });
  }
  return max;
}

function syncLimit() {
  state.maxStep = enabledLimit();
  if (state.step > state.maxStep) {
    state.step = state.maxStep;
    stop();
  }
}

function refreshLayouts() {
  syncLimit();
  state.layouts = plots.map((plot) => buildLayout(plot.metric));
}

function render() {
  plots.forEach((plot, index) => drawFrame(plot, state.layouts[index]));
  renderNote();
  progress.textContent = state.step ? `Experiment ${state.step}` : "Start";
  if (state.recording) paintRecording();
}

function stop() {
  clearTimeout(state.timer);
  state.timer = null;
}

function schedule() {
  syncLimit();
  if (state.maxStep < 1 || state.step >= state.maxStep) {
    state.timer = null;
    return;
  }
  const wait = state.notes && newBests().length ? HOLD_MS : STEP_MS;
  state.timer = setTimeout(() => {
    syncLimit();
    if (state.maxStep < 1 || state.step >= state.maxStep) {
      state.timer = null;
      render();
      return;
    }
    state.step += 1;
    const finished = state.step >= state.maxStep;
    render();
    playStepSound(finished);
    if (finished) {
      state.timer = null;
      if (state.recording) finishRecordingSoon();
      return;
    }
    schedule();
  }, wait);
}

function play() {
  if (state.timer) return;
  syncLimit();
  if (state.maxStep < 1) return;
  if (state.step >= state.maxStep) state.step = 0;
  schedule();
}

function restart() {
  stop();
  state.step = 0;
  render();
  play();
}

const query = new URLSearchParams(location.search);
let recordGain = null;
let recorder = null;
let recordChunks = [];
let recordCanvas = null;
let recordDone = null;
let paintGen = 0;
let paintTask = Promise.resolve();

const SVG_STYLE = ["fill", "stroke", "stroke-width", "stroke-dasharray", "stroke-linecap", "stroke-linejoin", "opacity", "font-family", "font-size", "font-weight"];

function inlineSvgStyles(source, clone) {
  const computed = getComputedStyle(source);
  for (const prop of SVG_STYLE) {
    const value = computed.getPropertyValue(prop);
    if (value) clone.style.setProperty(prop, value);
  }
  for (let index = 0; index < source.children.length; index += 1) {
    inlineSvgStyles(source.children[index], clone.children[index]);
  }
}

function svgImage(svg) {
  const clone = svg.cloneNode(true);
  inlineSvgStyles(svg, clone);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", "800");
  clone.setAttribute("height", "372");
  const xml = new XMLSerializer().serializeToString(clone);
  const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml;charset=utf-8" }));
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("chart"));
    };
    image.src = url;
  });
}

function wrapText(ctx, text, maxWidth) {
  const words = String(text || "").replace(/\s+/g, " ").trim().split(" ");
  const lines = [];
  let line = "";
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (line && ctx.measureText(next).width > maxWidth) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function recordingTitle() {
  const labels = [];
  for (const session of state.sessions) {
    if (state.hidden.has(session.id)) continue;
    const label = modelStyle(session.model).label;
    if (!labels.includes(label)) labels.push(label);
  }
  if (labels.length === 1) return labels[0];
  return "Session playback";
}

function recordingColor() {
  for (const session of state.sessions) {
    if (!state.hidden.has(session.id)) return modelStyle(session.model).color;
  }
  return "#1f1d18";
}

function recordingFrame() {
  const width = 1080;
  const pad = 56;
  const chartW = width - pad * 2;
  const chartH = chartW * (372 / 800);
  const noteLines = 2;
  const noteLineH = 32;
  const contentH = 58 + 44 + noteLines * noteLineH + 36 + chartH;
  let height = Math.ceil(pad + contentH + pad);
  if (height % 2) height += 1;
  return { width, height, pad, chartW, chartH, noteLines, noteLineH };
}

function paintRecording() {
  const gen = ++paintGen;
  const font = getComputedStyle(document.body).fontFamily;
  paintTask = svgImage(plots[0].svg).then((image) => {
    if (gen !== paintGen || !recordCanvas) return;
    const ctx = recordCanvas.getContext("2d");
    const frame = recordingFrame();
    const { width, height, chartW, chartH, noteLines, noteLineH } = frame;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, width, height);
    ctx.textBaseline = "top";
    let y = frame.pad;
    ctx.fillStyle = recordingColor();
    ctx.fillRect(56, y + 8, 18, 18);
    ctx.fillStyle = "#1f1d18";
    ctx.font = `600 34px ${font}`;
    ctx.fillText(recordingTitle(), 88, y);
    y += 58;
    ctx.fillStyle = "#6f6b62";
    ctx.font = `22px ${font}`;
    ctx.fillText(state.step ? `EXPERIMENT ${state.step}` : "START", 56, y);
    y += 44;
    const bests = state.notes ? newBests() : [];
    ctx.fillStyle = "#1f1d18";
    ctx.font = `22px ${font}`;
    if (bests.length) {
      const hit = bests.reduce((best, row) => (row.score > best.score ? row : best));
      const name = hit.experiment.title || hit.experiment.change || "";
      const lines = wrapText(ctx, `${hit.experiment.id} · ${hit.score.toFixed(3)}   ${name}`, chartW);
      lines.forEach((line, index) => ctx.fillText(line, 56, y + index * noteLineH));
    }
    y += noteLines * noteLineH + 36;
    ctx.drawImage(image, 56, y, chartW, chartH);
  }).catch((error) => {
    document.documentElement.dataset.paint = String(error);
  });
  return paintTask;
}

function videoName() {
  const slug = recordingTitle().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return `${slug || "playback"}.webm`;
}

function updateRecordButton() {
  recordButton.textContent = state.recording ? "Recording" : "Record";
  recordButton.setAttribute("aria-pressed", state.recording ? "true" : "false");
}

function setupRecorder(ctx) {
  const dest = ctx.createMediaStreamDestination();
  recordGain = ctx.createGain();
  recordGain.gain.value = 6;
  recordGain.connect(dest);
  recordCanvas = document.createElement("canvas");
  const frame = recordingFrame();
  recordCanvas.width = frame.width;
  recordCanvas.height = frame.height;
  const stream = recordCanvas.captureStream(30);
  dest.stream.getAudioTracks().forEach((track) => stream.addTrack(track));
  const types = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
  const mime = types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 6_000_000, audioBitsPerSecond: 128_000 } : undefined);
  recordChunks = [];
  recorder.ondataavailable = (event) => {
    if (event.data.size) recordChunks.push(event.data);
  };
  recorder.onstop = () => {
    const chunks = recordChunks;
    const type = recorder.mimeType || "video/webm";
    const keep = state.keepVideo;
    state.recording = false;
    recordGain = null;
    updateRecordButton();
    if (!keep || !chunks.length) return;
    const blob = new Blob(chunks, { type });
    if (query.has("save")) {
      fetch("/save-video", { method: "POST", body: blob })
        .then((response) => {
          document.documentElement.dataset.video = response.ok ? "saved" : "error";
        })
        .catch(() => {
          document.documentElement.dataset.video = "error";
        });
      return;
    }
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = videoName();
    link.click();
  };
}

function finishRecordingSoon() {
  clearTimeout(recordDone);
  recordDone = setTimeout(async () => {
    await paintTask;
    if (!state.recording || !recorder || recorder.state === "inactive") return;
    state.keepVideo = true;
    recorder.stop();
  }, 800);
}

function abortRecording() {
  clearTimeout(recordDone);
  if (!state.recording && (!recorder || recorder.state === "inactive")) return;
  state.keepVideo = false;
  state.recording = false;
  recordGain = null;
  if (recorder && recorder.state !== "inactive") recorder.stop();
  updateRecordButton();
}

async function beginRecording() {
  if (state.recording || !state.sessions.length || typeof MediaRecorder === "undefined") return;
  stop();
  const ctx = audio();
  if (!ctx) return;
  if (ctx.state !== "running") await ctx.resume();
  state.step = 0;
  state.recording = true;
  state.keepVideo = true;
  document.documentElement.dataset.video = "recording";
  try {
    setupRecorder(ctx);
  } catch {
    state.recording = false;
    recordGain = null;
    progress.textContent = "Recording is unavailable";
    document.documentElement.dataset.video = "error";
    return;
  }
  updateRecordButton();
  render();
  recorder.start(200);
  play();
}

notesButton.addEventListener("click", () => {
  state.notes = !state.notes;
  notesButton.setAttribute("aria-pressed", state.notes ? "true" : "false");
  notesButton.classList.toggle("is-off", !state.notes);
  render();
});

playButton.addEventListener("click", () => { audio(); play(); });
stopButton.addEventListener("click", () => { stop(); abortRecording(); });
restartButton.addEventListener("click", () => { abortRecording(); audio(); restart(); });
recordButton.addEventListener("click", beginRecording);
document.addEventListener("pointerdown", () => audio(), { once: true });

fetch("./data/sessions.json")
  .then((response) => {
    if (!response.ok) throw new Error(response.statusText);
    return response.json();
  })
  .then((data) => {
    state.sessions = data.sessions || [];
    const only = query.get("only");
    if (only) {
      let test = null;
      try { test = new RegExp(only, "i"); } catch { test = null; }
      if (test) {
        for (const session of state.sessions) {
          if (!test.test(`${session.model || ""} ${session.id}`)) state.hidden.add(session.id);
        }
      }
    }
    refreshLayouts();
    renderLegend();
    render();
    if (!query.has("record")) play();
  })
  .catch(() => {
    progress.textContent = "Could not load sessions";
  });

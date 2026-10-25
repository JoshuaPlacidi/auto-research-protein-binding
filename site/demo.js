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
    session.experiments.forEach((experiment, index) => {
      const x = experimentNumber(experiment, index);
      maxX = Math.max(maxX, x);
      const score = metricValue(experiment, metric);
      if (score === null) return;
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
      render();
    });
    item.append(button);
    legend.append(item);
  }
}

function render() {
  plots.forEach((plot, index) => drawFrame(plot, state.layouts[index]));
  renderNote();
  progress.textContent = state.step ? `Experiment ${state.step}` : "Start";
}

function stop() {
  clearTimeout(state.timer);
  state.timer = null;
}

function schedule() {
  const wait = state.notes && newBests().length ? HOLD_MS : STEP_MS;
  state.timer = setTimeout(() => {
    state.step += 1;
    render();
    if (state.step >= state.maxStep) {
      state.timer = null;
      return;
    }
    schedule();
  }, wait);
}

function play() {
  if (state.timer) return;
  if (state.step >= state.maxStep) state.step = 0;
  schedule();
}

function restart() {
  stop();
  state.step = 0;
  render();
  play();
}

notesButton.addEventListener("click", () => {
  state.notes = !state.notes;
  notesButton.setAttribute("aria-pressed", state.notes ? "true" : "false");
  notesButton.classList.toggle("is-off", !state.notes);
  render();
});

playButton.addEventListener("click", play);
stopButton.addEventListener("click", stop);
restartButton.addEventListener("click", restart);

fetch("./data/sessions.json")
  .then((response) => {
    if (!response.ok) throw new Error(response.statusText);
    return response.json();
  })
  .then((data) => {
    state.sessions = data.sessions || [];
    state.maxStep = 1;
    state.sessions.forEach((session) => {
      session.experiments.forEach((experiment, index) => {
        state.maxStep = Math.max(state.maxStep, experimentNumber(experiment, index));
      });
    });
    state.layouts = plots.map((plot) => buildLayout(plot.metric));
    renderLegend();
    render();
    play();
  })
  .catch(() => {
    progress.textContent = "Could not load sessions";
  });

const NS = "http://www.w3.org/2000/svg";
const WIDTH = 800;
const HEIGHT = 372;
const MARGIN = { top: 32, right: 12, bottom: 40, left: 54 };

const sessionList = document.querySelector("#sessions");
const detail = document.querySelector("#detail");
const showAll = document.querySelector("#show-all");

const state = {
  data: null,
  focus: null,
  selected: null,
  hover: null,
  hidden: new Set(),
};

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function experimentNumber(experiment, index) {
  const parsed = Number.parseInt(experiment.id, 10);
  return Number.isFinite(parsed) ? parsed : index + 1;
}

function metricValue(experiment, metric) {
  const value = experiment[metric];
  return experiment.status === "valid" && typeof value === "number" ? value : null;
}

function experimentTitle(experiment) {
  return experiment.title || experiment.change || "Untitled experiment";
}

function bestMetric(session, key) {
  const values = session.experiments.map((row) => row[key]).filter((value) => typeof value === "number");
  return values.length ? Math.max(...values) : null;
}

function fmtCount(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toLocaleString("en-GB");
}

const MODELS = [
  { test: /argon/, label: "Google DeepMind · Gemini 4 Argon", color: "#1A73E8" },
  { test: /gemini/, label: "Google DeepMind · Gemini 3.1 Pro", color: "#174EA6" },
  { test: /flint|recursive/, label: "Recursive · Flint", color: "#574D80" },
];
const OTHER_MODEL = { label: "Other", color: "#8A8478" };

function modelStyle(model) {
  const name = String(model || "").toLowerCase();
  return MODELS.find((entry) => entry.test.test(name)) || OTHER_MODEL;
}

function visibleSessions() {
  return state.data.sessions.filter((session) => !state.hidden.has(session.id));
}

function svgEl(name, attrs) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

const BASELINES = [
  { name: "One-hot VICReg", validation_auroc: 0.733, validation_auprc: 0.721 },
  { name: "Raw ESMC VICReg", validation_auroc: 0.809, validation_auprc: 0.803 },
];

const plots = [
  {
    svg: document.querySelector("#chart"),
    tooltip: document.querySelector("#chart-tooltip"),
    metric: "validation_auroc",
    title: "Validation AUROC",
    yLabel: "AUROC (Exponential)",
    baselines: BASELINES,
  },
  {
    svg: document.querySelector("#auprc"),
    tooltip: document.querySelector("#auprc-tooltip"),
    metric: "validation_auprc",
    title: "Validation AUPRC",
    yLabel: "AUPRC (Exponential)",
    baselines: BASELINES,
  },
];

const Y_STRETCH = 14;

function yUnits(auroc) {
  return Math.exp(Y_STRETCH * auroc);
}

function aurocFromUnits(units) {
  return Math.log(units) / Y_STRETCH;
}

function improvesSession(session, index, metric) {
  const score = metricValue(session.experiments[index], metric);
  if (score === null) return false;
  let best = null;
  for (let earlier = 0; earlier < index; earlier += 1) {
    const previous = metricValue(session.experiments[earlier], metric);
    if (previous === null) continue;
    best = best === null ? previous : Math.max(best, previous);
  }
  return best !== null && score > best;
}

function improvementPath(session, metric) {
  const path = [];
  let best = null;
  session.experiments.forEach((experiment, index) => {
    const score = metricValue(experiment, metric);
    if (score === null || (best !== null && score <= best)) return;
    path.push({
      x: experimentNumber(experiment, index),
      y: score,
      session,
      experiment,
    });
    best = score;
  });
  return path;
}

function buildLayout(sessions, metric, baselines) {
  const points = [];
  let maxX = 1;
  let yMin = 0.5;
  let yMax = 0.8;
  const values = [];

  for (const session of sessions) {
    session.experiments.forEach((experiment, index) => {
      maxX = Math.max(maxX, experimentNumber(experiment, index));
      const score = metricValue(experiment, metric);
      if (score === null) return;
      values.push(score);
      points.push({
        x: experimentNumber(experiment, index),
        y: score,
        session,
        experiment,
      });
    });
  }

  values.push(...baselines.map((baseline) => baseline[metric]).filter((value) => typeof value === "number"));
  if (values.length) {
    yMin = Math.min(...values);
    yMax = Math.max(...values);
    const low = yUnits(yMin);
    const high = yUnits(yMax);
    const span = Math.max(high - low, high * 0.04);
    const pad = span * 0.08;
    yMin = aurocFromUnits(Math.max(low - pad, low * 0.92));
    yMax = Math.min(1, aurocFromUnits(high + pad));
  }

  const innerW = WIDTH - MARGIN.left - MARGIN.right;
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;
  const xMax = Math.max(2, maxX);
  const low = yUnits(yMin);
  const high = yUnits(yMax);
  const xOf = (x) => MARGIN.left + ((x - 1) / (xMax - 1)) * innerW;
  const yOf = (y) => MARGIN.top + (1 - (yUnits(y) - low) / (high - low)) * innerH;
  return { points, xOf, yOf, yMin, yMax, xMax, innerW, innerH };
}

function pathFrom(points, layout) {
  return points.map((point, index) => {
    const command = index === 0 ? "M" : "L";
    return `${command}${layout.xOf(point.x).toFixed(2)} ${layout.yOf(point.y).toFixed(2)}`;
  }).join(" ");
}

function renderPlot(plot) {
  const sessions = visibleSessions();
  const layout = buildLayout(sessions, plot.metric, plot.baselines);
  plot.layout = layout;
  const focus = state.focus;
  const selected = state.selected;
  const hover = state.hover;
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

  const tickCount = 5;
  const yTicks = [];
  for (let index = 0; index <= tickCount; index += 1) {
    const units = yUnits(layout.yMin) + (yUnits(layout.yMax) - yUnits(layout.yMin)) * (index / tickCount);
    yTicks.push(aurocFromUnits(units));
  }
  for (const tick of yTicks) {
    const y = layout.yOf(tick);
    chart.append(svgEl("line", {
      class: "grid",
      x1: MARGIN.left,
      x2: WIDTH - MARGIN.right,
      y1: y,
      y2: y,
    }));
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
    const label = svgEl("text", {
      x: layout.xOf(x),
      y: HEIGHT - 18,
      "text-anchor": "middle",
    });
    label.textContent = String(x);
    chart.append(label);
  }
  const xAxis = svgEl("text", { x: WIDTH / 2, y: HEIGHT - 3, "text-anchor": "middle" });
  xAxis.textContent = "Experiment #";
  chart.append(xAxis);

  const byLevel = new Map();
  for (const baseline of plot.baselines) {
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
    const above = y - 8 - (rows.length - 1) * 16 >= MARGIN.top;
    rows.forEach((row, index) => {
      const label = svgEl("text", {
        class: "baseline-label",
        x: WIDTH - MARGIN.right,
        y: above ? y - 8 - (rows.length - 1 - index) * 16 : y + 16 + index * 16,
        "text-anchor": "end",
      });
      label.textContent = `${row.name} ${row[plot.metric].toFixed(3)}`;
      chart.append(label);
    });
  }

  sessions.forEach((session, index) => {
    const dim = Boolean(focus) && focus !== session.id;
    const color = modelStyle(session.model).color;
    const path = improvementPath(session, plot.metric);
    if (path.length > 1) {
      const line = svgEl("path", {
        class: `series${dim ? " is-dim" : ""}${focus === session.id ? " is-focus" : ""}`,
        d: pathFrom(path, layout),
        stroke: color,
      });
      chart.append(line);
    }
    for (const point of layout.points.filter((row) => row.session.id === session.id)) {
      const active = selected
        && selected.sessionId === session.id
        && selected.experimentId === point.experiment.id;
      const hovered = hover
        && hover.sessionId === session.id
        && hover.experimentId === point.experiment.id;
      const improved = improvesSession(session, session.experiments.indexOf(point.experiment), plot.metric);
      const circle = svgEl("circle", {
        class: `point${improved ? "" : " is-open"}${dim ? " is-dim" : ""}${active || hovered ? " is-active" : ""}`,
        cx: layout.xOf(point.x),
        cy: layout.yOf(point.y),
        r: active || hovered ? (improved ? 6 : 4.2) : (improved ? 5 : 3),
        fill: improved ? color : "none",
      });
      circle.style.setProperty("--point", color);
      chart.append(circle);
    }
  });
}

function renderPlots() {
  for (const plot of plots) renderPlot(plot);
}

function nearestPoint(event, plot) {
  if (!plot.layout) return null;
  const ctm = plot.svg.getScreenCTM();
  if (!ctm) return null;
  const cursor = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
  let best = null;
  let bestDist = 18;
  for (const point of plot.layout.points) {
    if (state.hidden.has(point.session.id)) continue;
    if (state.focus && point.session.id !== state.focus) continue;
    const dx = plot.layout.xOf(point.x) - cursor.x;
    const dy = plot.layout.yOf(point.y) - cursor.y;
    const dist = Math.hypot(dx, dy);
    if (dist < bestDist) {
      best = point;
      bestDist = dist;
    }
  }
  return best;
}

function placeTooltip(event, point, plot) {
  const title = experimentTitle(point.experiment);
  const tooltip = plot.tooltip;
  for (const other of plots) other.tooltip.hidden = other !== plot;
  tooltip.hidden = false;
  tooltip.innerHTML = `<span class="tip-score">${esc(point.experiment.id)} · ${esc(fmt(point.y))}</span><span class="tip-title">${esc(title)}</span>`;
  const box = plot.svg.getBoundingClientRect();
  const host = plot.svg.parentElement.getBoundingClientRect();
  const width = tooltip.offsetWidth;
  const height = tooltip.offsetHeight;
  const x = event.clientX - host.left + 12;
  let y = event.clientY - host.top - height - 10;
  if (y < 8) y = event.clientY - host.top + 14;
  tooltip.style.left = `${Math.min(Math.max(8, x), Math.max(8, box.width - width - 8))}px`;
  tooltip.style.top = `${Math.max(8, y)}px`;
}

function hideTooltip() {
  for (const plot of plots) plot.tooltip.hidden = true;
  if (state.hover) {
    state.hover = null;
    renderPlots();
  }
}

function renderSessions() {
  sessionList.replaceChildren();
  state.data.sessions.forEach((session) => {
    const row = document.createElement("tr");
    const hidden = state.hidden.has(session.id);
    if (hidden) row.classList.add("is-off");
    else if (state.focus && state.focus !== session.id) row.classList.add("is-dim");
    const model = document.createElement("th");
    model.scope = "row";
    model.className = "model";
    const swatch = document.createElement("button");
    swatch.type = "button";
    swatch.className = "swatch";
    swatch.style.color = modelStyle(session.model).color;
    swatch.title = hidden ? "Show on charts" : "Hide on charts";
    swatch.setAttribute("aria-pressed", hidden ? "false" : "true");
    swatch.setAttribute("aria-label", hidden ? `Show ${session.id} on charts` : `Hide ${session.id} on charts`);
    swatch.addEventListener("click", () => {
      if (hidden) state.hidden.delete(session.id);
      else {
        state.hidden.add(session.id);
        if (state.focus === session.id) state.focus = null;
      }
      renderSessions();
      renderPlots();
    });
    const link = document.createElement("a");
    link.href = `./session.html?id=${encodeURIComponent(session.id)}`;
    link.textContent = session.model || "unknown model";
    model.append(swatch, link);
    const cells = [
      session.id,
      fmtCount(session.experiments.filter((row) => row.status === "valid").length),
      fmtCount(session.context_tokens),
      fmt(bestMetric(session, "validation_auroc")),
      fmt(bestMetric(session, "validation_auprc")),
    ].map((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      return cell;
    });
    cells[0].replaceChildren();
    const sessionLink = document.createElement("a");
    sessionLink.href = `./session.html?id=${encodeURIComponent(session.id)}`;
    sessionLink.textContent = session.id;
    cells[0].append(sessionLink);
    row.append(model, ...cells);
    sessionList.append(row);
  });
}

function renderDetail() {
  if (!state.selected) {
    detail.innerHTML = "";
    return;
  }
  const session = state.data.sessions.find((row) => row.id === state.selected.sessionId);
  const experiment = session?.experiments.find((row) => row.id === state.selected.experimentId);
  if (!experiment) {
    detail.innerHTML = "";
    return;
  }
  const delta = experiment.auroc_vs_parent;
  const deltaText = typeof delta === "number"
    ? ` · ${delta > 0 ? "+" : ""}${delta.toFixed(3)} vs parent`
    : "";
  const runtime = typeof experiment.runtime_seconds === "number"
    ? ` · ${Math.round(experiment.runtime_seconds)} s`
    : "";
  detail.innerHTML = `
    <p class="kicker">${esc(session.id)} · ${esc(experiment.id)} · ${esc(experiment.status)}</p>
    <h2>${esc(experimentTitle(experiment))}</h2>
    <p class="meta">AUROC ${esc(fmt(experiment.validation_auroc))}${esc(deltaText)} · AUPRC ${esc(fmt(experiment.validation_auprc))} · parent ${esc(experiment.parent || "—")}${esc(runtime)}</p>
    <h3>Hypothesis</h3>
    <p>${esc(experiment.hypothesis || "Not recorded.")}</p>
    <h3>Change</h3>
    <p>${esc(experiment.change || "Not recorded.")}</p>
    ${experiment.error ? `<h3>Error</h3><p>${esc(experiment.error)}</p>` : ""}
  `;
}

function selectPoint(point) {
  state.selected = { sessionId: point.session.id, experimentId: point.experiment.id };
  state.focus = point.session.id;
  renderSessions();
  renderPlots();
  renderDetail();
}

function onPointerMove(event, plot) {
  const point = nearestPoint(event, plot);
  if (!point) {
    hideTooltip();
    return;
  }
  const next = { sessionId: point.session.id, experimentId: point.experiment.id };
  const same = state.hover
    && state.hover.sessionId === next.sessionId
    && state.hover.experimentId === next.experimentId;
  state.hover = next;
  if (!same) renderPlots();
  placeTooltip(event, point, plot);
}

for (const plot of plots) {
  plot.svg.addEventListener("pointermove", (event) => onPointerMove(event, plot));
  plot.svg.addEventListener("pointerleave", hideTooltip);
  plot.svg.addEventListener("click", (event) => {
    const point = nearestPoint(event, plot);
    if (point) selectPoint(point);
  });
}

showAll.addEventListener("click", () => {
  state.focus = null;
  state.hidden.clear();
  renderSessions();
  renderPlots();
});

fetch("./data/sessions.json")
  .then((response) => {
    if (!response.ok) throw new Error(response.statusText);
    return response.json();
  })
  .then((data) => {
    state.data = data;
    renderSessions();
    renderPlots();
    renderDetail();
  })
  .catch(() => {
    detail.innerHTML = `<p class="placeholder">Could not load data/sessions.json. Run <code>python site/build_summary.py</code> from the repo root, then open this page from the site folder.</p>`;
  });

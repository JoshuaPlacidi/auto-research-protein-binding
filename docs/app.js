const STEP_MS = 2000;
const chart = document.querySelector("#chart");
const lineage = document.querySelector("#lineage");
const panel = document.querySelector("#panel");
const changeLog = document.querySelector("#change-log");
const stats = document.querySelector("#stats");
const sessionLine = document.querySelector("#session-line");
const footnote = document.querySelector("#footnote");
const progress = document.querySelector("#progress");
const playButton = document.querySelector("#play");

const state = {
  data: null,
  index: -1,
  playing: true,
  speed: 1,
  timer: null,
};

function fmt(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return "Not recorded";
  const number = Number(value);
  if (number !== 0 && Math.abs(number) < 1e-4) return number.toExponential(2);
  return number.toFixed(digits);
}

function esc(value) {
  return String(value ?? "Not recorded")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function fmtInt(value) {
  if (value === null || value === undefined) return "Not recorded";
  return Number(value).toLocaleString("en-GB");
}

function rows() {
  return state.data.experiments;
}

function frameCount() {
  return rows().length + 1;
}

function scoreColor(row) {
  return row.status === "failed" ? "#a33b32" : "#243044";
}

function aurocLabel(row) {
  if (row.status === "failed" || row.validation_auroc === null || row.validation_auroc === undefined) return "Failed";
  const score = fmt(row.validation_auroc, 3);
  if (row.auroc_vs_parent === null || row.auroc_vs_parent === undefined) return score;
  const delta = Number(row.auroc_vs_parent);
  const sign = delta > 0 ? "+" : "";
  return `${score} (${sign}${delta.toFixed(3)})`;
}

function raisedBest(row, previous) {
  if (row.status !== "valid" || row.validation_auroc === null || row.validation_auroc === undefined) return false;
  if (row.validation_auroc !== row.best_so_far_auroc) return false;
  if (!previous || previous.best_so_far_auroc === null || previous.best_so_far_auroc === undefined) return true;
  return row.validation_auroc > previous.best_so_far_auroc;
}

function renderStats() {
  const data = state.data;
  const best = data.highest_auroc_experiment;
  const cards = [
    ["Valid experiments", String(data.valid_experiments)],
    ["Failed experiments", String(data.failed_experiments)],
    ["Backtracks", String((data.backtracks || []).length)],
    ["Highest AUROC", best ? `${fmt(best.validation_auroc, 3)} · ${best.experiment_id}` : "Not recorded"],
  ];
  stats.innerHTML = cards.map(([label, value]) =>
    `<article class="stat"><b>${value}</b><span>${label}</span></article>`
  ).join("");
}

function renderChart() {
  const items = rows();
  const visible = items.slice(0, Math.max(0, state.index + 1));
  const scores = visible.map((row) => row.validation_auroc).filter((value) => value !== null && value !== undefined);
  if (!scores.length) {
    chart.innerHTML = `<text class="tick" x="430" y="220" text-anchor="middle">Experiments appear as the replay advances.</text>`;
    return;
  }
  const allMin = Math.min(...scores);
  const allMax = Math.max(...scores);
  const min = allMin;
  const max = allMax;
  const wide = false;
  const span = Math.max(max - min, 1e-6);
  const y0 = min - span * 0.35;
  const y1 = max + span * 0.35;
  const left = 78;
  const right = wide ? 210 : 28;
  const top = 24;
  const bottom = 96;
  const width = 860;
  const height = 500;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const xAt = (i) => left + (items.length === 1 ? plotW / 2 : (i / (items.length - 1)) * plotW);
  const yAt = (loss) => top + ((y1 - loss) / (y1 - y0)) * plotH;

  const ticks = [min, (min + max) / 2, max];
  const grid = ticks.map((tick) => {
    const y = yAt(tick);
    return `<line class="grid" x1="${left}" x2="${width - right}" y1="${y}" y2="${y}"></line>
      <text class="tick" x="${left - 10}" y="${y + 4}" text-anchor="end">${fmt(tick, 3)}</text>`;
  }).join("");

  const bestPoints = visible.filter((row) => row.best_so_far_auroc !== null && row.best_so_far_auroc !== undefined);
  const bestLine = bestPoints.map((row) => {
    const i = items.indexOf(row);
    return `${xAt(i)},${yAt(row.best_so_far_auroc)}`;
  }).join(" ");

  const backArcs = visible.filter((row) => row.backtrack).map((row) => {
    const i = items.indexOf(row);
    const parentIndex = items.findIndex((item) => item.experiment_id === row.parent);
    const parent = parentIndex >= 0 ? items[parentIndex] : null;
    if (!parent || parent.validation_auroc === null || parent.validation_auroc === undefined || row.validation_auroc === null || row.validation_auroc === undefined) return "";
    const lift = Math.min(90, 28 + (i - parentIndex) * 4);
    const x1 = xAt(parentIndex);
    const y1 = yAt(parent.validation_auroc);
    const x2 = xAt(i);
    const y2 = yAt(row.validation_auroc);
    return `<path class="back-arc" d="M ${x1} ${y1} Q ${(x1 + x2) / 2} ${Math.min(y1, y2) - lift} ${x2} ${y2}"></path>`;
  }).join("");

  const dots = visible.map((row) => {
    const i = items.indexOf(row);
    const active = i === state.index ? "active" : "";
    const score = row.validation_auroc;
    const ring = row.backtrack ? `stroke="#c47b2b" stroke-width="2"` : "";
    if (score === null || score === undefined) {
      const y = height - bottom + 10;
      return `<rect class="dot ${active}" data-index="${i}" x="${xAt(i) - 4}" y="${y}" width="8" height="8" fill="#a33b32"></rect>`;
    }
    const label = i === state.index
      ? `<text class="tick" x="${xAt(i)}" y="${Math.max(16, yAt(score) - 12)}" text-anchor="middle">${fmt(score, 3)}</text>`
      : "";
    return `${label}<circle class="dot ${active}" data-index="${i}" cx="${xAt(i)}" cy="${yAt(score)}" r="${i === state.index ? 7 : 5}" fill="${scoreColor(row)}" ${ring}></circle>`;
  }).join("");

  let inset = "";
  if (wide) {
    const ix = width - 196;
    const iy = 36;
    const iw = 172;
    const ih = 120;
    const cSpan = Math.max(clusterMax - allMin, 1e-6);
    const cy0 = allMin - cSpan * 0.35;
    const cy1 = clusterMax + cSpan * 0.35;
    const insetX = (i) => ix + 28 + (items.length === 1 ? (iw - 36) / 2 : (i / (items.length - 1)) * (iw - 36));
    const insetY = (loss) => iy + 16 + ((cy1 - Math.min(loss, cy1)) / (cy1 - cy0)) * (ih - 28);
    const insetLine = bestPoints.map((row) => `${insetX(items.indexOf(row))},${insetY(row.best_so_far_auroc)}`).join(" ");
    const insetDots = visible.map((row) => {
      if (row.validation_auroc === null || row.validation_auroc === undefined || row.validation_auroc - allMin >= 0.01) return "";
      const i = items.indexOf(row);
      return `<circle data-index="${i}" class="dot" cx="${insetX(i)}" cy="${insetY(row.validation_auroc)}" r="3.5" fill="${scoreColor(row)}"></circle>`;
    }).join("");
    inset = `
      <rect x="${ix}" y="${iy}" width="${iw}" height="${ih}" rx="10" fill="#fffdf8" stroke="#e2d9cc"></rect>
      <text class="tick" x="${ix + 10}" y="${iy + 16}">Near 0.250</text>
      ${insetLine ? `<polyline class="best-line" points="${insetLine}"></polyline>` : ""}
      ${insetDots}
    `;
  }

  const labels = items.map((row, i) =>
    `<text class="tick" x="${xAt(i)}" y="${height - 28}" text-anchor="end" transform="rotate(-65 ${xAt(i)} ${height - 28})">${row.experiment_id}</text>`
  ).join("");

  chart.innerHTML = `
    ${grid}
    <line class="axis" x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}"></line>
    <line class="axis" x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}"></line>
    ${bestLine ? `<polyline class="best-line" points="${bestLine}"></polyline>` : ""}
    ${backArcs}
    ${dots}
    ${inset}
    ${labels}
    <text class="tick" x="16" y="${top + 8}">AUROC</text>
  `;
  chart.querySelectorAll(".dot").forEach((dot) => {
    dot.addEventListener("click", () => {
      pause();
      show(Number(dot.dataset.index));
    });
  });
}

function renderLineage() {
  const items = rows();
  const visible = items.slice(0, Math.max(0, state.index + 1));
  if (!visible.length) {
    lineage.innerHTML = `<text class="tick" x="430" y="100" text-anchor="middle">Parent links appear with each experiment.</text>`;
    return;
  }
  const left = 36;
  const right = 24;
  const width = 860;
  const y = 78;
  const xAt = (i) => left + (items.length === 1 ? (width - left - right) / 2 : (i / (items.length - 1)) * (width - left - right));
  const arcs = visible.filter((row) => row.backtrack).map((row) => {
    const i = items.indexOf(row);
    const parentIndex = items.findIndex((item) => item.experiment_id === row.parent);
    if (parentIndex < 0) return "";
    const span = i - parentIndex;
    const lift = Math.min(58, 16 + span * 3);
    const x1 = xAt(parentIndex);
    const x2 = xAt(i);
    const active = i === state.index;
    return `<path class="back-arc" d="M ${x1} ${y} Q ${(x1 + x2) / 2} ${y - lift} ${x2} ${y}" ${active ? 'stroke-width="2.6"' : ""}></path>
      <text class="tick" x="${(x1 + x2) / 2}" y="${y - lift - 3}" text-anchor="middle">${esc(row.parent)}</text>`;
  }).join("");
  const nodes = visible.map((row) => {
    const i = items.indexOf(row);
    const active = i === state.index ? "active" : "";
    const fill = row.status === "failed" ? "#a33b32" : "#243044";
    const ring = row.backtrack ? `stroke="#c47b2b" stroke-width="2"` : "";
    return `<circle class="dot ${active}" data-index="${i}" cx="${xAt(i)}" cy="${y}" r="${i === state.index ? 6.5 : 4.5}" fill="${fill}" ${ring}></circle>
      <text class="tick" x="${xAt(i)}" y="${y + 34}" text-anchor="end" transform="rotate(-65 ${xAt(i)} ${y + 34})">${esc(row.experiment_id)}</text>`;
  }).join("");
  lineage.innerHTML = `${arcs}${nodes}`;
  lineage.querySelectorAll(".dot").forEach((dot) => {
    dot.addEventListener("click", () => {
      pause();
      show(Number(dot.dataset.index));
    });
  });
}

function lineageText(row) {
  if (row.lineage === "starter") return "Recorded parent: starter. This experiment starts from the unmodified starter.";
  if (row.lineage === "continued") return `Recorded parent: ${row.parent}. That is the previous experiment, so this step continues from its model file.`;
  if (row.lineage === "backtrack") return `Recorded parent: ${row.parent}. That is earlier than the previous experiment, ${row.previous_experiment_id}. The controller copies ${row.parent}'s model file before applying this experiment's edits.`;
  return "Recorded parent: Not recorded.";
}

function metricCard(label, value) {
  return `<div class="metric"><b>${value}</b><span>${label}</span></div>`;
}

function renderPanel() {
  const items = rows();
  if (state.index < 0) {
    panel.innerHTML = `
      <p class="eyebrow">Recorded session</p>
      <h2>${state.data.session_id}</h2>
      <p>Replay of finished experiments. Incomplete runs are left out.</p>
      <p class="note">Recorded model: ${state.data.recorded_model}. Provider: ${state.data.recorded_provider}.</p>
      <p class="note">A flag saying the full history was included in each prompt: ${esc(state.data.full_history_included)}.</p>
      <p class="note">${esc(state.data.backtrack_definition)}</p>
    `;
    return;
  }
  if (state.index >= items.length) {
    const best = state.data.highest_auroc_experiment;
    panel.innerHTML = `
      <p class="eyebrow">Replay complete</p>
      <h2>Highest recorded AUROC</h2>
      <p>Experiment ${esc(best.experiment_id)} has the highest validation AUROC in this session, ${fmt(best.validation_auroc, 3)}.</p>
      <p class="note">That rank is the measured AUROC, not a keep decision. ${(state.data.backtracks || []).length} experiments record a parent earlier than the previous experiment.</p>
      <p class="note">${state.data.decision_recording}</p>
    `;
    return;
  }
  const row = items[state.index];
  const diff = row.source_diff
    ? `${row.source_diff.lines_added} lines added, ${row.source_diff.lines_removed} lines removed`
    : "Not recorded";
  panel.innerHTML = `
    <p class="eyebrow">${row.backtrack ? "Backtrack" : "Experiment"}</p>
    <h2>${esc(row.experiment_id)}</h2>
    <h3>Parent</h3>
    <p>${esc(lineageText(row))}</p>
    <h3>Hypothesis</h3>
    <p>${esc(row.hypothesis)}</p>
    <h3>Model change</h3>
    <p>${esc(row.change_summary)}</p>
    <p class="note">Recorded edits: ${row.n_edits ?? "Not recorded"}. Source diff versus the recorded parent's model file: ${esc(diff)}. Recorded training loss field: ${esc(row.recorded_training_loss)}.</p>
    ${row.error ? `<p class="note">Recorded error: ${esc(row.error)}</p>` : ""}
    <h3>Evaluation</h3>
    <div class="metrics">
      ${metricCard("AUROC", fmt(row.validation_auroc, 3))}
      ${metricCard("AUROC versus parent", fmt(row.auroc_vs_parent, 3))}
      ${metricCard("AUPRC", fmt(row.validation_auprc, 3))}
      ${metricCard("Peptide-macro AUROC", fmt(row.peptide_macro_auroc, 3))}
      ${metricCard("Unseen HLA AUROC", `${fmt(row.unseen_hla_auroc, 3)} (n=${row.unseen_hla_n ?? "Not recorded"})`)}
      ${metricCard("Validation loss", fmt(row.validation_loss, 4))}
      ${metricCard("Epochs completed / planned", `${row.epochs_completed ?? "Not recorded"} / ${row.epochs_planned ?? "Not recorded"}`)}
      ${metricCard("Training seconds", fmt(row.runtime_seconds, 1))}
      ${metricCard("Proposal prompt tokens", fmtInt(row.proposal_prompt_tokens))}
    </div>
    <h3>Decision</h3>
    <span class="decision">${row.decision}</span>
    <h3>Scientific reasoning</h3>
    <p>${esc(row.scientific_justification)}</p>
    <p class="note">${esc(row.decision_note)}</p>
    <h3>Recorded evidence for the proposal</h3>
    <p>${esc(row.proposal_evidence)}</p>
  `;
}

function renderChangeLog() {
  const items = rows();
  const last = state.index < 0 ? -1 : Math.min(state.index, items.length - 1);
  if (last < 0) {
    changeLog.innerHTML = `<li class="change-empty">Changes appear here as each experiment is replayed.</li>`;
    return;
  }
  const visible = items.slice(0, last + 1);
  changeLog.innerHTML = visible.map((row, index) => {
    const previous = index > 0 ? visible[index - 1] : null;
    const classes = ["change-row"];
    if (index === state.index) classes.push("current");
    if (row.status === "failed") classes.push("failed");
    else if (raisedBest(row, previous)) classes.push("best");
    return `<li class="${classes.join(" ")}" data-index="${index}">
      <span class="id">${esc(row.experiment_id)}</span>
      <span class="score">${esc(aurocLabel(row))}</span>
      <p>${esc(row.change_summary)}</p>
    </li>`;
  }).join("");
  changeLog.querySelectorAll(".change-row").forEach((row) => {
    row.addEventListener("click", () => {
      pause();
      show(Number(row.dataset.index));
    });
  });
  const current = changeLog.querySelector(".current") || changeLog.querySelector(".change-row:last-child");
  if (current) changeLog.scrollTop = Math.max(0, current.offsetTop - changeLog.clientHeight + current.offsetHeight + 8);
}

function renderProgress() {
  const total = rows().length;
  if (state.index < 0) progress.textContent = "Opening";
  else if (state.index >= total) progress.textContent = "Summary";
  else progress.textContent = `Experiment ${state.index + 1} of ${total}`;
}

function show(index) {
  state.index = index;
  renderChart();
  renderLineage();
  renderPanel();
  renderChangeLog();
  renderProgress();
}

function delay() {
  return STEP_MS / state.speed;
}

function schedule() {
  clearTimeout(state.timer);
  if (!state.playing) return;
  state.timer = setTimeout(() => {
    if (state.index >= rows().length) {
      pause();
      return;
    }
    show(state.index + 1);
    schedule();
  }, delay());
}

function pause() {
  state.playing = false;
  clearTimeout(state.timer);
  playButton.textContent = "Play";
}

function play() {
  state.playing = true;
  playButton.textContent = "Pause";
  if (state.index >= rows().length) show(-1);
  schedule();
}

playButton.addEventListener("click", () => state.playing ? pause() : play());
document.querySelector("#restart").addEventListener("click", () => {
  show(-1);
  play();
});
document.querySelector("#prev").addEventListener("click", () => {
  pause();
  show(Math.max(-1, state.index - 1));
});
document.querySelector("#next").addEventListener("click", () => {
  pause();
  show(Math.min(rows().length, state.index + 1));
});
document.querySelector("#speed").addEventListener("change", (event) => {
  state.speed = Number(event.target.value);
  if (state.playing) schedule();
});

fetch("./data/experiments.json")
  .then((response) => response.json())
  .then((data) => {
    state.data = data;
    const best = data.highest_auroc_experiment;
    sessionLine.textContent = `Session ${data.session_id}. Recorded model ${data.recorded_model} via ${data.recorded_provider}. ${data.excluded.length} incomplete experiment${data.excluded.length === 1 ? "" : "s"} excluded.`;
    footnote.textContent = `${data.metric_definition} ${data.running_best_definition} ${data.backtrack_definition} Highest AUROC is experiment ${best.experiment_id}. Excluded: ${data.excluded.map((row) => `${row.experiment_id} (${row.status})`).join(", ") || "none"}.`;
    renderStats();
    show(-1);
    play();
  })
  .catch(() => {
    sessionLine.textContent = "Could not load data/experiments.json. Run demo/build_demo_data.py, then open this page from the docs directory.";
  });
